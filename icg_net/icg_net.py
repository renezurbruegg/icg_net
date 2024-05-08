from __future__ import annotations
import hydra

import torch
import MinkowskiEngine as ME
from hydra.experimental import compose, initialize
from torch import nn
from torch.distributions import Bernoulli
from torch_scatter import scatter_mean
import numpy as np

from icg_net.model.icgnet import ICGNet
from icg_net.utils.grasps import decode_grasp_poses
import trimesh
import time

from icg_net.utils.mcubes.mesh_extractor import Generator3D
from icg_net.typing import Grasp, SceneEmbedding, ModelPredOut

from icg_net.utils.gripper import get_gripper_points, get_gripper_points_mask
from icg_net.utils.checkpoint import load_checkpoint_with_missing_or_exsessive_keys


def get_model(config, device):
    return ICGNetModule(
        config=config,
        device=device,
        grasp_each_object=True,
        n_grasps=4096,
        n_grasp_pred_orientations=6,
        gripper_offset=0.005,
        gripper_offset_perc=10.5,
        max_gripper_width=0.08,
        mesh_resolution=64,
        postprocess=False,
    ).eval()


def resolve_config(cfg, checkpoint=None, reproducible=True) -> ICGNet:

    model: ICGNet = hydra.utils.instantiate(cfg.model)
    checkpoint = checkpoint if checkpoint is not None else cfg.general.checkpoint
    if checkpoint is not None:
        print("FOUND CHECKPOINT. Loading from ", cfg.general.checkpoint)
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model, "model")
    else:
        print("[WARN] No cehckpoint specified!")
    model.repr_sort_idxs = reproducible
    return model


class ICGNetModule(nn.Module):
    def __init__(
        self,
        config="contact_packed_width_eval.yaml",
        device="cuda:0",
        grasp_each_object=True,
        n_grasps=32,
        n_orientations=12,
        max_gripper_width=0.075,
        use_predicted_width=True,
        mesh_resolution=32,
        occ_filter=True,
        gripper_depth=0.045,
        gripper_offset=0.014,
        gripper_offset_perc=0.1,
        n_grasp_pred_orientations=1,
        postprocess=False,
        full_width=False,
        coll_checks=True,
    ):

        if isinstance(config, str):
            with initialize():
                cfg = compose(config_name=config, return_hydra_config=True, overrides=[])
        else:
            cfg = config
        model: ICGNet = resolve_config(cfg)
        super().__init__()
        self.device = device
        self.model = model.to(device)
        self.cfg = cfg

        self.coll_checks = coll_checks

        self.full_width = full_width
        self.grasp_each_object = grasp_each_object
        self.postprocess = postprocess
        self.n_grasps = n_grasps
        self.n_orientations = n_orientations
        self.max_gripper_width = max_gripper_width
        self.use_predicted_width = use_predicted_width
        self.mesh_resolution = mesh_resolution
        self.occ_filter = occ_filter
        self.gripper_depth = gripper_depth
        self.n_grasp_pred_orientations = n_grasp_pred_orientations
        self.gripper_offset = gripper_offset  # 1cm additional offset
        self.gripper_offset_perc = gripper_offset_perc  # + 0.0 * width offset

    def prepare_input_data(self, coords, colors=None, normals=None, **kwargs) -> tuple[ME.SparseTensor, torch.Tensor]:
        data = []
        if self.cfg.data.get("add_colors", False):
            if colors is None:
                colors = torch.ones((coords.shape[0], 3)).to(self.device)  # todo, check normalization
            data.append(colors)
        else:
            data.append(torch.ones((coords.shape[0], 1)).to(self.device))

        if self.cfg.data.get("add_normals", False):
            if normals is None:
                print("No normals provided. Infering normals with o3d")
                import open3d as o3d

                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(coords.cpu().detach().numpy())
                pc.estimate_normals()

                if "camera_pos" not in kwargs:
                    print("WARNING: no camera pos given, Will not be able to compute normal direction accurately.")
                else:
                    pc.orient_normals_towards_camera_location(kwargs["camera_pos"])

                normals = torch.from_numpy(np.asarray(pc.normals)).float().to(self.device)

            data.append(normals)

        if self.cfg.data.get("add_z_coordinate", False):
            data.append(coords[..., -1].unsqueeze(-1))

        data = torch.cat(data, dim=-1)

        # Convert coordinates to minkowski tensors.
        raw_coordinates = coords
        coords, feats, _, voxel_assignement = ME.utils.sparse_quantize(
            coordinates=coords,
            features=data,
            quantization_size=self.cfg.data.voxel_size,
            return_index=True,
            return_inverse=True,
        )
        raw_coordinates = scatter_mean(raw_coordinates, voxel_assignement.to(self.device), dim=0).to(self.device)
        data = ME.SparseTensor(feats, ME.utils.batched_coordinates([coords]), device=self.device)
        return data, raw_coordinates, voxel_assignement

    @torch.no_grad()
    def encode_inputs(self, coords, colors=None, normals=None, **kwargs) -> SceneEmbedding:
        data, raw_coordinates, voxel_assignement = self.prepare_input_data(
            coords, colors=colors, normals=normals, **kwargs
        )

        output = self.model(data, [raw_coordinates])
        # These latents did not get assigned the "no object class"

        valid_latent_ids = (output["pred_logits"].argmax(-1) != (output["pred_logits"].size(-1) - 1)).squeeze(0)
        predicted_masks = output["pred_masks"][0]
        predicted_masks[..., ~valid_latent_ids] = -10000

        semseg_latent = output["pred_logits"].argmax(-1)
        predicted_instances = predicted_masks.argmax(dim=-1)
        semseg_points = semseg_latent[:, predicted_instances]

        valid_latent_ids[:] = False
        valid_latent_ids[predicted_instances.unique()] = True

        # Extract Latents
        shape_latents = output["all_queries"][-1][:, valid_latent_ids].squeeze(0)
        shape_pos_encs = output["positional_encodings"][-1][valid_latent_ids, :]

        scene_grasp_queries = output["scene_grasp_queries"][-1][:, valid_latent_ids, :]

        _, class_labels = torch.unique(predicted_instances, return_inverse=True)
        pointwise_labels = class_labels[voxel_assignement]

        return SceneEmbedding(
            scene_grasps=scene_grasp_queries,
            shape=shape_latents,
            pos_encodings=shape_pos_encs,
            class_labels=class_labels,
            voxelized_pc=raw_coordinates,
            pointwise_labels=pointwise_labels,
            voxel_assignement=voxel_assignement,
            semseg_points=semseg_points,
            semseg_latents=semseg_latent,
            semseg=semseg_latent[..., valid_latent_ids],
        )

    @torch.no_grad()
    def decode_occ(self, query_pts, embedding: SceneEmbedding, **kwargs) -> torch.Tensor:
        pts = query_pts.to(self.device)
        occ_values = self.model.decode_sdf(
            queries=[embedding.shape], query_positional_encoding=[embedding.pos_encodings], sdf_points=list(pts)
        )[0]

        # PointSDF prediction. Occupancy logits at pos '0'
        if occ_values.ndim == 3:
            occ_values = occ_values[..., 0]

        if "instance_id" in kwargs:
            occ_values = occ_values[kwargs["instance_id"]]
        else:
            # Scene level reconstruction
            occ_values = occ_values.max(dim=0)[0]

        return Bernoulli(logits=occ_values.squeeze())

    @torch.no_grad()
    def affordance_to_grasp_pose(
        self, affordance: torch.Tensor, normals: torch.Tensor, points: torch.Tensor
    ) -> list[Grasp]:
        if len(affordance.ravel()) == 0:
            return []

        instances = []
        if self.grasp_each_object:
            new_grasps = []
            new_idxs = []
            instances = []
            # lets only keep best grasp for each object
            for instance, g in enumerate(affordance):
                try:
                    max_vals, max_ind = torch.topk(g[..., :-1].sum(dim=1), k=min(self.n_grasps, len(g)))
                except Exception:
                    import pdb

                    pdb.set_trace()
                new_grasps.append(g[max_ind])
                new_idxs.append(max_ind)
                instances.append(instance + 0 * max_ind)
            # coresponding point.
            instances = torch.cat(instances)  # .cpu().numpy()
            points = points[torch.cat(new_idxs)]
            normals = normals[torch.cat(new_idxs)]
            affordance = torch.cat(new_grasps)
            n_grasps = len(affordance)
            _, latent_ids = affordance.max(0)

        else:
            affordance, latent_ids = affordance.max(0)
            n_grasps = self.n_grasps
            raise NotImplementedError("Instances not implemented for grasp_each_object=False")

        if affordance.size(-1) > self.n_orientations and self.use_predicted_width:
            width = affordance[..., self.n_orientations]  # * 1.2 + 0.01  # if possible, grasp 0.5mm from surface
            affordance = affordance[..., : self.n_orientations]
            if self.full_width:
                width = torch.clip(width, self.max_gripper_width, self.max_gripper_width)
            else:
                width = torch.clip(width * (1 + self.gripper_offset_perc), 0, self.max_gripper_width)
            slack_corr = torch.clip((width + self.gripper_offset), 0, self.max_gripper_width)
            # this is the slack we get
            slack = slack_corr - width  # + 0.002
        else:
            width = affordance[..., 0] * 0 + self.max_gripper_width

        # affordance has shape n_grasps x 12|13

        time.time()
        orientation, translation, scores, raw_points, ids, widths = decode_grasp_poses(
            points,
            width,  # *0+ 0.08,
            normals,
            affordance,
            n_grasps,
            slack=slack,
            agg="max",
            n_orientations=self.n_grasp_pred_orientations,
            return_np=False,
        )
        scores = 1 / (1 + torch.exp(-scores))

        return orientation, translation, scores, widths, instances[ids]

    @torch.no_grad()
    def _get_grasps(
        self,
        grasp_pts: torch.Tensor,
        grasp_normals,
        embeddings: SceneEmbedding,
        return_scene_grasps: bool = True,
        **kwargs,
    ):
        obj_grasp_poses, scene_grasp_poses = [], []

        grasp_pts = grasp_pts.to(self.device)

        if len(grasp_pts) == 0:
            print("Provided grasp points were empty!")

        scene_centric_grasps = self.model.decode_grasps(
            list(embeddings.scene_grasps),
            query_positional_encoding=list(embeddings.pos_encodings.unsqueeze(0)),
            grasp_points=[grasp_pts],
        )[0]

        if grasp_normals is None:
            # print("Infering grasp normals...")
            import open3d as o3d

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(grasp_pts.cpu().numpy())

            pc.estimate_normals()

            if "camera_pos" not in kwargs:
                print("WARNING: no camera pos given, Will not be able to compute normal direction accurately.")
            else:
                pc.orient_normals_towards_camera_location(kwargs["camera_pos"])
            grasp_normals = torch.from_numpy(np.asarray(pc.normals)).float().to(self.device)
        else:
            grasp_normals = grasp_normals.to(self.device)

            # scene_grasp_poses, object_centric_grasps = None, None
        if return_scene_grasps:
            # orientation, translation, scores, widths
            scene_grasp_info = self.affordance_to_grasp_pose(scene_centric_grasps, grasp_normals, grasp_pts)
            if self.occ_filter and len(scene_grasp_info) > 0:

                is_free = scene_grasp_info[2] > 0.1
                pts_on_gripper = get_gripper_points(scene_grasp_info[0][is_free], scene_grasp_info[1][is_free])
                gripper_mask = get_gripper_points_mask(pts_on_gripper, 0.05)

                is_free[is_free.clone()] = gripper_mask

                rtip = (
                    scene_grasp_info[0][is_free] @ torch.tensor([0, -0.04, 0.045]).to(grasp_pts.device)
                    + scene_grasp_info[1][is_free]
                )
                ltip = (  # center loc
                    scene_grasp_info[0][is_free] @ torch.tensor([0, 0.04, 0.045]).to(grasp_pts.device)
                    + scene_grasp_info[1][is_free]
                )
                if len(rtip) > 1 and self.coll_checks:
                    occ_per_latentr = self.model.decode_sdf(
                        [embeddings.shape],
                        query_positional_encoding=[embeddings.pos_encodings],
                        sdf_points=[rtip],
                    )[0].max(0)[0]

                    occ_per_latentl = self.model.decode_sdf(
                        [embeddings.shape],
                        query_positional_encoding=[embeddings.pos_encodings],
                        sdf_points=[ltip],
                    )[0].max(0)[0]

                    occ_per_lat = ~((occ_per_latentr.squeeze() < 0) * (occ_per_latentl.squeeze() < 0))
                    occ_per_lat = occ_per_lat.float()
                    occ_per_lat[occ_per_lat == 1] = 0.0  # 0.2
                    occ_per_lat[occ_per_lat == 0] = 1
                    is_free_float = is_free.float()
                    is_free_float[is_free.clone()] = occ_per_lat.float()
                else:
                    is_free_float = is_free.float()

                scene_grasp_info = [
                    scene_grasp_info[0],
                    scene_grasp_info[1],
                    scene_grasp_info[2] * (is_free_float + 0.0001),
                    scene_grasp_info[3],
                    scene_grasp_info[4],
                ]
                mask = scene_grasp_info[2] > kwargs.get("th", 0.3)
                scene_grasp_info = [s[mask] for s in scene_grasp_info]
            scene_grasp_poses = scene_grasp_info

        return scene_grasp_poses, obj_grasp_poses

    @torch.no_grad()
    def __call__(
        self,
        coords: torch.Tensor,
        colors: torch.Tensor | None = None,
        normals: torch.Tensor | None = None,
        grasp_pts: torch.Tensor | None = None,
        grasp_normals: torch.Tensor | None = None,
        return_meshes: bool = False,
        return_scene_grasps: bool = True,
        **kwargs,
    ) -> ModelPredOut:
        coords = coords.to(self.device)
        if normals is not None:
            normals = normals.to(self.device)

        self.model.eval()
        embeddings = self.encode_inputs(coords, colors=colors, normals=normals, **kwargs)

        # -------------------------------------------------------------
        # ------------------- Grasp Prediction-- ----------------------
        # -------------------------------------------------------------
        scene_grasp_poses, obj_grasp_poses = self._get_grasps(
            grasp_pts, grasp_normals, embeddings, return_scene_grasps, **kwargs
        )

        if kwargs.get("resample", False) and (scene_grasp_poses[2] > kwargs.get("th", 0.5)).sum() < 5:
            """Trigger resampling of surface points if not enough valid grasps are found."""
            try:
                import open3d as o3d

                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(grasp_pts[:-2].cpu().numpy())

                # find surface points
                scene_bounds = self.model.scene_bounds
                all_pts = []
                for idx, (latent, pos_enc) in enumerate(zip(embeddings.shape, embeddings.pos_encodings)):
                    n_pts = 2**14
                    pts = (
                        grasp_pts[torch.randint(0, len(grasp_pts), (n_pts,))].clone()
                        + torch.randn((n_pts, 3), device=grasp_pts.device) * 0.01
                    )
                    pts = pts.unsqueeze(0)
                    pts.requires_grad = True

                    def get_sdf_value(pts: torch.Tensor, idx=idx):
                        occ_values = self.model.decode_sdf(
                            [latent.unsqueeze(0)],
                            query_positional_encoding=[pos_enc.unsqueeze(0)],
                            sdf_points=list(pts.cuda()),
                            level=-1,
                        )[0]
                        return occ_values

                    with torch.enable_grad():
                        optimizer = torch.optim.AdamW([pts], lr=0.0005)
                        for i in range(20):
                            optimizer.zero_grad()
                            occ = get_sdf_value(pts).abs().clip(0.003)
                            loss = occ.mean()
                            with torch.no_grad():
                                out_of_bounds = torch.logical_or((pts < 0), (pts > 0.3))
                                out_of_bounds = out_of_bounds.any(-1)
                                out_of_bounds = torch.logical_or((out_of_bounds), pts[..., -1] < 0.05)

                                if out_of_bounds.any():
                                    pts[out_of_bounds.clone()] = (
                                        torch.rand(
                                            [*out_of_bounds[out_of_bounds].shape, 3], device=out_of_bounds.device
                                        )
                                        * 0.2
                                    )
                                if i % 2 == 0:
                                    all_pts.append(pts.squeeze()[occ.squeeze() <= 0.5].clone())
                                    pts += torch.randn(pts.shape, device=grasp_pts.device) * 0.001
                            # pts =pts + torch.randn(pts.shape, device=grasp_pts.device)*0.0001
                            loss.backward()
                            optimizer.step()

                cloud = o3d.geometry.PointCloud()
                all_pts_gpu = torch.cat([grasp_pts, torch.cat(all_pts)])
                all_pts = all_pts_gpu.cpu().detach().numpy()
                cloud.points = o3d.utility.Vector3dVector(all_pts)

                # o3d.visualization.draw([cloud, pc])

                cloud, ind = cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
                # pc, ind = pc.remove_radius_outlier(nb_points=30, radius=0.03)
                cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))

                normals = torch.from_numpy(np.asarray(cloud.normals)).to(all_pts_gpu.device).float()
                contacts = torch.from_numpy(np.asarray(cloud.points)).to(all_pts_gpu.device).float()
                sign_norm = get_sdf_value((contacts + normals * 0.005).unsqueeze(0)).squeeze() > 0
                sign_neg_norm = get_sdf_value((contacts - normals * 0.005).unsqueeze(0)).squeeze() > 0

                valid = sign_neg_norm != sign_norm
                valid = torch.logical_and(valid, contacts[..., -1] > 0.054)
                switch = sign_norm

                normals[switch, :] = -normals[switch, :]

                cloud.points = o3d.utility.Vector3dVector(contacts[valid].cpu().detach().numpy())
                cloud.normals = o3d.utility.Vector3dVector(normals[valid].cpu().detach().numpy())

                if valid.sum() > 5:
                    scene_grasp_poses, obj_grasp_poses = self._get_grasps(
                        contacts[valid], normals[valid], embeddings, return_scene_grasps, **kwargs
                    )
            except Exception as e:
                print("Error in resampling", e)

        # -------------------------------------------------------------
        # ------------------- Occupancy Prediction --------------------
        # -------------------------------------------------------------
        meshes = []

        if return_meshes:
            scene_bounds = self.model.scene_bounds
            for idx, (latent, pos_enc) in enumerate(zip(embeddings.shape, embeddings.pos_encodings)):

                @torch.no_grad()
                def get_sdf_value(pts: torch.Tensor, idx=idx):
                    occ_values = self.model.decode_sdf(
                        [latent.unsqueeze(0)],
                        query_positional_encoding=[pos_enc.unsqueeze(0)],
                        sdf_points=list(pts.cuda()),
                        level=-1,
                    )[0]
                    return occ_values.cpu().squeeze()

                gen = Generator3D(
                    get_sdf_value,
                    resolution0=self.mesh_resolution,  # 128, #32,
                    points_batch_size=4096 * 4,
                    scale=1,
                    translation=0,
                    threshold=0,
                )
                mesh = gen.generate_mesh(
                    volume_bounds=np.array(
                        [
                            (scene_bounds[0][0].cpu().numpy() - 0.05).squeeze().tolist(),
                            (scene_bounds[1][0].cpu().numpy() + 0.05).squeeze().tolist(),
                        ]
                    )
                )
                meshes.append(mesh)

            if not self.postprocess:
                meshes = [(m, i) for i, m in enumerate(meshes)]
            else:
                # Postprocess meshes based on connectivity
                time.time()
                # Fuse meshes, relabel instances
                col_man = trimesh.collision.CollisionManager()
                connections = []

                # import networkx as nx
                # coll_graph = nx.Graph()

                idx_cnter = 0
                meshes_filtered = []

                SPLIT_COMPONENTS = False

                for idx, mesh in enumerate(meshes):
                    if len(mesh.edges) == 0:
                        continue

                    if SPLIT_COMPONENTS:
                        components = mesh.split()
                        for c in components:
                            if len(c.vertices) < 5:
                                print("dropping", c, "too small")
                                continue
                            col_man.add_object(str(idx_cnter), c)
                            # coll_graph.add_node(idx_cnter)

                            meshes_filtered.append(c)
                            idx_cnter += 1
                    else:
                        meshes_filtered.append(mesh)
                        col_man.add_object(str(idx_cnter), mesh)
                        idx_cnter += 1

                removed = []
                for i, mesh in enumerate(meshes_filtered):
                    col_man.remove_object(str(i))
                    dist, name = col_man.min_distance_single(mesh, return_name=True)

                    if dist < 0.001:  # 8mm
                        # found connected component
                        # coll_graph.add_edge(int(name), i)
                        removed.append(i)
                        meshes_filtered[int(name)] = trimesh.util.concatenate([meshes_filtered[int(name)], mesh])
                        col_man.remove_object(name)
                        col_man.add_object(name, meshes_filtered[int(name)])
                        connections.append([int(name), i])
                    else:
                        col_man.add_object(str(i), mesh)
                        connections.append([i, i])

                import networkx as nx

                remaining_nodes = [i for i, m in enumerate(meshes_filtered) if i not in removed]

                g = nx.Graph()
                g.add_edges_from(connections)
                connected_components = list(nx.connected_components(g))
                mapping_dict = {}
                for i, c in enumerate(connected_components):
                    for idx in c:
                        mapping_dict[idx] = remaining_nodes[i]

                meshes = [(m, mapping_dict[i]) for i, m in enumerate(meshes_filtered) if i not in removed]

                # smooth
                # for m in meshes:
                #     trimesh.smoothing.filter_laplacian(m, iterations=1, lamb=0.5)
                if return_scene_grasps:
                    time.time()
                    grasp_ids = scene_grasp_poses[4].clone()

                    for src, tgt in mapping_dict.items():
                        scene_grasp_poses[4][grasp_ids == src] = tgt
                time.time()
                lbls = embeddings.class_labels.clone()

                for src, tgt in mapping_dict.items():
                    embeddings.class_labels[lbls == src] = tgt

        labels = embeddings.class_labels
        return ModelPredOut(
            embedding=embeddings,
            class_predictions=labels,
            scene_grasp_poses=scene_grasp_poses,
            reconstructions=meshes,
        )
