from __future__ import annotations
import matplotlib.pyplot as plt
import os
import torch
from icg_net.typing.types import Grasp
from scipy.spatial.transform import Rotation
import trimesh
from icg_net.utils.grasps import (
    convert_contact_to_grasp,
    create_vgn_gripper_marker,
    create_cnet_gripper_marker,
)


from torch import nn

import trimesh
import numpy as np
from icg_net.typing import ModelPredOut

tabmap = plt.get_cmap("tab10", 10)


def cmap(x):
    return tabmap((x + 1) / 10)


scoremap = plt.get_cmap("PiYG", 100)


class GraspVisualizer:
    def __init__(self, num_grasps=15) -> None:
        self.scan: tuple[np.typing.NDArray, np.typing.NDArray] | None = None
        self.grasps: tuple[np.typing.NDArray, np.typing.NDArray, np.typing.NDArray] | None = None

        self.raw_grasps: tuple[np.typing.NDArray, np.typing.NDArray] | None = None
        self.qual_cloud: tuple[np.typing.NDArray, np.typing.NDArray] | None = None
        self.scene_mesh: trimesh.Trimesh | None = None
        self.meshes: list[trimesh.Trimesh] = []
        self.invalid_raw_grasps = None

        self.num_grasps = num_grasps
        self.gripper_convention = "ours"
        self.widths = None
        self.grasp_instances = None

    @classmethod
    def from_prediction(
        cls,
        predictions: ModelPredOut,
        num_grasp: int = 15,
        filter_th=0.0,
        one_each_instance=False,
        max_grasps_per_instance=512,
    ):
        visualizer: GraspVisualizer = cls(num_grasps=num_grasp)
        visualizer.add_scan(predictions.embedding.voxelized_pc, predictions.class_predictions)
        visualizer.add_meshes(predictions.reconstructions)

        pred = predictions.scene_grasp_poses

        ori, contacts, colors, widths, instances, scores = [], [], [], [], [], []

        pred = []
        for i in range(len(predictions.scene_grasp_poses[0])):
            pred.append(
                Grasp(
                    predictions.scene_grasp_poses[0][i].cpu().numpy(),
                    predictions.scene_grasp_poses[1][i].cpu().numpy(),
                    predictions.scene_grasp_poses[2][i].cpu().numpy(),
                    predictions.scene_grasp_poses[3][i].cpu().numpy(),
                    predictions.scene_grasp_poses[4][i].cpu().numpy(),
                )
            )
        for grasp in pred:
            if grasp.score < filter_th:
                continue
            if len(ori) >= num_grasp:
                break
            if one_each_instance and grasp.instance_id in instances:
                continue
            if len([i for i in instances if i == grasp.instance_id]) >= max_grasps_per_instance:
                continue

            ori.append(grasp.orientation)
            contacts.append(grasp.position)
            scoremap(grasp.score)
            if grasp.score > 0.25:
                np.array([0, 1, 0, 1])
            elif grasp.score < 0.1:
                np.array([1, 0, 0, 1])
            else:
                np.array([1, 1, 0, 1])
                
            colors.append(cmap(grasp.instance_id))
            instances.append(grasp.instance_id)
            widths.append(grasp.width)
            scores.append(grasp.score)

        visualizer.grasps = (ori, contacts, colors)
        visualizer.widths = widths
        visualizer.grasp_instances = np.array(instances)
        visualizer.grasp_scores = np.array(scores)
        visualizer.semseg = predictions.embedding.semseg.cpu().numpy().squeeze()

        return visualizer

    def add_meshes(self, meshes: list[trimesh.Trimesh]):
        self.meshes = meshes

    def add_scan(self, scan: torch.tensor, labels: torch.tensor | None = None):
        if isinstance(scan, torch.Tensor):
            scan_np = scan.cpu().numpy()
        else:
            scan_np = scan

        if labels is not None:
            labels_np = labels.cpu().numpy()
            colors = cmap(labels_np)[..., :3]
        else:
            colors = np.ones_like(scan_np)
        self.scan = (scan_np, colors)

    def add_grasps(
        self,
        grasp_points: torch.tensor,
        grasp_labels: torch.tensor,
        grasp_type="occnet",
        mask: torch.tensor | None = None,
        grasp_normals: torch.tensor | None = None,
        obj_scene: str | None = None,
    ):
        if grasp_type == "acronym":
            # monkey patching for acronym
            obj_scene = obj_scene.replace("obj_scenes/", "") + "/meshes/scene.obj"

        if obj_scene is not None and os.path.exists(obj_scene):
            self.scene_mesh = trimesh.load(obj_scene)

        if grasp_type == "occnet" or grasp_type == "occnet_width":
            if grasp_labels.shape[-1] == 12:
                widths = grasp_labels[..., -1] * 0 + 0.08
                print("auto inferred width")
            else:
                widths = grasp_labels[..., -1]
                grasp_labels = grasp_labels[..., :-1]

            if grasp_labels.max() > 1 or grasp_labels.min() < 0:
                grasp_labels = torch.sigmoid(grasp_labels)
            grasp_qual, grasp_orientation = grasp_labels.max(-1)
            grasp_qual, grasp_inst = grasp_qual.max(0)
            grasp_orientation = torch.gather(grasp_orientation, 0, grasp_inst[None, ...]).squeeze()
            widths = torch.gather(widths, 0, grasp_inst[None, ...]).squeeze()

            score_cols = scoremap(grasp_qual.cpu().numpy())[..., :3]

            self.qual_cloud = (grasp_points.cpu().numpy(), score_cols)
            self.raw_grasps = (
                grasp_points[grasp_qual.to(grasp_points.device) > 0].cpu().numpy(),
                cmap(grasp_inst[grasp_qual.to(grasp_points.device) > 0].cpu().numpy()),
            )

            self.invalid_raw_grasps = (
                grasp_points[grasp_qual.to(grasp_points.device) == 0].cpu().numpy(),
                cmap(grasp_inst[grasp_qual.to(grasp_points.device) == 0].cpu().numpy()) * 0.4,
            )

            # Visualize grippers.
            n_grasps = min(self.num_grasps, (grasp_qual > 0.5).sum())
            grasp_qual, topk_idx = torch.topk(grasp_qual, n_grasps, dim=0)

            grasp_inst = torch.gather(grasp_inst, 0, topk_idx).cpu().numpy()
            grasp_orientation = torch.gather(grasp_orientation, 0, topk_idx).cpu().numpy()
            widths = torch.gather(widths, 0, topk_idx).cpu().numpy()
            grasp_points = grasp_points[topk_idx.cpu()].cpu().numpy()
            colors = cmap(grasp_inst)  # [..., :3]

            if len(grasp_points) == 0:
                return
            try:
                if grasp_normals is None:
                    if self.scene_mesh is not None:
                        (_, _, v_id) = self.scene_mesh.nearest.on_surface(grasp_points)
                        grasp_normals = self.scene_mesh.face_normals[v_id]
                    else:
                        import open3d as o3d

                        pc = o3d.geometry.PointCloud()
                        pc.points = o3d.utility.Vector3dVector(grasp_points)
                        pc.estimate_normals()
                        grasp_normals = np.asarray(pc.normals)
                else:
                    grasp_normals = grasp_normals.cpu().numpy()

                oris, contacts = [], []
                for i in range(len(grasp_points)):
                    ori, c = convert_contact_to_grasp(
                        grasp_normals[i], grasp_points[i], grasp_orientation[i], width_pred=widths[i]
                    )
                    oris.append(ori)
                    contacts.append(c)

                self.grasps = (oris, contacts, colors)

                self.widths = widths * 1.2 + 0.01
                self.widths[self.widths > 0.08] = 0.08

                self.grasps = (oris, contacts, colors)

            except Exception:
                pass  # o3d error on cluister

        elif grasp_type == "giga":
            grasp_quality = grasp_labels[..., -1]
            if grasp_quality.max() > 1 or grasp_quality.min() < 0:
                grasp_quality = torch.sigmoid(grasp_quality)
            grasp_inst = (grasp_quality + 0.001 * grasp_labels[..., :4].sum(-1)).argmax(0)
            grasp_labels = torch.gather(grasp_labels, 0, grasp_inst[None, ..., None].repeat_interleave(6, -1)).squeeze()

            grasp_orientation = grasp_labels[:, :4]
            grasp_width = grasp_labels[:, 4]
            grasp_qual = grasp_labels[:, -1].reshape(-1)

            score_cols = scoremap(grasp_qual.cpu().numpy())[..., :3]
            self.qual_cloud = (grasp_points.cpu().numpy(), score_cols)
            self.raw_grasps = (
                grasp_points[grasp_qual.to(grasp_points.device) > 0].cpu().numpy(),
                cmap(grasp_inst[grasp_qual.to(grasp_points.device) > 0].cpu().numpy()),
            )

            # Visualize grippers.
            grasp_qual, topk_idx = torch.topk(grasp_qual, min((grasp_qual > 0.5).sum(), self.num_grasps), dim=0)
            grasp_orientation = grasp_orientation[topk_idx].cpu().numpy()
            grasp_points = grasp_points[topk_idx.cpu()].cpu().numpy()
            colors = cmap(grasp_inst.cpu()[topk_idx.cpu()].numpy())  # [..., :3]

            oris, contacts = [], []
            for i in range(len(grasp_points)):
                ori, c = grasp_orientation[i], grasp_points[i]
                oris.append(ori)
                contacts.append(c)

            oris = np.stack(oris)
            contacts = np.stack(contacts)
            self.grasps = (oris, contacts, colors)

        elif grasp_type == "acronym":
            self.gripper_convention = "contact"
            if grasp_labels.size(-1) == 9:
                # These are gt labels. Last dim is instance_id.
                grasp_inst = grasp_labels[:, 8].long().cpu().numpy()
                grasp_qual = grasp_labels[:, 0].cpu().numpy()
                gripper_width = grasp_labels[:, 1].cpu().numpy()
                dir_labels = grasp_labels[:, 2:5].cpu().numpy()
                approach = grasp_labels[:, 5:8].cpu().numpy()
            else:
                if mask is not None:
                    grasp_labels = grasp_labels[:, mask]
                    grasp_points = grasp_points[mask.to(grasp_points.device)]
                grasp_quality = grasp_labels[..., 0]
                grasp_inst = grasp_quality.argmax(0)
                grasp_labels = torch.gather(
                    grasp_labels, 0, grasp_inst[None, ..., None].repeat_interleave(8, -1)
                ).squeeze()
                grasp_inst = grasp_inst.cpu().numpy()
                grasp_qual = grasp_labels[:, 0].cpu().numpy()
                gripper_width = grasp_labels[:, 1].cpu().numpy()
                dir_labels = grasp_labels[:, 2:5].cpu().numpy()
                approach = grasp_labels[:, 5:8].cpu().numpy()

            score_cols = scoremap(grasp_qual)[..., :3]
            grasp_points = grasp_points.cpu().numpy()
            self.qual_cloud = (grasp_points, score_cols)

            self.raw_grasps = (
                grasp_points[grasp_qual > 0],
                cmap(grasp_inst[grasp_qual > 0]),
            )
            colors = scoremap(grasp_qual)  # [..., :3]
            num_pos = (grasp_qual > 0).sum()
            _, pos_grasps = torch.topk(grasp_labels[:, 0], min(num_pos, self.num_grasps), dim=0)
            pos_grasps = pos_grasps.cpu().numpy()
            grasp_labels = grasp_labels.cpu().numpy()
            grasp_inst = grasp_inst[pos_grasps]
            grasp_qual = grasp_qual[pos_grasps]
            gripper_width = gripper_width[pos_grasps]
            approach = approach[pos_grasps]
            dir_labels = dir_labels[pos_grasps]
            grasp_points = grasp_points[pos_grasps]
            colors = colors[pos_grasps]

            rotations = np.stack([dir_labels, np.cross(approach, dir_labels), approach], axis=1).transpose(0, 2, 1)
            scene_pts = grasp_points
            scene_pts = scene_pts + dir_labels * gripper_width[:, None] / 2 - approach * 0.115  # 07
            rot_quat = Rotation.from_matrix(rotations).as_quat()
            self.grasps = (rot_quat, scene_pts, colors)

    def get_affordance_cloud(self):
        pass

    def get_instance_cloud(self):
        if self.raw_grasps is not None:
            cloud = np.concatenate([self.raw_grasps[0], 255 * self.raw_grasps[1][:, :3]], axis=-1)
        if self.invalid_raw_grasps is not None:
            cloud2 = np.concatenate([self.invalid_raw_grasps[0], 255 * self.invalid_raw_grasps[1][:, :3]], axis=-1)
            cloud = np.concatenate([cloud, cloud2], axis=0)
            # if self.scan is not None:
            #    c2 = np.concatenate([self.scan[0], 255 * self.scan[1][:, :3]], axis=-1)
            #    cloud = np.concatenate([cloud, c2], axis=0)
            return cloud[:, :3], cloud[:, 3:]
        return None

    def get_instance_cloud_wandb(self):
        import wandb

        if self.raw_grasps is not None:
            cloud = np.concatenate([self.raw_grasps[0], 255 * self.raw_grasps[1][:, :3]], axis=-1)
            if self.invalid_raw_grasps is not None:
                cloud2 = np.concatenate([self.invalid_raw_grasps[0], 255 * self.invalid_raw_grasps[1][:, :3]], axis=-1)
                cloud = np.concatenate([cloud, cloud2], axis=0)
        cloud = wandb.Object3D(cloud)
        return cloud

    def get_qual_cloud_wandb(self):
        import wandb

        if self.qual_cloud is not None:
            cloud = np.concatenate([self.qual_cloud[0], 255 * self.qual_cloud[1][:, :3]], axis=-1)
            cloud = wandb.Object3D(cloud)
        return cloud

    def get_qual_cloud(self):
        if self.qual_cloud is not None:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.qual_cloud[0])
            pcd.colors = o3d.utility.Vector3dVector(self.qual_cloud[1])
            return pcd
            return self.qual_cloud[0], self.qual_cloud[1]
        return None

    def get_grasp_mesh(self) -> trimesh.Trimesh | None:
        if self.grasps is None:
            return None

        grasps = []
        for i, (o, c, col) in enumerate(zip(self.grasps[0], self.grasps[1], self.grasps[2])):
            w = self.widths[i] if self.widths is not None else 0.08
            if self.gripper_convention == "ours":
                gripper = create_vgn_gripper_marker(color=col[:3], width=w)
            elif self.gripper_convention == "contact":
                gripper = create_cnet_gripper_marker(color=col[:3], width=w)

            se3_matrix = np.eye(4)
            se3_matrix[:3, -1] = c
            se3_matrix[:3, :3] = Rotation.from_quat(o).as_matrix()
            gripper_r = gripper.apply_transform(se3_matrix)
            grasps.append(gripper_r)
        grasp_scene = trimesh.util.concatenate(grasps)
        if self.scene_mesh is not None:
            grasp_scene = trimesh.util.concatenate([grasp_scene, self.scene_mesh])

        return grasp_scene

    def save(self, folder: str):
        import open3d as o3d

        os.makedirs(folder, exist_ok=True)

        if self.scan is not None:
            scan = o3d.geometry.PointCloud()
            scan.points = o3d.utility.Vector3dVector(self.scan[0])
            scan.colors = o3d.utility.Vector3dVector(self.scan[1] * 0)
            o3d.io.write_point_cloud(os.path.join(folder, "scan.ply"), scan)
            scan.colors = o3d.utility.Vector3dVector(self.scan[1])
            o3d.io.write_point_cloud(os.path.join(folder, "scan_inst.ply"), scan)

        for th in [0.5, 0.6, 0.7, 0.8]:
            import trimesh

            inst_scene = trimesh.Scene()
            qual_scene = trimesh.Scene()

            for i, (o, c, col, score) in enumerate(
                zip(self.grasps[0], self.grasps[1], self.grasps[2], self.grasp_scores)
            ):
                if score < th:
                    continue

                w = self.widths[i] if self.widths is not None else 0.08

                inst_color = cmap(self.grasp_instances[i])
                if self.gripper_convention == "ours":
                    gripper = create_vgn_gripper_marker(width=w, color=inst_color)
                # elif self.gripper_convention == "contact":
                #     gripper = create_cnet_gripper_marker(width=w)

                se3_matrix = np.eye(4)
                se3_matrix[:3, -1] = c
                se3_matrix[:3, :3] = o  # Rotation.from_quat(o).as_matrix()
                gripper_r = gripper.apply_transform(se3_matrix)
                inst_scene.add_geometry(gripper_r.copy())

                gripper_r.visual.face_colors = scoremap((score - 0.5) / (1 - 0.5))
                qual_scene.add_geometry(gripper_r)

            qual_scene.export(os.path.join(folder, "qual_scene_" + str(th) + ".ply"))
            inst_scene.export(os.path.join(folder, "inst_scene_" + str(th) + ".ply"))

            qual_scene.export(os.path.join(folder, "qual_scene_" + str(th) + ".obj"))
            inst_scene.export(os.path.join(folder, "inst_scene_" + str(th) + ".obj"))

            # Instanced Grouped
            th = 0.6
            scenes = {}
            for i, (o, c, col, score) in enumerate(
                zip(self.grasps[0], self.grasps[1], self.grasps[2], self.grasp_scores)
            ):
                if score < th:
                    continue

                w = self.widths[i] if self.widths is not None else 0.08
                inst = self.grasp_instances[i]
                if scenes.get(inst, None) is None:
                    scenes[inst] = trimesh.Scene()

                inst_color = cmap(self.grasp_instances[i])
                if self.gripper_convention == "ours":
                    gripper = create_vgn_gripper_marker(width=w, color=inst_color)
                # elif self.gripper_convention == "contact":
                #     gripper = create_cnet_gripper_marker(width=w)
                se3_matrix = np.eye(4)
                se3_matrix[:3, -1] = c
                se3_matrix[:3, :3] = o
                gripper_r = gripper.apply_transform(se3_matrix)
                scenes[inst].add_geometry(gripper_r.copy())
            for scene_id, scene in scenes.items():
                scene.export(os.path.join(folder, "inst_scene_" + str(scene_id) + ".obj"))

        if self.meshes is not None:
            for i, mesh in enumerate(self.meshes):
                mesh.visual.face_colors = cmap(i)
                mesh.export(os.path.join(folder, "mesh_" + str(i) + ".ply"))

    def show(self, ret_data=False, data_idx=0):
        import open3d as o3d

        clouds = []
        if self.scan is not None:
            scan = o3d.geometry.PointCloud()
            scan.points = o3d.utility.Vector3dVector(self.scan[0])
            scan.colors = o3d.utility.Vector3dVector(self.scan[1])
            clouds.append(scan)

        if self.qual_cloud is not None:
            qual_cloud = o3d.geometry.PointCloud()
            qual_cloud.points = o3d.utility.Vector3dVector(self.qual_cloud[0])
            qual_cloud.colors = o3d.utility.Vector3dVector(self.qual_cloud[1])
            clouds.append(qual_cloud)

        ass = self.get_instance_cloud()
        if ass is not None:
            qual_cloud = o3d.geometry.PointCloud()
            qual_cloud.points = o3d.utility.Vector3dVector(ass[0])
            qual_cloud.colors = o3d.utility.Vector3dVector(ass[1])
            clouds.append(qual_cloud)

        grasps = []
        for i, (o, c, col) in enumerate(zip(self.grasps[0], self.grasps[1], self.grasps[2])):
            w = self.widths[i] if self.widths is not None else 0.08

            if self.gripper_convention == "ours":
                gripper = create_vgn_gripper_marker(width=w)
            elif self.gripper_convention == "contact":
                gripper = create_cnet_gripper_marker(width=w)

            se3_matrix = np.eye(4)
            se3_matrix[:3, -1] = c
            se3_matrix[:3, :3] = o  # Rotation.from_quat(o).as_matrix()
            gripper_r = gripper.apply_transform(se3_matrix)
            mat1 = o3d.visualization.rendering.MaterialRecord()
            mat1.shader = "defaultLitTransparency"
            mat1.base_color = col
            o3d_mesh = gripper_r.as_open3d
            o3d_mesh = o3d_mesh.compute_vertex_normals()
            grasps.append({"name": str(data_idx) + "_gripper_" + str(i), "geometry": o3d_mesh, "material": mat1})


        if self.scene_mesh is not None:
            mat1 = o3d.visualization.rendering.MaterialRecord()
            mat1.shader = "defaultLitTransparency"
            m = self.scene_mesh.as_open3d
            m.compute_vertex_normals()
            grasps.append({"name": str(data_idx) + "scene", "geometry": m, "material": mat1})

        if self.meshes is not None:
            for mesh, idx in self.meshes:
                mat1 = o3d.visualization.rendering.MaterialRecord()
                mat1.shader = "defaultLitTransparency"
                mat1.base_color = cmap(idx)
                mat1.base_color = (mat1.base_color[0], mat1.base_color[1], mat1.base_color[2], 1)
                if len(mesh.vertices) == 0:
                    continue
                trimesh.smoothing.filter_laplacian(mesh, iterations=1)
                m = mesh.as_open3d
                m.compute_vertex_normals()
                grasps.append({"name": str(data_idx) + "mesh_" + str(idx), "geometry": m, "material": mat1})

        if not ret_data:
            o3d.visualization.draw(clouds + grasps)
        else:
            return clouds + grasps
        


class PointCloudVisualizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scan: tuple[np.typing.NDArray, np.typing.NDArray] | None = None
        self.occ_cloud: tuple[np.typing.NDArray, np.typing.NDArray] | None = None

    def add_classified_scan(self, scan: torch.tensor, labels: torch.tensor):
        scan_np = scan.cpu().numpy()
        labels_np = labels.cpu().numpy()
        colors = cmap(labels_np)[..., :3]
        self.scan = (scan_np, colors)

    def add_occupancy_cloud(self, coordinates: torch.tensor, occupancy: torch.tensor):
        # occuapncy has shape n_objects, n_pts
        occ, object_id = occupancy.max(dim=0)
        valid = (occ > 0.5).cpu()
        coordinates = coordinates.cpu()
        coordinates = coordinates[valid].cpu().numpy()
        object_id = object_id[valid].cpu().numpy()
        object_colors = cmap(object_id)[..., :3]
        self.occ_cloud = (coordinates, object_colors)

    def get_occ_cloud_wandb(self, num_pts=4096) -> tuple[any, any]:
        import wandb

        scan = None
        if self.scan is not None and len(self.occ_cloud[0]) > 0:
            idx = np.random.choice(np.arange(len(self.occ_cloud[0])), num_pts)
            scan = wandb.Object3D(
                np.hstack(
                    [
                        self.occ_cloud[0][idx],
                        self.occ_cloud[1][idx] * 255,
                    ]
                )
            )
        return scan

    def downsample(self, data, n_pts=4096):
        idx = np.random.choice(len(data), n_pts)
        return data[idx]

    def get_scan_cloud_wandb(self) -> tuple[any, any]:
        import wandb

        scan = None
        if self.scan is not None:
            scan = wandb.Object3D(
                self.downsample(
                    np.hstack(
                        [
                            self.scan[0],
                            self.scan[1] * 255,
                        ]
                    )
                )
            )
        return scan

    def show(self):
        import open3d as o3d

        scan_cloud = o3d.geometry.PointCloud()
        scan_cloud.points = o3d.utility.Vector3dVector(self.scan[0])
        scan_cloud.colors = o3d.utility.Vector3dVector(self.scan[1])
        occ_cloud = o3d.geometry.PointCloud()
        occ_cloud.points = o3d.utility.Vector3dVector(self.occ_cloud[0])
        occ_cloud.colors = o3d.utility.Vector3dVector(self.occ_cloud[1])
        o3d.visualization.draw([scan_cloud, occ_cloud])
