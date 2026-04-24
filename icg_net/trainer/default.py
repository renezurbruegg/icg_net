from __future__ import annotations
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
import os
import hydra
import MinkowskiEngine as ME
import numpy as np
import pytorch_lightning as pl
import torch
from typing import List, NamedTuple, Union
from torch import tensor
import numpy as np
import torch

# from occupancy_prediction.utils.timing import Timer
from icg_net.trainer.eval.metric import GraspMetrics, OccupancyMetrics, InstanceClassificationMetrics


from icg_net.trainer.criterion.criterion import SetCriterion
from icg_net.trainer.matcher.instance_matcher import HungarianMatcher
from icg_net.vis.visualizer import PointCloudVisualizer, GraspVisualizer
from icg_net.model.icgnet import ICGNetOutput

# from occupancy_prediction.model.mask3d import Mask3D, Mask3DOutput

import traceback
import trimesh
import random
import trimesh

class Timer:
    def __init__(self):
        self.timers = {}

    def __getitem__(self, key):
        if key not in self.timers:
            self.timers[key] = Timer()
        return self.timers[key]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def print_stats(self):
        for key, timer in self.timers.items():
            print(f"{key}: {timer}")

    def __str__(self):
        return str(self.timers)

# torch.set_float32_matmul_precision("medium")
class Queries(NamedTuple):
    shape_queries: List[List[torch.tensor]]
    object_grasp_queries: List[List[torch.tensor]] | None
    scene_grasp_queries: List[List[torch.tensor]] | None
    pos_encodings: List[torch.tensor]


def relabel_from_zero(data: torch.Tensor):
    """
    Relabels the data from 0 to n, where n is the number of unique elements in data.
    :param data: torch.Tensor
    :return: torch.Tensor
    """
    unique, inverse = torch.unique(data, return_inverse=True)
    return inverse


class RegularCheckpointing(pl.Callback):
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        general = pl_module.config.general
        trainer.save_checkpoint(f"{general.save_dir}/last-epoch.ckpt")
        print("Checkpoint created")


def get_data(
    feature_dim: int,
    spatial_dim: int = 3,
    with_coords: bool = False,
    scale=10,
    n_samples=100,
) -> ME.SparseTensor:
    coords = [
        torch.rand(n_samples, spatial_dim) * scale for _ in range(2)
    ]  # pylint: disable=no-member
    feats = [
        torch.rand(n_samples, feature_dim) for _ in range(2)
    ]  # pylint: disable=no-member
    coordinates, features = ME.utils.sparse_collate(
        coords=coords, feats=feats
    )  # pylint: disable=unb
    data = ME.SparseTensor(
        torch.cat([features, torch.cat(coords)], dim=-1), coordinates
    )
    ds_coordinates = [d[:, -spatial_dim:] for d in data.decomposed_features]

    coordinates, features = ME.utils.sparse_collate(
        coords=coords, feats=feats
    )  # pylint: disable=unb
    data = ME.SparseTensor(features, coordinates)

    if with_coords:
        return data, ds_coordinates
    return data


class GraspDetection(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # self.example_input_array = get_data(1, scale = 100, n_samples=1000)

        self.ss_warmup_epochs = 0
        if hasattr(config.general, "ss_warmup_epochs"):
            self.ss_warmup_epochs = config.general.ss_warmup_epochs

        if hasattr(config.general, "feature_interpolation"):
            self.feature_interpolation = config.general.feature_interpolation
        else:
            self.feature_interpolation = "test"  # None

        # self.example_input_array = get_data(1)
        self.with_aux_losses = config.general.with_aux_losses
        self.do_visualize = False
        self.show_semantic = True
        self.show_scene_grasps = True
        self.show_object_grasps = True

        self.local_vis = False

        self.debug_view_ds = False

        self.show_occ = True

        self.vis_per_batch = 2
        self.num_batch_to_vis = 4

        # Metrics
        self.obj_grasp_metrics = GraspMetrics()
        self.scene_grasp_metrics = GraspMetrics()

        # self.full_scene_grasp_metrics = GraspMetrics()
        self.full_occ_metrics = OccupancyMetrics()


        self.occ_metrics = OccupancyMetrics()
        self.instance_metrics = InstanceClassificationMetrics()

        self.save_hyperparameters()

        self.vis_attention = config.general.vis_attention
        self.config = config
        # model
        self.model: Mask3D = hydra.utils.instantiate(config.model)
        self.timer = Timer()

        # TODO, this ugly
        matcher: HungarianMatcher = hydra.utils.instantiate(config.matcher)
        weight_dict = {
            "loss_ce": matcher.cost_class,
            "loss_mask": matcher.cost_mask,
            "loss_dice": matcher.cost_dice,
        }

        for c, v in matcher.costs.items():
            weight_dict["loss_" + c] = v
            print("Registered Cost:", "loss_" + c, v)

        aux_weight_dict = {}
        for i in range(self.model.num_levels * self.model.num_decoders):
            if i not in self.config.general.ignore_mask_idx:
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            else:
                aux_weight_dict.update(
                    {k + f"_{i}": 0.0 for k, v in weight_dict.items()}
                )
        weight_dict.update(aux_weight_dict)

        self.criterion: SetCriterion = hydra.utils.instantiate(
            config.loss, matcher=matcher, weight_dict=weight_dict
        )

    def forward(
        self,
        voxelized_data: ME.SparseTensor,
        raw_coordinates: List[tensor],
        poses: Union[List[tensor], None] = None,
    ) -> Mask3DOutput:
        with self.timer["model_forward"]:
            poses = (
                [torch.from_numpy(p).to(self.device) for p in poses]
                if (poses is not None and len(poses) != 0 and poses[0] is not None)
                else None
            )

            x = self.model(
                voxelized_data,
                raw_coordinates=[raw_coordinates.to(self.device)],
                poses=poses,
            )
        return x

    def get_matched_latents(
        self,
        output: Mask3DOutput,
        indices: List[tuple[tensor, tensor]],
        valid_objects=None,
    ) -> tuple[Queries, List[torch.tensor]]:
        # get matchings using hung matcher
        if valid_objects is not None:
            src_indices = [
                ind[0][valid_objects[idx]] for idx, ind in enumerate(indices)
            ]
            tgt_indices = [
                ind[1][valid_objects[idx]] for idx, ind in enumerate(indices)
            ]
        else:
            src_indices = [ind[0] for ind in indices]
            tgt_indices = [ind[1] for ind in indices]

        shape_queries = [c.clone() for c in output["all_queries"]]
        object_grasp_queries = [c.clone() for c in output["object_grasp_queries"]]
        scene_grasp_queries = [c.clone() for c in output["scene_grasp_queries"]]
        pos_enc = [c.clone() for c in output["positional_encodings"]]

        ret_shape_queries = []
        ret_object_grasp_queries = []
        ret_scene_grasp_queries = []

        for stage_idx in range(len(shape_queries)):
            ret_shape_queries.append([])
            ret_object_grasp_queries.append([])
            ret_scene_grasp_queries.append([])

            for batch_idx, selected_idxs in enumerate(src_indices):
                ret_shape_queries[stage_idx].append(
                    shape_queries[stage_idx][batch_idx][selected_idxs]
                )
                if len(object_grasp_queries) > 0:
                    ret_object_grasp_queries[stage_idx].append(
                        object_grasp_queries[stage_idx][batch_idx][selected_idxs]
                    )
                if len(scene_grasp_queries) > 0:
                    ret_scene_grasp_queries[stage_idx].append(
                        scene_grasp_queries[stage_idx][batch_idx][selected_idxs]
                    )

                if stage_idx == 0:
                    pos_enc[batch_idx] = pos_enc[batch_idx][selected_idxs]

        src_indices = [ind[0] for ind in indices]
        tgt_indices = [ind[1] for ind in indices]

        return (
            Queries(
                ret_shape_queries,
                ret_object_grasp_queries,
                ret_scene_grasp_queries,
                pos_enc,
            ),
            tgt_indices,
            src_indices,
        )

    def get_query_predictions(
        self,
        queries: Queries,
        sdf_pts: list[torch.tensor] | None = None,
        obj_grasp_pts: list[torch.tensor] | None = None,
        scene_grasp_pts: list[torch.tensor] | None = None,
        additional_grasp_pts: list[torch.tensor] | None = None,
        normal_pts=None,
    ) -> tuple[
        List[List[torch.tensor]], List[List[torch.tensor]], List[List[torch.tensor]]
    ]:
        values = [[], [], []]
        if additional_grasp_pts is not None:
            values.append([])
        if normal_pts is not None:
            values.append([])

        with self.timer["query_decoding"]:
            if sdf_pts is not None:
                shape_queries = (
                    queries.shape_queries
                    if self.with_aux_losses
                    else [queries.shape_queries[-1]]
                )
                for idx, q in enumerate(shape_queries):  # Refinement Levels
                    values[0].append(
                        self.model.decode_sdf(
                            q, queries.pos_encodings, sdf_pts, level=idx
                        )
                    )

            if obj_grasp_pts is not None:
                for idx, q in enumerate(
                    queries.object_grasp_queries
                    if self.with_aux_losses
                    else [queries.object_grasp_queries[-1]]
                ):  # queries.object_grasp_queries:  # Refinement Levels
                    values[1].append(
                        self.model.decode_grasps(
                            q, queries.pos_encodings, obj_grasp_pts, level=idx
                        )
                    )

            if scene_grasp_pts is not None:
                for idx, q in enumerate(
                    queries.scene_grasp_queries
                    if self.with_aux_losses
                    else [queries.scene_grasp_queries[-1]]
                ):  # queries.scene_grasp_queries:  # Refinement Levels
                    values[2].append(
                        self.model.decode_grasps(
                            q, queries.pos_encodings, scene_grasp_pts, level=idx
                        )
                    )
            if additional_grasp_pts is not None:
                for q in (
                    queries.scene_grasp_queries
                    if self.with_aux_losses
                    else [queries.scene_grasp_queries[-1]]
                ):  # queries.scene_grasp_queries:  # Refinement Levels
                    values[3].append(
                        self.model.decode_grasps(
                            q,
                            queries.pos_encodings,
                            additional_grasp_pts,
                        )
                    )

            # if normal_pts is not None:
            # Normals disabled
            # shape_queries = queries.shape_queries if self.with_aux_losses else [queries.shape_queries[-1]]
            # for idx, q in enumerate(shape_queries):  # Refinement Levels
            #      values[-1].append(
            #         self.model.decode_normals(
            #             q,
            #             queries.pos_encodings,
            #             normal_pts,
            #         )
            #     )

        return values

    @torch.no_grad()
    def inference_step(
        self,
        data: ME.SparseTensor,
        raw_coordinates: torch.Tensor,
        grasp_query_pts: torch.Tensor = None,
        return_sdf_latents=True,
        return_queries=False,
        return_grasps=True,
        return_instances=True,
        return_latent_ids=False,
        query_mask=None,
        queries=None,
        pos_encodings=None,
        **kwargs,
    ):
        self.model.eval()
        data_train = ME.SparseTensor(
            coordinates=data.coordinates, features=data.features, device=self.device
        )
        output = self.forward(data_train, raw_coordinates=raw_coordinates)
        # These latents did not get assigned the "no object class"
        valid_latent_ids = (
            output["pred_logits"].argmax(-1) != (output["pred_logits"].size(-1) - 1)
        ).squeeze(0)
        print(
            "Found ",
            valid_latent_ids.sum().item(),
            " unique latents in the scene out of ",
            len(valid_latent_ids),
            " total",
        )
        predicted_masks = output["pred_masks"][0]
        predicted_masks[..., ~valid_latent_ids] = -10000
        predicted_instances = predicted_masks.argmax(dim=-1)
        print(
            "Found ",
            len(predicted_instances.unique()),
            " unique instances with mask correspondence",
        )
        valid_latent_ids[:] = False
        valid_latent_ids[predicted_instances.unique()] = True

        new_labels = relabel_from_zero(predicted_instances)
        # Extract Latents
        shape_latents = output["all_queries"][-1][:, valid_latent_ids].squeeze(0)
        shape_pos_encs = output["positional_encodings"][-1][valid_latent_ids, :]
        ret_data = [new_labels]

        if return_grasps:
            obj_grasp_queries = output["object_grasp_queries"][-1][
                :, valid_latent_ids, :
            ]
            scene_grasp_queries = output["scene_grasp_queries"][-1][
                :, valid_latent_ids, :
            ]
            query_pos_encodings = output["positional_encodings"][-1][
                valid_latent_ids, :
            ][None, ...]
            if grasp_query_pts is None:
                # print("No grasp query points provided, using raw coordinates")
                grasp_query_pts = raw_coordinates
            else:
                grasp_query_pts = grasp_query_pts.to(self.device)

            if len(grasp_query_pts) == 0:
                print("Provided grasp points were empty!")

            scene_centric_grasps = self.model.decode_grasps(
                list(scene_grasp_queries),
                query_positional_encoding=list(query_pos_encodings),
                grasp_points=[grasp_query_pts],
            )[0]
            ret_data += [scene_centric_grasps]

        if return_sdf_latents:
            ret_data.append([shape_latents, shape_pos_encs])

        if return_latent_ids:
            shape_ids = (
                torch.arange(0, self.model.num_queries)
                .unsqueeze(0)
                .repeat_interleave(len(output["instance_latents"]), 0)
                .to(self.device)
            )
            if query_mask is not None and len(query_mask) > 0:
                assert len(query_mask) == 1, "Only bs of 1 supported"
                mask = sum(shape_ids[0, :] == q for q in query_mask[0]).bool()
                shape_ids = shape_ids[:, ~mask]

            shape_ids = shape_ids[:, valid_latent_ids]
            ret_data.append(shape_ids)

        if return_queries:
            ret_data.append(output["query_info"])

        return ret_data  # scene_centric_grasps.sigmoid(), new_labels, shape_latents

    def training_step(self, batch, batch_idx):
        data, target, file_names, transforms = batch

        if len(target) == 0:
            print("no targets")
            return None

        if data.features.shape[0] > self.config.general.max_batch_size:
            raise RuntimeError("BATCH TOO BIG")

        raw_coordinates = data.features[:, -3:]
        data.features = data.features[:, :-3]

        data_train = ME.SparseTensor(
            coordinates=data.coordinates, features=data.features, device=self.device
        )

        try:
            output = self.forward(
                data_train, raw_coordinates=raw_coordinates, poses=data.poses
            )

            
            if self.criterion.matcher.costs.get("sdf_matching", 0) > 0 :
                with torch.no_grad():
                    n_sdf_pts_pos = 2*2048
                    n_sdf_pts_neg =  1024

                    sdf_match_pts = []
                    sdf_match_vals = []
                    for pts, gt_values in zip(data.sdf_queries, data.sdf_values):
                        scene_pos_idxs = []
                        for gt_inst in gt_values:
                            p_idxs = np.where(gt_inst > 0)[0]
                            scene_pos_idxs.append(np.random.choice(p_idxs, int(1 + n_sdf_pts_pos / len(gt_values))))

                        scene_neg = np.where(~(gt_values > 0).max(0))[0]

                        q_idxs =  np.concatenate([*scene_pos_idxs, np.random.choice(scene_neg, n_sdf_pts_neg)])
                        q_idxs = np.random.choice(q_idxs, n_sdf_pts_neg + n_sdf_pts_pos, replace=False)

                        sdf_match_pts.append(pts[q_idxs].to(self.device))
                        sdf_match_vals.append(torch.from_numpy(gt_values[:, q_idxs]).to(self.device))
                    

                    sdf_preds = self.get_query_predictions(sdf_pts=sdf_match_pts, queries =  Queries(output["all_queries"], None, None, output["positional_encodings"]))[0]
                    output["sdf_matching"]=sdf_preds
                    
                    for idx in range(len(target)):
                        target[idx]["sdf_matching"] = sdf_match_vals[idx]
                    
            with self.timer["query_association"]:
                # get matchings using hung matcher
                indices = self.criterion.get_matches(output, target, mask_type="masks")
                # limit num queries
                # compare max 5 predictions to avoid memory issues (e.g. acronym)
                for idx in range(len(indices)):
                    if len(indices[idx][0]) > 5:
                        selected = torch.randperm(len(indices[idx][0]))
                        selected = selected[:6]
                        # print("Found indices that were too large. Entries: ", len(indices[idx][0]))
                        indices[idx] = (
                            indices[idx][0][selected],
                            indices[idx][1][selected],
                        )
                # print("ind shapes.", [i[0].shape for i in indices])

            if self.ss_warmup_epochs <= self.current_epoch:
                queries, tgt_idxs, src_idxs = self.get_matched_latents(output, indices)

                sdf_points = data.sdf_queries
                grasp_points_object = data.grasp_points_object
                grasp_points_scene = data.grasp_points_scene

                normal_pts = None
                normal_vec = None
                normal_id = None
                if data.normals is not None:
                    normal_pts = [p["pts"] for p in data.normals]
                    normal_vec = [p["n_normals"].to(self.device) for p in data.normals]
                    normal_id = [p["n_ids"].to(self.device) for p in data.normals]

                if isinstance(sdf_points, torch.Tensor):
                    sdf_points = list(sdf_points)
                if isinstance(grasp_points_object, torch.Tensor):
                    grasp_points_object = list(grasp_points_object)
                if isinstance(grasp_points_scene, torch.Tensor):
                    grasp_points_scene = list(grasp_points_scene)

                sdf_points = (
                    [p.to(self.device) for p in sdf_points]
                    if sdf_points is not None
                    else None
                )
                grasp_points_object = (
                    [p.to(self.device) for p in grasp_points_object]
                    if grasp_points_object is not None
                    else None
                )
                grasp_points_scene = (
                    [p.to(self.device) for p in grasp_points_scene]
                    if grasp_points_scene is not None
                    else None
                )

                normal_pts = (
                    [p.to(self.device) for p in normal_pts]
                    if normal_pts is not None
                    else None
                )

                (
                    sdf_values,
                    grasp_values_object,
                    grasp_values_scene,
                    normal_pred,
                ) = self.get_query_predictions(
                    queries,
                    sdf_points,
                    grasp_points_object,
                    grasp_points_scene,
                    normal_pts=normal_pts,
                )

                if sdf_values is not None:
                    output["sdf_pred"] = sdf_values[-1]

                    for idx in range(len(sdf_values) - 1):
                        output["aux_outputs"][idx]["sdf_pred"] = sdf_values[idx]

                if grasp_points_object is not None:
                    output["object_grasps"] = grasp_values_object[-1]

                    for idx in range(len(grasp_values_object) - 1):
                        output["aux_outputs"][idx][
                            "object_grasps"
                        ] = grasp_values_object[idx]

                if grasp_points_scene is not None:
                    output["scene_grasps"] = grasp_values_scene[-1]

                    for idx in range(len(grasp_values_scene) - 1):
                        output["aux_outputs"][idx]["scene_grasps"] = grasp_values_scene[
                            idx
                        ]

                if normal_pred is not None and len(normal_pred) > 0:
                    output["normals"] = normal_pred[-1]
                    output["gt_normals"] = normal_vec
                    output["gt_normals_id"] = normal_id

                    for idx in range(len(normal_pred) - 1):
                        output["aux_outputs"][idx]["normals"] = normal_pred[idx]
                        output["aux_outputs"][idx]["gt_normals"] = normal_vec
                        output["aux_outputs"][idx]["gt_normals_id"] = normal_id

        except RuntimeError as run_err:
            print(run_err)
            if "only a single point gives nans in cross-attention" == run_err.args[0]:
                raise run_err
            else:
                raise run_err

        try:
            with self.timer["loss_calculation"]:
                losses = self.criterion(
                    output, target, mask_type="masks", indices=indices
                )

        except ValueError as val_err:
            print(f"ValueError: {val_err}")
            print(f"data shape: {data_train.shape}")
            print(f"data feat shape:  {data_train.features.shape}")
            print(f"data feat nans:   {data_train.features.isnan().sum()}")
            print(f"output: {output}")
            print(f"target: {target}")
            print(f"filenames: {file_names}")
            raise val_err

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                print("Dropping", k)
                losses.pop(k)

        logs = {f"train_{k}": v.detach().cpu().item() for k, v in losses.items()}
        self.log_dict(logs)

        if self.current_epoch % self.config["general"]["eval_train_rate"] == (
            self.config["general"]["eval_train_rate"] - 1
        ):
            print("Evaluating for training data")
            try:
                self.eval_metrics(
                    target,
                    raw_coordinates,
                    data,
                    output,
                    tgt_idxs,
                    src_idxs,
                    batch_idx,
                    file_names=file_names,
                    mode="train",
                )
            except Exception as e:
                print(traceback.format_exc())
                print("Failed to evaluate training data", str(e))
                if len(self.visualization_data) > 0:
                    self.logger.experiment.log(self.visualization_data)

        if self.model.grasp_decoder_type == "acronym":
            print("clearing cache....")
            torch.cuda.empty_cache()

        return sum(losses.values()) / len(losses.values())

    def validation_step(self, batch, batch_idx):
        # ugly
        # if batch_idx == 0:
        #     print("Calling test step for first validation batch")
        #     dl = self.test_dataloader()
        #     for idx, entry in enumerate(dl):
        #         print(f"TestStep {idx}/{len(dl)}")

        #         self.test_step(entry, idx)
        return self.eval_step(batch, batch_idx)

    def on_train_epoch_end(self, *args, **kwargs):
        # print("training epoch end")
        # train_loss = sum([out["loss"].cpu().item() for out in outputs]) / len(outputs)
        # results = {"train_loss_mean": train_loss}

        if self.current_epoch % self.config["general"]["eval_train_rate"] == (
            self.config["general"]["eval_train_rate"] - 1
        ):
            metrics = self.occ_metrics.get_metrics("train/recon")
            metrics = {
                **metrics,
                **self.obj_grasp_metrics.get_metrics("train/obj_grasp"),
            }
            metrics = {
                **metrics,
                **self.obj_grasp_metrics.get_metrics("train/obj_grasp"),
            }
        
        self.occ_metrics.reset()
        self.obj_grasp_metrics.reset()
        self.scene_grasp_metrics.reset()
        self.instance_metrics.reset()

        print("Training epoch end. Resetting metrics and empty cuda cache")
        torch.cuda.empty_cache()
        # self.log_dict(results)

    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        self.timer.print_stats()

        metrics = self.occ_metrics.get_metrics("val/recon")
        metrics = {**metrics, **self.obj_grasp_metrics.get_metrics("val/obj_grasp")}
        metrics = {**metrics, **self.scene_grasp_metrics.get_metrics("val/scene_grasp")}
        metrics = {**metrics, **self.instance_metrics.get_metrics("val/inst_segment")}
        # metrics = {**metrics, **self.full_scene_grasp_metrics.get_metrics("val/full_scene_grasp")}
        metrics = {**metrics, **self.full_occ_metrics.get_metrics("val/full_occ")}


        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=False)

        self.occ_metrics.reset()
        self.obj_grasp_metrics.reset()
        self.scene_grasp_metrics.reset()
        self.instance_metrics.reset()
        # self.full_scene_grasp_metrics.reset()
        self.full_occ_metrics.reset()
        print("Validation epoch end. Resetting metrics")
        # print(metrics)
        return metrics

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        data, target, file_names, transforms = batch

        if data.features.shape[0] > self.config.general.max_batch_size:
            raise RuntimeError("BATCH TOO BIG")

        raw_coordinates = data.features[:, -3:]
        data.features = data.features[:, :-3]

        data_train = ME.SparseTensor(
            coordinates=data.coordinates, features=data.features, device=self.device
        )

        try:
            output = self.forward(data_train, raw_coordinates=raw_coordinates)

            # These latents did not get assigned the "no object class"
            valid_latent_ids = (
                output["pred_logits"].argmax(-1) != (output["pred_logits"].size(-1) - 1)
            ).squeeze(0)
            predicted_masks = output["pred_masks"][0]
            predicted_masks[..., ~valid_latent_ids] = -10000
            predicted_instances = predicted_masks.argmax(dim=-1)
            valid_ids = predicted_instances.unique()
            valid_latent_ids[:] = False
            valid_latent_ids[predicted_instances.unique()] = True

            # Extract Latents
            queries, tgt_idxs, src_idxs = self.get_matched_latents(
                output, [(valid_ids, valid_ids)]
            )

            sdf_points = data.sdf_queries
            grasp_points_object = data.grasp_points_object
            grasp_points_scene = data.grasp_points_scene
            if isinstance(sdf_points, torch.Tensor):
                sdf_points = list(sdf_points)
            if isinstance(grasp_points_object, torch.Tensor):
                grasp_points_object = list(grasp_points_object)
            if isinstance(grasp_points_scene, torch.Tensor):
                grasp_points_scene = list(grasp_points_scene)

            sdf_points = (
                [p.to(self.device) for p in sdf_points]
                if sdf_points is not None
                else None
            )
            grasp_points_object = (
                [p.to(self.device) for p in grasp_points_object]
                if grasp_points_object is not None
                else None
            )
            grasp_points_scene = (
                [p.to(self.device) for p in grasp_points_scene]
                if grasp_points_scene is not None
                else None
            )

            d = self.get_query_predictions(
                queries,
                sdf_points,
                grasp_points_object,
                grasp_points_scene,
                [raw_coordinates.to(self.device)],
            )
            if len(d) < 4:
                d = d + [None]

            (
                sdf_values,
                grasp_values_object,
                grasp_values_scene,
                additional_scene_grasp,
            ) = d

            if sdf_values is not None:
                output["sdf_pred"] = sdf_values[-1]

                for idx in range(len(sdf_values) - 1):
                    output["aux_outputs"][idx]["sdf_pred"] = sdf_values[idx]

                if self.model.occ_decoder_type == "point2surf":
                    output["sdf_pred"] = [
                        s[..., 0][..., None] for s in output["sdf_pred"]
                    ]

                    for idx in range(len(sdf_values) - 1):
                        output["aux_outputs"][idx]["sdf_pred"] = [
                            s[..., 0][..., None]
                            for s in output["aux_outputs"][idx]["sdf_pred"]
                        ]

            if grasp_points_object is not None and len(grasp_points_object) > 0:
                output["object_grasps"] = grasp_values_object[-1]

                for idx in range(len(grasp_values_object) - 1):
                    output["aux_outputs"][idx]["object_grasps"] = grasp_values_object[
                        idx
                    ]

            if grasp_points_scene is not None:
                output["scene_grasps"] = grasp_values_scene[-1]

                for idx in range(len(grasp_values_scene) - 1):
                    output["aux_outputs"][idx]["scene_grasps"] = grasp_values_scene[idx]

            if additional_scene_grasp is not None:
                additional_scene_grasp = additional_scene_grasp[
                    -1
                ]  # only take the last one with refined scene grasps
                output["additional_scene_grasp"] = additional_scene_grasp

        except RuntimeError as run_err:
            print(run_err)
            if "only a single point gives nans in cross-attention" == run_err.args[0]:
                return None
            else:
                raise run_err

        try:
            self.show_test(
                raw_coordinates,
                data,
                output,
                src_idxs,
                batch_idx,
            )
        except Exception as e:
            print(traceback.format_exc())
            print("Failed to evaluate validation data", str(e))
            if len(self.visualization_data) > 0:
                self.logger.experiment.log(self.visualization_data)

    @torch.no_grad()
    def eval_step(self, batch, batch_idx, mode="val", forward_args=None):
        data, target, file_names, transforms = batch

        if len(target) == 0:
            print("no targets")
            return None

        if data.features.shape[0] > self.config.general.max_batch_size:
            raise RuntimeError("BATCH TOO BIG")

        raw_coordinates = data.features[:, -3:]
        data.features = data.features[:, :-3]

        data_train = ME.SparseTensor(
            coordinates=data.coordinates, features=data.features, device=self.device
        )

        try:
            output = self.forward(
                data_train,
                raw_coordinates=raw_coordinates,
                poses=data.poses,
            )

            # get matchings using hung matcher
            indices = self.criterion.get_matches(output, target, mask_type="masks")

            # valid_objects = []
            # for idx, (pred_idx, gt_idx) in enumerate(indices):
            #     valid = target[idx]["labels"] == 0
            #     valid = valid.to(pred_idx.device)
            #     valid_objects.append(valid)

            queries, tgt_idxs, src_idxs = self.get_matched_latents(
                output, indices, valid_objects=None
            )

            sdf_points = data.sdf_queries
            grasp_points_object = data.grasp_points_object
            grasp_points_scene = data.grasp_points_scene
            if isinstance(sdf_points, torch.Tensor):
                sdf_points = list(sdf_points)
            if isinstance(grasp_points_object, torch.Tensor):
                grasp_points_object = list(grasp_points_object)
            if isinstance(grasp_points_scene, torch.Tensor):
                grasp_points_scene = list(grasp_points_scene)

            sdf_points = (
                [p.to(self.device) for p in sdf_points]
                if sdf_points is not None
                else None
            )
            grasp_points_object = (
                [p.to(self.device) for p in grasp_points_object]
                if grasp_points_object is not None
                else None
            )
            grasp_points_scene = (
                [p.to(self.device) for p in grasp_points_scene]
                if grasp_points_scene is not None
                else None
            )

            d = self.get_query_predictions(
                queries,
                sdf_points,
                grasp_points_object,
                grasp_points_scene,
                [raw_coordinates.to(self.device)]
                if (mode != "train" and "occnet" in self.model.grasp_decoder_type)
                else None,
            )
            if len(d) < 4:
                d = d + [None]

            (
                sdf_values,
                grasp_values_object,
                grasp_values_scene,
                additional_scene_grasp,
            ) = d

            if sdf_values is not None:
                output["sdf_pred"] = sdf_values[-1]

                for idx in range(len(sdf_values) - 1):
                    output["aux_outputs"][idx]["sdf_pred"] = sdf_values[idx]

                if self.model.occ_decoder_type == "point2surf":
                    output["sdf_pred"] = [
                        s[..., 0][..., None] for s in output["sdf_pred"]
                    ]

                    for idx in range(len(sdf_values) - 1):
                        output["aux_outputs"][idx]["sdf_pred"] = [
                            s[..., 0][..., None]
                            for s in output["aux_outputs"][idx]["sdf_pred"]
                        ]

            if grasp_points_object is not None and len(grasp_points_object) > 0:
                output["object_grasps"] = grasp_values_object[-1]

                for idx in range(len(grasp_values_object) - 1):
                    output["aux_outputs"][idx]["object_grasps"] = grasp_values_object[
                        idx
                    ]

            if grasp_points_scene is not None:
                output["scene_grasps"] = grasp_values_scene[-1]

                for idx in range(len(grasp_values_scene) - 1):
                    output["aux_outputs"][idx]["scene_grasps"] = grasp_values_scene[idx]

            if additional_scene_grasp is not None:
                additional_scene_grasp = additional_scene_grasp[
                    -1
                ]  # only take the last one with refined scene grasps
                output["additional_scene_grasp"] = additional_scene_grasp

        except RuntimeError as run_err:
            print(run_err)
            if "only a single point gives nans in cross-attention" == run_err.args[0]:
                return None
            else:
                raise run_err

        try:
            losses = self.criterion(output, target, mask_type="masks", indices=indices)

        except ValueError as val_err:
            print(f"ValueError: {val_err}")
            print(f"data shape: {data_train.shape}")
            print(f"data feat shape:  {data_train.features.shape}")
            print(f"data feat nans:   {data_train.features.isnan().sum()}")
            print(f"output: {output}")
            print(f"target: {target}")
            print(f"filenames: {file_names}")
            raise val_err

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                print("dropping", k)
                losses.pop(k)
        logs = {}
        for k, v in losses.items():
            logs[f"{mode}_{k}"] = v.detach().cpu().item()
        self.log_dict(logs, batch_size=len(target))
        try:
            self.eval_metrics(
                target,
                raw_coordinates,
                data,
                output,
                tgt_idxs,
                src_idxs,
                batch_idx,
                file_names=file_names,
            )
        except Exception as e:
            print(traceback.format_exc())
            print("Failed to evaluate validation data", str(e))
            if len(self.visualization_data) > 0:
                self.logger.experiment.log(self.visualization_data)

    def eval_metrics(
        self,
        target,
        raw_coordinates,
        data,
        output,
        tgt_idxs,
        src_idxs,
        batch_iter_idx,
        file_names,
        mode="val",
    ):
        # Calculate metrics

        self.visualization_data = {}
        visualization_data = self.visualization_data

        obj_files = [
            os.path.join(
                os.path.dirname(f),
                "obj_scenes",
                os.path.basename(f).replace(".npz", ".obj").replace(".npy", ".obj"),
            )
            for f in file_names
        ]

        for batch_idx in range(len(target)):
            world_coordinates = raw_coordinates[data.coordinates[:, 0] == batch_idx]


            # Instance Metrics

            instance_lbl = target[batch_idx]["masks"][tgt_idxs].float().argmax(dim=0)
            latent_idxs = src_idxs[batch_idx]
            instance_pred = (
                output["pred_masks"][batch_idx].float()[:, latent_idxs].argmax(-1)
            )
            self.instance_metrics(instance_pred, instance_lbl)


            if "object_centric_labels" in target[batch_idx]:
                # object centric labels
                tgt = target[batch_idx]["object_centric_labels"].clone()
                pred = output["object_grasps"][batch_idx].clone()
                if min(tgt.shape) != 0 and min(pred.shape) != 0:
                    self.obj_grasp_metrics(
                        tgt, pred, decoder_type=self.model.grasp_decoder_type
                    )
            if "scene_centric_labels" in target[batch_idx]:
                # scene centric labels
                tgt = target[batch_idx]["scene_centric_labels"].clone()
                pred = output["scene_grasps"][batch_idx].clone()
                if min(tgt.shape) != 0 and min(pred.shape) != 0:
                    self.scene_grasp_metrics(tgt, pred, decoder_type=self.model.grasp_decoder_type)
                    # import pdb; pdb.set_trace()
                    # self.full_scene_grasp_metrics(pred.max(0)[0].unsqueeze(0),tgt.max(0)[0].unsqueeze(0), decoder_type=self.model.grasp_decoder_type)

            if "sdf" in target[batch_idx]:
                tgt = (target[batch_idx]["sdf"] > 0).long().squeeze().clone()
                pred = output["sdf_pred"][batch_idx].squeeze().clone()
                if min(tgt.shape) != 0 and min(pred.shape) != 0:
                    self.occ_metrics(
                        tgt,
                        pred,
                    )
                    self.full_occ_metrics(
                        tgt.max(0)[0],
                        pred.max(0)[0]
                    )


            if (
                batch_idx >= self.vis_per_batch
                or batch_iter_idx >= self.num_batch_to_vis
            ):
                continue

            gt_vis = PointCloudVisualizer()
            pred_vis = PointCloudVisualizer()
            sem_gt_vis = PointCloudVisualizer()
            sem_pred_vis = PointCloudVisualizer()

            if not os.path.exists(
                f"{self.config['general']['save_dir']}/visualizations/"
            ):
                os.makedirs(f"{self.config['general']['save_dir']}/visualizations/")

            if self.show_semantic and mode == "val":
                instance_lbl = target[batch_idx]["masks"].float().argmax(dim=0)

                semantic_lbl = target[batch_idx]["labels"][instance_lbl]
                gt_vis.add_classified_scan(world_coordinates, instance_lbl)
                sem_gt_vis.add_classified_scan(world_coordinates, semantic_lbl)

                latent_idxs = src_idxs[batch_idx]
                instance_pred = (
                    output["pred_masks"][batch_idx].float()[:, latent_idxs].argmax(-1)
                )

                
                semantic_pred = output["pred_logits"][batch_idx].argmax(-1)[latent_idxs][instance_pred]
                sem_pred_vis.add_classified_scan(world_coordinates, semantic_pred)

                
                pred_vis.add_classified_scan(world_coordinates, instance_pred)

                scan = gt_vis.get_scan_cloud_wandb()
                if scan is not None:
                    name = f"semseg/batch_{batch_iter_idx}_gt"
                    if name not in visualization_data:
                        visualization_data[name] = []
                    visualization_data[name].append(scan)

                scan = pred_vis.get_scan_cloud_wandb()
                if scan is not None:
                    name = f"semseg/batch_{batch_iter_idx}_pred"
                    if name not in visualization_data:
                        visualization_data[name] = []
                    visualization_data[name].append(scan)

                scan = sem_gt_vis.get_scan_cloud_wandb()
                if scan is not None:
                    name = f"classficiation/batch_{batch_iter_idx}_gt"
                    if name not in visualization_data:
                        visualization_data[name] = []
                    visualization_data[name].append(scan)

                scan = sem_pred_vis.get_scan_cloud_wandb()
                if scan is not None:
                    name = f"classficiation/batch_{batch_iter_idx}_pred"
                    if name not in visualization_data:
                        visualization_data[name] = []
                    visualization_data[name].append(scan)




            if self.show_occ and mode == "val":
                # Ground truth
                sdf_values = (target[batch_idx]["sdf"] > 0).float()
                gt_vis.add_occupancy_cloud(data.sdf_queries[batch_idx], sdf_values)

                # Prediction
                pred_vis.add_occupancy_cloud(
                    data.sdf_queries[batch_idx],
                    (output["sdf_pred"][batch_idx].sigmoid() > 0.5).squeeze(-1),
                )

                occ_gt = gt_vis.get_occ_cloud_wandb()
                if occ_gt is not None:
                    name = f"occupancy/batch_{batch_iter_idx}_gt"
                    if name not in visualization_data:
                        visualization_data[name] = []
                    visualization_data[name].append(occ_gt)

                occ_pred = pred_vis.get_occ_cloud_wandb()
                if occ_pred is not None:
                    name = f"occupancy/batch_{batch_iter_idx}_pred"
                    if name not in visualization_data:
                        visualization_data[name] = []
                    visualization_data[name].append(occ_pred)

                from icg_net.utils.mcubes.mesh_extractor import Generator3D

                idxs = src_idxs[-1]
                all_latents = output["all_queries"][-1][0][idxs]
                all_pos_enc = output["positional_encodings"][0][idxs]
                meshes = []
                for idx, (latent, pos_enc) in enumerate(zip(all_latents, all_pos_enc)):
                    print("Extracting mesh for instance", idx, "with query")

                    @torch.no_grad()
                    def get_sdf_value(pts: torch.Tensor, idx=idx):
                        occ_values = self.model.decode_sdf(
                            [all_latents],
                            query_positional_encoding=[all_pos_enc],
                            sdf_points=list(pts.cuda()),
                        )[0]
                        if self.model.occ_decoder_type == "point2surf":
                            occ_values = [o[..., 0] for o in occ_values]

                        # import pdb; pdb.set_trace()
                        return occ_values[idx].cpu()

                    gen = Generator3D(
                        get_sdf_value,
                        resolution0=64,  # 128, #32,
                        points_batch_size=4096 * 4,
                        scale=1,
                        translation=0,
                    )
                    mesh = gen.generate_mesh(
                        volume_bounds=np.array(
                            [
                                (self.model.scene_bounds[0][0].cpu().numpy() - 0.07)
                                .squeeze()
                                .tolist(),
                                (self.model.scene_bounds[1][0].cpu().numpy() + 0.07)
                                .squeeze()
                                .tolist(),
                            ]
                        )
                    )
                    meshes.append(mesh)
                scene = trimesh.Scene()
                for m in meshes:
                    scene.add_geometry(m)

                if self.local_vis:
                    scene.show()

                import wandb

                try:
                    os.makedirs(
                        f"{self.config['general']['save_dir']}/reconstructions/",
                        exist_ok=True,
                    )
                    scene.export(
                        f"{self.config['general']['save_dir']}/reconstructions/occ_{batch_iter_idx}_{batch_idx}_pred.obj"
                    )

                    # scene_wandb = wandb.Object3D(
                    #     f"{self.config['general']['save_dir']}/reconstructions/occ_{batch_iter_idx}_{batch_idx}_pred.obj"
                    # )
                    # name = f"val/mesh/batch_{batch_iter_idx}_pred"
                    # if name not in visualization_data:
                    #     visualization_data[name] = []
                    # visualization_data[name].append(scene_wandb)

                except Exception as e:
                    print("Error exporting mesh for idx", batch_iter_idx, e)

            # Scene grasps
            if self.show_scene_grasps and mode == "val" and "scene_grasps" in output:
                obj_scene = obj_files[batch_idx]

                grasp_gt_vis = GraspVisualizer(num_grasps=50)

                grasp_pts_scene = data.grasp_points_scene[batch_idx]
                if min(grasp_pts_scene.shape) == 0:
                    continue
                grasp_lbl = target[batch_idx]["scene_centric_labels"]

                grasp_gt_vis.add_grasps(
                    grasp_pts_scene,
                    grasp_lbl,
                    grasp_type=self.model.grasp_decoder_type,
                    obj_scene=obj_scene,
                )
                if self.debug_view_ds:
                    grasp_gt_vis_inv = GraspVisualizer(num_grasps=50)
                    # Invalid ones
                    grasp_gt_vis_inv.add_grasps(
                        grasp_pts_scene,
                        grasp_lbl,
                        grasp_type=self.model.grasp_decoder_type,
                        obj_scene=obj_scene,
                    )
                    import open3d as o3d

                    print("Visualizing scene grasps")
                    o3d.visualization.draw(
                        grasp_gt_vis.show(True, 0) + grasp_gt_vis_inv.show(True, 1)
                    )

                grasp_pred_vis = GraspVisualizer()

                grasp_lbl = output["scene_grasps"][batch_idx].squeeze(-1)

                grasp_pred_vis.add_grasps(
                    grasp_pts_scene,
                    grasp_lbl,
                    grasp_type=self.model.grasp_decoder_type,
                    # mask = target[batch_idx]["scene_centric_labels"][..., 0] == 1,
                    obj_scene=obj_scene,
                )

                # grasp_gt_vis.show()
                # grasp_pred_vis.show()
                # grasp_gt_vis.show()
                # import open3d as o3d
                # import pdb; pdb.set_trace()
                # grasp_gt_vis.show()

                # o3d.visualization.draw(grasp_gt_vis.show(True, 0) + grasp_pred_vis.show(True, 1))
                # Visualize
                pred_wandb = grasp_pred_vis.get_grasp_mesh()
                gt_wandb = grasp_gt_vis.get_grasp_mesh()

                grasp_gt_vis.add_scan(world_coordinates)
                instance_assignement = grasp_gt_vis.get_instance_cloud_wandb()
                # if self.debug_view_ds:
                #     import open3d as o3d
                #     print("Visualizing scene grasps")
                #     o3d.visualization.draw([*grasp_pred_vis.show(True, 0), *grasp_gt_vis.show(True, 1)])

                if instance_assignement is not None:
                    name = f"scene_grasps_assignement/batch_{batch_iter_idx}"
                    if name not in visualization_data:
                        visualization_data[name] = []
                    visualization_data[name].append(instance_assignement)

                if "additional_scene_grasp" in output:
                    pred_full_scene = output["additional_scene_grasp"][batch_idx]
                    full_scene_vis = GraspVisualizer(num_grasps=50)
                    full_scene_vis.add_grasps(
                        world_coordinates.to(self.device),
                        pred_full_scene.squeeze(-1),
                        grasp_type=self.model.grasp_decoder_type,
                        obj_scene=obj_scene,
                    )
                    aff_cloud = full_scene_vis.get_qual_cloud_wandb()

                    name = f"full_scene_grasp_eval/batch_{batch_iter_idx}"
                    if name not in visualization_data:
                        visualization_data[name] = []
                    visualization_data[name].append(aff_cloud)

                if pred_wandb is not None and gt_wandb is not None:
                    import wandb

                    pred_wandb.export(
                        f"{self.config['general']['save_dir']}/visualizations/grasp_{batch_iter_idx}_{batch_idx}_pred.obj"
                    )
                    gt_wandb.export(
                        f"{self.config['general']['save_dir']}/visualizations/grasp_{batch_iter_idx}_{batch_idx}_gt.obj"
                    )
                    # pred_wandb = wandb.Object3D(
                    #     f"{self.config['general']['save_dir']}/visualizations/grasp_{batch_iter_idx}_{batch_idx}_pred.obj"
                    # )
                    # gt_wandb = wandb.Object3D(
                    #     f"{self.config['general']['save_dir']}/visualizations/grasp_{batch_iter_idx}_{batch_idx}_gt.obj"
                    # )

                    # name = f"scene_grasps/batch_{batch_iter_idx}_pred"
                    # if name not in visualization_data:
                    #     visualization_data[name] = []
                    # visualization_data[name].append(pred_wandb)
                    # name = f"scene_grasps/batch_{batch_iter_idx}_gt"
                    # if name not in visualization_data:
                    #     visualization_data[name] = []
                    # visualization_data[name].append(gt_wandb)

            # Object grasps
            if self.show_object_grasps and mode == "val" and "object_grasps" in output:
                import wandb

                grasp_points_object = data.grasp_points_object[batch_idx]

                if min(grasp_points_object.shape) == 0:
                    continue

                obj_scene = obj_files[batch_idx]

                grasp_gt_vis = GraspVisualizer()

                grasp_lbl = target[batch_idx]["object_centric_labels"]
                grasp_gt_vis.add_grasps(
                    grasp_points_object,
                    grasp_lbl,
                    grasp_type=self.model.grasp_decoder_type,
                    obj_scene=obj_scene,
                )

                if self.debug_view_ds:
                    print("Showing Object Grasps")
                    grasp_gt_vis_inv = GraspVisualizer(num_grasps=50)
                    # Invalid ones
                    grasp_gt_vis_inv.add_grasps(
                        grasp_points_object,
                        grasp_lbl,
                        grasp_type=self.model.grasp_decoder_type,
                        obj_scene=obj_scene,
                    )
                    import open3d as o3d

                    o3d.visualization.draw(
                        grasp_gt_vis.show(True, 0) + grasp_gt_vis_inv.show(True, 1)
                    )

                grasp_pred_vis = GraspVisualizer()

                grasp_lbl = output["object_grasps"][batch_idx].squeeze(-1)
                grasp_pred_vis.add_grasps(
                    grasp_points_object,
                    grasp_lbl,
                    grasp_type=self.model.grasp_decoder_type,
                    obj_scene=obj_scene,
                )

                # Visualize
                pred_wandb = grasp_pred_vis.get_grasp_mesh()
                gt_wandb = grasp_gt_vis.get_grasp_mesh()

                grasp_gt_vis.add_scan(world_coordinates)
                instance_assignement = grasp_gt_vis.get_instance_cloud_wandb()
                if instance_assignement is not None:
                    name = f"obj_grasps_assignement/batch_{batch_iter_idx}"
                    if name not in visualization_data:
                        visualization_data[name] = []
                    visualization_data[name].append(instance_assignement)

                if pred_wandb is not None and gt_wandb is not None:
                    pred_wandb.export(
                        f"{self.config['general']['save_dir']}/visualizations/grasp_obj_{batch_iter_idx}_{batch_idx}_pred.obj"
                    )
                    gt_wandb.export(
                        f"{self.config['general']['save_dir']}/visualizations/grasp_obj_{batch_iter_idx}_{batch_idx}_gt.obj"
                    )
                    
                    # pred_wandb = wandb.Object3D(
                    #     f"{self.config['general']['save_dir']}/visualizations/grasp_obj_{batch_iter_idx}_{batch_idx}_pred.obj"
                    # )
                    # gt_wandb = wandb.Object3D(
                    #     f"{self.config['general']['save_dir']}/visualizations/grasp_obj_{batch_iter_idx}_{batch_idx}_gt.obj"
                    # )

                    # name = f"obj_grasps/batch_{batch_iter_idx}_pred"
                    # if name not in visualization_data:
                    #     visualization_data[name] = []
                    # visualization_data[name].append(pred_wandb)
                    # name = f"obj_grasps/batch_{batch_iter_idx}_gt"
                    # if name not in visualization_data:
                    #     visualization_data[name] = []
                    # visualization_data[name].append(gt_wandb)

        self.logger.experiment.log(visualization_data)

    def show_test(
        self,
        raw_coordinates,
        data,
        output,
        src_idxs,
        batch_iter_idx,
    ):
        # Calculate metrics

        self.visualization_data = {}
        visualization_data = self.visualization_data

        for batch_idx in range(1):
            world_coordinates = raw_coordinates[data.coordinates[:, 0] == batch_idx]

            pred_vis = PointCloudVisualizer()
            if "additional_scene_grasp" in output:
                pred_full_scene = output["additional_scene_grasp"][batch_idx]
                full_scene_vis = GraspVisualizer(num_grasps=50)
                full_scene_vis.add_grasps(
                    world_coordinates.to(self.device),
                    pred_full_scene.squeeze(-1),
                    grasp_type=self.model.grasp_decoder_type,
                    obj_scene=None, 
                )
                aff_cloud = full_scene_vis.get_qual_cloud_wandb()

                name = f"test_full_scene_grasp_eval/batch_{batch_iter_idx}"
                if name not in visualization_data:
                    visualization_data[name] = []
                visualization_data[name].append(aff_cloud)
                
            if self.show_semantic:
                latent_idxs = src_idxs[batch_idx]
                instance_pred = (
                    output["pred_masks"][batch_idx].float()[:, latent_idxs].argmax(-1)
                )
                # downsample both clouds
                num_pts = 4096
                if instance_pred.shape[0] > num_pts:
                    idxs = torch.randperm(instance_pred.shape[0])[:num_pts]
                    instance_pred = instance_pred[idxs]
                    world_coordinates = world_coordinates[idxs]
                pred_vis.add_classified_scan(world_coordinates, instance_pred)

                scan = pred_vis.get_scan_cloud_wandb()
                if scan is not None:
                    name = f"test_semseg/batch_{batch_iter_idx}_pred"
                    if name not in visualization_data:
                        visualization_data[name] = []
                    visualization_data[name].append(scan)
            
            if self.show_occ:
                # Prediction
                pred_vis.add_occupancy_cloud(
                    data.sdf_queries[batch_idx],
                    (output["sdf_pred"][batch_idx].sigmoid() > 0.5).squeeze(-1),
                )

                occ_pred = pred_vis.get_occ_cloud_wandb()
                if occ_pred is not None:
                    name = f"test_occupancy/batch_{batch_iter_idx}_pred"
                    if name not in visualization_data:
                        visualization_data[name] = []
                    visualization_data[name].append(occ_pred)

                from icg_net.utils.mcubes.mesh_extractor import Generator3D

                idxs = src_idxs[-1]
                all_latents = output["all_queries"][-1][0][idxs]
                all_pos_enc = output["positional_encodings"][0][idxs]
                meshes = []
                for idx, (latent, pos_enc) in enumerate(zip(all_latents, all_pos_enc)):
                    # print("Extracting mesh for instance", idx, "with query")

                    @torch.no_grad()
                    def get_sdf_value(pts: torch.Tensor, idx=idx):
                        occ_values = self.model.decode_sdf(
                            [all_latents],
                            query_positional_encoding=[all_pos_enc],
                            sdf_points=list(pts.cuda()),
                        )[0]
                        if self.model.occ_decoder_type == "point2surf":
                            occ_values = [o[..., 0] for o in occ_values]

                        # import pdb; pdb.set_trace()
                        return occ_values[idx].cpu()

                    gen = Generator3D(
                        get_sdf_value,
                        resolution0=64,  # 128, #32,
                        points_batch_size=4096 * 4,
                        scale=1,
                        translation=0,
                    )
                    mesh = gen.generate_mesh(
                        volume_bounds=np.array(
                            [
                                (self.model.scene_bounds[0][0].cpu().numpy() - 0.07)
                                .squeeze()
                                .tolist(),
                                (self.model.scene_bounds[1][0].cpu().numpy() + 0.07)
                                .squeeze()
                                .tolist(),
                            ]
                        )
                    )
                    meshes.append(mesh)
                scene = trimesh.Scene()
                for m in meshes:
                    scene.add_geometry(m)

                import wandb

                try:
                    os.makedirs(
                        f"{self.config['general']['save_dir']}/reconstructions/",
                        exist_ok=True,
                    )
                    scene.export(
                        f"{self.config['general']['save_dir']}/reconstructions/test_occ_{batch_iter_idx}_{batch_idx}_pred.obj"
                    )

                    scene_wandb = wandb.Object3D(
                        f"{self.config['general']['save_dir']}/reconstructions/test_occ_{batch_iter_idx}_{batch_idx}_pred.obj"
                    )
                    name = f"val/test_mesh/batch_{batch_iter_idx}_pred"
                    if name not in visualization_data:
                        visualization_data[name] = []
                    visualization_data[name].append(scene_wandb)

                except Exception as e:
                    print("Error exporting mesh for idx", batch_iter_idx, e)

        self.logger.experiment.log(visualization_data)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.config.optimizer, params=self.parameters()
        )
        if "steps_per_epoch" in self.config.scheduler.scheduler.keys():
            self.config.scheduler.scheduler.steps_per_epoch = len(
                self.train_dataloader()
            )
        lr_scheduler = hydra.utils.instantiate(
            self.config.scheduler.scheduler, optimizer=optimizer
        )
        scheduler_config = {"scheduler": lr_scheduler}
        scheduler_config.update(self.config.scheduler.pytorch_lightning_params)
        return [optimizer], [scheduler_config]

    def prepare_data(self):
        self.train_dataset = hydra.utils.instantiate(self.config.data.train_dataset)
        self.validation_dataset = hydra.utils.instantiate(
            self.config.data.validation_dataset
        )
        self.test_dataset = hydra.utils.instantiate(self.config.data.test_dataset)
        # self.labels_info = self.train_dataset.label_info

    def train_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.train_collation)
        print("train loadedr stup")
        return hydra.utils.instantiate(
            self.config.data.train_dataloader,
            self.train_dataset,
            collate_fn=c_fn,
        )

    def val_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        ds = hydra.utils.instantiate(
            self.config.data.validation_dataloader,
            self.validation_dataset,
            collate_fn=c_fn,
        )
        return ds

    def test_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.test_collation)
        return hydra.utils.instantiate(
            self.config.data.test_dataloader,
            self.test_dataset,
            collate_fn=c_fn,
        )
