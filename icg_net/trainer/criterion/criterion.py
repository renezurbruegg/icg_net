# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified for ICGNet
"""
MaskFormer criterion.
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn.functional as F
from torch import nn


@torch.jit.script
def soft_dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,  # ignored. Here for compatibility reasons
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numer = (inputs * targets).sum()
    denor = (inputs + targets).sum()
    loss = 1.0 - (2 * numer) / (denor + 0.01)
    return loss


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


@torch.jit.export
def sym_quat_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    quat_xyzw = targets
    loss1 = 1 - (inputs * quat_xyzw).sum(dim=-1).abs()
    quat_rot = torch.stack(
        (quat_xyzw[..., 1], -quat_xyzw[..., 0], quat_xyzw[..., 3], -quat_xyzw[..., 2]),
        -1,
    )
    loss2 = 1 - (inputs * quat_rot).sum(dim=-1).abs()
    loss = torch.min(loss1, loss2).mean() / num_masks
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


def sigmoid_ce_loss_mean(
    inputs: torch.Tensor,
    targets: torch.Tensor,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")
    return loss.mean()


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule
sigmoid_ce_loss_jit_mean = torch.jit.script(sigmoid_ce_loss_mean)  # type: torch.jit.ScriptModule


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        num_points,
        class_weights,
        sdf_loss_type,
        sdf_clip,
        grasp_loss_type=None,
        exclude_inter=[],
        regularizer=0.0,
        exclude_sdf_class=[]
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes - 1
        self.class_weights = class_weights
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.sdf_clip = sdf_clip
        self.losses = losses
        self.sdf_loss_type = sdf_loss_type
        self.grasp_loss_type = grasp_loss_type
        self.exclude_inter = exclude_inter
        self.regularizer = regularizer
        self.exclude_sdf_class = exclude_sdf_class

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef

        if self.class_weights != -1:
            assert len(self.class_weights) == self.num_classes, "CLASS WEIGHTS DO NOT MATCH"
            empty_weight[:-1] = torch.tensor(self.class_weights)

        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points

    def loss_affordance(self, outputs, targets, indices, num_masks, mask_type):
        loss_fn = (
            lambda x, t: soft_dice_loss(x, t, num_masks)
            if self.grasp_loss_type == "dice"
            else sigmoid_ce_loss_jit_mean(x, t)
        )
        loss = {}
        if "scene_grasps" in outputs:
            scene_grasps: List[torch.Tensor] = outputs["scene_grasps"]  # list wtih length of batchsize
            scene_targets: List[torch.Tensor] = [
                t["scene_centric_labels"][J] for t, (_, J) in zip(targets, indices)
            ]  # B x [n_inst, n_grasps, n_orientations]
            scene_loss = []
            for p, t in zip(scene_grasps, scene_targets):
                for pp, tt in zip(p, t):
                    if tt.sum() > 0:
                        scene_loss.append(loss_fn(pp, tt))
            if len(scene_loss) > 0:
                loss["loss_grasp_scene"] = torch.stack(scene_loss).mean()

        if "object_grasps" in outputs:
            object_grasps: List[torch.Tensor] = outputs["object_grasps"]  # list wtih length of batchsize
            object_targets: List[torch.Tensor] = [
                t["object_centric_labels"][J] for t, (_, J) in zip(targets, indices)
            ]  # B x [n_inst, n_grasps, n_orientations]
            object_loss = []
            for p, t in zip(object_grasps, object_targets):
                for pp, tt in zip(p, t):
                    if len(tt) > 0 and tt.sum() > 0:
                        l = loss_fn(pp, tt)
                        if not torch.isnan(l):
                            object_loss.append(l)

            if len(object_loss) > 0:
                loss["loss_grasp_object"] = torch.stack(object_loss).mean()

    

        return loss

    # loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    # return loss.mean(1).sum() / num_masks
    def loss_sdf(self, outputs, targets, indices, num_masks, mask_type):
        """L1 loss
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        if "sdf_pred" not in outputs or outputs["sdf_pred"] is None:
            return None

        pred_sdf: List[torch.Tensor] = outputs["sdf_pred"]  # list wtih length of batchsize
        # print("Sdf shape", [p.shape for p in pred_sdf])

        target_sdf = [t["sdf"][J] for t, (_, J) in zip(targets, indices)]
        lbls = [t["labels"][J] for t, (_, J) in zip(targets, indices)]

        if len(self.exclude_sdf_class) == 0 and torch.cat(lbls).max() != 0: # only calculate loss for non table classes DISABLED for now.
            # raise ValueError("not currently correctly implemented!!!!!")
            target_sdf = [t[lbls > 0] for t, lbls in zip(target_sdf, lbls)]
            pred_sdf = [p[lbls > 0] for p, lbls in zip(pred_sdf, lbls)]

        if self.sdf_loss_type != "point2surf":
            pred_sdf = [p.squeeze(-1) for p in pred_sdf]

        if self.sdf_loss_type == "l1":
            loss = torch.stack(
                [
                    F.l1_loss(
                        100 * p.clamp(-self.sdf_clip, self.sdf_clip),
                        100 * t.clamp(-self.sdf_clip, self.sdf_clip),
                    )
                    for p, t in zip(pred_sdf, target_sdf)
                ]
            ).mean()

        elif self.sdf_loss_type == "l2":
            loss = torch.stack(
                [
                    F.mse_loss(
                        100 * p.clamp(-self.sdf_clip, self.sdf_clip),
                        100 * t.clamp(-self.sdf_clip, self.sdf_clip),
                    )
                    for p, t in zip(pred_sdf, target_sdf)
                ]
            ).mean()
        elif self.sdf_loss_type == "point2surf":

            # loss_ce
            loss_ce = torch.stack(
                [sigmoid_ce_loss_jit(p[..., 0], (t >= 0).float(), 1) for p, t in zip(pred_sdf, target_sdf)]
            ).mean()
            
            loss_dist = torch.stack(
                [
                    F.mse_loss(
                        F.tanh(p[...,-1] * 50),
                        F.tanh(t.abs() * 50)
                    )
                    for p, t in zip(pred_sdf, target_sdf)
                ]
            ).mean()
            loss =  loss_ce + 0.5*loss_dist

        elif self.sdf_loss_type == "dice":
            loss = torch.stack([dice_loss(p, (t >= 0).float(), 1) for p, t in zip(pred_sdf, target_sdf)]).mean()

        elif self.sdf_loss_type == "bce":
            loss = torch.stack(
                [sigmoid_ce_loss_jit(p, (t >= 0).float(), 1) for p, t in zip(pred_sdf, target_sdf)]
            ).mean()
            # loss = torch.cat(
            #     [F.binary_cross_entropy_with_logits(p, (t >= 0).float(), reduction="none") for p, t in zip(pred_sdf, target_sdf)]
            # ).mean()
        elif self.sdf_loss_type == "both":
            loss = 0.2*torch.stack([dice_loss(p, (t >= 0).float(), 1) for p, t in zip(pred_sdf, target_sdf)]).mean()
            loss += 0.8*torch.stack(
                [sigmoid_ce_loss_jit(p, (t >= 0).float(), 1) for p, t in zip(pred_sdf, target_sdf)]
            ).mean()
        elif self.sdf_loss_type == "bce_weighted":
            losses = []
            for p, t in zip(pred_sdf, target_sdf):
                # balance
                for pp, tt in zip(p, t):
                    tgt_bin = (tt >= 0).float()
                    losses.append(
                        F.binary_cross_entropy_with_logits(pp, tgt_bin, pos_weight=1 / (0.001 + tgt_bin.mean()))
                    )
            loss = torch.stack(losses).mean()
        elif self.sdf_loss_type == "bce_resample":
            losses = []
            for p, t in zip(pred_sdf, target_sdf):
                # balance
                for pp, tt in zip(p, t):
                    # 50% close to object
                    tgt_mask = tt.abs() < 0.03  # close to object
                    # 50% random
                    tgt_mask[torch.where(~tgt_mask)[0][torch.randperm((~tgt_mask).sum())][: tgt_mask.sum()]] = True

                    losses.append(F.binary_cross_entropy_with_logits(pp[tgt_mask], (tt[tgt_mask] >= 0).float()))
            loss = torch.stack(losses).mean()
        else:
            raise ValueError("Unknown sdf loss type", self.sdf_loss_type)
        
        losses = {"loss_sdf": loss}

        if self.matcher.costs.get("scene_occ", 0.0) != 0.0:
            losses["loss_scene_occ"] = torch.stack([F.binary_cross_entropy_with_logits(p.max(0)[0],(t.max(0)[0]>=0).float()) for p, t in zip(pred_sdf, target_sdf)]).mean()

        return losses

    def loss_labels(self, outputs, targets, indices, num_masks, mask_type):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            self.empty_weight,
            ignore_index=253,
        )

        losses = {"loss_ce": loss_ce}
        return losses

    def loss_normal(self, outputs, targets, indices, num_masks, mask_type):
        if "gt_normals" not in outputs:
            return {}

        pred_normals = outputs["normals"]
        gt_normals = outputs["gt_normals"]
        gt_normals_id = outputs["gt_normals_id"]

        losses = []
        for b_idx in range(len(gt_normals)):
            pred_n = pred_normals[b_idx]
            gt_n = gt_normals[b_idx]
            ids = gt_normals_id[b_idx]
            for src_idx, gt_inst in enumerate(indices[b_idx][1]):
                gt_n_inst = gt_n[ids == gt_inst]
                pred_n_inst = pred_n[src_idx][ids == gt_inst]
                loss = (1 - torch.cosine_similarity(gt_n_inst, -pred_n_inst)).mean()
                losses.append(loss)

        return {"loss_normals": torch.stack(losses).mean()  }  if len(losses) > 0 else {}

    def loss_masks(self, outputs, targets, indices, num_masks, mask_type):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        loss_masks = []
        loss_dices = []

        for batch_id, (map_id, target_id) in enumerate(indices):
            # print("==> loss masks calc.")
            # print("shape", outputs["pred_masks"][batch_id].shape)
            map = outputs["pred_masks"][batch_id][:, map_id].T
            target_mask = targets[batch_id][mask_type][target_id]
            # print("mask shape", target_mask.shape)
            # print("shape opred", map.shape)
            # print("target", targets[batch_id][mask_type][target_id].shape)
            # print("vals", map,  targets[batch_id][mask_type][target_id])
            # print(self.num_points)
            if self.num_points != -1:
                point_idx = torch.randperm(target_mask.shape[1], device=target_mask.device)[
                    : int(self.num_points * target_mask.shape[1])
                ]
            else:
                # sample all points
                point_idx = torch.arange(target_mask.shape[1], device=target_mask.device)

            num_masks = target_mask.shape[0]
            map = map[:, point_idx]
            target_mask = target_mask[:, point_idx].float()

            dice_loss = dice_loss_jit(map, target_mask, num_masks)
            sigm_loss = sigmoid_ce_loss_jit(map, target_mask, num_masks)
            if not torch.isnan(dice_loss).any():
                loss_dices.append(dice_loss)
            else:
                loss_dices.append(torch.zeros_like(dice_loss))
                print("NaN Detected in dice loss!!!!", map.shape, map.unique())

            if not torch.isnan(sigm_loss).any():
                loss_masks.append(sigm_loss)
            else:
                loss_masks.append(torch.zeros_like(sigm_loss))
                print("NaN Detected in sigmoid loss!!!!", map.shape, map.unique())

        # del target_mask
        return {
            "loss_mask": torch.mean(torch.stack(loss_masks)),
            "loss_dice": torch.mean(torch.stack(loss_dices)),
        }

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, mask_type):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
            "sdf": self.loss_sdf,
            "grasps": self.loss_affordance,
            "normals": self.loss_normal
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, mask_type)

    def get_matches(self, outputs, targets, mask_type):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, mask_type)
        return indices

    def forward(self, outputs, targets, mask_type, indices=None, rematch_aux=False):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        if indices is None:
            indices = self.get_matches(outputs, targets, mask_type)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks, min=1).item()

        # Compute all the requested losses
        losses = {}
        # print("Forwarding", outputs.keys())
        for loss in self.losses:
            # print("Getting", loss)
            try:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, mask_type))
            except Exception as e:
                print("could not load loss for", loss)
                print(str(e))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                if rematch_aux:
                    indices = self.matcher(
                        aux_outputs, targets, mask_type
                    )  # TODO, why is this rematching here???????????

                for loss in self.losses:
                    #
                    if loss in self.exclude_inter:  # only for last layer
                        continue

                    try:
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, mask_type)
                        if l_dict is not None:
                            l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                            losses.update(l_dict)
                    except Exception as e:
                        print("could not load loss for", loss)
                        print(str(e))

        if self.regularizer > 0:
            norm = 0
            for query in outputs["all_queries"]:
                norm += query.norm(dim=-1).mean()
            losses["loss_lat_regularizer"] = self.regularizer * norm
        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


def build_6d_grasp(approach_dirs, base_dirs, contact_pts, thickness, gripper_depth=0.1034):
    """
    Build 6-DoF grasps + width from point-wise network predictions

    Arguments:
        approach_dirs {np.ndarray/torch.tensor} -- Nx3 approach direction vectors
        base_dirs {np.ndarray/torch.tensor} -- Nx3 base direction vectors
        contact_pts {np.ndarray/torch.tensor} -- Nx3 contact points
        thickness {np.ndarray/torch.tensor} -- Nx1 grasp width

    Keyword Arguments:
        use_tf {bool} -- whether inputs and outputs are tf tensors (default: {False})
        gripper_depth {float} -- distance from gripper coordinate frame to gripper baseline in m (default: {0.1034})

    Returns:
        np.ndarray -- Nx4x4 grasp poses in camera coordinates
    """
    grasps_R = torch.stack([base_dirs, torch.linalg.cross(approach_dirs, base_dirs), approach_dirs], dim=-1)
    grasps_t = contact_pts + thickness.unsqueeze(-1) / 2 * base_dirs - gripper_depth * approach_dirs
    ones = torch.ones((*contact_pts.shape[:-1], 1, 1), dtype=grasps_t.dtype, device=grasps_t.device)
    zeros = torch.zeros((*contact_pts.shape[:-1], 1, 3), dtype=grasps_t.dtype, device=grasps_t.device)
    homog_vec = torch.concat([zeros, ones], dim=-1)
    grasps = torch.concat([torch.concat([grasps_R, grasps_t.unsqueeze(-1)], dim=-1), homog_vec], dim=-2)

    return grasps


def homogenize(data: torch.Tensor):
    return torch.cat([data, torch.ones(data.shape[:-1], device=data.device).unsqueeze(-1)], -1)


def get_control_points(device):
    import numpy as np

    control_points = np.load("gripper_control_points/panda.npy")[:, :3]
    # sym
    control_points_sym = [
        [0, 0, 0],
        control_points[1, :],
        control_points[0, :],
        control_points[-1, :],
        control_points[-2, :],
    ]
    control_points_sym = np.tile(np.expand_dims(control_points_sym, 0), [1, 1])
    control_points_sym = torch.from_numpy(control_points_sym).to(device)

    control_points = [
        [0, 0, 0],
        control_points[0, :],
        control_points[1, :],
        control_points[-2, :],
        control_points[-1, :],
    ]
    control_points = np.tile(np.expand_dims(control_points, 0), [1, 1])
    control_points = torch.from_numpy(control_points).to(device)
    return homogenize(control_points).float(), homogenize(control_points_sym).float()


def grasp_loss_with_width(pred, tgt, loss_type = "bce"):
    bce_labels = tgt[:, :-1]
    bce_preds = pred[:, :-1]
    bce_loss = F.binary_cross_entropy_with_logits(bce_preds, (bce_labels > 0).float()).mean()
    
    if "both" == loss_type:
        bce_loss = bce_loss * 0.9 + 0.1 * soft_dice_loss(bce_preds, (bce_labels > 0).float(), 1).mean()

    width_labels = tgt[:, -1]
    width_preds = pred[:, -1]
    width_loss = (F.mse_loss(40 * width_preds, 40 * width_labels) * bce_labels.any(dim=-1)).mean()

    return {
        "loss_grasp_bce": bce_loss,
        "loss_grasp_width": width_loss,
    }


class SetCriterionWithWidth(SetCriterion):
    def loss_affordance(self, outputs, targets, indices, num_masks, mask_type):
        loss_fn = grasp_loss_with_width
        loss = {}
        if "scene_grasps" in outputs:
            scene_grasps: List[torch.Tensor] = outputs["scene_grasps"]  # list wtih length of batchsize
            scene_targets: List[torch.Tensor] = [
                t["scene_centric_labels"][J] for t, (_, J) in zip(targets, indices)
            ]  # B x [n_inst, n_grasps, n_orientations]
            scene_loss_bce = []
            scene_loss_width = []

            for p, t in zip(scene_grasps, scene_targets):
                for pp, tt in zip(p, t):
                    if tt.sum() > 0:
                        l = loss_fn(pp, tt, loss_type=self.grasp_loss_type)
                        scene_loss_bce.append(l["loss_grasp_bce"])
                        scene_loss_width.append(l["loss_grasp_width"])

            if len(scene_loss_width) > 0:
                loss["loss_grasp_scene"] = torch.stack(scene_loss_bce).mean()
                loss["loss_grasp_width_scene"] = torch.stack(scene_loss_width).mean()

            if self.matcher.costs.get("scene_grasp_aff", 0.0) != 0.0:

                full_scene_loss_bce = []
                full_scene_loss_width = []

                for p, t in zip(scene_grasps, scene_targets):
                        l = loss_fn(p.max(0)[0], t.max(0)[0], loss_type=self.grasp_loss_type)
                        full_scene_loss_bce.append(l["loss_grasp_bce"])
                        full_scene_loss_width.append(l["loss_grasp_width"])

                loss["loss_scene_grasp_aff"] = torch.stack(full_scene_loss_bce).mean()
                loss["loss_scene_grasp_width"] = torch.stack(full_scene_loss_width).mean()

        if "object_grasps" in outputs:
            object_grasps: List[torch.Tensor] = outputs["object_grasps"]  # list wtih length of batchsize
            object_targets: List[torch.Tensor] = [
                t["object_centric_labels"][J] for t, (_, J) in zip(targets, indices)
            ]  # B x [n_inst, n_grasps, n_orientations]
            object_loss = []
            object_loss_width = []
            for p, t in zip(object_grasps, object_targets):
                for pp, tt in zip(p, t):
                    if len(tt) > 0 and tt.sum() > 0:
                        l = loss_fn(pp, tt, loss_type=self.grasp_loss_type)
                        object_loss.append(l["loss_grasp_bce"])
                        object_loss_width.append(l["loss_grasp_width"])

            if len(object_loss) > 0:
                loss["loss_grasp_object"] = torch.stack(object_loss).mean()
                loss["loss_grasp_width_object"] = torch.stack(object_loss_width).mean()

        return loss


class SetCriterionGIGA(SetCriterion):
    def loss_affordance(self, outputs, targets, indices, num_masks, mask_type):
        # print("in loss. Labels: scene: ", scene_targets[0].shape, "object: ", object_targets[0].shape)
        # loss_fn = lambda x,t: dice_loss(x, t, 1) if self.grasp_loss_type == "dice" else sigmoid_ce_loss_jit(x, t, 1)

        qual_loss = lambda x, t: sigmoid_ce_loss_jit(x, t, num_masks)
        quat_loss = lambda x, t: sym_quat_loss(x, t, 1)
        width_loss = lambda x, t: F.mse_loss(40 * x, 40 * t)
        # torch.cat([rot, width,qual], axis = -1)
        loss = {}

        if "scene_grasps" in outputs:
            scene_grasps: List[torch.Tensor] = outputs["scene_grasps"]  # list wtih length of batchsize
            scene_targets: List[torch.Tensor] = [
                t["scene_centric_labels"][J] for t, (_, J) in zip(targets, indices)
            ]  # B x [n_inst, n_grasps, n_orientations]

            labels = [l[..., -1] == 1 for l in scene_targets]

            quat_losses = []
            width_losses = []
            qual_losses = []

            for p, t, l in zip(scene_grasps, scene_targets, labels):
                if l.sum() != 0:
                    quat_losses.append(quat_loss(p[l][:, :4], t[l][:, :4]))
                    width_losses.append(width_loss(p[l][:, 4][..., None], t[l][:, 4][..., None]))
                    if self.grasp_loss_type == "bce_balanced":
                        for pp, tgt_bin in zip(p[..., 5], t[..., 5]):
                            mloss = torch.topk(
                                F.binary_cross_entropy_with_logits(pp, (tgt_bin > 0).float(), reduction="none"),
                                k=min(64, len(pp)),
                            )[0]
                            qual_losses.append(mloss.mean())
                    else:
                        qual_losses.append(qual_loss(p[..., 5][..., None], t[..., 5][..., None]))
                # balance

            if len(quat_losses) != 0:
                scene_quat_loss = torch.stack(quat_losses).mean()
                scene_width_loss = torch.stack(width_losses).mean()
            scene_qual_loss = torch.stack(qual_losses).mean()

            loss["loss_scene_quat"] = scene_quat_loss if len(quat_losses) != 0 else 0 * scene_qual_loss
            loss["loss_scene_width"] = scene_width_loss if len(width_losses) != 0 else 0 * scene_qual_loss
            loss["loss_scene_qual"] = scene_qual_loss

        # loss["loss_grasp_scene"] = scene_loss

        if "object_grasps" in outputs:
            object_grasps: List[torch.Tensor] = outputs["object_grasps"]  # list wtih length of batchsize
            object_targets: List[torch.Tensor] = [
                t["object_centric_labels"][J] for t, (_, J) in zip(targets, indices)
            ]  # B x [n_inst, n_grasps, n_orientations]

            labels = [l[..., -1] == 1 for l in object_targets]

            quat_losses = []
            width_losses = []
            qual_losses = []

            for p, t, l in zip(object_grasps, object_targets, labels):
                if l.sum() != 0:
                    quat_losses.append(quat_loss(p[l][:, :4], t[l][:, :4]))
                    width_losses.append(width_loss(p[l][:, 4][..., None], t[l][:, 4][..., None]))
                    if self.grasp_loss_type == "bce_balanced":
                        for pp, tgt_bin in zip(p[..., 5], t[..., 5]):
                            mloss = torch.topk(
                                F.binary_cross_entropy_with_logits(pp, (tgt_bin > 0).float(), reduction="none"),
                                k=min(128, len(pp)),
                            )[0]
                            qual_losses.append(mloss.mean())
                    else:
                        qual_losses.append(qual_loss(p[..., 5][..., None], t[..., 5][..., None]))

            if len(quat_losses) != 0:
                object_quat_loss = torch.stack(quat_losses).mean()
                object_width_loss = torch.stack(width_losses).mean()
            object_qual_loss = torch.stack(qual_losses).mean()

            loss["loss_obj_quat"] = object_quat_loss if len(quat_losses) != 0 else 0 * object_qual_loss
            loss["loss_obj_width"] = object_width_loss if len(width_losses) != 0 else 0 * object_qual_loss
            loss["loss_obj_qual"] = object_qual_loss
        # for k,v in loss.items():
        #     print(k, ":", v)

        # print( {"loss_grasp_object": object_loss, "loss_grasp_scene": scene_loss})

        # scene_lables = np.concatenate([scene_grasp_qualities,  scene_offset_labels_pc, scene_dir_labels_pc, scene_approach_labels_pc, instance_labels], axis=1)

        # [t["scene_centric_labels"][J] for t, (_, J) in zip(targets, indices)] # B x [n_inst, n_grasps, n_orientations]

        # scene_loss = torch.stack(
        #     [
        #         loss_fn(p,t) for p,t in zip(scene_grasps, scene_targets)
        #     ]).mean()
        # loss["loss_grasp_scene"] = scene_loss

        # if "object_grasps" in outputs:
        #     object_grasps : List[torch.Tensor] = outputs["object_grasps"] # list wtih length of batchsize
        #     object_targets: List[torch.Tensor] = [t["object_centric_labels"][J] for t, (_, J) in zip(targets, indices)] # B x [n_inst, n_grasps, n_orientations]

        #     object_loss = torch.stack(
        #     [
        #         loss_fn(p,t) for p,t in zip(object_grasps, object_targets)
        #     ]).mean()
        #     loss["loss_grasp_object"] = object_loss
        # print( {"loss_grasp_object": object_loss, "loss_grasp_scene": scene_loss})

        return loss


@torch.jit.export
def contact_graspnet_loss(
    scene_grasp_qualities: torch.Tensor,
    scene_offset_labels_pc: torch.Tensor,
    scene_dir_labels_pc: torch.Tensor,
    scene_approach_labels_pc: torch.Tensor,
    pred_qual: torch.Tensor,
    pred_offsets: torch.Tensor,
    pred_dir: torch.Tensor,
    pred_approach: torch.Tensor,
    scene_contact_pts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    segmentation_loss = F.binary_cross_entropy_with_logits(pred_qual, scene_grasp_qualities, reduction="none")
    # only backprop top k=512
    segmentation_loss = torch.topk(segmentation_loss, k=min(len(segmentation_loss), 512)).values

    grasps_pred = build_6d_grasp(
        pred_approach,
        pred_dir,
        scene_contact_pts,
        pred_offsets,
    )
    grasps_gt = build_6d_grasp(
        scene_approach_labels_pc,
        scene_dir_labels_pc,
        scene_contact_pts,
        scene_offset_labels_pc,
    )  # TODO

    pos_gt_grasps_proj = grasps_gt[scene_grasp_qualities == 1]
    if len(pos_gt_grasps_proj != 0):
        control_points, control_points_sym = get_control_points(grasps_pred.device)

        # Here we calculate the distance to the closest positive grasp

        pred_control_points = (control_points @ grasps_pred.permute([0, 2, 1]))[..., :-1]  # 2224, n_pts, 4
        pos_gt_control_points = (control_points @ pos_gt_grasps_proj.permute([0, 2, 1]))[..., :-1]
        error = ((pred_control_points[None, ...] - pos_gt_control_points[:, None, ...]) ** 2).sum([-1, -2])
        # element wise error. shape: [n_pred, n_gt, n_gripper_sample_pts, 3]
        # entry at [x,x,0,3] will be center of gripper

        pred_control_points_sym = (control_points_sym @ grasps_pred.permute([0, 2, 1]))[..., :-1]  # 2224, n_pts, 4
        pos_gt_control_points_sym = (control_points_sym @ pos_gt_grasps_proj.permute([0, 2, 1]))[..., :-1]
        error_sym = ((pred_control_points_sym[None, ...] - pos_gt_control_points_sym[:, None, ...]) ** 2).sum([-1, -2])

        errors = torch.cat([error, error_sym], dim=0)
        smallest_dist = errors.min(dim=0).values.sqrt()  # this is the distance to the closest positive grasp
        confidence_weighted_dist = torch.sigmoid(pred_qual) * smallest_dist
        l_add_s = confidence_weighted_dist / (1 + scene_grasp_qualities.sum())
        # l_add_s = torch.topk(confidence_weighted_dist, k=min(512, len(confidence_weighted_dist))).values / (
        #     1 + scene_grasp_qualities.sum()
        # )  # this is l_add-s in the paper
        # import open3d as o3d

        # pts = scene_contact_pts

        # def show_cloud(*args):
        #     d = []
        #     for cloud in args:
        #         pcd = o3d.geometry.PointCloud()
        #         if isinstance(cloud, torch.Tensor):
        #             cloud = cloud.detach().cpu().numpy()
        #         pcd.points = o3d.utility.Vector3dVector(cloud)
        #         d.append(pcd)
        #     o3d.visualization.draw(d)

        # import pdb

        # pdb.set_trace()
        # lets directly to regression in width for now! # TODO
        l_width = (pred_offsets[scene_grasp_qualities == 1] - scene_offset_labels_pc[scene_grasp_qualities == 1]) ** 2
    else:
        l_add_s = torch.zeros(1, device=grasps_pred.device)
        l_width = torch.zeros(1, device=grasps_pred.device)

    return segmentation_loss.mean(), l_add_s.sum(), l_width.mean()


class SetCriterionAcronym(SetCriterion):
    def loss_affordance(self, outputs, targets, indices, num_masks, mask_type):
        # print("in loss. Labels: scene: ", scene_targets[0].shape, "object: ", object_targets[0].shape)
        loss_fn = lambda x, t: dice_loss(x, t, 1) if self.grasp_loss_type == "dice" else sigmoid_ce_loss_jit(x, t, 1)
        # bins = torch.Tensor(
        #     [
        #         0,
        #         0.00794435329,
        #         0.0158887021,
        #         0.0238330509,
        #         0.0317773996,
        #         0.0397217484,
        #         0.0476660972,
        #         0.055610446,
        #         0.0635547948,
        #         0.0714991435,
        #         0.08,
        #     ]
        # ).cuda()[None,:]
        loss = {}

        if "scene_grasps" in outputs:
            scene_grasps: List[torch.Tensor] = outputs["scene_grasps"]  # list with length of batchsize

            loss["loss_scene_seg"] = []
            loss["loss_scene_grasp"] = []
            loss["loss_scene_width"] = []

            for pred, t, ind in zip(scene_grasps, targets, indices):
                scene_targets = t["scene_centric_labels"]
                scene_pts = t["scene_centric_pts"]
                instance_labels = scene_targets[:, 8].long()
                for idx, target_ind in enumerate(ind[1]):
                    target_mask = instance_labels == target_ind
                    scene_contact_pts = scene_pts[target_mask]
                    # print("scene ctct pts", scene_contact_pts.shape)

                    (
                        scene_grasp_qualities,
                        scene_offset_labels_pc,
                        scene_dir_labels_pc,
                        scene_approach_labels_pc,
                    ) = (
                        scene_targets[target_mask, 0],
                        scene_targets[target_mask, 1],
                        scene_targets[target_mask, 2:5],
                        scene_targets[target_mask, 5:8],
                    )
                    if len(scene_grasp_qualities) == 0:
                        continue
                    # with torch.no_grad():
                    #     bin_width_labels = (
                    #         (scene_offset_labels_pc - bins).abs().argmin(0)
                    #     )

                    pred_qual, pred_offsets, pred_dir, pred_approach = (
                        pred[idx, target_mask, 0],
                        pred[idx, target_mask, 1],
                        pred[idx, target_mask, 2:5],
                        pred[idx, target_mask, 5:8],
                    )
                    l_seg, l_grasp, l_width = contact_graspnet_loss(
                        scene_grasp_qualities,
                        scene_offset_labels_pc,
                        scene_dir_labels_pc,
                        scene_approach_labels_pc,
                        pred_qual,
                        pred_offsets,
                        pred_dir,
                        pred_approach,
                        scene_contact_pts,
                    )
                    # if l_seg.isnan().any():
                    #     print("NAN DETECTED!!!!", l_seg)
                    #     print("WHy? Gt:", scene_grasp_qualities.shape, "values:", scene_grasp_qualities.unique())
                    #     print("WHy? Pred:", pred_qual.shape, "values:", pred_qual.unique())
                    loss["loss_scene_seg"].append(l_seg)
                    loss["loss_scene_grasp"].append(l_grasp)
                    loss["loss_scene_width"].append(l_width)

            if len(loss["loss_scene_seg"]) > 0:
                loss["loss_scene_seg"] = torch.mean(torch.stack(loss["loss_scene_seg"]))
                loss["loss_scene_grasp"] = torch.mean(torch.stack(loss["loss_scene_grasp"]))
                loss["loss_scene_width"] = torch.mean(torch.stack(loss["loss_scene_width"]))
            else:
                loss["loss_scene_seg"] = torch.zeros(1, device=self.empty_weight.device)
                loss["loss_scene_grasp"] = torch.zeros(1, device=self.empty_weight.device)
                loss["loss_scene_width"] = torch.zeros(1, device=self.empty_weight.device)

        if "object_grasps" in outputs:
            object_grasps: List[torch.Tensor] = outputs["object_grasps"]  # list with length of batchsize

            for pred, t, ind in zip(object_grasps, targets, indices):
                scene_targets = t["object_centric_labels"]
                scene_pts = t["object_centric_pts"]
                instance_labels = scene_targets[:, 8].long()
                for idx, target_ind in enumerate(ind[1]):
                    target_mask = instance_labels == target_ind
                    scene_contact_pts = scene_pts[target_mask]

                    (
                        scene_grasp_qualities,
                        scene_offset_labels_pc,
                        scene_dir_labels_pc,
                        scene_approach_labels_pc,
                    ) = (
                        scene_targets[target_mask, 0],
                        scene_targets[target_mask, 1],
                        scene_targets[target_mask, 2:5],
                        scene_targets[target_mask, 5:8],
                    )
                    # with torch.no_grad():
                    #     bin_width_labels = (
                    #         (scene_offset_labels_pc - bins).abs().argmin(0)
                    #     )

                    pred_qual, pred_offsets, pred_dir, pred_approach = (
                        pred[idx, target_mask, 0],
                        pred[idx, target_mask, 1],
                        pred[idx, target_mask, 2:5],
                        pred[idx, target_mask, 5:8],
                    )
                    l_seg, l_grasp, l_width = contact_graspnet_loss(
                        scene_grasp_qualities,
                        scene_offset_labels_pc,
                        scene_dir_labels_pc,
                        scene_approach_labels_pc,
                        pred_qual,
                        pred_offsets,
                        pred_dir,
                        pred_approach,
                        scene_contact_pts,
                    )
                    loss["object_seg"] = l_seg
                    loss["object_grasp"] = l_grasp
                    loss["object_width"] = l_width
        # print("Loss crerion:", loss)
        # scene_lables = np.concatenate([scene_grasp_qualities,  scene_offset_labels_pc, scene_dir_labels_pc, scene_approach_labels_pc, instance_labels], axis=1)

        # [t["scene_centric_labels"][J] for t, (_, J) in zip(targets, indices)] # B x [n_inst, n_grasps, n_orientations]

        # scene_loss = torch.stack(
        #     [
        #         loss_fn(p,t) for p,t in zip(scene_grasps, scene_targets)
        #     ]).mean()
        # loss["loss_grasp_scene"] = scene_loss

        # if "object_grasps" in outputs:
        #     object_grasps : List[torch.Tensor] = outputs["object_grasps"] # list wtih length of batchsize
        #     object_targets: List[torch.Tensor] = [t["object_centric_labels"][J] for t, (_, J) in zip(targets, indices)] # B x [n_inst, n_grasps, n_orientations]

        #     object_loss = torch.stack(
        #     [
        #         loss_fn(p,t) for p,t in zip(object_grasps, object_targets)
        #     ]).mean()
        #     loss["loss_grasp_object"] = object_loss
        # print( {"loss_grasp_object": object_loss, "loss_grasp_scene": scene_loss})

        return loss
