from __future__ import annotations
import torch
from torch import nn
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryPrecision, BinaryRecall
from torchmetrics.functional.classification import multiclass_jaccard_index

class OccupancyMetrics(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f1 = BinaryF1Score()
        self.acc = BinaryAccuracy()

    def forward(
        self,
        occupancy_labels: torch.tensor,
        icg_net: torch.tensor,
    ):
        icg_net = icg_net.sigmoid()
        if len(icg_net) == 0:
            return
            
        self.acc(icg_net, occupancy_labels)
        self.f1(icg_net, occupancy_labels)

    def reset(self):
        self.f1.reset()
        self.acc.reset()

    def get_metrics(self, prefix="") -> dict[str, torch.tensor]:
        return {
            f"{prefix}/acc": self.acc.compute(),
            f"{prefix}/f1": self.f1.compute(),
        }


class InstanceClassificationMetrics(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.miou_avg = 0
        self.miou_count = 0
        self.acc = BinaryAccuracy()

    def forward(
        self,
        pred: torch.tensor,
        lbl: torch.tensor,
    ):
        if lbl.max().item() > 0:
            self.miou_avg += multiclass_jaccard_index(pred, lbl, num_classes=lbl.max().item() + 1)
        else:
            self.miou_avg += 1
            
        self.miou_count += 1
        self.acc(pred==lbl, lbl >= 0) # ugly way of writin mean accuracy
        
        # self.miou(instance_prediction, instance_labels)
        # self.acc(instance_prediction, instance_labels)
    def get_metrics(self, prefix="") -> dict[str, torch.tensor]:
            return {
                f"{prefix}/acc": self.acc.compute(),
                f"{prefix}/miou": self.miou_avg / (self.miou_count+ 0.0001),
            }
    def reset(self):
        self.miou_avg = 0
        self.miou_count = 0
        self.acc.reset()


def sym_quat_loss(
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
    quat_xyzw = targets
    loss1 = 1 - (inputs * quat_xyzw).sum(dim=-1).abs()
    quat_rot = torch.stack(
        (quat_xyzw[..., 1], -quat_xyzw[..., 0], quat_xyzw[..., 3], -quat_xyzw[..., 2]),
        -1,
    )
    loss2 = 1 - (inputs * quat_rot).sum(dim=-1).abs()
    loss = torch.min(loss1, loss2).mean()
    return loss


class GraspMetrics(nn.Module):
    def __init__(self, topk=50):
        super().__init__()
        self.f1 = BinaryF1Score()
        self.acc = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.topk = topk

        # self.width_mse = Mse

        self.topk_acc = BinaryAccuracy()

    def reset(self):
        self.f1.reset()
        self.acc.reset()
        self.precision.reset()
        self.recall.reset()
        self.topk_acc.reset()

    def forward(
        self,
        grasp_labels: torch.tensor,
        grasp_prediction: torch.tensor,
        decoder_type: str,
        grasp_points: torch.tensor | None = None,
    ):
        if decoder_type == "occnet" or decoder_type == "occnet_width":
            if decoder_type == "occnet_width":
                grasp_prediction = grasp_prediction[..., :-1]
                grasp_labels = grasp_labels[..., :-1]

            pred_for_each_orient = grasp_prediction.sigmoid()  # [n_obj, grasps, n_orient]
            gt_for_each_orient = grasp_labels.long()  # [n_obj, grasps, n_orient]

            if pred_for_each_orient.ndim == 3:  # Calculate over all objects
                pred_for_each_orient = pred_for_each_orient.max(dim=0)[0]
                gt_for_each_orient = gt_for_each_orient.max(dim=0)[0]

            self.acc(pred_for_each_orient, gt_for_each_orient)
            self.precision(pred_for_each_orient, gt_for_each_orient)
            self.recall(pred_for_each_orient, gt_for_each_orient)
            self.f1(pred_for_each_orient, gt_for_each_orient)

            # topk
            top_grasps, top_idx = pred_for_each_orient.max(dim=-1, keepdim=True)
            top_grasps_labels = torch.gather(gt_for_each_orient, -1, top_idx)  # pylint: disable=no-member

            top_grasps = top_grasps.squeeze()
            top_grasps_labels = top_grasps_labels.squeeze()
            k = min(self.topk, len(top_grasps))
            top_grasps, idxs = torch.topk(top_grasps, dim=0, k=k)  # pylint: disable=no-member
            top_grasps_labels = top_grasps_labels[idxs]
            self.topk_acc(top_grasps, top_grasps_labels)

        if decoder_type == "acronym":
            gt_inst = grasp_labels[:, 8].long()
            gt_qual = grasp_labels[:, 0]
            pred_qual, pred_inst = grasp_prediction[..., 0].max(dim=0)
            pred_qual[gt_inst != pred_inst] = 1 * pred_qual[gt_inst != pred_inst]
            gt_qual[gt_inst != pred_inst] = 0
            correct = (pred_qual > 0) == (gt_qual > 0)

            self.acc(correct.float(), torch.ones_like(correct))
            # TODO
            self.precision(correct.float(), torch.ones_like(correct))
            self.recall(correct.float(), torch.ones_like(correct))
            self.f1(correct.float(), torch.ones_like(correct))

            k = min(self.topk, grasp_prediction.shape[1])

            # topk
            top_grasps, top_idx = torch.topk(grasp_prediction[..., 0].max(dim=0)[0], k=k)
            top_grasps_labels = gt_qual[top_idx]  # pylint: disable=no-member
            top_grasps[gt_inst[top_idx] != pred_inst[top_idx]] = 1
            top_grasps_labels[gt_inst[top_idx] != pred_inst[top_idx]] = 0
            correct = (top_grasps > 0) == (top_grasps_labels > 0)
            self.topk_acc(correct.float(), torch.ones_like(correct))

        if decoder_type == "giga":
            quat_pred = grasp_prediction[..., :4]
            width_pred = grasp_prediction[..., 4]
            qual_pred = grasp_prediction[..., -1].reshape(-1)

            quat_gt = grasp_labels[..., :4]
            width_gt = grasp_labels[..., 4]
            qual_gt = grasp_labels[..., -1].reshape(-1)
            self.acc(qual_pred, qual_gt)
            self.precision(qual_pred, qual_gt)
            self.recall(qual_pred, qual_gt)
            self.f1(qual_pred, qual_gt)

            qualities, idxs = torch.topk(qual_pred, k=min(len(qual_pred), self.topk))
            self.topk_acc(qualities, qual_gt[idxs])

            ((width_pred - width_gt) ** 2).mean().sqrt()
            sym_quat_loss(quat_pred.reshape(-1, 4), quat_gt.reshape(-1, 4))

    def get_metrics(self, prefix="") -> dict[str, torch.tensor | float]:
        return {
            f"{prefix}/acc": self.acc.compute(),
            f"{prefix}/f1": self.f1.compute(),
            f"{prefix}/precision": self.precision.compute(),
            f"{prefix}/recall": self.recall.compute(),
            f"{prefix}/topk_acc": self.topk_acc.compute(),
        }
