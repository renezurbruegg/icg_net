from __future__ import annotations


import torch
from torch import Tensor
import MinkowskiEngine as ME
import numpy as np
from typing import NamedTuple, TypedDict, List
import trimesh


class SceneEmbedding(NamedTuple):
    pos_encodings: torch.Tensor
    shape: torch.Tensor
    scene_grasps: torch.Tensor
    class_labels: torch.Tensor
    voxelized_pc: torch.Tensor
    pointwise_labels: torch.Tensor
    voxel_assignement: np.ndarray
    semseg_points: torch.Tensor
    semseg_latents: torch.Tensor
    semseg: torch.Tensor


class Grasp(NamedTuple):
    orientation: np.ndarray
    position: np.ndarray
    score: float
    width: float
    instance_id: int


class ModelPredOut(NamedTuple):
    class_predictions: torch.Tensor
    embedding: SceneEmbedding
    scene_grasp_poses: list[Grasp]
    reconstructions: list[trimesh.Trimesh]


class ICGNetOutput(TypedDict):
    pred_logits: Tensor
    pred_masks: Tensor
    instance_queries: Tensor
    all_queries: List[Tensor]
    instance_latents: Tensor
    intermittent_latents: List[Tensor]
    positional_encodings: Tensor
    aux_outputs: dict[str, Tensor]
    sampled_coords: np.typing.NDArray[np.float32]
    backbone_features: List[ME.SparseTensor]
    attention_info: List[tuple[Tensor | None, Tensor]]
    object_grasp_queries: List[Tensor]
    scene_grasp_queries: List[Tensor]
