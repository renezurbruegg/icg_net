"""Query refinement module."""

from __future__ import annotations
import torch
import torch.nn as nn
from icg_net.model.refinement.layers import (
    CrossAttentionLayer,
    FFNLayer,
)
# removed vis4d dependency import
from icg_net.model.resnet_fc import ResnetBlockFC

import torch
import torch.nn as nn
import MinkowskiEngine as ME

import torch.nn as nn
import MinkowskiEngine.MinkowskiOps as me
import MinkowskiEngine as ME

import torch.nn.functional as F
from torch_cluster import knn
import torch_scatter 

@torch.jit.script
def interpolate_grid(grid: torch.Tensor, coord: torch.Tensor,  min_coord: float, max_coord: float, method: str = "bilinear") -> torch.Tensor:
    pts_norm = (coord - min_coord) / ( max_coord -  min_coord) * 2 - 1
    dense_inter = F.grid_sample(
        grid[None, ...], 
        pts_norm[None, :, None, None, [2, 1, 0]],  # reverse last dim to match grid_sample
        mode=method,
        align_corners=False,
        padding_mode="border",
    )
    # shaped bs x n_pts x dense_feat_dim
    return dense_inter.squeeze(0).squeeze(-1).squeeze(-1).permute(1, 0)


class DenseVolumeInterpolator(nn.Module):
    
    def __init__(self, min_coord: float, max_coord: float):
        super().__init__()
        self.min_coord = min_coord
        self.max_coord = max_coord

    def forward(self, coordinates: list[torch.Tensor], grid: list[torch.Tensor], method:str = "bilinear") -> torch.Tensor:
        future = [torch.jit.fork(interpolate_grid, g, c, self.min_coord, self.max_coord, method) for c, g in zip(coordinates, grid)]
        return [torch.jit.wait(f) for f in future]
    
    
class PointNetKnnInterpolator(nn.Module):
    """Interpolator that uses a small pointnet and knn-neigbours to interpolate features from a sparse tensor to a set of points."""

    def __init__(self, feature_dim: int, k: int = 8):
        super().__init__()
        self.k = k
        self.pointnet = nn.Sequential(ResnetBlockFC(size_in = feature_dim + 3, size_out = feature_dim), ResnetBlockFC(size_in = feature_dim , size_out = feature_dim))
        

    def forward(self, features: ME.SparseTensor, pts: torch.Tensor):
        f = features.features  # Shape N_pts x 256
        coords = features.coordinates[:, 1:].float().contiguous()   # N_pts x 3 (ints)
        batch_idx = features.coordinates[:, 0]
        pts_coords = pts[:, 1:].contiguous()  # N_inter_pts x 3 (floats)
        pts_batch_idx = pts[:, 0].long()
        return self.knn_interpolate(f, coords, pts_coords, batch_idx, pts_batch_idx)

    def knn_interpolate(self, x: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor,
                    batch_x = None, batch_y = None, num_workers: int = 1):
    
        with torch.no_grad():
            assign_index = knn(pos_x, pos_y, self.k, batch_x=batch_x, batch_y=batch_y,
                            num_workers=num_workers)
            y_idx, x_idx = assign_index[0], assign_index[1]
            diff = pos_x[x_idx] - pos_y[y_idx]

        stacked_pts = torch.cat([x[x_idx], diff], 1) # shape n_pts x (feature_dim + 3)
        features = self.pointnet(stacked_pts)
        out = torch_scatter.scatter(features, y_idx, dim=0, dim_size=pos_y.size(0), reduce='max')
        return out


class SparseToDenseInterpolator(nn.Module):
    """Interpolation layer that attends to all sparse features."""

    def __init__(self, feature_dim: int, query_dim: int, dim_feedforward:int, pre_norm:bool = False, num_heads : int = 8, dropout = 0.0):
        super().__init__()

        self.feat_emb = nn.Linear(feature_dim, query_dim)
        self.num_heads = num_heads

        self.cross_attention = CrossAttentionLayer(
            d_model=query_dim, nhead=num_heads, dropout=dropout, normalize_before=pre_norm
        )

        self.ffn = FFNLayer(
            d_model=query_dim, dim_feedforward=dim_feedforward, dropout=dropout, normalize_before=pre_norm
        )

    def forward(
        self,
        queries: torch.Tensor,
        query_pos_encoding: torch.Tensor,
        features: torch.Tensor,
        feature_pos_encoding: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:

        refined_queries = self.cross_attention(
            queries, self.feat_emb(features), memory_mask = mask , query_pos=query_pos_encoding, pos=feature_pos_encoding
        )

        out = self.ffn(refined_queries)
        return out



class FeatureInterpolator(nn.Module):
    def __init__(
        self, min_coords, max_coords, unet_shapes, sparse_shapes, hidden_dim, quantization_size, sparse_interpolator
    ) -> None:
        super().__init__()
        self.dense_volume_inter = DenseVolumeInterpolator(min_coords, max_coords)

        self.dense_interpolations = nn.ModuleList()

        for in_dim in unet_shapes:
            self.dense_interpolations.append(ResnetBlockFC(in_dim, hidden_dim))

        self.point_interpolations = nn.ModuleList()
        for in_dim in sparse_shapes:
            self.point_interpolations.append(ResnetBlockFC(in_dim, hidden_dim))

        self.out_mlp = ResnetBlockFC(hidden_dim * 2, hidden_dim - 3)

        self.hidden_dim = hidden_dim
        self.sparse_interpolator = sparse_interpolator
        self.quantization_size = quantization_size

    def _dense_branch(self, xyz, features, stage: int):
        dense_inter = torch.stack(self.dense_volume_inter(xyz, features, method="bilinear"))
        vol_features = self.dense_interpolations[stage](dense_inter)
        return vol_features

    def _sparse_branch(self, xyz, sparse_coordinates, features, stage: int, batch_size: int):

        pts = ME.utils.batched_coordinates(
            [p / self.quantization_size for p in xyz],
            dtype=torch.float32,
            device=xyz[0].device,
        )

        point_feats = self.point_interpolations[stage](features)
        surface_features = self.sparse_interpolator(
            me.SparseTensor(
                point_feats,
                coordinates=sparse_coordinates,
                device=xyz[0].device,
            ),
            pts,
        ).reshape(batch_size, -1, self.hidden_dim)
        return surface_features

    def forward(self, points_pos_enc, xyz, sparse_coordinates, dense_features, multi_res_feats, batch_size, stage: int):

        vol_features = torch.jit.fork(self._dense_branch, xyz, dense_features, stage)
        surface_features = torch.jit.fork(
            self._sparse_branch, xyz, sparse_coordinates, multi_res_feats, stage, batch_size
        )

        vol_features, surface_features = torch.jit.wait(vol_features), torch.jit.wait(surface_features)

        if vol_features is not None and surface_features is not None:
            features = torch.cat([vol_features, surface_features], -1)
        elif vol_features is not None:
            features = torch.cat([vol_features], -1)
        elif surface_features is not None:
            features = torch.cat([surface_features], -1)

        pts_feats = self.out_mlp(features)
        points = torch.cat([torch.zeros_like(points_pos_enc[..., :3]), pts_feats], -1) + points_pos_enc

        return points