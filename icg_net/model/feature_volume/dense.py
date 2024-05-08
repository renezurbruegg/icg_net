from __future__ import annotations
from typing import List

from typing import List
import torch
import torch.nn as nn
import MinkowskiEngine.MinkowskiOps as me
import MinkowskiEngine as ME

import torch
from torch import nn
import torch.nn as nn
# from torch_geometric.nn.unpool import knn_interpolate

from icg_net.model.feature_volume.unet3d import UNet3D
from icg_net.model.me_utils import sparse_quantize, batched_coordinates



def list_to_chunks(data: list[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
    return torch.cat(data, dim=0), [len(d) for d in data]


def split_at_chunks(
    data: torch.Tensor, chunks: list[int]
) -> list[torch.Tensor]:
    return data.split(chunks, dim=0)


def sparse_to_dense(decomposed_features: list[torch.Tensor], decomposed_point_coords: list[torch.Tensor], min_coord:float, max_coord:float, dense_grid_resolution:float) -> torch.Tensor:

    all_feats, split_idxs = torch.cat(decomposed_features, dim=0), [len(d) for d in decomposed_features]
    all_coords = torch.cat(decomposed_point_coords, dim=0)
    # Remove points that are too close to the boundary
    valid = torch.logical_and((all_coords >= min_coord).all(1), (all_coords <= max_coord).all(1))
    all_coords = all_coords[valid]
    all_feats = all_feats[valid]
    
    split_idxs = [v.sum() for v in valid.split(split_idxs)]
    # add min and mox coord to the grid
    c_extended = torch.cat(
        [
            all_coords,
            torch.tensor([[min_coord, min_coord, min_coord]], device=all_coords.device),
            torch.tensor([[max_coord, max_coord, max_coord]], device=all_coords.device),
        ],
        dim=0,
    )
    feats_extended = torch.cat(
        [
            all_feats,
            torch.zeros((2, all_feats.size(-1)), device=all_feats.device),
        ],
        dim=0,
    )
    split_idxs[-1] += 2

    coordinates = []
    features = []
    for coords, feats in zip(c_extended.split(split_idxs), feats_extended.split(split_idxs)):
        # Make it Dense
        coords, feats = sparse_quantize(
            coordinates=coords, features=feats, quantization_size=dense_grid_resolution, device=feats.device
        )
        coordinates.append(coords)
        features.append(feats)

    features = torch.cat(features, dim=0)
    coordinates = batched_coordinates(coordinates, device=features.device)
    grid = me.SparseTensor(features=features, coordinates=coordinates, device=feats.device).dense()[0]
    return grid


class DenseFeatureExtractor(nn.Module):

  def __init__(self, quant_size: float, borders: List[float], f_maps, num_levels, out_dim, in_dim=1, with_norm = False):
    super().__init__()
    self.quant_size = quant_size
    self.min_coord, self.max_coord = borders

    self.dense_net = UNet3D(
        in_dim, out_dim, final_sigmoid=False, f_maps=f_maps, num_levels=num_levels, layer_order="crb", is_segmentation=False
    )
    

  def forward(self, sparse_points: ME.SparseTensor, coordinates: ME.SparseTensor):
        dense_grid_feats = sparse_to_dense(sparse_points.decomposed_features, coordinates.decomposed_features,
                                            min_coord=self.min_coord, max_coord=self.max_coord, dense_grid_resolution=self.quant_size)
        dense_feats, aux = self.dense_net(dense_grid_feats, with_aux = True)
        return dense_feats, aux
  