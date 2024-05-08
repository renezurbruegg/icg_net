"""ICGNet model."""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
import MinkowskiEngine.MinkowskiOps as me
import MinkowskiEngine as ME

from icg_net.minkowski.res16unet import Res16UNetBase


import sys

if sys.version_info >= (3, 8):
    from typing import TypedDict, overload  # pylint: disable=no-name-in-module
else:
    pass


from icg_net.model.feature_volume.dense import DenseFeatureExtractor

import torch
from torch import nn, Tensor
import torch.nn as nn

from icg_net.model.feature_volume.dense import DenseFeatureExtractor
from icg_net.model.interpolation.interpolator import DenseVolumeInterpolator


class CombinedFeatureVolume(nn.Module):
    def __init__(
        self,
        sparse_feature_extractor: Res16UNetBase,
        dense_decoder_cfg: dict,
        hidden_dim: int,
        hlevels: List[int],
        
    ):
        super().__init__()
        self.sparse_feature_extractor = sparse_feature_extractor
        self.repr_sort_idxs = True # Makes sure that the coordinates are sorted in the same way as the features


        self.sparse_backbone_feat_shapes = list(self.sparse_feature_extractor.PLANES[-5:])
        # dense feats
        if dense_decoder_cfg is not None:
            f_maps = dense_decoder_cfg.feature_maps
            num_levels = dense_decoder_cfg.depth

            self.dense_extractor = DenseFeatureExtractor(
                quant_size=dense_decoder_cfg.quantization_size,
                borders=dense_decoder_cfg.borders,
                f_maps=f_maps,
                num_levels=num_levels,
                out_dim=hidden_dim,
                in_dim=self.sparse_feature_extractor.in_channels,
            )

            self.unet_shapes = [f_maps * (2**i) for i in range((num_levels + 2) // 2)]
            self.unet_shapes.extend([f_maps * (2**i) for i in range((num_levels + 2) // 2)][::-1])

            self.dense_volume_inter = DenseVolumeInterpolator(
                min_coord=dense_decoder_cfg.borders[0], max_coord=dense_decoder_cfg.borders[1]
            )

            for idx, lvl in enumerate(hlevels):
                self.sparse_backbone_feat_shapes[lvl] += self.unet_shapes[-len(hlevels) + idx]

        print("Created CombinedFeatureVolume")
        print("Sparse Backbone: ", type(self.sparse_feature_extractor))

        print("Dense Backbone: ", type(self.dense_extractor))
        print("Dense Interpolator: ", type(self.dense_volume_inter))
        print("Dense feature shapes: ", self.sparse_backbone_feat_shapes)


    @property
    def dense_feature_shapes(self):
        return self.unet_shapes
    
    def forward(self, voxelized_data: ME.SparseTensor, raw_coordinates: List[Tensor]) -> ME.SparseTensor:
        # split coordiantes
        sparse_pcd_features, aux_sparse = self.sparse_feature_extractor(voxelized_data, repr_sort_idxs=self.repr_sort_idxs)
        
        if isinstance(sparse_pcd_features, tuple):
            coord_sort_map = [a[1] for a in aux_sparse]
            aux = [a[0] for a in aux_sparse]
            sparse_pcd_features[0]
    
        start_coords = me.SparseTensor(
            features = torch.cat(raw_coordinates),
            coordinates=aux[-1].coordinates,
            device=aux[-1].device
        )

        if not self.repr_sort_idxs:
            coordinates = start_coords
        else:
            # split coordiantes
            coordinates = me.SparseTensor(
                features=torch.cat(raw_coordinates)[coord_sort_map[-1]],
                coordinates=aux[-1].coordinates[coord_sort_map[-1]],
                device=aux[-1].device,
            )
        dense_feats, dense_aux = self.dense_extractor(voxelized_data, coordinates)


        return sparse_pcd_features, aux_sparse, dense_feats, dense_aux

        