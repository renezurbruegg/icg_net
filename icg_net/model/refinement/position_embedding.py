# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
from __future__ import annotations
import math
from typing import overload
import torch
from torch import nn
import numpy as np


# from utils.pc_util import shift_scale_points
def extract_scene_bounds(coords: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    bs = len(coords)
    assert bs > 0

    scene_bounds = [
        torch.zeros((bs, coords[0].shape[-1]), device=coords[0].device),
        torch.zeros((bs, coords[0].shape[-1]), device=coords[0].device),
    ]

    for idx, c in enumerate(coords):
        scene_bounds[0][idx, :] = c.min(0)[0]
        scene_bounds[1][idx, :] = c.max(0)[0]
    return tuple(scene_bounds)


def shift_scale_points(
    pred_xyz: torch.Tensor,
    src_range: tuple[torch.Tensor, torch.Tensor],
    dst_range: tuple[torch.Tensor, torch.Tensor] | None = None,
    normalize_mode: bool = True,
):
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    if dst_range is None:
        dst_range = (
            torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device),  # pylint: disable=no-member
            torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),  # pylint: disable=no-member
        )

    if pred_xyz.ndim == 4:
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]

    bs_bounds_min = src_range[0].shape[0]
    bs_pts = pred_xyz.shape[0]

    if src_range[0].shape != src_range[1].shape:
        raise ValueError(f"size of bounds inconsistent: { src_range[0].shape} != { src_range[1].shape}")

    if bs_bounds_min != bs_pts:
        raise ValueError(f"Batch size mismatch: {bs_bounds_min} != {bs_pts}")

    if src_range[0].shape[-1] != pred_xyz.shape[-1]:
        raise ValueError(f"Inconsistent position dim. {src_range[0].shape[-1]} != {pred_xyz.shape[-1]}")

    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]

    if normalize_mode == "max":
        src_diff = src_diff.max(-1, keepdim=True).values.repeat_interleave(3, -1)

    prop_xyz = (((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff) + dst_range[0][:, None, :]
    return prop_xyz


class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(
        self, temperature=10000, normalize=False, scale=None, pos_type="fourier", d_pos=None, d_in=3, gauss_scale=1.0, with_raw_pts = True, add_normalized_positions = False
    ):
        super().__init__()
        self.d_pos = d_pos
        self.temperature = temperature
        self.normalize = normalize
        self.with_raw_pts = with_raw_pts
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        assert pos_type in ["sine", "fourier", "periodic"]
        self.pos_type = pos_type
        self.scale = scale
        self.add_normalized_positions = add_normalized_positions
        if pos_type == "fourier":
            assert d_pos is not None
            assert d_pos % 2 == 0
            # define a gaussian matrix input_ch -> output_ch
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            self.register_buffer("gauss_B", B)
            self.d_pos = d_pos

        if pos_type == "periodic":
            assert d_pos is not None
            assert d_pos % 2 == 0
            self.d_pos = d_pos
            # define a gaussian matrix input_ch -> output_ch
            B = 2 ** (torch.arange(d_pos // 2).unsqueeze(0).repeat_interleave(d_in, 0) / (d_pos // 2))
            B *= gauss_scale
            self.register_buffer("gauss_B", B)

    def get_fourier_embeddings(
        self,
        xyz: torch.Tensor,
        num_channels: int | None = None,
        input_range: list[tuple[torch.Tensor, torch.Tensor]] = [],
    ):
        """Returns the fourier positional embeddings for the given xyz coordinates.
        Args:
            xyz: (B, N, 3) tensor of xyz coordinates
            num_channels: number of channels in the positional embedding
            input_range: [[B x 3], [B x 3]] - min and max XYZ coords
        Returns:
            (B, N, num_channels) tensor of positional embeddings
        """
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range, normalize_mode=self.normalize)
        
        if self.add_normalized_positions:
            xyz_scaled = xyz.clone()

        xyz *= 2 * np.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(bsize, npoints, d_out)
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x d_pos x npoints embedding
        final_embeds = torch.cat(final_embeds, dim=2)#
        if self.add_normalized_positions:
            final_embeds[..., :3] = xyz_scaled
        
        return final_embeds

    @overload
    def __call__(
        self,
        xyz: list[torch.Tensor],
        num_channels: int | None = None,
        input_range: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        ...

    @overload
    def __call__(
        self,
        xyz: torch.Tensor,
        num_channels: int | None = None,
        input_range: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        ...

    def __call__(
        self,
        xyz: torch.Tensor | list[torch.Tensor],
        num_channels: int | None = None,
        input_range: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        return self._call_impl(xyz, num_channels, input_range)

    def forward(self, xyz, num_channels, input_range):
        if isinstance(xyz, list):
            if not (len(xyz) == len(input_range[0]) == len(input_range[1])):
                raise ValueError(
                    "xyz, input_range[0], input_range[1] should be of same length. Current lengths are: ",
                    len(xyz),
                    len(input_range[0]),
                    len(input_range[1]),
                )
            ret = []
            for idx, coords in enumerate(xyz):
                ret.append(
                    self.forward_single(
                        coords[None, ...],
                        num_channels,
                        (input_range[0][idx].unsqueeze(0), input_range[1][idx].unsqueeze(0)),
                    ).squeeze()
                )
            return ret
        return self.forward_single(xyz, num_channels, input_range)

    def forward_single(self, xyz, num_channels=None, input_range=None):
        assert isinstance(xyz, torch.Tensor)
        in_dims = xyz.ndim
        if xyz.ndim == 2:  # single batch. need to unsqueeze
            xyz = xyz.unsqueeze(0)

        assert xyz.ndim == 3
        # xyz is batch x npoints x 3
        if self.pos_type == "fourier" or self.pos_type == "periodic":
            if self.normalize and input_range is None:
                raise ValueError("input_range should be passed if normalize is True")

            #with torch.no_grad():
            out = self.get_fourier_embeddings(xyz, num_channels, input_range)
        else:
            raise ValueError(f"Unknown {self.pos_type}")

        return out if in_dims == 3 else out.squeeze(0)

    def extra_repr(self):
        st = f"type={self.pos_type}, scale={self.scale}, normalize={self.normalize}"
        if hasattr(self, "gauss_B"):
            st += f", gaussB={self.gauss_B.shape}, gaussBsum={self.gauss_B.sum().item()}"
        return st
