"""Decoder modules for occupancy and grasp prediction models."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

import torch
from torch import nn
import math

from icg_net.model.resnet_fc import ResnetBlockFC


class OccnetWidthDecoder(nn.Module):
    """Decoder Predicting Grasps in GIGA format (xyzw, width, quality) shaped (4 + 1 + 1)"""

    def __init__(self, dim, c_dim, n_classes=None, n_blocks=5, with_scale=False, latent_product=False, with_pos=False):
        super().__init__()
        self.decoder_qual = OccnetDecoderFC(
            dim,
            c_dim,
            n_blocks=n_blocks,
            n_classes=12,
            with_scale=with_scale,
            latent_product=latent_product,
            with_pos=False,
        )
        self.decoder_width = OccnetDecoderFC(
            dim,
            c_dim,
            n_blocks=n_blocks,
            n_classes=1,
            with_scale=with_scale,
            latent_product=latent_product,
            with_pos=False,
        )

    def forward(self, query_points, latent, query_pos_enc=None, scale=None, q_sim=None):
        qual, width = (
            self.decoder_qual(query_points, latent, query_pos_enc=query_pos_enc, scale=scale, q_sim=q_sim),
            self.decoder_width(query_points, latent, query_pos_enc=query_pos_enc, scale=scale, q_sim=q_sim),
        )
        return torch.cat([qual, width], dim=-1)


class OccnetDecoderFC(nn.Module):
    """Decoder for ConvOccnet latent representations.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(
        self,
        dim=3,
        c_dim=32,
        hidden_size=32,
        n_blocks=5,
        sample_mode="bilinear",
        padding=0.1,
        n_classes=1,
        activation="ReLU",
        positional_encoder=None,
        with_scale=False,
        latent_product=False,
        with_pos=False,
    ):
        """TODO."""
        super().__init__()

        self.latent_product = latent_product
        self.c_dim = c_dim * (2 - latent_product) + 1 * with_scale

        self.n_blocks = n_blocks
        self.with_scale = with_scale

        if self.c_dim != 0:
            self.fc_c = nn.ModuleList([nn.Linear(self.c_dim, hidden_size) for _ in range(n_blocks)])

        self.fc_p = nn.Linear(dim + self.with_scale, hidden_size)
        # self.fc_p2 = nn.Linear(dim + self.with_scale, hidden_size)

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size, activation=activation) for _ in range(n_blocks)])

        self.with_pos = with_pos
        self.fc_out = nn.Linear(hidden_size, n_classes)

        self.actvn = getattr(nn, activation)()
        self.sample_mode = sample_mode
        self.padding = padding

    def __call__(
        self,
        query_points: torch.Tensor,
        c_plane: torch.Tensor,
        query_pos_enc: torch.Tensor,
        no_grad=False,
        scale=None,
        q_sim=None,
    ) -> torch.Tensor:
        return self._call_impl(
            query_points, c_plane, no_grad=no_grad, query_pos_enc=query_pos_enc, scale=scale, q_sim=q_sim
        )

    def forward(
        self,
        query_points: torch.Tensor,
        latent: torch.Tensor,
        query_pos_enc: torch.Tensor,
        no_grad=False,
        scale=None,
        q_sim=None,
    ) -> torch.Tensor:
        """TODO"""

        latent = torch.cat([latent, query_pos_enc], dim=-1)
        # if self.latent_product:
        # import pdb; pdb.set_trace()
        query_points_prod = query_points.repeat_interleave(latent.shape[0], 0)
        query_points = query_points_prod
        if q_sim is not None:
            query_points = (F.softmax(q_sim, dim=1) / math.sqrt(q_sim.size(-1))).unsqueeze(0).permute(
                [2, 1, 0]
            ) * query_points_prod
        if self.with_pos:
            query_points[..., :3] = query_points_prod[..., :3] - query_pos_enc[..., :3]
        # import pdb; pdb.set_trace()
        # query_points_prod[..., :-6] =  query_points[..., :-6]
        # query_points_prod[..., :-6] = query_pos_enc[..., :-3] * query_points[..., :-6]
        # query_points_prod[..., -3:] = query_pos_enc[..., -3:] - query_points[..., -6:-3]
        # query_points = query_points_prod

        if self.with_scale:
            latent = torch.cat([latent, scale.unsqueeze(0).repeat_interleave(latent.shape[0], 0)], dim=-1)
            query_points = torch.cat(
                [
                    query_points,
                    scale.unsqueeze(1)
                    .repeat_interleave(query_points.shape[1], 1)
                    .repeat_interleave(query_points.shape[0], 0),
                ],
                dim=-1,
            )

        # query_points = self.do1(query_points)
        # latent = self.do(latent)
        pts = query_points.float()
        net = self.fc_p(pts)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](latent)

            net = self.blocks[i](net)
            # if i == 2:
            #     net = net + self.fc_p2(pts)
            # net = self.do(net)

        out = self.fc_out(self.actvn(net))
        return out


class OccnetWidthDecoderAdabins(nn.Module):
    """Decoder Predicting Grasps in GIGA format (xyzw, width, quality) shaped (4 + 1 + 1)"""

    def __init__(self, dim, c_dim, n_classes=None, n_blocks=5, with_scale=False, latent_product=False, with_pos=False):
        super().__init__()

        self.num_bins = 20
        self.min_val = 0
        self.max_val = 0.08
        self.register_buffer("bins", torch.linspace(self.min_val, self.max_val, self.num_bins))

        self.decoder_qual = OccnetDecoderFC(
            dim,
            c_dim,
            n_blocks=n_blocks,
            n_classes=12,
            with_scale=with_scale,
            latent_product=latent_product,
            with_pos=with_pos,
        )

        self.decoder_width = OccnetDecoderFC(
            dim,
            c_dim,
            n_blocks=n_blocks,
            n_classes=20,
            with_scale=with_scale,
            latent_product=latent_product,
            with_pos=with_pos,
        )

    def forward(self, query_points, latent, query_pos_enc=None, scale=None, q_sim=None):
        qual, width_bins = (
            self.decoder_qual(query_points, latent, query_pos_enc=query_pos_enc, scale=scale, q_sim=q_sim),
            self.decoder_width(query_points, latent, query_pos_enc=query_pos_enc, scale=scale, q_sim=q_sim),
        )
        width = (self.bins * width_bins.softmax(-1)).sum(-1).unsqueeze(-1)
        return torch.cat([qual, width], dim=-1)
