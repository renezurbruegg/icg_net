"""Main components of the model."""

from __future__ import annotations
from typing import NamedTuple

import torch
from torch import nn


def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
    """Add positional embedding to the input tensor."""
    return tensor if pos is None else tensor + pos


class SelfAttentionLayer(nn.Module):
    """Self Attention Layer."""

    def __init__(
        self, d_model: int, nhead: int, dropout: float = 0.0, activation: str = "ReLU", normalize_before: bool = False
    ):
        """Initialize self attention layer."""
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ):
        if self.normalize_before:
            tgt = self.norm(tgt)

        q = k = with_pos_embed(tgt, query_pos)
        residual, _ = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)

        tgt = tgt + self.dropout(residual)

        if not self.normalize_before:
            tgt = self.norm(tgt)

        return tgt


class CrossAttentionLayer(nn.Module):
    """Cross Attention Layer."""

    def __init__(
        self, d_model: int, nhead: int, dropout: float = 0.0, activation: str = "ReLU", normalize_before: bool = False
    ) -> None:
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()
        self.normalize_before = normalize_before
        self.last_att_weights: torch.Tensor | None = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ):
        if self.normalize_before:
            tgt = self.norm(tgt)

        (residual, self.last_att_weights) = self.multihead_attn(
            query=with_pos_embed(tgt, query_pos),
            key=with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout(residual)

        if not self.normalize_before:
            tgt = self.norm(tgt)

        return tgt


class FFNLayer(nn.Module):
    """Simple Feedforward Neural Network Layer."""

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "ReLU",
        normalize_before: bool = False,
    ) -> None:
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = getattr(nn, activation)()
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt: torch.Tensor) -> torch.Tensor:
        """Pass the input through the feedforward layer.

        Args:
            tgt: the sequence to the feedforward layer (required).

        Returns:
            the output of the feedforward layer.
        """
        if self.normalize_before:
            tgt = self.norm(tgt)

        residual = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(residual)

        if not self.normalize_before:
            tgt = self.norm(tgt)

        return tgt


class GraspRefinementOut(NamedTuple):
    obj_query: torch.Tensor
    scene_query: torch.Tensor


class GraspRefinementLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        activation: str = "ReLU",
        with_mlp: bool = False,
        n_blocks: int = 2,
        with_obj_grasp_cross_attention: bool = False,
    ):
        super().__init__()

        self.query_transform_layer = FFNLayer(
            d_model, dim_feedforward=512, dropout=dropout, normalize_before=False, activation=activation
        )
        self.with_mlp = with_mlp
        self.self_attention = nn.ModuleList(
            SelfAttentionLayer(d_model, nhead, dropout, activation, normalize_before=False) for _ in range(n_blocks)
        )
        self.ffn_networks = nn.ModuleList(
            FFNLayer(d_model, dim_feedforward=512, dropout=dropout, normalize_before=False, activation=activation)
            for _ in range(n_blocks)
        )
        if with_obj_grasp_cross_attention:
            self.cross_attention = CrossAttentionLayer(d_model, nhead, dropout, activation, normalize_before=False)
            self.ffn_after = FFNLayer(
                d_model, dim_feedforward=512, dropout=dropout, normalize_before=False, activation=activation
            )
        else:
            self.cross_attention = None

        self.n_blocks = n_blocks

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        queries: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        padding_entries: torch.Tensor | None = None,
        rend_features: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        feature_pos_encoding: torch.Tensor | None = None,
    ) -> GraspRefinementOut:
        if self.with_mlp:
            refined_queries = self.query_transform_layer(
                queries
            )  # purely query centric. No positional encodings needed
        else:
            refined_queries = queries
        if self.cross_attention is not None:
            refined_queries = self.cross_attention(
                refined_queries,
                rend_features,
                memory_mask=attention_mask,
                query_pos=query_pos,
                pos=feature_pos_encoding,
            )
            refined_queries = self.ffn_after(refined_queries)
        collision_aware_queries = refined_queries.clone()
        for i in range(self.n_blocks):
            collision_aware_queries = self.self_attention[i](
                collision_aware_queries, tgt_mask=None, tgt_key_padding_mask=padding_entries, query_pos=query_pos
            )
            collision_aware_queries = self.ffn_networks[i](collision_aware_queries)

        return GraspRefinementOut(obj_query=refined_queries, scene_query=collision_aware_queries)
