"""Query refinement module."""

from __future__ import annotations
from typing import List, NamedTuple
import torch
import torch.nn as nn
import MinkowskiEngine.MinkowskiOps as me
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling
from icg_net.model.refinement.layers import (
    SelfAttentionLayer,
    CrossAttentionLayer,
    FFNLayer,
    GraspRefinementLayer,
)
from icg_net.model.resnet_fc import ResnetBlockFC

import sys
from icg_net.minkowski.modules.common import conv
from icg_net.minkowski.res16unet import sort_spare_tensor


from icg_net.minkowski.res16unet import sort_spare_tensor
class QueryRefinementPrediction(NamedTuple):
    """Prediction output of the query refinement module."""

    shape_queries: list[torch.Tensor]
    obj_grasp_queries: list[torch.Tensor]
    scene_grasp_queries: list[torch.Tensor]
    predictions_class: list[torch.Tensor]
    predictions_mask: list[torch.Tensor]
    query_pos_enc: torch.Tensor
    backbone_features: me.SparseTensor

    attention_info: list[tuple[torch.Tensor | None, torch.Tensor]]


class RefinementOut(NamedTuple):
    """Output of the refinement module."""

    queries: torch.Tensor
    attention_maps: torch.Tensor | None
    rend_features: torch.Tensor | None


class ImpRenderRefinement(nn.Module):
    """Refinement module for the implicit renderer."""

    def __init__(
        self,
        feature_dim: int,
        query_dim: int,
        with_feature_renderer: bool = True,
        with_query_renderer: bool = False,
        decoder_sa_layers: int = 0,
        num_heads: int = 8,
        dropout: float = 0.0,
        dim_feedforward: int = 2048,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.feature_renderer = None
        self.query_renderer = None

        self.pts_sa = None
        self.pts_ff = None

        if with_query_renderer:
            self.query_renderer = nn.Sequential(
                ResnetBlockFC(query_dim, query_dim),
                FFNLayer(d_model=query_dim, dim_feedforward=2 * query_dim),
            )

        if with_feature_renderer:
            self.feature_renderer = nn.Sequential(
                ResnetBlockFC(feature_dim, query_dim),
                FFNLayer(d_model=query_dim, dim_feedforward=2 * query_dim),
            )
        else:
            self.feature_renderer = nn.Sequential(nn.Linear(feature_dim, query_dim))

        if decoder_sa_layers > 0:
            self.pts_sa = nn.ModuleList(
                SelfAttentionLayer(d_model=query_dim, nhead=num_heads, dropout=dropout, normalize_before=pre_norm)
                for _ in range(decoder_sa_layers)
            )
            self.pts_ff = nn.ModuleList(
                FFNLayer(d_model=query_dim, dim_feedforward=2 * query_dim, dropout=dropout, normalize_before=pre_norm)
                for _ in range(decoder_sa_layers)
            )

        self.cross_attention = CrossAttentionLayer(
            d_model=query_dim, nhead=num_heads, dropout=dropout, normalize_before=pre_norm
        )
        self.self_attention = SelfAttentionLayer(
            d_model=query_dim, nhead=num_heads, dropout=dropout, normalize_before=pre_norm
        )
        self.ffn = FFNLayer(
            d_model=query_dim, dim_feedforward=dim_feedforward, dropout=dropout, normalize_before=pre_norm
        )

    def __call__(
        self,
        queries: torch.Tensor,
        query_pos_encoding: torch.Tensor,
        features: torch.Tensor,
        feature_pos_encoding: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> RefinementOut:
        return self._call_impl(queries, query_pos_encoding, features, feature_pos_encoding, attention_mask)

    def forward(
        self,
        queries: torch.Tensor,
        query_pos_encoding: torch.Tensor,
        features: torch.Tensor,
        feature_pos_encoding: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> RefinementOut:
        rend_features = self.feature_renderer(features) if self.feature_renderer is not None else features
        queries = self.query_renderer(queries) if self.query_renderer is not None else queries

        if self.pts_sa is not None and self.pts_ff is not None:
            for sa, ff in zip(self.pts_sa, self.pts_ff):
                rend_features = sa(rend_features, query_pos=feature_pos_encoding)
                rend_features = ff(rend_features)

        refined_queries = self.cross_attention(
            queries, rend_features, memory_mask=attention_mask, query_pos=query_pos_encoding, pos=feature_pos_encoding
        )
        self_att_queries = self.self_attention(refined_queries, query_pos=query_pos_encoding)

        return RefinementOut(
            queries=self.ffn(self_att_queries),
            attention_maps=self.cross_attention.last_att_weights,
            rend_features=rend_features,
        )


class QueryRefinementModule(nn.Module):
    """Query refinement module."""

    def __init__(
        self,
        num_classes: int,
        query_dim: int,
        sample_sizes: List[int],
        hlevels: List[int],
        feature_sizes: List[int],
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        pre_norm: bool = False,
        max_sample_size: int = 8192,
        with_grasps=False,
        w_feature_render_mlp=True,
        with_query_renderer=False,
        dim_feedforward=1024,
        decoder_sa_layers=0,
        with_obj_grasp_cross_attention=False,
        refine_position=False,
    ):
        super().__init__()
        self.max_sample_size = max_sample_size
        self.hlevels = hlevels
        self.num_classes = num_classes
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.sample_sizes = sample_sizes
        self.mask_dim = hidden_dim
        self.num_heads = num_heads
        self.with_grasps = with_grasps
        self.mask_mod_norm = nn.LayerNorm(hidden_dim)
        # Number af refinement levels.
        self.num_levels = len(self.hlevels)
        self.refine_position = refine_position

        # Final minkowski convolution head to get the final features with the same dimension as the query.
        self.mask_features_head = conv(
            feature_sizes[-1],
            self.mask_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            D=3,
        )
        # self.query_projection_sdf = self.query_projection

        self.mask_embed_head = nn.Sequential(
            ResnetBlockFC(hidden_dim, hidden_dim),
            ResnetBlockFC(hidden_dim, hidden_dim),
            ResnetBlockFC(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # self.class_embed_head = nn.Linear(hidden_dim, self.num_classes)
        self.class_embed_head = nn.Sequential(
            ResnetBlockFC(hidden_dim, hidden_dim),
            # ResnetBlockFC(hidden_dim, hidden_dim),
            # ResnetBlockFC(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, self.num_classes),
        )

        self.pooling = MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)

        self.refinement_stages = nn.ModuleList()
        for hlevel in self.hlevels:
            self.refinement_stages.append(
                ImpRenderRefinement(
                    feature_dim=feature_sizes[hlevel],
                    query_dim=query_dim,
                    with_feature_renderer=w_feature_render_mlp,
                    with_query_renderer=with_query_renderer,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    pre_norm=self.pre_norm,
                    dim_feedforward=dim_feedforward,
                    decoder_sa_layers=decoder_sa_layers,
                )
            )
        if self.refine_position:
            self.pos_feature_renderer = nn.Sequential(
                nn.Linear(feature_sizes[self.hlevels[0]], query_dim),
                nn.ReLU(),
                FFNLayer(d_model=query_dim, dim_feedforward=2 * query_dim),
            )
            self.pos_refine_layer = ImpRenderRefinement(
                feature_dim=query_dim,
                query_dim=query_dim,
                with_feature_renderer=False,
                with_query_renderer=False,
                num_heads=self.num_heads,
                dropout=self.dropout,
                pre_norm=self.pre_norm,
                dim_feedforward=dim_feedforward,
                decoder_sa_layers=False,
            )

        if self.with_grasps:
            self.grasp_refinement_layer = GraspRefinementLayer(
                d_model=self.mask_dim,
                nhead=self.num_heads,
                dropout=self.dropout,
                activation="ReLU",
                with_mlp=True,
                with_obj_grasp_cross_attention=with_obj_grasp_cross_attention,
            )

    def get_voxel_batch(
        self,
        decomposed_multi_res_features: list[torch.Tensor],
        decomposed_coords: list[torch.Tensor],
        decomposed_attn: list[torch.Tensor],
        pos_encodings_pcd: list[torch.Tensor],
        sample_size: int = sys.maxsize,
        is_eval: bool = False,
        device: str | torch.device = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        # Find biggest number of points over all multi_res_featureseliary features. Use this to upsample the predictions.
        curr_sample_size = max([pcd.shape[0] for pcd in decomposed_multi_res_features])

        if min([pcd.shape[0] for pcd in decomposed_multi_res_features]) == 1:
            raise RuntimeError("only a single point gives nans in cross-attention")

        # Make sure we do not end up with too many points. Upper limit by the provided sample sizes
        if not is_eval:
            curr_sample_size = min(curr_sample_size, sample_size)
        # Randomly sub or upsample pointcloud to the same size.
        rand_idx = []
        mask_idx: list[torch.Tensor] = []  # Mask to make sure we only attend to relevant voxels.
        for feat in decomposed_multi_res_features:
            pcd_size = feat.shape[0]
            if pcd_size <= curr_sample_size:
                # we do not need to sample
                # take all points and pad the rest with zeroes and mask it
                idx = torch.zeros(curr_sample_size, dtype=torch.long, device=device)  # pylint: disable=no-member
                midx = torch.ones(curr_sample_size, dtype=torch.bool, device=device)  # pylint: disable=no-member
                idx[:pcd_size] = torch.arange(pcd_size, device=device)  # pylint: disable=no-member
                midx[:pcd_size] = False  # attend to first points
            else:
                # we have more points in pcd as we like to sample
                # take a subset (no padding or masking needed)
                idx = torch.randperm(feat.shape[0], device=device)[:curr_sample_size]  # pylint: disable=no-member
                midx = torch.zeros(curr_sample_size, dtype=torch.bool, device=device)  # pylint: disable=no-member

            rand_idx.append(idx)
            mask_idx.append(midx)

        # decomposed_multi_res_features has shape n_voxels x 256
        batched_multi_res_features = torch.stack(  # pylint: disable=no-member
            [decomposed_multi_res_features[k][rand_idx[k], :] for k in range(len(rand_idx))]
        )
        batched_attn = torch.stack(  # pylint: disable=no-member
            [decomposed_attn[k][rand_idx[k], :] for k in range(len(rand_idx))]
        )
        batched_coords = torch.stack(  # pylint: disable=no-member
            [decomposed_coords[k][rand_idx[k], :] for k in range(len(rand_idx))]
        )
        batched_pos_enc = torch.stack(  # pylint: disable=no-member
            [pos_encodings_pcd[k][rand_idx[k], :] for k in range(len(rand_idx))]
        )
        batched_attn.permute((0, 2, 1))[batched_attn.sum(1) == rand_idx[0].shape[0]] = False
        return batched_multi_res_features, batched_attn, batched_coords, batched_pos_enc, mask_idx

    def __call__(
        self,
        pcd_features: me.SparseTensor,
        multi_res_features: list[me.SparseTensor],
        coordinates: list[torch.Tensor],
        coordinates_position_enc: list[list[torch.Tensor]],
        queries: torch.Tensor,
        queries_pos_enc: torch.Tensor,
        repr_sort_idxs: bool = False,
    ) -> QueryRefinementPrediction:
        return self._call_impl(
            pcd_features, multi_res_features, coordinates, coordinates_position_enc, queries, queries_pos_enc, repr_sort_idxs
        )

    def forward(
        self,
        pcd_features: me.SparseTensor,
        multi_res_features: list[me.SparseTensor],
        coordinates: list[me.SparseTensor],
        coordinates_position_enc: list[list[torch.Tensor]],
        queries: torch.Tensor,
        queries_pos_enc: torch.Tensor,
        repr_sort_idxs: bool = False,
    ) -> QueryRefinementPrediction:
        """Forward pass of the model."""
        # Get the final features from the encoder with the same shape as the positional encodings.
        mask_features = self.mask_features_head(pcd_features)

        # Drop these. Only for interpolation
        # multi_res_features = multi_res_features[3:]

        # TODO, remove!

        # Ignore projection of frequency components for now.

        object_grasp_queries = []
        scene_grasp_queries = []
        predictions_class = []
        predictions_mask = []
        intermittent_queries = []
        attention_info = []

        for i, hlevel in enumerate(self.hlevels):
            # get class and mask predictions. Also get the attention mask used for visualization.
            output_class, outputs_mask, attn_mask = self.mask_module(
                queries,
                mask_features,
                len(multi_res_features) - hlevel - 1,
                ret_attn_mask=True,
            )
            if repr_sort_idxs:
                attn_mask = sort_spare_tensor(attn_mask)[0]

            if repr_sort_idxs:
                attn_mask = sort_spare_tensor(attn_mask)[0]
            # Get the multi_res_featureseliary features for the current level.
            decomposed_multi_res_features: list[torch.Tensor] = multi_res_features[hlevel].decomposed_features
            decomposed_coords: list[torch.Tensor] = coordinates[hlevel].decomposed_features
            decomposed_attn: list[torch.Tensor] = attn_mask.decomposed_features

            # Up / Downsample voxels to have same number of points for all batches.
            # All components have the number of points/instances at the 2nd position.
            # I.E. [n_batch, n_inst/n_voxels, feature_dim]
            (
                batched_multi_res_features,
                batched_attn,
                batched_coords,
                batched_pos_enc,
                mask_idx,
            ) = self.get_voxel_batch(
                decomposed_multi_res_features,
                decomposed_coords,
                decomposed_attn,
                coordinates_position_enc[hlevel],
                sample_size=self.sample_sizes[hlevel],
                is_eval=not self.training,
                device=queries.device,
            )

            # Make sure to mask out the padded region.
            batched_attn = torch.logical_or(batched_attn, torch.stack(mask_idx)[..., None])  # pylint: disable=no-member

            ref_layer: ImpRenderRefinement = self.refinement_stages[i]  # type: ignore
            if self.refine_position and i == 0:
                queries_pos_enc, _, _ = self.pos_refine_layer.forward(
                    queries=queries_pos_enc,
                    query_pos_encoding=queries,
                    features=batched_pos_enc,
                    feature_pos_encoding=self.pos_feature_renderer(batched_multi_res_features),
                    attention_mask=batched_attn.repeat_interleave(self.pos_refine_layer.num_heads, dim=0).permute(
                        0, 2, 1
                    ),
                )

            queries, cross_att, rend_features = ref_layer.forward(
                queries=queries,
                query_pos_encoding=queries_pos_enc,
                features=batched_multi_res_features,
                feature_pos_encoding=batched_pos_enc,
                attention_mask=batched_attn.repeat_interleave(ref_layer.num_heads, dim=0).permute(0, 2, 1),
            )

            attention_info.append((cross_att, batched_coords))

            # Save Predictions
            predictions_class.append(output_class)
            predictions_mask.append(outputs_mask)
            intermittent_queries.append(queries.clone())

            if self.with_grasps:
                # Mask out instances that are not assigned to any class.
                padding_entries = output_class.argmax(-1) == (self.num_classes - 1)
                for idx in range(len(padding_entries)):  # Check that we do not pad all entries!
                    if padding_entries[idx].all():
                        padding_entries[idx] = ~padding_entries[idx]

                grasp_intermittent, grasp_refined = self.grasp_refinement_layer(
                    queries.clone(),
                    queries_pos_enc,
                    padding_entries=padding_entries,
                    rend_features=rend_features,
                    feature_pos_encoding=batched_pos_enc,
                    attention_mask=batched_attn.repeat_interleave(ref_layer.num_heads, dim=0).permute(0, 2, 1),
                )
                object_grasp_queries.append(grasp_intermittent)
                scene_grasp_queries.append(grasp_refined)

        # use second last for class pred
        q = intermittent_queries[-1]

        output_class, outputs_mask = self.mask_module(q, mask_features, 0, ret_attn_mask=False)
        predictions_class.append(output_class)
        predictions_mask.append(outputs_mask)

        return QueryRefinementPrediction(
            shape_queries=intermittent_queries,
            obj_grasp_queries=object_grasp_queries,
            scene_grasp_queries=scene_grasp_queries,
            predictions_class=predictions_class,
            predictions_mask=predictions_mask,
            attention_info=attention_info,
            query_pos_enc=queries_pos_enc,
            backbone_features=pcd_features,
        )

    def get_masks(self, query_feat: torch.Tensor, pointwise_feats: list[torch.Tensor]):
        query_feat = self.mask_mod_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)

        output_masks = []
        for i in range(len(pointwise_feats)):
            output_masks.append(pointwise_feats[i] @ mask_embed[i].T)
        output_masks = torch.cat(output_masks)  # pylint: disable=no-member

        return output_masks

    def mask_module(self, query_feat, mask_features, num_pooling_steps, ret_attn_mask=True):
        outputs_class = self.class_embed_head(query_feat)
        output_masks = self.get_masks(query_feat, mask_features.decomposed_features)
        outputs_mask = me.SparseTensor(
            features=output_masks,
            coordinate_manager=mask_features.coordinate_manager,
            coordinate_map_key=mask_features.coordinate_map_key,
        )

        if ret_attn_mask:
            attn_mask = outputs_mask
            for pool_level in range(num_pooling_steps):
                attn_mask = self.pooling(attn_mask.float())

            attn_mask = me.SparseTensor(
                features=(attn_mask.F.detach().sigmoid() < 0.5),
                coordinate_manager=attn_mask.coordinate_manager,
                coordinate_map_key=attn_mask.coordinate_map_key,
            )

            return outputs_class, outputs_mask.decomposed_features, attn_mask

        return outputs_class, outputs_mask.decomposed_features
