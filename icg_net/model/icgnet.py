"""ICGNet model."""

from __future__ import annotations
from typing import Optional, List
import torch
import torch.nn as nn
import MinkowskiEngine.MinkowskiOps as me
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling
from icg_net.minkowski.res16unet import Res16UNetBase

from icg_net.minkowski.res16unet import sort_spare_tensor

import sys

if sys.version_info >= (3, 8):
    from typing import TypedDict, overload  # pylint: disable=no-name-in-module
else:
    from typing_extensions import overload

from icg_net.model.decoders import (
    # AcronymDecoder,
    # DeepSDFDecoder,
    # GigaDecoder,
    OccnetDecoderFC,
    OccnetWidthDecoder,
    # ResnetBlockFC,
    # DynamicDecoder,
    # OccnetWidthDecoderAdabins,
)
from icg_net.third_party.pointnet2.pointnet2_utils import (
    furthest_point_sample as furthest_point_sample_gpu,
)
from icg_net.model.interpolation.interpolator import PointNetKnnInterpolator

from icg_net.model.refinement.position_embedding import PositionEmbeddingCoordsSine, extract_scene_bounds
from icg_net.utils.sampling import KMeans
from icg_net.model.interpolation.interpolator import FeatureInterpolator

import torch
from torch import nn, Tensor
from icg_net.model.refinement.query_refiner import QueryRefinementModule
import torch.nn as nn
import torch.nn.functional as F

from icg_net.model.feature_volume.combined import CombinedFeatureVolume

from icg_net.typing import ICGNetOutput

# Parse Backbone
from omegaconf.dictconfig import DictConfig
import hydra

from icg_net.model.decoders import OccnetWidthDecoder


def furthest_point_sample(points: torch.Tensor, num_samples: int):
    if points.is_cuda:
        return furthest_point_sample_gpu(points, num_samples)
    else:
        # TODO, only supporting gpu inference for now
        return furthest_point_sample_gpu(points.cuda(), num_samples).cpu()


class QueryInitializer(nn.Module):
    def __init__(
        self,
        query_dim: int,
        num_queries: int,
        learnable_queries: bool = True,
        pos_sampling_mode="cluster2D",
        normalize_embeddings=False,
        only_one_embedding: bool = False,
        feature_dim: int = 0,
    ) -> None:
        super().__init__()
        self.query_dim = query_dim
        self.learnable_queries = learnable_queries
        self.num_queries = num_queries
        self.pos_sampling_mode = pos_sampling_mode
        assert self.pos_sampling_mode in ["cluster2D", "cluster3D", "oracle", "fps"]
        self.only_one_embedding = only_one_embedding

        if self.learnable_queries:
            if not only_one_embedding:
                self.query_feat = nn.Embedding(
                    self.num_queries, query_dim, max_norm=True if normalize_embeddings else None
                )
            else:
                self.query_feat = nn.Embedding(1, query_dim, max_norm=True if normalize_embeddings else None)
        else:
            if not self.learnable_queries:
                self.embedding_mlp = nn.Sequential(
                    nn.Linear(feature_dim, query_dim), nn.ReLU(), nn.Linear(query_dim, query_dim)
                )

    def __call__(
        self,
        coordinates: List[Tensor],
        features: Optional[List[Tensor]] = None,
        poses: Optional[List[Tensor]] = None,
    ) -> tuple[Tensor, Tensor]:
        return self._call_impl(coordinates, features, poses)

    def forward(
        self,
        coordinates: List[Tensor],
        features: Optional[List[Tensor]] = None,
        poses: Optional[List[Tensor]] = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size = len(coordinates)

        fps_idx = [
            furthest_point_sample(coordinates[i][None, ...].float(), self.num_queries).squeeze(0)
            for i in range(batch_size)
        ]

        if self.pos_sampling_mode == "oracle":
            if poses is None:
                print("[ERROR] Can only use oracle clustering if poses are given!")
                sampled_coords: Tensor = torch.stack(  # pylint: disable=no-member
                    [coordinates[i][fps_idx[i].long(), :] for i in range(len(fps_idx))]
                )
            else:
                # assert poses is not None, "Can only use oracle clustering if poses are given."
                coords: List[Tensor] = []
                for i in range(len(fps_idx)):  # pylint: disable=consider-using-enumerate
                    coord = coordinates[i][fps_idx[i].long(), :]
                    pose = poses[i]
                    coord[: len(pose)] = pose
                    coords.append(coord)
                sampled_coords = torch.stack(coords)  # pylint: disable=no-member

        elif "cluster" in self.pos_sampling_mode:
            coords_sampled = []
            cluster_dim = 2 if "2D" in self.pos_sampling_mode else 3
            for i in range(len(fps_idx)):  # pylint: disable=consider-using-enumerate
                coord = coordinates[i][fps_idx[i].long(), :]

                coord = coord[:, :cluster_dim].contiguous().float()
                sampled_coords = KMeans(
                    coordinates[i][:, :cluster_dim].contiguous().float(),
                    K=coord.shape[0],
                    Niter=10,
                    c=coord,
                )[1]

                if cluster_dim == 2:  # only use xy coordinates, add z coordinate as mean value
                    coord = torch.cat(  # pylint: disable=no-member
                        [
                            coord,
                            torch.ones_like(coord[:, 0][:, None])  # pylint: disable=no-member
                            * coordinates[i][:, -1].mean(),
                        ],
                        dim=1,
                    )

                coords_sampled.append(coord)
            sampled_coords = torch.stack(coords_sampled)  # pylint: disable=no-member

        elif "fps" in self.pos_sampling_mode:
            sampled_coords = torch.stack(  # pylint: disable=no-member
                [coordinates[i][fps_idx[i].long(), :] for i in range(len(fps_idx))]
            )

        queries = (
            torch.zeros(  # pylint: disable=no-member
                (batch_size, self.num_queries, self.query_dim), device=coordinates[0].device
            )
            if not self.learnable_queries
            else self.query_feat.weight.clone()
            .unsqueeze(0)
            .repeat(batch_size, 1 if not self.only_one_embedding else self.num_queries, 1)
        )

        if not self.learnable_queries:
            for batch_idx, coord in enumerate(sampled_coords):
                feature_idx = (coordinates[batch_idx].unsqueeze(0) - coord.unsqueeze(1)).norm(dim=-1).argmax(1)
                queries[batch_idx] = self.embedding_mlp(features.decomposed_features[batch_idx][feature_idx])

        return queries, sampled_coords


def get_voxel_batch(
    decomposed_features: list[torch.Tensor],
    pos_encodings_pcd: list[torch.Tensor],
    sample_size: int = sys.maxsize,
    is_eval: bool = False,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    # Find biggest number of points over all multi_res_featureseliary features. Use this to upsample the predictions.
    curr_sample_size = max([pcd.shape[0] for pcd in decomposed_features])

    if min([pcd.shape[0] for pcd in decomposed_features]) == 1:
        raise RuntimeError("only a single point gives nans in cross-attention")

    # Make sure we do not end up with too many points. Upper limit by the provided sample sizes
    if not is_eval:
        curr_sample_size = min(curr_sample_size, sample_size)
    # Randomly sub or upsample pointcloud to the same size.
    rand_idx = []
    mask_idx: list[torch.Tensor] = []  # Mask to make sure we only attend to relevant voxels.
    for feat in decomposed_features:
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
        [decomposed_features[k][rand_idx[k], :] for k in range(len(rand_idx))]
    )
    batched_pos_enc = torch.stack(  # pylint: disable=no-member
        [pos_encodings_pcd[k][rand_idx[k], :] for k in range(len(rand_idx))]
    )

    return batched_multi_res_features, batched_pos_enc, mask_idx





class ICGNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_queries: int,
        sample_sizes: List[int],
        num_classes: int,
        hlevels: List[int],
        max_sample_size: int,
        backbone: Res16UNetBase,
        with_grasps: bool = True,
        # Queries
        learnable_embeddings: bool = True,
        pos_sampling_mode: str = "cluster2D",
        # Pos Encodings
        positional_encoding_type: str = "fourier",
        normalize_pos_enc: str = "max",
        gauss_scale: float = 1.0,
        # Arch choices
        grasp_decoder: str = "occnet",
        sdf_decoder: str = "occnet",
        num_heads: int = 8,
        shared_decoder: bool = True,
        num_decoders: int = 1,
        dropout: float = 0.0,
        pre_norm: bool = False,
        dim_feedforward: int = 1024,
        decoder_with_scale: bool = False,
        w_feature_render_mlp: bool = False,
        with_query_renderer: bool = False,
        use_pos_enc_projection: bool = False,
        decoder_sa_layers: int = 0,
        add_normalized_positions: bool = False,
        normalize_embeddings: bool = False,
        with_obj_grasp_cross_attention: bool = False,
        only_one_embedding: bool = False,
        refine_position: bool = False,
        learnable_pos_embeddings: bool = False,
        quantization_size: float = 0.005,
        interpolate_decoder_features: bool = True,
        interpolation_mode="none",
        dense_decoder_cfg: dict = None,
        repr_sort_idxs: bool = False,
    ):
        super().__init__()

        self.pooling = MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)

        self.repr_sort_idxs = repr_sort_idxs
        self.coord_sort_maps = None

        self.repr_sort_idxs = repr_sort_idxs
        if isinstance(backbone, DictConfig):
            backbone = hydra.utils.instantiate(backbone)

        self.feature_volume = CombinedFeatureVolume(
            sparse_feature_extractor=backbone,
            dense_decoder_cfg=dense_decoder_cfg,
            hidden_dim=hidden_dim,
            hlevels=hlevels,
        )

        self.add_normalized_positions = add_normalized_positions
        self.with_grasps = with_grasps
        self.max_sample_size = max_sample_size
        self.num_decoders = num_decoders
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.num_levels = len(hlevels)
        self.coord_sort_maps = None
        self.hlevels = hlevels

        self.only_one_embedding = only_one_embedding
        self.learnable_pos_embeddings = learnable_pos_embeddings
        self.interpolate_decoder_features = interpolate_decoder_features
        self.quantization_size = quantization_size

        if self.learnable_pos_embeddings:
            self.pos_embeddings = nn.Embedding(self.num_queries, hidden_dim, max_norm=True)

        self.interpolation_mode = interpolation_mode

        self.grasp_decoder_type = grasp_decoder
        self.occ_decoder_type = sdf_decoder

        # Positional Encoding
        self.pos_enc = PositionEmbeddingCoordsSine(
            pos_type=positional_encoding_type,
            d_pos=hidden_dim,
            gauss_scale=gauss_scale,
            normalize=normalize_pos_enc,
            add_normalized_positions=add_normalized_positions,
        )

        self.DECODER_START = -5
        self.scene_bounds: tuple[Tensor, Tensor] | None = None

        backbone_feat_shapes = self.feature_volume.sparse_backbone_feat_shapes

        self.features_each_level = [backbone_feat_shapes[l] for l in hlevels]

        # Parse Refinement Layers
        self.refinement_layers = nn.ModuleList()
        for decoder_id in range(num_decoders):
            if decoder_id > 0 and shared_decoder:
                self.refinement_layers.append(self.refinement_layers[0])
            else:
                self.refinement_layers.append(
                    QueryRefinementModule(
                        num_classes=num_classes,
                        decoder_sa_layers=decoder_sa_layers,
                        query_dim=hidden_dim,
                        sample_sizes=sample_sizes,
                        hlevels=hlevels,
                        feature_sizes=backbone_feat_shapes,
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        pre_norm=pre_norm,
                        max_sample_size=max_sample_size,
                        with_grasps=with_grasps,
                        w_feature_render_mlp=w_feature_render_mlp,
                        with_query_renderer=with_query_renderer,
                        dim_feedforward=dim_feedforward,
                        with_obj_grasp_cross_attention=with_obj_grasp_cross_attention,
                        refine_position=refine_position,
                    )
                )

        self.interpolation_fn = PointNetKnnInterpolator(feature_dim=hidden_dim, k=6)

        self.grasp_feature_interpolator = FeatureInterpolator(
            min_coords=dense_decoder_cfg.borders[0],
            max_coords=dense_decoder_cfg.borders[1],
            unet_shapes=self.feature_volume.dense_feature_shapes[-self.num_levels :],
            sparse_shapes=self.features_each_level,
            hidden_dim=hidden_dim,
            quantization_size=quantization_size,
            sparse_interpolator=self.interpolation_fn,
        )

        self.occupancy_feature_interpolator = FeatureInterpolator(
            min_coords=dense_decoder_cfg.borders[0],
            max_coords=dense_decoder_cfg.borders[1],
            unet_shapes=self.feature_volume.dense_feature_shapes[-self.num_levels :],
            sparse_shapes=self.features_each_level,
            hidden_dim=hidden_dim,
            quantization_size=quantization_size,
            sparse_interpolator=self.interpolation_fn,
        )

        self.dense_volume_interpolator = self.grasp_feature_interpolator.dense_volume_inter

        self.hidden_dim = hidden_dim

        # Set Decoders
        self.sdf_decoder = OccnetDecoderFC(
            dim=self.pos_enc.d_pos + 3,
            c_dim=hidden_dim,
            n_classes=1,
            n_blocks=5,
            with_scale=decoder_with_scale,
            latent_product=False,
            with_pos=add_normalized_positions,
        )

        self.grasp_decoder = OccnetWidthDecoder(
            dim=self.pos_enc.d_pos + 3,
            c_dim=hidden_dim,
            n_classes=12,
            n_blocks=5,
            with_scale=decoder_with_scale,
            latent_product=False,
        )

        # Query Initialization
        self.query_initializer = QueryInitializer(
            query_dim=hidden_dim,
            num_queries=num_queries,
            learnable_queries=learnable_embeddings,
            pos_sampling_mode=pos_sampling_mode,
            normalize_embeddings=normalize_embeddings,
            feature_dim=self.feature_volume.sparse_backbone_feat_shapes[-1],  # .PLANES[-1],
        )

        self.use_pos_enc_projection = use_pos_enc_projection

    @overload
    def pos_encode(self, coords: List[Tensor]) -> List[Tensor]: ...

    @overload
    def pos_encode(self, coords: Tensor) -> Tensor: ...

    def pos_encode(self, coords: List[Tensor] | Tensor) -> List[Tensor] | Tensor:
        """Encode coordinates into position encodings.

        Args:
            coords (List[Tensor] | Tensor): Coordinates to encode.

        Returns:
            List[Tensor] | Tensor: Encoded coordinates.
        """
        assert self.scene_bounds is not None, "Scene bounds must be set before encoding coordinates."

        encodings = self.pos_enc(xyz=coords, input_range=self.scene_bounds)
        if self.use_pos_enc_projection:
            if isinstance(encodings, Tensor):
                encodings = self.pos_enc_projection(encodings)
            else:
                encodings = [self.pos_enc_projection(e) for e in encodings]

        return encodings

    def _decode(
        self,
        queries: List[torch.Tensor],
        query_positional_encoding: List[torch.Tensor] | None,
        points: List[torch.Tensor],
        decoder: nn.Module,
        type="occ",
        level=-1,
    ) -> tuple[List[torch.Tensor], torch.Tensor]:
        """Decode queries and grasp points into sdf values.
        Args:
            queries (List[torch.Tensor]): Queries to decode. [(n_queries, query_dim), ...]
            query_positional_encoding (List[torch.Tensor]): Positional encodings for queries. [(n_queries, d_pos), ...]
            points (List[torch.Tensor]): points to decode. [(n_grasps, 3), ...]
        Returns:
            List[torch.Tensor]: Decoded values. [(n_queries, n_points, decoder_dim), ...]
            List[torch.Tensor]: Decoded bounds. []
        """
        bs = len(points)

        if bs == 0:
            return [], []

        if self.scene_bounds is None:
            raise ValueError("Scene bounds must be set before decoding grasps.")

        bounds = self.scene_bounds
        if bounds[0].shape[0] != bs:
            print(
                "WARNING: bounds and grasp points have different batch size. Lazy visualizing?",
                bounds[0].shape[0],
                bs,
            )
            bounds = (bounds[0][:bs], bounds[1][:bs])
            queries = queries[:bs]

        grasp_latents = queries

        xyz = [p.clone() for p in points]

        pts_pos_encodings = self.pos_enc(points, input_range=bounds)

        # Multi decoder setup
        if level > 0:
            level = level % len(self.hlevels)
        lvls = self.hlevels[: (level + 1)] if level != -1 else self.hlevels

        if self.interpolate_decoder_features:

            if isinstance(pts_pos_encodings, list):
                pts_pos_encodings = torch.stack(pts_pos_encodings, dim=0)

            idx = -len(self.hlevels) + level if level != -1 else -1

            if type == "occ":
                points = self.occupancy_feature_interpolator(
                    pts_pos_encodings,
                    xyz,
                    self.multi_res_feats[lvls[-1]].coordinates,
                    self.dense_aux[idx],
                    self.multi_res_feats[self.hlevels[idx]].features,
                    bs,
                    idx,
                )

            else:
                points = self.grasp_feature_interpolator(
                    pts_pos_encodings,
                    xyz,
                    self.multi_res_feats[lvls[-1]].coordinates,
                    self.dense_aux[idx],
                    self.multi_res_feats[self.hlevels[idx]].features,
                    bs,
                    idx,
                )

        scales = (bounds[1][:, None, :] - bounds[0][:, None, :]).max(-1).values
        points = [torch.cat([pt, (c - b) / s], axis=-1) for pt, c, s, b in zip(points, xyz, scales, bounds[0])]

        return [
            decoder(
                pts.unsqueeze(0),
                latent.unsqueeze(1),
                query_pos_enc=query_enc.unsqueeze(1),
                scale=s.unsqueeze(0),
            )
            for latent, pts, query_enc, s in zip(grasp_latents, points, query_positional_encoding, scales)
        ], scales

    def decode_normals(
        self,
        queries: List[torch.Tensor],
        query_positional_encoding: List[torch.Tensor] | None,
        sdf_points: List[torch.Tensor],
        level: int = -1,
    ):
        pts = [torch.autograd.Variable(p.data, requires_grad=True) for p in sdf_points]

        normals = []
        with torch.enable_grad():
            for idx, occ in enumerate(self.decode_sdf(queries, query_positional_encoding, pts, level)):
                d_output = torch.ones_like(occ[0], requires_grad=False, device=occ.device)
                gradients = torch.autograd.grad(
                    outputs=list(occ.sigmoid()),
                    inputs=[pts[idx] for _ in range(len(occ))],
                    grad_outputs=[d_output for _ in range(len(occ))],
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=False,
                    only_inputs=True,
                )

                # gradients have shape [n_latents, n_pts, 3]
                # normalize them.
                grads = F.normalize(torch.stack(gradients), dim=-1)
                normals.append(grads)
        return normals

    def decode_sdf(
        self,
        queries: List[torch.Tensor],
        query_positional_encoding: List[torch.Tensor] | None,
        sdf_points: List[torch.Tensor],
        level: int = -1,
    ) -> List[torch.Tensor]:
        return self._decode(queries, query_positional_encoding, sdf_points, self.sdf_decoder, type="occ", level=level)[
            0
        ]

    def decode_grasps(
        self,
        queries: List[torch.Tensor],
        query_positional_encoding: List[torch.Tensor] | None,
        grasp_points: List[torch.Tensor],
        level: int = -1,
    ) -> List[torch.Tensor]:
        preds, scales = self._decode(
            queries, query_positional_encoding, grasp_points, self.grasp_decoder, type="grasp", level=level
        )

        return preds

    def __call__(
        self,
        voxelized_data: ME.SparseTensor,
        raw_coordinates: List[Tensor],
        poses=Optional[List[Tensor]],
    ) -> ICGNetOutput:
        return self._call_impl(voxelized_data, raw_coordinates, poses)

    def forward(
        self,
        voxelized_data: ME.SparseTensor,
        raw_coordinates: List[Tensor],
        poses=Optional[List[Tensor]],
    ) -> ICGNetOutput:
        pcd_features: ME.SparseTensor
        aux: List[ME.SparseTensor]

        pcd_features, aux, dense_feats, dense_aux = self.feature_volume(voxelized_data, raw_coordinates)

        if type(pcd_features) == tuple:
            self.coord_sort_maps = [a[1] for a in aux]
            aux = [a[0] for a in aux]
            pcd_features = pcd_features[0]
        else:
            raw_coordinates = [c for c in raw_coordinates]

        start_coords = me.SparseTensor(
            features=torch.cat(raw_coordinates),
            coordinates=aux[-1].coordinates,
            device=aux[-1].device,
        )

        if self.coord_sort_maps is None:
            coordinates = start_coords
        else:
            coordinates = me.SparseTensor(
                features=torch.cat(raw_coordinates)[self.coord_sort_maps[-1]],
                coordinates=aux[-1].coordinates[self.coord_sort_maps[-1]],
                device=aux[-1].device,
            )

        self.multi_res_feats = aux

        if "dense" in self.interpolation_mode:

            self.dense_feats, self.dense_aux = (
                dense_feats,
                dense_aux,
            )  # self.dense_extractor(voxelized_data, coordinates)

            for idx, lvl in enumerate(self.hlevels):
                unet_coordinates = [
                    (coord * self.quantization_size) for coord in self.multi_res_feats[lvl].decomposed_coordinates
                ]
                features = self.dense_volume_interpolator(
                    unet_coordinates, self.dense_aux[-len(self.hlevels) + idx], method="nearest"
                )

                sparse_features = torch.cat([self.multi_res_feats[lvl].features, torch.cat(features)], -1)

                # update features
                self.multi_res_feats[lvl] = me.SparseTensor(
                    features=sparse_features,
                    coordinates=self.multi_res_feats[lvl].coordinates,
                    device=self.multi_res_feats[lvl].device,
                )

            pcd_features = self.multi_res_feats[-1]

        # convert voxelized_data to dense coordinates

        # Load coordinates for each feature level.
        with torch.no_grad():
            coords = [start_coords]
            for _ in reversed(range(len(aux) - 1)):
                if self.repr_sort_idxs:
                    coords.append(self.pooling(coords[-1]))
                else:
                    coords.append(sort_spare_tensor(self.pooling(coords[-1]))[0])
            coords.reverse()

        batched_coords = coordinates.decomposed_features
        # Lets always recompute the bounds for now. 
        self.scene_bounds = extract_scene_bounds(batched_coords)

        queries, sampled_coords = self.query_initializer(coordinates=batched_coords, features=pcd_features, poses=poses)

        if not self.learnable_pos_embeddings:
            query_pos_enc = self.pos_encode(sampled_coords)
        else:
            query_pos_enc = self.pos_embeddings.weight.clone().unsqueeze(0).repeat(queries.shape[0], 1, 1)
            if self.add_normalized_positions:
                enc = self.pos_encode(sampled_coords)
                query_pos_enc[:, :, :3] = enc[:, :, :3]

        # Get positional encodings for all coordiantes at all levels.

        self.pos_encodings_pcd: List[List[Tensor]] = []  # This has shape [n_levels x batch_size x (n_points x d_pos))]
        for c in coords:
            self.pos_encodings_pcd.append(self.pos_encode(c.decomposed_features))

        assert len(self.refinement_layers) > 0, "No refinement layers defined."

        for layer in self.refinement_layers:
            assert isinstance(layer, QueryRefinementModule)

            out = layer(
                pcd_features=pcd_features,
                multi_res_features=aux,
                coordinates=coords,
                coordinates_position_enc=self.pos_encodings_pcd,
                queries=queries,
                queries_pos_enc=query_pos_enc,
                repr_sort_idxs=self.repr_sort_idxs,
            )
            queries = out.shape_queries[-1]  # update queries.
            if queries.isnan().any():
                print("NaN in Queries!")

        query_pos_enc = out.query_pos_enc
        all_latents = [q + query_pos_enc for q in out.shape_queries]


        if self.repr_sort_idxs:
            for i in range(len(out.predictions_mask)):
                splits = [len(p) for p in out.predictions_mask[i]]
                out_idxs = self.coord_sort_maps[-1].split(splits)
                for j in range(len(out.predictions_mask[i])):
                    out_mask = out_idxs[j] - out_idxs[j].min()
                    preds = torch.zeros_like(out.predictions_mask[i][j])
                    preds[out_mask] = out.predictions_mask[i][j]
                    out.predictions_mask[i][j] = preds

        return {
            "pred_logits": out.predictions_class[-1],
            "pred_masks": out.predictions_mask[-1],
            "instance_queries": out.shape_queries[-1],  # Raw queries without positional encoding
            "all_queries": out.shape_queries,  # Raw queries without positional encoding
            "instance_latents": all_latents[
                -1
            ],  # These already have the positional encoding added. Only the final prediction.
            "intermittent_latents": all_latents,  # These already have the positional encoding added. All intermittend predictions.
            "positional_encodings": query_pos_enc,
            "aux_outputs": self._set_aux_loss(out.predictions_class, out.predictions_mask),
            "sampled_coords": sampled_coords.detach().cpu().numpy(),
            "backbone_features": pcd_features,
            "attention_info": out.attention_info,
            "object_grasp_queries": out.obj_grasp_queries,
            "scene_grasp_queries": out.scene_grasp_queries,
        }

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class: List[Tensor], outputs_seg_masks: List[Tensor]) -> List[dict[str, Tensor]]:
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a List.
        return [{"pred_logits": a, "pred_masks": b} for a, b, in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
