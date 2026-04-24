import logging
from itertools import product
from pathlib import Path
from random import random, sample, uniform
from typing import List, Optional, Tuple, Union
from random import choice
from copy import deepcopy
from random import randrange


import numpy
import torch
from datasets.random_cuboid import RandomCuboid

import albumentations as A
import numpy as np
import scipy
import volumentations as V
import yaml

# from yaml import CLoader as Loader
import os
from torch.utils.data import Dataset
from datasets.scannet200.scannet200_constants import (
    SCANNET_COLOR_MAP_200,
    SCANNET_COLOR_MAP_20,
)

logger = logging.getLogger(__name__)


class SemanticSegmentationDataset(Dataset):
    """Docstring for SemanticSegmentationDataset."""

    def __init__(
        self,
        dataset_name="scannet",
        data_dir: Optional[Union[str, Tuple[str]]] = "data/processed/scannet",
        label_db_filepath: Optional[
            str
        ] = "configs/scannet_preprocessing/label_database.yaml",
        # mean std values from scannet
        color_mean_std: Optional[Union[str, Tuple[Tuple[float]]]] = (
            (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
        ),
        mode: Optional[str] = "train",
        add_colors: Optional[bool] = True,
        add_normals: Optional[bool] = True,
        add_raw_coordinates: Optional[bool] = False,
        add_instance: Optional[bool] = False,
        num_labels: Optional[int] = -1,
        data_percent: Optional[float] = 1.0,
        ignore_label: Optional[Union[int, Tuple[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        image_augmentations_path: Optional[str] = None,
        instance_oversampling=0,
        place_around_existing=False,
        max_cut_region=0,
        point_per_cut=100,
        flip_in_center=False,
        noise_rate=0.0,
        resample_points=0.0,
        cache_data=False,
        add_unlabeled_pc=False,
        task="instance_segmentation",
        cropping=False,
        cropping_args=None,
        is_tta=False,
        crop_min_size=20000,
        crop_length=6.0,
        cropping_v1=True,
        reps_per_epoch=1,
        area=-1,
        on_crops=False,
        eval_inner_core=-1,
        filter_out_classes=[],
        label_offset=0,
        add_clip=False,
        is_elastic_distortion=True,
        color_drop=0.0,
        rand_rotate=False,
        rand_translate=False,
        num_sdf_points=50000,
        load_grasps=False,
        add_z_coordinate=False,
        add_all_coordinates=False,
        exclude_classes=[],
        scene_based=False,
        point_drop=0.05,
        balance_sdf=False,
        balance_grasps=False,
        num_grasp_points_obj=1024,
        num_grasp_points_scene=512,
    ):
        assert task in [
            "instance_segmentation",
            "semantic_segmentation",
        ], "unknown task"
        self.add_clip = add_clip
        self.dataset_name = dataset_name
        self.is_elastic_distortion = is_elastic_distortion
        self.color_drop = color_drop
        self.rand_rotate = rand_rotate
        self.rand_translate = rand_translate
        self.exclude_classes = exclude_classes
        self.scene_based = scene_based
        self.point_drop = point_drop
        self.balance_sdf = balance_sdf
        self.balance_grasps = balance_grasps

        self.num_grasp_points_obj = num_grasp_points_obj
        self.num_grasp_points_scene = num_grasp_points_scene

        self.add_z_coordinate = add_z_coordinate

        if self.dataset_name == "custom":
            self.color_map = {
                0: [0, 255, 0],  # ceiling
                1: [0, 0, 255],  # floor
                2: [0, 255, 255],  # wall
                3: [255, 255, 0],  # beam
                4: [255, 0, 255],  # column
                5: [100, 100, 255],  # window
                6: [200, 200, 100],  # door
                7: [170, 120, 200],  # table
                8: [255, 0, 0],  # chair
                9: [200, 100, 100],  # sofa
                10: [10, 200, 100],  # bookcase
                11: [200, 200, 200],  # board
                12: [50, 50, 50],  # clutter
            }
        else:
            assert False, "dataset not known"

        self.task = task
        self.add_all_coordinates = add_all_coordinates

        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset

        self.area = area
        self.eval_inner_core = eval_inner_core

        self.reps_per_epoch = reps_per_epoch

        self.cropping = cropping
        self.cropping_args = cropping_args
        self.is_tta = is_tta
        self.on_crops = on_crops

        self.crop_min_size = crop_min_size
        self.crop_length = crop_length

        self.version1 = cropping_v1

        self.random_cuboid = RandomCuboid(
            self.crop_min_size, crop_length=self.crop_length, version1=self.version1
        )

        self.mode = mode
        self.load_grasps = load_grasps
        self.data_dir = data_dir
        self.add_unlabeled_pc = add_unlabeled_pc
        if add_unlabeled_pc:
            self.other_database = self._load_yaml(
                Path(data_dir).parent / "matterport" / "train_database.yaml"
            )
        if type(data_dir) == str:
            self.data_dir = [self.data_dir]
        self.ignore_label = ignore_label
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.add_instance = add_instance
        self.add_raw_coordinates = add_raw_coordinates
        self.instance_oversampling = instance_oversampling
        self.place_around_existing = place_around_existing
        self.max_cut_region = max_cut_region
        self.point_per_cut = point_per_cut
        self.flip_in_center = flip_in_center
        self.noise_rate = noise_rate
        self.resample_points = resample_points
        self.num_sdf_points = num_sdf_points

        # loading database files
        self._data = []
        for database_path in self.data_dir:
            database_path = Path(database_path)
            if self.dataset_name != "s3dis":
                if not (database_path / f"{mode}_database.yaml").exists():
                    print(f"generate {database_path}/{mode}_database.yaml first")
                    exit()
                self._data.extend(
                    self._load_yaml(database_path / f"{mode}_database.yaml")
                )

        if data_percent < 1.0:
            self._data = sample(self._data, int(len(self._data) * data_percent))

        is_on_cluster = os.path.exists("/cluster/work/cvl/zrene")
        if is_on_cluster and False:  # True:
            print(
                "Monkey patching paths to local storage. First path:",
                self._data[0]["filepath"],
            )
            for e in self._data:
                # Remap to local storage-                                                                         /cluster/work/cvl/zrene/dataset/processed/
                e["filepath"] = (
                    e["filepath"]
                    .replace("/cluster/scratch/zrene/dataset/processed/", "./")
                    .replace("/cluster/work/cvl/zrene/dataset/processed/", "./")
                )
                e["raw_filepath"] = (
                    e["raw_filepath"]
                    .replace("/cluster/work/cvl/zrene/dataset/processed/", "./")
                    .replace("/cluster/work/cvl/zrene/dataset/processed/", "./")
                )
                e["instance_gt_filepath"] = (
                    e["instance_gt_filepath"]
                    .replace("/cluster/work/cvl/sszrene/dataset/processed/", "./")
                    .replace("/cluster/work/cvl/zrene/dataset/processed/", "./")
                )
            print("Patched now:", self.data[0]["filepath"])

        labels = self._load_yaml(Path(label_db_filepath))

        # if working only on classes for validation - discard others
        self._labels = labels  # self._select_correct_labels(labels, num_labels)

        if instance_oversampling > 0:
            self.instance_data = self._load_yaml(
                Path(label_db_filepath).parent / "instance_database.yaml"
            )

        if Path(str(color_mean_std)).exists():
            color_mean_std = self._load_yaml(color_mean_std)
            color_mean, color_std = (
                tuple(color_mean_std["mean"]),
                tuple(color_mean_std["std"]),
            )
        elif len(color_mean_std[0]) == 3 and len(color_mean_std[1]) == 3:
            color_mean, color_std = color_mean_std[0], color_mean_std[1]
        else:
            logger.error("pass mean and std as tuple of tuples, or as an .yaml file")
            # TODO: remove
            color_mean = (0, 0, 0)
            color_std = (1, 1, 1)

        # augmentations
        self.volume_augmentations = V.NoOp()
        if (volume_augmentations_path is not None) and (
            volume_augmentations_path != "none"
        ):
            self.volume_augmentations = V.load(
                Path(volume_augmentations_path), data_format="yaml"
            )
        self.image_augmentations = A.NoOp()
        if (image_augmentations_path is not None) and (
            image_augmentations_path != "none"
        ):
            self.image_augmentations = A.load(
                Path(image_augmentations_path), data_format="yaml"
            )
        # mandatory color augmentation
        if add_colors:
            self.normalize_color = A.Normalize(mean=color_mean, std=color_std)

        self.cache_data = cache_data

        # if self.cache_data:
        #     new_data = []
        #     for i in range(len(self._data)):
        #         self._data[i]['data'] = np.load(self.data[i]["filepath"].replace("../../", ""))
        #         if self.on_crops:
        #             if self.eval_inner_core == -1:
        #                 for block_id, block in enumerate(self.splitPointCloud(self._data[i]['data'])):
        #                     if len(block) > 10000:
        #                         new_data.append({
        #                             'instance_gt_filepath': self._data[i]['instance_gt_filepath'][block_id] \
        #                                 if len(self._data[i]['instance_gt_filepath']) > 0 else list(),
        #                             'scene': f"{self._data[i]['scene'].replace('.txt', '')}_{block_id}.txt",
        #                             'raw_filepath': f"{self.data[i]['filepath'].replace('.npy', '')}_{block_id}",
        #                             'data': block
        #                         })
        #                     else:
        #                         assert False
        #             else:
        #                 conds_inner, blocks_outer = self.splitPointCloud(self._data[i]['data'],
        #                                                                  size=self.crop_length,
        #                                                                  inner_core=self.eval_inner_core)

        #                 for block_id in range(len(conds_inner)):
        #                     cond_inner = conds_inner[block_id]
        #                     block_outer = blocks_outer[block_id]

        #                     if cond_inner.sum() > 10000:
        #                         new_data.append({
        #                             'instance_gt_filepath': self._data[i]['instance_gt_filepath'][block_id] \
        #                                 if len(self._data[i]['instance_gt_filepath']) > 0 else list(),
        #                             'scene': f"{self._data[i]['scene'].replace('.txt', '')}_{block_id}.txt",
        #                             'raw_filepath': f"{self.data[i]['filepath'].replace('.npy', '')}_{block_id}",
        #                             'data': block_outer,
        #                             'cond_inner': cond_inner
        #                         })
        #                     else:
        #                         assert False

        # if self.on_crops:
        #     self._data = new_data
        # new_data.append(np.load(self.data[i]["filepath"].replace("../../", "")))
        # self._data = new_data

    # def splitPointCloud(self, cloud, size=50.0, stride=50, inner_core=-1):
    #     if inner_core == -1:
    #         limitMax = np.amax(cloud[:, 0:3], axis=0)
    #         width = int(np.ceil((limitMax[0] - size) / stride)) + 1
    #         depth = int(np.ceil((limitMax[1] - size) / stride)) + 1
    #         cells = [(x * stride, y * stride) for x in range(width) for y in range(depth)]
    #         blocks = []
    #         for (x, y) in cells:
    #    s         xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
    #             ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
    #             cond = xcond & ycond
    #             block = cloud[cond, :]
    #             blocks.append(block)
    #         return blocks
    #     else:
    #         limitMax = np.amax(cloud[:, 0:3], axis=0)
    #         width = int(np.ceil((limitMax[0] - inner_core) / stride)) + 1
    #         depth = int(np.ceil((limitMax[1] - inner_core) / stride)) + 1
    #         cells = [(x * stride, y * stride) for x in range(width) for y in range(depth)]
    #         blocks_outer = []
    #         conds_inner = []
    #         for (x, y) in cells:
    #             xcond_outer = (cloud[:, 0] <= x + inner_core / 2. + size / 2) & (cloud[:, 0] >= x + inner_core / 2. - size / 2)
    #             ycond_outer = (cloud[:, 1] <= y + inner_core / 2. + size / 2) & (cloud[:, 1] >= y + inner_core / 2. - size / 2)

    #             cond_outer = xcond_outer & ycond_outer
    #             block_outer = cloud[cond_outer, :]

    #             xcond_inner = (block_outer[:, 0] <= x + inner_core) & (block_outer[:, 0] >= x)
    #             ycond_inner = (block_outer[:, 1] <= y + inner_core) & (block_outer[:, 1] >= y)

    #             cond_inner = xcond_inner & ycond_inner

    #             conds_inner.append(cond_inner)
    #             blocks_outer.append(block_outer)
    #         return conds_inner, blocks_outer

    def map2color(self, labels):
        output_colors = list()

        for label in labels:
            output_colors.append(self.color_map[label])

        return torch.tensor(output_colors)

    def __len__(self):
        if self.is_tta:
            return len(self.data)
        else:
            return self.reps_per_epoch * len(self.data)

    def __getitem__(self, idx: int):
        idx = idx % len(self.data)
        if self.is_tta:
            idx = idx % len(self.data)
        assert not self.cache_data, "Caching is not supported for now"

        poses = np.asarray(self.data[idx]["poses"])
        if self.cache_data:
            points = self.data[idx]["data"]
        else:
            assert not self.on_crops, "you need caching if on crops"
            points = np.load(self.data[idx]["filepath"].replace("../../", "")).astype(
                np.float
            )
            # load sdf and points
            file_name = os.path.basename(self.data[idx]["filepath"]).replace(".npy", "")
            dir_name = os.path.dirname(self.data[idx]["filepath"])

            sdf_archive = np.load(os.path.join(dir_name, "sdf", file_name, "0000.npz"))
            sdf_query_points, sdf_instance_values = (
                sdf_archive["points"],
                sdf_archive["sdf"],
            )

        # def show_cloud(cloud):
        #     import open3d as o3d
        #     pcd = o3d.geometry.PointCloud()
        #     if isinstance(cloud, torch.Tensor):
        #         cloud = cloud.deatch().cpu().numpy()

        #     pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
        #     o3d.visualization.draw_geometries([pcd])

        coordinates, color, normals, segments, labels = (
            points[:, :3],
            points[:, 3:6],
            points[:, 6:9],
            points[:, 9],
            points[:, 10:12],
        )

        # print("Trying to access,", np.unique(labels[:, 1].astype(int)))
        # print("SDF instances, ", sdf_instance_values.shape)
        # make sure to mask out invisible classes
        # sdf_instance_values = sdf_instance_values[np.unique(labels[:, 1].astype(int))]

        if len(self.exclude_classes) > 0:
            # Remove unwanted classes
            mask = np.isin(labels[:, 0], self.exclude_classes)
            if mask.any():
                points = points[~mask]
                coordinates, color, normals, segments, labels = (
                    points[:, :3],
                    points[:, 3:6],
                    points[:, 6:9],
                    points[:, 9],
                    points[:, 10:12],
                )

                labels[:, 0] = 0  # Remap all classes to class zero

                instance_mask = np.ones(sdf_instance_values.shape[0]).astype(np.bool)
                instance_mask[self.exclude_classes] = False

                sdf_instance_values = sdf_instance_values[instance_mask]

        if self.point_drop > 0.0:
            mask = np.random.rand(coordinates.shape[0]) > self.point_drop
            coordinates, color, normals, segments, labels = (
                coordinates[mask],
                color[mask],
                normals[mask],
                segments[mask],
                labels[mask],
            )

        if self.noise_rate != 0.0:
            coordinates += np.random.randn(*coordinates.shape) * self.noise_rate

        if self.load_grasps:
            # 0:3 contact, 3:6 y_axis, 6:7 width, 7:16 orientation


            # # Save Grasps:
            # processes_grasps_filepath_qual = self.save_dir / mode / "grasps" / f"{scene_name}_scene_quality.npy"
            # processes_grasps_filepath_qual_object = self.save_dir / mode / "grasps" / f"{scene_name}_object_quality.npy"
            # processes_grasps_filepath_query = self.save_dir / mode / "grasps" / f"{scene_name}_scene_query.npy"
            # processes_grasps_filepath_query_object = self.save_dir / mode / "grasps" / f"{scene_name}_object_query.npy"
            grasp_query_info_object = np.load(os.path.join(dir_name, "grasps", file_name +"_object_query.npy"))
            grasp_query_info_scene = np.load(os.path.join(dir_name, "grasps", file_name +"_scene_query.npy"))

            grasp_label_info_object = np.load(os.path.join(dir_name, "grasps", file_name +"_object_quality.npy")) #> 0
            grasp_label_info_scene = np.load(os.path.join(dir_name, "grasps", file_name +"_scene_quality.npy")) #> 0
            # shape of grasp labels is (2, num_instances, num_grasps, num_labels)
            try:
                object_centric_grasp_labels = grasp_label_info_object[np.unique(labels[:,1].astype(int))]#[0]
            except:
                print("Error indexing object grasps. Shapes:", grasp_label_info_object.shape, np.unique(labels[:,1].astype(int)).shape, "instances", np.unique(labels[:,1].astype(int)))
                return None
                
            scene_centric_grasp_labels = grasp_label_info_scene[np.unique(labels[:,1].astype(int))]#[1]
            mask = np.logical_or(grasp_label_info_scene[..., -1].argmax(0) != 0,   grasp_label_info_scene[0,..., -2] == 0)
                
            
            if (mask != False).any():
                scene_centric_grasp_labels = scene_centric_grasp_labels[:, mask]
                grasp_query_info_scene = grasp_query_info_scene[mask]

                
                if object_centric_grasp_labels.shape[1] != 0:
                    obj_mask = np.random.choice(np.arange(object_centric_grasp_labels.shape[1]), self.num_grasp_points_scene)
                    object_centric_grasp_labels = object_centric_grasp_labels[:, obj_mask]
                    grasp_query_info_object = grasp_query_info_object[obj_mask]

                if scene_centric_grasp_labels.shape[1] != 0:
                    scene_mask = np.random.choice(np.arange(scene_centric_grasp_labels.shape[1]), self.num_grasp_points_obj)
                    scene_centric_grasp_labels = scene_centric_grasp_labels[:, scene_mask]
                    grasp_query_info_scene = grasp_query_info_scene[scene_mask]
            
            else:
                print("No valid grasp points in scene")
                grasp_query_info_scene = 0*grasp_query_info_scene[:self.num_grasp_points_scene]
                scene_centric_grasp_labels = 0*scene_centric_grasp_labels[:self.num_grasp_points_scene]

                grasp_query_info_object = 0*grasp_query_info_object[:self.num_grasp_points_obj]
                object_centric_grasp_labels = 0*object_centric_grasp_labels[:self.num_grasp_points_obj]
                
            grasp_contact_pts_scene = grasp_query_info_scene
            grasp_contact_pts_object = grasp_query_info_object

        # print(
        #     "Scene centric labels",
        #     scene_centric_grasp_labels.shape,
        #     "sdfs",
        #     sdf_instance_values.shape,
        # )
        raw_color = color
        raw_normals = normals

        # if not self.add_colors:
        if np.all(color == 0):
            color = np.ones((len(color), 3))

        transform = np.eye(4)

        # volume and image augmentations for train
        if "train" in self.mode or "full" in self.mode or self.is_tta:
            if self.cropping:
                assert False, "Cropping is not supported for now"
                new_idx = self.random_cuboid(
                    coordinates,
                    labels[:, 1],
                    self._remap_from_zero(labels[:, 0].copy()),
                )

                coordinates = coordinates[new_idx]
                color = color[new_idx]
                labels = labels[new_idx]
                segments = segments[new_idx]
                raw_color = raw_color[new_idx]
                raw_normals = raw_normals[new_idx]
                normals = normals[new_idx]
                points = points[new_idx]

            augment = False

            if self.rand_rotate:
                angle = np.random.rand() * 2 * np.pi
                # Create random rotation matrix to rotates around z axis with angle
                transform[:3, :3] = np.array(
                    [
                        [np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1],
                    ]
                )

            if self.rand_translate:
                rand_translate = np.random.uniform(-0.5, 0.5, 3)  # sample +/- 0.5m
                transform[:3, 3] = rand_translate

            coordinates = (transform[:3, :3] @ coordinates.T).T + transform[:3, 3]
            poses = (transform[:3, :3] @ poses.T).T + transform[:3, 3]

            if self.load_grasps:
                grasp_contact_pts_scene = (
                    transform[:3, :3] @ grasp_contact_pts_scene.T
                ).T + transform[:3, 3]
                grasp_contact_pts_object = (
                    transform[:3, :3] @ grasp_contact_pts_object.T
                ).T + transform[:3, 3]

                # grasp_contact_normals = (transform[:3,:3] @ grasp_contact_normals.T).T

                # grasp_contact_pts_scene = np.concatenate([grasp_contact_pts_scene, table_points], axis=0)
                # grasp_contact_pts_object = np.concatenate([grasp_contact_pts_object, table_points], axis=0)

            if sdf_query_points is not None:
                sdf_query_points = (
                    transform[:3, :3] @ sdf_query_points.T
                ).T + transform[:3, 3]
            if raw_normals is not None:
                raw_normals = (transform[:3, :3] @ raw_normals.T).T
                normals = (transform[:3, :3] @ normals.T).T

            if random() < self.color_drop:
                color[:] = 1

        raw_coordinates = coordinates.copy()
        # normalize color information
        pseudo_image = (color * 255).astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])

        # prepare labels and map from 0 to 20(40)
        labels = labels.astype(np.int32)
        if labels.size > 0:
            labels[:, 0] = self._remap_from_zero(labels[:, 0])
            if not self.add_instance:
                # taking only first column, which is segmentation label, not instance
                labels = labels[:, 0].flatten()[..., None]

        labels = np.hstack((labels, segments[..., None].astype(np.int32)))
        instances = np.unique(labels[:, 1])
        if len(instances) != len(sdf_instance_values):
            # print(
            #     "MISSAMTCH IN INSTANCE NUMBER DETECTED. SDF: ",
            #     len(sdf_instance_values),
            #     " Points: ",
            #     instances,
            #     " len",
            #     len(instances),
            # )

            try:
                sdf_instance_values = sdf_instance_values[
                    np.unique(instances) - len(self.exclude_classes)
                ]
            except Exception as e:
                print(
                    str(e),
                    "Error indexing with",
                    np.unique(instances) - len(self.exclude_classes),
                    "sdf",
                    sdf_instance_values.shape,
                )
                sdf_instance_values = sdf_instance_values[: len(instances)]

        features = color
        if self.add_normals:
            features = np.hstack((features, normals))

        if self.add_z_coordinate:
            features = np.hstack((features, coordinates[:, 1][..., None]))

        if self.add_all_coordinates:
            features = np.hstack((features, coordinates))

        if self.add_raw_coordinates:
            if len(features.shape) == 1:
                features = np.hstack((features[None, ...], coordinates))
            else:
                features = np.hstack((features, coordinates))
        # self.balance_sdf = balance_sdf
        # self.balance_grasps = balance_grasps
        pos = np.max(sdf_instance_values, axis=0) > 0

        if not self.balance_sdf or pos.sum() == 0 or (~pos).sum() == 0:
            sdf_mask = np.random.choice(
                np.arange(len(sdf_query_points)),
                self.num_sdf_points,
                replace=len(sdf_query_points) < self.num_sdf_points,
            )
        else:
            sdf_mask_pos = np.random.choice(
                np.arange(len(sdf_query_points))[pos],
                self.num_sdf_points // 2,
                replace=pos.sum() < self.num_sdf_points // 2,
            )
            sdf_mask_neg = np.random.choice(
                np.arange(len(sdf_query_points))[~pos],
                self.num_sdf_points - self.num_sdf_points // 2,
                replace=(~pos).sum() < self.num_sdf_points - self.num_sdf_points // 2,
            )
            sdf_mask = np.concatenate((sdf_mask_pos, sdf_mask_neg))

        sdf_data = {
            "points": sdf_query_points[sdf_mask],
            "sdf_per_instance": sdf_instance_values[: len(instances)][
                ..., sdf_mask
            ],  # ignore un-observed instances. This could mess with l1,l2 scores.
        }

        grasp_input = None
        grasp_target = None

        # object_centric_grasp_labels # n__objs, n_grasps, 12
        # scene_centric_grasp_labels # n_objs, ???, 12
        if self.balance_grasps and self.load_grasps:
            pos_scene = scene_centric_grasp_labels[..., -1].max(0) > 0
            if pos_scene.sum() > 0 and (~pos_scene).sum() > 0:
                grasp_mask_pos_scene = np.random.choice(
                    np.arange(len(pos_scene))[pos_scene],
                    self.num_grasp_points_scene // 2,
                    replace=pos_scene.sum() < self.num_grasp_points_scene // 2,
                )
                grasp_mask_neg_scene = np.random.choice(
                    np.arange(len(pos_scene))[~pos_scene],
                    self.num_grasp_points_scene - self.num_grasp_points_scene // 2,
                    replace=(~pos_scene).sum()
                    < self.num_grasp_points_scene - self.num_grasp_points_scene // 2,
                )
                grasp_mask_scene = np.concatenate(
                    (grasp_mask_pos_scene, grasp_mask_neg_scene)
                )
            else:
                grasp_mask_scene = np.random.choice(
                    np.arange(len(grasp_contact_pts_scene)),
                    self.num_grasp_points_scene,
                )

            # print("SDF sdf_instance_values.shape")
            # print("laoded shapes scene", grasp_contact_pts_scene.shape, scene_centric_grasp_labels.shape)
            # print("laoded shapes object", grasp_contact_pts_object.shape, object_centric_grasp_labels.shape)
            # print("==")

            if len(grasp_contact_pts_scene) == 1 or len(grasp_contact_pts_object) < 20:
                return None  # Weird behaviour. Only one grasp?

            grasp_contact_pts_scene = grasp_contact_pts_scene[grasp_mask_scene]
            scene_centric_grasp_labels = scene_centric_grasp_labels[:, grasp_mask_scene]

            pos_object = object_centric_grasp_labels[..., -1].max(0) > 0
            if pos_object.sum() > 0 and (~pos_object).sum() > 0:
                grasp_mask_pos_object = np.random.choice(
                    np.arange(len(pos_object))[pos_object],
                    self.num_grasp_points_obj // 2,
                    replace=pos_object.sum() < self.num_grasp_points_obj // 2,
                )
                grasp_mask_neg_object = np.random.choice(
                    np.arange(len(pos_object))[~pos_object],
                    self.num_grasp_points_obj - self.num_grasp_points_obj // 2,
                    replace=(~pos_object).sum()
                    < self.num_grasp_points_obj - self.num_grasp_points_obj // 2,
                )
                grasp_mask_object = np.concatenate(
                    (grasp_mask_pos_object, grasp_mask_neg_object)
                )
            else:
                grasp_mask_object = np.random.choice(
                    np.arange(len(grasp_contact_pts_object)),
                    self.num_grasp_points_obj,
                )

            grasp_contact_pts_object = grasp_contact_pts_object[grasp_mask_object]
            object_centric_grasp_labels = object_centric_grasp_labels[
                :, grasp_mask_object
            ]

        if self.load_grasps:
            grasp_input = {
                "grasp_points_object": grasp_contact_pts_object,
                "grasp_points_scene": grasp_contact_pts_scene,
            }
            grasp_target = {
                "object_centric_labels": object_centric_grasp_labels,
                "scene_centric_labels": scene_centric_grasp_labels,
            }
        else:
            grasp_input = None
            grasp_target = None

        if self.dataset_name == "custom":
            return (
                coordinates,
                features,
                labels,
                self.data[idx]["filepath"],
                raw_color,
                raw_normals,
                raw_coordinates,
                idx,
                sdf_data,
                transform,
                grasp_input,
                grasp_target,
                poses,
            )
        else:
            return (
                coordinates,
                features,
                labels,
                self.data[idx]["raw_filepath"].split("/")[-2],
                raw_color,
                raw_normals,
                raw_coordinates,
                idx,
                transform,
            )

    @property
    def data(self):
        """database file containing information about preproscessed dataset"""
        return self._data

    @property
    def label_info(self):
        """database file containing information labels used by dataset"""
        return self._labels

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            # file = yaml.load(f, Loader=Loader)
            print("loading, ", filepath)
            file = yaml.load(f, Loader=yaml.FullLoader)
        return file

    def _select_correct_labels(self, labels, num_labels):
        number_of_validation_labels = 0
        number_of_all_labels = 0
        for (
            k,
            v,
        ) in labels.items():
            number_of_all_labels += 1
            if v["validation"]:
                number_of_validation_labels += 1

        if num_labels == number_of_all_labels:
            return labels
        elif num_labels == number_of_validation_labels:
            valid_labels = dict()
            for (
                k,
                v,
            ) in labels.items():
                if v["validation"]:
                    valid_labels.update({k: v})
            return valid_labels
        else:
            msg = f"""not available number labels, select from:
            {number_of_validation_labels}, {number_of_all_labels}"""
            raise ValueError(msg)

    def _remap_from_zero(self, labels):
        labels[~np.isin(labels, list(self.label_info.keys()))] = self.ignore_label
        # remap to the range from 0
        for i, k in enumerate(self.label_info.keys()):
            labels[labels == k] = i
        return labels

    def _remap_model_output(self, output):
        output = np.array(output)
        output_remapped = output.copy()
        for i, k in enumerate(self.label_info.keys()):
            output_remapped[output == i] = k
        return output_remapped

    def augment_individual_instance(
        self, coordinates, color, normals, labels, oversampling=1.0
    ):
        max_instance = int(len(np.unique(labels[:, 1])))
        # randomly selecting half of non-zero instances
        for instance in range(0, int(max_instance * oversampling)):
            if self.place_around_existing:
                center = choice(
                    coordinates[labels[:, 1] == choice(np.unique(labels[:, 1]))]
                )
            else:
                center = np.array([uniform(-5, 5), uniform(-5, 5), uniform(-0.5, 2)])
            instance = choice(choice(self.instance_data))
            instance = np.load(instance["instance_filepath"])
            # centering two objects
            instance[:, :3] = instance[:, :3] - instance[:, :3].mean(axis=0) + center
            max_instance = max_instance + 1
            instance[:, -1] = max_instance
            aug = V.Compose(
                [
                    V.Scale3d(),
                    V.RotateAroundAxis3d(rotation_limit=np.pi / 24, axis=(1, 0, 0)),
                    V.RotateAroundAxis3d(rotation_limit=np.pi / 24, axis=(0, 1, 0)),
                    V.RotateAroundAxis3d(rotation_limit=np.pi, axis=(0, 0, 1)),
                ]
            )(
                points=instance[:, :3],
                features=instance[:, 3:6],
                normals=instance[:, 6:9],
                labels=instance[:, 9:],
            )
            coordinates = np.concatenate((coordinates, aug["points"]))
            color = np.concatenate((color, aug["features"]))
            normals = np.concatenate((normals, aug["normals"]))
            labels = np.concatenate((labels, aug["labels"]))

        return coordinates, color, normals, labels


def elastic_distortion(pointcloud, granularity, magnitude, query_points=None):
    """Apply elastic distortion on sparse coordinate space.

    pointcloud: numpy array of (number of points, at least 3 spatial dims)
    granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: noise multiplier
    """
    blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
    blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
    blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
    coords = pointcloud[:, :3]
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
        noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(
            coords_min - granularity,
            coords_min + granularity * (noise_dim - 2),
            noise_dim,
        )
    ]
    interp = scipy.interpolate.RegularGridInterpolator(
        ax, noise, bounds_error=0, fill_value=0
    )
    pointcloud[:, :3] = coords + interp(coords) * magnitude
    return pointcloud


def crop(points, x_min, y_min, z_min, x_max, y_max, z_max):
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max and z_min < z_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, z_min = {z_min},"
            " x_max = {x_max}, y_max = {y_max}, z_max = {z_max})".format(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                z_min=z_min,
                z_max=z_max,
            )
        )
    inds = np.all(
        [
            (points[:, 0] >= x_min),
            (points[:, 0] < x_max),
            (points[:, 1] >= y_min),
            (points[:, 1] < y_max),
            (points[:, 2] >= z_min),
            (points[:, 2] < z_max),
        ],
        axis=0,
    )
    return inds


def flip_in_center(coordinates):
    # moving coordinates to center
    coordinates -= coordinates.mean(0)
    aug = V.Compose(
        [
            V.Flip3d(axis=(0, 1, 0), always_apply=True),
            V.Flip3d(axis=(1, 0, 0), always_apply=True),
        ]
    )

    first_crop = coordinates[:, 0] > 0
    first_crop &= coordinates[:, 1] > 0
    # x -y
    second_crop = coordinates[:, 0] > 0
    second_crop &= coordinates[:, 1] < 0
    # -x y
    third_crop = coordinates[:, 0] < 0
    third_crop &= coordinates[:, 1] > 0
    # -x -y
    fourth_crop = coordinates[:, 0] < 0
    fourth_crop &= coordinates[:, 1] < 0

    if first_crop.size > 1:
        coordinates[first_crop] = aug(points=coordinates[first_crop])["points"]
    if second_crop.size > 1:
        minimum = coordinates[second_crop].min(0)
        minimum[2] = 0
        minimum[0] = 0
        coordinates[second_crop] = aug(points=coordinates[second_crop])["points"]
        coordinates[second_crop] += minimum
    if third_crop.size > 1:
        minimum = coordinates[third_crop].min(0)
        minimum[2] = 0
        minimum[1] = 0
        coordinates[third_crop] = aug(points=coordinates[third_crop])["points"]
        coordinates[third_crop] += minimum
    if fourth_crop.size > 1:
        minimum = coordinates[fourth_crop].min(0)
        minimum[2] = 0
        coordinates[fourth_crop] = aug(points=coordinates[fourth_crop])["points"]
        coordinates[fourth_crop] += minimum

    return coordinates


def random_around_points(
    coordinates, color, normals, labels, rate=0.2, noise_rate=0, ignore_label=255
):
    coord_indexes = sample(
        list(range(len(coordinates))), k=int(len(coordinates) * rate)
    )
    noisy_coordinates = deepcopy(coordinates[coord_indexes])
    noisy_coordinates += np.random.uniform(
        -0.2 - noise_rate, 0.2 + noise_rate, size=noisy_coordinates.shape
    )

    if noise_rate > 0:
        noisy_color = np.random.randint(0, 255, size=noisy_coordinates.shape)
        noisy_normals = np.random.rand(*noisy_coordinates.shape) * 2 - 1
        noisy_labels = np.full(labels[coord_indexes].shape, ignore_label)

        coordinates = np.vstack((coordinates, noisy_coordinates))
        color = np.vstack((color, noisy_color))
        normals = np.vstack((normals, noisy_normals))
        labels = np.vstack((labels, noisy_labels))
    else:
        noisy_color = deepcopy(color[coord_indexes])
        noisy_normals = deepcopy(normals[coord_indexes])
        noisy_labels = deepcopy(labels[coord_indexes])

        coordinates = np.vstack((coordinates, noisy_coordinates))
        color = np.vstack((color, noisy_color))
        normals = np.vstack((normals, noisy_normals))
        labels = np.vstack((labels, noisy_labels))

    return coordinates, color, normals, labels


def random_points(
    coordinates, color, normals, labels, noise_rate=0.6, ignore_label=255
):
    max_boundary = coordinates.max(0) + 0.1
    min_boundary = coordinates.min(0) - 0.1

    noisy_coordinates = int((max(max_boundary) - min(min_boundary)) / noise_rate)

    noisy_coordinates = np.array(
        list(
            product(
                np.linspace(min_boundary[0], max_boundary[0], noisy_coordinates),
                np.linspace(min_boundary[1], max_boundary[1], noisy_coordinates),
                np.linspace(min_boundary[2], max_boundary[2], noisy_coordinates),
            )
        )
    )
    noisy_coordinates += np.random.uniform(
        -noise_rate, noise_rate, size=noisy_coordinates.shape
    )

    noisy_color = np.random.randint(0, 255, size=noisy_coordinates.shape)
    noisy_normals = np.random.rand(*noisy_coordinates.shape) * 2 - 1
    noisy_labels = np.full((noisy_coordinates.shape[0], labels.shape[1]), ignore_label)

    coordinates = np.vstack((coordinates, noisy_coordinates))
    color = np.vstack((color, noisy_color))
    normals = np.vstack((normals, noisy_normals))
    labels = np.vstack((labels, noisy_labels))
    return coordinates, color, normals, labels
