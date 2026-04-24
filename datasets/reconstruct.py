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
from datasets.scannet200.scannet200_constants import SCANNET_COLOR_MAP_200, SCANNET_COLOR_MAP_20

logger = logging.getLogger(__name__)


class ReconstructionDataset(Dataset):
    """Docstring for ReconstructionDataset. """

    def __init__(
        self,
        dataset_name="reconstruction",
        data_dir: Optional[Union[str, Tuple[str]]] = "data/processed/reconstruction",
        label_db_filepath: Optional[
            str
        ] = "configs/reconstruction/label_database.yaml",
        # mean std values from scannet
        color_mean_std: Optional[Union[str, Tuple[Tuple[float]]]] = (
            (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
        ),

        mode: Optional[str] = "train",
        add_colors: Optional[bool] = True,
        add_normals: Optional[bool] = True,
        add_z_coordinate= False,
        add_all_coordinates = False,
        add_raw_coordinates: Optional[bool] = False,
        add_instance: Optional[bool] = False,
        num_labels: Optional[int] = -1,
        data_percent: Optional[float] = 1.0,
        ignore_label: Optional[Union[int, Tuple[int]]] = 255,
        instance_oversampling=0,
        flip_in_center=False,
        noise_std=0.0,
        task="reconstruction",
        reps_per_epoch=1,
        color_drop=0.0,
        rand_rotate=False,
        rand_translate=False,
        num_sdf_points=50000,

        load_grasps=False,
        
        exclude_classes = []
    ):
        assert task in ["reconstruction"], "unknown task"
        self.dataset_name = dataset_name
        self.color_drop = color_drop
        self.rand_rotate = rand_rotate
        self.rand_translate = rand_translate
        self.exclude_classes = exclude_classes

        self.add_z_coordinate = add_z_coordinate
  
        self.color_map = {
            0: [0, 255, 0],        # ceiling
            1: [0, 0, 255],        # floor
            2: [0, 255, 255],      # wall
            3: [255, 255, 0],      # beam
            4: [255, 0, 255],      # column
            5: [100, 100, 255],    # window
            6: [200, 200, 100],    # door
            7: [170, 120, 200],    # table
            8: [255, 0, 0],        # chair
            9: [200, 100, 100],    # sofa
            10: [10, 200, 100],     # bookcase
            11: [200, 200, 200],    # board
            12: [50, 50, 50]      # clutter
        }

        self.task = task
        self.add_all_coordinates = add_all_coordinates

        self.reps_per_epoch = reps_per_epoch
        self.mode = mode
        self.load_grasps = load_grasps
        self.data_dir = data_dir

        if type(data_dir) == str:
            self.data_dir = [self.data_dir]

        self.ignore_label = ignore_label
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.add_instance = add_instance
        self.add_raw_coordinates = add_raw_coordinates
        self.instance_oversampling = instance_oversampling
        self.flip_in_center = flip_in_center
        self.noise_std = noise_std
        self.num_sdf_points = num_sdf_points

        # loading database files
        self._data = []
        for database_path in self.data_dir:
            database_path = Path(database_path)
            if not (database_path / f"{mode}_database.yaml").exists():
                print(f"generate {database_path}/{mode}_database.yaml first")
                exit()
            self._data.extend(self._load_yaml(database_path / f"{mode}_database.yaml"))

        if data_percent < 1.0:
            self._data = sample(self._data, int(len(self._data) * data_percent))
            

        # if working only on classes for validation - discard others
        self._labels = self._load_yaml(Path(label_db_filepath))

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
            color_mean = (0,0,0)
            color_std = (1,1,1)

        # augmentations
        self.volume_augmentations = V.NoOp()
        if add_colors:
            self.normalize_color = A.Normalize(mean=color_mean, std=color_std)

    def map2color(self, labels):
        output_colors = list()

        for label in labels:
            output_colors.append(self.color_map[label])

        return torch.tensor(output_colors)

    def __len__(self):
        if self.is_tta:
            return len(self.data)
        else:
            return self.reps_per_epoch*len(self.data)

    def __getitem__(self, idx: int):
        idx = idx % len(self.data)
        if self.is_tta:
            idx = idx % len(self.data)
        assert not self.cache_data, "Caching is not supported for now"

        poses = np.asarray(self.data[idx]['poses'])
        if self.cache_data:
            points = self.data[idx]['data']
        else:
            assert not self.on_crops, "you need caching if on crops"
            points = np.load(self.data[idx]["filepath"].replace("../../", "")).astype(np.float)
            # load sdf and points
            file_name= os.path.basename(self.data[idx]["filepath"]).replace(".npy", "")
            dir_name = os.path.dirname(self.data[idx]["filepath"])

            sdf_archive = np.load(os.path.join(dir_name, "sdf", file_name, "0000.npz"))
            sdf_query_points, sdf_instance_values = sdf_archive["points"], sdf_archive["sdf"]
            
        
        coordinates, color, normals, segments, labels = (
            points[:, :3],
            points[:, 3:6],
            points[:, 6:9],
            points[:, 9],
            points[:, 10:12],
        )

        # make sure to mask out obstructed classes
        sdf_instance_values = sdf_instance_values[np.unique(labels[:,1].astype(int))]

        if len(self.exclude_classes) > 0:

            # Remove unwanted classes
            mask = np.isin(labels[:,0], self.exclude_classes)
            if mask.any():
                points  = points[~mask]
                coordinates, color, normals, segments, labels = (
                    points[:, :3],
                    points[:, 3:6],
                    points[:, 6:9],
                    points[:, 9],
                    points[:, 10:12],
                )


                instance_mask = np.ones(sdf_instance_values.shape[0]).astype(np.bool)
                instance_mask[self.exclude_classes] = False
                sdf_instance_values = sdf_instance_values[instance_mask]
        
        
        grasp_contact_points = None
        grasp_contact_normals = None


        if self.load_grasps:
            grasp_query_info = np.load(os.path.join(dir_name, "grasps", file_name +"_query.npy"))

            grasp_label_info = np.load(os.path.join(dir_name, "grasps", file_name +"_quality.npy"))
            # shape of grasp labels is (2, num_instances, num_grasps, num_labels)
            object_centric_grasp_labels = grasp_label_info[0]
            scene_centric_grasp_labels = grasp_label_info[1]
            # 0:3 contact, 3:6 y_axis, 6:7 width, 7:16 orientation
            grasp_contact_points = grasp_query_info[:, 0:3]
            grasp_contact_normals = grasp_query_info[:, 3:6]
            
            if len(self.exclude_classes) > 0:
                mask = np.ones(len(object_centric_grasp_labels)).astype(np.bool)
                mask[self.exclude_classes] = False
                object_centric_grasp_labels = object_centric_grasp_labels[mask]
                scene_centric_grasp_labels = scene_centric_grasp_labels[mask]

            if 0 not in self.exclude_classes: # we do want to load the table class
                # points for table
                table_points = coordinates[labels[:,1] == 0,:]
                # selected points
                n_pts = len(grasp_contact_points) // 5
                selection_mask = np.random.choice(table_points.shape[0], n_pts, replace= table_points.shape[0] < n_pts)
                table_points = table_points[selection_mask]

                table_normals = np.zeros_like(table_points)
                table_normals[:, -1] = 1

                grasp_contact_points = np.concatenate([grasp_contact_points, table_points], axis=0)
                grasp_contact_normals = np.concatenate([grasp_contact_normals, table_normals], axis=0)

                pseudo_grasps = np.zeros_like(object_centric_grasp_labels[:, :n_pts, :])
                object_centric_grasp_labels = np.concatenate([object_centric_grasp_labels, pseudo_grasps], axis=1)
                scene_centric_grasp_labels = np.concatenate([scene_centric_grasp_labels, pseudo_grasps], axis=1)

        if (np.all(color == 0)):
            color = np.ones((len(color), 3))
        
        transform = np.eye(4)
        if "train" in self.mode: # only augment in training mode
            if self.rand_rotate:
                angle = np.random.rand() * 2 * np.pi
                # Create random rotation matrix to rotates around z axis with angle
                transform[:3,:3] = np.array([[np.cos(angle), -np.sin(angle), 0],
                                             [np.sin(angle), np.cos(angle), 0],
                                             [0, 0, 1]])

            if self.rand_translate:
                rand_translate = np.random.uniform(-0.5, 0.5, 3) # sample +/- 0.5m 
                transform[:3, 3] = rand_translate

            coordinates = (transform[:3,:3] @ coordinates.T).T  + transform[:3, 3]
            poses = (transform[:3,:3] @ poses.T).T + transform[:3, 3]

            if self.load_grasps:
                grasp_contact_points = (transform[:3,:3] @ grasp_contact_points.T).T  + transform[:3, 3]
                grasp_contact_normals = (transform[:3,:3] @ grasp_contact_normals.T).T


            if sdf_query_points is not None:
                sdf_query_points = (transform[:3,:3] @ sdf_query_points.T).T  + transform[:3, 3]

            if normals is not None:
                normals = (transform[:3,:3] @ normals.T).T


            if self.noise_std > 0:
                # Add random noise
                coordinates += np.random.noraml(0, self.noise_std, coordinates.shape)

            if self.flip_in_center:
                coordinates[:,:2] = -coordinates[:, :2]
                if sdf_query_points is not None:
                    sdf_query_points[:, :2] = -sdf_query_points[:, :2] 
                    
                if grasp_contact_points is not None:
                    grasp_contact_points[:, :2] = -grasp_contact_points[:, :2]
                    grasp_contact_normals[:, :2] = -grasp_contact_normals[:, :2]

            if random() < self.color_drop:
                color[:] = 255

        raw_coordinates = coordinates.copy()
        # normalize color information
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
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
            print("MISSAMTCH IN INSTANCE NUMBER DETECTED. SDF: ", len(sdf_instance_values), " Points: ", instances, " len", len(instances))

        features = np.zeros((features.shape[0], 0))
        
        if self.add_colors:
            features = np.hstack((features, color))
            
        if self.add_normals:
            features = np.hstack((features, normals))

        if self.add_z_coordinate:
            features =  np.hstack((features, coordinates[:, 1][..., None]))

        if self.add_all_coordinates:
            features = np.hstack((features, coordinates))

        if self.add_raw_coordinates:
            if len(features.shape) == 1:
                features = np.hstack((features[None, ...], coordinates))
            else:
                features = np.hstack((features, coordinates))

        sdf_mask = np.random.choice(np.arange(len(sdf_query_points)), self.num_sdf_points, replace=len(sdf_query_points) < self.num_sdf_points)

        sdf_data = {
            "points": sdf_query_points[sdf_mask],
            "sdf_per_instance": sdf_instance_values[:len(instances)][..., sdf_mask]
        } 

        grasp_input = None
        grasp_target = None

        if self.load_grasps:
            grasp_input = {
                "grasp_points": grasp_contact_points,
                "grasp_normals": grasp_contact_normals,
            }
            grasp_target = {
                "object_centric_labels": object_centric_grasp_labels,
                "scene_centric_labels": scene_centric_grasp_labels,
            }

        return coordinates, features, labels, self.data[idx]['filepath'], raw_coordinates, idx, sdf_data, transform, grasp_input, grasp_target, poses

    @property
    def data(self):
        """ database file containing information about preproscessed dataset """
        return self._data

    @property
    def label_info(self):
        """ database file containing information labels used by dataset """
        return self._labels

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            # file = yaml.load(f, Loader=Loader)
            print("loading, ",filepath)
            file = yaml.load(f)
        return file

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
