import MinkowskiEngine as ME
import numpy as np
import torch
from random import random


class VoxelizeCollateAcronym:
    def __init__(
        self,
        ignore_label=255,
        voxel_size=1,
        mode="test",
        batch_instance=False,
        task="reconstruction",
        ignore_class_threshold=100,
        filter_out_classes=[],
        label_offset=0,
        num_queries=None,
    ):
        # assert task in ["reconstruction"], "task not known"
        self.task = task
        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset
        self.voxel_size = voxel_size
        self.ignore_label = ignore_label
        self.mode = mode
        self.batch_instance = batch_instance
        self.ignore_class_threshold = ignore_class_threshold

        self.num_queries = num_queries

    def __call__(self, batch):
        batch = [*filter(lambda x: x is not None, batch)]  # remove invalid entries
        return voxelizeAcronym(
            batch,
            self.ignore_label,
            self.voxel_size,
            self.mode,
            task=self.task,
            ignore_class_threshold=self.ignore_class_threshold,
            filter_out_classes=self.filter_out_classes,
            label_offset=self.label_offset,
        )


def voxelizeAcronym(
    batch,
    ignore_label,
    voxel_size,
    mode,
    task,
    ignore_class_threshold,
    filter_out_classes,
    label_offset,
):
    (
        coordinates,
        features,
        labels,
        original_labels,
        inverse_maps,
        original_colors,
        original_normals,
        original_coordinates,
        idx,
        sdf_query_coords,
        sdf_values,
        transforms,
        poses,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    obj_grasp_pts = []
    scene_grasp_pts = []
    obj_grasp_labels = []
    scene_grasp_labels = []

    voxelization_dict = {
        "ignore_label": ignore_label,
        # "quantization_size": self.voxel_size,
        "return_index": True,
        "return_inverse": True,
    }

    full_res_coords = []
    full_res_features = []

    for sample in batch:
        idx.append(sample[7])
        original_coordinates.append(sample[6])
        original_labels.append(sample[2])
        full_res_coords.append(sample[6])
        original_colors.append(sample[4])
        original_normals.append(sample[5])
        full_res_features.append(sample[1][:, :-3])

        coords = np.floor(sample[0] / voxel_size)
        voxelization_dict.update(
            {
                "coordinates": torch.from_numpy(coords).to("cpu").contiguous(),
                "features": sample[1],
            }
        )

        assert len(sample) >= 9, "sample requires SDF information"

        data_dict = sample[8]
        queries = data_dict["points"]
        sdf_values.append(data_dict["sdf_per_instance"])
        sdf_query_coords.append(torch.from_numpy(queries).float())
        transforms.append(sample[9])

        # maybe this change (_, _, ...) is not necessary and we can directly get out
        # the sample coordinates?
        _, _, unique_map, inverse_map = ME.utils.sparse_quantize(**voxelization_dict)
        inverse_maps.append(inverse_map)

        sample_coordinates = coords[unique_map]
        coordinates.append(torch.from_numpy(sample_coordinates).int())
        sample_features = sample[1][unique_map]
        features.append(torch.from_numpy(sample_features).float())
        if len(sample[2]) > 0:
            sample_labels = sample[2][unique_map]
            labels.append(torch.from_numpy(sample_labels).long())

        grasp_input = sample[10]
        grasp_target = sample[11]

        if grasp_input is not None:
            if (
                "object_grasp_points" in grasp_input
                and grasp_input["object_grasp_points"] is not None
            ):
                obj_grasp_pts.append(
                    torch.from_numpy(grasp_input["object_grasp_points"]).float()
                )
            if (
                "scene_grasp_points" in grasp_input
                and grasp_input["scene_grasp_points"] is not None
            ):
                scene_grasp_pts.append(
                    torch.from_numpy(grasp_input["scene_grasp_points"]).float()
                )

        if grasp_target is not None:
            if grasp_target["object_labels"] is not None:
                obj_grasp_labels.append(
                    torch.from_numpy(grasp_target["object_labels"]).float()
                )
            if grasp_target["scene_labels"] is not None:
                scene_grasp_labels.append(
                    torch.from_numpy(grasp_target["scene_labels"]).float()
                )

        if len(sample) >= 13:
            pose = sample[12]
            poses.append(pose)

        # import pdb; pdb.set_trace()
        # grasp_points, grasp_normals, object_centric_grasps, scene_centric_grasps

    sdf_query_coords = torch.stack(sdf_query_coords)
    if len(scene_grasp_pts) > 0:
        scene_grasp_pts = torch.stack(scene_grasp_pts)
    if len(obj_grasp_pts) > 0:
        obj_grasp_pts = torch.stack(obj_grasp_pts)

    # # Subsample object centric grasp points
    # num_grasp_pts_obj = [len(o) for o in obj_grasp_pts]
    # max_num_pts = 3600
    # min_pts = min(min(num_grasp_pts_obj), max_num_pts)

    # # subsmaple
    # obj_grasp_pts_sub = []
    # obj_grasp_labels_sub  = []

    # for i, num_pts in enumerate(num_grasp_pts_obj):
    #     selection_mask = torch.randperm(num_pts)[:min_pts]
    #     obj_grasp_pts_sub.append(obj_grasp_pts[i][selection_mask])
    #     obj_grasp_labels_sub.append(obj_grasp_labels[i][:, selection_mask])

    # grasp_points_object = torch.stack(obj_grasp_pts_sub)
    # grasp_labels_object = obj_grasp_labels_sub

    # grasp_points_scene = torch.stack(scene_grasp_pts)
    # grasp_labels_scene = scene_grasp_labels
    # Concatenate all lists
    input_dict = {"coords": coordinates, "feats": features}
    if len(labels) > 0:
        input_dict["labels"] = labels
        coordinates, features, labels = ME.utils.sparse_collate(**input_dict)
    else:
        coordinates, features = ME.utils.sparse_collate(**input_dict)
        labels = torch.Tensor([])

    if mode == "test":
        for i in range(len(input_dict["labels"])):
            _, ret_index, ret_inv = np.unique(
                input_dict["labels"][i][:, 0], return_index=True, return_inverse=True
            )
            input_dict["labels"][i][:, 0] = torch.from_numpy(ret_inv)
            # input_dict["segment2label"].append(input_dict["labels"][i][ret_index][:, :-1])
    else:
        input_dict["segment2label"] = []

        if "labels" in input_dict:
            for i in range(len(input_dict["labels"])):
                # TODO BIGGER CHANGE CHECK!!!
                _, ret_index, ret_inv = np.unique(
                    input_dict["labels"][i][:, -1],
                    return_index=True,
                    return_inverse=True,
                )
                input_dict["labels"][i][:, -1] = torch.from_numpy(ret_inv)
                input_dict["segment2label"].append(
                    input_dict["labels"][i][ret_index][:, :-1]
                )

    if "labels" in input_dict:
        list_labels = input_dict["labels"]

        target = []
        target_full = []

        if len(list_labels[0].shape) == 1:
            for batch_id in range(len(list_labels)):
                label_ids = list_labels[batch_id].unique()
                if 255 in label_ids:
                    label_ids = label_ids[:-1]

                target.append(
                    {
                        "labels": label_ids,
                        "masks": list_labels[batch_id] == label_ids.unsqueeze(1),
                    }
                )
        else:
            if mode == "test":
                for i in range(len(input_dict["labels"])):
                    target.append(
                        {
                            "point2segment": input_dict["labels"][i][:, 0],
                        }
                    )
                    target_full.append(
                        {
                            "point2segment": torch.from_numpy(
                                original_labels[i][:, 0]
                            ).long()
                        }
                    )
            else:
                target = get_instance_masks(
                    list_labels,
                    list_segments=input_dict["segment2label"],
                    task=task,
                    ignore_class_threshold=ignore_class_threshold,
                    filter_out_classes=filter_out_classes,
                    label_offset=label_offset,
                )
                for i in range(len(target)):
                    target[i]["point2segment"] = input_dict["labels"][i][:, 2]
                if "train" not in mode:
                    target_full = get_instance_masks(
                        [torch.from_numpy(l) for l in original_labels],
                        task=task,
                        ignore_class_threshold=ignore_class_threshold,
                        filter_out_classes=filter_out_classes,
                        label_offset=label_offset,
                    )
                    for i in range(len(target_full)):
                        target_full[i]["point2segment"] = torch.from_numpy(
                            original_labels[i][:, 2]
                        ).long()
    else:
        target = []
        target_full = []
        coordinates = []
        features = []

    # obj_grasp_pts = []
    # scene_grasp_pts = []
    # obj_grasp_labels = []
    # scene_grasp_labels = []
    for idx in range(len(target)):
        target[idx]["sdf"] = torch.from_numpy(sdf_values[idx]).float()

        if obj_grasp_labels is not None and len(obj_grasp_labels) > 0:
            target[idx]["object_centric_labels"] = obj_grasp_labels[idx]
            target[idx]["object_centric_pts"] = obj_grasp_pts[idx]
        if scene_grasp_labels is not None and len(scene_grasp_labels) > 0:
            target[idx]["scene_centric_labels"] = scene_grasp_labels[idx]
            target[idx]["scene_centric_pts"] = scene_grasp_pts[idx]

    if "train" not in mode:
        return (
            NoGpu(
                coordinates,
                features,
                original_labels,
                inverse_maps,
                target_full,
                original_colors,
                original_normals,
                original_coordinates,
                idx,
                full_res_coords=full_res_coords,
                sdf_queries=sdf_query_coords,
                full_res_features=full_res_features,
                sdf_values=sdf_values,
                grasp_points_object=obj_grasp_pts,
                grasp_points_scene=scene_grasp_pts,
                poses=poses,
            ),
            target,
            [sample[3] for sample in batch],
            transforms,
        )
    else:
        return (
            NoGpu(
                coordinates,
                features,
                original_labels,
                inverse_maps,
                full_res_coords=full_res_coords,
                sdf_queries=sdf_query_coords,
                full_res_features=full_res_features,
                sdf_values=sdf_values,
                grasp_points_object=obj_grasp_pts,
                grasp_points_scene=scene_grasp_pts,
                poses=poses,
            ),
            target,
            [sample[3] for sample in batch],
            transforms,
        )


def get_instance_masks(
    list_labels,
    task,
    list_segments=None,
    ignore_class_threshold=100,
    filter_out_classes=[],
    label_offset=0,
):
    target = []

    for batch_id in range(len(list_labels)):
        label_ids = []
        masks = []
        segment_masks = []
        instance_ids = list_labels[batch_id][:, 1].unique()

        for instance_id in instance_ids:
            if instance_id == -1:
                continue

            # TODO is it possible that a ignore class (255) is an instance???
            # instance == -1 ???
            tmp = list_labels[batch_id][list_labels[batch_id][:, 1] == instance_id]
            label_id = tmp[0, 0]

            if (
                label_id in filter_out_classes
            ):  # floor, wall, undefined==255 is not included
                continue

            if (
                255 in filter_out_classes
                and label_id.item() == 255
                and tmp.shape[0] < ignore_class_threshold
            ):
                continue

            label_ids.append(label_id)
            masks.append(list_labels[batch_id][:, 1] == instance_id)

            if list_segments:
                segment_mask = torch.zeros(list_segments[batch_id].shape[0]).bool()
                segment_mask[
                    list_labels[batch_id][list_labels[batch_id][:, 1] == instance_id][
                        :, 2
                    ].unique()
                ] = True
                segment_masks.append(segment_mask)

        if len(label_ids) == 0:
            return list()

        label_ids = torch.stack(label_ids)
        masks = torch.stack(masks)
        if list_segments:
            segment_masks = torch.stack(segment_masks)

        if task == "semantic_segmentation":
            new_label_ids = []
            new_masks = []
            new_segment_masks = []
            for label_id in label_ids.unique():
                masking = label_ids == label_id

                new_label_ids.append(label_id)
                new_masks.append(masks[masking, :].sum(dim=0).bool())

                if list_segments:
                    new_segment_masks.append(
                        segment_masks[masking, :].sum(dim=0).bool()
                    )

            label_ids = torch.stack(new_label_ids)
            masks = torch.stack(new_masks)

            if list_segments:
                segment_masks = torch.stack(new_segment_masks)

                target.append(
                    {"labels": label_ids, "masks": masks, "segment_mask": segment_masks}
                )
            else:
                target.append({"labels": label_ids, "masks": masks})
        else:
            l = torch.clamp(label_ids - label_offset, min=0)

            if list_segments:
                target.append(
                    {"labels": l, "masks": masks, "segment_mask": segment_masks}
                )
            else:
                target.append({"labels": l, "masks": masks})
    return target


def make_crops(batch):
    new_batch = []
    # detupling
    for scene in batch:
        new_batch.append([scene[0], scene[1], scene[2]])
    batch = new_batch
    new_batch = []
    for scene in batch:
        # move to center for better quadrant split
        scene[0][:, :3] -= scene[0][:, :3].mean(0)

        # BUGFIX - there always would be a point in every quadrant
        scene[0] = np.vstack(
            (
                scene[0],
                np.array(
                    [
                        [0.1, 0.1, 0.1],
                        [0.1, -0.1, 0.1],
                        [-0.1, 0.1, 0.1],
                        [-0.1, -0.1, 0.1],
                    ]
                ),
            )
        )
        scene[1] = np.vstack((scene[1], np.zeros((4, scene[1].shape[1]))))
        scene[2] = np.concatenate((scene[2], np.full_like((scene[2]), 255)[:4]))

        crop = scene[0][:, 0] > 0
        crop &= scene[0][:, 1] > 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] > 0
        crop &= scene[0][:, 1] < 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] < 0
        crop &= scene[0][:, 1] > 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] < 0
        crop &= scene[0][:, 1] < 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

    # moving all of them to center
    for i in range(len(new_batch)):
        new_batch[i][0][:, :3] -= new_batch[i][0][:, :3].mean(0)
    return new_batch


class NoGpu:
    def __init__(
        self,
        coordinates,
        features,
        original_labels=None,
        inverse_maps=None,
        target_full=None,
        original_colors=None,
        original_normals=None,
        original_coordinates=None,
        idx=None,
        sdf_queries=None,
        sdf_values=None,
        grasp_points_object=None,
        grasp_points_scene=None,
        object_centric_labels=None,
        scene_centric_labels=None,
        poses=None,
        full_res_features=None,
        full_res_coords=None,
    ):
        """helper class to prevent gpu loading on lightning"""
        self.coordinates = coordinates
        self.features = features
        self.original_labels = original_labels
        self.inverse_maps = inverse_maps
        self.target_full = target_full
        self.original_colors = original_colors
        self.original_normals = original_normals
        self.original_coordinates = original_coordinates
        self.idx = idx
        self.sdf_queries = sdf_queries
        self.sdf_values = sdf_values
        self.grasp_points_scene = grasp_points_scene
        self.grasp_points_object = grasp_points_object
        self.object_centric_labels = object_centric_labels
        self.scene_centric_labels = scene_centric_labels
        self.poses = poses
        self.full_res_features = full_res_features
        self.full_res_coords = full_res_coords


class NoGpuMask:
    def __init__(
        self,
        coordinates,
        features,
        original_labels=None,
        inverse_maps=None,
        masks=None,
        labels=None,
    ):
        """helper class to prevent gpu loading on lightning"""
        self.coordinates = coordinates
        self.features = features
        self.original_labels = original_labels
        self.inverse_maps = inverse_maps

        self.masks = masks
        self.labels = labels
