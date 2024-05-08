"""Checkpoint utilities for loading and saving models."""

from typing import TYPE_CHECKING
import torch
from loguru import logger


def replace_key_start_in_checkpoint(checkpoint, old_start, new_start):
    # Rename keys that start with the old_start
    renamed_keys = {}
    for old_key in checkpoint.keys():
        if old_key.startswith(old_start):
            new_key = new_start + old_key[len(old_start) :]
            renamed_keys[old_key] = new_key

    # Rename the keys in the checkpoint dictionary
    for old_key, new_key in renamed_keys.items():
        checkpoint[new_key] = checkpoint.pop(old_key)


def load_checkpoint_with_missing_or_exsessive_keys(cfg, model: torch.nn.Module, module_name=""):
    state_dict = torch.load(cfg.general.checkpoint, "cuda:0")["state_dict"]
    correct_dict = dict(model.state_dict())

    if module_name != "":
        replace_key_start_in_checkpoint(state_dict, "model.", "")
        # Remap keys from before the refactoring
        replace_key_start_in_checkpoint(state_dict, "backbone.", "feature_volume.sparse_feature_extractor.")
        replace_key_start_in_checkpoint(state_dict, "dense_extractor.", "feature_volume.dense_extractor.")
        replace_key_start_in_checkpoint(state_dict, "dense_extractor.", "feature_volume.dense_extractor.")

        replace_key_start_in_checkpoint(
            state_dict, "dense_graps_interpolations", "grasp_feature_interpolator.dense_interpolations"
        )
        replace_key_start_in_checkpoint(
            state_dict, "point_interpolation_grasp", "grasp_feature_interpolator.point_interpolations"
        )
        replace_key_start_in_checkpoint(state_dict, "grasp_out_mlp", "grasp_feature_interpolator.out_mlp")

        replace_key_start_in_checkpoint(
            state_dict, "dense_occ_interpolations", "occupancy_feature_interpolator.dense_interpolations"
        )
        replace_key_start_in_checkpoint(
            state_dict, "point_interpolation_occ", "occupancy_feature_interpolator.point_interpolations"
        )
        replace_key_start_in_checkpoint(state_dict, "occ_out_mlp", "occupancy_feature_interpolator.out_mlp")

    # if parametrs not found in checkpoint they will be randomly initialized
    for key in state_dict.keys():
        if correct_dict.pop(key, None) is None:
            logger.warning(f"Key not found, it will be initialized randomly: {key}", correct_dict.keys())

    correct_dict = dict(model.state_dict())

    for key in correct_dict.keys():
        if key not in state_dict:
            if ".sparse_interpolator" not in key and "criterion" not in key:
                logger.warning(f"{key} not in loaded checkpoint")
            state_dict.update({key: correct_dict[key]})
        elif state_dict[key].shape != correct_dict[key].shape:
            logger.warning(
                f"incorrect shape Checkpoint {key}:{state_dict[key].shape} vs Model: {correct_dict[key].shape}"
            )
            state_dict.update({key: correct_dict[key]})

    # if we have more keys just discard them
    correct_dict = dict(model.state_dict())
    new_state_dict = dict()
    for key in state_dict.keys():
        if key in correct_dict.keys():
            new_state_dict.update({key: state_dict[key]})
        else:
            logger.warning(f"excessive key: {key}")
    model.load_state_dict(new_state_dict)
    return cfg, model
