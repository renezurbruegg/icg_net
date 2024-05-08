"""Utility functions from edge grasp net for gripper points."""

import torch


def get_gripper_points(rotation, pos):
    gripper_points_sim = (
        torch.tensor(
            [
                [
                    0,
                    0,
                    -0.02 - 0.022,
                ],
                [
                    0.03,
                    -0.09,
                    0.015 - 0.022,
                ],
                [
                    -0.030,
                    -0.09,
                    0.015 - 0.022,
                ],
                [
                    0.030,
                    0.09,
                    0.015 - 0.022,
                ],
                [
                    -0.030,
                    0.09,
                    0.015 - 0.022,
                ],
                [
                    0.005,
                    0.09,
                    0.078 - 0.022,
                ],
                [
                    0.005,
                    -0.09,
                    0.078 - 0.022,
                ],
            ]
        )
        .to(torch.float)
        .to(rotation.device)
    )

    num_p = gripper_points_sim.size(0)
    gripper_points_sim = gripper_points_sim.unsqueeze(dim=0).repeat(len(pos), 1, 1)
    gripper_points_sim = torch.einsum("pij,pjk->pik", rotation, gripper_points_sim.transpose(1, 2))
    gripper_points_sim = gripper_points_sim.transpose(1, 2)
    gripper_points_sim = gripper_points_sim + pos.unsqueeze(dim=1).repeat(1, num_p, 1)

    return gripper_points_sim


def get_gripper_points_mask(gripper_points_sim, threshold=0.053):
    z_value = gripper_points_sim[:, :, -1]
    # print('gripper max z value', z_value.max())
    z_mask = z_value > threshold
    z_mask = torch.all(z_mask, dim=1)

    return z_mask
