from __future__ import annotations

import trimesh
import numpy as np
from trimesh import transformations
from scipy.spatial.transform import Rotation


def create_vgn_gripper_marker(color=[0, 0, 255], tube_radius=0.002, sections=6, width=0.08) -> trimesh.Trimesh:
    return create_gripper_marker(
        color, tube_radius, sections, center_offset=[0, 0, 0], rotations=[0, 0, np.pi / 2], width=width
    )


def create_cnet_gripper_marker(color=[0, 0, 255], tube_radius=0.002, sections=6) -> trimesh.Trimesh:
    return create_gripper_marker(color, tube_radius, sections, center_offset=[0, 0, 0.07])


def create_our_gripper_marker(color=[0, 0, 255], tube_radius=0.002, sections=6) -> trimesh.Trimesh:
    return create_gripper_marker(
        color,
        tube_radius,
        sections,
        center_offset=[0, 0, -0.045],
        rotations=[0, 0, np.pi / 2],
    )


def create_gripper_marker(
    color=[0, 0, 255],
    tube_radius=0.002,
    sections=6,
    center_offset=[0, 0, 0],
    rotations=[0, 0, 0],
    width=0.08,
):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.
    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.
    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    finger_length = 0.05  # 5cm finger length
    gripper_finger_left = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [width / 2 + tube_radius, 0, finger_length],
            [width / 2 + tube_radius, 0, 0],
        ],
    )
    gripper_finger_right = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [-(width / 2 + tube_radius), 0, finger_length],
            [-(width / 2 + tube_radius), 0, 0],
        ],
    )

    # finger_contact_left = trimesh.creation.box(
    #     extents=([0.1, 0.001, 0.001]),
    #     transform=transformations.translation_matrix([-(width / 2 + tube_radius - 0.0025), 0, finger_center]),
    # )

    # finger_contact_right = trimesh.creation.box(
    #     extents=([0.1, 0.001, 0.001]),
    #     transform=transformations.translation_matrix([(width / 2 + tube_radius - 0.0025), 0, finger_center]),
    # )

    # handle
    gripper_handle = trimesh.creation.cylinder(
        radius=tube_radius, sections=sections, segment=[[0, 0, -0.07], [0, 0, 0]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[[-(width / 2 + tube_radius), 0, 0], [width / 2 + tube_radius, 0, 0]],
    )

    tmp = trimesh.util.concatenate(
        [
            gripper_handle,
            cb2,
            gripper_finger_right,
            gripper_finger_left,
            # finger_contact_left,
            # finger_contact_right,
        ]
    )
    tmp.visual.face_colors = color

    # center_offset[-1] -= 0.066  # Center gripper

    mat = transformations.translation_matrix(center_offset)
    mat[:3, :3] = transformations.euler_matrix(*rotations, "sxyz")[:3, :3]
    mat[:3, -1] = mat[:3, :3] @ np.asarray(center_offset)
    tmp.apply_transform(mat)

    # tmp = trimesh.util.concatenate([tmp, trimesh.creation.box([0.001, 0.01, 0.001])])  # shows in y direction
    # tmp.show()

    return tmp


import torch


def decode_grasp_poses(
    query_pts: torch.Tensor,
    width: torch.Tensor | None,
    normals: torch.Tensor,
    affordance: torch.Tensor,
    n_grasps=10,
    slack: torch.Tensor | None = None,
    max_width=0.08,
    n_orientations=1,
    agg="sum",
    return_np=True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(query_pts.shape) == 2:
        query_pts = query_pts.unsqueeze(0)

    if affordance.ndim == 3:
        affordance = affordance.squeeze(0)

    n_grasps = min(len(affordance.sum(dim=-1)), n_grasps // n_orientations)

    # Find best grasps
    if agg == "sum":
        topk_idx = torch.topk(affordance.sum(dim=-1), n_grasps).indices
    elif agg == "l2":
        topk_idx = torch.topk(affordance.norm(dim=-1), n_grasps).indices
    elif agg == "lp":
        topk_idx = torch.topk(affordance.norm(p=4, dim=-1), n_grasps).indices
    elif agg == "max":
        topk_idx = torch.topk(affordance.max(dim=-1).values, n_grasps).indices
    # elif agg == "entropy":
    #     topk_idx = torch.topk(
    #         affordance.max(dim=-1).values
    #         * Categorical(probs=affordance / affordance.norm(dim=-1, keepdim=True)).entropy(),
    #         n_grasps,
    #     ).indices
    pts = query_pts[..., topk_idx, :].squeeze(0)
    widths = width if width is not None else torch.tensor([max_width] * n_grasps, device=query_pts.device)
    widths = widths.squeeze()[topk_idx] if width is not None else widths

    slacks = slack if slack is not None else torch.tensor([0.0] * n_grasps, device=query_pts.device)
    slacks = slacks.squeeze()[topk_idx] if slack is not None else slacks

    normals = normals[..., topk_idx, :]

    ori = torch.topk(affordance[..., topk_idx, :], k=n_orientations, dim=-1)  #

    orientations = ori.indices

    # Make the right dimensions
    pts = pts.repeat(1, n_orientations).view(-1, 3)#.cpu().detach().numpy()
    normals = normals.repeat(1, n_orientations).view(-1, 3)#.cpu().detach().numpy()
    widths = widths.repeat(1, n_orientations).view(-1)#.cpu().detach().numpy()
    slacks = slacks.repeat(1, n_orientations).view(-1)#.cpu().detach().numpy()
    orientations = orientations.view(-1)#.cpu().detach().numpy()
    oris, contacts = [], []
    topk_idx = topk_idx.repeat_interleave(n_orientations).ravel()#.cpu().detach().numpy()


    # s = []
    # raw_points = []
    # for i in range(len(pts)):
    #     o, c, other = convert_contact_to_grasp(normals[i].cpu().detach().numpy(), pts[i].cpu().detach().numpy(), orientations[i].cpu().detach().numpy(), widths[i].cpu().detach().numpy(), offset=0.006 + slacks[i].cpu().detach().numpy() / 2)
    #     raw_points.append(pts[i])
    #     oris.append(o)
    #     contacts.append(c)
    #     s.append(other)
    oris, contacts = convert_contact_to_grasp_batched(normals, pts, orientations, widths, offset=0.006 + slacks / 2)


    if return_np:
        print("Returning numpy old mode")

        raw_points = []
        for i in range(len(pts)):
            o, c, other = convert_contact_to_grasp(normals[i].cpu().detach().numpy(), pts[i].cpu().detach().numpy(), orientations[i].cpu().detach().numpy(), widths[i].cpu().detach().numpy(), offset=0.006 + slacks[i].cpu().detach().numpy() / 2)
            raw_points.append(pts[i])
            oris.append(o)
            contacts.append(c)
        
        return (
            np.stack(oris),
            np.stack(contacts),
            ori.values.ravel().cpu().detach().numpy(),
            np.stack(raw_points),
            topk_idx.cpu().detach().numpy(),
            (widths + slacks).cpu().detach().numpy(),
        )
    return oris, contacts, ori.values.ravel(), pts, topk_idx, widths + slacks



def convert_contact_to_grasp_batched(
    normal: np.ndarray,
    contact_pt: np.ndarray,
    angle_id: int,
    width_pred: float = 0.08,
    offset=0.005,
    max_gripper_width=0.08
):
    # if width_pred is None:
    # width_pred = 0.08
    width_pred = (width_pred + 2 * offset).clip(0, max_gripper_width)
    gravity = torch.tensor([0.0, 0.0, 1.0], device=normal.device).repeat(len(normal), 1)
    y_axis = normal
    x_axis = torch.cross(gravity, y_axis)# torch.stack([-y_axis[:,1], y_axis[:,0], 0*y_axis[:,0]], dim=-1)
    # invalid_norms = y_axis[:, -2].abs() > 0.98
    # x_axis[invalid_norms, 0] = 1
    # x_axis[invalid_norms, 1:2] = 0
    z_axis = torch.cross(x_axis, y_axis)
    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)


    # R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)

    num_rotations = 12

    angle = -np.pi / 2 + angle_id / (num_rotations - 1) * (np.pi)
    rot = torch.eye(3, device=normal.device).repeat(len(normal), 1, 1)
    rot[:, 0, 0] = torch.cos(angle)
    rot[:, 0, 2] = torch.sin(angle)
    rot[:, 2, 0] = -torch.sin(angle)
    rot[:, 2, 2] = torch.cos(angle)


    ori = torch.bmm(R, rot)
    
    contact = contact_pt - (ori @  torch.stack([torch.zeros_like(offset), width_pred / 2 - offset, torch.zeros_like(offset) + 0.045],-1)[..., None]).squeeze(-1)

    # z_grasp = ori.as_matrix() @ np.array([0, 0, 1])
    # if z_grasp[-1] < 0.05:
    #     return ori.as_quat(), 5 * contact_pt
    # tf = Transform(ori, contact_pt - ori.apply(np.asarray([0, width_pred / 2 - NORMAL_OFFSET, 0.04])))
    # ori.apply(np.asarray([0,width_pred /2 - NORMAL_OFFSET,0.040])
    return ori, contact


def convert_contact_to_grasp(
    normal: np.ndarray,
    contact_pt: np.ndarray,
    angle_id: int,
    width_pred: float = 0.08,
    offset=0.005,
    max_gripper_width=0.08,
):
    # if width_pred is None:
    # width_pred = 0.08
    width_pred = min(max_gripper_width, width_pred + 2 * offset)

    y_axis = normal
    x_axis = np.r_[-y_axis[1], y_axis[0], 0]

    if np.abs(normal[-2]) > 0.98:  # top down grasp.
        x_axis = np.r_[1, 0, 0]

    z_axis = np.cross(x_axis, y_axis)

    R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)

    num_rotations = 12

    angle = -np.pi / 2 + angle_id / (num_rotations - 1) * (np.pi)
    ori = R * Rotation.from_euler("y", angle)

    ori.as_matrix() @ np.array([0, 0, 1])
    # if z_grasp[-1] < 0.05:
    #     return ori.as_quat(), 5 * contact_pt
    # tf = Transform(ori, contact_pt - ori.apply(np.asarray([0, width_pred / 2 - NORMAL_OFFSET, 0.04])))
    # ori.apply(np.asarray([0,width_pred /2 - NORMAL_OFFSET,0.040])
    return ori.as_quat(), contact_pt - ori.apply(np.asarray([0, width_pred / 2 - offset, 0.045])), {"ori": ori.as_matrix(), "width": width_pred, "angle": angle, "normal": normal, "contact": contact_pt, "R": np.vstack((x_axis, y_axis, z_axis)).T}
