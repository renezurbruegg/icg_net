""" This script saves all grasps in the respective canonical form."""

import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation
import pandas as pd
import tqdm
import pickle
import trimesh

from utils.grasp_utils import convert_contact_to_grasp
from utils.grasp_visualizer import create_our_gripper_marker, create_vgn_gripper_marker

STORE_LOCATION = "/data/raw/contact_single_object_packed_1M.pkl"
SCENE_TO_PARSE = "/data/raw/contact_single_object_packed_1M"
APPEND = False

df = pd.read_csv(os.path.join(SCENE_TO_PARSE, "grasps_candidate.csv"))

all_grasp_data = {}

def show_clouds(*args):
    scene = trimesh.Scene()
    for arg in args:
        scene.add_geometry(trimesh.PointCloud(vertices = arg))
    return scene

for scene in tqdm.tqdm(df.scene_id.unique()):
    grasps = df[df.scene_id == scene]
    pose_file = os.path.join(SCENE_TO_PARSE, "mesh_pose_list", f"{scene}.npz")
    file_path, scale, pose = np.load(pose_file, allow_pickle=True)["pc"][0]
    pose_mesh_pc = np.linalg.inv(pose)

    urdf_name = os.path.basename(file_path)

    x_axis = grasps[["ax_x", "ax_y", "ax_z"]].to_numpy()
    y_axis = grasps[["ay_x", "ay_y", "ay_z"]].to_numpy()
    z_axis = grasps[["az_x", "az_y", "az_z"]].to_numpy()
    orientations = np.stack([x_axis, y_axis, z_axis], axis=-1)

    contacts = grasps[["x", "y", "z"]].to_numpy()
    widths = grasps["width"].to_numpy()
    labels = grasps[[f"label_{i}" for i in range(12)]].to_numpy()

    widths *= 1 / scale
    contacts = (pose_mesh_pc[:3, :3] @ contacts.T).T + pose_mesh_pc[:-1, -1]
    contacts *= 1 / scale

    orientations = (pose_mesh_pc[:3, :3] @ orientations.T).T
    # poses = 
    poses = np.repeat(pose[None], len(orientations), 0)
    

    selected = []


    # import open3d as o3d
    # pc = o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(contacts)
    # pc.normals = o3d.utility.Vector3dVector(orientations[:,:,1])
    # o3d.visualization.draw_geometries([pc])

    # for idx in range(len(contacts)):
    #     markers = []

    #     scene = show_clouds(contacts)

    #     contact = contacts[idx]
    #     width = widths[idx]
    #     label = labels[idx]
    #     ori = orientations[idx]

    #     normal = orientations[idx][:, 1]


        # print(idx, label)
        # marker = create_our_gripper_marker(color = [1, 0, 0])
                
        # se3_matrix = np.eye(4)
        # se3_matrix[:3, -1] = contact
        # se3_matrix[:3, :3] = rot_mat = ori

        # markers.append(marker.apply_transform(se3_matrix))

        # for o_idx, ori_idx in enumerate(label):
        #     if True or ori_idx == 1:
        #         ori_q, pos = convert_contact_to_grasp(y_axis[idx], contact, angle_id = o_idx)
        #         marker = create_vgn_gripper_marker()
                
        #         se3_matrix = np.eye(4)
        #         se3_matrix[:3, -1] = pos
        #         se3_matrix[:3, :3] = rot_mat =  pose_mesh_pc[:3, :3] @ Rotation.from_quat(ori_q).as_matrix()
        #         gripper_r = marker.apply_transform(se3_matrix)
        #         markers.append(gripper_r)

        # for m in markers:
        #     scene.add_geometry(m)

        # scene.add_geometry(trimesh.load(file_path))
        # scene.show()

    if urdf_name not in all_grasp_data:
        all_grasp_data[urdf_name] = {
            "orientations": orientations,   
            "contacts": contacts,
            "widths": widths,
            "labels": labels,
            "poses": poses,
        }
    else:
        previous_data = all_grasp_data[urdf_name]
        all_grasp_data[urdf_name]["orientations"] = np.concatenate(
            [previous_data["orientations"], orientations]
        )
        all_grasp_data[urdf_name]["contacts"] = np.concatenate(
            [previous_data["contacts"], contacts]
        )
        all_grasp_data[urdf_name]["widths"] = np.concatenate(
            [previous_data["widths"], widths]
        )
        all_grasp_data[urdf_name]["labels"] = np.concatenate(
            [previous_data["labels"], labels]
        )
        all_grasp_data[urdf_name]["poses"] = np.concatenate(
            [previous_data["poses"], poses]
        )
        # Visualization related code

        # import trimesh
        # f = trimesh.load("/home/zrene/git/occupancy_prediction/" + file_path)
        # f.apply_scale(0.95)
        # f.apply_transform(pose)
        # trscene = trimesh.Scene()
        # trscene.add_geometry(f)
        # trscene.add_geometry(trimesh.points.PointCloud(all_grasp_data[urdf_name]["contacts"]))
        # trscene.show()
pickle.dump(all_grasp_data, open(STORE_LOCATION, "wb"))
