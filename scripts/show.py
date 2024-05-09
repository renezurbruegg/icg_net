import open3d as o3d
from icg_net import ICGNetModule, get_model
from icg_net.vis import GraspVisualizer
import argparse
import os
import numpy as np
import torch
import random
import glob


def set_seed(seed: int = 42) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == "__main__":
    set_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="/home/rene/ICRA_2024/icg_benchmark/data/51--0.656/config.yaml"
    )
    parser.add_argument("--path_to_pc", type=str, default="data/pc")
    args = parser.parse_args()

    model: ICGNetModule = get_model(
        args.config,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    i = 0
    for file in glob.glob(os.path.join(args.path_to_pc, "*.np*")):
        print("file", file)
        i += 1
        points = np.load(file)["pc"]
        normals = None
        pc, normals = points[:, :3], None  # points[:, 6:9]

        pc = torch.from_numpy(pc).float()

        o3dc = o3d.geometry.PointCloud()
        o3dc.points = o3d.utility.Vector3dVector(pc[..., :3].numpy())

        if normals is not None:
            o3dc.normals = o3d.utility.Vector3dVector(normals.numpy())
        else:
            o3dc.estimate_normals()
            try:
                f = file.replace("pc", "scenes")
                extrinsics = np.load(f)["extrinsics"]
                cam_loc = extrinsics[0, -3:]
                cam_loc[0] = -cam_loc[0]
                o3dc.orient_normals_towards_camera_location(cam_loc)
            except:
                pass

        # grasp downsample
        ds = o3dc.voxel_down_sample(voxel_size=0.001)
        # filter if needed (real world data)
        ds, _ = ds.remove_radius_outlier(nb_points=15, radius=0.005)
        ds = o3dc.voxel_down_sample(voxel_size=0.003)

        grasp_normals = torch.from_numpy(np.asarray(ds.normals)).float().cuda()
        grasp_pts = torch.from_numpy(np.asarray(ds.points)).float().cuda()

        out = model(
            torch.from_numpy(np.asarray(o3dc.points)).float(),
            normals=torch.from_numpy(np.asarray(o3dc.normals)).float(),
            grasp_pts=grasp_pts,
            grasp_normals=grasp_normals,
            visualize=True,
            n_grasps=512,
            each_object=True,
            return_meshes=True,
            return_scene_grasps=True,
        )

        GraspVisualizer.from_prediction(
            predictions=out, num_grasp=512, filter_th=0.2, max_grasps_per_instance=14
        ).show()
