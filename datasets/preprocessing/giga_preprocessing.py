import re
import os
from types import SimpleNamespace
import numpy as np
from fire import Fire
from natsort import natsorted
from loguru import logger
from base_preprocessing import BasePreprocessing
from scripts.save_mesh_scene import save_scene_mesh
from scripts.save_sdf_data_parallel import save_sdf
from scipy.spatial.transform import Rotation
import pandas as pd
import pickle


def show_pc(pc, colors=None):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


class CustomPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "./data/raw/custom",
        save_dir: str = "./data/processed/custom",
        mesh_root: str = "/home/zrene/git/GIGA",
        modes: tuple = ("train", "val"),
        store_location: str = None,
        n_jobs: int = -1,
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)

        valid_split_size = 0.1  # 10% of the data is used for validation

        self.store_location = store_location
        if store_location is not None:
            self.grasp_store_data = pickle.load(open(self.store_location, "rb"))
        else:
            self.grasp_store_data = None
        self.mesh_root = mesh_root
        self.class_map = {
            # N/A
            "table": 0,
            "other": 1,
            # 'other_2': 2,
            # 'wall': 2,
            # 'beam': 3,
            # 'column': 4,
            # 'window': 5,
            # 'door': 6,
            # 'table': 7,
            # 'chair': 8,
            # 'sofa': 9,
            # 'bookcase': 10,
            # 'board': 11,
            # 'clutter': 12,
            # 'stairs': 12  # stairs are also mapped to clutter
        }

        self.color_map = [
            [0, 255, 0],  # ceiling
            [0, 0, 255],  # floor
            [0, 255, 255],  # wall
            [255, 255, 0],  # beam
            [255, 0, 255],  # column
            [100, 100, 255],  # window
            [200, 200, 100],  # door
            [170, 120, 200],  # table
            [255, 0, 0],  # chair
            [200, 100, 100],  # sofa
            [10, 200, 100],  # bookcase
            [200, 200, 200],  # board
            [50, 50, 50],
        ]  # clutter

        # self.create_label_database()

        for mode in self.modes:
            filepaths = []
            files = sorted(
                [f.path for f in os.scandir(self.data_dir / "full_pointcloud")]
            )  # only 500 scenes for now
            if mode == "train":
                files = files[: int(len(files) * (1 - valid_split_size))]
            elif mode == "val":
                files = files[int(len(files) * (1 - valid_split_size)) :]

            for scene_path in files:
                filepaths.append(scene_path)
            logger.info("mode", mode, "files", len(filepaths))

            self.files[mode] = natsorted(filepaths)

        # logger.info(self.files)

        self.create_label_database()

    def create_label_database(self):
        label_database = dict()
        for class_name, class_id in self.class_map.items():
            label_database[class_id] = {
                "color": self.color_map[class_id],
                "name": class_name,
                "validation": True,
            }

        self._save_yaml(self.save_dir / "label_database.yaml", label_database)
        return label_database

    # def _buf_count_newlines_gen(self, fname):
    #     def _make_gen(reader):
    #         while True:
    #             b = reader(2 ** 16)
    #             if not b: break
    #             yield b

    #     with open(fname, "rb") as f:
    #         count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
    #     return count

    def process_file(self, filepath, mode):
        print("preocssing file", filepath, "mode", mode)

        # parser.add_argument("--num-proc", type=int, default=1)
        # parser.add_argument("raw", type=str)
        """process_file.

        Please note, that for obtaining segmentation labels ply files were used.

        Args:
            filepath: path to the main ply file
            mode: train, test or validation

        Returns:
            filebase: info about file
        """
        parent_dir = os.path.dirname(filepath)
        ds_root = os.path.dirname(parent_dir)

        print(parent_dir)

        scene_file = filepath.replace("full_pointcloud", "mesh_pose_list")
        scene_name = os.path.basename(scene_file)[:-4]
        print("saving scene file", scene_file, "exists?", os.path.exists(scene_file))
        save_scene_mesh(
            scene_file,
            SimpleNamespace(raw=self.save_dir / mode, mesh_root=self.mesh_root),
        )

        sdf_args = SimpleNamespace(
            raw=filepath,
            dataset=self.save_dir / mode,
            num_point_per_file=100000,
            num_file=1,
            mesh_root=self.mesh_root,
            uniform=True,
        )
        print("saving sdf file", scene_file, "exists?", os.path.exists(scene_file))
        if not os.path.exists(self.save_dir / mode / "sdf" / scene_name / "0000.npz"):
            save_sdf(scene_file, sdf_args)

        scene_np = np.load(scene_file, allow_pickle=True)
        object_poses = [p[2] for p in scene_np["pc"]]
        object_names = [os.path.basename(p[0]) for p in scene_np["pc"]]
        object_scales = [p[1] for p in scene_np["pc"]]
        object_positions = [p[:-1, -1] for p in object_poses]

        sdf_file = np.load(self.save_dir / mode / "sdf" / scene_name / "0000.npz")
        num_sdf_instances = sdf_file["sdf"].shape[0]
        num_mesh_instances = len(
            [
                f
                for f in os.listdir(self.save_dir / mode / "obj_scenes")
                if os.path.basename(scene_file)[:-4] + "_instance_" in f
            ]
        )
        assert (
            num_sdf_instances == num_mesh_instances
        ), "num_sdf_instances {} != num_mesh_instances {}".format(
            num_sdf_instances, num_mesh_instances
        )

        # save_sdf()

        # parser.add_argument("raw", type=str)
        # parser.add_argument("--dataset", type=str, default = None)
        # parser.add_argument("num_point_per_file", type=int)
        # parser.add_argument("num_file", type=int)
        # parser.add_argument("mesh_root", type=str, default = "")
        # parser.add_argument("--uniform", action='store_true', help='sample uniformly in the bbox, else sample in the tight bbox')

        filebase = {
            "filepath": filepath,
            "scene": os.path.basename(filepath),
            "mode": mode,
            "raw_filepath": str(filepath),
            "file_len": -1,
        }
        np_file = np.load(filepath)
        coords = np_file["pc"].astype(np.float32)
        colors = np_file["colors"].astype(np.float32)
        normals = np_file["normals"].astype(np.float32)
        instances = np_file["instances"].astype(np.int32).reshape(-1, 1)
        instance_class = (
            (np_file["instances"].astype(np.int32) > 0).astype(np.int32).reshape(-1, 1)
        )
        points = np.concatenate(
            [coords, colors, normals, instance_class, instances], axis=1
        )

        if len(np.unique(instances)) != num_sdf_instances:
            print(
                filepath,
                "num_sdf_instances {} != num_pts_instances {}. Occlusion?".format(
                    num_sdf_instances, np.max(instances)
                ),
            )

        if np.isnan(points).any():
            raise ValueError("FOUND NANS")

        points = np.hstack((points, np.ones(points.shape[0])[..., None]))
        points[:, [9, 10, -1]] = points[:, [-1, 9, 10]]  # move segments after RGB

        grasps = pd.read_csv(os.path.join(ds_root, "grasps.csv"))
        grasps = grasps[grasps["scene_id"] == os.path.basename(filepath)[:-4]]

        print(f"Found # {len(grasps)} grasps for {os.path.basename(filepath)}")

        quat = grasps[["qx", "qy", "qz", "qw"]].to_numpy()
        # x_axis = grasps[["ax_x", "ax_y", "ax_z"]].to_numpy()
        # y_axis = grasps[["ay_x", "ay_y", "ay_z"]].to_numpy()
        # z_axis = grasps[["az_x", "az_y", "az_z"]].to_numpy()
        # orientations = np.stack([x_axis, y_axis, z_axis], axis = -1)
        contacts = grasps[["x", "y", "z"]].to_numpy()
        widths = grasps["width"].to_numpy().reshape(-1, 1)
        labels = grasps[["label"]].to_numpy() == 1
        target = grasps["target"].to_numpy().astype(np.int32)
        # 4 + 1 + 1 =
        scene_centric_labels = np.zeros((num_sdf_instances, len(grasps), 6))
        object_centric_labels = np.zeros((num_sdf_instances, len(grasps), 6))

        for instance_id in np.unique(target):
            target_mask = target == instance_id
            target_labels = labels[target_mask]

            object_centric_succ = np.logical_or(
                target_labels == 1, target_labels == -1
            )  # success even if we collide with environment
            object_centric_labels[instance_id, target_mask] = np.concatenate(
                [quat[target_mask], widths[target_mask], object_centric_succ], axis=-1
            )
            scene_centric_labels[instance_id, target_mask] = np.concatenate(
                [quat, widths, labels], axis=-1
            )[target_mask]
            object_centric_query_pts = contacts

        if self.grasp_store_data:
            point_indices = [0] + [
                len(self.grasp_store_data.get(n, {}).get("contacts", []))
                for n in object_names
            ]

            point_cumsum = np.cumsum(point_indices)
            object_centric_query_pts = np.zeros((point_cumsum[-1], 3))
            # 4 + 1 + 1, pts, quats, widths, labels
            object_centric_labels = np.zeros((num_sdf_instances, sum(point_indices), 6))

            for target_id, name in enumerate(object_names):
                target_id = target_id + 1  # we skip table instance
                if name not in self.grasp_store_data:
                    print(f"Missing object {name} in grasp storage!")
                    continue
                grasp_data = self.grasp_store_data[name]
                contacts_n, widths_n, labels_n = (
                    grasp_data["contacts"] * object_scales[target_id - 1],
                    grasp_data["widths"] * object_scales[target_id - 1],
                    grasp_data["labels"],
                )
                widths_n = widths_n.reshape(-1, 1)
                pose = object_poses[target_id - 1]
                contacts_t = (pose[:-1, :-1] @ contacts_n.T).T + pose[:-1, -1]
                quat_t = (
                    Rotation.from_matrix(pose[:-1, :-1])
                    * Rotation.from_quat(grasp_data["quat"])
                ).as_quat()
                object_centric_labels[
                    target_id, point_cumsum[target_id - 1] : point_cumsum[target_id], :
                ] = np.concatenate([quat_t, widths_n, labels_n], axis=-1)

                corr = (0.08 - widths_n) / 2
                object_centric_query_pts[
                    point_cumsum[target_id - 1] : point_cumsum[target_id], :
                ] = (
                    contacts_t
                    + Rotation.from_quat(quat_t).as_matrix()[:, :, -2] * (corr - 0.003)
                    # + 0.006  # const offset
                )

                # path = os.path.join("/home/zrene/git/occupancy_prediction", scene_np['pc'][target_id-1][0])
                # import trimesh
                # f = trimesh.load(path)
                # #f.apply_scale(0.95)
                # pose = object_poses[target_id - 1]
                # f.apply_transform(pose)
                # trscene = trimesh.Scene()
                # trscene.add_geometry(f)
                # pts = ((pose[:-1,:-1] @ contacts.T).T + pose[:-1, -1])
                # trscene.add_geometry(trimesh.points.PointCloud(pts))
                # trscene.show()

            # Got grasp store. Load object specific grasps from there.

        # Save Grasps:
        processes_grasps_filepath_qual = (
            self.save_dir / mode / "grasps" / f"{scene_name}_scene_quality.npy"
        )
        processes_grasps_filepath_qual_object = (
            self.save_dir / mode / "grasps" / f"{scene_name}_object_quality.npy"
        )
        processes_grasps_filepath_query = (
            self.save_dir / mode / "grasps" / f"{scene_name}_scene_query.npy"
        )
        processes_grasps_filepath_query_object = (
            self.save_dir / mode / "grasps" / f"{scene_name}_object_query.npy"
        )

        if not processes_grasps_filepath_qual.parent.exists():
            processes_grasps_filepath_qual.parent.mkdir(parents=True, exist_ok=True)

        # grasp_labels = np.stack([object_centric_labels, scene_centric_labels])
        # shape of grasp labels is (2, num_instances, num_grasps, num_labels)
        np.save(processes_grasps_filepath_qual, scene_centric_labels)
        np.save(processes_grasps_filepath_qual_object, object_centric_labels)

        # general_grasp_data = np.concatenate([contacts, y_axis, widths[..., None], orientations.reshape(-1, 9)], axis = -1)
        np.save(processes_grasps_filepath_query, contacts)
        np.save(processes_grasps_filepath_query_object, object_centric_query_pts)
        # 0:3 contact, 3:6 y_axis, 6:7 width, 7:16 orientation

        filebase["file_len"] = len(points)
        # scene_name = os.path.basename(filepath).split(".")[0]

        gt_data = (points[:, -2] + 1) * 1000 + points[:, -1] + 1
        processed_filepath = self.save_dir / mode / f"{scene_name}.npy"
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        processed_gt_filepath = (
            self.save_dir / "instance_gt" / mode / f"{scene_name}.txt"
        )
        if not processed_gt_filepath.parent.exists():
            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_data.astype(np.int32), fmt="%d")
        filebase["instance_gt_filepath"] = str(processed_gt_filepath)

        filebase["color_mean"] = [
            float((points[:, 3]).mean()),
            float((points[:, 4]).mean()),
            float((points[:, 5]).mean()),
        ]
        filebase["color_std"] = [
            float(((points[:, 3]) ** 2).mean()),
            float(((points[:, 4]) ** 2).mean()),
            float(((points[:, 5]) ** 2).mean()),
        ]

        filebase["poses"] = [position.tolist() for position in object_positions]

        # if True:  # Visualize
        #     from grasp_visualizer import GraspVisualizer
        #     import open3d as o3d

        #     vis = GraspVisualizer()

        #     pts = points[:, :3]
        #     colors = points[:, 3:6]
        #     cloud = o3d.geometry.PointCloud()
        #     cloud.points = o3d.utility.Vector3dVector(pts)
        #     cloud.colors = o3d.utility.Vector3dVector(colors)
        #     vis.add_pointcloud(cloud)

        #     object = True
        #     tp = object_centric_labels if object else scene_centric_labels
        #     grasps = tp[0, ...]
        #     for g in tp:
        #         mask = g[:, -1] > grasps[:, -1]
        #         grasps[mask] = g[mask]

        #     pos_mask = grasps[:, -1] > 0
        #     pos_grasps = grasps[pos_mask]
        #     quat = pos_grasps[:, 0:4]
        #     pts = (
        #         contacts[pos_mask] if not object else object_centric_query_pts[pos_mask]
        #     )
        #     vis.add_grasps(list(zip(pts[:100], quat[:100])), widths=pos_grasps[:100, 4])
        #     vis.show()

        #     import pdb

        #     pdb.set_trace()

        return filebase

    # def compute_color_mean_std(
    #         self, train_database_path: str = ""
    # ):
    #     area_database_paths = [f for f in os.scandir(self.save_dir)
    #                            if f.name.startswith("Area_") and f.name.endswith(".yaml")]

    #     for database_path in area_database_paths:
    #         database = self._load_yaml(database_path.path)
    #         color_mean, color_std = [], []
    #         for sample in database:
    #             color_std.append(sample["color_std"])
    #             color_mean.append(sample["color_mean"])

    #         color_mean = np.array(color_mean).mean(axis=0)
    #         color_std = np.sqrt(np.array(color_std).mean(axis=0) - color_mean ** 2)
    #         feats_mean_std = {
    #             "mean": [float(each) for each in color_mean],
    #             "std": [float(each) for each in color_std],
    #         }
    #         self._save_yaml(self.save_dir / f"{database_path.name}_color_mean_std.yaml", feats_mean_std)

    #     for database_path in area_database_paths:
    #         all_mean, all_std = [], []
    #         for let_out_path in area_database_paths:
    #             if database_path == let_out_path:
    #                 continue

    #             database = self._load_yaml(let_out_path.path)
    #             for sample in database:
    #                 all_std.append(sample["color_std"])
    #                 all_mean.append(sample["color_mean"])

    #         all_color_mean = np.array(all_mean).mean(axis=0)
    #         all_color_std = np.sqrt(np.array(all_std).mean(axis=0) - all_color_mean ** 2)
    #         feats_mean_std = {
    #             "mean": [float(each) for each in all_color_mean],
    #             "std": [float(each) for each in all_color_std],
    #         }
    #         file_path = database_path.name.replace("_database.yaml", "")
    #         self._save_yaml(self.save_dir / f"{file_path}_color_mean_std.yaml", feats_mean_std)

    # @logger.catch
    # def fix_bugs_in_labels(self):
    #     pass

    # def joint_database(self, train_modes=("Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6")):
    #     for mode in train_modes:
    #         joint_db = []
    #         for let_out in train_modes:
    #             if mode == let_out:
    #                 continue
    #             joint_db.extend(self._load_yaml(self.save_dir / (let_out + "_database.yaml")))
    #         self._save_yaml(self.save_dir / f"train_{mode}_database.yaml", joint_db)

    # def _parse_scene_subscene(self, name):
    #     scene_match = re.match(r"scene(\d{4})_(\d{2})", name)
    #     return int(scene_match.group(1)), int(scene_match.group(2))

    def joint_database(self, train_modes=("train", "val")):
        joint_db = []
        for mode in train_modes:
            joint_db.extend(self._load_yaml(self.save_dir / (mode + "_database.yaml")))
        self._save_yaml(self.save_dir / "train_validation_database.yaml", joint_db)


if __name__ == "__main__":
    Fire(CustomPreprocessing)
