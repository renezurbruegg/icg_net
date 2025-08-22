import numpy as np
from scipy import ndimage
import torch.utils.data

from vgn.io import *
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, root_raw, augment=False, return_pos=False):
        self.root = root
        self.augment = augment
        self.return_pos = return_pos
        self.df = read_df(root_raw)
        voxel_size = 0.3/ 40
        self.df["x"] /= voxel_size
        self.df["y"] /= voxel_size
        self.df["z"] /= voxel_size
        self.df["width"] /= voxel_size
        self.df = self.df.rename(columns={"x": "i", "y": "j", "z": "k"})

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        done = False
        while not done:
            scene_id = self.df.loc[i, "scene_id"]
            ori = Rotation.from_quat(self.df.loc[i, "qx":"qw"].to_numpy(np.single))
            pos = self.df.loc[i, "i":"k"].to_numpy(np.single)
            width = self.df.loc[i, "width"].astype(np.single)
            label = self.df.loc[i, "label"].astype(np.long)

            try:
                voxel_grid = read_voxel_grid(self.root, scene_id)        
                done=True
            except FileNotFoundError as e:
                i += 1
                print("Did not find voxel grid for scene", scene_id, "next", i)
                done = False

        if self.augment:
            voxel_grid, ori, pos = apply_transform(voxel_grid, ori, pos)

        index = np.round(pos).astype(np.long)
        rotations = np.empty((2, 4), dtype=np.single)
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        rotations[0] = ori.as_quat()
        rotations[1] = (ori * R).as_quat()

        x, y, index = voxel_grid, (label, rotations, width), index
        if self.return_pos:
            return x, y, pos
        else:
            return x, y, index


def apply_transform(voxel_grid, orientation, position):
    angle = np.pi / 2.0 * np.random.choice(4)
    R_augment = Rotation.from_rotvec(np.r_[0.0, 0.0, angle])

    z_offset = np.random.uniform(6, 34) - position[2]

    t_augment = np.r_[0.0, 0.0, z_offset]
    T_augment = Transform(R_augment, t_augment)

    T_center = Transform(Rotation.identity(), np.r_[20.0, 20.0, 20.0])
    T = T_center * T_augment * T_center.inverse()

    # transform voxel grid
    T_inv = T.inverse()
    matrix, offset = T_inv.rotation.as_matrix(), T_inv.translation
    voxel_grid[0] = ndimage.affine_transform(voxel_grid[0], matrix, offset, order=0)

    # transform grasp pose
    position = T.transform_point(position)
    orientation = T.rotation * orientation

    return voxel_grid, orientation, position
