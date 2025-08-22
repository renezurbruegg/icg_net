from math import cos, sin
import time

import numpy as np
import open3d as o3d

from vgn.utils.transform import Transform


class CameraIntrinsic(object):
    """Intrinsic parameters of a pinhole camera model.

    Attributes:
        width (int): The width in pixels of the camera.
        height(int): The height in pixels of the camera.
        K: The intrinsic camera matrix.
    """

    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    def to_dict(self):
        """Serialize intrinsic parameters to a dict object."""
        data = {
            "width": self.width,
            "height": self.height,
            "K": self.K.flatten().tolist(),
        }
        return data

    @classmethod
    def from_dict(cls, data):
        """Deserialize intrinisic parameters from a dict object."""
        intrinsic = cls(
            width=data["width"],
            height=data["height"],
            fx=data["K"][0],
            fy=data["K"][4],
            cx=data["K"][2],
            cy=data["K"][5],
        )
        return intrinsic


class TSDFVolume(object):
    """Integration of multiple depth images using a TSDF."""

    def __init__(self, size, resolution):
        self.size = size
        self.resolution = resolution
        self.voxel_size = self.size / self.resolution
        self.sdf_trunc = 4 * self.voxel_size

        self._volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=self.size,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )

    def integrate(self, depth_img, intrinsic, extrinsic, rgb_img=None):
        """
        Args:
            depth_img: The depth image.
            intrinsic: The intrinsic parameters of a pinhole camera model.
            extrinsics: The transform from the TSDF to camera coordinates, T_eye_task.
        """
        if rgb_img is None:
            rgb_img = np.zeros_like(depth_img)[...,None].repeat(3, axis = -1).astype(np.uint8)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.empty_like(depth_img)),
            # o3d.geometry.Image(rgb_img),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False,
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )

        extrinsic = extrinsic.as_matrix()
        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def index_of(self, x, y, z):
        return x * self._volume.resolution*self._volume.resolution + y * self._volume.resolution + z
    
 
    def get_grid(self):
        # TODO(mbreyer) very slow (~35 ms / 50 ms of the whole pipeline)
        shape = (1, self.resolution, self.resolution, self.resolution)
        tsdf_grid = np.zeros(shape, dtype=np.float32)
        # print("extracting grid")
        # print("calling o3d")
        # c++ source code
        #
        # inline int IndexOf(int x, int y, int z) const {
        #     return x * resolution_ * resolution_ + y * resolution_ + z;
        # }
        # for (int x = 0; x < resolution_; x++) {
        #     for (int y = 0; y < resolution_; y++) {
        #         for (int z = 0; z < resolution_; z++) {
        #             const int ind = IndexOf(x, y, z);
        #             const float f = voxels_[ind].tsdf_;
        #             const float w = voxels_[ind].weight_;
        #             sharedvoxels_[ind] = Eigen::Vector2d(f, w);
        #         }
        #     }
        # }
        if o3d.__version__ > "0.13.0":
            print("getting grid the new way!")
            tsdf_vector = np.asarray(self._volume.extract_volume_tsdf()) # Needed to work with open3d > 0.14.1
            for x in range(self.resolution):
                for y in range(self.resolution):
                    for z in range(self.resolution):
                        ind = self.index_of(x, y, z)

                        f,w = tsdf_vector[ind][0], tsdf_vector[ind][1]
                        if w != 0 and f <= 0.98 and f >= -0.98:
                            tsdf_grid[0, x, y, z] = 0.5*(1 + f) # tsdf value

            return tsdf_grid
        else:
            print("getting grid th old way!")
            shape = (1, self.resolution, self.resolution, self.resolution)
            tsdf_grid = np.zeros(shape, dtype=np.float32)
            voxels = self._volume.extract_voxel_grid().get_voxels()
            for voxel in voxels:
                i, j, k = voxel.grid_index
                tsdf_grid[0, i, j, k] = voxel.color[0]
        return tsdf_grid
        # print("done")

        # print("done")
        return tsdf_grid


    def get_cloud(self):
        return self._volume.extract_point_cloud()


def create_tsdf(size, resolution, depth_imgs, intrinsic, extrinsics, rgb_imgs=None):
    tsdf = TSDFVolume(size, resolution)
    for i in range(depth_imgs.shape[0]):
        extrinsic = Transform.from_list(extrinsics[i])
        if rgb_imgs is not None:
            tsdf.integrate(depth_imgs[i], intrinsic, extrinsic, rgb_img = rgb_imgs[i])
        else:
            tsdf.integrate(depth_imgs[i], intrinsic, extrinsic)

    return tsdf


def camera_on_sphere(origin, radius, theta, phi):
    eye = np.r_[
        radius * sin(theta) * cos(phi),
        radius * sin(theta) * sin(phi),
        radius * cos(theta),
    ]
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])  # this breaks when looking straight down
    return Transform.look_at(eye, target, up) * origin.inverse()
