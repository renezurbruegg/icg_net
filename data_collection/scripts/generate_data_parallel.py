import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import open3d as o3d
import scipy.signal as signal
from tqdm import tqdm
import multiprocessing as mp
print("Starting imports")
from vgn.grasp import Grasp, Label
from vgn.io import *
from vgn.perception import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform
from vgn.utils.implicit import get_mesh_pose_list_from_world, get_scene_from_mesh_pose_list
import time
import trimesh
import yaml
print("Imports done")

OBJECT_COUNT_LAMBDA = 5
MAX_VIEWPOINT_COUNT = 6
NORMAL_OFFSET = 0.006

@dataclass
class GraspCandidate:
    contact: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray
    pitch_angles: List[float]
    widths: List[float]
    sucess: List[bool]
    target: List[int]
    
def main(args, rank):
    np.random.seed(args.seed + rank*100)
    print("Creating simulation environment")
    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.sim_gui, debug = args.debug, gripper_urdf=args.gripper_urdf, urdf_root=args.urdf_root)
    print("Simulation environment created")
    finger_depth = sim.gripper.finger_depth
    grasps_per_worker = args.num_grasps // args.num_proc
    pbar = tqdm(total=grasps_per_worker, disable=rank != 0)

    print("runnning with ", args.num_proc, " runners and ", grasps_per_worker, "grasps per worker")
    if rank == 0:
        (args.root / "scenes").mkdir(parents=True)
        (args.root / "info.yaml").write_text(yaml.dump(vars(args)))
        write_setup(
            args.root,
            sim.size,
            sim.camera.intrinsic,
            sim.gripper.max_opening_width,
            sim.gripper.finger_depth,
        )
        if args.save_scene:
            (args.root / "mesh_pose_list").mkdir(parents=True)
            (args.root / "pointcloud").mkdir(parents=True)
            (args.root / "full_pointcloud").mkdir(parents=True)


    for _ in range(grasps_per_worker // args.grasps_per_scene):
        # generate heap
        object_count = np.random.poisson( args.object_count_lambda) + 1
        sim.reset(object_count)
        sim.save_state()

          # render synthetic depth images
        depth_imgs, rgb_imgs, semseg_imgs, extrinsics = render_images(sim, args.num_views)
        mesh_pose_list = get_mesh_pose_list_from_world(sim.world, args.object_set)
        scene : trimesh.Trimesh = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False)

        if scene is None:
            continue

         # Load surface normals
        pointclouds = []
        bounding_box_no_table = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
        bounding_box_table = o3d.geometry.AxisAlignedBoundingBox(sim.lower - 0.05, sim.upper)
        
        instances = []
        for semseg, col, d,e in zip(semseg_imgs, rgb_imgs, depth_imgs, extrinsics):
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(col),
                o3d.geometry.Image(d),
                depth_scale=1.0,
                depth_trunc=2.0,
                convert_rgb_to_intensity=False,
            )

            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=sim.camera.intrinsic.width,
                height=sim.camera.intrinsic.height,
                fx=sim.camera.intrinsic.fx,
                fy=sim.camera.intrinsic.fy,
                cx=sim.camera.intrinsic.cx,
                cy=sim.camera.intrinsic.cy,
            )
            extrinsic = Transform.from_list(e).as_matrix()

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic)
            
            pcd_instances = o3d.geometry.PointCloud()
            pcd_instances.points = pcd.points
            inst = np.zeros((len(pcd.points), 3))
            inst[:,0] = np.asarray(semseg.ravel())
            pcd_instances.colors = o3d.utility.Vector3dVector(inst)

            points = np.asarray(pcd.points)
            
            table_mask = points[:,2] <= 0.053
            table_points = points[table_mask, :] 
            object_points = points[~table_mask, :] 
            normals = np.zeros_like(points)
            normals[table_mask, 2] = 1
            # import pdb; pdb.set_trace()
            # import matplotlib.pyplot as plt
            # plt.imshow(col)
            # plt.show()
            # plt.imshow(semseg)
            # plt.show()
            # o3d.visualization.draw_geometries([pcd_instances])
       

            (closest_points, distances, triangle_id) = scene.nearest.on_surface(object_points)
            normals[~table_mask, :] = scene.face_normals[triangle_id]
            pcd.normals = o3d.utility.Vector3dVector(normals)

            pointclouds.append(pcd.crop(bounding_box_table))
            instances.append(pcd_instances.crop(bounding_box_table))

        # Cobmine together
        pts, normals, colors, all_instances = [], [], [], []
        for pc, inst in zip(pointclouds, instances):
            pts.append(np.asarray(pc.points))
            normals.append(np.asarray(pc.normals))
            colors.append(np.asarray(pc.colors))
            all_instances.append(np.asarray(inst.colors)[:,0])
            # instances.append(np.ones_like(np.asarray(pc.points)) * inst)

        full_pc = o3d.geometry.PointCloud()
        full_pc.points = o3d.utility.Vector3dVector(np.concatenate(pts))
        full_pc.colors = o3d.utility.Vector3dVector(np.concatenate(colors))
        full_pc.normals = o3d.utility.Vector3dVector(np.concatenate(normals))
        pc = full_pc.crop(bounding_box_no_table).voxel_down_sample(voxel_size=0.0025)

        # o3d.visualization.draw_geometries([pc])
        if pc.is_empty():
            print("Point cloud empty, skipping scene")
            continue    

        # store the raw data
        scene_id = write_sensor_data(args.root, depth_imgs, extrinsics, rgb=rgb_imgs, semseg=semseg_imgs)
        if args.save_scene:
            write_point_cloud(args.root, scene_id, mesh_pose_list, name="mesh_pose_list")
            write_point_cloud(args.root, scene_id, np.asarray(pc.points), name="pointcloud", colors= np.asarray(pc.colors), normals = np.asarray(pc.normals))
            write_point_cloud(args.root, scene_id, np.asarray(full_pc.points), name="full_pointcloud", colors= np.asarray(full_pc.colors), normals = np.asarray(full_pc.normals), instances = np.concatenate(all_instances))

        # Do the grasping

        if not args.contact_based: # GIGA / VGN based grasps
            for _ in range(args.grasps_per_scene):
                # sample and evaluate a grasp point
                point, normal = sample_grasp_point(pc, finger_depth)

                    
                print("DEBUG DRAWING")
                point, normal = sample_grasp_point_contact(pc, horizontal_percentile=args.horizontal_percentile)

  # draw line with contact points and surface normal using pybullet
                sim.world.p.addUserDebugLine(point, point + normal*0.5, (1,0,0))

                grasp, label, target = evaluate_grasp_point(sim, point, normal)

                # store the sample
                write_grasp(args.root, scene_id, grasp, label, target=target)
                pbar.update()
        else:
            if args.sample_furthest:
                points, normals = sample_grasp_point_contact_furthest(pc, args.grasps_per_scene)

                for point,normal in zip(points, normals):
                    grasp, label, candidate = evaluate_grasp_point_contact(sim, point, normal, num_rotations=args.num_rotations, debug=args.debug)

                    if np.any(label):
                        succ += 1
                    for g,l in zip(grasp, label):
                        f_grasp = Grasp(g[0],g[1])
                        # store the sample
                        write_grasp(args.root, scene_id, f_grasp, l,  candidate)
                    if rank == 0:
                        pbar.update()
            else:
                for _ in range( args.grasps_per_scene):
                    if _ % 20 == 19:
                        print(f"Worker: {rank}, Grasp: {_}/{ args.grasps_per_scene}")
                    # sample and evaluate a grasp point

                    point, normal = sample_grasp_point_contact(pc, horizontal_percentile=args.horizontal_percentile)
                    # draw line with contact points and surface normal using pybullet
                    sim.world.p.addUserDebugLine(point, point + normal*0.5, (1,0,0))
                    grasp, label, candidate= evaluate_grasp_point_contact(sim, point, normal, num_rotations=args.num_rotations, debug=args.debug)
                    
                    for i, (g,l) in enumerate(zip(grasp, label)):
                        f_grasp = Grasp(g[0],g[1])
                        # store the sample
                        write_grasp(args.root, scene_id, f_grasp, l, candidate if i == 0 else None)
                    if rank == 0:
                        pbar.update()

    pbar.close()
    print('Process %d finished!' % rank)


def render_images(sim, n):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)
    rgb_imgs = np.empty((n, height, width, 3), np.uint8)
    semseg_imgs = np.empty((n, height, width), np.int32)

    for i in range(n):
        r = np.random.uniform(1.6, 2.4) * sim.size
        theta = np.random.uniform(0.0, np.pi / 4.0)
        phi = np.random.uniform(0.0, 2.0 * np.pi)

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        rgb, depth_img, semseg = sim.camera.render(extrinsic)

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img
        rgb_imgs[i] = rgb
        semseg_imgs[i] = semseg

    return depth_imgs, rgb_imgs, semseg_imgs, extrinsics

def render_side_images(sim, n=1, random=False):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, sim.size / 3])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)
    rgb_imgs = np.empty((n, height, width, 3), np.uint8)
    semseg_imgs = np.empty((n, height, width), np.int32)

    for i in range(n):
        if random:
            r = np.random.uniform(1.6, 2.4) * sim.size
            theta = np.random.uniform(np.pi / 4.0, 5.0 * np.pi / 12.0)
            phi = np.random.uniform(- 5.0 * np.pi / 5, - 3.0 * np.pi / 8.0)
        else:
            r = 2 * sim.size
            theta = np.pi / 3.0
            phi = - np.pi / 2.0

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        rgb, depth_img, semseg = sim.camera.render(extrinsic)

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img
        rgb_imgs[i] = rgb
        semseg_imgs[i] = semseg

    return depth_imgs, rgb_imgs, semseg_imgs, extrinsics


def sample_grasp_point_contact_furthest(pc, n_pts = 100):
    from dgl.geometry import farthest_point_sampler # type: ignore
    import torch  # type: ignore
    
    points = np.asarray(pc.points)
    normals = np.asarray(pc.normals)
    idxs = farthest_point_sampler(torch.from_numpy(points).unsqueeze(0), n_pts).squeeze()
    idxs = idxs.detach().cpu().numpy()
    points = points[idxs,:]
    normals = normals[idxs,:]
    return points, normals


def sample_grasp_point_contact(point_cloud,  horizontal_percentile = 0.5):
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)

    if np.random.random() < horizontal_percentile:
        points = points[np.abs(normals[:, 2]) < 0.5, :]
        normals = normals[np.abs(normals[:, 2]) < 0.5, :]
    else:
        points = points[np.abs(normals[:, 2]) > 0.5, :]
        normals = normals[np.abs(normals[:, 2]) > 0.5, :]

    if len(points) == 0:
        points = np.asarray(point_cloud.points)
        normals = np.asarray(point_cloud.normals)
        
    ok = False
    while not ok:
        # TODO this could result in an infinite loop, though very unlikely
        idx = np.random.randint(len(points))
        point, normal = points[idx], normals[idx]
        ok = True # normal[2] > -0.1  # make sure the normal is poitning upwards
    
    return point, normal


def grasp_from_contact(ori, pos, width:float, normal = None, z = None, finger_depth = 0.05):
    return Grasp(Transform(ori, pos + normal * width/2  +finger_depth*z), width)

def sample_grasp_point(point_cloud, finger_depth, eps=0.1):
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    ok = False
    while not ok:
        # TODO this could result in an infinite loop, though very unlikely
        idx = np.random.randint(len(points))
        point, normal = points[idx], normals[idx]
        ok = normal[2] > -0.1  # make sure the normal is poitning upwards
    grasp_depth = np.random.uniform(-eps * finger_depth, (1.0 + eps) * finger_depth)
    point = point + normal * grasp_depth
    return point, normal

def evaluate_grasp_point_contact(sim, pos, normal, num_rotations=12, debug = False) -> GraspCandidate:
    y_axis = normal
    x_axis = np.r_[-y_axis[1], y_axis[0], 0]
    
    if np.abs(normal[-1]) > 0.98: # top down grasp.
        x_axis = np.r_[1,0,0]

    z_axis = np.cross(x_axis, y_axis)
    R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)
    
    # try to grasp with different pitch angles
    pitches = np.linspace(-np.pi/2, np.pi/2, num_rotations)
    outcomes, widths = [], []

    grippers = []
    infos = []


    log_grasp_candidate = GraspCandidate(
        contact=pos,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        pitch_angles=[],
        widths=[],
        sucess=[],
        target = []
    )

    for pitch in pitches:
        ori = R * Rotation.from_euler("y", pitch)
        sim.restore_state()

        if debug:
            p = sim.world.get_client()
            p.removeAllUserDebugItems()
            
            p.addUserDebugLine(pos - y_axis*0.4, pos + y_axis*0.4, (0,1,0))
            p.addUserDebugLine(pos + 0.1*z_axis, pos - 0.1 * z_axis, (0,0, 1))
            p.addUserDebugLine(pos + 0.1*x_axis, pos - 0.1 * x_axis, (1,0, 0))

        tf = Transform(ori, pos - ori.apply(np.asarray([0,sim.gripper.max_opening_width /2 - NORMAL_OFFSET,0.040])))
        candidate = Grasp(tf, width=sim.gripper.max_opening_width)

        infos.append([tf])

        outcome, width, target = sim.execute_grasp(candidate, remove=False, with_target = True, table_col=not args.disable_table_collision)

        outcomes.append(outcome)
        widths.append(width)
        infos[-1].append(width)

        log_grasp_candidate.pitch_angles.append(pitch)
        log_grasp_candidate.widths.append(width)
        log_grasp_candidate.target.append(target)
        log_grasp_candidate.sucess.append((np.asarray(outcome)).astype(float))

    successes = (np.asarray(outcomes)).astype(float)
    

    return infos, successes, log_grasp_candidate

def evaluate_grasp_point(sim, pos, normal, num_rotations=6):
    # define initial grasp frame on object surface
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)

    # try to grasp with different yaw angles
    yaws = np.linspace(0.0, np.pi, num_rotations)
    outcomes, widths = [], []
    for yaw in yaws:
        ori = R * Rotation.from_euler("z", yaw)
        sim.restore_state()
        candidate = Grasp(Transform(ori, pos), width=sim.gripper.max_opening_width)
        outcome, width, target = sim.execute_grasp(candidate, remove=False, table_col=not args.disable_table_collision)
        outcomes.append(outcome)
        widths.append(width)

    # detect mid-point of widest peak of successful yaw angles
    # TODO currently this does not properly handle periodicity
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(float)
    if np.sum(successes):
        peaks, properties = signal.find_peaks(
            x=np.r_[0, successes, 0], height=1, width=1
        )
        idx_of_widest_peak = peaks[np.argmax(properties["widths"])] - 1
        ori = R * Rotation.from_euler("z", yaws[idx_of_widest_peak])
        width = widths[idx_of_widest_peak]

    return Grasp(Transform(ori, pos), width), int(np.max(outcomes)), target



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    parser.add_argument("--object-set", type=str, default="blocks")
    parser.add_argument("--num-grasps", type=int, default=10000)
    parser.add_argument("--grasps-per-scene", type=int, default=120)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--save-scene", action="store_true")
    parser.add_argument("--random", action="store_true", help="Add distrubation to camera pose")
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--seed", default=11, type=int)
    parser.add_argument("--num-views", default=1, type=int)
    parser.add_argument("--object-count-lambda", default=5, type=int, help="lambda for number of objects in poisson distribution")
    parser.add_argument("--debug", action="store_true")
    
    parser.add_argument("--contact-based", action="store_true")
    parser.add_argument("--normal-offset", default=0.006, type=float)
    parser.add_argument("--num-rotations", default=12, type=int)
    parser.add_argument("--sample-furthest", action="store_true")
    parser.add_argument("--horizontal-percentile", default=0.85, type=float)
    parser.add_argument("--disable-table-collision", default=False, action="store_true")

    parser.add_argument("--gripper-urdf", default="hand.urdf", type=str)
    parser.add_argument("--urdf-root", type=str, default="data/urdfs")

    args = parser.parse_args()
    args.save_scene = True
    
    t1 = time.perf_counter()
    if args.num_proc > 1:
        pool = mp.Pool(processes=args.num_proc)
        for i in range(args.num_proc):
            pool.apply_async(func=main, args=(args, i))
        pool.close()
        pool.join()
    else:
        main(args, 0)
