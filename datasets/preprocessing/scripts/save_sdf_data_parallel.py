from pysdf import SDF
import os
import glob
import time
import argparse
import numpy as np
import multiprocessing as mp
import time
from tqdm import tqdm

import trimesh

## occupancy related code
def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    The returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces, visual=g.visual)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def get_scene_from_mesh_pose_list(mesh_pose_list, scene_as_mesh=True, return_list=False, mesh_root = ""):
    # create scene from meshes
    scene = trimesh.Scene()
    mesh_list = []
    # Add table. Ugly fix imo
    pose = np.eye(4)
    pose[0,-1] = 0.15
    pose[1,-1] = 0.15
    pose[2,-1] = 0.05
    mesh_pose_list = [
          [
         "data/urdfs/setup/plane.obj",
         0.6, 
         pose
        ]
        , *mesh_pose_list
    ]
    
    for mesh_path, scale, pose in mesh_pose_list:
        if mesh_root != "":
            mesh_path = os.path.join(mesh_root, mesh_path)

        if os.path.splitext(mesh_path)[1] == '.urdf':
            obj = URDF.load(mesh_path)
            assert len(obj.links) == 1
            assert len(obj.links[0].visuals) == 1
            assert len(obj.links[0].visuals[0].geometry.meshes) == 1
            mesh = obj.links[0].visuals[0].geometry.meshes[0].copy()
        else:
            mesh = trimesh.load(mesh_path)

        mesh.apply_scale(scale)
        mesh.apply_transform(pose)
        scene.add_geometry(mesh)
        mesh_list.append(mesh)
    if scene_as_mesh:
        scene = as_mesh(scene)
    if return_list:
        return scene, mesh_list
    else:
        return scene


def sample_iou_points(mesh_list, bounds, num_point, padding=0.02, uniform=False, size=0.3):

    # Sample close to surface
    num_pts_to_sample = int(0.7 * num_point) //  len(mesh_list[1:]) 
    surface_dist = 0.007 
    shape_pts = np.concatenate([trimesh.sample.sample_surface_even(m, num_pts_to_sample)[0] for m in mesh_list[1:]]) # ignore table
    # Add noise
    shape_pts += np.random.randn(*shape_pts.shape) * surface_dist

    points = np.random.rand(num_point - shape_pts.shape[0], 3).astype(np.float32)
    if uniform:
        points *= size + 2 * padding
        points -= padding
    else:
        points = points * (bounds[[1]] + 2 * padding - bounds[[0]]) + bounds[[0]] - padding
    print(f"Sampled {points.shape} random and {shape_pts.shape} close to surface.")
    points = np.concatenate([shape_pts, points])
    np.random.shuffle(points)
    #import pdb; pdb.set_trace()
    sdf = np.zeros((len(mesh_list), points.shape[0]))
    
    # Only query points outside objects for all objects
    for idx, mesh in enumerate(mesh_list):
        f = SDF(mesh.vertices, mesh.faces)
        sdf[idx, :] = f(points)

    return points, sdf

def sample_sdf(mesh_pose_list_path, num_point, uniform, mesh_root = ""):
    mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
    scene, mesh_list = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=True, mesh_root = mesh_root)
    points, sdf = sample_iou_points(mesh_list, scene.bounds, num_point, uniform=uniform)
    return points, sdf

def save_sdf(mesh_pose_list_path, args):
    out = args.dataset if args.dataset is not None else args.raw
    name = os.path.basename(mesh_pose_list_path)[:-4]
    save_root = os.path.join(out, 'sdf', name)

    points, sdf = sample_sdf(mesh_pose_list_path, args.num_point_per_file * args.num_file, args.uniform, mesh_root = args.mesh_root)
    n_instances = sdf.shape[0]
    points = points.astype(np.float16).reshape(args.num_file, args.num_point_per_file, 3)
    sdf = sdf.reshape(args.num_file, n_instances, args.num_point_per_file)
    

    os.makedirs(save_root, exist_ok=True)
    for i in range(args.num_file):
        print(i)
        # print("sampling", np.min(points[i], axis = 0), np.max(points[i], axis = 0))
        np.savez(os.path.join(save_root, '%04d.npz' % i), points=points[i], sdf=sdf[i])

def log_result(result):
    g_completed_jobs.append(result)
    elapsed_time = time.time() - g_starting_time

    if len(g_completed_jobs) % 1000 == 0:
        msg = "%05d/%05d %s finished! " % (len(g_completed_jobs), g_num_total_jobs, result)
        msg = msg + 'Elapsed time: ' + \
                time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + '. '
        print(msg)

def main(args):
    mesh_list_files = glob.glob(os.path.join(args.raw, 'mesh_pose_list', '*.npz'))
    

    global g_completed_jobs
    global g_num_total_jobs
    global g_starting_time

    g_num_total_jobs = len(mesh_list_files)
    g_completed_jobs = []

    g_starting_time = time.time()

    if args.num_proc > 1:
        pool = mp.Pool(processes=args.num_proc) 
        print('Total jobs: %d, CPU num: %d' % (g_num_total_jobs, args.num_proc))
        for f in mesh_list_files:
            pool.apply_async(func=save_sdf, args=(f,args), callback=log_result)
        pool.close()
        pool.join()
    else:
        for f in tqdm(mesh_list_files):
            save_sdf(f, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("raw", type=str)
    parser.add_argument("--dataset", type=str, default = None)
    parser.add_argument("num_point_per_file", type=int)
    parser.add_argument("num_file", type=int)
    parser.add_argument("mesh_root", type=str, default = "")
    parser.add_argument("--uniform", action='store_true', help='sample uniformly in the bbox, else sample in the tight bbox')
    args = parser.parse_args()
    main(args)