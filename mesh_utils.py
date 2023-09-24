import argparse
import numpy as np
import torch
import plyfile
import skimage.measure
from tqdm import tqdm 
import yaml
import os.path as osp
import skimage
import time 

# 새로 추가
import mcubes
import open3d as o3d
import cv2


def convert_sigma_samples_to_ply(
    input_3d_sigma_array: np.ndarray,
    voxel_grid_origin,
    target,
    w2c,
    hwf,
    volume_size,
    ply_filename_out,
    level=5.0,
    offset=None,
    scale=None,):
    """
    Convert density samples to .ply
    :param input_3d_sdf_array: a float array of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :volume_size: a list of three floats
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()
    print(ply_filename_out)

    P_w2c = w2c
    H, W, focal = hwf
    K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    verts, faces, normals, values = skimage.measure.marching_cubes(
        input_3d_sigma_array, level=level, spacing=volume_size
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    # mesh_points = np.matmul(mesh_points, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))
    # mesh_points = np.matmul(mesh_points, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))


    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % str(ply_filename_out))
    ply_data.write(ply_filename_out)

    # remove noise in the mesh by keeping only the biggest cluster
    print('Removing noise ...')
    mesh = o3d.io.read_triangle_mesh(ply_filename_out)
    idxs, count, _ = mesh.cluster_connected_triangles()
    max_cluster_idx = np.argmax(count)
    triangles_to_remove = [i for i in range(len(faces_tuple)) if idxs[i] != max_cluster_idx]
    mesh.remove_triangles_by_index(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    print(f'Mesh has {len(mesh.vertices)/1e6:.2f} M vertices and {len(mesh.triangles)/1e6:.2f} M faces.')

    vertices_ = np.asarray(mesh.vertices).astype(np.float32)
    triangles = np.asarray(mesh.triangles)
    N_vertices = len(vertices_)
    vertices_homo = np.concatenate([vertices_, np.ones((N_vertices, 1))], 1) # (N, 4)

    non_occluded_sum = np.zeros((N_vertices, 1))
    v_color_sum = np.zeros((N_vertices, 3))

    for idx in tqdm(range(len(target))):
        image = np.array(target[idx])
        ## project vertices from world coordinate to camera coordinate
        vertices_cam = (P_w2c @ vertices_homo.T) # (3, N) in "right up back"
        vertices_cam[1:] *= -1 # (3, N) in "right down forward"
        ## project vertices from camera coordinate to pixel coordinate
        vertices_image = (K @ vertices_cam).T # (N, 3)
        depth = vertices_image[:, -1:]+1e-5 # the depth of the vertices, used as far plane
        vertices_image = vertices_image[:, :2]/depth
        vertices_image = vertices_image.astype(np.float32)
        vertices_image[:, 0] = np.clip(vertices_image[:, 0], 0, W-1)
        vertices_image[:, 1] = np.clip(vertices_image[:, 1], 0, H-1)    

        colors = []
        remap_chunk = int(3e4)
        for i in range(0, N_vertices, remap_chunk):
            colors += [cv2.remap(image[i], 
                                vertices_image[i:i+remap_chunk, 0],
                                vertices_image[i:i+remap_chunk, 1],
                                interpolation=cv2.INTER_LINEAR)[:, 0]]
        colors = np.vstack(colors) # (N_vertices, 3)
        print(colors.shape)
        time.sleep(60)

        non_occluded = np.ones_like(non_occluded_sum) * 0.1/depth
        non_occluded += opacity < 0.2

        v_color_sum += colors * non_occluded
        non_occluded_sum += non_occluded





    vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertex_all = np.empty(N_vertices, vertices_.dtype.descr)
    for prop in vertices_.dtype.names:
        vertex_all[prop] = vertices_[prop][:,0]
    face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = triangles
    _el_verts = plyfile.PlyElement.describe(vertex_all, "vertex")
    _el_faces = plyfile.PlyElement.describe(face, "face")
    _ply_data = plyfile.PlyData([_el_verts, _el_faces])
    print("cool!  saving clusted mesh to %s" % str(ply_filename_out))
    _ply_data.write(ply_filename_out)

    print(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )


def generate_and_write_mesh(i,bounding_box, target, c2w, hwf, num_pts, levels, chunk, device, ply_root, **render_kwargs):
    """
    Generate density grid for marching cubes
    :bounding_box: bounding box for meshing 
    :num_pts: Number of grid elements on each axis
    :levels: list of levels to write meshes for 
    :ply_root: string, path of the folder to save meshes to
    """
    
    P_c2w = c2w
    P_w2c = np.linalg.inv(P_c2w)[:3] # (3, 4)
    _hwf = hwf

    near = render_kwargs['near']
    bb_min = (*(bounding_box[0] + near).cpu().numpy(),)
    bb_max = (*(bounding_box[1] - near).cpu().numpy(),)

    x_vals = torch.tensor(np.linspace(bb_min[0], bb_max[0], num_pts))
    y_vals = torch.tensor(np.linspace(bb_min[1], bb_max[1], num_pts))
    z_vals = torch.tensor(np.linspace(bb_min[2], bb_max[2], num_pts))

    xs, ys, zs = torch.meshgrid(x_vals, y_vals, z_vals, indexing = 'ij')
    coords = torch.stack((xs, ys, zs), dim = -1)

    coords = coords.view(1, -1, 3).type(torch.FloatTensor).to(device)
    dummy_viewdirs = torch.tensor([0, 0, 1]).view(-1, 3).type(torch.FloatTensor).to(device)

    nerf_model = render_kwargs['network_fine']
    radiance_field = render_kwargs['network_query_fn']

    chunk_outs = []

    for k in tqdm(range(coords.shape[1] // chunk), desc = "Retrieving densities at grid points"):
        chunk_out = radiance_field(coords[:, k * chunk: (k + 1) * chunk, :], dummy_viewdirs, nerf_model)
        chunk_outs.append(chunk_out.detach().cpu().numpy()[:, :, -1])

    if not coords.shape[1] % chunk == 0:
        chunk_out = radiance_field(coords[:, (k+1) * chunk: , :], dummy_viewdirs, nerf_model)
        chunk_outs.append(chunk_out.detach().cpu().numpy()[:, :, -1])

    input_sigma_arr = np.concatenate(chunk_outs, axis = -1).reshape(num_pts, num_pts, num_pts)

    for level in levels:
        try:
            sizes = (abs(bounding_box[1] - bounding_box[0]).cpu()).tolist()
            convert_sigma_samples_to_ply(input_sigma_arr, list(bb_min), target, P_w2c, _hwf, sizes, osp.join(ply_root, f"test_mesh_{i}_{level}.ply"), level = level)
        except ValueError:
            print(f"Density field does not seem to have an isosurface at level {level} yet")
