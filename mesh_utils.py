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
import torch.nn.functional as F
from torch.distributions import Categorical
import pdb
from PIL import Image
from run_nerf_helpers import *

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    noise = 0.
    # sigma_loss = sigma_sparsity_loss(raw[...,3])
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = \
        alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]

    return weights.sum(1)

def convert_sigma_samples_to_ply(
    input_3d_sigma_array: np.ndarray,
    voxel_grid_origin,
    device,
    imgs_path,
    poses,
    hwf,
    nerf_model,
    radiance_field,
    volume_size,
    ply_filename_out,
    level=5.0,
    offset=None,
    scale=None,
    **render_kwargs):
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
    print(f"len(N_vertices) is {N_vertices}.")
    vertices_homo = np.concatenate([vertices_, np.ones((N_vertices, 1))], 1) # (N, 4)
    # print(f"target shape is{target.shape}")

    non_occluded_sum = np.zeros((N_vertices, 1))
    v_color_sum = np.zeros((N_vertices, 3))
    # print(f"type is {type(target)}")

    for idx in tqdm(range(len(imgs_path))):
        image = Image.open(imgs_path[idx]).convert('RGB')
        image = image.resize((W, H), Image.LANCZOS)
        image = np.array(image) 
        # print(f"@@image shape : {image.shape}") # (640, 360, 3) -> 

        P_c2w = poses[idx]
        P_w2c = np.linalg.inv(P_c2w)[:3] # (3, 4)
        ## project vertices from world coordinate to camera coordinate
        vertices_cam = (P_w2c @ vertices_homo.T) # (3, N) in "right up back" 
        # print(f"@@vertices_cam shape : {vertices_cam.shape}") # (3, 4, 250) -> (3, 9482)
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
            colors += [cv2.remap(image, 
                                vertices_image[i:i+remap_chunk, 0],
                                vertices_image[i:i+remap_chunk, 1],
                                interpolation=cv2.INTER_LINEAR)[:, 0]]
        colors = np.vstack(colors) # (N_vertices, 3)
        print(f"colors shape : {colors.shape}") # (9482, 3)

        rays_o = torch.FloatTensor(poses[idx][:3, -1]).expand(N_vertices, 3)
        ## ray's direction is the vector pointing from camera origin to the vertices
        rays_d = torch.FloatTensor(vertices_) - rays_o # (N_vertices, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        near = np.array([2.0, 6.0]).min() * torch.ones_like(rays_o[:, :1])
        # _near = near.cuda()
        ## the far plane is the depth of the vertices, since what we want is the accumulated
        ## opacity along the path from camera origin to the vertices
        far = torch.FloatTensor(depth) * torch.ones_like(rays_o[:, :1])
        rays = torch.cat([rays_o, rays_d, near, far], 1).cuda()
        # print(f"!!! rays.shape : {rays.shape}") # !!! rays.shape : torch.Size([9482, 8])


        t_vals = torch.linspace(0., 1., steps=64).cuda()
        z_vals = 1./(1./near.cuda() * (1.-t_vals) + 1./far.cuda() * (t_vals))
        z_vals = z_vals.expand([N_vertices, 64])

        pts = rays_o.cuda()[...,None,:] + rays_d.cuda()[...,None,:] * z_vals.cuda()[...,:,None]
        
        sh = rays_d.shape # [..., 3] ->확인됨
        # print(f"### sh.shape : {sh}")

        with torch.no_grad():
            raw = radiance_field(pts, rays_d.cuda(), nerf_model)
            #print(f"@@@ raw.shape : {raw.shape}") # torch.Size([9482, 64, 4])
            #print(f"@@@ raw : {raw}")
            weights = raw2outputs(raw, z_vals.cuda(), rays_d.cuda())
            # print(f"@@@ weights : {weights.shape}") # torch.Size([9482, 64])라서 raw2outputs의 return에 .sum(1)을 하였음
        opacity = weights.cpu().numpy()[:, np.newaxis] # (N_vertices, 1)
        print(f"opacity shape : {opacity.shape}")
        opacity = np.nan_to_num(opacity, 1)
            


        non_occluded = np.ones_like(non_occluded_sum) * 0.1/depth
        non_occluded += opacity < 0.2

        v_color_sum += colors * non_occluded
        non_occluded_sum += non_occluded

    v_colors = v_color_sum/non_occluded_sum
    v_colors = v_colors.astype(np.uint8)
    v_colors.dtype = [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertex_all = np.empty(N_vertices, vertices_.dtype.descr+v_colors.dtype.descr)
    for prop in vertices_.dtype.names:
        vertex_all[prop] = vertices_[prop][:,0]
    for prop in v_colors.dtype.names:
        vertex_all[prop] = v_colors[prop][:, 0]
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


def generate_and_write_mesh(i,
                            bounding_box, 
                            poses, 
                            imgs_path, 
                            hwf, 
                            num_pts, 
                            levels, 
                            chunk, 
                            device, 
                            ply_root, 
                            **render_kwargs):
    """
    Generate density grid for marching cubes
    :bounding_box: bounding box for meshing 
    :num_pts: Number of grid elements on each axis
    :levels: list of levels to write meshes for 
    :ply_root: string, path of the folder to save meshes to
    """

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
    '''
    print(f"##coords : {coords}")
    print(f"##coords.shape : {coords.shape}")
    print(f"##chunk : {chunk}")
    ##coords : tensor([[[-2.7596, -1.9699, -2.4187],
         [-2.7596, -1.9699, -2.4022],
         [-2.7596, -1.9699, -2.3856],
         ...,
         [ 2.3489,  2.6610,  1.7647],
         [ 2.3489,  2.6610,  1.7813],
         [ 2.3489,  2.6610,  1.7978]]])
    ##coords.shape : torch.Size([1, 16777216, 3])
    ##chunk : 32768
    '''
    chunk_outs = []

    for k in tqdm(range(coords.shape[1] // chunk), desc = "Retrieving densities at grid points"):
        chunk_out = radiance_field(coords[:, k * chunk: (k + 1) * chunk, :], dummy_viewdirs, nerf_model)
        chunk_outs.append(chunk_out.detach().cpu().numpy()[:, :, -1])

    if not coords.shape[1] % chunk == 0:
        chunk_out = radiance_field(coords[:, (k+1) * chunk: , :], dummy_viewdirs, nerf_model)
        chunk_outs.append(chunk_out.detach().cpu().numpy()[:, :, -1])

    input_sigma_arr = np.concatenate(chunk_outs, axis = -1).reshape(num_pts, num_pts, num_pts)
    '''
    print(f"##chunk_out : {chunk_out}")
    print(f"##chunk_out.shape : {chunk_out.shape}")
    ##chunk_out : tensor([[[1.7850, 1.1321, 0.7393, 0.2605],████████████████████| 512/512 [00:12<00:00, 42.39it/s]
         [1.7396, 1.1042, 0.7300, 0.3160],
         [1.6741, 1.0633, 0.7139, 0.3710],
         ...,
         [1.1437, 0.8011, 0.5585, 0.0391],
         [1.0863, 0.7637, 0.5336, 0.0523],
         [0.9940, 0.6975, 0.4910, 0.1351]]])
    ##chunk_out.shape : torch.Size([1, 32768, 4])
    '''

    for level in levels:
        try:
            sizes = (abs(bounding_box[1] - bounding_box[0]).cpu()).tolist()
            convert_sigma_samples_to_ply(input_sigma_arr, 
                                         list(bb_min), 
                                         device, 
                                         imgs_path, 
                                         poses, 
                                         hwf,
                                         nerf_model,
                                         radiance_field,
                                         sizes, 
                                         osp.join(ply_root, f"test_mesh_{i}_{level}.ply"), 
                                         level = level, 
                                         **render_kwargs)
        except ValueError:
            print(f"Density field does not seem to have an isosurface at level {level} yet")
