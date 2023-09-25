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

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # sigma_loss = sigma_sparsity_loss(raw[...,3])
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1) / torch.sum(weights, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map)
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    # Calculate weights sparsity loss
    try:
        entropy = Categorical(probs = torch.cat([weights, 1.0-weights.sum(-1, keepdim=True)+1e-6], dim=-1)).entropy()
    except:
        pdb.set_trace()
    sparsity_loss = entropy

    return rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss

def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                embed_fn=None,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, depth_map_0, acc_map_0, sparsity_loss_0 = rgb_map, depth_map, acc_map, sparsity_loss

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'depth_map' : depth_map, 'acc_map' : acc_map, 'sparsity_loss': sparsity_loss, 'weights' : weights}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['depth0'] = depth_map_0
        ret['acc0'] = acc_map_0
        ret['sparsity_loss0'] = sparsity_loss_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

@torch.no_grad()
def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'depth_map', 'acc_map', 'weights']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_path(render_poses, hwf, K, chunk, render_kwargs):
    H, W, focal = hwf
    near, far = render_kwargs['near'], render_kwargs['far']

    weights = []
    for i, c2w in enumerate(tqdm(render_poses)):
        _, _, _, weight, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        weights.append(weight.cpu().numpy())
    
    return weights


def convert_sigma_samples_to_ply(
    input_3d_sigma_array: np.ndarray,
    voxel_grid_origin,
    device,
    imgs_path,
    poses,
    w2c,
    hwf,
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
        print(f"@@image shape : {image.shape}") # (640, 360, 3) -> 

        P_c2w = poses[idx]
        P_w2c = np.linalg.inv(P_c2w)[:3] # (3, 4)
        ## project vertices from world coordinate to camera coordinate
        vertices_cam = (P_w2c @ vertices_homo.T) # (3, N) in "right up back" 
        print(f"@@vertices_cam shape : {vertices_cam.shape}") # (3, 4, 250) ->
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

        rays_o = torch.FloatTensor(poses[idx][:3, -1]).expand(N_vertices, 3)
        ## ray's direction is the vector pointing from camera origin to the vertices
        rays_d = torch.FloatTensor(vertices_) - rays_o # (N_vertices, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        near = np.array([2.0, 6.0]).min() * torch.ones_like(rays_o[:, :1])
        ## the far plane is the depth of the vertices, since what we want is the accumulated
        ## opacity along the path from camera origin to the vertices
        far = torch.FloatTensor(depth) * torch.ones_like(rays_o[:, :1])
        rays = torch.cat([rays_o, rays_d, near, far], 1).cuda()
        N_rays = rays.shape[0]

        t_vals = torch.linspace(0., 1., steps=N_vertices)
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
        z_vals = z_vals.expand([N_rays, N_vertices])

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

        dummy_viewdirs = torch.tensor([0, 0, 1]).view(-1, 3).type(torch.FloatTensor).to(device)

        nerf_model = render_kwargs['network_fine']
        radiance_field = render_kwargs['network_query_fn']

        raw = radiance_field(pts, dummy_viewdirs, nerf_model)

        _, _, _, weights, _, _ = raw2outputs(raw, z_vals, rays_d)
        opacity = weights.sum(1)
        opacity = opacity.cpu().numpy()[:, np.newaxis]
        print(f"opacity's shape is {opacity.shape}")
        opacity = np.nan_to_num(opacity, 1)

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


def generate_and_write_mesh(i,bounding_box, poses, imgs_path, c2w, hwf, num_pts, levels, chunk, device, ply_root, **render_kwargs):
    """
    Generate density grid for marching cubes
    :bounding_box: bounding box for meshing 
    :num_pts: Number of grid elements on each axis
    :levels: list of levels to write meshes for 
    :ply_root: string, path of the folder to save meshes to
    """
    
    P_w2c = np.linalg.inv(c2w.cpu().numpy().astype(np.float32))[:3] # (3, 4)

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
            convert_sigma_samples_to_ply(input_sigma_arr, list(bb_min), device, imgs_path, poses, P_w2c, hwf, sizes, osp.join(ply_root, f"test_mesh_{i}_{level}.ply"), level = level, **render_kwargs)
        except ValueError:
            print(f"Density field does not seem to have an isosurface at level {level} yet")
