import copy
import os

import cv2
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from gaussian_renderer import plane_depth_render
from utils.graphics_utils import fov2focal


def _cfg_get(config, key, default):
    return config.get(key, default) if config is not None else default


def camera_intrinsic_from_view(view):
    width = int(view.image_width)
    height = int(view.image_height)
    fx = fov2focal(view.FoVx, width)
    fy = fov2focal(view.FoVy, height)
    cx = width * 0.5
    cy = height * 0.5
    return width, height, fx, fy, cx, cy


def camera_extrinsic_from_view(view):
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = np.asarray(view.R).T
    extrinsic[:3, 3] = np.asarray(view.T)
    return extrinsic


def post_process_mesh(mesh, cluster_to_keep=1, min_triangles=50):
    if len(mesh.triangles) == 0:
        return mesh

    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        triangle_clusters, cluster_n_triangles, _ = mesh_0.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    if cluster_n_triangles.size == 0:
        return mesh_0

    keep = max(1, min(int(cluster_to_keep), cluster_n_triangles.size))
    threshold = np.sort(cluster_n_triangles.copy())[-keep]
    threshold = max(int(threshold), int(min_triangles))
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < threshold

    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    mesh_0.remove_duplicated_triangles()
    mesh_0.remove_duplicated_vertices()
    mesh_0.compute_vertex_normals()
    mesh_0.compute_triangle_normals()
    return mesh_0


def _depth_to_colormap(depth):
    valid = depth > 0
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    depth_i = np.zeros_like(depth, dtype=np.float32)
    d_min = depth[valid].min()
    d_max = depth[valid].max()
    depth_i[valid] = (depth[valid] - d_min) / (d_max - d_min + 1e-20)
    depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)


def extract_mesh_from_gaussians(
    config,
    gaussians,
    views,
    pipe,
    background,
    iteration,
):
    mesh_cfg = config.get("mesh_extraction", {})
    output_dir = os.path.join(
        config.model.model_path,
        _cfg_get(mesh_cfg, "output_dir", "mesh_extraction"),
        f"iteration_{iteration}",
    )
    os.makedirs(output_dir, exist_ok=True)

    voxel_size = float(_cfg_get(mesh_cfg, "voxel_size", 0.002))
    sdf_trunc = float(_cfg_get(mesh_cfg, "sdf_trunc", 4.0 * voxel_size))
    max_depth = float(_cfg_get(mesh_cfg, "max_depth", 5.0))
    depth_scale = float(_cfg_get(mesh_cfg, "depth_scale", 1000.0))
    num_cluster = int(_cfg_get(mesh_cfg, "num_cluster", 1))
    min_cluster_triangles = int(_cfg_get(mesh_cfg, "min_cluster_triangles", 50))
    mask_threshold = float(_cfg_get(mesh_cfg, "mask_threshold", 0.5))
    save_depth_maps = bool(_cfg_get(mesh_cfg, "save_depth_maps", True))
    max_views = int(_cfg_get(mesh_cfg, "max_views", 0))

    views = sorted(list(views), key=lambda cam: cam.uid)
    if max_views > 0:
        views = views[:max_views]

    if len(views) == 0:
        raise ValueError("mesh_extraction needs at least one camera view.")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    depth_dir = os.path.join(output_dir, "plane_depth")
    render_dir = os.path.join(output_dir, "renders")
    if save_depth_maps:
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(render_dir, exist_ok=True)

    with torch.no_grad():
        for idx, view in enumerate(tqdm(views, desc="Mesh extraction TSDF")):
            out = plane_depth_render(
                view,
                gaussians,
                pipe,
                background,
                return_depth_normal=False,
            )
            rendering = out["render"].clamp(0.0, 1.0)
            depth = out["plane_depth"].squeeze()

            if getattr(view, "gt_mask", None) is not None:
                depth = depth.clone()
                depth[view.gt_mask.squeeze().to(depth.device) < mask_threshold] = 0

            depth = depth.detach().float().cpu().numpy()
            depth[~np.isfinite(depth)] = 0
            depth[depth <= 0] = 0
            depth[depth > max_depth] = 0
            if np.count_nonzero(depth) == 0:
                continue

            color = (
                rendering.permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
                .clip(0.0, 1.0)
            )
            color = np.ascontiguousarray((color * 255).astype(np.uint8))
            depth_mm = np.ascontiguousarray((depth * depth_scale).astype(np.uint16))

            color_o3d = o3d.geometry.Image(color)
            depth_o3d = o3d.geometry.Image(depth_mm)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=depth_scale,
                depth_trunc=max_depth,
                convert_rgb_to_intensity=False,
            )

            width, height, fx, fy, cx, cy = camera_intrinsic_from_view(view)
            intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
            volume.integrate(rgbd, intrinsic, camera_extrinsic_from_view(view))

            if save_depth_maps:
                image_name = os.path.splitext(os.path.basename(str(view.image_name)))[0]
                if not image_name:
                    image_name = f"{idx:04d}"
                cv2.imwrite(os.path.join(depth_dir, f"{image_name}.png"), _depth_to_colormap(depth))
                cv2.imwrite(
                    os.path.join(render_dir, f"{image_name}.png"),
                    color[:, :, ::-1],
                )

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    raw_path = os.path.join(output_dir, "tsdf_fusion.ply")
    post_path = os.path.join(output_dir, "tsdf_fusion_post.ply")
    o3d.io.write_triangle_mesh(
        raw_path,
        mesh,
        write_triangle_uvs=True,
        write_vertex_colors=True,
        write_vertex_normals=True,
    )

    mesh = post_process_mesh(
        mesh,
        cluster_to_keep=num_cluster,
        min_triangles=min_cluster_triangles,
    )
    o3d.io.write_triangle_mesh(
        post_path,
        mesh,
        write_triangle_uvs=True,
        write_vertex_colors=True,
        write_vertex_normals=True,
    )

    vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32)
    faces = torch.tensor(np.asarray(mesh.triangles), dtype=torch.long)
    normals = torch.tensor(np.asarray(mesh.triangle_normals), dtype=torch.float32)
    if vertices.numel() == 0 or faces.numel() == 0:
        raise RuntimeError("TSDF mesh extraction produced an empty mesh.")
    print(
        "[Mesh extraction] post mesh shapes: "
        f"vertices={tuple(vertices.shape)}, "
        f"faces={tuple(faces.shape)}, "
        f"face_normals={tuple(normals.shape)}"
    )
    return vertices, faces, normals, {"raw_path": raw_path, "post_path": post_path}


def repeat_static_mesh(vertices, faces, normals, train_count, test_count, device="cuda"):
    vertices = vertices.to(device)
    faces = faces.to(device)
    normals = normals.to(device)
    return (
        faces,
        {
            "train_verts": vertices.unsqueeze(0).repeat(train_count, 1, 1),
            "train_normals": normals.unsqueeze(0).repeat(train_count, 1, 1),
            "test_verts": vertices.unsqueeze(0).repeat(test_count, 1, 1),
            "test_normals": normals.unsqueeze(0).repeat(test_count, 1, 1),
        },
    )
