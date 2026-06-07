#!/usr/bin/env python3
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch

sys.path.insert(0, str(Path.cwd()))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from mesh_extract import post_process_mesh
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def save_preview(mesh_path: Path, preview_path: Path) -> None:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"Mesh is empty: {mesh_path}")
    mesh.compute_vertex_normals()
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = float(np.linalg.norm(bbox.get_extent()))

    renderer = o3d.visualization.rendering.OffscreenRenderer(1280, 960)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.add_geometry("mesh", mesh, material)
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.scene.set_sun_light([0.5, -0.8, -1.0], [1.0, 1.0, 1.0], 75000)
    eye = center + np.array([0.9, -1.4, 0.6]) * max(extent, 1e-3)
    renderer.setup_camera(60.0, center, eye, [0.0, 0.0, 1.0])
    image = renderer.render_to_image()
    o3d.io.write_image(str(preview_path), image)


def extract_stream_mesh(
    dataset,
    pipe,
    iteration,
    name,
    num_cluster,
    voxel_size,
    block_count,
    depth_max,
    depth_scale,
    include_test,
    max_views,
    far_depth_threshold,
):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    kernel_size = dataset.kernel_size
    depth_name = "expected_depth" if dataset.depth_ratio < 0.5 else "median_depth"

    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    viewpoint_cam_list = list(scene.getTrainCameras())
    if include_test:
        viewpoint_cam_list.extend(list(scene.getTestCameras()))
    if max_views and 0 < max_views < len(viewpoint_cam_list):
        sample_idx = np.linspace(0, len(viewpoint_cam_list) - 1, num=max_views, dtype=int)
        viewpoint_cam_list = [viewpoint_cam_list[i] for i in sample_idx]

    print(f"using {len(viewpoint_cam_list)} cameras (include_test={include_test})")

    o3d_device = o3d.core.Device("CPU:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel_size,
        block_resolution=16,
        block_count=block_count,
        device=o3d_device,
    )

    used = 0
    skipped_empty = 0
    skipped_o3d = 0

    for idx, viewpoint_cam in enumerate(viewpoint_cam_list, start=1):
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size)
        color = torch.clamp(render_pkg["render"], min=0, max=1.0).cpu().numpy().transpose(1, 2, 0)
        depth = render_pkg[depth_name].clone()
        if viewpoint_cam.gt_mask is not None:
            depth[viewpoint_cam.gt_mask < 0.5] = 0
        depth = depth[0].cpu().numpy()
        if far_depth_threshold and far_depth_threshold > 0:
            depth[depth > far_depth_threshold] = 0
        torch.cuda.empty_cache()

        valid = np.isfinite(depth) & (depth > 0)
        if not np.any(valid):
            skipped_empty += 1
            continue

        depth_img = o3d.t.geometry.Image(np.ascontiguousarray(depth.astype(np.float32))).to(o3d_device)
        color_img = o3d.t.geometry.Image(np.ascontiguousarray(color.astype(np.float32))).to(o3d_device)
        intrinsic = o3d.core.Tensor(
            np.array(
                [
                    [viewpoint_cam.Fx, 0, viewpoint_cam.Cx],
                    [0, viewpoint_cam.Fy, viewpoint_cam.Cy],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
        )
        extrinsic = o3d.core.Tensor((viewpoint_cam.world_view_transform.T).cpu().numpy().astype(np.float64))
        try:
            frustum_block_coords = vbg.compute_unique_block_coordinates(
                depth_img, intrinsic, extrinsic, depth_scale, depth_max
            )
            vbg.integrate(
                frustum_block_coords,
                depth_img,
                color_img,
                intrinsic,
                extrinsic,
                depth_scale,
                depth_max,
            )
            used += 1
        except RuntimeError as exc:
            if "No block is touched in TSDF volume" in str(exc):
                skipped_o3d += 1
                continue
            raise

        if idx % 20 == 0 or idx == len(viewpoint_cam_list):
            print(
                f"processed {idx}/{len(viewpoint_cam_list)} used={used} skipped_empty={skipped_empty} skipped_o3d={skipped_o3d}",
                flush=True,
            )

    print(f"extracting mesh from used={used} skipped_empty={skipped_empty} skipped_o3d={skipped_o3d}")
    mesh = vbg.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    out_dir = Path(dataset.model_path)
    raw_path = out_dir / f"{name}.ply"
    post_path = out_dir / f"{name}_post.ply"
    preview_path = out_dir / f"{name}_post_preview.png"

    raw_legacy = mesh.to_legacy()
    o3d.io.write_triangle_mesh(str(raw_path), raw_legacy)
    post_mesh = post_process_mesh(raw_legacy, num_cluster)
    o3d.io.write_triangle_mesh(str(post_path), post_mesh)

    try:
        save_preview(post_path, preview_path)
    except Exception as exc:
        print(f"preview generation failed: {exc}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Streamed TSDF extraction for RaDe-GS outputs")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--voxel_size", default=0.01, type=float)
    parser.add_argument("--block_count", default=6000, type=int)
    parser.add_argument("--depth_max", default=8.0, type=float)
    parser.add_argument("--depth_scale", default=1.0, type=float)
    parser.add_argument("--include_test", action="store_true")
    parser.add_argument("--max_views", default=0, type=int)
    parser.add_argument("--far_depth_threshold", default=0.0, type=float)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    with torch.no_grad():
        extract_stream_mesh(
            model.extract(args),
            pipeline.extract(args),
            args.iteration,
            args.name,
            args.num_cluster,
            args.voxel_size,
            args.block_count,
            args.depth_max,
            args.depth_scale,
            args.include_test,
            args.max_views,
            args.far_depth_threshold,
        )
