#!/usr/bin/env python3
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import open3d as o3d
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


def extract_scalable_mesh(
    dataset,
    pipe,
    iteration,
    name,
    num_cluster,
    voxel_size,
    sdf_trunc,
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

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    used = 0
    skipped_empty = 0

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

        depth = np.ascontiguousarray(depth.astype(np.float32))
        color = np.ascontiguousarray(np.clip(color * 255.0, 0, 255).astype(np.uint8))
        depth_img = o3d.geometry.Image(depth)
        color_img = o3d.geometry.Image(color)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=color_img,
            depth=depth_img,
            depth_scale=depth_scale,
            depth_trunc=depth_max,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=depth.shape[1],
            height=depth.shape[0],
            fx=float(viewpoint_cam.Fx),
            fy=float(viewpoint_cam.Fy),
            cx=float(viewpoint_cam.Cx),
            cy=float(viewpoint_cam.Cy),
        )
        extrinsic = (viewpoint_cam.world_view_transform.T).cpu().numpy().astype(np.float64)
        volume.integrate(rgbd, intrinsic, extrinsic)
        used += 1

        if idx % 20 == 0 or idx == len(viewpoint_cam_list):
            print(
                f"processed {idx}/{len(viewpoint_cam_list)} used={used} skipped_empty={skipped_empty}",
                flush=True,
            )

    print(f"extracting mesh from used={used} skipped_empty={skipped_empty}")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    out_dir = Path(dataset.model_path)
    raw_path = out_dir / f"{name}.ply"
    post_path = out_dir / f"{name}_post.ply"
    preview_path = out_dir / f"{name}_post_preview.png"

    o3d.io.write_triangle_mesh(str(raw_path), mesh)
    post_mesh = post_process_mesh(mesh, num_cluster)
    o3d.io.write_triangle_mesh(str(post_path), post_mesh)

    try:
        save_preview(post_path, preview_path)
    except Exception as exc:
        print(f"preview generation failed: {exc}")


if __name__ == "__main__":
    parser = ArgumentParser(description="ScalableTSDFVolume extraction for RaDe-GS outputs")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--voxel_size", default=0.01, type=float)
    parser.add_argument("--sdf_trunc", default=0.0, type=float)
    parser.add_argument("--depth_max", default=8.0, type=float)
    parser.add_argument("--depth_scale", default=1.0, type=float)
    parser.add_argument("--include_test", action="store_true")
    parser.add_argument("--max_views", default=0, type=int)
    parser.add_argument("--far_depth_threshold", default=0.0, type=float)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    sdf_trunc = args.sdf_trunc if args.sdf_trunc > 0 else args.voxel_size * 5.0

    safe_state(args.quiet)
    with torch.no_grad():
        extract_scalable_mesh(
            model.extract(args),
            pipeline.extract(args),
            args.iteration,
            args.name,
            args.num_cluster,
            args.voxel_size,
            sdf_trunc,
            args.depth_max,
            args.depth_scale,
            args.include_test,
            args.max_views,
            args.far_depth_threshold,
        )
