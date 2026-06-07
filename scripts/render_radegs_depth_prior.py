#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def save_depth_vis(depth: np.ndarray, out_path: Path) -> None:
    valid = np.isfinite(depth) & (depth > 0)
    vis = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        vals = depth[valid]
        lo = float(np.percentile(vals, 2))
        hi = float(np.percentile(vals, 98))
        if hi <= lo:
            hi = lo + 1e-6
        scaled = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        vis = (scaled * 255.0).astype(np.uint8)
        vis[~valid] = 0
    imageio.imwrite(out_path, vis)


def main() -> None:
    parser = ArgumentParser(description="Render a single-frame RaDe-GS depth prior and pointmap")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--frame", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--far-depth-threshold", default=5.0, type=float)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    kernel_size = dataset.kernel_size
    depth_name = "expected_depth" if dataset.depth_ratio < 0.5 else "median_depth"
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    candidates = list(scene.getTrainCameras()) + list(scene.getTestCameras())
    viewpoint = None
    for cam in candidates:
        if getattr(cam, "image_name", None) == args.frame:
            viewpoint = cam
            break
    if viewpoint is None:
        raise KeyError(f"Frame {args.frame} not found in cameras")

    with torch.no_grad():
        render_pkg = render(viewpoint, gaussians, pipe, background, kernel_size)

    depth = render_pkg[depth_name].clone()
    if viewpoint.gt_mask is not None:
        depth[viewpoint.gt_mask < 0.5] = 0
    depth = depth[0].cpu().numpy().astype(np.float32)
    if args.far_depth_threshold and args.far_depth_threshold > 0:
        depth[depth > args.far_depth_threshold] = 0
    depth[~np.isfinite(depth)] = 0

    h, w = depth.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth
    x = (uu - float(viewpoint.Cx)) * z / float(viewpoint.Fx)
    y = (vv - float(viewpoint.Cy)) * z / float(viewpoint.Fy)
    pointmap = np.stack([-x, -y, z], axis=-1).astype(np.float32)
    pointmap[z <= 0] = np.nan

    w2c = viewpoint.world_view_transform.T.detach().cpu().numpy().astype(np.float32)
    c2w = np.linalg.inv(w2c).astype(np.float32)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{args.frame}_depth.npy", depth)
    np.save(out_dir / f"{args.frame}_pointmap.npy", pointmap)
    save_depth_vis(depth, out_dir / f"{args.frame}_depth_vis.png")
    meta = {
        "frame": args.frame,
        "depth_name": depth_name,
        "shape": [int(h), int(w)],
        "fx": float(viewpoint.Fx),
        "fy": float(viewpoint.Fy),
        "cx": float(viewpoint.Cx),
        "cy": float(viewpoint.Cy),
        "image_width": int(w),
        "image_height": int(h),
        "far_depth_threshold": float(args.far_depth_threshold),
        "valid_pixels": int(np.isfinite(pointmap[..., 2]).sum()),
        "camera_center": viewpoint.camera_center.detach().cpu().numpy().astype(float).tolist(),
        "w2c": w2c.tolist(),
        "c2w": c2w.tolist(),
    }
    (out_dir / f"{args.frame}_camera.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
