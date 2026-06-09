#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


def pointmap_to_radegs_camera(points: np.ndarray) -> np.ndarray:
    converted = points.copy()
    converted[:, 0] *= -1.0
    converted[:, 1] *= -1.0
    return converted


def transform_to_world(points: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    return (homog @ c2w.T)[:, :3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a pointmap/depth prior as a point cloud.")
    parser.add_argument("--pointmap", required=True)
    parser.add_argument("--camera-json", required=True)
    parser.add_argument("--image", default="")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--world-space", action="store_true")
    parser.add_argument("--radegs-camera", action="store_true")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--compare-mesh", default="")
    args = parser.parse_args()

    pointmap = np.load(args.pointmap).astype(np.float32)
    valid = np.isfinite(pointmap[..., 2]) & (pointmap[..., 2] > 0)
    if args.stride > 1:
        sampled = np.zeros_like(valid)
        sampled[:: args.stride, :: args.stride] = True
        valid &= sampled

    points = pointmap[valid].reshape(-1, 3).astype(np.float64)
    if args.world_space:
        camera_meta = json.loads(Path(args.camera_json).read_text())
        c2w = np.asarray(camera_meta["c2w"], dtype=np.float64)
        points = pointmap_to_radegs_camera(points)
        points = transform_to_world(points, c2w)
        coordinate_system = "radegs_world"
    elif args.radegs_camera:
        points = pointmap_to_radegs_camera(points)
        coordinate_system = "radegs_camera"
    else:
        coordinate_system = "sam3d_pointmap_camera"

    colors = None
    if args.image:
        image_pil = Image.open(args.image).convert("RGB")
        if image_pil.size != (pointmap.shape[1], pointmap.shape[0]):
            image_pil = image_pil.resize((pointmap.shape[1], pointmap.shape[0]), Image.BILINEAR)
        image = np.asarray(image_pil)
        colors = image[valid].reshape(-1, 3)

    cloud = trimesh.points.PointCloud(vertices=points, colors=colors)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    ply_path = output_prefix.with_suffix(".ply")
    cloud.export(ply_path)

    summary: dict[str, object] = {
        "pointmap": str(args.pointmap),
        "camera_json": str(args.camera_json),
        "image": args.image or None,
        "coordinate_system": coordinate_system,
        "num_points": int(points.shape[0]),
        "stride": int(args.stride),
        "bounds": np.array([points.min(axis=0), points.max(axis=0)]).tolist(),
    }

    if args.compare_mesh:
        mesh = trimesh.load(args.compare_mesh, force="mesh")
        mesh_bounds = mesh.bounds.astype(np.float64)
        cloud_center = points.mean(axis=0)
        mesh_center = np.asarray(mesh.vertices).mean(axis=0)
        summary["compare_mesh"] = {
            "path": str(args.compare_mesh),
            "bounds": mesh_bounds.tolist(),
            "bbox_center_distance": float(np.linalg.norm(np.array(summary["bounds"]).mean(axis=0) - mesh_bounds.mean(axis=0))),
            "mean_center_distance": float(np.linalg.norm(cloud_center - mesh_center)),
        }

    summary_path = output_prefix.parent / f"{output_prefix.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"ply": str(ply_path), "summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
