#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


def read_points3d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    points = []
    colors = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        points.append([float(parts[1]), float(parts[2]), float(parts[3])])
        colors.append([int(parts[4]), int(parts[5]), int(parts[6])])
    return np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.uint8)


def quat_wxyz_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.asarray(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def read_images(path: Path) -> list[dict]:
    entries = []
    data_lines = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        data_lines.append(line)
    for idx in range(0, len(data_lines), 2):
        parts = data_lines[idx].split()
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        image_name = parts[9]
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = quat_wxyz_to_rotmat(qw, qx, qy, qz)
        w2c[:3, 3] = [tx, ty, tz]
        c2w = np.linalg.inv(w2c)
        entries.append({"image_id": image_id, "image_name": image_name, "c2w": c2w})
    return sorted(entries, key=lambda item: item["image_id"])


def read_camera(path: Path) -> dict:
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        return {
            "width": float(parts[2]),
            "height": float(parts[3]),
            "fx": float(parts[4]),
            "fy": float(parts[5]),
            "cx": float(parts[6]),
            "cy": float(parts[7]),
        }
    raise RuntimeError(f"No camera entry found in {path}")


def sample_line(start: np.ndarray, end: np.ndarray, spacing: float, color: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vec = end - start
    length = float(np.linalg.norm(vec))
    if length < 1e-9:
        return start[None, :].astype(np.float32), color[None, :].astype(np.uint8)
    steps = max(2, int(np.ceil(length / spacing)) + 1)
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None]
    pts = start[None, :] * (1.0 - t) + end[None, :] * t
    cols = np.repeat(color[None, :], steps, axis=0)
    return pts.astype(np.float32), cols.astype(np.uint8)


def build_camera_preview(
    images: list[dict],
    cam: dict,
    scene_points: np.ndarray,
    frustum_stride: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    centers = np.asarray([item["c2w"][:3, 3] for item in images], dtype=np.float64)
    if len(centers) < 2:
        raise RuntimeError("Need at least two cameras to build a trajectory preview.")

    step_lengths = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    positive_steps = step_lengths[step_lengths > 1e-6]
    median_step = float(np.median(positive_steps)) if len(positive_steps) else 0.1

    if len(scene_points):
        bbox_min = scene_points.min(axis=0)
        bbox_max = scene_points.max(axis=0)
        bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
    else:
        bbox_diag = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))

    arrow_len = float(np.clip(max(median_step * 4.0, bbox_diag * 0.003), 0.12, 0.8))
    frustum_len = arrow_len * 0.65
    arrow_head = arrow_len * 0.28
    spacing = max(arrow_len / 10.0, 0.01)

    width, height = cam["width"], cam["height"]
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    image_corners = np.asarray(
        [
            [0.0, 0.0, frustum_len],
            [width, 0.0, frustum_len],
            [width, height, frustum_len],
            [0.0, height, frustum_len],
        ],
        dtype=np.float64,
    )
    frustum_corners_cam = np.stack(
        [
            (image_corners[:, 0] - cx) * image_corners[:, 2] / fx,
            (image_corners[:, 1] - cy) * image_corners[:, 2] / fy,
            image_corners[:, 2],
        ],
        axis=1,
    )

    point_chunks = []
    color_chunks = []

    trajectory_color = np.asarray([255, 220, 0], dtype=np.uint8)
    center_color = np.asarray([255, 0, 0], dtype=np.uint8)
    forward_color = np.asarray([0, 255, 0], dtype=np.uint8)
    frustum_color = np.asarray([0, 180, 255], dtype=np.uint8)

    for a, b in zip(centers[:-1], centers[1:]):
        pts, cols = sample_line(a, b, spacing, trajectory_color)
        point_chunks.append(pts)
        color_chunks.append(cols)

    for idx, item in enumerate(images):
        c2w = item["c2w"]
        center = c2w[:3, 3]
        right = c2w[:3, 0]
        up = c2w[:3, 1]
        forward = c2w[:3, 2]

        point_chunks.append(center[None, :].astype(np.float32))
        color_chunks.append(center_color[None, :])

        tip = center + forward * arrow_len
        pts, cols = sample_line(center, tip, spacing, forward_color)
        point_chunks.append(pts)
        color_chunks.append(cols)

        left_head = tip - forward * arrow_head + (up - right) * (arrow_head * 0.45)
        right_head = tip - forward * arrow_head + (up + right) * (arrow_head * 0.45)
        for head_pt in (left_head, right_head):
            pts, cols = sample_line(tip, head_pt, spacing, forward_color)
            point_chunks.append(pts)
            color_chunks.append(cols)

        if idx % frustum_stride != 0 and idx != len(images) - 1:
            continue
        world_corners = (c2w[:3, :3] @ frustum_corners_cam.T).T + center
        for corner in world_corners:
            pts, cols = sample_line(center, corner, spacing, frustum_color)
            point_chunks.append(pts)
            color_chunks.append(cols)
        for corner_a, corner_b in zip(world_corners, np.roll(world_corners, -1, axis=0)):
            pts, cols = sample_line(corner_a, corner_b, spacing, frustum_color)
            point_chunks.append(pts)
            color_chunks.append(cols)

    preview_points = np.concatenate(point_chunks, axis=0)
    preview_colors = np.concatenate(color_chunks, axis=0)
    summary = {
        "num_cameras": len(images),
        "median_camera_step": median_step,
        "scene_bbox_diag": bbox_diag,
        "arrow_length": arrow_len,
        "frustum_length": frustum_len,
        "frustum_stride": frustum_stride,
    }
    return preview_points, preview_colors, summary


def write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for xyz, rgb in zip(points, colors):
            f.write(f"{xyz[0]} {xyz[1]} {xyz[2]} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--frustum-stride", type=int, default=30)
    args = parser.parse_args()

    sparse_dir = args.dataset_dir / "sparse" / "0"
    out_dir = args.out_dir or (args.dataset_dir / "preview_ply")

    points, colors = read_points3d(sparse_dir / "points3D.txt")
    images = read_images(sparse_dir / "images.txt")
    cam = read_camera(sparse_dir / "cameras.txt")

    preview_points, preview_colors, summary = build_camera_preview(
        images=images,
        cam=cam,
        scene_points=points,
        frustum_stride=max(1, args.frustum_stride),
    )

    write_ascii_ply(out_dir / "init_points.ply", points, colors)
    write_ascii_ply(out_dir / "camera_trajectory_with_direction.ply", preview_points, preview_colors)
    write_ascii_ply(
        out_dir / "init_points_with_camera_trajectory.ply",
        np.concatenate([points, preview_points], axis=0),
        np.concatenate([colors, preview_colors], axis=0),
    )

    (out_dir / "preview_summary.json").write_text(
        json.dumps(
            {
                "dataset_dir": str(args.dataset_dir),
                "num_init_points": int(len(points)),
                "num_preview_points": int(len(preview_points)),
                **summary,
            },
            indent=2,
        )
    )
    print(json.dumps({"out_dir": str(out_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
