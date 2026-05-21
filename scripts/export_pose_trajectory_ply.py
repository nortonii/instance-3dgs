#!/usr/bin/env python3

import argparse
import json
import re
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def read_pose_file(path: Path) -> list[dict]:
    poses = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        poses.append(
            {
                "timestamp": float(parts[0]),
                "center": np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64),
                "rotation": Rotation.from_quat([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]).as_matrix(),
                "id": int(parts[8]) if len(parts) > 8 else None,
            }
        )
    return poses


def read_calib(path: Path) -> dict:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if fs.isOpened():
        try:
            camera_name = fs.getNode("camera_name").string()
            width = int(fs.getNode("image_width").real())
            height = int(fs.getNode("image_height").real())
            camera_matrix = fs.getNode("camera_matrix").mat()
            if camera_matrix is not None:
                fs.release()
                return {
                    "camera_name": camera_name,
                    "width": width,
                    "height": height,
                    "fx": float(camera_matrix[0, 0]),
                    "fy": float(camera_matrix[1, 1]),
                    "cx": float(camera_matrix[0, 2]),
                    "cy": float(camera_matrix[1, 2]),
                }
        except cv2.error:
            pass
        finally:
            fs.release()

    text = path.read_text(encoding="utf-8", errors="ignore")
    camera_name_match = re.search(r"camera_name:\s*(\S+)", text)
    width_match = re.search(r"image_width:\s*(\d+)", text)
    height_match = re.search(r"image_height:\s*(\d+)", text)
    data_match = re.search(r"camera_matrix:\s*.*?data:\s*\[\s*([^\]]+)\]", text, re.S)
    if not (camera_name_match and width_match and height_match and data_match):
        raise RuntimeError(f"Failed to parse calibration file: {path}")
    vals = [float(item.strip()) for item in data_match.group(1).replace("\n", " ").split(",") if item.strip()]
    if len(vals) != 9:
        raise RuntimeError(f"Expected 9 camera matrix values in {path}, got {vals}")
    return {
        "camera_name": camera_name_match.group(1),
        "width": int(width_match.group(1)),
        "height": int(height_match.group(1)),
        "fx": vals[0],
        "fy": vals[4],
        "cx": vals[2],
        "cy": vals[5],
    }


def nearest_index(times: list[float], stamp: float) -> int:
    idx = bisect_left(times, stamp)
    if idx <= 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    return idx - 1 if abs(times[idx - 1] - stamp) <= abs(times[idx] - stamp) else idx


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = (dst_centered.T @ src_centered) / src.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    s = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1
    rotation = u @ s @ vt
    variance = np.mean(np.sum(src_centered**2, axis=1))
    scale = float(np.trace(np.diag(singular_values) @ s) / variance)
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def align_poses(
    poses: list[dict],
    reference_poses: list[dict],
    max_time_diff: float,
) -> tuple[list[dict], dict]:
    ref_times = [item["timestamp"] for item in reference_poses]
    src_points = []
    dst_points = []
    matches = []
    for pose in poses:
        idx = nearest_index(ref_times, pose["timestamp"])
        dt = abs(reference_poses[idx]["timestamp"] - pose["timestamp"])
        if dt > max_time_diff:
            continue
        src_points.append(pose["center"])
        dst_points.append(reference_poses[idx]["center"])
        matches.append((pose, reference_poses[idx]))
    if len(src_points) < 3:
        raise RuntimeError("Need at least 3 timestamp matches to compute similarity alignment.")
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    scale, rotation, translation = umeyama_similarity(src, dst)

    aligned = []
    for pose in poses:
        aligned.append(
            {
                **pose,
                "center": scale * (rotation @ pose["center"]) + translation,
                "rotation": rotation @ pose["rotation"],
            }
        )

    aligned_src = np.asarray([item["center"] for item, _ in matches], dtype=np.float64)
    aligned_src = (scale * (rotation @ aligned_src.T)).T + translation
    aligned_dst = np.asarray([item["center"] for _, item in matches], dtype=np.float64)
    rmse = float(np.sqrt(np.mean(np.sum((aligned_src - aligned_dst) ** 2, axis=1))))
    summary = {
        "matched_count": len(matches),
        "max_time_diff": max_time_diff,
        "similarity_scale": scale,
        "similarity_rotation_matrix": rotation.tolist(),
        "similarity_translation": translation.tolist(),
        "aligned_rmse_translation_m": rmse,
    }
    return aligned, summary


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


def build_preview(
    poses: list[dict],
    calib: dict,
    frustum_stride: int,
    trajectory_color: np.ndarray,
    center_color: np.ndarray,
    forward_color: np.ndarray,
    frustum_color: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    centers = np.asarray([item["center"] for item in poses], dtype=np.float64)
    if len(centers) < 2:
        raise RuntimeError("Need at least two poses to build a trajectory preview.")

    step_lengths = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    positive_steps = step_lengths[step_lengths > 1e-6]
    median_step = float(np.median(positive_steps)) if len(positive_steps) else 0.1
    bbox_diag = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))

    arrow_len = float(np.clip(max(median_step * 4.0, bbox_diag * 0.01), 0.08, 1.2))
    frustum_len = arrow_len * 0.65
    arrow_head = arrow_len * 0.28
    spacing = max(arrow_len / 10.0, 0.01)

    width, height = calib["width"], calib["height"]
    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
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

    for a, b in zip(centers[:-1], centers[1:]):
        pts, cols = sample_line(a, b, spacing, trajectory_color)
        point_chunks.append(pts)
        color_chunks.append(cols)

    for idx, pose in enumerate(poses):
        center = pose["center"]
        rotation = pose["rotation"]
        right = rotation[:, 0]
        up = rotation[:, 1]
        forward = rotation[:, 2]

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

        if idx % frustum_stride != 0 and idx != len(poses) - 1:
            continue
        world_corners = (rotation @ frustum_corners_cam.T).T + center
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
        "num_poses": len(poses),
        "median_camera_step": median_step,
        "bbox_diag": bbox_diag,
        "arrow_length": arrow_len,
        "frustum_length": frustum_len,
        "frustum_stride": frustum_stride,
    }
    return preview_points, preview_colors, summary


def write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
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


def parse_color(text: str) -> np.ndarray:
    parts = [int(item.strip()) for item in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected color as r,g,b, got: {text}")
    return np.asarray(parts, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a pose trajectory and camera frustums as an ASCII PLY point preview.")
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--frustum-stride", type=int, default=30)
    parser.add_argument("--trajectory-color", default="255,220,0")
    parser.add_argument("--center-color", default="255,0,0")
    parser.add_argument("--forward-color", default="0,255,0")
    parser.add_argument("--frustum-color", default="0,180,255")
    parser.add_argument("--align-to")
    parser.add_argument("--align-max-time-diff", type=float, default=0.05)
    args = parser.parse_args()

    poses = read_pose_file(args.poses)
    calib = read_calib(args.calib)
    summary = {"poses_path": str(args.poses), "calib_path": str(args.calib)}

    if args.align_to:
        reference = read_pose_file(Path(args.align_to))
        poses, align_summary = align_poses(poses, reference, args.align_max_time_diff)
        summary["alignment"] = align_summary

    preview_points, preview_colors, preview_summary = build_preview(
        poses=poses,
        calib=calib,
        frustum_stride=max(1, args.frustum_stride),
        trajectory_color=parse_color(args.trajectory_color),
        center_color=parse_color(args.center_color),
        forward_color=parse_color(args.forward_color),
        frustum_color=parse_color(args.frustum_color),
    )
    write_ascii_ply(args.output_ply, preview_points, preview_colors)

    summary.update(preview_summary)
    summary["output_ply"] = str(args.output_ply)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
