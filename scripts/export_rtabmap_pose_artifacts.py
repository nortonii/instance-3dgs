#!/usr/bin/env python3

import argparse
import json
import math
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
        if len(parts) < 9:
            continue
        poses.append(
            {
                "timestamp": float(parts[0]),
                "x": float(parts[1]),
                "y": float(parts[2]),
                "z": float(parts[3]),
                "qx": float(parts[4]),
                "qy": float(parts[5]),
                "qz": float(parts[6]),
                "qw": float(parts[7]),
                "id": int(parts[8]),
            }
        )
    return poses


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_groundtruth(path: Path) -> list[dict]:
    gt = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        gt.append(
            {
                "timestamp": float(parts[0]),
                "x": float(parts[1]),
                "y": float(parts[2]),
                "z": float(parts[3]),
                "qx": float(parts[4]),
                "qy": float(parts[5]),
                "qz": float(parts[6]),
                "qw": float(parts[7]),
            }
        )
    return gt


def read_transforms(fs: cv2.FileStorage) -> dict[tuple[str, str], np.ndarray]:
    node = fs.getNode("trans_matrix")
    transforms = {}
    for i in range(node.size()):
        entry = node.at(i)
        parent = entry.getNode("parent_frame").string()
        child = entry.getNode("child_frame").string()
        transforms[(parent, child)] = entry.getNode("matrix").mat()
    return transforms


def find_transform(transforms: dict[tuple[str, str], np.ndarray], src: str, dst: str) -> np.ndarray:
    queue = [(src, np.eye(4, dtype=np.float64))]
    visited = {src}
    while queue:
        node, acc = queue.pop(0)
        if node == dst:
            return acc
        for (parent, child), mat in transforms.items():
            if parent == node and child not in visited:
                visited.add(child)
                queue.append((child, acc @ mat))
            if child == node and parent not in visited:
                visited.add(parent)
                queue.append((parent, acc @ np.linalg.inv(mat)))
    raise KeyError(f"No transform path from {src} to {dst}")


def resolve_gt_pose_transform(
    transforms: dict[tuple[str, str], np.ndarray],
    gt_pose_frame: str,
) -> np.ndarray:
    if gt_pose_frame == "camera":
        return np.eye(4, dtype=np.float64)
    if gt_pose_frame == "base":
        return find_transform(transforms, "base_link", "d400_color_optical_frame")
    if gt_pose_frame == "laser":
        return find_transform(transforms, "laser", "d400_color_optical_frame")
    raise ValueError(f"Unsupported gt pose frame: {gt_pose_frame}")


def transform_groundtruth(groundtruth: list[dict], gt_to_camera: np.ndarray) -> list[dict]:
    transformed = []
    for row in groundtruth:
        world_from_gt = np.eye(4, dtype=np.float64)
        world_from_gt[:3, :3] = Rotation.from_quat([row["qx"], row["qy"], row["qz"], row["qw"]]).as_matrix()
        world_from_gt[:3, 3] = [row["x"], row["y"], row["z"]]
        world_from_camera = world_from_gt @ gt_to_camera
        quat_xyzw = Rotation.from_matrix(world_from_camera[:3, :3]).as_quat()
        transformed.append(
            {
                "timestamp": row["timestamp"],
                "x": float(world_from_camera[0, 3]),
                "y": float(world_from_camera[1, 3]),
                "z": float(world_from_camera[2, 3]),
                "qx": float(quat_xyzw[0]),
                "qy": float(quat_xyzw[1]),
                "qz": float(quat_xyzw[2]),
                "qw": float(quat_xyzw[3]),
            }
        )
    return transformed


def nearest_index(times: list[float], stamp: float) -> int:
    idx = bisect_left(times, stamp)
    if idx <= 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    return idx - 1 if abs(times[idx - 1] - stamp) <= abs(times[idx] - stamp) else idx


def rmse_translation(poses: list[dict], gt: list[dict], max_delta: float) -> dict:
    gt_times = [p["timestamp"] for p in gt]
    matched = 0
    sq_err = 0.0
    max_err = 0.0
    for pose in poses:
        idx = nearest_index(gt_times, pose["timestamp"])
        delta = abs(gt[idx]["timestamp"] - pose["timestamp"])
        if delta > max_delta:
            continue
        err = math.sqrt(
            (pose["x"] - gt[idx]["x"]) ** 2
            + (pose["y"] - gt[idx]["y"]) ** 2
            + (pose["z"] - gt[idx]["z"]) ** 2
        )
        matched += 1
        sq_err += err * err
        max_err = max(max_err, err)
    return {
        "matched_count": matched,
        "rmse_translation_m": math.sqrt(sq_err / matched) if matched else None,
        "max_translation_m": max_err if matched else None,
        "max_delta_s": max_delta,
    }


def attach_manifest(poses: list[dict], manifest: list[dict]) -> list[dict]:
    manifest_by_stamp = {round(item["timestamp"], 6): item for item in manifest}
    enriched = []
    for pose in poses:
        item = manifest_by_stamp.get(round(pose["timestamp"], 6))
        out = dict(pose)
        if item:
            out.update(
                {
                    "frame_idx": item["frame_idx"],
                    "selected_index": item["selected_index"],
                    "rgb_file": item["rgb_file"],
                    "depth_file": item["depth_file"],
                    "matches_existing_fullspan_subset": item["matches_existing_fullspan_subset"],
                }
            )
        enriched.append(out)
    return enriched


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process RTAB-Map exported poses.")
    parser.add_argument("--robot-poses", type=Path, required=True)
    parser.add_argument("--camera-poses", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-time-diff", type=float, default=0.05)
    parser.add_argument("--groundtruth-frame", choices=("camera", "base", "laser"), default="camera")
    parser.add_argument("--transforms", type=Path)
    args = parser.parse_args()

    if args.groundtruth_frame != "camera" and args.transforms is None:
        raise ValueError("--transforms is required when --groundtruth-frame is not camera")

    robot_poses = read_pose_file(args.robot_poses)
    camera_poses = read_pose_file(args.camera_poses)
    manifest = read_manifest(args.manifest)
    groundtruth = read_groundtruth(args.groundtruth)
    if args.groundtruth_frame != "camera":
        trans_fs = cv2.FileStorage(str(args.transforms), cv2.FILE_STORAGE_READ)
        transforms = read_transforms(trans_fs)
        trans_fs.release()
        groundtruth = transform_groundtruth(
            groundtruth,
            resolve_gt_pose_transform(transforms, args.groundtruth_frame),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    camera_summary = rmse_translation(camera_poses, groundtruth, args.max_time_diff)
    camera_enriched = attach_manifest(camera_poses, manifest)
    robot_enriched = attach_manifest(robot_poses, manifest)

    fullspan_subset = [p for p in camera_enriched if p.get("matches_existing_fullspan_subset")]
    with (args.output_dir / "camera_poses_fullspan_subset.txt").open("w", encoding="utf-8") as f:
        for pose in fullspan_subset:
            f.write(
                f"{pose['timestamp']:.6f} {pose['x']} {pose['y']} {pose['z']} "
                f"{pose['qx']} {pose['qy']} {pose['qz']} {pose['qw']} "
                f"{pose.get('frame_idx', -1)} {pose['id']}\n"
            )

    write_jsonl(args.output_dir / "camera_poses.jsonl", camera_enriched)
    write_jsonl(args.output_dir / "robot_poses.jsonl", robot_enriched)

    summary = {
        "robot_pose_count": len(robot_poses),
        "camera_pose_count": len(camera_poses),
        "manifest_count": len(manifest),
        "coverage_ratio": (len(camera_poses) / len(manifest)) if manifest else None,
        "fullspan_subset_pose_count": len(fullspan_subset),
        "groundtruth_frame": "camera",
        "camera_translation_vs_gt": camera_summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
