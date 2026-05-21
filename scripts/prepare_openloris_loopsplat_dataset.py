#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import tarfile
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import py7zr
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp


def ensure_sequence_archive(package_tar: Path, sequence: str) -> Path:
    archive_path = package_tar.parent / f"{sequence}.7z"
    if archive_path.exists():
        return archive_path
    with tarfile.open(package_tar, "r") as tar:
        tar.extract(f"{sequence}.7z", path=package_tar.parent)
    return archive_path


def extract_targets(archive_path: Path, targets: list[str], out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seven_zip = shutil.which("7zz") or shutil.which("7z") or shutil.which("7za") or shutil.which("7zr")
    if not seven_zip:
        bundled = Path(os.environ.get("OPENLORIS_7ZZ_BIN", "")).expanduser() if os.environ.get("OPENLORIS_7ZZ_BIN") else None
        if bundled and bundled.exists():
            seven_zip = str(bundled)
    if not seven_zip:
        bundled = Path(__file__).resolve().parents[1] / "tools" / "7zip" / "7zz"
        if bundled.exists():
            seven_zip = str(bundled)
    if seven_zip:
        subprocess.run([seven_zip, "x", str(archive_path), f"-o{out_dir}", "-y", *targets], check=True)
        return
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extract(targets=targets, path=out_dir)


def read_timestamp_index(path: Path) -> list[tuple[float, str]]:
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ts, rel = line.split(maxsplit=1)
        entries.append((float(ts), rel))
    return entries


def read_groundtruth(path: Path) -> tuple[np.ndarray, np.ndarray, Rotation]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([float(x) for x in line.split()])
    data = np.asarray(rows, dtype=np.float64)
    return data[:, 0], data[:, 1:4], Rotation.from_quat(data[:, 4:8])


def read_sensor_camera(fs: cv2.FileStorage, node_name: str) -> dict:
    node = fs.getNode(node_name)
    intr = node.getNode("intrinsics").mat().reshape(-1)
    return {
        "width": int(node.getNode("width").real()),
        "height": int(node.getNode("height").real()),
        "fx": float(intr[0]),
        "fy": float(intr[1]),
        "cx": float(intr[2]),
        "cy": float(intr[3]),
        "model": node.getNode("model").string(),
    }


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
    queue = deque([(src, np.eye(4, dtype=np.float64))])
    visited = {src}
    while queue:
        node, acc = queue.popleft()
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


def nearest_match_indices(source_times: np.ndarray, target_times: np.ndarray, max_delta: float) -> np.ndarray:
    idxs = np.searchsorted(target_times, source_times)
    idxs = np.clip(idxs, 0, len(target_times) - 1)
    prev = np.clip(idxs - 1, 0, len(target_times) - 1)
    choose_prev = np.abs(target_times[prev] - source_times) <= np.abs(target_times[idxs] - source_times)
    idxs = np.where(choose_prev, prev, idxs)
    deltas = np.abs(target_times[idxs] - source_times)
    if np.any(deltas > max_delta):
        raise RuntimeError(f"Timestamp alignment exceeded max delta {max_delta}s (max observed {deltas.max():.6f}s)")
    return idxs


def parse_frame_indices(images_dir: Path) -> list[tuple[int, str]]:
    items = []
    for image_path in sorted(images_dir.glob("frame_*.png")):
        frame_idx = int(image_path.stem.split("_")[-1])
        items.append((frame_idx, image_path.name))
    if not items:
        raise RuntimeError(f"No frame_*.png images found in {images_dir}")
    return items


def write_assoc_file(path: Path, rows: list[tuple[float, str]]) -> None:
    with path.open("w") as f:
        f.write("# timestamp path\n")
        for ts, rel_path in rows:
            f.write(f"{ts:.9f} {rel_path}\n")


def write_groundtruth_file(path: Path, rows: list[tuple[float, np.ndarray]]) -> None:
    with path.open("w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, c2w in rows:
            quat = Rotation.from_matrix(c2w[:3, :3]).as_quat()
            tx, ty, tz = c2w[:3, 3]
            qx, qy, qz, qw = quat
            f.write(f"{ts:.9f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an OpenLORIS subset into a TUM-RGBD-style dataset for LoopSplat.")
    parser.add_argument("--package-tar", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--src-dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-timestamp-delta", type=float, default=0.05)
    parser.add_argument("--gt-pose-frame", choices=("camera", "base", "laser"), default="base")
    args = parser.parse_args()

    archive_path = ensure_sequence_archive(args.package_tar, args.sequence)
    meta_dir = args.output_dir.parent / f"_{args.sequence}_loopsplat_meta"
    extract_targets(
        archive_path,
        [
            f"{args.sequence}/sensors.yaml",
            f"{args.sequence}/trans_matrix.yaml",
            f"{args.sequence}/color.txt",
            f"{args.sequence}/aligned_depth.txt",
            f"{args.sequence}/groundtruth.txt",
        ],
        meta_dir,
    )
    seq_root = meta_dir / args.sequence

    sensor_fs = cv2.FileStorage(str(seq_root / "sensors.yaml"), cv2.FILE_STORAGE_READ)
    intr = read_sensor_camera(sensor_fs, "d400_color_optical_frame")
    sensor_fs.release()

    trans_fs = cv2.FileStorage(str(seq_root / "trans_matrix.yaml"), cv2.FILE_STORAGE_READ)
    transforms = read_transforms(trans_fs)
    trans_fs.release()
    gt_to_color = resolve_gt_pose_transform(transforms, args.gt_pose_frame)

    color_entries = read_timestamp_index(seq_root / "color.txt")
    depth_entries = read_timestamp_index(seq_root / "aligned_depth.txt")
    gt_times, gt_positions, gt_rotations = read_groundtruth(seq_root / "groundtruth.txt")
    frame_items = parse_frame_indices(args.src_dataset / "images")

    color_times = np.asarray([ts for ts, _ in color_entries], dtype=np.float64)
    depth_times = np.asarray([ts for ts, _ in depth_entries], dtype=np.float64)
    frame_indices = [frame_idx for frame_idx, _ in frame_items]
    frame_times = color_times[frame_indices]
    depth_match_indices = nearest_match_indices(frame_times, depth_times, args.max_timestamp_delta)

    interp_pos = np.stack([np.interp(frame_times, gt_times, gt_positions[:, i]) for i in range(3)], axis=1)
    interp_rot = Slerp(gt_times, gt_rotations)(frame_times).as_matrix()

    depth_targets = [f"{args.sequence}/{depth_entries[idx][1]}" for idx in depth_match_indices]
    stage_dir = args.output_dir.parent / f"_{args.sequence}_loopsplat_depth_extract"
    extract_targets(archive_path, depth_targets, stage_dir)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    rgb_dir = args.output_dir / "rgb"
    depth_dir = args.output_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    rgb_rows = []
    depth_rows = []
    gt_rows = []
    for (frame_idx, image_name), frame_time, depth_idx, rot, pos in zip(
        frame_items, frame_times, depth_match_indices, interp_rot, interp_pos
    ):
        rgb_src = args.src_dataset / "images" / image_name
        rgb_rel = f"rgb/{image_name}"
        shutil.copy2(rgb_src, args.output_dir / rgb_rel)

        depth_src = stage_dir / args.sequence / depth_entries[depth_idx][1]
        depth_rel = f"depth/{image_name}"
        shutil.move(depth_src, args.output_dir / depth_rel)

        world_from_gt = np.eye(4, dtype=np.float64)
        world_from_gt[:3, :3] = rot
        world_from_gt[:3, 3] = pos
        gt_rows.append((frame_time, world_from_gt @ gt_to_color))
        rgb_rows.append((frame_time, rgb_rel))
        depth_rows.append((frame_time, depth_rel))

    write_assoc_file(args.output_dir / "rgb.txt", rgb_rows)
    write_assoc_file(args.output_dir / "depth.txt", depth_rows)
    write_groundtruth_file(args.output_dir / "groundtruth.txt", gt_rows)
    write_groundtruth_file(args.output_dir / "pose.txt", gt_rows)

    metadata = {
        "sequence": args.sequence,
        "src_dataset": str(args.src_dataset),
        "package_tar": str(args.package_tar),
        "gt_pose_frame": args.gt_pose_frame,
        "num_frames": len(frame_items),
        "intrinsics": intr,
    }
    (args.output_dir / "openloris_loopsplat_metadata.json").write_text(json.dumps(metadata, indent=2))

    shutil.rmtree(meta_dir, ignore_errors=True)
    shutil.rmtree(stage_dir, ignore_errors=True)
    print(json.dumps({"output_dir": str(args.output_dir), "num_frames": len(frame_items), "intrinsics": intr}, indent=2))


if __name__ == "__main__":
    main()
