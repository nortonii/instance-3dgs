#!/usr/bin/env python3

import argparse
import json
import math
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def ensure_sequence_archive(package_tar: Path, sequence: str) -> Path:
    archive_path = package_tar.parent / f"{sequence}.7z"
    if archive_path.exists():
        return archive_path
    with tarfile.open(package_tar, "r") as tar:
        tar.extract(f"{sequence}.7z", path=package_tar.parent)
    return archive_path


def run_7za(seven_zip: Path, archive_path: Path, out_dir: Path, targets: list[str]) -> None:
    if not targets:
        return
    cmd = [str(seven_zip), "x", str(archive_path), f"-o{out_dir}", "-y", *targets]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_timestamp_index(path: Path) -> list[tuple[float, str]]:
    entries = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        stamp, rel = line.split(maxsplit=1)
        entries.append((float(stamp), rel))
    return entries


def parse_intrinsics(sensors_yaml: Path) -> dict[str, float]:
    text = sensors_yaml.read_text(encoding="utf-8", errors="ignore")
    match = re.search(
        r"d400_color_optical_frame:\s.*?width:\s*(\d+)\s.*?height:\s*(\d+)\s.*?data:\s*\[\s*([^\]]+)\]",
        text,
        re.S,
    )
    if not match:
        raise RuntimeError(f"Failed to parse color intrinsics from {sensors_yaml}")
    width, height = int(match.group(1)), int(match.group(2))
    vals = [float(v.strip()) for v in match.group(3).replace("\n", " ").split(",") if v.strip()]
    if len(vals) != 4:
        raise RuntimeError(f"Expected 4 intrinsics values in {sensors_yaml}, got {vals}")
    return {
        "width": width,
        "height": height,
        "fx": vals[0],
        "fy": vals[1],
        "cx": vals[2],
        "cy": vals[3],
    }


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


def write_groundtruth(path: Path, groundtruth: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in groundtruth:
            f.write(
                f"{row['timestamp']:.6f} {row['x']:.9f} {row['y']:.9f} {row['z']:.9f} "
                f"{row['qx']:.9f} {row['qy']:.9f} {row['qz']:.9f} {row['qw']:.9f}\n"
            )


def nearest_match_indices(source_times: list[float], target_times: list[float], max_delta: float) -> list[int]:
    matches = []
    j = 0
    for stamp in source_times:
        while j + 1 < len(target_times) and target_times[j + 1] <= stamp:
            j += 1
        candidates = [j]
        if j + 1 < len(target_times):
            candidates.append(j + 1)
        best = min(candidates, key=lambda idx: abs(target_times[idx] - stamp))
        delta = abs(target_times[best] - stamp)
        if delta > max_delta:
            raise RuntimeError(f"Depth alignment exceeded max delta {max_delta}s at stamp {stamp} (delta={delta})")
        matches.append(best)
    return matches


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an OpenLORIS RGB-D folder for rtabmap-rgbd_dataset.")
    parser.add_argument("--package-tar", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seven-zip", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-time-diff", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--gt-pose-frame", choices=("camera", "base", "laser"), default="base")
    args = parser.parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be positive")

    archive_path = ensure_sequence_archive(args.package_tar, args.sequence)
    meta_extract = args.output_dir.parent / f"_{args.sequence}_rtabmap_meta"
    frame_extract = args.output_dir.parent / f"_{args.sequence}_rtabmap_frames"
    shutil.rmtree(meta_extract, ignore_errors=True)
    shutil.rmtree(frame_extract, ignore_errors=True)
    meta_extract.mkdir(parents=True, exist_ok=True)
    frame_extract.mkdir(parents=True, exist_ok=True)

    meta_targets = [
        f"{args.sequence}/sensors.yaml",
        f"{args.sequence}/trans_matrix.yaml",
        f"{args.sequence}/color.txt",
        f"{args.sequence}/aligned_depth.txt",
        f"{args.sequence}/groundtruth.txt",
    ]
    run_7za(args.seven_zip, archive_path, meta_extract, meta_targets)

    seq_root = meta_extract / args.sequence
    intrinsics = parse_intrinsics(seq_root / "sensors.yaml")
    trans_fs = cv2.FileStorage(str(seq_root / "trans_matrix.yaml"), cv2.FILE_STORAGE_READ)
    transforms = read_transforms(trans_fs)
    trans_fs.release()
    groundtruth_raw = read_groundtruth(seq_root / "groundtruth.txt")
    groundtruth_camera = transform_groundtruth(
        groundtruth_raw,
        resolve_gt_pose_transform(transforms, args.gt_pose_frame),
    )
    color_entries = read_timestamp_index(seq_root / "color.txt")
    depth_entries = read_timestamp_index(seq_root / "aligned_depth.txt")

    selected_indices = list(range(0, len(color_entries), args.stride))
    selected_color_entries = [color_entries[i] for i in selected_indices]
    depth_match_indices = nearest_match_indices(
        [stamp for stamp, _ in selected_color_entries],
        [stamp for stamp, _ in depth_entries],
        args.max_time_diff,
    )

    color_targets = [f"{args.sequence}/{rel}" for _, rel in selected_color_entries]
    depth_targets = [f"{args.sequence}/{depth_entries[i][1]}" for i in depth_match_indices]
    for batch in chunked(color_targets + depth_targets, args.batch_size):
        run_7za(args.seven_zip, archive_path, frame_extract, batch)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    rgb_dir = args.output_dir / "rgb_sync"
    depth_dir = args.output_dir / "depth_sync"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "frame_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for out_idx, frame_idx in enumerate(selected_indices):
            stamp, color_rel = color_entries[frame_idx]
            depth_rel = depth_entries[depth_match_indices[out_idx]][1]
            stamp_name = f"{stamp:.6f}.png"
            shutil.move(str(frame_extract / args.sequence / color_rel), rgb_dir / stamp_name)
            shutil.move(str(frame_extract / args.sequence / depth_rel), depth_dir / stamp_name)
            fullspan_ref = (frame_idx % 30) in {0, 7, 15, 22}
            manifest.write(
                json.dumps(
                    {
                        "selected_index": out_idx,
                        "frame_idx": frame_idx,
                        "timestamp": stamp,
                        "color_relpath": color_rel,
                        "depth_relpath": depth_rel,
                        "rgb_file": f"rgb_sync/{stamp_name}",
                        "depth_file": f"depth_sync/{stamp_name}",
                        "matches_existing_fullspan_subset": fullspan_ref,
                    }
                )
                + "\n"
            )

    write_groundtruth(args.output_dir / "groundtruth.txt", groundtruth_camera)
    shutil.copy2(seq_root / "groundtruth.txt", args.output_dir / "groundtruth_raw.txt")
    shutil.copy2(seq_root / "trans_matrix.yaml", args.output_dir / "trans_matrix.yaml")
    recommended_args = [
        "--output",
        str(args.output_dir / "rtabmap_out"),
        "--output_name",
        args.sequence.replace("-", "_"),
        "--width",
        str(intrinsics["width"]),
        "--height",
        str(intrinsics["height"]),
        "--fx",
        str(intrinsics["fx"]),
        "--fy",
        str(intrinsics["fy"]),
        "--cx",
        str(intrinsics["cx"]),
        "--cy",
        str(intrinsics["cy"]),
        "--depth_factor",
        "1000",
        "--Rtabmap/DetectionRate",
        "0",
        "--Rtabmap/CreateIntermediateNodes",
        "true",
    ]
    (args.output_dir / "rtabmap_args.txt").write_text(" ".join(recommended_args) + "\n", encoding="utf-8")
    (args.output_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "sequence": args.sequence,
                "package_tar": str(args.package_tar),
                "archive_path": str(archive_path),
                "stride": args.stride,
                "max_time_diff": args.max_time_diff,
                "gt_pose_frame": args.gt_pose_frame,
                "num_color_frames_total": len(color_entries),
                "num_frames_selected": len(selected_indices),
                "intrinsics": intrinsics,
                "depth_factor": 1000,
                "rtabmap_args": recommended_args,
                "estimated_duration_s": None
                if len(color_entries) < 2
                else color_entries[-1][0] - color_entries[0][0],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    shutil.rmtree(meta_extract, ignore_errors=True)
    shutil.rmtree(frame_extract, ignore_errors=True)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "sequence": args.sequence,
                "num_frames_selected": len(selected_indices),
                "first_timestamp": selected_color_entries[0][0] if selected_color_entries else None,
                "last_timestamp": selected_color_entries[-1][0] if selected_color_entries else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
