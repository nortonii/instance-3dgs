#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import cv2
import numpy as np
import py7zr
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp


INVALID_DEPTH = 65535


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild an exported OpenLORIS subset initialization point cloud from all raw "
            "frames inside the subset time interval, using dense depth backprojection plus "
            "voxel downsampling."
        )
    )
    parser.add_argument("--base-dataset", required=True, type=Path)
    parser.add_argument("--out-dataset", required=True, type=Path)
    parser.add_argument("--package-tar", type=Path)
    parser.add_argument("--sequence", type=str)
    parser.add_argument(
        "--max-depth-m",
        type=float,
        default=0.0,
        help="Maximum valid depth in meters. Use 0 or a negative value to disable far-depth truncation.",
    )
    parser.add_argument("--max-timestamp-delta", type=float, default=0.05)
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Optional post-voxel random cap. 0 disables the cap.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=12,
        help="How many raw frames to accumulate before chunk-level voxel reduction.",
    )
    parser.add_argument(
        "--extend-after-frames",
        type=int,
        default=0,
        help="Extend the raw backprojection interval by this many color frames after the subset end.",
    )
    return parser.parse_args()


def ensure_sequence_archive(package_tar: Path, sequence: str) -> Path:
    archive_path = package_tar.parent / f"{sequence}.7z"
    if archive_path.exists():
        return archive_path
    with tarfile.open(package_tar, "r") as tar:
        tar.extract(f"{sequence}.7z", path=package_tar.parent)
    return archive_path


def extract_targets(archive_path: Path, targets: list[str], out_dir: Path, batch_size: int = 32) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seven_zip = os.environ.get("SEVEN_ZIP_BIN")
    if seven_zip:
        seven_zip = shutil.which(seven_zip) or seven_zip
    else:
        seven_zip = shutil.which("7zz") or shutil.which("7z") or shutil.which("7za") or shutil.which("7zr")
    if seven_zip:
        for start in range(0, len(targets), batch_size):
            batch = targets[start : start + batch_size]
            subprocess.run(
                [seven_zip, "x", str(archive_path), f"-o{out_dir}", "-y", *batch],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return
    for start in range(0, len(targets), batch_size):
        batch = targets[start : start + batch_size]
        with py7zr.SevenZipFile(archive_path, mode="r") as zf:
            zf.extract(targets=batch, path=out_dir)


def read_timestamp_index(path: Path) -> list[tuple[float, str]]:
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ts, rel = line.split(maxsplit=1)
        entries.append((float(ts), rel))
    return entries


def read_camera(path: Path) -> dict:
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        return {
            "raw": line,
            "width": int(parts[2]),
            "height": int(parts[3]),
            "fx": float(parts[4]),
            "fy": float(parts[5]),
            "cx": float(parts[6]),
            "cy": float(parts[7]),
        }
    raise RuntimeError(f"No camera entry found in {path}")


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
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    items = []
    for idx in range(0, len(lines), 2):
        parts = lines[idx].split()
        image_name = parts[9]
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = quat_wxyz_to_rotmat(qw, qx, qy, qz)
        w2c[:3, 3] = [tx, ty, tz]
        items.append(
            {
                "image_name": image_name,
                "w2c": w2c,
                "second_line": lines[idx + 1],
            }
        )
    return items


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


def parse_frame_index(image_name: str) -> int:
    return int(Path(image_name).stem.split("_")[-1])


def load_pose_anchor_items(base_dataset: Path, meta: dict) -> list[tuple[int, np.ndarray]]:
    dataset_candidates = [base_dataset]
    subset_source = meta.get("subset_source_dataset")
    if subset_source:
        subset_source_path = Path(subset_source).resolve()
        if subset_source_path.exists() and subset_source_path != base_dataset:
            dataset_candidates.append(subset_source_path)

    anchors: dict[int, np.ndarray] = {}
    for dataset_path in dataset_candidates:
        images_path = dataset_path / "sparse" / "0" / "images.txt"
        if not images_path.exists():
            continue
        for item in read_images(images_path):
            frame_idx = parse_frame_index(item["image_name"])
            anchors[frame_idx] = np.linalg.inv(item["w2c"])
    return sorted(anchors.items())


def structured_voxels(voxels: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(voxels).view([("x", np.int32), ("y", np.int32), ("z", np.int32)]).reshape(-1)


def reduce_voxels(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    voxels = np.floor(points / voxel_size).astype(np.int32)
    unique_voxels, inverse = np.unique(structured_voxels(voxels), return_inverse=True)
    counts = np.bincount(inverse).astype(np.int64)
    sum_xyz = np.stack([np.bincount(inverse, weights=points[:, axis]) for axis in range(3)], axis=1)
    sum_rgb = np.stack([np.bincount(inverse, weights=colors[:, axis]) for axis in range(3)], axis=1)
    voxel_xyz = np.stack([unique_voxels["x"], unique_voxels["y"], unique_voxels["z"]], axis=1).astype(np.int32)
    return voxel_xyz, counts, sum_xyz, sum_rgb


def merge_voxel_accumulators(
    voxel_xyz_a: np.ndarray | None,
    counts_a: np.ndarray | None,
    sum_xyz_a: np.ndarray | None,
    sum_rgb_a: np.ndarray | None,
    voxel_xyz_b: np.ndarray,
    counts_b: np.ndarray,
    sum_xyz_b: np.ndarray,
    sum_rgb_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if voxel_xyz_a is None:
        return voxel_xyz_b, counts_b, sum_xyz_b, sum_rgb_b
    merged_voxels = np.concatenate([voxel_xyz_a, voxel_xyz_b], axis=0)
    merged_counts = np.concatenate([counts_a, counts_b], axis=0)
    merged_sum_xyz = np.concatenate([sum_xyz_a, sum_xyz_b], axis=0)
    merged_sum_rgb = np.concatenate([sum_rgb_a, sum_rgb_b], axis=0)
    unique_voxels, inverse = np.unique(structured_voxels(merged_voxels), return_inverse=True)
    counts = np.bincount(inverse, weights=merged_counts).astype(np.int64)
    sum_xyz = np.stack([np.bincount(inverse, weights=merged_sum_xyz[:, axis]) for axis in range(3)], axis=1)
    sum_rgb = np.stack([np.bincount(inverse, weights=merged_sum_rgb[:, axis]) for axis in range(3)], axis=1)
    voxel_xyz = np.stack([unique_voxels["x"], unique_voxels["y"], unique_voxels["z"]], axis=1).astype(np.int32)
    return voxel_xyz, counts, sum_xyz, sum_rgb


def finalize_points(
    voxel_xyz: np.ndarray,
    counts: np.ndarray,
    sum_xyz: np.ndarray,
    sum_rgb: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = (sum_xyz / counts[:, None]).astype(np.float32)
    colors = np.clip(np.rint(sum_rgb / counts[:, None]), 0, 255).astype(np.uint8)
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(points), size=max_points, replace=False)
        points = points[sel]
        colors = colors[sel]
    return points, colors


def backproject_interval_points(
    extract_root: Path,
    sequence: str,
    raw_items: list[dict],
    camtoworlds: dict[int, np.ndarray],
    intr: dict,
    voxel_size: float,
    max_depth_m: float,
    max_points: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]
    width = intr["width"]
    height = intr["height"]
    norm_x = (np.arange(width, dtype=np.float32) - cx) / fx
    norm_y = (np.arange(height, dtype=np.float32) - cy) / fy
    grid_x = np.broadcast_to(norm_x[None, :], (height, width))
    grid_y = np.broadcast_to(norm_y[:, None], (height, width))

    global_voxel_xyz = None
    global_counts = None
    global_sum_xyz = None
    global_sum_rgb = None

    chunk_points = []
    chunk_colors = []
    for idx, item in enumerate(raw_items, start=1):
        rgb = np.array(Image.open(extract_root / sequence / item["color_rel"]).convert("RGB"))
        depth = np.array(Image.open(extract_root / sequence / item["depth_rel"]), dtype=np.uint16)
        z_raw = depth
        valid = (z_raw > 0) & (z_raw < INVALID_DEPTH)
        z = z_raw.astype(np.float32) / 1000.0
        if max_depth_m > 0:
            valid &= z <= max_depth_m
        if np.any(valid):
            pts_cam = np.stack([grid_x[valid] * z[valid], grid_y[valid] * z[valid], z[valid]], axis=1)
            c2w = camtoworlds[item["frame_idx"]]
            pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]
            cols = rgb[valid]
            chunk_points.append(pts_world.astype(np.float32))
            chunk_colors.append(cols.astype(np.uint8))

        if idx % chunk_size == 0 or idx == len(raw_items):
            if not chunk_points:
                continue
            chunk_point_array = np.concatenate(chunk_points, axis=0)
            chunk_color_array = np.concatenate(chunk_colors, axis=0)
            chunk_voxel_xyz, chunk_counts, chunk_sum_xyz, chunk_sum_rgb = reduce_voxels(
                chunk_point_array,
                chunk_color_array,
                voxel_size=voxel_size,
            )
            global_voxel_xyz, global_counts, global_sum_xyz, global_sum_rgb = merge_voxel_accumulators(
                global_voxel_xyz,
                global_counts,
                global_sum_xyz,
                global_sum_rgb,
                chunk_voxel_xyz,
                chunk_counts,
                chunk_sum_xyz,
                chunk_sum_rgb,
            )
            chunk_points = []
            chunk_colors = []

    if global_voxel_xyz is None:
        raise RuntimeError("No valid depth-backed points were generated from the raw interval.")
    return finalize_points(global_voxel_xyz, global_counts, global_sum_xyz, global_sum_rgb, max_points=max_points)


def write_points3d(path: Path, points_world: np.ndarray, colors: np.ndarray) -> None:
    with path.open("w") as f:
        f.write(
            "# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )
        for point_id, (xyz, rgb) in enumerate(zip(points_world, colors), start=1):
            f.write(f"{point_id} {xyz[0]} {xyz[1]} {xyz[2]} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])} 0\n")


def main() -> None:
    args = parse_args()
    base_dataset = args.base_dataset.resolve()
    out_dataset = args.out_dataset.resolve()

    meta_path = base_dataset / "openloris_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    package_tar = (args.package_tar or meta.get("package_tar"))
    sequence = args.sequence or meta.get("sequence")
    if not package_tar or not sequence:
        raise ValueError("package-tar and sequence must be provided directly or via base dataset metadata.")
    package_tar = Path(package_tar).resolve()

    base_items = read_images(base_dataset / "sparse" / "0" / "images.txt")
    camera = read_camera(base_dataset / "sparse" / "0" / "cameras.txt")
    frame_indices = np.asarray([parse_frame_index(item["image_name"]) for item in base_items], dtype=np.int64)
    first_frame_idx = int(frame_indices.min())
    last_frame_idx = int(frame_indices.max())

    color_meta_dir = Path(tempfile.mkdtemp(prefix="openloris_raw_interval_meta_"))
    raw_extract_dir = Path(tempfile.mkdtemp(prefix="openloris_raw_interval_frames_"))
    try:
        archive_path = ensure_sequence_archive(package_tar, sequence)
        extract_targets(
            archive_path,
            [f"{sequence}/color.txt", f"{sequence}/aligned_depth.txt"],
            color_meta_dir,
        )
        seq_meta = color_meta_dir / sequence
        color_entries = read_timestamp_index(seq_meta / "color.txt")
        depth_entries = read_timestamp_index(seq_meta / "aligned_depth.txt")
        color_times = np.asarray([ts for ts, _ in color_entries], dtype=np.float64)
        depth_times = np.asarray([ts for ts, _ in depth_entries], dtype=np.float64)

        if last_frame_idx >= len(color_entries):
            raise RuntimeError(f"Subset references frame {last_frame_idx}, but package only has {len(color_entries)} color entries.")

        requested_last_frame_idx = last_frame_idx + max(args.extend_after_frames, 0)
        interval_end_frame_idx = min(requested_last_frame_idx, len(color_entries) - 1)

        pose_anchor_items = load_pose_anchor_items(base_dataset, meta)
        if not pose_anchor_items:
            raise RuntimeError("No pose anchors found for interval interpolation.")
        pose_anchor_frame_indices = np.asarray([frame_idx for frame_idx, _ in pose_anchor_items], dtype=np.int64)
        if first_frame_idx < int(pose_anchor_frame_indices.min()) or interval_end_frame_idx > int(pose_anchor_frame_indices.max()):
            raise RuntimeError(
                "Requested interval exceeds available pose anchors: "
                f"[{first_frame_idx}, {interval_end_frame_idx}] vs "
                f"[{int(pose_anchor_frame_indices.min())}, {int(pose_anchor_frame_indices.max())}]"
            )

        pose_anchor_times = color_times[pose_anchor_frame_indices]
        pose_anchor_positions = np.asarray([c2w[:3, 3] for _, c2w in pose_anchor_items], dtype=np.float64)
        pose_anchor_rotations = Rotation.from_matrix(
            np.asarray([c2w[:3, :3] for _, c2w in pose_anchor_items], dtype=np.float64)
        )

        interval_indices = np.arange(first_frame_idx, interval_end_frame_idx + 1, dtype=np.int64)
        interval_times = color_times[interval_indices]
        interp_pos = np.stack(
            [np.interp(interval_times, pose_anchor_times, pose_anchor_positions[:, axis]) for axis in range(3)],
            axis=1,
        )
        interp_rot = Slerp(pose_anchor_times, pose_anchor_rotations)(interval_times).as_matrix()
        camtoworlds = {}
        for frame_idx, rot, pos in zip(interval_indices, interp_rot, interp_pos):
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = rot
            c2w[:3, 3] = pos
            camtoworlds[int(frame_idx)] = c2w

        depth_match_indices = nearest_match_indices(interval_times, depth_times, args.max_timestamp_delta)
        extract_targets(
            archive_path,
            [f"{sequence}/{color_entries[idx][1]}" for idx in interval_indices]
            + [f"{sequence}/{depth_entries[idx][1]}" for idx in depth_match_indices],
            raw_extract_dir,
        )
        raw_items = [
            {
                "frame_idx": int(frame_idx),
                "color_rel": color_entries[int(frame_idx)][1],
                "depth_rel": depth_entries[int(depth_idx)][1],
            }
            for frame_idx, depth_idx in zip(interval_indices, depth_match_indices)
        ]
        points_world, colors = backproject_interval_points(
            extract_root=raw_extract_dir,
            sequence=sequence,
            raw_items=raw_items,
            camtoworlds=camtoworlds,
            intr=camera,
            voxel_size=args.voxel_size,
            max_depth_m=args.max_depth_m,
            max_points=args.max_points,
            chunk_size=args.chunk_size,
        )
    finally:
        shutil.rmtree(color_meta_dir, ignore_errors=True)
        shutil.rmtree(raw_extract_dir, ignore_errors=True)

    if out_dataset.exists():
        shutil.rmtree(out_dataset)
    shutil.copytree(base_dataset, out_dataset, symlinks=True)
    write_points3d(out_dataset / "sparse" / "0" / "points3D.txt", points_world, colors)

    out_meta = dict(meta)
    out_meta.update(
        {
            "init_source_dataset": str(base_dataset),
            "init_raw_interval_start_frame": first_frame_idx,
            "init_raw_interval_end_frame": interval_end_frame_idx,
            "init_raw_interval_num_frames": interval_end_frame_idx - first_frame_idx + 1,
            "init_raw_interval_base_end_frame": last_frame_idx,
            "init_raw_interval_extend_after_frames": args.extend_after_frames,
            "init_voxel_size": args.voxel_size,
            "init_max_depth_m": None if args.max_depth_m <= 0 else args.max_depth_m,
            "init_max_timestamp_delta": args.max_timestamp_delta,
            "init_post_voxel_max_points": args.max_points,
            "num_init_points": int(len(points_world)),
        }
    )
    (out_dataset / "openloris_metadata.json").write_text(json.dumps(out_meta, indent=2))
    print(
        json.dumps(
            {
                "out_dataset": str(out_dataset),
                "interval_start_frame": first_frame_idx,
                "interval_end_frame": interval_end_frame_idx,
                "interval_num_frames": interval_end_frame_idx - first_frame_idx + 1,
                "base_end_frame": last_frame_idx,
                "extended_after_frames": args.extend_after_frames,
                "num_init_points": int(len(points_world)),
                "voxel_size": args.voxel_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
