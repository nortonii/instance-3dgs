#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import tarfile
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

try:
    import py7zr
except ImportError:
    py7zr = None


INVALID_DEPTH = 65535


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a denser first-half OpenLORIS dataset with GT poses and depth-backed "
            "initialization directly from the raw package."
        )
    )
    parser.add_argument("--base-dataset", required=True, type=Path)
    parser.add_argument("--out-dataset", required=True, type=Path)
    parser.add_argument("--package-tar", type=Path)
    parser.add_argument("--sequence", type=str)
    parser.add_argument("--gt-pose-frame", choices=("camera", "base", "laser"), default="base")
    parser.add_argument(
        "--time-fraction",
        type=float,
        default=0.5,
        help="Use this fraction of the base dataset time span, measured in raw frame index.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=0,
        help="Raw-frame stride for the exported subset. 0 derives a doubled-rate stride from the base dataset.",
    )
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--pixel-stride", type=int, default=16)
    parser.add_argument("--max-depth-m", type=float, default=15.0)
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument("--max-timestamp-delta", type=float, default=0.05)
    return parser.parse_args()


def ensure_sequence_archive(package_tar: Path, sequence: str) -> Path:
    archive_path = package_tar.parent / f"{sequence}.7z"
    if archive_path.exists():
        return archive_path
    with tarfile.open(package_tar, "r") as tar:
        tar.extract(f"{sequence}.7z", path=package_tar.parent)
    return archive_path


def extract_targets(archive_path: Path, targets: list[str], out_dir: Path, batch_size: int = 64) -> None:
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
        for start in range(0, len(targets), batch_size):
            batch = targets[start : start + batch_size]
            subprocess.run(
                [seven_zip, "x", str(archive_path), f"-o{out_dir}", "-y", *batch],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return

    if py7zr is None:
        raise RuntimeError("Neither a 7z executable nor py7zr is available for archive extraction.")

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


def read_images(path: Path) -> list[dict]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    items = []
    for idx in range(0, len(lines), 2):
        parts = lines[idx].split()
        image_name = parts[9]
        items.append({"image_name": image_name, "frame_idx": parse_frame_index(image_name)})
    return items


def parse_frame_index(image_name: str) -> int:
    return int(Path(image_name).stem.split("_")[-1])


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


def compute_dense_step(base_frame_indices: list[int]) -> int:
    diffs = np.diff(np.asarray(sorted(base_frame_indices), dtype=np.int64))
    mean_step = float(np.mean(diffs))
    dense_step = max(1, int(round(mean_step / 2.0)))
    return dense_step


def quat_from_rotmat(r: np.ndarray) -> tuple[float, float, float, float]:
    trace = np.trace(r)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (r[2, 1] - r[1, 2]) * s
        qy = (r[0, 2] - r[2, 0]) * s
        qz = (r[1, 0] - r[0, 1]) * s
    else:
        idx = int(np.argmax(np.diag(r)))
        if idx == 0:
            s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
            qw = (r[2, 1] - r[1, 2]) / s
            qx = 0.25 * s
            qy = (r[0, 1] + r[1, 0]) / s
            qz = (r[0, 2] + r[2, 0]) / s
        elif idx == 1:
            s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
            qw = (r[0, 2] - r[2, 0]) / s
            qx = (r[0, 1] + r[1, 0]) / s
            qy = 0.25 * s
            qz = (r[1, 2] + r[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
            qw = (r[1, 0] - r[0, 1]) / s
            qx = (r[0, 2] + r[2, 0]) / s
            qy = (r[1, 2] + r[2, 1]) / s
            qz = 0.25 * s
    return qw, qx, qy, qz


def backproject_depth_points(
    items: list[dict],
    intr: dict,
    pixel_stride: int,
    max_depth_m: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]

    points_world = []
    colors = []
    for item in items:
        rgb = np.array(Image.open(item["rgb_path"]).convert("RGB"))
        depth = np.array(Image.open(item["depth_path"]), dtype=np.uint16)
        ys = np.arange(0, depth.shape[0], pixel_stride)
        xs = np.arange(0, depth.shape[1], pixel_stride)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        z_raw = depth[grid_y, grid_x]
        valid = (z_raw > 0) & (z_raw < INVALID_DEPTH)

        z = z_raw.astype(np.float32) / 1000.0
        if max_depth_m > 0:
            valid &= z <= max_depth_m
        if not np.any(valid):
            continue

        x = (grid_x.astype(np.float32) - cx) * z / fx
        y = (grid_y.astype(np.float32) - cy) * z / fy
        pts_cam = np.stack([x, y, z], axis=-1)[valid]
        c2w = item["c2w"]
        pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]
        cols = rgb[grid_y, grid_x][valid]

        points_world.append(pts_world.astype(np.float32))
        colors.append(cols.astype(np.uint8))

    if not points_world:
        raise RuntimeError("No valid depth-backed points were generated for the subset.")

    points_world = np.concatenate(points_world, axis=0)
    colors = np.concatenate(colors, axis=0)
    if len(points_world) > max_points:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(points_world), size=max_points, replace=False)
        points_world = points_world[sel]
        colors = colors[sel]
    return points_world, colors


def write_colmap_text(out_dir: Path, intr: dict, items: list[dict], points_world: np.ndarray, colors: np.ndarray) -> None:
    sparse_dir = out_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    camera_line = (
        f"1 PINHOLE {intr['width']} {intr['height']} "
        f"{intr['fx']} {intr['fy']} {intr['cx']} {intr['cy']}"
    )
    (sparse_dir / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: 1\n"
        f"{camera_line}\n"
    )

    with (sparse_dir / "images.txt").open("w") as f:
        f.write(
            "# Image list with two lines of data per image:\n"
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n"
            "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
            f"# Number of images: {len(items)}, mean observations per image: 0\n"
        )
        for image_id, item in enumerate(items, start=1):
            w2c = np.linalg.inv(item["c2w"])
            qw, qx, qy, qz = quat_from_rotmat(w2c[:3, :3])
            tx, ty, tz = w2c[:3, 3]
            f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {item['image_name']}\n\n")

    with (sparse_dir / "points3D.txt").open("w") as f:
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

    base_meta_path = base_dataset / "openloris_metadata.json"
    base_meta = json.loads(base_meta_path.read_text()) if base_meta_path.exists() else {}
    package_tar_value = args.package_tar or base_meta.get("package_tar")
    sequence = args.sequence or base_meta.get("sequence")
    if not package_tar_value or not sequence:
        raise ValueError("package-tar and sequence must be provided directly or via the base dataset metadata.")
    package_tar = Path(package_tar_value).resolve()

    base_items = read_images(base_dataset / "sparse" / "0" / "images.txt")
    base_frame_indices = [item["frame_idx"] for item in base_items]
    if not base_frame_indices:
        raise RuntimeError(f"No frames found in {base_dataset / 'sparse/0/images.txt'}")

    start_frame = min(base_frame_indices)
    end_frame = max(base_frame_indices)
    half_end_frame = start_frame + int(np.floor((end_frame - start_frame) * args.time_fraction))
    frame_step = args.frame_step if args.frame_step > 0 else compute_dense_step(base_frame_indices)
    selected_frame_indices = list(range(start_frame, half_end_frame + 1, frame_step))
    if selected_frame_indices[-1] != half_end_frame:
        selected_frame_indices.append(half_end_frame)

    archive_path = ensure_sequence_archive(package_tar, sequence)
    meta_dir = out_dataset.parent / f"_{sequence}_dense_half_meta"
    extract_targets(
        archive_path,
        [
            f"{sequence}/sensors.yaml",
            f"{sequence}/trans_matrix.yaml",
            f"{sequence}/color.txt",
            f"{sequence}/aligned_depth.txt",
            f"{sequence}/groundtruth.txt",
        ],
        meta_dir,
    )
    seq_root = meta_dir / sequence

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
    color_times = np.asarray([ts for ts, _ in color_entries], dtype=np.float64)
    depth_times = np.asarray([ts for ts, _ in depth_entries], dtype=np.float64)

    if selected_frame_indices[-1] >= len(color_entries):
        raise RuntimeError(
            f"Requested frame {selected_frame_indices[-1]}, but package only has {len(color_entries)} color entries."
        )

    frame_times = color_times[selected_frame_indices]
    depth_match_indices = nearest_match_indices(frame_times, depth_times, args.max_timestamp_delta)
    interp_pos = np.stack([np.interp(frame_times, gt_times, gt_positions[:, i]) for i in range(3)], axis=1)
    interp_rot = Slerp(gt_times, gt_rotations)(frame_times).as_matrix()

    stage_dir = out_dataset.parent / f"_{sequence}_dense_half_extract"
    extract_targets(
        archive_path,
        [f"{sequence}/{color_entries[idx][1]}" for idx in selected_frame_indices]
        + [f"{sequence}/{depth_entries[idx][1]}" for idx in depth_match_indices],
        stage_dir,
    )

    if out_dataset.exists():
        shutil.rmtree(out_dataset)
    images_dir = out_dataset / "images"
    depth_dir = out_dataset / "depth_train_named"
    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for order_idx, (frame_idx, depth_idx, rot, pos) in enumerate(
        zip(selected_frame_indices, depth_match_indices, interp_rot, interp_pos)
    ):
        image_name = f"frame_{frame_idx:06d}.png"
        rgb_src = stage_dir / sequence / color_entries[frame_idx][1]
        depth_src = stage_dir / sequence / depth_entries[depth_idx][1]
        rgb_dst = images_dir / image_name
        depth_dst = depth_dir / image_name
        shutil.move(rgb_src, rgb_dst)
        shutil.move(depth_src, depth_dst)

        world_from_gt = np.eye(4, dtype=np.float64)
        world_from_gt[:3, :3] = rot
        world_from_gt[:3, 3] = pos
        items.append(
            {
                "order_idx": order_idx,
                "frame_idx": frame_idx,
                "image_name": image_name,
                "rgb_path": rgb_dst,
                "depth_path": depth_dst,
                "c2w": world_from_gt @ gt_to_color,
            }
        )

    train_items = [item for idx, item in enumerate(items) if idx % args.llffhold != 0]
    points_world, colors = backproject_depth_points(
        train_items,
        intr=intr,
        pixel_stride=args.pixel_stride,
        max_depth_m=args.max_depth_m,
        max_points=args.max_points,
    )
    write_colmap_text(out_dataset, intr=intr, items=items, points_world=points_world, colors=colors)

    step_counts = Counter(np.diff(np.asarray([item["frame_idx"] for item in items], dtype=np.int64)))
    out_meta = {
        "sequence": sequence,
        "package_tar": str(package_tar),
        "gt_pose_frame": args.gt_pose_frame,
        "subset_source_dataset": str(base_dataset),
        "subset_time_fraction": args.time_fraction,
        "subset_start_frame": start_frame,
        "subset_end_frame": half_end_frame,
        "subset_frame_step": frame_step,
        "subset_frame_step_counts": {str(k): int(v) for k, v in sorted(step_counts.items())},
        "llffhold": args.llffhold,
        "num_all_images": len(items),
        "num_train_images": len(train_items),
        "num_val_images": len(items) - len(train_items),
        "num_init_points": int(len(points_world)),
        "intrinsics": intr,
        "ignore_masks_available": False,
    }
    (out_dataset / "openloris_metadata.json").write_text(json.dumps(out_meta, indent=2))

    shutil.rmtree(meta_dir, ignore_errors=True)
    shutil.rmtree(stage_dir, ignore_errors=True)

    print(
        json.dumps(
            {
                "out_dataset": str(out_dataset),
                "num_all_images": len(items),
                "num_train_images": len(train_items),
                "num_val_images": len(items) - len(train_items),
                "num_init_points": int(len(points_world)),
                "subset_start_frame": start_frame,
                "subset_end_frame": half_end_frame,
                "subset_frame_step": frame_step,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
