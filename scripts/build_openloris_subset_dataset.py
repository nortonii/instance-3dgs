#!/usr/bin/env python3

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


INVALID_DEPTH = 65535


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a smaller OpenLORIS gsplat dataset subset from an exported dataset."
    )
    parser.add_argument("--src-dataset", required=True, type=Path)
    parser.add_argument("--out-dataset", required=True, type=Path)
    parser.add_argument("--start-image-index", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--val-count", type=int, default=5)
    parser.add_argument("--pixel-stride", type=int, default=16)
    parser.add_argument(
        "--max-depth-m",
        type=float,
        default=0.0,
        help="Maximum valid depth in meters. Use 0 or a negative value to disable far-depth truncation.",
    )
    parser.add_argument("--max-points", type=int, default=200000)
    return parser.parse_args()


def safe_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


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


def backproject_depth_points(
    rgb_dir: Path,
    depth_dir: Path,
    train_items: list[dict],
    intr: dict,
    pixel_stride: int,
    max_depth_m: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]

    points_world = []
    colors = []
    for item in train_items:
        image_name = item["image_name"]
        depth_path = depth_dir / image_name
        if not depth_path.exists():
            continue

        rgb = np.array(Image.open(rgb_dir / image_name).convert("RGB"))
        depth = np.array(Image.open(depth_path), dtype=np.uint16)
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
        c2w = np.linalg.inv(item["w2c"])
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


def write_colmap_text(out_dir: Path, camera_line: str, items: list[dict], points_world: np.ndarray, colors: np.ndarray) -> None:
    sparse_dir = out_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
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
            w2c = item["w2c"]
            r = w2c[:3, :3]
            tx, ty, tz = w2c[:3, 3]
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
            f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {item['image_name']}\n")
            f.write(f"{item['second_line']}\n")

    with (sparse_dir / "points3D.txt").open("w") as f:
        f.write(
            "# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )
        for point_id, (xyz, rgb) in enumerate(zip(points_world, colors), start=1):
            f.write(f"{point_id} {xyz[0]} {xyz[1]} {xyz[2]} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])} 0\n")


def main() -> None:
    args = parse_args()
    src = args.src_dataset
    dst = args.out_dataset

    all_items = read_images(src / "sparse" / "0" / "images.txt")
    if args.start_image_index < 0:
        raise ValueError("start-image-index must be non-negative.")
    if args.start_image_index >= len(all_items):
        raise ValueError(
            f"start-image-index {args.start_image_index} is out of range for dataset with {len(all_items)} images."
        )
    if args.start_image_index + args.max_images > len(all_items):
        raise ValueError(
            f"Requested images [{args.start_image_index}, {args.start_image_index + args.max_images}), "
            f"but source dataset only has {len(all_items)} images."
        )
    if args.val_count >= args.max_images:
        raise ValueError("val-count must be smaller than max-images.")

    selected = all_items[args.start_image_index : args.start_image_index + args.max_images]
    train_names = [item["image_name"] for item in selected[: args.max_images - args.val_count]]
    val_names = [item["image_name"] for item in selected[args.max_images - args.val_count :]]
    train_set = set(train_names)

    if dst.exists():
        shutil.rmtree(dst)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "depth_train_named").mkdir(parents=True, exist_ok=True)

    for item in selected:
        image_name = item["image_name"]
        safe_link((src / "images" / image_name).resolve(), dst / "images" / image_name)
        mask_path = src / "ignore_masks" / image_name
        if mask_path.exists():
            safe_link(mask_path.resolve(), dst / "ignore_masks" / image_name)
        if image_name in train_set:
            depth_path = src / "depth_train_named" / image_name
            if depth_path.exists():
                safe_link(depth_path.resolve(), dst / "depth_train_named" / image_name)

    camera = read_camera(src / "sparse" / "0" / "cameras.txt")
    train_items = [item for item in selected if item["image_name"] in train_set]
    points_world, colors = backproject_depth_points(
        rgb_dir=dst / "images",
        depth_dir=dst / "depth_train_named",
        train_items=train_items,
        intr=camera,
        pixel_stride=args.pixel_stride,
        max_depth_m=args.max_depth_m,
        max_points=args.max_points,
    )
    write_colmap_text(dst, camera["raw"], selected, points_world, colors)

    (dst / "splits.json").write_text(json.dumps({"train": train_names, "val": val_names}, indent=2))
    src_meta_path = src / "openloris_metadata.json"
    src_meta = json.loads(src_meta_path.read_text()) if src_meta_path.exists() else {}
    (dst / "openloris_metadata.json").write_text(
        json.dumps(
            {
                **src_meta,
                "subset_source_dataset": str(src),
                "subset_start_image_index": args.start_image_index,
                "subset_max_images": args.max_images,
                "subset_val_count": args.val_count,
                "subset_pixel_stride": args.pixel_stride,
                "subset_max_depth_m": None if args.max_depth_m <= 0 else args.max_depth_m,
                "num_all_images": len(selected),
                "num_train_images": len(train_names),
                "num_val_images": len(val_names),
                "num_init_points": int(len(points_world)),
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "out_dataset": str(dst),
                "num_all_images": len(selected),
                "num_train_images": len(train_names),
                "num_val_images": len(val_names),
                "num_init_points": int(len(points_world)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
