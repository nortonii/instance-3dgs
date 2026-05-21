#!/usr/bin/env python3

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


UINT64_MAX = "18446744073709551615"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def read_camera_line(path: Path) -> str:
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    raise RuntimeError(f"No camera entry found in {path}")


def read_images_txt(path: Path) -> list[dict]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    images = []
    for idx in range(0, len(lines), 2):
        pose = lines[idx].split()
        image_id = int(pose[0])
        qw, qx, qy, qz = map(float, pose[1:5])
        tx, ty, tz = map(float, pose[5:8])
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        w2c[:3, 3] = [tx, ty, tz]
        c2w = np.linalg.inv(w2c)
        images.append(
            {
                "image_id": image_id,
                "image_name": pose[9],
                "c2w": c2w,
            }
        )
    return images


def read_points3d_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    pts = []
    cols = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        cols.append([int(parts[4]), int(parts[5]), int(parts[6])])
    if not pts:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    return np.asarray(pts, dtype=np.float32), np.asarray(cols, dtype=np.uint8)


def write_image_list(path: Path, image_names: list[str]) -> None:
    path.write_text("\n".join(image_names) + "\n")


def count_model_images(images_txt: Path) -> int:
    return len(read_images_txt(images_txt))


def select_largest_model_txt(sparse_dir: Path, work_dir: Path, colmap_bin: str) -> Path:
    candidates = [path for path in sparse_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise RuntimeError(f"No sparse models found in {sparse_dir}")
    best_txt = None
    best_count = -1
    for candidate in sorted(candidates, key=lambda p: p.name):
        txt_dir = work_dir / f"txt_{candidate.name}"
        if txt_dir.exists():
            shutil.rmtree(txt_dir)
        txt_dir.mkdir(parents=True, exist_ok=True)
        run([colmap_bin, "model_converter", "--input_path", str(candidate), "--output_path", str(txt_dir), "--output_type", "TXT"])
        count = count_model_images(txt_dir / "images.txt")
        if count > best_count:
            best_count = count
            best_txt = txt_dir
    if best_txt is None:
        raise RuntimeError(f"Failed to select a text model from {sparse_dir}")
    return best_txt


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("Need at least three corresponding points for similarity alignment.")
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = (dst_centered.T @ src_centered) / src.shape[0]
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1
    rot = u @ s @ vt
    var_src = np.mean(np.sum(src_centered**2, axis=1))
    scale = np.trace(np.diag(d) @ s) / var_src
    trans = dst_mean - scale * (rot @ src_mean)
    return float(scale), rot, trans


def transform_c2w(c2w: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot @ c2w[:3, :3]
    out[:3, 3] = scale * (rot @ c2w[:3, 3]) + trans
    return out


def transform_points(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    return (scale * (rot @ points.T)).T + trans


def write_merged_model(
    out_dir: Path,
    camera_line: str,
    image_poses: dict[str, np.ndarray],
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
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
            f"# Number of images: {len(image_poses)}, mean observations per image: 0\n"
        )
        for image_id, image_name in enumerate(sorted(image_poses), start=1):
            w2c = np.linalg.inv(image_poses[image_name])
            qx, qy, qz, qw = Rotation.from_matrix(w2c[:3, :3]).as_quat()
            tx, ty, tz = w2c[:3, 3]
            f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {image_name}\n")
            f.write(f"0 0 {UINT64_MAX}\n")
    with (sparse_dir / "points3D.txt").open("w") as f:
        f.write(
            "# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )
        for point_id, (xyz, rgb) in enumerate(zip(points, colors), start=1):
            f.write(f"{point_id} {xyz[0]} {xyz[1]} {xyz[2]} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])} 0\n")


def build_segments(image_names: list[str], segment_size: int, overlap: int) -> list[list[str]]:
    if segment_size <= overlap:
        raise ValueError("segment_size must be larger than overlap")
    segments = []
    start = 0
    while start < len(image_names):
        end = min(start + segment_size, len(image_names))
        segments.append(image_names[start:end])
        if end == len(image_names):
            break
        start = end - overlap
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run segmented COLMAP on a long sequence and merge segments by overlap.")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--colmap-bin", default="colmap")
    parser.add_argument("--segment-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--sequential-overlap", type=int, default=20)
    parser.add_argument("--camera-line-path", type=Path, default=None)
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = dataset_dir / "images"
    image_names = sorted(path.name for path in image_dir.iterdir() if path.is_file())
    camera_line_path = args.camera_line_path or (dataset_dir / "sparse" / "0" / "cameras.txt")
    camera_line = read_camera_line(camera_line_path)
    segments = build_segments(image_names, args.segment_size, args.overlap)

    merged_poses: dict[str, np.ndarray] = {}
    merged_points = []
    merged_colors = []
    segment_summaries = []

    for segment_idx, segment_images in enumerate(segments):
        segment_dir = output_dir / f"segment_{segment_idx:03d}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        image_list_path = segment_dir / "image_list.txt"
        write_image_list(image_list_path, segment_images)
        database_path = segment_dir / "database.db"
        sparse_path = segment_dir / "sparse"
        sparse_path.mkdir(parents=True, exist_ok=True)
        use_gpu_flag = "1" if args.use_gpu else "0"

        run(
            [
                args.colmap_bin,
                "feature_extractor",
                "--database_path",
                str(database_path),
                "--image_path",
                str(image_dir),
                "--image_list_path",
                str(image_list_path),
                "--ImageReader.single_camera",
                "1",
                "--ImageReader.camera_model",
                "PINHOLE",
                "--ImageReader.camera_params",
                ",".join(camera_line.split()[4:8]),
                "--FeatureExtraction.use_gpu",
                use_gpu_flag,
            ]
        )
        run(
            [
                args.colmap_bin,
                "sequential_matcher",
                "--database_path",
                str(database_path),
                "--FeatureMatching.use_gpu",
                use_gpu_flag,
                "--SequentialMatching.overlap",
                str(args.sequential_overlap),
                "--SequentialMatching.quadratic_overlap",
                "1",
                "--SequentialMatching.loop_detection",
                "0",
            ]
        )
        run(
            [
                args.colmap_bin,
                "mapper",
                "--database_path",
                str(database_path),
                "--image_path",
                str(image_dir),
                "--output_path",
                str(sparse_path),
                "--Mapper.image_list_path",
                str(image_list_path),
                "--Mapper.multiple_models",
                "0",
                "--Mapper.ba_refine_focal_length",
                "0",
                "--Mapper.ba_refine_principal_point",
                "0",
                "--Mapper.ba_refine_extra_params",
                "0",
            ]
        )
        txt_model = select_largest_model_txt(sparse_path, segment_dir, args.colmap_bin)
        images = read_images_txt(txt_model / "images.txt")
        points, colors = read_points3d_txt(txt_model / "points3D.txt")

        if segment_idx == 0:
            scale = 1.0
            rot = np.eye(3, dtype=np.float64)
            trans = np.zeros(3, dtype=np.float64)
            common_image_names = [item["image_name"] for item in images]
            align_rmse = 0.0
        else:
            current_by_name = {item["image_name"]: item for item in images}
            common_image_names = sorted(set(current_by_name) & set(merged_poses))
            if len(common_image_names) < 3:
                raise RuntimeError(f"Segment {segment_idx} has only {len(common_image_names)} common images with previous segments.")
            src = np.stack([current_by_name[name]["c2w"][:3, 3] for name in common_image_names], axis=0)
            dst = np.stack([merged_poses[name][:3, 3] for name in common_image_names], axis=0)
            scale, rot, trans = umeyama_similarity(src, dst)
            aligned_src = transform_points(src, scale, rot, trans)
            align_rmse = float(np.sqrt(np.mean(np.sum((aligned_src - dst) ** 2, axis=1))))

        segment_pose_count = 0
        for item in images:
            global_pose = transform_c2w(item["c2w"], scale, rot, trans)
            if item["image_name"] not in merged_poses:
                merged_poses[item["image_name"]] = global_pose
                segment_pose_count += 1

        merged_points.append(transform_points(points, scale, rot, trans))
        merged_colors.append(colors)

        segment_summaries.append(
            {
                "segment_idx": segment_idx,
                "num_images_in_segment": len(segment_images),
                "num_registered_images": len(images),
                "num_points": int(len(points)),
                "num_common_images_for_alignment": len(common_image_names),
                "new_global_images_added": segment_pose_count,
                "alignment_scale": float(scale),
                "alignment_rmse": align_rmse,
                "segment_dir": str(segment_dir),
                "segment_start_image": segment_images[0],
                "segment_end_image": segment_images[-1],
            }
        )

    all_points = np.concatenate(merged_points, axis=0) if merged_points else np.empty((0, 3), dtype=np.float32)
    all_colors = np.concatenate(merged_colors, axis=0) if merged_colors else np.empty((0, 3), dtype=np.uint8)

    merged_dataset_dir = output_dir / "merged_dataset"
    shutil.copytree(dataset_dir, merged_dataset_dir, symlinks=True)
    write_merged_model(merged_dataset_dir, camera_line, merged_poses, all_points, all_colors)

    summary = {
        "dataset_dir": str(dataset_dir),
        "num_images_total": len(image_names),
        "segment_size": args.segment_size,
        "overlap": args.overlap,
        "sequential_overlap": args.sequential_overlap,
        "num_segments": len(segments),
        "num_merged_images": len(merged_poses),
        "num_merged_points": int(len(all_points)),
        "merged_dataset_dir": str(merged_dataset_dir),
        "segments": segment_summaries,
    }
    (output_dir / "segmented_colmap_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
