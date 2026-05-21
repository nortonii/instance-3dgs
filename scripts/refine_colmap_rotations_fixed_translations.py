#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


INVALID_POINT3D_IDS = {-1, 18446744073709551615}


def read_camera(path: Path) -> dict:
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        model = parts[1]
        if model != "PINHOLE":
            raise ValueError(f"Only PINHOLE cameras are supported, got {model}.")
        return {
            "camera_id": int(parts[0]),
            "model": model,
            "width": int(parts[2]),
            "height": int(parts[3]),
            "fx": float(parts[4]),
            "fy": float(parts[5]),
            "cx": float(parts[6]),
            "cy": float(parts[7]),
        }
    raise RuntimeError(f"No camera entry found in {path}")


def quat_wxyz_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def rotmat_to_quat_wxyz(rot: np.ndarray) -> tuple[float, float, float, float]:
    qx, qy, qz, qw = Rotation.from_matrix(rot).as_quat()
    return qw, qx, qy, qz


def read_points3d(path: Path) -> dict[int, np.ndarray]:
    points = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        points[int(parts[0])] = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
    return points


def parse_points2d(line: str) -> list[tuple[float, float, int]]:
    parts = line.split()
    if len(parts) % 3 != 0:
        raise ValueError(f"Invalid POINTS2D line: {line}")
    out = []
    for idx in range(0, len(parts), 3):
        x = float(parts[idx])
        y = float(parts[idx + 1])
        point3d_id = int(parts[idx + 2])
        out.append((x, y, point3d_id))
    return out


def read_images(path: Path) -> tuple[list[str], list[dict]]:
    header_lines = []
    data_lines = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            header_lines.append(line)
        else:
            data_lines.append(line.strip())

    images = []
    for idx in range(0, len(data_lines), 2):
        pose_parts = data_lines[idx].split()
        image_id = int(pose_parts[0])
        qw, qx, qy, qz = map(float, pose_parts[1:5])
        tx, ty, tz = map(float, pose_parts[5:8])
        images.append(
            {
                "image_id": image_id,
                "pose_line": data_lines[idx],
                "points2d_line": data_lines[idx + 1],
                "image_name": pose_parts[9],
                "camera_id": int(pose_parts[8]),
                "R": quat_wxyz_to_rotmat(qw, qx, qy, qz),
                "t": np.asarray([tx, ty, tz], dtype=np.float64),
            }
        )
    return header_lines, images


def read_frames(path: Path) -> tuple[list[str], dict[int, dict]]:
    if not path.exists():
        return [], {}
    header_lines = []
    frames = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            header_lines.append(line)
            continue
        parts = line.split()
        frame_id = int(parts[0])
        frames[frame_id] = {
            "frame_id": frame_id,
            "rig_id": int(parts[1]),
            "pose_prefix": parts[:2],
            "pose_suffix": parts[9:],
        }
    return header_lines, frames


def project_points(points_world: np.ndarray, rotation_wc: np.ndarray, center_world: np.ndarray, camera: dict) -> np.ndarray:
    points_cam = (rotation_wc @ (points_world - center_world).T).T
    z = points_cam[:, 2]
    if np.any(z <= 1e-9):
        raise ValueError("Points behind camera during projection.")
    x = camera["fx"] * (points_cam[:, 0] / z) + camera["cx"]
    y = camera["fy"] * (points_cam[:, 1] / z) + camera["cy"]
    return np.stack([x, y], axis=1)


def residuals_for_rotation(rotvec: np.ndarray, initial_rot: np.ndarray, center_world: np.ndarray, camera: dict, points_world: np.ndarray, points2d: np.ndarray) -> np.ndarray:
    delta_rot = Rotation.from_rotvec(rotvec).as_matrix()
    rotation_wc = delta_rot @ initial_rot
    projected = project_points(points_world, rotation_wc, center_world, camera)
    return (projected - points2d).reshape(-1)


def refine_image_rotation(image: dict, camera: dict, points3d: dict[int, np.ndarray]) -> dict:
    correspondences = []
    for x, y, point3d_id in parse_points2d(image["points2d_line"]):
        if point3d_id in INVALID_POINT3D_IDS or point3d_id not in points3d:
            continue
        correspondences.append((points3d[point3d_id], np.asarray([x, y], dtype=np.float64)))

    if len(correspondences) < 12:
        return {
            "image_id": image["image_id"],
            "image_name": image["image_name"],
            "num_correspondences": len(correspondences),
            "optimized": False,
        }

    points_world = np.stack([item[0] for item in correspondences], axis=0)
    points2d = np.stack([item[1] for item in correspondences], axis=0)
    center_world = -(image["R"].T @ image["t"])

    initial_err = residuals_for_rotation(np.zeros(3), image["R"], center_world, camera, points_world, points2d)
    result = least_squares(
        residuals_for_rotation,
        x0=np.zeros(3, dtype=np.float64),
        args=(image["R"], center_world, camera, points_world, points2d),
        loss="huber",
        f_scale=2.0,
        max_nfev=200,
    )

    refined_rot = Rotation.from_rotvec(result.x).as_matrix() @ image["R"]
    refined_t = -refined_rot @ center_world
    refined_err = residuals_for_rotation(result.x, image["R"], center_world, camera, points_world, points2d)

    image["R"] = refined_rot
    image["t"] = refined_t
    return {
        "image_id": image["image_id"],
        "image_name": image["image_name"],
        "num_correspondences": len(correspondences),
        "optimized": True,
        "initial_rmse": float(np.sqrt(np.mean(initial_err**2))),
        "final_rmse": float(np.sqrt(np.mean(refined_err**2))),
        "nfev": int(result.nfev),
        "success": bool(result.success),
    }


def write_images(path: Path, header_lines: list[str], images: list[dict]) -> None:
    with path.open("w") as f:
        for line in header_lines:
            f.write(f"{line}\n")
        for image in images:
            qw, qx, qy, qz = rotmat_to_quat_wxyz(image["R"])
            tx, ty, tz = image["t"]
            f.write(
                f"{image['image_id']} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {image['camera_id']} {image['image_name']}\n"
            )
            f.write(f"{image['points2d_line']}\n")


def write_frames(path: Path, header_lines: list[str], frames: dict[int, dict], images: list[dict]) -> None:
    if not frames:
        return
    image_by_id = {image["image_id"]: image for image in images}
    with path.open("w") as f:
        for line in header_lines:
            f.write(f"{line}\n")
        for frame_id in sorted(frames):
            frame = frames[frame_id]
            image = image_by_id[frame_id]
            qw, qx, qy, qz = rotmat_to_quat_wxyz(image["R"])
            tx, ty, tz = image["t"]
            suffix = " ".join(frame["pose_suffix"])
            f.write(f"{frame_id} {frame['rig_id']} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {suffix}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine COLMAP image rotations while keeping camera translations fixed.")
    parser.add_argument("--input-model", required=True, type=Path)
    parser.add_argument("--output-model", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()

    input_model = args.input_model
    output_model = args.output_model
    output_model.mkdir(parents=True, exist_ok=True)

    camera = read_camera(input_model / "cameras.txt")
    points3d = read_points3d(input_model / "points3D.txt")
    image_header, images = read_images(input_model / "images.txt")
    frame_header, frames = read_frames(input_model / "frames.txt")

    summary = []
    for image in images:
        summary.append(refine_image_rotation(image, camera, points3d))

    write_images(output_model / "images.txt", image_header, images)
    write_frames(output_model / "frames.txt", frame_header, frames, images)
    for name in ("cameras.txt", "points3D.txt", "rigs.txt"):
        src = input_model / name
        if src.exists():
            (output_model / name).write_text(src.read_text())

    optimized = [item for item in summary if item["optimized"]]
    report = {
        "num_images": len(images),
        "num_optimized_images": len(optimized),
        "mean_initial_rmse": float(np.mean([item["initial_rmse"] for item in optimized])) if optimized else None,
        "mean_final_rmse": float(np.mean([item["final_rmse"] for item in optimized])) if optimized else None,
        "images": summary,
    }
    summary_path = args.summary_json or (output_model / "rotation_refine_summary.json")
    summary_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
