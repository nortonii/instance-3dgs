#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

try:
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    if q.size != 4:
        raise ValueError(f"Expected quaternion of length 4, got shape {q.shape}")
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def maybe_from_stage1(stage1_root: Path | None, instance_dir: str, key: str):
    if stage1_root is None:
        return None
    output_path = stage1_root / instance_dir / "output.pt"
    if not output_path.exists() or torch is None:
        return None
    payload = torch.load(output_path, map_location="cpu")
    if isinstance(payload, dict) and key in payload:
        value = payload[key]
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        return value
    return None


def load_pose(mesh_dir: Path, stage1_root: Path | None) -> tuple[np.ndarray, np.ndarray, float]:
    meta = json.loads((mesh_dir / "meta.json").read_text())
    rotation = meta.get("rotation")
    translation = meta.get("translation")
    scale = meta.get("scale")

    if rotation is None:
        rotation = maybe_from_stage1(stage1_root, mesh_dir.name, "rotation")
    if translation is None:
        translation = maybe_from_stage1(stage1_root, mesh_dir.name, "translation")
    if scale is None:
        scale = maybe_from_stage1(stage1_root, mesh_dir.name, "scale")

    if rotation is None or translation is None or scale is None:
        raise KeyError(f"Missing pose fields for {mesh_dir}")

    rotation = np.asarray(rotation, dtype=np.float64).reshape(-1)
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    scale = float(np.asarray(scale, dtype=np.float64).reshape(-1)[0])
    return rotation, translation, scale


def apply_object_pose(vertices: np.ndarray, rotation: np.ndarray, translation: np.ndarray, scale: float) -> np.ndarray:
    center = vertices.mean(axis=0, keepdims=True)
    rot = quaternion_to_matrix(rotation)
    posed = (vertices - center) * scale
    posed = posed @ rot.T
    posed = posed + center + translation[None, :]
    return posed


def pointmap_to_radegs_camera(vertices: np.ndarray) -> np.ndarray:
    converted = vertices.copy()
    converted[:, 0] *= -1.0
    converted[:, 1] *= -1.0
    return converted


def transform_to_world(vertices: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    homog = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=vertices.dtype)], axis=1)
    return (homog @ c2w.T)[:, :3]


def simplify_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if target_faces <= 0 or len(mesh.faces) <= target_faces or o3d is None:
        return mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    simplified = o3d_mesh.simplify_quadric_decimation(target_faces)
    return trimesh.Trimesh(
        vertices=np.asarray(simplified.vertices),
        faces=np.asarray(simplified.triangles),
        process=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge posed SAM-3D instance meshes into one scene mesh.")
    parser.add_argument("--mesh-root", required=True)
    parser.add_argument("--stage1-root", default="")
    parser.add_argument("--camera-json", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--world-space", action="store_true")
    parser.add_argument("--target-faces", type=int, default=0)
    args = parser.parse_args()

    mesh_root = Path(args.mesh_root)
    stage1_root = Path(args.stage1_root) if args.stage1_root else None
    camera_meta = json.loads(Path(args.camera_json).read_text())
    c2w = np.asarray(camera_meta.get("c2w"), dtype=np.float64) if "c2w" in camera_meta else None

    meshes = []
    instances = []
    for mesh_dir in sorted(mesh_root.glob("mask_*")):
        mesh_path = mesh_dir / "mesh.ply"
        if not mesh_path.exists():
            continue

        mesh = trimesh.load(mesh_path, force="mesh")
        if mesh.is_empty:
            continue

        rotation, translation, scale = load_pose(mesh_dir, stage1_root)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        vertices = apply_object_pose(vertices, rotation, translation, scale)
        if args.world_space:
            if c2w is None:
                raise KeyError("camera json does not contain c2w")
            vertices = pointmap_to_radegs_camera(vertices)
            vertices = transform_to_world(vertices, c2w)

        posed_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=np.asarray(mesh.faces),
            visual=mesh.visual,
            process=False,
        )
        meshes.append(posed_mesh)
        instances.append(
            {
                "instance_dir": mesh_dir.name,
                "mesh_path": str(mesh_path),
                "num_vertices": int(len(posed_mesh.vertices)),
                "num_faces": int(len(posed_mesh.faces)),
                "scale": scale,
                "translation": translation.tolist(),
                "rotation": rotation.tolist(),
            }
        )

    if not meshes:
        raise RuntimeError(f"No meshes found under {mesh_root}")

    scene_mesh = trimesh.util.concatenate(meshes)
    scene_mesh = simplify_mesh(scene_mesh, args.target_faces)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    ply_path = output_prefix.with_suffix(".ply")
    glb_path = output_prefix.with_suffix(".glb")
    summary_path = output_prefix.parent / f"{output_prefix.name}_summary.json"

    scene_mesh.export(ply_path)
    scene_mesh.export(glb_path)

    summary = {
        "mesh_root": str(mesh_root),
        "stage1_root": str(stage1_root) if stage1_root else None,
        "camera_json": str(args.camera_json),
        "coordinate_system": "radegs_world" if args.world_space else "sam3d_pointmap_camera",
        "num_instances": len(instances),
        "num_vertices": int(len(scene_mesh.vertices)),
        "num_faces": int(len(scene_mesh.faces)),
        "bounds": np.asarray(scene_mesh.bounds).tolist(),
        "target_faces": args.target_faces,
        "instances": instances,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"ply": str(ply_path), "glb": str(glb_path), "summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
