#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge official 3DGS chunk point clouds into one merged Gaussian model."
    )
    parser.add_argument("--stitched-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iteration", type=int, required=True)
    return parser.parse_args()


def read_vertices(path: Path) -> np.ndarray:
    ply = PlyData.read(path)
    return ply["vertex"].data


def main() -> None:
    args = parse_args()
    stitched_summary = json.loads(args.stitched_summary.read_text())
    chunk_summaries = [Path(p) for p in stitched_summary["chunk_summaries"]]
    if not chunk_summaries:
        raise RuntimeError("No chunk summaries found in stitched summary.")

    vertex_arrays = []
    chunk_infos = []
    for chunk_summary_path in chunk_summaries:
        chunk_summary = json.loads(chunk_summary_path.read_text())
        chunk_result_dir = Path(chunk_summary["result_dir"])
        ply_path = chunk_result_dir / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
        if not ply_path.exists():
            raise FileNotFoundError(f"Missing chunk point cloud: {ply_path}")
        vertex_arrays.append(read_vertices(ply_path))
        chunk_infos.append(
            {
                "chunk_name": chunk_summary.get("chunk_name"),
                "result_dir": str(chunk_result_dir),
                "ply_path": str(ply_path),
                "num_gaussians": int(len(vertex_arrays[-1])),
            }
        )

    merged_vertices = np.concatenate(vertex_arrays, axis=0)

    out_dir = args.output_dir
    point_cloud_dir = out_dir / "point_cloud" / f"iteration_{args.iteration}"
    point_cloud_dir.mkdir(parents=True, exist_ok=True)
    out_ply = point_cloud_dir / "point_cloud.ply"
    PlyData([PlyElement.describe(merged_vertices, "vertex")]).write(out_ply)

    cfg_args = (
        "Namespace("
        f"sh_degree=3, "
        f"source_path='{stitched_summary['source_dataset']}', "
        f"model_path='{out_dir}', "
        "images='images', depths='', resolution=-1, white_background=False, "
        "train_test_exp=False, data_device='cuda', eval=False)"
    )
    (out_dir / "cfg_args").write_text(cfg_args)

    merged_info = {
        "source_dataset": stitched_summary["source_dataset"],
        "stitched_summary": str(args.stitched_summary),
        "iteration": args.iteration,
        "num_chunks": len(chunk_infos),
        "num_gaussians": int(len(merged_vertices)),
        "chunk_infos": chunk_infos,
        "merged_ply": str(out_ply),
    }
    (out_dir / "merged_chunks.json").write_text(json.dumps(merged_info, indent=2))

    first_cameras = Path(json.loads(chunk_summaries[0].read_text())["result_dir"]) / "cameras.json"
    if first_cameras.exists():
        shutil.copy2(first_cameras, out_dir / "cameras.json")

    print(json.dumps(merged_info, indent=2))


if __name__ == "__main__":
    main()
