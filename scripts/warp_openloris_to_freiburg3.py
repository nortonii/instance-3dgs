#!/usr/bin/env python3

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np


SRC_INTR = {
    "width": 848,
    "height": 480,
    "fx": 616.802734375,
    "fy": 435.0341796875,
    "cx": 616.7510375976562,
    "cy": 242.90113830566406,
}

TGT_INTR = {
    "width": 640,
    "height": 480,
    "fx": 535.4,
    "fy": 539.2,
    "cx": 320.1,
    "cy": 247.6,
}


def compute_transform(src_intr: dict, tgt_intr: dict) -> dict:
    sx = tgt_intr["fx"] / src_intr["fx"]
    sy = tgt_intr["fy"] / src_intr["fy"]
    resized_w = int(round(src_intr["width"] * sx))
    resized_h = int(round(src_intr["height"] * sy))
    crop_x = int(round(src_intr["cx"] * sx - tgt_intr["cx"]))
    crop_y = int(round(src_intr["cy"] * sy - tgt_intr["cy"]))
    pad_right = max(0, crop_x + tgt_intr["width"] - resized_w)
    pad_bottom = max(0, crop_y + tgt_intr["height"] - resized_h)
    return {
        "sx": sx,
        "sy": sy,
        "resized_w": resized_w,
        "resized_h": resized_h,
        "crop_x": crop_x,
        "crop_y": crop_y,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
    }


def warp_image(img: np.ndarray, is_depth: bool, cfg: dict, tgt_intr: dict) -> np.ndarray:
    interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
    warped = cv2.resize(img, (cfg["resized_w"], cfg["resized_h"]), interpolation=interp)
    if cfg["pad_right"] or cfg["pad_bottom"]:
        border_value = 0 if is_depth else [0, 0, 0]
        warped = cv2.copyMakeBorder(
            warped,
            0,
            cfg["pad_bottom"],
            0,
            cfg["pad_right"],
            cv2.BORDER_CONSTANT,
            value=border_value,
        )
    x0 = cfg["crop_x"]
    y0 = cfg["crop_y"]
    return warped[y0 : y0 + tgt_intr["height"], x0 : x0 + tgt_intr["width"]]


def write_calib_yaml(path: Path, camera_name: str, intr: dict) -> None:
    path.write_text(
        "\n".join(
            [
                "%YAML:1.0",
                "---",
                f"camera_name: {camera_name}",
                f"image_width: {intr['width']}",
                f"image_height: {intr['height']}",
                "camera_matrix:",
                "   rows: 3",
                "   cols: 3",
                f"   data: [ {intr['fx']:.16e}, 0., {intr['cx']:.16e}, 0.,",
                f"       {intr['fy']:.16e}, {intr['cy']:.16e}, 0., 0., 1. ]",
                "local_transform:",
                "   rows: 3",
                "   cols: 4",
                "   data: [ 0., 0., 1., 0., -1., 0., 0., 0., 0., -1., 0., 0. ]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Warp OpenLORIS RGB-D frames into a target image geometry.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-width", type=int, default=TGT_INTR["width"])
    parser.add_argument("--target-height", type=int, default=TGT_INTR["height"])
    parser.add_argument("--target-fx", type=float, default=TGT_INTR["fx"])
    parser.add_argument("--target-fy", type=float, default=TGT_INTR["fy"])
    parser.add_argument("--target-cx", type=float, default=TGT_INTR["cx"])
    parser.add_argument("--target-cy", type=float, default=TGT_INTR["cy"])
    parser.add_argument("--camera-name")
    args = parser.parse_args()

    tgt_intr = {
        "width": args.target_width,
        "height": args.target_height,
        "fx": args.target_fx,
        "fy": args.target_fy,
        "cx": args.target_cx,
        "cy": args.target_cy,
    }
    cfg = compute_transform(SRC_INTR, tgt_intr)
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    (args.output_dir / "rgb_sync").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "depth_sync").mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.source_dir / "groundtruth.txt", args.output_dir / "groundtruth.txt")
    raw_gt = args.source_dir / "groundtruth_raw.txt"
    if raw_gt.exists():
        shutil.copy2(raw_gt, args.output_dir / "groundtruth_raw.txt")
    trans_matrix = args.source_dir / "trans_matrix.yaml"
    if trans_matrix.exists():
        shutil.copy2(trans_matrix, args.output_dir / "trans_matrix.yaml")
    shutil.copy2(args.source_dir / "frame_manifest.jsonl", args.output_dir / "frame_manifest.jsonl")

    rgb_files = sorted((args.source_dir / "rgb_sync").glob("*.png"))
    for rgb_path in rgb_files:
        depth_path = args.source_dir / "depth_sync" / rgb_path.name
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            raise RuntimeError(f"Failed to read {rgb_path.name}")
        rgb_warped = warp_image(rgb, False, cfg, tgt_intr)
        depth_warped = warp_image(depth, True, cfg, tgt_intr)
        cv2.imwrite(str(args.output_dir / "rgb_sync" / rgb_path.name), rgb_warped)
        cv2.imwrite(str(args.output_dir / "depth_sync" / rgb_path.name), depth_warped)

    camera_name = args.camera_name or f"{args.output_dir.name}_calib"
    write_calib_yaml(args.output_dir / f"{camera_name}.yaml", camera_name, tgt_intr)

    info = {
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "source_intrinsics": SRC_INTR,
        "target_intrinsics": tgt_intr,
        "transform": cfg,
        "num_frames": len(rgb_files),
        "camera_name": camera_name,
        "calib_path": str(args.output_dir / f"{camera_name}.yaml"),
    }
    (args.output_dir / "warp_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
