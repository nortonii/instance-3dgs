#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

sys.path.append("notebook")
from inference import Inference  # noqa: E402


def load_label_map(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.array(Image.open(path))
    if raw.ndim == 2:
        return raw.astype(np.int32), np.ones(raw.shape, dtype=bool)
    if raw.ndim == 3 and raw.shape[2] >= 4:
        return raw[..., 0].astype(np.int32), raw[..., 3] > 0
    if raw.ndim == 3:
        return raw[..., 0].astype(np.int32), np.any(raw > 0, axis=-1)
    raise ValueError(f"Unsupported label map shape: {raw.shape}")


def build_masks(label_map: np.ndarray, valid_mask: np.ndarray, min_area: int) -> list[tuple[int, np.ndarray, int]]:
    ids, counts = np.unique(label_map[valid_mask], return_counts=True)
    masks = []
    for object_id, count in zip(ids.tolist(), counts.tolist()):
        if object_id <= 0 or count < min_area:
            continue
        masks.append((object_id, (label_map == object_id) & valid_mask, int(count)))
    return masks


def colorize_instances(label_map: np.ndarray, kept_ids: list[int], valid_mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    palette_size = max(kept_ids) + 1 if kept_ids else 1
    palette = np.random.default_rng(0).integers(30, 255, size=(palette_size, 3), dtype=np.uint8)
    for object_id in kept_ids:
        out[(label_map == object_id) & valid_mask] = palette[object_id]
    return out


def tensor_to_list(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--pointmap", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--min-area", type=int, default=4000)
    parser.add_argument("--min-valid-depth-ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-output-pt", action="store_true")
    parser.add_argument("--config", default="checkpoints/hf/pipeline.yaml")
    args = parser.parse_args()

    image = np.array(Image.open(args.image).convert("RGB"))
    label_map, label_valid = load_label_map(Path(args.label_map))
    pointmap = np.load(args.pointmap).astype(np.float32)
    valid_depth = np.isfinite(pointmap[..., 2]) & (pointmap[..., 2] > 0)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    masks = build_masks(label_map, label_valid, args.min_area)
    if args.limit and args.limit > 0:
        masks = masks[: args.limit]

    inference = Inference(args.config, compile=False)

    summary = []
    kept_ids = []
    for object_id, mask, area in masks:
        valid_ratio = float(valid_depth[mask].mean()) if mask.any() else 0.0
        if valid_ratio < args.min_valid_depth_ratio:
            continue

        kept_ids.append(object_id)
        out_dir = output_root / f"mask_{object_id:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(out_dir / "mask.png", (mask.astype(np.uint8) * 255))

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                output = inference(
                    image,
                    mask,
                    seed=args.seed,
                    stage1_only=args.stage1_only,
                    pointmap=torch.from_numpy(pointmap),
                )
        if args.save_output_pt:
            torch.save(output, out_dir / "output.pt")

        if not args.stage1_only:
            glb = output.get("glb")
            if glb is not None:
                glb.export(out_dir / "mesh.glb")
                glb.export(out_dir / "mesh.ply")
            gs = output.get("gs")
            if gs is not None:
                gs.save_ply(str(out_dir / "gaussian.ply"))

        info = {
            "object_id": int(object_id),
            "area": int(area),
            "valid_depth_ratio": valid_ratio,
            "keys": sorted(list(output.keys())) if isinstance(output, dict) else [],
        }
        if isinstance(output, dict):
            for key in ("rotation", "scale", "translation", "translation_scale", "voxel"):
                if key in output:
                    info[key] = tensor_to_list(output[key])
            if "iou" in output:
                info["iou"] = float(output["iou"])
        (out_dir / "meta.json").write_text(json.dumps(info, indent=2))
        summary.append(info)

    overlay = colorize_instances(label_map, kept_ids, label_valid)
    imageio.imwrite(output_root / "instance_overlay.png", overlay)
    imageio.imwrite(output_root / "instance_overlay_on_rgb.png", (0.55 * image + 0.45 * overlay).astype(np.uint8))
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "image": str(Path(args.image)),
                "label_map": str(Path(args.label_map)),
                "pointmap": str(Path(args.pointmap)),
                "min_area": args.min_area,
                "min_valid_depth_ratio": args.min_valid_depth_ratio,
                "num_input_instances": len(masks),
                "num_kept_instances": len(kept_ids),
                "kept_ids": kept_ids,
                "instances": summary,
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "num_input_instances": len(masks),
                "num_kept_instances": len(kept_ids),
                "output_root": str(output_root),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

