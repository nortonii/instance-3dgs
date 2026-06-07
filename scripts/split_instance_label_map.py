#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def load_label_map(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.array(Image.open(path))
    if raw.ndim == 2:
        return raw.astype(np.int32), np.ones(raw.shape, dtype=bool)
    if raw.ndim == 3 and raw.shape[2] >= 4:
        return raw[..., 0].astype(np.int32), raw[..., 3] > 0
    if raw.ndim == 3:
        return raw[..., 0].astype(np.int32), np.any(raw > 0, axis=-1)
    raise ValueError(f"Unsupported label map shape: {raw.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split an object-id label map into binary masks.")
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    label_map_path = Path(args.label_map)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label, valid = load_label_map(label_map_path)
    ids, counts = np.unique(label[valid], return_counts=True)

    entries = []
    for object_id, count in zip(ids.tolist(), counts.tolist()):
        if object_id <= 0 or count < args.min_area:
            continue
        mask = ((label == object_id) & valid).astype(np.uint8) * 255
        mask_dir = output_dir / f"mask_{object_id:03d}"
        mask_dir.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(mask_dir / "mask.png", mask)
        entries.append({"object_id": int(object_id), "area": int(count), "mask_dir": str(mask_dir)})
        if args.limit and len(entries) >= args.limit:
            break

    summary = {
        "label_map": str(label_map_path),
        "num_masks": len(entries),
        "min_area": args.min_area,
        "instances": entries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"output_dir": str(output_dir), "num_masks": len(entries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

