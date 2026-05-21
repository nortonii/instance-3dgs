#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SRC_DATASET="${SRC_DATASET:-${ROOT}/dataset_market1_1_fullspan}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
VAL_COUNT="${VAL_COUNT:-10}"
CHUNK_STRIDE="${CHUNK_STRIDE:-${CHUNK_SIZE}}"
ITERATIONS="${ITERATIONS:-40000}"
SAVE_ITERS="${SAVE_ITERS:-10000 20000 40000}"
RUN_TAG="${RUN_TAG:-maskignore}"
RESULT_PREFIX="${RESULT_PREFIX:-market1_1_chunked_official_vanilla}"
RESULT_SUFFIX="${RESULT_SUFFIX:-long40k}"
SUMMARY_SUFFIX="${SUMMARY_SUFFIX:-40k}"
MAX_CHUNKS="${MAX_CHUNKS:-0}"
FORCE_RENDER="${FORCE_RENDER:-1}"

RESULT_ROOT="${ROOT}/results"
CHUNK_DATA_ROOT="${ROOT}/chunk_datasets/${RESULT_PREFIX}_${RUN_TAG}_${RESULT_SUFFIX}"
STITCH_ROOT="${RESULT_ROOT}/${RESULT_PREFIX}_${RUN_TAG}_${RESULT_SUFFIX}_stitched"
SUMMARY_JSON="${RESULT_ROOT}/${RESULT_PREFIX}_${RUN_TAG}_${SUMMARY_SUFFIX}_stitched.json"

mkdir -p "${RESULT_ROOT}" "${CHUNK_DATA_ROOT}"

TOTAL_IMAGES="$("${PYTHON_BIN}" - <<'PY' "${SRC_DATASET}"
from pathlib import Path
import sys
print(len(list((Path(sys.argv[1]) / "images").glob("frame_*.png"))))
PY
)"

chunk_idx=0
start_idx=0
chunk_summary_files=()

while (( start_idx + CHUNK_SIZE <= TOTAL_IMAGES )); do
  if (( MAX_CHUNKS > 0 && chunk_idx >= MAX_CHUNKS )); then
    break
  fi

  chunk_name=$(printf "chunk%03d_%05d_%05d" "${chunk_idx}" "${start_idx}" "$((start_idx + CHUNK_SIZE - 1))")
  chunk_dataset="${CHUNK_DATA_ROOT}/${chunk_name}"
  chunk_result_prefix="${RESULT_PREFIX}_${chunk_name}"
  chunk_summary="${RESULT_ROOT}/${chunk_result_prefix}_${RUN_TAG}_${SUMMARY_SUFFIX}.json"

  echo "[chunk ${chunk_idx}] building subset ${chunk_name}"
  "${PYTHON_BIN}" "${ROOT}/scripts/build_openloris_subset_dataset.py" \
    --src-dataset "${SRC_DATASET}" \
    --out-dataset "${chunk_dataset}" \
    --start-image-index "${start_idx}" \
    --max-images "${CHUNK_SIZE}" \
    --val-count "${VAL_COUNT}"

  echo "[chunk ${chunk_idx}] training ${chunk_name}"
  ROOT="${ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  SRC_DATASET="${chunk_dataset}" \
  OFFICIAL_DATASET="${chunk_dataset}_official_rgba_maskignore" \
  CHUNK_NAME="${chunk_name}" \
  RESULT_PREFIX="${chunk_result_prefix}" \
  RUN_TAG="${RUN_TAG}" \
  ITERATIONS="${ITERATIONS}" \
  SAVE_ITERS="${SAVE_ITERS}" \
  RESULT_SUFFIX="${RESULT_SUFFIX}" \
  SUMMARY_SUFFIX="${SUMMARY_SUFFIX}" \
  FORCE_RENDER="${FORCE_RENDER}" \
  "${ROOT}/scripts/run_first50_gtpose_official_vanilla.sh"

  chunk_summary_files+=("${chunk_summary}")
  start_idx=$((start_idx + CHUNK_STRIDE))
  chunk_idx=$((chunk_idx + 1))
done

export ROOT RESULT_ROOT RESULT_PREFIX RUN_TAG RESULT_SUFFIX ITERATIONS SUMMARY_JSON STITCH_ROOT SRC_DATASET CHUNK_SIZE CHUNK_STRIDE VAL_COUNT
"${PYTHON_BIN}" - <<'PY' "${chunk_summary_files[@]}"
import csv
import json
import os
import shutil
import sys
from pathlib import Path

root = Path(os.environ["ROOT"])
result_root = Path(os.environ["RESULT_ROOT"])
result_prefix = os.environ["RESULT_PREFIX"]
run_tag = os.environ["RUN_TAG"]
result_suffix = os.environ["RESULT_SUFFIX"]
iterations = int(os.environ["ITERATIONS"])
summary_json = Path(os.environ["SUMMARY_JSON"])
stitch_root = Path(os.environ["STITCH_ROOT"])

summary_paths = [Path(p) for p in sys.argv[1:]]
if not summary_paths:
    raise SystemExit("No chunk summaries were produced.")

if stitch_root.exists():
    shutil.rmtree(stitch_root)
(stitch_root / "renders").mkdir(parents=True)
(stitch_root / "gt").mkdir(parents=True)

chunk_summaries = []
stitched_rows = []
stitched_name_map = []
global_idx = 0

for summary_path in summary_paths:
    chunk_summary = json.loads(summary_path.read_text())
    chunk_summaries.append(chunk_summary)
    chunk_name = chunk_summary["chunk_name"]
    result_dir = Path(chunk_summary["result_dir"])
    chunk_root = result_dir / "train" / f"ours_{iterations}"
    render_dir = chunk_root / "renders"
    gt_dir = chunk_root / "gt"
    quality_csv = chunk_root / "quality.csv"
    name_map = {row["name"]: row["image_name"] for row in csv.DictReader(quality_csv.open())}

    for render_path in sorted(render_dir.glob("*.png")):
        stem = render_path.stem
        out_name = f"{global_idx:05d}.png"
        shutil.copy2(render_path, stitch_root / "renders" / out_name)
        shutil.copy2(gt_dir / render_path.name, stitch_root / "gt" / out_name)
        stitched_name_map.append(
            {
                "index": global_idx,
                "chunk_name": chunk_name,
                "chunk_local_name": stem,
                "image_name": name_map[stem],
            }
        )
        global_idx += 1

    for row in csv.DictReader(quality_csv.open()):
        stitched_rows.append(
            {
                "chunk_name": chunk_name,
                "name": row["name"],
                "image_name": row["image_name"],
                "psnr": float(row["psnr"]),
                "ssim": float(row["ssim"]),
                "lpips": float(row["lpips"]),
            }
        )

with (stitch_root / "quality.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["chunk_name", "name", "image_name", "psnr", "ssim", "lpips"])
    writer.writeheader()
    writer.writerows(stitched_rows)

(stitch_root / "image_name_map.json").write_text(json.dumps(stitched_name_map, indent=2))

summary = {
    "source_dataset": os.environ["SRC_DATASET"] if "SRC_DATASET" in os.environ else None,
    "chunk_size": int(os.environ.get("CHUNK_SIZE", "0") or 0),
    "chunk_stride": int(os.environ.get("CHUNK_STRIDE", "0") or 0),
    "val_count": int(os.environ.get("VAL_COUNT", "0") or 0),
    "iterations": iterations,
    "run_tag": run_tag,
    "result_prefix": result_prefix,
    "result_suffix": result_suffix,
    "export_masked": False,
    "metric_masked": True,
    "num_chunks": len(chunk_summaries),
    "num_views": len(stitched_rows),
    "psnr": sum(r["psnr"] for r in stitched_rows) / len(stitched_rows),
    "ssim": sum(r["ssim"] for r in stitched_rows) / len(stitched_rows),
    "lpips": sum(r["lpips"] for r in stitched_rows) / len(stitched_rows),
    "chunk_summaries": [str(p) for p in summary_paths],
    "stitched_dir": str(stitch_root),
}
summary_json.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
