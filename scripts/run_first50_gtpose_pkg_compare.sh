#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SRC_DATASET="${SRC_DATASET:-${ROOT}/dataset_market1_1_first50_gtpose_colmapba_notrunc_v2}"
PKG_DATASET="${PKG_DATASET:-${ROOT}/$(basename "${SRC_DATASET}")_gspkg}"

RESULT_ROOT="${ROOT}/results"
RUN_TAG="${RUN_TAG:-maskignore}"
VANILLA_RESULT="${RESULT_ROOT}/market1_1_first50_gtpose_pkg_vanilla_${RUN_TAG}_long10k"
MCMC_RESULT="${RESULT_ROOT}/market1_1_first50_gtpose_pkg_mcmc_${RUN_TAG}_long10k"
COMPARE_JSON="${RESULT_ROOT}/market1_1_first50_gtpose_pkg_compare_${RUN_TAG}_10k.json"

ITERATIONS="${ITERATIONS:-10000}"
SAVE_ITERS="${SAVE_ITERS:-5000 10000}"
FORCE_RENDER="${FORCE_RENDER:-1}"

prepare_pkg_dataset() {
  export ROOT SRC_DATASET PKG_DATASET
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import shutil
from PIL import Image
import numpy as np
import os

root = Path(os.environ["ROOT"])
src = Path(os.environ["SRC_DATASET"])
out = Path(os.environ["PKG_DATASET"])
src_count = len(list((src / "images").glob("frame_*.png")))

if out.exists():
    image_count = len(list((out / "images").glob("frame_*.png")))
    mask_count = len(list((out / "images").glob("*_mask.png")))
    if image_count == src_count and mask_count == src_count and (out / "sparse/0").exists():
        print(f"Package dataset already ready at {out}")
        raise SystemExit(0)
    shutil.rmtree(out)

out.mkdir()
(out / "images").mkdir()
(out / "depths").mkdir()
shutil.copytree(src / "sparse", out / "sparse", symlinks=True)

for img in sorted((src / "images").glob("*.png")):
    dst = out / "images" / img.name
    try:
        dst.symlink_to(img.resolve())
    except OSError:
        shutil.copy2(img, dst)

    mask_src = src / "ignore_masks" / img.name
    if mask_src.exists():
        ignore_mask = np.array(Image.open(mask_src).convert("L"), dtype=np.uint8)
        valid_mask = 255 - ignore_mask
        Image.fromarray(valid_mask, mode="L").save(out / "images" / f"{img.stem}_mask.png")

for dep in sorted((src / "depth_train_named").glob("*.png")):
    dst = out / "depths" / dep.name
    try:
        dst.symlink_to(dep.resolve())
    except OSError:
        shutil.copy2(dep, dst)

print(f"Prepared package dataset at {out}")
PY
}

summarize_quality() {
 export ROOT RUN_TAG COMPARE_JSON PKG_DATASET
 "${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from pathlib import Path
 
root = Path(os.environ["ROOT"]) / "results"
run_tag = os.environ["RUN_TAG"]
compare_json = Path(os.environ["COMPARE_JSON"])

def load_quality(run_name: str):
    quality_path = root / run_name / f"ours_10000" / "quality.csv"
    rows = []
    with quality_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if k != "name" else v for k, v in row.items()})
    means = {
        "psnr": sum(r["psnr"] for r in rows) / len(rows),
        "ssim": sum(r["ssim"] for r in rows) / len(rows),
        "lpips": sum(r["lpips"] for r in rows) / len(rows),
        "num_views": len(rows),
    }
    return means

summary = {
    "dataset": os.environ["PKG_DATASET"],
    "iterations": 10000,
    "run_tag": run_tag,
    "mask_mode": "ignore",
    "vanilla": load_quality(f"market1_1_first50_gtpose_pkg_vanilla_{run_tag}_long10k"),
    "mcmc": load_quality(f"market1_1_first50_gtpose_pkg_mcmc_{run_tag}_long10k"),
}
compare_json.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
}

train_vanilla() {
  "${PYTHON_BIN}" -m gaussian_splatting.train \
    -s "${PKG_DATASET}" \
    -d "${VANILLA_RESULT}" \
    -i "${ITERATIONS}" \
    --mode densify \
    --device cuda \
    --no_depth_data \
    -o "mask_mode='ignore'" \
    --save_iterations ${SAVE_ITERS}
}

render_vanilla() {
  "${PYTHON_BIN}" -m gaussian_splatting.render \
    -s "${PKG_DATASET}" \
    -d "${VANILLA_RESULT}" \
    -i "${ITERATIONS}" \
    --device cuda
}

train_mcmc() {
  "${PYTHON_BIN}" -m gaussian_splatting_mcmc.train \
    -s "${PKG_DATASET}" \
    -d "${MCMC_RESULT}" \
    -i "${ITERATIONS}" \
    --mode base \
    --device cuda \
    --no_depth_data \
    -o "mask_mode='ignore'" \
    --save_iterations ${SAVE_ITERS}
}

render_mcmc() {
  "${PYTHON_BIN}" -m gaussian_splatting.render \
    -s "${PKG_DATASET}" \
    -d "${MCMC_RESULT}" \
    -i "${ITERATIONS}" \
    --device cuda
}

prepare_pkg_dataset

if [[ ! -f "${VANILLA_RESULT}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply" ]]; then
  train_vanilla
fi

if [[ "${FORCE_RENDER}" == "1" || ! -f "${VANILLA_RESULT}/ours_${ITERATIONS}/quality.csv" ]]; then
  render_vanilla
fi

if [[ ! -f "${MCMC_RESULT}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply" ]]; then
  train_mcmc
fi

if [[ "${FORCE_RENDER}" == "1" || ! -f "${MCMC_RESULT}/ours_${ITERATIONS}/quality.csv" ]]; then
  render_mcmc
fi

summarize_quality
