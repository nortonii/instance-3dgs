#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_OFFICIAL_REPO="${ROOT}/third_party/gaussian-splatting-official"
if [[ ! -d "${DEFAULT_OFFICIAL_REPO}" ]]; then
  DEFAULT_OFFICIAL_REPO="$(cd "${ROOT}/.." && pwd)/gaussian-splatting-official"
fi
OFFICIAL_REPO="${OFFICIAL_REPO:-${DEFAULT_OFFICIAL_REPO}}"
OFFICIAL_PATCH="${OFFICIAL_PATCH:-${ROOT}/patches/gaussian-splatting-official-cu124-compat.patch}"

SRC_DATASET="${SRC_DATASET:-${ROOT}/dataset_market1_1_first50_gtpose_colmapba_notrunc_v2}"
DATASET_BASENAME="${DATASET_BASENAME:-$(basename "${SRC_DATASET}")}"
OFFICIAL_DATASET="${OFFICIAL_DATASET:-${ROOT}/${DATASET_BASENAME}_official_rgba_maskignore}"

RESULT_ROOT="${ROOT}/results"
RESULT_PREFIX="${RESULT_PREFIX:-market1_1_first50_gtpose_official_vanilla}"
RUN_TAG="${RUN_TAG:-maskignore}"
RESULT_SUFFIX="${RESULT_SUFFIX:-long10k}"
SUMMARY_SUFFIX="${SUMMARY_SUFFIX:-10k}"
RESULT_DIR="${RESULT_ROOT}/${RESULT_PREFIX}_${RUN_TAG}_${RESULT_SUFFIX}"
SUMMARY_JSON="${RESULT_ROOT}/${RESULT_PREFIX}_${RUN_TAG}_${SUMMARY_SUFFIX}.json"

ITERATIONS="${ITERATIONS:-10000}"
SAVE_ITERS="${SAVE_ITERS:-5000 10000}"
FORCE_RENDER="${FORCE_RENDER:-1}"

ensure_official_patch() {
  if [[ ! -f "${OFFICIAL_PATCH}" ]]; then
    return
  fi
  if git -C "${OFFICIAL_REPO}" apply --check "${OFFICIAL_PATCH}" >/dev/null 2>&1; then
    git -C "${OFFICIAL_REPO}" apply "${OFFICIAL_PATCH}"
    echo "Applied official compatibility patch at ${OFFICIAL_REPO}"
    return
  fi
  if git -C "${OFFICIAL_REPO}" apply -R --check "${OFFICIAL_PATCH}" >/dev/null 2>&1; then
    return
  fi
  echo "Warning: could not verify compatibility patch state for ${OFFICIAL_REPO}" >&2
}

prepare_official_dataset() {
  export SRC_DATASET OFFICIAL_DATASET
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import os

src = Path(os.environ["SRC_DATASET"])
out = Path(os.environ["OFFICIAL_DATASET"])
src_count = len(list((src / "images").glob("frame_*.png")))

if out.exists():
    image_count = len(list((out / "images").glob("frame_*.png")))
    if image_count == src_count and (out / "sparse/0").exists():
        print(f"Official dataset already ready at {out}")
        raise SystemExit(0)
    shutil.rmtree(out)

out.mkdir(parents=True)
(out / "images").mkdir()
shutil.copytree(src / "sparse", out / "sparse", symlinks=True)

for img_path in sorted((src / "images").glob("*.png")):
    rgb = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    mask_path = src / "ignore_masks" / img_path.name
    if mask_path.exists():
        ignore_mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        valid_mask = 255 - ignore_mask
    else:
        valid_mask = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    masked_rgb = rgb.copy()
    masked_rgb[valid_mask == 0] = 0
    rgba = np.dstack([masked_rgb, valid_mask])
    Image.fromarray(rgba, mode="RGBA").save(out / "images" / img_path.name)

print(f"Prepared official RGBA dataset at {out}")
PY
}

train_official() {
  cd "${OFFICIAL_REPO}"
  PYTHONPATH="${OFFICIAL_REPO}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" train.py \
    -s "${OFFICIAL_DATASET}" \
    -m "${RESULT_DIR}" \
    --iterations "${ITERATIONS}" \
    --save_iterations ${SAVE_ITERS} \
    --test_iterations ${SAVE_ITERS}
}

render_official() {
  export OFFICIAL_REPO OFFICIAL_DATASET RESULT_DIR ITERATIONS SRC_DATASET
  PYTHONPATH="${OFFICIAL_REPO}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import sys
from pathlib import Path

import torch
import torchvision
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, os.environ["OFFICIAL_REPO"])

from arguments import GroupParams
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel

dataset = GroupParams()
dataset.sh_degree = 3
dataset.source_path = os.environ["OFFICIAL_DATASET"]
dataset.model_path = os.environ["RESULT_DIR"]
dataset.images = "images"
dataset.depths = ""
dataset.resolution = -1
dataset.white_background = False
dataset.train_test_exp = False
dataset.data_device = "cuda"
dataset.eval = False
src_dataset = Path(os.environ["SRC_DATASET"])

pipe = GroupParams()
pipe.convert_SHs_python = False
pipe.compute_cov3D_python = False
pipe.debug = False
pipe.antialiasing = False

iteration = int(os.environ["ITERATIONS"])
save_root = Path(dataset.model_path) / "train" / f"ours_{iteration}"
render_path = save_root / "renders"
gt_path = save_root / "gt"
render_path.mkdir(parents=True, exist_ok=True)
gt_path.mkdir(parents=True, exist_ok=True)

with torch.no_grad():
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    name_map = []
    for idx, view in enumerate(scene.getTrainCameras()):
        rendering = render(view, gaussians, pipe, background, use_trained_exp=False, separate_sh=False)["render"]
        gt = TF.to_tensor(Image.open(src_dataset / "images" / view.image_name).convert("RGB")).to(rendering.device)
        torchvision.utils.save_image(rendering, render_path / f"{idx:05d}.png")
        torchvision.utils.save_image(gt, gt_path / f"{idx:05d}.png")
        name_map.append({"index": idx, "image_name": view.image_name})

    (save_root / "image_name_map.json").write_text(json.dumps(name_map, indent=2))
    print(save_root)
PY
}

summarize_quality() {
  export RESULT_DIR ITERATIONS SUMMARY_JSON SRC_DATASET OFFICIAL_DATASET RUN_TAG
  "${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from gaussian_splatting.utils import psnr, ssim
from gaussian_splatting.utils.lpipsPyTorch import lpips

result_dir = Path(os.environ["RESULT_DIR"])
iteration = int(os.environ["ITERATIONS"])
src_dataset = Path(os.environ["SRC_DATASET"])
save_root = result_dir / "train" / f"ours_{iteration}"
render_dir = save_root / "renders"
gt_dir = save_root / "gt"
summary_json = Path(os.environ["SUMMARY_JSON"])
per_view_csv = save_root / "quality.csv"
name_map = {int(item["index"]): item["image_name"] for item in json.loads((save_root / "image_name_map.json").read_text())}

rows = []
for render_path in sorted(render_dir.glob("*.png")):
    gt_path = gt_dir / render_path.name
    render = TF.to_tensor(Image.open(render_path).convert("RGB")).unsqueeze(0).cuda()
    gt = TF.to_tensor(Image.open(gt_path).convert("RGB")).unsqueeze(0).cuda()
    image_name = name_map[int(render_path.stem)]
    ignore_mask_path = src_dataset / "ignore_masks" / image_name
    if ignore_mask_path.exists():
        ignore_mask = TF.to_tensor(Image.open(ignore_mask_path).convert("L")).unsqueeze(0).cuda()
        valid_mask = 1.0 - ignore_mask
        render = render * valid_mask
        gt = gt * valid_mask
    rows.append({
        "name": render_path.stem,
        "image_name": image_name,
        "psnr": float(psnr(render, gt).mean().item()),
        "ssim": float(ssim(render, gt).mean().item()),
        "lpips": float(lpips(render, gt).mean().item()),
    })

with per_view_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "image_name", "psnr", "ssim", "lpips"])
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "dataset": os.environ["OFFICIAL_DATASET"],
    "source_dataset": os.environ["SRC_DATASET"],
    "result_dir": str(result_dir),
    "chunk_name": os.environ.get("CHUNK_NAME"),
    "iterations": iteration,
    "run_tag": os.environ["RUN_TAG"],
    "export_masked": False,
    "metric_masked": True,
    "mask_mode": "ignore_via_premultiplied_rgba",
    "official_vanilla": {
        "psnr": sum(r["psnr"] for r in rows) / len(rows),
        "ssim": sum(r["ssim"] for r in rows) / len(rows),
        "lpips": sum(r["lpips"] for r in rows) / len(rows),
        "num_views": len(rows),
    },
}
summary_json.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
}

ensure_official_patch
prepare_official_dataset

if [[ ! -f "${RESULT_DIR}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply" ]]; then
  train_official
fi

if [[ "${FORCE_RENDER}" == "1" || ! -f "${RESULT_DIR}/train/ours_${ITERATIONS}/quality.csv" ]]; then
  render_official
  summarize_quality
fi
