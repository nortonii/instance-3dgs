#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bootstrap_from_scratch.sh <workspace_dir> [--radegs-env NAME] [--sam3d-env NAME] [--torch-cuda-tag cu121] [--torch-index-url URL]

This script clones upstream repositories, creates conda/mamba environments,
installs dependencies, applies the known SAM-3D checkout patch, runs smoke
tests, and writes configs/local.env for this wrapper repo.
EOF
}

if [ "$#" -lt 1 ]; then
  usage >&2
  exit 1
fi

WORKSPACE_DIR=$1
shift

RADEGS_ENV_NAME=radegs
SAM3D_ENV_NAME=sam3d-objects
TORCH_CUDA_TAG=cu121
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

while [ "$#" -gt 0 ]; do
  case "$1" in
    --radegs-env)
      RADEGS_ENV_NAME=$2
      shift 2
      ;;
    --sam3d-env)
      SAM3D_ENV_NAME=$2
      shift 2
      ;;
    --torch-cuda-tag)
      TORCH_CUDA_TAG=$2
      shift 2
      ;;
    --torch-index-url)
      TORCH_INDEX_URL=$2
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

CONDA_BIN=$(command -v mamba || command -v conda || true)
if [ -z "$CONDA_BIN" ]; then
  echo "Could not find mamba or conda in PATH. Install Miniforge/Mambaforge first." >&2
  exit 1
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd git
require_cmd nvcc

expected_cuda=""
case "$TORCH_CUDA_TAG" in
  cu118) expected_cuda="11.8" ;;
  cu121) expected_cuda="12.1" ;;
  cu124) expected_cuda="12.4" ;;
  cu126) expected_cuda="12.6" ;;
  cu128) expected_cuda="12.8" ;;
  cu130) expected_cuda="13.0" ;;
  *)
    echo "Unsupported TORCH_CUDA_TAG: $TORCH_CUDA_TAG" >&2
    exit 1
    ;;
esac

nvcc_version=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
if [ -z "$nvcc_version" ]; then
  echo "Could not parse nvcc version" >&2
  exit 1
fi
if [ "$nvcc_version" != "$expected_cuda" ]; then
  echo "nvcc version $nvcc_version does not match torch CUDA tag $TORCH_CUDA_TAG (expected $expected_cuda)." >&2
  echo "Either install matching CUDA toolkit or rerun with --torch-cuda-tag/--torch-index-url that matches this machine." >&2
  exit 1
fi

mkdir -p "$WORKSPACE_DIR"
WORKSPACE_DIR=$(cd "$WORKSPACE_DIR" && pwd)
RADEGS_REPO="$WORKSPACE_DIR/RaDe-GS"
SAM3D_REPO="$WORKSPACE_DIR/sam-3d-objects"
ENV_FILE="$REPO_ROOT/configs/local.env"

if [ ! -d "$RADEGS_REPO/.git" ]; then
  git clone --recursive https://github.com/HKUST-SAIL/RaDe-GS.git "$RADEGS_REPO"
fi
if [ ! -d "$SAM3D_REPO/.git" ]; then
  git clone https://github.com/facebookresearch/sam-3d-objects.git "$SAM3D_REPO"
fi

"$CONDA_BIN" env list | grep -qE "^${RADEGS_ENV_NAME}[[:space:]]" || "$CONDA_BIN" env create --name "$RADEGS_ENV_NAME" -f "$REPO_ROOT/envs/radegs.yml"
"$CONDA_BIN" env list | grep -qE "^${SAM3D_ENV_NAME}[[:space:]]" || "$CONDA_BIN" env create --name "$SAM3D_ENV_NAME" -f "$SAM3D_REPO/environments/default.yml"

"$CONDA_BIN" run -n "$RADEGS_ENV_NAME" python -m pip install --upgrade pip
"$CONDA_BIN" run -n "$RADEGS_ENV_NAME" python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
"$CONDA_BIN" run -n "$RADEGS_ENV_NAME" python -m pip install -r "$RADEGS_REPO/requirements.txt"
"$CONDA_BIN" run -n "$RADEGS_ENV_NAME" python -m pip install "$RADEGS_REPO/submodules/diff-gaussian-rasterization" --no-build-isolation
"$CONDA_BIN" run -n "$RADEGS_ENV_NAME" python -m pip install "$RADEGS_REPO/submodules/warp-patch-ncc" --no-build-isolation
"$CONDA_BIN" run -n "$RADEGS_ENV_NAME" python -m pip install "$RADEGS_REPO/submodules/simple-knn" --no-build-isolation
"$CONDA_BIN" run -n "$RADEGS_ENV_NAME" python -m pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation
"$CONDA_BIN" run -n "$RADEGS_ENV_NAME" python -m pip install "$RADEGS_REPO/submodules/tetra_triangulation" --no-build-isolation

"$CONDA_BIN" run -n "$SAM3D_ENV_NAME" python -m pip install --upgrade pip
"$CONDA_BIN" run -n "$SAM3D_ENV_NAME" python -m pip install -e "$SAM3D_REPO[dev]"
"$CONDA_BIN" run -n "$SAM3D_ENV_NAME" python -m pip install -e "$SAM3D_REPO[p3d]"

torch_triplet=$("$CONDA_BIN" run -n "$SAM3D_ENV_NAME" python - <<'PY'
import torch
print(f"torch-{torch.__version__.split('+')[0]}_cu{torch.version.cuda.replace('.', '')}")
PY
)
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/${torch_triplet}.html"
"$CONDA_BIN" run -n "$SAM3D_ENV_NAME" env PIP_FIND_LINKS="$PIP_FIND_LINKS" python -m pip install -e "$SAM3D_REPO[inference]"
(cd "$SAM3D_REPO" && "$CONDA_BIN" run -n "$SAM3D_ENV_NAME" ./patching/hydra)
"$CONDA_BIN" run -n "$SAM3D_ENV_NAME" python "$SCRIPT_DIR/patch_sam3d_checkout.py" --sam3d-repo "$SAM3D_REPO"

RADEGS_PYTHON=$("$CONDA_BIN" run -n "$RADEGS_ENV_NAME" python - <<'PY'
import sys
print(sys.executable)
PY
)
SAM3D_PYTHON=$("$CONDA_BIN" run -n "$SAM3D_ENV_NAME" python - <<'PY'
import sys
print(sys.executable)
PY
)

cat >"$ENV_FILE" <<EOF
RADEGS_REPO=${RADEGS_REPO}
RADEGS_ENV_NAME=${RADEGS_ENV_NAME}
RADEGS_PYTHON=${RADEGS_PYTHON}
SAM3D_REPO=${SAM3D_REPO}
SAM3D_ENV_NAME=${SAM3D_ENV_NAME}
SAM3D_PYTHON=${SAM3D_PYTHON}
SAM3D_CONFIG=${SAM3D_REPO}/checkpoints/hf/pipeline.yaml
FFMPEG_BIN=ffmpeg
COLMAP_BIN=colmap
EOF

"$CONDA_BIN" run -n "$RADEGS_ENV_NAME" python - <<'PY'
import torch
assert torch.cuda.is_available(), "RaDe-GS env cannot see CUDA"
print("RaDe-GS smoke test ok:", torch.__version__, torch.version.cuda)
PY

"$CONDA_BIN" run -n "$SAM3D_ENV_NAME" python - <<'PY'
import torch
assert torch.cuda.is_available(), "SAM3D env cannot see CUDA"
import sam3d_objects  # noqa: F401
print("SAM3D smoke test ok:", torch.__version__, torch.version.cuda)
PY

echo "Bootstrap complete."
echo "Generated env file: $ENV_FILE"
echo "Next: run ./scripts/bootstrap_checkpoints.sh $ENV_FILE hf"
