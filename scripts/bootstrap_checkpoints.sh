#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <env_file> <tag>" >&2
  echo "Example: $0 configs/local.env hf" >&2
  exit 1
fi

ENV_FILE=$1
TAG=$2

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

: "${SAM3D_REPO:?Missing SAM3D_REPO in env file}"
: "${SAM3D_ENV_NAME:?Missing SAM3D_ENV_NAME in env file}"

CONDA_BIN=${CONDA_BIN:-$(command -v mamba || command -v conda || true)}
if [ -z "$CONDA_BIN" ]; then
  echo "Could not find mamba or conda in PATH" >&2
  exit 1
fi

"$CONDA_BIN" run -n "$SAM3D_ENV_NAME" python -m pip install 'huggingface-hub[cli]<1.0'
"$CONDA_BIN" run -n "$SAM3D_ENV_NAME" hf download \
  --repo-type model \
  --local-dir "$SAM3D_REPO/checkpoints/${TAG}-download" \
  --max-workers 1 \
  facebook/sam-3d-objects

rm -rf "$SAM3D_REPO/checkpoints/${TAG}"
mv "$SAM3D_REPO/checkpoints/${TAG}-download/checkpoints" "$SAM3D_REPO/checkpoints/${TAG}"
rm -rf "$SAM3D_REPO/checkpoints/${TAG}-download"

python_path=$("$CONDA_BIN" run -n "$SAM3D_ENV_NAME" python - <<'PY'
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
SAM3D_PYTHON=${python_path}
SAM3D_CONFIG=${SAM3D_REPO}/checkpoints/${TAG}/pipeline.yaml
FFMPEG_BIN=${FFMPEG_BIN:-ffmpeg}
COLMAP_BIN=${COLMAP_BIN:-colmap}
EOF

echo "Downloaded checkpoints to $SAM3D_REPO/checkpoints/${TAG}"
echo "Updated $ENV_FILE with SAM3D_CONFIG"

