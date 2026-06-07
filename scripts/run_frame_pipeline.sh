#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 6 ]; then
  echo "Usage: $0 <env_file> <radegs_model_dir> <image_path> <frame_name> <label_map> <output_root> [extra sam3d args...]" >&2
  exit 1
fi

ENV_FILE=$1
MODEL_DIR=$2
IMAGE_PATH=$3
FRAME_NAME=$4
LABEL_MAP=$5
OUTPUT_ROOT=$6
shift 6

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

: "${RADEGS_REPO:?Missing RADEGS_REPO in env file}"
: "${RADEGS_PYTHON:?Missing RADEGS_PYTHON in env file}"
: "${SAM3D_REPO:?Missing SAM3D_REPO in env file}"
: "${SAM3D_PYTHON:?Missing SAM3D_PYTHON in env file}"
: "${SAM3D_CONFIG:=}"

SAM3D_CONFIG_ARGS=()
if [ -n "$SAM3D_CONFIG" ]; then
  SAM3D_CONFIG_ARGS=(--config "$SAM3D_CONFIG")
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEPTH_DIR="$OUTPUT_ROOT/depth_prior"
SPLIT_DIR="$OUTPUT_ROOT/split_masks"
STAGE1_DIR="$OUTPUT_ROOT/sam3d_stage1"
MESH_DIR="$OUTPUT_ROOT/sam3d_mesh"
SCENE_PREFIX="$OUTPUT_ROOT/${FRAME_NAME}_sam3d_mesh_scene"

mkdir -p "$OUTPUT_ROOT"

(
  cd "$RADEGS_REPO"
  "$RADEGS_PYTHON" "$SCRIPT_DIR/render_radegs_depth_prior.py" \
    -m "$MODEL_DIR" \
    --frame "$FRAME_NAME" \
    --out-dir "$DEPTH_DIR"
)

"$SAM3D_PYTHON" "$SCRIPT_DIR/split_instance_label_map.py" \
  --label-map "$LABEL_MAP" \
  --output-dir "$SPLIT_DIR"

(
  cd "$SAM3D_REPO"
  "$SAM3D_PYTHON" "$SCRIPT_DIR/run_sam3dobjects_with_pointmap.py" \
    --image "$IMAGE_PATH" \
    --label-map "$LABEL_MAP" \
    --pointmap "$DEPTH_DIR/${FRAME_NAME}_pointmap.npy" \
    --output-root "$STAGE1_DIR" \
    --stage1-only \
    --save-output-pt \
    "${SAM3D_CONFIG_ARGS[@]}" \
    "$@"
)

(
  cd "$SAM3D_REPO"
  "$SAM3D_PYTHON" "$SCRIPT_DIR/run_sam3dobjects_with_pointmap.py" \
    --image "$IMAGE_PATH" \
    --label-map "$LABEL_MAP" \
    --pointmap "$DEPTH_DIR/${FRAME_NAME}_pointmap.npy" \
    --output-root "$MESH_DIR" \
    --save-output-pt \
    "${SAM3D_CONFIG_ARGS[@]}" \
    "$@"
)

if ! find "$MESH_DIR" -path '*/mesh.ply' -type f | grep -q .; then
  echo "SAM-3D full decode did not produce any mesh.ply under $MESH_DIR" >&2
  exit 1
fi

"$SAM3D_PYTHON" "$SCRIPT_DIR/assemble_sam3d_scene.py" \
  --mesh-root "$MESH_DIR" \
  --stage1-root "$STAGE1_DIR" \
  --camera-json "$DEPTH_DIR/${FRAME_NAME}_camera.json" \
  --output-prefix "$SCENE_PREFIX"

echo "Scene outputs:"
echo "  ${SCENE_PREFIX}.ply"
echo "  ${SCENE_PREFIX}.glb"
echo "  ${SCENE_PREFIX}_summary.json"
