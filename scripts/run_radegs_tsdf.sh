#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 --env-file <env_file> <scalable|stream> <model_dir> <mesh_name> [script args...]" >&2
  echo "   or: $0 <scalable|stream> <radegs_repo> <radegs_python> <model_dir> <mesh_name> [script args...]" >&2
  exit 1
fi

if [ "${1:-}" = "--env-file" ]; then
  ENV_FILE=$2
  MODE=$3
  MODEL_DIR=$4
  MESH_NAME=$5
  shift 5
  if [ ! -f "$ENV_FILE" ]; then
    echo "Missing env file: $ENV_FILE" >&2
    exit 1
  fi
  set -a
  source "$ENV_FILE"
  set +a
  : "${RADEGS_REPO:?Missing RADEGS_REPO in env file}"
  : "${RADEGS_PYTHON:?Missing RADEGS_PYTHON in env file}"
else
  if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <scalable|stream> <radegs_repo> <radegs_python> <model_dir> <mesh_name> [script args...]" >&2
    exit 1
  fi
  MODE=$1
  RADEGS_REPO=$2
  RADEGS_PYTHON=$3
  MODEL_DIR=$4
  MESH_NAME=$5
  shift 5
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

case "$MODE" in
  scalable)
    SCRIPT_PATH="$SCRIPT_DIR/extract_radegs_scalable_tsdf.py"
    ;;
  stream)
    SCRIPT_PATH="$SCRIPT_DIR/extract_radegs_stream_tsdf.py"
    ;;
  *)
    echo "Unknown TSDF mode: $MODE" >&2
    exit 1
    ;;
esac

(
  cd "$RADEGS_REPO"
  "$RADEGS_PYTHON" "$SCRIPT_PATH" -m "$MODEL_DIR" --name "$MESH_NAME" "$@"
)
