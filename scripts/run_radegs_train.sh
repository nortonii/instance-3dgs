#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 --env-file <env_file> <dataset_dir> <model_dir> [train.py args...]" >&2
  echo "   or: $0 <radegs_repo> <radegs_python> <dataset_dir> <model_dir> [train.py args...]" >&2
  exit 1
fi

if [ "${1:-}" = "--env-file" ]; then
  ENV_FILE=$2
  DATASET_DIR=$3
  MODEL_DIR=$4
  shift 4
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
  if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <radegs_repo> <radegs_python> <dataset_dir> <model_dir> [train.py args...]" >&2
    exit 1
  fi
  RADEGS_REPO=$1
  RADEGS_PYTHON=$2
  DATASET_DIR=$3
  MODEL_DIR=$4
  shift 4
fi

(
  cd "$RADEGS_REPO"
  "$RADEGS_PYTHON" train.py -s "$DATASET_DIR" -m "$MODEL_DIR" "$@"
)
