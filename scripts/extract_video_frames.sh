#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <video> <output_dir> <fps> [ffmpeg extra args...]" >&2
  exit 1
fi

VIDEO=$1
OUTPUT_DIR=$2
FPS=$3
shift 3

mkdir -p "$OUTPUT_DIR"
FFMPEG_BIN=${FFMPEG_BIN:-ffmpeg}

"$FFMPEG_BIN" -y -i "$VIDEO" -vf "fps=${FPS}" "$@" "$OUTPUT_DIR/%05d.png"

