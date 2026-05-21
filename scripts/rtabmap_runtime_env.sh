#!/usr/bin/env bash
set -euo pipefail

ROOT="${RTABMAP_RUNTIME_ROOT:-/home/xuzhiyuan/work/ros-humble-rtabmap-runtime/root}"

if [[ ! -d "$ROOT" ]]; then
  echo "Runtime root not found: $ROOT" >&2
  exit 1
fi

EXTRA_LIBS="$(find "$ROOT" -type d \( -path '*/lib' -o -path '*/lib/*' -o -path '*/lib/x86_64-linux-gnu' \) | paste -sd: -)"
export PATH="$ROOT/opt/ros/humble/bin:$ROOT/usr/lib/p7zip:$ROOT/usr/bin:$ROOT/bin:${PATH:-}"
export LD_LIBRARY_PATH="$ROOT/opt/ros/humble/lib/x86_64-linux-gnu:$ROOT/opt/ros/humble/lib:$ROOT/usr/lib/x86_64-linux-gnu:$ROOT/usr/lib/x86_64-linux-gnu/blas:$ROOT/usr/lib/x86_64-linux-gnu/lapack:$ROOT/lib/x86_64-linux-gnu:$ROOT/usr/lib:$ROOT/lib:${EXTRA_LIBS}:${LD_LIBRARY_PATH:-}"
export QT_PLUGIN_PATH="$ROOT/usr/lib/x86_64-linux-gnu/qt5/plugins:${QT_PLUGIN_PATH:-}"
export XDG_DATA_DIRS="$ROOT/usr/share:$ROOT/opt/ros/humble/share:${XDG_DATA_DIRS:-/usr/local/share:/usr/share}"

exec "$@"
