#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


DEBUG_SNIPPETS = [
    "print(ss_generator.no_shortcut,self.ss_cfg_strength,self.ss_cfg_strength_pm)\n",
    "        print(ss_generator.no_shortcut,self.ss_cfg_strength,self.ss_cfg_strength_pm)\n",
    "        exit()\n",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove known stray debug lines from a SAM-3D checkout.")
    parser.add_argument("--sam3d-repo", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    target = Path(args.sam3d_repo) / "sam3d_objects" / "pipeline" / "inference_pipeline.py"
    text = target.read_text()
    found = [snippet for snippet in DEBUG_SNIPPETS if snippet in text]

    if not found:
        print(f"No known debug snippet found in {target}")
        return

    if args.check_only:
        print(f"Found {len(found)} known debug snippet(s) in {target}")
        return

    for snippet in found:
        text = text.replace(snippet, "")
    target.write_text(text)
    print(f"Patched {target}: removed {len(found)} known debug snippet(s)")


if __name__ == "__main__":
    main()

