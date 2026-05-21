#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from collections import deque
from pathlib import Path


INDEX_SOURCES = [
    ("jammy-main.txt", "http://archive.ubuntu.com/ubuntu/"),
    ("jammy-universe.txt", "http://archive.ubuntu.com/ubuntu/"),
    ("jammy-updates-main.txt", "http://archive.ubuntu.com/ubuntu/"),
    ("jammy-updates-universe.txt", "http://archive.ubuntu.com/ubuntu/"),
    ("ros2-jammy-main.txt", "http://packages.ros.org/ros2/ubuntu/"),
]

DEFAULT_SKIP_DEPS = {
    "adduser",
    "base-files",
    "base-passwd",
    "coreutils",
    "debconf",
    "debianutils",
    "dpkg",
    "gcc-12-base",
    "init-system-helpers",
    "libc6",
    "libcom-err2",
    "libcrypt1",
    "libgcc-s1",
    "libgssapi-krb5-2",
    "libk5crypto3",
    "libkeyutils1",
    "libkrb5-3",
    "libkrb5support0",
    "libnsl2",
    "libresolv2",
    "libselinux1",
    "libssl3",
    "libstdc++6",
    "libtirpc3",
    "libuuid1",
    "libx11-6",
    "lsb-base",
    "multiarch-support",
    "passwd",
    "perl-base",
    "tar",
    "zlib1g",
}


def parse_packages_file(path: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    current: dict[str, str] = {}
    last_key = None
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw_line:
            if current.get("Package") and current.get("Filename"):
                arch = current.get("Architecture", "")
                if arch in {"amd64", "all"}:
                    records[current["Package"]] = current
            current = {}
            last_key = None
            continue
        if raw_line.startswith((" ", "\t")) and last_key:
            current[last_key] += " " + raw_line.strip()
            continue
        key, value = raw_line.split(":", 1)
        current[key] = value.strip()
        last_key = key
    if current.get("Package") and current.get("Filename"):
        arch = current.get("Architecture", "")
        if arch in {"amd64", "all"}:
            records[current["Package"]] = current
    return records


def load_index(indexes_dir: Path) -> dict[str, tuple[dict[str, str], str]]:
    packages: dict[str, tuple[dict[str, str], str]] = {}
    for filename, base_url in INDEX_SOURCES:
        path = indexes_dir / filename
        if not path.exists():
            continue
        for pkg, record in parse_packages_file(path).items():
            packages[pkg] = (record, base_url)
    return packages


def strip_dep_name(token: str) -> str:
    token = re.sub(r"\[[^]]*\]", "", token)
    token = re.sub(r"\([^)]*\)", "", token)
    token = token.replace(":any", "").replace(":native", "")
    return token.strip()


def choose_alternative(expr: str, available: set[str], skip_deps: set[str]) -> str | None:
    for alt in expr.split("|"):
        name = strip_dep_name(alt)
        if name and name in available and name not in skip_deps:
            return name
    return None


def dependency_names(record: dict[str, str], available: set[str], skip_deps: set[str]) -> list[str]:
    deps: list[str] = []
    for field in ("Pre-Depends", "Depends"):
        if field not in record:
            continue
        for expr in record[field].split(","):
            name = choose_alternative(expr, available, skip_deps)
            if name:
                deps.append(name)
    return deps


def resolve_packages(
    initial_packages: list[str],
    package_index: dict[str, tuple[dict[str, str], str]],
    skip_deps: set[str],
    no_deps: bool,
) -> list[str]:
    available = set(package_index)
    resolved: list[str] = []
    seen: set[str] = set()
    queue = deque(initial_packages)

    while queue:
        pkg = queue.popleft()
        if pkg in seen or pkg in skip_deps:
            continue
        if pkg not in package_index:
            raise SystemExit(f"Package not found in cached indexes: {pkg}")
        seen.add(pkg)
        resolved.append(pkg)
        if no_deps:
            continue
        record, _ = package_index[pkg]
        for dep in dependency_names(record, available, skip_deps):
            if dep not in seen:
                queue.append(dep)
    return resolved


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract Jammy/ROS debs into a user-space root.")
    parser.add_argument("--indexes-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--no-deps", action="store_true")
    parser.add_argument("--skip-dep", action="append", default=[])
    parser.add_argument("packages", nargs="+")
    args = parser.parse_args()

    package_index = load_index(args.indexes_dir)
    if not package_index:
        raise SystemExit(f"No package indexes found under {args.indexes_dir}")

    skip_deps = DEFAULT_SKIP_DEPS | set(args.skip_dep)
    resolved = resolve_packages(args.packages, package_index, skip_deps, args.no_deps)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.root.mkdir(parents=True, exist_ok=True)

    for pkg in resolved:
        record, base_url = package_index[pkg]
        deb_name = Path(record["Filename"]).name
        deb_path = args.cache_dir / deb_name
        if not deb_path.exists():
            url = base_url + record["Filename"]
            print(f"Downloading {pkg} from {url}", file=sys.stderr)
            run(["curl", "-L", "--fail", "--silent", "--show-error", url, "-o", str(deb_path)])
        else:
            print(f"Using cached {deb_name}", file=sys.stderr)
        print(f"Extracting {deb_name}", file=sys.stderr)
        run(["dpkg-deb", "-x", str(deb_path), str(args.root)])

    print("\n".join(resolved))


if __name__ == "__main__":
    main()
