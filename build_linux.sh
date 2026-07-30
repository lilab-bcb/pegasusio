#!/usr/bin/env bash
set -euo pipefail

readonly project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly cibuildwheel_version="${CIBUILDWHEEL_VERSION:-4.1.1}"

# Match the Python versions supported by this project. cibuildwheel builds in
# manylinux containers and repairs each wheel with auditwheel automatically.
export CIBW_BUILD="${CIBW_BUILD:-cp311-manylinux_x86_64 cp312-manylinux_x86_64 cp313-manylinux_x86_64 cp314-manylinux_x86_64}"

cd "$project_dir"

if command -v pipx >/dev/null 2>&1; then
    runner=(pipx run "cibuildwheel==$cibuildwheel_version")
elif command -v uvx >/dev/null 2>&1; then
    runner=(uvx "cibuildwheel==$cibuildwheel_version")
elif python3 -c \
    'import importlib.metadata, sys; sys.exit(importlib.metadata.version("cibuildwheel") != sys.argv[1])' \
    "$cibuildwheel_version" >/dev/null 2>&1; then
    runner=(python3 -m cibuildwheel)
else
    echo "error: install pipx or uv, or install cibuildwheel==$cibuildwheel_version" >&2
    exit 1
fi

exec "${runner[@]}" \
    --platform linux \
    --archs x86_64 \
    --output-dir dist \
    .
