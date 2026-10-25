#!/usr/bin/env bash
# Apply the vendored-DeepEP patches and rebuild its wheel.
#
# DeepEP is a submodule of DeepInfer, so the changes live here as patches rather
# than as modifications to that checkout. Both steps matter: the fetch path has
# host-side branches in C++ and a JIT-compiled header, and the header ships
# inside the wheel.
#
# Required:
#   DEEPINFER_REPO  import root of the DeepInfer checkout
#
# Optional:
#   VENV_PATH       virtualenv to activate (must contain the built torch)
#   WHEEL_DIR       wheel output directory
set -euo pipefail

: "${DEEPINFER_REPO:?set DEEPINFER_REPO to the DeepInfer checkout}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="$(dirname "$SCRIPT_DIR")/patches"
DEEP_EP="$DEEPINFER_REPO/vendor/DeepEP"
WHEEL_DIR="${WHEEL_DIR:-$DEEPINFER_REPO/build/nvidia-x86_64/vendor-wheels}"

if [[ -n "${VENV_PATH:-}" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
fi

patch="$PATCH_DIR/deepep-engram-lsa-fetch.patch"
expected=f6aa38fe06cd4a0489e7730365d4e0114970a2f3
if [[ "$(git -C "$DEEP_EP" rev-parse HEAD)" != "$expected" ]]; then
  echo "Expected DeepEP $expected; initialize the pinned DeepInfer submodule." >&2
  exit 1
fi
if git -C "$DEEP_EP" apply --reverse --check "$patch" 2>/dev/null; then
  echo "DeepEP patch already applied."
else
  git -C "$DEEP_EP" apply --check "$patch"
  git -C "$DEEP_EP" apply "$patch"
fi
# Allows patching before DeepInfer build.sh builds all vendor wheels.
if [[ "${APPLY_ONLY:-0}" == 1 ]]; then
  exit 0
fi

mkdir -p "$WHEEL_DIR"
cd "$DEEP_EP"
python setup.py bdist_wheel --dist-dir "$WHEEL_DIR"
python -m pip install --no-deps --no-index --find-links "$WHEEL_DIR" \
  --force-reinstall deep_ep
