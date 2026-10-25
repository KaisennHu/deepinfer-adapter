#!/usr/bin/env bash
# Apply source changes only. Rebuild DeepInfer afterwards as documented.
set -euo pipefail
: "${DEEPINFER_REPO:?set DEEPINFER_REPO to the DeepInfer checkout}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$SCRIPT_DIR/../patches/deepinfer-pd-optimized.patch"
EXPECTED=9bd9cf243fc53ef9afbf8f04d391d989117d6ab9
if [[ "$(git -C "$DEEPINFER_REPO" rev-parse HEAD)" != "$EXPECTED" ]]; then
  echo "Expected DeepInfer $EXPECTED; use a separate checkout for this adapter." >&2
  exit 1
fi
if git -C "$DEEPINFER_REPO" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "DeepInfer patch already applied."
else
  git -C "$DEEPINFER_REPO" apply --check "$PATCH"
  git -C "$DEEPINFER_REPO" apply "$PATCH"
  echo "DeepInfer patch applied; rebuild the native extension before serving."
fi
