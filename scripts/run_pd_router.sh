#!/usr/bin/env bash
# vLLM Router itself is unmodified. P and D must be healthy before launch.
set -euo pipefail
: "${ROUTER_BIN:?set ROUTER_BIN to the vllm-router executable}"
exec "$ROUTER_BIN" --vllm-pd-disaggregation \
  --prefill "${PREFILL_URL:-http://127.0.0.1:18300}" \
  --decode "${DECODE_URL:-http://127.0.0.1:18400}" \
  --policy consistent_hash --host "${ROUTER_HOST:-127.0.0.1}" \
  --port "${ROUTER_PORT:-13010}" "$@"
