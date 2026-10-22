#!/usr/bin/env bash
# Launch the vLLM Router in front of one or more adapter workers.
#
# Every grpc:// worker must implement the vllm.Inference and vllm.Control
# services, which is what scripts/run_adapter.sh provides. With DP_SIZE=8 the
# URL list is grpc://127.0.0.1:50051,...,grpc://127.0.0.1:50058.
#
# Required:
#   MODEL_PATH   model directory, used by the router for tokenization
#   WORKER_URLS  grpc:// endpoints, comma or space separated
#
# Optional:
#   ROUTER_BIN   router executable; defaults to ./vllm-router in the repo root
#   SERVED_MODEL name advertised by /v1/models
set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH to the model directory}"
: "${WORKER_URLS:?set WORKER_URLS to the grpc:// endpoints}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER_ROOT="$(dirname "$SCRIPT_DIR")"

ROUTER_BIN="${ROUTER_BIN:-$ADAPTER_ROOT/vllm-router}"

# `--worker-urls` takes a list of bare values, so a single quoted
# "a,b,c" argument is rejected as one malformed URL. Split it ourselves.
# shellcheck disable=SC2206
worker_urls=(${WORKER_URLS//,/ })

# The tokenizer is used to decode the returned token ids. Prompts are not
# rendered here: grpc workers take token ids, so `/v1/completions` callers must
# send `prompt` as a list of integers.
export VLLM_ROUTER_TOKENIZER="$MODEL_PATH"
export VLLM_ROUTER_MODEL="${SERVED_MODEL:-$MODEL_PATH}"

exec "$ROUTER_BIN" \
  --host "${ROUTER_HOST:-127.0.0.1}" \
  --port "${ROUTER_PORT:-30000}" \
  --worker-urls "${worker_urls[@]}"
