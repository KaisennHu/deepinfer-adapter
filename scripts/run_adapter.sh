#!/usr/bin/env bash
# Launch the gRPC adapter in front of a DeepInfer engine.
#
# DeepInfer is SPMD: the server spawns one process per data-parallel rank and
# each rank listens on its own port (ADAPTER_PORT + rank). With DP_SIZE=8 the
# workers therefore occupy ports 50051..50058.
#
# ADAPTER_BACKEND=deepinfer (default) requires:
#   DEEPINFER_REPO  import root of a built DeepInfer checkout
#   MODEL_PATH      model directory (weights + tokenizer)
#
# ADAPTER_BACKEND=echo needs neither and serves synthetic tokens, which is
# enough to verify the router wiring on a machine without a GPU.
#
# Optional:
#   VENV_PATH         virtualenv to activate before launch
#   ENABLE_NUMA_SHIM  1 to prepend scripts/numa_shim to PYTHONPATH; see
#                     scripts/numa_shim/sitecustomize.py
#   EP_DISABLE_GIN    1 lets Engram buffer creation succeed on hosts without an
#                     IB device, but the first forward pass still fails because
#                     the fetch kernel only implements the GIN path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER_ROOT="$(dirname "$SCRIPT_DIR")"

ADAPTER_BACKEND="${ADAPTER_BACKEND:-deepinfer}"
ADAPTER_PORT="${ADAPTER_PORT:-50051}"
SERVED_MODEL="${SERVED_MODEL:-DeepSeek-V4.1-Flash}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
DP_SIZE="${DP_SIZE:-8}"
DEVICE_OFFSET="${DEVICE_OFFSET:-0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

if [[ -n "${VENV_PATH:-}" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
fi

PYTHONPATH="$ADAPTER_ROOT"
if [[ "$ADAPTER_BACKEND" == "echo" ]]; then
  # The echo backend is single-process; ranks would only duplicate the port.
  DP_SIZE=1
  DEVICE_OFFSET=0
else
  : "${DEEPINFER_REPO:?set DEEPINFER_REPO to the DeepInfer checkout}"
  : "${MODEL_PATH:?set MODEL_PATH to the model directory}"
  PYTHONPATH="$DEEPINFER_REPO:$PYTHONPATH"
  export DEEPINFER_ACCELERATOR="${DEEPINFER_ACCELERATOR:-nvidia}"
fi

if [[ "${ENABLE_NUMA_SHIM:-0}" == "1" ]]; then
  # sitecustomize must be importable at interpreter startup, hence a path entry
  # rather than an explicit import from the adapter.
  PYTHONPATH="$SCRIPT_DIR/numa_shim:$PYTHONPATH"
fi
export PYTHONPATH

exec python -u -m deepinfer_adapter.server \
  --port "$ADAPTER_PORT" \
  --model "$SERVED_MODEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --backend "$ADAPTER_BACKEND" \
  --deepinfer-model "${MODEL_PATH:-}" \
  --engine-type "${ENGINE_TYPE:-NORMAL}" \
  --dp-size "$DP_SIZE" \
  --device-offset "$DEVICE_OFFSET" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-seqs "$MAX_NUM_SEQS"
