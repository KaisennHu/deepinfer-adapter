#!/usr/bin/env bash
# Run one HTTP P/D worker. See README.md for the validated single-host profile.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER_ROOT="$(dirname "$SCRIPT_DIR")"
: "${PD_ROLE:?set PD_ROLE to prefill or decode}"
: "${DEEPINFER_REPO:?set DEEPINFER_REPO to the patched and built checkout}"
: "${MODEL_PATH:?set MODEL_PATH to the model directory}"
case "$PD_ROLE" in
  prefill) default_port=18300; default_offset=0; default_master=24333 ;;
  decode) default_port=18400; default_offset=4; default_master=24433 ;;
  *) echo "PD_ROLE must be prefill or decode" >&2; exit 2 ;;
esac
case "${PD_PROFILE:-none}" in
  none) ;;
  b300-1p1d) source "$ADAPTER_ROOT/configs/b300-1p1d.env" ;;
  *) echo "Unknown PD_PROFILE: $PD_PROFILE" >&2; exit 2 ;;
esac
if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
fi
export PYTHONPATH="$ADAPTER_ROOT:$DEEPINFER_REPO${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${ENABLE_NUMA_SHIM:-0}" == 1 ]]; then
  export PYTHONPATH="$SCRIPT_DIR/numa_shim:$PYTHONPATH"
fi
export DEEPINFER_ACCELERATOR="${DEEPINFER_ACCELERATOR:-nvidia}"
export EP_DISABLE_GIN="${EP_DISABLE_GIN:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
# DeepInfer also uses MASTER_PORT - 50: keep P/D rendezvous ranges distinct.
export MASTER_PORT="${MASTER_PORT:-$default_master}"
HANDOFF_DIR="${HANDOFF_DIR:-/dev/shm/di-pd}"
mkdir -p "$HANDOFF_DIR"
exec python -u -m deepinfer_adapter.pd_http \
  --role "$PD_ROLE" --host "${PD_HOST:-127.0.0.1}" \
  --port "${PD_PORT:-$default_port}" \
  --model-path "$MODEL_PATH" \
  --served-model "${SERVED_MODEL:-DeepSeek-V4.1-Flash-1017}" \
  --max-model-len "${MAX_MODEL_LEN:-262144}" \
  --dp-size "${DP_SIZE:-4}" --device-offset "${DEVICE_OFFSET:-$default_offset}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.8}" \
  --max-num-seqs "${MAX_NUM_SEQS:-32}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}" \
  --lookahead-tokens "${LOOKAHEAD_TOKENS:-1}" \
  --nextn "${NEXTN:-0}" --handoff-dir "$HANDOFF_DIR" \
  --handoff-timeout "${HANDOFF_TIMEOUT:-180}" "$@"
