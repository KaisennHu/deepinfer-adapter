#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Generate the Python bindings for the vLLM worker protocol served by
# deepinfer_adapter/server.py.
#
# The canonical schema lives in vllm-project/vllm: rust/proto/{inference,control}.proto.
# Copies are vendored under proto/ so the adapter does not depend on a vLLM
# checkout. Requires grpcio-tools matching the grpcio runtime:
#
#   python -m pip install "grpcio-tools==$(python -c 'import grpc; print(grpc.__version__)')"
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
proto_dir="$repo_root/proto"
out_dir="$repo_root/deepinfer_adapter/proto_gen"
well_known="$(python -c 'import os, grpc_tools; print(os.path.join(os.path.dirname(grpc_tools.__file__), "_proto"))')"

mkdir -p "$out_dir"
python -m grpc_tools.protoc \
  -I "$proto_dir" \
  -I "$well_known" \
  --python_out="$out_dir" \
  --pyi_out="$out_dir" \
  --grpc_python_out="$out_dir" \
  "$proto_dir/inference.proto" \
  "$proto_dir/control.proto"

echo "generated into $out_dir:"
ls -1 "$out_dir"
