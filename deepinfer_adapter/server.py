# SPDX-License-Identifier: Apache-2.0
"""Serve a DeepInfer engine as a vLLM-style gRPC worker.

Exposes the subset of the vLLM worker contract a router needs on a ``grpc://``
worker:

* ``grpc.health.v1.Health/Check`` with an empty service name, for liveness
* ``vllm.Inference/GenerateStream``, the request path (token ids in and out)
* ``vllm.Control/{GetServerInfo,GetModelInfo,GetKvEventSources}``, for the
  router's model discovery and KV source discovery

The router owns chat rendering and tokenization, so this worker is token-in and
token-out and carries no chat template. Generation is delegated to a
:class:`~deepinfer_adapter.backends.GenerationBackend`.

Run::

    python -m deepinfer_adapter.server --port 50051
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
from collections.abc import Iterator
from concurrent import futures
from pathlib import Path

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

# Generated bindings come from scripts/gen_proto.sh. protoc emits absolute
# imports, so the output directory itself has to be importable.
_PROTO_GEN = Path(__file__).resolve().parent / "proto_gen"
if str(_PROTO_GEN) not in sys.path:
    sys.path.insert(0, str(_PROTO_GEN))

import control_pb2  # noqa: E402
import control_pb2_grpc  # noqa: E402
import inference_pb2  # noqa: E402
import inference_pb2_grpc  # noqa: E402

from deepinfer_adapter import __version__  # noqa: E402
from deepinfer_adapter.backends import (  # noqa: E402
    DeepInferEngineBackend,
    EchoBackend,
    GenerateParams,
    GenerationBackend,
)

logger = logging.getLogger("deepinfer_adapter")

#: `StoppingCriteria.max_new_tokens` documents 0 as "engine default".
_DEFAULT_MAX_TOKENS = 16

_FINISH_REASONS = {
    "length": inference_pb2.FinishInfo.LENGTH,
    "stop": inference_pb2.FinishInfo.STOP,
    "abort": inference_pb2.FinishInfo.ABORTED,
}


class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    """Serves ``vllm.Inference`` on top of a generation backend."""

    def __init__(self, backend: GenerationBackend, model: str) -> None:
        self._backend = backend
        self._model = model

    def _params(self, request: inference_pb2.GenerateRequest) -> GenerateParams:
        sampling = request.sampling
        stopping = request.stopping
        stop_strings = list(stopping.stop_strings)
        stop_token_ids = list(stopping.stop_token_ids)

        # A backend that cannot honour a stop condition must reject the request.
        # Stripping it would change the output while still returning success.
        capabilities = self._backend.capabilities
        if stop_strings and not capabilities.stop_strings:
            raise ValueError(
                f"backend {type(self._backend).__name__} cannot honour stop_strings; "
                "remove `stop` or use stop_token_ids"
            )
        if stop_token_ids and not capabilities.stop_token_ids:
            raise ValueError(
                f"backend {type(self._backend).__name__} cannot honour stop_token_ids; "
                "remove `stop`"
            )

        return GenerateParams(
            request_id=request.request_id,
            model=request.model or self._model,
            temperature=request.temperature if request.HasField("temperature") else None,
            top_p=sampling.top_p if sampling.HasField("top_p") else None,
            top_k=sampling.top_k if sampling.HasField("top_k") else None,
            max_tokens=stopping.max_new_tokens or _DEFAULT_MAX_TOKENS,
            ignore_eos=stopping.ignore_eos,
            stop_token_ids=stop_token_ids,
            stop_strings=stop_strings,
        )

    def _events(
        self, request: inference_pb2.GenerateRequest
    ) -> Iterator[inference_pb2.GenerateResponse]:
        prompt_case = request.WhichOneof("prompt")
        if prompt_case != "token_ids":
            raise ValueError(f"DeepInfer accepts token-id prompts only, got {prompt_case!r}")
        prompt_ids = list(request.token_ids.ids)
        if not prompt_ids:
            raise ValueError("prompt must not be empty")
        # Validate before the first frame so a rejected request produces no output.
        params = self._params(request)

        yield inference_pb2.GenerateResponse(
            prompt_info=inference_pb2.PromptInfo(num_prompt_tokens=len(prompt_ids))
        )

        emitted = 0
        for chunk in self._backend.stream(prompt_ids, params):
            emitted += len(chunk.token_ids)
            finish_info = None
            if chunk.finish_reason is not None:
                finish_info = inference_pb2.FinishInfo(
                    num_output_tokens=emitted,
                    finish_reason=_FINISH_REASONS[chunk.finish_reason],
                )
            yield inference_pb2.GenerateResponse(
                outputs=inference_pb2.SequenceOutput(
                    index=0,
                    text=chunk.text,
                    num_tokens=len(chunk.token_ids),
                    token_ids=chunk.token_ids,
                    finish_info=finish_info,
                )
            )

    def Generate(
        self, request: inference_pb2.GenerateRequest, context: grpc.ServicerContext
    ) -> inference_pb2.GenerateResponse:
        prompt_info = None
        text_parts: list[str] = []
        token_ids: list[int] = []
        finish_info = None
        try:
            for event in self._events(request):
                if event.HasField("prompt_info"):
                    prompt_info = event.prompt_info
                if event.HasField("outputs"):
                    text_parts.append(event.outputs.text)
                    token_ids.extend(event.outputs.token_ids)
                    if event.outputs.HasField("finish_info"):
                        finish_info = event.outputs.finish_info
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        return inference_pb2.GenerateResponse(
            prompt_info=prompt_info,
            outputs=inference_pb2.SequenceOutput(
                index=0,
                text="".join(text_parts),
                num_tokens=len(token_ids),
                token_ids=token_ids,
                finish_info=finish_info,
            ),
        )

    def GenerateStream(
        self, request: inference_pb2.GenerateRequest, context: grpc.ServicerContext
    ) -> Iterator[inference_pb2.GenerateResponse]:
        try:
            yield from self._events(request)
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))


class ControlServicer(control_pb2_grpc.ControlServicer):
    """Serves the ``vllm.Control`` RPCs used for worker discovery.

    The remaining Control RPCs stay unimplemented.
    """

    def __init__(self, model: str, max_model_len: int, kv_block_size: int) -> None:
        self._model = model
        self._max_model_len = max_model_len
        self._kv_block_size = kv_block_size

    def GetServerInfo(
        self, request: control_pb2.GetServerInfoRequest, context: grpc.ServicerContext
    ) -> control_pb2.ServerInfo:
        del request, context
        return control_pb2.ServerInfo(
            engine_version=f"deepinfer-adapter/{__version__}",
            api_version="deepinfer-adapter/1",
            instance_id=socket.gethostname(),
            parallelism=control_pb2.ParallelismInfo(
                tensor_parallel_size=1, data_parallel_size=1, world_size=1
            ),
            max_model_len=self._max_model_len,
            kv_block_size=self._kv_block_size,
        )

    def GetModelInfo(
        self, request: control_pb2.GetModelInfoRequest, context: grpc.ServicerContext
    ) -> control_pb2.ModelInfo:
        del request, context
        return control_pb2.ModelInfo(
            model_id=self._model,
            served_model_name=self._model,
            supports_text_input=False,
            supports_token_ids_input=True,
        )

    def GetKvEventSources(
        self, request: control_pb2.GetKvEventSourcesRequest, context: grpc.ServicerContext
    ) -> control_pb2.GetKvEventSourcesResponse:
        # No KV event publisher yet: the router falls back to load-only routing.
        del request, context
        return control_pb2.GetKvEventSourcesResponse()


#: DeepInfer's default checkpoint; `DEEPINFER_HF_MODEL_DIR` points it at a local copy.
_DEFAULT_DEEPINFER_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"

#: Reported by the echo backend, which has no engine to ask.
_FALLBACK_MAX_MODEL_LEN = 8192
_FALLBACK_KV_BLOCK_SIZE = 64


def _build_backend(args: argparse.Namespace, rank: int) -> GenerationBackend:
    if args.backend == "echo":
        return EchoBackend()
    return DeepInferEngineBackend(
        model=args.deepinfer_model,
        tokenizer=args.deepinfer_tokenizer,
        engine_type=args.engine_type,
        engine_rank=rank,
        data_parallel_size=args.dp_size,
        device_index=args.device_offset + rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len or None,
        max_num_batched_tokens=args.max_num_batched_tokens,
        lookahead_tokens=args.lookahead_tokens,
    )


def build_server(
    backend: GenerationBackend,
    model: str,
    max_model_len: int,
    kv_block_size: int,
    address: str,
    max_workers: int = 8,
) -> tuple[grpc.Server, int]:
    """Attach the worker contract to ``backend``. Returns the server and its bound port.

    ``address`` may use port 0 to let the OS pick one, which the conformance
    suite relies on. Marking health ``SERVING`` here is safe because the caller
    builds the backend first: an engine that is still loading has not returned
    from its constructor.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    inference_pb2_grpc.add_InferenceServicer_to_server(
        InferenceServicer(backend, model), server
    )
    control_pb2_grpc.add_ControlServicer_to_server(
        ControlServicer(model, max_model_len, kv_block_size), server
    )

    port = server.add_insecure_port(address)
    if port == 0:
        raise RuntimeError(f"failed to bind {address}")

    # The router probes worker liveness with an empty service name.
    for service in ("", "vllm.Inference", "vllm.Control"):
        health_servicer.set(service, health_pb2.HealthCheckResponse.SERVING)
    return server, port


def _serve(args: argparse.Namespace, rank: int) -> None:
    """Serve one engine rank on its own gRPC port."""
    port = args.port + rank
    backend = _build_backend(args, rank)
    max_model_len = getattr(backend, "max_model_len", 0) or args.max_model_len or _FALLBACK_MAX_MODEL_LEN
    kv_block_size = getattr(backend, "kv_block_size", 0) or args.kv_block_size or _FALLBACK_KV_BLOCK_SIZE

    server, port = build_server(
        backend,
        args.model,
        max_model_len,
        kv_block_size,
        f"{args.host}:{port}",
        args.max_workers,
    )

    server.start()
    logger.info(
        "DeepInfer adapter rank %d/%d listening on %s (backend=%s, model=%s, max_model_len=%d)",
        rank,
        args.dp_size,
        address,
        args.backend,
        args.model,
        max_model_len,
    )
    try:
        server.wait_for_termination()
    finally:
        backend.close()


def _launch_ranks(args: argparse.Namespace) -> None:
    """Run one process per engine rank; every rank is its own gRPC worker."""
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_serve, args=(args, rank), daemon=False)
        for rank in range(args.dp_size)
    ]
    for process in processes:
        process.start()
    logger.info(
        "DeepInfer adapter serving %d ranks on %s:%d-%d",
        args.dp_size,
        args.host,
        args.port,
        args.port + args.dp_size - 1,
    )
    for process in processes:
        process.join()
    failed = [rank for rank, process in enumerate(processes) if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"engine ranks exited with an error: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="gRPC port the router dials; rank i listens on port + i",
    )
    parser.add_argument("--model", default="deepinfer", help="model id reported to the router")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help="context limit for the engine and for worker discovery; 0 leaves it to the engine",
    )
    parser.add_argument("--kv-block-size", type=int, default=0, help="0 asks the engine")
    parser.add_argument("--backend", choices=("echo", "deepinfer"), default="echo")
    parser.add_argument("--max-workers", type=int, default=8)

    deepinfer = parser.add_argument_group("deepinfer backend")
    deepinfer.add_argument(
        "--deepinfer-model",
        default=os.environ.get("DEEPINFER_HF_MODEL_DIR", _DEFAULT_DEEPINFER_MODEL),
    )
    deepinfer.add_argument("--deepinfer-tokenizer", default=None)
    deepinfer.add_argument(
        "--engine-type",
        choices=("NORMAL", "PROMPT", "COMPLETION"),
        default="NORMAL",
        help="DeepInfer engine type; PROMPT/COMPLETION are the disaggregated roles",
    )
    deepinfer.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="engine ranks to launch; one process per rank, each its own gRPC worker",
    )
    deepinfer.add_argument(
        "--device-offset", type=int, default=0, help="first accelerator index rank 0 maps to"
    )
    deepinfer.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    deepinfer.add_argument("--max-num-seqs", type=int, default=256)
    deepinfer.add_argument("--max-num-batched-tokens", type=int, default=32768)
    deepinfer.add_argument(
        "--lookahead-tokens",
        type=int,
        default=1,
        help="KV lookahead the scheduler reserves for speculative decoding",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    if args.backend == "deepinfer" and args.dp_size > 1:
        _launch_ranks(args)
        return
    _serve(args, rank=0)


if __name__ == "__main__":
    main()
