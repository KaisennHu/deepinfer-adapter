# SPDX-License-Identifier: Apache-2.0
"""Generation backends for the DeepInfer adapter.

The adapter is transport-only: it receives router-produced token ids and streams
token ids plus text back. A backend decides where those tokens come from.

``EchoBackend`` needs no engine and no GPU and exists to validate the
router -> adapter protocol path. ``DeepInferEngineBackend`` drives a real
DeepInfer engine and is only importable where a built ``deepinfer`` package
exists.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("deepinfer_adapter.backends")


@dataclass
class GenerateParams:
    """The subset of a generate request a backend needs."""

    request_id: str = ""
    model: str = ""
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int = 16
    ignore_eos: bool = False
    stop_token_ids: list[int] = field(default_factory=list)
    stop_strings: list[str] = field(default_factory=list)


@dataclass
class TokenChunk:
    """One streamed frame; ``finish_reason`` is None until the last frame."""

    token_ids: list[int]
    text: str
    finish_reason: str | None = None


@dataclass(frozen=True)
class BackendCapabilities:
    """Which ``GenerateRequest.stopping`` constraints a backend can honour.

    A backend that cannot honour a constraint must say so here; the server
    rejects the request instead of dropping it. Dropping a stop condition
    changes model output while still returning a successful response.
    """

    stop_strings: bool = False
    stop_token_ids: bool = False


class GenerationBackend:
    """Backend contract. One instance serves one worker process."""

    capabilities = BackendCapabilities()

    def stream(self, token_ids: list[int], params: GenerateParams) -> Iterator[TokenChunk]:
        raise NotImplementedError

    def close(self) -> None:
        """Release backend resources."""


class EchoBackend(GenerationBackend):
    """Connectivity backend: no engine, no GPU, deterministic output.

    Emits the prompt token ids in reverse, one frame per token, so a client can
    check that the router's token-in path delivered exactly the ids it sent.
    It honours ``stop_token_ids`` so the conformance suite can exercise both the
    accepted and the rejected stop paths without loading weights.
    """

    capabilities = BackendCapabilities(stop_strings=False, stop_token_ids=True)

    def stream(self, token_ids: list[int], params: GenerateParams) -> Iterator[TokenChunk]:
        ids = list(reversed(token_ids)) or [0]
        total = max(1, min(params.max_tokens, len(ids)))
        stop_ids = set(params.stop_token_ids)
        for i in range(total):
            token_id = ids[i % len(ids)]
            last = i + 1 == total
            if token_id in stop_ids:
                finish_reason = "stop"
            elif last:
                finish_reason = "length" if total == params.max_tokens else "stop"
            else:
                finish_reason = None
            yield TokenChunk(
                token_ids=[token_id], text=f"[{token_id}]", finish_reason=finish_reason
            )
            if finish_reason is not None:
                return


class _RequestStream:
    """Bridges one request between the engine step loop and its response stream.

    ``stream()`` drains this from the caller's thread while the step loop pushes
    into it from the engine thread.
    """

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.token_ids: list[int] = []
        self.text = ""
        self.finished = False
        self._queue: queue.Queue[TokenChunk | None] = queue.Queue()

    def push(self, chunk: TokenChunk) -> None:
        self._queue.put(chunk)
        if chunk.finish_reason is not None:
            self.finished = True
            self._queue.put(None)

    def drain(self) -> Iterator[TokenChunk]:
        while True:
            chunk = self._queue.get()
            if chunk is None:
                return
            yield chunk


#: Engine finish reasons (`csrc/engine/sequence.hpp`) folded onto the worker
#: contract's three outcomes.
_FINISH_REASONS = {"length": "length", "stop": "stop", "abort": "abort"}

_IDLE_POLL_SECONDS = 1.0

#: With several ranks every iteration joins a collective, so an idle rank has to
#: come back quickly or it stalls the ranks that do have work.
_DP_IDLE_POLL_SECONDS = 0.001


class DeepInferEngineBackend(GenerationBackend):
    """Drives a real DeepInfer ``LLMEngine`` as an online worker.

    DeepInfer ships an offline engine only: ``add_request`` plus ``step()``, whose
    outputs are pooled and released on the next ``step()``. Serving it means
    running a background ``step()`` loop that clones each step's outputs and fans
    them out to per-request streams.

    Every engine call happens on that one loop thread, because the engine is not
    thread-safe: submission, cancellation and shutdown are requests queued to it.

    Contract notes:

    - ``SamplingParams`` has no stop-sequence support, so both stop constraints
      are reported unsupported through ``capabilities`` and the server rejects
      requests that carry them instead of dropping them. Stopping relies on
      ``max_tokens`` and the model's EOS.
    - Prompt text never reaches this layer: the router renders and tokenizes.
    - DeepInfer runs one engine rank per process and one process per accelerator,
      so a multi-GPU deployment means ``data_parallel_size`` processes, each
      owning one rank and dialed by the router as its own worker (see
      ``server.py --dp-size``).
    """

    def __init__(
        self,
        model: str,
        tokenizer: str | None = None,
        engine_type: str = "NORMAL",
        engine_rank: int = 0,
        data_parallel_size: int = 1,
        device_index: int | None = None,
        **engine_kwargs: Any,
    ) -> None:
        # Device visibility has to be fixed before CUDA is first initialized, and
        # DeepInfer reads its knobs when the affected modules are imported, so both
        # happen before the imports below.
        if device_index is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
        os.environ.setdefault("DEEPINFER_USE_NEXTN_DECODE", "5")
        os.environ.setdefault("DEEPINFER_EXPERT_COUNT_WINDOW_SIZE", "3")

        import numpy as np
        import torch  # noqa: F401  loads libc10/libtorch that deepinfer._C links against

        from deepinfer._C.engine import RequestOutput
        from deepinfer.engine.arg_utils import EngineArgs
        from deepinfer.engine.engine_type import EngineType
        from deepinfer.engine.llm_engine import LLMEngine

        self._np = np
        self._request_output = RequestOutput

        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            engine_type=EngineType[engine_type],
            engine_rank=engine_rank,
            data_parallel_size=data_parallel_size,
            # With no context parallelism every rank owns one device and the
            # expert-parallel group spans all of them.
            expert_parallel_size=data_parallel_size,
            context_parallel_size=1,
            chunked_prefill=True,
            **engine_kwargs,
        )
        logger.info(
            "initializing DeepInfer engine (engine_type=%s, model=%s, rank=%d/%d, cuda_visible_devices=%s)",
            engine_type,
            model,
            engine_rank,
            data_parallel_size,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        self._engine = LLMEngine.from_engine_args(engine_args)
        self._tokenizer = self._engine.tokenizer

        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._stop = threading.Event()
        self._streams: dict[str, _RequestStream] = {}
        self._pending: list[tuple[str, Any, Any]] = []
        self._aborts: list[str] = []
        self._idle_poll = (
            _DP_IDLE_POLL_SECONDS if data_parallel_size > 1 else _IDLE_POLL_SECONDS
        )

        self._loop = threading.Thread(target=self._run_loop, name="deepinfer-step", daemon=True)
        self._loop.start()
        logger.info("DeepInfer engine ready")

    @property
    def max_model_len(self) -> int:
        """Runtime context limit the engine settled on, for worker discovery."""
        return self._engine.model_config.max_model_len

    @property
    def kv_block_size(self) -> int:
        """KV cache block size in tokens, for worker discovery."""
        return self._engine.cache_config.block_size

    def _sampling_params(self, params: GenerateParams) -> Any:
        from deepinfer.sampling_params import SamplingParams

        kwargs: dict[str, Any] = {
            "max_tokens": max(1, params.max_tokens),
            "ignore_eos": params.ignore_eos,
        }
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.top_k is not None:
            kwargs["top_k"] = params.top_k
        return SamplingParams(**kwargs)

    def stream(self, token_ids: list[int], params: GenerateParams) -> Iterator[TokenChunk]:
        state = _RequestStream(params.request_id or uuid.uuid4().hex)
        sampling_params = self._sampling_params(params)
        prompt = self._np.asarray(token_ids, dtype=self._np.int32)
        with self._lock:
            self._streams[state.request_id] = state
            self._pending.append((state.request_id, prompt, sampling_params))
        self._wakeup.set()
        try:
            yield from state.drain()
        finally:
            # Runs on client disconnect too, which is how cancellation reaches us.
            with self._lock:
                self._streams.pop(state.request_id, None)
                if not state.finished:
                    self._aborts.append(state.request_id)
            self._wakeup.set()

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                pending, self._pending = self._pending, []
                aborts, self._aborts = self._aborts, []
            try:
                for request_id, prompt, sampling_params in pending:
                    self._engine.add_request(
                        user_id="router",
                        request_id=request_id,
                        sampling_params=sampling_params,
                        prompt_token_ids=prompt,
                    )
                for request_id in aborts:
                    self._engine.abort_request(request_id)

                if self._engine.has_unfinished_requests():
                    outputs = self._engine.step()
                else:
                    outputs = self._engine.flush_inflight_reqs()
                for pointer in outputs:
                    self._dispatch(self._request_output.clone(pointer))
            except Exception:
                logger.exception("DeepInfer step loop failed")
                self._abort_all()
                return

            if not pending and not self._streams:
                self._wakeup.wait(self._idle_poll)
                self._wakeup.clear()

    def _dispatch(self, output: Any) -> None:
        state = self._streams.get(output.request_id)
        if state is None:
            return
        state.token_ids.append(output.token_id)
        text = self._tokenizer.decode(state.token_ids)
        delta = text[len(state.text) :]
        state.text = text

        finish_reason = None
        if output.finished:
            # `insufficient_system_resource` and friends are surfaced as a hard stop.
            finish_reason = _FINISH_REASONS.get(output.finish_reason, "stop")
            self._streams.pop(output.request_id, None)
        state.push(TokenChunk(token_ids=[output.token_id], text=delta, finish_reason=finish_reason))

    def _abort_all(self) -> None:
        for state in list(self._streams.values()):
            state.push(TokenChunk(token_ids=[], text="", finish_reason="abort"))
        self._streams.clear()

    def close(self) -> None:
        self._stop.set()
        self._abort_all()
        self._wakeup.set()
        self._loop.join(timeout=30)
        self._engine.close()
