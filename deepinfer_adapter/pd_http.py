# SPDX-License-Identifier: Apache-2.0
"""HTTP prefill/decode workers that plug DeepInfer into ``vllm-router`` PD mode.

The router speaks the vLLM PD protocol: it sends the request to a prefill
worker with ``max_tokens=1`` and ``kv_transfer_params.do_remote_decode=true``,
reads ``kv_transfer_params`` from that response, and forwards the original
request to a decode worker with those params attached. For NIXL pull mode the
router copies the prefill response verbatim, so both ends of the handoff are
ours and this module defines what the params mean.

Topology per side is one facade process (HTTP) plus one engine process per
data-parallel rank, because DeepInfer is SPMD: every rank must keep stepping so
the collectives inside ``step()`` line up. The facade gives a request to the
rank derived from its prompt, or an exact cached prefix when snapshot reuse is
enabled. Decode follows the source rank carried in the handoff metadata.

Handoff v1 goes through ``--handoff-dir`` (tmpfs by default): prefill copies the
block KV plus the request's full PE->CE state out of HBM, decode copies both
back into the slots it just allocated and resumes decoding without replaying
any token.
"""

from __future__ import annotations

import argparse
import asyncio
from array import array
import hashlib
import json
import logging
import multiprocessing as mp
import os
import pathlib
import queue
import threading
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from deepinfer_adapter.text_stream import TokenTextStream
from deepinfer_adapter.prefix_snapshot import PrefixSnapshots, yoco_snapshot_anchor

logger = logging.getLogger("deepinfer_adapter.pd_http")

_FINISH_REASONS = {"length": "length", "stop": "stop", "abort": "stop"}



def _allow_peer_fd_import() -> None:
    """DeepEP resolves sibling ranks with pidfd_getfd; yama needs opting in."""
    try:
        import ctypes

        prctl = ctypes.CDLL(None, use_errno=True).prctl
        prctl(0x59616D61, ctypes.c_ulong(-1), 0, 0, 0)  # PR_SET_PTRACER_ANY
    except Exception:  # pragma: no cover - best effort
        pass


def _sampling_params(temperature: float | None, max_tokens: int, ignore_eos: bool,
                     seed: int | None = None):
    from deepinfer.sampling_params import SamplingParams

    temperature = 0.0 if temperature is None else float(temperature)
    kwargs = {
        "max_tokens": max(1, max_tokens),
        "temperature": temperature,
        "ignore_eos": ignore_eos,
        "seq_seed": seed,
    }
    if temperature == 0.0:
        # Native temperature=0 becomes epsilon sampling. Explicit top-k=1 at
        # a positive temperature preserves true greedy selection, including ties.
        kwargs.update(temperature=1.0, top_k=1)
    return SamplingParams(**kwargs)


class _RankEngine:
    """One DeepInfer engine rank, driven by a single step loop."""

    def __init__(self, args: argparse.Namespace, rank: int) -> None:
        import numpy as np
        import torch  # noqa: F401  loads libc10 before deepinfer._C

        from deepinfer.engine.arg_utils import EngineArgs
        from deepinfer.engine.engine_type import EngineType
        from deepinfer.engine.llm_engine import LLMEngine

        self.rank = rank
        self.args = args
        self.role = args.role
        self.np = np
        self.torch = torch
        self.handoff_dir = pathlib.Path(args.handoff_dir)
        self.handoff_dir.mkdir(parents=True, exist_ok=True)

        # PROMPT 引擎关掉了 ZOS，空 rank 的 1-token 占位步和忙 rank 的真实 prompt 步
        # 走不同的 DeepGEMM kernel 路径，EP 的 NVLink barrier 会因此超时（4DP4EP 实测）。
        # 默认用 NORMAL + ZOS：占位步与真实步都由 GPU 图统一形状，DP 内各 rank 批次
        # 不一致时才安全；prefill 完成后我们立刻把 seq 摘走，不会真的进入 decode。
        default_engine = "NORMAL"
        engine_type = os.environ.get(f"PD_{args.role.upper()}_ENGINE", default_engine).upper()
        engine_args = EngineArgs(
            model=args.model_path,
            tokenizer=args.model_path,
            engine_type=EngineType[engine_type],
            engine_rank=rank,
            data_parallel_size=args.dp_size,
            expert_parallel_size=args.dp_size,
            context_parallel_size=1,
            chunked_prefill=True,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            lookahead_tokens=args.lookahead_tokens,
        )
        self.engine = LLMEngine.from_engine_args(engine_args)
        self.tokenizer = self.engine.tokenizer
        self.cache_engine = self.engine.driver_worker.cache_engine
        self.block_size = int(self.engine.cache_config.block_size)

        self._prefill_seqs: dict[str, tuple[object, int]] = {}
        snapshot_bytes = int(os.environ.get("PD_PREFIX_SNAPSHOT_BYTES", "0"))
        self._prefix_snapshots = PrefixSnapshots(snapshot_bytes) if self.role == "prefill" and snapshot_bytes > 0 else None
        self._prefix_keys: dict[str, bytes] = {}
        self._prefix_hits: dict[str, int] = {}
        self._yoco_snapshot_keep = 0
        if self._prefix_snapshots and os.environ.get("PD_YOCO_PREFIX_SNAPSHOTS", "0") == "1":
            from deepinfer import envs
            from deepinfer.worker.cache_engine import GPUCacheInterface, KVSTORE_GRAIN

            keep = envs.DEEPINFER_YOCO_PREFILL_KEEP_TOKENS
            history = GPUCacheInterface.swa_block_size_inference
            if (envs.DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE != "compact"
                    or not envs.DEEPINFER_PREFILL_STORE_EXTRA_STATE
                    or keep < history or self.block_size != KVSTORE_GRAIN
                    or GPUCacheInterface.swa_block_size_kvstore < history + keep + self.block_size - 1):
                raise ValueError("YOCO snapshots require compact replay and sufficient aligned ring history")
            self._yoco_snapshot_keep = keep
        elif self._prefix_snapshots and os.environ.get("DEEPINFER_PREFILL_STORE_EXTRA_STATE", "0") != "0":
            raise ValueError("Prefix snapshots currently require the ordinary inference state layout")
        self._pending_decodes: dict[str, tuple[dict, float]] = {}
        # 上一轮是否跑过 step()：ZOS 下 step 的输出可能推迟一次 post-process，
        # 需要在引擎转为空闲时用 flush_inflight_reqs() 收尾。
        self._step_pending = False
        self._last_stuck_warn = 0.0
        self._texts: dict[str, str] = {}
        self._token_cache: dict[str, list[int]] = {}
        self._decoders: dict[str, TokenTextStream] = {}
        self._ipc_sender = self._ipc_receiver = None
        if os.environ.get("PD_CUDA_IPC_POOL", "0") == "1":
            from deepinfer.worker.cache_engine import GPUCacheInterface
            from deepinfer_adapter.cuda_ipc_pool import Receiver, Sender

            pool_id = os.environ["PD_CUDA_IPC_POOL_ID"]
            if not pool_id or pathlib.Path(pool_id).name != pool_id or pool_id in (".", ".."):
                raise ValueError("A unique simple pool ID is required")
            if os.environ.get("PD_DECODE_RANK_POLICY") != "source":
                raise ValueError("GPU pool currently requires source-rank D routing")
            descriptor = self.handoff_dir / f"pool-{pool_id}-r{rank}.ipc"
            dimensions = (int(self.cache_engine.gpu_blocks.shape[1]),
                          GPUCacheInterface.total_bytes_inference,
                          (args.max_model_len + self.block_size - 1) // self.block_size, 2)
            started = time.monotonic()
            if self.role == "prefill":
                self._ipc_sender = Sender(descriptor, *dimensions)
            else:
                deadline = started + args.handoff_timeout
                while not descriptor.exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError("Producer GPU pool was not ready before startup deadline")
                    time.sleep(0.05)
                self._ipc_receiver = Receiver(descriptor, *dimensions)
            logger.info("IPC pool ready role=%s rank=%d dimensions=%s seconds=%.6f descriptor=%s",
                        self.role, rank, dimensions, time.monotonic() - started, descriptor)

        from deepinfer._C.engine import RequestOutput

        self._RequestOutput = RequestOutput
        logger.info(
            "rank %d ready role=%s block_size=%d max_model_len=%d",
            rank,
            self.role,
            self.block_size,
            int(self.engine.model_config.max_model_len),
        )

    # ------------------------------ prefill ------------------------------

    def _submit_prefill(self, cmd: dict) -> None:
        rid = cmd["request_id"]
        tokens = cmd["tokens"]
        # ignore_eos keeps the prompt engine from finalizing the sequence before
        # pop_processed_seqs() can hand it to us; it does not change the logits.
        # max_tokens 必须保持 1：多解一步会让 num_cached_tokens 越过 prompt 长度
        # （导出的是 prompt+K 的 KV、first_token 也变成第 K 个 token），D 侧接管
        # 时会直接触发 zos.cpp 的 "token_start < token_ids.size()" 断言。
        sp = _sampling_params(cmd.get("temperature"), 1, True, seed=cmd.get("seed"))
        token_array = self.np.asarray(tokens, dtype=self.np.int32)
        key = token_array.tobytes() if self._prefix_snapshots else b""
        limit = (4 * yoco_snapshot_anchor(len(tokens), self._yoco_snapshot_keep, self.block_size)
                 if self._yoco_snapshot_keep else None)
        cached = self._prefix_snapshots.find(key, max_prefix_bytes=limit) if self._prefix_snapshots else None
        seq = self.engine.add_request(
            user_id="pd-prefill",
            request_id=rid,
            sampling_params=sp,
            prompt_token_ids=token_array,
            enqueue=cached is None,
        )
        hit_tokens = 0
        if cached is not None:
            prefix, payload = cached
            if self.engine.scheduler.try_allocate_seq(seq):
                try:
                    dst = list(seq.block_table)
                    blocks = payload["blocks"]
                    state = payload["state"]
                    assert len(dst) >= len(blocks)
                    assert state is not None and int(seq.state_id) >= 0
                    state_dst = self.cache_engine.gpu_states[int(seq.state_id)]
                    self.cache_engine.gpu_blocks[dst[:len(blocks)]] = blocks.to(self.cache_engine.gpu_blocks.device)
                    hit_tokens = len(prefix) // 4
                    if self._yoco_snapshot_keep:
                        from deepinfer.worker.cache_engine import GPUCacheInterface

                        assert hit_tokens % self.block_size == 0
                        assert len(blocks) == hit_tokens // self.block_size
                        assert state.numel() == GPUCacheInterface.state_size_to_disk
                        # Producer history is restored below; consumer SWA and
                        # compression state are rebuilt from the aligned suffix.
                        state_dst.zero_()
                        GPUCacheInterface.load_state_from_store_to_prefill_cache(
                            int(seq.state_id), hit_tokens, state.to(state_dst.device),
                            self.cache_engine.gpu_states,
                        )
                    else:
                        assert state.shape == state_dst.shape
                        state_dst.copy_(state.to(state_dst.device))
                    seq.num_cached_tokens = hit_tokens
                    seq.swa_valid_start = 0
                except Exception:
                    self.engine.scheduler.free_seq(seq)
                    raise
            # A capacity miss resumes ordinary scheduling with an unallocated
            # sequence. A hit starts suffix prefill from complete KV + state.
            self.engine.scheduler.add_sequence(seq)
        if self._prefix_snapshots:
            self._prefix_keys[rid] = key
            self._prefix_hits[rid] = hit_tokens
            logger.info("prefix snapshot rid=%s reused_tokens=%d prompt_tokens=%d", rid, hit_tokens, len(tokens))
        self._prefill_seqs[rid] = (seq, len(tokens))

    def _export_ready(self, res_q) -> None:
        if not self._prefill_seqs:
            return
        # 逐条判定，不能用 pop_processed_seqs() 的「取走全部」语义：长 prompt 会
        # 分 chunk，同一 rank 上常常一路已经算完、另一路还在 chunk。整批等待时，
        # 已算完的那条会在下一轮 flush_inflight_reqs() 里被引擎 finalize 并回收
        # 块，之后就再也取不到，表现为整轮压测永久挂死。
        running = {str(seq.request_id) for seq in self.engine.scheduler.running}
        waiting = {str(seq.request_id) for seq in self.engine.scheduler.waiting}
        for rid, (seq, prompt_len) in list(self._prefill_seqs.items()):
            if int(seq.num_cached_tokens) < prompt_len:
                continue  # still chunking; exporting now would ship a partial KV
            if rid not in running:
                # 已不在 running = 要么被 preempt 到 waiting，要么已被 finalize 回收。
                if rid in waiting:
                    continue
                # 块已回收，KV 不可再用。宁可让这条请求报错，也不能让上游流永久
                # 挂住（整轮压测会卡死）。
                self._prefill_seqs.pop(rid, None)
                self._prefix_keys.pop(rid, None)
                self._prefix_hits.pop(rid, None)
                logger.error("prefill seq %s was finalized before KV export", rid)
                res_q.put(("error", rid, "prefill sequence finalized before KV export"))
                continue
            self._prefill_seqs.pop(rid, None)
            self._export_one(rid, seq, res_q)

    def _export_one(self, rid: str, seq, res_q) -> None:
        torch = self.torch
        from deepinfer.worker.cache_engine import GPUCacheInterface

        block_ids = list(seq.block_table)
        num_cached = int(seq.num_cached_tokens)
        first_token = int(seq.data.get_last_token_id())
        draft = [int(t) for t in seq.last_next_n_token]

        # Own one CPU copy; cloning again after the device transfer is redundant.
        gpu_kv = self.cache_engine.gpu_blocks[block_ids].detach()
        kv = gpu_kv.to("cpu", copy=True)
        state = None
        gpu_state = None
        if int(seq.state_id) >= 0:
            # TO_CE=True 才是 PE->CE 的完整 state（含 dropped consumer SWA 与
            # compression ring），长度等于 D 侧 total_bytes_inference。
            gpu_state = GPUCacheInterface.select_state_slice_to_store(
                int(seq.state_id), num_cached, self.cache_engine.gpu_states, TO_CE=True
            )
            state = gpu_state.detach().to("cpu", copy=True) if gpu_state is not None else None

        path = self.handoff_dir / f"{rid}-r{self.rank}.pt"
        tmp = self.handoff_dir / f"{rid}-r{self.rank}.pt.tmp"
        payload = {"blocks": kv, "state": state}
        pool_path = path.with_suffix(".pool")
        if self._ipc_sender and self._ipc_sender.publish(pool_path, gpu_kv, gpu_state):
            path = pool_path
            logger.info("IPC pool publish rank=%d rid=%s blocks=%d", self.rank, rid, len(block_ids))
        else:
            torch.save(payload, tmp)
            os.replace(tmp, path)
            if self._ipc_sender:
                logger.info("IPC pool full or oversize; CPU fallback rank=%d rid=%s", self.rank, rid)
        del gpu_kv, gpu_state

        cache_stored = False
        stored_tokens = 0
        if self._prefix_snapshots:
            key = self._prefix_keys.pop(rid)
            # The P side has computed exactly the input prefix. Its sampled
            # first output token is not part of this snapshot's cache identity.
            assert num_cached == len(key) // 4
            snapshot = payload
            stored_tokens = num_cached
            if self._yoco_snapshot_keep:
                stored_tokens = yoco_snapshot_anchor(num_cached, self._yoco_snapshot_keep, self.block_size)
                key = key[:stored_tokens * 4]
                if stored_tokens:
                    stored_state = GPUCacheInterface.select_state_slice_to_store(
                        int(seq.state_id), stored_tokens, self.cache_engine.gpu_states, TO_CE=False,
                    )
                    assert stored_state is not None
                    snapshot = {
                        # Own only the aligned blocks counted against the LRU budget.
                        "blocks": kv[:stored_tokens // self.block_size].clone(),
                        "state": stored_state.detach().to("cpu", copy=True),
                    }
            if stored_tokens:
                size = sum(t.numel() * t.element_size() for t in snapshot.values() if t is not None)
                cache_stored = self._prefix_snapshots.put(key, snapshot, size)

        meta = {
            "first_token": first_token,
            "first_text": self.tokenizer.decode([first_token]),
            "draft_tokens": draft,
            "num_cached": num_cached,
            "num_blocks": len(block_ids),
            "rank": self.rank,
            "handoff_file": str(path),
            "prefix_cache_stored": cache_stored,
            "prefix_stored_tokens": stored_tokens if cache_stored else 0,
            "prefix_cached_tokens": self._prefix_hits.pop(rid, 0),
        }
        self.engine.abort_request(rid)
        res_q.put(("prefill_done", rid, meta))

    # ------------------------------- decode ------------------------------

    def _submit_decode(self, cmd: dict, res_q) -> None:
        rid = cmd["request_id"]
        meta = cmd["meta"]
        max_new = int(cmd.get("max_tokens", 1)) - 1

        first_token = int(meta["first_token"])
        decoder = TokenTextStream(self.tokenizer)
        first_text = decoder.push(first_token, finished=max_new <= 0)
        self._decoders[rid] = decoder
        self._token_cache[rid] = [first_token]
        self._texts[rid] = first_text
        # The first token was sampled on the prefill side, so the decode side
        # reports it verbatim before any of its own steps run.
        res_q.put(("token", rid, first_token, first_text))

        if max_new <= 0:
            self._discard_handoff(meta)
            self._token_cache.pop(rid, None)
            self._texts.pop(rid, None)
            self._decoders.pop(rid, None)
            res_q.put(("finish", rid, "length", ""))
            return
        # 不能在 step 循环里阻塞等交接文件：一个 rank 卡住会让整组 EP 集合通信超时，
        # 整侧 rank 全被杀掉。这里只登记，由 _advance_decodes 每轮非阻塞推进。
        self._pending_decodes[rid] = (cmd, time.monotonic() + self.args.handoff_timeout)

    def _advance_decodes(self, res_q) -> None:
        for rid, (cmd, deadline) in list(self._pending_decodes.items()):
            meta = cmd["meta"]
            path = pathlib.Path(meta["handoff_file"])
            if not path.exists():
                if time.monotonic() > deadline:
                    self._pending_decodes.pop(rid, None)
                    res_q.put(("error", rid, f"handoff for {rid} never arrived at {path}"))
                continue
            # D 侧 KV 池被在跑的请求占满时不要硬失败，等块释放出来再接管
            # （vLLM 侧等价行为是把请求排在 waiting 里）。上限仍由 deadline 兜底。
            if self.engine.scheduler.get_num_free_gpu_blocks() < int(meta["num_blocks"]) + 1:
                if time.monotonic() > deadline:
                    self._pending_decodes.pop(rid, None)
                    res_q.put(("error", rid, "decode side has no free KV blocks"))
                continue
            pool_frame = None
            try:
                # The mapped storage stays alive after unlink until payload is
                # released, including the synchronous copies made by _adopt.
                if path.suffix == ".pool":
                    payload, pool_frame = self._ipc_receiver.load(path)
                else:
                    payload = self.torch.load(path, map_location="cpu", mmap=True)
            except Exception as exc:  # 半写状态：下一轮重试
                if path.suffix == ".pool" or time.monotonic() > deadline:
                    self._pending_decodes.pop(rid, None)
                    res_q.put(("error", rid, f"cannot read handoff {path}: {exc}"))
                continue
            self._pending_decodes.pop(rid, None)
            try:
                path.unlink()
                self._adopt(rid, cmd, payload, res_q)
            except Exception as exc:
                logger.exception("adopt failed on rank %d rid=%s", self.rank, rid)
                res_q.put(("error", rid, str(exc)))
            if pool_frame is not None:
                self.torch.cuda.current_stream().synchronize()
            del payload
            if pool_frame is not None:
                self._ipc_receiver.acknowledge(path, pool_frame)

    def _adopt(self, rid: str, cmd: dict, payload, res_q) -> None:
        meta = cmd["meta"]
        tokens = list(cmd["tokens"]) + [int(meta["first_token"])]
        max_new = int(cmd.get("max_tokens", 1)) - 1
        seq = self.engine.add_request(
            user_id="pd-decode",
            request_id=rid,
            sampling_params=_sampling_params(
                cmd.get("temperature"), max_new, bool(cmd.get("ignore_eos", False)),
                seed=cmd.get("seed"),
            ),
            prompt_token_ids=self.np.asarray(tokens, dtype=self.np.int32),
            enqueue=False,
        )
        if not self.engine.scheduler.try_allocate_seq(seq):
            raise RuntimeError("decode side ran out of KV blocks")

        dst_ids = list(seq.block_table)
        num_src = int(meta["num_blocks"])
        if len(dst_ids) < num_src:
            raise RuntimeError(f"dst blocks {len(dst_ids)} < src {num_src}")
        device = self.cache_engine.gpu_blocks.device
        src = payload["blocks"].to(device, non_blocking=False)
        expect = (num_src, self.cache_engine.gpu_blocks.shape[1])
        if tuple(src.shape) != expect:
            raise RuntimeError(f"kv shape mismatch {tuple(src.shape)} vs {expect}")
        # gpu_blocks[list] 是 advanced indexing，返回副本；只有 index assignment
        # 才会真正写回 KV cache。
        self.cache_engine.gpu_blocks[dst_ids[:num_src]] = src

        state_copied = False
        if payload.get("state") is not None and int(seq.state_id) >= 0:
            state_dst = self.cache_engine.gpu_states[int(seq.state_id)]
            state_src = payload["state"].to(device, non_blocking=False)
            if tuple(state_dst.shape) != tuple(state_src.shape):
                raise RuntimeError(
                    f"state shape mismatch {tuple(state_dst.shape)} vs {tuple(state_src.shape)}"
                )
            state_dst.copy_(state_src)
            state_copied = True

        # KV 与 state 都已落到本 rank 的槽位，解码进度直接接管 P 的计算进度；
        # 只剩 P 采样的那 1 个 token 需要 D 前向（即第一步 decode）。
        seq.num_cached_tokens = int(meta["num_cached"])
        seq.swa_valid_start = 0
        seq.last_next_n_token = [int(t) for t in meta.get("draft_tokens", [])]
        self.engine.scheduler.adopt_running_seq(seq)
        logger.info(
            "adopt rank=%d rid=%s len=%d cached=%d blocks=%d state_copied=%s",
            self.rank,
            rid,
            len(tokens),
            int(meta["num_cached"]),
            num_src,
            state_copied,
        )

    def _discard_handoff(self, meta: dict) -> None:
        path = pathlib.Path(meta["handoff_file"])
        try:
            if path.suffix == ".pool":
                self._ipc_receiver.discard(path)
            else:
                path.unlink()
        except FileNotFoundError:
            pass

    # -------------------------------- loop -------------------------------

    def _emit(self, output, res_q) -> None:
        rid = str(output.request_id)
        if rid not in self._token_cache:
            return
        token = int(output.token_id)
        self._token_cache[rid].append(token)
        decoder = self._decoders.get(rid)
        if decoder is None:
            decoder = self._decoders[rid] = TokenTextStream(self.tokenizer)
        delta = decoder.push(token, finished=output.finished)
        res_q.put(("token", rid, token, delta))
        if output.finished:
            self._token_cache.pop(rid, None)
            self._texts.pop(rid, None)
            self._decoders.pop(rid, None)
            res_q.put(
                ("finish", rid, _FINISH_REASONS.get(output.finish_reason, "stop"), "")
            )

    def _warn_if_stuck(self) -> None:
        """Engine idle but bookkeeping not empty: our export/adopt path is the blocker."""
        if not self._prefill_seqs and not self._pending_decodes:
            return
        now = time.monotonic()
        if now - self._last_stuck_warn < 5.0:
            return
        self._last_stuck_warn = now
        detail = ",".join(
            f"{rid}:{int(seq.num_cached_tokens)}/{plen}"
            for rid, (seq, plen) in list(self._prefill_seqs.items())[:4]
        )
        logger.warning(
            "rank %d idle but %d prefill(s) pending (%s) and %d decode(s) waiting",
            self.rank,
            len(self._prefill_seqs),
            detail,
            len(self._pending_decodes),
        )

    def run(self, cmd_q, res_q, stop_event) -> None:
        _allow_peer_fd_import()
        while not stop_event.is_set():
            if self._ipc_sender:
                self._ipc_sender.reap()
            worked = False
            while True:
                try:
                    cmd = cmd_q.get_nowait()
                except queue.Empty:
                    break
                worked = True
                if cmd["kind"] == "stop":
                    stop_event.set()
                    break
                if cmd["kind"] == "clear_prefix_cache":
                    if self._prefix_snapshots:
                        self._prefix_snapshots.clear()
                    res_q.put(("cache_cleared", cmd["request_id"], None))
                    continue
                try:
                    if self.role == "prefill":
                        self._submit_prefill(cmd)
                    else:
                        self._submit_decode(cmd, res_q)
                except Exception as exc:  # 单条请求的问题不该拖垮整侧 EP
                    logger.exception("submit failed on rank %d", self.rank)
                    res_q.put(("error", cmd["request_id"], str(exc)))

            if self.role == "decode":
                self._advance_decodes(res_q)

            try:
                # 每轮只能调用一次 has_unfinished_requests：它内部是一次 DP 集合通信
                # （ZOS 下走 gloo/int64），而 scheduler 里还有 int32 的投票集合通信，
                # 一旦各 rank 的调用次数不一致（例如按 worked 分叉），gloo 就会报
                # "Received data size doesn't match expected size. 8 vs 4"。
                busy = self.engine.has_unfinished_requests()
                if busy:
                    self._step_pending = True
                    for pointer in self.engine.step():
                        self._emit(self._RequestOutput.clone(pointer), res_q)
                elif self._step_pending:
                    # ZOS 会把 post-process 推迟一步：最后一个 token 与 finished 标记
                    # 还压在 inflight 缓冲里时，has_unfinished_requests() 就已经是 False。
                    # 必须把 flush 出来的输出也投递出去，否则这条请求的 finish 永远发不
                    # 出去，router/bench 会一直等流结束（表现为整轮压测挂死）。
                    self._step_pending = False
                    for pointer in self.engine.flush_inflight_reqs():
                        self._emit(self._RequestOutput.clone(pointer), res_q)
                if self.role == "prefill":
                    self._export_ready(res_q)
            except Exception:
                logger.exception("engine step failed on rank %d", self.rank)
                res_q.put(("engine_error", "", f"rank {self.rank} step failed"))
                stop_event.set()
                return

            if not busy and not worked:
                self._warn_if_stuck()
                time.sleep(0.001)


def _rank_main(rank: int, args: argparse.Namespace, cmd_q, res_q, stop_event) -> None:
    # 必须在加载模型（DeepEP 初始化）之前放行 pidfd_getfd，否则建 engine 时直接断言失败。
    _allow_peer_fd_import()
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s rank{rank} %(levelname)s %(name)s %(message)s",
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_offset + rank)
    os.environ["DEEPINFER_USE_NEXTN_DECODE"] = str(args.nextn)
    os.environ.setdefault("DEEPINFER_EXPERT_COUNT_WINDOW_SIZE", "3")
    os.environ.setdefault("DEEPINFER_PREFIX_CACHE", "0")
    os.environ.setdefault("DEEPINFER_ACCELERATOR", "nvidia")

    engine = _RankEngine(args, rank)
    if os.environ.get("DEEPINFER_PD_FREEZE_STARTUP_GC", "0") == "1":
        from deepinfer_adapter.startup_gc import freeze_startup_objects

        # This rank owns the engine until process exit. Freeze before readiness
        # so request-specific objects stay in the normal GC generations.
        logger.info("startup GC freeze rank=%d role=%s %s", rank, args.role, freeze_startup_objects())
    res_q.put(("rank_ready", rank))
    engine.run(cmd_q, res_q, stop_event)


# ------------------------------ facade ------------------------------


def _rank_for(tokens: list[int], dp_size: int) -> int:
    """Stable rank choice for a prompt; identical on the prefill and decode side."""
    head = ",".join(str(t) for t in tokens[:32])
    key = f"{len(tokens)}:{head}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big") % dp_size


class Facade:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.queues: list[mp.Queue] = []
        self.res_q: mp.Queue = mp.get_context("spawn").Queue()
        self.pending: dict[str, asyncio.Queue] = {}
        self.prefill_meta: dict[str, dict] = {}
        self.loop: asyncio.AbstractEventLoop | None = None
        self.stop_event = mp.get_context("spawn").Event()
        self.procs: list[mp.Process] = []
        self._lock = threading.Lock()
        self.ready_ranks: set[int] = set()
        snapshot_bytes = int(os.environ.get("PD_PREFIX_SNAPSHOT_BYTES", "0"))
        self.prefix_routes = (PrefixSnapshots(64 * 1024 * 1024)
                              if args.role == "prefill" and snapshot_bytes > 0 else None)

        ctx = mp.get_context("spawn")
        for rank in range(args.dp_size):
            cmd_q = ctx.Queue()
            proc = ctx.Process(
                target=_rank_main,
                args=(rank, args, cmd_q, self.res_q, self.stop_event),
                daemon=False,
            )
            proc.start()
            self.queues.append(cmd_q)
            self.procs.append(proc)

        self._drain = threading.Thread(target=self._drain_loop, name="pd-drain", daemon=True)
        self._drain.start()
        self._watchdog = threading.Thread(target=self._watch_ranks, name="pd-watchdog", daemon=True)
        self._watchdog.start()

    def is_ready(self) -> bool:
        with self._lock:
            initialized = len(self.ready_ranks) == self.args.dp_size
        return (initialized and not self.stop_event.is_set()
                and all(proc.is_alive() for proc in self.procs))

    def _watch_ranks(self) -> None:
        """A rank that dies takes its side down (EP collectives cannot be rebuilt per rank),
        so stop accepting work instead of letting requests hang on a queue nobody drains."""
        while True:
            time.sleep(1.0)
            dead = [i for i, p in enumerate(self.procs) if not p.is_alive()]
            if not dead:
                continue
            self.stop_event.set()
            logger.error("rank process(es) %s exited; failing pending requests", dead)
            with self._lock:
                pending = list(self.pending.items())
                self.pending.clear()
            for rid, q in pending:
                if self.loop is not None:
                    self.loop.call_soon_threadsafe(
                        q.put_nowait, ("engine_error", rid, f"rank(s) {dead} exited")
                    )
            return

    def _drain_loop(self) -> None:
        while True:
            try:
                message = self.res_q.get()
            except (EOFError, OSError):
                return
            if message[0] == "rank_ready":
                with self._lock:
                    self.ready_ranks.add(message[1])
                continue
            rid = message[1]
            with self._lock:
                target = self.pending.get(rid)
                if target is None:
                    continue
                if message[0] in {"finish", "error", "engine_error", "prefill_done", "cache_cleared"}:
                    self.pending.pop(rid, None)
            if self.loop is None:
                logger.warning("dropping result for %s before the loop started", rid)
                continue
            self.loop.call_soon_threadsafe(target.put_nowait, message)

    async def submit(self, rank: int, payload: dict) -> asyncio.Queue:
        rid = payload["request_id"]
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self.pending[rid] = q
        self.queues[rank].put(payload)
        return q

    def close(self) -> None:
        for q in self.queues:
            q.put({"kind": "stop"})
        for proc in self.procs:
            proc.join(timeout=30)


def _build_app(args: argparse.Namespace, facade: Facade):
    app = FastAPI(title=f"deepinfer-pd-{args.role}")
    model_name = args.served_model
    greedy_only = int(os.environ.get("DEEPINFER_SAMPLER_FORCE_TOPK", "0")) == 1
    round_robin_cold = os.environ.get("PD_COLD_ROUND_ROBIN", "0") == "1"
    yoco_keep = (int(os.environ["DEEPINFER_YOCO_PREFILL_KEEP_TOKENS"])
                 if args.role == "prefill" and os.environ.get("PD_YOCO_PREFIX_SNAPSHOTS", "0") == "1" else 0)
    next_cold_rank = 0

    @app.post("/reset_prefix_cache")
    async def reset_prefix_cache():
        nonlocal next_cold_rank
        with facade._lock:
            if facade.pending:
                return JSONResponse({"success": False}, status_code=409)
        if not facade.is_ready():
            return JSONResponse({"success": False}, status_code=503)
        if facade.prefix_routes:
            facade.prefix_routes.clear()
        replies = [await facade.submit(rank, {"kind": "clear_prefix_cache", "request_id": uuid.uuid4().hex})
                   for rank in range(args.dp_size)]
        messages = await asyncio.wait_for(asyncio.gather(*(q.get() for q in replies)), timeout=30)
        success = all(message[0] == "cache_cleared" for message in messages)
        if success:
            next_cold_rank = 0
        return {"success": success}

    @app.get("/health")
    async def health():
        ready = facade.is_ready()
        return JSONResponse(
            {"status": "ok" if ready else "unavailable", "role": args.role, "dp_size": args.dp_size},
            status_code=200 if ready else 503,
        )

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "deepinfer",
                    "root": args.model_path,
                    "max_model_len": args.max_model_len,
                }
            ],
        }

    def _completion_body(rid: str, tokens, text, ids, finish, model) -> dict:
        return {
            "id": f"cmpl-{rid}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "token_ids": ids,
                    "finish_reason": finish,
                    "stop_reason": None,
                }
            ],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": len(ids),
                "total_tokens": len(tokens) + len(ids),
            },
        }

    @app.post("/v1/completions")
    async def completions(request: Request):
        nonlocal next_cold_rank
        if not facade.is_ready():
            return JSONResponse({"error": "DeepInfer ranks are not ready"}, status_code=503)
        body = await request.json()
        prompt = body.get("prompt")
        if not isinstance(prompt, list):
            return JSONResponse(
                {"error": "DeepInfer PD workers accept token-id prompts only"}, status_code=400
            )
        tokens = [int(t) for t in prompt]
        model = body.get("model") or model_name
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 16)
        temperature = body.get("temperature")
        if greedy_only and temperature not in (None, 0):
            return JSONResponse(
                {"error": "This configuration supports temperature=0 only"}, status_code=400,
            )
        ignore_eos = bool(body.get("ignore_eos"))
        stream = bool(body.get("stream"))
        rid = uuid.uuid4().hex
        rank = _rank_for(tokens, args.dp_size)
        prefix_key = None
        cached_route = None
        if args.role == "prefill" and facade.prefix_routes:
            prefix_key = array("i", tokens).tobytes()
            limit = 4 * yoco_snapshot_anchor(len(tokens), yoco_keep) if yoco_keep else None
            cached_route = facade.prefix_routes.find(prefix_key, max_prefix_bytes=limit)
            if cached_route:
                rank = cached_route[1]
        if len(tokens) + max_tokens > args.max_model_len:
            # 超长 prompt 进 scheduler 只会永远留在 waiting（num_cached 一直是 0），
            # 引擎空转、上游流永久等 → 必须当场拒掉。
            return JSONResponse(
                {
                    "error": f"prompt_tokens {len(tokens)} + max_tokens {max_tokens} "
                    f"exceeds max_model_len {args.max_model_len}"
                },
                status_code=400,
            )

        if args.role == "prefill" and round_robin_cold and cached_route is None:
            rank = next_cold_rank
            next_cold_rank = (next_cold_rank + 1) % args.dp_size

        logger.info(
            "%s recv rid=%s tokens=%d max_tokens=%d rank=%d stream=%s",
            args.role,
            rid,
            len(tokens),
            max_tokens,
            rank,
            stream,
        )

        if args.role == "prefill":
            q = await facade.submit(
                rank,
                {
                    "kind": "prefill",
                    "request_id": rid,
                    "tokens": tokens,
                    "temperature": temperature,
                    "seed": body.get("seed"),
                },
            )
            message = await q.get()
            if message[0] in {"error", "engine_error"}:
                return JSONResponse({"error": message[2]}, status_code=500)
            meta = message[2]
            if prefix_key is not None and meta.get("prefix_cache_stored"):
                stored_tokens = meta.get("prefix_stored_tokens", len(tokens))
                facade.prefix_routes.put(prefix_key[:stored_tokens * 4], rank)
            first = int(meta["first_token"])
            text = meta.get("first_text", "")
            return JSONResponse(
                {
                    **_completion_body(rid, tokens, text, [first], "length", model),
                    "kv_transfer_params": {
                        "do_remote_decode": False,
                        "do_remote_prefill": True,
                        "handoff_file": meta["handoff_file"],
                        "first_token": first,
                        "draft_tokens": meta["draft_tokens"],
                        "num_cached": meta["num_cached"],
                        "num_blocks": meta["num_blocks"],
                        "source_rank": meta["rank"],
                        "seed": body.get("seed"),
                        "prefix_cached_tokens": meta.get("prefix_cached_tokens", 0),
                        "prefix_stored_tokens": meta.get("prefix_stored_tokens", 0),
                        # router 一般会原样透传 body；这里再带一份，防止 decode
                        # 侧拿不到 ignore_eos 而提前 EOS 截断。
                        "ignore_eos": ignore_eos,
                    },
                }
            )

        kv = body.get("kv_transfer_params") or {}
        if not kv.get("do_remote_prefill"):
            return JSONResponse(
                {"error": "decode worker needs kv_transfer_params from a prefill response"},
                status_code=400,
            )
        rank = int(kv.get("source_rank", rank))
        if not 0 <= rank < args.dp_size:
            return JSONResponse(
                {"error": f"invalid handoff source rank {rank}"}, status_code=400,
            )

        q = await facade.submit(
            rank,
            {
                "kind": "decode",
                "request_id": rid,
                "tokens": tokens,
                "temperature": temperature,
                "seed": kv.get("seed", body.get("seed")),
                "max_tokens": max_tokens,
                "ignore_eos": ignore_eos or bool(kv.get("ignore_eos")),
                "meta": kv,
            },
        )

        if not stream:
            ids: list[int] = []
            text = ""
            finish = "stop"
            while True:
                message = await q.get()
                if message[0] == "token":
                    ids.append(int(message[2]))
                    text += message[3]
                elif message[0] == "finish":
                    finish = message[2]
                    break
                elif message[0] in {"error", "engine_error"}:
                    return JSONResponse({"error": message[2]}, status_code=500)
            return JSONResponse(_completion_body(rid, tokens, text, ids, finish, model))

        async def relay():
            ids: list[int] = []
            text = ""
            finish = "stop"
            while True:
                message = await q.get()
                if message[0] == "token":
                    ids.append(int(message[2]))
                    text += message[3]
                    chunk = {
                        "id": f"cmpl-{rid}",
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "text": message[3],
                                "token_ids": [int(message[2])],
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif message[0] == "finish":
                    finish = message[2]
                    break
                elif message[0] in {"error", "engine_error"}:
                    yield f"data: {json.dumps({'error': message[2]})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
            final = _completion_body(rid, tokens, text, ids, finish, model)
            final["choices"][0]["token_ids"] = []
            final["choices"][0]["text"] = ""
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(relay(), media_type="text/event-stream")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("prefill", "decode"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model", default="DeepSeek-V4.1-Flash-1017")
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument("--dp-size", type=int, default=4)
    parser.add_argument("--device-offset", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--lookahead-tokens", type=int, default=1)
    parser.add_argument("--nextn", type=int, default=0)
    parser.add_argument("--handoff-dir", default="/dev/shm/di-pd")
    parser.add_argument("--handoff-timeout", type=float, default=180.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    import uvicorn

    facade = Facade(args)
    app = _build_app(args, facade)

    async def _serve():
        facade.loop = asyncio.get_running_loop()
        config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    try:
        asyncio.run(_serve())
    finally:
        facade.close()


if __name__ == "__main__":
    main()
