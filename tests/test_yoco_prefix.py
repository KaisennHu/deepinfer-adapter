"""CPU checks of cache identity, state ownership, replay bounds and HTTP routing."""
from array import array
import asyncio
import os
from pathlib import Path
import queue
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from fastapi.testclient import TestClient

from deepinfer_adapter.pd_http import _RankEngine, _build_app
from deepinfer_adapter.prefix_snapshot import PrefixSnapshots, yoco_snapshot_anchor


class YocoPrefixTests(unittest.TestCase):
    def test_all_supported_lengths_preserve_ring_and_replay_history(self):
        for length in range(1, 262145):
            anchor = yoco_snapshot_anchor(length, 140)
            if not anchor:
                self.assertLess(length, 268)
                continue
            self.assertEqual(anchor % 128, 0)
            self.assertGreaterEqual(length - anchor, 140)
            self.assertLessEqual(length - anchor, 267)
            self.assertGreaterEqual(anchor - 134, length - 402)
            for extra in (0, 1, 7, 128, 4096):
                self.assertLessEqual(anchor, length + extra - 140)

    def test_invalid_geometry(self):
        for arguments in [(-1, 140), (512, 0), (512, 140, 0)]:
            with self.assertRaises(ValueError):
                yoco_snapshot_anchor(*arguments)

    def test_short_fork_cannot_read_past_its_replay_limit(self):
        cache = PrefixSnapshots(10000)
        prefix = array('i', range(256)).tobytes()
        cache.put(prefix, 'long')
        short_fork = prefix + array('i', [999]).tobytes()
        self.assertIsNone(cache.find(short_fork, max_prefix_bytes=128 * 4))
        original = prefix + array('i', [998] * 140).tobytes()
        self.assertEqual(cache.find(original, max_prefix_bytes=256 * 4), (prefix, 'long'))
        cache.clear()
        self.assertIsNone(cache.find(original))

    def make_rank(self):
        rank = _RankEngine.__new__(_RankEngine)
        rank.np, rank.torch, rank.rank, rank.block_size = np, torch, 0, 128
        rank._yoco_snapshot_keep = 140
        rank._ipc_sender = rank._ipc_receiver = None
        rank._prefix_snapshots = PrefixSnapshots(100000)
        rank._prefix_keys, rank._prefix_hits, rank._prefill_seqs = {}, {}, {}
        rank.cache_engine = SimpleNamespace(gpu_blocks=torch.zeros((8, 4), dtype=torch.uint8),
                                           gpu_states=torch.full((1, 12), 99, dtype=torch.uint8))
        rank.tokenizer = SimpleNamespace(decode=lambda ids: str(ids))
        return rank

    def test_export_owns_aligned_prefix_separately_from_full_handoff(self):
        rank = self.make_rank()
        rank.cache_engine.gpu_blocks[:] = torch.arange(32, dtype=torch.uint8).reshape(8, 4)
        rank.engine = SimpleNamespace(abort_request=lambda rid: None)
        seq = SimpleNamespace(block_table=[0, 1, 2, 3], num_cached_tokens=396, state_id=0,
                              data=SimpleNamespace(get_last_token_id=lambda: 7), last_next_n_token=[])
        rank._prefix_keys['r'] = array('i', range(396)).tobytes()
        calls = []

        def select(state_id, position, states, TO_CE):
            calls.append((position, TO_CE))
            return torch.full((12 if TO_CE else 4,), 11 if TO_CE else 22, dtype=torch.uint8)

        fake = SimpleNamespace(GPUCacheInterface=SimpleNamespace(select_state_slice_to_store=select))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'deepinfer.worker.cache_engine': fake}):
            rank.handoff_dir = Path(directory)
            replies = queue.Queue()
            rank._export_one('r', seq, replies)
            meta = replies.get()[2]
            self.assertEqual(meta['prefix_stored_tokens'], 256)
            handoff = torch.load(meta['handoff_file'], weights_only=True)
            self.assertEqual(tuple(handoff['blocks'].shape), (4, 4))
            self.assertEqual(handoff['state'].tolist(), [11] * 12)
            key, snapshot = rank._prefix_snapshots.find(array('i', range(396)).tobytes())
            self.assertEqual(key, array('i', range(256)).tobytes())
            self.assertEqual(snapshot['state'].tolist(), [22] * 4)
            self.assertEqual(snapshot['blocks'].tolist(), handoff['blocks'][:2].tolist())
            self.assertEqual(snapshot['blocks'].untyped_storage().nbytes(), snapshot['blocks'].numel())
            self.assertEqual(rank._prefix_snapshots.bytes, len(key) + 8 + 4)
        self.assertEqual(calls, [(396, True), (256, False)])

    def test_restore_uses_native_producer_layout_and_clears_dropped_state(self):
        rank = self.make_rank()
        prefix = array('i', range(256)).tobytes()
        rank._prefix_snapshots.put(prefix, {'blocks': torch.full((2, 4), 17, dtype=torch.uint8),
                                          'state': torch.full((4,), 23, dtype=torch.uint8)})
        seq = SimpleNamespace(block_table=[3, 4, 5, 6], state_id=0, num_cached_tokens=0, swa_valid_start=-1)
        enqueued = []
        rank.engine = SimpleNamespace(add_request=lambda **kwargs: seq,
                                      scheduler=SimpleNamespace(try_allocate_seq=lambda seq: True,
                                                                add_sequence=enqueued.append))
        calls = []

        def restore(state_id, anchor, state, all_states):
            self.assertEqual(all_states.count_nonzero().item(), 0)
            calls.append((state_id, anchor, state.tolist()))
            all_states[state_id, :4] = state

        fake = SimpleNamespace(GPUCacheInterface=SimpleNamespace(state_size_to_disk=4,
                                    load_state_from_store_to_prefill_cache=restore))
        with patch.dict(sys.modules, {'deepinfer.worker.cache_engine': fake}), \
                patch('deepinfer_adapter.pd_http._sampling_params', return_value=None):
            rank._submit_prefill({'request_id': 'r', 'tokens': list(range(396))})
        self.assertEqual(calls, [(0, 256, [23] * 4)])
        self.assertEqual(seq.num_cached_tokens, 256)
        self.assertEqual(seq.swa_valid_start, 0)
        self.assertEqual(rank.cache_engine.gpu_blocks[3:5].tolist(), [[17] * 4] * 2)
        self.assertEqual(rank.cache_engine.gpu_states[0].tolist(), [23] * 4 + [0] * 8)
        self.assertEqual(enqueued, [seq])

    def test_http_routes_by_actual_anchor_and_rejects_too_short_replay(self):
        submissions = []

        async def submit(rank, payload):
            submissions.append(rank)
            q = asyncio.Queue()
            anchor = yoco_snapshot_anchor(len(payload['tokens']), 140)
            q.put_nowait(('prefill_done', payload['request_id'], {
                'first_token': 1, 'first_text': 'x', 'draft_tokens': [], 'rank': rank,
                'handoff_file': '/unused', 'num_cached': len(payload['tokens']), 'num_blocks': 4,
                'prefix_cache_stored': anchor > 0, 'prefix_stored_tokens': anchor,
            }))
            return q

        facade = SimpleNamespace(is_ready=lambda: True, prefix_routes=PrefixSnapshots(10000),
                                 pending={}, _lock=threading.Lock(), submit=submit)
        args = SimpleNamespace(role='prefill', dp_size=4, served_model='test', max_model_len=1024)
        with patch.dict(os.environ, {'PD_YOCO_PREFIX_SNAPSHOTS': '1',
                                    'DEEPINFER_YOCO_PREFILL_KEEP_TOKENS': '140', 'PD_COLD_ROUND_ROBIN': '1'}):
            client = TestClient(_build_app(args, facade))
        tokens = list(range(396))
        for prompt in (tokens, tokens, tokens[:257]):
            response = client.post('/v1/completions', json={'prompt': prompt, 'max_tokens': 1})
            self.assertEqual(response.status_code, 200)
        self.assertEqual(submissions, [0, 0, 1])
        self.assertEqual(list(facade.prefix_routes._entries), [array('i', tokens[:256]).tobytes()])


if __name__ == '__main__':
    unittest.main()
