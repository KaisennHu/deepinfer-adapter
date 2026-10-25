# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
from unittest.mock import patch
from array import array
import threading
from types import SimpleNamespace
import unittest

from fastapi.testclient import TestClient
from deepinfer_adapter.pd_http import _build_app, _rank_for
from deepinfer_adapter.prefix_snapshot import PrefixSnapshots


class PrefixRoutingTests(unittest.TestCase):
    def client(self, role, replies):
        self.submitted = []
        async def submit(rank, payload):
            self.submitted.append((rank, payload))
            queue = asyncio.Queue()
            for event in replies(payload):
                queue.put_nowait(event)
            return queue
        self.facade = SimpleNamespace(is_ready=lambda: True, prefix_routes=PrefixSnapshots(1024),
                                      pending={}, _lock=threading.Lock(), submit=submit)
        args = SimpleNamespace(role=role, dp_size=4, served_model="test", max_model_len=1024)
        return TestClient(_build_app(args, self.facade))

    def test_prefill_uses_the_rank_that_owns_a_complete_prefix(self):
        def replies(payload):
            yield ("prefill_done", payload["request_id"], {"first_token": 1, "first_text": "x",
                   "handoff_file": "/tmp/test", "draft_tokens": [], "num_cached": 4, "num_blocks": 1,
                   "rank": owner, "prefix_cache_stored": True, "prefix_cached_tokens": 3})
        client = self.client("prefill", replies)
        tokens = [10, 20, 30, 40]
        owner = (_rank_for(tokens, 4) + 1) % 4
        self.facade.prefix_routes.put(array("i", tokens[:3]).tobytes(), owner)
        response = client.post("/v1/completions", json={"prompt": tokens, "max_tokens": 1, "seed": 123})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.submitted[0][0], owner)
        self.assertEqual(response.json()["kv_transfer_params"]["prefix_cached_tokens"], 3)
        self.assertEqual(self.submitted[0][1]["seed"], 123)
        self.assertEqual(response.json()["kv_transfer_params"]["seed"], 123)

    def test_decode_follows_source_rank_and_rejects_out_of_range_rank(self):
        def replies(payload):
            yield ("token", payload["request_id"], 1, "x")
            yield ("finish", payload["request_id"], "length", "")
        client = self.client("decode", replies)
        tokens = [10, 20, 30, 40]
        source = (_rank_for(tokens, 4) + 1) % 4
        body = {"prompt": tokens, "max_tokens": 1,
                "kv_transfer_params": {"do_remote_prefill": True, "source_rank": source, "seed": 456}}
        self.assertEqual(client.post("/v1/completions", json=body).status_code, 200)
        self.assertEqual(self.submitted[0][0], source)
        self.assertEqual(self.submitted[0][1]["seed"], 456)
        body["kv_transfer_params"]["source_rank"] = 4
        self.assertEqual(client.post("/v1/completions", json=body).status_code, 400)
        self.assertEqual(len(self.submitted), 1)

    def test_singleton_mode_rejects_random_sampling_before_submission(self):
        def replies(payload):
            yield ("prefill_done", payload["request_id"], {
                "first_token": 1, "rank": 0, "draft_tokens": [],
                "handoff_file": "/unused", "num_cached": 1, "num_blocks": 1,
            })
        with patch.dict(os.environ, {"DEEPINFER_SAMPLER_FORCE_TOPK": "1"}):
            client = self.client("prefill", replies)
        body = {"prompt": [1], "max_tokens": 1, "temperature": 0.7}
        self.assertEqual(client.post("/v1/completions", json=body).status_code, 400)
        self.assertEqual(len(self.submitted), 0)
        body["temperature"] = 0
        self.assertEqual(client.post("/v1/completions", json=body).status_code, 200)
        with patch.dict(os.environ, {"DEEPINFER_SAMPLER_FORCE_TOPK": "0"}):
            client = self.client("prefill", replies)
        body["temperature"] = 0.7
        self.assertEqual(client.post("/v1/completions", json=body).status_code, 200)
        self.assertEqual(self.submitted[0][1]["temperature"], 0.7)

    def test_cache_reset_rejects_active_requests(self):
        client = self.client("prefill", lambda payload: [])
        self.facade.pending["active"] = object()
        response = client.post("/reset_prefix_cache")
        self.assertEqual(response.status_code, 409)
        self.assertFalse(response.json()["success"])


if __name__ == "__main__":
    unittest.main()
