"""Run against the isolated adapter copy, never the active serving sources."""
import asyncio
from array import array
import os
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from deepinfer_adapter.pd_http import _build_app, _rank_for
from deepinfer_adapter.prefix_snapshot import PrefixSnapshots


class ColdRoutingTests(unittest.TestCase):
    def client(self, enabled=True):
        self.requests = []

        async def submit(rank, payload):
            result = asyncio.Queue()
            if payload["kind"] == "clear_prefix_cache":
                result.put_nowait(("cache_cleared", payload["request_id"], None))
            else:
                self.requests.append((rank, payload))
                result.put_nowait(("prefill_done", payload["request_id"], {
                    "first_token": 1, "rank": rank, "draft_tokens": [],
                    "handoff_file": "/unused", "num_cached": 1, "num_blocks": 1,
                }))
            return result

        self.facade = SimpleNamespace(is_ready=lambda: True, prefix_routes=PrefixSnapshots(1024),
                                      pending={}, _lock=threading.Lock(), submit=submit)
        args = SimpleNamespace(role="prefill", dp_size=4, served_model="test", max_model_len=1024)
        with patch.dict(os.environ, {"PD_COLD_ROUND_ROBIN": "1" if enabled else "0"}):
            return TestClient(_build_app(args, self.facade))

    def send(self, client, tokens):
        response = client.post("/v1/completions", json={"prompt": tokens, "max_tokens": 1})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["kv_transfer_params"]["source_rank"], self.requests[-1][0])
        return self.requests[-1][0]

    def test_cold_requests_cycle_without_future_workload_information(self):
        client = self.client()
        self.assertEqual([self.send(client, [1, 2]) for _ in range(10)], [0, 1, 2, 3, 0, 1, 2, 3, 0, 1])

    def test_cached_owner_takes_priority_and_does_not_advance_cold_counter(self):
        client = self.client()
        self.assertEqual(self.send(client, [1, 2]), 0)
        self.facade.prefix_routes.put(array("i", [8, 9]).tobytes(), 3)
        self.assertEqual(self.send(client, [8, 9, 10]), 3)
        self.assertEqual(self.send(client, [4, 5]), 1)

    def test_reset_restarts_cold_rotation_after_cache_clear(self):
        client = self.client()
        self.send(client, [1, 2])
        self.send(client, [3, 4])
        self.assertTrue(client.post("/reset_prefix_cache").json()["success"])
        self.assertEqual(self.send(client, [5, 6]), 0)

    def test_oversize_request_does_not_advance_cold_counter(self):
        client = self.client()
        self.assertEqual(client.post("/v1/completions", json={"prompt": [1] * 1024, "max_tokens": 1}).status_code, 400)
        self.assertEqual(self.send(client, [1, 2]), 0)

    def test_disabled_mode_retains_hash_routing(self):
        client = self.client(False)
        for tokens in [[1, 2], [4, 5, 6], [1, 2]]:
            self.assertEqual(self.send(client, tokens), _rank_for(tokens, 4))


if __name__ == "__main__":
    unittest.main()
