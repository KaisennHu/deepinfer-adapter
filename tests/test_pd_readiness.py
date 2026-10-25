# SPDX-License-Identifier: Apache-2.0
import threading
from types import SimpleNamespace
import unittest

from fastapi.testclient import TestClient
from deepinfer_adapter.pd_http import Facade, _build_app


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(role="prefill", dp_size=4, served_model="test")
        self.facade = Facade.__new__(Facade)
        self.facade.args = self.args
        self.facade._lock = threading.Lock()
        self.facade.ready_ranks = set()
        self.facade.stop_event = threading.Event()
        self.facade.procs = [SimpleNamespace(is_alive=lambda: True) for _ in range(4)]
        self.client = TestClient(_build_app(self.args, self.facade))

    def test_all_ranks_must_finish_initializing(self):
        for rank in range(4):
            self.assertEqual(self.client.get("/health").status_code, 503)
            self.facade.ready_ranks.add(rank)
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_rank_exit_or_engine_stop_revokes_readiness(self):
        self.facade.ready_ranks.update(range(4))
        self.facade.procs[1] = SimpleNamespace(is_alive=lambda: False)
        self.assertEqual(self.client.get("/health").status_code, 503)
        self.facade.procs[1] = SimpleNamespace(is_alive=lambda: True)
        self.facade.stop_event.set()
        self.assertEqual(self.client.get("/health").status_code, 503)

    def test_requests_are_rejected_during_loading(self):
        response = self.client.post("/v1/completions", json={"prompt": [1], "max_tokens": 1})
        self.assertEqual(response.status_code, 503)
        self.assertIn("not ready", response.json()["error"])


if __name__ == "__main__":
    unittest.main()
