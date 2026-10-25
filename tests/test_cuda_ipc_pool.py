import json
from pathlib import Path
import tempfile
import unittest

from deepinfer_adapter.cuda_ipc_pool import Leases, Receiver


class PoolProtocolTests(unittest.TestCase):
    def test_bounded_and_stale_ack(self):
        leases = Leases(2, 2048)
        a, b = leases.reserve(2048), leases.reserve(1)
        self.assertIsNone(leases.reserve(1))
        self.assertIsNone(leases.reserve(2049))
        self.assertIsNone(leases.reserve(0))
        leases.acknowledge(a)
        c = leases.reserve(256)
        self.assertEqual(c["slot"], a["slot"])
        self.assertGreater(c["generation"], a["generation"])
        with self.assertRaises(ValueError):
            leases.acknowledge(a)
        self.assertEqual(len(leases.busy), 2)
        leases.acknowledge(c)
        leases.acknowledge(b)
        with self.assertRaises(ValueError):
            leases.acknowledge(b)

    def test_discard_validates_and_acknowledges_exact_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            receiver = Receiver.__new__(Receiver)
            receiver.descriptor = Path(directory) / "pool.ipc"
            receiver.generations = [0, 0]
            receiver.max_blocks = 2048
            frame = {"pool": str(receiver.descriptor), "slot": 0, "generation": 1, "blocks": 128, "state": True}
            path = Path(directory) / "request.pool"
            path.write_text(json.dumps(frame))
            receiver.discard(path)
            self.assertFalse(path.exists())
            self.assertEqual(json.loads(Path(str(path) + ".ack").read_text()), frame)
            for changes in ({}, {"pool": "other"}, {"slot": -1}, {"slot": True}, {"blocks": 2049}, {"state": 1}):
                path.write_text(json.dumps(dict(frame, **changes)))
                with self.assertRaises(ValueError):
                    receiver.frame(path)
            path.write_text(json.dumps(dict(frame, generation=2, blocks=1, state=False)))
            self.assertEqual(receiver.frame(path)["generation"], 2)


if __name__ == "__main__":
    unittest.main()
