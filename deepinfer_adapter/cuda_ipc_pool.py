"""Single-consumer GPU pool for same-rank local PD handoff.

Map once before readiness. Keep P alive until D exits. Slots are reusable only
after D synchronizes its reads and ACKs the exact generation; missing ACKs keep
slots occupied and force CPU fallback. Worker restart is not supported.
"""
import json
import os
from pathlib import Path


def _write(path, data):
    temporary = Path(str(path) + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


class Leases:
    def __init__(self, slots, max_blocks):
        if slots <= 0 or max_blocks <= 0:
            raise ValueError("Pool dimensions must be positive")
        self.max_blocks = max_blocks
        self.generations = [0] * slots
        self.busy = {}
        self.next_slot = 0

    def reserve(self, count):
        if count <= 0 or count > self.max_blocks:
            return None
        for offset in range(len(self.generations)):
            slot = (self.next_slot + offset) % len(self.generations)
            if slot not in self.busy:
                self.generations[slot] += 1
                self.busy[slot] = self.generations[slot]
                self.next_slot = (slot + 1) % len(self.generations)
                return {"slot": slot, "generation": self.busy[slot], "blocks": count}
        return None

    def acknowledge(self, frame):
        if self.busy.get(frame["slot"]) != frame["generation"]:
            raise ValueError("Unknown, duplicate or stale ACK")
        del self.busy[frame["slot"]]


class Sender:
    def __init__(self, descriptor, block_bytes, state_bytes, max_blocks, slots=2):
        import torch
        import torch.multiprocessing  # noqa: F401 registers CUDA reducers
        from multiprocessing.reduction import ForkingPickler

        self.descriptor = Path(descriptor)
        if self.descriptor.exists():
            raise ValueError("Pool descriptor already exists; use a unique run ID")
        self.leases = Leases(slots, max_blocks)
        self.block_bytes, self.state_bytes = block_bytes, state_bytes
        self.block_span = max_blocks * block_bytes
        self.pending = {}
        self.bytes = slots * (self.block_span + state_bytes)
        if block_bytes <= 0 or state_bytes < 0 or self.bytes > 1024**3:
            raise ValueError("Invalid pool dimensions or pool exceeds 1 GiB per rank")
        self.pool = torch.empty((slots, self.block_span + state_bytes), dtype=torch.uint8, device="cuda")
        _write(self.descriptor, bytes(ForkingPickler.dumps(self.pool)))

    def reap(self):
        released = 0
        for path, frame in list(self.pending.items()):
            ack = Path(str(path) + ".ack")
            if not ack.exists():
                continue
            if json.loads(ack.read_text()) != frame:
                raise ValueError("ACK does not match pending frame")
            self.leases.acknowledge(frame)
            ack.unlink()
            del self.pending[path]
            released += 1
        return released

    def publish(self, path, blocks, state):
        import torch

        path = Path(path)
        self.reap()
        if path in self.pending or path.exists() or Path(str(path) + ".ack").exists():
            raise ValueError("Duplicate pool handoff path")
        if blocks.dtype != torch.uint8 or blocks.ndim != 2 or blocks.shape[1] != self.block_bytes or not blocks.is_cuda:
            raise ValueError("Invalid CUDA KV blocks")
        if state is not None and (state.dtype != torch.uint8 or state.numel() != self.state_bytes or not state.is_cuda):
            raise ValueError("Invalid CUDA state")
        frame = self.leases.reserve(len(blocks))
        if frame is None:
            return False
        frame.update(pool=str(self.descriptor), state=state is not None)
        # Keep failed publications quarantined; never recycle an exported frame by timeout.
        self.pending[path] = frame
        slot = self.pool[frame["slot"]]
        slot[:blocks.numel()].view_as(blocks).copy_(blocks)
        if state is not None:
            slot[self.block_span:].copy_(state.reshape(-1))
        torch.cuda.current_stream().synchronize()
        _write(path, json.dumps(frame).encode())
        return True


class Receiver:
    def __init__(self, descriptor, block_bytes, state_bytes, max_blocks, slots=2):
        import torch
        import torch.multiprocessing  # noqa: F401 registers CUDA reducers
        from multiprocessing.reduction import ForkingPickler

        self.descriptor = Path(descriptor)
        self.block_bytes, self.state_bytes, self.max_blocks = block_bytes, state_bytes, max_blocks
        self.block_span = max_blocks * block_bytes
        self.pool = ForkingPickler.loads(self.descriptor.read_bytes())
        if self.pool.shape != (slots, self.block_span + state_bytes) or self.pool.dtype != torch.uint8 or not self.pool.is_cuda:
            raise ValueError("Pool layout mismatch")
        self.generations = [0] * slots

    def frame(self, path):
        frame = json.loads(Path(path).read_text())
        if set(frame) != {"pool", "slot", "generation", "blocks", "state"}:
            raise ValueError("Invalid pool frame fields")
        slot, generation, count = frame["slot"], frame["generation"], frame["blocks"]
        if (frame["pool"] != str(self.descriptor) or type(slot) is not int or not 0 <= slot < len(self.generations)
                or type(generation) is not int or generation <= self.generations[slot]
                or type(count) is not int or not 0 < count <= self.max_blocks or type(frame["state"]) is not bool):
            raise ValueError("Wrong pool, stale generation or invalid bounds")
        self.generations[slot] = generation
        return frame

    def load(self, path):
        frame = self.frame(path)
        slot = self.pool[frame["slot"]]
        payload = {"blocks": slot[:frame["blocks"] * self.block_bytes].view(frame["blocks"], self.block_bytes),
                   "state": slot[self.block_span:] if frame["state"] else None}
        return payload, frame

    @staticmethod
    def acknowledge(path, frame):
        # Caller must finish reads and release payload views before ACK.
        ack = Path(str(path) + ".ack")
        if ack.exists():
            raise ValueError("Duplicate ACK")
        _write(ack, json.dumps(frame).encode())

    def discard(self, path):
        frame = self.frame(path)
        Path(path).unlink()
        self.acknowledge(path, frame)
