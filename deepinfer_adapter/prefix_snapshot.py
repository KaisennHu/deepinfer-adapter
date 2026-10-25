# SPDX-License-Identifier: Apache-2.0
"""Bounded snapshots of fully computed prefixes, including their recurrent state."""
from collections import OrderedDict


def yoco_snapshot_anchor(num_tokens: int, keep_tokens: int, grain: int = 128) -> int:
    """Leave the entire consumer replay rectangle after an aligned prefix."""
    if num_tokens < 0 or keep_tokens <= 0 or grain <= 0:
        raise ValueError("Invalid YOCO snapshot geometry")
    return max(0, (num_tokens - keep_tokens) // grain * grain)


class PrefixSnapshots:
    """Match exact token bytes; a snapshot never substitutes for a partial prefix.

    Equal prompts are intentionally misses: their next-token logits have not
    been saved. A longer prompt can resume its suffix from the saved state.
    Payloads are owned by the caller and must remain immutable after insertion.
    """

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.bytes = 0
        self._entries = OrderedDict()

    def find(self, prompt: bytes, *, max_prefix_bytes: int | None = None):
        limit = len(prompt) - 1 if max_prefix_bytes is None else min(max_prefix_bytes, len(prompt) - 1)
        key = max((key for key in self._entries
                   if len(key) <= limit and prompt.startswith(key)), key=len, default=None)
        if key is None:
            return None
        payload, _ = self._entries[key]
        self._entries.move_to_end(key)
        return key, payload

    def put(self, prompt: bytes, payload, payload_bytes: int = 0) -> bool:
        size = len(prompt) + payload_bytes
        if not prompt or size > self.max_bytes:
            return False
        # Prefer the newest, longest prefix of a continuing conversation.
        # Removing an ancestor may cause a fork to miss; it cannot change outputs.
        for key in list(self._entries):
            if prompt.startswith(key):
                self.bytes -= self._entries.pop(key)[1]
        while self._entries and self.bytes + size > self.max_bytes:
            _, (_, evicted_size) = self._entries.popitem(last=False)
            self.bytes -= evicted_size
        self._entries[prompt] = (payload, size)
        self.bytes += size
        return True

    def clear(self):
        self._entries.clear()
        self.bytes = 0
