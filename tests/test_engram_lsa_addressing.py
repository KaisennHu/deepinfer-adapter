"""Addressing check for the patch in ``patches/deepep-engram-lsa-fetch.patch``.

The LSA fetch kernel computes where an Engram entry lives from its global index.
A wrong offset there does not crash -- it silently returns a neighbouring
embedding -- so it is worth checking the arithmetic without a GPU. This mirrors
the kernel literally: same shard split, same byte offsets, same skip rules.

The real kernel is exercised only on hardware; this test pins down the layout.
"""

import unittest

HIDDEN = 8              # stand-in for the real 256-byte vector
ENTRIES_PER_TOKEN = 3
NUM_RANKS = 8
ENTRIES_PER_LAYER = [7, 5]   # deliberately unequal, like the real 2-layer table
MAX_TOKENS = 4
EMPTY = 255             # marks a slot that must never be written


def _encode(rank: int, layer: int, local: int) -> bytes:
    return bytes([rank, layer, local % 251] + [0] * (HIDDEN - 3))


def _shard(rank: int) -> bytes:
    """One rank's storage: layers packed back to back, HIDDEN bytes per entry."""
    buf = bytearray([EMPTY] * (sum(ENTRIES_PER_LAYER) * HIDDEN))
    prefix = 0
    for layer, count in enumerate(ENTRIES_PER_LAYER):
        for local in range(count):
            off = (prefix + local) * HIDDEN
            buf[off:off + HIDDEN] = _encode(rank, layer, local)
        prefix += count
    return bytes(buf)


def _storage_offset(layer: int, local: int) -> int:
    return (sum(ENTRIES_PER_LAYER[:layer]) + local) * HIDDEN


def _recv_offset(layer: int, entry_idx: int) -> int:
    return (layer * ENTRIES_PER_TOKEN * MAX_TOKENS + entry_idx) * HIDDEN


def _fetch(indices, num_tokens: int) -> bytes:
    shards = [_shard(r) for r in range(NUM_RANKS)]
    recv = bytearray([EMPTY] * (len(ENTRIES_PER_LAYER) * ENTRIES_PER_TOKEN * MAX_TOKENS * HIDDEN))
    for layer, per_layer in enumerate(ENTRIES_PER_LAYER):
        for entry_idx in range(num_tokens * ENTRIES_PER_TOKEN):
            global_idx = indices[layer][entry_idx]
            if global_idx < 0:
                continue
            rank, local = divmod(global_idx, per_layer)
            if rank >= NUM_RANKS:
                continue
            src = _storage_offset(layer, local)
            dst = _recv_offset(layer, entry_idx)
            recv[dst:dst + HIDDEN] = shards[rank][src:src + HIDDEN]
    return bytes(recv)


class EngramLsaAddressingTest(unittest.TestCase):
    def test_entries_land_in_the_right_shard_and_slot(self):
        num_tokens = 3
        indices = []
        for layer, per_layer in enumerate(ENTRIES_PER_LAYER):
            total = per_layer * NUM_RANKS
            layer_idx = []
            for entry_idx in range(num_tokens * ENTRIES_PER_TOKEN):
                if entry_idx % 5 == 4:
                    layer_idx.append(-1)             # miss
                elif entry_idx == 7:
                    layer_idx.append(total + 3)      # beyond the last rank
                else:
                    layer_idx.append((entry_idx * 11 + layer * 3) % total)
            indices.append(layer_idx)

        recv = _fetch(indices, num_tokens)

        checked = skipped = 0
        for layer, per_layer in enumerate(ENTRIES_PER_LAYER):
            total = per_layer * NUM_RANKS
            for entry_idx in range(num_tokens * ENTRIES_PER_TOKEN):
                global_idx = indices[layer][entry_idx]
                got = recv[_recv_offset(layer, entry_idx):][:HIDDEN]
                if global_idx < 0 or global_idx >= total:
                    self.assertEqual(got, bytes([EMPTY] * HIDDEN))
                    skipped += 1
                    continue
                rank, local = divmod(global_idx, per_layer)
                self.assertEqual(got, _encode(rank, layer, local))
                checked += 1

        self.assertEqual(checked, 14)
        self.assertEqual(skipped, 4)


if __name__ == "__main__":
    unittest.main()
