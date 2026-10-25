# SPDX-License-Identifier: Apache-2.0
import unittest

from deepinfer_adapter.prefix_snapshot import PrefixSnapshots


class PrefixSnapshotTests(unittest.TestCase):
    def test_requires_an_exact_strict_prefix(self):
        cache = PrefixSnapshots(100)
        cache.put(b"abcd", "state")
        self.assertIsNone(cache.find(b"abcd"))
        self.assertIsNone(cache.find(b"abce-more"))
        self.assertIsNone(cache.find(b"abc"))
        self.assertEqual(cache.find(b"abcd-more"), (b"abcd", "state"))

    def test_extension_replaces_ancestor_and_preserves_other_conversations(self):
        cache = PrefixSnapshots(100)
        cache.put(b"ab", "old", 10)
        cache.put(b"xy", "other", 10)
        cache.put(b"abcd", "new", 10)
        self.assertEqual(cache.bytes, 26)
        self.assertEqual(cache.find(b"abcdef"), (b"abcd", "new"))
        self.assertIsNone(cache.find(b"ab-other-fork"))
        self.assertEqual(cache.find(b"xyz"), (b"xy", "other"))

    def test_budget_counts_payload_and_evicts_lru(self):
        cache = PrefixSnapshots(24)
        cache.put(b"ab", "a", 10)
        cache.put(b"xy", "b", 10)
        cache.find(b"abc")
        cache.put(b"pq", "c", 10)
        self.assertIsNone(cache.find(b"xyz"))
        self.assertEqual(cache.find(b"abc"), (b"ab", "a"))
        self.assertEqual(cache.bytes, 24)
        self.assertFalse(cache.put(b"oversized", "large", 100))
        self.assertEqual(cache.bytes, 24)
        cache.clear()
        self.assertEqual(cache.bytes, 0)
        self.assertIsNone(cache.find(b"abc"))


if __name__ == "__main__":
    unittest.main()
