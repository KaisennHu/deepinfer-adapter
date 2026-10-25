# SPDX-License-Identifier: Apache-2.0
import unittest

from deepinfer_adapter.pd_http import _sampling_params


class SamplingContractTests(unittest.TestCase):
    def test_zero_temperature_survives_native_parameter_normalization_as_greedy(self):
        for temperature in [None, 0, 0.0]:
            params = _sampling_params(temperature, 250, True, seed=879805729)
            self.assertEqual(params.top_k, 1)
            self.assertEqual(params.top_p, 1.0)
            self.assertGreater(params.temperature, 0)
            self.assertEqual(params.seq_seed, 879805729)
            self.assertEqual(params.max_tokens, 250)
            self.assertTrue(params.ignore_eos)

    def test_nonzero_temperature_and_explicit_zero_seed_are_preserved(self):
        params = _sampling_params(0.7, 8, False, seed=0)
        self.assertEqual(params.temperature, 0.7)
        self.assertEqual(params.top_k, -1)
        self.assertEqual(params.seq_seed, 0)
        self.assertFalse(params.ignore_eos)


if __name__ == "__main__":
    unittest.main()
