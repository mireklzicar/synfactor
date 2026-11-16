import unittest
import numpy as np

from experiments.docking.run import diverse_select, synthon_or_bits


class SynthonDiversityTests(unittest.TestCase):
    def test_missing_slot_and_descriptors_do_not_add_bits(self):
        features = np.zeros((4, 516), np.float32)
        features[0, [0, 1, 512]] = 3
        features[1, [1, 2, 513]] = 1
        features[2, [3, 514]] = 2
        features[3, 4] = 1
        rows = np.array([[0, 1, -1], [1, 2, 3]])
        actual = synthon_or_bits([1, 0], rows, features)
        self.assertEqual(actual.shape, (2, 512))
        self.assertEqual(np.flatnonzero(actual[0]).tolist(), [1, 2, 3, 4])
        self.assertEqual(np.flatnonzero(actual[1]).tolist(), [0, 1, 2])

    def test_identical_bits_match_cached_product_acquisition(self):
        rng = np.random.default_rng(137)
        features = rng.integers(0, 4, size=(7, 516)).astype(np.float32)
        features[rng.random(features.shape) < .9] = 0
        rows = np.array([[0, 1, -1], [1, 2, 3], [2, 4, -1], [4, 5, 6], [0, 3, 5]])
        bits = synthon_or_bits(np.arange(len(rows)), rows, features)
        packed = np.packbits(bits.astype(np.uint8), axis=1, bitorder="big")
        ids, scores, references = np.array([1, 2, 3, 4]), np.array([7., 6.8, 6.5, 6.4]), np.array([0])
        old = diverse_select(ids, scores, 3, packed, "big", references, .5, "cpu")
        new = diverse_select(ids, scores, 3, None, None, references, .5, "cpu",
                             synthon_rows=rows, synthon_features=features)
        np.testing.assert_array_equal(old, new)

    def test_only_requested_rows_are_read(self):
        class ShortlistOnly:
            def __getitem__(self, ids):
                np.testing.assert_array_equal(ids, [42, 9])
                return np.array([[0, 1, -1], [1, 1, -1]])
        result = synthon_or_bits([42, 9], ShortlistOnly(), np.ones((2, 516)))
        self.assertEqual(result.shape, (2, 512))


if __name__ == "__main__":
    unittest.main()
