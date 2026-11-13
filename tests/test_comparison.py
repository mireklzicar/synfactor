"""Scientific invariants for the pilot baselines."""
from collections import Counter
from itertools import product
import unittest

import numpy as np

from experiments.baselines import basis_indices, integer_histogram, shi_predict


class ComparisonTests(unittest.TestCase):
    def test_basis_equation_and_interaction(self):
        rows = np.array(list(product(range(3), range(4), range(2))))
        basis, refs = basis_indices(rows, ["r"] * len(rows))
        additive = (5 + rows @ np.array([2., -3., 4.]))[:, None]
        np.testing.assert_allclose(shi_predict(additive, basis, refs), additive)
        interaction = additive + (rows[:, 0] * rows[:, 1])[:, None]
        pred = shi_predict(interaction, basis, refs)
        np.testing.assert_allclose(pred, additive)
        self.assertGreater(np.max(np.abs(pred - interaction)), 0)

    def test_two_slots_and_noninteger_rejection(self):
        rows = np.array([(a, b, -1) for a, b in product(range(2), range(3))])
        basis, refs = basis_indices(rows, ["r"] * len(rows))
        y = (7 + 2 * rows[:, 0] - rows[:, 1])[:, None]
        np.testing.assert_allclose(shi_predict(y, basis, refs), y)
        with self.assertRaises(ValueError):
            integer_histogram([[.5]])

    def test_histogram_matches_enumeration_with_negative_contributions(self):
        slots = [[-2, 0, 0, 1], [0, 3], [-1, 2]]
        expected = Counter(sum(x) + 8 for x in product(*slots))
        self.assertEqual(integer_histogram(slots, 8), expected)


if __name__ == "__main__":
    unittest.main()
