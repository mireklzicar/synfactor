"""Algebra/serialization checks independent of any teacher download."""

import itertools
import unittest

import numpy as np
import torch

from chemistry import assemble, cap_synthon
from model import SynFactor, cartesian_blocks, score_tuples
from score import top_hits


class FactorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_cached_predictions_match_network_for_two_and_three_slots(self):
        model = SynFactor(8, tasks=2).eval()
        x = torch.randn(9, 8)
        tuples = torch.tensor([[0, 3, -1], [1, 4, 7], [2, 5, 8]])
        model.y_mean.copy_(torch.tensor([2.0, -3.0]))
        model.y_scale.copy_(torch.tensor([0.7, 2.0]))
        with torch.no_grad():
            factors = torch.cat(
                [model.factor(x[s * 3 : (s + 1) * 3], s) for s in range(3)]
            )
            a, b = model.affine()
            direct = model(x[tuples.clamp_min(0)], tuples >= 0)
            cached = score_tuples(factors.half(), tuples, a, b)
        torch.testing.assert_close(direct, cached, atol=5e-4, rtol=5e-4)

    def test_cartesian_and_topk_match_independent_tuple_enumeration(self):
        for sizes in ((5, 7), (3, 4, 5)):
            slots = [torch.randn(n, 16) for n in sizes]
            combinations = list(itertools.product(*[range(n) for n in sizes]))
            truth = (
                torch.tensor(
                    [
                        float(
                            torch.stack([slots[s][i] for s, i in enumerate(ids)])
                            .prod(0)
                            .sum()
                        )
                        for ids in combinations
                    ]
                )
                * 0.4
                - 1.3
            )
            matrix = torch.empty(int(np.prod(sizes[:-1])), sizes[-1])
            for i, j, block in cartesian_blocks(slots, 0.4, -1.3, block=2):
                matrix[i : i + len(block), j : j + block.shape[1]] = block
            torch.testing.assert_close(matrix.flatten(), truth, atol=2e-6, rtol=1e-5)
            for minimize in (False, True):
                values, ids = top_hits(slots, 0.4, -1.3, 8, minimize, block=2)
                expected, order = torch.topk(truth, 8, largest=not minimize)
                np.testing.assert_allclose(values, expected.numpy(), atol=2e-6)
                self.assertEqual(
                    ids.tolist(), [list(combinations[i]) for i in order.tolist()]
                )

    def test_missing_slot_uses_multiplicative_identity(self):
        factors = torch.ones(3, 1, 16)
        factors[2] *= 9
        values = score_tuples(factors, torch.tensor([[0, 1, -1], [0, 1, 2]]), 1.0, 0.0)
        torch.testing.assert_close(values[:, 0], torch.tensor([16.0, 144.0]))

    def test_connector_capping_and_assembly(self):
        self.assertEqual(cap_synthon("CN[U]"), "CN")
        self.assertEqual(cap_synthon("c1ccn([U])c1"), "c1cc[nH]c1")
        product = assemble(["CN[U]", "CC(=O)[U]"], "[U]-[*:1].[U]-[*:2]>>[*:1]-[*:2]")
        self.assertEqual(product, "CNC(C)=O")


if __name__ == "__main__":
    unittest.main()
