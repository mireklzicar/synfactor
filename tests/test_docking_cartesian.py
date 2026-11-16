import itertools
import unittest

import numpy as np
import torch

from experiments.docking.factor_model import FactorRegressor, _FactorNetwork
from experiments.docking.cartesian import stream_topk, ReactionGrid, IncompleteSweepError


def fixture(device="cpu", constant=False):
    torch.manual_seed(31)
    model = FactorRegressor(np.ones((15, 3), np.float32), 3, rank=4,
                            additive=True, device=device)
    model.model = _FactorNetwork(3, 3, 4, 8, True).to(device)
    model.cached_factors = torch.randn(3, 15, 4, device=device)
    model.cached_additive = torch.randn(3, 15, device=device)
    model.y_mean, model.y_scale = 3.25, 1.7
    with torch.no_grad():
        model.model.bias.fill_(.4)
        model.model.reaction_bias.weight[:, 0] = torch.tensor([.2, -.3, 1.1], device=device)
        if constant:
            model.cached_factors.zero_(); model.cached_additive.zero_()
            model.model.bias.zero_(); model.model.reaction_bias.weight.zero_()
    grids = [ReactionGrid(2, (np.array([6, 1, 3]), np.array([11, 8]), np.array([0, 14, 9]))),
             ReactionGrid(0, (np.array([10, 4]), np.array([5, 12, 2, 7])))]
    return model, grids


class CartesianTests(unittest.TestCase):
    def test_full_scores_topk_and_exclusions_match_independent_enumeration(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            model, grids = fixture(device)
            rows, reactions = [], []
            for grid in sorted(grids, key=lambda x: x.reaction_index):
                for row in itertools.product(*grid.slots):
                    rows.append(list(row)+[-1]*(3-len(row))); reactions.append(grid.reaction_index)
            rows, reactions = np.asarray(rows), np.asarray(reactions)
            expected = model.predict(rows, reactions)
            excluded = [2, 10, 17]
            eligible = np.array([i for i in range(len(rows)) if i not in excluded])
            order = eligible[np.argsort(-expected[eligible], kind="stable")]
            for k in (7, 100):
                result = stream_topk(model, grids, k, excluded_tuples=rows[excluded],
                                     excluded_reaction_ids=reactions[excluded], block_rows=2, block_cols=2)
                wanted = order[:k]
                np.testing.assert_array_equal(result["tuples"], rows[wanted])
                np.testing.assert_array_equal(result["reaction_ids"], reactions[wanted])
                np.testing.assert_allclose(result["scores"], expected[wanted], atol=3e-6, rtol=2e-6)
                self.assertEqual(result["diagnostics"]["visited_products"], len(rows))
                self.assertTrue(result["diagnostics"]["complete"])

    def test_exact_ties_are_block_size_independent_and_keep_first_cartesian_routes(self):
        model, grids = fixture(constant=True)
        a = stream_topk(model, grids, 13, block_rows=1, block_cols=1)
        b = stream_topk(model, grids, 13, block_rows=64, block_cols=64)
        np.testing.assert_array_equal(a["tuples"], b["tuples"])
        np.testing.assert_array_equal(a["reaction_ids"], b["reaction_ids"])
        self.assertEqual(a["reaction_ids"].tolist(), [0]*8+[2]*5)
        self.assertEqual(a["tuples"][0].tolist(), [10, 5, -1])

    def test_timeout_raises_and_restores_precision(self):
        model, grids = fixture()
        original = torch.backends.cuda.matmul.fp32_precision
        with self.assertRaises(IncompleteSweepError) as context:
            stream_topk(model, grids, 3, max_seconds=1e-12)
        self.assertFalse(context.exception.diagnostics["complete"])
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, original)

    def test_invalid_grid_and_exclusion_rejected(self):
        model, grids = fixture()
        with self.assertRaises(ValueError):
            stream_topk(model, [dict(reaction_index=0, slots=[[1, 1], [2, 3]])])
        with self.assertRaises(ValueError):
            stream_topk(model, grids, excluded_tuples=[[1, 2, -1]], excluded_reaction_ids=[0])


if __name__ == "__main__":
    unittest.main()
