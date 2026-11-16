import unittest

import numpy as np
import torch

from experiments.docking.factor_model import FactorRegressor


class DockingFactorTests(unittest.TestCase):
    def test_cached_network_agreement_and_absent_third_slot(self):
        rng = np.random.default_rng(5)
        features = rng.normal(size=(24, 6)).astype(np.float32)
        rows = rng.integers(0, 24, size=(80, 3))
        rows[:40, 2] = -1
        reactions = rng.integers(0, 2, 80)
        y = features[rows[:, 0], 0] - features[rows[:, 1], 1]
        for additive in (False, True):
            model = FactorRegressor(features, 2, rank=4, hidden=16, additive=additive, device="cpu")
            model.fit(rows, reactions, y, epochs=2)
            with torch.no_grad():
                direct = model.model(model.x, torch.tensor(rows), torch.tensor(reactions))
                direct = direct.numpy() * model.y_scale + model.y_mean
            np.testing.assert_allclose(model.predict(rows, reactions, batch_size=13), direct, atol=2e-6, rtol=2e-6)
            # Row 0 must never supply the missing third factor.
            cached = model.predict(rows[:40], reactions[:40])
            model.cached_factors[2, 0] = 1e5
            if model.cached_additive is not None:
                model.cached_additive[2, 0] = 1e5
            np.testing.assert_allclose(model.predict(rows[:40], reactions[:40]), cached)

    def test_learns_low_rank_interactions(self):
        rng = np.random.default_rng(7)
        features = rng.normal(size=(35, 3)).astype(np.float32)
        rows = rng.integers(0, 35, size=(700, 3))
        rows[:350, 2] = -1
        y = features[rows[:, 0], 0] * features[rows[:, 1], 1]
        y[350:] *= 1 + .2 * features[rows[350:, 2], 2]
        train, test = np.arange(550), np.arange(550, 700)
        reg = FactorRegressor(features, 1, rank=4, hidden=32, device="cpu", seed=7)
        reg.fit(rows[train], np.zeros(len(train), int), y[train], epochs=100, patience=20, batch_size=256)
        pred = reg.predict(rows[test], np.zeros(len(test), int))
        mse = np.mean((pred-y[test])**2)
        baseline = np.mean((y[train].mean()-y[test])**2)
        self.assertLess(mse, baseline * .5)

    def test_rejects_invalid_indexes(self):
        reg = FactorRegressor(np.ones((4, 2)), 1, device="cpu")
        with self.assertRaises(ValueError):
            reg.predict([[-1, 0, -1]], [0])


if __name__ == "__main__":
    unittest.main()
