"""Independent Gaussian/logdet checks for approximate CP batch information."""
import unittest

import numpy as np
import torch

from experiments.docking.factor_model import FactorRegressor
from experiments.docking.information import LinearInformation


class DockingInformationTests(unittest.TestCase):
    def test_greedy_gains_equal_independent_logdet_with_unlabeled_conditioning(self):
        rng = np.random.default_rng(13)
        train = rng.normal(size=(21, 5))
        y = rng.normal(size=len(train))
        pool = rng.normal(size=(12, 5))
        condition = rng.normal(size=(4, 5))
        acquisition = LinearInformation(train, y, y+.1, noise_variance=.3, prior_precision=2.)
        chosen, info = acquisition.select(pool, np.zeros(len(pool)), 7, quality_weight=0,
                                         condition_features=condition)
        scale = acquisition.scale.numpy().astype(float)
        xt, xc, xp = train/scale, condition/scale, pool/scale
        precision = .3*2*np.eye(5) + xt.T@xt + xc.T@xc
        posterior = np.linalg.inv(precision)
        # Independent determinant lemma, using a candidate Gram matrix only in
        # this small test. Production never constructs a pool-square matrix.
        gram = xp[chosen]@posterior@xp[chosen].T
        expected = .5*np.linalg.slogdet(np.eye(len(chosen))+gram)[1]
        self.assertAlmostEqual(info['total_conditional_gain_nats'], expected, places=6)
        np.testing.assert_allclose(info['initial_mean_latent_variance'],
                                   .3*np.mean(np.sum((xp@posterior)*xp, axis=1)), rtol=2e-6)
        self.assertLess(info['final_mean_latent_variance'], info['initial_mean_latent_variance'])
        self.assertEqual(info['conditioned_count'], 4)
        # Calling select does not mutate the fitted acquired-only precision.
        again, repeated = acquisition.select(pool, np.zeros(len(pool)), 7, quality_weight=0,
                                              condition_features=condition)
        np.testing.assert_array_equal(chosen, again)
        self.assertEqual(info['conditional_gain_nats'], repeated['conditional_gain_nats'])

    def test_duplicate_information_diminishes_and_batch_avoids_redundancy(self):
        train = np.eye(2)
        acquisition = LinearInformation(train, [0., 1.], [0., 1.], noise_variance=1.)
        pool = np.array([[1., 0.], [1., 0.], [0., 1.]])
        chosen, info = acquisition.select(pool, [0., 0., 0.], 3, quality_weight=0)
        np.testing.assert_array_equal(chosen, [0, 2, 1])
        self.assertLess(info['conditional_gain_nats'][2], info['conditional_gain_nats'][0])
        conditioned, reduced = acquisition.select(pool, [0., 0., 0.], 1,
                                                  quality_weight=0, condition_features=pool[:1])
        self.assertEqual(conditioned.tolist(), [2])
        self.assertLess(reduced['initial_mean_latent_variance'], info['initial_mean_latent_variance'])

    def test_redundant_onehot_intercept_stays_stable_at_low_noise(self):
        # Intercept equals sum of one-hot columns: exactly singular feature Gram.
        base = np.array([[1., 0., 1.], [0., 1., 1.]])
        train = np.tile(base, (50, 1))
        y = np.tile([0., 1.], 50)
        acquisition = LinearInformation(train, y, y, noise_variance=1e-6)
        chosen, info = acquisition.select(base, [0., 0.], 2, quality_weight=0)
        self.assertEqual(len(set(chosen)), 2)
        # Fifty identical observations of each category give variance/noise
        # approaching 1/50 as observation noise tends to zero.
        self.assertAlmostEqual(info['initial_mean_latent_variance']/1e-6, 1/50, places=6)
        self.assertTrue(np.isfinite(info['total_conditional_gain_nats']))

    def test_quality_only_and_empirical_noise_use_acquired_inputs(self):
        train = np.array([[1., 2.], [3., 1.], [-1., .5]])
        observed, predictions = np.array([1., 3., 2.]), np.array([.5, 3.5, 2.])
        acquisition = LinearInformation(train, observed, predictions)
        self.assertAlmostEqual(acquisition.noise_variance, 1/6)
        pool = np.array([[100., 0.], [0., 1.], [1., 1.]])
        before = acquisition.scale.clone()
        chosen, _ = acquisition.select(pool, [0., 2., 1.], 2, information_weight=0, quality_weight=1)
        np.testing.assert_array_equal(chosen, [1, 2])
        torch.testing.assert_close(before, acquisition.scale)
        expected_scale = np.sqrt(np.mean(train**2, axis=0))
        np.testing.assert_allclose(acquisition.scale.numpy(), expected_scale, rtol=1e-6)
        with self.assertRaises(ValueError):
            acquisition.select(pool, [0., 2., 1.], 4)
        with self.assertRaises(ValueError):
            acquisition.select(pool, [0., 2., np.nan], 1)
        with self.assertRaises(ValueError):
            acquisition.select(pool, [0., 2., 1.], 1, information_weight=0, quality_weight=0)

    def test_cp_features_reconstruct_predictions_and_mask_missing_third_slot(self):
        rng = np.random.default_rng(18)
        features = rng.normal(size=(12, 4)).astype(np.float32)
        rows = rng.integers(0, 12, size=(30, 3))
        rows[:15, 2] = -1
        reactions = rng.integers(0, 3, len(rows))
        y = rng.normal(size=len(rows)).astype(np.float32)
        for additive in (False, True):
            model = FactorRegressor(features, 3, rank=4, hidden=8, additive=additive, device='cpu')
            model.fit(rows, reactions, y, epochs=1)
            phi = model.acquisition_features(rows, reactions, batch_size=7)
            weights = np.concatenate([np.ones(model.rank+(3 if additive else 0)),
                model.model.reaction_bias.weight.detach().numpy().ravel(),
                [float(model.model.bias.detach())]])
            reconstruction = (phi@weights)*model.y_scale + model.y_mean
            np.testing.assert_allclose(reconstruction, model.predict(rows, reactions), atol=1e-6)
            if additive:
                np.testing.assert_array_equal(phi[:15, model.rank+2], 0)
            original = phi[:15].copy()
            model.cached_factors[2, 0] = 1e8
            if additive:
                model.cached_additive[2, 0] = 1e8
            np.testing.assert_array_equal(model.acquisition_features(rows[:15], reactions[:15]), original)


if __name__ == '__main__':
    unittest.main()
