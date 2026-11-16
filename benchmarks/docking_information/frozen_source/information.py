"""Cheap information-directed batches for frozen neural CP representations.

This is conditional Bayesian linear information gain, NOT exact information
about neural-network weights or docking outcomes. For acquired features Phi,
we use a Gaussian linear model with isotropic weight precision lambda and
homoscedastic residual variance sigma². Its precision is
    Lambda = lambda I + Phi.T Phi / sigma².
A noisily observed candidate contributes 0.5 log(1 + phi.T Lambda^-1 phi /
sigma²) nats. Greedy Sherman–Morrison updates condition the remaining batch
without its labels. Neural-network predictions remain the quality objective;
this module never fits or substitutes a new predictive mean.

Use only a bounded candidate pool, e.g. 10k elites plus 10k random candidates.
The covariance and cached candidate-feature matrix cost O(d² + pool_size*d),
not a pool_size² kernel. Recreate after each neural-network refit.
"""
from __future__ import annotations

import math
import numpy as np
import torch


class LinearInformation:
    """Frozen-feature Gaussian uncertainty approximation using acquired data.

    Features are RMS-scaled using acquired rows only (no centering; retain the
    intercept). Noise is acquired residual MSE, floored at noise_floor times
    acquired target SD, and at variance 1e-8. This empirical in-sample noise is
    an acquisition heuristic, not calibrated Bayesian posterior uncertainty.
    """

    def __init__(self, train_features, observed, predicted, *, prior_precision=1.,
                 noise_variance=None, noise_floor=.05, device="cpu"):
        if not math.isfinite(prior_precision) or prior_precision <= 0:
            raise ValueError("prior_precision must be finite and positive")
        if not math.isfinite(noise_floor) or noise_floor < 0:
            raise ValueError("noise_floor must be finite and nonnegative")
        self.device = torch.device(device)
        x = self._features(train_features)
        y = np.asarray(observed, dtype=np.float64)
        means = np.asarray(predicted, dtype=np.float64)
        if len(x) < 2 or y.shape != (len(x),) or means.shape != y.shape:
            raise ValueError("Need at least two acquired feature/label/prediction rows")
        if not np.isfinite(y).all() or not np.isfinite(means).all():
            raise ValueError("Acquired labels and predictions must be finite")
        self.dimension = x.shape[1]
        self.acquired_count = len(x)
        self.scale = x.square().mean(0).sqrt().clamp_min(1e-3)
        self.prior_precision = float(prior_precision)
        if noise_variance is None:
            noise_variance = max(float(np.mean((y-means)**2)),
                                 float((noise_floor*np.std(y))**2), 1e-8)
            self.noise_source = "acquired residual MSE with acquired-SD floor"
        else:
            self.noise_source = "supplied constant"
        if not math.isfinite(noise_variance) or noise_variance <= 0:
            raise ValueError("noise_variance must be finite and positive")
        self.noise_variance = float(noise_variance)
        normalized = (x / self.scale).double()
        # A = sigma² Lambda. Thus phi.T A^-1 phi is variance/noise,
        # and an unlabeled hypothetical observation adds phi phi.T to A.
        self.precision = normalized.T @ normalized
        self.precision.diagonal().add_(self.prior_precision*self.noise_variance)

    def _features(self, features):
        value = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if value.ndim != 2 or value.shape[1] < 1 or not bool(torch.isfinite(value).all()):
            raise ValueError("Features must be a finite [rows, dimensions] matrix")
        if hasattr(self, "dimension") and value.shape[1] != self.dimension:
            raise ValueError("Feature dimension differs from acquired representation")
        return value

    @torch.no_grad()
    def select(self, pool_features, predicted, count, *, information_weight=1.,
               quality_weight=.25, condition_features=None):
        """Return (pool row positions, diagnostics); never requires pool labels.

        condition_features may contain an already selected exploitation batch.
        These unlabeled rows condition covariance before information selection.
        Exclude acquired/already selected identities from the pool in the driver.
        Quality is (NN mean - median)/(95th-5th percentile), clipped to [-2,2].
        Information is actual conditional gain in nats, without rescaling.
        """
        if any(not math.isfinite(w) or w < 0 for w in (information_weight, quality_weight)):
            raise ValueError("Acquisition weights must be finite and nonnegative")
        if information_weight == quality_weight == 0:
            raise ValueError("At least one acquisition weight must be positive")
        x = self._features(pool_features) / self.scale
        means = torch.as_tensor(predicted, dtype=torch.float32, device=self.device)
        if means.shape != (len(x),) or not bool(torch.isfinite(means).all()):
            raise ValueError("Need one finite neural prediction per pool row")
        if not isinstance(count, (int, np.integer)) or count < 0 or count > len(x):
            raise ValueError("count must lie between zero and candidate pool size")
        precision = self.precision.clone()
        conditioned = 0
        if condition_features is not None:
            condition = (self._features(condition_features) / self.scale).double()
            conditioned = len(condition)
            precision.add_(condition.T @ condition)
        # Solve small matrices in float64; GPU batch rank-one updates use float32.
        cholesky = torch.linalg.cholesky(precision)
        # Whiten in float64 before casting: one-hot + intercept columns are
        # linearly dependent, so explicitly casting A^-1 to float32 can create
        # catastrophic cancellation. The whitened prior covariance is identity.
        x = torch.linalg.solve_triangular(cholesky, x.T.double(), upper=False).T.float()
        projected = x.clone()
        variance_ratio = x.square().sum(1)
        initial = variance_ratio.clone()
        if len(means):
            low, median, high = torch.quantile(means, torch.tensor([.05, .5, .95], device=self.device))
            quality = ((means-median)/(high-low).clamp_min(1e-6)).clamp(-2, 2)
        else:
            quality = means
        selected, gains, variances = [], [], []
        unavailable = torch.zeros(len(x), dtype=torch.bool, device=self.device)
        for _ in range(count):
            information = .5*torch.log1p(variance_ratio)
            acquisition = quality_weight*quality + information_weight*information
            acquisition.masked_fill_(unavailable, -torch.inf)
            j = int(acquisition.argmax())
            selected.append(j)
            gains.append(float(information[j]))
            variances.append(float(variance_ratio[j])*self.noise_variance)
            unavailable[j] = True
            # Cov(phi_i, phi_j)/sigma² and Woodbury denominator, conditioned on
            # all earlier selected rows. Clone row before mutating its matrix.
            cross = projected @ x[j]
            row = projected[j].clone()
            denominator = 1. + variance_ratio[j]
            projected.sub_(cross[:, None]*row[None, :] / denominator)
            variance_ratio.sub_(cross.square()/denominator).clamp_min_(0)
        diagnostics = dict(
            approximation="conditional Gaussian linear model on frozen learned CP features",
            dimension=self.dimension, acquired_count=self.acquired_count,
            pool_size=len(x), selected_count=len(selected),
            conditioned_count=conditioned, noise_variance=self.noise_variance,
            noise_source=self.noise_source, prior_precision=self.prior_precision,
            information_weight=float(information_weight), quality_weight=float(quality_weight),
            conditional_gain_nats=gains, total_conditional_gain_nats=float(sum(gains)),
            selected_conditional_latent_variance=variances,
            initial_mean_latent_variance=float(initial.mean())*self.noise_variance if len(x) else 0.,
            final_mean_latent_variance=float(variance_ratio.mean())*self.noise_variance if len(x) else 0.,
            feature_scaling="acquired-only RMS, minimum1e-3; no centering",
        )
        return np.asarray(selected, dtype=np.int64), diagnostics
