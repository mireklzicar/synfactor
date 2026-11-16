# Information-directed acquisition

The optional information arm asks which additional observations would reduce
uncertainty in a small conditional model of the newly fitted neural factors.
It is **not an exact posterior over neural-network weights**, and a large
information score is not a guarantee of better docking or portfolio results.

For each candidate, `FactorRegressor.acquisition_features()` exposes the rank-R
factor products, optional additive terms, reaction indicators and an intercept.
With this representation frozen for one round, a Gaussian linear surrogate has
precision

```text
Λ = λ I + Φobservedᵀ Φobserved / σ²
gain(x | observed) = ½ log(1 + φxᵀ Λ⁻¹ φx / σ²)
```

The feature scales and residual-noise estimate use **charged observations
only**. The noise estimate is acquired residual MSE with a floor; it is an
empirical acquisition heuristic, not calibrated uncertainty. Neural predictions
remain the score objective; the linear surrogate does not replace the predictive
mean. The representation and uncertainty approximation are rebuilt after every
neural refit.

Within a batch, covariance is conditioned after each hypothetical selection
using rank-one updates. Labels are not needed for this conditional covariance
under the Gaussian approximation. Selecting two nearly redundant feature vectors
therefore has diminishing information gain. The implementation uses a small
feature covariance and a bounded pool, not a quadratic candidate kernel.

## Fixed tested policy

At the 10,000-query budget, charge the shared 1,000 warm observations and then
repeat nine 1,000-query batches:

- 700 selections for predicted utility with the existing soft diversity penalty;
- 200 information-directed selections, conditioned on those 700 choices;
- 100 random, previously unqueried products.

The information pool mixes approximately 10,000 predicted elites with 10,000
random candidates. Previously charged and already selected identities are
excluded. Its greedy criterion is conditional gain in nats plus 0.25 times
robustly normalized neural utility. Thus this is a quality/information tradeoff,
not unconstrained maximum information over the entire space.

```bash
python -m experiments.docking.run \
  --cache runs/docking_data/1M --out runs/information_kif11 \
  --target kif11 --seed 0 --rank 64 --additive --epochs 180 \
  --diversity .5 --objective raw --information-fraction .2 \
  --information-pool 20000 --information-quality .25
```

`--information-quality 0` gives a pure information criterion within the
information pool. `--information-fraction 0` preserves the prior default.
Information acquisition currently requires one CP model and no reranker;
chemical diversity and random acquisition remain separate controls.

Every round records selected conditional variances, marginal gains, noise,
feature dimension and pool size. Summing gains across rounds is descriptive:
each neural refit changes the representation, so it is not a single coherent
information reduction about a fixed neural-network parameter vector.

## Six paired Freedom 1M experiments

The fixed policy was tested on KIF11/PYRD/TGFR1, seeds 0 and 1, with identical
warm starts and budgets to the original CP64 pilot. All calls were verified
against the source oracle. All six OPDiv100 solves reached certified optimality.

| Comparison | Top-100 score wins | GPDiv100 wins | Certified OPDiv100 wins |
| --- | ---: | ---: | ---: |
| Information arm vs previous CP64 + diversity | 5 / 6 | 3 / 6 | 3 / 6 |
| Information arm vs matched GA-DCSO v0 | 3 / 6 | 6 / 6 | 6 / 6 |

The information arm took **5.65–6.02 seconds** per run and selected 1,800
information-directed products per run. Relative to the existing CP policy,
mean GPDiv changed by **−0.01283** and mean OPDiv by **−0.01526**. Despite five
top-100 wins, its mean top-100 score also worsened slightly because KIF11 seed 1
lost more than the other cells gained.

This supports keeping information acquisition **optional**, rather than claiming
it consistently improves the existing policy. No information hyperparameter
tuning was performed on these six results. See
[`benchmarks/docking_information`](../../benchmarks/docking_information/README.md)
for records and the full numerical comparison.

Tests independently check the determinant-lemma gain, diminishing duplicate gain,
conditioning on unlabeled exploitation choices, acquired-only normalization,
mixed-arity feature reconstruction and singular reaction/intercept designs.

## Full-space prior sensitivity

The first full-Freedom TYK2 run exposed a calibration issue: reaction-indicator
columns absent from the observations receive the RMS floor of 0.001. Under the
isotropic prior in scaled coordinates, this gives unseen reactions a very large
latent variance. First observations of 69 new reactions accounted for 88.48% of
the run's reported information gain. Those nats describe this chosen prior;
they are not measured improvements in predictive accuracy.

The ongoing fixed-policy pilot retains this implementation for comparability.
A separate development variant should give reaction indicators an explicit
unit scale/prior, then test whether information acquisition improves the score
and portfolio objectives. The existing default remains zero information share.
