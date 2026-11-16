# Surrogate overoptimization, informative batches and shifting search distributions

Literature checked 2026-11-16 for the TYK2 development experiment. These works
motivate design choices; the pilot is not a reproduction of their algorithms.

## Optimizing prediction errors

[Conservative Objective Models (Trabucco et al., ICML 2021)](https://proceedings.mlr.press/v139/trabucco21a.html)
directly studies the failure where optimizing a learned predictor drives search
to overestimated out-of-distribution designs. It adds a conservative training
objective. Its setting is offline optimization, whereas our new docking results
arrive sequentially. A practical implication for this pilot is to check selected
candidate calibration and use independently trained models; disagreement alone
is neither a calibrated posterior nor the COM objective.

[Conditioning by Adaptive Sampling (Brookes et al., ICML 2019)](https://proceedings.mlr.press/v97/brookes19a/brookes19a.pdf)
frames design as estimating a distribution conditioned on desired properties,
with controls on excursions into regions where the predictive oracle is
unreliable. It explains why unrestricted maximization of a surrogate can be
inferior to controlled adaptive search. We use observed-hit neighbors alongside
global proposals; this is not an implementation of CbAS.

[TuRBO (Eriksson et al., NeurIPS 2019)](https://proceedings.neurips.cc/paper/2019/hash/6c990b7aca7bc7058f5e98ea909e924b-Abstract.html)
uses local probabilistic models and allocation among local searches to address
heterogeneous global landscapes and excessive exploration. Its continuous
trust regions are not directly our discrete reaction space. The relevant idea
is combining broad discovery with focused exploitation around observed hits.
The authors' framework has an executable
[BoTorch tutorial](https://botorch.org/docs/v0.17.0/tutorials/turbo_1).

## Information gain needs a decision objective

[Max-value Entropy Search (Wang and Jegelka, ICML 2017)](https://proceedings.mlr.press/v70/wang17e.html)
seeks information about the maximum function value, rather than all model
parameters. This distinction matters when the goal is finding top-scoring
molecules rather than making the entire space predictable. The
[BoTorch MES tutorial](https://botorch.org/docs/v0.17.0/tutorials/max_value_entropy)
provides the posterior-based acquisition and sequential batch conditioning.
An arbitrary CP/XGB disagreement score cannot be called MES or optimal
information gain. Extending decision-focused acquisition to a constrained
100-member portfolio would require explicitly modelling that portfolio utility.

[BatchBALD (Kirsch et al., NeurIPS 2019)](https://proceedings.nips.cc/paper/2019/hash/95323660ed2124450caaac2c46b5ed90-Abstract.html)
jointly scores informative batches to avoid the redundancy of independently
selecting high-uncertainty points. Its original evaluation concerns Bayesian
classification, so it is not a drop-in regression policy for our two models.
The author's [technical explanation](https://blackhc.github.io/BatchBALD/)
describes greedy selection conditioned on previously selected points.

[Deep Batch Active Learning for Regression (Holzmüller et al., JMLR 2023)](https://jmlr.org/papers/v24/22-0937.html)
is more directly relevant to our regression setting. It organizes neural
feature kernels, posterior transformations and batch selection approaches,
including methods that account for both information and redundancy. True
product-fingerprint GPDiv is a useful diversity heuristic, but geometric
dissimilarity alone does not establish Bayesian information optimality.

[MolPAL (Graff et al., Chemical Science 2021)](https://pubs.rsc.org/en/content/articlehtml/2021/sc/d0sc06805e)
compares models and acquisition strategies for docking. Greedy acquisition
performs strongly, with UCB improving some cases, and model representation
matters. This supports keeping a separate raw-score arm that allows close
analogues instead of imposing the diversity goal on every query.

## Reweighting the changing search distribution

[Autofocused Oracles (Fannjiang and Listgarten, NeurIPS 2020)](https://arxiv.org/abs/2006.08052)
studies retraining a predictive oracle as the search distribution moves. Its
importance weights reflect the search-versus-training input distributions;
this is not equivalent to simply weighting observations by their score.

Our proposed rank-tail weighting is a simpler controlled heuristic: retain
every valid observed docking score, increase weights smoothly toward the
observed top tail, and cap the max/min ratio at2 initially (at most4 in explicit
variants). The target docking function is unchanged; the acquired input
distribution shifts. Old low-score observations still provide evidence against
false positives, so neither age-based deletion nor hard top-tail-only training
is the default. Weighted and unweighted models must be compared on subsequent
observations without treating a retrospective selected-pool test as a live
acquisition improvement.

## Current experimental interpretation

The two models train directly on the same acquired valid docking observations.
CP64 proposes globally; full-product Morgan XGBoost ranks a bounded pool with
the archived GA-v14 tree hyperparameters. There is no distillation. Raw-score
selection permits close analogues. Portfolio selection uses successive strict
GPDiv100 groups plus bounded, quality-qualified cross-model disagreement.
No druglike gate is used. These are practical development policies, not claims
of theoretically optimal active learning.
