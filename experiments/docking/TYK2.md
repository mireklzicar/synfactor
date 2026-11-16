# TYK2: independent CP proposals and full-product ranking

This development experiment uses the shared seed-0 warm start (1,000 charged
attempts), then at most nine batches of 1,000 live Glide queries. Both models
train directly from this branch's acquired docking scores. There is no
distillation, druglike filter, archival winner seeding or reaction quota.
Failed docking attempts consume budget but do not become regression targets.

## Models and candidate generation

- Neural CP64 with additive terms uses capped-synthon Morgan counts and four
  descriptors. Its acquired-data validation split selects the training epoch.
- XGBoost uses assembled full-product Morgan radius-2, 2,048-bit, nonchiral
  fingerprints. The GA-v14 hyperparameters are retained: 256 trees, depth 8,
  learning rate 0.05, row and column subsampling 0.8, squared-error objective,
  histogram training, XGBoost 3.2.0. This changes the representation relative
  to the archived GA model; identical tree settings alone do not isolate the
  search algorithm.
- A streaming GPU sweep of 296,350,180,179 Freedom routes retains 100,000 CP
  candidates. Add up to 512 one-slot neighbors of each of the best 100 observed
  hits, and 10,000 uniformly sampled routes. Native assembly and full-product
  XGBoost reranking operate only on this bounded pool.
- Exclude previously charged routes and canonical molecules. Native assembly
  is checked against recorded structures; submitted SMILES are also checked by
  the remote docking bridge. No docking-derived validity filter is used before
  querying.

## Distinct objectives

`raw` takes 900 highest-XGBoost candidates, allowing close analogues, plus 100
quality-qualified CP/XGBoost rank-disagreement candidates.

`portfolio` takes nine successive greedy 100-member portfolios, each satisfying
strict full-product Tanimoto < 0.30, plus 100 disagreement candidates. Molecular
duplicates are excluded across groups, but close analogues in different groups
are allowed. The final GPDiv/OPDiv evaluation uses the entire acquired set.

Disagreement is a heuristic, not calibrated Bayesian information gain.
Fingerprint diversity likewise does not establish information optimality.

Uniform training is the reference. `--tail-weight 2` retains all valid labels
while smoothly increasing XGBoost weights toward the observed upper tail, with
maximum/minimum weight ratio 2. `--weight-cp` applies these weights to CP too.
These are explicit variants, not a reproduction of Autofocused Oracles.

## Run

These research commands require the local SynthonBench data, native assembler,
prepared Freedom cache and an already configured remote Glide oracle. They are
not part of the standalone public ADMET example.

Install the docking requirements first. For this CPU XGBoost reranker alone,
`pip install xgboost-cpu==3.2.0 scikit-learn==1.9.1` suffices in addition to the
ordinary repository requirements; CP training and Cartesian scoring use CUDA.

```bash
python -m experiments.docking.tyk2_run \
  --cache runs/docking_fullspace/freedom \
  --out runs/tyk2_raw_s0 --policy raw --rounds 1
```

This prepares a proposal. Add `--execute` to submit it. Start with one adaptive
round; extend the same unchanged configuration using `--rounds 2`, then
`--rounds 9`. Use a different output directory for policy or weighting changes.
Configuration hashes, proposals and completed responses are durable; unresolved
remote work must be reconciled before any retry. Pilot runs use an immutable
source snapshot so later code edits cannot silently alter the experiment.

```bash
python -m experiments.docking.tyk2_evaluate \
  --history runs/tyk2_raw_s0/state.json \
  --charged 2000 --out runs/tyk2_raw_s0/evaluation_2k --optimal
```

Evaluation requires the separate Orion/OPDiv environment. It reconstructs
matched charged prefixes of GA-v14 and both SpaceHASTEN variants, checks the
warm/grid identities, canonicalizes molecules and reports raw top-100, GPDiv100
and optionally full-domain SCIP OPDiv100 bounds/certification. All are unfiltered
by druglikeness. Lower raw docking means and higher diversity utilities are
better. A one-seed development win is not a multi-target confirmation result.

## Optional fiber proposal helper

`tyk2_fibers.cp_fiber_candidates` fixes all but one slot of observed hit anchors
and ranks every compatible replacement with cached CP factors. Anchors may be
raw top hits or a greedy diverse set of observed hits. This is a label-free,
bounded candidate-generation helper, not an additional source of docking labels.
It is not enabled in the two frozen pilot arms. On the warm-start model it
generated about 76,000 routes from 1.6 million substitutions in under a second;
any acquisition benefit still needs a separate experiment.

Enable these proposals with `--fiber-anchors raw` or `--fiber-anchors gpdiv`.
`--disagreement-count 0` gives the raw policy 1,000 greedy exploitation queries.
`--chiral-xgb` is an independent optional predictor-feature variant; acquisition
and portfolio evaluation still use the original nonchiral fingerprints.

## Explicit explore-then-exploit development branch

`tyk2_branch` can preserve a verified prefix of the portfolio policy and start a
separate continuation. Inherited queries remain charged; no later parent labels
are copied. This is a development branch, not an independent replicate.

```bash
python -m experiments.docking.tyk2_branch \
  --parent runs/tyk2_portfolio_s0 --inherited-rounds 3 \
  --out runs/tyk2_explore_exploit_s0
python -m experiments.docking.tyk2_run \
  --out runs/tyk2_explore_exploit_s0 --policy raw \
  --fiber-anchors gpdiv --disagreement-count 0 --rounds 4
```

The example inherits 4,000 attempts and prepares one new 1,000-query batch.
Only `--execute` submits it; continuing through `--rounds 9` reaches 10,000
total attempts, including the inherited prefix. Lineage binds parent identity,
query evidence and history hashes. Optional model settings must remain fixed
through this continuation, and all branches should be disclosed in reporting.
