# Budgeted docking search with SynFactor

This research pilot fits neural synthon factors **during search**, using only
charged oracle queries. It starts with 1,000 random products, then repeats
fit → score the space → acquire 1,000 products nine times. The total budget is
10,000 unique product IDs. This is separate from the fixed ADMET-teacher demo.

The [initial results and query traces](../../benchmarks/docking_pilot/README.md)
compare against unchanged archived GA-DCSO implementations and separately label
adapted SpaceHASTEN and historical comparisons.

## Selected method

- Capped-synthon Morgan count features (512 bins) and four RDKit descriptors.
- A shared `516 → 128 → 128` neural network, three rank-64 slot heads,
  optional additive slot heads, and learned reaction offsets.
- Product utility is the sum of elementwise slot-factor products, additive
  terms and the reaction offset. Missing third slots use the identity factor.
- Fit from scratch each round using acquired labels only: AdamW, squared error,
  deterministic 80/20 acquired-data split, at most 180 epochs and patience 15.
  Feature and target normalization use the fitting partition only.
- Cache all synthon factors, then score every candidate with batched GPU
  gathers and factor products. No per-product neural encoder is used.
- Acquire 900 predicted high-value products plus 100 uniformly random unqueried
  products. The selected 1M configuration applies a soft diversity penalty of
  0.5 against acquired elites and the developing batch. Acquisition uses the
  existing **RDKit path/2048** fingerprint cache, not Morgan fingerprints.

The driver also supports CP16 → top-10% CP64 reranking, product-fingerprint
LightGBM reranking, ensembles and disagreement exploration. These were tried
on KIF11 seed 0; the simpler CP64 plus diversity performed better for the
selected portfolio objective. See every usable development run in
[`development.csv`](../../benchmarks/docking_pilot/development.csv).

## Data and objectives

This integration expects the local SynthonBench workspace. The ordinary
SynFactor demo remains standalone; these optional experiment commands require
the score tables, space schemas and existing fingerprint caches. No baseline
implementation is copied into SynFactor.

| Input | Meaning |
| --- | --- |
| `scores/glide_1M.parquet` | 990,610 product IDs with precomputed measured Glide scores |
| `scores/surrogate_10M.parquet` | Frozen docking-surrogate predictions, not measured docking |
| `scores/surrogate_100M.parquet` | Frozen docking-surrogate predictions; check duplicate-ID policy in the report |
| `spaces/*.synthons.tsv`, `spaces/reactions.tsv` | Reaction grids and assembly rules |
| `synthon_space_*_enum_fps` | Precomputed enumerated RDKit path fingerprints |

Large tables also require the released `.sidx` sorted-ID index for independent
oracle/identity checks and `.topk` sidecar for exact global-tail evaluation.
Preparation writes a sparse first-source-row alias map; the search rejects
noncanonical aliases and refuses unaudited large caches. Re-running preparation
on an existing cache performs the identity audit without rebuilding the arrays.

**The historical `*_glide_le` column is not heavy-atom ligand efficiency.**
It stores `-raw Glide - MolLogP`, with zero on filtering/docking failure.
The primary comparison instead uses **negative raw Glide**, with nonfinite
scores replaced by zero and still charged. `--objective stored` reproduces the
historical objective. Surrogate tables always use negative predicted score.
Do not compare these tracks as if they optimized the same objective.

The optimizer never uses target-dependent docking-success/validity flags to
prefilter candidates. `--chemical-filter` is an explicitly separate variant
using the deterministic structural filter; it was not used in the primary
comparison. Valid-only scores are reported post hoc as a secondary analysis.

Some synthon IDs are reused for different attachment structures. Preparation
resolves these using the recorded product structure and reaction assembly,
never docking scores. Historical RDKit stereoretention differences are counted
when a unique connectivity match is used. Fingerprint alignment and packed-bit
order are audited before using the cache.

## Run the selected 1M experiment

From the repository root, after installing the ordinary requirements:

```bash
pip install -r experiments/docking/requirements.txt

python -m experiments.docking.prepare \
  --data /path/to/hf_synthonbench_dataset \
  --fingerprints /path/to/synthon_bench/data \
  --scale 1M --out runs/docking_data/1M

python -m experiments.docking.run \
  --cache runs/docking_data/1M --out runs/my_cp64_kif11 \
  --target kif11 --seed 0 --budget 10000 --batch 1000 \
  --rank 64 --additive --epochs 180 --diversity .5 --objective raw
```

Use `pyrd` or `tgfr1` for the other targets. CPU is supported with `--device cpu`.
For the development cascade, use `--rank 16 --additive --rerank cp64-gbm
--screen-fraction .1`; the optional LightGBM requirement is needed for that
reranker. Existing output traces are never overwritten.

For a larger plain-CP run, prepare `--scale 10M` or `100M` with
`--skip-fingerprints`, then run with `--diversity 0`. This does not require a
full product-fingerprint array. Preparation still reads tuple identities and
builds a separate frozen-oracle cache; it is not a full-space live-docking API.
The current scorer visits the enumerated rows. Full Freedom/Enamine integration
needs streaming Cartesian matrix scoring and on-demand product assembly.

An additional development mode, `--diversity .5 --diversity-features synthon-or`,
approximates candidate fingerprints by OR-combining nonzero capped-synthon
Morgan-count bins (512 bits) only for the acquisition shortlist and observed
elites. It supports diversity selection without a full product fingerprint
cache. This is an acquisition approximation, not a reconstructed full-product
Morgan fingerprint; final portfolio evaluation still uses real product
Morgan radius-2/2048 fingerprints. The frozen 1M comparison uses the original
`--diversity-features product` default.

## GA-DCSO controls

```bash
python -m experiments.docking.baseline_run \
  --workspace /path/to/SynthonBench_ICML \
  --target kif11 --seed 0 --method ga_dcso --budget 10000 \
  --out runs/ga_raw_kif11

python -m experiments.docking.baseline_run \
  --workspace /path/to/SynthonBench_ICML \
  --target kif11 --seed 0 --method ga_dcso_rdk_v12 --budget 10000 \
  --out runs/ga12_raw_kif11
```

The adapter imports the unchanged archived implementation and public benchmark
harness. v0 uses its default 10 × 1,000 schedule; v12 uses 20 × 500 and
XGBoost CUDA. This archived v12 is itself a benchmark adaptation of production
v12, not the complete production optimizer. Both use the same raw-score oracle policy. Missing required
estimators fail explicitly rather than silently substituting random search.
The six primary CP/v0 comparisons share exactly the same ordered first 1,000
product IDs. v12 retains its own native initialization and schedule.

The unchanged local SpaceHASTEN-style adaptation can be rerun with
`python -m experiments.docking.spacehasten_run --workspace /path/to/SynthonBench_ICML
--target kif11 --seed 0 --out runs/sh_raw_kif11`. It requires the legacy
`synthon_bench` source as well as the public harness and indexed oracle.
Its product-weighted per-reaction initialization is retained; it is not the
native FTrees/SpaceLight implementation.

## Portfolio evaluation

Use a separate environment for OPDiv: its OR-Tools dependency uses NumPy 2,
whereas the main ADMET environment is pinned to NumPy 1.26.

```bash
python -m venv .venv-eval
.venv-eval/bin/pip install -r experiments/docking/requirements-eval.txt
# Install the local OPDiv source used by the benchmark:
.venv-eval/bin/pip install /path/to/opdiv

.venv-eval/bin/python -m experiments.docking.evaluate \
  --score-table /path/to/hf_synthonbench_dataset/scores/glide_1M.parquet \
  --target kif11 --observed runs/my_cp64_kif11/observed.csv \
  --out runs/my_cp64_kif11/evaluation --ks 100 --thresholds .4 \
  --max-seconds 30
```

For 10M/100M use `experiments.docking.large_evaluate` with the same arguments;
its default is greedy-only, with `--optimal` enabling exact OPDiv. It scans
only relevant Parquet columns, materializes queried rows, and reads the frozen
top-k sidecar for global hit recovery. Never load the entire 100M structure
table into pandas to evaluate a 10,000-query archive.

Evaluation deduplicates canonical isomeric molecules, retaining the first
queried representative. It recomputes **Morgan radius 2 / 2048-bit**
fingerprints, without chirality. OPDiv100 maximizes mean negative docking score
of exactly 100 molecules subject to every pair having Tanimoto ≤ 0.4; GPDiv is
the corresponding greedy portfolio. Exact optimization considers the entire
eligible queried archive, not a top-score prefix. Solver status, bound and gap
are retained; a feasible solution is not called a certified optimum.

Search time includes repeated fitting, whole-space scoring, selection and table
queries. It excludes cache preparation, interpreter/import startup, and post-hoc
portfolio evaluation. These are precomputed oracle lookups, **not 10,000 live
docking calculations in seconds**. Runtime includes at most the configured
number of rounds; the wall-time check occurs between rounds.

## Checks and artifacts

```bash
python -m unittest discover -s tests -v
.venv-eval/bin/python -m unittest discover -s tests -p test_docking_evaluate.py -v
python -m experiments.docking.report
```

Budget tests instrument truth access and reject reads outside charged oracle
queries; tests also cover duplicate calls, nonfinite values, mixed-arity
scoring, warm starts, shuffled grid alignment and molecular deduplication.
Reports preserve compressed ordered query traces, score summaries, exact
solver bounds, run configurations, source hashes and the frozen protocol.
