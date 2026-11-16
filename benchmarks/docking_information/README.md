# Information-acquisition pilot

**Information acquisition was inexpensive, but did not consistently improve the existing SynFactor policy.** Six fixed Freedom 1M runs compared an optional information arm with the frozen CP64 + diversity policy and matched-budget GA-DCSO v0. There was no information hyperparameter tuning.

[Paired numerical results](paired.csv) · [summary and per-run bounds](summary.json) · [protocol](protocol.json) · [oracle audit](audit.json) · [method definition](../../experiments/docking/INFORMATION.md) · [original docking pilot](../docking_pilot/README.md)

| Information arm compared with | Top-100 score wins | GPDiv100 wins | Certified OPDiv100 wins |
| --- | ---: | ---: | ---: |
| Frozen CP64 + diversity | 5 / 6 | 3 / 6 | 3 / 6 |
| Matched GA-DCSO v0 | 3 / 6 | 6 / 6 | 6 / 6 |

Despite five top-100 wins over frozen CP, the mean raw score **worsened by 0.01443**: the KIF11 seed 1 loss exceeded the other gains. Mean GPDiv and OPDiv changed by **−0.01283** and **−0.01526**, respectively. Against GA-v0, mean GPDiv and OPDiv improved by 0.23434 and 0.22098. All six new OPDiv solves reached certified optimality over their queried archives. These results support keeping the information arm optional.

## Fixed comparison

KIF11, PYRD and TGFR1, seeds 0 and 1; 990,610 candidate product IDs. Each run starts from the same ordered 1,000 observations as its frozen CP control and makes exactly **10,000 unique charged queries**. Docking failures consume queries and receive zero optimization utility. Evaluation uses finite raw Glide scores, canonical isomeric molecule deduplication, and Morgan radius 2/2048/no-chirality portfolios of 100 molecules with pairwise **Tc ≤ 0.4**. This is the frozen-table pilot convention, separate from the strict Tc < 0.30 convention of the full-Freedom Orion study.

The neural model is rank 64 CP with additive terms, 180 maximum epochs, and diversity strength 0.5. After the initial 1,000 queries, each of nine batches contains:

- 700 choices for predicted utility and the existing soft diversity penalty;
- 200 information-directed choices, conditioned on those 700;
- 100 random unqueried products.

The information pool combines approximately 10,000 elites and 10,000 random candidates. Acquisition adds Gaussian conditional gain in nats to 0.25 times normalized neural utility. Uncertainty uses the frozen learned factor representation, acquired-only feature scaling and acquired residual noise. This is **approximate conditional linear-model information gain**, not exact information about neural-network weights or true docking outcomes.

Search took **5.65–6.02 seconds** per run on the RTX 4090, including fitting, scoring, acquisition and cached oracle queries. This excludes imports/cache loading, cache preparation and portfolio evaluation; it is not live docking time. Each run selected 1,800 information-directed products. Summed gains were 26.72–31.73 nats. Because each refit changes the representation, this sum is descriptive rather than one coherent information reduction about a fixed parameter vector.

## Reproduce

Use the training and evaluation environments in [environment.json](environment.json), and prepare the audited Freedom 1M cache following the [docking instructions](../../experiments/docking/README.md). The exact source used for these runs is preserved in `frozen_source/`; its file hashes match `protocol.json`.

From the repository root, one cell can be rerun as follows, using a fresh output directory:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python \
  -m benchmarks.docking_information.frozen_source.run \
  --cache runs/docking_data/1M_v2 --out runs/reproduce_information_kif11_s0 \
  --target kif11 --seed 0 --budget 10000 --batch 1000 \
  --rank 64 --additive --epochs 180 --diversity .5 --objective raw \
  --information-fraction .2 --information-pool 20000 --information-quality .25

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv-eval/bin/python \
  -m benchmarks.docking_information.frozen_source.evaluate \
  --score-table /path/to/hf_synthonbench_dataset/scores/glide_1M.parquet \
  --target kif11 --observed runs/reproduce_information_kif11_s0/observed.csv \
  --out runs/reproduce_information_kif11_s0/evaluation \
  --ks 100 --thresholds .4 --max-seconds 30
```

Repeat for `kif11`, `pyrd`, `tgfr1` and seeds 0,1. Solver time is limited per optimization solve; graph construction is additional. Numeric reproducibility depends on the recorded environments and hardware.

## Included records

`cells/` contains six configurations, run summaries, evaluation summaries, compressed ordered query traces and compressed per-round fit/information histories. Traces contain only call number, cached row index, hexadecimal benchmark product ID and observed utility. **No molecular structures, product fingerprint arrays or vendor synthon files are included.** The existing baseline evidence is preserved in the [original pilot](../docking_pilot/README.md).

`audit.json` verifies all queried utilities against the charged oracle, unique query counts and unchanged source hashes. `paired.csv` records raw scores, GPDiv, OPDiv bounds/status, runtime and information gains for every pair. The code tests cover oracle isolation, diminishing information for duplicate features, unlabeled batch conditioning, determinant-lemma agreement and stable singular feature designs.
