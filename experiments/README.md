# Initial prior-art comparison

These optional experiments compare teacher approximation on fully enumerable
reaction grids. They are separate from the minimal SynFactor demo and from the
993-test-route experiment in the main figure.

## Methods and scope

| Method | What is actually run | Labels / representation |
| --- | --- | --- |
| Shi Eq. 11 | Basis-product formula from Shi et al. (2000) | Full-product teacher evaluated on one reference product and each single-slot substitution; no fitting |
| Shi + calibration | Eq. 11 plus per-reaction, per-endpoint affine calibration | Same basis products plus training labels; calibration never uses test labels |
| Learned additive | Ridge fit of reaction-slot-specific synthon lookup values | All training labels; penalty selected on validation; useful flexible additive control within these known grids |
| CP16 | The repo's neural rank-16 CP model | Frozen capped-synthon embeddings from each endpoint's molecular teacher; RDKit fingerprint/descriptor features for classical descriptors |
| APEX pilot | Upstream APEX encoder, LOO ridge probe, and associative factorizer | Molecular surrogate trained from scratch on these teacher labels, then embedding reconstruction; reduced training budget |
| APEX molecular surrogate | Full-product encoder + linear probe before factorization | Diagnostic of the first APEX stage; requires assembled products and is **not** a fast synthon scorer |
| Histogram control | Integer convolution of additive descriptor contributions | A limited control motivated by SpaceProp; **not the native SpaceProp implementation** |

The Shi formula implemented in `baselines.py` is

```text
Q_hat(i_1, ..., i_K)
  = sum_j Q(reference partners with slot j replaced by i_j)
    - (K - 1) Q(all reference partners).
```

References are the smallest heavy-atom-count synthons among the selected rows
in each slot, chosen before any labels are inspected. These are **assembled
basis products**, not predictions on isolated fragments. The paper also
reports linear calibration; we show the raw formula and a train-only
calibrated variant separately. We do not reproduce its tree filtering engine.

The APEX adapter imports actual upstream modules at commit
`80e544eff6bbc2b09022a4f780fc9665256a884e`. It keeps the native 64-dimensional,
four-layer architectures, noisy molecular-encoder training, leave-one-out
ridge probe and normalized embedding-reconstruction loss. It changes the data
loader, labels, validation protocol and training budget (1,500 steps per stage
instead of the upstream example's 100,000). Variance denominators are clamped
at `1e-8` for numerical stability. Factorizer selection uses validation
embedding error. This is an **initial ADMET adaptation**, not a reproduction of
APEX's published docking results or a benchmark of its CUDA top-k engine.
CP16 uses pretrained foundation features; this APEX pilot trains its molecular
encoder from scratch. Their training histories are not matched.

SpaceProp handles chemically decomposable descriptors through fragment
internal and boundary contributions. It does not accept arbitrary OpenADMET
neural predictions as an exact descriptor calculator. A native installation
and license were not available for this run. Our control checks additive
basis-product contributions against every product in the pilot, and convolves
integer HAC/HBA/HBD counts. A second HAC-only control uses the bundled larger
space, restricted to schemas for which the code can establish atom
conservation. It counts combinations, not deduplicated valid molecules.
Neither control implements SpaceProp's general boundary handling.

## Reproduce

First install the main repo's requirements and optional OpenADMET dependencies
as described in the main README. Reporting additionally needs Matplotlib and
the APEX pilot needs TensorBoard. Upstream APEX is CC BY-NC 4.0; its source
and weights are not included in this MIT-licensed repo.

```bash
python -m experiments.compare --device cuda --out runs/comparison
python -m experiments.histograms --out runs/comparison

pip install -r experiments/requirements.txt
python -m experiments.prepare_apex /tmp/synfactor-apex
python -m experiments.apex_pilot --upstream /tmp/synfactor-apex \
    --device cuda --steps 1500 --out runs/comparison
python -m experiments.report --out runs/comparison
```

`prepare_apex` pins the checkout and changes one eager `apex_topk` import into
a lazy import, allowing training without compiling the unused search extension.
It does not modify the models. The adapter does not download published APEX
weights. Use a **new output directory** when changing the seed, epochs, step
budget or method; existing prediction/checkpoint caches are reused on resume.
To reuse an existing deployment, `compare` accepts `--models-root` (one folder
per repository) and `--foundation-root` (parent of `foundations/`). If a local
model lacks recipe YAML, target names are accepted only when its checkpoint
hash matches the independently verified `benchmarks/adapter_smoke.json`.

For a lightweight, teacher-download-free start:

```bash
python -m experiments.compare --descriptors-only --device cpu --out runs/descriptors
python -m experiments.histograms --out runs/descriptors
```

`report` currently expects the full ADMET run because its constrained query
uses hERG and LogD. All methods can train/score on CPU; GPU is the default.

## Protocol and interpretation

Seed 38026 selects six two-slot reactions (20 × 20) and both three-slot
reactions (10 × 10 × 10) from the bundled example. All 4,400 products assemble
successfully and are unique canonical isomeric SMILES. All 290 unique basis
products are forced into training. The other products use a deterministic
canonical-SMILES hash split: 3,180 training, 415 validation, 805 test in total.
No basis-product label is used as a held-out result. Synthons and reactions
occur in every split: this tests **unseen combinations of known synthons**, not
new-synthon or new-reaction generalization. There is one seed, one small space,
and no experimental ADMET data.

Nine pinned OpenADMET models supply 18 outputs. Six RDKit descriptors supply
additional controls. Teacher labels are calculated for the entire grid to
permit exact auditing, but fitting uses only the training labels and model
selection uses only validation. Shi Eq. 11 needs only the 290 basis labels;
learned methods use 3,180 fitting labels plus 415 validation labels. Thus this
is not a matched-oracle-budget comparison. Feature normalization uses training
synthons and target normalization uses training products.

The report includes global, two-/three-slot and per-reaction Spearman/MAE/RMSE,
both-tail top-5% recall for ADMET, and one held-out constrained query:
minimize hERG prediction while LogD lies between the validation teacher's
25th/75th percentiles. Candidate ranking is exhaustive over only the held-out
pilot products. This is a retrieval-quality check, not a trillion-scale search
or timing claim. Histogram correctness is audited against complete-grid
descriptor evaluations; those full-grid audits are not generalization tests.

Correlation alone is misleading for discrete descriptors: tiny numerical
errors can break ties and reduce Spearman even when MAE is negligible. Use
absolute errors and exact histogram counts for those controls.

## Sources

- Shi et al. (2000), [Efficient combinatorial filtering for desired molecular
  properties of reaction products](https://doi.org/10.1016/S1093-3263(00)00069-3),
  especially Eq. 11 and the basis-product/reference-product construction.
- Pedawi et al., [APEX](https://arxiv.org/abs/2510.24380),
  [upstream implementation](https://github.com/NumerionLabs/apex).
- Bellmann, Klein and Rarey (2022), *Calculating and Optimizing Physicochemical
  Property Distributions of Large Combinatorial Fragment Spaces*, and Lubbers,
  Lessel and Rarey (2024), *Enhanced Calculation of Property Distributions in
  Chemical Fragment Spaces*. See the [official SpaceProp page](https://www.zbh.uni-hamburg.de/forschung/amd/software/spaceprop.html).

Recorded results are in [`benchmarks/comparison_pilot`](../benchmarks/comparison_pilot/).
