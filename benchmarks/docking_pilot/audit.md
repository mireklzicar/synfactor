# Independent audit of the frozen 1M docking comparison

Audited 2026-11-16. Scope: six CP64 + additive, diversity 0.5 runs versus six unchanged archived GA-v0 runs, with KIF11 seed0 archived GA-v12 benchmark implementation as a supplementary control. No retuning performed.

## Findings

- No oracle-label leakage found in the inspected search path. Only `BudgetOracle.query` supplies raw scores to fitting; nonfinite scores become zero utility and consume budget. Descriptor/fingerprint features are computed from structures. Early stopping uses a 20% split of already charged observations. Evaluation reads full labels only after committed query traces. All 120,000 recorded CP/GA utilities agree with their source raw oracle (CP float32 rounding <4.8e-7; GA exact at CSV precision).
- Every trace has exactly 10,000 distinct product IDs. In each of the six paired cells, the first 1,000 ordered IDs are identical. Subsequent acquisition is method-specific. Same seeds do not imply identical later random draws.
- All six CP configurations are identical except target, seed, and output. Frozen protocol identifies KIF11 seed0 as the development cell; it is a development recheck, not an independent held-out success. The other five cells follow the frozen settings. Two seeds and three targets are preliminary evidence, not a robust generalization claim.
- The stored historical `glide_le` objective differs from raw Glide. Primary comparisons here correctly rerun GA-v0 and v12 against raw Glide, rather than crediting an objective mismatch. A fresh raw-objective SpaceHASTEN-style adapted control is now available for KIF11 seed0; historical utility-objective archives remain contextual.
- CP64 uses capped-synthon Morgan counts512 plus four descriptors, a trainable MLP, multiplicative rank64 factors, additive heads, and reaction offsets. It is not a frozen Chemeleon model. Diversity acquisition uses precomputed full-product RDKit path fingerprints2048; final OPDiv uses separately recomputed Morgan radius2/2048, no chirality. The acquisition therefore relies on enumerated product structures/cache at this scale.
- No chemistry or target-valid prefilter is applied in these runs. Target-valid includes chemistry eligibility AND docking success; using it prospectively would leak unavailable outcomes. Finite raw scores from filter-failing molecules are deliberately included in the primary analysis.

## Scores and chemistry-eligible caveat

Raw top100 is canonical-molecule deduplicated; valid-only top100 below is the supplementary route-level statistic among target-valid rows, so it must not be labeled the same molecular statistic. Lower raw energies are better.

|Target|Seed|CP raw top100|GA raw top100|CP valid-only top100|GA valid-only top100|CP valid calls|GA valid calls|
|---|---:|---:|---:|---:|---:|---:|---:|
|kif11|0|-8.611840|-8.148659|-7.422325|-7.127482|2068|2853|
|kif11|1|-8.725321|-8.686641|-7.215603|-6.982206|1346|2060|
|pyrd|0|-11.061502|-11.127849|-10.715873|-10.491401|2693|3143|
|pyrd|1|-11.089027|-11.179258|-10.680011|-10.498190|2559|3397|
|tgfr1|0|-9.584070|-9.500328|-9.055944|-8.784456|2464|3150|
|tgfr1|1|-9.540248|-9.324081|-8.994450|-8.761921|2560|3435|

CP wins primary raw top100 in 4/6 cells (both PYRD cells lose), supplementary target-valid route top100 in 6/6, but obtains fewer target-valid calls in 6/6. Thus the caveat is lower eligible-hit yield, not failure of the best eligible scores against GA-v0. KIF11 seed0 frozen CP valid-only top100 is -7.422325 versus archived v12 -7.420133: essentially tied at the scale of this single run, despite a small numerical CP advantage. Earlier development CP results are not the final frozen result.

## OPDiv audit

OPDiv maximizes mean negative raw energy over exactly 100 queried canonical molecules with every pair satisfying Morgan Tanimoto <=0.4. Equality is permitted. Canonical isomeric SMILES deduplication keeps the first queried representative; full finite parseable archives are used, without score-prefix truncation. This is optimal selection within each queried archive, not the global 1M library optimum. GPDiv is score-ordered greedy selection under the same constraint.

All 12 full-archive OPDiv solves reached `optimal`; lower/upper gaps are <6e-9 under the package numerical certification tolerance. CP lower bound exceeds GA upper bound in all six cells. See `opdiv_full_archive.csv` and `opdiv_paired.csv`; solver time limit was 30s per solve, excluding graph/model construction. Archived v12 KIF11 seed0 also reached optimal: OPDiv=7.664118274 versus CP=8.063439023. No feasible solution was relabeled as an optimum. Independent checks of all 12 saved OPDiv portfolios confirm exactly 100 unique canonical molecules, pairwise Tc<=0.4, and recorded objective equal to the mean saved utility.

## Runtime and provenance limits

CP measured search-loop runtime is 5.39–5.76s versus GA-v0 10.07–12.64s; archived v12 took 42.16s. These are cached-oracle search runtimes, not live docking wall times. CP timing excludes initial cache loading, imported dependencies, original product enumeration/fingerprint generation, cache preparation, and post-hoc portfolio evaluation. The 1M_v2 preparation manifest records 5.77s using an already existing fingerprint cache; that is not the cost of creating the original cache. Hardware differs by algorithm (CP CUDA, GA-v0 CPU, v12 CUDA XGBoost), and timing is single-run evidence.

The frozen `factor_model.py` SHA matches current source. Before the later 100M identity repair, reverting the two lazy-fingerprint-loading lines in `run.py` reproduced the frozen SHA. That exact primary-run driver is preserved as `frozen_run.py`, with SHA `c362865bfb9c3a324c1c3e806379b6cc1749ee368dbce189987a27ccaf4e63a7`. All six runs have diversity=0.5, so lazy fingerprint loading leaves their behavior unchanged. Current `run.py` additionally handles duplicate product IDs for 100M; the archived driver, rather than current source, defines the exact primary-run provenance. Native baselines record archived source hashes/commits and package versions.

## Fresh raw SpaceHASTEN-style adapted control

`runs/docking_baselines/sh_raw_kif11_s0` reruns the unchanged local `SpaceHASTENIterative` and product-weighted allocator, not native FTrees/SpaceLight. The optimizer uses original legacy SynthonSpace semantics (first-row SMILES lookup for reused IDs); the shared public oracle/harness charges 10,000 unique raw-docking queries, with 179 nonfinite outcomes penalized to zero. All original defaults remain, including 32 warm-up observations per reaction and native per-reaction refitting. These warm starts differ from GA/SynFactor. No extra uncharged seed labels are supplied. Runtime was 85.34s.

Raw molecular top100=-8.186424; GPDiv100/Tc.4=7.187912; target-valid route top100=-6.144262, with 1,120 target-valid calls. OPDiv timed out with a feasible portfolio: lower bound=7.188611601, upper bound=7.412668343, gap=0.224056742. This is not a certified OPDiv value. Nevertheless the frozen SynFactor KIF11 seed0 lower bound 8.063439023 exceeds the SpaceHASTEN-style upper bound, certifying superiority for these two queried archives under this metric. Only one target/seed is available for this baseline.

Final-driver validation after canonical-ID changes reproduced the complete 10,000-row KIF11 seed0 frozen query trace exactly (including order and utilities). No primary metric rerun was needed.
