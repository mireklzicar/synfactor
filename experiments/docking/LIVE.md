# Full Freedom Dock-12 integration

This optional research integration performs neural fitting and Cartesian scoring
on the local CUDA machine, then sends only selected route IDs to the existing
licensed Dock-12 oracle host. It requires the local SynthonBench V2 workspace,
Freedom release, archived warm cohort, prepared grids and authorized remote
oracle installation. It is not part of the standalone public quick-start demo.

## Full-space scoring qualification

The pinned Freedom Space 5 release contains **123 reactions, 1,822,466 typed
synthon rows and 296,350,180,179 Cartesian routes**. These are route counts,
not a claim that every route gives a different molecule.

The measured RTX 4090 qualification fit CP64 to the shared 1,000 TYK2 warm
observations and visited the entire release:

| Stage | Measured result |
| --- | ---: |
| One-time feature/cache preparation | 128.8 s |
| CP64 fitting and factor-cache creation | 5.32 s |
| Complete Cartesian score sweep + top-20,000 retention | **19.82 s** |
| Loading + fit + sweep | 27.28 s |
| Peak sweep GPU allocation | 7.13 GiB |

All 1,000 charged warm routes were excluded from acquisition. Returned scores
agreed with direct tuple prediction on 100 checks within 1.91e−6. This was an
actual complete sweep, not a trillion-scale extrapolation. It did **not** dock
296B products or evaluate a molecular encoder for every product.

`cartesian.py` streams two-slot factor matrix products and three-slot pair-factor
matrix products in bounded blocks, retaining global top-k routes. An exact
running cutoff avoids sorting blocks that cannot enter top-k. Every supplied
Cartesian combination is still visited. A time-limited incomplete sweep raises
an exception rather than returning a misleading global result. Exact floating
ties use reaction and local Cartesian order; float32 matrix accumulation can
differ slightly from elementwise tuple accumulation.

```bash
python -m experiments.docking.fullspace \
  --library /path/to/freedom5_skach --out runs/fullspace/freedom
```

Only unique source synthon structures are encoded before expansion to typed
rows. Capping failures retain the valid original synthon representation and
are counted, rather than silently deleting reactions. No docking labels are
used in feature preparation.

## Oracle and campaign contract

The Dock-12 roster is TYK2, JNK1, p38α, KIF11, PYRD/DHODH, TGFR1/TGFBR1,
ALDH1, HTR1A, DRD3, PPARα, AChE and BRD4. KIF11/PYRD/TGFR1 use the frozen
SpaceHASTEN continuity grids. The BRD4 grid must match the Orion Navigator/
SpaceHASTEN archives; another public-method campaign uses a different grid.

Each seed/target receives **10,000 charged route attempts**: 1,000 shared warm
observations plus nine adaptive batches of 1,000. The information experiment
uses 700 exploitation/diversity, 200 information-directed and 100 random
selections per adaptive batch. Failed observations stay charged and return
zero training utility. The complete raw Freedom release is searched without
a prospective druglike gate.

The oracle uses unchanged Navigator 0.4.0 native assembly and the existing
Glide harness: HTVS, OPLS_2005, Epik classic at pH 7, one stereoisomer and one
requested pose, 32 workers. A separate four-route TYK2 engineering smoke
matched all four archived structures **and scores exactly**; its four calls
are separate from optimization budgets and do not enter adaptive training.

`remote_oracle.py` provides read-only preflight unless `--execute` is supplied.
It pins source, space/index, grid and warm-cohort hashes; checks cross-batch
identity and the 9,000-adaptive-attempt cap; retains raw scores and pose files;
and refuses to automatically resubmit uncertain work. The remote runtime must
provide `uv` and standard system tools including `ldconfig`. The local
transport explicitly retains both user-bin and system-sbin directories.

```bash
# Read-only check of all twelve targets:
python -m experiments.docking.live \
  --cache runs/fullspace/freedom --out runs/live_info20 \
  --host ubuntu@YOUR_DOCKING_HOST --key /path/to/existing/key

# Same command with --execute runs the seed-0 suite:
python -m experiments.docking.live \
  --cache runs/fullspace/freedom --out runs/live_info20 \
  --host ubuntu@YOUR_DOCKING_HOST --key /path/to/existing/key --execute
```

`--targets tyk2` restricts the roster; `--seed` selects a frozen warm cohort.
`--information-fraction 0` is the acquisition control. Default sweep deadline
is 300 seconds. The one-hour cell budget sums recorded GPU search and oracle
stage times and is checked between batches; it does not interrupt in-flight
docking. It excludes transport, preflight and assembly overhead, so it is not
a strict end-to-end wall-time cap. Both stage timings are recorded separately. The remote bridge's configured root and installed asset paths must
match the authorized host; do not overwrite another campaign's working files.

For sustained execution, use a frozen source snapshot and retain its hashes.
Local state, committed proposals and normalized responses are resumable, but
changed science/source settings or an unresolved remote submission stop resume.
Infrastructure failures proven to precede docking require an explicit recorded
reconciliation; they are not silently converted into failed molecular scores.

## Matched evaluation

```bash
.venv-eval/bin/python -m experiments.docking.live_evaluate \
  --history runs/live_info20/sf_freedom_seed0_tyk2_info20/state.json \
  --target tyk2 --seed 0 --grid-sha256 GRID_SHA \
  --warm-sha256 WARM_CSV_SHA --out runs/live_info20/tyk2_evaluation
```

This reports unique-molecule top-100/top-1,000 scores and GPDiv/OPDiv100 using
**strict Morgan Tanimoto < 0.30**, radius 2, 2048 bits, no chirality. This differs
from the subset pilot's inclusive ≤0.40 threshold. Both all-valid and the exact
production `druglike-v1` gate are evaluated on original molecular forms.
Failures and retrospectively excluded chemistry remain charged.

Comparators are the archived Orion `ga_dcso_v14_screening`, `spacehasten_fast`
and `spacehasten_accurate` methods. Grid, warm cohort, archive hash, gate and
fingerprint contracts are checked. Historical optimized bounds are reused only
when compatible; otherwise the same current evaluator recomputes both sides.
Feasible solutions, bounds and certified optima remain distinct. These checks
are needed before calling any live target a win.

Matched comparison also verifies the original warm-source hashes and every
charged batch response against the history. Keep `config.json`, preflight and
batch evidence beside `state.json`; supplying hashes on the command line cannot
replace that evidence. `--skip-baselines` permits standalone evaluation of a
copied history without asserting a matched historical comparison.

Dense unfiltered archives may leave loose CP-SAT bounds at the initial
30-second solver limit. `live_refine.py` optionally uses the original Orion
SCIP routines for exact score-aware domination, clique constraints and a
stronger full-domain bound. It requires `pyscipopt==6.3.0`, SciPy and the
recorded SynthonBench V2 analysis sources in the evaluation environment.
Restricted-support optimization improves only the feasible solution; its
bound is never treated as a global certificate. Refinement writes separate
files and keeps the initial evaluation intact.

```bash
.venv-eval/bin/python -m experiments.docking.live_refine \
  --molecules /path/to/cell/evaluation_live/synfactor__all-valid__molecules.csv.gz \
  --out /path/to/cell/refinement/synfactor__all-valid.json
```

The first [TYK2 matched comparison](../../benchmarks/docking_fullspace/tyk2_seed0/README.md)
uses these refinements alongside the initial certified druglike portfolios.

## Unattended completion

`live_monitor.py` can supervise an already running campaign. It pins the process
identity and evaluator source, evaluates completed targets serially, and writes
`monitor_progress.md` plus machine-readable state in the campaign directory.
It never submits molecules, restarts the campaign or retries uncertain docking.

```bash
python -m experiments.docking.live_monitor \
  --campaign /absolute/path/to/live_info20 --pid CAMPAIGN_PID \
  --evaluator-root /absolute/path/to/frozen/evaluator/source \
  --eval-python /absolute/path/to/.venv-eval/bin/python \
  --host ubuntu@YOUR_DOCKING_HOST --key /path/to/existing/key
```

If this campaign started a previously stopped instance, the optional
`--restore-instance --instance-audit /path/to/start_audit.json` restores that
state after the campaign exits and evaluation finishes. The audit must record
`instance_id`, `profile`, `region`, `public_ip`, `previous_state: "stopped"` and
`start_requested_by_this_campaign: true`. Restoration verifies the pinned audit,
instance identity and remote idleness twice; unrelated work or uncertain
inspection prevents shutdown and is recorded in the progress report. AWS stop
acknowledgement and verified stopped state are reported separately.

`live_refine_watch.py` can additionally refine all four methods' unfiltered
portfolios as targets finish. It runs only local CPU jobs, using a frozen
source directory containing `live_refine.py`, `live_refine_watch.py`,
`live_evaluate.py` and `live_monitor.py`:

```bash
.venv-eval/bin/python -m experiments.docking.live_refine_watch \
  --campaign /absolute/path/to/live_info20 \
  --frozen-root /absolute/path/to/frozen/refinement/source \
  --eval-python /absolute/path/to/.venv-eval/bin/python
```

Its separate `refined_progress.md` reports full-domain intervals and initial
druglike results. Failed refinements remain explicit; it neither retries
docking nor changes the search policy. Local refinement can finish after the
remote docking instance has been restored to its stopped state.
