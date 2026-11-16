# Full-Freedom scoring and Dock-12 qualification

Measured on 2026-11-16 with an RTX 4090. SynFactor CP64 + additive terms was
fitted to the shared TYK2 seed-0 cohort: 1,000 charged observations, 997 valid.
The GPU then scored **all 296,350,180,179 routes** in the pinned Freedom release
and retained the top 20,000 unqueried routes.

| Stage | Seconds |
| --- | ---: |
| One-time label-free synthon feature preparation | 128.76 |
| Neural fitting and factor caching | 5.32 |
| Complete Cartesian scoring and top-20,000 selection | **19.82** |
| Loading, fitting and scoring | 27.28 |

The sweep used 7.13 GiB peak allocated GPU memory. All 15,817 score blocks were
visited; the 1,000 warm routes were excluded. Direct tuple predictions on 100
returned routes agreed within 1.91e-6. See [measured records](first_sweep_summary.json)
and [space/cache provenance](space_summary.json). This measures surrogate
scoring, not docking 296 billion molecules. Routes need not be unique molecules.
The exact [qualification script](frozen_qualification.py) and
[source hashes](source_sha256.json) are retained. That script uses the recorded
local cache/benchmark paths; it is separate from the portable quick-start demo.

The separate remote oracle qualification assembled and docked four known TYK2
routes. All four molecular structures and Glide scores matched the archived
warm observations exactly; [qualification record](smoke_validation.json).
These four engineering calls do not enter adaptive training or search budgets.

The live pilot uses twelve Dock-12 targets, seed 0, 10,000 charged attempts per
target (1,000 shared warm + 9 × 1,000 adaptive). Each batch allocates 700 choices
to score/diversity, 200 to conditional information gain, and 100 to random
exploration. The full Cartesian space is scored after each neural refit.
The campaign was launched on 2026-11-16; this qualification record alone does
not establish a docking-score or portfolio improvement over other methods.

The [first completed target, TYK2 seed 0](tyk2_seed0/README.md), has a matched
10,000-query comparison and certified/bounded portfolio results: stronger than
both SpaceHASTEN variants on OPDiv, weaker than GA-DCSO v14 on unfiltered scores
and OPDiv, and slightly stronger than GA on druglike OPDiv. The remaining
eleven targets were still running when this first-target report was written.

See [implementation and protocol](../../experiments/docking/LIVE.md) for
reproduction requirements and matched GA-DCSO v14 / SpaceHASTEN evaluation.
The full-space experiment needs the original Freedom release, prepared docking
grids and licensed oracle installation; these assets are not bundled here.
