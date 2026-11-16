"""Raw-oracle adapter for the unchanged local SpaceHASTEN-style baseline.

This is the ExtraTrees/product-weighted benchmark adaptation, not native
SpaceHASTEN/FTrees/SpaceLight. No optimizer parameters are changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from .baseline_run import source_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/home/mirekl/SynthonBench_ICML"))
    parser.add_argument("--target", choices=["kif11", "pyrd", "tgfr1"], default="kif11")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "observed.csv").exists():
        parser.error("Refusing to overwrite an existing trace")
    bench = args.workspace / "synthon_bench"
    public = args.workspace / "synthonbench"
    internal = args.workspace / "synthon_bench_internal"
    sys.path[:0] = [str(bench / "src"), str(public / "src"), str(internal / "archived/src")]
    from synthonbench import load_space
    from synthonbench.task import GlobalSynthonTask
    from synthonbench.runner import run_optimizer
    from synthon_bench.baselines import spacehasten_iterative, global_routing
    from synthon_bench.api import assembly
    from synthon_bench.api import space as legacy_space
    from synthon_bench_internal.bigscale import load_bigscale
    data = args.workspace / "hf_synthonbench_dataset"
    table = data / "scores/glide_1M.parquet"
    column = f"{args.target}_glide_raw"
    start = time.perf_counter()
    space = load_space("1M", data_dir=data, auto_download=False)
    optimizer_space = legacy_space.SynthonSpace(data / "spaces/synthon_space_1M.synthons.tsv")
    indexed = load_bigscale(table, space, objective_column=column, direction="minimize",
                           report_column=column, report_direction="minimize", default=0.)

    class Oracle:
        direction = "minimize"
        nonfinite = 0

        def __call__(self, reaction_id, synthon_ids):
            value = indexed(reaction_id, synthon_ids)
            if not np.isfinite(value):
                self.nonfinite += 1
                return 0.
            return value

    oracle = Oracle()
    task = GlobalSynthonTask(space, oracle, budget=args.budget, seed=args.seed,
                            direction="minimize", task_id=f"glide_1M_{args.target}_raw")
    optimizer = global_routing.create_routed_global_optimizer(
        "spacehasten_iterative", allowed_reactions=task.allowed_reactions, space=optimizer_space,
        seed=args.seed, allocator_policy="product_weighted", inner_kwargs={})
    setup = time.perf_counter() - start
    print("Running unchanged SpaceHASTEN-style adaptation on raw Glide", flush=True)
    start = time.perf_counter()
    try:
        result = run_optimizer(optimizer, task, method="spacehasten_iterative_product_weighted")
    finally:
        if task.trace:
            pd.DataFrame(task.trace).to_csv(args.out / "observed.csv", index=False)
    summary = {"method": "SpaceHASTEN-style adapted ExtraTrees product-weighted",
               "not_native_spacehasten": True, "target": args.target, "seed": args.seed,
               "budget": args.budget, "submitted_calls": result.submitted_calls,
               "unique_queried_products": len(result.observed_ids),
               "duplicate_attempts": result.duplicate_attempts,
               "setup_seconds": setup, "run_seconds": time.perf_counter()-start,
               "objective_column": column, "direction": "minimize",
               "nonfinite_raw_score_policy": 0., "nonfinite_calls": oracle.nonfinite,
               "warmstart": "unchanged 32 per reaction with product-weighted allocation; all queries charged",
               "eligibility": "no chemistry/docking-success prefilter",
               "diagnostics": result.diagnostics,
               "space_semantics": "legacy SynthonSpace for optimizer, preserving original reused-ID SMILES lookup",
               "source": source_provenance(bench, [spacehasten_iterative, global_routing, assembly, legacy_space])}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k: summary[k] for k in ["submitted_calls", "run_seconds", "nonfinite_calls"]}), flush=True)


if __name__ == "__main__":
    main()
