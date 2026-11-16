"""Run an unchanged archived GA-DCSO against a raw-docking table oracle.

This changes the objective supplied to the historical benchmark, not the GA
implementation. Failed/nonfinite table scores are returned as raw 0.0 and still
consume a query. Full measured values are retained in the frozen source table
for a separate post-hoc evaluator; no true-score eligibility filter is applied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd


def source_provenance(root, modules):
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    files = {}
    for module in modules:
        path = Path(inspect.getfile(module))
        files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"repository": str(root), "commit": commit, "loaded_source_sha256": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/home/mirekl/SynthonBench_ICML"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--target", choices=["kif11", "pyrd", "tgfr1"], default="kif11")
    parser.add_argument("--scale", choices=["1M", "10M", "100M"], default="1M")
    parser.add_argument("--kind", choices=["glide", "surrogate"], default="glide")
    parser.add_argument("--method", default="ga_dcso",
                        choices=["ga_dcso", "ga_dcso_rdk_v12", "ga_dcso_rdk_v12_extrees_fp"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--failure-raw-score", type=float, default=0.)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.budget <= 0 or not np.isfinite(args.failure_raw_score):
        parser.error("Budget must be positive and failure score finite")
    if args.kind == "glide" and args.scale != "1M":
        parser.error("Measured Glide table is only available at 1M")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "observed.csv").exists():
        parser.error("Output already contains a query trace; use a new output directory")
    internal = args.workspace / "synthon_bench_internal"
    public = args.workspace / "synthonbench"
    sys.path[:0] = [str(internal / "archived/src"), str(public / "src")]
    from synthonbench import load_space
    from synthonbench import ids as id_module, space as space_module, task as task_module, runner as runner_module
    from synthonbench.task import GlobalSynthonTask
    from synthonbench.runner import run_optimizer
    from synthon_bench_internal.methods import create_method
    from synthon_bench_internal import bigscale as bigscale_module
    from synthon_bench_internal.baselines import ga_dcso, features
    from synthon_bench_internal.bigscale import load_bigscale

    data = args.data_dir or args.workspace / "hf_synthonbench_dataset"
    table = data / "scores" / ("glide_1M.parquet" if args.kind == "glide" else f"surrogate_{args.scale}.parquet")
    column = f"{args.target}_glide_raw" if args.kind == "glide" else f"{args.target}_predicted_dock_score"
    load_started = time.perf_counter()
    space = load_space(args.scale, data_dir=data, auto_download=False)
    indexed = load_bigscale(table, space, objective_column=column, direction="minimize",
                           report_column=column, report_direction="minimize", default=args.failure_raw_score)

    class FiniteRawOracle:
        direction = "minimize"

        def __init__(self):
            self.nonfinite_count = 0

        def __call__(self, reaction_id, synthon_ids):
            value = indexed(reaction_id, synthon_ids)
            if not np.isfinite(value):
                self.nonfinite_count += 1
                return args.failure_raw_score
            return value

    oracle = FiniteRawOracle()
    task = GlobalSynthonTask(space, oracle, budget=args.budget, seed=args.seed,
                            direction="minimize", task_id=f"{args.kind}_{args.scale}_{args.target}_raw")
    optimizer = create_method(args.method, space=space, allowed_reactions=task.allowed_reactions,
                              seed=args.seed, budget=args.budget)
    estimator = getattr(optimizer, "_Estimator", None)
    if args.method == "ga_dcso" and estimator is None:
        raise RuntimeError("GA-DCSO could not import ExtraTrees; refusing a silent random fallback")
    if args.method == "ga_dcso_rdk_v12":
        from synthon_bench_internal.baselines.ga_dcso_rdk import _load_xgboost
        if _load_xgboost() is None:
            raise RuntimeError("Native v12 requires XGBoost; refusing an ExtraTrees substitution")
    setup_seconds = time.perf_counter() - load_started
    print(f"Starting unchanged {type(optimizer).__name__}: {column}, budget={args.budget}, seed={args.seed}", flush=True)
    started = time.perf_counter()
    try:
        result = run_optimizer(optimizer, task, method=args.method)
    finally:
        if task.trace:
            pd.DataFrame(task.trace).to_csv(args.out / "observed.csv", index=False)
    elapsed = time.perf_counter() - started
    modules = [ga_dcso, features, bigscale_module]
    if args.method != "ga_dcso":
        from synthon_bench_internal.baselines import ga_dcso_rdk, ga_dcso_rdk_v12
        modules.extend([ga_dcso_rdk, ga_dcso_rdk_v12])
    summary = {
        "method": args.method, "implementation": f"{type(optimizer).__module__}.{type(optimizer).__name__}",
        "target": args.target, "scale": args.scale, "kind": args.kind, "seed": args.seed,
        "objective_column": column, "objective_direction": "minimize", "utility": "negative raw score",
        "budget": args.budget, "submitted_calls": result.submitted_calls,
        "unique_queried_products": len(result.observed_ids), "duplicate_attempts": result.duplicate_attempts,
        "best_utility": result.best_utility, "best_raw_score": task.best_raw_score,
        "setup_seconds": setup_seconds, "run_seconds": elapsed,
        "nonfinite_oracle_values_replaced": oracle.nonfinite_count,
        "nonfinite_raw_score_policy": args.failure_raw_score,
        "eligibility": "no chemistry or docking-success prefilter; all allowed reaction tuples",
        "oracle_access": "unique charged query via unchanged public GlobalSynthonTask/run_optimizer",
        "batch_sizes": getattr(optimizer, "batch_sizes", None),
        "diagnostics": result.diagnostics,
        "source": source_provenance(internal, modules),
        "public_harness": source_provenance(public, [id_module, space_module, task_module, runner_module]),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn")},
        "score_table": str(table.resolve()),
        "score_index_manifest": json.loads(Path(str(table) + ".sidx/manifest.json").read_text()),
    }
    if args.method == "ga_dcso_rdk_v12":
        import xgboost
        summary["versions"]["xgboost"] = xgboost.__version__
        summary["xgboost_package_path"] = str(Path(xgboost.__file__).resolve())
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("submitted_calls", "run_seconds", "best_raw_score", "nonfinite_oracle_values_replaced")}), flush=True)


if __name__ == "__main__":
    main()
