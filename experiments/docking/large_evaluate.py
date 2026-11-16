"""Memory-bounded post-hoc evaluation of 10M/100M frozen-surrogate runs.

Only queried records are materialized; exact reference tails come from the
released top-k sidecar. Portfolio definitions are shared with evaluate.py.
This evaluator is intentionally separate from the optimizer and its budget.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import mmap
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .evaluate import molecular_archive, portfolio_metrics, read_observed


def indexed_values(path, column, ids):
    """Released indexed oracle uses the first row for duplicate 128-bit IDs."""
    directory = Path(str(path) + ".sidx")
    if not directory.exists():
        return None
    hi = np.load(directory / "keys_hi.npy", mmap_mode="r")
    lo = np.load(directory / "keys_lo.npy", mmap_mode="r")
    values = np.load(directory / f"vals__{column}.npy", mmap_mode="r")
    result = np.full(len(ids), np.nan)
    # Small independent batches plus explicit mmap page eviction prevent random
    # binary searches from retaining gigabytes of index pages in resident RAM.
    for offset in range(0, len(ids), 128):
        batch = ids[offset:offset+128]
        wanted_hi = np.asarray([int(pid[:16], 16) for pid in batch], np.uint64)
        wanted_lo = np.asarray([int(pid[16:], 16) for pid in batch], np.uint64)
        left = np.searchsorted(hi, wanted_hi, "left")
        right = np.searchsorted(hi, wanted_hi, "right")
        for j, (start, end) in enumerate(zip(left, right)):
            row = int(start) + int(np.searchsorted(lo[start:end], wanted_lo[j], "left"))
            if row < end and lo[row] == wanted_lo[j]:
                result[offset+j] = values[row]
        for array in (hi, lo, values):
            if hasattr(array._mmap, "madvise"):
                array._mmap.madvise(mmap.MADV_DONTNEED)
    return result


def load_queried(path, target, ids, batch_size=200000):
    """Arrow membership pushdown, no full-table pandas or read-ahead buffers."""
    column = f"{target}_predicted_dock_score"
    dataset = ds.dataset(path, format="parquet")
    if column not in dataset.schema.names:
        raise ValueError("large_evaluate expects a frozen surrogate score table")
    columns = ["product_id", "smiles", column]
    scanner = dataset.scanner(
        columns=columns, filter=ds.field("product_id").isin(ids),
        batch_size=batch_size, use_threads=False,
        batch_readahead=0, fragment_readahead=0,
        fragment_scan_options=ds.ParquetFragmentScanOptions(pre_buffer=False),
    )
    frames = [batch.to_pandas() for batch in scanner.to_batches() if batch.num_rows]
    matched = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    alias_rows = int(matched.product_id.duplicated().sum())
    duplicate_ids = int(matched.loc[matched.product_id.duplicated(), "product_id"].nunique())
    # Dataset scanner preserves physical row order with threading disabled.
    # Public ScoreTableIndex.get selects the first identical sorted key, hence
    # the first source row. Validate against that independent released index.
    matched = matched.drop_duplicates("product_id", keep="first")
    expected = indexed_values(path, column, matched.product_id.tolist())
    if expected is not None and not np.allclose(matched[column], expected, atol=1e-12, rtol=0, equal_nan=True):
        raise ValueError("First source rows do not match released indexed oracle")
    if alias_rows and expected is None:
        raise ValueError("Duplicate product IDs require independent .sidx verification")
    matched["product_id"] = matched.product_id.astype(str)
    n_matched = len(matched)
    # Re-establish submitted order for canonical-molecule deduplication.
    queried = matched.set_index("product_id").reindex(ids).reset_index()
    queried.attrs["product_id_identity"] = dict(policy="first source row, matching released .sidx",
        duplicate_queried_ids=duplicate_ids, excluded_alias_rows=alias_rows,
        indexed_oracle_verified=expected is not None)
    return queried, column, n_matched


def reference_tail(path, column, table_rows, count=1000):
    directory = Path(str(path) + ".topk")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = [entry for entry in manifest["entries"]
               if entry["score_column"] == column and entry["direction"] == "minimize"]
    if len(entries) != 1:
        raise ValueError(f"Need exactly one minimizing top-k sidecar for {column}")
    entry = entries[0]
    needed = min(count, table_rows)
    if entry["n_rows"] != table_rows or entry["top_n"] < needed:
        raise ValueError("Top-k manifest row count or tail capacity does not match")
    source = directory / entry["path"]
    tail = pd.read_csv(source, nrows=needed, usecols=["rank", "product_id", column], dtype={"product_id": str})
    if len(tail) != needed or tail.product_id.duplicated().any():
        raise ValueError("Incomplete or duplicate reference tail")
    if not np.array_equal(tail["rank"].to_numpy(), np.arange(1, needed+1)):
        raise ValueError("Reference tail is not a consecutive ordered prefix")
    values = tail[column].to_numpy(float)
    if not np.isfinite(values).all() or np.any(np.diff(values) < 0):
        raise ValueError("Reference tail is nonfinite or not minimizing score order")
    expected = indexed_values(path, column, tail.product_id.tolist())
    if expected is not None and not np.allclose(values, expected, atol=1e-8, rtol=0, equal_nan=True):
        raise ValueError("Released top-k prefix includes a noncanonical alias score; rebuild canonical tail")
    return tail, {"indexed_oracle_verified": expected is not None, "manifest": str(manifest_path.resolve()),
                  "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  "path": str(source.resolve()), "reference_rows": len(tail),
                  "tie_policy": "released sidecar order: score then product_id",
                  "prefix_sha256": hashlib.sha256(tail.to_csv(index=False).encode()).hexdigest()}


def bounded_score_metrics(queried, ids, column, charged, table_rows, matched, tail):
    raw = queried[column].to_numpy(float)
    finite = np.isfinite(raw)
    scores = {"matched_ids": matched, "finite_raw_scores": int(finite.sum()),
              "nonfinite_raw_scores": int((~finite).sum())}
    ranked = np.sort(raw[finite])
    for k in (1, 10, 100, 1000):
        values = ranked[:k]
        scores[f"route_raw_top{k}_mean"] = float(values.mean()) if len(values) else None
        scores[f"route_raw_top{k}_count"] = len(values)
    observed = set(ids)
    for k in (100, 1000):
        actual_k = min(k, len(tail))
        hits = len(observed & set(tail.product_id.iloc[:actual_k]))
        recall = hits / actual_k if actual_k else None
        scores[f"global_raw_top{k}_hits"] = hits
        scores[f"global_raw_top{k}_recall"] = recall
        scores[f"global_raw_top{k}_dcrf"] = recall * table_rows / charged if recall is not None and charged else None
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-table", type=Path, required=True)
    parser.add_argument("--target", choices=["kif11", "pyrd", "tgfr1"], required=True)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=200000)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[.4])
    parser.add_argument("--ks", type=int, nargs="+", default=[100])
    parser.add_argument("--optimal", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_seconds <= 0 or any(k < 1 for k in args.ks) or any(not 0 <= t <= 1 for t in args.thresholds):
        parser.error("Need positive batch size/time/k and thresholds in [0,1]")
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    ids, metadata = read_observed(args.observed)
    if not ids:
        raise ValueError("No queried product IDs")
    table_rows = pq.ParquetFile(args.score_table).metadata.num_rows
    queried, column, matched = load_queried(args.score_table, args.target, ids, args.batch_size)
    if args.observed.suffix == ".csv":
        observed = pd.read_csv(args.observed)
        if "utility" in observed and not np.allclose(observed.utility, -queried[column], atol=1e-5, rtol=0):
            raise ValueError("Observed utilities disagree with indexed oracle: invalidate and rerun search")
    tail, reference = reference_tail(args.score_table, column, table_rows)
    scores = bounded_score_metrics(queried, ids, column, metadata["submitted_calls"], table_rows, matched, tail)
    archive, identity = molecular_archive(queried, column, args.target, "finite")
    for k in (1, 10, 100, 1000):
        values = archive.raw_score.iloc[:k]
        scores[f"molecular_raw_top{k}_mean"] = float(values.mean()) if len(values) else None
        scores[f"molecular_raw_top{k}_count"] = len(values)
    archive.to_csv(args.out / "eligible_molecules.csv.gz", index=False)
    summary = {
        "target": args.target, "score_table": str(args.score_table.resolve()),
        "score_kind": "frozen_docking_surrogate", "raw_score_column": column,
        "table_rows": table_rows, "observed_source": str(args.observed.resolve()),
        "observed_sha256": hashlib.sha256(args.observed.read_bytes()).hexdigest(),
        "archive": metadata, "scores": scores, "identity": identity,
        "product_id_identity": queried.attrs["product_id_identity"],
        "portfolio_eligibility": "finite",
        "fingerprint": {"type": "Morgan", "radius": 2, "n_bits": 2048, "chirality": False},
        "portfolio_utility": "negative frozen docking-surrogate score; higher is better",
        "duplicates": "canonical isomeric molecule; first representative in supplied ID order retained (historical JSON archives are sorted sets, not query order)",
        "ties": "raw score then product_id; equality to similarity ceiling is allowed",
        "reference_tail": reference,
        "loading": {"membership_filter": "pyarrow.dataset product_id.isin(queried_ids)",
                    "use_threads": False, "batch_size": args.batch_size,
                    "pre_buffer": False, "batch_readahead": 0, "fragment_readahead": 0,
                    "materialized_queried_rows": len(queried)},
        "time_limit_scope": "seconds per CP-SAT solve; graph/model construction excluded",
        "opdiv_version": importlib.metadata.version("opdiv"),
    }
    def save():
        summary["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        summary["wall_seconds"] = time.perf_counter()-started
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    save()
    records, portfolios, graph_seconds = portfolio_metrics(archive, args.thresholds, args.ks, args.max_seconds, args.optimal)
    pd.DataFrame(records).to_csv(args.out / "portfolio_metrics.csv", index=False)
    pd.DataFrame(portfolios).to_csv(args.out / "portfolios.csv", index=False)
    summary.update(portfolios=records, fingerprint_graph_seconds=graph_seconds)
    save()
    print(f"Saved {args.out}; peak RSS {summary['peak_rss_mib']:.0f} MiB; {summary['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
