"""Post-hoc docking and molecular portfolio evaluation, isolated from search.

Reads full score tables ONLY after an optimizer has committed its queried IDs.
Historical Glide runs optimize stored LE but report raw docking energies. Both
are reported here, with explicitly separate eligibility and ranking semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def read_observed(path: str | Path):
    """Read our ordered CSV, archived GA JSON, or adapted SH queried CSV/TXT."""
    path = Path(path)
    metadata = {}
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            ids = payload
        else:
            key = next((k for k in ("evaluated_product_ids", "queried_product_ids", "observed_ids") if k in payload), None)
            if key is None:
                raise ValueError(f"No observed product-ID array in {path}")
            ids = payload[key]
            metadata = {k: payload[k] for k in ("method", "dataset", "target", "seed", "budget", "submitted_calls", "unique_queried_count") if k in payload}
    elif path.suffix in ("txt", ".txt"):
        ids = path.read_text().splitlines()
    else:
        frame = pd.read_csv(path, dtype={"product_id": str})
        if "product_id" not in frame:
            raise ValueError("Observed CSV must contain product_id")
        ids = frame.product_id.tolist()
    ids = [str(x).strip() for x in ids if str(x).strip()]
    unique = list(dict.fromkeys(ids))
    charged = int(metadata.get("submitted_calls", metadata.get("unique_queried_count", len(unique))))
    if charged < len(unique):
        raise ValueError("Reported charged calls are smaller than unique queried product IDs")
    return unique, {**metadata, "submitted_calls": charged, "input_rows": len(ids), "unique_observed_ids": len(unique)}


def load_scores(path, target):
    available = set(pq.read_schema(path).names)
    raw = f"{target}_glide_raw"
    kind = "measured_glide"
    if raw not in available:
        raw = f"{target}_predicted_dock_score"
        kind = "frozen_docking_surrogate"
    if raw not in available:
        raise ValueError(f"No docking score column for {target}")
    wanted = ["product_id", "smiles", "canonical_smiles", raw, f"{target}_glide_le", f"{target}_valid", "passes_exp27_exact_filter"]
    frame = pd.read_parquet(path, columns=[x for x in wanted if x in available])
    if frame.product_id.duplicated().any():
        raise ValueError("Score table must have unique product IDs")
    frame["product_id"] = frame.product_id.astype(str)
    return frame, raw, kind


def score_metrics(table, ids, raw_column, target, charged):
    """Rank scores independently of molecular graph/portfolio construction."""
    queried = table.set_index("product_id").reindex(ids)
    raw = queried[raw_column].to_numpy(float)
    finite = np.isfinite(raw)
    result = {"matched_ids": int(queried.index.isin(table.product_id).sum()),
              "finite_raw_scores": int(finite.sum()), "nonfinite_raw_scores": int((~finite).sum())}
    for k in (1, 10, 100, 1000):
        values = np.sort(raw[finite])[:k]
        result[f"route_raw_top{k}_mean"] = float(values.mean()) if len(values) else None
        result[f"route_raw_top{k}_count"] = len(values)
    valid_column = f"{target}_valid"
    if valid_column in queried:
        valid = queried[valid_column].fillna(False).to_numpy(bool)
        result["target_valid_count"] = int(valid.sum())
        for k in (1, 10, 100, 1000):
            values = np.sort(raw[finite & valid])[:k]
            result[f"valid_route_raw_top{k}_mean"] = float(values.mean()) if len(values) else None
            result[f"valid_route_raw_top{k}_count"] = len(values)
    le_column = f"{target}_glide_le"
    rankings = {"raw": (raw_column, True)}
    if le_column in table:
        le = queried[le_column].to_numpy(float)
        valid_le = le[np.isfinite(le)]
        result["best_le"] = float(valid_le.max()) if len(valid_le) else None
        result["top100_le_mean"] = float(np.sort(valid_le)[-100:].mean()) if len(valid_le) else None
        result["zero_le_count"] = int((le == 0).sum())
        rankings["le"] = (le_column, False)
    observed = set(ids)
    for name, (column, ascending) in rankings.items():
        finite_table = table[np.isfinite(table[column].to_numpy(float))]
        # Fixed deterministic tie policy; no optimizer order enters exact tails.
        order = finite_table.sort_values([column, "product_id"], ascending=[ascending, True], kind="stable")
        for k in (100, 1000):
            actual_k = min(k, len(order))
            hits = len(observed & set(order.product_id.iloc[:actual_k]))
            recall = hits / actual_k if actual_k else None
            result[f"global_{name}_top{k}_hits"] = hits
            result[f"global_{name}_top{k}_recall"] = recall
            # Historical benchmark uses all product IDs as N, including invalid
            # or filtered records; retain that denominator, and disclose it.
            result[f"global_{name}_top{k}_dcrf"] = recall * len(table) / charged if recall is not None and charged else None
    return queried.reset_index(), result


def molecular_archive(queried, raw_column, target, eligible="finite"):
    """Deduplicate canonical molecules, keeping the first queried representative."""
    selected, seen, invalid, duplicate, ineligible = [], set(), 0, 0, 0
    for row in queried.to_dict("records"):
        if not np.isfinite(row[raw_column]):
            continue
        valid_value = row.get(f"{target}_valid", False)
        if eligible == "target-valid" and not (pd.notna(valid_value) and bool(valid_value)):
            ineligible += 1
            continue
        smiles = row.get("smiles")
        if not isinstance(smiles, str) or not smiles:
            smiles = row.get("canonical_smiles")
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            invalid += 1
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if canonical in seen:
            duplicate += 1
            continue
        seen.add(canonical)
        selected.append({"product_id": row["product_id"], "smiles": smiles,
                         "canonical_smiles": canonical, "raw_score": float(row[raw_column]),
                         "utility": -float(row[raw_column])})
    archive = pd.DataFrame(selected, columns=["product_id", "smiles", "canonical_smiles", "raw_score", "utility"])
    archive = archive.sort_values(["raw_score", "product_id"], kind="stable").reset_index(drop=True)
    return archive, {"invalid_smiles_excluded": invalid, "duplicate_molecules_excluded": duplicate,
                     "eligibility_excluded": ineligible, "unique_eligible_molecules": len(archive)}


def conflict_graphs(smiles, thresholds):
    """Build bool graphs with strict Tc > threshold; no floating epsilon."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    graphs = {float(t): np.zeros((len(fps), len(fps)), dtype=bool) for t in thresholds}
    for i, fp in enumerate(fps):
        values = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, fps[i+1:]), dtype=np.float64)
        for threshold, graph in graphs.items():
            graph[i, i+1:] = values > threshold
            graph[i+1:, i] = graph[i, i+1:]
    return graphs


def portfolio_metrics(archive, thresholds, ks, max_seconds, optimal=True):
    from opdiv import select

    start = time.perf_counter()
    graphs = conflict_graphs(archive.smiles, thresholds)
    graph_seconds = time.perf_counter() - start
    utilities = archive.utility.to_numpy(float)
    records, portfolios = [], []
    for threshold, graph in graphs.items():
        for k in ks:
            for method in (("greedy", "optimal") if optimal else ("greedy",)):
                started = time.perf_counter()
                result = select(utilities, k=k, conflicts=graph, method=method,
                                **({"time_limit": max_seconds} if method == "optimal" else {}))
                records.append({"metric": "GPDiv" if method == "greedy" else "OPDiv",
                                "k": k, "max_similarity": threshold, "candidate_count": len(archive),
                                "status": result.status, "selected_count": len(result.indices),
                                "value": result.value,
                                "metric_value": result.value if method == "greedy" or result.status == "optimal" else None,
                                "feasible_lower_bound": result.value,
                                "value_is_certified_opdiv": method == "optimal" and result.status == "optimal",
                                "upper_bound": result.upper_bound, "gap": result.gap,
                                "wall_seconds": time.perf_counter() - started})
                for rank, index in enumerate(result.indices):
                    portfolios.append({"metric": records[-1]["metric"], "k": k, "max_similarity": threshold,
                                       "rank": rank + 1, **archive.iloc[index].to_dict()})
                print(f"{records[-1]['metric']} k={k} Tc={threshold}: {result.status}, value={result.value}, gap={result.gap}", flush=True)
    return records, portfolios, graph_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-table", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=20)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[.4, .6])
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 100])
    parser.add_argument("--greedy-only", action="store_true")
    parser.add_argument("--eligibility", choices=["finite", "target-valid"], default="finite")
    args = parser.parse_args()
    if args.max_seconds <= 0 or any(k < 1 for k in args.ks) or any(not 0 <= t <= 1 for t in args.thresholds):
        parser.error("Need positive time/k and thresholds in [0,1]")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ids, metadata = read_observed(args.observed)
    table, raw_column, kind = load_scores(args.score_table, args.target)
    queried, scores = score_metrics(table, ids, raw_column, args.target, metadata["submitted_calls"])
    archive, identity = molecular_archive(queried, raw_column, args.target, args.eligibility)
    for k in (1, 10, 100, 1000):
        values = archive.raw_score.iloc[:k]
        scores[f"molecular_raw_top{k}_mean"] = float(values.mean()) if len(values) else None
        scores[f"molecular_raw_top{k}_count"] = len(values)
    archive.to_csv(args.out / "eligible_molecules.csv.gz", index=False)
    # Write basic audit before solver work, so partial evaluations remain useful.
    summary = {"target": args.target, "score_table": str(args.score_table.resolve()),
               "score_kind": kind, "raw_score_column": raw_column, "table_rows": len(table),
               "observed_source": str(args.observed.resolve()),
               "observed_sha256": hashlib.sha256(args.observed.read_bytes()).hexdigest(),
               "archive": metadata, "scores": scores, "identity": identity,
               "portfolio_eligibility": args.eligibility,
               "fingerprint": {"type": "Morgan", "radius": 2, "n_bits": 2048, "chirality": False},
               "portfolio_utility": "negative raw score; higher is better",
               "duplicates": "canonical isomeric molecule; first queried representative retained",
               "ties": "raw score then product_id; equality to similarity ceiling is allowed",
               "time_limit_scope": "seconds per CP-SAT solve; graph construction/model construction excluded",
               "opdiv_version": importlib.metadata.version("opdiv")}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    records, portfolios, graph_seconds = portfolio_metrics(archive, args.thresholds, args.ks, args.max_seconds, not args.greedy_only)
    pd.DataFrame(records).to_csv(args.out / "portfolio_metrics.csv", index=False)
    pd.DataFrame(portfolios).to_csv(args.out / "portfolios.csv", index=False)
    summary.update(portfolios=records, fingerprint_graph_seconds=graph_seconds, wall_seconds=time.perf_counter()-started)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Saved post-hoc evaluation to {args.out}", flush=True)


if __name__ == "__main__":
    main()
