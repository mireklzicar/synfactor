"""Export small, reviewable pilot results and a held-out constrained query."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.stats import spearmanr

from .compare import dump
from .baselines import shi_predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/comparison"))
    parser.add_argument("--report", type=Path, default=Path("benchmarks/comparison_pilot"))
    args = parser.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    data = np.load(args.out / "comparison.npz")
    apex = np.load(args.out / "apex_predictions.npz")
    products = pd.read_csv(args.out / "products.csv")
    names, y = data["names"].tolist(), data["y"]
    predictions = {k: data[k] for k in ("shi_eq11", "shi_calibrated", "learned_additive", "cp16")}
    predictions["apex_pilot"] = apex["apex"]
    predictions["apex_molecular_surrogate"] = apex["molecular_surrogate"]
    basis = np.load(args.out / "basis.npz")
    checks = {}
    for method in ("shi_eq11", "shi_calibrated", "learned_additive", "apex_pilot"):
        pred = predictions[method]
        reconstructed = shi_predict(pred, basis["basis"], basis["refs"])
        np.testing.assert_allclose(pred, reconstructed, rtol=2e-5, atol=2e-4)
        checks[method] = float(np.max(np.abs(pred - reconstructed)))
    assert products.groupby("smiles").split.nunique().max() == 1
    used_basis = np.unique(basis["basis"][basis["basis"] >= 0])
    assert (products.iloc[used_basis].split == "train").all()
    assert all(np.isfinite(v).all() for v in predictions.values())
    dump(args.report / "integrity_checks.json", {
        "canonical_product_split_disjoint": True,
        "all_basis_products_training_only": True,
        "all_predictions_finite": True,
        "additive_reconstruction_max_abs_errors": checks,
    })
    metrics = pd.concat([pd.read_csv(args.out / "metrics.csv"), pd.read_csv(args.out / "apex_metrics.csv")])
    metrics.to_csv(args.report / "metrics.csv", index=False)
    table = metrics[metrics.arity == "all"].pivot(index="endpoint", columns="method", values="spearman")
    table.to_csv(args.report / "spearman.csv")
    per_reaction = []
    for rid, group in products.groupby("reaction_id"):
        ids = group.index[(group.split == "test") & ~group.smiles.duplicated()].values
        for method, pred in predictions.items():
            for j, name in enumerate(names):
                a, b = y[ids, j], pred[ids, j]
                rho = float(spearmanr(a, b).statistic) if np.std(a) and np.std(b) else None
                per_reaction.append(dict(reaction_id=rid, method=method, endpoint=name, n=len(ids),
                                         spearman=rho, rmse=float(np.sqrt(np.mean((a-b)**2)))))
    pd.DataFrame(per_reaction).to_csv(args.report / "per_reaction_metrics.csv", index=False)
    test = (products.split == "test").values & ~products.smiles.duplicated().values
    val = (products.split == "validation").values
    h = names.index("herg-chemeleon-baseline/pchembl_value_mean")
    d = names.index("permeability-logd-ppb-chemeleon-baseline/logD")
    lo, hi = np.quantile(y[val, d], [.25, .75])
    yt = y[test]
    feasible = (yt[:, d] >= lo) & (yt[:, d] <= hi)
    k = min(20, int(feasible.sum()))
    true_best = np.flatnonzero(feasible)[np.argsort(yt[feasible, h], kind="stable")[:k]]
    query, retrieval = [], []
    for method, pred in predictions.items():
        pt = pred[test]
        pf = (pt[:, d] >= lo) & (pt[:, d] <= hi)
        chosen = np.flatnonzero(pf)[np.argsort(pt[pf, h], kind="stable")[:k]]
        query.append(dict(method=method, selected=len(chosen),
                          teacher_feasible_precision=float(feasible[chosen].mean()) if len(chosen) else 0,
                          true_top20_recall=len(set(chosen) & set(true_best)) / k,
                          mean_teacher_herg=float(yt[chosen, h].mean()) if len(chosen) else None))
        for j, name in enumerate(names):
            n = max(1, int(np.ceil(len(yt) * .05)))
            # Continuous ADMET endpoints: deterministic ties by row order.
            if name.startswith("RDKit/"):
                continue
            for direction in (-1, 1):
                ideal = set(np.argsort(direction * yt[:, j], kind="stable")[:n])
                selected = set(np.argsort(direction * pt[:, j], kind="stable")[:n])
                retrieval.append(dict(method=method, endpoint=name,
                                      direction="highest" if direction == -1 else "lowest",
                                      k=n, recall=len(ideal & selected) / n,
                                      wasserstein=float(wasserstein_distance(yt[:, j], pt[:, j]))))
    pd.DataFrame(query).to_csv(args.report / "constrained_query.csv", index=False)
    pd.DataFrame(retrieval).to_csv(args.report / "retrieval.csv", index=False)
    dump(args.report / "query_definition.json", {
        "objective": "minimize teacher hERG pChEMBL prediction",
        "constraint": "LogD between validation-teacher 25th and 75th percentiles",
        "logd_lower": float(lo), "logd_upper": float(hi), "k": k,
        "test_candidates": len(yt), "teacher_feasible_candidates": int(feasible.sum()),
        "oracle_top20_mean_herg": float(yt[true_best, h].mean()),
        "scope": "805 held-out products; brute-force query evaluation, NOT apex_topk timing or trillion-scale retrieval",
    })
    for name in ("design.json", "descriptor_additivity.csv", "descriptor_histograms.json", "full_space_hac.json",
                 "apex_provenance.json", "apex_surrogate_training.json", "apex_factorizer_training.json", "training.json"):
        shutil.copyfile(args.out / name, args.report / name)
    teachers = {p.name: json.loads(p.read_text()) for p in args.out.glob("*_teacher.json")}
    dump(args.report / "teacher_provenance.json", teachers)
    if (args.out / "foundation_assets.json").exists():
        shutil.copyfile(args.out / "foundation_assets.json", args.report / "foundation_assets.json")
    import torch
    environment = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pandas", "scipy", "rdkit", "tensorboard", "matplotlib")}
    environment.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
    dump(args.report / "environment.json", environment)
    sources = list(Path("experiments").glob("*.py")) + [Path(n) for n in ("model.py", "chemistry.py", "teachers.py", "data/reactions.tsv", "data/synthon_space_1M.tsv")]
    dump(args.report / "source_hashes.json", {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    aliases = {
        "pchembl_value_mean": "hERG", "logD": "LogD (CheMeleon)",
        "caco2_atob_LogPapp": "Caco-2 A→B", "caco2_btoa_LogPapp": "Caco-2 B→A",
        "mppb_LogUnbound": "Mouse PPB", "hppb_LogUnbound": "Human PPB",
        "LOG_CLint_HLM": "Human clearance", "LOG_CLint_RLM": "Rat clearance",
        "LOG_CLint_MLM": "Mouse clearance", "pEC50": "PXR",
    }
    labels = []
    for name in names:
        repo, target = name.split("/")
        if "foundation" in repo:
            label = ("LogD" if repo.startswith("logd") else "MBPB") + (" (mini.)" if "minimol" in repo else " (surr.)")
        else:
            label = aliases.get(target, target.replace("OPENADMET_LOGAC50_", ""))
        labels.append(label)
    methods = ["shi_eq11", "shi_calibrated", "learned_additive", "apex_molecular_surrogate", "apex_pilot", "cp16"]
    values = table.reindex(index=names, columns=methods).values
    fig, ax = plt.subplots(figsize=(9, 10))
    im = ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(names)), labels)
    ax.set_xticks(range(6), ["Shi\nEq. 11", "Shi\ncalibrated", "Learned\nadditive", "APEX\nfull mol.*", "APEX\npilot", "CP16"])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for i in range(len(names)):
        for j in range(6):
            ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center", color="white" if values[i,j] > .7 else "black", fontsize=10)
    ax.axhline(5.5, color="black", linewidth=1.5)
    ax.set_title("Teacher agreement on held-out combinations\n805 test products · one seed · eight reaction grids", pad=42)
    fig.colorbar(im, ax=ax, fraction=.035, label="Test Spearman ρ")
    fig.text(.02, .02, "*Full-molecule APEX surrogate is a diagnostic, not a cached synthon scorer.\nAPEX: native modules, short ADMET training run. Descriptor ties: see MAE/exactness audits.", fontsize=9)
    fig.tight_layout(rect=[0, .06, 1, 1])
    fig.savefig(args.report / "comparison.png", dpi=180)
    fig.savefig(args.report / "comparison.pdf")
    plt.close(fig)
    print(table.round(3).to_string())
    print(pd.DataFrame(query).to_string(index=False))


if __name__ == "__main__":
    main()
