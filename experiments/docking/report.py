"""Collect the local docking pilot into small, reviewable result artifacts.

This script only reads completed runs. It does not fit models or select settings.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd


def read(path):
    return json.loads(path.read_text())


def extract(folder, method, group, target, seed, scale="1M"):
    evaluation = folder / "evaluation/summary.json"
    if not evaluation.exists():
        evaluation = folder / "evaluation_opt/summary.json"
    if not evaluation.exists():
        evaluation = folder / "summary.json"
    e = read(evaluation)
    scores = e["scores"]
    s = read(folder / "summary.json") if evaluation.parent != folder else {}
    row = dict(group=group, method=method, scale=scale, target=target, seed=seed,
               score_kind=e["score_kind"], calls=e["archive"]["submitted_calls"],
               search_seconds=s.get("wall_seconds", s.get("run_seconds")),
               raw_top100=scores["molecular_raw_top100_mean"],
               raw_top1000=scores["molecular_raw_top1000_mean"],
               best_raw=scores["molecular_raw_top1_mean"],
               global_top1000_recall=scores.get("global_raw_top1000_recall"),
               valid_route_top100=scores.get("valid_route_raw_top100_mean"),
               valid_count=scores.get("target_valid_count"), source=str(folder))
    row["gpdiv100_tc04"] = next(p["value"] for p in e["portfolios"]
                                     if p["metric"] == "GPDiv" and p["k"] == 100 and p["max_similarity"] == .4)
    exact = folder / "evaluation_opt/summary.json"
    if exact.exists():
        p = next(p for p in read(exact)["portfolios"] if p["metric"] == "OPDiv"
                 and p["k"] == 100 and p["max_similarity"] == .4)
        row.update(opdiv100_tc04=p["value"] if p["status"] == "optimal" else None,
                   opdiv_lower_bound=p["value"], opdiv_status=p["status"],
                   opdiv_upper_bound=p["upper_bound"], opdiv_gap=p["gap"])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("benchmarks/docking_pilot"))
    args = parser.parse_args()
    out, runs = args.out, args.runs
    out.mkdir(parents=True, exist_ok=True)
    records, archives = [], []
    for target in ("kif11", "pyrd", "tgfr1"):
        for seed in (0, 1):
            for method, relative in (("SynFactor CP64 + diversity", f"docking_confirm/cp64_div05_{target}_s{seed}"),
                                     ("GA-DCSO v0", f"docking_baselines/ga_raw_{target}_s{seed}")):
                folder = runs / relative
                records.append(extract(folder, method, "matched_raw", target, seed))
                archives.append(folder)
    stronger = runs / "docking_baselines/ga12_raw_kif11_s0"
    records.append(extract(stronger, "GA-DCSO v12", "raw_stronger_control", "kif11", 0))
    archives.append(stronger)
    for target in ("kif11", "pyrd", "tgfr1"):
        for method, prefix in (("GA-DCSO v12 (historical LLE objective)", "ga12"),
                               ("SpaceHASTEN-style (historical LLE objective)", "sh")):
            folder = runs / f"docking_baselines/{prefix}_{target}_s0"
            records.append(extract(folder, method, "historical_different_objective", target, 0))
            archives.append(folder)
    fresh_sh = runs / "docking_baselines/sh_raw_kif11_s0"
    if (fresh_sh / "evaluation_opt/summary.json").exists():
        records.append(extract(fresh_sh, "SpaceHASTEN-style", "raw_adapted_control", "kif11", 0))
        archives.append(fresh_sh)
    for scale in ("10M", "100M"):
        # A separately marked corrected run takes precedence over development.
        cp = runs / f"docking_pilot/cp64_{scale}_kif11_s0"
        corrected = runs / f"docking_pilot/cp64_{scale}_kif11_s0_unique"
        if (corrected / "evaluation/summary.json").exists():
            cp = corrected
        valid_cp = not (cp / "INVALIDATED.json").exists() and (scale != "100M" or cp == corrected)
        if valid_cp and (cp / "evaluation/summary.json").exists():
            records.append(extract(cp, "SynFactor CP64, no diversity", "surrogate_scale", "kif11", 0, scale))
            archives.append(cp)
        for method, prefix in (("GA-DCSO v12", "ga12"), ("GA-DCSO v0", "ga0")):
            folder = runs / f"docking_baselines/{prefix}_{scale}_kif11_s0"
            if (folder / "summary.json").exists():
                records.append(extract(folder, method, "surrogate_scale", "kif11", 0, scale))
                archives.append(folder)
        diverse = runs / f"docking_pilot/cp64_div05_{scale}_kif11_s0"
        if (diverse / "evaluation/summary.json").exists():
            records.append(extract(diverse, "SynFactor CP64 + diversity", "surrogate_scale_followup", "kif11", 0, scale))
            archives.append(diverse)
        synthon_diverse = runs / f"docking_pilot/cp64_synthondiv05_{scale}_kif11_s0"
        if (synthon_diverse / "evaluation/summary.json").exists():
            records.append(extract(synthon_diverse, "SynFactor CP64 + synthon-OR diversity", "surrogate_scale_followup", "kif11", 0, scale))
            archives.append(synthon_diverse)
    results = pd.DataFrame(records)
    results.to_csv(out / "results.csv", index=False)
    development = []
    for folder in sorted((runs / "docking_pilot").iterdir()):
        if folder.name.endswith("_v2") or folder.name.startswith("raw_"):
            if (folder / "evaluation/summary.json").exists():
                development.append(extract(folder, folder.name, "development", "kif11", 0))
    pd.DataFrame(development).to_csv(out / "development.csv", index=False)
    for name in ("frozen_protocol.json", "opdiv_paired.csv", "opdiv_full_archive.csv", "audit.md", "frozen_run.py"):
        source = runs / "docking_confirm" / name
        if source.exists():
            if source.suffix == ".csv":
                (out / name).write_text(source.read_text())
            else:
                shutil.copyfile(source, out / name)
    for name in ("product_identity_audit.json", "molecular_order_audit.json", "sh_trace_audit.json"):
        source = runs / "docking_baselines" / name
        if source.exists():
            shutil.copyfile(source, out / name)
    # Preserve query order and run metadata without bundling the large libraries.
    provenance = []
    for folder in archives:
        dest = out / "runs" / folder.name
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "config.json", "history.json"):
            if (folder / name).exists():
                shutil.copyfile(folder / name, dest / name)
        for sub in ("evaluation", "evaluation_opt"):
            for name in ("summary.json", "portfolio_metrics.csv"):
                source = folder / sub / name
                if source.exists():
                    (dest / sub).mkdir(exist_ok=True)
                    shutil.copyfile(source, dest / sub / name)
        source = folder / "observed.csv"
        if source.exists():
            content = source.read_bytes()
            (dest / "observed.csv.gz").write_bytes(gzip.compress(content, mtime=0))
            provenance.append(dict(run=folder.name, observed_sha256=hashlib.sha256(content).hexdigest()))
    manifest = dict(query_archives=provenance, source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(Path("experiments/docking").glob("*.py"))},
                    note="Original frozen_run.py is archived with its verified frozen SHA. Later driver changes add lazy fingerprint loading and duplicate-ID handling for 100M; see audit.md and README.md.")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    preparation = {}
    for scale, cache_name in (("1M", "1M_v2"), ("10M", "10M"), ("100M", "100M_cp")):
        cache = runs / "docking_data" / cache_name
        if (cache / "manifest.json").exists():
            preparation[scale] = {k: v for k, v in read(cache / "manifest.json").items() if k != "layout"}
            if (cache / "product_aliases.json").exists():
                preparation[scale]["product_aliases_sha256"] = hashlib.sha256((cache / "product_aliases.json").read_bytes()).hexdigest()
    (out / "preparation.json").write_text(json.dumps(preparation, indent=2)+"\n")
    invalid = runs / "docking_pilot/cp64_100M_kif11_s0/INVALIDATED.json"
    if invalid.exists():
        shutil.copyfile(invalid, out / "invalidated_100M.json")
    plot(results, out)
    print(results.to_string(index=False))


def plot(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "pdf.fonttype": 42})
    frame = results[results.group == "matched_raw"]
    labels = [f"{t.upper()} · {s}" for t in ("kif11", "pyrd", "tgfr1") for s in (0, 1)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    for ax, metric, title in zip(axes[:2], ("raw_top100", "opdiv100_tc04"),
                                 ("Mean best 100 docking scores ↓", "Certified OPDiv100, Tc ≤ 0.4 ↑")):
        for i, (target, seed) in enumerate((t, s) for t in ("kif11", "pyrd", "tgfr1") for s in (0, 1)):
            pair = frame[(frame.target == target) & (frame.seed == seed)]
            ga = pair[pair.method == "GA-DCSO v0"][metric].iloc[0]
            cp = pair[pair.method == "SynFactor CP64 + diversity"][metric].iloc[0]
            ax.plot([ga, cp], [i, i], color="#b6b6b6", lw=2, zorder=1)
            ax.scatter(ga, i, marker="s", color="#c87934", s=45, label="GA-DCSO v0" if i == 0 else None)
            ax.scatter(cp, i, color="#187f87", s=45, label="SynFactor CP64" if i == 0 else None)
        ax.set_yticks(range(6), labels)
        ax.invert_yaxis(); ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
        ax.grid(axis="x", alpha=.15)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, -.31), frameon=False)
    scales = results[(results.group == "surrogate_scale") & results.method.str.startswith("SynFactor")]
    timing = [("1M: measured table", float(frame[frame.method.str.startswith("SynFactor")].search_seconds.median()))]
    timing += [(f"{r.scale}: surrogate table", r.search_seconds) for r in scales.itertuples()]
    axes[2].barh(np.arange(len(timing)), [v for _, v in timing], color="#187f87", height=.55)
    axes[2].set_yticks(np.arange(len(timing)), [l for l, _ in timing], fontsize=9)
    axes[2].invert_yaxis(); axes[2].set_xlabel("Search wall time (seconds)")
    for i, (_, value) in enumerate(timing):
        axes[2].text(value+.5, i, f"{value:.1f} s", va="center", fontsize=9)
    axes[2].set_xlim(0, max(v for _, v in timing)*1.3)
    axes[2].set_title("10,000 queries on one RTX 4090", loc="left", fontweight="bold", fontsize=11)
    fig.suptitle("Freedom pilot: repeated neural factor fitting + whole-space scoring", fontsize=14, y=1.01)
    fig.text(.01, -.01, "1M: 3 targets × 2 seeds; same 1,000-query start. Search excludes preparation and post-hoc portfolio evaluation.\n"
             "KIF11 seed 0 was used for method development. Large-scale runs use frozen surrogate scores, not new docking.", fontsize=9)
    fig.tight_layout(w_pad=2)
    for extension in ("png", "pdf"):
        fig.savefig(out / f"comparison.{extension}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
