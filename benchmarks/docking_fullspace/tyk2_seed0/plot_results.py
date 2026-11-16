"""Recreate the aggregate-only first-target figure; no private assets needed."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
data = pd.read_csv(root / "comparison.csv").set_index(["method", "domain"])
methods = ["synfactor", "ga_dcso_v14_screening", "spacehasten_fast", "spacehasten_accurate"]
labels = ["SynFactor + information", "GA-DCSO v14", "SpaceHASTEN fast", "SpaceHASTEN accurate"]
colors = ["#17877c", "#7854a1", "#c58b27", "#4e80aa"]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for panel, (ax, domain) in enumerate(zip(axes, ["all-valid", "all-valid", "druglike-v1"])):
    for i, (method, color) in enumerate(zip(methods, colors)):
        row = data.loc[method, domain]
        if panel == 0:
            low = high = row.molecular_top100_mean
        else:
            low, high = row.OPDiv_lower, max(row.OPDiv_lower, row.OPDiv_upper)
        ax.plot([low, high], [i, i], color=color, lw=3)
        ax.scatter(low, i, color=color, s=45, zorder=3)
        if high - low > 1e-6:
            ax.scatter(high, i, facecolor="white", edgecolor=color, s=45, zorder=3)
    ax.set_yticks(np.arange(len(methods)), labels)
    ax.grid(axis="x", color="#dddddd", alpha=.7)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].invert_yaxis()
axes[0].set_title("Unfiltered top-100 docking score")
axes[0].set_xlabel("Mean energy; lower is better")
axes[1].set_title("Unfiltered OPDiv100")
axes[2].set_title("Druglike OPDiv100")
for ax in axes[1:]:
    ax.set_xlabel("Mean utility; higher is better")
fig.suptitle("TYK2 · full Freedom · 10,000 charged attempts · seed 0", fontsize=13)
fig.text(.5, .015, "OPDiv lines show optimization bounds, not statistical confidence intervals. Strict Morgan Tc < 0.30.",
         ha="center", fontsize=9, color="#444444")
fig.tight_layout(rect=(0, .05, 1, .96))
for extension in ("png", "svg"):
    fig.savefig(root / f"comparison.{extension}", dpi=180, bbox_inches="tight")
svg = root / "comparison.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
