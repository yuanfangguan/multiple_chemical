import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── 1. Collect data ──────────────────────────────────────────────────────────
# Map alpha value → directory
dirs = {
    0.1: "new_nonparrallel_noRATA_alpha0.1",
    0.5: "new_nonparrallel_noRATA_alpha0.5",
    1.0: "new_nonparrallel_noRATA",          # alpha=1
    5.0: "new_nonparrallel_noRATA_alpha5",
    10.0: "new_nonparrallel_noRATA_alpha10",
}

records = []
for alpha, directory in dirs.items():
    files = sorted(glob.glob(os.path.join(directory, "evaluation.tsv.*")))
    if not files:
        print(f"WARNING: no files found in {directory}")
        continue

    pearson_vals, cosine_vals = [], []
    for f in files:
        df = pd.read_csv(f, sep="\t")
        # Normalise column names (strip whitespace)
        df.columns = df.columns.str.strip()
        # Keep only the MEAN row
        mean_row = df[df.iloc[:, 0].str.strip().str.upper() == "MEAN"]
        if mean_row.empty:
            print(f"WARNING: no MEAN row in {f}")
            continue
        pearson_vals.append(float(mean_row["pearson"].values[0]))
        cosine_vals.append(float(mean_row["cosine"].values[0]))

    if pearson_vals:
        records.append({
            "alpha": alpha,
            "pearson_mean": np.mean(pearson_vals),
            "pearson_std":  np.std(pearson_vals),
            "cosine_mean":  np.mean(cosine_vals),
            "cosine_std":   np.std(cosine_vals),
        })

df_plot = pd.DataFrame(records).sort_values("alpha").reset_index(drop=True)
print(df_plot.to_string(index=False))

# ── 2. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

x = df_plot["alpha"]
x_log = np.log10(x)          # log-spaced x positions for even visual spacing
x_labels = [str(a) for a in x]

colors = {"pearson": "#2563EB", "cosine": "#DC2626"}

for ax, metric, label, ylim in zip(
    axes,
    ["pearson", "cosine"],
    ["Pearson Correlation (↑)", "Cosine Distance (↓)"],
    [(0.55, 0.65), (0.15, 0.25)],
):
    means = df_plot[f"{metric}_mean"]
    stds  = df_plot[f"{metric}_std"]
    color = colors[metric]

    ax.errorbar(
        x_log, means, yerr=stds,
        fmt="o-", color=color, linewidth=2, markersize=7,
        capsize=5, elinewidth=1.4, ecolor=color, alpha=0.85,
    )
    ax.fill_between(x_log, means - stds, means + stds, alpha=0.12, color=color)

    # Mark alpha=1 with a dashed vertical line
    ax.axvline(np.log10(1.0), color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(np.log10(1.0) + 0.03, ylim[0] + 0.02, "α=1", color="gray", fontsize=9)

    ax.set_xticks(x_log)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Alpha (α)", fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_ylim(ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("MEAN Performance across Alpha Values", fontsize=13, y=1.01)
plt.tight_layout()

out_path = "alpha_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out_path}")
