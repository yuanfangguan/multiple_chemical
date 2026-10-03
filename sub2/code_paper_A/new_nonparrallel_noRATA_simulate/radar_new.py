import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

# === Configuration ===
INPUT_CSV = "3d_grid_predictions.csv"
RADAR_DIR = "radar_overlays"
EXCLUDE_LABELS = ["pleasantness", "intensity"]
META_COLS = [
    "stimulus",
    "smiles1",
    "smiles2",
    "smiles3",
    "f1",
    "f2",
    "f3",
    "dil1",
    "dil2",
    "dil3",
]
NUM_OVERLAYS = 5

# === Global Styling ===
plt.rcParams.update({"font.size": 20})


# Helper function to format small floats nicely as decimals instead of scientific notation
def format_decimal(value):
    # Format to a maximum of 8 decimal places, then strip unnecessary trailing zeros
    s = f"{value:.8f}".rstrip("0")
    if s.endswith("."):
        s += "0"  # keep 0.0 instead of just 0.
    return s


def generate_radar_overlays(csv_path, output_dir):
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Identify valid smell labels
    all_smell_cols = [c for c in df.columns if c not in META_COLS]
    smell_labels = [
        c
        for c in all_smell_cols
        if c.lower() not in [x.lower() for x in EXCLUDE_LABELS]
    ]

    stimuli = df["stimulus"].unique()

    for stim_id in stimuli:
        df_stim = df[df["stimulus"] == stim_id].copy()

        # Identify top 8 descriptors for this stimulus
        top_8_labels = (
            df_stim[smell_labels]
            .max()
            .sort_values(ascending=False)
            .head(8)
            .index.tolist()
        )
        num_vars = len(top_8_labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        # --- DYNAMIC SCALING CALCULATION ---
        local_max = df_stim[top_8_labels].values.max()
        chart_limit = local_max * 1.1 if local_max > 0 else 1.0

        for factor in ["f1", "f2", "f3"]:
            others = [f for f in ["f1", "f2", "f3"] if f != factor]

            f_other1_val = df_stim[others[0]].median()
            f_other2_val = df_stim[others[1]].median()

            df_slice = df_stim[
                (df_stim[others[0]] == f_other1_val)
                & (df_stim[others[1]] == f_other2_val)
            ].sort_values(by=factor)

            if df_slice.empty:
                df_slice = df_stim.sort_values(by=factor)

            unique_vals = sorted(df_slice[factor].unique())
            indices = np.linspace(
                0,
                len(unique_vals) - 1,
                min(len(unique_vals), NUM_OVERLAYS),
                dtype=int,
            )
            selected_levels = [unique_vals[i] for i in indices]

            fig, ax = plt.subplots(figsize=(14, 10), subplot_kw=dict(polar=True))
            colors = cm.plasma(np.linspace(0, 0.8, len(selected_levels)))

            for i, val in enumerate(selected_levels):
                row = df_slice[df_slice[factor] == val].iloc[0]
                values = row[top_8_labels].values.tolist()
                values += values[:1]

                # --- CHANGED: Call format_decimal helper instead of using :.1e ---
                label_text = f"{format_decimal(val)}"

                ax.plot(
                    angles,
                    values,
                    color=colors[i],
                    linewidth=3,
                    label=label_text,
                    alpha=0.9,
                )
                ax.fill(angles, values, color=colors[i], alpha=0.1)

            # --- Axis Styling ---
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])

            ax.set_xticklabels(top_8_labels, fontsize=20)
            ax.tick_params(axis="both", which="major", labelsize=20, pad=15)

            # APPLY DYNAMIC LIMIT
            ax.set_ylim(0, chart_limit)
            ax.set_rticks(np.linspace(0, local_max, 5))

            # --- Title & Legend Fix ---
            plt.title(f"Stimulus: {stim_id}\nVarying {factor}", size=24, pad=50)

            plt.legend(
                loc="center left",
                bbox_to_anchor=(1.2, 0.5),
                title=f"Dilution ({factor})",
                title_fontsize=20,
                frameon=True,
                prop={"size": 16},  # Slightly smaller legend text if lines get long
            )

            filename = f"radar_stim_{stim_id}_vary_{factor}.png"
            plt.savefig(
                os.path.join(output_dir, filename), dpi=150, bbox_inches="tight"
            )
            plt.close()

        print(f"✅ Finished dynamic radar plots for Stimulus: {stim_id}")


if __name__ == "__main__":
    generate_radar_overlays(INPUT_CSV, RADAR_DIR)
