import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Directory definitions ===
base_dirs = {
    "Descriptors": "../new_descriptors",
    "MACCS": "../new_maccs",
    "Mordred": "../new_mordred",
    "Morgan": "../new_morgan",
    "RDKitFP": "../new_rdkitfp"
}

mean_dirs = {
    "Descriptors": "../new_descriptors_meanonly",
    "MACCS": "../new_maccs_meanonly",
    "Mordred": "../new_mordred_meanonly",
    "Morgan": "../new_morgan_meanonly",
    "RDKitFP": "../new_rdkitfp_meanonly"
}

max_dirs = {
    "Descriptors": "../new_descriptors_maxonly",
    "MACCS": "../new_maccs_maxonly",
    "Mordred": "../new_mordred_maxonly",
    "Morgan": "../new_morgan_maxonly",
    "RDKitFP": "../new_rdkitfp_maxonly"
}

def load_data(folder, model_type):
    dfs = []
    for i in range(5):
        path = os.path.join(folder, f"evaluation.tsv.{i}")
        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t')
            df["type"] = model_type
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# === Combine all data ===
all_data = []
for name in base_dirs:
    assemble = load_data(base_dirs[name], "Assemble")
    mean = load_data(mean_dirs[name], "Mean")
    max_ = load_data(max_dirs[name], "Max")
    for df, t in zip([mean, max_, assemble], ["Mean", "Max", "Assemble"]):
        df["feature"] = name
        df["group"] = t
    all_data.append(pd.concat([mean, max_, assemble]))

df_all = pd.concat(all_data)
df_all = df_all[~df_all["label"].isin(["ALL", "MEAN"])]

# === Plot settings ===
sns.set_context("talk", font_scale=1.1)
palette = {"Mean": "#8B5CF6", "Max": "#EF4444", "Assemble": "#3B82F6"}

# Prepare subplots — 5 panels horizontally
features = list(base_dirs.keys())
fig, axes = plt.subplots(
    nrows=1,
    ncols=len(features),
    figsize=(len(features) * 3.8, max(6, len(df_all["label"].unique()) * 0.25 + 3)),
    sharey=True
)

if len(features) == 1:
    axes = [axes]

# === Plot each feature in a separate panel ===
for ax, feature_name in zip(axes, features):
    df_feature = df_all[df_all["feature"] == feature_name]
    sns.boxplot(
        data=df_feature,
        y="label",       # smell types vertically
        x="pearson",     # Pearson correlation
        hue="group",
        palette=palette,
        linewidth=0.8,
        fliersize=2,
        width=0.6,
        dodge=0.6,
        ax=ax
    )

    ax.set_title(feature_name, fontsize=15)
    ax.set_xlabel("Pearson correlation (XGBoost)", fontsize=12)
    ax.set_ylabel("")
    ax.legend_.remove()

axes[0].set_ylabel("Smell type", fontsize=13)

# Add global legend at bottom-right
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(
    handles, labels,
    title="Type",
    loc="lower right",
    fontsize=11,
    title_fontsize=12
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("pearson_correlation_by_smell_all_features_side_by_side.png", dpi=300, bbox_inches='tight')
plt.show()

