import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define data directories
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

# Load and concatenate all data
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

# Remove summary rows (ALL / MEAN)
df_all = df_all[~df_all["label"].isin(["ALL", "MEAN"])]

# Style settings
sns.set_context("talk", font_scale=1.1)
palette = {"Mean": "#8B5CF6", "Max": "#EF4444", "Assemble": "#3B82F6"}

# Generate one vertical plot per feature
for feature_name in base_dirs.keys():
    df_feature = df_all[df_all["feature"] == feature_name]

    plt.figure(figsize=(7, max(6, len(df_feature["label"].unique()) * 0.25)))  # auto height
    sns.boxplot(
        data=df_feature,
        y="label",             # smell types vertically
        x="cosine",            # cosine distance horizontally
        hue="group",
        palette=palette,
        linewidth=0.8,
        fliersize=2,
        width=0.6,
        dodge=0.6
    )

    plt.title(f"Cosine Distance with LightGBM ({feature_name})", fontsize=18)
    #plt.xlabel("Cosine distance", fontsize=14)
    plt.ylabel("Smell type", fontsize=14)

    # Legend at bottom-right corner
    plt.legend(
        title="Type",
        fontsize=11,
        title_fontsize=12,
        loc="lower right",
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(f"cosine_distance_by_smell_{feature_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

print("✅ All 5 vertical cosine plots saved with legend at bottom right.")

