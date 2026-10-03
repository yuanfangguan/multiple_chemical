import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base_dir = "shap_outputs.0"  # adjust as needed
feature_sets = ["maccs", "morgan", "rdkitfp", "descriptors", "mordred"]
variants = ["avg", "max"]

summary_records = []

for fset in feature_sets:
    for variant in variants:
        shap_path = os.path.join(base_dir, fset, variant)
        if not os.path.exists(shap_path):
            continue

        shap_files = [f for f in os.listdir(shap_path) if f.startswith("shap_values") and f.endswith(".csv")]
        all_shap_vals = []

        for file in shap_files:
            df = pd.read_csv(os.path.join(shap_path, file))
            df = df.drop(columns=["sample_index"], errors="ignore")
            mean_abs = df.abs().mean()
            all_shap_vals.append(mean_abs)

        if all_shap_vals:
            global_importance = pd.concat(all_shap_vals, axis=1).mean(axis=1).sort_values(ascending=False)
            top_features = global_importance.head(20)
            summary_records.append({
                "feature_set": fset,
                "variant": variant,
                "top_features": top_features
            })

            # Plot top-20 features
            plt.figure(figsize=(6, 4))
            top_features[::-1].plot.barh()
            plt.title(f"{fset} ({variant}) — Top 20 Mean |SHAP| Features")
            plt.xlabel("Mean |SHAP| value")
            plt.tight_layout()
            outdir = os.path.join("paper_figures", fset)
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(os.path.join(outdir, f"shap_global_{variant}.png"), dpi=300)
            plt.close()

            print(f"✅ Saved global SHAP summary for {fset} ({variant})")

