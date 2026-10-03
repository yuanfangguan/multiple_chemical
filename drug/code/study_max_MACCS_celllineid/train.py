import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_distances
import joblib
import os
from sklearn.preprocessing import OneHotEncoder

# ========================
# Paths
# ========================
train_file = "train.csv"
test_file = "test.csv"
drug_file = "../../data/drug.csv"
feature_file = "../../data/features/features_maccs.csv"

# ========================
# Load data
# ========================
train_df = pd.read_csv(train_file, low_memory=False)
test_df = pd.read_csv(test_file, low_memory=False)
drug_df = pd.read_csv(drug_file, dtype=str)
feat_df = pd.read_csv(feature_file, low_memory=False)

# Ensure IDs are strings
feat_df["molecule"] = feat_df["molecule"].astype(str)
drug_df["cid"] = drug_df["cid"].astype(str)
drug_df["dname"] = drug_df["dname"].astype(str)

# ========================
# Map drug names to CID
# ========================
drug_map = dict(zip(drug_df["dname"], drug_df["cid"]))
for df in [train_df, test_df]:
    df["cid_row"] = df["drug_row"].map(drug_map)
    df["cid_col"] = df["drug_col"].map(drug_map)

train_df = train_df.dropna(subset=["cid_row", "cid_col"])
test_df = test_df.dropna(subset=["cid_row", "cid_col"])
print(f"✅ Train after filtering: {len(train_df)}, Test: {len(test_df)}")

# ========================
# Merge features (use MAX fusion instead of concatenation)
# ========================
def merge_features(df):
    # Merge to bring both feature sets
    df1 = df.merge(feat_df, left_on="cid_row", right_on="molecule", how="left", suffixes=("", "_row"))
    df2 = df1.merge(feat_df, left_on="cid_col", right_on="molecule", how="left", suffixes=("_row", "_col"))

    # Select numeric feature columns from both molecules
    exclude_cols = {"drug_row", "drug_col", "cid_row", "cid_col", "block_id",
                    "molecule", "molecule_row", "molecule_col"}
    all_cols = [c for c in df2.columns if c not in exclude_cols]
    num_cols_row = [c for c in all_cols if c.endswith("_row")]
    num_cols_col = [c for c in all_cols if c.endswith("_col")]

    # Ensure matching features
    base_names = [c.replace("_row", "") for c in num_cols_row if c.replace("_row", "_col") in num_cols_col]

    # Compute elementwise max for each feature
    X = pd.DataFrame({
        name: np.maximum(pd.to_numeric(df2[f"{name}_row"], errors="coerce").fillna(0),
                         pd.to_numeric(df2[f"{name}_col"], errors="coerce").fillna(0))
        for name in base_names
    })

    X.columns = [f"f{i}" for i in range(X.shape[1])]
    return df2, X

train_full, X_train = merge_features(train_df)
test_full, X_test = merge_features(test_df)

# ========================
# Add cell line one-hot encoding
# ========================
print("🧬 Adding cell line one-hot encoding...")
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe.fit(train_full[["cell_line_name"]])

train_ohe = pd.DataFrame(ohe.transform(train_full[["cell_line_name"]]),
                         columns=[f"cell_{c}" for c in ohe.categories_[0]])
test_ohe = pd.DataFrame(ohe.transform(test_full[["cell_line_name"]]),
                        columns=[f"cell_{c}" for c in ohe.categories_[0]])

X_train = pd.concat([X_train, train_ohe.reset_index(drop=True)], axis=1)
X_test = pd.concat([X_test, test_ohe.reset_index(drop=True)], axis=1)
X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)

print(f"✅ After alignment: X_train {X_train.shape}, X_test {X_test.shape}")

# ========================
# Output setup
# ========================
targets = ["css_ri", "synergy_zip", "synergy_loewe", "synergy_hsa", "synergy_bliss"]
os.makedirs("results", exist_ok=True)
summary_records = []

# ========================
# Train models per target
# ========================
for target in targets:
    if target not in train_full.columns:
        print(f"⚠️ Target '{target}' not found — skipping.")
        continue

    print(f"\n🚀 Training model for {target} ...")

    train_full[target] = pd.to_numeric(train_full[target], errors="coerce")
    test_full[target] = pd.to_numeric(test_full[target], errors="coerce")

    train_mask = train_full[target].notna()
    test_mask = test_full[target].notna()
    y_train = train_full.loc[train_mask, target].values
    y_test = test_full.loc[test_mask, target].values
    X_train_sub = X_train.loc[train_mask].reset_index(drop=True)
    X_test_sub = X_test.loc[test_mask].reset_index(drop=True)

    if len(y_train) == 0 or len(y_test) == 0:
        print(f"⚠️ No valid data for {target}, skipping.")
        continue

    model = CatBoostRegressor(
        iterations=1000,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=200
    )
    model.fit(X_train_sub, y_train)

    preds_train = model.predict(X_train_sub)
    preds_test = model.predict(X_test_sub)

    # ========================
    # Dataset-level evaluation
    # ========================
    print(f"📊 Evaluating {target} per study_name ...")
    study_metrics = []
    for study, sub in test_full.loc[test_mask].groupby("study_name"):
        y_true_study = sub[target].values
        idx = sub.index
        y_pred_study = preds_test[np.isin(test_full.loc[test_mask].index, idx)]
        if len(y_true_study) > 2:
            r, _ = pearsonr(y_true_study, y_pred_study)
            cos = cosine_distances(y_true_study.reshape(1, -1), y_pred_study.reshape(1, -1))[0][0]
            study_metrics.append({"study_name": study, "pearson_r": r, "cosine_distance": cos})

    if study_metrics:
        df_study = pd.DataFrame(study_metrics)
        mean_r = df_study["pearson_r"].mean()
        mean_cos = df_study["cosine_distance"].mean()
        df_study.to_csv(f"results/evaluation_{target}_by_study.csv", index=False)
    else:
        mean_r, mean_cos = np.nan, np.nan

    print(f"✅ {target} — Train r: {pearsonr(y_train, preds_train)[0]:.4f}, "
          f"Mean Test r across studies: {mean_r:.4f}")
    print(f"✅ {target} — Mean Test cosine distance: {mean_cos:.4f}")

    model.save_model(f"results/model_{target}.cbm")
    joblib.dump(model, f"results/model_{target}.pkl")
    summary_records.append({
        "target": target,
        "train_r": pearsonr(y_train, preds_train)[0],
        "mean_test_r_by_study": mean_r,
        "mean_test_cos_by_study": mean_cos
    })

# ========================
# Combined summary
# ========================
summary_df = pd.DataFrame(summary_records)
summary_df.to_csv("results/evaluation_summary_by_study.csv", index=False)
print("\n💾 All models trained and saved.")
print(summary_df)

