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
feature_file = "../../data/features/features_rdkitfp.csv"

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
# Convert to single-molecule training format
# ========================
def expand_to_single(df, feat_df):
    """Duplicate each pair into two single-molecule rows (row/col)."""
    df_expanded = pd.DataFrame({
        "cid": np.concatenate([df["cid_row"].values, df["cid_col"].values]),
        "cell_line_name": np.concatenate([df["cell_line_name"].values, df["cell_line_name"].values]),
        "study_name": np.concatenate([df["study_name"].values, df["study_name"].values])
    })
    # Expand targets as well
    for target in ["css_ri", "synergy_zip", "synergy_loewe", "synergy_hsa", "synergy_bliss"]:
        if target in df.columns:
            df_expanded[target] = np.concatenate([df[target].values, df[target].values])
    # Merge molecule features
    df_expanded = df_expanded.merge(feat_df, left_on="cid", right_on="molecule", how="left")
    return df_expanded

train_single = expand_to_single(train_df, feat_df)
test_single = expand_to_single(test_df, feat_df)
print(f"✅ Single-molecule train shape: {train_single.shape}, test shape: {test_single.shape}")

# ========================
# Prepare feature matrix
# ========================
exclude_cols = {"cid", "molecule", "study_name"}
num_cols = [c for c in train_single.columns if c not in exclude_cols and train_single[c].dtype != "object"]
X_train_base = train_single[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
X_test_base = test_single[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

# One-hot encode cell line
print("🧬 Adding cell line one-hot encoding...")
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe.fit(train_single[["cell_line_name"]])

train_ohe = pd.DataFrame(ohe.transform(train_single[["cell_line_name"]]),
                         columns=[f"cell_{c}" for c in ohe.categories_[0]])
test_ohe = pd.DataFrame(ohe.transform(test_single[["cell_line_name"]]),
                        columns=[f"cell_{c}" for c in ohe.categories_[0]])

X_train = pd.concat([X_train_base.reset_index(drop=True), train_ohe.reset_index(drop=True)], axis=1)
X_test = pd.concat([X_test_base.reset_index(drop=True), test_ohe.reset_index(drop=True)], axis=1)
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
    if target not in train_single.columns:
        print(f"⚠️ Target '{target}' not found — skipping.")
        continue

    print(f"\n🚀 Training model for single-molecule {target} ...")

    train_single[target] = pd.to_numeric(train_single[target], errors="coerce")
    test_single[target] = pd.to_numeric(test_single[target], errors="coerce")

    mask_train = train_single[target].notna()
    y_train = train_single.loc[mask_train, target].values
    X_train_sub = X_train.loc[mask_train].reset_index(drop=True)

    if len(y_train) == 0:
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

    model.save_model(f"results/model_{target}.cbm")
    joblib.dump(model, f"results/model_{target}.pkl")

    preds_train = model.predict(X_train_sub)
    print(f"✅ {target} — Train r: {pearsonr(y_train, preds_train)[0]:.4f}")
    summary_records.append({"target": target, "train_r": pearsonr(y_train, preds_train)[0]})

# ========================
# Robust pair-level averaging
# ========================
print("\n🔄 Generating pair-level predictions by averaging two molecules...")
pair_results = []

for target in targets:
    model_path = f"results/model_{target}.pkl"
    if not os.path.exists(model_path):
        continue
    model = joblib.load(model_path)

    # Predict for all single-molecule entries in test set
    preds_single = model.predict(X_test)
    test_single_pred = test_single.copy()
    test_single_pred["pred"] = preds_single

    # Merge predictions for row and col drugs
    df_test = test_df.copy()
    df_test = df_test.merge(
        test_single_pred[["cid", "cell_line_name", "pred"]],
        left_on=["cid_row", "cell_line_name"],
        right_on=["cid", "cell_line_name"],
        how="left"
    ).rename(columns={"pred": "cid_row_pred"}).drop(columns=["cid"])

    df_test = df_test.merge(
        test_single_pred[["cid", "cell_line_name", "pred"]],
        left_on=["cid_col", "cell_line_name"],
        right_on=["cid", "cell_line_name"],
        how="left"
    ).rename(columns={"pred": "cid_col_pred"}).drop(columns=["cid"])

    df_test["pred_avg"] = df_test[["cid_row_pred", "cid_col_pred"]].mean(axis=1)

    # Evaluate if ground truth exists
    if target in df_test.columns:
        df_test[target] = pd.to_numeric(df_test[target], errors="coerce")
        valid = df_test[target].notna()
        if valid.sum() > 3:
            r, _ = pearsonr(df_test.loc[valid, target], df_test.loc[valid, "pred_avg"])
            cos = cosine_distances(df_test.loc[valid, target].values.reshape(1, -1),
                                   df_test.loc[valid, "pred_avg"].values.reshape(1, -1))[0][0]
            print(f"✅ Pair-level {target} — Test r: {r:.4f}, cosine: {cos:.4f}")
            pair_results.append({"target": target, "test_r_avg": r, "cosine": cos})

pair_summary = pd.DataFrame(pair_results)
pair_summary.to_csv("results/evaluation_pairlevel_summary.csv", index=False)
print("\n💾 All models trained. Pair-level summary:")
print(pair_summary)

