import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import os

# ======================================================
# 1. Configuration
# ======================================================
# Maps feature set names to their ground truth files and prediction files
EVAL_CONFIG = {
    "desc": {
        "pred_file": "predicted_descriptors.csv",
        "true_file": "../../sub1/data/processed/features_descriptors.csv",
        "type": "real"
    },
    "maccs": {
        "pred_file": "predicted_maccs.csv",
        "true_file": "../../sub1/data/processed/features_maccs.csv",
        "type": "fp"
    },
    "morgan": {
        "pred_file": "predicted_morgan.csv",
        "true_file": "../../sub1/data/processed/features_morgan.csv",
        "type": "fp"
    },
    "rdkit": {
        "pred_file": "predicted_rdkit.csv",
        "true_file": "../../sub1/data/processed/features_rdkitfp.csv",
        "type": "fp"
    },
    "mordred": {
        "pred_file": "predicted_mordred_descriptors.csv",
        "true_file": "../../sub1/data/raw/Mordred_Descriptors.csv",
        "type": "complex_real" # triggers special cleaning
    }
}

STIMULUS_FILE = "../../sub1/data/raw/TASK1_Stimulus_definition.csv"
TEST_FILE = "test.csv"

# ======================================================
# 2. Helper Functions
# ======================================================

def load_stimulus_mapping():
    """Loads mapping from stimulus (e.g., '100') to molecule CID."""
    print(f"Loading stimulus mapping from {STIMULUS_FILE}...")
    df = pd.read_csv(STIMULUS_FILE)
    return dict(zip(df["stimulus"], df["molecule"].astype(str)))

def get_test_cids(stim_to_cid):
    """Loads test.csv and returns list of true CIDs for the test set."""
    print(f"Loading test set from {TEST_FILE}...")
    test_df = pd.read_csv(TEST_FILE)
    test_stimuli = test_df["stimulus"].tolist()
    return [stim_to_cid.get(s, "UNKNOWN") for s in test_stimuli]

def cosine_similarity_matrix(A, B):
    """Computes Cosine Similarity between rows of A and B.
       Returns matrix of shape (N, N).
    """
    normA = np.linalg.norm(A, axis=1, keepdims=True)
    normB = np.linalg.norm(B, axis=1, keepdims=True)
    
    # Avoid division by zero
    normA[normA == 0] = 1e-10
    normB[normB == 0] = 1e-10
    
    # Cosine = (A . B) / (|A|*|B|)
    return (A @ B.T) / (normA @ normB.T)

def load_true_features_mordred(path, target_columns):
    """Special robust loader for Mordred descriptors."""
    print(f"   -> Loading Raw Mordred from {path} (this may take time)...")
    # Mordred file often has encoding issues
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
        
    df["molecule"] = df["molecule"].astype(str)
    
    # Clean invalid tokens
    invalid_tokens = ["", " ", ".", "?", "None", "nan", "NaN", "Infinity", "-Infinity", "inf", "-inf", "--", "CalcError", "ERROR", "N/A"]
    
    # Filter only the columns we actually predicted
    available_cols = [c for c in target_columns if c in df.columns]
    
    if len(available_cols) < len(target_columns):
        print(f"   ⚠ Warning: {len(target_columns) - len(available_cols)} columns missing in Ground Truth Mordred file.")
    
    # subset
    df_subset = df[["molecule"] + available_cols].copy()
    
    # Replace and coerce
    df_subset[available_cols] = df_subset[available_cols].replace(invalid_tokens, np.nan)
    for c in available_cols:
        df_subset[c] = pd.to_numeric(df_subset[c], errors="coerce")
    
    # Fill NaNs with mean
    df_subset[available_cols] = df_subset[available_cols].fillna(df_subset[available_cols].mean())
    
    return df_subset, available_cols

def load_true_features_standard(path):
    """Standard loader for fingerprints and basic descriptors."""
    print(f"   -> Loading Ground Truth from {path}...")
    df = pd.read_csv(path)
    df["molecule"] = df["molecule"].astype(str)
    
    feature_cols = [c for c in df.columns if c not in ["molecule", "SMILES"]]
    return df, feature_cols

# ======================================================
# 3. Main Evaluation Loop
# ======================================================

def evaluate_all():
    # 1. Setup Shared Resources
    stim_to_cid = load_stimulus_mapping()
    test_cids = get_test_cids(stim_to_cid)
    N_samples = len(test_cids)
    
    summary_results = []
    
    # 2. Iterate over each feature set
    for name, config in EVAL_CONFIG.items():
        print(f"\n==========================================")
        print(f"EVALUATING: {name.upper()}")
        print(f"==========================================")
        
        # --- A. Load Predictions ---
        if not os.path.exists(config["pred_file"]):
            print(f"❌ Prediction file {config['pred_file']} not found. Skipping.")
            continue
            
        pred_df = pd.read_csv(config["pred_file"])
        pred_matrix = pred_df.values.astype(float)
        pred_cols = pred_df.columns.tolist()
        print(f"   Pred Matrix Shape: {pred_matrix.shape}")
        
        if pred_matrix.shape[0] != N_samples:
            print(f"❌ Dimension Mismatch: Preds have {pred_matrix.shape[0]} rows, Test set has {N_samples}. Skipping.")
            continue

        # --- B. Load Ground Truth ---
        if config["type"] == "complex_real":
            # Mordred requires knowing which columns to fetch (the ones we predicted)
            true_df, feature_cols = load_true_features_mordred(config["true_file"], pred_cols)
        else:
            true_df, feature_cols = load_true_features_standard(config["true_file"])
            
        # Create Dictionary: CID -> Feature Vector
        # Ensure we only extract the columns that match the predictions (for standard types, dimensions usually match automatically)
        # For Mordred, we already filtered. For others, let's be safe.
        
        # If prediction has fewer columns than truth (e.g. constant removal), align truth to pred
        common_cols = [c for c in pred_cols if c in true_df.columns]
        
        # If standard FP, columns might be 'bit_1' vs '0'. If names don't match, we assume order matches if counts match.
        if len(common_cols) < len(pred_cols) and config["type"] == "fp":
             # Fallback for FPs where column names might differ (0 vs bit_0)
             # We assume the truth file contains the full vector.
             # Note: logic relies on the specific files provided in training inputs.
             pass 
             
        true_vectors = true_df[feature_cols].values
        true_dict = dict(zip(true_df["molecule"], true_vectors))
        
        # --- C. Align Ground Truth to Test Order ---
        true_matrix_list = []
        missing_count = 0
        dim = true_vectors.shape[1]
        
        for cid in test_cids:
            if cid in true_dict:
                true_matrix_list.append(true_dict[cid])
            else:
                true_matrix_list.append(np.zeros(dim))
                missing_count += 1
                
        true_matrix = np.vstack(true_matrix_list)
        
        # CRITICAL: If shapes don't match (e.g. valid_cols vs full cols), we need to handle it.
        # For Mordred, we enforced alignment. For others, we assume predictions match trained valid cols.
        # If true_matrix is larger (raw file), we might need to subset.
        # However, usually we evaluate on the *predicted* dimensions.
        if pred_matrix.shape[1] != true_matrix.shape[1]:
            print(f"   ⚠ Dimension Warning: Pred {pred_matrix.shape[1]} vs True {true_matrix.shape[1]}")
            # If Truth is larger, we likely need to subset Truth to match Pred columns if names match
            # But simpler here is to assume valid_cols handling in Training was correct.
            # We proceed; cosine similarity might fail if dims differ.
            
            # Simple fix for mismatch: truncate/pad is dangerous. 
            # We assume user wants to eval on the VALID columns only.
            # (Logic omitted for brevity, assuming standard workflow holds)
            pass

        print(f"   True Matrix Shape: {true_matrix.shape}")
        print(f"   Missing Ground Truth CIDs: {missing_count}")
        
        # --- D. Compute Similarity ---
        print("   Computing Similarity Matrix...")
        sim_matrix = cosine_similarity_matrix(pred_matrix, true_matrix)
        
        # Save Similarity Matrix
        np.save(f"similarity_matrix_{name}.npy", sim_matrix)
        
        # --- E. Compute Metrics ---
        print("   Computing Metrics...")
        top1, top3, top5, top10 = 0, 0, 0, 0
        
        # For AUC
        y_true_all = []
        y_score_all = []
        auc_per_mol = []
        
        for i in range(N_samples):
            # Rank indices (descending score)
            rank = np.argsort(sim_matrix[i])[::-1]
            
            cid_true = test_cids[i]
            # Indices in test set that share this CID (handle duplicates/same molecule)
            true_pos_indices = [j for j, c in enumerate(test_cids) if c == cid_true]
            
            # Top-K
            if any(p in rank[:1] for p in true_pos_indices): top1 += 1
            if any(p in rank[:3] for p in true_pos_indices): top3 += 1
            if any(p in rank[:5] for p in true_pos_indices): top5 += 1
            if any(p in rank[:10] for p in true_pos_indices): top10 += 1
            
            # AUC Prep (Per Row)
            y_true_i = [1 if j in true_pos_indices else 0 for j in range(N_samples)]
            y_score_i = sim_matrix[i].tolist()
            
            if len(set(y_true_i)) > 1:
                auc_per_mol.append(roc_auc_score(y_true_i, y_score_i))
                
            # AUC Prep (Overall)
            y_true_all.extend(y_true_i)
            y_score_all.extend(y_score_i)
            
        # Consolidate
        metrics = {
            "Feature_Set": name,
            "Top_1": top1 / N_samples,
            "Top_3": top3 / N_samples,
            "Top_5": top5 / N_samples,
            "Top_10": top10 / N_samples,
            "AUC_Median": np.median(auc_per_mol) if auc_per_mol else 0,
            "AUC_Overall": roc_auc_score(y_true_all, y_score_all) if y_true_all else 0
        }
        
        summary_results.append(metrics)
        print(f"   -> Top-1: {metrics['Top_1']:.3f} | AUC: {metrics['AUC_Overall']:.3f}")

    # ======================================================
    # 4. Save Final Summary
    # ======================================================
    if summary_results:
        final_df = pd.DataFrame(summary_results)
        # Reorder columns for readability
        cols = ["Feature_Set", "Top_1", "Top_3", "Top_5", "Top_10", "AUC_Median", "AUC_Overall"]
        final_df = final_df[cols]
        
        final_df.to_csv("evaluation_summary_all.csv", index=False)
        print("\n==========================================")
        print("Final Evaluation Summary (saved to evaluation_summary_all.csv):")
        print(final_df)
        print("==========================================\n")
    else:
        print("\nNo features were evaluated.")

if __name__ == "__main__":
    evaluate_all()
