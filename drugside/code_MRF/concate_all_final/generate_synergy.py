import pandas as pd

# 1. Load the Single Drug Data and Predictions
# We need to map the SMILES to its predicted "Baseline" value
df_single_smiles = pd.read_csv("test_single_drugs.csv")
df_single_preds = pd.read_csv("pred_single_fold0.csv")

# Create a dictionary: {SMILES: predicted_frequency}
# Note: In test_single_drugs, drug_1_smiles == drug_2_smiles
baseline_map = dict(zip(df_single_smiles['drug_1_smiles'], 
                        df_single_preds['mean_reporting_frequency_pred']))

# 2. Load the Pair Predictions
df_pair_smiles = pd.read_csv("test_missing_pairs.csv")
df_pair_preds = pd.read_csv("pred_fold0.csv")

# Combine them into one working dataframe
df_combo = pd.DataFrame({
    'drug_1': df_pair_smiles['drug_1_smiles'],
    'drug_2': df_pair_smiles['drug_2_smiles'],
    'p_combo': df_pair_preds['mean_reporting_frequency_pred']
})

# 3. Calculate Interaction Scores
def calculate_scores(row):
    p_a = baseline_map.get(row['drug_1'], 0.0)
    p_b = baseline_map.get(row['drug_2'], 0.0)
    p_combo = row['p_combo']
    
    # Highest Single Agent (HSA) Score: Excess risk over the most toxic drug
    hsa_score = p_combo - max(p_a, p_b)
    
    # Additive Score: Excess risk over the sum (bliss-like)
    # Note: Use with caution as frequencies are probabilities/rates
    additive_score = p_combo - (p_a + p_b)
    
    return pd.Series([p_a, p_b, hsa_score, additive_score])

df_combo[['p_a', 'p_b', 'hsa_score', 'additive_score']] = df_combo.apply(calculate_scores, axis=1)

# 4. Save and Inspect
df_combo.to_csv("interaction_synergy_scores.csv", index=False)

print("Top 5 Synergistic (High Risk) Combinations:")
print(df_combo.sort_values('hsa_score', ascending=False).head())
