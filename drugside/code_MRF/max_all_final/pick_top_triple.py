import pandas as pd

# 1. Load Data
train_df = pd.read_csv("train_final.csv")
triple_df = pd.read_csv("triple_drug_predictions.csv")

# 2. Build the SMILES -> Name Mapping (as we did for the heatmap)
df_names = pd.read_csv("../../data/TWOSIDES.csv")
df_smiles = pd.read_csv("../../data/TWOSIDES_with_smiles.csv")

name_map = {}
for _, row in df_names.iterrows():
    name_map[row['drug_1_rxnorn_id']] = row['drug_1_concept_name']
    name_map[row['drug_2_rxnorm_id']] = row['drug_2_concept_name']

smiles_to_name = {}
for _, row in df_smiles.iterrows():
    id1, id2 = row['drug_1_rxnorn_id'], row['drug_2_rxnorm_id']
    s1, s2 = row['drug_1_smiles'], row['drug_2_smiles']
    if pd.notna(s1): smiles_to_name[s1] = name_map.get(id1, "Unknown")
    if pd.notna(s2): smiles_to_name[s2] = name_map.get(id2, "Unknown")

def get_name(s):
    return smiles_to_name.get(s, s[:10])

# 3. Create a lookup for original (Double) MRF values
# We sort the SMILES pair to ensure the lookup works regardless of order
train_df['pair_key'] = train_df.apply(lambda r: tuple(sorted([r.drug_1_smiles, r.drug_2_smiles])), axis=1)
doublet_lookup = dict(zip(train_df['pair_key'], train_df['mean_reporting_frequency']))

# 4. Find the best added drug for each base pair
# Group by the two base drugs and find the one with the max predicted_mrf
best_triples = triple_df.sort_values('predicted_mrf', ascending=False).drop_duplicates(['base_drug_1', 'base_drug_2'])

final_rows = []

for _, row in best_triples.iterrows():
    s1, s2, s3 = row['base_drug_1'], row['base_drug_2'], row['added_drug_3']
    new_val = row['predicted_mrf']
    
    # Get the original value
    pair_key = tuple(sorted([s1, s2]))
    old_val = doublet_lookup.get(pair_key, 0.0)
    
    # Only keep if the triple is actually HIGHER than the double
    if new_val > old_val:
        final_rows.append({
            'Drug 1': get_name(s1),
            'Drug 2': get_name(s2),
            'Added Drug 3': get_name(s3),
            'Original MRF': round(old_val, 5),
            'Triple MRF': round(new_val, 5),
            'Increase': round(new_val - old_val, 5)
        })

# 5. Output Results
results_df = pd.DataFrame(final_rows)
results_df = results_df.sort_values('Increase', ascending=False) # Sort by most synergistic

results_df.to_csv("top_triple_synergy.csv", index=False)

print("Top 5 Synergistic Triples:")
print(results_df.head())
