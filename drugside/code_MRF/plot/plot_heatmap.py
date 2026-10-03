import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Build the SMILES -> Name Mapping
print("Building drug name mapping...")
# Load the name reference
df_names = pd.read_csv("../../data/TWOSIDES.csv")
# Load the SMILES reference
df_smiles = pd.read_csv("../../data/TWOSIDES_with_smiles.csv")

# Create a mapping of RxNorm ID to Concept Name
name_map = {}
for _, row in df_names.iterrows():
    name_map[row['drug_1_rxnorn_id']] = row['drug_1_concept_name']
    name_map[row['drug_2_rxnorm_id']] = row['drug_2_concept_name']

# Create a mapping of SMILES to Concept Name using RxNorm ID as the bridge
smiles_to_name = {}
for _, row in df_smiles.iterrows():
    s1, s2 = row['drug_1_smiles'], row['drug_2_smiles']
    id1, id2 = row['drug_1_rxnorn_id'], row['drug_2_rxnorm_id']
    
    if pd.notna(s1) and id1 in name_map:
        smiles_to_name[s1] = name_map[id1]
    if pd.notna(s2) and id2 in name_map:
        smiles_to_name[s2] = name_map[id2]

def get_name(smiles):
    return smiles_to_name.get(smiles, smiles[:15] + "...") # Fallback to truncated SMILES

# 2. Load and combine interaction data
train_df = pd.read_csv("../max_all_final/train_final.csv")[['drug_1_smiles', 'drug_2_smiles', 'mean_reporting_frequency']]
train_df.columns = ['d1', 'd2', 'mrf']

novel_df = pd.read_csv("../max_all_final/novel_pair_predictions.csv")[['drug_1_smiles', 'drug_2_smiles', 'predicted_mrf']]
novel_df.columns = ['d1', 'd2', 'mrf']

full_df = pd.concat([train_df, novel_df])

# 3. Select Top 40 drugs and create matrix
all_drugs_series = pd.concat([full_df['d1'], full_df['d2']])
top_drugs = all_drugs_series.value_counts().head(20).index.tolist()

subset_df = full_df[full_df['d1'].isin(top_drugs) & full_df['d2'].isin(top_drugs)].copy()
matrix_data = subset_df.pivot_table(index='d1', columns='d2', values='mrf')
matrix_data = matrix_data.reindex(index=top_drugs, columns=top_drugs).fillna(0)

# 4. Apply the Names to the Matrix
drug_names = [get_name(s) for s in top_drugs]
matrix_data.index = drug_names
matrix_data.columns = drug_names

# 5. Plotting the Clustermap
g = sns.clustermap(
    matrix_data,
    method='ward',
    cmap="YlOrRd",
    figsize=(16, 16),
    annot=False,
    linewidths=.5,
    cbar_kws={'label': 'Mean Reporting Frequency'},
    dendrogram_ratio=(.1, .1)
)

# Rotate labels for better readability
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=20)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=20)

g.fig.suptitle('Hierarchically Clustered Drug Interaction Map (Concept Names)', fontsize=22, y=1.02)

plt.savefig("clustered_drug_names_heatmap.png", bbox_inches='tight', dpi=300)
print("✔ Plot saved as clustered_drug_names_heatmap.png")
plt.show()
