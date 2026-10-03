import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 1. Load Data
try:
    df_pair_smiles = pd.read_csv("test_missing_pairs.csv")
    df_pair_preds = pd.read_csv("pred_fold0.csv")
    df_single_smiles = pd.read_csv("test_single_drugs.csv")
    df_single_preds = pd.read_csv("pred_single_fold0.csv")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# 2. Build SMILES -> Drug Name Mapping
print("Building drug name mapping...")
df_names = pd.read_csv("../../data/TWOSIDES.csv")
df_smiles_ref = pd.read_csv("../../data/TWOSIDES_with_smiles.csv")

name_map = {}
for _, row in df_names.iterrows():
    name_map[row['drug_1_rxnorn_id']] = row['drug_1_concept_name']
    name_map[row['drug_2_rxnorm_id']] = row['drug_2_concept_name']

smiles_to_name = {}
for _, row in df_smiles_ref.iterrows():
    s1, s2 = row['drug_1_smiles'], row['drug_2_smiles']
    id1, id2 = row['drug_1_rxnorn_id'], row['drug_2_rxnorm_id']
    if pd.notna(s1) and id1 in name_map:
        smiles_to_name[s1] = name_map[id1]
    if pd.notna(s2) and id2 in name_map:
        smiles_to_name[s2] = name_map[id2]

def get_name(smiles):
    return smiles_to_name.get(smiles, smiles[:15] + "...")

# 3. Strict Additive Synergy Calculation
baseline_map = dict(zip(df_single_smiles['drug_1_smiles'],
                        df_single_preds['mean_reporting_frequency_pred']))

df = pd.DataFrame({
    'drug_1': df_pair_smiles['drug_1_smiles'],
    'drug_2': df_pair_smiles['drug_2_smiles'],
    'p_combo': df_pair_preds['mean_reporting_frequency_pred']
})

def get_strict_synergy(row):
    p_a = baseline_map.get(row['drug_1'])
    p_b = baseline_map.get(row['drug_2'])
    if p_a is None or p_b is None:
        return None
    return row['p_combo'] - (p_a + p_b)

df['score'] = df.apply(get_strict_synergy, axis=1)
df = df.dropna(subset=['score'])

# 4. Map SMILES to names
df['drug_1'] = df['drug_1'].map(get_name)
df['drug_2'] = df['drug_2'].map(get_name)

# Remove all pairs containing diethylene glycol
df = df[~((df['drug_1'].str.contains('diethylene glycol', case=False, na=False)) |
          (df['drug_2'].str.contains('diethylene glycol', case=False, na=False)))]
# 5. Filter and Prune
threshold = 0.3
df_filtered = df[df['score'] > threshold].copy()

t1 = df_filtered[['drug_1', 'drug_2', 'score']].rename(columns={'drug_1': 'source', 'drug_2': 'target'})
t2 = df_filtered[['drug_2', 'drug_1', 'score']].rename(columns={'drug_2': 'source', 'drug_1': 'target'})
combined = pd.concat([t1, t2])

final_edges = (
    combined.groupby('source', group_keys=False)
    .apply(lambda x: x.sort_values('score', ascending=False).head(50))
    .reset_index(drop=True)
)
final_edges['edge_key'] = final_edges.apply(lambda x: tuple(sorted([x['source'], x['target']])), axis=1)
final_edges = final_edges.drop_duplicates('edge_key')

# 6. Graph Building
G = nx.from_pandas_edgelist(final_edges, 'source', 'target', ['score'])
G.remove_nodes_from(list(nx.isolates(G)))

# 7. Visualization
plt.figure(figsize=(22, 16), dpi=300)
pos = nx.kamada_kawai_layout(G)

FONT_SIZE = 20

degrees = dict(G.degree())
node_sizes = [v * 100 for v in degrees.values()]
weights = [G[u][v]['score'] for u, v in G.edges()]

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#e74c3c', alpha=0.9, edgecolors='black')
nx.draw_networkx_edges(G, pos, width=2.5, alpha=0.5, edge_color=weights, edge_cmap=plt.cm.Reds)

sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=threshold, vmax=max(weights) if weights else 1))
sm._A = []
cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.02, pad=0.02)
cbar.set_label('Super-Additive Risk Score', fontsize=FONT_SIZE, fontweight='bold')
cbar.ax.tick_params(labelsize=FONT_SIZE)

# 8. Label Top 5 Hubs with Drug Names
top_5_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
labels = {node: node for node, deg in top_5_nodes}  # names already mapped

nx.draw_networkx_labels(G, pos, labels,
                        font_size=FONT_SIZE,
                        font_weight="bold",
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.5'))

plt.title(f"Strict Super-Additive DDI Risk Network (Threshold > {threshold})",
          fontsize=FONT_SIZE + 4, fontweight='bold', pad=30)

plt.axis('off')
plt.tight_layout()

plt.savefig("strict_synergy_big_font.png", bbox_inches='tight')
print(f"✔ Done! Top 5 hubs labeled with drug names.")
plt.show()
