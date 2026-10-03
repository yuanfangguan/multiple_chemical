import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 1. Load and Merge Data
# Assuming missing_pairs and pred_fold0 are aligned row-wise
df_smiles = pd.read_csv("test_missing_pairs.csv")
df_preds = pd.read_csv("pred_fold0.csv")

# Combine relevant columns
df = pd.DataFrame({
    'drug_1': df_smiles['drug_1_smiles'],
    'drug_2': df_smiles['drug_2_smiles'],
    'score': df_preds['mean_reporting_frequency_pred']
})

# 2. Filter by Global Threshold
df_filtered = df[df['score'] > 0.5].copy()

# 3. Keep Top 10 Connections per Drug
def get_top_n(group, n=10):
    return group.sort_values('score', ascending=False).head(n)

# Combine directed-like edges to group by "any drug node"
temp_1 = df_filtered[['drug_1', 'drug_2', 'score']].rename(columns={'drug_1': 'source', 'drug_2': 'target'})
temp_2 = df_filtered[['drug_2', 'drug_1', 'score']].rename(columns={'drug_2': 'source', 'drug_1': 'target'})
combined = pd.concat([temp_1, temp_2])

# Fix Pandas warning by explicitly selecting columns and resetting index
top_edges = (
    combined.groupby('source', group_keys=False)
    .apply(lambda x: get_top_n(x, n=10))
    .reset_index(drop=True)
)

# Remove duplicates (A-B and B-A)
top_edges['edge_key'] = top_edges.apply(lambda x: tuple(sorted([x['source'], x['target']])), axis=1)
final_edges = top_edges.drop_duplicates('edge_key')

# 4. Build NetworkX Graph
G = nx.from_pandas_edgelist(final_edges, 'source', 'target', ['score'])

# FIX: networkx updated isolated_nodes to isolates
isolates = list(nx.isolates(G))
G.remove_nodes_from(isolates)

# 5. Visualization Setup
plt.figure(figsize=(16, 12))
# Using kamada_kawai_layout for a more "Nature-style" organic look
pos = nx.kamada_kawai_layout(G)

# Node styling
degrees = dict(G.degree())
node_sizes = [v * 100 for v in degrees.values()]

# Edge styling (color by prediction intensity)
edges = G.edges()
weights = [G[u][v]['score'] for u, v in edges]

# Draw
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#1f77b4', alpha=0.7)
nx.draw_networkx_edges(G, pos, width=1.2, alpha=0.4, edge_color=weights, edge_cmap=plt.cm.YlOrRd)

# 6. Labeling (Top 10 most connected drugs)
top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
labels = {node: f"Drug_{i+1}\n({node[:8]}...)" for i, (node, deg) in enumerate(top_nodes)}

nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight="bold",
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))

plt.title(f"Predicted Drug Interaction Network\n(Threshold > 0.4, Nodes: {G.number_of_nodes()})", fontsize=16)
plt.axis('off')
plt.tight_layout()

# Save for Paper
plt.savefig("ddi_network_high_risk.png", dpi=300, bbox_inches='tight')
print(f"✔ Network saved: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
plt.show()
