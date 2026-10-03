import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Configuration & Data Loading ---
pred_prefix = "predictions.csv."
test_prefix = "test.csv."
folds = range(5)
output_dir = "./"
if not os.path.exists(output_dir): os.makedirs(output_dir)

# Set global font scale for Seaborn
sns.set(font_scale=1.5) # This scales up all text elements globally
sns.set_style("whitegrid")

all_results = []

print("Loading data and calculating correlations per fold...")
for i in folds:
    pred_file = f"{pred_prefix}{i}"
    test_file = f"{test_prefix}{i}"
    
    if os.path.exists(pred_file) and os.path.exists(test_file):
        df_p = pd.read_csv(pred_file).set_index('stimulus')
        df_t = pd.read_csv(test_file).set_index('stimulus')
        df_t = df_t.reindex(df_p.index)
        
        for col in df_p.columns:
            if df_t[col].std() > 0:
                corr = df_t[col].corr(df_p[col])
                all_results.append({'Fold': i, 'Descriptor': col, 'Correlation': corr})

df_results = pd.DataFrame(all_results)

# --- 2. Hedonic Groups ---
groups = {
    'Pleasant': [
        'Berry', 'BrownSpice', 'Buttery', 'Caramellic', 'Citrus', 'Coconut', 
        'Cooling', 'Cucumber', 'Dairy', 'Floral', 'Fruity', 'Green', 'Herbal', 
        'Nutty', 'Peach', 'Pine', 'Powdery', 'Roasted', 'Sweet', 'Tropical', 
        'Vanilla', 'Woody'
    ],
    'Unpleasant': [
        'Ammonia', 'Animal', 'Burnt', 'Cheesy', 'Fecal', 'Fishy', 'Garlic.Onion', 
        'Medicinal', 'Metallic', 'Musty', 'Phenolic', 'Plastic', 'Rotten.Decay', 
        'Rubber', 'Sharp', 'Smoky', 'Sour', 'Sulfurous'
    ],
    'Neutral': [
        'Alcoholic', 'Chlorine', 'Earthy', 'Fatty', 'Fermented', 'Grainy', 
        'Mushroom', 'Ozone', 'Waxy'
    ]
}

# --- 3. Plotting with Enlarged Fonts ---
# Increased figsize for better spacing
fig, axes = plt.subplots(3, 1, figsize=(18, 24), sharey=True)

colors = {'Pleasant': 'Greens', 'Neutral': 'Blues', 'Unpleasant': 'Reds'}

for ax, (group_name, desc_list) in zip(axes, groups.items()):
    subset = df_results[df_results['Descriptor'].isin(desc_list)]
    
    # Sort by median correlation
    if not subset.empty:
        order = subset.groupby('Descriptor')['Correlation'].median().sort_values(ascending=False).index
        
        sns.boxplot(
            data=subset, x='Descriptor', y='Correlation', 
            ax=ax, palette=colors[group_name], order=order, showfliers=False,
            linewidth=2.5 # Thicker box lines
        )
        
        sns.stripplot(
            data=subset, x='Descriptor', y='Correlation', 
            ax=ax, order=order, color=".2", size=6, alpha=0.6
        )
        
        # Text Customization
        ax.set_title(f'{group_name}', fontsize=30, fontweight='bold', pad=20)
        ax.set_ylabel('Correlation ($r$)', fontsize=30, labelpad=15)
        ax.set_xlabel('', fontsize=30)
        
        # X-axis tick labels (the smell names)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=30)
        
        # Y-axis tick labels (the numbers)
        ax.tick_params(axis='y', labelsize=30)
        
        ax.set_ylim(0, 1.0)

# Adjust layout to prevent overlap of large text
plt.tight_layout()
plt.subplots_adjust(hspace=0.4) # Add space between the three subgraphs

# Save with high resolution
output_path = os.path.join(output_dir, "smell_analysis.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Done! Plot saved with enlarged fonts to: {output_path}")
