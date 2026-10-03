import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import glob

# Configuration
output_dir = "scientific_plots"
if not os.path.exists(output_dir): os.makedirs(output_dir)

# 1. Load Data
files = glob.glob("smells_detailed.csv.*")
if not files:
    print("Error: No data files found.")
    exit()

df = pd.concat([pd.read_csv(f) for f in files])

# 2. Define Criteria
df['is_emergent_pred'] = df['mixture_pred'] > (df['max_component_pred'] + 0.2)
df['is_emergent_gold'] = df['gold_standard'] > (df['max_component_pred'] + 0.2)
df['is_shielded_pred'] = df['mixture_pred'] < (df['mean_component_pred'] - 0.1)
df['is_shielded_gold'] = df['gold_standard'] < (df['mean_component_pred'] - 0.1)

# A more professional "Nature" style qualitative palette
# Set3 is often used in scientific papers for categorical data
color_palette = px.colors.qualitative.Set3 

def save_detailed_sankey(df_subset, category_col, title, filename):
    top_odors = df_subset['odor_type'].value_counts().nlargest(10).index.tolist()
    df_plot = df_subset[df_subset['odor_type'].isin(top_odors)].copy()
    
    # Define Nodes
    outcomes = ["VALIDATED (TRUE)", "FALSE POSITIVE"]
    nodes = top_odors + [title.upper()] + outcomes
    
    # Map colors to odors
    color_map = {odor: color_palette[i % len(color_palette)] for i, odor in enumerate(top_odors)}
    
    sources, targets, values, link_colors = [], [], [], []
    
    # --- Tier 1: Odor -> Category Node ---
    for odor in top_odors:
        count = len(df_plot[df_plot['odor_type'] == odor])
        if count > 0:
            sources.append(nodes.index(odor))
            targets.append(nodes.index(title.upper()))
            values.append(count)
            # Full odor color with 0.5 opacity
            c = color_map[odor].replace('rgb', 'rgba').replace(')', ', 0.5)')
            link_colors.append(c)
            
    # --- Tier 2: Category Node -> Outcomes (Weighted by Odor Color) ---
    # To keep the color all the way to the end, we break Tier 2 down by odor
    for odor in top_odors:
        odor_subset = df_plot[df_plot['odor_type'] == odor]
        valid_count = len(odor_subset[odor_subset[category_col]])
        invalid_count = len(odor_subset) - valid_count
        
        # Validated path for this specific odor
        if valid_count > 0:
            sources.append(nodes.index(title.upper()))
            targets.append(nodes.index("VALIDATED (TRUE)"))
            values.append(valid_count)
            link_colors.append(color_map[odor].replace('rgb', 'rgba').replace(')', ', 0.5)'))
            
        # False positive path for this specific odor
        if invalid_count > 0:
            sources.append(nodes.index(title.upper()))
            targets.append(nodes.index("FALSE POSITIVE"))
            values.append(invalid_count)
            # Slightly lower opacity for false positives to visually "dim" them
            link_colors.append(color_map[odor].replace('rgb', 'rgba').replace(')', ', 0.2)'))

    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 25, thickness = 20,
            line = dict(color = "white", width = 1),
            label = [n.upper() for n in nodes],
            # Use light grey for the central "processing" nodes, colored for odors
            color = [color_map.get(n, "#D3D3D3") for n in nodes]
        ),
        link = dict(source = sources, target = targets, value = values, color = link_colors)
    )])

    fig.update_layout(
        title_text=f"MECHANISTIC VALIDATION: {title.upper()}",
        font=dict(size=18, family="Arial", color="black"),
        width=1200, height=800,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Save Outputs
    fig.write_html(os.path.join(output_dir, f"{filename}.html"))
    try:
        # Scale=2 is usually plenty for a clean PNG without crashing the browser engine
        fig.write_image(os.path.join(output_dir, f"{filename}.png"), scale=2)
        print(f"Successfully exported {filename}.png")
    except:
        print(f"PNG failed, but {filename}.html is ready.")

# Run
save_detailed_sankey(df[df['is_emergent_pred']], 'is_emergent_gold', "Emergence", "nature_emergence")
save_detailed_sankey(df[df['is_shielded_pred']], 'is_shielded_gold', "Shielding", "nature_shielding")
