import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import os

# === Configuration ===
INPUT_CSV = "3d_grid_predictions.csv"
FIGURE_DIR = "figures"
META_COLS = ['stimulus', 'smiles1', 'smiles2', 'smiles3', 'f1', 'f2', 'f3', 'dil1', 'dil2', 'dil3']

def generate_3d_plots(csv_path, output_dir):
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    smell_labels = [c for c in df.columns if c not in META_COLS]
    stimuli = df['stimulus'].unique()

    for stim_id in stimuli:
        df_stim = df[df['stimulus'] == stim_id].copy()
        
        chem1 = df_stim['smiles1'].iloc[0]
        chem2 = df_stim['smiles2'].iloc[0]
        chem3 = df_stim['smiles3'].iloc[0]

        for label in smell_labels:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            x = np.log2(df_stim['f1'])
            y = np.log2(df_stim['f2'])
            z = np.log2(df_stim['f3'])
            
            # The 'Intensity' value for the specific smell
            intensity = df_stim[label]

            # Logic: Color comes from intensity values (using the 'magma' map)
            # Logic: Size comes from intensity values (scaled for visibility)
            sizes = (intensity * 200) + 10 

            scatter = ax.scatter(x, y, z, 
                                 c=intensity, 
                                 s=sizes, 
                                 cmap='magma', 
                                 alpha=0.6, 
                                 edgecolors='w', 
                                 linewidth=0.5)

            # Axis Labeling
            ax.set_xlabel(f"Log2 Conc: {chem1}", fontsize=8)
            ax.set_ylabel(f"Log2 Conc: {chem2}", fontsize=8)
            ax.set_zlabel(f"Log2 Conc: {chem3}", fontsize=8)
            
            plt.title(f"Stimulus: {stim_id}\nSmell Profile: {label}", fontsize=12, pad=25)
            
            # 1. Colorbar for Intensity
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=20, pad=0.1)
            cbar.set_label(f'{label} Intensity (Color)', rotation=270, labelpad=15)

            # 2. Custom Size Legend
            # We create "proxy artists" to show what different sizes mean
            max_int = intensity.max()
            mid_int = max_int / 2
            min_int = intensity.min() if intensity.min() > 0 else 0.1
            
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label=f'High ({max_int:.1f})',
                       markerfacecolor='gray', markersize=np.sqrt((max_int * 200) + 10)),
                Line2D([0], [0], marker='o', color='w', label=f'Med ({mid_int:.1f})',
                       markerfacecolor='gray', markersize=np.sqrt((mid_int * 200) + 10)),
                Line2D([0], [0], marker='o', color='w', label='Low',
                       markerfacecolor='gray', markersize=np.sqrt(10))
            ]
            
            ax.legend(handles=legend_elements, title="Intensity (Size)", 
                      loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

            # Save
            clean_label = "".join([c if c.isalnum() else "_" for c in label])
            filename = f"stim_{stim_id}_{clean_label}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
            plt.close(fig) 
            
        print(f"✅ Finished plots for Stimulus: {stim_id}")

if __name__ == "__main__":
    generate_3d_plots(INPUT_CSV, FIGURE_DIR)
