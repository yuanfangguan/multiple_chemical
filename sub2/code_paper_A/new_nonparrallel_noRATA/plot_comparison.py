import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import r2_score

# Configuration
pred_prefix = "predictions.csv."
test_prefix = "test.csv."
folds = range(5)
output_dir = "smell_plots"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load and combine all folds
all_preds = []
all_tests = []

print("Loading data...")
for i in folds:
    pred_file = f"{pred_prefix}{i}"
    test_file = f"{test_prefix}{i}"
    
    if os.path.exists(pred_file) and os.path.exists(test_file):
        df_p = pd.read_csv(pred_file).set_index('stimulus')
        df_t = pd.read_csv(test_file).set_index('stimulus')
        
        # Ensure indices match
        df_t = df_t.reindex(df_p.index)
        
        all_preds.append(df_p)
        all_tests.append(df_t)
    else:
        print(f"Warning: Missing files for fold {i}")

# Combine into single DataFrames
preds_combined = pd.concat(all_preds)
tests_combined = pd.concat(all_tests)

# Get list of descriptors (columns)
descriptors = preds_combined.columns

print(f"Generating plots for {len(descriptors)} descriptors...")

for col in descriptors:
    plt.figure(figsize=(6, 6))
    
    y_true = tests_combined[col]
    y_pred = preds_combined[col]
    
    # Scatter plot
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, edgecolor=None)
    
    # Calculate limits for diagonal line
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], color='red', linestyle='--', lw=1, label='y=x')
    
    # Correlation calculation
    corr = y_true.corr(y_pred)
    
    plt.title(f"{col} (Corr: {corr:.3f})")
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # Save plot
    clean_name = col.replace('.', '_')
    plt.savefig(os.path.join(output_dir, f"{clean_name}_plot.png"), bbox_inches='tight')
    plt.close()

print(f"Done! Plots saved in '{output_dir}/' directory.")
