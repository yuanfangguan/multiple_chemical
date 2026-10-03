import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def extract_data():
    folders = {
        "../concate_all": "Two-drug",
        "../max_all": "Max",
        "../mean_all": "Mean",
        "../max_mean_all": "Max and Mean",
        "../single_all": "Single"
    }
    results = []
    for path, label in folders.items():
        files = glob.glob(os.path.join(path, "metrics_fold*.txt"))
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        if "cosine similarity" in line.lower():
                            value = float(line.split(":")[1].strip())
                            results.append({"Algorithm": label, "Cosine Similarity": value})
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return pd.DataFrame(results)

def create_large_font_plot(df):
    if df.empty:
        print("No data found!")
        return

    # 1. Set global scaling: 'talk' or 'poster' makes everything bigger
    sns.set_context("talk", font_scale=1.2) 
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(14, 9)) # Increased figure size for readability

    order = ["Two-drug", "Max", "Mean", "Max and Mean", "Single"]
    
    # 2. Create Plot
    ax = sns.boxplot(
        x="Algorithm", 
        y="Cosine Similarity", 
        data=df, 
        order=order, 
        palette="Set2",
        linewidth=2.5,  # Thicker box lines
        fliersize=0     # Hide outliers to show them via stripplot
    )

    # Add points with high visibility
    sns.stripplot(
        x="Algorithm", 
        y="Cosine Similarity", 
        data=df, 
        order=order, 
        color=".1", 
        size=10,        # Larger dots
        alpha=0.7,
        jitter=True
    )

    # 3. Explicitly set very large font sizes
    plt.title('Mean reporting frequency', fontsize=28,  pad=30)
    plt.ylabel('Cosine Similarity Score', fontsize=22, labelpad=15)
    plt.xlabel('Algorithm', fontsize=22, labelpad=15)
    
    # Scale the tick labels (the names of the algorithms on the axis)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.tight_layout()
    
    # Save with high DPI for clarity
    save_path = "mean_reporting_frequency_large.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved with large fonts to: {save_path}")
    plt.show()

if __name__ == "__main__":
    df = extract_data()
    create_large_font_plot(df)
