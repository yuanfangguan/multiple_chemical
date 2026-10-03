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
                        # CHANGED: Now looking for Pearson correlation
                        if "pearson correlation" in line.lower():
                            value = float(line.split(":")[1].strip())
                            results.append({"Algorithm": label, "Correlation": value})
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return pd.DataFrame(results)

def create_large_font_plot(df):
    if df.empty:
        print("No data found! Check if 'Pearson correlation' exists in your files.")
        return

    # Set global scaling
    sns.set_context("talk", font_scale=1.2) 
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(14, 9))

    order = ["Two-drug", "Max", "Mean", "Max and Mean", "Single"]
    
    # Create Plot
    ax = sns.boxplot(
        x="Algorithm", 
        y="Correlation",  # Updated column name
        data=df, 
        order=order, 
        palette="Set2",
        linewidth=2.5,
        fliersize=0 
    )

    # Add points
    sns.stripplot(
        x="Algorithm", 
        y="Correlation", # Updated column name
        data=df, 
        order=order, 
        color=".1", 
        size=10,
        alpha=0.7,
        jitter=True
    )

    # UPDATED: Titles and Labels for Correlation
    plt.title('Mean reporting frequency', fontsize=28, pad=30)
    plt.ylabel('Pearson Correlation', fontsize=22, labelpad=15)
    plt.xlabel('Algorithm', fontsize=22, labelpad=15)
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.tight_layout()
    
    save_path = "mean_reporting_frequency_correlation.png"
    plt.savefig(save_path, dpi=300)
    print(f"Correlation plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    df = extract_data()
    create_large_font_plot(df)
