import matplotlib.pyplot as plt

def compare_avg_max(fset):
    avg_path = f"paper_figures/{fset}/shap_global_avg.png"
    max_path = f"paper_figures/{fset}/shap_global_max.png"
    img_avg = plt.imread(avg_path)
    img_max = plt.imread(max_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_avg)
    axes[0].set_title(f"{fset} - Mean Pooling")
    axes[1].imshow(img_max)
    axes[1].set_title(f"{fset} - Max Pooling")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"paper_figures/{fset}_avg_vs_max.png", dpi=300)
    plt.close()

