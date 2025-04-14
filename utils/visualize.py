import os
import matplotlib.pyplot as plt
import seaborn as sns

def create_plots(epochs_range, baseline_metrics, ndlinear_metrics,
                                   base_params, nd_params, save_path="results"):
    
    sns.set_style("whitegrid")
    os.makedirs(save_path, exist_ok=True)

    # Define professional color scheme
    baseline_color = '#1f77b4'  # Blue
    ndlinear_color = '#ff7f0e'  # Orange

    # Common styling elements
    title_fontsize = 16
    subtitle_fontsize = 12
    axis_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 11
    linewidth = 2.5

    # 1. TRAINING LOSS COMPARISON
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, baseline_metrics['train_loss'],
             label=f'Baseline ({base_params:,} params)',
             color=baseline_color, linewidth=linewidth)
    plt.plot(epochs_range, ndlinear_metrics['train_loss'],
             label=f'NdLinear ({nd_params:,} params)',
             color=ndlinear_color, linewidth=linewidth)

    plt.xlabel('Epoch', fontsize=axis_fontsize)
    plt.ylabel('Loss', fontsize=axis_fontsize)
    plt.title('Training Loss Comparison', fontsize=title_fontsize, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize, frameon=True, facecolor='white', edgecolor='lightgray')

    # Annotate final values
    plt.annotate(f"{baseline_metrics['train_loss'][-1]:.4f}",
                 xy=(epochs_range[-1], baseline_metrics['train_loss'][-1]),
                 xytext=(5, 5), textcoords='offset points', fontsize=10,
                 color=baseline_color, fontweight='bold')
    plt.annotate(f"{ndlinear_metrics['train_loss'][-1]:.4f}",
                 xy=(epochs_range[-1], ndlinear_metrics['train_loss'][-1]),
                 xytext=(5, -15), textcoords='offset points', fontsize=10,
                 color=ndlinear_color, fontweight='bold')

    # Add descriptive subtitle
    plt.figtext(0.5, 0.01,
                f"NdLinear achieves {(baseline_metrics['train_loss'][-1] - ndlinear_metrics['train_loss'][-1]):.4f} lower training loss with {base_params - nd_params:,} fewer parameters",
                ha="center", fontsize=subtitle_fontsize, fontstyle='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{save_path}/training_loss.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Parameter Efficiency Plot
    plt.figure(figsize=(8, 6))
    base_efficiency = baseline_metrics['test_acc'][-1] / (base_params / 1000)
    nd_efficiency = ndlinear_metrics['test_acc'][-1] / (nd_params / 1000)

    bars = plt.bar([0, 1], [base_efficiency, nd_efficiency],
                  color=[baseline_color, ndlinear_color], width=0.6)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}% per 1K params',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xticks([0, 1], ['Baseline', 'NdLinear'], fontsize=tick_fontsize)
    plt.ylabel('Accuracy (%) per 1,000 Parameters', fontsize=axis_fontsize)
    plt.title('Parameter Efficiency Comparison', fontsize=title_fontsize, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # Add efficiency gain annotation
    efficiency_gain = (nd_efficiency / base_efficiency - 1) * 100
    plt.figtext(0.5, 0.01,
                f"NdLinear is {efficiency_gain:.2f}% more parameter-efficient than Baseline",
                ha="center", fontsize=subtitle_fontsize, fontstyle='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{save_path}/parameter_efficiency.png", dpi=300, bbox_inches='tight')
    plt.show()