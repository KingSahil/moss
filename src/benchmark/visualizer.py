import logging
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

def plot_latency_graphs(
    latencies: Dict[str, List[float]], 
    output_path: str = "latency_comparison.png"
) -> None:
    """
    Plots benchmark latency comparisons (median/P99 bars and distribution box plots)
    using the Digital Ledger design system aesthetic.
    """
    logger.info("Plotting latency charts to: %s", output_path)
    
    # Configure global Matplotlib styles to match Digital Ledger aesthetics
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['text.color'] = '#071e27'
    plt.rcParams['axes.labelcolor'] = '#071e27'
    plt.rcParams['xtick.color'] = '#071e27'
    plt.rcParams['ytick.color'] = '#071e27'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#f3faff')
    ax1.set_facecolor('#ffffff')
    ax2.set_facecolor('#ffffff')
    
    # Color palette
    colors = {
        "Chroma (Ephemeral)": "#005db7",     # Aquatic Blue
        "Pinecone (Serverless)": "#00838f",   # Muted Teal
        "Moss (Local-First)": "#004f45"       # Botanical Green
    }
    
    # Filter only configured systems
    systems = [sys for sys in latencies.keys() if len(latencies[sys]) > 0]
    if not systems:
        logger.warning("No latency data available to plot.")
        plt.close()
        return

    p50_vals = [np.percentile(latencies[sys], 50) for sys in systems]
    p99_vals = [np.percentile(latencies[sys], 99) for sys in systems]
    
    x = np.arange(len(systems))
    width = 0.35
    
    # 1. Latency Bar Chart (P50 and P99)
    rects1 = ax1.bar(x - width/2, p50_vals, width, label='P50 (Median)', color='#005db7', alpha=0.85)
    rects2 = ax1.bar(x + width/2, p99_vals, width, label='P99 (99th %tile)', color='#004f45', alpha=0.9)
    
    ax1.set_title("Query Latency Comparison (Lower is Better)", fontsize=13, fontweight='bold', pad=15)
    ax1.set_ylabel("Latency (milliseconds)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems, fontweight='semibold')
    ax1.legend(frameon=False)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Hide top and right spines
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
        
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f'{height:.2f}ms',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='semibold'
            )
                        
    autolabel(rects1, ax1)
    autolabel(rects2, ax1)

    # 2. Box Plot of Query Latencies (excluding outliers for clean representation)
    data_to_plot = [latencies[sys] for sys in systems]
    box = ax2.boxplot(data_to_plot, patch_artist=True, showfliers=False)
    ax2.set_xticks(range(1, len(systems) + 1))
    ax2.set_xticklabels(systems)
    
    box_colors = [colors.get(sys, "#90a4ae") for sys in systems]
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('#071e27')
        
    for median in box['medians']:
        median.set_color('#ffffff')
        median.set_linewidth(2)
        
    ax2.set_title("Query Latency Distribution (No Outliers)", fontsize=13, fontweight='bold', pad=15)
    ax2.set_ylabel("Latency (milliseconds)")
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Latency graphs updated successfully.")
