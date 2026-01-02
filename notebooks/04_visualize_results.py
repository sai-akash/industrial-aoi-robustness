import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load results
results_df = pd.read_csv("results/robustness_results.csv")
print(results_df)

# Set up the plot style
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Degradation curves
ax1 = axes[0]
colors = {"brightness": "#e74c3c", "noise": "#3498db", "blur": "#2ecc71"}

for deg_type in ["brightness", "noise", "blur"]:
    data = results_df[results_df["degradation"] == deg_type]
    ax1.plot(data["severity"], data["auroc"], 
             marker='o', linewidth=2, markersize=8,
             label=deg_type.capitalize(), color=colors[deg_type])

# Add baseline
baseline = results_df[results_df["degradation"] == "none"]["auroc"].values[0]
ax1.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.3f})')

ax1.set_xlabel("Severity Level", fontsize=12)
ax1.set_ylabel("AUROC", fontsize=12)
ax1.set_title("PatchCore Robustness Under Industrial Degradations", fontsize=14)
ax1.legend(loc='lower left')
ax1.set_ylim(0.4, 1.0)
ax1.set_xticks([1, 2, 3, 4, 5])

# Plot 2: Heatmap
ax2 = axes[1]
pivot_data = results_df[results_df["degradation"] != "none"].pivot(
    index="degradation", columns="severity", values="auroc"
)
sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="RdYlGn", 
            vmin=0.5, vmax=1.0, ax=ax2, cbar_kws={'label': 'AUROC'})
ax2.set_title("Sensitivity Heatmap", fontsize=14)
ax2.set_xlabel("Severity Level", fontsize=12)
ax2.set_ylabel("Degradation Type", fontsize=12)

plt.tight_layout()
plt.savefig("results/robustness_analysis.png", dpi=150, bbox_inches='tight')
plt.show()

df = pd.read_csv("results/robustness_results.csv")

print("Deployment Thresholds (AUROC > 0.90):")
print("=" * 40)

for deg_type in ["brightness", "noise", "blur"]:
    data = df[df["degradation"] == deg_type]
    safe_levels = data[data["auroc"] >= 0.90]["severity"].tolist()
    if safe_levels:
        max_safe = max(safe_levels)
        print(f"{deg_type.capitalize()}: Safe up to severity {max_safe}")
    else:
        print(f"{deg_type.capitalize()}: Not safe at any severity")

print("\nVisualization saved to results/robustness_analysis.png")