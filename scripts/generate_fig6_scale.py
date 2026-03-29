"""
Figure 6: Cross-model comparison — GPT-2 vs Pythia-2.8B
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Load GPT-2 results
with open("results/03_entanglement/entanglement_results.json") as f:
    gpt2 = json.load(f)
gpt2_baseline_bias = gpt2["baselines"]["bias"]
gpt2_heads = gpt2["ablation_results"]

# Load Pythia results
with open("results/14_pythia_scale/pythia_scale_results.json") as f:
    pythia = json.load(f)
pythia_heads = pythia["head_results"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: Pareto comparison
ax = axes[0]

# GPT-2 heads
for key, d in gpt2_heads.items():
    bias_red = d["bias_reduction"] / gpt2_baseline_bias * 100
    ppl_change = d.get("ppl_increase_pct", 0)
    color = 'red' if key == 'L10H9' else 'purple' if key == 'L0H8' else 'lightblue'
    size = 80 if key in ['L10H9', 'L0H8'] else 15
    alpha = 0.9 if key in ['L10H9', 'L0H8'] else 0.3
    ax.scatter(bias_red, ppl_change, c=color, s=size, alpha=alpha, marker='o', zorder=5 if size > 15 else 1)

# Pythia heads
for key, d in pythia_heads.items():
    bias_red = d["bias_reduction_pct"]
    ppl_change = d["ppl_change_pct"]
    color = 'darkred' if key == 'L22H30' else 'darkviolet' if key == 'L0H1' else 'lightyellow'
    size = 80 if key in ['L22H30', 'L0H1'] else 15
    alpha = 0.9 if key in ['L22H30', 'L0H1'] else 0.15
    ax.scatter(bias_red, ppl_change, c=color, s=size, alpha=alpha, marker='s', zorder=5 if size > 15 else 1)

# Labels
for key, label, dx, dy in [('L10H9', 'L10H9\n(GPT-2)', 3, 3),
                              ('L0H8', 'L0H8\n(GPT-2)', 3, -5)]:
    if key in gpt2_heads:
        d = gpt2_heads[key]
        bx = d["bias_reduction"] / gpt2_baseline_bias * 100
        by = d.get("ppl_increase_pct", 0)
        ax.annotate(label, (bx, by), fontsize=8, fontweight='bold',
                   xytext=(dx, dy), textcoords='offset points')

for key, label, dx, dy in [('L22H30', 'L22H30\n(Pythia)', -35, 3),
                              ('L0H1', 'L0H1\n(Pythia)', 3, -5)]:
    if key in pythia_heads:
        d = pythia_heads[key]
        ax.annotate(label, (d["bias_reduction_pct"], d["ppl_change_pct"]),
                   fontsize=8, fontweight='bold',
                   xytext=(dx, dy), textcoords='offset points')

ax.axhline(y=5, color='green', linestyle=':', alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel("Bias Reduction (%)")
ax.set_ylabel("PPL Change (%)")
ax.set_title("Separability Spectrum:\nGPT-2 (circles) vs Pythia-2.8B (squares)")
ax.set_ylim(-5, 100)

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='GPT-2 separable'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='GPT-2 entangled'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='darkred', markersize=8, label='Pythia separable'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='darkviolet', markersize=8, label='Pythia entangled'),
]
ax.legend(handles=legend_elements, fontsize=8, loc='upper right')

# Plot 2: Layer-level bias reduction comparison
ax = axes[1]

# GPT-2: average bias reduction by layer
gpt2_by_layer = {}
for key, d in gpt2_heads.items():
    layer = d["layer"]
    bias_red = d["bias_reduction"] / gpt2_baseline_bias * 100
    if layer not in gpt2_by_layer:
        gpt2_by_layer[layer] = []
    gpt2_by_layer[layer].append(bias_red)

gpt2_layers = sorted(gpt2_by_layer.keys())
gpt2_max_by_layer = [max(gpt2_by_layer[l]) for l in gpt2_layers]

# Pythia: average bias reduction by layer
pythia_by_layer = {}
for key, d in pythia_heads.items():
    layer = d["layer"]
    if layer not in pythia_by_layer:
        pythia_by_layer[layer] = []
    pythia_by_layer[layer].append(d["bias_reduction_pct"])

pythia_layers = sorted(pythia_by_layer.keys())
pythia_max_by_layer = [max(pythia_by_layer[l]) for l in pythia_layers]

# Normalize to relative depth
gpt2_depth = [l / 11 * 100 for l in gpt2_layers]  # 12 layers, max = 11
pythia_depth = [l / 31 * 100 for l in pythia_layers]  # 32 layers, max = 31

ax.plot(gpt2_depth, gpt2_max_by_layer, 'o-', color='steelblue', label='GPT-2 (12L)', markersize=6)
ax.plot(pythia_depth, pythia_max_by_layer, 's-', color='coral', label='Pythia-2.8B (32L)', markersize=4)
ax.set_xlabel("Relative Layer Depth (%)")
ax.set_ylabel("Max Bias Reduction in Layer (%)")
ax.set_title("Late-Layer Localization\nAcross Model Scales")
ax.legend(fontsize=9)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

# Plot 3: Summary comparison table
ax = axes[2]
ax.axis('off')

data = [
    ['', 'GPT-2\n(12L, 117M)', 'Pythia-2.8B\n(32L, 2.8B)'],
    ['Best separable', 'L10H9', 'L22H30'],
    ['Bias reduction', '35.3%', '52.6%'],
    ['PPL change', '-1.0%', '-0.4%'],
    ['Pronoun accuracy', '100%', '100%'],
    ['Layer depth', '83%', '69%'],
    ['Most entangled', 'L0H8', 'L0H1'],
    ['Entangled PPL cost', '+84%', '+64%'],
    ['Entangled layer', 'L0', 'L0'],
]

table = ax.table(cellText=data, loc='center', cellLoc='center',
                  colWidths=[0.35, 0.35, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

# Style header
for i in range(3):
    table[0, i].set_facecolor('#E8E8E8')
    table[0, i].set_text_props(fontweight='bold')

# Color best values
for row in [2, 3, 4]:
    table[row, 2].set_facecolor('#D4EDDA')  # Pythia wins

ax.set_title("Cross-Model Comparison", fontsize=13, pad=20)

plt.suptitle("Scale Replication: Separability Spectrum Generalizes Across Architectures",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig6_scale_comparison.png")
plt.savefig(FIG_DIR / "fig6_scale_comparison.pdf")
plt.close()
print("Saved fig6_scale_comparison")
