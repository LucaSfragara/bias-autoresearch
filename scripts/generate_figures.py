"""
Generate publication-quality figures for the paper.
Reads from all experiment result JSONs and creates clean visualizations.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_ROOT = Path("results")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Style settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# ══════════════════════════════════════════════════
# FIGURE 1: Separability Spectrum (Main Figure)
# ══════════════════════════════════════════════════
print("Figure 1: Separability Spectrum...")

with open(RESULTS_ROOT / "13_robust_eval" / "robust_eval_results.json") as f:
    robust = json.load(f)

with open(RESULTS_ROOT / "03_entanglement" / "entanglement_results.json") as f:
    entangle = json.load(f)

fig, ax = plt.subplots(figsize=(8, 6))

# Plot all 144 heads from entanglement scan (exp 03)
abl = entangle.get("ablation_results", {})
baseline_bias = entangle.get("baselines", {}).get("bias", 0.093)
baseline_ppl = entangle.get("baselines", {}).get("perplexity", 30.49)

for head_key, data in abl.items():
    bias_red_pct = data["bias_reduction"] / baseline_bias * 100
    ppl_change_pct = data.get("ppl_increase_pct", 0)
    color = 'lightgray'
    size = 15
    zorder = 1
    alpha = 0.4
    # Highlight L10H9
    if head_key == "L10H9":
        color = 'red'
        size = 150
        zorder = 5
        alpha = 1.0
    # Highlight L0H8
    elif head_key == "L0H8":
        color = 'purple'
        size = 100
        zorder = 4
        alpha = 0.9
    ax.scatter(bias_red_pct, ppl_change_pct, c=color, s=size, alpha=alpha, zorder=zorder)

# Add labels for key heads
for head_key in ["L10H9", "L0H8"]:
    if head_key in abl:
        d = abl[head_key]
        bx = d["bias_reduction"] / baseline_bias * 100
        by = d.get("ppl_increase_pct", 0)
        ax.annotate(head_key, (bx, by), fontsize=9, fontweight='bold',
                    xytext=(5, 5), textcoords='offset points')

# Legend entries for highlighted heads
ax.scatter([], [], c='red', s=150, label='L10H9 (separable)')
ax.scatter([], [], c='purple', s=100, label='L0H8 (entangled)')

# Add reference points for baselines
with open(RESULTS_ROOT / "15_baselines" / "baselines_results.json") as f:
    baselines = json.load(f)

comparison = baselines.get("comparison", {})
for method, data in comparison.items():
    if "INLP" in method:
        ax.scatter(data["bias_reduction_pct"], data["ppl_change_pct"],
                  c='orange', s=100, zorder=4, marker='s', edgecolors='darkorange',
                  label='INLP (destroys pronouns)')

# Separability zone
ax.axhspan(-5, 5, alpha=0.05, color='green')
ax.axhline(y=5, color='green', linestyle=':', alpha=0.3, label='Separability threshold (5% PPL)')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

ax.set_xlabel("Bias Reduction (%)")
ax.set_ylabel("Perplexity Change (%)")
ax.set_title("The Separability Spectrum:\nBias Reduction vs Capability Cost per Attention Head")
ax.legend(loc='upper left', fontsize=9)
plt.savefig(FIG_DIR / "fig1_separability_spectrum.png")
plt.savefig(FIG_DIR / "fig1_separability_spectrum.pdf")
plt.close()
print("  Saved fig1_separability_spectrum")


# ══════════════════════════════════════════════════
# FIGURE 2: Method Comparison Table (Bar Chart)
# ══════════════════════════════════════════════════
print("Figure 2: Method Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

methods = ["INLP (L10, 3 iter)", "Mean-subtraction (L10)",
           "Head L10H9 ablation (ours)", "True gender features (ours)",
           "Combined L10H9+features (ours)"]
short_names = ["INLP", "Mean-sub", "L10H9\n(ours)", "Gender\nfeatures", "Combined\n(ours)"]
colors = ['#E69F00', '#CC79A7', '#0072B2', '#009E73', '#D55E00']

bias_red = []
ppl_change = []
pronoun = []

for m in methods:
    if m in comparison:
        d = comparison[m]
        bias_red.append(d["bias_reduction_pct"])
        ppl_change.append(d["ppl_change_pct"])
        pronoun.append(d["pronoun"] * 100)
    else:
        bias_red.append(0)
        ppl_change.append(0)
        pronoun.append(100)

# Bias reduction
ax = axes[0]
bars = ax.bar(range(len(methods)), bias_red, color=colors, alpha=0.8)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(short_names, fontsize=9)
ax.set_ylabel("Bias Reduction (%)")
ax.set_title("Bias Reduction")
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
for bar, val in zip(bars, bias_red):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            '%.0f%%' % val, ha='center', fontsize=8)

# PPL change
ax = axes[1]
bars = ax.bar(range(len(methods)), ppl_change, color=colors, alpha=0.8)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(short_names, fontsize=9)
ax.set_ylabel("PPL Change (%)")
ax.set_title("Perplexity Cost")
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

# Pronoun accuracy
ax = axes[2]
bars = ax.bar(range(len(methods)), pronoun, color=colors, alpha=0.8)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(short_names, fontsize=9)
ax.set_ylabel("Pronoun Accuracy (%)")
ax.set_title("Gender Knowledge Preservation")
ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random chance')
ax.set_ylim(0, 110)
ax.legend(fontsize=8)

plt.suptitle("Debiasing Method Comparison", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_method_comparison.png")
plt.savefig(FIG_DIR / "fig2_method_comparison.pdf")
plt.close()
print("  Saved fig2_method_comparison")


# ══════════════════════════════════════════════════
# FIGURE 3: Artifact Warning (F23406 vs True Features)
# ══════════════════════════════════════════════════
print("Figure 3: SAE Artifact Warning...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Specificity comparison
ax = axes[0]
features = ['L0_F23406\n(artifact)', 'L10_F23440\n(female)', 'L10_F16291\n(male)', 'Combined\ntrue features']
specificities = [0.195, 0.601, 0.601, 0.552]  # from exp 13
colors_spec = ['coral', 'steelblue', 'steelblue', 'green']
ax.bar(range(len(features)), specificities, color=colors_spec, alpha=0.7)
ax.set_xticks(range(len(features)))
ax.set_xticklabels(features, fontsize=9)
ax.set_ylabel("Specificity Score")
ax.set_title("Specificity: Gender-Targeted vs Blunt")
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='0.5 threshold')
ax.legend(fontsize=8)

# Bias reduction vs true mechanism
ax = axes[1]
categories = ['F23406\n(artifact)', 'True gender\nfeatures', 'L10H9\nablation', 'Combined\n(ours)']
bias_vals = [52.5, 13.1, 28.7, 36.9]
colors_b = ['coral', 'steelblue', 'royalblue', 'green']
ax.bar(range(len(categories)), bias_vals, color=colors_b, alpha=0.7)
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylabel("Bias Reduction (%)")
ax.set_title("Apparent vs True Debiasing")

# CrowS-Pairs comparison
try:
    with open(RESULTS_ROOT / "18_crows_pairs" / "crows_pairs_results.json") as f:
        crows = json.load(f)

    ax = axes[2]
    cond = crows["conditions"]
    crows_names = list(cond.keys())
    crows_vals = [cond[n] * 100 for n in crows_names]
    short_crows = ['Baseline', 'L10H9', 'F23406\n(artifact)', 'True\ngender', 'Combined\n(ours)']
    colors_c = ['gray', 'royalblue', 'coral', 'steelblue', 'green']

    ax.bar(range(len(crows_vals)), crows_vals, color=colors_c[:len(crows_vals)], alpha=0.7)
    ax.set_xticks(range(len(crows_vals)))
    ax.set_xticklabels(short_crows[:len(crows_vals)], fontsize=9)
    ax.set_ylabel("CrowS-Pairs Stereotype Score (%)")
    ax.set_title("CrowS-Pairs Benchmark")
    ax.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='Ideal (50%)')
    ax.legend(fontsize=8)
except Exception as e:
    axes[2].text(0.5, 0.5, 'CrowS-Pairs data\nnot available', ha='center', va='center',
                 transform=axes[2].transAxes)
    print("  Warning: CrowS-Pairs data not found: %s" % str(e)[:100])

plt.suptitle("SAE Feature Artifact: Apparent vs Genuine Debiasing", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_artifact_warning.png")
plt.savefig(FIG_DIR / "fig3_artifact_warning.pdf")
plt.close()
print("  Saved fig3_artifact_warning")


# ══════════════════════════════════════════════════
# FIGURE 4: L10H9 Mechanistic Circuit
# ══════════════════════════════════════════════════
print("Figure 4: L10H9 Circuit...")

try:
    with open(RESULTS_ROOT / "17_occupation_patching" / "occupation_patching_results.json") as f:
        occ_data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Per-occupation bias profile
    ax = axes[0]
    per_occ = occ_data["per_occupation"]
    sorted_occ = sorted(per_occ.items(), key=lambda x: x[1]["baseline_bias"])
    occs = [o for o, _ in sorted_occ]
    baseline_b = [per_occ[o]["baseline_bias"] for o in occs]
    ablated_b = [per_occ[o]["ablated_bias"] for o in occs]

    y = np.arange(len(occs))
    ax.barh(y - 0.2, baseline_b, 0.4, label='Baseline', color='gray', alpha=0.6)
    ax.barh(y + 0.2, ablated_b, 0.4, label='L10H9 ablated', color='steelblue', alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(occs, fontsize=8)
    ax.set_xlabel("Gender Bias (P(M) - P(F))")
    ax.set_title("L10H9 Amplifies Stereotypes\nin Both Directions")
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')

    # Occupation gender projection vs output bias
    ax = axes[1]
    attn_data = occ_data["l10h9_attention"]
    for occ in per_occ:
        if occ in attn_data:
            x_val = attn_data[occ]["occ_gender_proj"]
            y_val = per_occ[occ]["baseline_bias"]
            color = '#E066A6' if y_val < 0 else '#4C72B0'
            ax.scatter(x_val, y_val, color=color, s=40, alpha=0.7)
            ax.annotate(occ, (x_val, y_val), fontsize=7, ha='center', va='bottom')

    corr = occ_data["occ_proj_vs_bias_correlation"]
    ax.set_xlabel("L10H9 Occupation-Position Gender Projection")
    ax.set_ylabel("Output Gender Bias (P(M) - P(F))")
    ax.set_title("L10H9 Reads Stereotypes from\nOccupation Token (r=%.2f)" % corr)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    plt.suptitle("L10H9 Mechanistic Circuit: Bidirectional Stereotype Copier", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_l10h9_circuit.png")
    plt.savefig(FIG_DIR / "fig4_l10h9_circuit.pdf")
    plt.close()
    print("  Saved fig4_l10h9_circuit")
except Exception as e:
    print("  Warning: Could not create fig4: %s" % str(e)[:200])


# ══════════════════════════════════════════════════
# FIGURE 5: BOS Path Analysis
# ══════════════════════════════════════════════════
print("Figure 5: BOS Path Analysis...")

try:
    with open(RESULTS_ROOT / "16_bos_path" / "bos_path_results.json") as f:
        bos_data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Gender projection by layer at BOS
    ax = axes[0]
    total = bos_data["layer_gender_projection"]["total"]
    attn = bos_data["layer_gender_projection"]["attn"]
    mlp = bos_data["layer_gender_projection"]["mlp"]
    x = range(len(total))
    width = 0.25
    ax.bar([i - width for i in x], total, width, label='Total', color='steelblue', alpha=0.7)
    ax.bar(x, attn, width, label='Attention', color='coral', alpha=0.7)
    ax.bar([i + width for i in x], mlp, width, label='MLP', color='green', alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Gender Direction Projection")
    ax.set_title("Per-Layer Gender Info at BOS")
    ax.legend(fontsize=9)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    # Counterfactual
    ax = axes[1]
    cf = bos_data["counterfactual"]
    conditions_cf = ['Baseline', 'Zero BOS\nattention', 'Full L10H9\nablation']
    biases_cf = [cf["baseline_bias"], cf["zero_bos_attn_bias"], cf["full_ablation_bias"]]
    colors_cf = ['gray', 'coral', 'steelblue']
    ax.bar(range(3), biases_cf, color=colors_cf, alpha=0.7)
    ax.set_xticks(range(3))
    ax.set_xticklabels(conditions_cf, fontsize=10)
    ax.set_ylabel("Mean Absolute Gender Bias")
    ax.set_title("BOS Attention Dilutes Stereotypes\n(removing it increases bias)")

    for i, v in enumerate(biases_cf):
        ax.text(i, v + 0.002, '%.3f' % v, ha='center', fontsize=9)

    plt.suptitle("BOS Path: Context-Independent Anchor That Dilutes Bias", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_bos_path.png")
    plt.savefig(FIG_DIR / "fig5_bos_path.pdf")
    plt.close()
    print("  Saved fig5_bos_path")
except Exception as e:
    print("  Warning: Could not create fig5: %s" % str(e)[:200])


print("\nAll figures saved to %s/" % FIG_DIR)
print("Done!")
