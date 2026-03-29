"""
Experiment 25: Matched Effect-Size Dose-Response

For each method, sweep intervention strength on dev set to find settings
hitting 10%, 25%, 40% absolute bias reduction targets.
At each matched level, run full capability evaluation.
Generates Pareto frontier plots.
"""

import torch
import numpy as np
import json
import sys
from pathlib import Path
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import (
    get_prompts, eval_bias, full_eval, print_results, results_to_json,
    eval_wikitext_ppl, eval_winogender
)

RESULTS_DIR = Path("results/25_dose_response")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2", device=device)


def scale_head(z, hook, head_idx, alpha):
    z[:, :, head_idx, :] *= alpha
    return z


dev_prompts = get_prompts("dev")

print("=" * 70)
print("EXPERIMENT 25: MATCHED EFFECT-SIZE DOSE-RESPONSE")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# BASELINE
# ═══════════════════════════════════════════════════════════════
baseline = eval_bias(model, dev_prompts)
baseline_abs = baseline["abs_bias"]
baseline_ppl_arr = eval_wikitext_ppl(model, hooks=None, n_sentences=200)
baseline_ppl = float(np.mean(baseline_ppl_arr))
print("Baseline abs bias: %.4f" % baseline_abs)
print("Baseline PPL:      %.2f" % baseline_ppl)

# ═══════════════════════════════════════════════════════════════
# METHOD 1: L10H9 HEAD SCALING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("METHOD 1: L10H9 HEAD SCALING (alpha sweep)")
print("=" * 70)

alphas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
head_scaling_sweep = []

for alpha in alphas:
    hooks = [("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=alpha))]
    bias = eval_bias(model, dev_prompts, hooks=hooks)
    ppl_arr = eval_wikitext_ppl(model, hooks=hooks, n_sentences=100)
    ppl_mean = float(np.mean(ppl_arr))
    reduction = (baseline_abs - bias["abs_bias"]) / baseline_abs * 100
    ppl_change = (ppl_mean - baseline_ppl) / baseline_ppl * 100

    head_scaling_sweep.append({
        "alpha": alpha,
        "abs_bias": bias["abs_bias"],
        "signed_bias": bias["signed_bias"],
        "reduction_pct": float(reduction),
        "ppl_mean": ppl_mean,
        "ppl_change_pct": float(ppl_change),
        "gender_mass": bias["total_gender_mass"],
    })
    print("  alpha=%.1f: bias_red=%.1f%%, PPL_Δ=%+.1f%%, signed=%+.4f" % (
        alpha, reduction, ppl_change, bias["signed_bias"]))

# ═══════════════════════════════════════════════════════════════
# METHOD 2: L10H9 + OTHER HEADS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("METHOD 2: MULTI-HEAD ABLATION")
print("=" * 70)

# Test ablating L10H9 + other top heads from discovery
additional_heads = [(11, 0), (5, 9), (10, 0)]  # L11H0, L5H9, L10H0

multi_head_sweep = []
for extra_heads in [[], [(11, 0)], [(11, 0), (5, 9)], [(11, 0), (5, 9), (10, 0)]]:
    all_hooks = [("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.0))]
    for layer, head in extra_heads:
        all_hooks.append(("blocks.%d.attn.hook_z" % layer,
                          partial(scale_head, head_idx=head, alpha=0.0)))

    label = "L10H9" + ("+" + "+".join("L%dH%d" % (l, h) for l, h in extra_heads) if extra_heads else "")
    bias = eval_bias(model, dev_prompts, hooks=all_hooks)
    ppl_arr = eval_wikitext_ppl(model, hooks=all_hooks, n_sentences=100)
    ppl_mean = float(np.mean(ppl_arr))
    reduction = (baseline_abs - bias["abs_bias"]) / baseline_abs * 100
    ppl_change = (ppl_mean - baseline_ppl) / baseline_ppl * 100

    multi_head_sweep.append({
        "label": label,
        "abs_bias": bias["abs_bias"],
        "signed_bias": bias["signed_bias"],
        "reduction_pct": float(reduction),
        "ppl_mean": ppl_mean,
        "ppl_change_pct": float(ppl_change),
    })
    print("  %s: bias_red=%.1f%%, PPL_Δ=%+.1f%%" % (label, reduction, ppl_change))

# ═══════════════════════════════════════════════════════════════
# FIND MATCHED EFFECT-SIZE SETTINGS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MATCHED EFFECT-SIZE COMPARISONS")
print("=" * 70)

targets = [10, 25, 40]

def find_closest(sweep, target):
    """Find sweep point closest to target reduction %."""
    return min(sweep, key=lambda x: abs(x["reduction_pct"] - target))


matched_results = {}
for target in targets:
    print("\n--- Target: %d%% bias reduction ---" % target)
    matched = {}

    # Head scaling
    closest = find_closest(head_scaling_sweep, target)
    matched["head_scaling"] = closest
    print("  Head scaling: alpha=%.1f → %.1f%% red, PPL %+.1f%%" % (
        closest["alpha"], closest["reduction_pct"], closest["ppl_change_pct"]))

    matched_results["%d_pct" % target] = matched

# ═══════════════════════════════════════════════════════════════
# FULL EVAL ON KEY SETTINGS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FULL EVALUATION ON KEY DOSE SETTINGS")
print("=" * 70)

key_configs = [
    ("Baseline", None),
    ("L10H9 alpha=0.5", [("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.5))]),
    ("L10H9 alpha=0.0", [("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.0))]),
]

full_eval_results = {}
for label, hooks in key_configs:
    print("\n--- %s ---" % label)
    results = full_eval(model, hooks=hooks, split="dev", capability="full", verbose=True)
    print_results(results, label=label)
    full_eval_results[label] = results_to_json(results)

# ═══════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: Dose-response curve
ax = axes[0]
alphas_plot = [s["alpha"] for s in head_scaling_sweep]
reductions = [s["reduction_pct"] for s in head_scaling_sweep]
ppls = [s["ppl_change_pct"] for s in head_scaling_sweep]
ax.plot(alphas_plot, reductions, 'o-', color='steelblue', label='Bias reduction', markersize=6)
ax2 = ax.twinx()
ax2.plot(alphas_plot, ppls, 's-', color='coral', label='PPL change', markersize=5)
ax.set_xlabel("Scaling alpha")
ax.set_ylabel("Bias reduction (%)", color='steelblue')
ax2.set_ylabel("PPL change (%)", color='coral')
ax.set_title("L10H9 Dose-Response")
ax.invert_xaxis()
ax.legend(loc='upper left', fontsize=9)
ax2.legend(loc='upper right', fontsize=9)

# Plot 2: Pareto frontier (bias reduction vs PPL cost)
ax = axes[1]
for s in head_scaling_sweep:
    ax.scatter(s["reduction_pct"], s["ppl_change_pct"],
              c='steelblue', s=60, alpha=0.7, zorder=5)
    ax.annotate("%.1f" % s["alpha"], (s["reduction_pct"], s["ppl_change_pct"]),
               fontsize=7, textcoords='offset points', xytext=(3, 3))
for s in multi_head_sweep:
    ax.scatter(s["reduction_pct"], s["ppl_change_pct"],
              c='coral', s=80, marker='s', alpha=0.7, zorder=5)
ax.set_xlabel("Absolute Bias Reduction (%)")
ax.set_ylabel("PPL Change (%)")
ax.set_title("Pareto Frontier")
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax.axhline(y=5, color='red', linestyle=':', alpha=0.3, label='5% PPL threshold')
ax.legend(fontsize=9)

# Plot 3: Signed bias trajectory
ax = axes[2]
signed = [s["signed_bias"] for s in head_scaling_sweep]
ax.plot(alphas_plot, signed, 'o-', color='purple', markersize=6)
ax.set_xlabel("Scaling alpha")
ax.set_ylabel("Signed bias (+ = male skew)")
ax.set_title("Signed Bias Trajectory")
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax.axhline(y=baseline["signed_bias"], color='black', linestyle=':', alpha=0.3, label='Baseline')
ax.invert_xaxis()
ax.legend(fontsize=9)

plt.suptitle("Matched Effect-Size Dose-Response", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "dose_response.png", dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════
save_data = {
    "baseline_abs_bias": baseline_abs,
    "baseline_ppl": baseline_ppl,
    "head_scaling_sweep": head_scaling_sweep,
    "multi_head_sweep": multi_head_sweep,
    "matched_results": matched_results,
    "full_eval_results": full_eval_results,
}

with open(RESULTS_DIR / "dose_response_results.json", "w") as f:
    json.dump(save_data, f, indent=2, default=str)

print("\nResults saved to %s" % RESULTS_DIR)
print("\n✓ Experiment 25 complete.")
