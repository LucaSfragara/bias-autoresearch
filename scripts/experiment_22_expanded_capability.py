"""
Experiment 22: Capability Validation + Benchmark Deep-Dive

Validates the reusable evaluator and produces detailed benchmark breakdowns.
Runs full_eval on baseline + all interventions (L10H9, combined, F23406).

Additional deep-dive:
- Full StereoSet gender (intrasentence + intersentence)
- Per-occupation breakdown of capability changes
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
    full_eval, print_results, results_to_json, eval_bias,
    get_prompts, get_gender_ids
)

RESULTS_DIR = Path("results/22_expanded_capability")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2", device=device)

# ═══════════════════════════════════════════════════════════════
# INTERVENTION HOOKS
# ═══════════════════════════════════════════════════════════════

def scale_head(z, hook, head_idx, alpha):
    z[:, :, head_idx, :] *= alpha
    return z


def make_sae_hook(model, device):
    """Try to load SAE for combined intervention. Returns hook or None."""
    try:
        from sae_lens import SAE
        sae = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id="blocks.10.hook_resid_pre",
            device=device,
        )

        def scale_sae_features(resid, hook, sae=sae):
            sae_acts = sae.encode(resid)
            feat_23440 = sae_acts[:, :, 23440:23441]
            feat_16291 = sae_acts[:, :, 16291:16292]
            dir_23440 = sae.W_dec[23440]
            dir_16291 = sae.W_dec[16291]
            adj = -0.5 * feat_23440 * dir_23440 + -0.5 * feat_16291 * dir_16291
            return resid + adj

        return scale_sae_features
    except Exception as e:
        print("SAE loading failed: %s" % e)
        return None


INTERVENTIONS = {
    "Baseline": None,
    "L10H9 ablation": [("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.0))],
}

# Try to add combined intervention
sae_hook = make_sae_hook(model, device)
if sae_hook:
    INTERVENTIONS["Combined (L10H9 + SAE)"] = [
        ("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.0)),
        ("blocks.10.hook_resid_pre", sae_hook),
    ]
    # F23406 artifact
    try:
        from sae_lens import SAE
        sae_l0 = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id="blocks.0.hook_resid_pre",
            device=device,
        )

        def clamp_f23406(resid, hook, sae=sae_l0):
            sae_acts = sae.encode(resid)
            feat = sae_acts[:, :, 23406:23407]
            feat_dir = sae.W_dec[23406]
            adj = -feat * feat_dir
            return resid + adj

        INTERVENTIONS["F23406 artifact"] = [("blocks.0.hook_resid_pre", clamp_f23406)]
    except Exception as e:
        print("L0 SAE loading failed: %s" % e)

print("=" * 70)
print("EXPERIMENT 22: CAPABILITY VALIDATION")
print("=" * 70)
print("Interventions to test: %s" % list(INTERVENTIONS.keys()))

# ═══════════════════════════════════════════════════════════════
# FULL EVAL ON ALL INTERVENTIONS (dev split)
# ═══════════════════════════════════════════════════════════════
all_results = {}
for name, hooks in INTERVENTIONS.items():
    print("\n" + "=" * 70)
    print("EVALUATING: %s" % name)
    print("=" * 70)
    results = full_eval(model, hooks=hooks, split="dev", capability="full", verbose=True)
    print_results(results, label=name)
    all_results[name] = results_to_json(results)

# ═══════════════════════════════════════════════════════════════
# PER-OCCUPATION BREAKDOWN
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PER-OCCUPATION BIAS BREAKDOWN (dev split)")
print("=" * 70)

from eval_utils import load_splits
splits = load_splits()
dev_occs = splits["dev"]["occupations"]
dev_tmpls = splits["dev"]["templates"]
male_ids, female_ids = get_gender_ids(model)

hooks_l10h9 = INTERVENTIONS["L10H9 ablation"]

per_occ_results = {}
print("\n%-15s %10s %10s %10s %10s" % ("Occupation", "Base_sign", "Abl_sign", "Base_abs", "Abl_abs"))
print("-" * 57)

for occ in dev_occs:
    occ_prompts = ["The %s %s" % (occ, t) for t in dev_tmpls]

    base = eval_bias(model, occ_prompts)
    ablated = eval_bias(model, occ_prompts, hooks=hooks_l10h9)

    per_occ_results[occ] = {
        "baseline_signed": base["signed_bias"],
        "baseline_abs": base["abs_bias"],
        "ablated_signed": ablated["signed_bias"],
        "ablated_abs": ablated["abs_bias"],
        "abs_reduction_pct": (base["abs_bias"] - ablated["abs_bias"]) / (base["abs_bias"] + 1e-10) * 100,
    }

    print("%-15s %+9.4f %+9.4f %10.4f %10.4f" % (
        occ, base["signed_bias"], ablated["signed_bias"],
        base["abs_bias"], ablated["abs_bias"]))

# ═══════════════════════════════════════════════════════════════
# STEREOSET (if available)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEREOSET EVALUATION")
print("=" * 70)

try:
    from datasets import load_dataset
    from eval_utils import pseudo_log_likelihood

    ds = load_dataset("stereoset", "intersentence", split="validation", trust_remote_code=True)

    # Filter to gender
    gender_examples = [ex for ex in ds if ex["bias_type"] == "gender"]
    print("StereoSet gender examples: %d" % len(gender_examples))

    stereoset_results = {}
    for int_name, hooks in [("Baseline", None), ("L10H9 ablation", hooks_l10h9)]:
        stereo_scores = []
        for ex in gender_examples:
            context = ex["context"]
            sentences = ex["sentences"]
            labels = sentences["gold_label"]
            texts = sentences["sentence"]

            plls = {}
            for text, label in zip(texts, labels):
                full_text = context + " " + text
                pll = pseudo_log_likelihood(model, full_text, hooks)
                plls[label] = pll

            if "stereotype" in plls and "anti-stereotype" in plls:
                stereo_scores.append(1 if plls["stereotype"] > plls["anti-stereotype"] else 0)

        score = np.mean(stereo_scores) if stereo_scores else 0
        stereoset_results[int_name] = float(score)
        print("  %s: stereotype score = %.1f%% (%d examples)" % (int_name, score * 100, len(stereo_scores)))

    all_results["stereoset"] = stereoset_results
except Exception as e:
    print("StereoSet evaluation failed: %s" % e)

# ═══════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FULL COMPARISON TABLE")
print("=" * 70)

metrics = [
    ("signed_bias", "Signed bias", "%.4f"),
    ("abs_bias", "Abs bias", "%.4f"),
    ("total_gender_mass", "Gender mass", "%.4f"),
    ("wikitext_ppl", "WikiText PPL", "%.2f"),
    ("lambada_acc", "LAMBADA acc", "%.3f"),
    ("winogender_male_pref", "Winogender M%", "%.3f"),
    ("winobias_type1_gap", "WinoBias T1 gap", "%.3f"),
    ("winobias_type2_gap", "WinoBias T2 gap", "%.3f"),
    ("gap_overall", "GAP overall", "%.3f"),
    ("crows_pairs_score", "CrowS-Pairs", "%.3f"),
]

header = "%-20s" + " %12s" * len(all_results)
print(header % tuple(["Metric"] + list(all_results.keys())))
print("-" * (20 + 13 * len(all_results)))

for key, label, fmt in metrics:
    vals = []
    for name in all_results:
        v = all_results[name].get(key, None)
        if v is not None:
            vals.append(fmt % v)
        else:
            vals.append("N/A")
    row = "%-20s" + " %12s" * len(vals)
    print(row % tuple([label] + vals))

# ═══════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Per-occupation signed bias
ax = axes[0, 0]
occs = list(per_occ_results.keys())
base_signed = [per_occ_results[o]["baseline_signed"] for o in occs]
abl_signed = [per_occ_results[o]["ablated_signed"] for o in occs]
x = np.arange(len(occs))
ax.bar(x - 0.2, base_signed, 0.4, label='Baseline', color='gray', alpha=0.7)
ax.bar(x + 0.2, abl_signed, 0.4, label='L10H9 ablated', color='steelblue', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(occs, rotation=45, ha='right', fontsize=8)
ax.set_ylabel("Signed bias (P(M) - P(F))")
ax.set_title("Per-Occupation Signed Bias")
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.legend(fontsize=8)

# Plot 2: Capability metrics comparison
ax = axes[0, 1]
cap_metrics = ["wikitext_ppl", "lambada_acc", "winogender_male_pref", "crows_pairs_score"]
cap_labels = ["PPL/10", "LAMBADA", "Wino M%", "CrowS"]
for i, (name, res) in enumerate(all_results.items()):
    vals = []
    for m in cap_metrics:
        v = res.get(m, 0)
        if m == "wikitext_ppl":
            v = v / 10  # Scale PPL to fit
        vals.append(v)
    ax.bar(np.arange(len(vals)) + i * 0.2, vals, 0.18, label=name, alpha=0.7)
ax.set_xticks(np.arange(len(cap_labels)))
ax.set_xticklabels(cap_labels)
ax.set_title("Capability Metrics")
ax.legend(fontsize=7, loc='upper right')

# Plot 3: WinoBias pro vs anti
ax = axes[1, 0]
wb_configs = ["type1_pro", "type1_anti", "type2_pro", "type2_anti"]
wb_labels = ["T1 Pro", "T1 Anti", "T2 Pro", "T2 Anti"]
for i, (name, res) in enumerate(all_results.items()):
    vals = [res.get("winobias_%s" % c, 0) for c in ["type1_pro_acc", "type1_anti_acc", "type2_pro_acc", "type2_anti_acc"]]
    ax.bar(np.arange(4) + i * 0.2, vals, 0.18, label=name, alpha=0.7)
ax.set_xticks(np.arange(4))
ax.set_xticklabels(wb_labels)
ax.set_ylabel("Accuracy")
ax.set_title("WinoBias: Pro vs Anti-Stereotypical")
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.legend(fontsize=7)

# Plot 4: Abs bias reduction by occupation
ax = axes[1, 1]
reductions = [per_occ_results[o]["abs_reduction_pct"] for o in occs]
colors = ['green' if r > 0 else 'red' for r in reductions]
ax.barh(range(len(occs)), reductions, color=colors, alpha=0.7)
ax.set_yticks(range(len(occs)))
ax.set_yticklabels(occs, fontsize=8)
ax.set_xlabel("Abs bias reduction (%)")
ax.set_title("Per-Occupation Bias Reduction (L10H9)")
ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

plt.suptitle("Experiment 22: Expanded Capability Validation", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "expanded_capability.png", dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════
save_data = {
    "interventions": list(all_results.keys()),
    "full_eval_results": all_results,
    "per_occupation": per_occ_results,
}

with open(RESULTS_DIR / "expanded_capability_results.json", "w") as f:
    json.dump(save_data, f, indent=2, default=str)

print("\nResults saved to %s" % RESULTS_DIR)
print("\n✓ Experiment 22 complete.")
