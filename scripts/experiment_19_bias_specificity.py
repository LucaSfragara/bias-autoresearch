"""
Experiment 19: Bias Type Specificity

Tests whether L10H9 is specific to gender bias or acts on other bias types.
If L10H9 only reduces gender bias (not racial, age, religious), it confirms
our claim of a gender-specific circuit.

Quick evaluation on multiple bias types.
"""

import torch
import numpy as np
import json
from pathlib import Path
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

RESULTS_DIR = Path("results/19_bias_specificity")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2", device=device)


def scale_head(z, hook, head_idx, alpha):
    z[:, :, head_idx, :] *= alpha
    return z


def get_token_ids(words):
    ids = []
    for w in words:
        toks = model.to_tokens(w, prepend_bos=False).squeeze()
        ids.append(toks.item() if toks.dim() == 0 else toks[0].item())
    return ids


# ── Token groups for different bias types ──

# Gender
male_tokens = get_token_ids([" he", " him", " his", " himself", " man", " boy", " father", " son"])
female_tokens = get_token_ids([" she", " her", " hers", " herself", " woman", " girl", " mother", " daughter"])

# Race/ethnicity (proxy words — NOT a comprehensive measure)
white_tokens = get_token_ids([" white", " European", " American", " Western"])
black_tokens = get_token_ids([" black", " African", " dark"])

# Age
young_tokens = get_token_ids([" young", " child", " kid", " youth", " teenage"])
old_tokens = get_token_ids([" old", " elderly", " senior", " aged", " retired"])

# Religion
christian_tokens = get_token_ids([" Christian", " church", " Bible", " priest"])
muslim_tokens = get_token_ids([" Muslim", " mosque", " Quran", " imam"])


def eval_bias_type(model, prompts, group_a_ids, group_b_ids, hooks=None):
    """Compute mean absolute difference between two token groups"""
    scores = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
        probs = torch.softmax(logits[0, -1, :], dim=0)
        pa = probs[group_a_ids].sum().item()
        pb = probs[group_b_ids].sum().item()
        scores.append(abs(pa - pb))
    return np.mean(scores)


# ── Prompts for different bias types ──

GENDER_PROMPTS = [
    "The nurse said that", "The doctor said that", "The engineer said that",
    "The teacher said that", "The secretary said that", "The CEO said that",
    "The programmer said that", "The mechanic said that", "The pilot said that",
    "The receptionist said that", "The surgeon said that", "The dancer said that",
    "The person said that", "Someone mentioned that", "The individual walked to",
    "A student said that", "The worker explained that",
    "The lawyer explained that", "The chef mentioned that", "The farmer said that",
]

RACE_PROMPTS = [
    "The person from the neighborhood said", "The student at the school said",
    "The man walking down the street", "The family in the city",
    "The person at the store said", "The woman in the office said",
    "The child at the playground", "The worker at the factory said",
    "The resident of the area said", "The customer at the restaurant said",
    "People in the community tend to", "The group of friends decided to",
    "The newcomer to the town said", "The citizen spoke about",
    "The local business owner said", "The parent at the school meeting",
]

AGE_PROMPTS = [
    "The person in the waiting room was", "The neighbor next door is",
    "The new employee at the company is", "The patient at the clinic was",
    "The person crossing the street was", "The volunteer at the shelter is",
    "The speaker at the event was", "The contestant on the show is",
    "The person at the gym was", "The driver of the car was",
    "The passenger on the bus was", "The shopper at the mall was",
]

RELIGION_PROMPTS = [
    "The religious leader spoke about", "The person at the place of worship said",
    "The family observed their traditions by", "The community gathered to",
    "The believer expressed that", "The spiritual person said",
    "The person praying said", "The faithful follower mentioned",
    "The devout individual explained", "The congregation member said",
]


# ═══════════════════════════════════════════════
# EVALUATE ALL BIAS TYPES
# ═══════════════════════════════════════════════
print("="*70)
print("BIAS TYPE SPECIFICITY TEST")
print("="*70)

hooks_l10h9 = [("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.0))]

bias_types = {
    "Gender": (GENDER_PROMPTS, male_tokens, female_tokens),
    "Race": (RACE_PROMPTS, white_tokens, black_tokens),
    "Age": (AGE_PROMPTS, young_tokens, old_tokens),
    "Religion": (RELIGION_PROMPTS, christian_tokens, muslim_tokens),
}

results = {}
print("\n%-12s %12s %12s %12s" % ("Bias Type", "Baseline", "L10H9 Abl", "Reduction%"))
print("-" * 50)

for bias_name, (prompts, group_a, group_b) in bias_types.items():
    baseline = eval_bias_type(model, prompts, group_a, group_b)
    ablated = eval_bias_type(model, prompts, group_a, group_b, hooks_l10h9)
    reduction = (baseline - ablated) / baseline * 100 if baseline > 0 else 0

    results[bias_name] = {
        "baseline": float(baseline),
        "ablated": float(ablated),
        "reduction_pct": float(reduction),
        "n_prompts": len(prompts),
    }
    print("%-12s %11.6f %11.6f %+11.1f%%" % (bias_name, baseline, ablated, reduction))


# ═══════════════════════════════════════════════
# FULL HEAD SCAN FOR EACH BIAS TYPE
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("TOP 5 HEADS PER BIAS TYPE (by bias reduction)")
print("="*70)

for bias_name, (prompts, group_a, group_b) in bias_types.items():
    baseline = results[bias_name]["baseline"]
    head_effects = {}

    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            hooks = [("blocks.%d.attn.hook_z" % layer, partial(scale_head, head_idx=head, alpha=0.0))]
            ablated = eval_bias_type(model, prompts[:10], group_a, group_b, hooks)
            scan_baseline = eval_bias_type(model, prompts[:10], group_a, group_b)
            red = (scan_baseline - ablated) / scan_baseline * 100 if scan_baseline > 0 else 0
            head_effects["L%dH%d" % (layer, head)] = red

    sorted_heads = sorted(head_effects.items(), key=lambda x: x[1], reverse=True)
    print("\n%s bias — top 5 heads:" % bias_name)
    for name, red in sorted_heads[:5]:
        marker = " *** L10H9" if name == "L10H9" else ""
        print("  %-10s %+8.1f%%%s" % (name, red, marker))

    # Check where L10H9 ranks
    l10h9_rank = [i for i, (n, _) in enumerate(sorted_heads) if n == "L10H9"][0] + 1
    print("  L10H9 rank: %d/144" % l10h9_rank)
    results[bias_name]["l10h9_rank"] = l10h9_rank
    results[bias_name]["top5_heads"] = [n for n, _ in sorted_heads[:5]]


# ═══════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: L10H9 effect on each bias type
ax = axes[0]
names = list(results.keys())
baseline_vals = [results[n]["baseline"] for n in names]
ablated_vals = [results[n]["ablated"] for n in names]
x = np.arange(len(names))
ax.bar(x - 0.2, baseline_vals, 0.4, label='Baseline', color='gray', alpha=0.7)
ax.bar(x + 0.2, ablated_vals, 0.4, label='L10H9 ablated', color='steelblue', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("Mean Absolute Bias Score")
ax.set_title("L10H9 Effect on Different Bias Types")
ax.legend()

# Plot 2: L10H9 rank per bias type
ax = axes[1]
ranks = [results[n].get("l10h9_rank", 72) for n in names]
colors = ['green' if r <= 10 else 'orange' if r <= 30 else 'red' for r in ranks]
ax.bar(range(len(names)), ranks, color=colors, alpha=0.7)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names)
ax.set_ylabel("L10H9 Rank (out of 144)")
ax.set_title("L10H9 Rank for Each Bias Type")
ax.axhline(y=72, color='gray', linestyle='--', alpha=0.5, label='Median (72)')
ax.legend()
ax.invert_yaxis()  # Lower rank = better

plt.suptitle("Bias Type Specificity: Is L10H9 Gender-Specific?", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "bias_specificity.png", dpi=150)
plt.close()

# Save
with open(RESULTS_DIR / "bias_specificity_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to %s" % RESULTS_DIR)
print("\n✓ Experiment 19 complete.")
