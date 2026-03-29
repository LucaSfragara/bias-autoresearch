"""
Experiment 04: Cross-Bias Analysis
- Run activation patching for race and religion bias (not just gender)
- Compare which circuits overlap across bias types
- Test: does ablating a gender-bias head affect racial bias? (side effects)
- Key question: are bias circuits shared or independent?
"""

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

RESULTS_DIR = Path("results/04_cross_bias")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading GPT-2 small...")
model = HookedTransformer.from_pretrained("gpt2", device=device)
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

# ── Define token sets for each bias type ──

# Race-related tokens
RACE_TOKEN_SETS = {
    "white": [],
    "black": [],
}
# We use name-based proxies and descriptors
white_words = [" White", " European", " Caucasian", " American"]
black_words = [" Black", " African", " dark"]

for w in white_words:
    toks = model.to_tokens(w, prepend_bos=False).squeeze()
    if toks.dim() == 0:
        RACE_TOKEN_SETS["white"].append(toks.item())
for w in black_words:
    toks = model.to_tokens(w, prepend_bos=False).squeeze()
    if toks.dim() == 0:
        RACE_TOKEN_SETS["black"].append(toks.item())

print(f"Race tokens - white: {len(RACE_TOKEN_SETS['white'])}, black: {len(RACE_TOKEN_SETS['black'])}")

# Religion tokens
RELIGION_TOKEN_SETS = {
    "christian": [],
    "muslim": [],
}
christian_words = [" Christian", " church", " Bible", " Jesus", " Christmas"]
muslim_words = [" Muslim", " mosque", " Quran", " Islamic"]

for w in christian_words:
    toks = model.to_tokens(w, prepend_bos=False).squeeze()
    if toks.dim() == 0:
        RELIGION_TOKEN_SETS["christian"].append(toks.item())
for w in muslim_words:
    toks = model.to_tokens(w, prepend_bos=False).squeeze()
    if toks.dim() == 0:
        RELIGION_TOKEN_SETS["muslim"].append(toks.item())

print(f"Religion tokens - christian: {len(RELIGION_TOKEN_SETS['christian'])}, muslim: {len(RELIGION_TOKEN_SETS['muslim'])}")

# Gender tokens (reload)
with open("results/00_setup/gender_token_ids.json") as f:
    gender_token_ids = json.load(f)
male_ids = gender_token_ids["male"]
female_ids = gender_token_ids["female"]


# ── Contrastive prompts per bias type ──
BIAS_PROMPTS = {
    "gender": [
        ("The nurse said that", "The person said that"),
        ("The doctor said that", "The person said that"),
        ("The engineer discussed", "The person discussed"),
        ("The secretary filed", "The person filed"),
        ("The CEO addressed", "The person addressed"),
        ("The teacher graded", "The person graded"),
        ("The programmer wrote", "The person wrote"),
        ("The homemaker prepared", "The person prepared"),
    ],
    "race": [
        ("The Black man walked into the store and", "The man walked into the store and"),
        ("The White man walked into the store and", "The man walked into the store and"),
        ("The African American student studied hard and", "The student studied hard and"),
        ("The Asian student studied hard and", "The student studied hard and"),
        ("People from the inner city tend to", "People from the suburbs tend to"),
        ("The immigrant worker was known to", "The worker was known to"),
        ("In the diverse neighborhood, residents", "In the neighborhood, residents"),
        ("The minority candidate applied and", "The candidate applied and"),
    ],
    "religion": [
        ("The Muslim man prayed and", "The man prayed and"),
        ("The Christian man prayed and", "The man prayed and"),
        ("At the mosque, people gathered to", "At the building, people gathered to"),
        ("At the church, people gathered to", "At the building, people gathered to"),
        ("The Jewish family celebrated and", "The family celebrated and"),
        ("The Hindu festival involved", "The festival involved"),
        ("The religious leader preached about", "The leader spoke about"),
        ("The devout believer said that", "The person said that"),
    ],
}


def compute_bias_score_generic(model, prompt, group_a_ids, group_b_ids):
    """Generic bias score: P(group_a) - P(group_b)."""
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)
    probs = torch.softmax(logits[0, -1, :], dim=0)
    pa = probs[group_a_ids].sum().item()
    pb = probs[group_b_ids].sum().item()
    return abs(pa - pb)


# ── Head patching for each bias type ──
print("\n" + "="*60)
print("HEAD PATCHING ACROSS BIAS TYPES")
print("="*60)


def patch_head_z(corrupted_z, hook, clean_cache, head_idx):
    corrupted_z[:, :, head_idx, :] = clean_cache[hook.name][:, :, head_idx, :]
    return corrupted_z


def run_head_patching_for_bias_type(bias_type, pairs):
    """Run head-level activation patching for a given bias type."""
    head_results = np.zeros((n_layers, n_heads, len(pairs)))

    # Choose appropriate token sets for measuring bias
    if bias_type == "gender":
        ga, gb = male_ids, female_ids
    elif bias_type == "race":
        ga, gb = RACE_TOKEN_SETS["white"], RACE_TOKEN_SETS["black"]
    elif bias_type == "religion":
        ga, gb = RELIGION_TOKEN_SETS["christian"], RELIGION_TOKEN_SETS["muslim"]
    else:
        raise ValueError(f"Unknown bias type: {bias_type}")

    for pair_idx, (biased_prompt, clean_prompt) in enumerate(pairs):
        biased_tokens = model.to_tokens(biased_prompt)
        clean_tokens = model.to_tokens(clean_prompt)

        if biased_tokens.shape[1] != clean_tokens.shape[1]:
            continue

        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_tokens)
            biased_logits = model(biased_tokens)

        probs_b = torch.softmax(biased_logits[0, -1, :], dim=0)
        baseline_bias = abs(probs_b[ga].sum().item() - probs_b[gb].sum().item())

        for layer in range(n_layers):
            for head in range(n_heads):
                hook_name = f"blocks.{layer}.attn.hook_z"
                with torch.no_grad():
                    patched_logits = model.run_with_hooks(
                        biased_tokens,
                        fwd_hooks=[(hook_name, partial(patch_head_z,
                                                        clean_cache=clean_cache, head_idx=head))]
                    )
                probs_p = torch.softmax(patched_logits[0, -1, :], dim=0)
                patched_bias = abs(probs_p[ga].sum().item() - probs_p[gb].sum().item())
                head_results[layer, head, pair_idx] = baseline_bias - patched_bias

    return np.mean(np.abs(head_results), axis=2)


bias_type_results = {}
for bias_type, pairs in BIAS_PROMPTS.items():
    print(f"\nPatching for {bias_type} bias...")
    head_imp = run_head_patching_for_bias_type(bias_type, pairs)
    bias_type_results[bias_type] = head_imp
    top5 = np.argsort(head_imp.flatten())[::-1][:5]
    print(f"  Top 5 heads: {[(idx // n_heads, idx % n_heads) for idx in top5]}")


# ── Cross-bias overlap analysis ──
print("\n" + "="*60)
print("CROSS-BIAS CIRCUIT OVERLAP")
print("="*60)


def get_top_k_heads(importance_matrix, k=10):
    """Get top-k important heads as a set of (layer, head) tuples."""
    flat = importance_matrix.flatten()
    top_k_indices = np.argsort(flat)[::-1][:k]
    return set((idx // n_heads, idx % n_heads) for idx in top_k_indices)


for k in [5, 10, 15, 20]:
    print(f"\nTop-{k} head overlap:")
    top_sets = {}
    for bias_type in ["gender", "race", "religion"]:
        top_sets[bias_type] = get_top_k_heads(bias_type_results[bias_type], k)

    for bt1 in ["gender", "race", "religion"]:
        for bt2 in ["gender", "race", "religion"]:
            if bt1 >= bt2:
                continue
            overlap = top_sets[bt1] & top_sets[bt2]
            jaccard = len(overlap) / len(top_sets[bt1] | top_sets[bt2]) if len(top_sets[bt1] | top_sets[bt2]) > 0 else 0
            print(f"  {bt1} ∩ {bt2}: {len(overlap)} heads, Jaccard={jaccard:.2f}")
            if overlap:
                print(f"    Shared heads: {sorted(overlap)}")


# ── Cross-bias side effects: ablating gender-bias heads and measuring race/religion bias ──
print("\n" + "="*60)
print("CROSS-BIAS SIDE EFFECTS")
print("="*60)

# Get top gender-bias heads
gender_top_heads = sorted(
    [(l, h, bias_type_results["gender"][l, h]) for l in range(n_layers) for h in range(n_heads)],
    key=lambda x: x[2], reverse=True
)[:10]

print("Ablating top 10 gender-bias heads and measuring cross-bias effects:")


def zero_ablate_head(z, hook, head_idx):
    z[:, :, head_idx, :] = 0.0
    return z


# Baseline bias scores for race and religion
race_prompts_eval = [p[0] for p in BIAS_PROMPTS["race"]]
religion_prompts_eval = [p[0] for p in BIAS_PROMPTS["religion"]]
gender_prompts_eval = [p[0] for p in BIAS_PROMPTS["gender"]]


def mean_bias_on_prompts(model, prompts, ga, gb, hooks=None):
    scores = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            if hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                logits = model(tokens)
        probs = torch.softmax(logits[0, -1, :], dim=0)
        scores.append(abs(probs[ga].sum().item() - probs[gb].sum().item()))
    return np.mean(scores)


baseline_gender_bias = mean_bias_on_prompts(model, gender_prompts_eval, male_ids, female_ids)
baseline_race_bias = mean_bias_on_prompts(model, race_prompts_eval,
                                            RACE_TOKEN_SETS["white"], RACE_TOKEN_SETS["black"])
baseline_religion_bias = mean_bias_on_prompts(model, religion_prompts_eval,
                                                RELIGION_TOKEN_SETS["christian"], RELIGION_TOKEN_SETS["muslim"])

print(f"\nBaselines: gender={baseline_gender_bias:.6f}, race={baseline_race_bias:.6f}, religion={baseline_religion_bias:.6f}")

cross_effects = []
for layer, head, imp in gender_top_heads:
    hook_name = f"blocks.{layer}.attn.hook_z"
    hooks = [(hook_name, partial(zero_ablate_head, head_idx=head))]

    abl_gender = mean_bias_on_prompts(model, gender_prompts_eval, male_ids, female_ids, hooks=hooks)
    abl_race = mean_bias_on_prompts(model, race_prompts_eval,
                                      RACE_TOKEN_SETS["white"], RACE_TOKEN_SETS["black"], hooks=hooks)
    abl_religion = mean_bias_on_prompts(model, religion_prompts_eval,
                                          RELIGION_TOKEN_SETS["christian"], RELIGION_TOKEN_SETS["muslim"], hooks=hooks)

    result = {
        "layer": int(layer),
        "head": int(head),
        "gender_bias_change": float(abl_gender - baseline_gender_bias),
        "race_bias_change": float(abl_race - baseline_race_bias),
        "religion_bias_change": float(abl_religion - baseline_religion_bias),
    }
    cross_effects.append(result)

    print(f"  L{layer}H{head}: gender={result['gender_bias_change']:+.6f}, "
          f"race={result['race_bias_change']:+.6f}, religion={result['religion_bias_change']:+.6f}")


# ── Visualizations ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (bias_type, matrix) in enumerate(bias_type_results.items()):
    im = axes[i].imshow(matrix, aspect='auto', cmap='hot')
    axes[i].set_xlabel("Head")
    axes[i].set_ylabel("Layer")
    axes[i].set_title(f"{bias_type.capitalize()} Bias Head Importance")
    plt.colorbar(im, ax=axes[i])

plt.suptitle("Head Importance Across Bias Types (GPT-2 Small)", fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "cross_bias_heatmaps.png", dpi=150)
plt.close()

# Cross-effects bar chart
fig, ax = plt.subplots(figsize=(12, 6))
head_labels = [f"L{r['layer']}H{r['head']}" for r in cross_effects]
x = np.arange(len(head_labels))
width = 0.25

ax.bar(x - width, [r["gender_bias_change"] for r in cross_effects], width, label="Gender bias change", color="blue")
ax.bar(x, [r["race_bias_change"] for r in cross_effects], width, label="Race bias change", color="red")
ax.bar(x + width, [r["religion_bias_change"] for r in cross_effects], width, label="Religion bias change", color="green")

ax.set_xlabel("Ablated Head (top gender-bias heads)")
ax.set_ylabel("Bias Change (negative = reduced)")
ax.set_title("Cross-Bias Side Effects: Ablating Gender-Bias Heads")
ax.set_xticks(x)
ax.set_xticklabels(head_labels, rotation=45)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig(RESULTS_DIR / "cross_bias_side_effects.png", dpi=150)
plt.close()


# ── Save results ──
with open(RESULTS_DIR / "cross_bias_results.json", "w") as f:
    json.dump({
        "bias_type_head_importance": {k: v.tolist() for k, v in bias_type_results.items()},
        "cross_effects": cross_effects,
        "baselines": {
            "gender": float(baseline_gender_bias),
            "race": float(baseline_race_bias),
            "religion": float(baseline_religion_bias),
        },
    }, f, indent=2)

print(f"\nResults saved to {RESULTS_DIR}")
print("\n✓ Experiment 04 complete.")
