"""
Experiment 03: Entanglement Analysis (THE CORE CONTRIBUTION)
- For each bias-relevant head, ablate it and measure:
  1. Bias reduction (gender bias score)
  2. Perplexity change (general capability)
  3. Gendered pronoun resolution (syntactic capability)
  4. Subject-verb agreement (grammatical capability)
- Compute selectivity index: bias_reduction / capability_degradation
- Classify heads as "separable" vs "entangled"
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

RESULTS_DIR = Path("results/03_entanglement")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading GPT-2 small...")
model = HookedTransformer.from_pretrained("gpt2", device=device)
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

with open("results/00_setup/gender_token_ids.json") as f:
    gender_token_ids = json.load(f)
male_ids = gender_token_ids["male"]
female_ids = gender_token_ids["female"]

# Load head importance from experiment 01
with open("results/01_activation_patching/patching_results.json") as f:
    patching_results = json.load(f)

top_heads = patching_results["top_heads_zero"]
print(f"Top heads from patching: {[(h['layer'], h['head']) for h in top_heads[:10]]}")


# ── Ablation hook ──
def zero_ablate_head(z, hook, head_idx):
    """Zero out a single attention head's z (pre-projection output).
    z shape: (batch, pos, n_heads, d_head)
    """
    z[:, :, head_idx, :] = 0.0
    return z


# ── 1. Bias evaluation under ablation ──
print("\n" + "="*60)
print("1. BIAS SCORE UNDER HEAD ABLATION")
print("="*60)

BIAS_PROMPTS = [
    "The nurse said that",
    "The doctor said that",
    "The engineer said that",
    "The secretary said that",
    "The CEO said that",
    "The teacher said that",
    "The programmer said that",
    "The homemaker said that",
    "The mechanic fixed the car and",
    "The receptionist answered the phone and",
    "The pilot announced that",
    "The librarian recommended that",
    "The construction worker said that",
    "The nanny mentioned that",
    "The surgeon explained that",
    "The beautician suggested that",
]


def compute_mean_bias(model, prompts, hooks=None):
    """Compute mean absolute gender bias across prompts."""
    scores = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            if hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                logits = model(tokens)
        probs = torch.softmax(logits[0, -1, :], dim=0)
        male_p = probs[male_ids].sum().item()
        female_p = probs[female_ids].sum().item()
        scores.append(abs(male_p - female_p))
    return np.mean(scores), np.std(scores), scores


# Baseline bias
baseline_bias, baseline_bias_std, baseline_bias_scores = compute_mean_bias(model, BIAS_PROMPTS)
print(f"Baseline mean |bias|: {baseline_bias:.6f} (std: {baseline_bias_std:.6f})")


# ── 2. Perplexity evaluation ──
print("\n" + "="*60)
print("2. PERPLEXITY UNDER HEAD ABLATION")
print("="*60)

# Use a set of general English sentences for perplexity
PPL_SENTENCES = [
    "The cat sat on the mat and looked out the window at the birds.",
    "In the beginning, there was nothing but darkness and silence.",
    "The stock market experienced significant volatility throughout the trading session.",
    "She walked through the garden, admiring the flowers that bloomed in spring.",
    "The committee decided to postpone the meeting until further notice.",
    "Scientists have discovered a new species of deep-sea fish near the thermal vents.",
    "The children played in the park while their parents watched from the bench.",
    "According to the latest report, unemployment rates have decreased significantly.",
    "The old library contained thousands of books dating back centuries.",
    "After careful consideration, the board approved the new budget proposal.",
    "The river flowed gently through the valley, reflecting the morning sunlight.",
    "Technology continues to reshape how we communicate and interact with each other.",
    "The musician performed a beautiful solo that captivated the entire audience.",
    "Research suggests that regular exercise can improve both physical and mental health.",
    "The detective examined the evidence carefully before drawing any conclusions.",
    "Several factors contributed to the decline in agricultural productivity last year.",
    "The architect designed a building that harmonized with the surrounding landscape.",
    "Despite the challenges, the team managed to complete the project on time.",
    "The professor explained the complex theory in simple, accessible terms.",
    "Climate change poses significant threats to biodiversity across the globe.",
]


def compute_perplexity(model, sentences, hooks=None):
    """Compute mean perplexity across sentences."""
    total_loss = 0
    total_tokens = 0

    for sent in sentences:
        tokens = model.to_tokens(sent)
        with torch.no_grad():
            if hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                logits = model(tokens)

        # Cross-entropy loss
        log_probs = torch.log_softmax(logits[0, :-1, :], dim=-1)
        targets = tokens[0, 1:]
        token_losses = -log_probs.gather(1, targets.unsqueeze(1)).squeeze()
        total_loss += token_losses.sum().item()
        total_tokens += len(targets)

    avg_loss = total_loss / total_tokens
    ppl = np.exp(avg_loss)
    return ppl


baseline_ppl = compute_perplexity(model, PPL_SENTENCES)
print(f"Baseline perplexity: {baseline_ppl:.2f}")


# ── 3. Pronoun resolution evaluation ──
print("\n" + "="*60)
print("3. PRONOUN RESOLUTION UNDER HEAD ABLATION")
print("="*60)

# Test: does the model correctly assign higher probability to the right pronoun
# given an explicitly gendered antecedent?
PRONOUN_RESOLUTION_TESTS = [
    # (prompt, correct_pronoun_token, incorrect_pronoun_token)
    ("The man went to the store. When he arrived,", " he", " she"),
    ("The woman went to the store. When she arrived,", " she", " he"),
    ("The boy played in the yard. Then", " he", " she"),
    ("The girl played in the yard. Then", " she", " he"),
    ("My father always told me that", " he", " she"),
    ("My mother always told me that", " she", " he"),
    ("The king ruled the land. Everyone respected", " him", " her"),
    ("The queen ruled the land. Everyone respected", " her", " him"),
    ("John walked to work.", " He", " She"),
    ("Mary walked to work.", " She", " He"),
    ("The husband cooked dinner for", " his", " her"),
    ("The wife cooked dinner for", " her", " his"),
    ("My grandfather used to say that", " he", " she"),
    ("My grandmother used to say that", " she", " he"),
    ("The prince fought bravely.", " He", " She"),
    ("The princess fought bravely.", " She", " He"),
]


def compute_pronoun_accuracy(model, tests, hooks=None):
    """Compute accuracy on pronoun resolution task."""
    correct = 0
    total = 0
    margins = []

    for prompt, correct_tok, incorrect_tok in tests:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            if hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                logits = model(tokens)

        probs = torch.softmax(logits[0, -1, :], dim=0)

        correct_id = model.to_tokens(correct_tok, prepend_bos=False).squeeze()
        incorrect_id = model.to_tokens(incorrect_tok, prepend_bos=False).squeeze()

        if correct_id.dim() > 0:
            correct_id = correct_id[0]
        if incorrect_id.dim() > 0:
            incorrect_id = incorrect_id[0]

        p_correct = probs[correct_id].item()
        p_incorrect = probs[incorrect_id].item()

        if p_correct > p_incorrect:
            correct += 1
        margins.append(p_correct - p_incorrect)
        total += 1

    return correct / total, np.mean(margins)


baseline_pronoun_acc, baseline_pronoun_margin = compute_pronoun_accuracy(model, PRONOUN_RESOLUTION_TESTS)
print(f"Baseline pronoun resolution accuracy: {baseline_pronoun_acc:.2%}")
print(f"Baseline pronoun margin: {baseline_pronoun_margin:.4f}")


# ── 4. Subject-verb agreement evaluation ──
print("\n" + "="*60)
print("4. SUBJECT-VERB AGREEMENT UNDER HEAD ABLATION")
print("="*60)

# Test: does the model prefer grammatically correct verb forms?
AGREEMENT_TESTS = [
    # (prompt, correct_token, incorrect_token)
    ("The cat", " is", " are"),
    ("The cats", " are", " is"),
    ("The dog", " runs", " run"),
    ("The dogs", " run", " runs"),
    ("The child", " was", " were"),
    ("The children", " were", " was"),
    ("The man", " is", " are"),
    ("The men", " are", " is"),
    ("The woman", " has", " have"),
    ("The women", " have", " has"),
    ("The boy who lives next door", " is", " are"),
    ("The boys who live next door", " are", " is"),
    ("The student in the classroom", " works", " work"),
    ("The students in the classroom", " work", " works"),
    ("The bird on the branch", " sings", " sing"),
    ("The birds on the branch", " sing", " sings"),
]


def compute_agreement_accuracy(model, tests, hooks=None):
    """Compute accuracy on subject-verb agreement task."""
    correct = 0
    total = 0

    for prompt, correct_tok, incorrect_tok in tests:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            if hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                logits = model(tokens)

        probs = torch.softmax(logits[0, -1, :], dim=0)

        correct_id = model.to_tokens(correct_tok, prepend_bos=False).squeeze()
        incorrect_id = model.to_tokens(incorrect_tok, prepend_bos=False).squeeze()

        if correct_id.dim() > 0:
            correct_id = correct_id[0]
        if incorrect_id.dim() > 0:
            incorrect_id = incorrect_id[0]

        if probs[correct_id].item() > probs[incorrect_id].item():
            correct += 1
        total += 1

    return correct / total


baseline_agreement = compute_agreement_accuracy(model, AGREEMENT_TESTS)
print(f"Baseline agreement accuracy: {baseline_agreement:.2%}")


# ── 5. MAIN LOOP: Ablate each head and measure everything ──
print("\n" + "="*60)
print("5. COMPREHENSIVE ABLATION SWEEP")
print("="*60)

# We test ALL heads, not just top ones, to get the full picture
ablation_results = {}

for layer in tqdm(range(n_layers), desc="Ablation sweep (layers)"):
    for head in range(n_heads):
        hook_name = f"blocks.{layer}.attn.hook_z"
        hooks = [(hook_name, partial(zero_ablate_head, head_idx=head))]

        # Bias
        abl_bias, abl_bias_std, _ = compute_mean_bias(model, BIAS_PROMPTS, hooks=hooks)

        # Perplexity
        abl_ppl = compute_perplexity(model, PPL_SENTENCES, hooks=hooks)

        # Pronoun resolution
        abl_pronoun, abl_pronoun_margin = compute_pronoun_accuracy(model, PRONOUN_RESOLUTION_TESTS, hooks=hooks)

        # Agreement
        abl_agreement = compute_agreement_accuracy(model, AGREEMENT_TESTS, hooks=hooks)

        # Compute deltas
        bias_reduction = baseline_bias - abl_bias  # positive = bias reduced
        ppl_increase = abl_ppl - baseline_ppl  # positive = capability degraded
        pronoun_drop = baseline_pronoun_acc - abl_pronoun  # positive = capability degraded
        agreement_drop = baseline_agreement - abl_agreement  # positive = capability degraded

        # Selectivity index: bias_reduction / max(capability_degradation, epsilon)
        # Higher = more separable (can remove bias without hurting capabilities)
        capability_loss = max(
            ppl_increase / baseline_ppl,  # normalize by baseline
            pronoun_drop,
            agreement_drop,
            1e-6  # avoid division by zero
        )
        selectivity = bias_reduction / capability_loss if bias_reduction > 0 else 0.0

        ablation_results[f"L{layer}H{head}"] = {
            "layer": layer,
            "head": head,
            "bias_original": float(baseline_bias),
            "bias_ablated": float(abl_bias),
            "bias_reduction": float(bias_reduction),
            "ppl_original": float(baseline_ppl),
            "ppl_ablated": float(abl_ppl),
            "ppl_increase": float(ppl_increase),
            "ppl_increase_pct": float(ppl_increase / baseline_ppl * 100),
            "pronoun_acc_original": float(baseline_pronoun_acc),
            "pronoun_acc_ablated": float(abl_pronoun),
            "pronoun_drop": float(pronoun_drop),
            "agreement_original": float(baseline_agreement),
            "agreement_ablated": float(abl_agreement),
            "agreement_drop": float(agreement_drop),
            "selectivity": float(selectivity),
        }

# ── Analysis ──
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

# Sort by bias reduction
sorted_by_bias = sorted(ablation_results.items(), key=lambda x: x[1]["bias_reduction"], reverse=True)

print("\nTop 15 heads by BIAS REDUCTION:")
print(f"{'Head':<10} {'Bias Red.':>10} {'PPL +%':>8} {'Pron Drop':>10} {'Agr Drop':>10} {'Select.':>10} {'Type':>12}")
print("-" * 75)
for name, r in sorted_by_bias[:15]:
    # Classify
    if r["bias_reduction"] > 0.001:
        if r["ppl_increase_pct"] < 2 and r["pronoun_drop"] < 0.05 and r["agreement_drop"] < 0.05:
            htype = "SEPARABLE"
        elif r["pronoun_drop"] > 0.1:
            htype = "ENTANGLED-syn"
        elif r["ppl_increase_pct"] > 5:
            htype = "ENTANGLED-ppl"
        elif r["agreement_drop"] > 0.1:
            htype = "ENTANGLED-agr"
        else:
            htype = "MIXED"
    else:
        htype = "NO-EFFECT"

    print(f"{name:<10} {r['bias_reduction']:>10.6f} {r['ppl_increase_pct']:>7.2f}% "
          f"{r['pronoun_drop']:>10.4f} {r['agreement_drop']:>10.4f} "
          f"{r['selectivity']:>10.4f} {htype:>12}")

# Find the "best" head to ablate: max bias reduction with min capability loss
print("\nTop 10 heads by SELECTIVITY (bias reduction / capability loss):")
sorted_by_selectivity = sorted(
    [(n, r) for n, r in ablation_results.items() if r["bias_reduction"] > 0.0005],
    key=lambda x: x[1]["selectivity"], reverse=True
)
for name, r in sorted_by_selectivity[:10]:
    print(f"  {name}: selectivity={r['selectivity']:.4f}, "
          f"bias_red={r['bias_reduction']:.6f}, ppl+={r['ppl_increase_pct']:.2f}%")


# ── Visualizations ──
# 1. Scatter: bias reduction vs capability loss
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

bias_reds = [r["bias_reduction"] for r in ablation_results.values()]
ppl_incs = [r["ppl_increase_pct"] for r in ablation_results.values()]
pronoun_drops = [r["pronoun_drop"] for r in ablation_results.values()]
agreement_drops = [r["agreement_drop"] for r in ablation_results.values()]
layers = [r["layer"] for r in ablation_results.values()]

sc = axes[0].scatter(bias_reds, ppl_incs, c=layers, cmap='viridis', alpha=0.7, s=30)
axes[0].set_xlabel("Bias Reduction")
axes[0].set_ylabel("Perplexity Increase (%)")
axes[0].set_title("Bias Reduction vs Perplexity Cost")
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.3)
plt.colorbar(sc, ax=axes[0], label="Layer")

# Annotate top heads
for name, r in sorted_by_bias[:5]:
    axes[0].annotate(name, (r["bias_reduction"], r["ppl_increase_pct"]), fontsize=7)

sc = axes[1].scatter(bias_reds, pronoun_drops, c=layers, cmap='viridis', alpha=0.7, s=30)
axes[1].set_xlabel("Bias Reduction")
axes[1].set_ylabel("Pronoun Resolution Drop")
axes[1].set_title("Bias Reduction vs Pronoun Resolution Cost")
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.3)
plt.colorbar(sc, ax=axes[1], label="Layer")

for name, r in sorted_by_bias[:5]:
    axes[1].annotate(name, (r["bias_reduction"], r["pronoun_drop"]), fontsize=7)

sc = axes[2].scatter(bias_reds, agreement_drops, c=layers, cmap='viridis', alpha=0.7, s=30)
axes[2].set_xlabel("Bias Reduction")
axes[2].set_ylabel("Agreement Accuracy Drop")
axes[2].set_title("Bias Reduction vs Agreement Cost")
axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
axes[2].axvline(x=0, color='gray', linestyle='--', alpha=0.3)
plt.colorbar(sc, ax=axes[2], label="Layer")

for name, r in sorted_by_bias[:5]:
    axes[2].annotate(name, (r["bias_reduction"], r["agreement_drop"]), fontsize=7)

plt.suptitle("Entanglement Analysis: Bias Reduction vs Capability Costs", fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "entanglement_scatter.png", dpi=150)
plt.close()

# 2. Heatmaps
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

bias_red_matrix = np.zeros((n_layers, n_heads))
ppl_inc_matrix = np.zeros((n_layers, n_heads))
pronoun_matrix = np.zeros((n_layers, n_heads))
selectivity_matrix = np.zeros((n_layers, n_heads))

for name, r in ablation_results.items():
    l, h = r["layer"], r["head"]
    bias_red_matrix[l, h] = r["bias_reduction"]
    ppl_inc_matrix[l, h] = r["ppl_increase_pct"]
    pronoun_matrix[l, h] = r["pronoun_drop"]
    selectivity_matrix[l, h] = r["selectivity"]

for ax, matrix, title, cmap in [
    (axes[0, 0], bias_red_matrix, "Bias Reduction", "RdBu_r"),
    (axes[0, 1], ppl_inc_matrix, "Perplexity Increase (%)", "Reds"),
    (axes[1, 0], pronoun_matrix, "Pronoun Resolution Drop", "Reds"),
    (axes[1, 1], selectivity_matrix, "Selectivity Index", "Greens"),
]:
    if cmap == "RdBu_r":
        vmax = np.max(np.abs(matrix))
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(matrix, aspect='auto', cmap=cmap)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

plt.suptitle("Head-Level Entanglement Analysis (GPT-2 Small)", fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "entanglement_heatmaps.png", dpi=150)
plt.close()

# ── Save results ──
with open(RESULTS_DIR / "entanglement_results.json", "w") as f:
    json.dump({
        "baselines": {
            "bias": float(baseline_bias),
            "perplexity": float(baseline_ppl),
            "pronoun_accuracy": float(baseline_pronoun_acc),
            "agreement_accuracy": float(baseline_agreement),
        },
        "ablation_results": ablation_results,
    }, f, indent=2)

print(f"\nAll results saved to {RESULTS_DIR}")
print("\n✓ Experiment 03 complete.")
