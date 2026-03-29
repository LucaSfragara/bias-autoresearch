"""
Experiment 16: BOS Path Patching — Tracing Gender Information to L10H9

L10H9 attends ~63% to the BOS token. This experiment traces how gender-relevant
information accumulates at the BOS position across layers 0-9, explaining the
mechanistic pathway by which L10H9 implements gender bias.

1. Layer-by-layer residual stream at BOS: what does each layer write?
2. Activation patching at BOS: which layers' BOS contributions matter for bias?
3. Gender direction projection: project each layer's BOS output onto gender subspace
4. Causal path: patch BOS residual at each layer between male/female contexts
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

RESULTS_DIR = Path("results/16_bos_path")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2", device=device)
print("Loaded GPT-2: %d layers, %d heads" % (model.cfg.n_layers, model.cfg.n_heads))

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

# Gender tokens
male_words = [" he", " him", " his", " himself", " man", " boy", " male", " father",
              " son", " brother", " husband", " king"]
female_words = [" she", " her", " hers", " herself", " woman", " girl", " female", " mother",
                " daughter", " sister", " wife", " queen"]

male_ids = []
female_ids = []
for w in male_words:
    toks = model.to_tokens(w, prepend_bos=False).squeeze()
    male_ids.append(toks.item() if toks.dim() == 0 else toks[0].item())
for w in female_words:
    toks = model.to_tokens(w, prepend_bos=False).squeeze()
    female_ids.append(toks.item() if toks.dim() == 0 else toks[0].item())


# ── Prompts ──
OCCUPATIONS = [
    "nurse", "doctor", "engineer", "secretary", "CEO", "teacher",
    "programmer", "homemaker", "mechanic", "receptionist", "pilot",
    "librarian", "surgeon", "nanny", "janitor", "hairdresser",
    "plumber", "dancer", "firefighter", "babysitter", "accountant",
    "lawyer", "chef", "farmer", "dentist", "professor",
]

TEMPLATES = [
    "The %s said that",
    "The %s walked to the",
    "The %s announced that",
    "The %s explained that",
    "The %s mentioned that",
]

bias_prompts = []
for occ in OCCUPATIONS:
    for tmpl in TEMPLATES:
        bias_prompts.append(tmpl % occ)

# Paired prompts for patching (explicitly gendered vs neutral)
MALE_PROMPTS = [
    "The man said that",
    "The boy walked to the",
    "The father announced that",
    "The husband explained that",
    "The king mentioned that",
    "He said that",
    "The brother walked to the",
    "The son announced that",
]

FEMALE_PROMPTS = [
    "The woman said that",
    "The girl walked to the",
    "The mother announced that",
    "The wife explained that",
    "The queen mentioned that",
    "She said that",
    "The sister walked to the",
    "The daughter announced that",
]


def gender_bias_score(logits):
    """Signed: positive = male-biased"""
    probs = torch.softmax(logits, dim=-1)
    pm = probs[male_ids].sum().item()
    pf = probs[female_ids].sum().item()
    return pm - pf


def eval_bias(model, prompts, hooks=None):
    scores = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
        scores.append(abs(gender_bias_score(logits[0, -1, :])))
    return np.mean(scores)


# ═══════════════════════════════════════════════
# 1. CACHE RESIDUAL STREAM AT BOS ACROSS LAYERS
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("1. RESIDUAL STREAM DECOMPOSITION AT BOS POSITION")
print("="*70)

# For a set of bias prompts, cache what each layer writes to BOS
sample_prompts = bias_prompts[:30]

# Collect per-layer contributions at BOS position
layer_bos_contributions = {layer: [] for layer in range(n_layers)}
mlp_bos_contributions = {layer: [] for layer in range(n_layers)}
attn_bos_contributions = {layer: [] for layer in range(n_layers)}

for prompt in tqdm(sample_prompts, desc="Caching BOS residuals"):
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    for layer in range(n_layers):
        # Attention output at BOS position (pos 0)
        attn_out = cache["blocks.%d.hook_attn_out" % layer][0, 0, :]  # [d_model]
        mlp_out = cache["blocks.%d.hook_mlp_out" % layer][0, 0, :]    # [d_model]
        attn_bos_contributions[layer].append(attn_out.cpu())
        mlp_bos_contributions[layer].append(mlp_out.cpu())
        layer_bos_contributions[layer].append((attn_out + mlp_out).cpu())

# Compute gender direction from the unembedding
# Gender direction = mean(male_unembed) - mean(female_unembed)
W_U = model.W_U  # [d_model, vocab]
male_unembed = W_U[:, male_ids].mean(dim=1)   # [d_model]
female_unembed = W_U[:, female_ids].mean(dim=1)  # [d_model]
gender_dir = (male_unembed - female_unembed)
gender_dir = gender_dir / gender_dir.norm()
gender_dir_cpu = gender_dir.cpu()

print("\nGender direction norm: %.4f" % (male_unembed - female_unembed).norm().item())

# Project each layer's BOS contribution onto gender direction
print("\nLayer-by-layer gender projection at BOS (mean across %d prompts):" % len(sample_prompts))
print("%-8s %12s %12s %12s %12s" % ("Layer", "Total", "Attn", "MLP", "Cumulative"))
print("-" * 60)

layer_gender_proj = []
attn_gender_proj = []
mlp_gender_proj = []
cumulative = 0

for layer in range(n_layers):
    total_projs = [torch.dot(c, gender_dir_cpu).item() for c in layer_bos_contributions[layer]]
    attn_projs = [torch.dot(c, gender_dir_cpu).item() for c in attn_bos_contributions[layer]]
    mlp_projs = [torch.dot(c, gender_dir_cpu).item() for c in mlp_bos_contributions[layer]]

    mean_total = np.mean(total_projs)
    mean_attn = np.mean(attn_projs)
    mean_mlp = np.mean(mlp_projs)
    cumulative += mean_total

    layer_gender_proj.append(mean_total)
    attn_gender_proj.append(mean_attn)
    mlp_gender_proj.append(mean_mlp)

    print("  L%-5d %+11.4f %+11.4f %+11.4f %+11.4f" % (
        layer, mean_total, mean_attn, mean_mlp, cumulative))

# Also check the embedding layer
embed_projs = []
for prompt in sample_prompts:
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    embed = cache["hook_embed"][0, 0, :] + cache["hook_pos_embed"][0, 0, :]
    embed_projs.append(torch.dot(embed.cpu(), gender_dir_cpu).item())
print("\n  Embed  %+11.4f (token + positional embedding at BOS)" % np.mean(embed_projs))


# ═══════════════════════════════════════════════
# 2. ACTIVATION PATCHING AT BOS
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("2. ACTIVATION PATCHING AT BOS (male→female context)")
print("="*70)

# For each layer, patch the BOS residual from a male prompt into a female prompt
# and measure how much bias changes. This shows which layers' BOS info matters.

def patch_bos_residual(resid, hook, clean_bos, layer_idx):
    """Replace BOS position residual with the clean (male) version"""
    resid[:, 0, :] = clean_bos[layer_idx]
    return resid

# Use matched pairs
n_pairs = min(len(MALE_PROMPTS), len(FEMALE_PROMPTS))

patching_results = {}
for patch_layer in tqdm(range(n_layers), desc="Patching BOS"):
    bias_changes = []

    for i in range(n_pairs):
        male_tokens = model.to_tokens(MALE_PROMPTS[i])
        female_tokens = model.to_tokens(FEMALE_PROMPTS[i])

        # Get male BOS residual at this layer
        with torch.no_grad():
            _, male_cache = model.run_with_cache(male_tokens)

        male_bos = {}
        for l in range(n_layers):
            male_bos[l] = male_cache["blocks.%d.hook_resid_pre" % l][0, 0, :].clone()

        # Run female prompt clean
        with torch.no_grad():
            female_logits = model(female_tokens)
        clean_bias = gender_bias_score(female_logits[0, -1, :])

        # Run female prompt with male BOS patched at this layer
        hooks = [("blocks.%d.hook_resid_pre" % patch_layer,
                  partial(patch_bos_residual, clean_bos=male_bos, layer_idx=patch_layer))]
        with torch.no_grad():
            patched_logits = model.run_with_hooks(female_tokens, fwd_hooks=hooks)
        patched_bias = gender_bias_score(patched_logits[0, -1, :])

        # How much did patching BOS shift bias toward male?
        bias_changes.append(patched_bias - clean_bias)

    mean_shift = np.mean(bias_changes)
    std_shift = np.std(bias_changes)
    patching_results[patch_layer] = {
        "mean_bias_shift": float(mean_shift),
        "std": float(std_shift),
    }

print("\nBOS patching results (male→female, positive = more male after patch):")
print("%-8s %15s %10s %s" % ("Layer", "Mean Bias Shift", "Std", ""))
print("-" * 55)
for layer in range(n_layers):
    r = patching_results[layer]
    bar = "+" * int(max(0, r["mean_bias_shift"]) * 200)
    bar_neg = "-" * int(max(0, -r["mean_bias_shift"]) * 200)
    print("  L%-5d %+14.6f %9.6f %s%s" % (
        layer, r["mean_bias_shift"], r["std"], bar, bar_neg))


# ═══════════════════════════════════════════════
# 3. HEAD-LEVEL BOS PATCHING IN KEY LAYERS
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("3. HEAD-LEVEL BOS PATCHING (which heads write gender info to BOS?)")
print("="*70)

# For layers that showed high patching effect, drill into individual heads
# We patch individual head outputs at BOS position

def patch_head_bos(z, hook, head_idx, clean_head_bos):
    """Replace a single head's output at BOS with the male version"""
    z[:, 0, head_idx, :] = clean_head_bos
    return z

# Find top patching layers
sorted_layers = sorted(patching_results.items(), key=lambda x: abs(x[1]["mean_bias_shift"]), reverse=True)
top_layers = [l for l, _ in sorted_layers[:5]]
print("Drilling into top-5 patching layers: %s" % top_layers)

head_patching_results = {}
for layer in tqdm(top_layers, desc="Head-level BOS patching"):
    for head in range(n_heads):
        bias_changes = []

        for i in range(n_pairs):
            male_tokens = model.to_tokens(MALE_PROMPTS[i])
            female_tokens = model.to_tokens(FEMALE_PROMPTS[i])

            with torch.no_grad():
                _, male_cache = model.run_with_cache(male_tokens)

            male_head_bos = male_cache["blocks.%d.attn.hook_z" % layer][0, 0, head, :].clone()

            with torch.no_grad():
                female_logits = model(female_tokens)
            clean_bias = gender_bias_score(female_logits[0, -1, :])

            hooks = [("blocks.%d.attn.hook_z" % layer,
                      partial(patch_head_bos, head_idx=head, clean_head_bos=male_head_bos))]
            with torch.no_grad():
                patched_logits = model.run_with_hooks(female_tokens, fwd_hooks=hooks)
            patched_bias = gender_bias_score(patched_logits[0, -1, :])

            bias_changes.append(patched_bias - clean_bias)

        key = "L%dH%d" % (layer, head)
        head_patching_results[key] = {
            "layer": layer, "head": head,
            "mean_bias_shift": float(np.mean(bias_changes)),
            "std": float(np.std(bias_changes)),
        }

# Show top heads
sorted_heads = sorted(head_patching_results.items(), key=lambda x: abs(x[1]["mean_bias_shift"]), reverse=True)
print("\nTop 15 heads by BOS patching effect:")
print("%-10s %15s %10s" % ("Head", "Bias Shift", "Std"))
print("-" * 40)
for name, r in sorted_heads[:15]:
    print("%-10s %+14.6f %9.6f" % (name, r["mean_bias_shift"], r["std"]))


# ═══════════════════════════════════════════════
# 4. L10H9 ATTENTION DECOMPOSITION
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("4. L10H9 ATTENTION SOURCE DECOMPOSITION")
print("="*70)

# What information does L10H9 actually read from each position?
# Decompose L10H9's output into contributions from each source position

decomp_prompts = bias_prompts[:20]

per_position_contributions = []  # list of dicts: {pos: gender_proj}

for prompt in tqdm(decomp_prompts, desc="L10H9 decomposition"):
    tokens = model.to_tokens(prompt)
    n_tok = tokens.shape[1]

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    # L10H9 attention pattern at last position
    attn = cache["blocks.10.attn.hook_pattern"][0, 9, -1, :]  # [n_tok]

    # Value vectors at each position
    v = cache["blocks.10.attn.hook_v"][0, :, 9, :]  # [n_tok, d_head]

    # OV matrix for head 9
    W_O = model.blocks[10].attn.W_O[9]  # [d_head, d_model]

    # Per-position contribution to output
    contributions = {}
    for pos in range(n_tok):
        # What this position contributes: attn_weight * v * W_O
        contrib = attn[pos] * v[pos] @ W_O  # [d_model]
        gender_proj = torch.dot(contrib.cpu(), gender_dir_cpu).item()
        contributions[pos] = gender_proj

    per_position_contributions.append(contributions)

# Average across prompts (positions: 0=BOS, 1=The, 2=occupation, 3=said, 4=that)
print("\nL10H9 output decomposition by source position (gender projection):")
print("%-10s %12s %12s" % ("Position", "Gender Proj", "Description"))
print("-" * 40)

max_pos = min(6, min(len(c) for c in per_position_contributions))
pos_labels = ["BOS", "The", "occupation", "verb", "that/the", "pos5"]
for pos in range(max_pos):
    projs = [c[pos] for c in per_position_contributions if pos in c]
    mean_proj = np.mean(projs)
    label = pos_labels[pos] if pos < len(pos_labels) else "pos%d" % pos
    bar = "M" * int(max(0, mean_proj) * 500) + "F" * int(max(0, -mean_proj) * 500)
    print("  %-8s %+11.6f  %s %s" % (label, mean_proj, label, bar))


# ═══════════════════════════════════════════════
# 5. WHAT'S SPECIAL ABOUT BOS? COMPARE INFORMATION CONTENT
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("5. BOS AS GENDER-NEUTRAL ANCHOR vs INFORMATION AGGREGATOR")
print("="*70)

# Hypothesis 1: BOS accumulates gender-stereotypical info from context
# Hypothesis 2: BOS is gender-neutral; L10H9 uses it as a default/anchor
# Test: compare BOS residual gender projection across male/female contexts

male_bos_projs = []
female_bos_projs = []
neutral_bos_projs = []

for prompt in MALE_PROMPTS:
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    # Residual at BOS just before L10
    bos_resid = cache["blocks.10.hook_resid_pre"][0, 0, :]
    male_bos_projs.append(torch.dot(bos_resid.cpu(), gender_dir_cpu).item())

for prompt in FEMALE_PROMPTS:
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    bos_resid = cache["blocks.10.hook_resid_pre"][0, 0, :]
    female_bos_projs.append(torch.dot(bos_resid.cpu(), gender_dir_cpu).item())

neutral_prompts = ["The person said that", "Someone mentioned that",
                   "The individual walked to", "A student said that",
                   "The worker explained that", "The employee stated that",
                   "The professional noted that", "The citizen reported that"]
for prompt in neutral_prompts:
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    bos_resid = cache["blocks.10.hook_resid_pre"][0, 0, :]
    neutral_bos_projs.append(torch.dot(bos_resid.cpu(), gender_dir_cpu).item())

print("BOS residual at L10 entry — gender direction projection:")
print("  Male contexts:    mean=%+.6f (std=%.6f)" % (np.mean(male_bos_projs), np.std(male_bos_projs)))
print("  Female contexts:  mean=%+.6f (std=%.6f)" % (np.mean(female_bos_projs), np.std(female_bos_projs)))
print("  Neutral contexts: mean=%+.6f (std=%.6f)" % (np.mean(neutral_bos_projs), np.std(neutral_bos_projs)))

separation = np.mean(male_bos_projs) - np.mean(female_bos_projs)
print("\n  Male-Female separation at BOS: %+.6f" % separation)
if abs(separation) > 0.5:
    print("  → BOS AGGREGATES gender info from context (Hypothesis 1)")
else:
    print("  → BOS is relatively NEUTRAL — L10H9 uses it as anchor (Hypothesis 2)")

# Also check: does BOS projection correlate with bias in occupation prompts?
occ_bos_projs = []
occ_biases = []
for prompt in bias_prompts[:50]:
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
        logits = model(tokens)
    bos_resid = cache["blocks.10.hook_resid_pre"][0, 0, :]
    bos_proj = torch.dot(bos_resid.cpu(), gender_dir_cpu).item()
    bias = gender_bias_score(logits[0, -1, :])
    occ_bos_projs.append(bos_proj)
    occ_biases.append(bias)

correlation = np.corrcoef(occ_bos_projs, occ_biases)[0, 1]
print("\n  Correlation(BOS gender proj, output bias): r=%.4f" % correlation)
if abs(correlation) > 0.5:
    print("  → BOS position carries occupation-specific gender signal!")
else:
    print("  → BOS position is NOT strongly occupation-specific")


# ═══════════════════════════════════════════════
# 6. COUNTERFACTUAL: ZERO BOS ATTENTION IN L10H9
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("6. COUNTERFACTUAL: SUPPRESS L10H9's BOS ATTENTION")
print("="*70)

# Instead of ablating the entire head, just zero out its attention to BOS
# This isolates BOS's role specifically

def zero_bos_attention(pattern, hook, head_idx):
    """Zero out attention to BOS for a specific head, renormalize"""
    # pattern: [batch, n_heads, q_pos, k_pos]
    bos_attn = pattern[:, head_idx, :, 0].clone()
    pattern[:, head_idx, :, 0] = 0
    # Renormalize remaining attention
    remaining = pattern[:, head_idx, :, 1:].sum(dim=-1, keepdim=True)
    remaining = remaining.clamp(min=1e-8)
    pattern[:, head_idx, :, 1:] = pattern[:, head_idx, :, 1:] / remaining * (bos_attn.unsqueeze(-1) + remaining)
    # Actually just renormalize all
    pattern[:, head_idx, :, :] = pattern[:, head_idx, :, :] / pattern[:, head_idx, :, :].sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return pattern

eval_prompts = bias_prompts[:50]
baseline_bias = eval_bias(model, eval_prompts)

hooks_zero_bos = [("blocks.10.attn.hook_pattern",
                    partial(zero_bos_attention, head_idx=9))]
zero_bos_bias = eval_bias(model, eval_prompts, hooks_zero_bos)

# Compare with full head ablation
def scale_head(z, hook, head_idx, alpha):
    z[:, :, head_idx, :] *= alpha
    return z

hooks_full_ablate = [("blocks.10.attn.hook_z",
                       partial(scale_head, head_idx=9, alpha=0.0))]
full_ablate_bias = eval_bias(model, eval_prompts, hooks_full_ablate)

print("Baseline bias:              %.6f" % baseline_bias)
print("Zero BOS attn in L10H9:     %.6f (%.1f%% reduction)" % (
    zero_bos_bias, (baseline_bias - zero_bos_bias) / baseline_bias * 100))
print("Full L10H9 ablation:        %.6f (%.1f%% reduction)" % (
    full_ablate_bias, (baseline_bias - full_ablate_bias) / baseline_bias * 100))

bos_fraction = (baseline_bias - zero_bos_bias) / max(1e-8, baseline_bias - full_ablate_bias) * 100
print("\nBOS accounts for %.1f%% of L10H9's total bias effect" % bos_fraction)


# ═══════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Gender projection by layer at BOS
ax = axes[0, 0]
x = range(n_layers)
ax.bar(x, layer_gender_proj, color='steelblue', alpha=0.7, label='Total')
ax.bar(x, attn_gender_proj, color='coral', alpha=0.5, width=0.4, align='edge', label='Attn')
ax.bar(x, mlp_gender_proj, color='green', alpha=0.5, width=-0.4, align='edge', label='MLP')
ax.set_xlabel("Layer")
ax.set_ylabel("Gender Direction Projection")
ax.set_title("Per-Layer Gender Info Written to BOS")
ax.legend(fontsize=8)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

# Plot 2: BOS patching effect by layer
ax = axes[0, 1]
shifts = [patching_results[l]["mean_bias_shift"] for l in range(n_layers)]
stds = [patching_results[l]["std"] for l in range(n_layers)]
colors = ['red' if abs(s) > 0.01 else 'steelblue' for s in shifts]
ax.bar(range(n_layers), shifts, color=colors, alpha=0.7)
ax.errorbar(range(n_layers), shifts, yerr=stds, fmt='none', color='black', capsize=2)
ax.set_xlabel("Layer")
ax.set_ylabel("Bias Shift (male→female patch)")
ax.set_title("BOS Activation Patching by Layer")
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

# Plot 3: BOS gender projection by context type
ax = axes[1, 0]
categories = ['Male\ncontext', 'Female\ncontext', 'Neutral\ncontext']
means = [np.mean(male_bos_projs), np.mean(female_bos_projs), np.mean(neutral_bos_projs)]
stds_ctx = [np.std(male_bos_projs), np.std(female_bos_projs), np.std(neutral_bos_projs)]
colors_ctx = ['royalblue', 'hotpink', 'gray']
ax.bar(categories, means, color=colors_ctx, alpha=0.7)
ax.errorbar(categories, means, yerr=stds_ctx, fmt='none', color='black', capsize=5)
ax.set_ylabel("Gender Direction Projection at BOS (L10)")
ax.set_title("BOS Gender Content by Context")
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

# Plot 4: BOS projection vs output bias (scatter)
ax = axes[1, 1]
ax.scatter(occ_bos_projs, occ_biases, alpha=0.5, s=20, color='steelblue')
ax.set_xlabel("BOS Gender Projection (L10 entry)")
ax.set_ylabel("Output Gender Bias (P(M)-P(F))")
ax.set_title("BOS Info vs Output Bias (r=%.3f)" % correlation)
# Fit line
if len(occ_bos_projs) > 2:
    z = np.polyfit(occ_bos_projs, occ_biases, 1)
    p = np.poly1d(z)
    xs = np.linspace(min(occ_bos_projs), max(occ_bos_projs), 100)
    ax.plot(xs, p(xs), 'r-', alpha=0.5)

plt.suptitle("BOS Path Analysis: How Gender Info Reaches L10H9", fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "bos_path_analysis.png", dpi=150)
plt.close()
print("\nPlots saved to %s" % RESULTS_DIR)


# ── Save results ──
results = {
    "layer_gender_projection": {
        "total": layer_gender_proj,
        "attn": attn_gender_proj,
        "mlp": mlp_gender_proj,
    },
    "embed_gender_projection": float(np.mean(embed_projs)),
    "bos_patching_by_layer": {str(k): v for k, v in patching_results.items()},
    "head_patching_top15": {name: r for name, r in sorted_heads[:15]},
    "bos_context_comparison": {
        "male_mean": float(np.mean(male_bos_projs)),
        "female_mean": float(np.mean(female_bos_projs)),
        "neutral_mean": float(np.mean(neutral_bos_projs)),
        "male_female_separation": float(separation),
    },
    "bos_bias_correlation": float(correlation),
    "counterfactual": {
        "baseline_bias": float(baseline_bias),
        "zero_bos_attn_bias": float(zero_bos_bias),
        "zero_bos_reduction_pct": float((baseline_bias - zero_bos_bias) / baseline_bias * 100),
        "full_ablation_bias": float(full_ablate_bias),
        "full_ablation_reduction_pct": float((baseline_bias - full_ablate_bias) / baseline_bias * 100),
        "bos_fraction_of_total": float(bos_fraction),
    },
}

with open(RESULTS_DIR / "bos_path_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("1. Layer gender projection: identifies which layers write gender info to BOS")
print("2. BOS patching: shows causal effect of each layer's BOS content on bias")
print("3. Context comparison: tests whether BOS aggregates context or stays neutral")
print("4. Counterfactual: %.1f%% of L10H9's bias comes from BOS attention" % bos_fraction)
print("\n✓ Experiment 16 complete.")
