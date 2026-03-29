"""
Experiment 01: Component Importance via Zero-Ablation + Activation Patching
- Zero-ablate each layer/head/MLP and measure change in gender bias score
- Also use mean-ablation (replace with mean activation across dataset) for comparison
- Produces heatmaps of component importance for bias

Two complementary approaches:
1. Zero ablation: zero out component, measure bias change
   → Tells us which components are NECESSARY for bias
2. Activation patching with mean-ablation: replace component with its mean
   → More controlled, avoids destroying information flow
"""

import torch
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

RESULTS_DIR = Path("results/01_activation_patching")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading GPT-2 small...")
model = HookedTransformer.from_pretrained("gpt2", device=device)
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
d_model = model.cfg.d_model
d_head = model.cfg.d_head
print(f"Model: {n_layers} layers, {n_heads} heads, d_model={d_model}, d_head={d_head}")

# Verify hook names
print("Available hook types (layer 0):")
test_tokens = model.to_tokens("test")
with torch.no_grad():
    _, test_cache = model.run_with_cache(test_tokens)
for k in sorted(test_cache.keys()):
    if "0" in k and ("attn" in k or "mlp" in k or "resid" in k):
        print(f"  {k}: {test_cache[k].shape}")
del test_cache

with open("results/00_setup/gender_token_ids.json") as f:
    gender_token_ids = json.load(f)
male_ids = gender_token_ids["male"]
female_ids = gender_token_ids["female"]

# ── Bias prompts for evaluation ──
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
    "The surgeon explained that",
    "The nanny mentioned that",
    "The janitor cleaned the floor and",
    "The hairdresser styled the hair and",
    "The plumber fixed the pipe and",
    "The dancer performed and",
    "The firefighter rescued the person and",
    "The babysitter watched the kids and",
]


def compute_bias_score(model, prompt, hooks=None):
    """Compute signed gender bias score: P(male) - P(female)."""
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        if hooks:
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        else:
            logits = model(tokens)
    probs = torch.softmax(logits[0, -1, :], dim=0)
    return probs[male_ids].sum().item() - probs[female_ids].sum().item()


def compute_mean_abs_bias(model, prompts, hooks=None):
    """Mean absolute bias across prompts."""
    scores = [abs(compute_bias_score(model, p, hooks)) for p in prompts]
    return np.mean(scores), scores


# ── Baseline ──
baseline_bias, baseline_scores = compute_mean_abs_bias(model, BIAS_PROMPTS)
# Also compute signed scores for each prompt
baseline_signed = [compute_bias_score(model, p) for p in BIAS_PROMPTS]
print(f"\nBaseline mean |bias|: {baseline_bias:.6f}")
print(f"Per-prompt signed bias:")
for p, s in zip(BIAS_PROMPTS, baseline_signed):
    direction = "M" if s > 0 else "F"
    print(f"  {p:<45} {s:+.4f} ({direction})")


# ═══════════════════════════════════════════════
# APPROACH 1: ZERO ABLATION
# ═══════════════════════════════════════════════
print("\n" + "="*60)
print("ZERO-ABLATION: HEAD-LEVEL")
print("="*60)


def zero_ablate_head_z_hook(z, hook, head_idx):
    """Zero out one head's pre-projection output (hook_z).
    z shape: (batch, pos, n_heads, d_head)
    Zeroing hook_z for a head = zeroing its contribution to the residual stream.
    """
    z[:, :, head_idx, :] = 0.0
    return z


head_bias_change = np.zeros((n_layers, n_heads))
head_bias_ablated = np.zeros((n_layers, n_heads))

for layer in tqdm(range(n_layers), desc="Zero-ablating heads"):
    for head in range(n_heads):
        hook_name = f"blocks.{layer}.attn.hook_z"
        hooks = [(hook_name, partial(zero_ablate_head_z_hook, head_idx=head))]
        abl_bias, _ = compute_mean_abs_bias(model, BIAS_PROMPTS, hooks=hooks)
        head_bias_change[layer, head] = baseline_bias - abl_bias  # positive = bias reduced
        head_bias_ablated[layer, head] = abl_bias

print("\nTop 20 heads by bias reduction when zero-ablated:")
flat = head_bias_change.flatten()
top_idx = np.argsort(flat)[::-1][:20]
top_heads = []
for idx in top_idx:
    l, h = idx // n_heads, idx % n_heads
    change = head_bias_change[l, h]
    remaining = head_bias_ablated[l, h]
    top_heads.append({"layer": int(l), "head": int(h), "bias_reduction": float(change),
                       "remaining_bias": float(remaining)})
    print(f"  L{l}H{h}: bias_reduction={change:.6f}, remaining={remaining:.6f}")

# Also find heads that INCREASE bias when ablated (potentially "debiasing" heads)
print("\nTop 10 heads that INCREASE bias when ablated (anti-bias heads):")
bottom_idx = np.argsort(flat)[:10]
for idx in bottom_idx:
    l, h = idx // n_heads, idx % n_heads
    change = head_bias_change[l, h]
    print(f"  L{l}H{h}: bias_change={change:.6f} (ablation increases bias)")


# ── MLP zero ablation ──
print("\n" + "="*60)
print("ZERO-ABLATION: MLP LAYER-LEVEL")
print("="*60)


def zero_ablate_mlp_hook(output, hook):
    """Zero out entire MLP output."""
    output[:] = 0.0
    return output


mlp_bias_change = np.zeros(n_layers)
for layer in tqdm(range(n_layers), desc="Zero-ablating MLPs"):
    hook_name = f"blocks.{layer}.hook_mlp_out"
    hooks = [(hook_name, zero_ablate_mlp_hook)]
    abl_bias, _ = compute_mean_abs_bias(model, BIAS_PROMPTS, hooks=hooks)
    mlp_bias_change[layer] = baseline_bias - abl_bias

print("\nMLP importance for gender bias:")
for layer in range(n_layers):
    bar = "█" * int(abs(mlp_bias_change[layer]) * 200)
    sign = "+" if mlp_bias_change[layer] > 0 else "-"
    print(f"  Layer {layer:2d}: {mlp_bias_change[layer]:+.6f} {bar}")


# ── Attention output (summed) zero ablation ──
print("\n" + "="*60)
print("ZERO-ABLATION: ATTENTION (SUMMED) LAYER-LEVEL")
print("="*60)


def zero_ablate_attn_hook(output, hook):
    """Zero out entire attention output."""
    output[:] = 0.0
    return output


attn_bias_change = np.zeros(n_layers)
for layer in tqdm(range(n_layers), desc="Zero-ablating attention"):
    hook_name = f"blocks.{layer}.hook_attn_out"
    hooks = [(hook_name, zero_ablate_attn_hook)]
    abl_bias, _ = compute_mean_abs_bias(model, BIAS_PROMPTS, hooks=hooks)
    attn_bias_change[layer] = baseline_bias - abl_bias

print("\nAttention layer importance for gender bias:")
for layer in range(n_layers):
    print(f"  Layer {layer:2d}: attn={attn_bias_change[layer]:+.6f}, mlp={mlp_bias_change[layer]:+.6f}")


# ═══════════════════════════════════════════════
# APPROACH 2: MEAN-ABLATION (more controlled)
# ═══════════════════════════════════════════════
print("\n" + "="*60)
print("MEAN-ABLATION: Computing mean activations...")
print("="*60)

# Compute mean activations across a corpus of neutral prompts
NEUTRAL_PROMPTS = [
    "The cat sat on the mat.",
    "It was a sunny day outside.",
    "The meeting started at noon.",
    "They walked along the path.",
    "The book was on the table.",
    "It began to rain heavily.",
    "The car drove down the road.",
    "We had dinner together.",
    "The music played softly.",
    "Time passed slowly that day.",
] + BIAS_PROMPTS  # include bias prompts too for a broader mean

# Collect mean head z activations
mean_head_z = {}
for layer in range(n_layers):
    hook_name = f"blocks.{layer}.attn.hook_z"
    means = []
    for prompt in NEUTRAL_PROMPTS:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        # Average over batch and sequence position: shape (n_heads, d_head)
        means.append(cache[hook_name].mean(dim=[0, 1]))
    mean_head_z[layer] = torch.stack(means).mean(dim=0)  # (n_heads, d_head)


def mean_ablate_head_z_hook(z, hook, head_idx, mean_z):
    """Replace one head's z with its mean activation."""
    z[:, :, head_idx, :] = mean_z[head_idx]
    return z


head_mean_abl_change = np.zeros((n_layers, n_heads))

for layer in tqdm(range(n_layers), desc="Mean-ablating heads"):
    for head in range(n_heads):
        hook_name = f"blocks.{layer}.attn.hook_z"
        hooks = [(hook_name, partial(mean_ablate_head_z_hook, head_idx=head,
                                      mean_z=mean_head_z[layer]))]
        abl_bias, _ = compute_mean_abs_bias(model, BIAS_PROMPTS, hooks=hooks)
        head_mean_abl_change[layer, head] = baseline_bias - abl_bias

print("\nTop 20 heads by bias reduction (MEAN ablation):")
flat_mean = head_mean_abl_change.flatten()
top_idx_mean = np.argsort(flat_mean)[::-1][:20]
for idx in top_idx_mean:
    l, h = idx // n_heads, idx % n_heads
    print(f"  L{l}H{h}: {head_mean_abl_change[l, h]:.6f}")


# ═══════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Head zero-ablation heatmap
vmax = max(np.max(np.abs(head_bias_change)), 1e-6)
im0 = axes[0, 0].imshow(head_bias_change, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[0, 0].set_xlabel("Head")
axes[0, 0].set_ylabel("Layer")
axes[0, 0].set_title("Head Zero-Ablation: Bias Change\n(red=ablation reduces bias, blue=increases)")
plt.colorbar(im0, ax=axes[0, 0])

# 2. Head mean-ablation heatmap
vmax2 = max(np.max(np.abs(head_mean_abl_change)), 1e-6)
im1 = axes[0, 1].imshow(head_mean_abl_change, aspect='auto', cmap='RdBu_r', vmin=-vmax2, vmax=vmax2)
axes[0, 1].set_xlabel("Head")
axes[0, 1].set_ylabel("Layer")
axes[0, 1].set_title("Head Mean-Ablation: Bias Change\n(red=ablation reduces bias, blue=increases)")
plt.colorbar(im1, ax=axes[0, 1])

# 3. Layer-level: Attention vs MLP
x = np.arange(n_layers)
width = 0.35
axes[1, 0].bar(x - width/2, attn_bias_change, width, label='Attention', alpha=0.8, color='steelblue')
axes[1, 0].bar(x + width/2, mlp_bias_change, width, label='MLP', alpha=0.8, color='coral')
axes[1, 0].set_xlabel("Layer")
axes[1, 0].set_ylabel("Bias Change (+ = reduced)")
axes[1, 0].set_title("Attention vs MLP: Bias Contribution by Layer")
axes[1, 0].legend()
axes[1, 0].set_xticks(x)
axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.3)

# 4. Correlation between zero and mean ablation
zero_flat = head_bias_change.flatten()
mean_flat = head_mean_abl_change.flatten()
axes[1, 1].scatter(zero_flat, mean_flat, alpha=0.5, s=20)
axes[1, 1].set_xlabel("Zero-ablation bias change")
axes[1, 1].set_ylabel("Mean-ablation bias change")
axes[1, 1].set_title("Zero vs Mean Ablation Correlation")
# Add identity line
lims = [min(zero_flat.min(), mean_flat.min()), max(zero_flat.max(), mean_flat.max())]
axes[1, 1].plot(lims, lims, 'r--', alpha=0.3)

# Annotate top heads
for idx in top_idx[:5]:
    l, h = idx // n_heads, idx % n_heads
    axes[1, 1].annotate(f"L{l}H{h}", (zero_flat[idx], mean_flat[idx]), fontsize=7)

plt.suptitle("Component Importance for Gender Bias (GPT-2 Small)", fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "component_importance.png", dpi=150)
plt.close()
print(f"Saved component importance plot")


# ── Per-prompt analysis: which heads matter for which prompts? ──
print("\n" + "="*60)
print("PER-PROMPT HEAD IMPORTANCE")
print("="*60)

# For top 10 heads, show per-prompt effect
top10_heads = [(idx // n_heads, idx % n_heads) for idx in top_idx[:10]]
per_prompt_effects = {}

for l, h in top10_heads:
    hook_name = f"blocks.{l}.attn.hook_z"
    hooks = [(hook_name, partial(zero_ablate_head_z_hook, head_idx=h))]
    effects = []
    for prompt in BIAS_PROMPTS:
        orig = compute_bias_score(model, prompt)
        abl = compute_bias_score(model, prompt, hooks=hooks)
        effects.append({"prompt": prompt, "original": orig, "ablated": abl, "change": orig - abl})
    per_prompt_effects[f"L{l}H{h}"] = effects

# Show for top 3 heads
for head_name in list(per_prompt_effects.keys())[:3]:
    print(f"\n{head_name}:")
    for e in per_prompt_effects[head_name]:
        arrow = "→" if abs(e["change"]) > 0.01 else "·"
        print(f"  {arrow} {e['prompt']:<40} {e['original']:+.4f} → {e['ablated']:+.4f} (Δ={e['change']:+.4f})")


# ── Save results ──
results = {
    "baseline_mean_abs_bias": float(baseline_bias),
    "baseline_signed_scores": list(zip(BIAS_PROMPTS, [float(s) for s in baseline_signed])),
    "head_zero_ablation": head_bias_change.tolist(),
    "head_mean_ablation": head_mean_abl_change.tolist(),
    "mlp_zero_ablation": mlp_bias_change.tolist(),
    "attn_zero_ablation": attn_bias_change.tolist(),
    "top_heads_zero": top_heads,
    "per_prompt_effects": per_prompt_effects,
}

with open(RESULTS_DIR / "patching_results.json", "w") as f:
    json.dump(results, f, indent=2)

np.save(RESULTS_DIR / "head_zero_ablation.npy", head_bias_change)
np.save(RESULTS_DIR / "head_mean_ablation.npy", head_mean_abl_change)

print(f"\nAll results saved to {RESULTS_DIR}")
print("\n✓ Experiment 01 complete.")
