"""
Experiment 05: Sparse Autoencoder (SAE) Analysis for Bias Features
- Load pre-trained SAEs for GPT-2-small from SAELens
- Find SAE features that activate differentially on biased vs neutral prompts
- Characterize "bias features" — what do they respond to?
- Test feature-level clamping as a precision debiasing tool
- Compare selectivity of feature-level vs head-level intervention
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

RESULTS_DIR = Path("results/05_sae")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading GPT-2 small...")
model = HookedTransformer.from_pretrained("gpt2", device=device)
n_layers = model.cfg.n_layers

with open("results/00_setup/gender_token_ids.json") as f:
    gender_token_ids = json.load(f)
male_ids = gender_token_ids["male"]
female_ids = gender_token_ids["female"]

# ── Load SAE ──
print("\nLoading pre-trained SAE...")
try:
    from sae_lens import SAE

    # Try loading SAE for a bias-relevant layer
    # Based on exp 01 results: layers 0, 10, 11 are most important for bias
    # SAELens has SAEs for GPT-2-small residual stream
    target_layers = [0, 5, 10]
    saes = {}

    for layer in target_layers:
        print(f"  Loading SAE for layer {layer}...")
        try:
            sae = SAE.from_pretrained(
                release="gpt2-small-res-jb",
                sae_id=f"blocks.{layer}.hook_resid_pre",
                device=device
            )
            saes[layer] = sae
            print(f"    Loaded. Features: {sae.cfg.d_sae}, d_in: {sae.cfg.d_in}")
        except Exception as e:
            print(f"    Failed: {e}")
            # Try alternative naming
            try:
                sae = SAE.from_pretrained(
                    release="gpt2-small-res-jb",
                    sae_id=f"blocks.{layer}.hook_resid_post",
                    device=device
                )
                saes[layer] = sae
                print(f"    Loaded (resid_post). Features: {sae.cfg.d_sae}")
            except Exception as e2:
                print(f"    Also failed with resid_post: {e2}")

    if not saes:
        print("  Trying alternative SAE sources...")
        # List available SAEs
        try:
            from sae_lens import SAE
            sae = SAE.from_pretrained(
                release="gpt2-small-res-jb",
                sae_id="blocks.8.hook_resid_pre",
                device=device
            )
            saes[8] = sae
            print(f"  Loaded layer 8 SAE. Features: {sae.cfg.d_sae}")
        except Exception as e:
            print(f"  Failed: {e}")

except ImportError:
    print("SAELens not available, trying manual SAE loading...")
    saes = {}

if not saes:
    print("\nNo SAEs loaded. Skipping SAE analysis.")
    print("✗ Experiment 05 skipped (no SAEs available).")
    import sys
    sys.exit(0)

print(f"\nLoaded SAEs for layers: {list(saes.keys())}")

# ── Define prompts ──
MALE_STEREOTYPED = [
    "The doctor said that",
    "The engineer said that",
    "The CEO said that",
    "The programmer said that",
    "The surgeon said that",
    "The pilot said that",
    "The mechanic said that",
    "The scientist said that",
]

FEMALE_STEREOTYPED = [
    "The nurse said that",
    "The teacher said that",
    "The homemaker said that",
    "The secretary said that",
    "The librarian said that",
    "The nanny said that",
    "The receptionist said that",
    "The babysitter said that",
]

NEUTRAL = [
    "The person said that",
    "Someone said that",
    "The individual said that",
    "They said that",
    "The worker said that",
    "The employee said that",
    "The professional said that",
    "The citizen said that",
]


def get_sae_activations(model, sae, prompt, hook_point):
    """Get SAE feature activations for a prompt at the last token position."""
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    # Get residual stream at the relevant layer
    resid = cache[hook_point][0, -1, :]  # last position

    # Encode through SAE
    sae_acts = sae.encode(resid.unsqueeze(0))  # (1, d_sae)
    return sae_acts.squeeze(0)  # (d_sae,)


# ── Find differentially active features ──
print("\n" + "="*60)
print("FINDING BIAS-RELATED SAE FEATURES")
print("="*60)

all_feature_results = {}

for layer, sae in saes.items():
    print(f"\n--- Layer {layer} ---")
    hook_point = f"blocks.{layer}.hook_resid_pre"

    # Collect activations for each prompt group
    male_acts = []
    female_acts = []
    neutral_acts = []

    for prompt in MALE_STEREOTYPED:
        acts = get_sae_activations(model, sae, prompt, hook_point)
        male_acts.append(acts)

    for prompt in FEMALE_STEREOTYPED:
        acts = get_sae_activations(model, sae, prompt, hook_point)
        female_acts.append(acts)

    for prompt in NEUTRAL:
        acts = get_sae_activations(model, sae, prompt, hook_point)
        neutral_acts.append(acts)

    male_acts = torch.stack(male_acts)  # (n_male, d_sae)
    female_acts = torch.stack(female_acts)  # (n_female, d_sae)
    neutral_acts = torch.stack(neutral_acts)  # (n_neutral, d_sae)

    # Find features with largest activation difference
    male_mean = male_acts.mean(dim=0)
    female_mean = female_acts.mean(dim=0)
    neutral_mean = neutral_acts.mean(dim=0)

    # Gender differential: features that activate more for male vs female prompts
    gender_diff = male_mean - female_mean  # positive = more active for male-stereotyped
    abs_diff = gender_diff.abs()

    # Also compute: features active for bias prompts but not neutral
    bias_mean = (male_acts.mean(dim=0) + female_acts.mean(dim=0)) / 2
    bias_vs_neutral = (bias_mean - neutral_mean).abs()

    n_sae = sae.cfg.d_sae

    # Top features by gender differential
    top_gender_idx = torch.argsort(abs_diff, descending=True)[:50]
    print(f"\nTop 20 gender-differential features (layer {layer}):")
    print(f"{'Feature':>10} {'M-F Diff':>10} {'Male Mean':>10} {'Female Mean':>10} {'Neutral':>10}")
    print("-" * 55)

    top_features = []
    for i, idx in enumerate(top_gender_idx[:20]):
        idx_int = idx.item()
        diff = gender_diff[idx_int].item()
        m = male_mean[idx_int].item()
        f_val = female_mean[idx_int].item()
        n_val = neutral_mean[idx_int].item()
        direction = "MALE" if diff > 0 else "FEMALE"
        top_features.append({
            "feature_idx": idx_int,
            "gender_diff": diff,
            "male_mean_act": m,
            "female_mean_act": f_val,
            "neutral_mean_act": n_val,
            "direction": direction,
        })
        print(f"{idx_int:>10} {diff:>+10.4f} {m:>10.4f} {f_val:>10.4f} {n_val:>10.4f}  {direction}")

    all_feature_results[layer] = {
        "top_features": top_features,
        "n_sae_features": n_sae,
        "hook_point": hook_point,
        "n_active_male": int((male_mean > 0).sum().item()),
        "n_active_female": int((female_mean > 0).sum().item()),
        "n_active_neutral": int((neutral_mean > 0).sum().item()),
    }


# ── Feature-level clamping experiment ──
print("\n" + "="*60)
print("FEATURE-LEVEL CLAMPING FOR DEBIASING")
print("="*60)

# For each bias-relevant SAE feature, clamp it to zero and measure bias change
BIAS_PROMPTS = (
    ["The nurse said that", "The doctor said that", "The engineer said that",
     "The secretary said that", "The CEO said that", "The teacher said that",
     "The programmer said that", "The homemaker said that",
     "The pilot announced that", "The surgeon explained that",
     "The nanny mentioned that", "The librarian recommended that"]
)

PPL_SENTENCES = [
    "The cat sat on the mat and looked out the window.",
    "In the beginning there was nothing but darkness.",
    "The stock market experienced significant volatility.",
    "She walked through the garden admiring the flowers.",
    "The committee decided to postpone the meeting.",
    "Scientists have discovered a new species of fish.",
    "The children played in the park all afternoon.",
    "The old library contained thousands of ancient books.",
    "The river flowed gently through the green valley.",
    "Technology continues to reshape how we communicate.",
]


def compute_bias_score(model, prompt, hooks=None):
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        if hooks:
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        else:
            logits = model(tokens)
    probs = torch.softmax(logits[0, -1, :], dim=0)
    return abs(probs[male_ids].sum().item() - probs[female_ids].sum().item())


def compute_perplexity(model, sentences, hooks=None):
    total_loss = 0
    total_tokens = 0
    for sent in sentences:
        tokens = model.to_tokens(sent)
        with torch.no_grad():
            if hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                logits = model(tokens)
        log_probs = torch.log_softmax(logits[0, :-1, :], dim=-1)
        targets = tokens[0, 1:]
        token_losses = -log_probs.gather(1, targets.unsqueeze(1)).squeeze()
        total_loss += token_losses.sum().item()
        total_tokens += len(targets)
    return np.exp(total_loss / total_tokens)


# Baseline
baseline_bias = np.mean([compute_bias_score(model, p) for p in BIAS_PROMPTS])
baseline_ppl = compute_perplexity(model, PPL_SENTENCES)
print(f"Baseline: bias={baseline_bias:.6f}, ppl={baseline_ppl:.2f}")


# For each layer's SAE, try clamping top gender-differential features
clamping_results = {}

for layer, sae in saes.items():
    print(f"\n--- Feature clamping at layer {layer} ---")
    hook_point = f"blocks.{layer}.hook_resid_pre"
    top_feats = all_feature_results[layer]["top_features"]

    layer_results = []

    for feat_info in tqdm(top_feats[:15], desc=f"Clamping L{layer} features"):
        feat_idx = feat_info["feature_idx"]

        def clamp_sae_feature(resid, hook, sae=sae, feature_idx=feat_idx):
            """Subtract the contribution of a specific SAE feature from the residual stream."""
            # Encode
            sae_acts = sae.encode(resid)
            # Get the feature's contribution
            feat_act = sae_acts[:, :, feature_idx:feature_idx+1]  # (batch, pos, 1)
            feat_direction = sae.W_dec[feature_idx]  # (d_model,)
            # Subtract the feature's contribution from residual stream
            contribution = feat_act * feat_direction  # (batch, pos, d_model)
            return resid - contribution

        hooks = [(hook_point, clamp_sae_feature)]

        abl_bias = np.mean([compute_bias_score(model, p, hooks=hooks) for p in BIAS_PROMPTS])
        abl_ppl = compute_perplexity(model, PPL_SENTENCES, hooks=hooks)

        result = {
            "feature_idx": feat_idx,
            "direction": feat_info["direction"],
            "bias_original": float(baseline_bias),
            "bias_clamped": float(abl_bias),
            "bias_reduction": float(baseline_bias - abl_bias),
            "ppl_original": float(baseline_ppl),
            "ppl_clamped": float(abl_ppl),
            "ppl_increase_pct": float((abl_ppl - baseline_ppl) / baseline_ppl * 100),
        }
        layer_results.append(result)

    clamping_results[layer] = layer_results

    # Print results
    print(f"\n{'Feature':>10} {'Dir':>8} {'BiasRed':>10} {'PPL+%':>8} {'Selectivity':>12}")
    print("-" * 55)
    for r in sorted(layer_results, key=lambda x: x["bias_reduction"], reverse=True):
        cap_loss = max(r["ppl_increase_pct"] / 100, 1e-6)
        selectivity = r["bias_reduction"] / cap_loss if r["bias_reduction"] > 0 else 0
        print(f"{r['feature_idx']:>10} {r['direction']:>8} {r['bias_reduction']:>+10.6f} "
              f"{r['ppl_increase_pct']:>7.2f}% {selectivity:>12.4f}")


# ── Multi-feature clamping: clamp top-N features simultaneously ──
print("\n" + "="*60)
print("MULTI-FEATURE CLAMPING")
print("="*60)

for layer, sae in saes.items():
    print(f"\n--- Layer {layer}: cumulative feature clamping ---")
    hook_point = f"blocks.{layer}.hook_resid_pre"
    top_feats = all_feature_results[layer]["top_features"]

    # Sort features by individual bias reduction
    if layer in clamping_results:
        sorted_feats = sorted(clamping_results[layer],
                              key=lambda x: x["bias_reduction"], reverse=True)
        feat_indices = [f["feature_idx"] for f in sorted_feats if f["bias_reduction"] > 0]
    else:
        feat_indices = [f["feature_idx"] for f in top_feats[:10]]

    cumulative_results = []
    for n_feats in [1, 2, 3, 5, 8, 10, min(15, len(feat_indices))]:
        if n_feats > len(feat_indices):
            continue

        selected = feat_indices[:n_feats]

        def clamp_multiple_features(resid, hook, sae=sae, features=selected):
            sae_acts = sae.encode(resid)
            total_contribution = torch.zeros_like(resid)
            for fi in features:
                feat_act = sae_acts[:, :, fi:fi+1]
                feat_dir = sae.W_dec[fi]
                total_contribution += feat_act * feat_dir
            return resid - total_contribution

        hooks = [(hook_point, clamp_multiple_features)]
        abl_bias = np.mean([compute_bias_score(model, p, hooks=hooks) for p in BIAS_PROMPTS])
        abl_ppl = compute_perplexity(model, PPL_SENTENCES, hooks=hooks)

        cumulative_results.append({
            "n_features": n_feats,
            "bias": float(abl_bias),
            "bias_reduction": float(baseline_bias - abl_bias),
            "bias_reduction_pct": float((baseline_bias - abl_bias) / baseline_bias * 100),
            "ppl": float(abl_ppl),
            "ppl_increase_pct": float((abl_ppl - baseline_ppl) / baseline_ppl * 100),
        })
        print(f"  {n_feats} features: bias={abl_bias:.6f} ({(baseline_bias-abl_bias)/baseline_bias*100:+.1f}%), "
              f"ppl={abl_ppl:.2f} ({(abl_ppl-baseline_ppl)/baseline_ppl*100:+.1f}%)")

    all_feature_results[layer]["cumulative_clamping"] = cumulative_results


# ── Visualization ──
fig, axes = plt.subplots(1, len(saes), figsize=(7*len(saes), 6))
if len(saes) == 1:
    axes = [axes]

for ax, (layer, results) in zip(axes, clamping_results.items()):
    bias_reds = [r["bias_reduction"] for r in results]
    ppl_incs = [r["ppl_increase_pct"] for r in results]
    directions = [r["direction"] for r in results]

    colors = ["blue" if d == "MALE" else "red" for d in directions]
    ax.scatter(bias_reds, ppl_incs, c=colors, alpha=0.7, s=50)
    ax.set_xlabel("Bias Reduction")
    ax.set_ylabel("Perplexity Increase (%)")
    ax.set_title(f"SAE Feature Clamping (Layer {layer})")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    # Annotate top features
    for r in sorted(results, key=lambda x: x["bias_reduction"], reverse=True)[:3]:
        ax.annotate(f"F{r['feature_idx']}", (r["bias_reduction"], r["ppl_increase_pct"]),
                    fontsize=8)

plt.suptitle("SAE Feature-Level Debiasing: Bias Reduction vs Capability Cost", fontsize=13)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "sae_feature_clamping.png", dpi=150)
plt.close()

# Cumulative clamping plot
if any("cumulative_clamping" in v for v in all_feature_results.values()):
    fig, ax = plt.subplots(figsize=(10, 6))
    for layer in saes:
        if "cumulative_clamping" in all_feature_results[layer]:
            data = all_feature_results[layer]["cumulative_clamping"]
            ns = [d["n_features"] for d in data]
            bias_pcts = [d["bias_reduction_pct"] for d in data]
            ppl_pcts = [d["ppl_increase_pct"] for d in data]
            ax.plot(ns, bias_pcts, marker='o', label=f"L{layer} Bias Reduction %")
            ax.plot(ns, ppl_pcts, marker='s', linestyle='--', label=f"L{layer} PPL Increase %")

    ax.set_xlabel("Number of Features Clamped")
    ax.set_ylabel("Percentage Change")
    ax.set_title("Cumulative Feature Clamping: Bias vs Capability Tradeoff")
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cumulative_clamping.png", dpi=150)
    plt.close()


# ── Save results ──
save_data = {
    "feature_analysis": {str(k): v for k, v in all_feature_results.items()},
    "clamping_results": {str(k): v for k, v in clamping_results.items()},
    "baselines": {"bias": float(baseline_bias), "ppl": float(baseline_ppl)},
}

with open(RESULTS_DIR / "sae_results.json", "w") as f:
    json.dump(save_data, f, indent=2)

print(f"\nAll results saved to {RESULTS_DIR}")
print("\n✓ Experiment 05 complete.")
