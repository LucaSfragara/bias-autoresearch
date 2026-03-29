"""
Experiment 17: Occupation-Position Activation Patching

Confirms that L10H9 reads its gender-stereotype signal from the occupation
token position. By swapping activations at the occupation position between
stereotypically male and female occupations, we can flip L10H9's output.

Also: per-occupation bias profile and how L10H9 modulates each.
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

RESULTS_DIR = Path("results/17_occupation_patching")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2", device=device)
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

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


def gender_bias_score(logits):
    """Signed: positive = male-biased"""
    probs = torch.softmax(logits, dim=-1)
    return probs[male_ids].sum().item() - probs[female_ids].sum().item()


def scale_head(z, hook, head_idx, alpha):
    z[:, :, head_idx, :] *= alpha
    return z


# ═══════════════════════════════════════════════
# 1. PER-OCCUPATION BIAS PROFILE
# ═══════════════════════════════════════════════
print("="*70)
print("1. PER-OCCUPATION BIAS PROFILE (baseline vs L10H9 ablated)")
print("="*70)

OCCUPATIONS = [
    "nurse", "doctor", "engineer", "secretary", "CEO", "teacher",
    "programmer", "homemaker", "mechanic", "receptionist", "pilot",
    "librarian", "surgeon", "nanny", "janitor", "hairdresser",
    "plumber", "dancer", "firefighter", "babysitter", "accountant",
    "lawyer", "chef", "farmer", "dentist", "professor",
]

TEMPLATE = "The %s said that"

occ_results = {}
for occ in tqdm(OCCUPATIONS, desc="Per-occupation"):
    prompt = TEMPLATE % occ
    tokens = model.to_tokens(prompt)

    # Baseline
    with torch.no_grad():
        logits = model(tokens)
    baseline_bias = gender_bias_score(logits[0, -1, :])

    # L10H9 ablated
    hooks = [("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.0))]
    with torch.no_grad():
        logits_abl = model.run_with_hooks(tokens, fwd_hooks=hooks)
    ablated_bias = gender_bias_score(logits_abl[0, -1, :])

    occ_results[occ] = {
        "baseline_bias": float(baseline_bias),
        "ablated_bias": float(ablated_bias),
        "l10h9_effect": float(baseline_bias - ablated_bias),
    }

# Sort by baseline bias
sorted_occ = sorted(occ_results.items(), key=lambda x: x[1]["baseline_bias"])

print("\n%-15s %12s %12s %12s %s" % ("Occupation", "Baseline", "Ablated", "L10H9 Effect", "Direction"))
print("-" * 70)
for occ, r in sorted_occ:
    direction = "male→" if r["l10h9_effect"] > 0.005 else ("←female" if r["l10h9_effect"] < -0.005 else "neutral")
    print("%-15s %+11.4f %+11.4f %+11.4f  %s" % (
        occ, r["baseline_bias"], r["ablated_bias"], r["l10h9_effect"], direction))

# Count
male_push = sum(1 for r in occ_results.values() if r["l10h9_effect"] > 0.005)
female_push = sum(1 for r in occ_results.values() if r["l10h9_effect"] < -0.005)
neutral = len(occ_results) - male_push - female_push
print("\nL10H9 pushes: %d male, %d female, %d neutral" % (male_push, female_push, neutral))


# ═══════════════════════════════════════════════
# 2. OCCUPATION-POSITION ACTIVATION PATCHING
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("2. OCCUPATION-POSITION ACTIVATION PATCHING")
print("="*70)

# Patch the occupation position (pos 2) residual stream from one occupation
# into a prompt with a different occupation, at the layer just before L10

# Source/target pairs: stereotypically female → stereotypically male and vice versa
PATCH_PAIRS = [
    ("nurse", "engineer"),
    ("secretary", "mechanic"),
    ("homemaker", "plumber"),
    ("nanny", "surgeon"),
    ("babysitter", "pilot"),
    ("hairdresser", "firefighter"),
    ("receptionist", "programmer"),
    ("dancer", "farmer"),
]

# We need to verify that occupation is at position 2 (BOS, The, <occ>)
# Check tokenization
print("\nVerifying tokenization:")
for occ1, occ2 in PATCH_PAIRS[:3]:
    t1 = model.to_tokens("The %s said that" % occ1)
    t2 = model.to_tokens("The %s said that" % occ2)
    print("  '%s': %d tokens, '%s': %d tokens" % (occ1, t1.shape[1], occ2, t2.shape[1]))

# Find occupation token position for each prompt
def find_occ_position(prompt, occ):
    """Find which position the occupation token starts at"""
    tokens = model.to_tokens(prompt)
    occ_tokens = model.to_tokens(" " + occ, prepend_bos=False).squeeze()
    if occ_tokens.dim() == 0:
        occ_tokens = occ_tokens.unsqueeze(0)
    # Occupation should be at position 2 for "The <occ> said that"
    # But some occupations might be multi-token
    return 2, occ_tokens.shape[0]

patching_results = []

for occ_source, occ_target in tqdm(PATCH_PAIRS, desc="Occupation patching"):
    prompt_source = TEMPLATE % occ_source
    prompt_target = TEMPLATE % occ_target

    tok_source = model.to_tokens(prompt_source)
    tok_target = model.to_tokens(prompt_target)

    # Only patch if both are single-token occupations (same total length)
    if tok_source.shape[1] != tok_target.shape[1]:
        print("  Skipping %s→%s (length mismatch)" % (occ_source, occ_target))
        continue

    occ_pos = 2  # position of occupation token

    # Get source activations at occupation position
    with torch.no_grad():
        _, source_cache = model.run_with_cache(tok_source)

    # Baseline: run target prompt clean
    with torch.no_grad():
        target_logits = model(tok_target)
    clean_bias = gender_bias_score(target_logits[0, -1, :])

    # Patch at each layer before L10
    layer_effects = {}
    for patch_layer in range(n_layers):
        source_resid = source_cache["blocks.%d.hook_resid_pre" % patch_layer][0, occ_pos, :].clone()

        def patch_occ(resid, hook, src=source_resid, pos=occ_pos):
            resid[:, pos, :] = src
            return resid

        hooks = [("blocks.%d.hook_resid_pre" % patch_layer, patch_occ)]
        with torch.no_grad():
            patched_logits = model.run_with_hooks(tok_target, fwd_hooks=hooks)
        patched_bias = gender_bias_score(patched_logits[0, -1, :])

        layer_effects[patch_layer] = float(patched_bias - clean_bias)

    patching_results.append({
        "source": occ_source,
        "target": occ_target,
        "clean_target_bias": float(clean_bias),
        "layer_effects": layer_effects,
    })

# Aggregate across all pairs
print("\nOccupation patching effect by layer (mean across pairs):")
print("Layer  Mean Effect  Direction")
print("-" * 40)
for layer in range(n_layers):
    effects = [pr["layer_effects"][layer] for pr in patching_results]
    mean_eff = np.mean(effects)
    bar = "F" * int(max(0, -mean_eff) * 200) + "M" * int(max(0, mean_eff) * 200)
    print("  L%-3d %+10.6f  %s" % (layer, mean_eff, bar))


# ═══════════════════════════════════════════════
# 3. L10H9 READS FROM OCCUPATION: CAUSAL VERIFICATION
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("3. CAUSAL VERIFICATION: L10H9 reads stereotype from occupation")
print("="*70)

# For each occupation, cache L10H9's attention weight to position 2 (occupation)
# and the value vector it reads from there. Project onto gender direction.

W_U = model.W_U
male_unembed = W_U[:, male_ids].mean(dim=1)
female_unembed = W_U[:, female_ids].mean(dim=1)
gender_dir = (male_unembed - female_unembed)
gender_dir = gender_dir / gender_dir.norm()
gender_dir_cpu = gender_dir.cpu()

W_O = model.blocks[10].attn.W_O[9]  # [d_head, d_model]

occ_attn_data = {}
for occ in tqdm(OCCUPATIONS, desc="L10H9 attention"):
    prompt = TEMPLATE % occ
    tokens = model.to_tokens(prompt)
    n_tok = tokens.shape[1]

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    # L10H9 attention at last position
    attn = cache["blocks.10.attn.hook_pattern"][0, 9, -1, :]  # [n_tok]
    v = cache["blocks.10.attn.hook_v"][0, :, 9, :]  # [n_tok, d_head]

    # Contribution from occupation position (pos 2)
    occ_contrib = attn[2] * v[2] @ W_O  # [d_model]
    occ_gender_proj = torch.dot(occ_contrib.cpu(), gender_dir_cpu).item()

    # Total output gender projection
    total_output = torch.zeros(model.cfg.d_model)
    for pos in range(n_tok):
        total_output += (attn[pos] * v[pos] @ W_O).cpu()
    total_gender_proj = torch.dot(total_output, gender_dir_cpu).item()

    occ_attn_data[occ] = {
        "attn_to_occ": float(attn[2].item()),
        "attn_to_bos": float(attn[0].item()),
        "occ_gender_proj": float(occ_gender_proj),
        "total_gender_proj": float(total_gender_proj),
        "occ_fraction": float(occ_gender_proj / (total_gender_proj + 1e-10)),
    }

# Sort by occupation gender projection
sorted_occ_attn = sorted(occ_attn_data.items(), key=lambda x: x[1]["occ_gender_proj"])
print("\n%-15s %8s %8s %12s %12s %8s" % (
    "Occupation", "Attn→occ", "Attn→BOS", "Occ Gender", "Total Gender", "Occ Frac"))
print("-" * 75)
for occ, r in sorted_occ_attn:
    print("%-15s %7.1f%% %7.1f%% %+11.4f %+11.4f %7.1f%%" % (
        occ, r["attn_to_occ"]*100, r["attn_to_bos"]*100,
        r["occ_gender_proj"], r["total_gender_proj"],
        r["occ_fraction"]*100))

# Correlation between occupation position contribution and output bias
occ_projs = [occ_attn_data[occ]["occ_gender_proj"] for occ in OCCUPATIONS]
output_biases = [occ_results[occ]["baseline_bias"] for occ in OCCUPATIONS]
corr = np.corrcoef(occ_projs, output_biases)[0, 1]
print("\nCorrelation(occ_position gender proj, output bias): r=%.4f" % corr)


# ═══════════════════════════════════════════════
# 4. ATTENTION REDISTRIBUTION: WHAT HAPPENS WHEN BOS IS REMOVED
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("4. ATTENTION REDISTRIBUTION AFTER BOS ZEROING")
print("="*70)

# When we zero BOS attention, where does the weight go?
def capture_pattern_after_bos_zero(pattern, hook, head_idx):
    """Zero BOS attention and renormalize, return modified pattern"""
    pattern[:, head_idx, :, 0] = 0
    row_sums = pattern[:, head_idx, :, :].sum(dim=-1, keepdim=True).clamp(min=1e-8)
    pattern[:, head_idx, :, :] = pattern[:, head_idx, :, :] / row_sums
    return pattern

print("\nAttention redistribution for L10H9 (last position):")
print("%-15s %8s → %8s  %8s → %8s" % (
    "Occupation", "BOS(orig)", "BOS(mod)", "Occ(orig)", "Occ(mod)"))
print("-" * 60)

for occ in OCCUPATIONS[:10]:
    prompt = TEMPLATE % occ
    tokens = model.to_tokens(prompt)

    # Original attention
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    orig_attn = cache["blocks.10.attn.hook_pattern"][0, 9, -1, :].cpu().numpy()

    # Modified (BOS zeroed)
    modified_attn = orig_attn.copy()
    modified_attn[0] = 0
    if modified_attn.sum() > 0:
        modified_attn = modified_attn / modified_attn.sum()

    print("%-15s %7.1f%%    %7.1f%%   %7.1f%%    %7.1f%%" % (
        occ, orig_attn[0]*100, modified_attn[0]*100,
        orig_attn[2]*100, modified_attn[2]*100))


# ═══════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Per-occupation bias (baseline vs ablated)
ax = axes[0, 0]
occs = [o for o, _ in sorted_occ]
baseline_biases = [occ_results[o]["baseline_bias"] for o in occs]
ablated_biases = [occ_results[o]["ablated_bias"] for o in occs]
x = np.arange(len(occs))
ax.barh(x - 0.2, baseline_biases, 0.4, label='Baseline', color='steelblue', alpha=0.7)
ax.barh(x + 0.2, ablated_biases, 0.4, label='L10H9 ablated', color='coral', alpha=0.7)
ax.set_yticks(x)
ax.set_yticklabels(occs, fontsize=7)
ax.set_xlabel("Gender Bias (P(M)-P(F))")
ax.set_title("Per-Occupation Bias: Baseline vs L10H9 Ablated")
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
ax.legend(fontsize=8)

# Plot 2: L10H9 effect by occupation
ax = axes[0, 1]
effects = [occ_results[o]["l10h9_effect"] for o in occs]
colors = ['royalblue' if e > 0 else 'hotpink' for e in effects]
ax.barh(x, effects, color=colors, alpha=0.7)
ax.set_yticks(x)
ax.set_yticklabels(occs, fontsize=7)
ax.set_xlabel("L10H9 Effect on Bias")
ax.set_title("L10H9 Effect: >0 pushes male, <0 pushes female")
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

# Plot 3: Occupation position patching by layer
ax = axes[1, 0]
for pr in patching_results:
    effects_list = [pr["layer_effects"][l] for l in range(n_layers)]
    ax.plot(range(n_layers), effects_list, alpha=0.3, linewidth=1)
mean_effects = [np.mean([pr["layer_effects"][l] for pr in patching_results]) for l in range(n_layers)]
ax.plot(range(n_layers), mean_effects, 'k-', linewidth=2, label='Mean')
ax.set_xlabel("Patch Layer")
ax.set_ylabel("Bias Shift")
ax.set_title("Occupation Patching Effect by Layer")
ax.legend()
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

# Plot 4: Occupation gender projection vs output bias
ax = axes[1, 1]
for occ in OCCUPATIONS:
    color = 'hotpink' if occ_results[occ]["baseline_bias"] < 0 else 'royalblue'
    ax.scatter(occ_attn_data[occ]["occ_gender_proj"], occ_results[occ]["baseline_bias"],
               color=color, s=40, alpha=0.7)
    ax.annotate(occ, (occ_attn_data[occ]["occ_gender_proj"], occ_results[occ]["baseline_bias"]),
                fontsize=6, ha='center', va='bottom')
ax.set_xlabel("L10H9 Occupation Position Gender Projection")
ax.set_ylabel("Output Gender Bias")
ax.set_title("L10H9 Reads Stereotypes from Occupation (r=%.3f)" % corr)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

plt.suptitle("L10H9 Occupation-Position Mechanism", fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "occupation_patching.png", dpi=150)
plt.close()

# ── Save ──
results = {
    "per_occupation": occ_results,
    "patching_by_layer": [
        {k: v for k, v in pr.items()} for pr in patching_results
    ],
    "l10h9_attention": occ_attn_data,
    "occ_proj_vs_bias_correlation": float(corr),
}

with open(RESULTS_DIR / "occupation_patching_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to %s" % RESULTS_DIR)
print("\n✓ Experiment 17 complete.")
