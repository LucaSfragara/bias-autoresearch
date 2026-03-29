"""
Experiment 23: Proper Path Patching

Replaces gradient-based EAP (exp 10) with actual activation patching.
Tests:
1. L10H9 SUFFICIENCY: Patch L10H9 output from clean→corrupted, measure bias recovery
2. L10H9 NECESSITY: In clean run, corrupt L10H9 output, measure bias loss
3. UPSTREAM SOURCES: For each head in L0-L9, patch clean→corrupted, measure bias restoration
4. MEDIATION: Which upstream heads contribute through L10H9?

Clean = occupation prompts ("The nurse said that")
Corrupted = neutral prompts ("The person said that")
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

RESULTS_DIR = Path("results/23_path_patching")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2", device=device)

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

# Token IDs
he_id = model.to_tokens(" he", prepend_bos=False).squeeze().item()
she_id = model.to_tokens(" she", prepend_bos=False).squeeze().item()
him_id = model.to_tokens(" him", prepend_bos=False).squeeze().item()
her_id = model.to_tokens(" her", prepend_bos=False).squeeze().item()
male_ids = [he_id, him_id]
female_ids = [she_id, her_id]


def gender_bias(logits):
    """Signed bias: P(male) - P(female)."""
    probs = torch.softmax(logits, dim=-1)
    pm = sum(probs[i].item() for i in male_ids)
    pf = sum(probs[i].item() for i in female_ids)
    return pm - pf


# ═══════════════════════════════════════════════════════════════
# PROMPTS: Clean (occupation) vs Corrupted (neutral)
# ═══════════════════════════════════════════════════════════════
OCCUPATIONS = [
    "nurse", "doctor", "engineer", "secretary", "CEO",
    "teacher", "programmer", "homemaker", "mechanic", "receptionist",
    "pilot", "librarian", "surgeon", "nanny", "janitor",
    "plumber", "dancer", "firefighter", "chef", "dentist",
]

TEMPLATE = "The %s said that"

# Filter to single-token occupations (same sequence length as "person")
person_len = model.to_tokens(TEMPLATE % "person").shape[1]
valid_occs = []
for occ in OCCUPATIONS:
    if model.to_tokens(TEMPLATE % occ).shape[1] == person_len:
        valid_occs.append(occ)

print("Valid single-token occupations: %d / %d" % (len(valid_occs), len(OCCUPATIONS)))
print("  Occupations: %s" % ", ".join(valid_occs))

# ═══════════════════════════════════════════════════════════════
# CACHE CLEAN AND CORRUPTED ACTIVATIONS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CACHING CLEAN AND CORRUPTED ACTIVATIONS")
print("=" * 70)

clean_caches = []
corrupted_caches = []
clean_biases = []
corrupted_biases = []

for occ in valid_occs:
    clean_prompt = TEMPLATE % occ
    corrupted_prompt = TEMPLATE % "person"

    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

    clean_biases.append(gender_bias(clean_logits[0, -1, :]))
    corrupted_biases.append(gender_bias(corrupted_logits[0, -1, :]))
    clean_caches.append(clean_cache)
    corrupted_caches.append(corrupted_cache)

mean_clean_bias = np.mean(clean_biases)
mean_corrupted_bias = np.mean(corrupted_biases)
total_effect = mean_clean_bias - mean_corrupted_bias
print("Mean clean bias:     %+.6f" % mean_clean_bias)
print("Mean corrupted bias: %+.6f" % mean_corrupted_bias)
print("Total effect (clean - corrupted): %+.6f" % total_effect)

# ═══════════════════════════════════════════════════════════════
# TEST 1: L10H9 SUFFICIENCY
# Patch ONLY L10H9 output from clean into corrupted run.
# If L10H9 is sufficient, this should restore most of the bias.
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 1: L10H9 SUFFICIENCY (clean L10H9 → corrupted)")
print("=" * 70)

sufficiency_biases = []
for i, occ in enumerate(valid_occs):
    corrupted_tokens = model.to_tokens(TEMPLATE % "person")
    clean_z = clean_caches[i]["blocks.10.attn.hook_z"][0, :, 9, :].clone()

    def patch_l10h9(z, hook, clean_act=clean_z):
        z[:, :, 9, :] = clean_act
        return z

    hooks = [("blocks.10.attn.hook_z", patch_l10h9)]
    with torch.no_grad():
        patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=hooks)
    sufficiency_biases.append(gender_bias(patched_logits[0, -1, :]))

mean_sufficiency = np.mean(sufficiency_biases)
recovery_pct = (mean_sufficiency - mean_corrupted_bias) / (total_effect + 1e-10) * 100
print("After patching L10H9: bias = %+.6f" % mean_sufficiency)
print("Recovery: %.1f%% of total effect" % recovery_pct)
print("Interpretation: L10H9 alone %s the occupation-specific bias" % (
    "RECOVERS most of" if recovery_pct > 50 else
    "partially recovers" if recovery_pct > 20 else
    "does NOT recover"))

# ═══════════════════════════════════════════════════════════════
# TEST 2: L10H9 NECESSITY
# In clean run, corrupt ONLY L10H9 with its corrupted-run output.
# If L10H9 is necessary, this should eliminate the bias.
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: L10H9 NECESSITY (corrupted L10H9 → clean)")
print("=" * 70)

necessity_biases = []
for i, occ in enumerate(valid_occs):
    clean_tokens = model.to_tokens(TEMPLATE % occ)
    corrupted_z = corrupted_caches[i]["blocks.10.attn.hook_z"][0, :, 9, :].clone()

    def patch_l10h9_corrupt(z, hook, corr_act=corrupted_z):
        z[:, :, 9, :] = corr_act
        return z

    hooks = [("blocks.10.attn.hook_z", patch_l10h9_corrupt)]
    with torch.no_grad():
        patched_logits = model.run_with_hooks(clean_tokens, fwd_hooks=hooks)
    necessity_biases.append(gender_bias(patched_logits[0, -1, :]))

mean_necessity = np.mean(necessity_biases)
removal_pct = (mean_clean_bias - mean_necessity) / (total_effect + 1e-10) * 100
print("After corrupting L10H9: bias = %+.6f" % mean_necessity)
print("Removal: %.1f%% of total effect" % removal_pct)
print("Interpretation: Corrupting L10H9 %s the occupation-specific bias" % (
    "ELIMINATES most of" if removal_pct > 50 else
    "partially removes" if removal_pct > 20 else
    "does NOT remove"))

# ═══════════════════════════════════════════════════════════════
# TEST 3: UPSTREAM HEAD PATCHING
# For each head in L0-L9, patch clean output into corrupted run,
# measure how much bias is restored at the final output.
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 3: UPSTREAM HEAD PATCHING (clean head → corrupted)")
print("=" * 70)

upstream_effects = {}
for layer in range(10):  # L0 through L9
    for head in range(n_heads):
        name = "L%dH%d" % (layer, head)
        patched_biases = []

        for i, occ in enumerate(valid_occs):
            corrupted_tokens = model.to_tokens(TEMPLATE % "person")
            clean_z = clean_caches[i]["blocks.%d.attn.hook_z" % layer][0, :, head, :].clone()

            def patch_head(z, hook, h=head, act=clean_z):
                z[:, :, h, :] = act
                return z

            hooks = [("blocks.%d.attn.hook_z" % layer, patch_head)]
            with torch.no_grad():
                patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=hooks)
            patched_biases.append(gender_bias(patched_logits[0, -1, :]))

        mean_patched = np.mean(patched_biases)
        recovery = (mean_patched - mean_corrupted_bias) / (total_effect + 1e-10) * 100
        upstream_effects[name] = {
            "layer": layer,
            "head": head,
            "patched_bias": float(mean_patched),
            "recovery_pct": float(recovery),
        }

    print("  Layer %d complete" % layer)

# Sort by recovery
sorted_upstream = sorted(upstream_effects.items(), key=lambda x: abs(x[1]["recovery_pct"]), reverse=True)
print("\nTop 15 upstream heads by |recovery|:")
print("%-10s %12s" % ("Head", "Recovery%"))
print("-" * 24)
for name, d in sorted_upstream[:15]:
    print("%-10s %+11.1f%%" % (name, d["recovery_pct"]))

# ═══════════════════════════════════════════════════════════════
# TEST 4: MEDIATION THROUGH L10H9
# For top upstream heads, check if their effect is mediated through L10H9.
# Patch upstream head (clean→corrupted) AND simultaneously corrupt L10H9.
# If the upstream effect disappears, it's mediated through L10H9.
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 4: MEDIATION THROUGH L10H9")
print("=" * 70)

top_upstream = [n for n, d in sorted_upstream[:10] if abs(d["recovery_pct"]) > 2]

mediation_results = {}
for up_name in top_upstream:
    up = upstream_effects[up_name]
    up_layer, up_head = up["layer"], up["head"]

    direct_biases = []  # upstream + L10H9 corrupted
    for i, occ in enumerate(valid_occs):
        corrupted_tokens = model.to_tokens(TEMPLATE % "person")
        clean_up_z = clean_caches[i]["blocks.%d.attn.hook_z" % up_layer][0, :, up_head, :].clone()
        corrupted_l10h9_z = corrupted_caches[i]["blocks.10.attn.hook_z"][0, :, 9, :].clone()

        def patch_upstream(z, hook, h=up_head, act=clean_up_z):
            z[:, :, h, :] = act
            return z

        def corrupt_l10h9(z, hook, act=corrupted_l10h9_z):
            z[:, :, 9, :] = act
            return z

        hooks = [
            ("blocks.%d.attn.hook_z" % up_layer, patch_upstream),
            ("blocks.10.attn.hook_z", corrupt_l10h9),
        ]
        with torch.no_grad():
            patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=hooks)
        direct_biases.append(gender_bias(patched_logits[0, -1, :]))

    mean_direct = np.mean(direct_biases)
    direct_recovery = (mean_direct - mean_corrupted_bias) / (total_effect + 1e-10) * 100
    total_recovery = up["recovery_pct"]
    mediated_pct = (1 - direct_recovery / (total_recovery + 1e-10)) * 100 if abs(total_recovery) > 0.5 else 0

    mediation_results[up_name] = {
        "total_recovery": float(total_recovery),
        "direct_recovery": float(direct_recovery),
        "mediated_pct": float(mediated_pct),
    }
    print("  %-10s total: %+.1f%%, direct (bypassing L10H9): %+.1f%%, mediated: %.0f%%" % (
        up_name, total_recovery, direct_recovery, mediated_pct))

# ═══════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: Sufficiency + Necessity bar chart
ax = axes[0]
labels = ["Corrupted\n(person)", "Sufficiency\n(patch L10H9)", "Clean\n(occupation)", "Necessity\n(corrupt L10H9)"]
values = [mean_corrupted_bias, mean_sufficiency, mean_clean_bias, mean_necessity]
colors = ['gray', 'steelblue', 'green', 'coral']
ax.bar(range(4), values, color=colors, alpha=0.7)
ax.set_xticks(range(4))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Mean signed bias")
ax.set_title("L10H9 Sufficiency & Necessity")
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

# Plot 2: Upstream head effects heatmap
ax = axes[1]
effect_matrix = np.zeros((10, n_heads))
for name, d in upstream_effects.items():
    effect_matrix[d["layer"], d["head"]] = d["recovery_pct"]
im = ax.imshow(effect_matrix, aspect='auto', cmap='RdBu_r', vmin=-20, vmax=20)
ax.set_xlabel("Head")
ax.set_ylabel("Layer")
ax.set_title("Upstream Head Effects\n(% bias recovery when patched)")
plt.colorbar(im, ax=ax)

# Plot 3: Mediation
ax = axes[2]
if mediation_results:
    med_names = list(mediation_results.keys())[:8]
    total_vals = [mediation_results[n]["total_recovery"] for n in med_names]
    direct_vals = [mediation_results[n]["direct_recovery"] for n in med_names]
    x = np.arange(len(med_names))
    ax.bar(x - 0.2, total_vals, 0.4, label='Total effect', color='steelblue', alpha=0.7)
    ax.bar(x + 0.2, direct_vals, 0.4, label='Direct (bypass L10H9)', color='coral', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(med_names, rotation=45, fontsize=8)
    ax.set_ylabel("Recovery %")
    ax.set_title("Mediation Through L10H9")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

plt.suptitle("Path Patching: L10H9 Circuit Analysis", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "path_patching.png", dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════
save_data = {
    "occupations": valid_occs,
    "total_effect": float(total_effect),
    "mean_clean_bias": float(mean_clean_bias),
    "mean_corrupted_bias": float(mean_corrupted_bias),
    "sufficiency": {
        "patched_bias": float(mean_sufficiency),
        "recovery_pct": float(recovery_pct),
    },
    "necessity": {
        "patched_bias": float(mean_necessity),
        "removal_pct": float(removal_pct),
    },
    "upstream_top15": {n: upstream_effects[n] for n, _ in sorted_upstream[:15]},
    "mediation": mediation_results,
}

with open(RESULTS_DIR / "path_patching_results.json", "w") as f:
    json.dump(save_data, f, indent=2)

print("\nResults saved to %s" % RESULTS_DIR)
print("\n✓ Experiment 23 complete.")
