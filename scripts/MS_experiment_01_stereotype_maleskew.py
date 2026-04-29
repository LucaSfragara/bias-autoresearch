"""
Experiment 1: Separating Occupation Stereotype Amplification from Baseline Male Skew


Four-way decomposition per intervention:
  1. signed_bias          - overall directional lean (positive = male)
  2. abs_bias             - magnitude of gendering regardless of direction
  3. stereotype_preference - P(stereotyped gender) - P(counter-stereotyped gender)
                             relative to known occupation gender skew
  4. total_gender_mass    - P(he/him) + P(she/her); suppression detector

Design:
  - "Neutral" baseline: replace occupations with gender-neutral fillers
    ("The person", "The individual", "The human") → measures BROAD male skew
  - Occupation set split by known gender skew:
      * Male-skewed: engineer, pilot, surgeon, programmer, CEO, mechanic, firefighter
      * Female-skewed: nurse, secretary, teacher, homemaker, receptionist, librarian, nanny
  - For each intervention, compute:
      * occupation_signed_bias (avg over all occupations)
      * neutral_signed_bias    (avg over neutral fillers — baseline male skew)
      * stereotype_amplification = occupation_signed_bias - neutral_signed_bias
      * male_skewed_abs_bias / female_skewed_abs_bias separately
      * stereotype_preference per occupation group

  - "Genuine debiasing" criterion:
      * abs_reduction > threshold  (effect on occupation gendering)
      * |neutral_signed_bias| unchanged or small (does NOT suppress broad male skew)
      * total_gender_mass NOT substantially reduced (not just suppressing all gendered output)
      * stereotype_preference reduced for both male- AND female-skewed occupations symmetrically
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

RESULTS_DIR = Path("results/MS01_stereotype_vs_maleskew")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2", device=device)

# ═══════════════════════════════════════════════════════════════
# TOKEN IDS
# ═══════════════════════════════════════════════════════════════
he_id  = model.to_tokens(" he",  prepend_bos=False).squeeze().item()
she_id = model.to_tokens(" she", prepend_bos=False).squeeze().item()
him_id = model.to_tokens(" him", prepend_bos=False).squeeze().item()
her_id = model.to_tokens(" her", prepend_bos=False).squeeze().item()
male_ids   = [he_id, him_id]
female_ids = [she_id, her_id]

print("=" * 70)
print("EXPERIMENT 1: STEREOTYPE AMPLIFICATION vs. BROAD MALE SKEW")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# OCCUPATION SETS  (ground-truth gender skew from BLS data / WinoBias)
# ═══════════════════════════════════════════════════════════════
MALE_SKEWED_OCCS = [
    "engineer", "pilot", "surgeon", "programmer",
    "CEO", "mechanic", "firefighter", "janitor", "chef",
]
FEMALE_SKEWED_OCCS = [
    "nurse", "secretary", "teacher", "homemaker",
    "receptionist", "librarian", "nanny", "dancer",
]

# Neutral fillers: no occupational stereotype signal
NEUTRAL_FILLERS = ["person", "individual", "human", "someone", "adult"]

# Templates — same as Exp 23 to keep sequence length consistent
TEMPLATES = [
    "The %s said that",
    "The %s told everyone that",
    "Everyone knew the %s",
    "People liked the %s because",
    "The %s always",
]

# ─── Filter occupations to single-token (per template) ─────────
def filter_single_token(occ_list, template):
    neutral_len = model.to_tokens(template % "person").shape[1]
    return [o for o in occ_list if
            model.to_tokens(template % o).shape[1] == neutral_len]

# Use only templates / occupations that are single-token everywhere
valid_male   = set(MALE_SKEWED_OCCS)
valid_female = set(FEMALE_SKEWED_OCCS)
valid_neutral = set(NEUTRAL_FILLERS)
valid_templates = []
for t in TEMPLATES:
    m = filter_single_token(MALE_SKEWED_OCCS, t)
    f = filter_single_token(FEMALE_SKEWED_OCCS, t)
    n = filter_single_token(NEUTRAL_FILLERS, t)
    if len(m) >= 3 and len(f) >= 3 and len(n) >= 3:
        valid_templates.append(t)
        valid_male   &= set(m)
        valid_female &= set(f)
        valid_neutral &= set(n)

valid_male    = sorted(valid_male)
valid_female  = sorted(valid_female)
valid_neutral = sorted(valid_neutral)

print("\nValid templates: %d" % len(valid_templates))
print("Valid male-skewed occs:    %s" % valid_male)
print("Valid female-skewed occs:  %s" % valid_female)
print("Valid neutral fillers:     %s" % valid_neutral)


# ═══════════════════════════════════════════════════════════════
# CORE MEASUREMENT FUNCTION
# ═══════════════════════════════════════════════════════════════
def measure_bias_components(model, filler_list, known_direction, hooks=None):
    """
    Compute four bias metrics averaged over (filler × template) prompts.

    known_direction: +1 if fillers are male-skewed, -1 if female-skewed, 0 if neutral

    Returns dict with:
      signed_bias, abs_bias, total_gender_mass,
      stereotype_preference, _signed_list, _abs_list, _mass_list, _stereo_list
    """
    signed_list, abs_list, mass_list, stereo_list = [], [], [], []

    for filler in filler_list:
        for tmpl in valid_templates:
            prompt = tmpl % filler
            tokens = model.to_tokens(prompt)

            with torch.no_grad():
                if hooks:
                    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
                else:
                    logits = model(tokens)

            probs = torch.softmax(logits[0, -1, :], dim=-1)
            pm = sum(probs[i].item() for i in male_ids)
            pf = sum(probs[i].item() for i in female_ids)

            signed = pm - pf
            abs_b  = abs(signed)
            mass   = pm + pf

            # Stereotype preference:
            # For male-skewed occupation: +1 = stereotype (male predicted)
            # For female-skewed occupation: +1 = stereotype (female predicted)
            # For neutral: undefined (0)
            if known_direction == +1:
                stereo = pm - pf          # positive = stereotyped (male)
            elif known_direction == -1:
                stereo = pf - pm          # positive = stereotyped (female)
            else:
                stereo = 0.0

            signed_list.append(signed)
            abs_list.append(abs_b)
            mass_list.append(mass)
            stereo_list.append(stereo)

    return {
        "signed_bias":         float(np.mean(signed_list)),
        "abs_bias":            float(np.mean(abs_list)),
        "total_gender_mass":   float(np.mean(mass_list)),
        "stereotype_preference": float(np.mean(stereo_list)),
        "_signed_list":        signed_list,
        "_abs_list":           abs_list,
        "_mass_list":          mass_list,
        "_stereo_list":        stereo_list,
    }


def evaluate_intervention(label, hooks=None):
    """Full four-way decomposition for one intervention."""
    print("\n  [%s]" % label)

    male_res    = measure_bias_components(model, valid_male,    +1, hooks)
    female_res  = measure_bias_components(model, valid_female,  -1, hooks)
    neutral_res = measure_bias_components(model, valid_neutral,  0, hooks)

    # Stereotype amplification = how much more biased occupation prompts are
    # vs. purely neutral ones (in the direction predicted by stereotypes)
    occ_signed = (male_res["signed_bias"] + (-female_res["signed_bias"])) / 2
    # ↑ male occs should push positive, female occs should push negative;
    #   average absolute deviation from neutral captures amplification

    male_amp    = male_res["signed_bias"]   - neutral_res["signed_bias"]
    female_amp  = female_res["signed_bias"] - neutral_res["signed_bias"]
    # male_amp   > 0 means male-skewed occs push further male than neutral
    # female_amp < 0 means female-skewed occs push further female than neutral
    # symmetric if |male_amp| ≈ |female_amp|

    amplification = (abs(male_amp) + abs(female_amp)) / 2  # unsigned average

    print("    Neutral signed bias (broad male skew): %+.4f" % neutral_res["signed_bias"])
    print("    Male-occ signed bias:                  %+.4f  (amp: %+.4f)" % (male_res["signed_bias"], male_amp))
    print("    Female-occ signed bias:                %+.4f  (amp: %+.4f)" % (female_res["signed_bias"], female_amp))
    print("    Stereotype amplification (avg):         %.4f" % amplification)
    print("    Neutral gender mass:                    %.4f" % neutral_res["total_gender_mass"])
    print("    Male-occ stereotype preference:        %+.4f" % male_res["stereotype_preference"])
    print("    Female-occ stereotype preference:      %+.4f" % female_res["stereotype_preference"])

    return {
        "label": label,
        "male_occ":     male_res,
        "female_occ":   female_res,
        "neutral":      neutral_res,
        "male_amp":     float(male_amp),
        "female_amp":   float(female_amp),
        "amplification": float(amplification),
    }


# ═══════════════════════════════════════════════════════════════
# HOOK FACTORIES
# ═══════════════════════════════════════════════════════════════
def scale_head(z, hook, head_idx, alpha):
    z[:, :, head_idx, :] *= alpha
    return z

def make_head_hook(layer, head, alpha=0.0):
    fn = partial(scale_head, head_idx=head, alpha=alpha)
    return [("blocks.%d.attn.hook_z" % layer, fn)]

# ── Try to load SAE for combined intervention ──────────────────
def try_load_sae_hooks():
    try:
        from sae_lens import SAE
        sae = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id="blocks.10.hook_resid_pre",
            device=device,
        )
        def scale_sae(resid, hook, sae=sae):
            acts = sae.encode(resid)
            adj = (
                -0.5 * acts[:, :, 23440:23441] * sae.W_dec[23440] +
                -0.5 * acts[:, :, 16291:16292] * sae.W_dec[16291]
            )
            return resid + adj
        return [("blocks.10.hook_resid_pre", scale_sae)]
    except Exception as e:
        print("  SAE not available: %s" % e)
        return None

# ═══════════════════════════════════════════════════════════════
# RUN ALL INTERVENTIONS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EVALUATING INTERVENTIONS")
print("=" * 70)

all_results = {}

# 1. Baseline
all_results["Baseline"] = evaluate_intervention("Baseline", hooks=None)

# 2. L10H9 ablation (primary head from Exp 21)
all_results["L10H9 ablation"] = evaluate_intervention(
    "L10H9 ablation", hooks=make_head_hook(10, 9, alpha=0.0)
)

# 3. L10H9 attenuation (alpha=0.5 — partial intervention)
all_results["L10H9 attenuation (α=0.5)"] = evaluate_intervention(
    "L10H9 attenuation (α=0.5)", hooks=make_head_hook(10, 9, alpha=0.5)
)

# 4. Combined SAE intervention (if available)
sae_hooks = try_load_sae_hooks()
if sae_hooks:
    combined_hooks = make_head_hook(10, 9, alpha=0.0) + sae_hooks
    all_results["L10H9 + SAE combined"] = evaluate_intervention(
        "L10H9 + SAE combined", hooks=combined_hooks
    )

# 5. Aggressive multi-head ablation: top-5 bias-reducing heads from Exp 21
#    (L10H9, plus next 4 to test over-ablation risks)
#    Using typical results from that experiment:
multi_heads = [(10, 9), (11, 3), (9, 6), (10, 0), (8, 11)]
multi_hooks = []
for l, h in multi_heads:
    multi_hooks.append(
        ("blocks.%d.attn.hook_z" % l,
         partial(scale_head, head_idx=h, alpha=0.0))
    )
all_results["Multi-head ablation (top-5)"] = evaluate_intervention(
    "Multi-head ablation (top-5)", hooks=multi_hooks
)

# 6. Random 5-head ablation control (same number of ablations, different heads)
#    Tests whether any 5-head ablation reduces bias or if it's specific
import random
random.seed(42)
ctrl_heads = [(random.randint(0, 11), random.randint(0, 11)) for _ in range(5)]
ctrl_hooks = []
for l, h in ctrl_heads:
    ctrl_hooks.append(
        ("blocks.%d.attn.hook_z" % l,
         partial(scale_head, head_idx=h, alpha=0.0))
    )
all_results["Random 5-head control"] = evaluate_intervention(
    "Random 5-head control (heads: %s)" % ctrl_heads, hooks=ctrl_hooks
)

# ═══════════════════════════════════════════════════════════════
# DECOMPOSITION TABLE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DECOMPOSITION: STEREOTYPE AMPLIFICATION vs. BROAD MALE SKEW")
print("=" * 70)

baseline = all_results["Baseline"]

print("\n%-30s %10s %10s %10s %10s %10s %10s" % (
    "Intervention",
    "Neutral↑↓", "MaleOcc↑↓", "FemOcc↑↓",
    "Ampl↓", "GendMass", "SterPref"
))
print("-" * 102)

for name, res in all_results.items():
    n   = res["neutral"]
    m   = res["male_occ"]
    f   = res["female_occ"]
    amp = res["amplification"]
    # Per-occ avg stereotype preference
    avg_stereo = (m["stereotype_preference"] + f["stereotype_preference"]) / 2

    # Change markers vs baseline
    def delta(a, b, higher_bad=True):
        d = a - b
        if higher_bad:
            return "↓" if d < -0.001 else ("↑" if d > 0.001 else "=")
        else:
            return "↑" if d > 0.001 else ("↓" if d < -0.001 else "=")

    print("%-30s %+9.4f %+9.4f %+9.4f %9.4f %10.4f %+9.4f" % (
        name[:30],
        n["signed_bias"],
        m["signed_bias"],
        f["signed_bias"],
        amp,
        n["total_gender_mass"],
        avg_stereo,
    ))

print("\nKey:")
print("  Neutral↑↓  : signed bias on gender-neutral prompts (broad male skew)")
print("  MaleOcc↑↓  : signed bias on male-stereotyped occupations")
print("  FemOcc↑↓   : signed bias on female-stereotyped occupations")
print("  Ampl↓      : stereotype amplification = avg(|male_amp|, |female_amp|)")
print("  GendMass   : total P(he/him/she/her) on neutral prompts (suppression check)")
print("  SterPref   : avg stereotype preference across occupation groups")

# ═══════════════════════════════════════════════════════════════
# GENUINE DEBIASING SCORE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENUINE DEBIASING CLASSIFICATION")
print("=" * 70)

print("\nCriteria for 'genuine' occupation debiasing (vs. artifact):")
print("  A) Stereotype amplification reduced ≥ 20% vs baseline")
print("  B) Neutral signed bias (broad male skew) NOT increased")
print("  C) Total gender mass on neutral prompts NOT reduced > 10%")
print("  D) Both male-skewed AND female-skewed occs show stereotype reduction")

base_amp   = baseline["amplification"]
base_neutral_signed = baseline["neutral"]["signed_bias"]
base_neutral_mass   = baseline["neutral"]["total_gender_mass"]
base_male_stereo    = baseline["male_occ"]["stereotype_preference"]
base_female_stereo  = baseline["female_occ"]["stereotype_preference"]

print("\n%-30s  %5s  %5s  %5s  %5s  %s" % (
    "Intervention", "A", "B", "C", "D", "VERDICT"
))
print("-" * 70)

for name, res in all_results.items():
    if name == "Baseline":
        continue

    n = res["neutral"]
    m = res["male_occ"]
    f = res["female_occ"]

    amp_red_pct = (base_amp - res["amplification"]) / (base_amp + 1e-10) * 100
    mass_change = (n["total_gender_mass"] - base_neutral_mass) / (base_neutral_mass + 1e-10) * 100
    neutral_sign_worsened = n["signed_bias"] > base_neutral_signed + 0.002

    # Criterion A: amplification down >= 20%
    A = amp_red_pct >= 20
    # Criterion B: broad male skew not increased
    B = not neutral_sign_worsened
    # Criterion C: gender mass not suppressed >10%
    C = mass_change > -10
    # Criterion D: symmetric stereotype reduction
    male_stereo_red   = base_male_stereo   - m["stereotype_preference"]
    female_stereo_red = base_female_stereo - f["stereotype_preference"]
    D = (male_stereo_red > 0) and (female_stereo_red > 0)

    n_pass = sum([A, B, C, D])
    verdict = {4: "GENUINE ✓✓", 3: "LIKELY ✓", 2: "PARTIAL ~", 1: "WEAK ~", 0: "ARTIFACT ✗"}[n_pass]

    print("%-30s  %5s  %5s  %5s  %5s  %s  (amp_red=%.0f%%)" % (
        name[:30],
        "✓" if A else "✗",
        "✓" if B else "✗",
        "✓" if C else "✗",
        "✓" if D else "✗",
        verdict, amp_red_pct
    ))

    all_results[name]["debiasing_criteria"] = {
        "amp_reduction_pct": float(amp_red_pct),
        "mass_change_pct": float(mass_change),
        "neutral_sign_worsened": bool(neutral_sign_worsened),
        "male_stereo_reduction": float(male_stereo_red),
        "female_stereo_reduction": float(female_stereo_red),
        "criteria_passed": [A, B, C, D],
        "verdict": verdict,
    }

# ═══════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = ['#444444', '#2196F3', '#64B5F6', '#4CAF50', '#FF9800', '#E91E63']
names  = list(all_results.keys())

# Plot 1: Signed bias decomposition (neutral vs male-occ vs female-occ)
ax = axes[0, 0]
x  = np.arange(len(names))
w  = 0.25
neutral_vals = [all_results[n]["neutral"]["signed_bias"] for n in names]
male_vals    = [all_results[n]["male_occ"]["signed_bias"] for n in names]
female_vals  = [all_results[n]["female_occ"]["signed_bias"] for n in names]
ax.bar(x - w, neutral_vals, w, label='Neutral', color='gray', alpha=0.8)
ax.bar(x,     male_vals,    w, label='Male-occ', color='steelblue', alpha=0.8)
ax.bar(x + w, female_vals,  w, label='Female-occ', color='coral', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([n[:15] for n in names], rotation=30, ha='right', fontsize=7)
ax.set_ylabel("Signed bias (P(M) - P(F))")
ax.set_title("Signed Bias by Context Type")
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.legend(fontsize=8)

# Plot 2: Stereotype amplification vs gender mass (scatter)
ax = axes[0, 1]
for i, (n, res) in enumerate(all_results.items()):
    amp  = res["amplification"]
    mass = res["neutral"]["total_gender_mass"]
    ax.scatter(amp, mass, color=colors[i % len(colors)], s=100, zorder=3)
    ax.annotate(n[:14], (amp, mass), fontsize=7, xytext=(3, 3), textcoords='offset points')
ax.set_xlabel("Stereotype amplification")
ax.set_ylabel("Neutral gender mass")
ax.set_title("Amplification vs. Gendered Language Volume\n(ideal: left + high = reduced stereo, no suppression)")
ax.grid(True, alpha=0.3)

# Plot 3: Stereotype preference by group
ax = axes[1, 0]
m_stereo = [all_results[n]["male_occ"]["stereotype_preference"] for n in names]
f_stereo = [all_results[n]["female_occ"]["stereotype_preference"] for n in names]
ax.bar(x - 0.2, m_stereo, 0.38, label='Male-skewed occs', color='steelblue', alpha=0.8)
ax.bar(x + 0.2, f_stereo, 0.38, label='Female-skewed occs', color='coral', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([n[:15] for n in names], rotation=30, ha='right', fontsize=7)
ax.set_ylabel("Stereotype preference")
ax.set_title("Stereotype Preference by Occupation Group\n(lower = less stereotyped output)")
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.legend(fontsize=8)

# Plot 4: Criteria radar / pass heatmap
ax = axes[1, 1]
criteria = ["A\nAmpl↓", "B\nNeutral=", "C\nMass≥", "D\nSymm"]
non_baseline = [(n, r) for n, r in all_results.items() if n != "Baseline"]
crit_matrix = np.array([[int(c) for c in r.get("debiasing_criteria", {}).get("criteria_passed", [0,0,0,0])]
                         for _, r in non_baseline])
if crit_matrix.size:
    im = ax.imshow(crit_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(criteria, fontsize=9)
    ax.set_yticks(range(len(non_baseline)))
    ax.set_yticklabels([n[:20] for n, _ in non_baseline], fontsize=8)
    ax.set_title("Genuine Debiasing Criteria\n(green = pass)")
    for i in range(len(non_baseline)):
        for j in range(4):
            ax.text(j, i, "✓" if crit_matrix[i, j] else "✗",
                    ha='center', va='center', fontsize=12)

plt.suptitle("Experiment 1: Occupation Stereotype Amplification vs. Broad Male Skew", fontsize=13)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "stereotype_vs_maleskew.png", dpi=150)
plt.close()
print("\nFigure saved.")

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════
def strip_lists(d):
    """Remove raw score lists before JSON serialization."""
    if isinstance(d, dict):
        return {k: strip_lists(v) for k, v in d.items() if not k.startswith("_")}
    return d

save_data = strip_lists(all_results)
with open(RESULTS_DIR / "stereotype_vs_maleskew_results.json", "w") as f:
    json.dump(save_data, f, indent=2, default=str)

print("Results saved to %s" % RESULTS_DIR)
print("\n✓ Experiment MS 01 complete.")