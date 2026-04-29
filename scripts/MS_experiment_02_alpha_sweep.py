"""
Experiment 2: Graduated Alpha Sweep + Multi-Metric Pareto Analysis

Systematically maps how each metric (amplification, residual skew, gender mass)
responds as L10H9 output is progressively attenuated from α=1.0 (intact) to
α=0.0 (fully ablated).

Key questions:
  Q1. Is there an α where amplification drops substantially before residual
      skew changes significantly? (= a "selective" debiasing regime)
  Q2. Does gender mass (gendered language volume) decline proportionally to
      amplification, or does it stay flat? (flat mass = genuine, not suppression)
  Q3. Is the dose-response curve monotone or does it have a threshold?

Also sweeps across all 12 heads in L10 simultaneously to identify which
exhibit the "selective" property and which merely suppress gendered output.

Additionally:
  - Pareto frontier: for each head, plot (amplification_reduction,
    mass_suppression) to find heads that reduce stereotype without suppression.
  - Cross-layer comparison: repeat the sweep for L10H9, L9H6, L11H3 (top
    three heads from Exp 21) and overlay their response curves.
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

RESULTS_DIR = Path("results/MS_02_alpha_sweep")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2", device=device)

he_id  = model.to_tokens(" he",  prepend_bos=False).squeeze().item()
she_id = model.to_tokens(" she", prepend_bos=False).squeeze().item()
him_id = model.to_tokens(" him", prepend_bos=False).squeeze().item()
her_id = model.to_tokens(" her", prepend_bos=False).squeeze().item()
male_ids   = [he_id, him_id]
female_ids = [she_id, her_id]

print("=" * 70)
print("EXPERIMENT 2: GRADUATED ALPHA SWEEP + PARETO ANALYSIS")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# OCCUPATION PAIRS + NEUTRAL 
# ═══════════════════════════════════════════════════════════════
TEMPLATE = "The %s said that"
neutral_len = model.to_tokens(TEMPLATE % "person").shape[1]

MALE_OCCS_ALL   = ["engineer", "pilot", "surgeon", "programmer",
                    "CEO", "mechanic", "firefighter", "janitor", "chef"]
FEMALE_OCCS_ALL = ["nurse", "secretary", "teacher", "homemaker",
                    "receptionist", "librarian", "nanny", "dancer"]
NEUTRAL_ALL     = ["person", "individual", "human"]

def valid_occs(lst):
    return [o for o in lst
            if model.to_tokens(TEMPLATE % o).shape[1] == neutral_len]

MALE_OCCS    = valid_occs(MALE_OCCS_ALL)
FEMALE_OCCS  = valid_occs(FEMALE_OCCS_ALL)
NEUTRAL_OCCS = valid_occs(NEUTRAL_ALL)

print("Male-skewed:   %s" % MALE_OCCS)
print("Female-skewed: %s" % FEMALE_OCCS)
print("Neutral:       %s" % NEUTRAL_OCCS)


# ═══════════════════════════════════════════════════════════════
# CORE METRIC FUNCTION
# ═══════════════════════════════════════════════════════════════
def compute_metrics(hooks=None):
    """Returns (amplification, residual_skew, gender_mass_occ, neutral_signed, neutral_mass)."""
    def sbias_mass(occ, hooks):
        tokens = model.to_tokens(TEMPLATE % occ)
        with torch.no_grad():
            if hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                logits = model(tokens)
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        pm = sum(probs[i].item() for i in male_ids)
        pf = sum(probs[i].item() for i in female_ids)
        return pm - pf, pm + pf

    m_biases, m_masses = zip(*[sbias_mass(o, hooks) for o in MALE_OCCS])
    f_biases, f_masses = zip(*[sbias_mass(o, hooks) for o in FEMALE_OCCS])
    n_biases, n_masses = zip(*[sbias_mass(o, hooks) for o in NEUTRAL_OCCS])

    bm, bf, bn = np.mean(m_biases), np.mean(f_biases), np.mean(n_biases)
    mass_occ = np.mean(list(m_masses) + list(f_masses))
    mass_n   = np.mean(n_masses)

    amp  = (bm - bf) / 2
    skew = (bm + bf) / 2

    return float(amp), float(skew), float(mass_occ), float(bn), float(mass_n)


def scale_head(z, hook, head_idx, alpha):
    z[:, :, head_idx, :] *= alpha
    return z


# ═══════════════════════════════════════════════════════════════
# BASELINE
# ═══════════════════════════════════════════════════════════════
base_amp, base_skew, base_mass, base_n_signed, base_n_mass = compute_metrics()
print("\nBaseline:")
print("  Amplification:   %.4f" % base_amp)
print("  Residual skew:   %.4f" % base_skew)
print("  Occ gender mass: %.4f" % base_mass)
print("  Neutral signed:  %.4f" % base_n_signed)
print("  Neutral mass:    %.4f" % base_n_mass)


# ═══════════════════════════════════════════════════════════════
# ALPHA SWEEP: PRIMARY HEADS
# ═══════════════════════════════════════════════════════════════
ALPHAS = np.linspace(1.0, 0.0, 21)   # 1.0, 0.95, ..., 0.0

PRIMARY_HEADS = [
    (10, 9,  "L10H9"),   # primary
    (11, 3,  "L11H3"),
    (9,  6,  "L9H6"),
    (10, 0,  "L10H0"),   # control: different L10 head
    (5,  5,  "L5H5"),    # mid-layer control
]

sweep_results = {}

print("\n" + "=" * 70)
print("ALPHA SWEEP")
print("=" * 70)

for layer, head, label in PRIMARY_HEADS:
    amps, skews, masses, n_signed = [], [], [], []

    for alpha in ALPHAS:
        fn = partial(scale_head, head_idx=head, alpha=float(alpha))
        hooks = [("blocks.%d.attn.hook_z" % layer, fn)]
        amp, skew, mass, ns, nm = compute_metrics(hooks)
        amps.append(amp)
        skews.append(skew)
        masses.append(mass)
        n_signed.append(ns)

    sweep_results[label] = {
        "layer": layer, "head": head,
        "alphas":   ALPHAS.tolist(),
        "amps":     amps,
        "skews":    skews,
        "masses":   masses,
        "n_signed": n_signed,
    }

    # Find "selective" regime: where amp drops >20% but skew changes <5%
    amp_thresh = base_amp * 0.80   # 20% below baseline
    skew_tol   = abs(base_skew) * 0.05
    selective_alphas = [
        ALPHAS[i] for i in range(len(ALPHAS))
        if amps[i] < amp_thresh and abs(skews[i] - base_skew) < skew_tol
    ]
    print("  %-8s: amp @ α=0: %.4f (Δ=%.0f%%)  skew @ α=0: %.4f  "
          "selective αs: %s" % (
        label,
        amps[-1], (base_amp - amps[-1]) / (base_amp + 1e-10) * 100,
        skews[-1],
        selective_alphas[:3] if selective_alphas else "None"
    ))


# ═══════════════════════════════════════════════════════════════
# ALL L10 HEADS: Pareto at α=0 (full ablation)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PARETO ANALYSIS: ALL L10 HEADS (α=0 ablation)")
print("=" * 70)

pareto = {}
for head in range(model.cfg.n_heads):
    fn = partial(scale_head, head_idx=head, alpha=0.0)
    hooks = [("blocks.10.attn.hook_z", fn)]
    amp, skew, mass, ns, nm = compute_metrics(hooks)

    amp_red  = (base_amp  - amp)  / (base_amp  + 1e-10) * 100
    mass_chg = (mass - base_mass) / (base_mass + 1e-10) * 100
    skew_chg = skew - base_skew

    pareto["L10H%d" % head] = {
        "amp_reduction_pct": float(amp_red),
        "mass_change_pct":   float(mass_chg),
        "skew_change":       float(skew_chg),
        "final_amp":         float(amp),
        "final_skew":        float(skew),
    }
    print("  L10H%-2d  amp_red=%+5.1f%%  mass_chg=%+5.1f%%  skew_Δ=%+.4f" % (
        head, amp_red, mass_chg, skew_chg))

# Pareto-dominant heads (high amp_red, low |mass_chg|, low skew worsening)
pareto_dominant = sorted(
    [(n, d) for n, d in pareto.items()
     if d["amp_reduction_pct"] > 10 and d["mass_change_pct"] > -10],
    key=lambda x: x[1]["amp_reduction_pct"],
    reverse=True
)
print("\nPareto-dominant L10 heads (>10% amp reduction, <10% mass suppression):")
for name, d in pareto_dominant:
    print("  %-8s  amp_red=%.1f%%  mass=%.1f%%  skew_Δ=%+.4f" % (
        name, d["amp_reduction_pct"], d["mass_change_pct"], d["skew_change"]))


# ═══════════════════════════════════════════════════════════════
# SELECTIVITY INDEX
# For each head, SI = (amp_reduction%) / (|skew_change| + |mass_change%| + 1)
# Higher SI = more specific to stereotype amplification
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SELECTIVITY INDEX (all L10 heads + top heads from other layers)")
print("=" * 70)

all_heads_si = {}
for name, d in pareto.items():
    si = d["amp_reduction_pct"] / (abs(d["skew_change"]) * 100 + abs(d["mass_change_pct"]) + 1)
    all_heads_si[name] = float(si)

# Also compute for top non-L10 heads
for layer, head, label in PRIMARY_HEADS[1:]:
    fn = partial(scale_head, head_idx=head, alpha=0.0)
    hooks = [("blocks.%d.attn.hook_z" % layer, fn)]
    amp, skew, mass, ns, nm = compute_metrics(hooks)
    amp_red  = (base_amp  - amp)  / (base_amp  + 1e-10) * 100
    mass_chg = (mass - base_mass) / (base_mass + 1e-10) * 100
    skew_chg = skew - base_skew
    si = amp_red / (abs(skew_chg) * 100 + abs(mass_chg) + 1)
    all_heads_si[label] = float(si)
    pareto[label] = {
        "amp_reduction_pct": float(amp_red),
        "mass_change_pct":   float(mass_chg),
        "skew_change":       float(skew_chg),
    }

sorted_si = sorted(all_heads_si.items(), key=lambda x: x[1], reverse=True)
print("%-10s  %10s" % ("Head", "Selectivity Index"))
print("-" * 22)
for name, si in sorted_si[:10]:
    print("%-10s  %10.3f" % (name, si))


# ═══════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Alpha sweep curves — amplification
ax = axes[0, 0]
for layer, head, label in PRIMARY_HEADS:
    r = sweep_results[label]
    amp_norm = [a / (base_amp + 1e-10) for a in r["amps"]]
    ax.plot(r["alphas"], amp_norm, marker='o', markersize=3, label=label)
ax.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.6)
ax.axhline(0.8, color='red', ls=':', lw=0.8, alpha=0.6, label='−20% threshold')
ax.set_xlabel("α (head scaling factor)")
ax.set_ylabel("Normalized amplification (1=baseline)")
ax.set_title("Stereotype Amplification vs. α")
ax.legend(fontsize=8)
ax.invert_xaxis()
ax.grid(True, alpha=0.3)

# Plot 2: Alpha sweep — residual skew (should be flat for genuine debiasing)
ax = axes[0, 1]
for layer, head, label in PRIMARY_HEADS:
    r = sweep_results[label]
    skew_norm = [s / (abs(base_skew) + 1e-10) for s in r["skews"]]
    ax.plot(r["alphas"], skew_norm, marker='o', markersize=3, label=label)
ax.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.6)
ax.set_xlabel("α (head scaling factor)")
ax.set_ylabel("Normalized residual skew (1=baseline)")
ax.set_title("Residual Male Skew vs. α\n(flat = no artifact; drifting = confound)")
ax.legend(fontsize=8)
ax.invert_xaxis()
ax.grid(True, alpha=0.3)

# Plot 3: Pareto scatter (amp_reduction vs mass_change for all L10 heads)
ax = axes[1, 0]
l10_heads = ["L10H%d" % h for h in range(model.cfg.n_heads)]
x_vals = [pareto[n]["amp_reduction_pct"] for n in l10_heads]
y_vals = [pareto[n]["mass_change_pct"]   for n in l10_heads]
colors = ['green' if x > 10 and y > -10 else 'gray' for x, y in zip(x_vals, y_vals)]
ax.scatter(x_vals, y_vals, color=colors, s=80, zorder=3)
for i, n in enumerate(l10_heads):
    ax.annotate(n, (x_vals[i], y_vals[i]), fontsize=7,
                xytext=(3, 2), textcoords='offset points')
ax.axvline(10, color='red', ls=':', lw=0.8, alpha=0.6)
ax.axhline(-10, color='red', ls=':', lw=0.8, alpha=0.6)
ax.set_xlabel("Amplification reduction (%)")
ax.set_ylabel("Gender mass change (%)")
ax.set_title("Pareto: Stereotype Reduction vs. Suppression\n(green = Pareto-dominant; top-right is ideal)")
ax.grid(True, alpha=0.3)

# Plot 4: Selectivity index bar chart
ax = axes[1, 1]
si_names = [n for n, _ in sorted_si[:10]]
si_vals  = [v for _, v in sorted_si[:10]]
colors4  = ['#2196F3' if n == 'L10H9' else '#90CAF9' if n.startswith('L10') else '#FFCC80'
             for n in si_names]
ax.barh(range(len(si_names)), si_vals, color=colors4, alpha=0.8)
ax.set_yticks(range(len(si_names)))
ax.set_yticklabels(si_names, fontsize=9)
ax.set_xlabel("Selectivity Index\n(amp_reduction / (skew_change + mass_suppression + 1))")
ax.set_title("Head Selectivity: Stereotype-Specific vs. Generic Suppression")

plt.suptitle("Experiment 2: Graduated Alpha Sweep + Pareto Analysis", fontsize=13)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "alpha_sweep_pareto.png", dpi=150)
plt.close()
print("\nFigure saved.")


# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════
save_data = {
    "baseline": {
        "amplification": base_amp,
        "residual_skew": base_skew,
        "gender_mass": base_mass,
        "neutral_signed": base_n_signed,
        "neutral_mass": base_n_mass,
    },
    "alpha_sweep": sweep_results,
    "pareto_l10": pareto,
    "selectivity_index": dict(sorted_si),
    "pareto_dominant_l10": [(n, d) for n, d in pareto_dominant],
}

with open(RESULTS_DIR / "alpha_sweep_results.json", "w") as f:
    json.dump(save_data, f, indent=2, default=str)

print("Results saved to %s" % RESULTS_DIR)
print("\n✓ Experiment 2 complete.")