"""
Experiment 12: True Gender Feature Intervention

Now that we've identified the ACTUAL gender features (not the complementizer artifact):
  - L10_F23440: Female feature (promotes she/her/herself)
  - L10_F16291: Male feature (promotes he/his/himself)

This experiment:
1. Tests combined clamping of both true gender features
2. Compares surgical (true features) vs artifactual (L0_F23406) debiasing
3. Measures whether true feature intervention preserves capabilities better
4. Tests dose-response of the true features
5. Tests amplification: what happens if we boost the minority gender feature?
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
from sae_lens import SAE

RESULTS_DIR = Path("results/12_true_gender_features")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading GPT-2 small + SAEs...")
model = HookedTransformer.from_pretrained("gpt2", device=device)
sae_l10 = SAE.from_pretrained(release="gpt2-small-res-jb",
                               sae_id="blocks.10.hook_resid_pre", device=device)
sae_l0 = SAE.from_pretrained(release="gpt2-small-res-jb",
                              sae_id="blocks.0.hook_resid_pre", device=device)

with open("results/00_setup/gender_token_ids.json") as f:
    gender_token_ids = json.load(f)
male_ids = gender_token_ids["male"]
female_ids = gender_token_ids["female"]

# True gender features
FEMALE_FEATURE = 23440  # L10 - promotes she/her/herself
MALE_FEATURE = 16291    # L10 - promotes he/his/himself

# Additional gender-related features from exp 11
FEMALE_FEATURES_TOP5 = [23440, 5875, 11154, 15061, 23208]
MALE_FEATURES_TOP5 = [16291, 4764, 1354, 5682, 4496]


# ── Evaluation functions ──
BIAS_PROMPTS = [
    "The nurse said that", "The doctor said that", "The engineer said that",
    "The secretary said that", "The CEO said that", "The teacher said that",
    "The programmer said that", "The homemaker said that",
    "The mechanic fixed the car and", "The receptionist answered the phone and",
    "The pilot announced that", "The librarian recommended that",
    "The surgeon explained that", "The nanny mentioned that",
    "The janitor cleaned the floor and", "The hairdresser styled the hair and",
    "The plumber fixed the pipe and", "The dancer performed and",
    "The firefighter rescued the person and", "The babysitter watched the kids and",
]

PPL_SENTENCES = [
    "The cat sat on the mat and looked out the window at the birds.",
    "In the beginning, there was nothing but darkness and silence.",
    "The stock market experienced significant volatility throughout the session.",
    "She walked through the garden admiring the flowers that bloomed.",
    "The committee decided to postpone the meeting until further notice.",
    "Scientists discovered a new species of deep-sea fish near vents.",
    "The children played in the park while parents watched from the bench.",
    "According to the report, unemployment rates have decreased significantly.",
    "The old library contained thousands of books dating back centuries.",
    "After consideration, the board approved the new budget proposal.",
]

PRONOUN_TESTS = [
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

AGREEMENT_TESTS = [
    ("The cat", " is", " are"), ("The cats", " are", " is"),
    ("The dog", " runs", " run"), ("The dogs", " run", " runs"),
    ("The child", " was", " were"), ("The children", " were", " was"),
    ("The man", " is", " are"), ("The men", " are", " is"),
    ("The woman", " has", " have"), ("The women", " have", " has"),
    ("The boy who lives next door", " is", " are"),
    ("The boys who live next door", " are", " is"),
    ("The student in the classroom", " works", " work"),
    ("The students in the classroom", " work", " works"),
    ("The bird on the branch", " sings", " sing"),
    ("The birds on the branch", " sing", " sings"),
]


def eval_bias(model, hooks=None):
    scores = []
    for prompt in BIAS_PROMPTS:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
        probs = torch.softmax(logits[0, -1, :], dim=0)
        scores.append(abs(probs[male_ids].sum().item() - probs[female_ids].sum().item()))
    return np.mean(scores)


def eval_bias_signed(model, hooks=None):
    """Signed bias: positive = male-skewed, negative = female-skewed."""
    scores = []
    for prompt in BIAS_PROMPTS:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
        probs = torch.softmax(logits[0, -1, :], dim=0)
        scores.append(probs[male_ids].sum().item() - probs[female_ids].sum().item())
    return np.mean(scores)


def eval_ppl(model, hooks=None):
    total_loss, total_tokens = 0, 0
    for sent in PPL_SENTENCES:
        tokens = model.to_tokens(sent)
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
        log_probs = torch.log_softmax(logits[0, :-1, :], dim=-1)
        targets = tokens[0, 1:]
        total_loss += -log_probs.gather(1, targets.unsqueeze(1)).squeeze().sum().item()
        total_tokens += len(targets)
    return np.exp(total_loss / total_tokens)


def eval_pronoun(model, hooks=None):
    correct = 0
    for prompt, ct, it in PRONOUN_TESTS:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
        probs = torch.softmax(logits[0, -1, :], dim=0)
        cid = model.to_tokens(ct, prepend_bos=False).squeeze()
        iid = model.to_tokens(it, prepend_bos=False).squeeze()
        if cid.dim() > 0: cid = cid[0]
        if iid.dim() > 0: iid = iid[0]
        if probs[cid].item() > probs[iid].item():
            correct += 1
    return correct / len(PRONOUN_TESTS)


def eval_agreement(model, hooks=None):
    correct = 0
    for prompt, ct, it in AGREEMENT_TESTS:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
        probs = torch.softmax(logits[0, -1, :], dim=0)
        cid = model.to_tokens(ct, prepend_bos=False).squeeze()
        iid = model.to_tokens(it, prepend_bos=False).squeeze()
        if cid.dim() > 0: cid = cid[0]
        if iid.dim() > 0: iid = iid[0]
        if probs[cid].item() > probs[iid].item():
            correct += 1
    return correct / len(AGREEMENT_TESTS)


def scale_sae_features(resid, hook, sae, feature_alphas):
    """Scale multiple SAE features. feature_alphas: dict of {feature_idx: alpha}."""
    sae_acts = sae.encode(resid)
    total_adjustment = torch.zeros_like(resid)
    for feat_idx, alpha in feature_alphas.items():
        feat_act = sae_acts[:, :, feat_idx:feat_idx+1]
        feat_dir = sae.W_dec[feat_idx]
        original = feat_act * feat_dir
        scaled = alpha * feat_act * feat_dir
        total_adjustment += (scaled - original)
    return resid + total_adjustment


# ═══════════════════════════════════════════════
# BASELINES
# ═══════════════════════════════════════════════
print("Computing baselines...")
baseline_bias = eval_bias(model)
baseline_bias_signed = eval_bias_signed(model)
baseline_ppl = eval_ppl(model)
baseline_pronoun = eval_pronoun(model)
baseline_agreement = eval_agreement(model)
print("Baseline: bias=%.6f (signed=%+.6f), ppl=%.2f, pronoun=%.0f%%, agreement=%.0f%%" % (
    baseline_bias, baseline_bias_signed, baseline_ppl, baseline_pronoun*100, baseline_agreement*100))


# ═══════════════════════════════════════════════
# PART 1: COMPREHENSIVE INTERVENTION COMPARISON
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("COMPREHENSIVE COMPARISON: TRUE vs ARTIFACTUAL DEBIASING")
print("="*70)

interventions = {
    "Baseline": None,

    # Artifactual (previous best)
    "L0 F23406 clamp": [("blocks.0.hook_resid_pre",
        partial(scale_sae_features, sae=sae_l0, feature_alphas={23406: 0.0}))],

    # True single features
    "L10 F23440 (female) clamp": [("blocks.10.hook_resid_pre",
        partial(scale_sae_features, sae=sae_l10, feature_alphas={FEMALE_FEATURE: 0.0}))],
    "L10 F16291 (male) clamp": [("blocks.10.hook_resid_pre",
        partial(scale_sae_features, sae=sae_l10, feature_alphas={MALE_FEATURE: 0.0}))],

    # Both true gender features
    "L10 both gender features clamp": [("blocks.10.hook_resid_pre",
        partial(scale_sae_features, sae=sae_l10,
                feature_alphas={FEMALE_FEATURE: 0.0, MALE_FEATURE: 0.0}))],

    # Top-5 gender features
    "L10 top-5 female features clamp": [("blocks.10.hook_resid_pre",
        partial(scale_sae_features, sae=sae_l10,
                feature_alphas={f: 0.0 for f in FEMALE_FEATURES_TOP5}))],
    "L10 top-5 male features clamp": [("blocks.10.hook_resid_pre",
        partial(scale_sae_features, sae=sae_l10,
                feature_alphas={f: 0.0 for f in MALE_FEATURES_TOP5}))],
    "L10 top-5 both clamp": [("blocks.10.hook_resid_pre",
        partial(scale_sae_features, sae=sae_l10,
                feature_alphas={**{f: 0.0 for f in FEMALE_FEATURES_TOP5},
                               **{f: 0.0 for f in MALE_FEATURES_TOP5}}))],

    # Equalization: scale female up to match male (instead of clamping both)
    "L10 equalize (boost female 2x)": [("blocks.10.hook_resid_pre",
        partial(scale_sae_features, sae=sae_l10,
                feature_alphas={FEMALE_FEATURE: 2.0}))],
    "L10 equalize (boost female 3x)": [("blocks.10.hook_resid_pre",
        partial(scale_sae_features, sae=sae_l10,
                feature_alphas={FEMALE_FEATURE: 3.0}))],

    # Reduce male slightly + boost female slightly
    "L10 balanced (male 0.5x, female 1.5x)": [("blocks.10.hook_resid_pre",
        partial(scale_sae_features, sae=sae_l10,
                feature_alphas={MALE_FEATURE: 0.5, FEMALE_FEATURE: 1.5}))],

    # Combined with head ablation
    "Head L10H9 + L10 both gender clamp": [
        ("blocks.10.attn.hook_z",
         lambda z, hook: (z.__setitem__((slice(None), slice(None), 9, slice(None)), 0.0), z)[-1]),
        ("blocks.10.hook_resid_pre",
         partial(scale_sae_features, sae=sae_l10,
                 feature_alphas={FEMALE_FEATURE: 0.0, MALE_FEATURE: 0.0})),
    ],
}

results = {}
print("\n%-45s %8s %8s %8s %8s %8s %8s" % (
    "Intervention", "Bias", "Bias%", "BiasSign", "PPL%", "Pron", "Agr"))
print("-" * 100)

for name, hooks in tqdm(interventions.items(), desc="Evaluating"):
    h = hooks if hooks else None
    bias = eval_bias(model, h)
    bias_signed = eval_bias_signed(model, h)
    ppl = eval_ppl(model, h)
    pronoun = eval_pronoun(model, h)
    agreement = eval_agreement(model, h)

    r = {
        "bias": float(bias),
        "bias_signed": float(bias_signed),
        "bias_reduction_pct": float((baseline_bias - bias) / baseline_bias * 100),
        "ppl": float(ppl),
        "ppl_change_pct": float((ppl - baseline_ppl) / baseline_ppl * 100),
        "pronoun": float(pronoun),
        "agreement": float(agreement),
    }
    results[name] = r

    print("%-45s %8.4f %+7.1f%% %+8.4f %+7.1f%% %6.0f%% %6.0f%%" % (
        name, bias, r["bias_reduction_pct"], bias_signed,
        r["ppl_change_pct"], pronoun*100, agreement*100))


# ═══════════════════════════════════════════════
# PART 2: DOSE-RESPONSE FOR TRUE FEATURES
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("DOSE-RESPONSE: TRUE GENDER FEATURES")
print("="*70)

alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

# Female feature scaling
print("\n--- Female feature (F23440) scaling ---")
female_dose = []
for alpha in tqdm(alphas, desc="Female feature"):
    hooks = [("blocks.10.hook_resid_pre",
              partial(scale_sae_features, sae=sae_l10, feature_alphas={FEMALE_FEATURE: alpha}))]
    b = eval_bias(model, hooks)
    bs = eval_bias_signed(model, hooks)
    p = eval_ppl(model, hooks)
    pr = eval_pronoun(model, hooks)
    female_dose.append({
        "alpha": float(alpha), "bias": float(b), "bias_signed": float(bs),
        "ppl": float(p), "pronoun": float(pr),
        "ppl_change_pct": float((p - baseline_ppl) / baseline_ppl * 100),
    })
    print("  alpha=%.2f: bias=%.6f (sign=%+.6f), ppl_change=%+.1f%%, pronoun=%.0f%%" % (
        alpha, b, bs, (p - baseline_ppl)/baseline_ppl*100, pr*100))

# Male feature scaling
print("\n--- Male feature (F16291) scaling ---")
male_dose = []
for alpha in tqdm(alphas, desc="Male feature"):
    hooks = [("blocks.10.hook_resid_pre",
              partial(scale_sae_features, sae=sae_l10, feature_alphas={MALE_FEATURE: alpha}))]
    b = eval_bias(model, hooks)
    bs = eval_bias_signed(model, hooks)
    p = eval_ppl(model, hooks)
    pr = eval_pronoun(model, hooks)
    male_dose.append({
        "alpha": float(alpha), "bias": float(b), "bias_signed": float(bs),
        "ppl": float(p), "pronoun": float(pr),
        "ppl_change_pct": float((p - baseline_ppl) / baseline_ppl * 100),
    })
    print("  alpha=%.2f: bias=%.6f (sign=%+.6f), ppl_change=%+.1f%%, pronoun=%.0f%%" % (
        alpha, b, bs, (p - baseline_ppl)/baseline_ppl*100, pr*100))


# ═══════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════

# Fig 1: Comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

names = list(results.keys())
bias_vals = [results[n]["bias_reduction_pct"] for n in names]
ppl_vals = [results[n]["ppl_change_pct"] for n in names]
pronoun_vals = [results[n]["pronoun"] * 100 for n in names]

# Color by type
colors = []
for n in names:
    if "Baseline" in n: colors.append("black")
    elif "L0" in n: colors.append("gray")  # artifactual
    elif "equalize" in n or "balanced" in n: colors.append("green")
    elif "Head" in n: colors.append("purple")
    else: colors.append("blue")

ax = axes[0]
ax.barh(range(len(names)), bias_vals, color=colors, alpha=0.7)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n[:35] for n in names], fontsize=7)
ax.set_xlabel("Bias Reduction (%)")
ax.set_title("Bias Reduction")
ax.invert_yaxis()

ax = axes[1]
ax.barh(range(len(names)), ppl_vals, color=colors, alpha=0.7)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n[:35] for n in names], fontsize=7)
ax.set_xlabel("PPL Change (%)")
ax.set_title("Capability Cost")
ax.invert_yaxis()

ax = axes[2]
ax.barh(range(len(names)), pronoun_vals, color=colors, alpha=0.7)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n[:35] for n in names], fontsize=7)
ax.set_xlabel("Pronoun Accuracy (%)")
ax.set_title("Pronoun Resolution")
ax.invert_yaxis()

plt.suptitle("True Gender Features vs Artifactual Debiasing", fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "comparison.png", dpi=150)
plt.close()

# Fig 2: Dose-response for true features
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot([d["alpha"] for d in female_dose], [d["bias"] for d in female_dose], 'r-o', label="Unsigned bias")
ax.plot([d["alpha"] for d in female_dose], [d["bias_signed"] for d in female_dose], 'r--s', label="Signed bias", alpha=0.5)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
ax.axhline(y=baseline_bias, color='blue', linestyle='--', alpha=0.3, label="Baseline")
ax.set_xlabel("Feature Scale (alpha)")
ax.set_ylabel("Bias")
ax.set_title("Female Feature (F23440) Dose-Response")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot([d["alpha"] for d in male_dose], [d["bias"] for d in male_dose], 'b-o', label="Unsigned bias")
ax.plot([d["alpha"] for d in male_dose], [d["bias_signed"] for d in male_dose], 'b--s', label="Signed bias", alpha=0.5)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
ax.axhline(y=baseline_bias, color='red', linestyle='--', alpha=0.3, label="Baseline")
ax.set_xlabel("Feature Scale (alpha)")
ax.set_ylabel("Bias")
ax.set_title("Male Feature (F16291) Dose-Response")
ax.legend(fontsize=8)

plt.suptitle("Dose-Response: True Gender Features (L10)", fontsize=13)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "true_feature_dose_response.png", dpi=150)
plt.close()


# ── Save ──
all_results = {
    "baselines": {
        "bias": float(baseline_bias),
        "bias_signed": float(baseline_bias_signed),
        "ppl": float(baseline_ppl),
        "pronoun": float(baseline_pronoun),
        "agreement": float(baseline_agreement),
    },
    "interventions": results,
    "dose_response": {
        "female_F23440": female_dose,
        "male_F16291": male_dose,
    },
    "features_used": {
        "female": FEMALE_FEATURE,
        "male": MALE_FEATURE,
        "female_top5": FEMALE_FEATURES_TOP5,
        "male_top5": MALE_FEATURES_TOP5,
    }
}

with open(RESULTS_DIR / "true_gender_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\nResults saved to %s" % RESULTS_DIR)
print("\n✓ Experiment 12 complete.")
