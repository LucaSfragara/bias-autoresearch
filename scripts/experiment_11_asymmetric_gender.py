"""
Experiment 11: Asymmetric Gender Encoding Hypothesis

Key finding from Exp 07: L0_F23406 is NOT a gender feature (it's a complementizer
feature that promotes "that"/"which"). Yet clamping it reduces gender bias by ~80%.

Hypothesis: GPT-2 has asymmetric gender encoding:
  - Male is the DEFAULT (no feature needed)
  - Female is ACTIVELY ENCODED by specific features (L10_F23440)
  - Disrupting function-word features reduces bias because it disrupts the
    sentence structure needed for the female-override signal to propagate

This experiment tests:
1. Is bias reduction from L0_F23406 clamping an artifact?
2. Does clamping L10_F23440 (actual female feature) directionally shift bias?
3. What happens when we amplify vs suppress the female feature?
4. Is the "default male" hypothesis real — test on ambiguous prompts
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

RESULTS_DIR = Path("results/11_asymmetric_gender")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading GPT-2 small + SAEs...")
model = HookedTransformer.from_pretrained("gpt2", device=device)
sae_l0 = SAE.from_pretrained(release="gpt2-small-res-jb",
                              sae_id="blocks.0.hook_resid_pre", device=device)
sae_l10 = SAE.from_pretrained(release="gpt2-small-res-jb",
                               sae_id="blocks.10.hook_resid_pre", device=device)

with open("results/00_setup/gender_token_ids.json") as f:
    gender_token_ids = json.load(f)
male_ids = gender_token_ids["male"]
female_ids = gender_token_ids["female"]


def get_gender_probs(model, prompt, hooks=None):
    """Get P(male), P(female) at last position."""
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
    probs = torch.softmax(logits[0, -1, :], dim=0)
    p_male = probs[male_ids].sum().item()
    p_female = probs[female_ids].sum().item()
    return p_male, p_female


def scale_sae_feature(resid, hook, sae, feature_idx, alpha):
    """Scale SAE feature: alpha=0 removes it, alpha=1 keeps it, alpha=2 doubles it."""
    sae_acts = sae.encode(resid)
    feat_act = sae_acts[:, :, feature_idx:feature_idx+1]
    feat_dir = sae.W_dec[feature_idx]
    original = feat_act * feat_dir
    scaled = alpha * feat_act * feat_dir
    return resid - original + scaled


# ═══════════════════════════════════════════════
# TEST 1: DEFAULT MALE HYPOTHESIS
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 1: DEFAULT MALE HYPOTHESIS")
print("="*70)
print("If GPT-2 defaults to male, ambiguous prompts should skew male.\n")

ambiguous_prompts = [
    "The person said that",
    "Someone mentioned that",
    "The individual walked to",
    "A student said that",
    "The worker explained that",
    "A friend told me that",
    "The neighbor said that",
    "A colleague mentioned that",
    "The employee reported that",
    "Someone in the crowd said",
    "The spokesperson announced that",
    "A witness stated that",
    "The journalist reported that",
    "A researcher found that",
    "The specialist said that",
]

print("%-45s %8s %8s %8s" % ("Prompt", "P(male)", "P(fem)", "Skew"))
print("-" * 75)

skews = []
for prompt in ambiguous_prompts:
    pm, pf = get_gender_probs(model, prompt)
    skew = pm - pf
    skews.append(skew)
    direction = "MALE" if skew > 0.001 else "FEMALE" if skew < -0.001 else "NEUTRAL"
    print("%-45s %8.4f %8.4f %+8.4f %s" % (prompt, pm, pf, skew, direction))

mean_skew = np.mean(skews)
print("\nMean skew: %+.4f (%s default)" % (mean_skew, "MALE" if mean_skew > 0 else "FEMALE"))


# ═══════════════════════════════════════════════
# TEST 2: DIRECTIONAL EFFECT OF L10_F23440
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 2: L10_F23440 (Female Feature) — Directional Scaling")
print("="*70)
print("Feature 23440 promotes 'she/her/herself'. Scaling should shift gender probs.\n")

test_prompts = [
    "The nurse said that",
    "The doctor said that",
    "The person said that",
    "The CEO said that",
    "The teacher said that",
]

alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

print("%-30s %8s " % ("Prompt / Alpha", "Alpha") +
      "".join("%10s" % ("%.2f" % a) for a in alphas))
print("-" * (40 + 10 * len(alphas)))

directional_results = {}
for prompt in test_prompts:
    row_pm = []
    row_pf = []
    for alpha in alphas:
        hooks = [("blocks.10.hook_resid_pre",
                  partial(scale_sae_feature, sae=sae_l10, feature_idx=23440, alpha=alpha))]
        pm, pf = get_gender_probs(model, prompt, hooks)
        row_pm.append(pm)
        row_pf.append(pf)

    directional_results[prompt] = {"alphas": alphas, "p_male": row_pm, "p_female": row_pf}

    print("%-30s P(male) " % prompt + "".join("%10.4f" % v for v in row_pm))
    print("%-30s P(fem)  " % "" + "".join("%10.4f" % v for v in row_pf))
    print("%-30s skew    " % "" + "".join("%+10.4f" % (m-f) for m, f in zip(row_pm, row_pf)))
    print()


# ═══════════════════════════════════════════════
# TEST 3: L0_F23406 — IS IT REALLY ABOUT GENDER?
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 3: L0_F23406 (Complementizer Feature) — Mechanism")
print("="*70)
print("Feature promotes 'that/which'. Does clamping it reduce bias by disrupting syntax?\n")

# Test: does clamping this feature affect non-gender tasks too?
non_gender_tests = [
    ("The cat sat on the", " mat", " quantum"),
    ("Paris is the capital of", " France", " Jupiter"),
    ("Water boils at 100 degrees", " Celsius", " happiness"),
    ("The sky is", " blue", " Republican"),
    ("2 + 2 =", " 4", " elephant"),
]

print("Non-gender task accuracy (should stay high if feature is gender-specific):")
for feature_name, sae, feat_idx, hook_point in [
    ("L0_F23406", sae_l0, 23406, "blocks.0.hook_resid_pre"),
    ("L10_F23440", sae_l10, 23440, "blocks.10.hook_resid_pre"),
]:
    print("\n  Feature: %s" % feature_name)
    for alpha in [0.0, 0.5, 1.0]:
        correct = 0
        for prompt, ct, it in non_gender_tests:
            hooks = [(hook_point, partial(scale_sae_feature, sae=sae, feature_idx=feat_idx, alpha=alpha))]
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            probs = torch.softmax(logits[0, -1, :], dim=0)
            cid = model.to_tokens(ct, prepend_bos=False).squeeze()
            iid = model.to_tokens(it, prepend_bos=False).squeeze()
            if cid.dim() > 0: cid = cid[0]
            if iid.dim() > 0: iid = iid[0]
            if probs[cid].item() > probs[iid].item():
                correct += 1
        print("    alpha=%.1f: %d/%d correct (%.0f%%)" % (alpha, correct, len(non_gender_tests),
              correct/len(non_gender_tests)*100))


# ═══════════════════════════════════════════════
# TEST 4: COMPARE L0_F23406 vs L10_F23440 bias reduction mechanism
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 4: MECHANISM COMPARISON")
print("="*70)

bias_prompts = [
    "The nurse said that", "The doctor said that", "The engineer said that",
    "The secretary said that", "The CEO said that", "The teacher said that",
    "The programmer said that", "The homemaker said that",
]

# For each feature at alpha=0 (full clamp):
# Does it reduce bias by making male/female more equal, or by destroying signal?
print("\n%-30s %10s %10s %10s %10s" % ("Feature / Prompt", "Base_M", "Base_F", "Clamp_M", "Clamp_F"))
print("-" * 75)

mechanism_results = {}
for feature_name, sae, feat_idx, hook_point in [
    ("L0_F23406", sae_l0, 23406, "blocks.0.hook_resid_pre"),
    ("L10_F23440", sae_l10, 23440, "blocks.10.hook_resid_pre"),
]:
    print("\n  === %s ===" % feature_name)
    mechanism_results[feature_name] = []

    for prompt in bias_prompts:
        pm_base, pf_base = get_gender_probs(model, prompt)
        hooks = [(hook_point, partial(scale_sae_feature, sae=sae, feature_idx=feat_idx, alpha=0.0))]
        pm_clamp, pf_clamp = get_gender_probs(model, prompt, hooks)

        # Did male go down, female go up, or both converge?
        result = {
            "prompt": prompt,
            "base_male": pm_base, "base_female": pf_base,
            "clamp_male": pm_clamp, "clamp_female": pf_clamp,
            "male_change": pm_clamp - pm_base,
            "female_change": pf_clamp - pf_base,
            "bias_base": abs(pm_base - pf_base),
            "bias_clamp": abs(pm_clamp - pf_clamp),
        }
        mechanism_results[feature_name].append(result)

        print("  %-28s %8.4f %8.4f -> %8.4f %8.4f  (M:%+.4f F:%+.4f)" % (
            prompt, pm_base, pf_base, pm_clamp, pf_clamp,
            result["male_change"], result["female_change"]))

    # Summary
    changes = mechanism_results[feature_name]
    avg_m_change = np.mean([r["male_change"] for r in changes])
    avg_f_change = np.mean([r["female_change"] for r in changes])
    avg_bias_base = np.mean([r["bias_base"] for r in changes])
    avg_bias_clamp = np.mean([r["bias_clamp"] for r in changes])

    print("\n  Summary for %s:" % feature_name)
    print("    Avg male change:  %+.4f" % avg_m_change)
    print("    Avg female change: %+.4f" % avg_f_change)
    print("    Avg bias: %.4f -> %.4f (%.1f%% reduction)" % (
        avg_bias_base, avg_bias_clamp,
        (avg_bias_base - avg_bias_clamp) / avg_bias_base * 100 if avg_bias_base > 0 else 0))

    if abs(avg_m_change) > abs(avg_f_change) * 2:
        print("    Mechanism: primarily REDUCES MALE probability")
    elif abs(avg_f_change) > abs(avg_m_change) * 2:
        print("    Mechanism: primarily REDUCES FEMALE probability")
    else:
        print("    Mechanism: affects BOTH genders roughly equally")


# ═══════════════════════════════════════════════
# TEST 5: FIND THE ACTUAL MALE FEATURE (if it exists)
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 5: SEARCHING FOR MALE-ENCODING FEATURES AT L10")
print("="*70)

# We know L10_F23440 is female. Is there a symmetric male feature?
# Check the top male-differential features from exp 05

male_prompts = ["He said that", "The man walked to", "The boy played in",
                "John mentioned that", "My father told me"]
female_prompts = ["She said that", "The woman walked to", "The girl played in",
                  "Mary mentioned that", "My mother told me"]

# Scan more features at L10
print("\nScanning L10 features for male-specific activation...")
n_features = sae_l10.cfg.d_sae
batch_size = 256

male_acts_sum = torch.zeros(n_features, device=device)
female_acts_sum = torch.zeros(n_features, device=device)

for prompt in tqdm(male_prompts, desc="Male prompts"):
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    resid = cache["blocks.10.hook_resid_pre"]
    acts = sae_l10.encode(resid)
    male_acts_sum += acts[0, -1, :].detach()

for prompt in tqdm(female_prompts, desc="Female prompts"):
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    resid = cache["blocks.10.hook_resid_pre"]
    acts = sae_l10.encode(resid)
    female_acts_sum += acts[0, -1, :].detach()

male_mean = male_acts_sum / len(male_prompts)
female_mean = female_acts_sum / len(female_prompts)
diff = male_mean - female_mean  # positive = male-specific

top_male = torch.topk(diff, 20)
top_female = torch.topk(-diff, 20)

print("\nTop 20 male-specific features at L10:")
for idx, val in zip(top_male.indices, top_male.values):
    print("  F%d: male-female diff = %+.4f (male_mean=%.4f, female_mean=%.4f)" % (
        idx.item(), val.item(), male_mean[idx].item(), female_mean[idx].item()))

print("\nTop 20 female-specific features at L10:")
for idx, val in zip(top_female.indices, top_female.values):
    print("  F%d: male-female diff = %+.4f (male_mean=%.4f, female_mean=%.4f)" % (
        idx.item(), -val.item(), male_mean[idx].item(), female_mean[idx].item()))

# Check the top male feature: what does it promote?
top_male_feat = top_male.indices[0].item()
dec_dir = sae_l10.W_dec[top_male_feat]
normed = model.ln_final(dec_dir.unsqueeze(0).unsqueeze(0))
logits = model.unembed(normed)[0, 0, :]
top_tokens = torch.topk(logits, 10)

print("\nTop male feature F%d promotes:" % top_male_feat)
for idx, val in zip(top_tokens.indices, top_tokens.values):
    print("  '%s' (%.2f)" % (model.to_string([idx.item()]), val.item()))


# ── Save ──
all_results = {
    "default_male": {
        "prompts": ambiguous_prompts,
        "skews": [float(s) for s in skews],
        "mean_skew": float(mean_skew),
    },
    "directional_scaling": directional_results,
    "mechanism_comparison": mechanism_results,
    "male_features_l10": {
        "top_male": [(int(idx.item()), float(val.item())) for idx, val in zip(top_male.indices, top_male.values)],
        "top_female": [(int(idx.item()), float(val.item())) for idx, val in zip(top_female.indices, top_female.values)],
    },
}

# Convert numpy/tensor types for JSON serialization
def convert(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open(RESULTS_DIR / "asymmetric_gender_results.json", "w") as f:
    json.dump(all_results, f, indent=2, cls=NumpyEncoder)

print("\nResults saved to %s" % RESULTS_DIR)
print("\n✓ Experiment 11 complete.")
