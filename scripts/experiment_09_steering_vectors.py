"""
Experiment 09: Steering Vectors & Dose-Response Analysis
- Instead of binary ablation (0 or 1), continuously scale intervention strength
- Activation steering: add/subtract gender directions in the residual stream
- SAE feature scaling: multiply feature activations by factor alpha
- Dose-response curves: bias vs capability as function of intervention strength
- Compare ablation vs steering vs SAE clamping at matched bias-reduction levels
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

RESULTS_DIR = Path("results/09_steering")
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


def eval_bias(model, hooks=None):
    scores = []
    for prompt in BIAS_PROMPTS:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
        probs = torch.softmax(logits[0, -1, :], dim=0)
        scores.append(abs(probs[male_ids].sum().item() - probs[female_ids].sum().item()))
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


# ═══════════════════════════════════════════════
# PART 1: COMPUTE GENDER STEERING VECTOR
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("COMPUTING GENDER STEERING VECTORS")
print("="*70)

# Contrastive pairs for steering vector computation
MALE_PROMPTS = [
    "He is a", "The man said", "The boy went to", "His father told him",
    "The king ruled", "John walked to", "The husband cooked",
    "My grandfather said", "The prince fought", "He drove his car",
    "The gentleman entered", "The actor performed", "My brother told me",
    "The waiter served", "He went to his office",
]
FEMALE_PROMPTS = [
    "She is a", "The woman said", "The girl went to", "Her mother told her",
    "The queen ruled", "Mary walked to", "The wife cooked",
    "My grandmother said", "The princess fought", "She drove her car",
    "The lady entered", "The actress performed", "My sister told me",
    "The waitress served", "She went to her office",
]

# Collect mean activations for male vs female prompts at each layer
n_layers = model.cfg.n_layers
steering_vectors = {}

for layer in tqdm(range(n_layers), desc="Computing steering vectors"):
    hook_point = "blocks.%d.hook_resid_post" % layer
    male_acts = []
    female_acts = []

    for prompt in MALE_PROMPTS:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        male_acts.append(cache[hook_point][0, -1, :].cpu())

    for prompt in FEMALE_PROMPTS:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        female_acts.append(cache[hook_point][0, -1, :].cpu())

    male_mean = torch.stack(male_acts).mean(0)
    female_mean = torch.stack(female_acts).mean(0)
    steering_vec = male_mean - female_mean  # points from female to male

    steering_vectors[layer] = {
        "vector": steering_vec,
        "norm": steering_vec.norm().item(),
        "cosine_with_unembed_he": 0.0,  # will compute below
        "cosine_with_unembed_she": 0.0,
    }

# Check which layer's steering vector aligns with he/she in vocab space
he_id = model.to_tokens(" he", prepend_bos=False).squeeze().item()
she_id = model.to_tokens(" she", prepend_bos=False).squeeze().item()

for layer in range(n_layers):
    sv = steering_vectors[layer]["vector"].to(device)
    # Project through LN + unembed
    normed = model.ln_final(sv.unsqueeze(0).unsqueeze(0))
    logits = model.unembed(normed)[0, 0, :]
    he_logit = logits[he_id].item()
    she_logit = logits[she_id].item()
    steering_vectors[layer]["he_logit_projection"] = he_logit
    steering_vectors[layer]["she_logit_projection"] = she_logit
    steering_vectors[layer]["he_she_gap"] = he_logit - she_logit

print("\nSteering vector properties per layer:")
print("%-8s %10s %12s %12s %12s" % ("Layer", "Norm", "he_proj", "she_proj", "gap"))
print("-" * 60)
for layer in range(n_layers):
    sv = steering_vectors[layer]
    print("L%-7d %10.3f %12.3f %12.3f %12.3f" % (
        layer, sv["norm"], sv["he_logit_projection"],
        sv["she_logit_projection"], sv["he_she_gap"]))

# Find the layer with strongest gender signal
best_layer = max(range(n_layers), key=lambda l: abs(steering_vectors[l]["he_she_gap"]))
print("\nStrongest gender steering vector: Layer %d (gap=%.3f)" % (
    best_layer, steering_vectors[best_layer]["he_she_gap"]))


# ═══════════════════════════════════════════════
# PART 2: DOSE-RESPONSE CURVES
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("DOSE-RESPONSE ANALYSIS")
print("="*70)

# Method 1: Head ablation with scaling (partial ablation)
def scale_head(z, hook, head_idx, alpha):
    """Scale head output by alpha (0 = full ablation, 1 = no change, >1 = amplification)."""
    z[:, :, head_idx, :] *= alpha
    return z

# Method 2: Activation steering (add/subtract steering vector)
def steer_residual(resid, hook, direction, strength):
    """Add strength * direction to residual stream at all positions."""
    resid[:, :, :] += strength * direction.to(resid.device)
    return resid

# Method 3: SAE feature scaling
def scale_sae_feature(resid, hook, sae, feature_idx, alpha):
    """Scale SAE feature contribution by alpha (0 = full clamp, 1 = no change)."""
    sae_acts = sae.encode(resid)
    feat_act = sae_acts[:, :, feature_idx:feature_idx+1]
    feat_dir = sae.W_dec[feature_idx]
    # Remove original and add scaled version
    original_contribution = feat_act * feat_dir
    scaled_contribution = alpha * feat_act * feat_dir
    return resid - original_contribution + scaled_contribution

# Baselines
print("Computing baselines...")
baseline_bias = eval_bias(model)
baseline_ppl = eval_ppl(model)
baseline_pronoun = eval_pronoun(model)
print("Baseline: bias=%.6f, ppl=%.2f, pronoun=%.0f%%" % (
    baseline_bias, baseline_ppl, baseline_pronoun * 100))

# Alphas to test (intervention strength)
alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
steering_strengths = [-2.0, -1.5, -1.0, -0.8, -0.5, -0.3, -0.1, 0.0,
                      0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

# Dose-response: Head L10H9 scaling
print("\n--- Dose-response: Head L10H9 scaling ---")
head_dose = []
for alpha in tqdm(alphas, desc="Head scaling"):
    hooks = [("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=alpha))]
    b = eval_bias(model, hooks)
    p = eval_ppl(model, hooks)
    pr = eval_pronoun(model, hooks)
    head_dose.append({
        "alpha": alpha, "bias": float(b), "ppl": float(p), "pronoun": float(pr),
        "bias_reduction_pct": float((baseline_bias - b) / baseline_bias * 100),
        "ppl_change_pct": float((p - baseline_ppl) / baseline_ppl * 100),
    })
    print("  alpha=%.1f: bias=%.6f (%.1f%%), ppl=%.2f (%+.1f%%)" % (
        alpha, b, (baseline_bias - b) / baseline_bias * 100, p,
        (p - baseline_ppl) / baseline_ppl * 100))

# Dose-response: SAE Feature 23406 scaling
print("\n--- Dose-response: SAE L0 F23406 scaling ---")
sae_dose = []
for alpha in tqdm(alphas, desc="SAE feature scaling"):
    hooks = [("blocks.0.hook_resid_pre",
              partial(scale_sae_feature, sae=sae_l0, feature_idx=23406, alpha=alpha))]
    b = eval_bias(model, hooks)
    p = eval_ppl(model, hooks)
    pr = eval_pronoun(model, hooks)
    sae_dose.append({
        "alpha": alpha, "bias": float(b), "ppl": float(p), "pronoun": float(pr),
        "bias_reduction_pct": float((baseline_bias - b) / baseline_bias * 100),
        "ppl_change_pct": float((p - baseline_ppl) / baseline_ppl * 100),
    })
    print("  alpha=%.1f: bias=%.6f (%.1f%%), ppl=%.2f (%+.1f%%)" % (
        alpha, b, (baseline_bias - b) / baseline_bias * 100, p,
        (p - baseline_ppl) / baseline_ppl * 100))

# Dose-response: Activation steering at best layer
print("\n--- Dose-response: Activation steering at Layer %d ---" % best_layer)
steer_vec = steering_vectors[best_layer]["vector"]
steer_norm = steer_vec.norm()
steer_unit = steer_vec / steer_norm  # unit direction

steering_dose = []
for strength in tqdm(steering_strengths, desc="Steering"):
    hook_name = "blocks.%d.hook_resid_post" % best_layer
    hooks = [(hook_name, partial(steer_residual, direction=steer_unit, strength=strength))]
    b = eval_bias(model, hooks)
    p = eval_ppl(model, hooks)
    pr = eval_pronoun(model, hooks)
    steering_dose.append({
        "strength": float(strength), "bias": float(b), "ppl": float(p), "pronoun": float(pr),
        "bias_reduction_pct": float((baseline_bias - b) / baseline_bias * 100),
        "ppl_change_pct": float((p - baseline_ppl) / baseline_ppl * 100),
    })
    print("  strength=%.1f: bias=%.6f (%.1f%%), ppl=%.2f (%+.1f%%)" % (
        strength, b, (baseline_bias - b) / baseline_bias * 100, p,
        (p - baseline_ppl) / baseline_ppl * 100))

# Also try steering at layer 10 (the known important layer)
print("\n--- Dose-response: Activation steering at Layer 10 ---")
steer_vec_l10 = steering_vectors[10]["vector"]
steer_unit_l10 = steer_vec_l10 / steer_vec_l10.norm()

steering_l10_dose = []
for strength in tqdm(steering_strengths, desc="Steering L10"):
    hooks = [("blocks.10.hook_resid_post",
              partial(steer_residual, direction=steer_unit_l10, strength=strength))]
    b = eval_bias(model, hooks)
    p = eval_ppl(model, hooks)
    pr = eval_pronoun(model, hooks)
    steering_l10_dose.append({
        "strength": float(strength), "bias": float(b), "ppl": float(p), "pronoun": float(pr),
        "bias_reduction_pct": float((baseline_bias - b) / baseline_bias * 100),
        "ppl_change_pct": float((p - baseline_ppl) / baseline_ppl * 100),
    })


# ═══════════════════════════════════════════════
# PART 3: COMPARATIVE ANALYSIS AT MATCHED BIAS LEVELS
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("COMPARATIVE: METHODS AT MATCHED BIAS REDUCTION")
print("="*70)

# Find the settings that achieve ~25%, ~50%, ~75% bias reduction for each method
target_reductions = [25, 50, 75]

def find_closest(dose_results, target_pct, key="bias_reduction_pct"):
    """Find the dose that achieves closest to target bias reduction."""
    best = None
    best_diff = float('inf')
    for entry in dose_results:
        diff = abs(entry[key] - target_pct)
        if diff < best_diff:
            best_diff = diff
            best = entry
    return best

print("\n%-30s %10s %10s %10s %10s" % ("Method @ Target", "Actual%", "PPL%", "Pronoun", "Setting"))
print("-" * 80)

for target in target_reductions:
    # Head
    h = find_closest(head_dose, target)
    print("Head L10H9 @ %d%%         %10.1f%% %10.1f%% %10.0f%% alpha=%.1f" % (
        target, h["bias_reduction_pct"], h["ppl_change_pct"], h["pronoun"]*100, h["alpha"]))

    # SAE
    s = find_closest(sae_dose, target)
    print("SAE L0 F23406 @ %d%%      %10.1f%% %10.1f%% %10.0f%% alpha=%.1f" % (
        target, s["bias_reduction_pct"], s["ppl_change_pct"], s["pronoun"]*100, s["alpha"]))

    # Steering
    st = find_closest(steering_dose, target)
    print("Steering L%d @ %d%%       %10.1f%% %10.1f%% %10.0f%% str=%.1f" % (
        best_layer, target, st["bias_reduction_pct"], st["ppl_change_pct"],
        st["pronoun"]*100, st["strength"]))
    print()


# ═══════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════

# Fig 1: Dose-response curves (bias vs alpha/strength)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.plot([d["alpha"] for d in head_dose], [d["bias"] for d in head_dose], 'b-o', label="Bias")
ax.axhline(y=baseline_bias, color='gray', linestyle='--', alpha=0.3)
ax2 = ax.twinx()
ax2.plot([d["alpha"] for d in head_dose], [d["ppl"] for d in head_dose], 'r--s', label="PPL", alpha=0.7)
ax.set_xlabel("Head Scale (alpha)")
ax.set_ylabel("Bias (blue)", color='blue')
ax2.set_ylabel("Perplexity (red)", color='red')
ax.set_title("Head L10H9 Dose-Response")

ax = axes[1]
ax.plot([d["alpha"] for d in sae_dose], [d["bias"] for d in sae_dose], 'b-o', label="Bias")
ax.axhline(y=baseline_bias, color='gray', linestyle='--', alpha=0.3)
ax2 = ax.twinx()
ax2.plot([d["alpha"] for d in sae_dose], [d["ppl"] for d in sae_dose], 'r--s', label="PPL", alpha=0.7)
ax.set_xlabel("Feature Scale (alpha)")
ax.set_ylabel("Bias (blue)", color='blue')
ax2.set_ylabel("Perplexity (red)", color='red')
ax.set_title("SAE L0 F23406 Dose-Response")

ax = axes[2]
ax.plot([d["strength"] for d in steering_dose], [d["bias"] for d in steering_dose], 'b-o', label="Bias")
ax.axhline(y=baseline_bias, color='gray', linestyle='--', alpha=0.3)
ax2 = ax.twinx()
ax2.plot([d["strength"] for d in steering_dose], [d["ppl"] for d in steering_dose], 'r--s', label="PPL", alpha=0.7)
ax.set_xlabel("Steering Strength")
ax.set_ylabel("Bias (blue)", color='blue')
ax2.set_ylabel("Perplexity (red)", color='red')
ax.set_title("Activation Steering L%d Dose-Response" % best_layer)

plt.suptitle("Dose-Response: Bias vs Capability Cost", fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "dose_response_curves.png", dpi=150)
plt.close()

# Fig 2: Pareto frontier — bias reduction vs PPL cost for all three methods
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot([d["bias_reduction_pct"] for d in head_dose],
        [d["ppl_change_pct"] for d in head_dose],
        'b-o', label="Head L10H9 scaling", markersize=5)
ax.plot([d["bias_reduction_pct"] for d in sae_dose],
        [d["ppl_change_pct"] for d in sae_dose],
        'r-s', label="SAE L0 F23406 scaling", markersize=5)
ax.plot([d["bias_reduction_pct"] for d in steering_dose],
        [d["ppl_change_pct"] for d in steering_dose],
        'g-^', label="Steering (L%d)" % best_layer, markersize=5)
ax.plot([d["bias_reduction_pct"] for d in steering_l10_dose],
        [d["ppl_change_pct"] for d in steering_l10_dose],
        'm-v', label="Steering (L10)", markersize=5, alpha=0.7)

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel("Bias Reduction (%)", fontsize=12)
ax.set_ylabel("Perplexity Change (%)", fontsize=12)
ax.set_title("Pareto Frontier: Bias Reduction vs Capability Cost\n(Continuous Intervention Strength)", fontsize=13)
ax.legend(fontsize=10)

# Shade favorable region
ax.fill_between([0, 100], [-15, -15], [5, 5], alpha=0.05, color='green')
ax.set_xlim(-20, 100)
ax.set_ylim(-15, max(50, max(d["ppl_change_pct"] for d in head_dose + sae_dose + steering_dose) * 1.1))

plt.tight_layout()
plt.savefig(RESULTS_DIR / "pareto_frontier_continuous.png", dpi=150)
plt.close()

# Fig 3: Pronoun accuracy vs bias reduction
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot([d["bias_reduction_pct"] for d in head_dose],
        [d["pronoun"] * 100 for d in head_dose],
        'b-o', label="Head L10H9", markersize=5)
ax.plot([d["bias_reduction_pct"] for d in sae_dose],
        [d["pronoun"] * 100 for d in sae_dose],
        'r-s', label="SAE L0 F23406", markersize=5)
ax.plot([d["bias_reduction_pct"] for d in steering_dose],
        [d["pronoun"] * 100 for d in steering_dose],
        'g-^', label="Steering L%d" % best_layer, markersize=5)

ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel("Bias Reduction (%)", fontsize=12)
ax.set_ylabel("Pronoun Resolution Accuracy (%)", fontsize=12)
ax.set_title("Pronoun Accuracy vs Bias Reduction", fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "pronoun_vs_bias.png", dpi=150)
plt.close()


# ── Save all results ──
results = {
    "baselines": {
        "bias": float(baseline_bias),
        "ppl": float(baseline_ppl),
        "pronoun": float(baseline_pronoun),
    },
    "steering_vectors": {
        str(l): {
            "norm": sv["norm"],
            "he_logit_projection": sv["he_logit_projection"],
            "she_logit_projection": sv["she_logit_projection"],
            "he_she_gap": sv["he_she_gap"],
        } for l, sv in steering_vectors.items()
    },
    "best_steering_layer": best_layer,
    "dose_response": {
        "head_L10H9": head_dose,
        "sae_L0_F23406": sae_dose,
        "steering_best_layer": steering_dose,
        "steering_L10": steering_l10_dose,
    },
}

with open(RESULTS_DIR / "steering_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to %s" % RESULTS_DIR)
print("\n✓ Experiment 09 complete.")
