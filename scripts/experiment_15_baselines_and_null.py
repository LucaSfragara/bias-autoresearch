"""
Experiment 15: Debiasing Baselines + Jaccard Null Model

1. INLP (Iterative Null-space Projection) — linear probe debiasing baseline
2. Mean-subtraction debiasing — simplest possible baseline
3. Jaccard null model — expected overlap under random head selection
4. Comparison table of all methods
"""

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from functools import partial
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from sae_lens import SAE
from sklearn.linear_model import LogisticRegression

RESULTS_DIR = Path("results/15_baselines")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading GPT-2 small + SAEs...")
model = HookedTransformer.from_pretrained("gpt2", device=device)
sae_l10 = SAE.from_pretrained(release="gpt2-small-res-jb",
                               sae_id="blocks.10.hook_resid_pre", device=device)

with open("results/00_setup/gender_token_ids.json") as f:
    gender_token_ids = json.load(f)
male_ids = gender_token_ids["male"]
female_ids = gender_token_ids["female"]


# ── Evaluation ──
OCCUPATIONS = [
    "nurse", "doctor", "engineer", "secretary", "CEO", "teacher",
    "programmer", "homemaker", "mechanic", "receptionist", "pilot",
    "librarian", "surgeon", "nanny", "janitor", "hairdresser",
    "plumber", "dancer", "firefighter", "babysitter", "accountant",
    "lawyer", "chef", "farmer", "dentist", "professor",
]

TEMPLATES = [
    "The %s said that", "The %s walked to the", "The %s announced that",
    "The %s explained that", "The %s mentioned that",
    "The %s reported that", "The %s decided to",
    "The %s thought about", "The %s was known for", "The %s told the patient that",
]

bias_prompts = [tmpl % occ for occ in OCCUPATIONS for tmpl in TEMPLATES]
print("Using %d bias prompts" % len(bias_prompts))

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
    "The river flows through the valley creating a natural boundary.",
    "Music has the ability to evoke powerful emotions.",
    "The professor explained quantum entanglement to the class.",
    "The ancient ruins tell the story of a lost civilization.",
    "The startup raised fifty million dollars in funding.",
]

PRONOUN_TESTS = [
    ("The man went to the store. When he arrived,", " he", " she"),
    ("The woman went to the store. When she arrived,", " she", " he"),
    ("The boy played in the yard. Then", " he", " she"),
    ("The girl played in the yard. Then", " she", " he"),
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
    for prompt in bias_prompts:
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
# PART 1: INLP — Linear Probe Debiasing
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("INLP: ITERATIVE NULL-SPACE PROJECTION")
print("="*70)

# Step 1: Collect residual stream activations for male vs female prompts
male_prompts = [
    "He said that", "The man walked to", "The boy played in",
    "His father told him that", "The king ruled the land.",
    "John mentioned that", "My brother told me", "The husband cooked",
    "The actor performed brilliantly.", "My grandfather said",
    "He drove to his office.", "The gentleman entered the room.",
    "His son played outside.", "The prince fought bravely.",
    "The waiter brought the menu.",
]

female_prompts = [
    "She said that", "The woman walked to", "The girl played in",
    "Her mother told her that", "The queen ruled the land.",
    "Mary mentioned that", "My sister told me", "The wife cooked",
    "The actress performed brilliantly.", "My grandmother said",
    "She drove to her office.", "The lady entered the room.",
    "Her daughter played outside.", "The princess fought bravely.",
    "The waitress brought the menu.",
]

# Collect activations at multiple layers
INLP_LAYERS = [0, 4, 8, 10, 11]

for target_layer in INLP_LAYERS:
    hook_point = "blocks.%d.hook_resid_post" % target_layer
    print("\n--- INLP at Layer %d ---" % target_layer)

    activations = []
    labels = []

    for prompt in male_prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        act = cache[hook_point][0, -1, :].cpu().numpy()
        activations.append(act)
        labels.append(0)  # male

    for prompt in female_prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        act = cache[hook_point][0, -1, :].cpu().numpy()
        activations.append(act)
        labels.append(1)  # female

    X = np.array(activations)
    y = np.array(labels)

    # Train linear probe
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, y)
    probe_acc = clf.score(X, y)
    print("  Linear probe accuracy: %.1f%%" % (probe_acc * 100))

    # INLP: iteratively project out gender direction
    n_iterations = 3
    directions = []
    X_proj = X.copy()

    for i in range(n_iterations):
        clf_iter = LogisticRegression(max_iter=1000, C=1.0)
        clf_iter.fit(X_proj, y)
        acc = clf_iter.score(X_proj, y)
        print("  Iteration %d: probe acc=%.1f%%" % (i+1, acc*100))

        if acc < 0.55:
            print("  Probe near chance, stopping.")
            break

        # Get gender direction (normal to decision boundary)
        w = clf_iter.coef_[0]
        w = w / np.linalg.norm(w)
        directions.append(w)

        # Project out this direction
        X_proj = X_proj - np.outer(X_proj @ w, w)

    if not directions:
        print("  No directions found, skipping.")
        continue

    # Apply INLP projection as a hook
    P = np.eye(X.shape[1])
    for w in directions:
        P = P - np.outer(w, w)
    P_tensor = torch.tensor(P, dtype=torch.float32, device=device)

    def inlp_hook(resid, hook, projection_matrix):
        # Project out gender directions at last position
        resid[:, -1, :] = resid[:, -1, :] @ projection_matrix.T
        return resid

    hooks = [(hook_point, partial(inlp_hook, projection_matrix=P_tensor))]

    bias = eval_bias(model, hooks)
    ppl = eval_ppl(model, hooks)
    pronoun = eval_pronoun(model, hooks)

    print("  After INLP (layer %d):" % target_layer)
    print("    Bias: %.4f, PPL: %.2f, Pronoun: %.0f%%" % (bias, ppl, pronoun*100))


# ═══════════════════════════════════════════════
# PART 2: MEAN-SUBTRACTION DEBIASING
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("MEAN-SUBTRACTION DEBIASING")
print("="*70)

# Compute the mean gender direction at each layer
for target_layer in [0, 10, 11]:
    hook_point = "blocks.%d.hook_resid_post" % target_layer
    print("\n--- Mean subtraction at Layer %d ---" % target_layer)

    male_acts = []
    female_acts = []

    for prompt in male_prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        male_acts.append(cache[hook_point][0, -1, :].cpu())

    for prompt in female_prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        female_acts.append(cache[hook_point][0, -1, :].cpu())

    male_mean = torch.stack(male_acts).mean(0)
    female_mean = torch.stack(female_acts).mean(0)
    gender_dir = (male_mean - female_mean)
    gender_dir = gender_dir / gender_dir.norm()

    # Hook: project out gender direction at all positions
    def mean_sub_hook(resid, hook, direction):
        proj = (resid @ direction.to(resid.device)).unsqueeze(-1) * direction.to(resid.device)
        return resid - proj

    hooks = [(hook_point, partial(mean_sub_hook, direction=gender_dir))]

    bias = eval_bias(model, hooks)
    ppl = eval_ppl(model, hooks)
    pronoun = eval_pronoun(model, hooks)

    print("  After mean-subtraction (layer %d):" % target_layer)
    print("    Bias: %.4f, PPL: %.2f, Pronoun: %.0f%%" % (bias, ppl, pronoun*100))


# ═══════════════════════════════════════════════
# PART 3: JACCARD NULL MODEL
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("JACCARD NULL MODEL")
print("="*70)

# Our claim: gender/race/religion top-k heads have Jaccard overlap = 0.00
# Null hypothesis: if each bias type independently selects k heads from 144 total,
# what is the expected Jaccard overlap?

total_heads = 144  # 12 layers x 12 heads
k_values = [5, 10, 15, 20]

print("Expected Jaccard overlap under random selection (from %d heads):" % total_heads)
print("%-8s %12s %15s %15s" % ("k", "E[Jaccard]", "P(J=0)", "P(J>0)"))
print("-" * 55)

null_results = {}
for k in k_values:
    # Monte Carlo simulation
    n_sims = 100000
    jaccards = []
    zero_count = 0

    for _ in range(n_sims):
        set_a = set(np.random.choice(total_heads, k, replace=False))
        set_b = set(np.random.choice(total_heads, k, replace=False))
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        j = intersection / union if union > 0 else 0
        jaccards.append(j)
        if j == 0:
            zero_count += 1

    expected_j = np.mean(jaccards)
    p_zero = zero_count / n_sims
    null_results[k] = {
        "expected_jaccard": float(expected_j),
        "p_zero": float(p_zero),
        "p_nonzero": float(1 - p_zero),
    }
    print("k=%-5d %12.4f %15.4f %15.4f" % (k, expected_j, p_zero, 1 - p_zero))

# Three-way: P(all three pairwise Jaccards = 0)
print("\nThree-way (gender/race/religion) P(all J=0):")
for k in k_values:
    n_sims = 100000
    all_zero = 0
    for _ in range(n_sims):
        sets = [set(np.random.choice(total_heads, k, replace=False)) for _ in range(3)]
        j01 = len(sets[0] & sets[1]) / len(sets[0] | sets[1]) if len(sets[0] | sets[1]) > 0 else 0
        j02 = len(sets[0] & sets[2]) / len(sets[0] | sets[2]) if len(sets[0] | sets[2]) > 0 else 0
        j12 = len(sets[1] & sets[2]) / len(sets[1] | sets[2]) if len(sets[1] | sets[2]) > 0 else 0
        if j01 == 0 and j02 == 0 and j12 == 0:
            all_zero += 1
    p_all_zero = all_zero / n_sims
    null_results["%d_three_way" % k] = float(p_all_zero)
    print("  k=%d: P(all three J=0) = %.4f" % (k, p_all_zero))


# ═══════════════════════════════════════════════
# PART 4: FULL COMPARISON TABLE
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("FULL COMPARISON TABLE")
print("="*70)

baseline_bias = eval_bias(model)
baseline_ppl = eval_ppl(model)
baseline_pronoun = eval_pronoun(model)

def scale_head(z, hook, head_idx, alpha):
    z[:, :, head_idx, :] *= alpha
    return z

def scale_sae_features(resid, hook, sae, feature_alphas):
    sae_acts = sae.encode(resid)
    total_adjustment = torch.zeros_like(resid)
    for feat_idx, alpha in feature_alphas.items():
        feat_act = sae_acts[:, :, feat_idx:feat_idx+1]
        feat_dir = sae.W_dec[feat_idx]
        original = feat_act * feat_dir
        scaled = alpha * feat_act * feat_dir
        total_adjustment += (scaled - original)
    return resid + total_adjustment

# Collect INLP result at best layer (layer 10)
hook_point = "blocks.10.hook_resid_post"
X_inlp = []
y_inlp = []
for prompt in male_prompts:
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    X_inlp.append(cache[hook_point][0, -1, :].cpu().numpy())
    y_inlp.append(0)
for prompt in female_prompts:
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    X_inlp.append(cache[hook_point][0, -1, :].cpu().numpy())
    y_inlp.append(1)
X_inlp = np.array(X_inlp)
y_inlp = np.array(y_inlp)

P_inlp = np.eye(X_inlp.shape[1])
X_proj = X_inlp.copy()
for _ in range(3):
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_proj, y_inlp)
    if clf.score(X_proj, y_inlp) < 0.55:
        break
    w = clf.coef_[0]
    w = w / np.linalg.norm(w)
    P_inlp = P_inlp - np.outer(w, w)
    X_proj = X_proj - np.outer(X_proj @ w, w)

P_tensor = torch.tensor(P_inlp, dtype=torch.float32, device=device)

def inlp_hook_final(resid, hook, projection_matrix):
    resid[:, -1, :] = resid[:, -1, :] @ projection_matrix.T
    return resid

# Mean subtraction at L10
male_acts = []
female_acts = []
for prompt in male_prompts:
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    male_acts.append(cache["blocks.10.hook_resid_post"][0, -1, :].cpu())
for prompt in female_prompts:
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    female_acts.append(cache["blocks.10.hook_resid_post"][0, -1, :].cpu())
gender_dir = (torch.stack(male_acts).mean(0) - torch.stack(female_acts).mean(0))
gender_dir = gender_dir / gender_dir.norm()

def mean_sub_final(resid, hook, direction):
    proj = (resid @ direction.to(resid.device)).unsqueeze(-1) * direction.to(resid.device)
    return resid - proj

ALL_METHODS = {
    "Baseline": None,
    "INLP (L10, 3 iter)": [("blocks.10.hook_resid_post", partial(inlp_hook_final, projection_matrix=P_tensor))],
    "Mean-subtraction (L10)": [("blocks.10.hook_resid_post", partial(mean_sub_final, direction=gender_dir))],
    "Head L10H9 ablation (ours)": [("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.0))],
    "True gender features (ours)": [("blocks.10.hook_resid_pre",
        partial(scale_sae_features, sae=sae_l10, feature_alphas={23440: 0.0, 16291: 0.0}))],
    "Combined L10H9+features (ours)": [
        ("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.0)),
        ("blocks.10.hook_resid_pre",
         partial(scale_sae_features, sae=sae_l10, feature_alphas={23440: 0.0, 16291: 0.0})),
    ],
}

comparison_results = {}
print("\n%-40s %10s %10s %10s" % ("Method", "Bias Red%", "PPL Change%", "Pronoun%"))
print("-" * 75)

for name, hooks in tqdm(ALL_METHODS.items(), desc="Comparing"):
    h = hooks if hooks else None
    bias = eval_bias(model, h)
    ppl = eval_ppl(model, h)
    pronoun = eval_pronoun(model, h)

    bias_red = (baseline_bias - bias) / baseline_bias * 100 if name != "Baseline" else 0
    ppl_chg = (ppl - baseline_ppl) / baseline_ppl * 100 if name != "Baseline" else 0

    comparison_results[name] = {
        "bias": float(bias), "bias_reduction_pct": float(bias_red),
        "ppl": float(ppl), "ppl_change_pct": float(ppl_chg),
        "pronoun": float(pronoun),
    }
    print("%-40s %+9.1f%% %+9.1f%% %9.0f%%" % (name, bias_red, ppl_chg, pronoun*100))


# ── Save ──
all_results = {
    "comparison": comparison_results,
    "jaccard_null": null_results,
    "baseline": {
        "bias": float(baseline_bias),
        "ppl": float(baseline_ppl),
        "pronoun": float(baseline_pronoun),
    },
}

with open(RESULTS_DIR / "baselines_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\nResults saved to %s" % RESULTS_DIR)
print("\n✓ Experiment 15 complete.")
