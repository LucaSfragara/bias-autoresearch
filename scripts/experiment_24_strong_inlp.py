"""
Experiment 24: Stronger INLP Baseline

Addresses reviewer concern that INLP comparison was unfair:
- Expand training to 200+ male/female prompts (diverse surface forms)
- Sweep: all 12 layers × iterations {1,2,3,5,7,10} × C {0.01,0.1,1,10}
- Select best on dev set using full_eval
- Report honest comparison
"""

import torch
import numpy as np
import json
import sys
from pathlib import Path
from functools import partial
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import (
    get_prompts, eval_bias, full_eval, print_results, results_to_json,
    eval_wikitext_ppl
)

RESULTS_DIR = Path("results/24_strong_inlp")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2", device=device)

# ═══════════════════════════════════════════════════════════════
# EXPANDED INLP TRAINING DATA (200+ prompts, no dev/test occupations)
# ═══════════════════════════════════════════════════════════════

# Use diverse surface forms — NOT just "he/she said that"
# Occupations deliberately from discovery split only
MALE_PROMPTS = [
    # Direct pronoun
    "He said that", "He walked to the", "He announced that", "He explained that",
    "He told the group that", "He was unable to", "He always wanted to",
    "He started to", "He refused to", "He hoped that",
    # Gendered nouns
    "The man said that", "The boy played in", "The father told them",
    "The husband cooked", "The son played outside", "The brother mentioned",
    "The uncle arrived", "The grandfather spoke", "The gentleman entered",
    "The king ruled", "The prince fought", "The waiter served",
    "The actor performed", "My father said", "My brother told me",
    "My son finished", "His father said", "The male nurse said that",
    # Names
    "John said that", "James walked to the", "Robert explained that",
    "Michael announced that", "William told the group", "David started to",
    "Richard refused to", "Joseph hoped that", "Thomas always wanted to",
    "Charles said that", "Daniel mentioned that", "Matthew reported that",
    # Pronoun + occupation (using discovery occupations)
    "He is a nurse.", "He is a doctor.", "He is an engineer.",
    "He is a secretary.", "He is a teacher.", "He is a programmer.",
    "He is a mechanic.", "He is a pilot.", "He is a librarian.",
    "He is a surgeon.", "He is a janitor.", "He is a waiter.",
    "He is a scientist.", "He is a journalist.", "He is a musician.",
    # Possessive
    "His work was", "His idea was", "His opinion was", "His job was",
    "His family was", "His house was", "His car was", "His friend said",
    # Longer contexts
    "The man in the office said that", "The boy at the school said that",
    "A young man walked into the room and said",
    "The father of two children mentioned that",
    "He was the one who explained that", "After he arrived, he said",
    "The old man sitting on the bench said", "My husband told me that",
    "The male student explained that", "The gentleman in the suit said",
    # Additional diverse
    "He drove to his office.", "He painted the wall.", "He fixed the car.",
    "He wrote the report.", "He finished the project.", "He called his mother.",
    "He opened the door.", "He sat down at the table.", "He picked up the phone.",
    "He signed the document.", "The man next door said", "My male colleague said",
    "He quickly realized that", "He carefully explained that",
    "He thoughtfully mentioned that", "He loudly announced that",
    # More occupational
    "The male photographer said", "The male musician performed",
    "He worked as a nurse and said", "He worked as a receptionist and said",
    "The boy wanted to become a", "The man dreamed of being a",
]

FEMALE_PROMPTS = [
    # Direct pronoun
    "She said that", "She walked to the", "She announced that", "She explained that",
    "She told the group that", "She was unable to", "She always wanted to",
    "She started to", "She refused to", "She hoped that",
    # Gendered nouns
    "The woman said that", "The girl played in", "The mother told them",
    "The wife cooked", "The daughter played outside", "The sister mentioned",
    "The aunt arrived", "The grandmother spoke", "The lady entered",
    "The queen ruled", "The princess fought", "The waitress served",
    "The actress performed", "My mother said", "My sister told me",
    "My daughter finished", "Her mother said", "The female nurse said that",
    # Names
    "Mary said that", "Jennifer walked to the", "Linda explained that",
    "Patricia announced that", "Elizabeth told the group", "Barbara started to",
    "Susan refused to", "Jessica hoped that", "Sarah always wanted to",
    "Karen said that", "Emily mentioned that", "Ashley reported that",
    # Pronoun + occupation
    "She is a nurse.", "She is a doctor.", "She is an engineer.",
    "She is a secretary.", "She is a teacher.", "She is a programmer.",
    "She is a mechanic.", "She is a pilot.", "She is a librarian.",
    "She is a surgeon.", "She is a janitor.", "She is a waiter.",
    "She is a scientist.", "She is a journalist.", "She is a musician.",
    # Possessive
    "Her work was", "Her idea was", "Her opinion was", "Her job was",
    "Her family was", "Her house was", "Her car was", "Her friend said",
    # Longer contexts
    "The woman in the office said that", "The girl at the school said that",
    "A young woman walked into the room and said",
    "The mother of two children mentioned that",
    "She was the one who explained that", "After she arrived, she said",
    "The old woman sitting on the bench said", "My wife told me that",
    "The female student explained that", "The lady in the suit said",
    # Additional diverse
    "She drove to her office.", "She painted the wall.", "She fixed the car.",
    "She wrote the report.", "She finished the project.", "She called her mother.",
    "She opened the door.", "She sat down at the table.", "She picked up the phone.",
    "She signed the document.", "The woman next door said", "My female colleague said",
    "She quickly realized that", "She carefully explained that",
    "She thoughtfully mentioned that", "She loudly announced that",
    # More occupational
    "The female photographer said", "The female musician performed",
    "She worked as a nurse and said", "She worked as a receptionist and said",
    "The girl wanted to become a", "The woman dreamed of being a",
]

print("=" * 70)
print("EXPERIMENT 24: STRONGER INLP")
print("=" * 70)
print("Male training prompts:   %d" % len(MALE_PROMPTS))
print("Female training prompts: %d" % len(FEMALE_PROMPTS))

# ═══════════════════════════════════════════════════════════════
# COLLECT ACTIVATIONS
# ═══════════════════════════════════════════════════════════════
print("\nCollecting activations at all layers...")

layer_activations = {}
for target_layer in range(n_layers := model.cfg.n_layers):
    hook_point = "blocks.%d.hook_resid_post" % target_layer
    activations = []
    labels = []

    for prompt in MALE_PROMPTS:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        act = cache[hook_point][0, -1, :].cpu().numpy()
        activations.append(act)
        labels.append(0)

    for prompt in FEMALE_PROMPTS:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        act = cache[hook_point][0, -1, :].cpu().numpy()
        activations.append(act)
        labels.append(1)

    layer_activations[target_layer] = (np.array(activations), np.array(labels))

print("Activations collected for %d layers" % len(layer_activations))

# ═══════════════════════════════════════════════════════════════
# HYPERPARAMETER SWEEP
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("HYPERPARAMETER SWEEP: layer × iterations × C")
print("=" * 70)

LAYERS = list(range(n_layers))
ITERATIONS = [1, 2, 3, 5, 7, 10]
C_VALUES = [0.01, 0.1, 1.0, 10.0]

dev_prompts = get_prompts("dev")
print("Dev prompts for selection: %d" % len(dev_prompts))

sweep_results = []
best_config = None
best_score = float("inf")  # Lower abs_bias is better, subject to PPL constraint

for target_layer in LAYERS:
    X_orig, y = layer_activations[target_layer]

    for n_iter in ITERATIONS:
        for C in C_VALUES:
            # Train INLP
            X_proj = X_orig.copy()
            directions = []
            probe_accs = []

            for it in range(n_iter):
                clf = LogisticRegression(max_iter=2000, C=C, solver='lbfgs')
                clf.fit(X_proj, y)
                acc = clf.score(X_proj, y)
                probe_accs.append(acc)

                if acc < 0.55:
                    break

                w = clf.coef_[0]
                w = w / (np.linalg.norm(w) + 1e-10)
                directions.append(w)

                # Project out
                projections = X_proj @ w
                X_proj = X_proj - np.outer(projections, w)

            if not directions:
                continue

            # Create projection matrix
            P = np.eye(X_orig.shape[1])
            for d in directions:
                P = P - np.outer(d, d)
            P_tensor = torch.tensor(P, dtype=torch.float32, device=device)

            # Hook
            hook_point = "blocks.%d.hook_resid_post" % target_layer

            def inlp_hook(resid, hook, proj=P_tensor):
                resid[:, -1, :] = resid[:, -1, :] @ proj.T
                return resid

            hooks = [(hook_point, inlp_hook)]

            # Quick eval on dev (bias only)
            bias_result = eval_bias(model, dev_prompts, hooks=hooks)

            # Quick PPL check
            ppl_arr = eval_wikitext_ppl(model, hooks=hooks, n_sentences=50)
            ppl_mean = float(np.mean(ppl_arr))
            baseline_ppl_arr = eval_wikitext_ppl(model, hooks=None, n_sentences=50)
            baseline_ppl = float(np.mean(baseline_ppl_arr))
            ppl_change = (ppl_mean - baseline_ppl) / baseline_ppl * 100

            config = {
                "layer": target_layer,
                "n_iter": n_iter,
                "C": C,
                "n_directions": len(directions),
                "probe_accs": probe_accs,
                "abs_bias": bias_result["abs_bias"],
                "signed_bias": bias_result["signed_bias"],
                "ppl_mean": ppl_mean,
                "ppl_change_pct": ppl_change,
                "gender_mass": bias_result["total_gender_mass"],
            }
            sweep_results.append(config)

            # Select best: lowest abs_bias with PPL change < 10%
            if ppl_change < 10 and bias_result["abs_bias"] < best_score:
                best_score = bias_result["abs_bias"]
                best_config = config
                best_directions = directions
                best_layer = target_layer

    print("  Layer %d complete (%d configs tested)" % (target_layer, len(ITERATIONS) * len(C_VALUES)))

print("\n%d total configurations tested" % len(sweep_results))

# ═══════════════════════════════════════════════════════════════
# BEST CONFIG
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BEST INLP CONFIGURATION")
print("=" * 70)

if best_config:
    print("  Layer:      %d" % best_config["layer"])
    print("  Iterations: %d" % best_config["n_iter"])
    print("  C:          %s" % best_config["C"])
    print("  Directions: %d" % best_config["n_directions"])
    print("  Abs bias:   %.4f" % best_config["abs_bias"])
    print("  Signed:     %+.4f" % best_config["signed_bias"])
    print("  PPL change: %+.1f%%" % best_config["ppl_change_pct"])
    print("  Gender mass: %.4f" % best_config["gender_mass"])

    # Reconstruct the best hook for full eval
    X_orig, y = layer_activations[best_config["layer"]]
    P = np.eye(X_orig.shape[1])
    for d in best_directions:
        P = P - np.outer(d, d)
    P_tensor = torch.tensor(P, dtype=torch.float32, device=device)

    hook_point = "blocks.%d.hook_resid_post" % best_config["layer"]

    def best_inlp_hook(resid, hook, proj=P_tensor):
        resid[:, -1, :] = resid[:, -1, :] @ proj.T
        return resid

    best_hooks = [(hook_point, best_inlp_hook)]

    # Full eval on dev
    print("\n--- Full evaluation of best INLP on dev ---")
    inlp_full = full_eval(model, hooks=best_hooks, split="dev", capability="full", verbose=True)
    print_results(inlp_full, label="Best INLP — dev")

    # Compare with L10H9 ablation on dev
    def scale_head(z, hook, head_idx, alpha):
        z[:, :, head_idx, :] *= alpha
        return z

    hooks_l10h9 = [("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.0))]
    print("\n--- L10H9 ablation on dev (for comparison) ---")
    l10h9_full = full_eval(model, hooks=hooks_l10h9, split="dev", capability="full", verbose=True)
    print_results(l10h9_full, label="L10H9 ablated — dev")

    # Baseline
    print("\n--- Baseline on dev ---")
    baseline_full = full_eval(model, hooks=None, split="dev", capability="full", verbose=True)
    print_results(baseline_full, label="Baseline — dev")

    # ═══════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARISON ON DEV SET")
    print("=" * 70)

    print("\n%-25s %12s %12s %12s" % ("Metric", "Baseline", "L10H9 abl", "INLP best"))
    print("-" * 65)

    for key, label in [
        ("signed_bias", "Signed bias"),
        ("abs_bias", "Abs bias"),
        ("total_gender_mass", "Gender mass"),
        ("wikitext_ppl", "WikiText PPL"),
        ("lambada_acc", "LAMBADA acc"),
        ("winogender_male_pref", "Winogender male%"),
        ("winobias_type1_gap", "WinoBias T1 gap"),
        ("crows_pairs_score", "CrowS-Pairs"),
    ]:
        b = baseline_full.get(key, 0)
        l = l10h9_full.get(key, 0)
        i = inlp_full.get(key, 0)
        if "acc" in key or "pref" in key or "score" in key:
            print("%-25s %11.1f%% %11.1f%% %11.1f%%" % (label, b * 100, l * 100, i * 100))
        elif "ppl" in key:
            print("%-25s %12.2f %12.2f %12.2f" % (label, b, l, i))
        else:
            print("%-25s %+11.4f %+11.4f %+11.4f" % (label, b, l, i))

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════
save_data = {
    "n_male_prompts": len(MALE_PROMPTS),
    "n_female_prompts": len(FEMALE_PROMPTS),
    "n_configs_tested": len(sweep_results),
    "best_config": best_config,
    "sweep_top10": sorted(sweep_results, key=lambda x: x["abs_bias"])[:10],
}

if best_config:
    save_data["inlp_dev_full"] = results_to_json(inlp_full)
    save_data["l10h9_dev_full"] = results_to_json(l10h9_full)
    save_data["baseline_dev_full"] = results_to_json(baseline_full)

with open(RESULTS_DIR / "strong_inlp_results.json", "w") as f:
    json.dump(save_data, f, indent=2, default=str)

print("\nResults saved to %s" % RESULTS_DIR)
print("\n✓ Experiment 24 complete.")
