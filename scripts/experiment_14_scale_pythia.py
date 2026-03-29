"""
Experiment 14: Scale Replication on Pythia-2.8B

Replicate core findings on a larger model to show generalization:
1. Head-level ablation scan to find separable bias heads
2. Entanglement analysis (bias reduction vs PPL for each head)
3. Verify late-layer localization pattern
4. Compare to GPT-2-small findings
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

RESULTS_DIR = Path("results/14_pythia_scale")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Try Pythia-2.8B first, fall back to 1.4B or GPT-2-medium
MODEL_NAME = None
for candidate in ["pythia-2.8b", "pythia-1.4b", "gpt2-medium"]:
    try:
        print("Trying to load %s..." % candidate)
        model = HookedTransformer.from_pretrained(candidate, device=device)
        MODEL_NAME = candidate
        print("Loaded %s: %d layers, %d heads, d_model=%d" % (
            candidate, model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_model))
        break
    except Exception as e:
        print("  Failed: %s" % str(e)[:100])
        continue

if MODEL_NAME is None:
    print("No larger model available. Exiting.")
    exit(1)

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

# Gender tokens — need to find these for the loaded model's tokenizer
male_words = [" he", " him", " his", " himself", " man", " boy", " male", " father",
              " son", " brother", " husband", " king"]
female_words = [" she", " her", " hers", " herself", " woman", " girl", " female", " mother",
                " daughter", " sister", " wife", " queen"]

male_ids = []
female_ids = []
for w in male_words:
    toks = model.to_tokens(w, prepend_bos=False).squeeze()
    if toks.dim() == 0:
        male_ids.append(toks.item())
    else:
        male_ids.append(toks[0].item())

for w in female_words:
    toks = model.to_tokens(w, prepend_bos=False).squeeze()
    if toks.dim() == 0:
        female_ids.append(toks.item())
    else:
        female_ids.append(toks[0].item())

print("Male token IDs (%d): %s" % (len(male_ids), male_ids[:5]))
print("Female token IDs (%d): %s" % (len(female_ids), female_ids[:5]))


# ── Evaluation prompts ──
OCCUPATIONS = [
    "nurse", "doctor", "engineer", "secretary", "CEO", "teacher",
    "programmer", "homemaker", "mechanic", "receptionist", "pilot",
    "librarian", "surgeon", "nanny", "janitor", "hairdresser",
    "plumber", "dancer", "firefighter", "babysitter", "accountant",
    "lawyer", "chef", "farmer", "dentist", "professor",
]

TEMPLATES = [
    "The %s said that",
    "The %s walked to the",
    "The %s announced that",
    "The %s explained that",
    "The %s mentioned that",
]

bias_prompts = []
for occ in OCCUPATIONS:
    for tmpl in TEMPLATES:
        bias_prompts.append(tmpl % occ)
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
    "Music has the ability to evoke powerful emotions and bring people together.",
    "The professor explained the concept of quantum entanglement to the class.",
    "The ancient ruins tell the story of a civilization that flourished.",
    "The startup raised fifty million dollars in its latest round of funding.",
]


def eval_bias(model, prompts, hooks=None):
    scores = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
        probs = torch.softmax(logits[0, -1, :], dim=0)
        pm = probs[male_ids].sum().item()
        pf = probs[female_ids].sum().item()
        scores.append(abs(pm - pf))
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


def scale_head(z, hook, head_idx, alpha):
    z[:, :, head_idx, :] *= alpha
    return z


# ═══════════════════════════════════════════════
# BASELINE
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("BASELINE: %s" % MODEL_NAME)
print("="*70)

baseline_bias = eval_bias(model, bias_prompts)
baseline_ppl = eval_ppl(model)
print("Baseline: bias=%.6f, ppl=%.2f" % (baseline_bias, baseline_ppl))

# Check default male hypothesis
ambiguous = ["The person said that", "Someone mentioned that",
             "The individual walked to", "A student said that",
             "The worker explained that"]
for prompt in ambiguous:
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)
    probs = torch.softmax(logits[0, -1, :], dim=0)
    pm = probs[male_ids].sum().item()
    pf = probs[female_ids].sum().item()
    print("  %-40s P(M)=%.4f P(F)=%.4f skew=%+.4f" % (prompt, pm, pf, pm-pf))


# ═══════════════════════════════════════════════
# FULL HEAD SCAN
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("HEAD ABLATION SCAN (%d layers x %d heads = %d)" % (n_layers, n_heads, n_layers * n_heads))
print("="*70)

# Use a smaller prompt set for the full scan
scan_prompts = bias_prompts[:50]
scan_baseline = eval_bias(model, scan_prompts)
scan_baseline_ppl = eval_ppl(model)

head_results = {}
total_heads = n_layers * n_heads

for layer in tqdm(range(n_layers), desc="Layer scan"):
    for head in range(n_heads):
        hooks = [("blocks.%d.attn.hook_z" % layer, partial(scale_head, head_idx=head, alpha=0.0))]
        bias = eval_bias(model, scan_prompts, hooks)
        ppl = eval_ppl(model, hooks)
        bias_red = (scan_baseline - bias) / scan_baseline * 100
        ppl_change = (ppl - scan_baseline_ppl) / scan_baseline_ppl * 100

        key = "L%dH%d" % (layer, head)
        head_results[key] = {
            "layer": layer, "head": head,
            "bias": float(bias), "bias_reduction_pct": float(bias_red),
            "ppl": float(ppl), "ppl_change_pct": float(ppl_change),
        }

# Find separable heads (high bias reduction, low PPL cost)
sorted_by_bias = sorted(head_results.items(), key=lambda x: x[1]["bias_reduction_pct"], reverse=True)

print("\nTop 15 heads by bias reduction:")
print("%-10s %12s %12s %12s" % ("Head", "Bias Red%", "PPL Change%", "Category"))
print("-" * 50)
for name, r in sorted_by_bias[:15]:
    cat = "SEPARABLE" if r["ppl_change_pct"] < 5 else "ENTANGLED"
    print("%-10s %+11.1f%% %+11.1f%% %12s" % (name, r["bias_reduction_pct"], r["ppl_change_pct"], cat))

# Find most separable (high bias reduction + low PPL cost)
for name, r in sorted_by_bias:
    if r["ppl_change_pct"] < 2 and r["bias_reduction_pct"] > 5:
        print("\nBest separable head: %s (bias_red=%.1f%%, ppl=%.1f%%)" % (
            name, r["bias_reduction_pct"], r["ppl_change_pct"]))
        break

# Layer-level localization
print("\nBias reduction by layer (mean of all heads):")
for layer in range(n_layers):
    layer_heads = [r for name, r in head_results.items() if r["layer"] == layer]
    avg_bias = np.mean([r["bias_reduction_pct"] for r in layer_heads])
    max_bias = max([r["bias_reduction_pct"] for r in layer_heads])
    max_head = [r["head"] for r in layer_heads if r["bias_reduction_pct"] == max_bias][0]
    bar = "#" * int(max(0, avg_bias) * 2)
    print("  L%-3d: avg=%+6.1f%%, max=%+6.1f%% (H%d) %s" % (
        layer, avg_bias, max_bias, max_head, bar))


# ═══════════════════════════════════════════════
# ENTANGLEMENT ANALYSIS ON TOP HEADS
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("ENTANGLEMENT ANALYSIS (Top 10 bias heads, full prompt set)")
print("="*70)

top_heads = [(r["layer"], r["head"]) for _, r in sorted_by_bias[:10]]

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
]

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

baseline_pronoun = eval_pronoun(model)

entanglement_results = {}
print("%-10s %12s %12s %12s %12s" % ("Head", "Bias Red%", "PPL Change%", "Pronoun", "Category"))
print("-" * 60)

for layer, head in tqdm(top_heads, desc="Entanglement"):
    hooks = [("blocks.%d.attn.hook_z" % layer, partial(scale_head, head_idx=head, alpha=0.0))]
    bias = eval_bias(model, bias_prompts, hooks)
    ppl = eval_ppl(model, hooks)
    pronoun = eval_pronoun(model, hooks)

    bias_red = (baseline_bias - bias) / baseline_bias * 100
    ppl_change = (ppl - baseline_ppl) / baseline_ppl * 100

    key = "L%dH%d" % (layer, head)
    cat = "SEPARABLE" if ppl_change < 5 and pronoun >= 0.9 else "ENTANGLED"
    entanglement_results[key] = {
        "layer": layer, "head": head,
        "bias_reduction_pct": float(bias_red),
        "ppl_change_pct": float(ppl_change),
        "pronoun": float(pronoun),
        "category": cat,
    }
    print("%-10s %+11.1f%% %+11.1f%% %11.0f%% %12s" % (
        key, bias_red, ppl_change, pronoun*100, cat))


# ═══════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════

# Heatmap: bias reduction per head
bias_matrix = np.zeros((n_layers, n_heads))
ppl_matrix = np.zeros((n_layers, n_heads))
for name, r in head_results.items():
    bias_matrix[r["layer"], r["head"]] = r["bias_reduction_pct"]
    ppl_matrix[r["layer"], r["head"]] = r["ppl_change_pct"]

fig, axes = plt.subplots(1, 2, figsize=(16, max(6, n_layers * 0.4)))

ax = axes[0]
im = ax.imshow(bias_matrix, aspect='auto', cmap='RdBu_r', vmin=-20, vmax=20)
ax.set_xlabel("Head")
ax.set_ylabel("Layer")
ax.set_title("%s: Bias Reduction per Head (%%)" % MODEL_NAME)
plt.colorbar(im, ax=ax)

ax = axes[1]
im = ax.imshow(ppl_matrix, aspect='auto', cmap='RdYlGn_r', vmin=-5, vmax=50)
ax.set_xlabel("Head")
ax.set_ylabel("Layer")
ax.set_title("%s: PPL Change per Head (%%)" % MODEL_NAME)
plt.colorbar(im, ax=ax)

plt.suptitle("Scale Replication: %s (%d layers, %d heads)" % (MODEL_NAME, n_layers, n_heads), fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "scale_heatmap.png", dpi=150)
plt.close()

# Pareto plot
fig, ax = plt.subplots(figsize=(10, 7))
for name, r in head_results.items():
    color = 'red' if name in [k for k, v in entanglement_results.items() if v["category"] == "SEPARABLE"] else 'gray'
    alpha = 0.8 if color == 'red' else 0.2
    size = 50 if color == 'red' else 10
    ax.scatter(r["bias_reduction_pct"], r["ppl_change_pct"], c=color, s=size, alpha=alpha)

for key, r in entanglement_results.items():
    ax.annotate(key, (head_results[key]["bias_reduction_pct"], head_results[key]["ppl_change_pct"]),
                fontsize=7, ha='center')

ax.set_xlabel("Bias Reduction (%)")
ax.set_ylabel("PPL Change (%)")
ax.set_title("%s: Bias-Capability Tradeoff (Separable in Red)" % MODEL_NAME)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "scale_pareto.png", dpi=150)
plt.close()


# ── Save ──
results = {
    "model": MODEL_NAME,
    "n_layers": n_layers,
    "n_heads": n_heads,
    "baseline": {
        "bias": float(baseline_bias),
        "ppl": float(baseline_ppl),
        "pronoun": float(baseline_pronoun),
    },
    "head_results": head_results,
    "entanglement": entanglement_results,
    "top_separable": [k for k, v in entanglement_results.items() if v["category"] == "SEPARABLE"],
    "top_entangled": [k for k, v in entanglement_results.items() if v["category"] == "ENTANGLED"],
}

with open(RESULTS_DIR / "pythia_scale_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to %s" % RESULTS_DIR)
print("\n✓ Experiment 14 complete.")
