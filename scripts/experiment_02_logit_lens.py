"""
Experiment 02: Logit Lens Analysis - Bias Formation Dynamics
- Project intermediate residual stream to vocabulary space at each layer
- Track when gendered tokens gain/lose probability
- Visualize bias "crystallization" across layers
"""

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

RESULTS_DIR = Path("results/02_logit_lens")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading GPT-2 small...")
model = HookedTransformer.from_pretrained("gpt2", device=device)
n_layers = model.cfg.n_layers

with open("results/00_setup/gender_token_ids.json") as f:
    gender_token_ids = json.load(f)

male_ids = gender_token_ids["male"]
female_ids = gender_token_ids["female"]

# Prompts to analyze: stereotypically gendered occupations
PROMPTS = {
    "nurse": "The nurse said that",
    "doctor": "The doctor said that",
    "engineer": "The engineer said that",
    "secretary": "The secretary said that",
    "CEO": "The CEO said that",
    "teacher": "The teacher said that",
    "programmer": "The programmer said that",
    "homemaker": "The homemaker said that",
    "mechanic": "The mechanic said that",
    "receptionist": "The receptionist said that",
    "pilot": "The pilot said that",
    "librarian": "The librarian said that",
    # Control: explicitly gendered
    "he_explicit": "He said that",
    "she_explicit": "She said that",
    # Control: neutral
    "person": "The person said that",
    "someone": "Someone said that",
}


def logit_lens_analysis(model, prompt, male_ids, female_ids):
    """
    At each layer, project the residual stream to vocab space via the unembedding matrix.
    Returns per-layer bias scores and top predicted tokens.
    """
    tokens = model.to_tokens(prompt)

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    layer_data = []
    for layer in range(n_layers):
        # Get residual stream after this layer
        resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]  # last token position

        # Apply layer norm (important for accurate projection)
        normed = model.ln_final(resid)

        # Project to vocabulary space
        logits = model.unembed(normed.unsqueeze(0).unsqueeze(0))[0, 0, :]
        probs = torch.softmax(logits, dim=0)

        male_prob = probs[male_ids].sum().item()
        female_prob = probs[female_ids].sum().item()
        bias_score = male_prob - female_prob

        # Top 5 predicted tokens
        top5_indices = torch.topk(probs, 5).indices.tolist()
        top5_tokens = [model.to_string([idx]) for idx in top5_indices]
        top5_probs = [probs[idx].item() for idx in top5_indices]

        layer_data.append({
            "layer": layer,
            "male_prob": male_prob,
            "female_prob": female_prob,
            "bias_score": bias_score,
            "top5_tokens": top5_tokens,
            "top5_probs": top5_probs,
        })

    return layer_data


# ── Run logit lens on all prompts ──
print("\nRunning logit lens analysis...")
all_results = {}

for name, prompt in tqdm(PROMPTS.items(), desc="Logit lens"):
    all_results[name] = logit_lens_analysis(model, prompt, male_ids, female_ids)

# ── Visualization 1: Bias score across layers for each prompt ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Male-stereotyped occupations
ax = axes[0, 0]
male_stereo = ["doctor", "engineer", "CEO", "programmer", "mechanic", "pilot"]
for name in male_stereo:
    scores = [d["bias_score"] for d in all_results[name]]
    ax.plot(range(n_layers), scores, marker='o', markersize=3, label=name)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel("Layer")
ax.set_ylabel("Bias Score (+ = male)")
ax.set_title("Male-Stereotyped Occupations")
ax.legend(fontsize=8)

# Female-stereotyped occupations
ax = axes[0, 1]
female_stereo = ["nurse", "secretary", "teacher", "homemaker", "receptionist", "librarian"]
for name in female_stereo:
    scores = [d["bias_score"] for d in all_results[name]]
    ax.plot(range(n_layers), scores, marker='o', markersize=3, label=name)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel("Layer")
ax.set_ylabel("Bias Score (+ = male)")
ax.set_title("Female-Stereotyped Occupations")
ax.legend(fontsize=8)

# Controls
ax = axes[1, 0]
controls = ["he_explicit", "she_explicit", "person", "someone"]
for name in controls:
    scores = [d["bias_score"] for d in all_results[name]]
    ax.plot(range(n_layers), scores, marker='o', markersize=3, label=name)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel("Layer")
ax.set_ylabel("Bias Score (+ = male)")
ax.set_title("Control Prompts")
ax.legend(fontsize=8)

# P(male) and P(female) for one example
ax = axes[1, 1]
for name, color_m, color_f in [("nurse", "blue", "red"), ("doctor", "cyan", "orange")]:
    male_probs = [d["male_prob"] for d in all_results[name]]
    female_probs = [d["female_prob"] for d in all_results[name]]
    ax.plot(range(n_layers), male_probs, color=color_m, linestyle='-', marker='o', markersize=3, label=f"{name} P(male)")
    ax.plot(range(n_layers), female_probs, color=color_f, linestyle='--', marker='s', markersize=3, label=f"{name} P(female)")
ax.set_xlabel("Layer")
ax.set_ylabel("Probability")
ax.set_title("P(male) vs P(female) Across Layers")
ax.legend(fontsize=8)

plt.suptitle("Logit Lens: Gender Bias Formation Across Layers (GPT-2 Small)", fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "logit_lens_bias_formation.png", dpi=150)
plt.close()

# ── Visualization 2: Heatmap of bias scores ──
all_names = list(PROMPTS.keys())
bias_matrix = np.zeros((len(all_names), n_layers))
for i, name in enumerate(all_names):
    for j in range(n_layers):
        bias_matrix[i, j] = all_results[name][j]["bias_score"]

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(bias_matrix, aspect='auto', cmap='RdBu_r',
               vmin=-np.max(np.abs(bias_matrix)), vmax=np.max(np.abs(bias_matrix)))
ax.set_yticks(range(len(all_names)))
ax.set_yticklabels(all_names)
ax.set_xlabel("Layer")
ax.set_ylabel("Prompt")
ax.set_title("Logit Lens: Bias Score Heatmap (blue=female, red=male)")
plt.colorbar(im, ax=ax, label="Bias Score")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "logit_lens_heatmap.png", dpi=150)
plt.close()

# ── Visualization 3: Bias crystallization point ──
# For each prompt, find the layer where bias becomes "stable"
# (defined as the first layer after which the sign doesn't change)
print("\nBias crystallization analysis:")
print(f"{'Prompt':<20} {'Crystal. Layer':>15} {'Final Bias':>12} {'Direction':>10}")
print("-" * 60)

crystallization = {}
for name in all_names:
    scores = [all_results[name][j]["bias_score"] for j in range(n_layers)]
    final_sign = 1 if scores[-1] > 0 else -1

    crystal_layer = 0
    for layer in range(n_layers):
        sign = 1 if scores[layer] > 0 else -1
        if sign != final_sign:
            crystal_layer = layer + 1

    direction = "male" if final_sign > 0 else "female"
    crystallization[name] = {
        "crystal_layer": crystal_layer,
        "final_bias": scores[-1],
        "direction": direction,
    }
    print(f"{name:<20} {crystal_layer:>15} {scores[-1]:>12.4f} {direction:>10}")

# ── Save results ──
save_results = {
    "prompts": PROMPTS,
    "per_prompt_per_layer": {name: data for name, data in all_results.items()},
    "crystallization": crystallization,
    "bias_matrix": bias_matrix.tolist(),
}

with open(RESULTS_DIR / "logit_lens_results.json", "w") as f:
    json.dump(save_results, f, indent=2)

print(f"\nResults saved to {RESULTS_DIR}")
print("\n✓ Experiment 02 complete.")
