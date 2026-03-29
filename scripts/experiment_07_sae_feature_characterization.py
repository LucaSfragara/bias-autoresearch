"""
Experiment 07: SAE Feature Characterization
- What does SAE Feature 23406 (Layer 0) actually encode?
- What does Feature 23440 (Layer 10) encode?
- Find top activating examples from a corpus
- Test on diverse prompts to understand the feature's semantics
- Visualize feature activation patterns
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
from sae_lens import SAE

RESULTS_DIR = Path("results/07_feature_characterization")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model and SAEs...")
model = HookedTransformer.from_pretrained("gpt2", device=device)
sae_l0 = SAE.from_pretrained(release="gpt2-small-res-jb",
                              sae_id="blocks.0.hook_resid_pre", device=device)
sae_l10 = SAE.from_pretrained(release="gpt2-small-res-jb",
                               sae_id="blocks.10.hook_resid_pre", device=device)
print("Loaded.")

# Key features to characterize
FEATURES = {
    "L0_F23406": (sae_l0, 23406, "blocks.0.hook_resid_pre"),
    "L0_F14924": (sae_l0, 14924, "blocks.0.hook_resid_pre"),  # top male-differential
    "L0_F15603": (sae_l0, 15603, "blocks.0.hook_resid_pre"),  # top female-differential
    "L10_F23440": (sae_l10, 23440, "blocks.10.hook_resid_pre"),
    "L10_F11154": (sae_l10, 11154, "blocks.10.hook_resid_pre"),
    "L10_F13794": (sae_l10, 13794, "blocks.10.hook_resid_pre"),  # top male L10
}

# ── Corpus of diverse prompts to test feature activation ──
TEST_PROMPTS = {
    # Gendered occupations
    "nurse_said": "The nurse said",
    "doctor_said": "The doctor said",
    "engineer_said": "The engineer said",
    "secretary_said": "The secretary said",
    "ceo_said": "The CEO said",
    "teacher_said": "The teacher said",
    "programmer_said": "The programmer said",
    "homemaker_said": "The homemaker said",
    "nanny_said": "The nanny said",
    "surgeon_said": "The surgeon said",
    "pilot_said": "The pilot said",
    "receptionist_said": "The receptionist said",

    # Explicitly gendered
    "he_said": "He said",
    "she_said": "She said",
    "man_walked": "The man walked",
    "woman_walked": "The woman walked",
    "boy_played": "The boy played",
    "girl_played": "The girl played",
    "father_told": "My father told",
    "mother_told": "My mother told",

    # Gender-neutral
    "person_said": "The person said",
    "someone_said": "Someone said",
    "they_said": "They said",
    "worker_said": "The worker said",

    # Non-gendered contexts
    "cat_sat": "The cat sat on",
    "weather_today": "The weather today is",
    "stock_market": "The stock market",
    "python_code": "Write a Python function",
    "history_rome": "In ancient Rome",
    "math_equation": "The equation x squared",
    "food_recipe": "To make a good pasta",
    "music_beethoven": "Beethoven composed his",

    # Race-related
    "black_man": "The Black man walked",
    "white_man": "The White man walked",
    "asian_student": "The Asian student",
    "hispanic_community": "The Hispanic community",

    # Religion-related
    "muslim_prayed": "The Muslim man prayed",
    "christian_prayed": "The Christian man prayed",
    "jewish_family": "The Jewish family",

    # Names (gendered)
    "john_said": "John said that",
    "mary_said": "Mary said that",
    "james_walked": "James walked to",
    "emily_walked": "Emily walked to",

    # Possessive/pronoun contexts
    "his_car": "He drove his car",
    "her_car": "She drove her car",
    "his_office": "He went to his office",
    "her_office": "She went to her office",
}


def get_feature_activation(model, sae, prompt, hook_point, feature_idx):
    """Get activation of a specific SAE feature for a prompt, at each token position."""
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    resid = cache[hook_point]  # (1, seq_len, d_model)
    sae_acts = sae.encode(resid)  # (1, seq_len, d_sae)

    # Per-position activation
    per_pos = sae_acts[0, :, feature_idx].detach().cpu().numpy()  # (seq_len,)

    # Last position (used for next-token prediction)
    last_pos = per_pos[-1]

    # Max across positions
    max_act = per_pos.max()

    # Token strings
    token_strs = [model.to_string([t.item()]) for t in tokens[0]]

    return {
        "last_pos_activation": float(last_pos),
        "max_activation": float(max_act),
        "per_position": per_pos.tolist(),
        "tokens": token_strs,
    }


# ── Characterize each feature ──
print("\n" + "="*70)
print("FEATURE CHARACTERIZATION")
print("="*70)

all_characterizations = {}

for feat_name, (sae, feat_idx, hook_point) in FEATURES.items():
    print(f"\n--- {feat_name} (feature {feat_idx}) ---")

    activations = {}
    for prompt_name, prompt in TEST_PROMPTS.items():
        result = get_feature_activation(model, sae, prompt, hook_point, feat_idx)
        activations[prompt_name] = result

    # Sort by last-position activation
    sorted_by_last = sorted(activations.items(), key=lambda x: x[1]["last_pos_activation"], reverse=True)

    print(f"\nTop 15 activating prompts (last position):")
    print(f"{'Prompt':<30} {'Last Pos':>10} {'Max':>10}")
    print("-" * 55)
    for name, data in sorted_by_last[:15]:
        print(f"{name:<30} {data['last_pos_activation']:>10.4f} {data['max_activation']:>10.4f}")

    print(f"\nBottom 5 (lowest activation):")
    for name, data in sorted_by_last[-5:]:
        print(f"{name:<30} {data['last_pos_activation']:>10.4f} {data['max_activation']:>10.4f}")

    # Categorize: which types of prompts activate this feature?
    categories = {
        "male_occupation": ["doctor_said", "engineer_said", "ceo_said", "programmer_said",
                           "surgeon_said", "pilot_said"],
        "female_occupation": ["nurse_said", "secretary_said", "teacher_said", "homemaker_said",
                             "nanny_said", "receptionist_said"],
        "explicit_male": ["he_said", "man_walked", "boy_played", "father_told",
                         "john_said", "james_walked", "his_car", "his_office"],
        "explicit_female": ["she_said", "woman_walked", "girl_played", "mother_told",
                           "mary_said", "emily_walked", "her_car", "her_office"],
        "neutral": ["person_said", "someone_said", "they_said", "worker_said"],
        "non_gendered": ["cat_sat", "weather_today", "stock_market", "python_code",
                        "history_rome", "math_equation", "food_recipe", "music_beethoven"],
    }

    category_means = {}
    for cat, prompts in categories.items():
        vals = [activations[p]["last_pos_activation"] for p in prompts if p in activations]
        category_means[cat] = float(np.mean(vals)) if vals else 0.0

    print(f"\nCategory mean activations:")
    for cat, mean_val in sorted(category_means.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(mean_val * 5)
        print(f"  {cat:<25} {mean_val:>8.4f} {bar}")

    all_characterizations[feat_name] = {
        "feature_idx": feat_idx,
        "activations": {k: {"last": v["last_pos_activation"], "max": v["max_activation"]}
                       for k, v in activations.items()},
        "category_means": category_means,
        "top_activating": [(name, data["last_pos_activation"]) for name, data in sorted_by_last[:10]],
    }


# ── Decoder direction analysis ──
# What does the feature's decoder direction look like in vocabulary space?
print("\n" + "="*70)
print("DECODER DIRECTION ANALYSIS")
print("="*70)
print("(What tokens does each feature point toward in vocabulary space?)")

for feat_name, (sae, feat_idx, hook_point) in FEATURES.items():
    print(f"\n--- {feat_name} ---")

    # Get the decoder direction for this feature
    dec_dir = sae.W_dec[feat_idx]  # (d_model,)

    # Project through the unembedding matrix to get vocabulary logits
    # Need to apply layer norm first for accurate projection
    # For layer 0, this is less meaningful (early layer)
    # For layer 10, this is more meaningful

    normed = model.ln_final(dec_dir.unsqueeze(0).unsqueeze(0))
    vocab_logits = model.unembed(normed)[0, 0, :]  # (vocab_size,)

    # Top tokens this feature pushes toward
    top_k = 20
    top_indices = torch.topk(vocab_logits, top_k).indices
    bottom_indices = torch.topk(-vocab_logits, top_k).indices

    print(f"  Top {top_k} tokens PROMOTED by this feature:")
    promoted = []
    for idx in top_indices:
        tok = model.to_string([idx.item()])
        logit = vocab_logits[idx].item()
        promoted.append((tok, logit))
        print(f"    '{tok}' ({logit:.3f})")

    print(f"  Top {top_k} tokens SUPPRESSED by this feature:")
    suppressed = []
    for idx in bottom_indices:
        tok = model.to_string([idx.item()])
        logit = vocab_logits[idx].item()
        suppressed.append((tok, logit))
        print(f"    '{tok}' ({logit:.3f})")

    all_characterizations[feat_name]["promoted_tokens"] = promoted
    all_characterizations[feat_name]["suppressed_tokens"] = suppressed


# ── Per-token activation visualization for key prompts ──
print("\n" + "="*70)
print("PER-TOKEN ACTIVATION PATTERNS")
print("="*70)

key_prompts = [
    "The nurse said that she was tired",
    "The doctor said that he was tired",
    "The engineer designed a new bridge",
    "The homemaker prepared a meal for the family",
    "The person walked to the store",
    "The cat sat on the mat",
]

for feat_name, (sae, feat_idx, hook_point) in list(FEATURES.items())[:3]:
    print(f"\n--- {feat_name} ---")
    for prompt in key_prompts:
        result = get_feature_activation(model, sae, prompt, hook_point, feat_idx)
        tokens = result["tokens"]
        acts = result["per_position"]
        print(f"  '{prompt}':")
        for tok, act in zip(tokens, acts):
            bar = "█" * int(act * 3) if act > 0 else ""
            print(f"    {tok:<15} {act:>8.4f} {bar}")


# ── Visualization ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for ax_idx, (feat_name, data) in enumerate(all_characterizations.items()):
    if ax_idx >= 6:
        break
    ax = axes[ax_idx // 3, ax_idx % 3]

    cats = list(data["category_means"].keys())
    vals = [data["category_means"][c] for c in cats]
    colors = ['blue' if 'male' in c else 'red' if 'female' in c else 'gray' if 'neutral' in c else 'green'
              for c in cats]

    ax.barh(cats, vals, color=colors, alpha=0.7)
    ax.set_xlabel("Mean Activation")
    ax.set_title(f"{feat_name}")
    ax.invert_yaxis()

plt.suptitle("SAE Feature Category Activation Profiles", fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "feature_category_profiles.png", dpi=150)
plt.close()

# ── Save ──
with open(RESULTS_DIR / "feature_characterization.json", "w") as f:
    json.dump(all_characterizations, f, indent=2)

print(f"\nAll results saved to {RESULTS_DIR}")
print("\n✓ Experiment 07 complete.")
