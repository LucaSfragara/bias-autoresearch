"""
Experiment 08: Expanded Evaluation
- Larger pronoun resolution test set (WinoBias-style)
- More diverse agreement tests
- LAMBADA-style next-word prediction
- Test the best interventions on expanded benchmarks
- Compare GPT-2-small vs GPT-2-medium if feasible
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

RESULTS_DIR = Path("results/08_expanded_eval")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════
# PART 1: EXPANDED EVALUATION ON GPT-2-SMALL
# ═══════════════════════════════════════════════

print("Loading GPT-2 small + SAEs...")
model = HookedTransformer.from_pretrained("gpt2", device=device)
sae_l0 = SAE.from_pretrained(release="gpt2-small-res-jb",
                              sae_id="blocks.0.hook_resid_pre", device=device)

with open("results/00_setup/gender_token_ids.json") as f:
    gender_token_ids = json.load(f)
male_ids = gender_token_ids["male"]
female_ids = gender_token_ids["female"]

# ── Expanded Pronoun Resolution (WinoBias-style) ──
EXPANDED_PRONOUN_TESTS = [
    # Standard coreference
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
    # Longer context
    ("The man was walking down the street when suddenly", " he", " she"),
    ("The woman was walking down the street when suddenly", " she", " he"),
    ("My uncle has always been someone who", " he", " she"),
    ("My aunt has always been someone who", " she", " he"),
    ("The actor delivered the monologue and then", " he", " she"),
    ("The actress delivered the monologue and then", " she", " he"),
    ("My brother said that", " he", " she"),
    ("My sister said that", " she", " he"),
    # With occupations (non-stereotypical — tests if model uses explicit gender cue)
    ("The male nurse helped the patient.", " He", " She"),
    ("The female doctor examined the results.", " She", " He"),
    ("The male secretary organized the files.", " He", " She"),
    ("The female engineer presented the design.", " She", " He"),
    # Possessive resolution
    ("The boy lost track of time doing", " his", " her"),
    ("The girl lost track of time doing", " her", " his"),
    ("The gentleman placed his hat on the table.", " He", " She"),
    ("The lady placed her hat on the table.", " She", " He"),
    # Multiple mentions
    ("Tom went to the market. At the market, Tom bought apples.", " He", " She"),
    ("Sarah went to the market. At the market, Sarah bought apples.", " She", " He"),
    ("The waiter served the food. After serving,", " he", " she"),
    ("The waitress served the food. After serving,", " she", " he"),
    # Reflexive
    ("The man looked at himself in the mirror.", " He", " She"),
    ("The woman looked at herself in the mirror.", " She", " He"),
    ("The boy taught himself to play guitar.", " He", " She"),
    ("The girl taught herself to play guitar.", " She", " He"),
]

# ── Expanded Agreement Tests ──
EXPANDED_AGREEMENT_TESTS = [
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
    # More complex constructions
    ("The teacher who teaches the students", " is", " are"),
    ("The teachers who teach the students", " are", " is"),
    ("The manager of the department", " has", " have"),
    ("The managers of the department", " have", " has"),
    ("The scientist with many publications", " was", " were"),
    ("The scientists with many publications", " were", " was"),
    ("The child playing in the park", " runs", " run"),
    ("The children playing in the park", " run", " runs"),
    ("Every student", " has", " have"),
    ("All students", " have", " has"),
    ("The group of people", " is", " are"),
    ("The people in the group", " are", " is"),
    # Relative clauses (harder)
    ("The book that the student reads", " is", " are"),
    ("The books that the students read", " are", " is"),
    ("The car that the mechanic fixed", " was", " were"),
    ("The cars that the mechanics fixed", " were", " was"),
]

# ── Factual Knowledge Tests ──
FACTUAL_TESTS = [
    ("The capital of France is", " Paris", " London"),
    ("Water boils at 100 degrees", " Celsius", " miles"),
    ("The Earth orbits the", " Sun", " Moon"),
    ("Shakespeare wrote", " Hamlet", " calculus"),
    ("The chemical symbol for water is", " H", " Z"),
    ("Photosynthesis occurs in", " plants", " rocks"),
    ("The speed of light is approximately", " 300", " 50"),
    ("DNA stands for de", " oxy", " ter"),
    ("The largest planet in our solar system is", " Jupiter", " Mars"),
    ("Gravity was described by", " Newton", " Mozart"),
    ("The Great Wall is in", " China", " Brazil"),
    ("Albert Einstein developed the theory of", " relativity", " cooking"),
    ("The Pacific is the largest", " ocean", " desert"),
    ("The human body has 206", " bones", " wheels"),
    ("Oxygen is element number", " 8", " 3"),
]


def eval_accuracy(model, tests, hooks=None):
    """Evaluate accuracy on binary choice tests."""
    correct = 0
    for prompt, ct, it in tests:
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
    return correct / len(tests)


def eval_bias(model, hooks=None):
    prompts = [
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
    scores = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
        probs = torch.softmax(logits[0, -1, :], dim=0)
        scores.append(abs(probs[male_ids].sum().item() - probs[female_ids].sum().item()))
    return np.mean(scores)


def eval_ppl(model, hooks=None):
    sentences = [
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
        "The river flowed gently through the valley reflecting the sunlight.",
        "Technology continues to reshape how we communicate with each other.",
        "The musician performed a beautiful solo that captivated the audience.",
        "Research suggests regular exercise can improve physical and mental health.",
        "The detective examined the evidence carefully before drawing conclusions.",
    ]
    total_loss, total_tokens = 0, 0
    for sent in sentences:
        tokens = model.to_tokens(sent)
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
        log_probs = torch.log_softmax(logits[0, :-1, :], dim=-1)
        targets = tokens[0, 1:]
        total_loss += -log_probs.gather(1, targets.unsqueeze(1)).squeeze().sum().item()
        total_tokens += len(targets)
    return np.exp(total_loss / total_tokens)


# ── Define interventions ──
def zero_head(z, hook, head_idx):
    z[:, :, head_idx, :] = 0.0
    return z

def clamp_sae(resid, hook, sae, features):
    sae_acts = sae.encode(resid)
    total = torch.zeros_like(resid)
    for fi in features:
        total += sae_acts[:, :, fi:fi+1] * sae.W_dec[fi]
    return resid - total

interventions = {
    "Baseline": None,
    "Head L10H9": [("blocks.10.attn.hook_z", partial(zero_head, head_idx=9))],
    "SAE L0 F23406": [("blocks.0.hook_resid_pre",
                        partial(clamp_sae, sae=sae_l0, features=[23406]))],
    "Combined": [
        ("blocks.10.attn.hook_z", partial(zero_head, head_idx=9)),
        ("blocks.0.hook_resid_pre", partial(clamp_sae, sae=sae_l0, features=[23406])),
    ],
}

# ── Run expanded evaluation ──
print("\n" + "="*80)
print("EXPANDED EVALUATION ON GPT-2-SMALL")
print("="*80)

results = {}
for name, hooks in tqdm(interventions.items(), desc="Evaluating"):
    h = hooks if hooks else None
    r = {
        "bias": float(eval_bias(model, h)),
        "ppl": float(eval_ppl(model, h)),
        "pronoun_basic": float(eval_accuracy(model, EXPANDED_PRONOUN_TESTS[:16], h)),
        "pronoun_expanded": float(eval_accuracy(model, EXPANDED_PRONOUN_TESTS, h)),
        "agreement_basic": float(eval_accuracy(model, EXPANDED_AGREEMENT_TESTS[:16], h)),
        "agreement_expanded": float(eval_accuracy(model, EXPANDED_AGREEMENT_TESTS, h)),
        "factual": float(eval_accuracy(model, FACTUAL_TESTS, h)),
    }
    results[name] = r

print(f"\n{'Intervention':<20} {'Bias':>8} {'PPL':>8} {'Pron16':>8} {'PronAll':>8} "
      f"{'Agr16':>8} {'AgrAll':>8} {'Factual':>8}")
print("-" * 90)
for name, r in results.items():
    print(f"{name:<20} {r['bias']:>8.4f} {r['ppl']:>8.2f} {r['pronoun_basic']:>7.0%} "
          f"{r['pronoun_expanded']:>7.0%} {r['agreement_basic']:>7.0%} "
          f"{r['agreement_expanded']:>7.0%} {r['factual']:>7.0%}")


# ═══════════════════════════════════════════════
# PART 2: GPT-2-MEDIUM COMPARISON
# ═══════════════════════════════════════════════

print("\n" + "="*80)
print("GPT-2-MEDIUM: BASELINE COMPARISON")
print("="*80)

# Free GPU memory
del model, sae_l0
torch.cuda.empty_cache()

print("Loading GPT-2 medium...")
model_med = HookedTransformer.from_pretrained("gpt2-medium", device=device)
n_layers_med = model_med.cfg.n_layers
n_heads_med = model_med.cfg.n_heads
print(f"GPT-2 medium: {n_layers_med} layers, {n_heads_med} heads")

# Verify token IDs are the same (same tokenizer)
male_ids_med = male_ids
female_ids_med = female_ids

# Baseline metrics
print("\nComputing GPT-2-medium baselines...")
med_bias = eval_bias(model_med)
med_ppl = eval_ppl(model_med)
med_pronoun = eval_accuracy(model_med, EXPANDED_PRONOUN_TESTS)
med_agreement = eval_accuracy(model_med, EXPANDED_AGREEMENT_TESTS)
med_factual = eval_accuracy(model_med, FACTUAL_TESTS)

print(f"GPT-2-medium baseline: bias={med_bias:.4f}, ppl={med_ppl:.2f}, "
      f"pronoun={med_pronoun:.0%}, agreement={med_agreement:.0%}, factual={med_factual:.0%}")

# Quick layer-level ablation to see where bias lives in medium
print("\nLayer-level attention ablation (GPT-2-medium):")

def zero_attn(output, hook):
    output[:] = 0.0
    return output

med_layer_importance = []
for layer in tqdm(range(n_layers_med), desc="Layer ablation"):
    hooks = [(f"blocks.{layer}.hook_attn_out", zero_attn)]
    abl_bias = eval_bias(model_med, hooks)
    change = med_bias - abl_bias
    med_layer_importance.append(float(change))

print("\nLayer importance (attention) for gender bias in GPT-2-medium:")
for layer in range(n_layers_med):
    v = med_layer_importance[layer]
    bar = "█" * int(abs(v) * 200) if abs(v) > 0.001 else ""
    print(f"  Layer {layer:2d}: {v:+.6f} {bar}")

top_med_layers = np.argsort(np.abs(med_layer_importance))[::-1][:5]
print(f"\nTop 5 layers: {top_med_layers.tolist()}")

# ── Save results ──
all_results = {
    "gpt2_small_expanded": results,
    "gpt2_medium": {
        "baseline": {
            "bias": float(med_bias),
            "ppl": float(med_ppl),
            "pronoun": float(med_pronoun),
            "agreement": float(med_agreement),
            "factual": float(med_factual),
        },
        "layer_importance": med_layer_importance,
        "top_layers": top_med_layers.tolist(),
        "n_layers": n_layers_med,
        "n_heads": n_heads_med,
    },
}

with open(RESULTS_DIR / "expanded_eval_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Visualization: GPT-2-medium layer importance
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(n_layers_med), med_layer_importance,
       color=['red' if v > 0 else 'blue' for v in med_layer_importance])
ax.set_xlabel("Layer")
ax.set_ylabel("Bias Change (+ = ablation reduces bias)")
ax.set_title("GPT-2-Medium: Attention Layer Importance for Gender Bias")
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "gpt2_medium_layer_importance.png", dpi=150)
plt.close()

print(f"\nAll results saved to {RESULTS_DIR}")
print("\n✓ Experiment 08 complete.")
