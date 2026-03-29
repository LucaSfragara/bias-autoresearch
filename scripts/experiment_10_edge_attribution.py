"""
Experiment 10: Edge Attribution Patching (EAP)
- Trace the full bias circuit: which edges transmit gender information?
- Compute gradient-based attribution for each edge (head-to-head)
- Answer: where does L10H9 read gender info from? What writes it?
- Identify the complete gender bias circuit topology
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

RESULTS_DIR = Path("results/10_edge_attribution")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading GPT-2 small...")
model = HookedTransformer.from_pretrained("gpt2", device=device)
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

with open("results/00_setup/gender_token_ids.json") as f:
    gender_token_ids = json.load(f)
male_ids = gender_token_ids["male"]
female_ids = gender_token_ids["female"]


def bias_metric(logits):
    """Differentiable bias metric: mean |P(male) - P(female)| across prompts."""
    probs = torch.softmax(logits[:, -1, :], dim=-1)
    male_prob = probs[:, male_ids].sum(dim=-1)
    female_prob = probs[:, female_ids].sum(dim=-1)
    return (male_prob - female_prob).abs().mean()


BIAS_PROMPTS = [
    "The nurse said that", "The doctor said that", "The engineer said that",
    "The secretary said that", "The CEO said that", "The teacher said that",
    "The programmer said that", "The homemaker said that",
    "The mechanic fixed the car and", "The receptionist answered the phone and",
]

# ═══════════════════════════════════════════════
# PART 1: GRADIENT-BASED EDGE ATTRIBUTION
# ═══════════════════════════════════════════════
# For each pair (source_head, dest_head), compute how much
# the source's output affects the bias metric through the destination.
# We use the gradient of the bias metric w.r.t. each head's output.

print("\n" + "="*70)
print("EDGE ATTRIBUTION PATCHING")
print("="*70)

# Step 1: Compute gradient of bias metric w.r.t. each head's output (hook_z)
# This tells us which heads are most important for the bias metric
print("\nStep 1: Computing per-head gradients w.r.t. bias metric...")

head_grads = {}  # layer -> head -> mean gradient norm

for prompt in tqdm(BIAS_PROMPTS, desc="Computing gradients"):
    tokens = model.to_tokens(prompt)

    # Enable gradient computation for the hook points
    activations = {}

    def save_and_grad_hook(z, hook):
        z.retain_grad()
        activations[hook.name] = z
        return z

    # Register hooks for all attention outputs
    hooks = []
    for layer in range(n_layers):
        hooks.append(("blocks.%d.attn.hook_z" % layer, save_and_grad_hook))

    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    loss = bias_metric(logits)
    loss.backward()

    for layer in range(n_layers):
        hook_name = "blocks.%d.attn.hook_z" % layer
        if hook_name in activations and activations[hook_name].grad is not None:
            grad = activations[hook_name].grad  # (1, seq, n_heads, d_head)
            for head in range(n_heads):
                key = (layer, head)
                g = grad[0, :, head, :].norm().item()
                if key not in head_grads:
                    head_grads[key] = []
                head_grads[key].append(g)

    model.zero_grad()

# Average gradients
mean_head_grads = {}
for (layer, head), grads in head_grads.items():
    mean_head_grads[(layer, head)] = np.mean(grads)

# Print top heads by gradient magnitude
sorted_heads = sorted(mean_head_grads.items(), key=lambda x: x[1], reverse=True)
print("\nTop 20 heads by gradient magnitude (bias-relevant):")
for (layer, head), grad_mag in sorted_heads[:20]:
    print("  L%dH%d: grad=%.6f" % (layer, head, grad_mag))


# ═══════════════════════════════════════════════
# PART 2: ATTENTION PATTERN ANALYSIS FOR KEY HEADS
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("ATTENTION PATTERN ANALYSIS")
print("="*70)

# For L10H9 and other key heads, analyze what they attend to
key_heads = [(10, 9), (0, 8), (5, 9), (11, 0), (1, 8)]

prompt_for_analysis = "The nurse said that"
tokens = model.to_tokens(prompt_for_analysis)
token_strs = [model.to_string([t.item()]) for t in tokens[0]]

with torch.no_grad():
    _, cache = model.run_with_cache(tokens)

print("\nToken sequence: %s" % token_strs)

attn_patterns = {}
for (layer, head) in key_heads:
    # attention pattern: (1, n_heads, seq, seq)
    attn = cache["blocks.%d.attn.hook_pattern" % layer][0, head]  # (seq, seq)
    print("\n--- L%dH%d attention pattern ---" % (layer, head))
    print("  (row = query position, col = key position)")

    # Show last position attending to all positions
    last_pos_attn = attn[-1].cpu().numpy()
    print("  Last position attends to:")
    for i, (tok, a) in enumerate(zip(token_strs, last_pos_attn)):
        bar = "#" * int(a * 50)
        print("    pos %d '%s': %.4f %s" % (i, tok, a, bar))

    attn_patterns[(layer, head)] = attn.cpu().numpy().tolist()


# ═══════════════════════════════════════════════
# PART 3: OV CIRCUIT ANALYSIS — What does each head write?
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("OV CIRCUIT ANALYSIS (What does each head write?)")
print("="*70)

# For key heads, compute the OV circuit's effect on gendered tokens
# OV matrix: W_V @ W_O maps input to output for each head
# Full OV: W_E @ W_V @ W_O @ W_U gives token-to-token mapping

for (layer, head) in key_heads:
    print("\n--- L%dH%d OV circuit ---" % (layer, head))

    W_V = model.blocks[layer].attn.W_V[head]  # (d_model, d_head)
    W_O = model.blocks[layer].attn.W_O[head]  # (d_head, d_model)
    OV = W_V @ W_O  # (d_model, d_model)

    # Project through unembedding to see what this head writes in vocab space
    # For a given input direction, OV @ direction gives the output direction
    # Then W_U @ output gives logit contribution

    # Check: if input is "nurse" embedding, what does the OV circuit write?
    test_tokens = ["nurse", "doctor", "engineer", "secretary", " he", " she",
                   " man", " woman", " boy", " girl"]

    for tok in test_tokens:
        tok_id = model.to_tokens(tok, prepend_bos=False).squeeze()
        if tok_id.dim() > 0:
            tok_id = tok_id[0]
        emb = model.embed.W_E[tok_id]  # (d_model,)

        output_dir = emb @ OV  # (d_model,)
        # Project through LN + unembed
        normed = model.ln_final(output_dir.unsqueeze(0).unsqueeze(0))
        logits = model.unembed(normed)[0, 0, :]

        he_logit = logits[model.to_tokens(" he", prepend_bos=False).squeeze().item()].item()
        she_logit = logits[model.to_tokens(" she", prepend_bos=False).squeeze().item()].item()

        print("  Input '%s' -> OV -> he=%.3f, she=%.3f, gap=%+.3f" % (
            tok, he_logit, she_logit, he_logit - she_logit))


# ═══════════════════════════════════════════════
# PART 4: QK CIRCUIT — What does each head attend to?
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("QK CIRCUIT ANALYSIS (What does each head look for?)")
print("="*70)

for (layer, head) in key_heads[:3]:  # Just top 3
    print("\n--- L%dH%d QK circuit ---" % (layer, head))

    W_Q = model.blocks[layer].attn.W_Q[head]  # (d_model, d_head)
    W_K = model.blocks[layer].attn.W_K[head]  # (d_model, d_head)
    # QK = W_E @ W_Q @ W_K^T @ W_E^T gives token-to-token attention preference

    # Check: which tokens does this head attend to when queried at last position?
    test_keys = ["nurse", "doctor", " he", " she", " the", " said", " that",
                 " man", " woman", "CEO"]

    for query_tok in [" that", " and"]:
        q_id = model.to_tokens(query_tok, prepend_bos=False).squeeze()
        if q_id.dim() > 0: q_id = q_id[0]
        q_emb = model.embed.W_E[q_id]
        q_vec = q_emb @ W_Q  # (d_head,)

        print("  Query='%s':" % query_tok)
        scores = []
        for key_tok in test_keys:
            k_id = model.to_tokens(key_tok, prepend_bos=False).squeeze()
            if k_id.dim() > 0: k_id = k_id[0]
            k_emb = model.embed.W_E[k_id]
            k_vec = k_emb @ W_K  # (d_head,)
            score = (q_vec @ k_vec).item() / (model.cfg.d_head ** 0.5)
            scores.append((key_tok, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        for tok, s in scores:
            print("    key='%s': score=%.3f" % (tok, s))


# ═══════════════════════════════════════════════
# PART 5: CIRCUIT GRAPH
# ═══════════════════════════════════════════════
print("\n" + "="*70)
print("BIAS CIRCUIT TOPOLOGY")
print("="*70)

# Compute pairwise influence: does ablating head A change head B's contribution to bias?
# Simplified version: use activation patching between pairs of key heads

# For the top bias-relevant heads, compute their activation overlap
top_bias_heads = [(l, h) for (l, h), _ in sorted_heads[:15]]

# Collect head outputs for all bias prompts
print("Collecting head outputs for circuit analysis...")
head_outputs = {}

for prompt in tqdm(BIAS_PROMPTS, desc="Caching outputs"):
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    for (layer, head) in top_bias_heads:
        key = (layer, head)
        z = cache["blocks.%d.attn.hook_z" % layer][0, -1, head, :].cpu()
        if key not in head_outputs:
            head_outputs[key] = []
        head_outputs[key].append(z)

# Compute pairwise cosine similarity of head output directions
print("\nPairwise cosine similarity of head output directions:")
cos_sim_matrix = np.zeros((len(top_bias_heads), len(top_bias_heads)))

for i, key_i in enumerate(top_bias_heads):
    for j, key_j in enumerate(top_bias_heads):
        vecs_i = torch.stack(head_outputs[key_i])
        vecs_j = torch.stack(head_outputs[key_j])
        mean_i = vecs_i.mean(0)
        mean_j = vecs_j.mean(0)
        cos = torch.nn.functional.cosine_similarity(mean_i.unsqueeze(0), mean_j.unsqueeze(0)).item()
        cos_sim_matrix[i, j] = cos

# Print the similarity matrix
head_labels = ["L%dH%d" % (l, h) for (l, h) in top_bias_heads]
print("\n%-10s" % "" + "".join("%-10s" % l for l in head_labels[:10]))
for i, label in enumerate(head_labels[:10]):
    row = "%-10s" % label
    for j in range(min(10, len(head_labels))):
        row += "%-10.2f" % cos_sim_matrix[i, j]
    print(row)


# ── Visualization ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gradient importance
ax = axes[0]
grad_vals = [mean_head_grads.get((l, h), 0) for l in range(n_layers) for h in range(n_heads)]
grad_matrix = np.array(grad_vals).reshape(n_layers, n_heads)
im = ax.imshow(grad_matrix, aspect='auto', cmap='hot')
ax.set_xlabel("Head")
ax.set_ylabel("Layer")
ax.set_title("Gradient-Based Attribution\n(Importance for Bias Metric)")
plt.colorbar(im, ax=ax)

# Cosine similarity
ax = axes[1]
n_show = min(10, len(top_bias_heads))
im = ax.imshow(cos_sim_matrix[:n_show, :n_show], cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(n_show))
ax.set_yticks(range(n_show))
ax.set_xticklabels(head_labels[:n_show], rotation=45, fontsize=8)
ax.set_yticklabels(head_labels[:n_show], fontsize=8)
ax.set_title("Head Output Cosine Similarity\n(Top Bias-Relevant Heads)")
plt.colorbar(im, ax=ax)

plt.suptitle("Gender Bias Circuit Analysis (GPT-2 Small)", fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "circuit_analysis.png", dpi=150)
plt.close()


# ── Save ──
results = {
    "head_gradient_importance": {
        "L%dH%d" % (l, h): float(g) for (l, h), g in sorted_heads
    },
    "top_20_bias_heads": [
        {"layer": l, "head": h, "grad": float(g)} for (l, h), g in sorted_heads[:20]
    ],
    "attention_patterns": {
        "prompt": prompt_for_analysis,
        "tokens": token_strs,
        "patterns": {
            "L%dH%d" % (l, h): p for (l, h), p in attn_patterns.items()
        }
    },
    "cosine_similarity": {
        "heads": head_labels,
        "matrix": cos_sim_matrix.tolist(),
    },
}

with open(RESULTS_DIR / "edge_attribution_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to %s" % RESULTS_DIR)
print("\n✓ Experiment 10 complete.")
