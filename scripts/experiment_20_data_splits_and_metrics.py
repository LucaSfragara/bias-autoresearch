"""
Experiment 20: Data Splits + Comprehensive Metrics Validation

Creates and validates:
- 3 non-overlapping data splits (discovery/dev/test)
- Reusable evaluator with 4 bias + capability + coreference metrics
- Baseline results on all splits

This is the foundation for all subsequent experiments (21-26).
"""

import torch
import json
import sys
from pathlib import Path
from functools import partial

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import (
    load_splits, get_prompts, full_eval, print_results, results_to_json
)

RESULTS_DIR = Path("results/20_data_splits")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# VALIDATE SPLITS
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("EXPERIMENT 20: DATA SPLITS + METRICS VALIDATION")
print("=" * 70)

splits = load_splits()
all_occs = set()
for name in ["discovery", "dev", "test"]:
    s = splits[name]
    occs = set(s["occupations"])
    tmpls = set(s["templates"])
    prompts = get_prompts(name)
    print("\n%s split:" % name)
    print("  Occupations: %d" % len(occs))
    print("  Templates:   %d" % len(tmpls))
    print("  Prompts:     %d" % len(prompts))
    print("  Sample:      %s" % prompts[0])

    # Check no overlap
    overlap = all_occs & occs
    if overlap:
        print("  WARNING: overlapping occupations with previous splits: %s" % overlap)
    else:
        print("  No occupation overlap with previous splits")
    all_occs |= occs

print("\nTotal unique occupations: %d" % len(all_occs))

# ═══════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════
print("\nLoading GPT-2...")
from transformer_lens import HookedTransformer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2", device=device)
print("Model loaded on %s" % device)

# ═══════════════════════════════════════════════════════════════
# BASELINE EVALUATION ON ALL SPLITS
# ═══════════════════════════════════════════════════════════════
all_results = {}

for split_name in ["discovery", "dev", "test"]:
    print("\n" + "=" * 70)
    print("BASELINE: %s split (full evaluation)" % split_name)
    print("=" * 70)

    results = full_eval(model, hooks=None, split=split_name, capability="full", verbose=True)
    print_results(results, label="Baseline — %s" % split_name)
    all_results["baseline_%s" % split_name] = results_to_json(results)

# ═══════════════════════════════════════════════════════════════
# QUICK INTERVENTION CHECK (L10H9 on dev)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("L10H9 ABLATION: dev split (full evaluation)")
print("=" * 70)


def scale_head(z, hook, head_idx, alpha):
    z[:, :, head_idx, :] *= alpha
    return z


hooks_l10h9 = [("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.0))]
results_l10h9 = full_eval(model, hooks=hooks_l10h9, split="dev", capability="full", verbose=True)
print_results(results_l10h9, label="L10H9 ablated — dev")
all_results["l10h9_dev"] = results_to_json(results_l10h9)

# ═══════════════════════════════════════════════════════════════
# COMPUTE DELTAS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DELTAS (L10H9 vs Baseline on dev)")
print("=" * 70)

baseline = all_results["baseline_dev"]
l10h9 = all_results["l10h9_dev"]

print("  Signed bias:  %+.4f → %+.4f (Δ = %+.4f)" % (
    baseline["signed_bias"], l10h9["signed_bias"],
    l10h9["signed_bias"] - baseline["signed_bias"]))
print("  Abs bias:      %.4f →  %.4f (Δ = %+.4f, %.1f%% reduction)" % (
    baseline["abs_bias"], l10h9["abs_bias"],
    l10h9["abs_bias"] - baseline["abs_bias"],
    (baseline["abs_bias"] - l10h9["abs_bias"]) / baseline["abs_bias"] * 100))
print("  Gender mass:   %.4f →  %.4f" % (baseline["total_gender_mass"], l10h9["total_gender_mass"]))

if "wikitext_ppl" in baseline and "wikitext_ppl" in l10h9:
    ppl_delta = (l10h9["wikitext_ppl"] - baseline["wikitext_ppl"]) / baseline["wikitext_ppl"] * 100
    print("  WikiText PPL:  %.2f →  %.2f (%+.1f%%)" % (
        baseline["wikitext_ppl"], l10h9["wikitext_ppl"], ppl_delta))

if "lambada_acc" in baseline and "lambada_acc" in l10h9:
    print("  LAMBADA acc:   %.1f%% →  %.1f%%" % (
        baseline["lambada_acc"] * 100, l10h9["lambada_acc"] * 100))

if "winobias_type1_gap" in baseline and "winobias_type1_gap" in l10h9:
    print("  WinoBias T1 gap: %.1f pp → %.1f pp" % (
        baseline["winobias_type1_gap"] * 100, l10h9["winobias_type1_gap"] * 100))

if "crows_pairs_score" in baseline and "crows_pairs_score" in l10h9:
    print("  CrowS-Pairs:   %.1f%% → %.1f%%" % (
        baseline["crows_pairs_score"] * 100, l10h9["crows_pairs_score"] * 100))

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════
with open(RESULTS_DIR / "exp20_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\nResults saved to %s" % RESULTS_DIR)
print("\n✓ Experiment 20 complete.")
