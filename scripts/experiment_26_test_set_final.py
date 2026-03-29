"""
Experiment 26: Test-Set Final Evaluation

Held-out test set evaluation for the paper's main results table.
- All methods on test split (250 prompts, 25 never-seen occupations)
- Full capability suite with bootstrap CIs
- GPT-2 and (if available) Pythia-2.8B
"""

import torch
import numpy as np
import json
import sys
from pathlib import Path
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import (
    full_eval, print_results, results_to_json
)

RESULTS_DIR = Path("results/26_test_final")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


def scale_head(z, hook, head_idx, alpha):
    z[:, :, head_idx, :] *= alpha
    return z


# ═══════════════════════════════════════════════════════════════
# GPT-2
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("EXPERIMENT 26: TEST-SET FINAL EVALUATION — GPT-2")
print("=" * 70)

model = HookedTransformer.from_pretrained("gpt2", device=device)

# Define interventions
INTERVENTIONS = {
    "Baseline": None,
    "L10H9 ablation": [("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.0))],
    "L10H9 alpha=0.5": [("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.5))],
}

# Try SAE-based interventions
try:
    from sae_lens import SAE
    sae_l10 = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.10.hook_resid_pre",
        device=device,
    )

    def sae_gender_hook(resid, hook, sae=sae_l10):
        sae_acts = sae.encode(resid)
        f23440 = sae_acts[:, :, 23440:23441]
        f16291 = sae_acts[:, :, 16291:16292]
        adj = -0.5 * f23440 * sae.W_dec[23440] + -0.5 * f16291 * sae.W_dec[16291]
        return resid + adj

    INTERVENTIONS["Combined (L10H9 + gender features)"] = [
        ("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.0)),
        ("blocks.10.hook_resid_pre", sae_gender_hook),
    ]

    sae_l0 = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.0.hook_resid_pre",
        device=device,
    )

    def artifact_hook(resid, hook, sae=sae_l0):
        sae_acts = sae.encode(resid)
        f23406 = sae_acts[:, :, 23406:23407]
        adj = -f23406 * sae.W_dec[23406]
        return resid + adj

    INTERVENTIONS["F23406 artifact (control)"] = [("blocks.0.hook_resid_pre", artifact_hook)]
    print("SAE interventions loaded successfully")
except Exception as e:
    print("SAE loading failed (continuing without): %s" % e)

# Try INLP if results exist
try:
    inlp_results = json.load(open("results/24_strong_inlp/strong_inlp_results.json"))
    best = inlp_results["best_config"]
    if best:
        from sklearn.linear_model import LogisticRegression
        # Reconstruct INLP projection (need to retrain)
        print("INLP best config: layer=%d, iter=%d, C=%s" % (best["layer"], best["n_iter"], best["C"]))
        # We'd need the training data and retraining — mark as TODO
        print("  (INLP hook reconstruction deferred — use exp 24 results directly)")
except Exception:
    pass

# Run all GPT-2 evaluations
gpt2_results = {}
for name, hooks in INTERVENTIONS.items():
    print("\n" + "=" * 70)
    print("GPT-2 TEST SET: %s" % name)
    print("=" * 70)
    results = full_eval(model, hooks=hooks, split="test", capability="full",
                        n_boot=10000, verbose=True)
    print_results(results, label="GPT-2 — %s" % name)
    gpt2_results[name] = results_to_json(results)

# ═══════════════════════════════════════════════════════════════
# PYTHIA-2.8B (if enough VRAM)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LOADING PYTHIA-2.8B...")
print("=" * 70)

del model
torch.cuda.empty_cache()

pythia_results = {}
try:
    model_pythia = HookedTransformer.from_pretrained("pythia-2.8b", device=device)
    print("Pythia-2.8B loaded")

    PYTHIA_INTERVENTIONS = {
        "Baseline": None,
        "L22H30 ablation": [("blocks.22.attn.hook_z", partial(scale_head, head_idx=30, alpha=0.0))],
    }

    for name, hooks in PYTHIA_INTERVENTIONS.items():
        print("\n" + "=" * 70)
        print("PYTHIA TEST SET: %s" % name)
        print("=" * 70)
        results = full_eval(model_pythia, hooks=hooks, split="test", capability="full",
                            n_boot=10000, verbose=True)
        print_results(results, label="Pythia — %s" % name)
        pythia_results[name] = results_to_json(results)

    del model_pythia
    torch.cuda.empty_cache()

except Exception as e:
    print("Pythia-2.8B failed: %s" % e)
    print("Continuing with GPT-2 results only")

# ═══════════════════════════════════════════════════════════════
# MAIN RESULTS TABLE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PAPER MAIN RESULTS TABLE (TEST SET)")
print("=" * 70)

all_results = {}
for name, res in gpt2_results.items():
    all_results["GPT2_%s" % name] = res
for name, res in pythia_results.items():
    all_results["Pythia_%s" % name] = res

metrics = [
    ("signed_bias", "Signed bias"),
    ("abs_bias", "Absolute bias"),
    ("total_gender_mass", "Gender mass"),
    ("stereotype_preference", "Stereo pref"),
    ("wikitext_ppl", "WikiText PPL"),
    ("lambada_acc", "LAMBADA acc"),
    ("winogender_male_pref", "Winogender M%"),
    ("winobias_type1_gap", "WinoBias T1 gap"),
    ("winobias_type2_gap", "WinoBias T2 gap"),
    ("gap_overall", "GAP overall"),
    ("crows_pairs_score", "CrowS-Pairs"),
]

print("\n%-25s" % "Metric", end="")
for name in all_results:
    print(" %18s" % name[:18], end="")
print()
print("-" * (25 + 19 * len(all_results)))

for key, label in metrics:
    print("%-25s" % label, end="")
    for name, res in all_results.items():
        v = res.get(key, None)
        ci = res.get(key + "_ci", None)
        if v is None:
            print(" %18s" % "N/A", end="")
        elif ci:
            if isinstance(v, float) and abs(v) < 1:
                print(" %7.4f [%5.3f,%5.3f]" % (v, ci[0], ci[1]), end="")
            else:
                print(" %7.2f [%5.1f,%5.1f]" % (v, ci[0], ci[1]), end="")
        else:
            if isinstance(v, float) and abs(v) < 1:
                print(" %18.4f" % v, end="")
            else:
                print(" %18.2f" % v, end="")
    print()

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════
save_data = {
    "gpt2": gpt2_results,
    "pythia": pythia_results,
    "all_results": all_results,
}

with open(RESULTS_DIR / "test_final_results.json", "w") as f:
    json.dump(save_data, f, indent=2, default=str)

print("\nResults saved to %s" % RESULTS_DIR)
print("\n✓ Experiment 26 complete.")
