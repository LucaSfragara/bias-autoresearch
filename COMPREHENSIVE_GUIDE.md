# Comprehensive Guide: Mechanistic Interpretability for Bias Localization in LLMs

**Authors**: Luca Sfragara, Ayela Chughtai, Pantelis Emmanouil, Shriya Karam

This document explains every experiment, every result, why each decision was made, and how the whole project fits together. Written so that any team member can fully understand the work without reading a single line of code.

---

## Table of Contents

1. [What This Project Is About](#1-what-this-project-is-about)
2. [The Core Question](#2-the-core-question)
3. [Tools and Infrastructure](#3-tools-and-infrastructure)
4. [Phase 1: Discovery (Experiments 00-09)](#4-phase-1-discovery-experiments-00-09)
5. [Phase 2: Deep Analysis (Experiments 10-19)](#5-phase-2-deep-analysis-experiments-10-19)
6. [The Reviewer Critique](#6-the-reviewer-critique)
7. [Phase 3: Methodological Fixes (Experiments 20-27)](#7-phase-3-methodological-fixes-experiments-20-27)
8. [The Final Results](#8-the-final-results)
9. [What the Results Actually Mean](#9-what-the-results-actually-mean)
10. [Key Concepts Glossary](#10-key-concepts-glossary)

---

## 1. What This Project Is About

When you give GPT-2 the prompt "The nurse said that", it's more likely to continue with "she" than "he". When you give it "The engineer said that", the opposite happens. This is **occupation-conditioned gender bias** -- the model has learned statistical associations between occupations and gender from its training data.

Our project asks: **can we find exactly where in the model this bias lives, and can we surgically remove it without breaking the model's ability to do everything else?**

We use a technique called **mechanistic interpretability (mech interp)** -- opening up the transformer's internals and studying individual components (attention heads, MLP layers, individual neurons/features) to understand what they do.

### What we study
- **Model**: GPT-2-small (117M parameters, 12 layers, 12 attention heads per layer = 144 total heads)
- **Also**: Pythia-2.8B (32 layers, 32 heads per layer = 1024 total heads) for scale replication
- **Bias type**: Gender bias conditioned on occupation tokens
- **Method**: Ablation (zeroing out components), activation patching, SAE feature analysis, linear probing

### What we found (one-sentence summary)
Some bias-carrying heads can be cleanly removed ("separable"), others are so entangled with general language modeling that removing them destroys fluency -- this forms a **separability spectrum** that gradient importance cannot predict.

---

## 2. The Core Question

Imagine the model's computation as a pipeline: tokens go in, pass through 12 layers of processing, and probabilities come out. At each layer, 12 attention heads read the input and write information into the "residual stream" (a running vector that accumulates information). The final residual stream gets converted to token probabilities.

**Our question**: Which of these 144 attention heads are responsible for making the model prefer "he" after "engineer" and "she" after "nurse"?

But there's a catch: a head might carry bias information AND be critical for grammar, fluency, or word prediction. Removing it would fix bias but break the model. This is the **separability problem**.

---

## 3. Tools and Infrastructure

### Libraries
- **TransformerLens** (Neel Nanda): Lets us hook into any internal activation in GPT-2/Pythia, modify it during a forward pass, and see how the output changes. This is the core tool for all ablation and patching experiments.
- **SAE-Lens**: Loads Sparse Autoencoders (SAEs) trained on GPT-2's internal representations. SAEs decompose the model's internal vectors into interpretable "features" -- individual directions that correspond to concepts.
- **HuggingFace Datasets**: Source for WikiText-103, LAMBADA, BLiMP, WinoBias, and other benchmarks.

### Hardware
- **Remote**: SSH to an A40 GPU (48GB VRAM) at `root@69.30.85.65 -p 22056`
- GPT-2-small fits in ~500MB VRAM, Pythia-2.8B needs ~11GB

### Key technique: Ablation
"Ablation" means zeroing out a component's output. If we zero out head L10H9 (layer 10, head 9) and bias drops, that head was contributing to bias. If perplexity (PPL) barely changes, the head wasn't critical for general language modeling. This is the basis of the separability spectrum.

Concretely, we hook into `blocks.10.attn.hook_z` (the output of L10's attention heads) and multiply head 9's output by 0 (or some alpha between 0 and 1 for partial ablation).

---

## 4. Phase 1: Discovery (Experiments 00-09)

### Experiment 00: Setup
**What**: Load GPT-2-small via TransformerLens, verify everything works.
**Why**: Baseline infrastructure check.

### Experiment 01: Activation Patching (Head Scan)
**What**: For each of the 144 attention heads in GPT-2, zero it out and measure how much the gender bias changes across ~20 occupation prompts (e.g., "The nurse said that", "The doctor said that").

**How bias is measured**: For each prompt, look at the probability the model assigns to the next token being a male pronoun (he, him, his, himself) vs female (she, her, hers, herself). The "absolute bias" is the average |P(male) - P(female)| across all prompts.

**Result**: L10H9 stands out. Ablating it reduces absolute bias by ~25% while barely touching perplexity. Most other high-bias heads cause massive PPL increases when ablated.

**Why this matters**: This is the first evidence of the separability spectrum. Not all bias heads are equal.

### Experiment 02: Logit Lens
**What**: The "logit lens" technique reads the residual stream at each layer and projects it to token probabilities (as if that layer were the last). This shows how the model's prediction evolves layer by layer.

**Result**: Gender bias emerges primarily in layers 9-11. Earlier layers have roughly equal male/female probabilities; the gap opens up in the late layers.

**Why this matters**: Confirms that the bias-relevant computation happens late in the network, consistent with L10H9 being in layer 10.

### Experiment 03: Entanglement Analysis
**What**: Systematically measure the trade-off between bias reduction and capability damage for each head. Plot "bias reduction %" vs "PPL increase %" for all 144 heads.

**Key result**:
- **L0H8**: Highest bias reduction (35.8%) but catastrophic PPL increase (+103%). Deeply entangled.
- **L10H9**: Strong bias reduction (25%) with minimal PPL increase (+1.9%). Cleanly separable.
- **L0H9**: 29% bias reduction but +22.8% PPL. Not separable.

**Why this matters**: This defines the **separability spectrum** -- the central finding of the paper. Gradient importance (how much a head's output correlates with bias) does NOT predict separability. L0H8 has the highest gradient importance but is the most entangled.

### Experiment 04: Cross-Bias Generalization
**What**: Test whether the bias pattern holds across different sentence templates, not just "The [occupation] said that".

**Result**: The bias generalizes across templates. "The nurse walked to the" shows similar gender skew to "The nurse said that".

**Why this matters**: Our findings aren't an artifact of a specific template.

### Experiment 05: SAE Feature Analysis
**What**: Use Sparse Autoencoders to decompose the model's internal representations into interpretable features. Look for features that correlate with gender bias.

**How SAEs work**: An SAE takes a model's internal activation vector (e.g., the residual stream at layer 10) and decomposes it into a sparse combination of ~25,000 "features". Each feature is a direction in activation space that tends to fire for specific concepts.

**Result**: Found L0_F23406 -- a feature at layer 0 that, when suppressed, appears to reduce bias by 80%. This looked like a huge win.

**Why this matters (and the trap)**: This result is misleading. See experiment 07.

### Experiment 06: Combined Interventions
**What**: Test combining L10H9 ablation with other interventions (other heads, SAE features).

**Result**: Combinations give incrementally better bias reduction but with increasing capability cost.

### Experiment 07: SAE Feature Characterization (The Artifact Discovery)
**What**: Investigate what L0_F23406 actually does. Look at what tokens activate it most. Look at what happens to overall gender probability mass (P(male) + P(female)) when it's suppressed.

**Result**: L0_F23406 is NOT a gender feature. Its top-activating tokens are "that", "which", "who" -- it's a **complementizer/relative pronoun feature**. When you suppress it, you suppress all pronoun probabilities (both male and female). The "bias reduction" was fake -- it was just suppressing all gender tokens equally.

**Key metric**: Gender probability mass dropped from 0.189 to 0.112 (a 40.7% drop). The model wasn't less biased; it was just less likely to produce ANY gendered pronoun.

**Why this matters**: This is a **major cautionary tale** for SAE-based debiasing. A feature can look like it "removes bias" while actually just suppressing the relevant output space. You MUST check gender mass, not just bias delta.

### Experiment 08: Expanded Evaluation (GPT-2-medium)
**What**: Run the same analysis on GPT-2-medium (345M params, 24 layers) to see if findings generalize to a slightly larger model.

**Result**: Late-layer localization confirmed -- bias-carrying heads are again in the final third of the network.

### Experiment 09: Steering Vectors
**What**: Test an alternative intervention: instead of ablating heads, create a "steering vector" (the difference between male-prompt and female-prompt activations) and subtract it from the residual stream.

**Result**: Head scaling (ablation with alpha < 1) works better than SAE feature clamping, which works better than steering vectors. Steering vectors are too blunt -- they affect too many aspects of the model's behavior.

---

## 5. Phase 2: Deep Analysis (Experiments 10-19)

### Experiment 10: Edge Attribution Patching (EAP)
**What**: Use gradient-based attribution to estimate which edges in the computational graph (head-to-head connections) contribute most to bias.

**How EAP works**: Instead of actually patching each connection (computationally expensive), use gradients as a first-order approximation. For each edge, compute: (activation difference) * (gradient of bias metric w.r.t. that edge).

**Result**: L10H9's OV circuit (the "what to write" part of attention) has strong gender-copying behavior. It reads occupation information and writes gender-associated information.

**Limitation identified later**: EAP is gradient-based, not causal. It can miss nonlinear interactions and give false positives. This is why experiment 23 was created.

### Experiment 11: Asymmetric Gender Analysis
**What**: Check whether the model has a "male default" -- i.e., does it tend to produce male pronouns more than female ones across most occupations?

**Result**: Yes. Across most occupations, P(he) > P(she). The model has an overall male skew, and occupation-specific effects modulate this baseline. For stereotypically female occupations (nurse, secretary), female pronouns are boosted, but the baseline is still male-leaning.

**Why this matters**: This means "reducing absolute bias" might not mean what you think. If you reduce |P(male) - P(female)| but both genders drop equally, the model might still be male-default. You need to check **signed bias** (P(male) - P(female), keeping the sign).

### Experiment 12: True Gender Features (SAE)
**What**: After discovering F23406 is an artifact, find the REAL gender features in the SAE at layer 10.

**Result**:
- **L10_F23440**: Female feature. Top activations on "she", "her", "herself". Suppressing it reduces female pronoun probabilities.
- **L10_F16291**: Male feature. Top activations on "he", "his", "himself".

**How "combined intervention" works**: Ablate L10H9 (zero its output) AND scale down these two gender features by 50%. This gives ~28.5% bias reduction (vs 25% for L10H9 alone).

### Experiment 13: Robust Evaluation
**What**: Add bootstrap confidence intervals (resample the data 10,000 times, compute the metric each time, report the 2.5th-97.5th percentile range). Add specificity scores. Compute z-scores for statistical significance.

**Result**: L10H9's bias reduction is highly significant (z=19.09). Bootstrap 95% CI for absolute bias reduction excludes zero.

**Limitation**: Only used ~30 WikiText sentences for PPL and ~40 hand-coded pronoun tests for coreference. These were upgraded in experiments 20-26.

### Experiment 14: Pythia-2.8B Scale Replication
**What**: Run the head scan on Pythia-2.8B (a completely different model architecture, 6x larger) to test if the separability spectrum is a general phenomenon.

**How**: Same approach -- zero each of the 1024 heads, measure bias change and PPL change.

**Result**: **L22H30** achieves 40.6% absolute bias reduction with only +0.2% PPL increase on the held-out test set. The separability spectrum exists in Pythia too.

**Why this matters**: The finding isn't GPT-2-specific. Different architecture, different training data, different scale -- same phenomenon.

### Experiment 15: Baselines and INLP
**What**: Compare our mechanistic approach against INLP (Iterative Null-space Projection), a standard debiasing baseline.

**How INLP works**:
1. Collect activations at a specific layer for male-associated and female-associated prompts
2. Train a linear probe (logistic regression) to distinguish male from female activations
3. Project the activations onto the null space of the probe's weight vector (remove the direction the probe uses to classify gender)
4. Repeat: train a new probe on the projected activations, project again
5. After N iterations, the model's activations at that layer have had N gender-discriminating directions removed

**Result (original)**: INLP with ~30 training prompts gives moderate bias reduction. But the comparison was not fair -- too few training prompts, no hyperparameter tuning.

### Experiment 16: BOS Path Analysis
**What**: Investigate the BOS (Beginning of Sequence) token's role. L10H9 attends heavily to the BOS token position. Is BOS carrying gender information?

**Result**: BOS attention is context-independent. The model always puts high attention on BOS regardless of prompt content. When we zero out BOS attention, bias actually INCREASES by 30.6%, because attention redistributes to occupation tokens (which carry occupation-gender associations).

**Original interpretation (wrong)**: "BOS is a causal pathway for bias"
**Revised interpretation (exp 27)**: BOS is an attention sink -- it absorbs excess attention as a normalization mechanism.

### Experiment 17: Occupation Patching
**What**: Test whether the occupation token is the source of the bias signal. Replace the activation at the occupation position in one run with the activation from a different run.

**Result**: Occupation identity causally determines gender bias (correlation r=0.69 between occupation activation and output bias). Patching is bidirectional -- swapping "nurse" activations into a "doctor" run makes the model behave as if the occupation were "nurse".

### Experiment 18: CrowS-Pairs
**What**: Evaluate on CrowS-Pairs, a bias benchmark with paired sentences where one is more stereotypical than the other (e.g., "Women are not good at math" vs "Men are not good at math").

**How scoring works**: For each pair, compute the pseudo-log-likelihood (sum of log P(token_i | prefix) for each token). The model "prefers" whichever sentence has higher PLL. A perfectly unbiased model would prefer the stereotypical sentence 50% of the time.

**Result (original)**: 50 hand-coded gender pairs. Baseline: ~60% stereotype preference. L10H9 ablation: ~59%.

**Limitation**: Only 50 pairs, hand-coded. Upgraded to 262 real pairs from the published dataset in experiments 22/26.

### Experiment 19: Cross-Bias Specificity
**What**: Is L10H9 specific to gender bias, or does it also carry race, age, religion bias?

**Result**: L10H9 is rank #2 for gender bias reduction but rank #129-132 for race/age/religion. It appears gender-specific.

**Caveat**: Non-gender bias baselines are tiny (~0.00005 for race vs ~0.117 for gender), making these rankings potentially unstable. Moved to appendix.

---

## 6. The Reviewer Critique

A trusted reviewer identified 16 serious issues. The most important ones:

### Issue 1: Signed bias not reported
We only reported **absolute bias** (|P(male) - P(female)|). This hides the fact that our interventions barely change the **direction** of bias. L10H9 ablation: signed bias goes from +0.074 to +0.074 (literally unchanged). The model is still male-default; we just reduced the variance between occupations.

### Issue 2: Too few prompts, no data splits
Original experiments used ~20 prompts, and the same prompts were used for discovery AND evaluation. This is overfitting -- the head that looks best on your discovery prompts might not generalize to new prompts.

### Issue 3: Weak INLP baseline
We compared against INLP with ~30 training prompts and minimal tuning. A properly tuned INLP would likely beat our approach on bias metrics.

### Issue 4: Overstated claims
"Complete circuit" (it's partial), "debiasing" (it barely changes the male default), "capability-preserving" (we only checked PPL on 30 sentences).

### Issue 5: Gradient-based EAP instead of causal patching
Experiment 10 used gradient approximations instead of actual activation patching. Gradients can be misleading.

### Issue 6: CrowS-Pairs too small
Only 50 hand-coded pairs instead of the full ~262-pair published dataset.

### Issue 7: No coreference benchmarks
We never tested whether the intervention affects pronoun coreference resolution -- exactly the task that gender bias most directly affects.

---

## 7. Phase 3: Methodological Fixes (Experiments 20-27)

### Experiment 20: Data Splits + Reusable Evaluator

**What we built**: Two critical pieces of infrastructure.

#### Data Splits (`data/splits.json`)

Three non-overlapping sets of occupations and templates:

| Split | Purpose | Occupations | Templates | Total Prompts |
|-------|---------|-------------|-----------|---------------|
| **Discovery** | Find which heads carry bias | 20 (nurse, doctor, engineer, ...) | 10 ("said that", "walked to the", ...) | 200 |
| **Dev** | Tune intervention parameters | 20 (hairdresser, plumber, dancer, ...) | 10 ("mentioned that", "reported that", ...) | 200 |
| **Test** | Final held-out evaluation | 25 (pharmacist, veterinarian, geologist, ...) | 10 ("believed that", "whispered that", ...) | 250 |

Zero overlap in occupations or templates between any two splits. This means test-set results are genuinely on data the model's bias was never evaluated on during development.

Each prompt has the form: `"The [occupation] [template]"` e.g., `"The nurse said that"`, `"The pharmacist believed that"`.

#### Reusable Evaluator (`scripts/eval_utils.py`)

A single function `full_eval(model, hooks, split, ...)` that runs ALL benchmarks in one call:

**Bias metrics (4)**:
- `signed_bias`: Mean of (P(male) - P(female)) across all prompts. Positive = male-skewed. This is the metric the reviewer demanded.
- `abs_bias`: Mean of |P(male) - P(female)|. The original metric. Measures magnitude of bias regardless of direction.
- `total_gender_mass`: Mean of (P(male) + P(female)). Catches the F23406-style artifact where suppressing all gender tokens looks like debiasing.
- `stereotype_preference`: Fraction of prompts where P(male) > P(female). 1.0 = always male, 0.5 = balanced, 0.0 = always female.

**General capability (3)**:
- `wikitext_ppl`: Perplexity on 1000 sentences from WikiText-103 validation set. Measures general language modeling quality. Lower = better. Computed as exp(mean cross-entropy loss per sentence).
- `lambada_acc`: Accuracy on 500 LAMBADA examples. Each example is a passage where the last word requires long-range context to predict. Tests contextual understanding.
- `blimp_acc`: Accuracy on BLiMP minimal pairs (anaphor_gender_agreement + subject-verb agreement). Tests grammatical knowledge. For each pair of sentences (one grammatical, one not), check if the model assigns higher likelihood to the grammatical one.

**Coreference benchmarks (3)**:
- `winogender`: 240 occupation-pronoun sentence pairs from Winogender. For each pair (male version / female version of same sentence), find the pronoun position and check which gender the model prefers. Reports overall male preference rate + breakdown by pronoun form (he/she, him/her, his/her).
- `winobias`: WinoBias coreference resolution. Sentences with occupation-pronoun pairs where the correct referent is either stereotypical ("pro") or counter-stereotypical ("anti"). Type 1 = syntactically ambiguous, Type 2 = syntactically unambiguous. Reports accuracy on each condition + the gap (pro - anti). A large gap means the model relies on stereotypes rather than syntax.
- `gap`: GAP dataset -- 2000 real Wikipedia sentences with ambiguous pronoun reference. Replace the pronoun with each candidate name, check which replacement the model prefers via pseudo-log-likelihood. Reports masculine/feminine accuracy splits.

**Bias benchmark (1)**:
- `crows_pairs`: 262 real gender sentence pairs from the published CrowS-Pairs dataset. For each pair, compute PLL of stereotypical vs anti-stereotypical sentence. Report fraction preferring stereotype. 50% = no bias.

**All metrics get bootstrap 95% CIs** (resample 10,000 times).

**Why `full_eval` matters**: Every experiment from 21-26 calls this same function. There's no ad-hoc metric computation anywhere. If a metric is wrong, it's wrong everywhere and easy to fix in one place.

---

### Experiment 21: Head Re-Discovery on Discovery Set

**What**: Repeat the 144-head scan from experiment 01, but now using 200 discovery prompts (10x more than original) and the standardized evaluator.

**Why**: The original scan used ~20 prompts. With so few, random variation could make a mediocre head look important. We need to confirm L10H9 still stands out with a proper sample size.

**Method**: For each of 144 heads:
1. Zero it out (alpha=0)
2. Run all 200 discovery prompts through the model
3. Measure the 4 bias metrics + quick PPL check (50 WikiText sentences for speed)
4. Compare to baseline

**Results (top 5 by absolute bias reduction)**:

| Rank | Head | Abs Bias Reduction | PPL Change | Separable? |
|------|------|-------------------|------------|------------|
| 1 | L0H8 | 35.8% | +103% | NO |
| **2** | **L10H9** | **32.9%** | **+1.9%** | **YES** |
| 3 | L0H9 | 29.0% | +22.8% | NO |
| 4 | L11H0 | 28.2% | +7.7% | Borderline |
| 5 | L5H9 | 24.0% | Unknown | TBD |

**L10H9 is confirmed as the top separable head.** It's rank #2 overall but rank #1 among heads that don't destroy the model.

**Bootstrap CIs on top 10**: All top-10 heads had confidence intervals excluding zero, confirming the bias reduction is statistically robust.

**Known issue**: The quick PPL comparison in this scan has a bug -- baseline PPL was computed with 200 WikiText sentences but per-head scans used 50 sentences. The first 50 sentences have different mean PPL than all 200, causing every head to spuriously show ~-35% PPL "improvement". **This does not affect bias rankings** (which don't depend on PPL). The full_eval on top 3 heads uses 1000 sentences and is correct.

---

### Experiment 22: Expanded Capability Validation

**What**: Run `full_eval` on baseline + all interventions (L10H9 ablation, combined L10H9+SAE features, F23406 artifact control) on the dev set. Also do a per-occupation breakdown.

**Why**: Validate that the evaluator works end-to-end and produce detailed benchmark breakdowns.

**Results (dev set, selected metrics)**:

| Metric | Baseline | L10H9 Ablation | Combined | F23406 (artifact) |
|--------|----------|----------------|----------|-------------------|
| Signed bias | +0.074 | +0.074 | +0.073 | +0.050 |
| Abs bias | 0.123 | 0.090 | 0.084 | 0.082 |
| Gender mass | 0.196 | 0.165 | 0.160 | 0.137 |
| WikiText PPL | 123.6 | 126.0 | 126.0 | 124.5 |
| LAMBADA acc | 24.8% | 25.0% | 25.0% | 23.8% |
| Winogender M% | 78.8% | 86.7% | 90.8% | 77.5% |
| CrowS-Pairs | 60.7% | 59.2% | 57.6% | 61.1% |

**Critical observation -- signed bias**: L10H9 ablation doesn't change signed bias at all (+0.074 -> +0.074). The intervention reduces the variance between occupations but doesn't change the overall male-default direction.

**Critical observation -- Winogender**: L10H9 ablation INCREASES male preference from 78.8% to 86.7%. This means L10H9 was actually partially counteracting the male default -- it was boosting female pronouns for some occupations. Removing it makes the model more uniformly male-preferring.

**Per-occupation breakdown**: Showed that L10H9 ablation reduces bias for most occupations but can increase it for a few. The effect is not uniform.

**StereoSet**: Attempted but failed due to label format mismatch in the dataset. Not critical since CrowS-Pairs and WinoBias provide bias benchmarking.

---

### Experiment 23: Proper Path Patching (Causal Circuit Analysis)

**What**: Replace the gradient-based EAP from experiment 10 with actual activation patching to get causal (not correlational) evidence for the bias circuit.

**Why**: The reviewer correctly noted that gradient-based attribution can be misleading. We need to actually intervene on activations and measure downstream effects.

**Method**:

Two distributions of prompts:
- **Clean**: "The nurse said that", "The doctor said that", etc. (prompts where occupation identity matters)
- **Corrupted**: "The person said that" (neutral occupation, no gender signal)

Only occupations that tokenize to a single token were used (14 valid: nurse, doctor, engineer, secretary, CEO, teacher, programmer, mechanic, pilot, surgeon, dancer, firefighter, chef, dentist).

**Four tests**:

#### Test 1: Sufficiency (can L10H9 alone create the bias?)
- Run the corrupted prompt ("The person...")
- But patch in L10H9's output from the clean run
- If L10H9 is sufficient, this should restore the occupation-specific bias

**Result: 19.8% recovery.** L10H9 alone can only recover about 1/5 of the bias signal. It's NOT sufficient on its own.

#### Test 2: Necessity (does the bias need L10H9?)
- Run the clean prompt
- But corrupt L10H9's output (replace with corrupted-run values)
- If L10H9 is necessary, this should remove the occupation-specific bias

**Result: 31.8% removal.** Corrupting L10H9 removes about 1/3 of the bias. It's partially necessary but not fully.

**What sufficiency + necessity together mean**: L10H9 is one important node in a distributed circuit. It contributes meaningfully but is not the whole story. The original "complete circuit" claim was overstated.

#### Test 3: Upstream sources (who feeds information to L10H9?)
- For each head in layers 0-9, patch its clean output into the corrupted run
- Measure how much bias recovers at the final output

**Top upstream heads**:
| Head | Recovery % | Interpretation |
|------|-----------|----------------|
| L9H7 | +37.3% | Strongest upstream source |
| L8H5 | -27.8% | Counter-signal (anti-bias) |
| L9H2 | +25.1% | Second upstream source |
| L4H3 | +19.0% | Early-layer contributor |
| L4H7 | +17.5% | Early-layer contributor |

#### Test 4: Mediation (does upstream information route through L10H9?)
- For each upstream head: patch it clean while simultaneously corrupting L10H9
- The "direct" effect (bypassing L10H9) tells us how much of that head's contribution goes around L10H9

**Key mediation results**:
| Head | Total Effect | Direct (bypass L10H9) | % Mediated Through L10H9 |
|------|-------------|----------------------|--------------------------|
| L4H3 | +19.0% | +3.3% | **83%** |
| L4H11 | -16.2% | +0.5% | **103%** |
| L4H7 | +17.5% | +5.0% | **71%** |
| L3H6 | +12.8% | +3.3% | **74%** |
| L8H5 | -27.8% | -9.2% | **67%** |
| L9H7 | +37.3% | +21.4% | **43%** |

**Interpretation**: Layer 3-4 heads funnel their bias signal heavily through L10H9 (70-100% mediated). Later heads (L8, L9) have more independent paths to the output. This makes sense: early heads compute occupation-gender associations and write them to the residual stream; L10H9 reads this information and relays it toward the output; late heads have shorter paths to the output and can bypass L10H9.

**The partial circuit**:
```
L3H6 ──┐
L4H3 ──┤ (70-100% mediated)
L4H7 ──┼──> L10H9 ──> Output (19.8% sufficiency)
L4H11 ─┘
L8H5 ──────────────> Output (independent path, counter-signal)
L9H7 ──────────────> Output (43% mediated, 57% direct)
```

---

### Experiment 24: Stronger INLP Baseline

**What**: Properly tune INLP as a fair comparison baseline.

**Why**: The original INLP (exp 15) used only ~30 training prompts with minimal tuning. The reviewer correctly noted this was an unfair comparison.

**Method**:
1. Created 95 male-associated + 95 female-associated training prompts with diverse surface forms (not from dev/test occupations)
2. Swept: 12 layers x 6 iteration counts (1,2,3,5,7,10) x 4 regularization values (C=0.01,0.1,1,10) = 288 configurations
3. Selected best configuration on dev set by abs_bias

**Best configuration**: Layer 10, 10 iterations, C=1.0

**Results**:
- **Abs bias**: 0.012 (91% reduction from baseline 0.123). Crushes our 27% from L10H9.
- **Signed bias**: +0.001 (essentially zero -- INLP actually fixes the male default!)
- **PPL**: 123.6 (unchanged from baseline)
- **Gender mass**: 0.093 (halved from 0.196 baseline)
- **LAMBADA accuracy**: 19.4% (dropped from 24.8%)
- **Winogender male pref**: 69.2% (moved toward 50%, the only method that does this)

**The INLP tradeoff**: INLP dominates on pure bias metrics. But it halves gender probability mass (the model becomes less likely to produce ANY gendered pronoun) and hurts LAMBADA (contextual word prediction). It also removes 10 directions from the residual stream at layer 10, which is a more invasive intervention than zeroing one head.

**Probe accuracy trajectory**: The probes start at 100% accuracy (gender is linearly separable in layer 10) and drop to 87% after 10 iterations of null-space projection. This means even after removing 10 gender-discriminating directions, there's still some residual gender information.

**Honest comparison**: INLP is better for bias reduction. Our approach is better for understanding the mechanism and preserving the model's ability to use gender information when appropriate. The right choice depends on the deployment scenario.

---

### Experiment 25: Matched Effect-Size Dose-Response

**What**: Instead of just ablation (on/off), sweep L10H9's scaling factor from 1.0 (no change) to 0.0 (full ablation) in steps of 0.1. At each setting, measure all metrics.

**Why**: We need to know (1) is the dose-response linear? (2) what's the cost per unit of bias reduction? (3) where's the sweet spot?

**L10H9 alpha sweep results**:

| Alpha | Abs Bias Red. | Signed Bias | PPL Change |
|-------|--------------|-------------|------------|
| 1.0 | 0% | +0.074 | 0% |
| 0.5 | 11.2% | +0.074 | minimal |
| 0.0 | 26.6% | +0.074 | +1.9% |

**Key finding 1 -- Linear dose-response**: Each 0.1 decrease in alpha gives ~2.7% bias reduction. The effect is remarkably linear. This means you can choose your desired bias-capability tradeoff point smoothly.

**Key finding 2 -- Signed bias is flat**: Signed bias literally does not change across the entire sweep (+0.074 at every alpha). L10H9 ablation reduces gender-variance-between-occupations without changing the male-default direction.

**Multi-head ablation**:
| Heads Ablated | Bias Reduction | PPL Change |
|---------------|---------------|------------|
| L10H9 only | 26.6% | +1.9% |
| L10H9 + L11H0 | 25.3% | +26.8% |
| L10H9 + L11H0 + L5H9 | 36.5% | +34.9% |
| L10H9 + L11H0 + L5H9 + L10H0 | 37.9% | +36.1% |

**Key finding 3 -- Diminishing returns**: Adding more heads doesn't proportionally increase bias reduction but dramatically increases PPL cost. L10H9 alone is the best trade-off. This reinforces the separability spectrum: L10H9 is the one cleanly removable head.

**Known issue**: PPL numbers in the sweep have the same n_sentences mismatch bug as exp 21 (baseline 200 sentences, sweep 100 sentences), making the absolute PPL values unreliable. The relative ordering and the full_eval results at the bottom (1000 sentences) are correct.

---

### Experiment 26: Test-Set Final Evaluation (The Paper's Main Table)

**What**: Run all methods on the held-out test set (250 prompts, 25 occupations NEVER seen during any development). Full capability evaluation with bootstrap CIs on every metric. Also evaluate Pythia-2.8B.

**Why**: This is the only evaluation that matters for the paper. Everything before this was discovery and tuning on discovery/dev sets.

**GPT-2 Test-Set Results**:

| Metric | Baseline | L10H9 Ablated | L10H9 alpha=0.5 | Combined | F23406 (ctrl) |
|--------|----------|---------------|-----------------|----------|---------------|
| **Signed bias** | +0.059 [0.049, 0.070] | +0.052 [0.045, 0.059] | -- | +0.052 [0.046, 0.059] | +0.027 [0.018, 0.035] |
| **Abs bias** | 0.079 [0.071, 0.087] | 0.059 [0.053, 0.065] | -- | 0.056 [0.051, 0.062] | 0.043 [0.036, 0.051] |
| **Gender mass** | 0.189 | 0.152 | -- | 0.149 | 0.112 |
| **WikiText PPL** | 123.6 [110.6, 138.4] | 126.0 [112.6, 141.1] | -- | 126.0 [112.5, 141.4] | 124.5 [111.7, 139.5] |
| **LAMBADA** | 24.8% [21.2%, 28.6%] | 25.0% | -- | 25.0% | 23.8% |
| **BLiMP mean** | 98.2% | 97.8% | -- | 97.8% | 97.9% |
| **Winogender M%** | 78.8% [73.3%, 83.8%] | 86.7% | -- | 90.8% | 77.5% |
| **WinoBias T1 gap** | 0.5 pp | -0.8 pp | -- | -1.5 pp | 0.5 pp |
| **WinoBias T2 gap** | 24.2 pp | 18.9 pp | -- | 18.4 pp | 24.2 pp |
| **GAP overall** | 70.6% [68.5%, 72.7%] | 71.0% | -- | 71.1% | 70.8% |
| **CrowS-Pairs** | 60.7% [54.9%, 66.4%] | 59.2% | -- | 57.6% | 61.1% |

(Numbers in brackets are bootstrap 95% CIs. `--` = not run for this variant.)

**Pythia-2.8B Test-Set Results**:

| Metric | Baseline | L22H30 Ablated |
|--------|----------|----------------|
| **Signed bias** | +0.032 [0.023, 0.041] | +0.021 [0.015, 0.026] |
| **Abs bias** | 0.056 [0.049, 0.063] | 0.033 [0.029, 0.038] |
| **WikiText PPL** | 59.6 | 59.7 (+0.2%) |
| **LAMBADA** | 52.8% | 52.8% |
| **Winogender M%** | 74.2% | 80.0% |
| **CrowS-Pairs** | 64.5% | 63.7% |

**Reading the results**:

1. **L10H9 ablation works**: 25.3% abs bias reduction, +1.9% PPL, LAMBADA/BLiMP/GAP all unchanged.

2. **The signed bias caveat is real**: Signed bias barely changes (0.059 -> 0.052). The model is still male-default after intervention.

3. **Winogender gets WORSE**: Male preference goes from 78.8% to 86.7%. L10H9 was partially helping produce female pronouns in stereotypically female contexts. Removing it makes the model more uniformly male.

4. **WinoBias T2 gap improves**: From 24.2pp to 18.9pp. The intervention reduces the model's reliance on stereotypes for syntactically unambiguous coreference.

5. **F23406 is confirmed as artifact**: It shows 45% abs bias reduction but gender mass drops from 0.189 to 0.112 and CrowS-Pairs is unchanged (61.1% vs 60.7%). It's suppressing all gender tokens, not fixing bias.

6. **Pythia replicates beautifully**: L22H30 gives 40.6% abs bias reduction with +0.2% PPL. Even better separability than GPT-2's L10H9.

---

### Experiment 27: BOS Reframing

**What**: Re-analyze the BOS (Beginning of Sequence) findings from experiments 16-17 in light of the "attention sink" literature (Xiao et al. 2023).

**Why**: Our original claim was that BOS serves as a "causal pathway" for bias. A reviewer would immediately question this. The attention sink literature provides a better explanation.

**Attention sinks explained**: In causal transformers, the BOS token is always in the attention window for every position. Many heads learn to dump excess attention on BOS as a normalization mechanism -- it's like a garbage collector for attention that doesn't need to go anywhere specific. This is NOT the same as BOS carrying meaningful information.

**Results**:
- **101 out of 144 heads** (70%) in GPT-2 have >50% BOS attention. This is the norm, not a special property of bias heads.
- **L10H9's BOS attention (47%) is actually LOWER than other L10 heads** (mean 85.3%, z-score = -3.96). L10H9 attends LESS to BOS than typical heads, meaning it attends MORE to content tokens (like occupation tokens).
- **Attention entropy**: L10H9 has the highest attention entropy (1.36) among L10 heads, meaning it distributes attention more broadly rather than dumping it on BOS.

**Revised interpretation**: L10H9 is unusual BECAUSE it DOESN'T use BOS as a sink. It actually reads content tokens and uses that information to produce gendered outputs. BOS zeroing increases bias because it forces attention to redistribute to occupation tokens, amplifying the gender signal.

---

## 8. The Final Results

### The Five Paper Claims (Revised and Honest)

**Claim 1: The Separability Spectrum**
> Bias-carrying attention heads exist on a spectrum from cleanly separable to deeply entangled. L10H9 achieves 25% absolute bias reduction with +1.9% PPL, while L0H8 achieves 36% but causes +103% PPL. Gradient importance does not predict separability -- L0H8 has the highest gradient but is most entangled.

**Claim 2: SAE Feature Artifact Warning**
> SAE features that appear to debias may actually suppress all gender probabilities. L0_F23406 reduces absolute bias 45% but shows zero improvement on CrowS-Pairs and halves gender probability mass. Always check gender mass and independent bias benchmarks.

**Claim 3: Partial Mechanistic Circuit**
> L10H9 is one node in a distributed bias circuit (19.8% sufficiency, 31.8% necessity). Layer 3-4 heads are heavily mediated through L10H9 (>70%). The circuit is partial and distributed, not complete.

**Claim 4: Honest Baseline Comparison**
> Dev-optimized INLP achieves 91% abs bias reduction with 0% PPL change, vastly outperforming our mechanistic approach (25%). However, INLP halves gender probability mass and drops LAMBADA accuracy, while L10H9 ablation preserves both. The right method depends on deployment constraints.

**Claim 5: Signed Bias Caveat**
> Our intervention reduces absolute bias (occupation-to-occupation variance) but NOT signed bias (male-default direction). Only INLP substantially reduces signed bias. Head ablation makes the model more uniformly male-default while reducing occupation-specific variation.

### What "works" and what doesn't

**Works**:
- L10H9 ablation reliably reduces absolute gender bias by ~25% across templates, occupations, and models
- The intervention barely touches PPL, LAMBADA, BLiMP, or GAP
- The separability spectrum is real and replicates on Pythia-2.8B
- The partial circuit (L3/L4 -> L10H9 -> output) is causally supported

**Doesn't work**:
- The intervention doesn't fix the male-default direction
- Winogender male preference actually increases
- Multi-head ablation has diminishing returns with increasing cost
- SAE-based combined intervention adds only ~3% over L10H9 alone

---

## 9. What the Results Actually Mean

### For the paper
We have a solid mechanistic interpretability case study, not a debiasing breakthrough. The contribution is the **separability spectrum** as an organizing concept + the **artifact warning** for SAE-based interventions + **honest methodology** (signed bias, proper splits, comprehensive benchmarks).

### For debiasing practitioners
If you want maximum bias reduction and don't care about gender mass, use INLP. If you want surgical intervention with minimal side effects and are willing to accept a smaller effect, head ablation works. If you want to understand WHY the model is biased, mechanistic interpretability reveals the circuit.

### For the field
The signed bias finding is important: reducing |P(M) - P(F)| is not the same as making the model fair. A model that always says "he" for every occupation has zero absolute bias but maximum signed bias. Our intervention moves the model slightly toward this degenerate case.

---

## 10. Key Concepts Glossary

**Absolute bias**: Mean of |P(male) - P(female)| across prompts. Measures size of bias regardless of direction.

**Signed bias**: Mean of (P(male) - P(female)). Positive = male-skewed. Measures direction and size.

**Gender mass**: Mean of (P(male) + P(female)). If this drops, the intervention is suppressing all gender tokens, not fixing bias.

**Perplexity (PPL)**: exp(mean cross-entropy loss). Measures how surprised the model is by real text. Lower = better. A PPL increase of +1.9% means the model is slightly worse at predicting text.

**Ablation**: Setting a component's output to zero. "L10H9 ablation" = multiply L10H9's output vector by 0 during forward pass.

**Alpha/scaling**: Multiplying a component's output by a factor between 0 and 1. Alpha=0 is full ablation, alpha=0.5 is 50% strength.

**Activation patching**: Replacing one component's activation in one forward pass with its value from a different forward pass. Used to test causal relationships.

**Sufficiency**: If you patch a component from clean->corrupted run and bias recovers, that component is sufficient to create bias.

**Necessity**: If you corrupt a component in the clean run and bias disappears, that component is necessary for bias.

**Mediation**: If head A's effect mostly disappears when head B is corrupted, A's effect is mediated through B.

**SAE (Sparse Autoencoder)**: Neural network that decomposes model activations into sparse, interpretable features. Each feature is a direction in activation space.

**INLP (Iterative Null-space Projection)**: Debiasing method that iteratively finds and removes linear directions that encode a protected attribute (gender).

**Bootstrap CI**: Confidence interval estimated by resampling the data 10,000 times and computing the metric each time. The 2.5th to 97.5th percentile gives a 95% CI.

**Pseudo-log-likelihood (PLL)**: For a sentence, sum of log P(token_i | all tokens before i). Higher = model finds the sentence more likely. Used for CrowS-Pairs and GAP scoring.

**Attention sink**: The phenomenon where transformer heads dump excess attention on the BOS token as a normalization mechanism, not because BOS contains useful information.

**Residual stream**: The running activation vector that passes through all layers. Each layer reads from and writes to the residual stream. Think of it as the "main bus" of the transformer.

**Hook**: In TransformerLens, a function that intercepts and optionally modifies activations during a forward pass. This is how we implement ablation, patching, and other interventions.

---

## Appendix: File Structure

```
mech_interp_bias/
  data/
    splits.json                          # Occupation/template splits
  scripts/
    eval_utils.py                        # Reusable evaluator (full_eval)
    experiment_00_setup.py               # Infrastructure check
    experiment_01_activation_patching.py  # Initial head scan
    experiment_02_logit_lens.py          # Layer-by-layer bias emergence
    experiment_03_entanglement.py        # Separability spectrum
    experiment_04_cross_bias.py          # Template generalization
    experiment_05_sae.py                 # SAE feature search (finds F23406)
    experiment_06_combined_intervention.py # Multi-intervention combos
    experiment_07_sae_feature_characterization.py # F23406 = artifact
    experiment_08_expanded_eval.py       # GPT-2-medium check
    experiment_09_steering_vectors.py    # Steering vector comparison
    experiment_10_edge_attribution.py    # Gradient-based EAP
    experiment_11_asymmetric_gender.py   # Male-default analysis
    experiment_12_true_gender_features.py # Real gender SAE features
    experiment_13_robust_evaluation.py   # Bootstrap CIs, z-scores
    experiment_14_scale_pythia.py        # Pythia-2.8B replication
    experiment_15_baselines_and_null.py  # Original INLP
    experiment_16_bos_path.py            # BOS analysis
    experiment_17_occupation_patching.py # Occupation causal role
    experiment_18_crows_pairs.py         # Original CrowS-Pairs (50 pairs)
    experiment_19_bias_specificity.py    # Cross-bias specificity
    experiment_20_data_splits_and_metrics.py # Data splits + evaluator validation
    experiment_21_head_rediscovery.py    # 144-head rescan (200 prompts)
    experiment_22_expanded_capability.py # Full benchmark validation
    experiment_23_path_patching.py       # Causal circuit analysis
    experiment_24_strong_inlp.py         # Tuned INLP baseline
    experiment_25_matched_dose_response.py # Alpha sweep + multi-head
    experiment_26_test_set_final.py      # HELD-OUT TEST SET (main table)
    experiment_27_bos_reframing.py       # Attention sink reanalysis
  results/
    20_data_splits/                      # Baseline validation
    21_head_rediscovery/                 # 144-head scan results
    22_expanded_capability/              # Full benchmark suite
    23_path_patching/                    # Causal circuit evidence
    24_strong_inlp/                      # INLP sweep (288 configs)
    25_dose_response/                    # Alpha sweep + Pareto frontier
    26_test_final/                       # PAPER MAIN TABLE
    27_bos_reframing/                    # Attention sink analysis
  FINDINGS_SYNTHESIS.md                  # Paper-oriented findings summary
  COMPREHENSIVE_GUIDE.md                 # This document
```
