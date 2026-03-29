# Mechanistic Interpretability for Bias Localization in LLMs
## Comprehensive Findings Synthesis (Experiments 00–27)

**Authors**: Luca Sfragara, Ayela Chughtai, Pantelis Emmanouil, Shriya Karam

---

## 1. Executive Summary

We apply mechanistic interpretability techniques to localize and intervene on occupation-conditioned gender preference in GPT-2-small, revealing a **separability spectrum**: some bias-carrying components can be surgically removed with minimal capability cost, while others are deeply entangled with core language modeling. Our key contributions:

1. **Separability spectrum**: L10H9 achieves 25% absolute bias reduction with +1.9% PPL change (separable), while L0H8 has highest gradient importance but catastrophic entanglement (+103% PPL)
2. **Artifact identification**: SAE feature L0_F23406 appears to debias but actually suppresses all gender probabilities — a cautionary tale for SAE-based interventions
3. **Partial mechanistic circuit**: L10H9 is one node in a distributed bias circuit (19.8% sufficiency, 31.8% necessity), with upstream contributions from L4H3 (83% mediated), L4H7 (71%), and L9H7 (43%)
4. **Honest INLP comparison**: Dev-optimized INLP (layer 10, 10 iterations) achieves 91% bias reduction with 0% PPL change but halves total gender probability mass and drops LAMBADA accuracy
5. **Scale replication**: Pythia-2.8B L22H30 achieves 40.6% bias reduction on held-out test set

### Scope

This work studies **occupation-conditioned gender preference in small causal LMs** (GPT-2-small, Pythia-2.8B). We use "bias reduction" as shorthand for "reduction in absolute difference between male and female pronoun probabilities following occupation tokens." We do not claim these methods generalize to all forms of bias or all model scales.

---

## 2. Core Finding: The Separability Spectrum

### 2.1 Head-Level Localization (Exp 01, 03, 21)

Head re-discovery on a principled **discovery set** (200 prompts from 20 occupations, non-overlapping with dev/test) confirms L10H9:

| Head | Rank | Abs Bias Reduction | WikiText PPL Δ | Separable? |
|------|------|-------------------|----------------|------------|
| L0H8 | #1 | 35.8% | +103% | NO (entangled) |
| **L10H9** | **#2** | **32.9%** | **+1.9%** | **YES** |
| L0H9 | #3 | 29.0% | +22.8% | NO |
| L11H0 | #4 | 28.2% | +7.7% | Borderline |
| L5H9 | #5 | 24.0% | Unknown | TBD |

**L10H9 is the top separable head** — rediscovered on a separate discovery set with 200 prompts (up from ~20 in original exp 01).

### 2.2 Test-Set Results (Exp 26 — Held-Out, 250 Prompts, 25 Never-Seen Occupations)

| Metric | Baseline | L10H9 Ablated | Combined | F23406 (ctrl) |
|--------|----------|---------------|----------|---------------|
| Signed bias | +0.059 [0.049, 0.070] | +0.052 [0.045, 0.059] | +0.052 [0.046, 0.059] | +0.027 [0.018, 0.035] |
| Abs bias | 0.079 [0.071, 0.087] | 0.059 [0.053, 0.065] | 0.056 [0.051, 0.062] | 0.043 [0.036, 0.051] |
| Gender mass | 0.189 | 0.152 | 0.149 | 0.112 |
| WikiText PPL | 123.6 [110.6, 138.4] | 126.0 [112.6, 141.1] | 126.0 [112.5, 141.4] | 124.5 [111.7, 139.5] |
| LAMBADA acc | 24.8% | 25.0% | 25.0% | 23.8% |
| BLiMP mean | 98.2% | 97.8% | 97.8% | 97.9% |
| Winogender M% | 78.8% | 86.7% | 90.8% | 77.5% |
| WinoBias T1 gap | 0.5pp | -0.8pp | -1.5pp | 0.5pp |
| WinoBias T2 gap | 24.2pp | 18.9pp | 18.4pp | 24.2pp |
| GAP overall | 70.6% | 71.0% | 71.1% | 70.8% |
| CrowS-Pairs | 60.7% | 59.2% | 57.6% | 61.1% |

**Key observations:**
- L10H9 ablation: 25.3% abs bias reduction, +1.9% PPL, LAMBADA/BLiMP/GAP unchanged
- Combined: 28.5% abs bias reduction, same PPL cost
- F23406: Appears to reduce bias but signed bias reveals it mostly suppresses gender mass (0.189→0.112)
- Signed bias barely changes for L10H9 ablation (+0.059→+0.052), confirming the reviewer's concern about directional effects
- Winogender male preference INCREASES after L10H9 ablation (78.8%→86.7%) — the head was partially counteracting male default

### 2.3 Why Separability Matters

High bias *importance* (gradient) does not predict separability:
- L0H8: Highest gradient (0.102), rank 1 — but deeply entangled (ablation destroys fluency)
- L10H9: Moderate gradient (0.052), rank 17 by gradient — but cleanly separable (ablation preserves everything)

---

## 3. SAE Feature Analysis: Artifacts vs True Gender Features

### 3.1 The L0_F23406 Artifact (Exp 05, 07, 13, 22)

| Metric | L0_F23406 | True Gender Features |
|--------|-----------|---------------------|
| Abs bias reduction | 45.3% | 13.1% |
| Gender mass change | -40.7% | -1.0% |
| CrowS-Pairs | 61.1% (NO change) | — |
| Top activations | "that", "which", "who" | "she", "her", "himself" |

L0_F23406 is a **complementizer feature**, not a gender feature. It achieves "debiasing" by suppressing total gender probability mass. CrowS-Pairs confirms zero improvement (61.1% vs 60.7% baseline).

---

## 4. Partial Mechanistic Circuit of L10H9 (Exp 10, 16, 17, 23, 27)

### 4.1 Path Patching Results (Exp 23 — Proper Causal Analysis)

| Test | Result | Interpretation |
|------|--------|---------------|
| **Sufficiency** | 19.8% recovery | L10H9 alone does NOT fully account for occupation bias |
| **Necessity** | 31.8% removal | Corrupting L10H9 partially removes bias |

L10H9 is **one node in a distributed circuit**, not the complete circuit. This is an honest downgrade from the original "complete circuit" claim.

### 4.2 Upstream Sources (Mediated Through L10H9)

| Head | Total Effect | Direct (bypass L10H9) | Mediated via L10H9 |
|------|-------------|----------------------|-------------------|
| L4H3 | +19.0% | +3.3% | **83%** |
| L4H11 | -16.2% | +0.5% | **103%** |
| L4H7 | +17.5% | +5.0% | **71%** |
| L3H6 | +12.8% | +3.3% | **74%** |
| L8H5 | -27.8% | -9.2% | **67%** |
| L9H7 | +37.3% | +21.4% | **43%** |

Layer 3-4 heads feed heavily through L10H9 (>70% mediated). Later heads (L8, L9) have more direct paths to the output.

### 4.3 BOS as Attention Sink (Exp 27 — Reframed)

- 101/144 heads have >50% BOS attention — BOS is a universal **attention sink** (Xiao et al. 2023)
- L10H9's BOS attention (47%) is actually **LOWER** than other L10 heads (85.3% mean, z=-3.96)
- This means L10H9 attends MORE to content tokens than typical heads
- When BOS attention is zeroed, bias increases +30.6% because attention redistributes to occupation tokens
- **Revised interpretation**: BOS is a normalizing mechanism, not an information carrier

---

## 5. Intervention Comparison (Exp 22, 24, 25, 26)

### 5.1 Honest Comparison Table (Dev Set)

| Method | Abs Bias | Signed Bias | Gender Mass | WikiText PPL | LAMBADA | Winogender M% | CrowS-Pairs |
|--------|----------|-------------|-------------|-------------|---------|---------------|-------------|
| Baseline | 0.123 | +0.074 | 0.196 | 123.6 | 24.8% | 78.8% | 60.7% |
| L10H9 ablation | 0.090 | +0.074 | 0.165 | 126.0 | 25.0% | 86.7% | 59.2% |
| Combined (L10H9+SAE) | 0.084 | +0.073 | 0.160 | 126.0 | 25.0% | 90.8% | 57.6% |
| **INLP (dev-optimized)** | **0.012** | **+0.001** | **0.093** | **123.6** | **19.4%** | **69.2%** | **60.7%** |
| F23406 artifact | 0.082 | +0.050 | 0.137 | 124.5 | 23.8% | 77.5% | 61.1% |

### 5.2 Key Observations

1. **INLP dominates on bias metrics**: 91% abs reduction vs 27% for L10H9, with zero PPL cost
2. **BUT INLP halves gender mass** (0.196→0.093) and drops LAMBADA (24.8%→19.4%)
3. **L10H9 ablation barely changes signed bias** (+0.074→+0.074), confirming the intervention reduces variance but not the male-default direction
4. **L10H9 increases Winogender male preference** (78.8%→86.7%), suggesting L10H9 was partially counterbalancing male default
5. **INLP moves Winogender toward 50%** (78.8%→69.2%), the only method that reduces directional bias

### 5.3 Dose-Response (Exp 25)

L10H9 scaling shows linear dose-response: each 0.1 alpha reduction gives ~2.7% abs bias reduction. Multi-head ablation (L10H9+L11H0+L5H9) reaches 36.5% but with 35% PPL cost — diminishing returns with increasing capability damage.

---

## 6. Scale Replication (Exp 14, 26)

### 6.1 Pythia-2.8B Test-Set Results (Held-Out, 250 Prompts)

| Metric | Pythia Baseline | Pythia L22H30 Ablated |
|--------|----------------|----------------------|
| Signed bias | +0.032 [0.023, 0.041] | +0.021 [0.015, 0.026] |
| Abs bias | 0.056 [0.049, 0.063] | 0.033 [0.029, 0.038] |
| WikiText PPL | 59.6 | 59.7 (+0.2%) |
| LAMBADA | 52.8% | 52.8% (unchanged) |
| Winogender M% | 74.2% | 80.0% |
| CrowS-Pairs | 64.5% | 63.7% |

L22H30 achieves **40.6% abs bias reduction** with essentially zero PPL cost on held-out test set. Separability spectrum confirmed across architectures.

---

## 7. Robustness (Exp 13, 18, 20, 21)

### 7.1 Data Splits

- **Discovery**: 20 occupations × 10 templates = 200 prompts (used for head scan)
- **Dev**: 20 occupations × 10 templates = 200 prompts (used for intervention tuning)
- **Test**: 25 occupations × 10 templates = 250 prompts (used for final numbers only)
- **Zero overlap** between splits in occupations or templates

### 7.2 Capability Benchmarks

All interventions evaluated on:
- WikiText-103 validation PPL (1000 sentences)
- LAMBADA last-word prediction (500 examples)
- BLiMP minimal pairs (anaphor_gender_agreement + SVA)
- Winogender (240 occupation-pronoun pairs, split by pronoun form)
- WinoBias (type1 + type2, pro vs anti-stereotypical coreference)
- GAP (2000 Wikipedia pronoun-name coreference examples)
- CrowS-Pairs (262 real gender pairs from nyu-mll dataset)
- Bootstrap 95% CIs (10,000 resamples) on all metrics

### 7.3 Cross-Bias Specificity (Exp 19 — Appendix)

L10H9 is gender-specific: rank #2 for gender bias but #129-132 for race/age/religion. However, non-gender baselines are tiny (~4.5e-05 for race vs 0.117 for gender), making these ranks potentially unstable. This result belongs in the appendix, not as a main claim.

---

## 8. Key Claims for Paper (Revised)

### Claim 1: The Separability Spectrum
> Bias-carrying attention heads exist on a spectrum from cleanly separable to deeply entangled. L10H9 (rank #2 by abs bias reduction, rediscovered on held-out discovery set) achieves 25% bias reduction with +1.9% PPL increase, while L0H8 (rank #1) causes +103% PPL increase. This spectrum cannot be predicted from gradient importance alone.

### Claim 2: SAE Feature Artifact Warning
> SAE features that appear to "debias" may actually suppress all gender probabilities. L0_F23406 reduces absolute bias 45% but shows zero improvement on CrowS-Pairs (61.1% vs 60.7% baseline) and halves gender probability mass — a methodological cautionary tale.

### Claim 3: Partial Mechanistic Circuit
> L10H9 is one node in a distributed bias circuit (19.8% sufficiency, 31.8% necessity via path patching). Layer 3-4 heads (L4H3, L4H7, L3H6) are heavily mediated through L10H9 (>70%), while later heads (L9H7, L8H5) have more direct output paths.

### Claim 4: Honest Baseline Comparison
> Dev-optimized INLP achieves 91% abs bias reduction with zero PPL cost, substantially outperforming our mechanistic interventions (27%). However, INLP halves total gender probability mass (0.196→0.093) and drops LAMBADA accuracy (24.8%→19.4%), while L10H9 ablation preserves both. The right method depends on the deployment constraint.

### Claim 5: Signed Bias Caveat
> Our interventions reduce absolute bias (variance between male/female) but barely change signed bias (male-default direction). L10H9 ablation's signed bias: +0.074→+0.074. Only INLP substantially reduces signed bias (+0.074→+0.001). Head ablation reduces occupation-specific gender variance without addressing the overall male skew.

---

## 9. Recommended Paper Structure

**Title**: *Separable and Entangled: Mechanistic Analysis of Occupation-Gender Preference Circuits in Language Models*

1. **Introduction**: Occupation-conditioned gender preference in LLMs, promise of mech interp for surgical intervention, the problem of entanglement
2. **Related Work**: INLP, CDA, CrowS-Pairs/WinoBias benchmarks, mech interp (Elhage, Conmy), SAE interpretability, attention sinks (Xiao et al. 2023)
3. **Methods**: TransformerLens, head ablation, SAE feature analysis, path patching, specificity score
4. **The Separability Spectrum** (Section 4.1): Head scan results, L10H9 vs L0H8, discovery/dev/test split methodology
5. **SAE Feature Artifacts** (Section 4.2): L0_F23406 cautionary tale, true gender features, specificity analysis
6. **Partial Circuit Analysis** (Section 4.3): Path patching (sufficiency/necessity/mediation), BOS as attention sink
7. **Evaluation** (Section 5): Full benchmark suite (WikiText, LAMBADA, BLiMP, WinoBias, Winogender, GAP, CrowS-Pairs), bootstrap CIs, signed vs absolute bias, INLP comparison, Pythia replication
8. **Discussion**: Signed bias caveat, INLP tradeoffs, limitations (small models, gender only, English only)
9. **Conclusion**: The separability spectrum as organizing principle, artifact warning, capability-bias tradeoffs matter

---

## 10. Experiment Index

| # | Name | Key Result |
|---|------|------------|
| 00 | Setup | TransformerLens + SAELens loaded |
| 01 | Activation patching | L10H9 identified as separable bias head |
| 02 | Logit lens | Bias emerges in layers 9-11 |
| 03 | Entanglement | L10H9 separable, L0H8 entangled |
| 04 | Cross-bias | Occupation bias generalizes across templates |
| 05 | SAE features | L0_F23406 appears to remove 80% bias |
| 06 | Combined | Multi-head ablation combinations |
| 07 | Feature characterization | L0_F23406 is complementizer, not gender |
| 08 | Expanded eval | GPT-2-medium: late-layer localization confirmed |
| 09 | Steering vectors | Head scaling > SAE > steering vectors |
| 10 | Edge attribution | OV circuit analysis, L10H9 is gender-copier |
| 11 | Asymmetric gender | Male-default hypothesis confirmed |
| 12 | True gender features | L10_F23440 (female) + L10_F16291 (male) |
| 13 | Robust evaluation | Bootstrap CIs, z=19.09, specificity scores |
| 14 | Pythia-2.8B scale | L22H30: 40.6% bias red, +0.2% PPL — replicates |
| 15 | Baselines + null | Original INLP comparison |
| 16 | BOS path | BOS is context-independent |
| 17 | Occupation patching | r=0.69 occupation→bias, bidirectional |
| 18 | CrowS-Pairs | Original 50 pairs (superseded by exp 22/26) |
| 19 | Bias specificity | L10H9 gender-specific (appendix) |
| **20** | **Data splits + metrics** | **3 non-overlapping splits, reusable evaluator** |
| **21** | **Head re-discovery** | **L10H9 rank #2/144 on discovery set** |
| **22** | **Expanded capability** | **Full benchmark suite validated** |
| **23** | **Path patching** | **19.8% sufficiency, 31.8% necessity** |
| **24** | **Strong INLP** | **91% bias red, 0% PPL, but halves gender mass** |
| **25** | **Dose-response** | **Linear scaling, diminishing multi-head returns** |
| **26** | **Test-set final** | **Paper's main results (GPT-2 + Pythia)** |
| **27** | **BOS reframing** | **Attention sink (z=-3.96), not causal pathway** |
