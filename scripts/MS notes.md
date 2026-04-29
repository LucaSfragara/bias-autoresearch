# Notes on separating overall male skew with bias reduction

**Authors**: Luca Sfragara, Ayela Chughtai, Pantelis Emmanouil, Shriya Karam

---

## Experiment 1: Disentangling stereotype amplification vs. broader male skew

Main question: Does L10H9 (and other interventions) genuinely reduce occupation-triggered stereotype amplification, or does it merely suppress gendered language output / leave a broad male-leaning tendency intact?

L10H9 head ablation, attenuation, and multi-head ablation all show that occupation-specific words amplify bias (shown by the positive male-occ signed bias and negative female-occ signed bias). However, these interventions also reduce overall gender mass relative to the baseline. The neutral signed bias remains positive and close to baseline levels, suggesting that a general male-skewed prior persists even when the head is modified. Together, these results imply that L10H9 is not solely responsible for encoding stereotype-specific amplification; its removal also partially suppresses overall gendered signal, rather than cleanly isolating and eliminating occupation-conditioned bias.

The main tension is between criteria A (stereotype amplification) and C (total gender mass): getting enough amplification reduction requires enough head suppression to dent the gender mass. This suggests L10H9 is not purely a "stereotype amplification head" in isolation; it contributes to the model's general gendered-language machinery as well, and those two functions are entangled within the same head.

## Experiment 2: Graduated alpha sweep and Pareto analysis of head selectivity

Main question: How does attenuating heads to different degrees impact bias amplification, residual skew, and gender mass? 

L10H9 is the most effective head for reducing stereotype amplification: full ablation (α=0) cuts amplification by 46%, while no other head — within Layer 10 or across layers — exceeds ~4%. However, this comes at the cost of a 22.4% reduction in total gendered pronoun mass and a slight male-ward shift in residual skew (from −0.019 to −0.002), meaning L10H9 normally provides a partial female-ward correction that counteracts the model's generic male prior. Its selectivity index (1.816) is highest among all tested heads, confirming it offers the best ratio of debiasing benefit to collateral damage — but the Pareto frontier is empty, meaning no head achieves >10% amplification reduction without >10% mass suppression. The amplification and gendered-language functions are entangled within L10H9, ruling out a simple scaling intervention that cleanly isolates stereotype reduction.








