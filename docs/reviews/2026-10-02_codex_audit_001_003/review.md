# Experiments 001–003: what failed, what remains supported, and what to do next

Audit of `/workspace/Causal-Concordance` at `40a9393`, 2026-10-02. Scope: Qwen3.6-27B, WorkspaceBench brew dev; 003 uses 82 gated pairs from 33 tables, with directions fitted from 231 renders across 48 dev tables. The intervention regime is stock bf16 through block 39 and true float32 replay above it. No new model forward or external judge call was run for this audit.

Reproducible checks: [audit script](audit_003.py), [computed results and source hashes](2026-10-02_003_audit_results.json), [execution log](2026-10-02_003_audit_reproduction.log).

**Main conclusion: 003 failed to produce a reliable reference for ranking readers. It did not establish that the intermediate is unused, nor that the whole project failed.** The engine checks passed, the tested one-dimensional directions usually have small positive effects, and the assignment of a binary “used” label is unstable across fitting data. Some report claims also exceed the measurements, and one registered prediction was scored incorrectly.

## 1. What actually ran

003 fits an additive regression from start colour, intermediate colour and answer colour to activations on other tables. For a colour contrast, it takes the difference between two fitted coefficient vectors, normalizes it, and changes the recipient activation's coordinate along that direction to the donor's coordinate. It then measures the change in the full-sequence donor-versus-recipient answer margin.

The ablation arm sets that coordinate to its fitting-set mean in both prompts and measures how much their answer-margin gap shrinks. These are one-cell edits. They are not a whole content-subspace replacement or a clamp of a J-derived span throughout downstream computation.

“Reader-free” is correct: the direction does not come from J, NLA, Oracle or SAE. But reader independence does not establish that a direction isolates the intermediate's causal role. The regression controls the fitted additive associations with input and answer labels; it does not guarantee a pure semantic intervention or remove table-dependent interactions.

## 2. The measured effects are small, not generally zero

At the final `:` cell, the following medians reproduce from raw 003 edits and the stored 001 reference:

| Layer | Whole-cell patch share | Fitted intermediate-direction share | Positive intermediate-direction effect | Donor answer log-probability increases |
|---|---:|---:|---:|---:|
| 40 | 0.350 | 0.0308 | 70/82 | 70/82 |
| 42 | 0.476 | 0.0344 | 69/82 | 69/82 |
| 44 | 0.503 | 0.0496 | 74/82 | 74/82 |

The share is a fraction of the natural pair's answer-margin gap; it is not accuracy, a flip rate, or a fraction of the model's computation. Positive mean effects occur in 28–29 of the 33 tables. Thus the result is not solely one positive outlier, though the largest effects are concentrated.

The threshold for “used” was 0.10 plus the null/damage checks. A typical 0.03–0.05 effect therefore fails that magnitude threshold even when it is positive. Most labels at `:` become **unresolved**, not evidence that the variable is absent or unused.

The positive-control result is relevant: the start-associated direction at the start token has median share 0.076 against a whole-cell share around 0.23, and only 28% receive the “used” label. This shows limited recovery even in the proposed control setting. It weakens the inference from a small intermediate-direction edit to a broad statement about the intermediate's role.

At L60, the answer-associated direction has a large median share (about 0.62), but many of these edits trigger the registered damage flag. This is evidence that the assay can produce large effects in a late answer-associated direction; it is not an unqualified clean causal reference.

## 3. Why the reliability gate failed

The disjoint reliability fits each use 77 renders from about 16 tables, fitting 34 regression coefficients per activation coordinate. Some colours have only two or three examples in a role. Full design-matrix rank does not mean those coefficients are estimated precisely.

Independent recomputation from the saved coefficient arrays gives:

| Layer at `:` | Median cosine between the two intermediate directions | Median absolute difference between their effect shares |
|---|---:|---:|
| 40 | 0.365 | 0.0255 |
| 42 | 0.361 | 0.0278 |
| 44 | 0.343 | 0.0349 |

Those effect differences are substantial relative to the median primary effect of 0.03–0.05. The instability is therefore not merely an unfortunate property of Cohen's κ.

Across L42 and L44, the two fits label 11 and 13 pair-layer cases positive. Only **5** are positive under both, **14** under one only, and **145** under neither. That produces 91% raw agreement but κ≈0.37: agreement is dominated by negative calls. The continuous effect correlation is about 0.41. These are 164 pair-layer observations from 82 pairs, not 164 independent problems.

The primary fit uses twice as many tables, so this does not directly measure its own reliability. It does show that the current evidence is insufficient to certify stable per-item labels.

**Inference, not an established explanation:** finite-sample direction estimation and table-dependent activation structure are plausible causes. There is no need to posit that the model has no intermediate representation. The high held-out direction AUC is compatible with low causal-edit reliability: several different directions can discriminate the same labels while perturbing downstream computation differently.

## 4. The necessity prediction was incorrectly marked false

Design L4 asks whether the ablation gap shrinkage exceeds its random-control q95 in at least 30% of pairs. `labels.py:650` instead reads `frac_used_necessary`, which additionally requires the swap to pass the 0.10 “used” threshold. These are different events.

The raw ablation records give:

| Layer at `:` | Above registered ablation-null q95 | Percentage | Reported “used AND necessary” |
|---|---:|---:|---:|
| 40 | 71/82 | 86.6% | 11.0% |
| 42 | 72/82 | 87.8% | 14.6% |
| 44 | 76/82 | 92.7% | 18.3% |

**L4 passes its written criterion.** This correction does not fix G4 or establish strong necessity: the median gap shrinkage remains about 3–5%, and the ablation control has a serious specificity limitation.

The swap null is norm-matched. **The ablation null is not.** It clamps a random coordinate to its own mean, usually making a much smaller edit. Reconstructing the actual random draws and prompt-only edit sizes from cached activations gives a median real-to-random ablation RMS-norm ratio of approximately **21×, 19× and 20×** at L40, L42 and L44. The computation combines the member and partner edit norms and compares the real intervention with the eight random controls; this is an input-dose comparison, not a new behavioral measurement.

Consequently, “the fitted-coordinate ablation beats these random-coordinate ablations” is observed, but an equally large generic perturbation has not been ruled out. Add norm-matched ablation perturbations before upgrading this to causal specificity. Also, the raw necessity rows contain margins and fidelity only: they do not contain the promised separate answer-probability or damage measurements for the two ablated runs. Damage on a swap is not a substitute for damage on an ablation.

## 5. Two label-inference problems survive the numerical gate checks

**The pooled swap null does not automatically give calibrated per-item p-values.** Each item gets eight random controls matched to that item's real edit size. Analysis then pools all items' shares within a cell/layer/content and compares each item to that mixture. Edit sizes, answer gaps and local sensitivities differ between items. Pooling does not establish that this mixture is the correct null for each item.

At `:` L42, individual eight-control q95 estimates range from roughly 0.00008 to 0.068. The eight-draw estimates themselves are noisy, but they demonstrate why an exchangeability assumption needs checking. A direct per-item Monte Carlo test with eight draws has minimum nonzero p=1/9; the much smaller pooled p-values do not acquire per-item calibration just by having more other-item draws.

This does not erase the descriptive positive effects. It does mean that the “BH-significant per-item ground truth” claim requires additional calibration. G4 also uses unadjusted significance with the primary fit's pooled null, rather than repeating the complete primary labelling pipeline for each reliability fit.

**Family-level decodability was promoted to item-level labels.** At `labels.py:570`, a small whole-cell share assigns “decodable, not needed” to every relevant item; no held-out per-item probe probability is consulted. The repaired probe scores around 0.79–0.84 establish strong dataset-level decodability at the `</think>` newline, not that every one of those items passes the spec's item-level criterion.

The defensible observation is: **this position supports decoding on the dataset, while donor whole-cell patches often have small net effects on the specified answer.** A small net donor-patch effect also cannot exclude cancellation between components. Keep this as a readable/low-patch-leverage pattern until the stronger labels are justified.

## 6. “Other content carries 0.21–0.42” is not measured

The attribution code sets `other = S_full - max(S_start, S_z, S_answer)` (`labels.py:629`). The report interprets that value as influence outside all three content directions.

That conclusion is invalid. Even an additive two-direction system with individual effects 1 and 1 has whole-cell effect 2 and `full - max = 1`, despite there being no effect outside the joint span. Nonlinearity and interactions make the subtraction still less informative.

The supported wording is: **none of the three individually tested directions reproduces the full-cell effect.** To test the missing explanation, actually patch their joint span and its complement. Even then, report each intervention's effect and interactions; do not assume additive causal credit.

One qualitative correction also matters: in table 0383, the `white` partner follows **white → teal → gold**, not an orange answer. The strongest z-swap example raises orange mass to about 0.23 and gold to about 0.19, while brown remains about 0.40. The report's prose calls orange the partner answer across these examples; that is wrong for the white partner. The margin effect can be real while the output shift is only partly specific to the intended answer.

## 7. What 001 and 002 still tell us

**The probe repair improved a real measurement problem.** Multiple control draws, within-fold AUC and retuning C under the null are implemented. Controls now average near chance; off-path membership at `:` L40–44 passes the current family-level rule narrowly. The saved solver and grouped-null checks support keeping these as descriptive decodability results. They do not manufacture item-level causal labels.

**002 shows a difference between colour scores, mentions, and faithful role claims.** The lenses' scores contain activation-sensitive intermediate information below the top-10 cutoff. On the ten verbalizer tables, NLA mentions the intermediate much more often than it assigns the correct role; Oracle often emits incompatible candidate chains under the whole-readout rubric. These are useful instrument observations. The Oracle prompt asks for distinct concepts, so the penalty for one globally inconsistent chain is a property of this scoring task, not a universal statement about its usefulness for exploration.

One recurring summary claim needs correction: **positive tracking is not necessarily a ranking flip.** The 002 statistic is a difference-in-differences of scores. It can be positive while the same colour remains ranked above the other in both prompts. For J_NP cosine at `:` L40:

- 48/48 tables have positive mean tracking;
- only 90/183 natural partner comparisons actually reverse the two colours' ranking in the correct direction;
- 35/48 tables have at least one such reversal, and 8/48 reverse for every tested partner.

At L42 the corresponding pairwise reversals are 72/183. This preserves the evidence for activation-sensitive information but corrects the near-perfect-flipping interpretation in STATUS and the motivation for 003.

The existing 002 overlap table remains a small descriptive result: readers can overlap in mentioning a colour without agreeing on its role. It does not yet give an agreement-versus-causal-truth atlas. The proposed “W≈0” interpretation in report 003 is also premature: the causal labels failed reliability, the comparison changes position, reader tracking is not the registered hit statistic, and the reader and intervention populations differ (48 versus 33 tables). Describe the two measured patterns separately.

## 8. Highest-value next action

**Repair the analysis and validate the intervention assay on a small dev slice before scoring readers against 003.** Do not launch a large causal ranking or select a more flexible method just to obtain larger effects.

First, make the corrections above using existing files: L4, ranking-change wording, item-label scope, “other” attribution, and the outlier's answer identity. Keep continuous effects and the reliability failure visible. This costs no model forwards.

The next GPU experiment should be a small, separately designed **L42 assay check** on a fixed set of distinct dev tables, with the original start-token control retained:

1. Compare the existing 1-D edit with a reader-free intermediate-associated span, the joint fitted content span, their complements, and the whole-cell patch. Measure the complement directly. Use matched-rank and matched-edit-norm controls where applicable.
2. Repeat the central edits under two disjoint direction fits. Test whether expanding the span improves reproducibility as well as effect size.
3. Add ablation controls matched to the real edit magnitudes and log damage on both ablated prompts.

The experiment distinguishes useful branches:

- **A stable associated span recovers much more influence:** the one-direction assay was too restrictive; proceed to validation of the span's specificity.
- **The joint content span works but the intermediate span does not:** the relevant influence is not isolated by the present intermediate estimator. Do not label it “z unused.”
- **The span and complement effects interact strongly:** additive attribution is inadequate; report that interaction rather than allocating a fictitious “other” share.
- **Directions remain unstable or the control setting is poorly recovered:** improve fitting support/regularization and reassess the assay before using it as reference truth. Additional independent tables may help more than extra prompts from the same tables.
- **Norm-matched ablations remove the apparent specificity:** retract the stronger necessity interpretation while retaining the measured swap effects.

DAS remains a later alternative to compare, not an automatic rescue. A larger optimized effect is not by itself a better truth label.

## 9. Audit limits and housekeeping

I independently recomputed the central effects, necessity counts, cross-fit geometry, ablation input doses and score-ranking reversals from saved artifacts, and read the 003 measurement and analysis code. I did not rerun the model, validate every judge annotation, or independently certify the entire inherited replay engine.

Two secondary issues remain: the design promises full-sequence colour distributions, but `colour_mass` still records first-token colour mass (including the first token of multi-token ` Teal`); and the report's 37,299 “edited forwards” is a count of edit records, with a necessity record containing two scored runs. There are 43,941 scored edit evaluations by that counting convention, before extra multi-token continuations. These do not explain G4, but should be corrected for provenance.

STATUS and the handoff documents retain stale sections that say probes/readers have not run after their new top blocks say they are complete. That makes the project look more contradictory than it is. The reliable current scientific status is: **descriptive reader differences and local intervention effects exist; a stable content-specific causal adjudication benchmark does not yet exist.**
