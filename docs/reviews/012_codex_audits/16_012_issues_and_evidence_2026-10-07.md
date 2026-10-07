# 012: issues, evidence, and recommended corrections

Independent Codex audit for Vishesh, 2026-10-07. Rechecked repository HEAD `2f72ea4` on `012-reader-concordance`. The affected canonical F2/R code, statistics and ledger claims remain unchanged at this inspection. This report consolidates the actionable issues from reviews 09–15. It distinguishes implementation errors, limits on validation, and unsupported inferences; these are not interchangeable findings.

Repository and canonical artifacts were read-only. Only this report and the Codex context ledger were written. No project workflow or GPU experiment was executed. Recommendations below are for Claude Code, subject to the PI's research decisions.

## Priority assessment

The highest-priority issue is a confirmed position-filter error in F2 and R. Existing outputs suffice to correct it. It changes the basis for the proposed NLA-specific finding. The AR work has useful calibration evidence, but does not meet its validation bar and cannot yet establish the historical training or reward objective. The robust intervention results remain useful; these issues do not invalidate all of 012.

| ID | Issue | Type | Confidence / priority |
|---|---|---|---|
| 1 | Reader and model statistics average different positions | Confirmed implementation error | High; correct first |
| 2 | Positive F2 interpretation persists after its supporting comparison fails | Stale scientific interpretation | High; correct with issue 1 |
| 3 | Reader specificity inferred from different significance outcomes | Invalid decision rule | High; replace before another specificity claim |
| 4 | Intervention tracking can be mistaken for natural concept mediation | Scope/inference limit | High; limits the central claim |
| 5 | AR “held-out” validation reuses loader-selection texts | Validation-independence limit | High; clarify immediately |
| 6 | Scaled profile is treated as stronger evidence than it supplies | Calibration/inference limit; gate already marked failed | High about procedure, unresolved cause |
| 7 | Candidate metrics use inconsistent centering/FVE/retrieval conventions | Comparison mismatch | High; does not invalidate leading candidate directly |
| 8 | Validation metric is promoted into AR loss and Oracle reward | Unsupported inference chain | High about missing links; objective itself unresolved |

## 1. F2 and R compare different sets of positions

**Issue.** The same filter is applied to two different row schemas. Model rows have `is_read_minus_1`; reader rows do not. The code excludes the predecessor of a final `is` only from model averages. The reader statistic therefore includes a cell absent from its output-facing comparator.

**Why I believe it.** The filter is visible in [stageF_analysis.py](/workspace/causal-concordance/scripts/012_reader_concordance/stageF_analysis.py:82) and [stageR_analysis.py](/workspace/causal-concordance/scripts/012_reader_concordance/stageR_analysis.py:29). Raw JSONL schema inspection confirms the missing reader field. Independent CPU recomputation reproduced published points before applying one shared `(item, pos)` inclusion set. Root separately checked the critical points rather than accepting the subagent's conclusion.

F2 is affected on 52/82 primary items and 40/64 toward-target items. Its prespecified common exclusion gives:

| NLA L42 statistic | Published | Shared intended mask |
|---|---:|---:|
| AUC | 0.7202380952 | 0.5853174603 |
| Partial Spearman rho | 0.5359641359 | 0.1742155802 |

The corrected AUC interval is [0.4419311309, 0.7676415344]; corrected partial interval is [−0.1651917446, 0.3923664825]. AUC uses 64 items, partial 63 complete cases. Keeping the predecessor for both instead gives model AUC 0.7400793651 versus NLA 0.7202380952, and NLA partial 0.1647985553 [−0.2527750305, 0.4140126718]. Thus the positive beyond-output interpretation fails under either aligned analysis.

R inherits the same error, affecting 33/146 primary and 23/109 toward-target items. Under the literal §16 all-early-position endpoint:

| Reader | Published partial rho | Shared-mask rho | Shared-mask 95% interval |
|---|---:|---:|---|
| NLA L42 | 0.3441045214 | −0.0643491055 | [−0.3314121392, 0.1614803978] |
| Oracle L44 | 0.4023550634 | 0.1187179166 | [−0.2628674886, 0.3859361969] |

R's wording also references F2, so inheriting F2's exclusion is a reasonable alternative endpoint. Excluding the predecessor from both again gives intervals spanning zero. R's existing negative replication verdict survives; its positive partial point estimates do not.

**Consequence.** The omitted model position is informative. This is a substantive baseline mismatch, not the cosmetic plotting issue already mentioned in the ledger. It can make reader scores appear to contain information absent from the output baseline simply because the latter was denied the same cells.

**Recommended correction.** Join explicit position metadata, declare the endpoint, apply one common mask, and assert identical included positions for each reader/comparator pair. Recompute CPU summaries and update dependent reports/pages. No new model captures or reader run is needed. Do not silently pick a new endpoint based on the better result; retain F2's intended endpoint and disclose the R ambiguity.

**Confidence:** high in both diagnosis and numerical effect. These corrections remove support for a particular positive claim; they do not establish that NLA has no additional information.

## 2. The proposed replication still rests on the erroneous F2 headline

**Issue.** The ledger describes F2 as an informative positive hint awaiting fresh hidden-bridge items, and explains R's failure partly through the changed family. That framing omits the error in the original supporting comparison.

**Evidence.** The canonical F2 JSON still reports NLA 0.72 AUC / 0.54 partial, while the corrected intended-mask values are 0.59 / 0.17 and fail the planned signal criterion. The ledger's R qualification and stopping point still propose fresh hidden-bridge replication.

**Consequence.** More compute could be spent confirming an effect whose existing evidence changes materially after a cheap analysis correction. R's explicit-source broadcast family really does differ from hidden-bridge multihop, but that difference does not rescue F2's original evidence. The broadcast bank was already examined model-side in A, so R also is not wholly untouched confirmation.

**Recommended wording.** “The original F2 positive interpretation did not survive correction of a position-mask mismatch. R finds strong shared tracking but no demonstrated additional NLA association under aligned output controls.” A future hidden-bridge study could ask a new, well-specified question; it should not be justified solely as replication of the old headline.

**Confidence:** high. This is a research-prioritization consequence of issue 1, not a second independent implementation defect.

## 3. “NLA significant, other readers nonsignificant” does not show specificity

**Issue.** [Spec §16](/workspace/causal-concordance/docs/design_specs/012_reader_concordance.md:513) and [the verdict code](/workspace/causal-concordance/scripts/012_reader_concordance/stageR_analysis.py:65) permit “NLA-specific” when NLA passes its threshold and other readers fail theirs.

**Reason.** A difference in significance can reflect different uncertainty, even when the underlying effects are similar. The scientific question requires a direct paired reader contrast. The present negative R outcome did not trigger this invalid positive label, but the rule would mislicense a future conclusion.

**Existing evidence.** R's all-early AUC is 0.9468344156 for NLA and 0.9472402597 for Oracle L44. Their paired difference is −0.0004058442 [−0.0191390254, 0.0197368421], using a 10,000-draw source-cluster bootstrap. These results do not support specificity. There are only 16 source clusters despite 109 items, which limits generalization.

**Recommended correction.** State the between-reader contrast and a relevant effect threshold in advance. Match items and cells and use paired cluster uncertainty. Preserve “unresolved” where appropriate; lack of a detected difference is not universal equivalence.

**Confidence:** high in the statistical objection.

## 4. J-derived intervention labels do not establish natural bridge mediation

**Issue.** Behavioral labels are reader-free, which is valuable, but the intervention directions are derived from J rows. Readers and model output may all respond to the imposed signal or its downstream propagation. This differs from showing that a particular naturally occurring cell mediates a hidden concept in the clean computation.

**Evidence.** In R the source is explicit, the intervention writes every prompt position, and the model's early-position bridge-output AUC is 0.9626623377, alongside NLA/Oracle AUCs about 0.947. Strong reader agreement is therefore compatible with shared sensitivity to effective steering. F2/R “edit size” controls summarize writes at the read position, rather than the local dose at every analyzed early cell. Partial correlations do not remove every output-facing, dose or intervention-geometry explanation.

**Consequence.** These results support outcome tracking under the tested interventions. They do not yield per-cell semantic ground truth, establish a natural intermediate mechanism, or prove readers reveal information inaccessible at the output. Conversely, the strong output baseline does not prove readers literally copy it.

**Recommended next scientific check.** If we need to distinguish these explanations, use a small target-balanced intervention reference selected independently of J—such as a natural state transfer or class-mean intervention—with its own leverage and specificity checks. Relation-dependent answers under the same concept change would be useful against generic answer attraction. This is a proposed experiment for Claude, not an experiment executed here.

**Confidence:** high about the inference limit; which competing mechanism holds remains unresolved.

## 5. The AR spans are not held out from local loader selection

**Issue.** “1,024 held-out spans” can sound like independent validation of the chosen loader. They are recrops of the same twelve deterministic responses used during 011's convention selection.

**Why I believe it.** E0 explicitly uses `011 ar_vectors.gen_texts (12 VAL_PROMPTS, greedy 160)`. Source handling is in [stageE0_ar_profile.py](/workspace/causal-concordance/scripts/012_reader_concordance/stageE0_ar_profile.py:105). Audit 15 reconstructed 768/768 earlier saved 011 crops from E0's response texts. E0 contains 792 distinct preceding-token target positions among 1,024 overlapping crops. Its retrieval summary evaluates 1,008 rows in 42 groups of 24.

**Consequence.** Additional layers and crop lengths are useful diagnostics, but not independent text-level validation of choices already tuned on these contexts. The crop count must not be read as 1,024 independent contexts. This does not establish contamination of the AR's original training data; that is a separate, unchecked question.

**Recommended correction.** Call these reused calibration contexts. If independent instrument validation is needed, freeze the forward/metric convention, fit scales on calibration texts and evaluate on genuinely separate response-level texts. An official reference input/output fixture is more direct for checking forward equivalence.

**Confidence:** high in the reuse diagnosis; it does not prove that the measured agreement is entirely caused by reuse.

## 6. Scaled FVE does not validate the loader or identify a scale-invariant loss

**Evidence.** The saved profile passes scaled-FVE tolerance at 12/12 layers, retrieval tolerance at 8/12, and unscaled-FVE tolerance at 2/12. The acceptance rule requires at least 10 retrieval matches. The canonical “not validated” verdict is correct.

The calculation in [white_metrics](/workspace/causal-concordance/scripts/012_reader_concordance/stageE0_ar_profile.py:49) fits one scalar independently per layer, on the same samples being scored. The scalars range from 0.1127607423 to 2.3344610489. With T = W(y−μ), P = W(ŷ−μ), it fits alpha = <P,T>/<P,P>. Scaled FVE equals the squared cosine between flattened target and prediction arrays. The raw-space correction is μ + alpha(ŷ−μ).

**Issue.** Interpreting this as a single global output correction or proof of the published evaluation/loss convention is too strong. One scalar per layer is a legitimate exploratory calibration; it is not an implementation error by itself. But amplitude-fitted directional agreement does not establish correct native activation scale. The observed depth trend is compatible with a scale-invariant loss, missing transforms, normalization and evaluation differences; it does not identify which one occurred.

**Consequence.** The loader fails the agreed bar, while its exact failure cause remains unknown. Matching a profile on another distribution would not uniquely identify an implementation even if all tolerances passed. The ledger's statement that the result “implies” published FVEs are scale-aligned should be a hypothesis.

**Recommended action.** Preserve the failed gate and specify “per-layer, same-sample affine calibration.” Obtain the official formula/fixture before interpreting the scale as historical training evidence. Scale alone does not invalidate normalized concept differences: a common positive scale can cancel within a layer. Directional and input-dependent error still matter.

**Confidence:** high in the calculation and gate status; low confidence in any specific explanation of the mismatch.

## 7. The J-FVE candidate comparison changes more than scale

**Issue.** E0a separately subtracts target and prediction sample means for candidates (a), (c) and (d), then reuses their E0 retrieval scores on the earlier vectors. See [centering code](/workspace/causal-concordance/scripts/012_reader_concordance/stageE0a_jfve_scaled.py:77) and [retrieval reuse](/workspace/causal-concordance/scripts/012_reader_concordance/stageE0a_jfve_scaled.py:92).

**Reason.** Positive global scaling preserves cosine retrieval; separate mean subtraction does not. It also removes a vector mean error, effectively adding an intercept adjustment. These can be complementary metrics, but do not describe the same transformed vectors and cannot be justified solely by scale invariance.

**Boundary.** Candidate (b), J(y−μ) versus J(ŷ−μ), uses the same external mean and is not directly affected. It remains the leading tested candidate: FVE matches at 8/11 layers and retrieval at 10/11, still below the 9/11 FVE requirement. The dense candidate has only three layers, so cannot attain a nine-layer threshold by construction. Its three FVE mismatches are informative only for those conditions.

**Recommended correction.** Document the exact transformation for each metric; if claiming matched FVE/retrieval geometry, recompute them on the same transformed vectors. Do not equate the full J transform with 007's orthogonal dense projector or 008's sparse component. Avoid claiming that this candidate search uniquely eliminated alternatives.

**Confidence:** high in the comparison mismatch, with limited impact on the leading candidate's current verdict.

## 8. A plausible J-FVE definition is not the AR loss or Oracle reward

**Issue.** The design says identifying J-transformed J-FVE would establish an explicit J training objective. The ledger then says the AR “was then trained” with that term. Neither inference follows from a validation-profile match alone.

**Evidence supporting the hypothesis.** Metadata names a J checkpoint, a J mean prefix, J evaluation metrics and `loss_mix_lambda=0.25`. This is substantive evidence for a J-aware setup. It should not be dismissed.

**Missing evidence.** We do not have the exact active loss branch, weighting formula or gradient computation. Evaluation metrics can be logged without entering optimization. Nor does “joint bullet-FVE” establish a reward mixing J and raw FVE: “joint” could refer to reconstructing from multiple bullets together. Even a J-trained AR could be scored using ordinary whitened FVE in the Oracle reward, creating an indirect bias distinct from an explicit J reward.

**Consequence.** The chain remains: candidate evaluation metric → active AR objective → Oracle reward → explanation of 007. Each arrow requires evidence. A confirmed J objective would weaken independence of Oracle–J agreement, but would not by itself explain 007's quantitative dense-projection effect.

**Recommended action.** Ask for the exact AR training loss, validation FVE/J-FVE and Oracle reward definitions as separate questions, plus an official forward fixture. Pending that, say: “The metadata and profile support a J-aware reconstructor hypothesis; the exact objective and reward dependence remain unverified.”

**Confidence:** high that the current evidence does not establish the chain. Whether the suspected training term was active remains unresolved.

## Related earlier 012 wording that should be corrected with the above

These issues were established in reviews 09–12 and were not newly discovered in this pass:

- **Unequal naming opportunities:** “Oracle-only 7, NLA-only 0” pools four Oracle samples across two layers against two NLA samples. At equal opportunities the counts are Oracle40-only7/NLA-only2, or Oracle44-only6/NLA-only1. It does not establish zero unique NLA coverage.
- **Non-flip is not no influence:** all 14 Oracle40 non-flipping examples that name the target have positive answer-margin changes (approximately 0.115–5.471 nats). They are not automatically false-positive detections of irrelevant content.
- **No demonstrated ensemble gain is not “adds nothing”:** the output-plus-readers AUC increment has interval [−0.02273, 0.02099]. The experiment has not demonstrated an improvement; it has not established universal redundancy or that all readers carry exactly one signal.
- **H does not isolate the cause of disagreement with 008:** its late-band answer control fails the power gate, and the comparison changes numerical regime and atom convention as well as positions. A family-only explanation and “positions ruled out” remain unsupported.

## What remains supported

The corrections do not erase intervention leverage: R has 39/146 successful J swaps versus zero random-control successes. F1 still has 35/82 successes without directly editing the read cell, versus 38/82 with all positions edited and zero with only that cell edited. This supports downstream propagation of intervention effects; it does not show the read cell is unnecessary for later computation. Readers track many successful interventions. Mid-layer AR calibration is encouraging. The correct conclusion is bounded progress with unresolved specificity, not a failed project.

## Handoff order and evidence trail

1. Correct the shared-position analyses and dependent F/R claims using existing outputs.
2. Replace the significance-based specificity rule and inaccurate coverage/ensemble language.
3. Keep AR gate failure explicit; separate validation, objective and reward hypotheses.
4. Only then choose the next experiment according to the remaining scientific uncertainty.

Supporting reports: [F2 audit 10](/workspace/codex_reports/10_012_stageF_audit_2026-10-07.md), [D/Q audit 12](/workspace/codex_reports/12_012_DQ_audit_2026-10-07.md), [integrated new-run review 13](/workspace/codex_reports/13_012_new_runs_review_2026-10-07.md), [R audit 14](/workspace/codex_reports/14_012_stageR_audit_2026-10-07.md), [AR audit 15](/workspace/codex_reports/15_012_AR_profile_audit_2026-10-07.md). These preserve exact numerical values, cohort definitions and recomputation methods. Statistics here derive from full-precision artifacts/recomputations; confidence intervals are clustered as documented there. No findings require a new experiment to establish the existing analysis errors.
