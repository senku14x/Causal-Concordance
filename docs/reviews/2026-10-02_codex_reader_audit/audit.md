# Audit of the completed Causal Concordance pilot

Prepared by Codex on 2026-10-02 for the researcher and the CC instance implementing the project.

**Assessment: the completed runs contain useful pilot observations, but the report overstates several conclusions. They do not yet establish which readers faithfully identify causally influential content.** The immediate priority is to improve the scoring and interpretation of the existing reader outputs before expanding the causal benchmark.

This audit inspected the raw reader descriptions, probe results, intervention records, analysis code, and finished report. It also recomputed descriptive comparisons from saved outputs. No model inference, training, or intervention runs were launched, and the project repository was not edited.

**Scope and completed work**

The model is Qwen3.6-27B. Intervention measurements use SEC replay: stock bf16 through block 39, true float32 from block 40. Layer numbers denote zero-based decoder-block outputs. The scientific measurements reviewed here are exploratory dev-set results; admission checks also covered test candidates.

| Component | Completed scope | What it supports |
|---|---|---|
| Engine and numerical controls | Self-patch, embedding anchor, prefix identity, and numerical-noise checks | Confidence in the tested implementation invariants |
| Whole-cell interventions | Brew: 82 forward pairs from 33 tables; chain: 48 pairs from 9 step-sequence groups | Influence of the combined natural pair difference at a cell |
| Decodability probes | 231 natural brew prompts from 48 tables; grouped cross-validation | Held-out linear decodability under the fitted probe procedure |
| Reader snapshot | 10 brew tables, three emission positions, original/partner/substituted prompts, real/shuffled/no-injection arms | Preliminary reader responses to controlled activation changes |
| Content-specific causal adjudication | Not completed | No validated used/necessary labels or causal reader rankings yet |

The saved reader protocol has NLA at L42, Oracle on the L20–60 step-four grid, and four-sample Oracle runs at L40/L44. It does not contain an Oracle L42 arm. The new same-L42 preference discussed with the researcher has therefore not been evaluated in these outputs.

**1. Colour mentions are being interpreted too strongly**

The snapshot scorer checks whether a colour occurs anywhere in a description. It does not establish that the reader assigns that colour the correct role, endorses it, or describes the computation correctly. It also compares long verbalizer descriptions with top-ten token lists, without the original spec's matched claim budgets or full closed-set scoring.

At the final colon, NLA L42 mentions the true intermediate in 17/40 samples, compared with 8/40 shuffled-activation samples. There are four samples on each of ten tables, not forty independent problems. J top-ten outputs name none of the true intermediate colours at that cell on these ten tables.

The distinction between mention and interpretation is substantive. In table `brew-s2-r10d2-0104`, the actual trajectory is:

> orange → green → black

One counted NLA hit instead describes:

> Green → Yellow → Red

It includes the correct intermediate's name while assigning it the input role and inventing the remaining trajectory. All 17 NLA explanations counted as intermediate mentions at this cell were inspected; the count cannot be presented as 17 correct intermediate explanations.

The report's statement that the disagreement holds on all ten tables is also too broad: only six tables have any intermediate mention among the four NLA colon samples. And the statement that J names no colour should be restricted to the scored palette: J_NP cosine includes `YELLOW` on table `0069` at L42, but yellow is absent from the scorer's twelve-colour dictionary. This does not change the true-intermediate hit count.

Source: [frozen raw reader outputs](evidence/results/001_leverage_profile/analysis/snapshot.json), [scoring implementation](evidence/scripts/001_leverage_profile/leverage_profile.py), and [original report section 8](evidence/docs/reports/001_leverage_profile.md).

**2. A useful activation-sensitive signal survives a stronger descriptive comparison**

The report emphasizes real-versus-shuffled marginal naming rates. A more discriminative existing contrast keeps the table fixed and changes its starting colour, which changes the correct intermediate. This audit computed, separately for each table:

`tracking = [(member mentions old z − partner mentions old z) + (partner mentions new z − member mentions new z)] / 2`

Each mention rate averages the four descriptions available for that condition. A positive value means mentions shift toward the intermediate appropriate to the changed input. Both intermediates remain present in the same rule table.

| Reader and cell | Mean tracking score | Positive tables | Negative tables | Ties |
|---|---:|---:|---:|---:|
| NLA L42, assistant newline | 0.3500 | 8 | 1 | 1 |
| NLA L42, end-of-thinking newline | 0.3000 | 6 | 1 | 3 |
| NLA L42, final colon | 0.1625 | 7 | 1 | 2 |
| Oracle L40, final colon | 0.1375 | 4 | 1 | 5 |
| Oracle L44, final colon | 0.2375 | 9 | 0 | 1 |

This is evidence of some activation-sensitive, item-specific content. It does not establish faithful role assignment or causal use. Oracle and NLA here also read different layers, so this is not a matched-layer ranking. The contrast was selected during this audit and is exploratory.

The result is compatible with readers recovering part of a relevant signal and elaborating it into an inaccurate narrative. It argues against dismissing all outputs as activation-independent stories.

Source: [audit results](reader_audit_results.json), including per-table values and exploratory uncertainty calculations.

**3. Reader agreement itself has not yet been adequately measured**

The snapshot summary reports marginal naming rates, not matched atomic claims or a full agreement analysis. The outputs already suggest overlap at one level: J often lists words meaning second, cycle, and again, while verbalizers describe an iterative transformation. They can agree about task structure while disagreeing about entities, roles, and specific transitions.

Thus the current output does not establish that the readers recover the same mechanism, or that a reader naming more colours is the more faithful reader. The next analysis should distinguish task structure, entity identity, and entity role.

**4. Whole-cell effects do not adjudicate the named intermediate**

The full-cell transplant changes all dimensions of the state at a selected token and layer. In the start-pair construction, the input, intermediate, answer, and other downstream state differences co-vary. A large patch effect cannot determine which of these accounts for the effect.

Combining a strong intermediate probe with a strong whole-cell patch establishes two observations: the intermediate is decodable, and the combined paired state difference can affect the answer. It does not establish that the intermediate caused the answer change. A reader-blind mean-difference direction will not automatically remove this confounding either; content specificity still needs an independent test.

Likewise, weak answer-probe performance at L42 does not establish that the answer is absent or has not yet been computed. The report should say that the tested linear probe decodes it less accurately there.

At the final colon, the saved probe results are:

| Probe target | L42 | L60 |
|---|---:|---:|
| Starting colour, twelve-way balanced accuracy | 0.805 | 0.529 |
| Intermediate colour, twelve-way balanced accuracy | 0.757 | 0.678 |
| Final answer, twelve-way balanced accuracy | 0.174 | 0.832 |
| Off-trajectory membership, mean AUC | 0.619 | 0.763 |

Chance is approximately 0.083 for the balanced accuracies and 0.5 for AUC. Off-trajectory membership passes the existing significance-plus-selectivity rule only at L60's final colon and end-of-thinking newline. L60 success does not establish a validated L42 not-needed class. A failed L42 criterion is also not evidence of absence: the substitution-direction check provides some evidence of a weaker signal there.

Source: [frozen probe results](evidence/results/001_leverage_profile/analysis/probes.json) and [pilot summary](evidence/results/001_leverage_profile/analysis/summary.json).

**5. The reconstruction conclusion is too categorical**

The report concludes that FVE reflects the shared template rather than item-specific content because own-cell and shuffled-cell FVE are close. Close means do not establish zero discriminative information.

For the same shuffled explanation, this audit compared its FVE against its actual activation source and against the incorrect target cell. The actual source wins in **244/360 comparisons**, with a mean FVE advantage of **0.0157**. Seven of the ten target-table averages are positive. The observations share tables, roles, positions, and sampled explanations; 360 is not an independent sample size, and the fixed shuffle also reuses source tables.

The supported conclusion is that reconstruction provides a small source-preference signal on these examples. Its usefulness for faithful claim verification or best-of-N selection remains untested. It is not justified to conclude that reconstruction has no item-specific information.

Source: the `fve` section of [audit results](reader_audit_results.json).

**6. The damage explanation is incomplete**

At the brew final colon, L42, 55/82 forward patches exceed the registered off-pair-mass increase threshold. The report suggests this is less concerning because probability moves to intermediate colours. However, its decomposition uses only the top five tokens.

Recomputing that decomposition at L44 gives:

| Quantity | Mean probability mass |
|---|---:|
| All saved top-five tokens combined | 0.7721 |
| Intermediate colours within the saved top five | 0.0909 |
| Recorded total off-pair mass | 0.3447 |

About 0.228 of total probability lies outside the saved top-five list. These displayed quantities do not establish that most total off-pair mass is on intermediates. A claim about the increase in intermediate mass additionally requires a corresponding clean-versus-edited decomposition, which this table does not supply.

The implementation's off-pair statistic uses next-token mass outside the accepted answers' first-token sets; it is not a full-sequence answer-distribution statistic. These are reasons to improve the diagnostic, not reasons to waive the registered flag. Probability on the wrong intermediate can still indicate failure of selective transfer, even if outputs remain grammatical and colour-related.

Source: `damage_L44` in [audit results](reader_audit_results.json) and [damage implementation](evidence/scripts/common/regime.py).

**7. Statistical and reporting issues to repair**

- The report places intervals for table-level means beside pair-level medians. These are different estimands; report each with its own uncertainty or identify them separately.
- Chain's analysis clusters by item, although its independent split unit is the step-sequence group. There are nine groups, not twelve independent items.
- Real and control probes tune regularization through inner cross-validation. Permutation probes reuse regularization chosen from real labels. The permutation distribution therefore does not reproduce the full fitting procedure; its significance claims need checking before use as a label gate.
- Some membership random-label controls are unusually strong, reaching AUC 0.843 in one condition. With many correlated checks this does not prove leakage. It does require diagnosis rather than a blanket statement that the controls sit at chance; the report's low control range applies to the twelve-way tasks.
- No detected substitution tracking on ten tables is an exploratory null, not proof that readers cannot report the content. The stronger probe result at L60 cannot establish the sensitivity of the L42 verbalizers.
- No prompt-only reader comparison, matched claim-budget evaluation, or content-specific causal reader ranking was completed in this pilot.

**Recommended next action**

Re-score the existing original/partner descriptions for the colour they actually identify as the intermediate, with ambiguity and abstention allowed. Keep separate scores for simple mention, correct role assignment, and contradicted claims. Compare changes within the same table and use comparable claim budgets before ranking readers. Record this as an exploratory scoring refinement; freeze it before held-out evaluation.

In parallel, correct the report's causal and null-result wording and audit the probe controls. Then select one concrete reader disagreement for a content-specific intervention that distinguishes intermediate information from answer information. J-coordinate swaps or downstream clamps can be experimental arms, while the reference labels remain independent of J.

The defensible current statement is: **on this small brew pilot, readers share task-structure information, show some activation-sensitive colour signal, and often surround it with incorrect detail. Whether their claims track causally influential content remains unresolved.**

**Reproduction and provenance**

The [analysis script](reader_audit.py) reads only the frozen evidence shipped with this report and writes [reader_audit_results.json](reader_audit_results.json). Run:

```bash
python /workspace/codex_reports/reader_audit.py
```

Dependencies are Python and NumPy. The script reproduces the within-table mention contrasts, real-versus-shuffled contrasts, FVE source comparison, and top-five probability decomposition. Qualitative role judgments and the broader methodological review are author assessments documented above, not outputs of an automated judge.

The JSON also includes table-bootstrap intervals and unadjusted sign-flip diagnostics. These are exploratory, not registered confirmatory tests. Ten tables provide limited uncertainty resolution. The fixed activation derangement links tables, so the simple per-table intervals for shuffled comparisons do not fully account for that dependence. No claim here relies on crossing a significance threshold in those diagnostics.

[source_manifest.json](source_manifest.json) records the original paths, SHA256 hashes, capture time, and project commit. The frozen evidence includes the raw reader snapshot, probe results, intervention records needed by the script, relevant code, and the report being audited. These copies preserve the reviewed state if the working project changes later.
