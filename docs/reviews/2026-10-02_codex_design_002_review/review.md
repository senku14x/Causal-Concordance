# Review of design 002 and the overnight work

Reviewed 2026-10-02, around 12:42 UTC. Scope: design 002 v2, its newly written scoring code and validation artifacts, and design 001 amendment A2 / repaired probe code. The repository is being edited concurrently. Design v2 and author codes were committed as `1a8fc59`; `claim_scoring.py` was still being developed. Source hashes and verified counts are in [the checks JSON](2026-10-02_design_002_checks.json). [Reproduction script](review_design_002.py) makes no model or API calls.

**Current decision, updated at 12:46 UTC: proceed with the corrected v2.1 data collection; fix variant truth keys and error categories before reporting scores, and omit or redesign the censored cross-table role control.** Claude independently caught the missing partner candidates during this review and committed `2e03624`, expanding each table's set to include both trajectories. I verified that all member/partner role targets are now included. Its partial five-candidate run was archived. There is no reason to rerun those corrected generations merely because the remaining analysis fixes are pending. No causal labels are established by this experiment.

**Handoff for Claude — still open at the recheck:**

1. `validation/key.jsonl` still has 25 partner rows with the original trajectory (20 validation, 5 pilot). Repair before any truth-dependent score.
2. Under v2.1 every correct trajectory colour is in the set. An explicit non-candidate role value (`none`) is therefore an incorrect explicit claim, not abstention. This can be fixed in analysis without new generations; keep conflicts separate.
3. The role-based own-z versus other-tables' z control still excludes **45/90** other-table target comparisons from its allowed answers. Do not interpret that contrast as evidence against colour preferences. Keep the unrestricted mention-based control for this run, or redesign extraction before a later role-based control.
4. Validate the judge before using scores; add version-aware cache checks before any further schema/prompt change. The current manual archival of the five-candidate run avoids mixing that particular run with v2.1, but the cache itself still keys only on readout ID.

The findings below preserve what was inspected in v2; resolution notes distinguish fixes already made from outstanding work. These are concrete scoring issues, not a reason to abandon the project.

I have not modified experiment code, stopped processes, or launched model/API runs.

## 1. Partner targets excluded from allowed answers — fixed for paired tracking in v2.1

**Resolution:** v2.1 uses one common union of both pair paths and the original off-path options. No member/partner role target is excluded in the ten tables. The counts below describe the superseded v2 rule. The cross-table control remains censored (45/90 targets), because other tables' intermediates are not necessarily in this union.

Design §2 / §3 gives every variant the original item's five candidates. `make_sets.py:58` and `claim_scoring.py:86` implement exactly that. The JSON schema prohibits every other colour.

Across the ten saved tables, the original candidate set omits:

| Partner role | Tables with the correct colour excluded |
|---|---:|
| Start | 4/10 |
| Intermediate | **8/10** |
| Answer | 6/10 |

Example: `brew-s2-r10d2-0009` has candidates `[red, purple, pink, white, gold]`. Its original trajectory is `gold → red → purple`; its partner is **`black → orange → green`**. The judge cannot emit any of the partner's three correct role values. A perfectly faithful explanation of the partner is forced to return `none` for all three.

This invalidates the proposed symmetric start-swap tracking statistic: its partner-to-new-intermediate term is structurally censored for eight tables. It also compromises the cross-table colour-preference contrast whenever another table's intermediate is outside the recipient's candidate set.

**Fix:** separate extracting a reader's claims from evaluating them against task options. Prefer extraction of explicit colour names and supporting text spans, normalized afterward, with an explicit value for an out-of-palette colour. A fixed common colour vocabulary with an `other_explicit_colour` field also works for this pilot. Do not condition extraction on the original five options.

Keep the registered five-candidate closed-set task as a separate analysis, with candidates rebuilt for each actual render and an assertion that start, intermediate and answer are included. For symmetric tracking, both original and partner targets must be scoreable under the same extraction rule on both readouts; merely switching to five different options for each arm does not guarantee this. Record the scoring change in the design.

**Tiny acceptance check before another full judge pass:** feed three explicit chains, including `black → orange → green` for table 0009, through the extraction-to-scoring path. All three roles must survive extraction, and partner accuracy must be computed against the partner trajectory. Include a member readout that explicitly names the partner's intermediate, to test both cross-terms in tracking.

## 2. Blocking if used for correctness: the validation truth key ignores the variant

`make_sets.py` writes `start`, `z`, and `answer` from the original table for every row, including `partner` rows.

Verified in the saved `key.jsonl`: **25 partner rows have all three truth values wrong**, comprising 5 pilot and 20 validation rows. This does not itself invalidate blind judge-versus-author agreement, which compares extracted claims. It does invalidate any role-correctness/confusion calculation that uses this key.

**Fix:** resolve truth by `(table, variant)`: member and off-path-substituted prompts retain the original trajectory; partner prompts use `t['partner']`. Shuffled readouts are scored against the recipient variant for the primary shuffled control. If donor fidelity is also examined, use a separate, explicitly named donor truth field. No-activation readouts need an explicit recipient assignment when scored.

Regenerate the key and test it against snapshot metadata. Author codes made under the censored five-option schema need revisiting if the extraction schema expands; they cannot recover colours already collapsed into `none`.

## 3. Blocking for error-rate interpretation: an explicit wrong claim becomes an abstention

The prompt says that a role assigned a colour outside the five candidates becomes `none`. Design §6 then treats `none` as an abstention; only another in-set candidate counts as contradicted.

Thus “the intermediate is yellow” is counted as abstention even if the correct intermediate is green. Conversely, a correct partner intermediate outside the original options is also abstention. This conflates a reader declining to claim something, an explicit false claim, and a censored true claim.

**Preferred general fix:** preserve the explicit colour and score it after extraction. **Minimal fix for the already running v2.1:** because all correct role values are now candidates, count `none` as an explicit out-of-set error, not an abstention. No new generation is required for that aggregate distinction. Exact shared wrong-colour identities remain unavailable for these rows. Use distinct outcomes for:

- a correct asserted role value;
- an incorrect asserted role value, including out-of-set values;
- no role asserted / cannot determine a role;
- multiple incompatible assertions.

Conflicts can remain their own reported category. Do not silently treat them as the reader voluntarily abstaining. Keep generic examples, negations, and considered alternatives distinguishable from assertions about this item where the wording supports the distinction. Put a few such cases in the small parser/judge check set.

## 4. Fix before a corrected rerun: cache entries are not tied to the scoring version

`claim_scoring.py:_run` resumes by readout ID alone. Successful rows do not record the prompt hash or text hash. The final manifest records the *current* prompt hash. A rerun after editing the prompt, schema or candidates can therefore silently retain old scores while documenting the new prompt.

**Fix:** key/reject cached rows using the readout text hash, candidate/schema version, prompt hash and judge settings, or use immutable versioned output directories. Preserve old runs as superseded. The pilot currently deletes `pilot_claims.jsonl` on each invocation; archive each pilot pass instead, so prompt-development history remains reviewable. Include pilot calls and retries when reporting total spend.

The run is inexpensive, so this is primarily a provenance and correctness problem, not a compute problem.

## 5. Validate before judging the full grid

The written procedure judges all 3,930 outputs before assessing the 60-row validation set. Run the frozen judge on those 60 rows first, examine extraction disagreements, and only then run the grid if the scorer is adequate. Show role-specific agreement and the rate of explicit colour assignments; a mostly-abstaining validation sample does not adequately validate positive claims by itself.

Claude's blind codes are useful independent annotations, but they are another model's annotations, not human ground truth. The promised PI spot-check remains useful when the PI returns. A disagreement should prompt inspection of the text, not automatic deference to either model. If validation feedback changes the prompt, treat that set as development material and draw fresh validation examples for the changed procedure.

## 6. Important interpretation fixes; these do not block collecting outputs

- **Bridge attribution:** design §9 says role agreement below 0.8 between stock and SEC readouts flags judge noise. These are separately generated reader texts. Disagreement could come from the reader, the small activation change, or the judge. Call it end-to-end bridge disagreement. Rejudging identical text isolates judge instability; comparing separately generated texts does not.
- **Reader comparison:** putting a lens's top candidate beside a verbalizer's correctly assigned intermediate is informative, but these remain different targets. “One output per role” does not erase differences in how much text or how many claims the verbalizer produced. Report entity detection and role attribution separately; do not present this pilot as a fully matched reader ranking.
- **Actual overlap:** §6 lists within-reader contrasts, but no explicit reader-pair overlap table. Add an exploratory paired table on common items/cells: both detect the entity, only A, only B, neither. For text readers, separately compare asserted role values and shared incorrect values. Keep Oracle L40/L44 versus NLA L42 comparisons labelled as different-layer comparisons. This addresses the user's overlap question without requiring causal labels.
- **Prompt-only:** the proposed full-problem baseline measures prompt solvability. It is not a mathematical ceiling, nor does it isolate whether a reader internally guessed from its own priors. Report it as a task baseline alongside activation controls. Compare reader and baseline results on the same ten tables as well as any broader baseline summary. For a later pre-start-cell comparison, the baseline must be limited to the prefix visible at that cell.
- **Sample scope:** many samples, layers and variants still come from ten independent tables. Keep this dev scoring experiment descriptive; do not interpret thousands of judged readouts as thousands of independent task examples.

## 7. Probe amendment A2: useful repairs, with residual checks

A2 fixes three specific problems: one unlucky control draw shared across fits, pooling incomparable fold decision scores, and failing to retune C under the null. Those changes are worth running. A high single control AUC was not, by itself, proof of leakage; the fresh-control diagnostic appropriately investigates that possibility.

Two limitations still need attention before using the results to certify labels:

1. The accelerated span solver is mathematically reasonable, but the amendment reports only 93.5% held-out prediction agreement with the original solver at C=1. Existing tests exercise C=.001 and .01 only. Inspect convergence warnings and, on a few real folds spanning the selected C values, compare objective values and predictions. A changed numerical estimator is acceptable if it is stable and consistently applied; the current evidence does not establish equivalence across the whole C grid.
2. The 12-way permutation null still shuffles labels across individual renders, whereas the data and splits are grouped by table. Applying the same training pipeline to real and permuted data fixes tuning asymmetry; it does not establish the exchangeability of that row-level shuffle. Before treating its p-values as calibrated, justify the null or test it on synthetic table-correlated data and compare with an appropriate grouped randomization. The binary membership null uses a joint table permutation, which is a different procedure. Also report how many randomizations fail class-support checks rather than hiding the reduced effective null sample.

These are reasons to qualify the probe inference and make small diagnostic checks, not to discard its descriptive predictions or interrupt an otherwise useful run.

## Suggested overnight sequence

1. Candidate handling for paired tracking is now fixed. Correct variant truth and error categories; omit/redesign the censored cross-table role control. Add the small end-to-end scoring checks and version-aware caching for future changes.
2. Finish the existing probe repair and record its remaining numerical/null limitations.
3. Run a frozen judge validation pass, then the corrected descriptive reader scoring, candidate ranks, and prompt-only baseline.
4. Produce the common-cell overlap tables and keep qualitative examples beside them.
5. Leave any claimed causal ranking, W estimate, and expansion to test-split causal labels for the separately designed label experiment. 002 cannot establish those.

This is a worthwhile overnight block once the scoring bugs are repaired. Its strongest potential result is a reliable description of what readers say, which claims follow activation changes, and where their outputs overlap. Which named content causally drives the answer remains a subsequent question.
