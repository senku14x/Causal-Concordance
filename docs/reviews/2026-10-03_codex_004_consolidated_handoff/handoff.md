# Consolidated handoff for experiment 004

Prepared 2026-10-03 against Causal-Concordance commit `a2ef513`, with a clean working tree at inspection. This reconciles the two supplied reviews with the earlier Codex review and the current code. It is the recommended scope for the next design revision, not a claim that implementation or model runs have occurred.

**Revise 004 and run a bounded primary diagnostic.** Keep 33 dev tables, the final-colon L42 cell, the separate start-token L40 reference, the existing estimator, associated spans and directly measured complements. Repair paired controls and interpretation. Add a small PCA comparison, not an expanded experiment containing every suggested control. Assign no labels and rank no readers against these interventions.

## Project decision

The primary project now compares native reader configurations: useful task information, errors, overlap, and complementarity. Causal experiments are a focused validation branch. State this priority in the current amendment/status documents. Numbering 004 and 005 does not authorize a return to making a complete causal label bank the prerequisite for reader comparison.

If 004 is successful, it identifies a promising intervention construction for further specificity and held-out validation. It does not automatically make z-span labels valid. If it is unsuccessful or inconclusive, retain the descriptive findings and proceed with task-level reader evaluation rather than automatically adding another estimator, layer sweep, or task family.

## Before expensive work

1. **Verify the raw-output mirror first.** The local handoff still says `outputs/` exists only on this nonpersistent instance and the destination is pending. Verify the actual off-instance state rather than assuming that documentation is current. Use the already authorized private destination, record a file manifest with sizes and hashes and the corresponding code/registry versions, and verify retrieval of sample artifacts. A local archive or Git commit of summaries is not an off-instance mirror. If the destination remains unresolved, report that specifically while continuing CPU preparation; do not invent a destination or claim the mirror is complete.
2. Preserve 001–003 raw outputs and historical designs. Add dated corrections instead of retrospectively rewriting what was registered. Regenerated analyses need versioned provenance and a comparison against the prior summaries.
3. Make the 004 design, arm manifest, scoring contract, and CPU checks concrete before its GPU smoke. Follow the existing design-commit and PI launch instruction workflow. The handoff itself does not launch anything.

## Scope and intervention set

- Seed-select one of the existing 82 gated pairs per table: 33 pairs. Freeze the exact manifest, seed, pair identities, and exclusions before 004 results. Primary summaries weight tables equally. Shared fitted coefficients remain a source of dependence; intervals conditional on these fits do not include all fitting uncertainty.
- Primary: output of decoder block 42 at the final colon, same SEC regime as 003. Patch once at this cell and allow downstream computation to run normally. Reapply the same specified intervention when teacher-forcing each answer continuation for scoring; this is not a clamp across downstream layers.
- Reference: start-colour token, L40, using locally fitted start-associated directions. It tests recovery at a site with known whole-cell leverage. It is not guaranteed that an additive start span should recover all of that leverage.
- Keep the original two-fold primary fit and the original two disjoint one-fold stress fits. Each excludes the evaluation table. Keep the estimator unchanged so the first comparison isolates expansion of its span.
- At the primary cell measure: z 1-D, z-associated span Z, joint start/z/answer span J, Z complement, J complement, whole-cell replacement. Add start-only and answer-only span swaps as descriptive comparisons, under the primary and the two stress fits. These inexpensive comparisons identify whether a successful joint span could be explained by the answer span alone.
- At the start reference measure start 1-D, start span, its complement, and whole-cell replacement, with disjoint-fit repeats and the corresponding controls/clamps below. Do not silently expand this into a second full three-role survey.
- Defer L44, secondary pairs, dose sweeps, and extra reader runs. If a later L44 extension is justified, fit L44 spans rather than transporting L42 directions without a separate validation.

Call Z the **fitted intermediate-associated span**. Reader independence and additive regression adjustment do not establish semantic isolation. The cached L42 fits already have smallest z/start principal angles of 34–36 degrees and z/answer angles of 30–38 degrees. These are geometric proximity measurements, not evidence of semantic equivalence or exact shared dimensions.

## Controls to implement

### Existing orientation and rank controls

Keep the random rank-matched projection and random edit-norm-matched direction controls for the central swap arms. Keep 16 draws as descriptive reference distributions. Keep directly measured random complements for the joint complement; scope any inference about the Z complement to the controls actually run. Save all draws, edit sizes, and seeds.

The objection to relying only on ambient random directions is valid: a fitted span may outperform them merely because it follows variation in the data. The statement that it must win is too strong. Also, k/5120 is the expected fraction of **squared norm** captured by a uniform random rank-k projector, not the norm fraction or the behavioral effect. For k=11 it is about 0.00215; the corresponding RMS norm fraction is about 0.0464.

### Add a fitted PCA comparator

Use a PCA basis built from the fitting activations at the same cell, with **within-table centering** to target the variation relevant to these within-table pairs. Give tables equal weight in the covariance: weight each centered row by 1/sqrt(number of fitting rows in its table). Fit independently for the primary and each stress fit, without evaluation-table activations or outcomes.

For each required rank k_Z or k_J, take the top-k PCA directions and measure their donor-coordinate swap. Also measure a dose-matched version pointing along that PCA-projected donor difference, rescaled to the corresponding content edit's norm. The rescaled edit is a perturbation control, not a natural donor-coordinate patch. If the projected difference is numerically zero, record the control as undefined instead of dividing by zero. Record actual ranks and norms.

One PCA basis gives nested ranks. **Do not create sixteen copies of deterministic PCA or duplicate a full null grid for this comparator.** Apply the same small comparison to the start span at the start reference. Full control repeats around the added start-only and answer-only swaps at the primary cell are not needed for their descriptive role comparison.

A PCA result equal to or larger than Z weakens the claim that the label-associated construction offers anything specific beyond a general subspace aligned with the data. Beating PCA is a useful discriminator, not proof that Z isolates the intermediate. Compare unscaled and dose-matched results so a difference in edit magnitude is visible.

Defer label-shuffled regression spans. They are a sensible possible follow-up, but a valid construction must define how labels, unequal table sizes, role associations, and fitting support are preserved or broken. Adding sixteen arbitrary shuffles now is not required to answer the bounded diagnostic. Revisit them if a stable Z advantage over PCA makes semantic specificity the next binding uncertainty.

### Replace the paired ablation size control

For content basis Q with orthonormal columns and that fit's mean mu, set

    a_x = Q^T (h_x - mu)
    real_delta_x = -Q a_x

For each random draw sample a Haar-uniform orthonormal d-by-k basis U (for example, Gaussian QR with the diagonal-sign correction) and use

    control_delta_x = -U a_x

for both member and partner with the **same U**. This preserves both edit norms, their mutual inner product, and the norm of their difference. It is preferable to merely using the same random direction with two positive magnitudes, which can destroy the opposing signs of a contrast-removing edit. This control randomizes orientation; it is not itself a clamp to a random subspace mean.

Keep the separate random-subspace-to-own-mean clamp as a second comparison. Use each fit's own fitting-set mean. The inherited `Directions.fitmean` hardcodes `both` and must not be reused unchanged for r1/r2.

Freeze U and the intervention identity across all teacher-forced continuation forwards. Matching sequence-length sources remains required. Use 16 controls per type for the primary fits. For disjoint stress fits repeat the real swaps, complements, and clamps; do not imply that a primary-fit null calibrates the stress fits, and do not replicate every control grid unless separately budgeted. Compare controls using N for ablations and S for swaps, not a mixture of endpoints.

## Scoring and specificity

Keep the existing full-sequence LSE answer margin and clean whole-run counterfactual gap G. Recompute whole-cell patches under the same scoring path. Save raw margins, Delta m, G, and signed shares; do not replace them with only recovery ratios or pass flags.

For every scored prompt and edit, save all twelve raw accepted-spelling masses q(c), their total M, and optionally q(c)/M. These are accepted spelling-prefix events under the existing scorer; no answer terminator is included. Verify token-sequence events are disjoint before using 1-M, and call it mass outside the accepted spelling set. Keep next-token diagnostics separate from full-sequence quantities.

**Continuation trap:** 003 captured donor continuations only for the two endpoint answers. All-colour scoring needs donor captures for every multi-token colour spelling, including ` Teal` even when neither endpoint is teal. Use matching sequence lengths and the same random draw throughout.

For swaps, report where positive colour-mass gains go. Let

    d_c = q_edit(c) - q_clean(c)
    T_positive = sum_c max(d_c, 0)
    F_target = max(d_a_prime, 0) / T_positive

Treat F_target as a composition diagnostic only when T_positive >= 0.01; otherwise mark it as too little transferred mass for this diagnostic. Save the raw changes in all cases. Report gains to the intended answer, donor intermediate, donor start, and other colours. When roles name the same colour, show those overlapping labels without double-counting mass. This describes redistribution of measured answer probability, not literal causal transport of probability mass.

Do not automatically require most gains to land on a_prime merely to acknowledge a margin effect. Use that requirement for a separately named **target-directed recovery qualification**: F_target > 0.5, an increase in donor answer probability beyond the measured numerical floor, and no relevant damage flag. This prevents a margin improvement driven by suppressing the original answer or promoting the donor's intermediate from being sold as clean answer transfer.

For this dev diagnostic, preregister two warning flags:

- mass outside accepted spellings increases by more than 0.05;
- on a swap, both endpoint log probabilities decrease by more than 0.05 nats.

These are warnings about this answer distribution, not established tests of broad capability damage. Preserve the existing non-answer retention diagnostic where it has positions to evaluate; at the final colon it may be unavailable. Log both ablated prompts separately. Loss of the currently correct answer during an ablation is expected and is not by itself evidence of generic damage.

Report all-pair effects, flagged cases, and qualified effects separately. **Do not silently drop flagged rows from the headline median or denominator.** Damaged or misdirected effects cannot support an unqualified clean-recovery conclusion, even when their margin share is large.

## Reliability and statistical interpretation

- The minimum plus-one rank statistic with 16 draws is 1/17. Save effects, empirical exceedance counts, and the rank statistic, but call these descriptive random-orientation comparisons. No per-item significance labels, pooled p-values, FDR labels, or claim that these controls define the entire null of no causal content. Exceeding empirical q95 is an exploratory heuristic.
- Define the recovery subset once using remeasured S_full >= 0.10. State its size, report signed S for every pair, and do not clip recovery ratios above 1 or below 0. Specify each branch's denominator; control and damage fractions should retain the full primary sample with missing cases counted explicitly.
- Compare Z against remeasured 1-D using paired Delta S, not only separate medians. Likewise report paired differences against unscaled and dose-matched PCA. Give table-level intervals and all-pair scatter plots.
- Report correlation, signed bias, median absolute cross-fit difference, and the effect range for both S and N. Use **0.05 share units as the proposed dev material-disagreement tolerance**, frozen before the run. This is an operational threshold, not a universal reliability standard. Undefined or low correlation among nearly constant, agreeing effects is not automatic failure.
- Retain the half-sized fits as a low-data stress test. To describe the high-recovery result as reproducible under that stress test, require the substantive recovery conclusion in both disjoint fits as well as material absolute agreement. A failure under those smaller fits is not a direct estimate of the larger primary fit's reliability.
- Report signed I = S_span + S_complement - S_full, median |I|, and the fraction with |I| >= 0.15, with uncertainty and sign information. The interaction is nonadditivity for these interventions and this margin endpoint.

## Replace the branch interpretations

Treat branches as descriptions that can coexist, not a mandatory five-way diagnosis.

| Observation | Licensed interpretation and next action |
|---|---|
| Z recovery >= 0.5, with positive paired improvement over 1-D, reproduced under both stress fits and material absolute agreement | The one-direction construction misses influence recoverable by the larger associated span. Report control and specificity qualifiers before choosing further validation. |
| The above also exceeds random comparators and PCA, with predominantly target-directed, unflagged effects | Promising candidate for further content-specific and held-out validation. No automatic causal labels. |
| Joint recovery >= 0.5 while Z recovery < 0.25 | Influence is recovered by the broader fitted span but not isolated by the current Z estimator. Check answer-only/start-only effects, controls, and reproducibility before explaining why. |
| Large signed or heterogeneous absolute interaction | Additive attribution is inadequate for the affected pairs and endpoint. Report the directly measured effects rather than assigning residual causal credit. |
| Material disagreement between disjoint-fit effects | The present fitted intervention is not reproducible under the low-data stress test. Larger fitting support or regularization is a candidate follow-up, not a confirmed remedy. |
| Poor start-span recovery despite a stable full-cell reference | The fitted span poorly recovers leverage at that site. This alone does not identify instability, implementation failure, or nonlinear encoding. |
| Content ablation does not clearly outperform the paired rotated controls | Specificity is not established. This is not proof that the real effect is generic. |
| No clear pattern or thresholds not met | Inconclusive. Return to the reader panel unless a bounded follow-up would change the decision. |

Use the original 0.5/0.25 recovery and 0.15 interaction thresholds as dev heuristics, not significance cutoffs. Show the continuous paired evidence and uncertainty beside every flag. A stable larger effect without target direction or an advantage over PCA remains a descriptive geometry result. Neither failure nor a small complement effect establishes absence or natural non-use.

## Preflight and implementation contract

1. Freeze eligible-colour rules and rank tolerance. Record both render counts and distinct-table support per role/colour. Keep the original >=2-render rule initially to isolate the intervention change; flag weak table support rather than silently changing the estimator. If support is inadequate for a requested contrast, mark it missing and preserve the denominator. Record rank and eligible-set differences across fits.
2. Use float64 construction of orthonormal bases with a declared SVD tolerance; the existing geometry audit used relative cutoff 1e-6. Measure numerical identities after conversion to the actual write dtype as well. Test Q^T Q, projector idempotence, 1-D containment where applicable, and span-plus-complement reconstruction.
3. Handle zero or numerically negligible requested writes with explicit no-op/read-back checks. rho and cosine are undefined at exactly zero. Define the negligible-write handling relative to floating-point resolution; do not turn arbitrary large errors into no-ops.
4. Keep exact self-swap checks for genuinely identical code paths. Replace bitwise equality of a general full-rank projector and donor assignment with declared state and downstream tolerances. A special-case direct assignment can be bitwise, but does not test generic projection arithmetic. Set tolerances from numerical preflight/noise measurements, not from whichever result would make a later gate pass.
5. Extend the reader/label import-boundary check to the actual 004 entry point and transitive helper modules. Do not inherit reader code through the mixed 001 module just to reuse material helpers; factor the minimal reader-free helpers needed. This is dependency hygiene, not evidence that earlier label calculations actually invoked a reader.
6. Keep outputs under 004 with complete manifests: source hashes, pair selection, fit membership, colour support, means, bases, ranks, seeds, control construction, scoring spellings, model pins, and code version. Do not overwrite 001–003 artifacts.
7. Enumerate the arm list before timing. Count intervention conditions, scored prompts, and continuation/upper-stack forwards separately, including both members of every ablation. Do not reuse the draft's 60-run start-reference allowance or its 30-minute estimate without measuring the complete two-pair smoke. All-colour scoring requires the multi-token continuation on every scored evaluation.
8. Inspect those two pairs end to end, including colour distributions and write fidelity, before running the remaining primary tables. Log intermediate outputs and retain failures. Include fixed-seed random qualitative examples alongside median, extreme, and failed cases in the final report.

## Suggestions deferred or rejected

**Do not add L2 to the primary run.** Earlier whole-cell L0–L4 effects around 1 are real in the saved report, but they do not guarantee that a fitted additive identity span must work at L2. Failure would not by itself prove the method is broken, and early success plus late failure would not uniquely identify transformed content. More practically, the current true-fp32 replay begins above block 39; early edits introduce the lower-stack numerical regime, whose saved noise measurements are much larger. Establishing a comparable early assay is additional work, not a free positive control. Keep L40 and revisit an early or planted control only if its result would resolve a remaining ambiguity.

**Do not add leave-one-table-out 23/24-table stress fits yet.** This is a reasonable conditional learning-curve experiment, but it changes fit construction and requires separate fits for each held-out table. Its larger sample does not guarantee stability, nor is failure of the old 1-D fit proof the span test is predetermined. If the core result is promising but unstable, compare nested 16-table and 23/24-table subsets with only central edits on fixed evaluation tables. That would directly test whether additional support helps.

**No automatic alpha=2 dose arm.** A small dose-response can be informative later, but it expands the current question and can worsen distribution shift. If needed, alpha=0, 0.5, 1 is an initial local check; do not make it a prerequisite now.

**Do not adopt "same positive random direction in both ablations" as the paired fix.** Use the rotated-coordinate construction above, which preserves the actual relationship between the edits.

## Verified inherited issues and separate cleanup

The checkout is at `a2ef513`; `aa229ae` applied the earlier Codex audit and `a2ef513` added the draft. The following distinctions matter so that the cleanup list is accurate rather than copied wholesale.

| Item from the pasted status list | What is verified now | Action and relation to 004 |
|---|---|---|
| Raw-output mirror | Local handoff explicitly says not mirrored; no completion record found in the inspected tree. Remote state was not queried. | Verify actual authorized remote mirror first. |
| 002 start-tracking fix | The wording error is corrected. The lens loop separately computes z/start/answer tracking; verbalizer `start_swap_tracking_roles` measures intermediate-role tracking under a start-input change. The exact additional bug intended by the reviewer was not identified from that label alone. | Independently reproduce one member/partner example, state which quantity is intended, and attach a failing example before changing it. Do not claim a specific fix is outstanding or complete without that evidence. This need not block the brew span measurement. |
| Restore G4 | The on-disk 003 design is unchanged from its original design commit. It says per-content kappa >=0.7; code uses simplified unadjusted per-item p-values for the repeats and advertises a pooled z/L42–44 flag. PI guidance also had a stronger general gates-before-outcomes instruction. | Record the design/code/process discrepancy and preserved failed result. Do not "restore" a guessed rule into a historical design. Resolve forward-looking rules explicitly; no causal reader scoring is authorized by the old G4 result. |
| Import boundary | Test covers only named `scripts/common` modules. 003 imports the mixed 001 module, which contains a reader import inside its snapshot function. | Make 004 label-safe and test its transitive dependencies. The current test is not evidence of experiment-entry-point isolation; no evidence here that 003 executed reader code for labels. |
| Obsolete analysis decisions | 001 `stage_analysis` still emits `brew_supplies_not_needed_class` from a family-level membership result plus substitution thresholds, and its power flag counts pairs. | Correct or deprecate before regenerating/reusing those decisions. Keep the measurements; do not promote them to item-level labels. Do not run old analysis code as part of 004 initialization. |
| Negligibility check | `stats.equivalence` and its call on table-mean S exist. | A signed-mean equivalence test is not missing, but cannot establish negligible effects for most individual cases because signs can cancel. Keep fraction |S|<epsilon and absolute-effect summaries. A different alleged implementation bug needs an explicit counterexample. |
| Chain parser | `re.search` returns the first integer anywhere in the output. This can mis-score explanations containing multiple numbers. | Specify intended answer extraction and re-audit saved decodes before using chain admission. It does not affect a brew-only 004 launch unless shared code changes introduce a dependency. |
| Report 001 leftovers | For example, "about half the gap in every pair" is stronger than 100% of pairs above the toward threshold. Other old not-needed-class language persists. | Correct summaries from raw evidence; neither rewrite data nor imply these prose fixes require new GPU runs. |

Complete the mirror and dependencies needed by 004 before expensive execution. Keep unrelated historical repairs in a separate tracked list with evidence and provenance; do not make the entire backlog a prerequisite for this bounded brew diagnostic.

## Requested deliverable from CC

Produce the revised design and exact arm manifest, implement the necessary checks, record the verified backup status, and run the authorized primary experiment after the full-path smoke passes. Report the continuous effects, controls, reproducibility, and qualified interpretations, including negative and inconclusive results. End with whether a focused causal follow-up is justified; do not automatically schedule a causal-label-based reader ranking.
