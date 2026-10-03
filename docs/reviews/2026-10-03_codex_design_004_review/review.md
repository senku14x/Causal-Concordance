# Review of the 004 span assay design

Review of design v1 on 2026-10-03. Recommendation: run a corrected, bounded diagnostic at `:` L42 with the separate start-token L40 control. Defer L44 and second pairs until the primary results identify a reason to add them. The experiment can diagnose limitations of the existing assay; it cannot establish that a successful span isolates the intermediate or make a failed intervention evidence of non-use.

The highest-priority changes are to the branch interpretations, reliability criterion, and implementation gates. This review does not authorize or launch a model run, and does not change the experiment design in place.

## Layer and position scope

Keep the primary intervention at the output of block 42, at the final colon, and let downstream computation run normally. This holds the cell fixed while changing the intervention geometry, directly addressing the question left by 003. The existing whole-cell effect at that cell provides room to compare edits.

A clamp reapplied across multiple downstream layers would ask a different question about influence under sustained removal and possible reintroduction. It is not a repair to this single-cell comparison. The start-token L40 experiment is a separate control, not a simultaneous edit. If L44 is later run, construct the spans from L44 coefficients and report it separately; do not silently reuse L42 directions there.

004 should also cease to be a gate for all reader evaluation. Reader scoring against causal labels remains conditional on a validated reference. Native-configuration reader accuracy, errors, overlap, and complementarity can proceed independently, as agreed in the current project discussion.

## Geometry already available without a model run

I computed this from the saved 003 coefficients. I used float64 SVD of the centered coefficient vectors for colours with at least two fitting examples, with relative singular-value cutoff 1e-6. Centering spans the same eligible contrasts as subtraction of one eligible reference. This is an audit choice, not a silently adopted rank rule for 004.

Reproduction: [script](check_004_span_geometry.py), [results and input hashes](2026-10-03_004_span_geometry.json). Source hashes were unchanged during the calculation. No GPU forward or external call was used.

- The z span at `:` L42 has rank 11 in every primary and disjoint fit checked.
- The smallest principal angles between the primary z and start spans are 33.6–36.1 degrees across the three evaluation folds. Between z and answer they are 30.2–38.3 degrees. Thus the proposed less-than-45-degree warning already fires for both comparisons in every fold. This measures geometrical proximity; it does not establish shared semantic content or an exact subspace intersection.
- Cross-fit mean squared principal cosine for the z span is 0.304–0.331. For the start span at the start token L40 it is 0.740–0.763. This statistic is 1 for identical equal-rank subspaces and 0 for orthogonal ones. The comparison shows a more reproducible fitted geometry for the start control, not known behavioral reliability for either span.
- One single-fold answer fit has only 11 eligible colours, giving rank 10 rather than 11. Report eligible sets and ranks when comparing joint spans across fits; not all changes are rotations of the same fixed-rank object.

These observations support retaining the control and measuring specificity. They do not determine whether the proposed behavioral experiment will succeed.

## Changes needed before interpreting the run

**1. A fitted z span is an intermediate-associated span, not an isolated intermediate variable.** The regression controls additive label associations; projection onto its coefficient span can still change answer- or start-associated coordinates. The observed cross-role geometry makes this a concrete concern. Branch A should say that the wider fitted span recovers more influence than the pair-specific direction, with specificity still unresolved.

Add start-only and answer-only span swaps at the primary cell as inexpensive diagnostic arms. Without these, a successful joint span could be explained entirely by the answer span. Two additional swaps per fit are cheap compared with the controls already planned. Their descriptive comparison need not trigger a whole second control grid. Do not orthogonalize z against the other roles after seeing effects and present that as the same intervention.

**2. Branch E confuses failure to establish specificity with evidence of a generic explanation.** Failing to exceed a noisy random q95 in half the pairs can occur with weak but specific effects, heterogeneous effects, or limited sensitivity. Rename it "specificity not established." Evidence that random controls reproduce the content intervention needs an explicit comparison of the content-minus-control effects and uncertainty; a claim of equivalence needs an equivalence margin. The stronger 003 necessity interpretation has already been withdrawn, so 004 need not manufacture a binary vindication/retraction decision.

**3. Branch D combines different failures.** Low cross-fit agreement is an estimation/reproducibility problem. Low start-span recovery can instead reflect an incomplete span, context-dependent geometry, redundancy, or nonadditivity. The input colour affects the problem by construction, but its influence is not known to be mediated mostly by an additive fitted span at this particular cell. Report "unstable estimates" and "control recovery inadequate" separately. The latter does not specifically prescribe more fitting data or regularization.

**4. Branch A needs absolute agreement as well as correlation.** S2 = 2*S1 has r = 1 despite systematic disagreement in intervention strength. Use the already proposed median absolute cross-fit difference in the decision rule, with a prespecified material tolerance. Inspect raw margin changes alongside normalized shares, since a common denominator G can affect correlations. Undefined r on nearly constant effects should not automatically count as instability.

The current primary fit uses two folds; the two reliability fits use one fold each. Keep stating that this is a stress test with less fitting data, not a direct reliability estimate for the full primary fit. A half-size failure does not by itself establish that adding dimensions cannot work.

**5. No triggered branch means inconclusive, not partially adequate.** For example, both spans could recover little with moderate r and no large interaction. Branch B's interpretation also needs to acknowledge instability or damage if those occur concurrently. Put the continuous quantities, uncertainty, and damage next to every branch flag.

**6. Sixteen random draws cannot support a per-item p below 0.05.** The stated Monte Carlo formula is correct, but its minimum is 1/17 = 0.0588. The interpolated q95 comparison is not a calibrated 5% per-item test. Keep 16 if this is a descriptive control comparison and explicitly label it that way. If a 5% per-item test is required, increase and predefine the draw count; 39 permits p = 0.025, although its tail estimates remain coarse. No multiple-testing claim follows automatically. With no labels being assigned, extra draws are not inherently necessary.

**7. Interactions may cancel in the signed median.** Keep signed I, and also report median |I|, the fraction with |I| >= 0.15, and positive/negative cases. Opposite interactions across tables can give median I near zero while invalidating additive attribution on many items. The measured nonadditivity is for this intervention pair and log-probability-margin endpoint; it is not a complete mechanism.

**8. Recovery is a ratio to a reference intervention, not fraction of content recovered.** S_edit/S_full can exceed 1 or be negative. Preserve signed S and raw margin changes for every pair, including those omitted from the ratio because S_full < 0.10. Define the denominator for each branch explicitly: the ratio subset versus all 33 primary pairs. A successful wider edit should show an actual paired improvement over 1-D, not only pass a standalone threshold. Record damage alongside the improvement.

## Implementation changes and checks

**Full-rank projection should agree numerically, not necessarily bitwise.** In finite precision, h + Q Qᵀ(h_d - h) need not equal direct assignment h_d bitwise, even when Q is a square orthogonal basis. The subtraction/addition can differ even with the identity basis. Keep exact self/no-op tests where the code path is genuinely identical. For the algebraic identity check, test state error and downstream margin error against a declared, measured numerical tolerance. A hardcoded full-rank path that directly assigns the donor is a valid optimization but does not test general projection arithmetic.

**Make span sanity algebraic.** Assert QᵀQ ≈ I, P² ≈ P, u is contained in the intended span when its two colours are eligible, and delta_span + delta_complement ≈ delta. Because u belongs to the span, uᵀPh ≈ uᵀh already preserves the original score. A newly trained finite-sample probe on a larger projection is not guaranteed to outperform the 1-D classifier on held-out data, so "at least as good separation" is an unsuitable hard gate unless its scoring rule is specified.

**Freeze rank construction and eligible colours.** Specify the SVD/QR tolerance, eligible-reference rule, and missing-direction treatment. Do not count tiny numerical singular values as real dimensions or tune the rank after measuring behavioral effects.

**Use each fit's own mean in reliability ablations.** The inherited `Directions.fitmean` currently hardcodes the `both` fit. Reusing it for r1/r2 would not implement the specified disjoint-fit mean clamps. Every fit needs its own span, eligible set, and mean, all excluding the evaluation table.

**Specify paired random-control coupling.** Necessity compares two edited runs. Drawing independent random directions for the two prompts versus sharing a direction produces different distributions of gap changes. Freeze and record that choice. If stronger matching is wanted, jointly rotate the two real edit vectors using the same random orthogonal map; this preserves each norm and their mutual inner product. Norm matching alone tests perturbation size, not every possible nuisance. Keep a given random edit consistent across every teacher-forced continuation; never resample per answer spelling.

**Full-sequence mass and candidate normalization are different quantities.** Record raw accepted-spelling masses, their total, and optionally the distribution conditional on the 12-colour set. If reporting 1 minus their total, verify the accepted token-sequence events are disjoint and call it mass outside the accepted spelling set. A softmax normalized across colours discards the evidence of mass leaving that set. The inherited scorer measures spelling-prefix likelihoods without an answer terminator; do not silently describe these as exact complete-answer probabilities.

**All-colour scoring requires matching donor captures for all continuations.** 003 precomputed continuation sources only for the two endpoint answers. With `extra_spellings` for all colours, ` Teal` is scored even when neither endpoint is teal. Capture its donor state at the matching sequence length as well, and apply the same intervention for its entire score. The existing scorer supports extra spelling sets; the donor-cache preparation must match that usage.

## Budget and scope

The 30-minute estimate is not supported by the listed operations yet. The start-token specification says to run the same edit set, but its approximately 60-run allowance does not cover that same control/necessity grid. Necessity reliability is promised in section 6 but absent from the count: even just three content clamps under two additional fits costs 12 extra scored evaluations per pair before controls. Scoring all 12 colours also makes the multi-token continuation recurring rather than an occasional endpoint cost. With the present scorer, every score includes a base forward plus at least the ` Teal` continuation, not a small ten-percent surcharge in forward count. The quoted 0.12 seconds may itself time a different unit; measure the actual new path.

Write an explicit arm manifest, smoke-test two pairs with all requested scoring, count scored evaluations and upper-stack replays separately, then estimate wall time. Prioritize the 33 primary tables and the start control. Defer L44, extra pairs, and a broad rank sweep.

One pair per table avoids duplicate-table observations in the primary summary. It does not make fitted predictions fully independent: many evaluation cases share fitted coefficients. Table resampling is a useful conditional-on-fits summary; fitting uncertainty is addressed separately by the disjoint-fit diagnostic. Do not describe the resulting intervals as including all training uncertainty.

The qualitative panel should include fixed-seed random examples and at least one failure, in addition to the largest, median, and smallest effects. Selecting only by effect size cannot show what a typical unselected output looks like.

Proceed after these targeted clarifications. If the primary run cannot establish a stable, specific assay, record that outcome and return to reader reliability and complementarity; do not automatically expand the experiment into an exhaustive label-construction project.
