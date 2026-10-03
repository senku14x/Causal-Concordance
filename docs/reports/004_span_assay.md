# 004 — Span assay at `:` L42: report (dev)

*Run 2026-10-03 (start 01:39 UTC) at commit `9da6b23`, clean tree; design `docs/design_specs/004_span_assay.md` v2 with
the pre-run clarifications of §13. 33 pairs, one per table, 55.8 min. Dev split only. Reader-free. This is a descriptive
diagnostic: it **assigns no labels and ranks no readers**. All numbers are for Qwen3.6-27B, WorkspaceBench brew start
pairs, the SEC regime, the final `:` at L42 (reference: the start-colour token at L40), and these 33 pairs.*

Numbers come from `results/004_span_assay/analysis/{summary,per_pair,examples}.json` (built by
`scripts/004_span_assay/span_assay.py analysis`). Raw rows are in `outputs/004_span_assay/run/pairs/` and on the public
mirror. "S" is the signed share of the natural answer-margin gap G moved by an edit (not an accuracy). "Recovery" is
S(edit) / S(whole cell) per pair, unclipped, on the recovery subset. Intervals are 95% table-bootstrap intervals (one
pair per table, so tables = pairs). "Joint span" is the fitted start/z/answer-associated span, not the J-lens.

## 1. Summary

- **The whole-cell patch at `:` L42 moves a median 0.43 [0.37, 0.49] of the gap** (range 0.23–0.65; all 33 pairs are in
  the recovery subset).
- **Fitted spans recover more than the 1-D direction, and the joint span recovers most of the cell.** Median recovery:
  1-D z direction 0.06 [0.01, 0.09]; z-associated span (rank 11) 0.21 [0.16, 0.31]; joint span (rank 33) 0.72
  [0.66, 0.81]. The z span beats the 1-D direction in 32/33 pairs (paired ΔS +0.063 [+0.049, +0.081]).
- **Generic subspaces of the same rank do as well or better.** PCA of the same activations recovers 0.58 at rank 11 and
  0.89 at rank 33. Rescaled to the content edit's norm, PCA still edges out the fitted spans: z span − PCA ΔS −0.023
  [−0.035, −0.010] (PCA larger in 24/33); joint − PCA −0.016 [−0.036, −0.007] (23/33). Recovery tracks how much of the
  pair difference Δ a subspace contains: z span 35% of ‖Δ‖², PCA-11 59%, joint 66%, PCA-33 77% (medians).
- **The added probability mostly does not go to the donor's answer.** z span: median F_target 0.13 (defined in 32 pairs), target-directed in
  4/33 pairs; its largest single colour gain is the *recipient's own intermediate* in 18/33 pairs. Joint span: F_target
  0.35, 10/33. Whole cell: 0.64, 12/33 (11/33 flagged for mass outside the accepted spellings).
- **Random controls do nothing, and the real edits clear them.** Rank- and norm-matched random edits have median S
  ≤ 0.005. Real z-span and joint-span swaps exceed the control q95 in 31–33/33 pairs. Clamping the z span to its
  fitting mean removes 0.12 [0.11, 0.13] of the gap and exceeds the paired-rotation q95 in 31/33 (joint: 0.22, 33/33).
- **The z span and its complement are far from additive:** S(Z) + S(complement) falls 0.18 [0.15, 0.21] short of the
  whole cell; |I| ≥ 0.15 in 67% of pairs.
- **Reproducibility:** the z span agrees between the two half-sized fits only at a low level (recovery 0.17 / 0.17;
  median |ΔS| 0.049). The joint span reaches ≥ 0.5 in both (0.73 / 0.63) but its median |ΔS| is 0.052, missing the
  pre-set 0.05 tolerance by 0.002, so by the frozen rule it is not reproduced.
- **Reference (start token L40):** whole cell 0.20 [0.16, 0.26]; the start span recovers 0.42 [0.36, 0.55] (30 pairs),
  PCA of the same rank 0.57.
- **Predictions:** S1 no (0.21 < 0.25), S2 yes (0.72), S3 no (0.42), S4 no, S5 no (low-effect agreement), S6 yes (94%).
- **Reading (design §6):** a stable larger effect without target direction or an advantage over PCA is a descriptive
  geometry result. Within these assays, the fitted spans move the answer roughly as much as any data-aligned subspace
  holding the same share of the pair difference; nothing here isolates influence specific to the intermediate. 004
  nominates no intervention construction for further validation.

## 2. Material and run

- **Pairs:** one of the 82 gated 003 dev pairs per table (seed 20261007), 33 tables
  (`results/004_span_assay/manifest_pairs.json`, frozen before the smoke).
- **Fits:** 003's least-squares coefficients. Primary = the two folds not containing the table; r1, r2 = each of those
  folds alone. Ranks: z span 11 in every fit; joint span 33 (primary), 32–33 (r1, r2); start span at the reference 11.
- **Arms, controls, scoring:** as in design §3–§4 (334 conditions, 476 scored prompts, 956 forwards per pair; 31,548
  forwards). Every scored prompt records the full-sequence probability of all 24 accepted spellings of the 12 colours,
  including the two-token " Teal".
- **Gates:** the preflight passes (`preflight.json`); the smoke (v2, after the external review) passes every gate
  (`smoke_checks.json`). In the run every write passed fidelity (rho/kappa checked at ≥ 1e-6 ‖h‖, read-back bound
  below). There were no quarantined rows and no missing 1-D directions.
- **Mass outside the accepted spellings:** about 14% on clean prompts, mostly formatting tokens (`<|im_end|>`,
  newlines, " **"; `smoke_outside_mass_tokens.json`).

## 3. Recovery

Median S over all 33 pairs, and median recovery on the recovery subset (`:` 33 pairs, reference 30), primary fit:

| Edit at `:` L42 | k | median S [95% CI] | median recovery [95% CI] | r1 / r2 recovery |
|---|---|---|---|---|
| whole cell | 5120 | 0.43 [0.37, 0.49] | 1 | — |
| 1-D z direction (003) | 1 | 0.03 [0.00, 0.04] | 0.06 [0.01, 0.09] | 0.02 / 0.01 |
| z-associated span | 11 | 0.10 [0.07, 0.13] | 0.21 [0.16, 0.31] | 0.17 / 0.17 |
| start-associated span | 11 | 0.11 [0.07, 0.15] | 0.21 [0.17, 0.36] | 0.28 / 0.16 |
| answer-associated span | 11 | 0.05 [0.04, 0.09] | 0.14 [0.10, 0.20] | 0.11 / 0.11 |
| joint span | 33 | 0.32 [0.27, 0.35] | 0.72 [0.66, 0.81] | 0.73 / 0.63 |
| complement of the z span | 5109 | 0.12 [0.09, 0.16] | 0.33 [0.18, 0.36] | 0.32 / 0.32 |
| complement of the joint span | 5087 | 0.04 [0.02, 0.06] | 0.09 [0.05, 0.12] | 0.09 / 0.07 |
| PCA, rank 11 | 11 | 0.24 [0.21, 0.29] | 0.58 [0.53, 0.68] | 0.53 / 0.35 |
| PCA, rank 11, z-span edit norm | 11 | 0.14 [0.11, 0.15] | 0.29 [0.24, 0.35] | 0.30 / 0.21 |
| PCA, rank 33 | 33 | 0.37 [0.32, 0.45] | 0.89 [0.84, 0.96] | 0.78 / 0.77 |
| PCA, rank 33, joint-span edit norm | 33 | 0.34 [0.29, 0.36] | 0.80 [0.72, 0.86] | 0.69 / 0.66 |

The 1-D z swap is negative in 9/33 pairs; the z span in 1/33; the joint span in none. The joint span exceeds the whole
cell in 1/33 pairs.

## 4. Fitted spans against generic subspaces

Paired differences in S (primary fit, all 33 pairs; median [95% CI], and the number of pairs where the second edit is
larger):

| Comparison | ΔS | second larger |
|---|---|---|
| z span − 1-D z | +0.063 [+0.049, +0.081] | 1/33 |
| joint − z span | +0.165 [+0.148, +0.209] | 0/33 |
| z span − PCA-11 | −0.116 [−0.157, −0.082] | 29/33 |
| z span − PCA-11 at the z-span norm | −0.023 [−0.035, −0.010] | 24/33 |
| joint − PCA-33 | −0.060 [−0.082, −0.039] | 30/33 |
| joint − PCA-33 at the joint norm | −0.016 [−0.036, −0.007] | 23/33 |

The stress fits give the same signs. Two of the norm-matched intervals cross zero there (r1 joint − PCA-33 dose
−0.012 [−0.027, +0.012]; r2 z span − PCA-11 dose −0.012 [−0.040, +0.004]).

**Why PCA wins unscaled:** the top PCA directions of the fitting activations, centred within table, hold more of the
natural pair difference than the fitted content spans of the same rank. Median share of ‖Δ‖², from the prompt-length
captures:

| Subspace | 1-D z | z span | start span (`:`) | answer span | joint | PCA-11 | PCA-33 |
|---|---|---|---|---|---|---|---|
| share of ‖Δ‖² | 0.10 | 0.35 | 0.34 | 0.28 | 0.66 | 0.59 | 0.77 |

At matched edit norm the gap narrows to −0.02 S but does not reverse. The fitted labels therefore buy no extra
behavioural effect per unit of edit over generic high-variance directions, at either rank.

**Random controls** (primary fit, 16 draws each; descriptive, no per-item significance):
- the controls are essentially inert: median S ≤ 0.005 for rank-matched projections and norm-matched directions;
- the real edit exceeds the control q95 in: 1-D z 24/33 (rank) and 21/33 (norm); z span 32/33 and 31/33; joint span
  33/33 and 33/33; start span 31/33 and 28/33;
- the random complement of a rank-33 subspace keeps ≈ 99% of Δ and moves 0.43 (≈ the whole cell); the real joint
  complement is below its q05 in 33/33.

These comparisons show the fitted spans sit in directions that matter for this endpoint, unlike random directions. PCA
shows the same is true of generic data-aligned directions.

## 5. Where the added probability goes

Median positive colour-mass gains (probability units) per swap, primary fit, all pairs. Role definitions: a′ = the
donor's answer; z′ = the donor's intermediate; recipient z = the member's own intermediate. Each colour is counted once,
under its first role in that order.

| Edit | to a′ | to z′ | to recipient z | total positive gain | median F_target (n defined) | target-directed | flagged |
|---|---|---|---|---|---|---|---|
| 1-D z | 0.001 | 0.001 | 0.007 | 0.028 | 0.07 (21) | 1 | 0 |
| z span | 0.007 | 0.006 | 0.014 | 0.067 | 0.13 (32) | 4 | 0 |
| PCA-11 | 0.035 | 0.027 | 0.035 | 0.191 | 0.25 (33) | 6 | 4 |
| PCA-11, z-span norm | 0.011 | 0.014 | 0.016 | 0.078 | 0.17 (32) | 3 | 0 |
| joint span | 0.064 | 0.029 | 0.028 | 0.231 | 0.35 (33) | 10 | 6 |
| PCA-33 | 0.093 | 0.029 | 0.027 | 0.301 | 0.49 (33) | 13 | 7 |
| PCA-33, joint norm | 0.081 | 0.030 | 0.030 | 0.272 | 0.41 (33) | 12 | 5 |
| whole cell | 0.176 | 0.023 | 0.025 | 0.361 | 0.64 (33) | 12 | 11 |
| start span (reference) | 0.008 | 0.016 | 0.000 | 0.081 | 0.06 (30) | 5 | 2 |
| whole cell (reference) | 0.025 | 0.046 | 0.003 | 0.181 | 0.26 (32) | 6 | 3 |

- **The single largest gain** goes to: under the z span, the recipient's own intermediate in 18/33 pairs, a′ in 9 and
  z′ in 6; under the joint span, a′ 16, recipient z 9, z′ 8; under the whole cell, a′ 24, z′ 5, recipient z 3.
- **The z-span swap raises the donor's answer**, but most of what it moves goes elsewhere, chiefly to the recipient's
  own intermediate. The data do not say why: semantic contamination, a partial change in computation, or something
  else.
- **The whole-cell patch at the start token** raises the donor's intermediate (0.046) more than the donor's answer
  (0.025).

Flagged rows (mass outside the accepted spellings up > 0.05, or both endpoints down > 0.05 nats) stay in every median;
none of the z-span rows is flagged.

## 6. Necessity

Clamping the span coordinates to that fit's own mean in both the member's and the partner's run. N = 1 − G_abl/G, the
share of the gap removed:

| Span | N, primary [95% CI] | r1 / r2 | above paired-rotation q95 | above random-clamp q95 |
|---|---|---|---|---|
| 1-D z (`:`) | 0.027 [0.009, 0.038] | 0.017 / 0.010 | 23/33 | 25/33 |
| z span (`:`) | 0.118 [0.108, 0.135] | 0.104 / 0.083 | 31/33 | 32/33 |
| joint span (`:`) | 0.223 [0.184, 0.268] | 0.215 / 0.216 | 33/33 | 33/33 |
| start span (reference) | 0.094 [0.051, 0.143] | 0.110 / 0.083 | 28/33 | 30/33 |

The paired rotation keeps both runs' edit norms and their inner product and randomizes only the orientation. The real
clamps clear it in most pairs. This is evidence that the fitted spans' orientation matters relative to these controls.
It is not a per-item significance claim, a natural-route necessity claim, or a certificate of semantic purity, and
004 ran no PCA clamp to compare against. No ablated prompt was flagged.

## 7. Additivity

I = S(span) + S(complement) − S(whole cell), primary fit:

| Span | signed I | |I| | fraction |I| ≥ 0.15 |
|---|---|---|---|
| z span (`:`) | −0.178 [−0.209, −0.145] | 0.188 | 0.67 [0.48, 0.82] |
| joint span (`:`) | −0.075 [−0.120, −0.045] | 0.075 | 0.18 [0.06, 0.30] |
| start span (reference) | −0.060 [−0.089, −0.036] | 0.065 | 0.12 [0.03, 0.24] |

For the z span, the span and its complement each move little while the two together (the whole cell) move much more.
The z span and its complement do not partition the whole-cell effect additively for this endpoint. The joint span and
its complement come closer, with a smaller negative interaction.

## 8. Reproducibility (r1 vs r2, all 33 pairs)

| Quantity | Pearson r | median \|S_r1 − S_r2\| | median recovery r1 / r2 | verdict (design §13) |
|---|---|---|---|---|
| S, 1-D z | 0.22 | 0.023 | 0.02 / 0.01 | low-effect agreement |
| S, z span | 0.53 | 0.049 | 0.17 / 0.17 | low-effect agreement |
| S, joint span | 0.77 | 0.052 | 0.73 / 0.63 | not reproduced (tolerance missed by 0.002) |
| S, PCA-33 | 0.81 | 0.039 | 0.78 / 0.77 | high recovery reproduced |
| S, start span (reference) | 0.90 | 0.017 | 0.44 / 0.35 | low-effect agreement |
| N, z span | 0.38 | 0.050 | — | — |
| N, joint span | 0.79 | 0.034 | — | — |

The joint span's miss is marginal; it is reported under the frozen rule, not re-argued. The half-sized fits test
stability at low data; they do not estimate the primary fit's reliability.

## 9. Reference cell (start token L40)

- **The whole-cell effect is small:** median S 0.20 [0.16, 0.26]; 3 pairs fall below 0.10 (0066, 0316, 0323).
- **On the remaining 30 pairs**, recovery is: start 1-D 0.38 [0.27, 0.44]; start span 0.42 [0.36, 0.55]; PCA-11 0.57
  [0.45, 0.64]; complement 0.25 [0.14, 0.34]. The start span beats the 1-D direction by ΔS +0.009 [+0.003, +0.015].
  Against PCA the differences are not distinguishable from zero (−0.016 [−0.035, +0.007]; dose-matched −0.009
  [−0.024, +0.009]).
- **The positive control is weak:** the fitted start span recovers under half of a whole-cell effect that is itself
  small. This alone does not identify why.

## 10. Predictions (priors written before any 004 run)

| # | Prediction | Prior | Value | Outcome |
|---|---|---|---|---|
| S1 | median z-span recovery ≥ 0.25 | 45% | 0.21 | does not hold |
| S2 | median joint-span recovery ≥ 0.5 | 40% | 0.72 | holds |
| S3 | median start-span recovery at the reference ≥ 0.5 | 55% | 0.42 | does not hold |
| S4 | z span beats unscaled PCA-11 (paired median) | 50% | ΔS −0.116 | does not hold |
| S5 | z span reproducible under the stress fits (≥ 0.5 in both, median \|ΔS\| ≤ 0.05) | 35% | 0.17 / 0.17; 0.049 | does not hold (low-effect agreement) |
| S6 | z-span clamp above the paired-rotation q95 in ≥ 50% of pairs | 45% | 94% | holds |

## 11. Interpretation (design §6)

| Observation in the design | Here | Licensed reading |
|---|---|---|
| Z recovery ≥ 0.5, reproduced | no (0.21) | — |
| J ≥ 0.5 while Z < 0.25 | yes (0.72, 0.21) | The broader fitted span recovers the cell; the z-span estimator does not isolate it. Start-only (0.21) and answer-only (0.14) spans do not carry it alone either. Controls are cleared, but same-rank PCA does as well or better, and the reproducibility rule is missed narrowly. |
| Large or heterogeneous \|I\| | yes for the z span (67% ≥ 0.15) | Additive attribution between the z span and its complement is inadequate for these pairs and this endpoint. Only the measured effects are reported. |
| Material r1/r2 disagreement | marginal for the joint span (0.052) | Not reproduced under the low-data stress test, by a hair. |
| Poor start-span recovery with a stable whole-cell reference | 0.42 on a small, stable reference | The fitted span recovers that site's leverage poorly; cause not identified. |
| Ablation not clearly above the paired rotations | the opposite: 94% | Orientation specificity relative to paired rotations is observed for the z span. That is not semantic isolation. |
| Larger effect without target direction or advantage over PCA | yes | **A descriptive geometry result.** |

What this does and does not say:
- **It does say:** within these assays, editing the fitted spans moves the answer about as much as generic
  data-aligned subspaces holding the same share of the pair difference, slightly less at matched norm. Most of the
  z-span edit's colour movement does not go to the donor's answer.
- **It does not say** that the intermediate is unused at `:` L42, or that no estimator could isolate it. It also does
  not say that the joint span "contains the computation"; the joint-span result is matched by PCA.

## 12. Qualitative examples

Fixed-seed random pairs beside the median, extreme and flagged cases (primary fit, `:` L42). The top-4 colours are
accepted-spelling masses; the full set is in `results/004_span_assay/analysis/examples.{json,md}`.

| Pair (why) | recipient → donor path | S: 1-D / z span / PCA-11 / joint / whole | clean top 4 | after the z-span swap | after the joint swap |
|---|---|---|---|---|---|
| 0069 (random) | black→teal→**orange**; red→green→**pink** | −0.03 / 0.02 / 0.34 / 0.49 / 0.61 | orange 0.80, red 0.05, teal 0.01, pink 0.01 | orange 0.81, red 0.06, teal 0.02, pink 0.01 | orange 0.48, pink 0.18, red 0.06, purple 0.06 |
| 0228 (random; median z-span recovery) | brown→blue→**pink**; white→purple→**red** | 0.04 / 0.10 / 0.22 / 0.33 / 0.48 | pink 0.79, blue 0.04, teal 0.01, white 0.00 | pink 0.70, blue 0.08, white 0.01, teal 0.01 | pink 0.62, blue 0.11, white 0.03, red 0.02 |
| 0299 (random) | pink→purple→**gray**; green→blue→**black** | 0.05 / 0.21 / 0.41 / 0.41 / 0.52 | gray 0.79, purple 0.05, gold 0.04, black 0.01 | gray 0.68, purple 0.08, gold 0.06, black 0.05 | gray 0.57, black 0.17, purple 0.08, gold 0.03 |
| 0383 (max z-span recovery) | green→pink→**brown**; black→purple→**orange** | 0.28 / 0.45 / 0.60 / 0.53 / 0.58 | brown 0.79, orange 0.08, pink 0.02, gold 0.01 | brown 0.41, orange 0.33, pink 0.08, gold 0.07 | orange 0.44, brown 0.38, pink 0.04, gold 0.03 |
| 0142 (min z-span recovery) | orange→blue→**brown**; white→teal→**orange** | 0.00 / −0.01 / 0.15 / 0.15 / 0.29 | brown 0.72, orange 0.04, teal 0.03, green 0.03 | brown 0.69, orange 0.04, teal 0.03, green 0.03 | brown 0.50, blue 0.09, orange 0.09, teal 0.06 |
| 0012 (flagged whole cell) | purple→white→**black**; red→pink→**purple** | 0.13 / 0.17 / 0.24 / 0.32 / 0.35 | black 0.68, white 0.07, teal 0.03, brown 0.03 | black 0.48, **white 0.19**, purple 0.08, pink 0.04 | black 0.35, purple 0.20, white 0.09, pink 0.07 |
| 0041 (flagged PCA, joint, whole) | teal→black→**gray**; green→red→**blue** | 0.04 / 0.10 / 0.46 / 0.43 / 0.57 | gray 0.65, gold 0.10, black 0.07, orange 0.03 | gray 0.48, **black 0.16**, gold 0.09, orange 0.09 | black 0.27, blue 0.20, orange 0.11, red 0.08 |

In 0012 and 0041 the z-span swap mostly raises the recipient's own intermediate (white, black). In 0383 it moves
probability to the donor's answer, orange. In 0142, which has the smallest z-span effect, the joint-span swap raises
the recipient's intermediate (blue) as much as the donor's answer.

## 13. Limits

- **Scope:** dev only; one pair per table; one cell and layer (`:` L42) plus one reference (start token L40); one
  endpoint (the full-sequence a′-vs-a margin over the accepted spellings); the SEC regime.
- **Fits:** spans come from 003's least-squares estimator with reference coding and a ≥ 2-occurrence rule. A different
  estimator (regularized, DAS-trained, more data) could behave differently; 004 does not test that.
- **The PCA comparator** is unsupervised but fitted on the same activations, with the evaluated table excluded. It is
  a strong generic baseline, not a random one.
- **Control statistics** are descriptive random-orientation comparisons: no per-item significance, pooled p-values or
  FDR.
- **The share of ‖Δ‖²** is computed from the 001/003 prompt-length captures, not from every continuation.

## 14. Next

The design's framing (still awaiting PI confirmation) says that if 004 fails or is inconclusive, the project returns
to task-level reader evaluation and does not add estimators automatically. 004 gives a clear descriptive answer: larger
fitted spans recover more, but no more than generic subspaces. It nominates no intervention construction. Options for
the PI:
1. **Return to the reader panel** (the primary line per the handoff): native reader configurations, errors, overlap,
   complementarity. Causal labels from these span constructions should not be used to score readers.
2. **A bounded follow-up, only if it would change a decision:** for example, whether a supervised method (DAS) at
   `:` L42 beats PCA at matched rank and norm. Not proposed by default.

## 15. Silly mistakes (this experiment)

All are also in CLAUDE.md:
- The first preflight imported the 003 module, which is mixed via 001.
- `brew_material` cited a test file that did not exist.
- The preflight's rank printout had colliding keys.
- The smoke inspection read the random-complement control in the wrong tail.
- From the external smoke review:
  - S5 was coded with S1's threshold and the difference of median recoveries;
  - the run path recorded fidelity but did not enforce G1;
  - the fidelity exemption (1e-4) was wider than the measured resolution;
  - my smoke summary said "fidelity holds on every write" and "25 spellings" (it is 24);
  - the smoke analysis reported a Pearson r from two points.

All were fixed before the 33-pair run; the v1 smoke outputs are kept as superseded.
- Writing this report: eight numbers were hand-copied from three-decimal printouts and double-rounded (for example
  0.205 → 0.21 where the data give 0.2047), and two F_target "defined" counts were written from memory (17 and 29;
  the data give 21 and 32). All were caught by regenerating every table from the JSON before commit.
