# 004 — Span assay diagnostic at `:` L42 — design v2 (revised to the consolidated handoff of 2026-10-03)

*v1 (a2ef513) superseded; changes in §12. Dev only. Reader-free. **Assigns no labels and ranks no readers.** The
project's primary line is the comparison of native reader configurations: task information, errors, overlap,
complementarity. Causal experiments are a focused validation branch; this framing comes from the handoff and awaits
PI confirmation. If 004 succeeds, it nominates an intervention construction for further validation. If it fails or is
inconclusive, the project returns to task-level reader evaluation and does not add estimators automatically.*

## 1. Question

Does expanding 003's one-direction intervention to its fitted associated spans recover more of the whole-cell effect,
reproducibly? And is that more than a generic, data-aligned subspace of the same size or edit norm would give?

## 2. Material (frozen before any result)

- **Pairs:** one of the 82 gated 003 pairs per table, drawn with seed 20261007 → 33 pairs from 33 tables. The manifest
  (pair ids, seed, exclusions) is written to `results/004_span_assay/manifest_pairs.json` before the smoke run.
  Summaries weight tables equally.
- **Primary cell:** the output of block 42 at the final `:`, SEC regime as in 003. One patch at that cell; downstream
  computation runs normally. The same intervention, with the same random draws, is reapplied in every teacher-forced
  continuation forward. Donor sources are captured at the matching sequence length.
- **Reference cell:** the start-colour token at L40, with start-associated spans. This is a site with known
  whole-cell leverage. An additive span is not guaranteed to recover all of it.
- **Fits:**
  - the 003 two-fold primary fit and the two disjoint one-fold stress fits, all excluding the evaluation table;
  - estimator unchanged: least squares with reference coding, a colour eligible if it appears ≥ 2 times in the role;
  - per fit: render counts, distinct-table support per role and colour, ranks and eligible sets.
- **Out of scope:** L44, a second pair per table, dose sweeps, L2, leave-one-table-out fits, label-shuffled spans,
  readers.

**Wording.** Z is the *fitted intermediate-associated span*: reader-free and additively adjusted, not semantically
isolated. At L42 its smallest principal angles to the start and answer spans are 30–38°, which is geometric proximity
only.

## 3. Arms

Δ = h_donor − h. Bases are orthonormalized in float64 (SVD, relative cutoff 1e-6).

| Cell | Arm | Edit | Fits |
|---|---|---|---|
| `:` L42 | z 1-D (remeasured) | h + (û·Δ) û | primary, r1, r2 |
| | Z span | h + P_Z Δ (rank k_Z ≤ 11) | primary, r1, r2 |
| | J span (start + z + answer) | h + P_J Δ (k_J ≤ 33) | primary, r1, r2 |
| | Z complement | h + (I − P_Z) Δ | primary, r1, r2 |
| | J complement | h + (I − P_J) Δ | primary, r1, r2 |
| | start-only span, answer-only span | h + P_S Δ, h + P_A Δ (descriptive) | primary, r1, r2 |
| | whole cell | h_donor, through the same scoring path | — |
| start token L40 | start 1-D, start span, its complement, whole cell | as above | primary, r1, r2 |

**Swap controls** (primary fit only, 16 draws, seeds saved; for the 1-D, Z and J arms at `:` and the start span at the
reference):
- **random rank-matched projection:** h + P_R Δ, R Haar-random of the same rank;
- **edit-norm-matched random direction:** h + ‖P_S Δ‖ r;
- **random complement:** h + (I − P_R) Δ, for the J complement only.

Note: k/5120 is the expected fraction of the *squared* norm captured by a random rank-k projection (k = 11 →
0.00215; RMS norm fraction 0.046), not of the effect.

**PCA comparator** (deterministic, one basis per fit, nested ranks):
- built from the fitting activations at the same cell, centred within table, each row weighted 1/√(rows in its
  table);
- arms: top-k_Z and top-k_J PCA donor-coordinate swaps (unscaled), and dose-matched versions (the PCA-projected Δ
  rescaled to the corresponding content edit's norm);
- the dose-matched version is undefined when the projected Δ is numerically zero, and is recorded as undefined;
- run for the primary and both stress fits, at `:` (k_Z, k_J) and at the reference (k_start).

**Necessity** (`:` arms: 1-D, Z, J; reference: the start span), for each fit with its own fitting mean μ:

    a_x = Qᵀ(h_x − μ)
    real edit: −Q a_x, applied in the member's run and in the partner's run

N = 1 − G_abl / G.

Necessity controls, primary fit, 16 draws, in both runs:
- (a) **paired rotation:** −U a_x with the *same* Haar U (Gaussian QR, sign-corrected) for member and partner. This
  preserves both edit norms, their inner product and the norm of their difference. It randomizes orientation; it is
  not a clamp.
- (b) a random rank-k subspace clamped to that fit's own mean.

The stress fits repeat the real swaps, complements and clamps only.

## 4. Scoring (every scored prompt and edit)

- The full-sequence LSE margin m (accepted spellings, every token scored) and the clean whole-run gap G. Saved per run:
  raw m, Δm, G and the signed S. No clipping.
- **All 12 colours' accepted-spelling masses q(c)**, their sum M, and q/M.
  - This needs donor captures and continuation forwards for every multi-token spelling (" Teal") on every
    evaluation.
  - Token-sequence events are verified disjoint before 1 − M is used, and 1 − M is called "mass outside the accepted
    spelling set".
  - Next-token diagnostics are kept separate.
- **Composition of the swap gains:**
  - d_c = q_edit(c) − q_clean(c);
  - T₊ = Σ max(d_c, 0);
  - F_target = max(d_{a′}, 0) / T₊, defined only when T₊ ≥ 0.01;
  - gains reported to a′, the donor z, the donor start and other colours, with overlapping roles labelled and not
    double-counted.
- **Target-directed recovery qualification:** F_target > 0.5, **and** log P(a′) up by more than the numerical floor,
  **and** no warning flag.
- **Warning flags** (dev, for this answer distribution only):
  - mass outside the accepted spellings up by more than 0.05;
  - on a swap, both endpoint log-probabilities down by more than 0.05 nats.
  Both ablated prompts are logged separately. Losing the current answer under ablation is expected and is not a flag.
- **Reporting:** all-pair effects, flagged cases and qualified effects are reported separately. Flagged rows are
  never dropped from headline medians or denominators.

## 5. Analysis (continuous first; thresholds are dev heuristics)

- **Recovery subset:** pairs with remeasured S_full ≥ 0.10, defined once, size stated. Recovery = S_arm / S_full per
  pair, unclipped. Signed S is reported for every pair.
- **Paired comparisons:** Δ S for (Z − 1-D), (J − Z), (Z − PCA_k unscaled), (Z − PCA_k dose-matched), and the
  corresponding J comparisons. Medians with table-bootstrap intervals; all-pair scatter plots.
- **Controls:** per pair, the effect, the exceedance count over the 16 draws, and the rank statistic (minimum 1/17).
  These are descriptive random-orientation comparisons: no per-item significance, pooled p-values or FDR, and no
  claim that they define the null of no causal content.
- **Reproducibility** (S and N): correlation, signed bias, median |ΔS| between r1 and r2, and the effect range.
  Material-disagreement tolerance: **0.05 share units** (dev, operational). A high-recovery result counts as
  reproducible under the stress test only if both r1 and r2 reach it and they agree materially. Failure under these
  half-sized fits does not estimate the primary fit's reliability.
- **Interaction:** I = S_span + S_complement − S_full: signed median, median |I|, the fraction with |I| ≥ 0.15, with
  intervals.

## 6. Interpretations (coexisting descriptions, not a forced diagnosis)

| Observation | Licensed reading |
|---|---|
| Z recovery ≥ 0.5, positive paired gain over 1-D, reproduced in both stress fits, with material agreement | The 1-D construction misses influence that the larger associated span recovers. Report the control and specificity qualifiers. |
| … and it exceeds the random comparators and PCA, with mostly target-directed, unflagged effects | A promising candidate for further content-specific, held-out validation. **No automatic labels.** |
| J ≥ 0.5 while Z < 0.25 | Recovered by the broader fitted span but not isolated by the current Z estimator. Check start-only and answer-only, controls and reproducibility before explaining it. |
| Large or heterogeneous \|I\| | Additive attribution is inadequate for those pairs and this endpoint. Report the measured effects only. |
| Material r1/r2 disagreement | Not reproducible under the low-data stress test. More support or regularization is a candidate follow-up, not a remedy. |
| Poor start-span recovery with a stable whole-cell reference | The fitted span recovers that site's leverage poorly. This alone does not identify the cause. |
| Ablation not clearly above the paired rotations | Specificity not established. Not proof that the effect is generic. |
| No clear pattern | Inconclusive. Return to the reader panel unless a bounded follow-up would change a decision. |

A stable larger effect without target direction, or without an advantage over PCA, is a descriptive geometry result.
No outcome establishes absence or natural non-use.

## 7. Predictions (priors; before any 004 run)

| # | Prediction | Prior |
|---|---|---|
| S1 | Median Z recovery at `:` L42 ≥ 0.25 | 45% |
| S2 | Median J recovery ≥ 0.5 | 40% |
| S3 | Median start-span recovery at the reference ≥ 0.5 | 55% |
| S4 | Z beats unscaled PCA-k_Z in the paired median | 50% |
| S5 | Z is reproducible under the stress fits (both r1 and r2 reach the conclusion; material agreement) | 35% |
| S6 | The Z clamp exceeds the paired-rotation q95 in ≥ 50% of pairs | 45% |

## 8. Preflight (CPU, before the GPU smoke)

1. **Colour eligibility:** frozen per fit, with render and table support recorded; missing contrasts are kept in the
   denominators.
2. **Basis identities** in float64 and after conversion to the write dtype: QᵀQ = I, projector idempotence, 1-D
   containment in Z, and span plus complement = Δ (tolerances fixed in the preflight from measured float error).
3. **Zero or negligible writes:** read-back no-op checks, with a negligible threshold relative to float resolution.
4. **Dependency hygiene:**
   - factor the minimal reader-free helpers (materials, gating, render keys) out of the mixed 001 module into
     `scripts/common/`;
   - extend the import-boundary test to the 004 entry point and its transitive imports.
5. **Arm manifest** (`results/004_span_assay/arm_manifest.json`): every condition, with counts of intervention
   conditions, scored prompts and forwards (both ablated prompts and every " Teal" continuation included). Timing is
   measured on the smoke, not taken from v1's estimate.

## 9. Gates and launch

- **G1 fidelity:** on every write, with zero-write handling.
- **G2 (smoke, 2 pairs, end to end):**
  - an exact self-swap on identical code paths;
  - the whole cell through the general projector (I) matches direct assignment within the preflight tolerance;
  - S_full matches 003 / 001 for the pair within tolerance;
  - colour distributions, flags and fidelity inspected by hand.
- **Launch:** design commit → preflight → smoke → **PI go** → run.

## 10. Outputs

- `outputs/004_span_assay/` (raw);
- `results/004_span_assay/{manifest_pairs, arm_manifest, preflight, smoke_checks, analysis/*}`;
- the report, with fixed-seed random qualitative examples beside the median, extreme and failed cases;
- 001–003 artifacts untouched.

## 11. Prerequisite (infrastructure)

**Resolved 2026-10-03:** the PI made the dataset public. `outputs/` (43 files, 232 MB) was uploaded with `MANIFEST.json` (sizes, sha256, repo commit, registry hash); 3 sample files were retrieved and their hashes verified. The note below is kept for history: **the raw-output mirror was not complete** (checked 2026-10-03). The private HF dataset `senku21x/causal-concordance-outputs`
holds only a README, because the account's private storage is full; `outputs/` is now 222 MB. An off-instance copy
needs a PI decision on the destination: free or upgrade the HF storage, or Git LFS in the private GitHub repo.

## 12. Changes from v1

- Project framing.
- 33-pair manifest frozen.
- Start-only and answer-only descriptive arms.
- PCA comparator (unscaled and dose-matched).
- Paired-rotation necessity control, replacing v1's norm-matched perturbation.
- All-colour full-sequence scoring with every continuation.
- Composition diagnostics and target-directed qualification.
- Two warning flags.
- Descriptive control statistics only; no per-item significance.
- Unclipped recovery; paired differences.
- A 0.05 material-disagreement tolerance.
- Coexisting interpretations.
- Preflight and the import-boundary work.
- Deferred: L44, a second pair per table, L2, leave-one-table-out fits, label-shuffled spans, dose sweeps.
- Each fit uses its own mean: the inherited `Directions.fitmean` reads "both" and is not reused.
