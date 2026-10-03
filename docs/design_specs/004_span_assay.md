# 004 — Span assay check at `:` L42 (is the 1-D label assay too narrow, unstable, or right?) — design v1, DRAFT for PI approval

*2026-10-03. Dev only. Reader-free. A small methods check that follows the external audit of 001–003
(`docs/reviews/2026-10-02_codex_audit_001_003/`, §8). It assigns **no labels**: it decides how labels should be built
before any reader is scored. Reader scoring against labels moves from "004" to 005.*

## 1. Question

003's single reader-free z direction at `:` L42 carries a small, mostly positive share of the pair difference (median
0.034; positive in 84% of pairs) against a whole-cell share of 0.48. At the start token, where the start colour
matters by construction, the start direction recovered only a third of the cell (0.076 of 0.23). And the per-pair
effects were unstable across disjoint fits (r = 0.41). Three explanations make different predictions:

| Explanation | Predicts |
|---|---|
| **Too narrow:** content lives in a multi-dimensional span, not one direction | span patches recover much more than 1-D; stable across fits |
| **Not isolated:** the influence is carried jointly by start / z / answer content, or by a format the z estimator misses | the joint content span recovers much, the z span little |
| **Interaction:** span and complement effects are not additive | S_span + S_complement differs strongly from S_full |
| **Estimator instability:** the directions are too noisy to define a label | effects differ across disjoint fits even for spans; the start-token control is still poorly recovered |
| **Generic perturbation:** the 003 necessity result reflects edit size, not content | norm-matched ablations match the content ablation |

## 2. Material

- **Pairs:** one gated dev brew start pair per table, drawn by seed from the 82 used in 003: **33 pairs, 33 distinct
  tables**, so pairs are independent at the table level. A second pair per table where one exists (up to 33 more) is a
  secondary within-table check.
- **Cells:**
  - `:` at **L42** (primary);
  - the start-colour token at **L40** (positive control, content = start);
  - `:` at L44 (secondary, with the same edits as L42 if budget allows; reported separately).
- **Directions:** the same SEC captures, folds and least-squares fit as 003 (§3), reused unchanged. Each pair uses fits
  that exclude its table.
- **Regime:** SEC (true float32 from block 40). Writes go through float32 hooks, with fidelity measured on the
  realized write (as in 003).

## 3. Edits (all at one cell; recipient = member, donor = partner; Δ = h_donor − h)

| Edit | Definition | Rank |
|---|---|---|
| **z 1-D** | 003's swap: h + (û·Δ) û | 1 |
| **z span** | h + P_Z Δ, where P_Z projects on the span of the fitted z-colour contrasts {B_z[c] − B_z[ref]} (orthonormalized; colours with ≥ 2 occurrences in the fit set) | ≤ 11 |
| **joint content span** | h + P_J Δ, J = span of the start, z and answer contrasts | ≤ 33 |
| **z-span complement** | h + (I − P_Z) Δ: everything in the pair difference except the z span | 5120 − k_Z |
| **joint complement** | h + (I − P_J) Δ | 5120 − k_J |
| **whole cell** | h_donor (re-measured here, not taken from 001) | 5120 |

At the start token, the same set is run with the **start** span in place of the z span.

**Controls:**
- **Rank-matched random subspace** (16 draws per edit): h + P_R Δ, with R a uniformly random subspace of the same rank
  as the span. This measures how much of the pair difference any k-dimensional subspace carries.
- **Norm-matched random direction** (16 draws per edit): h + ‖P_S Δ‖ · r, with r a random unit vector. This is a
  perturbation of the same size that carries no pair content.
- **Random complement** (8 draws): h + (I − P_R) Δ for a random R of rank k_J. It is the complement-side counterpart.

## 4. Necessity (norm-matched, damage logged)

For the z 1-D direction, the z span and the joint span (and the start span at the start token), clamp the span to its
fitting-set mean in **both** runs: h′ = h − P_S (h − μ_S). Then G_abl = m(partner′) − m(member′) and
N = 1 − G_abl / G.

Controls for each, 16 draws, in both runs:
- (a) **rank-matched random subspace clamped to its own mean;**
- (b) **norm-matched random perturbation:** h + ‖P_S (h − μ_S)‖ · r, the same edit size as the real clamp on that run.

Damage (§5) is logged on **both** ablated runs.

## 5. Measurements on every run

- The margin m: full sequence, LSE over the accepted spellings, every token scored. S = Δm / G.
- **The full-sequence 12-colour answer distribution**, every colour scored over all of its spellings (including the
  multi-token " Teal"), clean and edited. This replaces 003's first-token mass (audit §9).
- Non-colour mass at the answer position; Δlog P(a) and Δlog P(a′).
- Realized-write fidelity (ρ, κ).
- Edit norms: requested and realized.

## 6. Reliability

Every content edit in §3 (not the whole cell) is repeated under the **two disjoint single-fold fits** (as 003).
Reported per edit type:
- the per-pair r and median |ΔS| between the fits;
- principal angles between the two fits' spans;
- the same for necessity N.

## 7. Primary quantities and decision rules (fixed before any 004 result)

Per edit type, over the 33 primary pairs:
- **recovery** = median over pairs of S_edit / S_full (pairs with S_full < 0.10 excluded, and their number reported);
- **stability** = r between fits, and median |ΔS| relative to the median S;
- **interaction** I = S_span + S_complement − S_full, median and interval, reported for the z span and the joint span;
- **excess over controls** = S_edit minus the 95th percentile of its rank-matched and norm-matched random draws, per
  pair. Per-pair Monte Carlo p = (1 + #draws ≥ S) / 17. No pooled null (audit §5).

Branch rules, at `:` L42:

| Branch | Rule | Reading |
|---|---|---|
| A: too narrow | z-span recovery ≥ 0.5, z span beats both controls in ≥ 50% of pairs, stability r ≥ 0.7 | Single directions were too restrictive; validate the z span's specificity next |
| B: not isolated | joint-span recovery ≥ 0.5 and z-span recovery < 0.25 | The influence is not isolated by the z estimator. **Not** "z unused" |
| C: interaction | \|median I\| ≥ 0.15 for the z span or the joint span | Additive attribution is inadequate; report the interaction |
| D: unstable | stability r < 0.5 for the z span or the joint span, **or** start-span recovery at the start token < 0.5 | Improve the fitting support or regularization before using the assay as a reference |
| E: generic | the z-span or z 1-D clamp does not exceed the norm-matched perturbation's q95 in ≥ 50% of pairs | Retract 003's necessity interpretation; keep the swap effects |

Branches are not exclusive; every applicable one is reported. If none applies, the assay is partially adequate. The
report then states which quantities clear which thresholds.

## 8. Predictions (priors; written before any 004 run)

| # | Prediction | Prior |
|---|---|---|
| S1 | z-span recovery at `:` L42 ≥ 0.25 | 45% |
| S2 | Joint-span recovery at `:` L42 ≥ 0.5 | 40% |
| S3 | Start-span recovery at the start token L40 ≥ 0.5 | 55% |
| S4 | Span effects are more stable than 1-D effects (r higher by ≥ 0.2) | 60% |
| S5 | \|median I\| ≥ 0.15 for the joint span | 35% |
| S6 | The z 1-D clamp does not beat the norm-matched perturbation in ≥ 50% of pairs (branch E) | 50% |

## 9. Gates

- **G1 fidelity:** ρ ∈ [0.9, 1.1], κ ≥ 0.99 on every write.
- **G2 engine (smoke, 2 pairs):**
  - a self-swap is bitwise identical to clean;
  - the whole-cell patch reproduces 001's S for the pair to 1e-6 (the same regime and sources);
  - a full-span patch with rank 5120 equals the whole cell bitwise.
- **G3 span sanity:**
  - the rank of each fitted span is reported;
  - held-out z-class separation by the projection on the z span is at least as good as the 1-D direction;
  - principal angles between the z, start and answer spans are reported, and overlap is flagged if the smallest
    angle is below 45°.

## 10. Budget

Per pair at `:` L42:
- content edits: 6;
- reliability: 10;
- swap controls: 3 spans × (16 + 16) + 8 = 104;
- necessity: 3 spans × (1 + 32) × 2 runs = 198;
- start-token block: about 60.

That is about 380 runs; with " Teal" continuations about 420 × 0.12 s ≈ 50 s per pair, so about **30 min for 33
pairs**. Adding the secondary second-pair-per-table set and L44 brings it to about 1.5 h. GPU only; no API cost.

## 11. Outputs

- `outputs/004_span_assay/` (raw);
- `results/004_span_assay/analysis/{summary.json, per_pair.json, qualitative.json}`;
- `docs/reports/004_span_assay.md`, with qualitative pairs: the colour distribution before and after the z span, the
  joint span and the complement for the largest, median and smallest pairs.

## 12. What 004 cannot do

It assigns no labels and scores no reader. A large span effect is not truth by itself: a more flexible edit can move
the answer more without isolating content (audit §8). It is one model, brew, dev, 33 tables, and one cell plus a
control.

## 13. Changes log

*(none yet)*
