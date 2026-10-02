# 003 — Reader-free causal labels at the brew answer cells

*Qwen3.6-27B · WorkspaceBench brew · dev only · design `docs/design_specs/003_labels.md` (v1) · 2026-10-02 ·
SEC regime (true float32 from block 40).*

## Scope

- **Pairs:** 82 gated brew dev start pairs (33 tables). Recipient = member, donor = partner.
- **Directions:** fitted on the 231 natural dev renders (48 tables), cross-fitted over 3 table folds, so no pair
  sees its own table.
- **Cells:**
  - the three emission cells at L40/42/44;
  - `:` at L60;
  - the start-colour token at L40 (positive control).
- **No reader is used anywhere.** The label code loads no reader module (checked transitively).
- **Wording:** "used" = the pair difference along that content direction demonstrably moves this answer. It is not
  workspace membership, and it is not the model's natural route.

**Gate G4 failed, so these labels are descriptive only.** They are not used to score readers (design §6; SPEC §6
reliability gate).

## Provenance

| Stage | Output | Notes |
|---|---|---|
| capture | `outputs/003_labels/captures/start_token.npz` | 231 renders, SEC, start-colour token |
| directions | `outputs/003_labels/directions.npz`, `analysis/directions_validity.json` | 3 folds × {both, single, single} fits; full-rank least squares (34/34) |
| smoke | `analysis/smoke_checks.json` | 3 pairs; all checks pass |
| run | `outputs/003_labels/run/edits.jsonl` | 82 pairs, 37,299 edited forwards, 92 min on GPU |
| analysis | `analysis/{summary.json, labels.json}` | — |

Design commit 6a2a097, before any run; gates committed in 9ee4467, before the main run.
- The run manifest records 9ee4467 as **dirty**: the analysis stage was added to `labels.py` while the run was in
  progress.
- The running process had loaded the code as committed at 9ee4467, and the edit added analysis functions only. So
  the measurement code that ran is the committed code; the manifest alone cannot show that.

## 1. Gates

| Gate | Result |
|---|---|
| G1 fidelity (every edit) | **Pass:** 0 / 37,299 failures; ρ 0.9996–1.0024, κ ≥ 0.998 |
| G2 engine (smoke) | **Pass:** clean margins identical to 001; self-swaps bitwise identical to clean |
| G3 direction validity | **Pass:** held-out separation of z contrasts, median AUC 1.00 at every emission cell L40–44 (0.97 at L60; 0.85 at the start token) |
| G4 label reliability | **Fail** (details below) |

**G4 detail.** At `:` L42–44, the "used" call for z under two disjoint fitting subsets (one fold each, about 16
tables):
- κ = **0.37**;
- raw agreement 0.91;
- "used" in 7% and 8% of pairs under the two fits;
- r = 0.41 between the per-pair S values of the two fits.

At the assistant newline, r = 0.68, κ = 0.32. With "used" this rare, κ is low even at high agreement. But r = 0.41
shows that the per-pair effect really does depend on which tables the direction is fitted on.

## 2. What carries the leverage? Median S by content direction (82 pairs; 95% bootstrap over tables)

| Cell | z direction | start direction | answer direction | Whole cell (001) | Null q95 (z) |
|---|---|---|---|---|---|
| `:` L40 | 0.031 [0.023, 0.044] | 0.009 | 0.003 | 0.35 | 0.020 |
| `:` L42 | 0.034 [0.026, 0.049] | 0.012 | 0.003 | 0.48 | 0.021 |
| `:` L44 | **0.050 [0.034, 0.065]** | 0.010 | 0.006 | 0.50 | 0.026 |
| `:` L60 | 0.004 | 0.006 | **0.62 [0.56, 0.64]** | 0.74 | 0.018 |
| assistant newline L40 / 42 / 44 | 0.026 / 0.024 / 0.035 | 0.002 / 0.004 / 0.008 | 0.004 / 0.003 / 0.010 | 0.29 / 0.29 / 0.32 | ≈ 0.02 |
| `</think>` newline L40 / 42 / 44 | −0.003 / −0.001 / −0.000 | ≈ 0 | ≈ 0 | −0.00 / 0.01 / 0.03 | ≈ 0.005 |
| start token L40 | 0.007 | **0.076 [0.056, 0.096]** | 0.001 | 0.23 | 0.008 (start: 0.022) |

- **At `:` and the assistant newline, L40–44:** the z direction is the only content direction whose median clearly
  exceeds its random-direction null. It is larger than the start and answer directions at every layer. In 46–53 of
  82 pairs it is the largest of the three.
- **But it carries only 3–5% of the gap**, against 29–50% for the whole cell. About 0.21–0.42 of the gap ("other")
  lies outside all three single content directions.
- **At `:` L60** the answer direction carries most of the cell (0.62 of 0.74), and z does nothing.
- **Positive control:** the start direction at the start token carries 0.076 of that cell's 0.23. That is clearly
  above null but under half the cell, so a single linear content direction recovers only part of the leverage even
  where the content is known to matter.
- **Direction geometry:**
  - the three content directions are nearly orthogonal (median pairwise cosine about 0);
  - the naive centroid-difference z direction is almost the same as the decorrelated one (cos 0.94–0.95 at the
    emission cells) and gives the same S. With 48 tables, decorrelation changes little.

## 3. Labels

| Cell | z used | z used + necessary | z decodable, not needed | z unresolved | start used | answer used |
|---|---|---|---|---|---|---|
| `:` L40 | 11% | 11% | 2% | 87% | 2% | 0% |
| `:` L42 | 15% | 15% | 0% | 85% | 5% | 0% |
| `:` L44 | **18%** | **18%** | 0% | 82% | 7% | 1% |
| assistant newline L40 / 42 / 44 | 10 / 9 / 17% | same | 0–1% | 83–91% | 0–4% | 1–2% |
| `</think>` newline L40 / 42 / 44 | 0% | 0% | **100 / 87 / 70%** | 0–30% | 0% | 0% |
| `:` L60 | 5% | — | 0% | 95% | 6% | 44% (52% of these swaps flagged for damage) |
| start token L40 (start content) | — | — | — | — | **28%** | — |

- **Necessity.** Clamping the z direction to its fitting-set mean in both runs shrinks the gap by a median 3–5% at
  `:` L40–44. The 95th percentile of random clamps is 0.001. Every "used" z label at those cells also passes
  necessity.
- **Concentrated in a few tables.** At `:` L44, 5 of 82 pairs have S_z > 0.2, and "used" pairs come from 11 of 33
  tables. Table 0383 has a median S_z of 0.36 over 4 pairs; most tables sit near 0.
- **At the `</think>` newline the whole cell carries no leverage**, while z is strongly decodable there (probe 0.79–
  0.84). That gives the cleanest **decodable-not-needed** label of the project so far, for natural pair content.
- **The damage flag at `:` L60** catches answer swaps that push mass to non-colour tokens (e.g. "other" rising from
  0.07 to 0.29). Those labels stay unresolved.

## 4. Predictions

| # | Prediction | Prior | Outcome |
|---|---|---|---|
| L1 | z swap at `:` L42/44 used in ≥ 50% of pairs | 55% | **False** (15%, 18%) |
| L2 | `:` L40–44 median S_z > median S_start | 65% | **True** (0.031–0.050 vs 0.009–0.012) |
| L3 | `:` L42–44 median S_z > median S_answer | 55% | **True** (0.034–0.050 vs 0.003–0.006) |
| L4 | z necessity at `:` L40–44 above the null in ≥ 30% of pairs | 35% | **False** (11–18%) |
| L5 | G4 κ ≥ 0.7 for z at `:` L42–44 | 50% | **False** (0.37) |
| L6 | Start swap at the start token used in ≥ 70% | 75% | **False** (28%) |
| L7 | `:` median S_z at L60 > at L44 | 50% | **False** (0.004 vs 0.050) |
| L8 | The naive z direction gives a larger S than the decorrelated one at `:` L42–44 | 60% | **False** (0.039 vs 0.034; 0.049 vs 0.050) |

## 5. Qualitative (`:` L44; colour mass at the answer position, clean → after the z swap)

| Pair | S_z | S_start | S_answer | S_full | Clean (top) | After the z swap | Label |
|---|---|---|---|---|---|---|---|
| 0383 → white | **0.56** | −0.01 | 0.02 | 0.66 | brown 0.79, other 0.10, orange 0.08 | brown 0.40, orange 0.23, gold 0.19 | used + necessary |
| 0383 → black | 0.38 | 0.21 | 0.01 | 0.60 | brown 0.79 | brown 0.51, orange 0.29 | used + necessary |
| 0383 → teal | 0.33 | 0.04 | 0.00 | 0.60 | brown 0.79 | brown 0.44, orange 0.31 | used + necessary |
| 0041 → gray (median) | 0.05 | −0.01 | 0.03 | 0.37 | gray 0.65, gold 0.10 | gray 0.60, black 0.11 | unresolved |
| 0312 → green | −0.01 | −0.00 | −0.03 | 0.60 | purple 0.87 | purple 0.86 | unresolved |
| 0377 → red | −0.03 | 0.00 | −0.00 | 0.25 | gray 0.81 | gray 0.80 | unresolved |

In 0383 the single z direction moves most of the cell's effect, toward the partner's answer (orange). In most pairs
it moves almost nothing, while the whole cell moves 0.25–0.60.

## 6. Limitations

- **G4 failed: the labels are descriptive only.** Directions fitted on about 16 tables vary enough that per-pair
  effects correlate only at r = 0.41 between fits. The primary directions (about 32 tables) are less noisy, but their
  reliability is not established.
- **Single linear directions only.** A large whole-cell effect with a small 1-D effect is "unresolved". The leverage
  may sit in a multi-dimensional or non-linear format, in content other than start, z and answer, or be spread across
  directions that differ by table. 003 cannot tell these apart.
- **Scope:** one model, brew dev, the forward direction only, provisional τ and ε′, 8 random directions per test.

## 7. Interpretation and next

Under these conditions:
- the intermediate at `:` and at the assistant newline, L40–44, has a **small, consistent, direction-specific** influence
  on the answer: above norm-matched random directions, above the start and answer directions, and necessary by
  clamping. It is 3–5% of the gap in the median and large in a few tables;
- at the `</think>` newline the intermediate is **decodable but not needed**;
- at L60 the answer content carries the cell;
- most of the emission cells' leverage at L40–44 is **not** captured by any single content direction.

**For the readers (descriptive, because of G4):**
- In 002 the J lenses tracked z at `:` *and* at the `</think>` newline, about equally (47–48 of 48 tables at both).
  At the first, z has a small demonstrated influence; at the second, it is decodable and not needed.
- So at these cells, lens z-tracking does not separate used from not-needed content. This is the W ≈ 0 pattern of
  SPEC §8 A2, and it is an observation, not a score.
- The verbalizers rarely claim z in either place.

**Next options (PI's choice):**
1. **Subspace labels:** swap the whole z subspace (11 dimensions), and a joint start + z + answer subspace, to measure
   how much of the cell's leverage is linearly reachable through content subspaces. This is the "used, non-1-D"
   bucket.
2. **DAS on a subset** as the second label method (SPEC §6).
3. **More fitting data for reliability:** more partner renders per table, or a ridge/shrinkage estimator, and then
   re-check G4 before scoring readers. Test-split tables stay held out.
4. **004 descriptive:** set the 002 reader outputs against these labels, as a descriptive W pattern only.

## 8. Breakages and silly mistakes

- The run manifest is dirty, because the analysis stage was written during the run (provenance note above). Run
  stages should be started from a clean tree.
- The analysis was dry-run on 2 pairs to catch code errors before the run finished. One known
weakness, now documented: κ is uninformative when one label dominates, so agreement and r(S₁, S₂) are reported
beside it.
