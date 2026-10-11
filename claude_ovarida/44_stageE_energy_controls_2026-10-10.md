# 44. 012 Stage E §17: energy-matched removal controls and flip overlaps (2026-10-10)

**Scope.**
- Design: `docs/design_specs/012E_ar_concept_swaps.md` §17 (`60dc326`, written before this run). EXPLORATORY, POST
  HOC.
- Setting: the same 82 items, regime and sizes as report 43.
- Numbers: every number comes from `results/012_reader_concordance/analysis/stageE_step4.json` (`energy`, `overlap`),
  rounded half-up by script.
- Determinism: AR_full@J re-run in the same process reproduces report 43's Δm exactly (maximum abs difference 0.0).

## Bottom line

1. **The J result survives an energy-matched control.**
   - AR_emrem removes a random subspace sized so that it takes out the same share of the AR difference's energy as
     007's dense J subspace. Median removed share 0.579, against J's 0.579; cos to the original 0.649.
   - **It keeps the whole effect, and more:** 45/82 flips, median Δm +8.43. AR_full gets 37/82 and +5.25; J removal
     gets 0/82 and +0.42.
   - Pre-stated reading: **J-specific**, at both sizes and in every subset. The effect does not break under any removal
     of this size; it breaks under J removal.
2. **The Δ-matched PCA-k arm could not be matched.**
   - The cross-fitted PCA basis of clean activations has rank about 618. Even at full rank it captures a median of only
     0.322 of the AR difference's energy, against J's 0.579. So the arm used the maximum rank on all 984 item-layers
     (flagged "unmatched").
   - It removes less than J, so it is not an energy match. Its "J-specific" reading (31/82 flips, median +3.34) only
     shows that removing the top-variance activation subspace, which overlaps J at 0.28 against 0.20 random, does not
     break the handle.
3. **Overlap.**
   - All 37 of AR_full@J's flips are also J-row flips. The J-row swap has 9 more.
   - 27 of the 37 are Japan targets.
   - AR_full@J and the answer-concept control AR_answer@J flip 31 items in common; each flips 6 the other doesn't; 39
     items flip under neither.

## 1. §17 readings

| Arm @ size | Subset | n | X − AR_Jrem, Δm [95% CI] | Median Δm: X / full / Jrem | Flips: X / full | Reading |
|---|---|---|---|---|---|---|
| AR_emrem@J | all | 82 | 9.27 [5.00, 12.51] | 8.43 / 5.25 / 0.42 | 45 / 37 | J-specific |
| AR_emrem@J | named_place | 49 | 14.40 [10.75, 16.61] | 17.25 / 13.05 / 0.74 | 44 / 37 | J-specific |
| AR_emrem@J | other_bridge | 33 | 1.66 [0.80, 2.87] | 0.70 / 0.41 / 0.06 | 1 / 0 | J-specific |
| AR_emrem@J | japan_target | 27 | 17.20 [11.29, 19.08] | 18.90 / 15.96 / 0.86 | 27 / 27 | J-specific |
| AR_emrem@J | not_japan | 55 | 5.38 [3.11, 8.03] | 2.48 / 1.66 / 0.19 | 18 / 10 | J-specific |
| AR_emrem@J | covered | 51 | 13.72 [9.38, 16.19] | 16.23 / 12.82 / 0.73 | 42 / 36 | J-specific |
| AR_emrem@J | uncovered | 31 | 1.96 [1.00, 3.34] | 0.70 / 0.41 / 0.06 | 3 / 1 | J-specific |
| AR_emrem@J | rescued_concept | 13 | 13.17 [8.24, 18.18] | 15.27 / 11.21 / 0.86 | 10 / 9 | J-specific |
| AR_pcarem@J | all | 82 | 5.85 [2.39, 8.54] | 3.34 / 5.25 / 0.42 | 31 / 37 | J-specific |
| AR_pcarem@J | named_place | 49 | 9.27 [4.94, 11.64] | 11.52 / 13.05 / 0.74 | 31 / 37 | J-specific |
| AR_pcarem@J | other_bridge | 33 | 0.77 [0.44, 1.23] | 0.43 / 0.41 / 0.06 | 0 / 0 | J-specific |
| AR_pcarem@J | japan_target | 27 | 12.52 [6.18, 14.61] | 13.45 / 15.96 / 0.86 | 24 / 27 | J-specific |
| AR_pcarem@J | not_japan | 55 | 2.57 [1.30, 4.12] | 1.19 / 1.66 / 0.19 | 7 / 10 | J-specific |
| AR_pcarem@J | covered | 51 | 8.87 [4.39, 11.37] | 11.38 / 12.82 / 0.73 | 30 / 36 | J-specific |
| AR_pcarem@J | uncovered | 31 | 0.88 [0.51, 1.40] | 0.43 / 0.41 / 0.06 | 1 / 1 | J-specific |
| AR_pcarem@J | rescued_concept | 13 | 7.74 [3.40, 12.39] | 9.84 / 11.21 / 0.86 | 6 / 9 | J-specific |
| AR_emrem@C | all | 82 | 7.85 [3.97, 10.81] | 6.25 / 3.77 / 0.33 | 39 / 30 | J-specific |
| AR_emrem@C | named_place | 49 | 12.38 [8.45, 14.76] | 13.78 / 10.34 / 0.62 | 39 / 30 | J-specific |
| AR_emrem@C | other_bridge | 33 | 1.11 [0.63, 1.67] | 0.74 / 0.35 / 0.22 | 0 / 0 | J-specific |
| AR_emrem@C | japan_target | 27 | 14.17 [8.07, 16.14] | 16.47 / 11.43 / 0.62 | 23 / 18 | J-specific |
| AR_emrem@C | not_japan | 55 | 4.74 [2.32, 7.58] | 2.55 / 1.51 / 0.30 | 16 / 12 | J-specific |
| AR_emrem@C | covered | 51 | 11.95 [7.91, 14.38] | 13.13 / 10.00 / 0.64 | 37 / 29 | J-specific |
| AR_emrem@C | uncovered | 31 | 1.09 [0.60, 1.76] | 0.74 / 0.35 / 0.22 | 2 / 1 | J-specific |
| AR_emrem@C | rescued_concept | 13 | 13.09 [9.01, 17.35] | 14.23 / 12.63 / 0.91 | 11 / 10 | J-specific |
| AR_pcarem@C | all | 82 | 4.67 [2.26, 6.56] | 2.98 / 3.77 / 0.33 | 23 / 30 | J-specific |
| AR_pcarem@C | named_place | 49 | 7.35 [4.52, 9.33] | 7.84 / 10.34 / 0.62 | 23 / 30 | J-specific |
| AR_pcarem@C | other_bridge | 33 | 0.69 [0.41, 1.02] | 0.47 / 0.35 / 0.22 | 0 / 0 | J-specific |
| AR_pcarem@C | japan_target | 27 | 8.01 [4.06, 9.76] | 8.79 / 11.43 / 0.62 | 12 / 18 | J-specific |
| AR_pcarem@C | not_japan | 55 | 3.03 [1.32, 5.18] | 1.31 / 1.51 / 0.30 | 11 / 12 | J-specific |
| AR_pcarem@C | covered | 51 | 7.14 [4.35, 9.07] | 7.52 / 10.00 / 0.64 | 22 / 29 | J-specific |
| AR_pcarem@C | uncovered | 31 | 0.61 [0.37, 0.94] | 0.47 / 0.35 / 0.22 | 1 / 1 | J-specific |
| AR_pcarem@C | rescued_concept | 13 | 8.01 [4.51, 12.10] | 8.26 / 12.63 / 0.91 | 7 / 10 | J-specific |

**Diagnostics** (medians over 984 item-layers):
- J share of u: 0.579;
- emrem removed share: 0.579, cos to u 0.649;
- PCA-k: rank 618, removed share 0.322, cos to u 0.823;
- PCA basis overlap with J: 0.280 (random 0.20).

**Unexplained.** Removing random energy makes the edit *stronger*: 45 vs 37 flips at J size, 39 vs 30 at C size.
Report 43 saw the same with the 1,024-dim random removal (41 vs 37). Random removal leaves u's J share unchanged in
expectation, and the edit size is fixed per (layer, position), so neither renormalisation nor the J share explains it.
It is recorded as unexplained.

## 2. Flip-set overlaps (report 43 item 2)

| Pair | Both | Only first | Only second |
|---|---|---|---|
| J-row & AR_full@J | 37 | 9 | 0 |
| J-row & AR_full@C | 28 | 18 | 2 |
| J-row & AR_answer@J | 36 | 10 | 1 |
| J-row & C_white | 36 | 10 | 4 |
| AR_full@J & AR_full@C | 26 | 11 | 4 |
| AR_full@J & AR_answer@J | 31 | 6 | 6 |
| AR_full@J & C_white | 33 | 4 | 7 |

- **AR_full@J vs AR_answer@J:**

  | | AR_answer flips | AR_answer does not |
  |---|---|---|
  | AR_full flips | 31 | 6 |
  | AR_full does not | 6 | 39 |

- **AR_full@J flips by target:** 27 of 37 on Japan targets.
- **J-row flips that AR_full@J misses** (9): fjords-capital, ikea-capital, lego-capital, windycity-state,
  zeus-country, christmas-season, nile-desert, southpole-bird, valentines-month.

**Readings** (observations):
- The Oracle's concept differences flip a *subset* of the items J rows flip, at J size, and never an item J rows can't.
- The bridge edit and the answer edit flip largely the same items (31 of 37 each). That is consistent with both acting
  through the same answer-facing route on these items. It is not a test of it.

## 3. What this changes in report 43

- **§17 read.** The J statement in report 43 can now be scoped: under an energy-matched random control, removing J
  removes the effect and removing the same energy at random does not.
- **The PCA-k arm could not be energy-matched.** It adds only that removing the top-variance activation subspace does
  not break the handle.
- The corrections the PI asked for are applied in report 43 itself (dated).

## Next

The Stage I/J delayed-question design (reconstructor edits at clue positions only, two relations per held state) is
next, as the PI set out. It is not drafted yet. I'll draft it on your word.

## Caveat added 2026-10-11 (012E §18, report 49)

- **The arms were matched at the clamp target, not in effective push.** A per-layer clamp accumulates the part of
  its direction that persists across layers.
- On 10 seeded items, at L47:
  - AR_emrem pushes about 1.6× AR_full's amount along u (≈ 1/cos(u′, u));
  - AR_rrem about 1.13×;
  - AR_Jrem about 0.7×, with no accumulation.
- The persistent part of u appears to lie in J.
- **So the pre-stated "J-specific" reading above compares arms at unequal effective push.** It stands as "removing J
  removes the effect at matched clamp size". Whether a J-removed direction would act at a matched effective push is
  untested.
- The larger effect of the random-removal arms is explained by the accumulation
  (`results/012_reader_concordance/analysis/stageE_clampcheck.json` `summary`).
