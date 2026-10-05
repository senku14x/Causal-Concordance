# 006 — Do the verbalizers' likelihoods track z at the brew `:` cell? (design v1, pre-registered before any run)

**Status.**
- PI go 2026-10-05: "run a 006 but on a separate branch … not vibecoded".
- Branch `006-reader-likelihood`, off `005-amended-full-run` @ ff7c322.
- `[PI]` prior fields are left blank. The agent's priors below are committed before any 006 GPU run.
- Dev split only, as 001–004 (AMENDMENT_02 governs any test-split use).

## 1. Question and why

- **002 found:**
  - the verbalizers rarely *claim* the intermediate colour z in its role at the `:` cell (NLA 5%, Oracle 0–3%);
  - the token lenses' colour scores *track* z within tables (J_NP cosine L40: 48 of 48 tables).
- **005 found:** naming is a blunt reader measure. A reader's next-token likelihood at a forced answer position responds where its sampled text does not (§16.7, full run P11′).
- **006 asks:** scored by likelihood, as the lenses were, do the NLA and the Oracle track z at the `:` cell?
  - **If yes:** 002's "verbalizers miss z" becomes "they register z but do not say it".
  - **If no:** 002's result stands with a sensitive instrument.

## 2. Material (frozen)

- **Activations:** the saved 001 captures `outputs/001_leverage_profile/run/captures_brew.npz`.
  - Shape [281 renders, 3 cells (`emit_asst`, `emit_think`, `emit_colon`), 5 layers (40, 42, 44, 56, 60), 5120], float32.
  - These are SEC-regime captures, the ones 002's lens scores used.
  - The reader-input convention is the stock forward. 002's bridge measured stock vs SEC cos ≥ 0.99985, and that deviation is accepted and stated.
- **Renders:** the 48 member and 183 partner renders of the 48 dev tables (`captures_brew_meta.json`). Members and partners differ in the start colour, so z and the answer differ. The 50 "substituted" renders are not used.
- **Colours:** the 12 WorkspaceBench colours. Spellings " c" and " C".
  - " Teal" is two tokens. Every token of every spelling is scored (the non-negotiable), so nothing is skipped.

## 3. Readers and reads

| Reader | Layers | Note |
|---|---|---|
| NLA `nla-rl` iter_000400 (`readers.NLArl`; the project's NLA since 005) | L42 | 002's verbalizer readouts came from `nla-L42`, a different checkpoint, so comparisons with 002's claim rates cross checkpoints |
| Oracle `olens_s3d_rl600` (`readers.OracleLens`) | L40, L44 (primary); L56, L60 (secondary) | on-contract layers present in the captures |

- **Cells:** `emit_colon` is primary; `emit_asst` and `emit_think` are secondary.
- **Forced answer position:** the reader's answer is teacher-forced to start with a fixed prefix. The colour word is then scored as a continuation.
  - Primary prefixes (validated on planted concepts in 005): Oracle `-`; NLA `<explanation>\nThe text is about`.
  - Secondary prefixes: Oracle `- The colour is`; NLA `<explanation>\nThe colour is`.
- **Score:** LL_x(c) = log-sum-exp over c's spellings of the full-sequence log-probability of the spelling given activation x. Every token of the spelling is scored, from float32 logits computed from the final-norm output.

## 4. Statistics

**Tracking (002's lens statistic, reused exactly).** Per table and target colour role t ∈ {z, start, answer}:
- take every partner p whose t differs from the member m's;
- D_t = [LL_m(t_m) − LL_m(t_p)] − [LL_p(t_m) − LL_p(t_p)];
- average over those partners, giving one value per table;
- report the mean over the 48 tables with a 95% cluster bootstrap CI (10,000 draws, seed 20261004), and the sign counts (+ / − / 0).

**Forced choice (absolute).** On member renders:
- the share of tables where z has the highest LL among the 5 candidates (start, z, answer, and two off-path colours drawn by 002's seeded rule). Chance is 0.2.
- Also for start and answer.

**Controls.**
- **Shuffled:** D_t recomputed with table k's colours and table k+1's activations (002's derangement). Expected ≈ 0.
- **No-activation reference:** LL on the dev-mean activation for that cell and layer.

**Lens comparison (descriptive).** 002's lens tracking counts, from `results/002_claim_scoring/analysis/scores.json`, shown next to the readers'.

## 5. Gates (checked before any outcome is read)

| Gate | Rule |
|---|---|
| G-score1 | The scorer agrees with an independent computation: unit tests on CPU, and on GPU, for single-token spellings, `score_seq` vs 005's `score_next`: **max ≤ 0.25 nats and median ≤ 0.05 (amended v1.1, see below)** |
| G-score2 | float32 log-probs vs the reader's own logits: max \|difference\| ≤ 1.0 nat (as in 005) |
| G-pc | Positive control, per reader-layer: on natural "Tell me about {colour}" activations at that layer (stock forward, last token), 12-way forced-choice ID among the colours ≥ 0.5 (chance 0.083). A reader-layer that fails is reported as uninformative |

## 6. Pre-registered rules and priors

| ID | Rule | Agent prior | `[PI]` |
|---|---|---|---|
| T1 | **NLA L42 tracks z at `:`**: mean D_z CI above 0 | 0.55 | |
| T2 | **Oracle tracks z at `:`** (L40 or L44, each tested; Holm over the two): CI above 0 | 0.60 | |
| T3 | **Start is not tracked** at `:` for either reader (CI includes 0 or lies below), as for the lenses | 0.55 | |
| T4 | Forced-choice ID of z on member renders at `:` > 0.4 (any primary reader-layer) | 0.25 | |
| T5 | G-pc passes: Oracle L40/L44 · NLA L42 | 0.85 · 0.65 | |

**Family:** T1 and T2 are the primary family (Holm across NLA L42, Oracle L40, Oracle L44). Everything else is
secondary or descriptive.

**Reading:**
- T1 or T2 supported, with the shuffled control ≈ 0: the reader's likelihood carries item-specific z information at that cell. Wording: "decodable from the reader's likelihood", not "the reader represents z".
- Not supported, with G-pc passing: no evidence that the reader registers z there, with a demonstrated-sensitive instrument.
- G-pc failing: uninformative for that reader.

## 7. Cost, code, outputs

- **Reads:** 231 renders × 3 cells × (1 NLA + 4 Oracle layers) × 24 spellings ≈ 83k scored continuations, forward-only, plus 12 positive-control activations per layer. Estimate ≈ 15–25 GPU minutes. No subject-model writes (G0 not applicable).
- **Code:**
  - `scripts/common/readers.py` `score_seq` (Oracle and NLA): full-sequence continuation scoring, with its core in a pure function unit-tested on CPU;
  - `scripts/006_reader_likelihood/reader_likelihood.py` (stages: preflight, run);
  - `reader_likelihood_analysis.py`.
- **Outputs:** `outputs/006_reader_likelihood/<run_id>/`; results in `results/006_reader_likelihood/analysis/summary.json`.

## 8. What this cannot show

- Likelihood tracking is decodability through the reader, not the reader's claim, and not use by the model.
- The colour words are 12 fixed tokens. A reader may register "a colour change" without its tokens following z.
- SEC captures, not stock (cos ≥ 0.99985).
- The dev split only, one family (brew), one cell type primary.

## Amendment v1.1 (2026-10-05, after G-score1 stopped the first launch, before any outcome was read)

- **What happened.** The first launch stopped at G-score1. `score_seq` and `score_next` differed by up to 0.10 nats against the 1e-3 tolerance.
- **The diagnostic** (4 activations, Oracle L40, `-` prefix), |difference| on single-token spellings:

| Comparison | Median | Max |
|---|---|---|
| `score_seq` vs `score_next` | 0.023 | 0.10 |
| `score_next` with batch 4 vs batch 1 (same scorer, same length) | 0.029 | 0.10 |
| `score_seq` padded vs unpadded | 0.020 | 0.12 |

- The scores range from −13 to −1 nats, so a position or target bookkeeping error would show as nats, not hundredths.
- **Reading.** The discrepancy is the bf16 floor for a change of batch shape or sequence length, documented in CLAUDE.md's pitfalls. My 1e-3 tolerance ignored that floor.
- **Change.** G-score1 passes if max |difference| ≤ 0.25 nats (the documented floor) and the median ≤ 0.05.
- **Effect on the statistics.** The bf16 noise adds variance to D_t, not bias. The bookkeeping itself stays covered by the exact CPU tests.
