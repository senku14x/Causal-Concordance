# 42. 012 Stage E, step 4: parity, noise floors, class-mean lever and the 5-item smoke (2026-10-10)

**Scope.** Step 4 of `docs/design_specs/012E_ar_concept_swaps.md` as amended in §15 (`6a46d75`). EXPLORATORY. The
coverage rescue and every class-mean result are POST HOC. Step 4 stops here for the PI.
- **Numbers:** every number comes from `results/012_reader_concordance/analysis/stageE_step4.json`, rounded half-up by
  script, unless marked otherwise.
- **Runs:** all on Jan's `ModelRunner` (bf16, `BandEditor`, float32 final logits), band L36–47, every prompt position,
  prefill only, using both GPUs.

## Bottom line

1. **Our L36–47 J-row swap reproduces Jan's Stage A exactly.** It flips 46/82 (random 3/82), with 100% per-item flip
   agreement and identical generated text on every item. G0 is bitwise: the α = 0 edit changes no log-prob and no
   token.
2. **The noise floor is small.** The identity clamp moves the margin by a median of 0.001 nats (Jan's swap) or 0.000
   (our additive clamp); 90th percentile 0.04, maximum 0.13; the greedy output is unchanged on 87/87 items.
3. **The class-mean swap fails the lever rule.** It flips 10/51 covered items (20%; random 1/51); the rule needs ≥ 30%
   and ≥ 20 points above random. **Under fix 2, the primary therefore uses J-row-matched edit sizes.** The class-mean
   arm is reported as secondary and scoped to its covered set.
4. **Smoke (5 items, observations only).** Every arm runs and logs its sizes.
   - The full reconstructor vector moves the margin far above the random nulls (Δm +1.0 to +9.0 nats; 1/5 flips).
   - With J removed, it falls to about the J-complement null (−0.06 to +1.98).
   - Removing a random 1,024-dim subspace instead leaves it intact (+1.2 to +10.2).
   - The mismatched-concept control is about 0.
   - Your whitened coordinate swap flips 2/5.
   - All of this is at class-mean-matched sizes. At J-row sizes, the primary configuration, the J-removed vector gives
     +0.04 to +1.01, and the J-row-size nulls were not in the smoke (see "For the full run").

## 1. G-parity and G0

| | Ours (L36–47) | Stage A `j_swap_L36-47` |
|---|---|---|
| J-row swap flips (cohort) | 46 / 82 | 46 / 82 |
| Random control flips | 3 / 82 | 3 / 82 |
| Per-item flip agreement | 100% (J and random) | |
| Identical generated text (clean, J, random) | 100% / 100% / 100% | |

- G0: the α = 0 edit's maximum |Δ log-prob| is 0.0, and the greedy output is identical on every item.
- The J-row lever passes the lever rule at L36–47 (56%, 52 points above random). This is the stage's power reference.
- **Edit direction:** 68 of the 82 items are toward_tgt by the clean-coordinate diagnostic. All 46 flips are among
  them; the random control's 3 are as well.

## 2. Identity-clamp noise floors (all 87 evaluated items)

| Clamp | Median abs(Δm) | 90th pct | Max | Greedy unchanged |
|---|---|---|---|---|
| Jan's `SwapEdit` written back to its own clean coordinates | 0.001 | 0.041 | 0.129 | 87 / 87 |
| Our additive clamp with k = 0 | 0.000 | 0.037 | 0.129 | 87 / 87 |

These are the bf16-write floors for the two edit families in this regime. Every Δm below should be read against them.

## 3. Class-mean coverage after the rescue (POST HOC)

**Rescue.** It recovered 6 of the 9 named places: Brazil, Canada, China, Fuji, Olympus and Sydney. Denmark, Miami and
Nile still fell short, keeping 9, 8 and 8 prompts against a minimum of 10.

| | Covered | Total |
|---|---|---|
| Concepts | 52 | 89 |
| Items with both class means | 51 | 82 |
| … bridge is a named place | 46 | 49 |
| … of which countries | 41 | 42 |
| … bridge is anything else | **5** | **33** |
| … Japan target | 27 | 27 |

**The class-mean arm is effectively a named-place arm.** Of the 51 items it covers, 46 have named-place bridges and
27 have Japan targets. Every class-mean result is scoped to this set.

**Style control** (descriptive): the easy-prompt class mean against the original class mean for the same concept,
centred, final token.

| Layer | Japan | Egypt | Italy | Cross-concept median | Cross-concept max |
|---|---|---|---|---|---|
| L36 | 0.70 | 0.74 | 0.48 | −0.07 | 0.72 |
| L40 | 0.70 | 0.69 | 0.48 | −0.05 | 0.69 |
| L44 | 0.62 | 0.68 | 0.45 | −0.04 | 0.65 |

- Same-concept agreement is well above the cross-concept median.
- But it is only moderate (0.45–0.74), and the highest cross-concept cosine reaches the same level.
- **So the prompt style moves the class mean.** The 13 of 51 covered items that involve a rescued concept should be
  reported separately whenever class-mean results are used.

## 4. The class-mean lever rule (fix 2)

| Class-mean swap, covered items (n = 51) | Flips | Median Δm (nats) |
|---|---|---|
| Swap | 10 / 51 (20%) | +2.61 |
| True source / random target | 1 / 51 | +0.96 |
| Named-place bridges | 10 / 46 swap, 1 / 46 random | |
| Other bridges | 0 / 5 swap, 0 / 5 random | |
| Japan target / not Japan | 5 / 27, 5 / 24 | |
| Within the class-mean swap's own toward_tgt items (28) | 9 / 28 swap, 1 / 28 random | |

- **Verdict:** the rule fails (20% < 30%). The size reference for the primary therefore switches to the J-row swap
  (fix 2, fixed before this run).
- Within toward_tgt the swap reaches 32%. That subset is not the rule's set, so this is descriptive only.
- **Observation:** the model's own class-mean directions are a weaker lever on these prompts than its J rows: 10/51
  against 46/82 for J rows on the full cohort, 37/49 at L24–35 on named-place items (`stageA_splits.json`). This is not
  a like-for-like comparison: the sets differ and the class means are fit on generated prompts.

## 5. Edit sizes at every edited position

Realised ‖Δh‖ / ‖h‖ per layer: median over items of the median over prompt positions, with the read position in
brackets where logged.

| Arm (run) | L36 | L40 | L44 | L47 |
|---|---|---|---|---|
| J_L36-47 (parity, n 82) | 0.028 (0.056) | 0.007 (0.010) | 0.003 (0.003) | 0.003 (0.004) |
| rand_L36-47 (parity, n 82) | 0.076 (0.088) | 0.065 (0.104) | 0.065 (0.089) | 0.069 (0.082) |
| CM_swap (cmlever, n 51) | 0.096 (0.142) | 0.023 (0.027) | 0.015 (0.022) | 0.035 (0.054) |
| CM_rand (cmlever, n 51) | 0.088 (0.148) | 0.078 (0.128) | 0.075 (0.103) | 0.066 (0.111) |
| AR_full (smoke, n 5) | 0.066 | 0.011 | 0.013 | 0.029 |
| AR_Jrem (smoke, n 5) | 0.066 | 0.023 | 0.012 | 0.029 |
| AR_Jrem_Jsize (smoke, n 5) | 0.033 | 0.009 | 0.006 | 0.005 |
| AR_answer (smoke, n 5) | 0.066 | 0.012 | 0.011 | 0.028 |
| AR_mismatched (smoke, n 5) | 0.066 | 0.017 | 0.013 | 0.028 |
| rand_full_0 (smoke, n 5) | 0.066 | 0.092 | 0.042 | 0.056 |
| rand_Jcomp_0 (smoke, n 5) | 0.066 | 0.092 | 0.045 | 0.055 |
| C_white (smoke, n 5) | 0.039 | 0.017 | 0.010 | 0.006 |

**What the table shows.**
- **First edited layer.** Each arm is matched to its reference *at the clamp target*, per position and layer. At L36
  every additive smoke arm has exactly the reference's size (0.066).
- **Later layers.** The realised edit is whatever is needed to hold the coordinate at its target:
  - concept directions persist through the network, so little re-application is needed (about 0.01 at L40);
  - random directions decay, so the clamp keeps re-adding them (about 0.09 at L40).
- **So the random nulls receive more total perturbation than the arms they control for.** That makes them conservative
  nulls: a concept arm that beats them does so with less total edit.
- **J-row sizes.** The J-row swap's realised edit is small after L36 (0.003–0.007 from L40 on). The primary
  configuration (AR arms at J-row size) therefore edits much less than the class-mean-sized smoke arms.
- **Write fidelity.** The largest coordinate error after the bf16 write is about 0.1 or less (in coordinate units
  along unit directions) for every arm: maximum 0.102, for AR_Jrem_Jsize. Per-arm maxima are in the JSON.

## 6. Smoke (5 items drawn with seed 11 from the 51 covered items; class-mean-matched sizes)

Δm toward y_b in nats; `*` marks a flip (y_b in the greedy output).

| Arm | fjords-capital | basketball-players | giza-continent | southpole-bird | dual-photosynthesis-opposite | flips |
|---|---|---|---|---|---|---|
| J_L36-47 | 8.03* | 0.97 | 10.89* | 6.76* | 5.24 | 3/5 |
| rand_L36-47 | -0.86 | 0.96 | -1.22 | 1.91 | 1.24 | 0/5 |
| CM_swap | -1.24 | 1.78 | 0.19 | 0.82 | 5.18 | 0/5 |
| CM_rand | 1.39 | 0.53 | 0.96 | 0.28 | 4.25 | 0/5 |
| CM_swap_Jrem | -0.65 | 0.12 | 0.18 | 0.85 | -0.04 | 0/5 |
| AR_full | 8.96* | 3.05 | 1.02 | 3.27 | 5.34 | 1/5 |
| AR_Jrem | 0.82 | 0.74 | -0.06 | 0.30 | 1.98 | 0/5 |
| AR_rrem | 10.22* | 3.70 | 1.20 | 3.50 | 5.56 | 1/5 |
| AR_Jrem_Jsize | 0.16 | 0.04 | 0.37 | 0.59 | 1.01 | 0/5 |
| AR_starts | 8.72* | 3.50 | 0.98 | 3.09 | 5.41 | 1/5 |
| AR_starts_Jrem | 0.50 | 1.13 | -0.06 | 0.54 | 1.84 | 0/5 |
| AR_mentions | 4.84 | 1.04 | 0.95 | 2.81 | 4.40 | 0/5 |
| AR_mentions_Jrem | 0.60 | 0.17 | -0.06 | -0.16 | 1.50 | 0/5 |
| AR_template | 7.73* | 5.22* | 0.96 | 3.64 | 7.29* | 3/5 |
| AR_template_Jrem | 1.10 | 1.32 | -0.23 | 0.62 | 3.20 | 0/5 |
| AR_word | 8.22* | 5.57* | 1.05 | 3.34 | 6.43* | 3/5 |
| AR_word_Jrem | 0.63 | 1.62 | -0.27 | 0.49 | 2.26 | 0/5 |
| AR_answer | 6.37* | 0.80 | 2.16 | 3.84 | 7.43* | 2/5 |
| AR_mismatched | 1.01 | 0.06 | -0.08 | -0.73 | 0.47 | 0/5 |
| AR_mismatched_Jrem | 0.18 | 0.15 | -0.09 | -0.13 | 0.92 | 0/5 |
| rand_full_0 | 0.08 | -0.23 | -0.31 | 1.45 | 0.43 | 0/5 |
| rand_Jcomp_0 | 0.40 | -0.06 | 0.35 | 1.42 | 0.80 | 0/5 |
| rand_full_1 | 0.69 | 1.72 | 0.27 | 0.78 | 1.03 | 0/5 |
| rand_Jcomp_1 | 0.18 | 0.32 | 0.20 | -0.35 | 0.14 | 0/5 |
| C_white | 10.31* | 1.24 | 10.63* | 4.08 | -0.97 | 2/5 |

**What the smoke checks.** The mechanics work:
- every arm runs and logs its edit at every position;
- the J-row reference flips here as in parity;
- the concept-specific controls (mismatched) and the nulls stay near the noise floor's scale (within about ±2 nats).

**Observations** (5 items; hypotheses, not results):
- **The full reconstructor difference is a strong margin handle** at class-mean size, far above both random nulls. It
  rarely flips (1/5), which would be a "weak handle" by the §9 reading if it held at n = 82.
- **Removing J collapses it.** Every J-removed arm falls to about the J-complement null, while removing a random
  1,024-dim subspace leaves it intact. That is the J-attribution pattern of §9: the effect needs the J component *at
  the edit layer*.
- **Single words and templates flip more than phrases** (3/5 each vs 1/5) at the same size. This is consistent with
  the step-2 finding that short inputs are more answer-facing.
- **Your whitened coordinate swap flips 2/5** here, where it did nothing on brew in 011.

## For the full run (decisions for the PI)

1. **Primary at J-row-matched sizes (fix 2), with nulls at the same sizes.**
   - The smoke drew its J-complement and full random nulls at class-mean sizes only.
   - The full run adds both random nulls and the mismatched null at J-row sizes, so the primary contrast (AR J-removed
     at J-row size against the J-complement null at J-row size) is like-for-like.
   - The class-mean-sized arms stay as a secondary, on the 51 covered items, with the 13 rescued-concept items reported
     separately.
2. **Runtime.** About 30 s per item for all arms in this regime, so about 21 minutes for 82 items split across both
   GPUs. With the J-row-size nulls added, about 25–30 minutes.
3. **No other change:** the arms, scores (Δm for the relative test, flips for the absolute bar), reporting within
   toward_tgt, and the bridge-type and Japan breakdowns stay as fixed in §9 and §15.

**Go for the full multihop run with item 1 added?**
