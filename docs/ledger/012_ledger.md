# 012 research ledger

Chronological record of what was run, why, what was found, and anything odd. UTC times. Newest at the bottom.
Spec: `docs/design_specs/012_reader_concordance.md` (v1 → v1.1 §11 → v1.2 §12). Every result here is EXPLORATORY.

## 2026-10-06

**22:20 — phrase-lens read.** Cloned `chen-jan/phrase-lens` @ ea85583, read-only, to `/workspace/sources/ext/phrase-lens-src`; Jan cleared its use.
- The banks are package data.
- All 96 ST items are WSB multihop items. The 67 MT items are Jan's own; 16 are flagged.
- No fitted tensors and no results are shipped.
- Outlook: report 32. Jan's writeup was read the same evening (report 32 §8).

**22:46 — Stage A smoke** (4 items, J swap L24–47, Jan's code unmodified).
- Banks built; 10 s per 4 items.
- Odd: `chem-bones-Z`'s clean prompt already answers y_b (" 26"), and Jan's code counted it as a success → rates now count clean-correct items only.
- The other 3 continuations were unchanged.
- My note explained them geometrically (near-parallel rows). **That was wrong**; see 23:10.

**22:56–23:08 — Stage A full** (8 runs, about 80 s each; `outputs/012_reader_concordance/stageA/`).
- Summary: `results/012_reader_concordance/analysis/stageA_summary.json`.
- Seeded continuation samples: `results/012_reader_concordance/qualitative/stageA_samples.jsonl`.
- **Clean-correct, n = 82:**

  | Arm | Success | Random |
  |---|---|---|
  | J swap L24–47 | 0.585 | 0.061 |
  | J swap L24–39 | 0.549 | 0.012 |
  | J swap L24–35 | 0.463 | 0.012 |
  | logit-row swap | 0.00 | — |

  J ablation survival 0.72 (random 1.00).
- Reproduces Jan's ~59% figure. The early partial (2 of 18) was the alphabetically first, harder items.

**23:10 — the independent review (Codex) and the PI's go.**
- Adopted:
  - the contamination rule plus clean-target-absent;
  - the render contract (captures from the exact labelled runs);
  - the reflection identity (my geometric note withdrawn);
  - withdrawal of the two invalid Stage C rules;
  - the d_r statistic;
  - a fixed cohort that keeps failures and controls;
  - a common downstream band for both readers (L24–35 works, so it is the primary band);
  - narrowed label wording;
  - per-target and pooled reporting.
- Not adopted tonight: per-cell mediation transfers; Q3/Q4 endpoints.
- Spec v1.2 written and committed before any B/C code ran.

**23:20 — Stage A final** (`stageA_summary.py` now also excludes clean continuations that already contain y_b; there were none on multihop and 1 on broadcast). Clean-correct, n = 82:

| Arm | Success | Random |
|---|---|---|
| J L24–47 | 0.585 | 0.061 |
| J L24–39 | 0.549 | 0.012 |
| J L24–35 | 0.463 | 0.012 |
| J L36–47 | 0.561 | 0.037 |
| **J L40–48** | **0.549** | **0.024** |
| logit L24–47 | 0.00 | — |

- J ablation L24–47: survival 0.72 (random 1.00).
- Broadcast J L24–47 (n = 146 clean-correct, 45 clean-incorrect): 0.315 vs random 0.021; Jan's figure shows about 26%.
- **Finding: L40–48 at every position works on multihop.** So 008's brew arm-A null is not explained by the band alone. What remains: positions (008 edited 3 cells), family (brew), and dose. My report-32 inference ("very likely band/positions") was too strong, as the review said.
- The "unchanged" category (swap left the greedy answer unchanged) holds 31–42 of 82 items depending on band.

**~23:45 — Stage B/C smoke** (4 items; the code was built by a subagent).
- **G-parity passed:** clean, J and random greedy texts 12/12 identical to Stage A; budgets and prompt token counts identical; paired success table identical.
- **Read position:** 4/4 match WSB's `last_word_token`.
- **G0 failed as first coded.** The cause is not the edit: Jan's `continuation_scores` scores from bf16 lm_head logits, so every m sits on a 1/16-nat grid. This violates our float32-logit rule.
  - **Fix (decided before the full run):**
    - add a float32 margin (final-norm output × W_U in float32) in the same edited forward, and keep Jan's bf16 m as `m_bf16_jan`;
    - G0 = an α = 0 edit reproduces clean exactly;
    - the identity clamp's bf16 write effect is recorded as a noise floor, not used as a gate.
- **Odd and important (n = 3):** after the J swap, every reader at L40/L44 reads *more* src.
  - On 3 of 4 items the clean pinv coordinates already favour tgt (e.g. atomic-26 at L24: c_s −0.3, c_t 1.9). The swap is a reflection of the src−tgt contrast, so it moves those states *toward src*.
  - Plausibly this explains many "unchanged" outcomes. Readers reading more src would then be following the actual edit.
  - Added diagnostics: per-item clean c_s and c_t per band layer, uᵀh, and an `edit_direction` label (toward_tgt / toward_src).
- **Naming:** answer naming (y_a/y_b, e.g. Au/Ag) added alongside concept naming. Example: the Oracle at L56 says "Ag" after the gold→silver swap without naming "silver".
- The subagent ran Stage C on the smoke captures after G0 stopped the queue, as a code test of the reader path (the captures had passed parity). Accepted as a code test; none of those numbers are used.
- Full-run estimate: about 0.5–0.7 GPU-h for 87 items.

**23:24–23:55 — Stage B-lite and C pilot, full run** (87 covered items; 0.53 GPU-h). Code: `stageB_capture.py`, `stageC_readers.py`, `stageC_nla.py`, `c012.py`, `stageBC_queue.sh`, `stageBC_summary.py`.

*Gates*
- **G-parity:** 0 mismatches over 87 items (clean, J and random greedy identical to Stage A). Paired tables identical: L24–35 48 fail / 39 success; L24–47 37 / 50.
- **G0** (α = 0): exact. Read position 87/87 match WSB.
- **Identity-clamp floor:** |Δm| median 0.001, max 0.145.

*Model side* (n = 82 clean-correct; float32 Δm)
- J L24–35: median +5.04 nats (success +15.0, fail +1.6); random +0.28.
- J L24–47: +8.77 (success +15.9, fail +2.0); random +0.79.
- **Edit direction:**
  - L24–35: 64 toward_tgt (36 succeed / 28 fail), 18 toward_src (2 / 16).
  - L24–47: 68 toward_tgt (48 / 20), 14 toward_src (0 / 14).
  - The reflection's direction explains about a third of failures. **Methodological finding about the paper's swap operator:** it moves the state toward whichever concept the clean state loads on less (in pinv coordinates). So "swap src→tgt" is not guaranteed to push toward tgt.

*Readers* (Stage D, `stageD_analysis.py`, POST HOC: the direction stratification and edit-size control were chosen after seeing the smoke and the full-run summary)
- Within toward_tgt items, the J-swap d_r is larger after working swaps than after failed ones for **every reader**, with cluster-bootstrap CIs well above 0.
- AUC for predicting a working swap:
  - Oracle 0.95–0.96 (L40/L44);
  - NLA 0.95 (L42), 0.97 (L44, off-training);
  - J_NP and J_CB about 0.88–0.91;
  - logit lens 0.88–0.92.
- The partial Spearman of d_r with Δm, controlling for edit size, is 0.79–0.91.
- **Oracle vs NLA:** AUC difference +0.008 [−0.024, +0.045] (L40) and +0.016 [−0.017, +0.063] (L44). No detectable difference, so the prospective "Oracle > NLA" prediction is not supported.
- **Naming** (bridge names in sampled readouts):
  - Oracle L40: clean 121 src / 7 tgt of 164 → after J 35 / 70.
  - NLA L42: clean 117 / 2 → after J 37 / 61.
- The random-target control is not neutral for readers: its d_r comes mostly from LL(src) falling, since the reflection removes the src coordinate.

*Interpretive worry (open)*
- The plain logit lens tracks swap success almost as well as the Oracle and NLA. The read cell is the answer-producing token.
- So readers may be tracking answer-facing content (y_b) rather than the bridge.
- **Check running now:** reader LL for answers (y_a vs y_b) on the same captures, plus the model's own read-position output for bridge and answer tokens.
