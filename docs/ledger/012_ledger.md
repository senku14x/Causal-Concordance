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

**~00:30 (2026-10-07) — the output-facing check** (answer-vs-bridge rescoring of the same saved captures plus the model's own read-position output: 6.7 GPU-min, parity 87/87 bitwise; analysis `stageD_answers.py` → `results/012_reader_concordance/analysis/stageD_answers.json`; POST HOC).
- **The model's own output at the read cell already predicts swap success:** AUC for its bridge-token shift and its answer-token shift is about 0.91–0.92, within toward_tgt items.
- **Readers' answer shifts (y_b vs y_a) track success as well as their bridge shifts:** AUC about 0.83–0.96 against 0.88–0.97. The two shifts correlate with ρ 0.6–0.92.
- **Beyond the model's own output and the edit size,** the bridge shift keeps a modest partial association with Δm: ρ about 0.3–0.57, with cluster-bootstrap CIs mostly above 0. Several touch 0 (Oracle L44, L52, L56; J_NP and J_CB L44; logit L44).
- **Reading:** readers do follow states the model demonstrably acts on. But at this cell (the answer-producing token) most of that tracking is shared with output-facing content: the model is about to say y_b, and every reader, including the plain logit lens, sees it. This is the 006b / 009 confound reappearing on multihop. No reader stands out (Oracle ≈ NLA ≈ token lenses).
- **What would separate bridge reading from answer reading:** read cells where the bridge is present but the answer is not yet output-facing:
  - an earlier position (e.g. the bridge-cue tokens);
  - prompts where the answer position comes later.
- First-token NA: the token lenses and the model output have no answer score for 2 items whose answers share a first digit token (chem-bones-Z, double-dice-faces).
- Correction to the entry above: the model's own read-cell output predicts swap success with AUC **0.97 (bridge) / 0.96 (answer) at L24–35** and 0.91 / 0.92 at L24–47. Values from `stageD_answers.json`.

**~00:15 — Results page published:** https://claude.ai/artifact/NoevKvBkXNUTvtpnJPVsuq
- Built by `scripts/012_reader_concordance/make_page.py`, only from committed JSON, with inline SVG charts. Source: `results/012_reader_concordance/page/index.html`.
- The visual render check was skipped because `/venv/page` doesn't exist on this instance; only a structural check was done (the SVGs are well-formed).

**Where this leaves 012 (for the PI, morning of 2026-10-07)**
- Established tonight (exploratory, scoped to Qwen3.6-27B, 82 ST multihop items, final-token reads):
  1. J-row swaps at L24–47 change the answer on 0.585 of items, against 0.06 random; logit rows do nothing.
  2. The swap operator has a direction set by the clean state, which explains about a third of the failures.
  3. Every reader's src→tgt shift tracks whether the answer changed. Oracle ≈ NLA, with no detectable difference.
  4. Most of that tracking is shared with the model's own output at the read cell.
- **Not established:** that any reader reads the bridge *as distinct from* the upcoming answer.
- **Next, in order of information per GPU-hour:**
  1. Read cells where the bridge is present but the answer is not yet output-facing: earlier prompt positions (the cue tokens), position-split edits (read position only vs everything else), or prompts with a delayed answer slot.
  2. The whitened swap (Camila's a; needs Σ for L24–47).
  3. The class-mean arm (J-independent in construction, Fig. 16 split).
  4. The brew arm-A rerun at every position.
  5. MT.
  6. Stage E when the loader arrives.
- **Odd and open:**
  - Jan's scorer uses bf16 logits; ours are float32.
  - The random-target control is not neutral for readers (it removes src).
  - 2 items have no first-token answer score.
  - Japan is the target in 28 of 87 covered items; reported pooled.

## 2026-10-07

**00:25 — Stage F pre-registered** (spec §13), after the PI said "keep on working as long as it makes sense". Goal: separate bridge reading from answer-facing reading.
- **F1:** position-split edits (read position only / all but the read position).
- **F2:** reads at early prompt positions, with the model's own per-position output as the covariate.
- Expected cost: under 1 GPU-h. Built by the same subagent on the Stage B/C code.

**00:30–01:15 — Stage F run** (subagent; about 42.5 GPU-min). Gates:
- write masks exact: the max write outside the mask is 0.0 at every band layer for every item;
- G0 (α = 0, allbutread mask) bitwise on 87/87;
- clean, J and random arms bitwise-identical to Stage B on 87/87.

Analysis: `stageF_analysis.py` → `results/012_reader_concordance/analysis/stageF_analysis.json`. Rules from spec §13. Cohort: 82 clean-correct items.

*F1, position split* (J swap L24–35):

| Arm | Flips | Median Δm |
|---|---|---|
| all positions | 38/82 | +5.05 |
| **all but the read position** | **35/82** | +4.42 |
| **read position only** | **0/82** | +0.14 |
| random controls | 1/82, 1/82, 0/82 | — |

- **The answer change is carried from earlier positions to the answer token by the model itself;** the read cell's own edit is not sufficient.
- **Readers at the unwritten read cell** (all-but-read arm), AUC for that arm's success: Oracle L40/L44 0.96 / 0.96, NLA L42 0.94, J_NP / J_CB 0.94–0.95, logit 0.93–0.94.
  - **Rule (i) met:** readers follow states the model computed, not only written coordinates.
  - The model's own read-cell output under this arm: AUC 0.97 (bridge and answer).
  - Partial Spearman with Δm, given the model's read-cell output and the write norm:
    - Oracle 0.48 [0.18, 0.63] (L40) and 0.44 [0.13, 0.60] (L44);
    - **NLA 0.46 [0.18, 0.60]**;
    - token lenses 0.34–0.41, with L44 CIs touching 0.
- **Read-position-only edits** change no answers, yet readers shift at the written cell (J_NP +3.08, Oracle L40 +1.11, NLA +0.68). That shift is detection of the write itself.

*F2, early positions* (all-position J swap L24–35; prompt positions before the read token, minus the token before " is"; mean per item; toward_tgt n = 64):
- The model's own next-token output shift at those positions does not track success: AUC 0.55 (bridge) / 0.56 (answer). So these positions are not output-facing, as intended.
- **Readers there shift toward tgt about equally on working and failed swaps:**
  - J_NP / J_CB AUC 0.59–0.64, logit 0.47–0.52, Oracle L40/L44 0.48 / 0.58. Verdict: **not outcome-specific**. They detect the written edit whether or not the model uses it.
  - **NLA L42: AUC 0.72 [0.58, 0.84], partial 0.54 [0.20, 0.69].** Verdict: **intermediate** (between the 0.65 and 0.75 bars, partial CI above 0). The only reader with some outcome-specific signal at early positions. Single reader, single layer, post hoc interest; flagged for a test with fresh data.
- Bug, cosmetic: the per-position profile in `stageF_analysis.json` is keyed by absolute position, not offset from the end, because the reader rows carry no `n_prompt`. Not used in any verdict.

*What Stage F adds:*
- At the answer token, readers track a model-computed state, and the Oracle and NLA keep a moderate association with the answer change beyond the model's own output there.
- At written earlier positions, readers mostly see the write, not the use. The NLA is the possible exception.
- This is the first sign in the project of an NLA property the Oracle doesn't share. It is weak and needs fresh data before any claim.

**01:20 — Stage G pre-registered** (spec §14): whitened J-row swaps at L24–35, Camila's request (a). Σ by 008's recipe. Measured: flips, Δm, direction, readers at the read cell (likelihood only). Rule: ±10 points of flip rate against the raw swap.

**01:25–01:45 — Stage G run** (subagent; about 10.7 GPU-min). Analysis: `stageG_analysis.py` → `results/012_reader_concordance/analysis/stageG_analysis.json`.
- **Σ for L24–35:** 008's recipe, imported unchanged. 128 UltraChat conversations, 50,199 tokens; Ledoit–Wolf λ 0.0033–0.0053; G-Σ passes; round trip ≤ 2.4e-14.
- **Gates:** G0 (α = 0) bitwise on 87/87; |R·A − I| ≤ 1.6e-14; clean identical to Stage B.

**Result** (82 clean-correct items, all positions, L24–35):

| Arm | Flips |
|---|---|
| whitened J swap | **26/82 (0.317)** |
| raw J swap (reference) | 38/82 (0.463) |
| whitened random control | 3/82 |

- **§14 verdict: "whitening hurts" on this family** (−14.6 points).
- The whitened flips are a strict subset of the raw flips (26 both, 12 raw-only, 0 whitened-only).
- Direction from the whitened coordinates: 64 toward_tgt (25 flip), 18 toward_src (1 flips). It agrees with the raw direction on 76/82.
- **Readers on whitened-swap states,** AUC within toward_tgt: Oracle 0.92–0.93, NLA L42 0.91, J_NP / J_CB 0.85–0.87, logit 0.79–0.83. Same pattern as the raw swap.

**Odd:** the whitened random control's realised edits are large (median max ‖Δh‖/‖h‖ 0.227, against 0.079 for whitened J). Its Δm is +2.87 despite 3 flips. Any Δm comparison against this control needs a size-matched check.

**Note for Camila:** on Qwen ST multihop at L24–35, the paper-style coordinate swap works better in raw coordinates than in whitened ones. Her experience that whitening gives cleaner results came from swaps along Oracle-reconstructor vectors (her request b), which still waits on the loader.

**Σ file:** 5.0 GB, mirrored with the outputs.

**01:55 — Stage H pre-registered** (spec §15): 008's arm A on brew, rerun at every position (L24–47 and L40–48), with the answer-atom control. Question: positions or family?

**02:00–02:15 — Stage H run** (subagent; about 12 GPU-min; `stageH.py` → `results/012_reader_concordance/analysis/stageH_summary.json`).
- **Setup:** 008's 82 gated brew dev pairs (33 tables), with G taken from 008's recorded values (our recomputation is within 0.24 nats). All 12 colour words resolve to single " w" tokens. G0 bitwise on 82/82.

Median S [table-cluster 95% CI]:

| Arm | S | Flips |
|---|---|---|
| answer atom, L40–48, every position | **0.073 [0.056, 0.089]** | 0/82 |
| answer atom, L24–47 | 0.155 [0.115, 0.231] | 0/82 |
| z atom, L24–47 | 0.019 [0.004, 0.028] | 0/82 |
| z atom, L40–48 | 0.005 [0.002, 0.012] | 0/82 |
| random controls | −0.006 to 0.032 | — |

- 008's 3-cell answer control was 0.010; 008's class-mean full-direction lever was 0.21.
- **§15 verdict: "family explains it."** Every-position editing lifts the answer-atom control from 0.010 to 0.073, still below the 0.15 bar. So 008's brew null is not about positions. The z arm is not read under the rule.
- Descriptive, outside the rule: even at L24–47, where the answer control just passes, the z-atom swap moves S by only 0.019 and flips nothing.
- The J-row edits at the `:` cell are about 1–2% of ‖h‖. Brew's colour concepts load weakly on their J rows, so the coordinate swap barely edits anything.
- **Contrast with multihop:** every-position J-row swaps there change about half the answers. Consistent with the paper's report that swap success tracks workspace loading. Loading itself was not measured here.
- The CLAUDE.md mistakes-log row on report 32's "band/positions" inference stands: positions are now ruled out as the explanation too.

**Stopping point for tonight.** The next steps need PI input or external material:
1. the class-mean arm (J-independent in construction): needs about 10–16 gated, cue-varied prompts per concept, i.e. drafting;
2. MT: needs directions for multi-token concepts (class means, or Jan's template tensors);
3. Stage E (Camila's reconstructor vectors): needs her loader;
4. a fresh-data test of the NLA's early-position signal (Stage F2, "intermediate").

**02:30 — Stage Q: reader agreement and ensembles** (CPU, POST HOC, the PI's Q1/Q2/Q4; `stageQ_concordance.py` → `results/012_reader_concordance/analysis/stageQ_concordance.json`). Cohort: J swap L24–35, toward_tgt items, n = 64 (36 working).
- **Agreement:**
  - Readers' bridge d_r agree strongly across items (Spearman 0.83–0.90; Oracle L44 vs NLA 0.90, J_NP vs NLA 0.84, logit vs Oracle 0.83).
  - "Names tgt" agrees only moderately: κ Oracle L40 vs NLA 0.36, Oracle L44 vs NLA 0.51, Oracle L40 vs L44 0.63.
- **Coverage:**
  - Working swaps not naming tgt: Oracle L40 2/36, Oracle L44 2/36, NLA 7/36.
  - Failed swaps naming tgt anyway: Oracle L40 14/28, Oracle L44 8/28, NLA 8/28.
  - Working swaps named by only one family: Oracle 7, NLA 0.
  - No reader has d_r ≤ 0 on any working swap.
- **Ensembles** (against "the model's answer changed"):
  - best single reader Oracle L44 AUC 0.961;
  - family-balanced rank ensemble 0.951 [0.87, 0.99]; ensemble minus best [−0.04, +0.01];
  - leave-one-family-out 0.94–0.96;
  - the model's own read-cell output alone 0.969, plus the readers 0.971.
  - **On this task, ensembling readers adds nothing** beyond the best reader or the model's own output; the readers share one signal.

**02:35 — Hand-read sample** (seed 7: 4 working, 2 failed; Oracle L40 and NLA L42, clean vs J at L24–35).
- **The Oracle mostly names the answer after the swap, not the bridge:**
  - Kremlin→Japan: "Japanese… Buddhism";
  - kangaroo→Japan: "Tokyo… 东京 (Tokyo)";
  - Colosseum→Japan: "Tokyo".
- **It also names the new concept when the answer did not change:** Great Wall→Egypt (failed): "Egypt's longest river is the Nile". This fits "failed swaps naming tgt" (14/28 at L40) and F2: the Oracle detects the edit.
- **NLA L42 readouts are mostly format boilerplate** ("Q&A trivia format establishes a factual answer is expected…"). Concept content appears only sometimes and late ("likely Tokyo. The clue references Japan's landmark…").
- Oracle readouts are often partly in Chinese (罗马, 日语, 悉尼).
- One odd clean Oracle readout: pred-valentines-prevmonth (February) talks about October / September / November.
- Takeaway: the likelihood numbers overstate how "conceptual" the readouts are. The Oracle's tgt content is largely answer-facing, and the NLA's is diluted by boilerplate.

**02:55 — Stage E0 pre-registered** (spec §17), after the PI asked to try building the loader.
- **Findings so far:**
  - The official loader is not public: `local-workspace` has no code (checked).
  - The LoRA covers all 64 blocks, so "truncation" is not the issue.
  - The README is partly stale ("17-row head" vs `heads.pt`'s one head and 12 tags).
  - **Clue:** the published `val_jfve` and `val_jret1` exist for L20–60 but not L63, and the Neuronpedia J-lens matrices cover blocks 0–62. So "jfve" is very likely computed through the J-lens matrices, i.e. a J-lens-transformed reconstruction target with weight `loss_mix_lambda` 0.25.
- **Plan:** test our loader across all 12 layers against the published profile, and test four jfve definitions. Runs after Stage R (one GPU).

**02:45–03:30 — Stage R run** (subagent; about 41 GPU-min; F2 protocol on Jan's broadcast ST bank). Analysis: `stageR_analysis.py` → `results/012_reader_concordance/analysis/stageR_analysis.json`.
- **Gates:** G0 bitwise on 192/192; clean text, budget and clean_correct match Stage A's broadcast rows on 192/192.
- **Cohort:** 146 clean-correct items (1 dropped because its clean output contains y_b); 109 toward_tgt (32 working), 37 toward_src (7 working).
- J swap L24–35 success 39/146 (0.267); random 0/146.

*Early-position test:*
- The model's own early-position output **does** track success here: AUC 0.96 (bridge) / 0.94 (answer), against 0.55 / 0.56 on multihop.
- The concept is explicit in these short prompts, so the early positions are outcome-informative.
- Every reader's early-position AUC is about 0.91–0.95.
- Partial Spearman beyond the model's early output: NLA L42 0.34 [−0.21, 0.60], Oracle L44 0.40 [−0.04, 0.63], others 0.24–0.30, all CIs including 0.
- **§16 verdict: "does not replicate."**
- **Qualification (written after seeing the result):** the broadcast early positions are not the non-output-facing setting that made F2 informative. So this is a weak test of the hint, not a clean refutation. The F2 NLA hint stays unconfirmed; a proper test needs fresh *hidden-bridge* items.

**03:35–03:50 — Stage E0 run** (subagent; about 8 GPU-min). `stageE0_ar_profile.py` imports 011's `ar_vectors` unchanged. Output: `results/012_reader_concordance/analysis/stageE0_ar_profile.json`. 1,024 held-out spans of 1–32 tokens (mean 16.4); all 12 tag layers. The 9 missing whiteners were downloaded from the public `local-workspace` dataset.

*Verdicts:*

| Check | Met at | Rule | Result |
|---|---|---|---|
| scaled FVE within 0.04 of published | 12/12 layers | ≥ 10/12 | met |
| ret@1 (micro-batches of 24) within 0.05 | **8/12** | ≥ 10/12 | **missed** |
| unscaled FVE within 0.04 | 2/12 | ≥ 10/12 | missed |

- The ret@1 misses are L20–32, each 0.05–0.11 short.
- **§17 verdict: not validated, even up to scale.** Stage E stays blocked under the rule.
- **jfve: open.** Unscaled candidate FVEs miss everywhere. Candidate (b), the μ-removed J transform, matches jret1 at 10/11 layers but not jfve.

*Observation:* the best-fitting output scale rises steadily with depth (0.11 at L20 → 2.33 at L63) instead of being one global constant.
- That is what a scale-invariant training loss would produce (the head's output norm roughly fixed while true activation norms grow with depth). Anthropic's own reconstructor uses "cosine error in the whitened metric".
- This plausibly explains 011's "unexplained global output scale" (post hoc interpretation, untested).
- It also implies the published FVEs are scale-aligned, which our scaled FVE matches at 12/12. The jfve candidates should be compared the same way.
- The AR config's jspace file has the same sha256 as our J_NP pin.

**03:55–04:05 — §17a run** (POST HOC; subagent; about 4 GPU-min; `stageE0a_jfve_scaled.py`; results under `jfve_scaled` in `stageE0_ar_profile.json`). The spans were regenerated deterministically; L44's scaled FVE reproduces E0 exactly.

| Candidate | Scaled FVE within 0.04 | jret1 within 0.07 |
|---|---|---|
| (a) raw J transform | 0/11 | 9/11 |
| **(b) J transform of the mean-centred activation** | **8/11** | **10/11** |
| (c) J-lens logits | 0/11 | 8/11 |
| (d) 007's dense J subspace | 0/3 | 3/3 |

- Candidate (b) is **within 0.04 at every layer from L32 to L60** (e.g. L44 0.564 vs 0.589; L48 0.622 vs 0.608). It misses only L20–28 (0.26–0.37 vs 0.39–0.48), the same early layers where our loader's retrieval also falls short.
- **By rule (9/11): jfve stays open.**
- **Post hoc reading:** it is likely that the AR's "jfve" is the FVE of the J-lens transform J_ℓ(y − μ). The AR was then trained with a J-lens-transformed reconstruction term (`loss_mix_lambda` 0.25).
- The Oracle is RL-trained on a "joint bullet-FVE" computed on this AR. If that FVE includes the J term, the Oracle is rewarded partly for reconstructing J-lens content, which would be a design reason for 007's finding that the Oracle's z reading is concentrated in J.
- **Question for the PI to put to Camila / WSB:**
  1. Does the AR loss include λ = 0.25 on J_ℓ(y − μ)?
  2. Does the Oracle's joint bullet-FVE reward include the J part?

**Page v4 published** (section 9: the reconstructor).

**Stopping point (04:05).** GPU idle. Remaining items need the PI or outside material:
- Camila's loader, or her answer on the J-loss;
- drafting gated cue-varied prompts (class-mean arm, MT);
- fresh hidden-bridge items to test the NLA early-position hint properly.

**(PI: "okay do 1") — Stage I pre-registered** (spec §18): the delayed-question variant.
- 168 templated prompts from 28 of Jan's country-clue items (`stageI_prompts.py`).
- Swap the hidden country only up to the end of a carrier sentence; check whether the answers to all three later questions follow.
- Read at the hold tokens, where the model cannot yet be "about to say" the answer.
- Hand-written content: one fact table (capital / language / continent for 23 countries). The gate checks that Qwen knows every tgt fact.

**Stage I run** (subagent; about 19 GPU-min). Analysis: `stageI_analysis.py` → `results/012_reader_concordance/analysis/stageI_analysis.json`.

*Gates:* G0, write masks and prefix tokenization all pass. The hold states are **bitwise identical across the three questions** of every (clue, carrier), 56/56.
- **Clean gate: only 35 of 168 prompts pass** (12 capital, 23 language, **0 continent**; 29 hold states over 20 clues).
  - 98 fall below 9/10 at T 0.8; 27 are wrong when greedy; 8 already contain y_b.
  - Qwen's short continuations of these delayed prompts wander.
- Tgt fact checks: 26/27 pass ("continent of Japan" fails).

*I-1, lever: PASS.*
- J_pre (edit only before the question) flips 25/35 gated trials (0.71); rand_pre 0/35; J_all 28/35.
- Ungated: 92/154 vs 1/154.
- 69% of gated hold states flip every gated question.
- **The model holds the swapped country through the carrier and uses it for whichever question follows. A swap made before the question exists acts at the bridge level.**

*I-2, non-output-facing check: FAILS.* The model's own next-token bridge shift at H2 predicts hold success (AUC 0.82 gated, 0.76 ungated). The swapped country leaks into H2's next-token tail (009's lesson). **So I-3 cannot be passed by rule.**

*I-3, readers at H2* (the carrier's final "."):
- They do **not** track hold success: AUC 0.45–0.66 across readers, cohorts and strata; median d_r about 0 to 2.
- Sampled readouts at H2 almost never name either country (Oracle L44 clean "neither" 55/58; NLA 58/58).
- **Readers at the hold token see nothing of the held country, though the model demonstrably uses it.** The likely reason is that it is stored at the clue tokens and read back by attention, not carried in H2's residual. That reason is inferred, not tested.

*H1, the clue's final "." (secondary; a written position; the model's own output at H1 was not measured):*
- Every reader's d_r predicts hold success on the gated cohort: AUC 0.88–0.93 (CIs about [0.65, 1.0]; 29 holds, 22 successes). d_r is about 11–14 after working swaps vs 3–8 after failed ones.
- No reader stands out: the logit lens is 0.88–0.90.
- Ungated: 0.63–0.75, with Oracle / NLA 0.71–0.75 against token lenses 0.63–0.66.
- Not decisive: the position is written (F2: readers detect writes), the gated n is small, and there is no output control.

*Post hoc, labelled:*
- The edit direction measured at H2 does not predict flips (15 of 22 gated toward_src trials flip). H2 is the wrong token to define direction for these prompts. The analysis reports both the pre-registered toward_tgt stratum and all holds.
- Japan is the target in 19 of the 35 gated trials; no-Japan AUCs are 0.39–0.51, n small.

*What Stage I adds:*
- (1) Strong model-side evidence: the hidden country is held and used at the bridge level, across a carrier, for any question asked later.
- (2) At a token where it is held but neither written nor (much) output-facing, **no reader sees it.** That is a limitation of reading one token's residual: the information lives in context (earlier tokens), not at the read cell.
- (3) Where the swap is written (the clue end), readers track success, but without a reader-specific advantage, and confounded by the write.

**(PI: "go") — Stage J pre-registered** (spec §19).
- **J1:** a relaxed clean gate (fixed 8-token budget, ≥ 8/10 at T 0.8), re-scoring Stage I's flips. A stated amendment, written after seeing I's 35/168 pass rate.
- **J2:** swap only the clue's own tokens (positions before the clue's final "."), then read at that "." (H1) and at the carrier end (H2). Both are computed by the model, not written. The model's own output at H1 is the output-facing control.
- **Correction logged:** my message suggested "swap only the tokens after the clue and read at the clue end". In a causal LM such edits cannot reach the clue end. The corrected design edits the clue tokens before H1.

## Corrections after the independent Codex audit (2026-10-07; the PI shared reviews 09–15)

All points were checked. Accepted:

1. **A real bug in Stages F2 and R (mine).** `keep()` excluded the token before " is" using `is_read_minus_1`, which only the model rows carry. Reader averages therefore included a position the model covariate did not.
   - **Fixed:** one shared (item, pos) mask, built from the model rows and applied to both. `stageF_analysis.py` and `stageR_analysis.py` re-run; the numbers match the audit's independent recomputation.
   - **F2, NLA L42:** AUC 0.72 → **0.59 [0.44, 0.77]**; partial 0.54 → **0.17 [−0.17, 0.39]**. Verdict "intermediate" → **"not outcome-specific"**. **The NLA early-position hint was an artifact of this bug; withdrawn.**
   - **R:** NLA partial 0.34 → −0.06 [−0.37, 0.23], Oracle L44 0.40 → 0.07. The verdict ("does not replicate") is unchanged, and no positive residual estimate remains.
2. **"NLA-specific if NLA passes and others fail" (§16, also §18–19) is an invalid specificity rule:** it compares two significance outcomes. It is replaced by a paired reader contrast with a stated threshold (spec §20). Existing paired check from the audit: R NLA − Oracle L44 AUC −0.0004 [−0.019, 0.020].
3. **Scope.** J-derived intervention labels show outcome tracking under these interventions, not natural mediation of the bridge. Wording narrowed on the page and here.
4. **E0's "held-out spans" were reused calibration contexts:** 1,024 overlapping crops from the same 12 responses 011 used to select the loader conventions; 792 distinct target positions; retrieval on 1,008 rows. Not independent validation.
5. **"One global scale" was wrong:** E0 fits one scalar *per layer*, on the same samples it scores (scaled FVE = squared cosine). The depth trend is *compatible* with a scale-invariant loss but does not identify it. "Implies the published FVEs are scale-aligned" is withdrawn as too strong.
6. **E0a candidates (a), (c) and (d) subtracted separate sample means** and reused E0's retrieval values computed on different vectors, so their FVE and retrieval entries describe different geometries. Candidate (b) is unaffected. E0's ledger line "unscaled candidates miss everywhere" was also stale: (b) matched at L52 and L56.
7. **The J-objective chain is not established.** Each link needs its own evidence: a candidate jfve metric → an active AR training loss → the Oracle's reward → an explanation of 007. "The AR was then trained with a J term" is withdrawn. New wording: "the metadata and profile support a J-aware reconstructor hypothesis; the exact objective and the reward's dependence on J remain unverified." Questions for the PI to ask separately: the AR training loss, the definitions of validation FVE and J-FVE, the Oracle reward, and an official input/output fixture.
8. **Earlier wording corrected:**
   - "Oracle-only 7, NLA-only 0" pooled 4 Oracle samples (2 layers) against 2 NLA samples. At equal opportunity it is Oracle L40-only 7 vs NLA-only 2 (or L44 6 vs 1).
   - The 14 "failed swaps naming tgt" all have positive answer-margin changes, so they are not false positives.
   - "Ensembling adds nothing" → "no demonstrated gain" (increment CI [−0.023, 0.021]).
   - Stage H: "family, not positions" is too strong. H also changed the numeric regime and the atom convention, and its late-band answer control failed the power bar. "Not established which difference explains 008's null."

**Still supported:**
- the J-swap lever (Stage A: 48/82 vs random; R: 39/146 vs 0);
- F1 propagation (35/82 without editing the read cell; 0/82 editing only it);
- readers track successful interventions;
- the Stage D output-facing result;
- Stage I's model-side finding.

**Additions after report 34** (2026-10-07; the PI checked report 34 and approved this housekeeping; the lines above are left as written):
1. **ledger:186 kept F2's other readers at their pre-correction values.** Corrected (`stageF_analysis.json` `F2.*.auc`): J_NP / J_CB 0.47–0.51, logit 0.39–0.43, Oracle L40 / L44 0.40 / 0.47 (NLA 0.59). Every reader is "not outcome-specific".
2. **Pattern, recorded, not interpreted:**
   - 8 of the 9 corrected F2 partials are negative (e.g. Oracle L40 −0.38 [−0.50, −0.07], logit L40 −0.32 [−0.65, 0.01]); the NLA (+0.17) is the exception.
   - The readers' d_r correlate 0.75–0.98 (Stage Q, read cell), so these are not nine independent tests.
   - Stage J's H2 no-Japan AUCs are 0.30–0.36 for every reader (`stageJ_analysis.json` `readers.*|H2.auc_no_japan`).
3. **ledger:250's "0.83–0.90" was written from the printed subset** (`stageQ_concordance.py:115–116`). The full matrix runs from 0.75 (logit–NLA) to 0.98 (Oracle L40–L44); the cross-family maximum is 0.96 (J_CB–logit). The script now also prints the cross-family min / max.
4. **Numbers that were quoted but missing from JSON are now in JSON** (CPU reruns; every existing value is unchanged, checked key by key):
   - J1 re-scored J_pre 42/60, rand_pre 0/60, J_all 50/60 (`stageJ_analysis.json` `J1_rescored_stageI_arms`);
   - R: J swap 39/146, random 0/146 (`stageR_analysis.json` `J_L24-35_success`, `rand_L24-35_success`);
   - Q naming at equal opportunity: Oracle L40-only 7 vs NLA-only 2; Oracle L44-only 6 vs NLA-only 1 (`naming_equal_opportunity`);
   - Q: output plus readers minus output alone, +0.002 [−0.020, +0.020] (`auc.model_output_plus_family_ensemble_minus_model_output*`; our seeded bootstrap; audit 12 reported [−0.023, 0.021] from its own bootstrap); Oracle L44 − output [−0.06, +0.02]; NLA − output [−0.10, +0.03].
5. **Spec §20 paired contrast for R** (`stageR_analysis.json` `paired_nla_minus_oracle`):
   - NLA − Oracle L44: 0.000 [−0.018, +0.019], n 109. Both AUCs are 0.9505 under the corrected mask; audit 14's −0.0004 was on the pre-fix endpoint.
   - NLA − Oracle L40: +0.03 [+0.01, +0.07]. The CI excludes 0 but |ΔAUC| < 0.10, so this is not "specific" under §20.
6. **`audit_note` fields** were added to `stageH_summary.json` (the rule's label "family explains it" is not a supported explanation) and `stageE0_ar_profile.json` (`global_scale` is per layer; the spans are reused calibration contexts) by `annotate_audit_notes.py`. Labels and key names are unchanged.
7. **Page v8:**
   - the early-position AUC range and the read-cell partial range are now computed from JSON (they were hard-coded as "0.5–0.64" and "0.3–0.5");
   - the E0 caption's "held-out" became "reused calibration contexts";
   - the Stage J H2 sentence now gives the model's own H2 output (0.70).
8. **H2 is output-facing by our rule:** the model's own H2 output gives 0.70 in J and 0.82 in I, both ≥ 0.65. Report 34's first draft said it was not; corrected.
9. **`/workspace/.env` was missing on this instance.** The OpenRouter key was re-entered from the PI's message. It is in no mirror.

**Stage J run** (the last subagent task; about 25 GPU-min). Analysis is mine, CPU only: `stageJ_analysis.py` → `results/012_reader_concordance/analysis/stageJ_analysis.json`.

*Gates:* G0 bitwise 168/168; no write at positions ≥ H1; clean H1/H2 captures bitwise equal to Stage I; holds identical across questions.

*J1, relaxed gate (8-token budget, ≥ 8/10):*
- 60/168 pass (capital 24, language 36, continent 0); 27 clues.
- Flips on the J1 cohort: J_pre 42/60 (0.70), rand_pre 0/60, J_all 50/60. This confirms Stage I on about twice the trials.

*J2, clue-only edit* (positions before H1; H1, the carrier and the question are never written):
- **J-1: PASS.** J_clue flips 36/60 (0.60), median Δm +8.94; rand_clue 0/60. 48 holds, 31 successes.
- **J-2: FAILS, narrowly.** The model's own next-token bridge shift at H1 predicts hold success with AUC 0.69 (bar < 0.65); at H2, 0.70.
- **J-3: not met** (J-2 failed). Descriptive:
  - **At H1** (computed by the model, not written), every reader's d_r predicts hold success: Oracle L40 / L44 0.75 / 0.80, NLA L42 0.79, J_NP / J_CB 0.79–0.81, **logit lens 0.79–0.81**. d_r is about 9–11 after working swaps vs 3–4 after failed ones.
  - Without Japan targets: 0.57–0.67.
  - **At H2,** every reader is about 0.46–0.49: no signal.
- **Paired Oracle − NLA at H1** (§20): L40 −0.04 [−0.16, 0.08], L44 +0.01 [−0.10, 0.11]. **Unresolved, with no detected difference.**

*Reading:*
- The model holds and uses the swapped country from the clue tokens alone (strong).
- At the clue's end, which the model computes from the edited clue, readers track whether the swap works about as well as the plain logit lens.
- The model's own next-token output there carries a similar signal, and the effect leans on Japan-target items.
- At the carrier end nothing is readable.
- **No evidence that any reader reads the hidden step beyond output-facing content, and no Oracle/NLA difference.**

**STOP (PI: limit nearly gone; no more subagents).** GPU idle.

**NLA repo read in full** (`ceselder/qwen3.6-27b-nla-rl` @ 5a13b7ec, at the PI's request).
- **Contents:** RL adapters `iter_000100`–`000800` (the README lists only up to 600); `ar_reconstructor/` (43-block truncated backbone plus value head, `saved_at_step` 600, "latest only"); SFT-init and SFT checkpoints of the reconstructor; `nla_meta.yaml`; `run_config.yaml` (reward = −reconstruction MSE, no J term); `fve_trajectory.json`; `data/example_activations.parquet` (64 layer-42 activations with source text, a ready validation set).
- **The README says to "pair any AV adapter with the provided (final) AR".** The project's earlier framing that "AR step 600 is not step-paired with our AV iter_000400", treated as a blocker for the reconstructor / hallucination test, was an unchecked assumption.
- Keeping iter_000400 for the reading experiments remains justified, since it is WSB's and Cathryn's NLA.

**Stage K pre-registered** (spec §21; the PI's "go" on report 34 (e)1). This is the smoke validating the nla-rl reconstructor (AR) against the card's 76.6% FVE at step 400.
- Reference read from `fve_trajectory.json`, `run_config.yaml` and EasyNLA's eval loop:
  - FVE = 1 − MSE(n(pred), n(gold)) / B, with n scaling to norm √5120;
  - B is the predict-the-mean baseline on the scored set itself;
  - 128 held-out rows; T 1.0, top_p 1, top_k 0, 256 tokens.
- Band: [71.6, 81.6] on the parquet's 64 rows. Extraction rate ≥ 0.90. Matched − shuffled ≥ 20 points.
- `/workspace/.env` recreated. The AR download (35.4 GB, pinned 5a13b7ec) was started in the background before the run.

**Stage K run** (about 4.4 GPU-min: AV 2.2, AR 2.2). Results: `results/012_reader_concordance/analysis/stageK_ar_smoke.json`.

*Gates:*
- AV injection healthy: the write's norm relative to the residual is 0.99–1.01, and its cosine with the activation is ≥ 0.99998, on every chunk.
- Extraction 64/64 (and 256/256 over all samples).
- AR load: only `model.norm.weight` and `lm_head.weight` are reported missing. EasyNLA replaces both with identity, so this is expected.

*§21 verdict: **validated.***
- Primary FVE (first sample per row, B on the 64 rows = 0.462): **78.1%**, row-bootstrap [75.3, 80.7]. The band is [71.6, 81.6]; the reference is 76.6% at step 400.
- FVE over all 4 samples: 76.6%.
- Shuffled control: −89.8%, so matched − shuffled is 167.9 points (rule: ≥ 20).
- Per-row FVE: median 0.81, range 0.30–0.91.
- Raw AR output norm median 71.9 (≈ √5120 = 71.6); gold norm median 88.2.

*Caveat:* the 64 parquet rows may have been training rows, so the in-band FVE validates our loading path, not held-out generalisation.

*Qualitative (seeded sample `results/012_reader_concordance/qualitative/stageK_samples.jsonl`):* explanations are genre and continuation descriptions. Row 4 (FVE 0.81) is already a concrete example for the hallucination test: the source is a Seattle-area Holi event (Lake Sammamish, "Vibha Seattle"); the explanation says "Washington DC" and names an organisation "Ynacharya for India". Both are contradicted by the text, yet reconstruction is good.

**STOP (the PI's rule: report after the smoke).** Next, on the PI's go: the main hallucination test (pre-register per §21's decisions), then the late-layer H2 read with Oracle L52–60 (the PI's item 8).
