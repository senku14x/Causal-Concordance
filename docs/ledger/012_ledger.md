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
