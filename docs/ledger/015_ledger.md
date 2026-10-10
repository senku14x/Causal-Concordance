# 015 ledger: the hallucination test on the Oracle Lens, with cross-scoring

Append-only. Spec: `docs/design_specs/015_oracle_hallucination.md`. Branch `015-oracle-hallucination`. EXPLORATORY.

## 2026-10-11: spec approved and committed

- The PI approved the spec with six changes, applied before commit:
  - R1 on all four cells;
  - 013's exact metric, with B stated;
  - reader-specific claims only for the pre-named contrasts;
  - the per-share requirement;
  - F's composition logged;
  - the NLA P redo as a post hoc arm, with the sign convention stated.
- B computed from the gold targets alone before any text was scored:
  - Oracle AR: 0.7032, so ±2 points = ±0.0070 cosine;
  - NLA AR: 0.2818, which reproduces 013's `baseline_B`, so ±2 points = ±0.0028 cosine.
- No judge call or reconstructor forward has run.

## 2026-10-11: step 2, R1 on all four cells (O versions, clean first samples, no judge calls)

- **Stop gate** (Oracle texts × Oracle AR): +88.6 [81.0, 96.1], retrieval@1 0.78. Pass.
- **Other cells:**
  - Oracle texts × NLA AR: +75.6 [66.9, 84.8], retrieval 0.68;
  - NLA texts × NLA AR: +130.4 [118.0, 142.9], retrieval 0.75;
  - NLA texts × Oracle AR: +9.7 [7.4, 12.0], retrieval 0.26.
  - All CIs exclude 0. The last cell is weak (cosine 0.067).
- **Determinism:** a true rerun is bitwise identical. Batch-layout noise per text: 90th pct 0.11, max 0.35 points.
- **G-repro (NLA side):** 1.9e-7. **CW-20 consistency:** +0.215, cos 0.466, scale 0.439 (hers +0.21, 0.46, 0.434).
- **Material fetched:** the NLA reconstructor (`ar_reconstructor/`, `nla_meta.yaml` at `5a13b7ec`) was downloaded; it
  was not on this instance.
- **Correction before commit:** my first "determinism" check compared two batch layouts. It is now the identical call
  twice, with the layout comparison reported separately.
- Report: `claude_ovarida/45_015_r1_2026-10-11.md`. No judge call has run.


## 2026-10-11: step 3, smoke — judges, hand checks, and choices fixed before any score

Written after the judge calls and the hand checks, **before any reconstructor forward on a smoke version and before
any Δ was computed**.

**Judges (20 Oracle readouts by 013's rule; the same 20 (item, state, sample) for the NLA).**
- G1 (Oracle A): 20/20 parse with verbatim spans. B labels, K kinds and answer spans parse 20/20 for both readers.
- Spend: about $0.05.

**Hand checks** (`results/015_oracle_hallucination/qualitative/handcheck_smoke.md`, all 20 Oracle readouts and the
first 10 NLA explanations).
- **G2, A's contradicted spans on Oracle readouts: pass.** There are 32 spans, and 26 are clearly right.
  - Borderline (4): two invented follow-on questions in bordeaux ("What fruit is used to make sherry?…", "What is the
    primary fruit used to make beer?"); eiffel rand "they joined the Eurozone in 2002" (euro adopted 1999, cash 2002);
    carnival "The question asks which ocean".
  - Wrong (1): bordeaux "the primary fruit used to produce sherry" (sherry is made from grapes).
  - Arguable (1): bordeaux "specifically, the black grape".
  - Precision is 26/32 = 0.81 strict and 30/32 = 0.94 with borderlines counted correct. Threshold 0.80. The 10 extra
    readouts of §8 are checked at the full run.
  - Recall is not gated. Misses seen: aztec "Modern Spanish" and "The primary language spoken in Mexico today is
    Spanish" (against the continuation "Nahuatl"); rhyme "the square of what comes after 3 is 9"; roman "(Answer: III)".
- **G2k, kind tags: 28/32 = 0.88, pass.** But **all 3 "misquote" tags are wrong**: they are wrong answers opening a
  bullet ("Pt", "**Pt**", "Stockholm …"), which are wrong_entity. Roman "三 (three) …" is tagged false_fact; I would
  call it wrong_entity.
  - So for Oracle texts, the per-kind "misquote" is a mis-tagged wrong_entity in this smoke. It is reported as
    unreliable unless the full-run tag check says otherwise.
- **G2a, answer spans: pass.**
  - Oracle: 50 spans, 49 are answer statements (0.98). The error is aztec 'Nahuatl', which the readout gives as "the
    old language", not as the answer. 44/50 (0.88) are exactly the shortest span; 5 are over-long (they carry a
    parenthetical or a "because …" clause).
  - Oracle recall misses (not gated): roman "5 (the number of Olympic rings)"; carnival bullet 4 "Atlantic Ocean";
    sunrise "is west" / "is West"; smallest bullet 3 "Europe"; bordeaux "**Answer: grapes**".
  - NLA: 10 spans in 10 explanations, all answer statements (1.00); 8/10 exactly shortest.
  - NLA recall misses in 3 of 10: aztec ('most likely "Spanish" or "Nahuatl"'), fjords ('e.g., "Oslo"'), redplanet J
    ('e.g., "two"'). Atomic's "Cu" is correctly not an answer statement (continuation "Au").
  - Recall misses leave answer text in CA, so they bias Δ_A toward 0.
- **F (20 Oracle):** 18 are acceptable minimal factual rewrites. Most of F's edit is deleting follow-on questions,
  placeholders and invented hints (PI change 5 logs the split).
  - Tallest is questionable: it inserts the prompt's own sentence ("Fact: The continent of the world's tallest mountain
    is") three times. That adds prompt text, which 013 found the reconstructor rewards.
  - Roman is a minor over-edit: it rewrites the correct "equal to the Roman numeral V" and "5 (…)" to "V".
- **P (20 Oracle): two defects.**
  - **8/20 P texts come back wrapped in PROMPT_P's `<<<` … `>>>` delimiters.**
  - The realised edit size is often off even after the re-ask. Examples: redplanet J changed 9 words against F's 31;
    tulips 9 vs 16.
  - The P* texts (NLA) carry no delimiters.
  - 013's own P has the same delimiter defect on 4/522 texts (lisbon J 0, tajmahal rand 1, olympia J 0, caesar
    rand 1); none is in the smoke. 013's recorded scores are not changed.

**Choices fixed now** (the spec leaves them open, or they fix a defect; POST HOC where marked).
1. **Delimiter strip (POST HOC, fixes a defect):** one leading `<<<` and one trailing `>>>` are stripped from every P
   and P* before scoring. words_changed and the match ratio are recomputed on the stripped text. The re-ask is not
   redone. P's B labels were judged on the wrapped text and are kept.
2. **PA construction:** 5 seeds, as for R1 and RA.
   - Answer spans are placed longest first. Each goes to a seeded random answer-free host (an Oracle bullet's content
     after `- `, or an NLA sentence) with at least as many tokens. The first l tokens are cut at the tokenizer's
     offsets.
   - If any seed runs out of hosts, PA is not computable for that text.
   - Smoke coverage: Oracle 10/20, NLA 12/20. An Oracle readout whose bullets all open with the answer has no host, so
     **Δ_AP is measured on readouts with more wrong or placeholder bullets.**
3. **RA:** built as R1 (clause pool = all clause units). On Oracle texts, a bullet's first clause unit includes the
   `- ` mark, which the Oracle AR does not score. So RA can delete about one scored token fewer than CA per
   bullet-initial clause it takes.
4. **New kinds:** wrong_entity and false_fact use 013's construction with new salts (clause 61 / 71, sentence 81 / 91).
   013's kinds keep 013's salts. The rebuilt NLA versions equal 013's stored texts for 20/20 smoke explanations.
5. **Emptied bullets:** a line with no word character after deletion (e.g. a lone `-`) is dropped from the Oracle AR's
   average. A version with no piece left is degenerate and counted.
6. **J share under the Oracle AR:** 013's form on unit-normalised raw output, centred at the mean of the unit-normalised
   raw L44 gold over the 261 targets (direction only, so the AR's per-layer scale cancels).
7. **Per-share version:** a ratio of means, Σ Δ / Σ deleted token share, with an item-cluster bootstrap. A per-text
   ratio is unstable for small deletions. For the reader contrast it is computed per reader, then differenced.
8. **English-only subset:** the (item, state, sample) whose Oracle readout has no non-Latin script. The NLA side uses
   the same keys, so the subset stays paired.
9. **Added descriptive check:** F − P on the texts whose P is within 20% of F's edit size (Oracle P; NLA P*).

Nothing has been scored on smoke versions yet.

## 2026-10-11: step 3, smoke scored and analysed (report `claude_ovarida/46_015_smoke_2026-10-11.md`)

- **Scoring:** both scorers, every version of both readers' 20 paired texts (1,170 unique texts, 1,822 entries; Oracle AR
  102 s on GPU 0, NLA AR 178 s on GPU 1). Results `results/015_oracle_hallucination/analysis/oh_smoke.json`.
- **Checks:**
  - G-repro exact: the NLA AR reproduces 013's MSE at 0.0 on 252 version values and on F/P, and the versions are
    identical 20/20;
  - Oracle O vs step 2: max 0.25 points (layout);
  - rewrite gates pass for both readers;
  - G3 κ 0.67.
- **Descriptive, n 20 (paired; 13 (item, state)):**
  - Oracle texts × Oracle AR: Δ_A +15.9 [+7.0, +24.1] (n 19), Δ_AP +5.3 [−0.7, +12.1] (n 10, unresolved), primary
    +1.1 [−6.2, +8.5];
  - NLA texts × NLA AR: primary +29.1 [+9.1, +54.1] (013's pattern);
  - F − P: Oracle −7.6 [−12.4, −3.5] (Oracle AR), NLA +7.8 [+3.7, +11.9] (NLA AR).
- **Found in analysis:**
  - Under the Oracle AR, every NLA version lies within −171.0 to −164.1 points, so the paired reader contrast there is
    mostly the Oracle side (report 46 §4a).
  - The Oracle's F mostly deletes unflagged follow-on text (§2c).
  - **013's controls:** in 013's eligible set, R1 is empty for at least one seed on 12/474 explanations and RS1 on
    75/474. An empty text scores far below C1, so this biases 013's Δ toward 0. Observation only; 013 is unchanged.
- **Open for the PI** (report 46 §4):
  - 4a: scale-free descriptive view of the reader contrasts;
  - 4b: a correction-only Fc arm;
  - 4c: PA coverage;
  - 4d: Oracle misquote tags;
  - 4e: a note in the 013 ledger.
- Full run not started.

## 2026-10-11: full-run amendment (written and committed before any full-run judge call)

**The PI's go:** "okay start it", in reply to report 46. No choice was made among §4's options, so my recommendations are
adopted. All of the following are POST HOC relative to the spec and none changes a pre-registered quantity.
- **4a. Scale-free reader view (descriptive):** each reader's Δ divided by its cell's step-2 matched − shuffled mean
  (`r1.json`), × 100, then differenced (NLA − Oracle) as §7. The §7 claim rule is unchanged and uses the raw contrasts.
- **4b. Fc arm (Oracle readouts with ≥ 1 flagged span).**
  - Judge A, temperature 0, with the readout's own A spans listed. Exact text: `oh.py` `PROMPT_FC`. It corrects each
    listed span in place, leaves a span that can't be corrected in place (invented context) unchanged, and deletes or
    adds nothing.
  - Pc: PROMPT_P with Fc's realised edit size, the same 20% re-ask, delimiters stripped.
  - Reported: Fc − Pc, Fc − O, Pc − O, F − P on the same readouts, and Fc − Pc with Pc within 20%. Both scorers,
    descriptive.
  - Checks: Fc keeps the line count; hand check of 10 Fc.
- **4c.** Δ_A on the PA-computable subset is reported next to Δ_AP. PA is unchanged.
- **4d.** The Oracle per-kind misquote results are reported as unreliable, unless the 10-readout extra hand check shows
  correct misquote tags.
- **4e.** A note in the 013 ledger (appended today).

**Mechanics (no effect on any number):**
- The judge runs 16 threads in chunks of 64. It aborts if the projected spend passes $10; the projection over-estimates,
  because Oracle rows come first and cost more.
- The NLA AR runs in 3 shards (unique text index mod 3), one text per forward as before, so a shard can't change a
  score: shards 0 and 1 on GPU 1, shard 2 on GPU 0 alongside the Oracle AR (about 91 GB).
- The Oracle AR accumulates running sums.
- The analysis asserts that every version entry is scored exactly once per scorer.

**Regression check of the new analysis code on the smoke:** 0 differences in every previously committed field of
`oh_smoke.json`. The regenerated file adds `dA_on_PA_subset`, the scaled contrasts and the Fc fields (empty in the smoke).

**Correction to report 46 (my mistake, found by the new field):**
- §2a said "the position-matched control eats most of that gap". That compared Δ_A on 19 readouts with Δ_AP on a
  different 10.
- On the same 10, Δ_A is +7.1 [+2.1, +12.4] against Δ_AP +5.3 [−0.7, +12.1]. So the position control accounts for about
  a quarter of the answer effect.
- Report 46 is corrected in place, marked with the date.

**G2's 10 extra readouts:** `random.Random(15).sample` of the non-smoke Oracle readouts (`oh.py handset --subset
extra10`). Hand-checked before the full analysis is opened.
