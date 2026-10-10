# 015: The hallucination test on the Oracle Lens, with cross-scoring (design spec, 2026-10-11)

**Status:** APPROVED by the PI on 2026-10-11 with six changes, applied below. Committed before any judge call or
reconstructor forward. Branch `015-oracle-hallucination`, off `014-typo` at `b975986`. EXPLORATORY.
- **Source:** the PI's brief (2026-10-11).
- **Reuses** the 013 scripts with a reader switch (`NLAH_READER=oracle`), as 014 did with `NLAH_FAMILY`. Outputs go to
  `outputs/015_oracle_hallucination/`, results to `results/015_oracle_hallucination/`, ledger
  `docs/ledger/015_ledger.md`.
- **Committed and approved before any judge call or reconstructor forward.** Deviations from the brief are marked
  *deviation*, with the reason.

## 1. Question

013 found that the NLA's false claims carry above-average signal for the NLA's own reconstructor, almost all of it from
misquotes of the prompt. CW-20 found close to the opposite for the Oracle Lens under the Oracle's reconstructor: the
rebuild leans on the answer at the start of each bullet, and the wrong pieces count for no more than random text.

The two reconstructors are trained differently:
- the NLA's maps a description to the activation it describes;
- the Oracle's maps a span to the activation *before* it.

So the contrast could be about the readers' texts or about the reconstructors. **015 runs the Oracle through 013's
exact pipeline on 013's items and states, and scores every text of both readers with both reconstructors.** A 2 × 2
(text source × scorer) is what can license a reader-specific statement.

**Second question (the answer arm).** Does the Oracle's reconstructor reward the answer itself, or just whatever text
opens a bullet? CW-20 could not separate the two, because the answer always opens the bullet. A position-matched
control does.

## 2. Material (all saved; nothing new is generated from the subject model)

**Items and states:** 012's 87 covered multihop items; states clean, `J_L24-35` and `rand_L24-35` (013's).

**Oracle readouts:** `outputs/012_reader_concordance/stageC/readouts_oracle.jsonl`, `layer == 44`, the three states,
2 samples each: 522 readouts (checked: 87 × 3 × 2).
- Made by `stageC_readers.py` (`ol.read_batch`, k = 2, seed `C.SEED`).
- Every readout parses into 4–5 bullets: median 4, none empty (checked).
- Bullets are lines starting with `- `; the text is the whole readout.
- Each bullet continues the prompt, usually an answer and then a follow-on, and is often cut off mid-sentence by the
  generation budget.
- **294 of the 522 readouts (56%) contain CJK characters** (checked; CW-12 reported 30–72% by piece).
- L40 readouts are a replication set, used only if the PI asks.

**NLA explanations:** 013's 522 (`hallucination.explanations()`), with their judge outputs, span kinds and rewrites
(`outputs/013_nla_hallucination/judge/`, `rewrite/`).

**Gold activations:** Stage B `captures.pt` `X[item, state, L]`. L44 is the Oracle's read layer and the Oracle AR's
target; L42 is the NLA's. Both are the final-token block outputs the readers read.

**The model's continuation per state:** `stageB/items.jsonl` `arms[state].text`, as in 013.

**Unit and clusters:** one readout or explanation is the unit; bootstrap clusters are items (87).

## 3. Scorers

### 3a. The Oracle AR (scores any text at L44)

**Loader:** as accepted for L36–60, directions only (`CLAUDE.md` 012 E0, `d4b9860`).
- `[Layer 44]` tag + span, read `ln_pre`, raw head output (011 `ar_vectors.attach_ar` / `ar_forward`, unchanged).

**Feeding a text:**
- **Oracle readouts:** bullet by bullet, without the list mark, predictions averaged (CW-20).
- **NLA explanations:** sentence by sentence (013's splitter), predictions averaged.
- Any piece longer than 32 AR tokens is split at clause boundaries (`,` `;` `:`; else at 32 tokens). The share of pieces
  split is logged.
- Pieces are tokenized as written, with no added leading space (CW-20's `cw20_ar.py`:
  `tok(text, add_special_tokens=False)` after stripping `- `).
- A version in which every piece is empty is degenerate and is counted.

**Primary metric: 013's exact form, in the Oracle AR's whitened, centred space (PI change 2).**
- Whiten with `whitening_iolens_pooled_L44` and centre at its mean μ: ŷ_w = W(ŷ − μ), y_w = W(y − μ).
- Normalise both to unit norm: n(·) = · / ‖·‖.
- MSE = ‖n(ŷ_w) − n(y_w)‖² = 2 − 2 cos.
- B = the predict-the-mean baseline of the 261 normalised targets: B = mean_i ‖n(y_w,i) − mean_j n(y_w,j)‖².
- FVE = 1 − MSE / B. Retrieval ranks by FVE, which is equivalent to ranking by cosine.

**B is computed from the gold targets alone** (before any text is scored):

| Scorer | B | ±2 FVE points in cosine | FVE = 0 at cosine |
|---|---|---|---|
| Oracle AR (L44, whitened, centred) | 0.7032 | ±0.0070 | 0.648 |
| NLA AR (L42, 013's B) | 0.2818 | ±0.0028 | 0.859 |

The NLA value reproduces 013's recorded `baseline_B` (0.28180).
- Since FVE = 1 − (2 − 2 cos)/B, a change of 2 points is a cosine change of 0.01 × B. The same number of points is a
  2.5× larger cosine change under the Oracle AR, which is one more reason points are never compared across scorers.
- **Pre-stated:** the Oracle-AR level may be negative (CW-20's mean cosine 0.46 gives FVE about −0.54 here). Only
  differences are read. **The mean cosine is reported next to every FVE.**

**Secondary metric (CW-20's):** whitened FVE, corpus-centred at μ, after one global least-squares scale fitted on the
O versions of the set, plus the mean cosine. Reported so our numbers sit beside hers on the 61 items we share (WSB
multihop items 31–100 that are among the 87).

**Whole-readout check:** the O version scored as one span, once (CW-20 found it near zero).

### 3b. The NLA AR (scores any text at L42)

013 §5 unchanged: the final nla-rl `ar_reconstructor` through EasyNLA's `NLACriticModel` + `critic_predict`, critic
template, `mse_scale` √5120; FVE against the 261 L42 targets' predict-the-mean baseline (B = 0.2818, as 013). Any text
is fed whole. The mean cosine is reported next to every FVE.

**Caveats (stated, not fixed):**
- the two scorers sit on their native layers (L42 vs L44), and the texts were read at those layers;
- **the NLA AR is an English critic, and 56% of Oracle readouts contain Chinese.** Every cross-scored result is also
  reported on the English-only subset (readouts with no CJK or other non-Latin script; *deviation*: the brief asks only
  for a descriptive flag).

### 3c. J share (secondary, descriptive)

As 013 §5: the share of each reconstruction's centred raw output in 007's dense J basis at the scorer's layer
(`J_raw_L44` for the Oracle AR, `J_raw_L42` for the NLA AR). Reported for O, C1, R1, F and P, next to the gold
activations' own share.

## 4. Judges

013 §3 unchanged: `google/gemini-3.8-flash` (A: spans, rewrites) and `z-ai/glm-5.3-flash` (B: sentence labels, kinds,
answer spans); temperature 0, reasoning "low"; every call cached and cost-logged; key in `/workspace/.env`.
- 013's `DEFS` and `STATE_NOTE` are used verbatim.
- **Input framing for Oracle readouts.** The only change is the description line before the text. Exact text:
  > READOUT (text written by another system from the model's internal activation at the last prompt token. It is a
  > list of bullets; each bullet continues the prompt, usually with an answer and then a follow-on. Bullets may be cut
  > off mid-sentence. Some text may be in Chinese or another language: judge it by its meaning):

  The NLA framing is 013's, plus the same final sentence about other languages.
- **Prompts A, B and K:** 013's `PROMPT_A`, `PROMPT_B` and `PROMPT_K`, with "EXPLANATION" replaced by "READOUT" for
  Oracle texts. Sentences come from 013's splitter (newlines split bullets).
- **New: the answer-span prompt (judge B), for both readers.** Exact text, with `{inputs}` built as above:

  > You find statements of the model's answer in an automatically generated text about a language model's internal
  > state.
  >
  > {inputs}
  >
  > An ANSWER STATEMENT is a span that states the model's continuation (shown above) as the answer or as the next words,
  > in this or other wording, in any language. Copy the shortest span that does this: the answer word or phrase,
  > together with any words in the same clause whose only job is to present it as the answer (for example "the answer
  > is", "the next word is"). Do not include follow-on questions or other content. If a bullet or sentence opens with
  > the answer, the span is that opening. A span that states a DIFFERENT answer is not an answer statement.
  >
  > Return JSON only: {"answer_spans": [...]}. Each span is copied EXACTLY, character for character, from the text.
  > Spans must not overlap. An empty list is allowed.

- **Rewrites (013 A2).** F's prompt is 013's. P is 013's with one fix (*brief*): the target edit size passed is F's
  realised word-level edit distance. The realised P distance is then checked, and P is re-asked once (same prompt plus
  "Your previous paraphrase changed {n} words; change about {target}") if it is off by more than 20%. The match is
  reported.
- **NLA P redo for parity (PI: yes; a new POST HOC NLA arm, PI change 6).** The NLA's P is redone the same way, so both
  readers' F − P use matched paraphrases.
  - 013's P changed about 47 words against F's 61 (`rewrite_all.json` `words_changed_mean`).
  - **013's recorded F − P (+3.1 [1.3, 4.9]) stays as recorded.** The redo is reported as a separate arm, P*.

## 5. Versions

Built deterministically. No LLM writes new text, except F and P.

**Shared with 013, unchanged in construction:**
- O; C1 / C2;
- R1 / R2 (token-matched random clause deletions, 5 seeds);
- K1, RS1;
- per-kind deletions with clause and sentence random controls. All four kinds (misquote, invented_context,
  wrong_entity, false_fact) are run (*deviation*: 013 ran misquote and invented_context only). For the Oracle,
  wrong_entity (a wrong answer opening a bullet) is the kind CW-20's "wrong pieces" mostly were.
- Token counts use the AV tokenizer for NLA texts and the AR tokenizer (Qwen) for Oracle texts. Deletions keep bullet
  structure: a bullet emptied by a deletion is dropped from the Oracle AR's averaging.

**New, the answer arm (both readers):**
- **CA:** O with judge B's answer spans deleted.
- **RA:** token-matched random clause deletion (5 seeds, built as R1).
- **PA (position-matched):** for each answer span (length l tokens), delete the first l tokens of a different bullet
  (Oracle) or sentence (NLA) that contains no answer span.
  - Hosts are chosen seeded, without replacement.
  - Not computable if there are fewer answer-free hosts than answer spans (counted).
  - On the Oracle, an answer-free bullet usually opens with a *wrong* answer or a placeholder. So CA − PA reads as "the
    true answer vs other text at the same position".
- **Eligibility:** ≥ 1 answer span, and CA keeps ≥ 10 tokens.

**Rewrites:** F, P (matched as §4) and O.
- **F's composition is logged per readout (PI change 5):** tokens deleted vs tokens corrected, from a word-level
  alignment of O and F (deleted runs vs substituted runs). The Oracle's F is expected to mostly delete follow-on
  questions.

**Non-English flag:** each readout and version carries `has_non_english` (CJK or other non-Latin script, by regex).

## 6. Cross-scoring

Every version of both readers' texts is scored by both scorers:

| Texts | Scored by |
|---|---|
| Oracle | the Oracle AR (§3a, bullets) and the NLA AR (§3b, whole text) |
| NLA | the NLA AR (013's own scores are reproduced, G-repro) and the Oracle AR (§3a, sentences) |

## 7. Decision rules (fixed now)

**Bands, within each scorer, in its own FVE points (013 §6):**

| Band | Condition |
|---|---|
| H-signal | ≥ +2 with the CI above 0 |
| H-padding | ≤ −2 with the CI below 0 |
| No difference | CI inside ±2 |
| Unresolved | otherwise |

Cluster bootstrap by item, 2,000 draws, seed 0.

**Primary (Oracle):** Δ = FVE(R1) − FVE(C1) for Oracle texts under the Oracle AR, pooled over states, on eligible
readouts (013's definition).

**Co-primary (the answer arm, Oracle texts under the Oracle AR):**
- (a) CA against RA: Δ_A = FVE(RA) − FVE(CA);
- (b) Δ_AP = FVE(PA) − FVE(CA). Positive means deleting the true answer costs more than deleting other text at the
  same position.

**Points are never compared across scorers.**

**Sign convention against CW-20 (PI change 6).** All our deltas are FVE(control) − FVE(cleaned). CW-20 reports the
change from removing a piece against random removals. So her "removing the answer −7.7" corresponds to our Δ_A ≈ +7.7
(in her metric, not ours).

**Reader-specific claims only for pre-named contrasts (PI change 3):**
- **(a) the answer arm, Δ_A and Δ_AP: the main contrast;**
- **(b) the primary false-claim Δ.**

Every other contrast is descriptive.

**Deletion size (PI change 4).** For every reader comparison, each Δ is also reported divided by the deleted token share
(tokens deleted / tokens in O). The NLA's flagged text is about 64% of its tokens (013); CW-20 found 13% of the Oracle's
text wrong. **A reader-specific claim must hold on both the raw and the per-share version.**

**A reader-specific claim** (pre-stated; pre-named contrasts only) needs, for a given contrast:
- under one scorer: a paired difference between NLA texts and Oracle texts on the same (item, state), averaged over
  the 2 samples, with a CI excluding 0 and |difference| ≥ 2 points;
- the same sign under the other scorer;
- the same sign on the English-only subset (PI: yes);
- the same sign, CI excluding 0, on the per-share version (PI change 4).

If it holds under one scorer and reverses or vanishes under the other, it is attributed to the reconstructor.

**Pre-stated expectations (written to be wrong):**
- Oracle false claims read as padding or no difference under the Oracle AR;
- under the Oracle AR, deleting the answer costs more than RA and more than PA;
- under the NLA AR, misquotes carry the rebuild for NLA texts (013 replicates).

## 8. Gates (fixed now)

- **R1, the de-risk, on all four cells (text source × scorer), O versions only, no judge calls (PI change 1).** On the
  87 clean first samples, compare the own text against a shuffled text (another item, same state, seeded derangement).
  - **The stop gate is the Oracle AR × Oracle texts cell.** It must pass both conditions below.
  - The other three cells are sensitivity checks. A cell whose matched − shuffled CI does not exclude 0 cannot support
    or refute a reader-specific claim, and is reported as such.

  Pass conditions for the gate cell:
  - matched − shuffled with a CI above 0 and ≥ 10 points;
  - retrieval@1 among the 87 clean targets ≥ 0.5 (rank by the §3a metric). CW-20 gives 0.84 among 70.

  **If R1 fails at L44, stop.**
- **Determinism:** O's Oracle-AR score on the 87 clean first samples is identical across two runs.
- **G-repro (NLA side):** NLA texts' O scores under the NLA AR reproduce 013's to ≤ 1e-6.
- **G1–G3 (013 §7)**, for the new judge calls too:
  - G1: JSON parse plus verbatim spans, ≥ 90%;
  - G2: my hand check of A's contradicted spans on Oracle readouts, precision ≥ 0.80 (20 smoke readouts, then 10 more);
  - G2k: kind tags, agreement ≥ 0.80 or the per-kind results are reported as unreliable;
  - G2a (new): my hand check of the answer spans on the 20 smoke readouts and 10 NLA explanations, precision ≥ 0.80;
  - G3: A–B κ, descriptive.
- **Rewrite gates (013 A2):** F's contradicted-sentence rate ≤ 0.10; P keeps ≥ 0.7 of O's; my hand check of 20 F and
  10 P; P's edit-size match reported.
- **Consistency with CW-20 (not a gate):** on the 61 shared items, our clean L44 level and cosine under her metric,
  next to hers (+0.21, cos 0.46). Her write-ups are her own samples, so the comparison is approximate.

## 9. Order of work, with stops

1. **This spec:** PI approval, then commit.
2. **R1 de-risk on all four cells** (O versions, clean first samples, no judge calls), plus the determinism check,
   the whole-readout check and the CW-20 consistency numbers. **Stop and report all four cells before any judge
   call.**
3. **Smoke:** 20 Oracle readouts by 013's rule (`random.Random(13)`, stratified 7 clean / 7 J / 6 random). By
   construction these are the same (item, state, sample) as 013's NLA smoke, so the smoke is paired.
   - All versions, both scorers, the hand checks.
   - Stop and report.
4. **Full run** on the PI's go:
   - all 522 Oracle readouts;
   - the NLA answer-arm labels (and the NLA P redo, if chosen);
   - cross-scoring of all 1,044 texts' versions.
   - **Spend cap $10**; the run aborts if the projected spend passes it.

## 10. Cost (estimates)

- **Judges:** about 6 calls per Oracle readout (A, B, K, answer, F, P plus re-ask), plus about 2 per NLA explanation
  (answer, P redo). About 4,200 calls. At 013's realised cost (about $2.35 for its judge calls),
  about $3–5; every call is cost-logged.
- **GPU** (two in parallel: the Oracle AR on one, the NLA AR on the other):
  - Oracle AR: about 75 versions × 522 Oracle readouts × about 5 pieces, plus NLA texts at about 8 pieces. About
    500k short-span forwards at the measured rate (about 107 spans/s): about 1.3 GPU-h.
  - NLA AR: about 75 versions × 522 Oracle texts, whole text: about 1 GPU-h.
  - In parallel, about 1.5 h wall-clock.

## 11. Not claimed

- **Whether either reader reads what the model uses.** The reconstructors are the readers' own co-trained critics, so
  this measures what each critic rewards, not what Qwen computes.
- **Anything beyond** multihop at the final token, L42/L44, these two readers, these judges and these samples.
- **The Oracle's reward definition** (the published 0.487 "joint" figure) and its reconstructor's loss.
- **Cross-scoring is limited:**
  - the layer mismatch (L42 vs L44);
  - the NLA AR scoring Chinese text;
  - the Oracle AR scoring prose sentences it wasn't trained on (it was trained on assistant-text spans).

## 12. Decisions (PI, 2026-10-11)

1. **Approved, with six changes** (applied above):
   - R1 on all four cells;
   - 013's exact metric with B stated;
   - reader-specific claims only for the pre-named contrasts;
   - the per-share requirement;
   - F's composition logged;
   - the NLA P redo as a post hoc arm, with the sign convention stated.
2. **NLA P redo:** yes.
3. **English-only subset as a required condition:** yes.
4. **All four span kinds:** yes, descriptive.
5. **Branch:** `015-oracle-hallucination`, off `014-typo` at `b975986`.
