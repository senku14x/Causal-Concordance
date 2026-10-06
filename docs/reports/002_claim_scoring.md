# 002 — Reader claim scoring (role-aware, closed-set)

*Qwen3.6-27B · WorkspaceBench brew · dev only · design `docs/design_specs/002_claim_scoring.md` (v2.2) ·
2026-10-02.*

## Scope

- **Material:**
  - verbalizers: the 3,930 saved 001 snapshot readouts on 10 brew tables;
  - token lenses: the saved 001 activation captures on all 48 dev tables.
  - No new reader generations.
- **What is scored:** what each reader **claims**, not what the model uses. No causal label exists yet; 002 cannot
  rank readers causally, estimate W, or say which named content drives the answer.
- **Status:** descriptive and exploratory. With 10 tables for the verbalizers, thousands of judged readouts are not
  thousands of independent examples.
- **Wording:**
  - "role-correct" = the judge reads the text as putting the true colour in that role;
  - "tracks" = the claim or score moves the right way when, within one table, the start colour is swapped (which
    changes the correct intermediate).

## Provenance

| Stage | Output | Notes |
|---|---|---|
| Sets | `validation/{pilot,validation,key}.jsonl` | seed 20261004; key made variant-aware after the design review (25 partner rows had been wrong) |
| Judge pilot | `judge/pilot_claims.jsonl` (pass 2) | pass 1 was overwritten (an early-code slip, now fixed); prompt frozen at sha e1cc74d7 |
| Judge | `judge/claims.jsonl`, `manifest_judge.json` | `google/gemini-3.8-flash` via OpenRouter (WorkspaceBench's judge), reasoning minimal, temperature 0, strict JSON enum schema; 3,930 / 3,930 valid, 0 errors |
| Discarded | `judge/discarded_5cand_partial.jsonl` | 302 readouts judged with the wrong candidate sets (v2), stopped; see §10 |
| Lens scores | `lens/colour_scores.json` | all 12 colours × 5 lens readouts × 3 cells × 5 layers × 281 renders; " Teal" (two tokens) skipped, " teal" kept |
| Prompt-only | `prompt_only/colour_logprobs.json` | stock model, 231 natural dev renders × 3 questions |
| Analysis | `analysis/{scores.json, readout_scores.jsonl, qualitative.json}` | — |

**Spend:** $7.98 in total.
- $7.37: main judge run;
- $0.555: discarded partial run;
- $0.058: pilot passes;
- $0.001: acceptance check.

The design estimated $2.35. The difference is about 385 reasoning tokens per call, which Gemini spends even at
"minimal" effort.

## 1. What was run

1. **Judge:** each verbalizer readout goes to Gemini 3.8 Flash with only the table's closed candidate set:
   - the member's path, the partner's path and WSB's off-path options, 6–8 colours;
   - the judge returns which candidate the text puts in START, INTERMEDIATE and ANSWER, or "none" (an explicit value
     outside the set), "cannot tell" or "conflicting".
2. **Lenses, judge-free:** each lens's score for every colour at every cell, judged by the A01 rule. Hit = the
   colour ranks first among the render's 5 candidates (start, z, answer, two off-path colours) **and** scores above
   the same lens on the dev-mean activation.
3. **Prompt-only task baseline:** the stock model, given the full prompt, is asked for each colour.

## 2. Judge validity (gate passed)

| Set | Role | Exact agreement with the author | Cohen's κ |
|---|---|---|---|
| Validation, 60 (stratified; mostly non-claims) | intermediate | 0.77 | **0.67** |
| | start / answer | 0.75 / 0.75 | 0.66 / 0.60 |
| Positive-claim supplement, 30 (the judge assigned a candidate; author blind to the judge's labels) | intermediate | 0.83 | **0.81** |
| | start / answer | 0.77 / 0.87 | 0.70 / 0.73 |

- Gate: κ ≥ 0.6 on the intermediate role. Judge errors: 0%.
- Acceptance check (three synthetic chains for table 0009, including the partner's black → orange → green): passed.
- **Direction of disagreement:** where the author coded "conflicting" because an Oracle sample's bullets disagree,
  the judge sometimes picked the dominant colour (all 5 disagreements in the positive set). So judge counts of
  role-correct and role-incorrect claims may run slightly high relative to strict coding.
- The author codes are another model's annotations, not human ground truth; the PI spot-check is still open.

## 3. What the verbalizers claim (the intermediate role, member activation)

| Reader, cell | Arm | z correct | incorrect (other candidate + non-candidate) | no claim | conflicting | z in the wrong role | **mention(z)**, the 001 measure |
|---|---|---|---|---|---|---|---|
| NLA L42, assistant newline | real | **0.15** | 0.60 (0.12 + 0.47) | 0.05 | 0.20 | 0.07 | 0.55 |
| | shuffled | 0.07 | 0.57 | 0.10 | 0.25 | 0.05 | 0.25 |
| NLA L42, `</think>` newline | real | **0.15** | 0.72 (0.45 + 0.28) | 0.00 | 0.12 | **0.28** | 0.60 |
| | shuffled | 0.05 | 0.68 | 0.07 | 0.20 | 0.00 | 0.20 |
| NLA L42, `:` | real | **0.05** | 0.42 | 0.15 | 0.38 | 0.17 | 0.43 (corrected 2026-10-06, report 31: 0.425 was written 0.42) |
| | shuffled | 0.00 | 0.47 | 0.20 | 0.33 | 0.00 | 0.20 |
| Oracle L40, `:` (k = 4) | real | **0.00** | 0.12 | 0.03 | **0.85** | 0.05 | 0.53 |
| | shuffled | 0.00 | 0.23 | 0.00 | 0.78 | 0.00 | 0.38 |
| | none | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| Oracle L44, `:` (k = 4) | real | 0.03 | 0.17 | 0.23 | 0.57 | 0.00 | 0.30 |

40 readouts per row (10 tables × 4 samples). WSB-grid Oracle rows (k = 1, L20–60) are on the results page; the
role-correct rate there is 0–0.10 at every layer.

- **The 001 mention rates overstate what the verbalizers claim.**
  - NLA at `:`: z is mentioned in 42% of readouts but put in the intermediate role in 5%.
  - Oracle L40: z is mentioned in 53%, never placed in its role. 85% of the Oracle's samples contradict themselves
    (different colours for the same role across bullets).
- **The NLA places z in the wrong role** in up to 28% of readouts (`</think>` cell), usually as the start of a
  two-step chain. This matches the role reading of report 001 §8.
- **Real vs shuffled:** role-correct z is higher with the real activation for the NLA (0.05–0.15 vs 0.00–0.07) and
  not at all for the Oracle.

## 4. Do the claims follow the activation? (within table, 10 tables)

| Reader, cell | Start-swap tracking on role claims (mean; tables + / − / tie) | Real − shuffled | End-to-end bridge (real vs SEC capture, same intermediate label) |
|---|---|---|---|
| NLA, assistant newline | **+0.14** (6 / 2 / 2) | +0.08 (2 / 2 / 6) | — |
| NLA, `</think>` newline | +0.03 (3 / 4 / 3) | +0.10 (4 / 1 / 5) | — |
| NLA, `:` | +0.08 (4 / 2 / 4) | +0.05 (2 / 0 / 8) | **0.48** |
| Oracle L40, `:` | +0.04 (2 / 0 / 8) | 0.00 (0 / 0 / 10) | 0.81 |
| Oracle L44, `:` | +0.01 (2 / 1 / 7) | +0.03 (1 / 0 / 9) | 0.63 |

Bootstrap intervals and unadjusted sign-flip p-values are in `scores.json`; none is confirmatory.

- **Weak at best.** On mentions, report 001's audit follow-up found tracking of 0.16–0.35. On **role claims** it
  shrinks to 0.01–0.14.
- **Bridge:** the bridge compares two separately generated readouts from nearly identical activations. The NLA's
  intermediate label agrees only 48% of the time. Most of that is sampling variability in the reader, not the
  capture regime (stock vs SEC activations agree at cos ≥ 0.99985).

## 5. Token lenses: closed-set scores (48 dev tables; judge-free)

Hit rate on z at the `:` cell (A01 rule; chance with 5 candidates is about 0.2 before the reference requirement):

| Lens | L40 | L42 | L44 | L56 | L60 |
|---|---|---|---|---|---|
| Logit lens | 0.28 | 0.27 | 0.24 | 0.12 | 0.07 |
| J_NP (standard) | 0.37 | 0.35 | 0.36 | 0.11 | 0.07 |
| J_NP (cosine) | **0.45** | **0.43** | **0.39** | 0.10 | 0.08 |
| J_CB | 0.38 | 0.29 | 0.35 | 0.14 | 0.06 |
| R_CB | 0.43 | 0.33 | 0.34 | 0.16 | 0.06 |

The answer hit rate is 0.18–0.24 at L40–44 and 0.81–0.89 at L56–60.

**Post hoc (added after seeing the table above; exploratory): within-table tracking and a shuffled baseline.**
Per table, take the lens's score for the member's colour minus the partner's colour, on the member render minus on
the partner render. Counts are tables with positive / negative values, out of 48:

| `:` cell | z | start | answer | Member hit on z, real / shuffled |
|---|---|---|---|---|
| J_NP cosine, L40 | **48 / 0** | 21 / 27 | 42 / 6 | 0.46 / 0.17 |
| J_CB, L40 | **48 / 0** | 19 / 29 | 40 / 8 | 0.33 / 0.08 |
| Logit lens, L40 | 44 / 4 | 22 / 26 | 32 / 16 | 0.12 / 0.10 |
| J_NP cosine, L44 | **47 / 1** | 25 / 23 | 38 / 10 | 0.38 / 0.15 |
| J_CB, L60 | 47 / 1 | 22 / 26 | **48 / 0** | 0.00 / 0.08 |

The full grid (every lens, cell and layer) is in `scores.json` → `lens_post_hoc`. The two newline cells show the
same pattern, with member hits up to 0.58.

- **The lenses carry item-specific information about z at L40–44.** It sits below their top-10 cutoff, where task
  words dominate ("second", "again"). Report 001's "no lens names a colour at L40–44" is true of the top-10 lists
  and false of the lenses' colour scores.
- **The signal is z-specific, not a generic trajectory signal.** The start colour is not tracked (at chance), although
  it is the most linearly decodable content at that cell (probe 0.81). The answer is partly tracked at L40–44 and
  fully at L60.
- **Scope:** this is a ranking among candidate colours (budget 1, no role), not a claim a lens "makes" in its
  readout. It is a different target from a verbalizer's explicit role claim, so this is not a matched reader ranking.

## 6. Prompt-only task baseline

The stock model, given the full prompt and asked:

| Question | All 48 tables (231 renders) | The 10 snapshot tables (20 renders) |
|---|---|---|
| Start | 1.00 | 1.00 |
| Intermediate | **1.00** | 1.00 |
| Answer | 0.89 | 1.00 |

The 231 renders include partners that failed the gate. So z is fully inferable from the prompt. This is a task
baseline, not a ceiling on readers, and it measures the full prompt, not the prefix visible at a cell.

## 7. Overlap between readers (10 tables, member activation; exploratory)

| Pair | Level | Both | Only first | Only second | Neither |
|---|---|---|---|---|---|
| NLA L42 `:` vs Oracle L40 `:` | mentions z | 6 | 0 | 1 | 3 |
| | asserts z in role | **0** | 2 | 0 | 8 |
| | the same wrong intermediate asserted | 3 tables | | | |
| NLA L42 `:` vs Oracle L44 `:` | asserts z in role | 1 | 1 | 0 | 8 |
| | the same wrong intermediate | 4 tables | | | |
| NLA L42 `:` vs J_NP cosine L42 top-10 | mentions z | 0 | 6 | 0 | 4 |

The two verbalizers agree on *mentioning* z far more than on *claiming* it, and they share wrong intermediates in 3–4
of 10 tables. This is the motto's case in miniature: agreement on mentions is not agreement on claims, and some of
it is shared error. (Different layers; 10 tables.)

## 8. Predictions

| # | Prediction | Prior | Outcome |
|---|---|---|---|
| R1 | NLA L42 `:` role-correct(z), real, < 0.25 | 60% | **True** (0.05) |
| R2 | Oracle L40 `:` real − shuffled ≥ 0.10 | 40% | **False** (0.00) |
| R3 | NLA start-swap tracking (roles) positive in ≥ 7/10 tables, assistant newline | 45% | **False** (6/10) |
| R4 | Judge–author κ, intermediate ≥ 0.6 | 70% | **True** (0.67) |
| R5 | Lens hit on z at L40–44 ≤ 0.2 for every lens; answer hit ≥ 0.8 at L60 | 70% | **False**: z hit reaches 0.56 (assistant newline); the answer part holds (0.85–0.89) |
| R6 | Prompt-only picks z in ≥ 0.95 | 85% | **True** (1.00) |
| R7 | "Conflicting" is the most common Oracle contract outcome | 50% | **True** (177/240) |

## 9. Qualitative samples (full texts and judge outputs in `analysis/qualitative.json`; browsable on the results page)

- **NLA, role-correct** (0104 partner, black → white → orange): "…blue→white→red→black, so … 'white' was first
  step…". The judge reads intermediate = white (correct), start "none" (blue), answer conflicting. A correct
  intermediate inside an otherwise invented chain.
- **NLA, z in the wrong role** (0027 substituted, orange → pink → black): "'Pink → Yellow' established; 'Pink traced to
  the second iteration'…". Here z (pink) is the start of the NLA's chain.
- **NLA, another candidate** (0069 member, black → teal → orange): "'purple → green → [next step result]' …
  nextColor('green') yields 'red'…". The judge reads intermediate = green; the true z is teal.
- **Oracle, conflicting** (0071 member, blue → gold → brown): four bullets with four different chains ("Blue → Green →
  Red", "Yellow → Red → 1", "0 → Purple → 2", "2 → Blue → Red").
- **Oracle, role-correct** (0041 substituted, teal → black → gray): "…the chain is Poison -> Black -> White…". z
  (black) is in the right slot of an invented chain. The author coded this sample "conflicting", because another
  bullet makes red the step-2 colour.

## 10. Breakages and silly mistakes (logged in CLAUDE.md)

- **Candidate sets (v2):** partner readouts were judged against the member item's 5 options, which need not contain
  the partner's path, so tracking could not be scored on roles. Found while writing the lens stage. 302 readouts
  ($0.555) were discarded; v2.1 uses one set per table. The external review found the same flaw independently.
- **Truth key:** 25 partner rows of `key.jsonl` carried the member trajectory. The analysis resolved truth by
  variant all along; the file is fixed.
- **"None" counted as abstention** in the v2 design. Under v2.1 it is an incorrect claim (review point 3).
- **Pilot pass 1 overwritten** by the pilot stage; passes are now archived, and its $0.031 is added to spend by hand.
- **Cost estimate 3× low:** reasoning tokens at "minimal" effort.
- **Prompt-only rows:** the score dictionaries overwrote the truth fields "start" and "answer". The truth was
  recovered from the capture metadata by render key; the stage code is fixed.
- **Process:** a `pkill -f` pattern matched its own shell twice (no harm; patterns now bracketed).

## 11. Limitations

- **Sample:** 10 tables for the verbalizers; 1–4 samples per cell; one prompt template; dev only.
- **Judge:** one judge (WSB's). It agrees with the author's blind codes at κ 0.67 / 0.81 and is somewhat more
  lenient on conflicting samples. Wrong colours outside the candidate set keep no identity.
- **Different targets:** lens ranks among candidates vs verbalizer role claims; different layers (Oracle L40/L44,
  NLA L42).
- **No causal label:** whether z at `:` L40–44 drives the answer is still the open question. 002 shows which readers
  carry or claim z, not whether z is used.
- **Post hoc:** the lens tracking and shuffled baseline were added after the hit rates were seen; they are labelled as
  such.

## 12. Interpretation and next

- Under role-aware scoring the picture from 001 changes:
  - the **verbalizers** mention the intermediate often but rarely claim it in its role. Their outputs are dominated
    by invented or self-contradictory chains, with weak activation tracking. This is clearest for the Oracle.
  - the **token lenses**, read inside the candidate set rather than as top-10 lists, carry the intermediate
    specifically: positive start-swap tracking (a difference-in-differences of scores) in 47–48 of 48 tables at
    L40–44, and none for the start. Actual rank reversals are rarer: for J_NP cosine at `:` L40, 90 of 183 partner
    comparisons reverse, and 8 of 48 tables reverse for every partner. (Corrected after the audit of 2026-10-02.)
  - Scope: one model, brew, dev, descriptive.
- The disagreement "the NLA names z and the J lenses do not" (001 §13) does not survive this scoring. At the claim
  level the NLA rarely asserts z correctly. At the score level the J lenses carry z. The better-supported
  disagreement is between **readout format and content**: readers whose stated outputs miss z can still carry it, and
  readers whose outputs mention z mostly do not claim it.
- **Next (for the PI):** the label experiment (003). Is z at `:` L40–44 causally influential, given that a z-specific
  lens direction exists there? Reader-free mean-difference directions and necessity tests, with J kept out of the
  labels. 002's scorer is frozen for held-out use, but an extraction schema that keeps the identity of out-of-set
  colours is recommended before any test-split scoring (review point 1).
