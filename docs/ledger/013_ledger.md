# 013 research ledger (NLA hallucination test)

Branch `013-nla-hallucination` (off `012-reader-concordance`). Plan for the PI: `/workspace/claude_suggestions/013_plan_nla_hallucination_2026-10-07.md` (outside the repo, as the PI asked). Everything here is EXPLORATORY.

## 2026-10-07

**Scope (PI).** Only the post-005 NLA (nla-rl + iter_000400, L42); 001/002's older NLA is out. Multihop is primary; brew is kept only if a de-risk passes. Every readout is saved.

**De-risk** (rules fixed in the plan §4 before the run; script `scripts/013_nla_hallucination/derisk.py`; about 10 GPU-min).
- **Brew readouts:** all 231 of 007's `:` L42 renders, 2 samples each (WSB sampling), saved to `outputs/013_nla_hallucination/derisk/brew_readouts_nla_L42.jsonl`. Injection healthy on every chunk. Extraction 1.0.
- **Hand-read**, coded *before* any FVE was seen: `results/013_nla_hallucination/qualitative/brew_handread_codes.jsonl`, 32 seeded member readouts.
  - On-task (colour transformation): 30/32. The off-task two describe hex XOR (#5) and binary NOT (#6).
  - The hidden colour's word appears somewhere in 13/32, in any role. Colour words are everywhere in these readouts, so this is not evidence of reading z.
  - **Mostly contradicted: 29/32.** Almost every readout invents a multi-part worksheet ("Answer 1: Green", "Round 2 Answer:", "Part B") and colour chains that don't match the table. The one claim usually true is that the final "Answer:" needs a colour after two steps.
- **AR FVE** (`results/013_nla_hallucination/analysis/derisk.json`; first sample; B on each set's own activations; derangement control):

  | Set | FVE | 95% | Shuffled | Gap | B | Absolute MSE |
  |---|---|---|---|---|---|---|
  | brew (231) | **−556.2%** | [−568.1, −544.9] | −583.2% | 27.0 | 0.031 | 0.201 |
  | multihop (87, 012 clean readouts) | **49.9%** | [46.0, 53.6] | −79.0% | 129.0 | 0.285 | 0.143 |

**Verdicts by the pre-set rules:**
- **Brew: fails R1** (FVE far below 40%) and **fails R3** (29/32 mostly contradicted; the bar is ≤ 16). It passes R2 (30/32 on task). **Brew is dropped from the hallucination test,** as the PI expected.
- **Multihop: passes R1.** It is the test set.

**Reading:**
- The reconstructor's absolute error on brew (0.20) is close to multihop's (0.14). But brew's `:` activations hardly differ across renders (B 0.031 against 0.285), so that error is about 6.6× the brew-to-brew variance. The reconstruction is worse than predicting the brew mean, so it can't resolve brew renders, and no cleaning effect could show there.
- Multihop's 50% is below the FineWeb validation's 78% (off-distribution), but it leaves room to detect a cleaning effect.

**013 spec pre-registered** (`docs/design_specs/013_nla_hallucination.md`, commit 1c802b3) before any judge call. The pipeline is `scripts/013_nla_hallucination/hallucination.py`.

**Smoke** (20 explanations, 7 clean / 7 J / 6 random; judges $0.031; reconstructor 24 s). Results: `results/013_nla_hallucination/analysis/hallucination_smoke.json`.

*Gates:*
- G-repro exact: max |ΔMSE| 0.0 over 4 shared explanations.
- G1: A parses 20/20.
- **G2 (my hand check, before any Δ): A's contradicted-span precision 62/63 = 0.98.** About 5 claims were missed or put under "unsupported" (`results/013_nla_hallucination/qualitative/handcheck_smoke_codes.md`).
- G3: A–B sentence κ 0.74 (111 sentences).

*Numbers (smoke, n small; observation only):*
- Contradicted spans cover a mean 61% of each explanation's tokens.
- FVE: O 38.6%, **C1 −24.6%**, R1 4.5%, C2 −27.3%, R2 −7.8%.
- **Δ = FVE_R1 − FVE_C1 = +29.1 [9.1, 54.1]** (n 19; 1 explanation cleaned away). By the §6 rule this reads "H-signal".
- Per state: clean +9.5 [−22.2, 34.9]; J +46.1 [5.2, 87.3]; random +34.9 [−5.5, 76.8].
- K1 is computable on only 4/20, because the unflagged text is usually shorter than the target.
- J share: no shift (ΔJ_C1 − ΔJ_R1 −0.011 [−0.034, 0.005]).

*Why the smoke verdict can't be read as "confabulation carries signal" yet:*
1. **Many contradicted spans are misquotes of the prompt that still carry the bridge or answer** (hand check, kind ii). For example "the country that has the Tokyo tower [Japan]" after the Russia → Japan swap: in that explanation C1 keeps 59.8 while R1 falls to 23.4, and in others deleting such sentences removes the bridge. Deleting kind (ii) removes true content with the falsehood. Kind (i), invented context such as "Hint" headers and "Think Fast", is the pure confabulation.
2. **The deletions are large** (C1 keeps a median of about 40% of tokens), so many versions sit in a regime where the reconstruction is worse than the mean (FVE down to −158%). Means are dominated by a few explanations.
3. The token-matched random control removes clause fragments spread across sentences, while C1 removes whole sentences, so the two differ in structure as well as in content.

**Proposed amendment for the full run** (the PI's decision; written before any full-run judge call; post hoc, labelled):
- judge B tags each of A's contradicted spans as kind (i) invented context, (ii) a misquote or paraphrase of the prompt, (iii) a wrong entity or answer, or (iv) false world knowledge;
- report Δ separately for deleting kind (i) only and kind (ii) only, each against its own matched random control;
- add a sentence-level random control alongside the clause-level one.

The pre-registered primary (all contradicted spans) stays as recorded.

**STOP after the smoke** (the PI's rule).

**Full deletion run** (522 explanations; amendment A1; judges $0.82; reconstructor 26.6 GPU-min). Results: `results/013_nla_hallucination/analysis/hallucination_all.json`.

*Gates:*
- G-repro exact on 87/87.
- G1 0.964; G3 κ 0.72.
- G2 0.99 (89/90); G2k 0.83 (75/90), with the invented_context tag's precision 0.62 (`qualitative/handcheck_full_codes.md`).

*Counts:* 474 eligible; 1 had nothing to clean; 29 were cleaned away. Contradicted text covers a mean 64% of tokens.

*FVE means (eligible):* O 47.3; C1 −25.1; R1 3.5; C2 −27.5; R2 −2.3; K1 41.8 (n 87).

**Pre-registered primary:** Δ = FVE_R1 − FVE_C1 = **+28.6 [24.0, 33.4]**, n 474. The §6 label is "H-signal". The same holds in each state (clean +27.6, J +29.1, random +29.0) and with the sentence-level control (+26.0 [21.1, 31.3]).

**Split by kind** (A1, post hoc; the §6 bands applied descriptively):

| Kind | n | Share of tokens deleted | Δ clause control | Δ sentence control |
|---|---|---|---|---|
| **invented_context** | 354 | 0.25 | **−1.9 [−4.3, 0.3]** (unresolved) | **−0.1 [−1.8, 1.7]** (no difference beyond ±2) |
| **misquote** | 453 | 0.49 | **+35.8 [31.1, 40.8]** | **+36.6 [31.7, 41.6]** |

**Reading:**
- The primary "signal" comes from misquotes: paraphrases of the prompt, often distorted, that still carry the bridge or answer.
- Pure invented context (headers, quiz names, framings) carries no more signal than the same amount of random text. Within ±2 FVE points by the sentence control, the reconstructor treats it as padding.
- The invented_context tag's contamination (precision 0.62) is by misquotes, which would push its Δ *up*, so the padding reading isn't an artifact of that contamination.

*J share:* deleting contradicted text lowers the reconstruction's J share 0.010 [0.007, 0.014] more than random deletion does. That is tiny (gold 0.329, O 0.339).

**Rewrite smoke** (amendment A2; 20 explanations; judges $0.055; `results/013_nla_hallucination/analysis/rewrite_smoke.json`).
- Gates pass: B's contradicted-sentence rate is O 0.59, **F 0.01**, P 0.59. I hand-checked 20 F and 10 P: F factual 20/20; P keeps its claims 10/10. Words changed: F 60, P 45.
- **ΔFVE = FVE(F) − FVE(P) = +7.8 [3.7, 11.9]**, giving the reading "removing hallucinations improves reconstruction". P − O is −0.8 [−2.5, 0.9]: rewording alone does nothing.
- **ΔJ = +0.009 [0.002, 0.018]**, giving "more J" (O 0.351 → F 0.359; gold 0.338). Small.
- *Confound check (post hoc):* F added the correct answer in 4/20 (Cu → Au, Nahuatl, 'C', nine). Without them, ΔFVE is still +8.3 [3.7, 12.7] (n 16).
- *Still a caveat:* F also restores the prompt's exact wording when correcting misquotes. Part of the gain may be "a more accurate description of the input" rather than "less hallucination", and the two can't be separated here.
- In the J state, correcting a misquote back to the prompt removes swap-carrying text, e.g. Kremlin J: F − P = +3.0; red planet J: −19.2.

**STOP after the rewrite smoke** (the PI's rule). Next, on the PI's go: the full F/P run on 522 (about $1.5, about 10 GPU-min).

**Full rewrite run** (amendment A2; 522 explanations; judges $1.398; total 013 judge spend about $2.35). Results: `results/013_nla_hallucination/analysis/rewrite_all.json`. Hand-check file `qualitative/rewrite_handcheck_all.md`.

*Gates pass:* B's contradicted-sentence rate is O 0.60, **F 0.01**, P 0.59. F and P were produced for 522/522. Words changed: F 61, P 47.

*FVE means:* O 47.3, F 48.9, P 45.8. *J share means:* gold 0.329, O 0.339, F 0.356, P 0.336.

**Co-primary results (A2):**
- **ΔFVE = F − P = +3.1 [1.3, 4.9]**, giving the band reading "removing hallucinations improves reconstruction". Its parts:
  - F − O = +1.6 [−0.0, 3.3];
  - P − O = −1.4 [−1.9, −1.1]: paraphrasing itself costs about 1.4 points, so about half of F − P is the paraphrase penalty F avoids.
- **ΔJ = F − P = +0.020 [0.016, 0.024]**, giving "more J" (F − O = +0.017 [0.014, 0.021]).
  - Note: the original reconstruction's J share (0.339) is already above the gold activation's (0.329), and F moves it *further* above (0.356). "More J" is not "closer to the true activation in J".
- **Per state:**
  - clean: ΔFVE +5.5 [3.1, 7.9], ΔJ +0.019;
  - J swap: +3.0 [0.7, 5.1], +0.018;
  - random swap: +0.8 [−1.8, 3.4], +0.022.
  - There was no paired contrast between states, so no state-specific claim.
- **The smoke's +7.8 shrank to +3.1 on the full set** (n = 20 vs 522).

**Test-half restriction** (POST HOC, `analysis/testhalf.json`; 007's items 51–100, to line up with the numbers shared earlier):
- **Deletion:**
  - primary +30.0 [23.5, 36.4];
  - invented context with the sentence control: test +0.4 [−1.6, 2.7], dev −0.5 [−3.1, 2.3];
  - misquotes: test +34.5, dev +39.1.
- **Rewrite:** ΔFVE test +3.8 [0.8, 6.5], dev +2.4 [−0.1, 4.8]; ΔJ test +0.024, dev +0.015.

**STOP.**
