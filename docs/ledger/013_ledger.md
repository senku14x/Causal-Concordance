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
