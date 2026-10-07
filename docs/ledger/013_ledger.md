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
