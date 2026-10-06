# 009 — The brew-code bank, reader-free de-risk: the gate passes at depth 1; the output-facing check fails

**Run info.**
- **Design:**
  - `docs/design_specs/008_009_trackb_brewcode.md` Part 2, with the execution notes in §E-009 (early stop, collision
    rule, gate decode check);
  - reader rules in `009b_reader_phase.md`.
- **Code:** `scripts/009_brewcode/` (`bank.py`, `gate.py`, `derisk.py`).
- **Runs:** gate `g2` (depth 2, stopped early), gate `g1` (depth 1), outfacing `o1`, captures and probes `c1`, profile
  `p1`.
- **Results:** `results/009_brewcode/analysis/{gate_g1, gate_g2, outfacing, probes}.json`.
- **Label: EXPLORATORY.** Dev tables only. The 150 test tables at each depth are generated, hashed and untouched.
- Every number below is from the JSON, rounded half-up.

## 1. Summary

- **The family's premise fails.** Making the answer a two-digit code does not take the hidden colour out of the model's
  output distribution. It only makes it improbable.
  - Under the final-code question, at the answer slot:
    - **the hidden colour is the top colour in 98% of renders;**
    - it ranks a median **11th of all 248,320 tokens** (IQR 8th–13th);
    - the model's own D for it is **+11.9 nats** [11.7, 12.1], positive in 240 of 240 tables. Brew's 006b value was 3.8.
  - Yet the 12 colours together get only about **0.05%** of the probability (median 5.1e-4; max 2.8e-3).
  - The model says the code; its distribution still points at the colour.
- **By the pre-set rule (design §2.5.1), the output-facing check fails on all five criteria, so readers do not run on
  this bank.**
- **The gate works at depth 1, not at depth 2.**
  - Depth 2 (two stirs + a code lookup, no chain of thought): **0 of 77** pairs passed after 30 tables; final-code renders
    passed 11%. The early-stop rule moved to depth 1.
  - Depth 1: **771 of 1,200** candidate pairs pass, across 241 of 300 tables. Final-code renders pass 80%, start-code
    99.5%.
- **The model computes the hidden colour even when the question doesn't need it.**
  - A reader-free probe decodes it under both questions at 110 of 112 (cell × layer) sites.
  - When the answer needs it, decoding is near-perfect: balanced accuracy 0.86–1.00 at the answer-side cells, L40–60.
  - When it doesn't, decoding is weaker but well above chance (1/12 ≈ 0.08): 0.45–0.76 at the answer-side cells, L40–48.
  - Under the start-code question it then fades at L52–60 (0.22–0.28 at `:`, where it no longer passes).
- **The model's own output already follows the question.**
  - At the answer slot the hidden colour's D is +11.9 under final-code but only +2.8 under start-code. Under start-code,
    the start colour dominates instead (D +14.1; the top colour in 100% of renders).
  - So a reader whose hidden-colour signal "follows the task" could be following output-facing content. That is exactly
    what this family was meant to rule out.

## 2. The gate

| Run | Depth | Tables | Final-code renders pass | Start-code renders pass | Pairs in / candidates | Decision |
|---|---|---|---|---|---|---|
| g2 | 2 | 30 (stopped) | 11% (greedy 58%; mean 4.5/10 correct) | 98% | 0 / 77 | early stop → depth 1 |
| g1 | 1 | 300 | 80% (greedy 99.6%; 9.1/10) | 99.5% | **771 / 1,200** (241 tables) | ok |

- **Depth 2 is beyond Qwen3.6-27B without chain of thought in this format:** three lookups, two stirs plus the code.
- **The gate is conservative.**
  - In about 4% of samples, the EOS suppression makes the model run digits on after a correct code ("8787…"), and the
    strict parser rejects them (design §E-009).
  - So no wrong answer passes, but some valid pairs are lost.
  - A pair needs both questions to pass, so this selection is the same for both arms of the task contrast.
- **Forced continuations name colours.** In the gate's 16-token greedy decodes (EOS suppressed), 53% of final-code
  renders name a colour after the code, and 39% of start-code renders do.

## 3. The output-facing check (the model's own output; SEC regime; 241 tables × up to 4 partners)

### 3.1 D statistic

Mean over tables [95% table-cluster CI], collision-free pairs:

| Question | Slot | D, hidden colour | D, start colour |
|---|---|---|---|
| final-code | assistant newline | +11.6 [11.4, 11.8] | +3.7 [3.5, 3.9] |
| final-code | `</think>` | +12.4 [12.2, 12.7] | +5.5 [5.3, 5.8] |
| final-code | **`:` (answer slot)** | **+11.9 [11.7, 12.1]** | +6.3 [6.0, 6.5] |
| final-code | after the forced answer | +4.1 [3.9, 4.2] | +3.5 [3.4, 3.7] |
| start-code | assistant newline | +2.0 [1.9, 2.2] | +11.8 [11.6, 12.0] |
| start-code | `</think>` | +3.5 [3.3, 3.7] | +17.6 [17.4, 17.8] |
| start-code | **`:` (answer slot)** | **+2.8 [2.6, 3.0]** | **+14.1 [13.9, 14.3]** |
| start-code | after the forced answer | +0.6 [0.5, 0.6] | +6.5 [6.3, 6.6] |

- D = [LL_m(t_m) − LL_m(t_p)] − [LL_p(t_m) − LL_p(t_p)].
- At depth 1 the hidden colour is the final colour, so z = f.
- The all-pairs values are within 0.6 nats of these (e.g. 11.3 at `:`).
- The hidden colour's D is positive in 240 of 240 tables at every final-code slot.

### 3.2 How much the colours actually get (member renders; total probability on the 12 colours)

| Question | Slot | Median | 90th pct | Max |
|---|---|---|---|---|
| final-code | answer slot | 5.1e-4 | 1.5e-3 | 2.8e-3 |
| final-code | after the forced answer | 2.1e-8 | 4.9e-8 | 1.6e-7 |
| start-code | answer slot | 1.6e-3 | 3.0e-3 | 5.5e-3 |
| start-code | after the forced answer | 2.9e-8 | 9.0e-8 | 2.8e-7 |

### 3.3 Rank of the hidden colour (" z") among all 248,320 tokens at the answer slot

| Question | 10th pct | 25th | Median | 75th | 90th | Top colour of the 12 is… |
|---|---|---|---|---|---|---|
| final-code | 7 | 8 | **11** | 13 | 19 | the hidden colour in **98%** of renders |
| start-code | 28 | 47 | 82 | 138 | 206 | the start colour in **100%** |

### 3.4 Forced continuations (24 greedy tokens after the teacher-forced code, EOS suppressed)

- **Final-code:** **50%** spell the hidden colour.
- **Start-code:** 0% do.
- Most continuations either restate the problem ("The potion starts out gray. According to the rules: 'A gray potion
  turns teal…'") or open an explanation ("**Step-by-step Deduction:**…"). A few are digit run-ons.
- This is the model explaining itself once it is not allowed to stop. It is not part of its answer.

### 3.5 Verdict

| Criterion (design §2.5.1) | Bar | Value | Pass |
|---|---|---|---|
| D, hidden colour, at the answer slot | CI inside ±0.5 | [11.7, 12.1] | no |
| D, final colour | CI inside ±0.5 | [11.7, 12.1] (same colour at depth 1) | no |
| median rank of z | > 100 | 11 | no |
| median rank of f | > 100 | 11 | no |
| continuations spelling z or f | ≤ 20% | 50% | no |

## 4. Is the hidden colour present? (reader-free probes on SEC captures)

12-way logistic probe on PCA-256, 5 table folds; "decodes" = BA CI lower bound > 0.233, control ≈ chance.

| Cell | Question | L40 | L42 | L44 | L48 | L52 | L56 | L60 |
|---|---|---|---|---|---|---|---|---|
| last question token | final | 0.92 | 0.91 | 0.94 | 0.94 | 0.93 | 0.98 | 0.97 |
| last question token | start | 0.66 | 0.72 | 0.76 | 0.78 | 0.74 | 0.61 | 0.56 |
| assistant newline | final | 0.98 | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |
| assistant newline | start | 0.76 | 0.76 | 0.75 | 0.68 | 0.56 | 0.50 | 0.40 |
| `</think>` | final | 0.86 | 0.91 | 0.94 | 0.98 | 0.99 | 1.00 | 1.00 |
| `</think>` | start | 0.45 | 0.53 | 0.61 | 0.55 | 0.33 | 0.30 | 0.26 |
| `:` | final | 0.98 | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |
| `:` | start | 0.51 | 0.51 | 0.57 | 0.59 | 0.28 | 0.23 | 0.22 |

- Permuted-label controls are 0.06–0.11 everywhere.
- The start colour is decoded at 0.99–1.00 under the start-code question.
- **Reading:**
  - the model computes the hidden colour whether or not the answer needs it;
  - when the answer doesn't need it, the colour is weaker at L40–48 and fades by L52–60 at the answer-side cells;
  - the "present under both questions" condition for a reader comparison holds at 110 of 112 sites.

## 5. Single-cell leverage (patch profile)

**Method.**
- The whole-cell patch share S: the partner's cell is patched into the member at one cell and one layer, scored on the
  code-answer margin (full sequence, every token), SEC regime.
- 48 gated pairs per question, one per table first.
- The table shows median S [95% table-cluster CI]. Gates G0 and same-source are exact (Δm = 0) on the first 2 pairs per
  question.
- The answer gap G has median 14.5 nats (final-code) and 18.0 (start-code).

| Question | Cell | L40 | L42 | L44 | L48 | L52 | L56 | L60 |
|---|---|---|---|---|---|---|---|---|
| final-code | assistant newline | 0.16 | 0.19 | **0.30** | 0.21 | 0.14 | 0.13 | 0.09 |
| final-code | `</think>` | **0.00** | **0.00** | **0.01** | 0.12 | 0.12 | 0.11 | 0.04 |
| final-code | `:` | 0.12 | 0.19 | 0.35 | 0.46 | **0.48** | 0.42 | 0.45 |
| start-code | assistant newline | 0.04 | 0.02 | 0.02 | 0.01 | 0.00 | 0.00 | 0.00 |
| start-code | `</think>` | −0.01 | −0.01 | 0.01 | 0.18 | 0.13 | 0.11 | 0.06 |
| start-code | `:` | 0.06 | 0.06 | 0.08 | 0.27 | 0.38 | 0.33 | 0.34 |

**Reading.**
- **The code bank reproduces brew's structure.** Under final-code at L40–44, the `</think>` cell has no single-cell
  leverage (S 0.00–0.01), while the hidden colour is decodable there (probe BA 0.86–0.94). Brew's `</think>` showed the
  same (S 0.01 at L42).
- `:` and the assistant newline carry a real share (0.12–0.35 at L40–44), rising at `:` to about 0.48 by L52.
- **Under start-code, the answer-side cells carry little until L48** (`:` 0.06–0.08 at L40–44). The start code appears
  to be read out late.
- So at the readers' layers (L40–44), the code bank offers:
  - a cell with the hidden colour **present and with leverage** (`:`, final-code);
  - a cell with it **present and without single-cell leverage** (`</think>`, final-code);
  - cells where it is **present but not needed by the question** (all three cells, start-code).
- These are the contrasts a reader analysis needs. What the bank cannot do is make the colour absent from the
  output-facing distribution (§3).

## 6. What this means

**Observed:**
- The answer-slot distribution of Qwen3.6-27B carries the hidden colour sharply in its tail when the answer is that
  colour's code.
- It does so with almost no probability mass.
- **Changing the answer's format did not remove the colour from the output-facing distribution.**
- This is consistent with the model computing "the code of red" through a state in which "red" is readable from the
  final layer. On brew the same state shows up as z being the runner-up answer.

**Interpretation (not established):**
- An intermediate that the model represents in a nameable form may always leave a trace like this at the answer slot.
- If so, no family with a nameable hidden step can make "the reader reads output-facing content" impossible.
- At best it can make that content improbable, as here.
- Report 29 qualifies how much this matters for the readers. On brew, the mid-layer readers' per-pair z signal tracked
  what the J lens reads (r 0.63–0.80) much more than the model's own output (r 0.2–0.3). Only the late Oracle (L56/L60)
  followed the output.

**Consequences for the plan:**
- **Readers do not run on this bank under the committed rule.** The rule asked for the colour to be absent from the
  output-facing distribution by D and rank. By that standard it is present, and more sharply than on brew.
- **The task contrast is confounded at the output too.** The model's own D for the hidden colour moves from +11.9
  (final-code) to +2.8 (start-code). A reader that "follows the task" could be following that.
- **The bank is still useful without readers.**
  - It has a large, well-gated set of pairs (771 at depth 1, 241 tables, folds by table, an untouched test split).
  - The hidden colour is present but unneeded under the start-code question (BA 0.45–0.76 at L40–48), a clean
    "computed but not needed" condition for model-side work.
  - Track B's swaps and the 011 Oracle-space swaps can run here, where the colour is never the answer.

## 7. Options (the PI's call)

1. **Keep the rule.**
   - No readers on this bank. This report is the finding.
   - Use the bank for model-side questions: Track B and 011 swaps where the colour is never the answer; the leverage of
     the hidden colour under each question.
2. **Run the readers as a labelled post hoc analysis**, with the model's own per-pair D as a covariate (006b's logic).
   - Does a reader's task contrast exceed what the model's output-facing contrast predicts?
   - About 2 GPU-h. It cannot be confirmatory, because the rule failed.
3. **Redesign the family.** Make the hidden step something the model cannot name in a single token, or make the answer
   slot's distribution independent of it (for example, an answer that must be emitted before the hidden step is
   computed). Uncertain: any nameable intermediate may leave the same trace.

## 8. Execution notes and deviations (all recorded before outcomes)

- **Depth-2 early stop** after 30 tables (Wilson upper bound of the pair share 0.048 < 0.25; §E-009).
- **Collision-free primary D:** 13% of depth-1 pairs have one member's hidden colour equal to the other's start colour.
- **Gate decode check:** the EOS-suppression digit run-ons are rejected by the strict parser, so the gate is
  conservative.
- **Probes** use logistic regression on train-fold PCA-256 (the fork's choice, documented in the code). The design said
  "logistic".
