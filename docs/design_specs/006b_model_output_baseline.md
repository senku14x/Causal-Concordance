# 006b — Is the readers' z tracking just the model's own output distribution? (design, pre-registered before the run)

**Status.**
- The follow-up asked for by the PI-forwarded 006 review (2026-10-05). The PI's go: "one final run for me".
- Branch `005-amended-full-run`.
- Exploratory in the sense that 006's outcomes are known. The rules and priors below are fixed before 006b's own numbers exist.
- `[PI]` priors are left blank.

## Question

- **What 006 found:** at the brew `:` cell the readers' colour likelihoods follow the cell's own intermediate colour z. At the `</think>` cell, where 001's patch share is 0.01, the Oracle's follows it twice as strongly.
- **The boring alternative:** the model's *own* next-token distribution at the cell already ranks z high, for example as the runner-up answer. A reader that reads off the cell's output-facing content would then track z without any held intermediate.
- **006b measures** the model's final-layer log-probs for the 12 colours at the three cells, with 006's D statistic. Then it asks whether the readers' D_z goes beyond the model's.

## Material and measurement

- **Renders:** the same 231 member and partner renders.
  - Rebuilt with `brew_material.brew_pairs` and matched to the 001 capture keys (`ids_key`). All 231 must match.
  - Cells come from the render's regions: assistant newline (158), `</think>` newline (162), `:` (164). They are asserted on every render.
- **Model output:** the stock forward (bf16 matmuls), float32 logits from the final-norm output, in the subject model's text stack.
  - Per render and cell, the sequence is truncated after the cell. Each spelling (" c", " C") is appended and scored over every token (`readers.cont_batch` / `seq_logprob_from_hidden`, unit-tested in 006).
  - The colour score LL_model(c) is the log-sum-exp over the two spellings.
- **Statistics:**
  - **D_t** for t ∈ {z, start, answer}, per table, exactly as in 006. For start and answer, collision-free pairs only (the review's census).
  - **Rank:** the rank of z among the 12 colours in the model's distribution at `:` on member renders, and the share of tables where z ranks second, behind the answer ("runner-up").
  - **Reader beyond model** (the key test), per primary reader-layer (NLA L42, Oracle L40, Oracle L44; 006 prefix p0; `:` cell):
    - **primary:** on the pairs where the model's own D_z ≤ 0 (its output does not move toward the cell's z), the mean reader D_z, with a 95% table-cluster bootstrap CI (10,000 draws, seed 20261004). Fewer than 15 such pairs makes it "not testable";
    - **secondary:** the OLS intercept of reader D_z on model D_z across the 183 pairs, plus r. The intercept extrapolates when model D_z sits far from 0; the synthetic check before the run showed this, which is why it is secondary.

## Gate

- **G-score-m:** the scoring path checked against the full model's own logits at `:`, over the single-token spellings of 4 renders.
- Pass: max |difference| ≤ 0.25 nats and median ≤ 0.05 (006 v1.1's bf16 floor).

## Rules and priors

| ID | Rule | Agent prior | `[PI]` |
|---|---|---|---|
| B1 | The model's own D_z at `:` has its CI above 0 (the output distribution tracks z) | 0.75 | |
| B2 | Reader beyond model at `:`: on pairs with model D_z ≤ 0, the reader's mean D_z CI lies above 0 (≥ 15 pairs, else not testable). Per reader-layer, Holm over the three (one-sided bootstrap p, α 0.025) | NLA 0.45 · Oracle 0.60 | |
| B3 | At `</think>` (patch share 0.01) the model's own D_z CI includes 0 or lies below it, while the readers' does not (006) | 0.50 | |
| B4 | z is the runner-up (rank 2) in the model's distribution at `:` in ≥ 50% of tables | 0.35 | |

**Reading:**
- **B2 holds:** the readers carry z information beyond what the model's own output distribution at that cell carries.
- **B2 fails while B1 holds:** the readers' z tracking is explained by the model's output distribution (the runner-up-answer reading).
- **B3 holds:** at `</think>` the readers track z while the model's own output there does not.

## Cost

About 231 renders × 3 cells × 24 spellings ≈ 16,600 scored continuations of about 165 tokens. That is roughly 10 GPU minutes. No writes (G0 not applicable).

**Code:**
- `scripts/006_reader_likelihood/model_output.py` (GPU);
- `scripts/006_reader_likelihood/model_output_analysis.py`, with output in `results/006_reader_likelihood/analysis/model_output.json`.
