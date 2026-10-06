# 009b — reader-phase decision rules (written before any reader score on the 009 bank)

**Status.**
- Exploratory.
- Committed with `scripts/009_brewcode/read009.py` and `read009_analysis.py`, before any reader has scored a 009
  activation.
- Readers run only after design §2.6's go/no-go passes on dev:
  - the gate yields ≥ 200 pairs;
  - the output-facing check passes;
  - at least one emit cell × layer in L40/42/44 has z decodable under both tasks.
- Wording:
  - "decodable through the reader's likelihood";
  - a cell has "demonstrated / no single-cell leverage";
  - never "the model uses".

## What is measured

- **Readers and layers:**
  - the Oracle `olens_s3d_rl600` at L40 and L44 (on-contract), prefix "-";
  - the NLA (`nla-rl` iter_000400) at L42 (its training layer) and L44 (off-training), prefix
    "<explanation>\nThe text is about";
  - comparators:
    - the token lenses (J_NP, J_CB and R_CB cosine; the logit lens) at L40/42/44 on the whole activation;
    - the prompt-only floor per cell (stock model, WSB template);
    - the model's own next-token output at each cell (the 009 output-facing run).
- **Activations:**
  - SEC captures of the gated dev renders, both tasks, at the 3 emit cells;
  - 007's pieces, built with 007's bases: whole, J (the raw-W_U J subspace at the reader's layer), rest_J,
    R0–R2 (random 1,024-dim), rest_R0.
- **Statistic:** 006's D for z, f and start (start and f on collision-free pairs) as a mean over tables, with:
  - a 95% table-cluster bootstrap (10,000 draws, seed 20261004) and sign counts;
  - a 200-derangement null on the whole piece.

## Gates (per reader, before any outcome is read)

- **G-score1:** the fast 2-row scorer vs the reader's own 24-row `score_seq` on 32 activations; median |Δ| ≤ 0.05 and
  p99 ≤ 0.25 nats. These are per-reader noise-aware rules (report 25), never a fixed maximum.
- **G-score2:** float32 vs the reader's own logits, max ≤ 1.0 nat.
- **Positive control** per reader × layer × piece, on 006's natural "Tell me about {colour}" activations through the
  same operators: LOO-prior ID ≥ 0.5 **and** ≥ 50/66 positive pairwise double differences.
  - A null on a failing piece is "uninformative".
  - The NLA at L44 is off its training layer, so its positive control is the test of whether it reads L44 at all.

## Primary family and decision rule

- **Contrast.** At each emit cell × reader layer where the z probe decodes under **both** tasks (`probes.json`
  `z_decodes_both_tasks`), on the whole activation:
  - Δ = D_z(final-code) − D_z(start-code), paired by table.
- **Family:** every eligible (reader, layer, cell). Holm over the family on two-sided bootstrap p; δ = 0.5 nats.

| Category | Rule |
|---|---|
| **follows the task** | Δ 95% CI lower bound > 0 **and** Δ ≥ 0.5 (plus Holm p < 0.05 for the family claim) |
| **task-blind** | Δ 90% CI inside (−0.5, +0.5) **and** D_z(start) 95% CI > 0: z is decodable through the reader under both tasks, equally |
| **absent** | D_z CIs include 0 under both tasks |
| **higher under start** | Δ 95% CI upper bound < 0 and Δ ≤ −0.5 |
| **unresolved** | otherwise |

**What each category means here.**
- Under the final-code task the answer depends on z. Under the start-code task it does not; the start code is read
  directly.
- With z decodable by a reader-free probe under both tasks, and the model's own next-token output putting negligible
  mass on colours (the output-facing check), a **task-blind** reader reports content that is present whether or not the
  answer depends on it.
- A reader that **follows the task** is the first evidence that a reader's z signal tracks where the answer depends on
  z, with the output-facing route closed by design.
- The comparators qualify each reading:
  - the model's own output contrast should be ≈ 0;
  - the prompt-only floor's contrast shows what the text alone gives away;
  - the probe's decodability under each task is reported beside it.

## Secondary (descriptive)

- **J dependence on the new family, per reader:** D(rest_R0) − D(rest_J) (P3-like) and D(J) − mean D(R0–R2).
  Does the Oracle's 007 J dependence replicate where z is not an answer candidate?
- **D_f and D_start** per task: what the reader tracks besides z.
- **The NLA at L42 vs L44, and the Oracle at L40 vs L44:** layer effects.
- **Qualitative samples** at every emit cell (`</think>` included), for whole, J and rest_J, on 8 fixed-seed tables:
  member + first partner, k = 2.

## Not claimed

- Anything beyond Qwen3.6-27B, this bank and dev tables.
- That "task-blind" means the reader is wrong. It reports present content; whether that content is used is a property of
  the model, not of the reader.
- Content-level use: the task contrast is a design-level manipulation of what the answer needs, checked by gates and
  probes, not a per-cell content label.
