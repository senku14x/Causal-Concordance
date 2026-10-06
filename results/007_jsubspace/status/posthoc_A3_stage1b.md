# 007 post hoc (A2 + A3) — NLA L42, Stage 1b (brew dev, `</think>`) — status (2026-10-06)

**Label: post hoc, dev only.** The first A2 rerun stopped at G-score1 (max 0.280 > 0.25). Report 25 traced that to the
NLA's bf16 batch-layout noise. Amendment A3 (71f26c7, adopted before this rerun) replaced the max with median ≤ 0.05 and
p99 ≤ 0.25 over 32 activations, for this rerun only. The scorer (006's exact path), the statistics and the other gates
are unchanged.

**Gates — all pass.** G-score1 (A3) median 0.0366, p99 0.1993, max 0.2549 (736 values); G-score2 0.081;
G-repro against 006's `</think>` NLA likelihoods max 0.052, median 0.0000, D_z 1.452 vs 1.452; G-inject incl.
scale check (exact). GPU time 125.1 min.

**P1-P5 for the NLA at `</think>` (post hoc full family, Holm/TOST over 15).**

| contrast | estimate [95% CI] | p | category | pieces failing the positive control |
|---|---|---|---|---|
| P1 | +0.29 [-0.13, +0.72] | 0.1650 | **unresolved** | PCA |
| P2 | +0.37 [+0.07, +0.66] | 0.0150 | **uninformative** | R0, R1, R2, R3, R4, R6, R7, R8, R9 |
| P3 | +1.01 [+0.74, +1.26] | 0.0001 | **uninformative** | rest_J |
| P4 | +2.13 [+1.47, +2.82] | 0.0001 | **+** | — |
| P5 | +1.06 [+0.45, +1.70] | 0.0004 | **+** | R0_c |

Mapping: Raw contradicted, Visible contradicted, H_J unresolved. Oracle (same family): L40 H_J supported, L44 H_J supported.

**NLA D_z at `</think>` (nats).** whole 1.45; J 0.76; rest of J 0.21; PCA-4 0.47; random
(mean of 10) 0.40; rest of random 1.21; mean-removed 3.58; generic-mean-removed 0.42; J∘c 2.74;
random∘c 1.68. J piece roles: D_start -0.02, D_answer 0.73, D_off -0.10.

**Observations (post hoc, dev only).**
- At `</think>` the NLA depends on J more than at `:`. Removing J leaves almost nothing (0.21 vs 1.21 for removing a random
  slice; P3 +1.01, but uninformative because rest of J fails the positive control with a null D), and J beats random
  after the same mean removal (P5 `+`).
- J alone still keeps only about half of the whole activation's signal (0.76 vs 1.45). Mean removal more than doubles it
  (3.58; P4 `+`), the visibility pattern, which the Oracle shows at `</think>` L40 too.
- Mixed evidence, no model supported. The Oracle's clean J pattern is still not mirrored by the NLA.
