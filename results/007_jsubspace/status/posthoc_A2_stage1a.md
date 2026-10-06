# 007 post hoc (A2) — NLA L42, Stage 1a (brew dev, `:`) — status (2026-10-06)

**Label: post hoc** (amendment A2, adopted before this rerun after the PI's "rerun properly"). The pre-registered record
(Oracle only, the NLA's contrasts as p = 1) is unchanged in `results/007_jsubspace/status/stage1a.md`.

**What ran.** The NLA (`nla-rl` iter_000400) on 006's exact scorer (24 rows per activation, `acts_per_batch` 2, original
log-prob), all 26 pieces × 231 renders, no-activation row, the positive control (35 piece types), qualitative samples.
No subject forward (saved pieces and u_gen). GPU time 134.9 min.

**Gates — all pass.** G-score1 max 0.2465 / median 0.0416 (006: 0.2465 / 0.041); G-score2 0.031;
**G-repro max 0.152 / median 0.0000, D_z 1.862 vs 006 1.862** (the gate the 2-row scorer failed);
G-inject incl. scale check (exact) and no-activation vs direct (exact).

**P1-P5 for the NLA (post hoc full family: Holm and TOST over all 15 contrasts with real p-values).**

| contrast | estimate [95% CI] | p (two-sided) | category | pieces failing the positive control |
|---|---|---|---|---|
| P1 | +0.96 [+0.42, +1.49] | 0.0004 | **uninformative** | PCA: null_uninformative |
| P2 | +0.69 [+0.31, +1.09] | 0.0002 | **uninformative** | R0: stands_flagged, R1: null_uninformative, R2: null_uninformative, R3: null_uninformative, R4: null_uninformative, R6: stands_flagged, R7: stands_flagged, R8: stands_flagged, R9: null_uninformative |
| P3 | +0.17 [-0.12, +0.47] | 0.2508 | **unresolved** | rest_J: stands_flagged |
| P4 | +0.36 [-0.25, +0.94] | 0.2428 | **unresolved** | all informative |
| P5 | +0.52 [-0.11, +1.17] | 0.1068 | **unresolved** | R0_c: stands_flagged |

Mapping: Raw unresolved, Visible unresolved, H_J unresolved. The Oracle's categories are unchanged in
the full family (L40: H_J supported; L44: unresolved).

**NLA D_z by piece (nats).** whole 1.86; J 0.95; rest of J 0.63; PCA-4 -0.02; random (mean of 10)
0.26; rest of random (3) 0.80; mean-removed 2.22; generic-mean-removed 0.89; J∘c 1.60; random∘c (3) 1.07;
LL_gain 0.60; rest of LL 1.92.

**Oracle–NLA per-pair agreement (pair-level D_z, Pearson r; §11.1).** NLA vs Oracle L40: whole 0.24, J 0.31,
rest of J 0.08, mean-removed 0.44; vs Oracle L44: whole 0.30, J 0.37,
rest of J 0.24, mean-removed 0.33.

**Observations (post hoc; dev only so far).**
- The NLA's z signal is not concentrated in J: J keeps about half of the whole activation's D_z, removing J costs no more
  than removing a random 1,024-dim slice (P3 ≈ 0), and mean removal slightly raises it. J is enriched over random
  slices (+0.69 [+0.31, +1.09]), but most random pieces fail the NLA's positive control, so P2 is uninformative.
- **J dependence is reader-specific:** strong for the Oracle, weak for the NLA, at the same cell and the same pieces.
- The readers' per-pair agreement is low throughout (r 0.08-0.44), highest on the mean-removed piece.
