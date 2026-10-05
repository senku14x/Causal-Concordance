# 007 Stage 2b — status (the paper's §3.3 test as an adapted selectivity test; 2026-10-05)

**What ran.** Class-mean z directions at `:` L40/42/44 (dev SEC captures, leave-the-pair's-table-out; reader-free),
split by nonnegative gradient pursuit (k = 25, NNLS refit, folded gain atoms) with J_NP and with J_CB; the swap
"remove the member's z component, add the partner's" along the full direction, its J part and its non-J part, every
edit rescaled to the full direction's magnitude, plus 4 norm-matched random directions; all 33 pairs (A1); 004's gates
and damage readouts. GPU time 4.1 min.

**Gates / checks.** G1 on every write, 0 rows quarantined. Smoke check on 2 pairs (reported only, A1): median
full-direction S 0.090 (criterion ≥ 0.02). Stage 0: the J part holds a median 4–7% of the direction's energy; the colour's
own token atom is in the k = 25 support for 0–3 of 12 colours.

| layer | lens | full S | J-part S | non-J S | random S | J − non-J [95% CI] | F_target J / non-J | damage J / non-J | reading |
|---|---|---|---|---|---|---|---|---|---|
| L40 | J_NP | 0.0426 | 0.0143 | 0.0416 | 0.0014 | -0.0234 [-0.0295, -0.0116] | 0.060 / 0.058 | 0.00 / 0.00 | not demonstrated under this adaptation |
| L40 | J_CB | 0.0426 | 0.0194 | 0.0373 | 0.0014 | -0.0231 [-0.0312, -0.0063] | 0.041 / 0.057 | 0.00 / 0.00 | not demonstrated under this adaptation |
| L42 | J_NP | 0.0552 | 0.0170 | 0.0559 | 0.0013 | -0.0282 [-0.0552, -0.0147] | 0.044 / 0.075 | 0.00 / 0.00 | not demonstrated under this adaptation |
| L42 | J_CB | 0.0552 | 0.0139 | 0.0559 | 0.0013 | -0.0345 [-0.0438, -0.0158] | 0.043 / 0.068 | 0.00 / 0.00 | not demonstrated under this adaptation |
| L44 | J_NP | 0.0630 | 0.0125 | 0.0615 | -0.0004 | -0.0421 [-0.0722, -0.0279] | 0.030 / 0.094 | 0.00 / 0.00 | not demonstrated under this adaptation |
| L44 | J_CB | 0.0630 | 0.0176 | 0.0634 | -0.0004 | -0.0467 [-0.0610, -0.0279] | 0.053 / 0.098 | 0.00 / 0.00 | not demonstrated under this adaptation |

**Observations.**
- At matched magnitude the J part moves the answer margin less than the non-J part, at every layer with both lenses
  (CIs exclude 0). It is above the norm-matched random directions, but far below half of the full direction's effect.
- No swap flips the top-1 answer (all effects small: full-direction S 0.04–0.06).
- Pre-registered reading (A1): **privilege not demonstrated under this adaptation**, with J and both lenses. The
  non-J part, not the J part, carries most of the class-mean direction's effect here.
- Stage 2 (fixed projection) and Stage 2b (adaptive pursuit) need not agree (A1). Here both say the J subspace gets
  no special model-side purchase at `:` beyond its content.
