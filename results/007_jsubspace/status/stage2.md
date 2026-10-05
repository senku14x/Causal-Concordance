# 007 Stage 2 — status (brew Part B, model side; 2026-10-05)

**What ran.** 004's 33 frozen pairs, the final `:` cell at L40/42/44, SEC regime, 004's write path (apply_write +
Fidelity G1) and answer-margin scoring; 34 subspace swaps per layer (whole cell, J, rest of J, PCA-k at k = 1/2/4/8/16
with the table-held-out basis, random 1,024 × 10, Δ-matched random × 10, random 512 × 3, random 2,560 × 3). GPU time
22.8 min (+ 4.1 min for 2b in the same session).

**Gates.** Smoke on 2 pairs: G0 (zero write) dm = 0.0 exactly at all three layers; G-repro (whole-cell S at L42 vs 004)
identical to every digit (0.33997, 0.34781); G1 fidelity enforced on every write — 0 rows quarantined; G-repro checked on
all 33 pairs (max |ΔS| 0.0000). Basis QR error ≤ 2.4e-6. **All pass.**

**Primary (recovery = S / S_full on the recovery subset; medians over pairs, 95% CI; Δ-share in parentheses).**

| layer | S_full | J | PCA-4 | Δ-matched random | random 1,024 | rest of J | R_J − R_PCA4 | R_J − R_Rdelta | reading | J residual vs isotonic curve (secondary) | interaction J |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L40 | 0.277 (001: 0.35) | 0.188 (0.272) | 0.149 (0.293) | 0.119 (0.276) | 0.081 (0.201) | 0.369 (0.728) | +0.005 [-0.066, +0.059] | +0.067 [+0.041, +0.110] | **mixed** | +0.070 [+0.021, +0.107] | -0.112 |
| L42 | 0.429 (001: 0.48) | 0.094 (0.238) | 0.109 (0.294) | 0.080 (0.243) | 0.069 (0.201) | 0.757 (0.762) | -0.022 [-0.086, +0.009] | +0.020 [-0.005, +0.030] | **tracks delta-share** | +0.008 [-0.017, +0.024] | -0.067 |
| L44 | 0.491 (001: 0.5) | 0.210 (0.276) | 0.174 (0.295) | 0.107 (0.278) | 0.069 (0.200) | 0.392 (0.724) | +0.051 [-0.064, +0.130] | +0.091 [+0.024, +0.163] | **mixed** | +0.099 [+0.002, +0.138] | -0.163 |

**Observations (demonstrated influence on this answer margin; J geometry, never a label):**
- At L42 J's recovery tracks its content share, matching PCA-4 and the Δ-matched random subspaces within ±0.10:
  004's conclusion extends to a fixed basis that was not fitted on the data.
- At L40 and L44 J recovers modestly more than random subspaces with the same Δ-share (+0.07 / +0.09; the isotonic
  residual agrees, +0.07 / +0.10) but not more than PCA-4, and below the pre-registered 0.10 margin: "mixed", not
  "J beyond content share".
- J and its complement are far from additive (I = −0.07 to −0.16), as in 004.
- Whole-cell swaps are outside-spelling flagged in 9–33% of rows (001's damage note); subspace swaps rarely.
- S_full at L40 is 0.28 here vs 001's median 0.35 (different pair sets: 004's 33 vs 001's 82).
