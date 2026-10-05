# 007 Stage 1b — status (brew dev, `</think>`: lower observed patch leverage; 2026-10-05)

**What ran.** As Stage 1a at the `</think>` cell (001's whole-cell patch share 0.01, vs 0.48 at `:`): the Oracle L40/L44
on all 26 pieces × 231 renders, no-activation row, PCA positive controls re-scored with the `</think>` PCA basis (the other
piece types' positive control is cell-independent and comes from Stage 1a), prompt-only floor. The NLA is disabled (A1).
GPU time 9.7 min.

**Gates.** Oracle L40: G-fastpath 0.138, G-score1 0.118, G-score2 0.069,
G-repro max 0.141 / median 0.021 (D_z 6.012 vs 006 6.013), G-inject; Oracle L44: G-fastpath
0.100, G-score1 0.076, G-score2 0.080, G-repro max 0.142 (D_z 5.970 vs 5.966), G-inject — **all pass**.

**P1-P5 at `</think>` (secondary: outside the Holm family; categories use the same rules, Holm over 15 with the NLA's
5 as p = 1, for comparability).**

| reader | P1 | P2 | P3 | P4 | P5 | mapping |
|---|---|---|---|---|---|---|
| Oracle L40 | +7.37 [+6.54, +8.24] **+** | +6.87 [+6.14, +7.66] **+** | +3.22 [+2.84, +3.62] **+** | +1.63 [+0.73, +2.55] **+** | +5.05 [+4.30, +5.86] **+** | Raw: contradicted, Visible: contradicted, H_J: supported |
| Oracle L44 | +7.72 [+6.91, +8.57] **+** | +7.77 [+7.08, +8.53] **+** | +3.54 [+3.08, +4.06] **+** | -1.80 [-2.38, -1.21] **−** | +4.05 [+3.31, +4.81] **+** | Raw: contradicted, Visible: contradicted, H_J: supported |

**D_z (nats).** `</think>`, L40: whole 6.01 / J 7.77 / rest_J 0.72 / PCA 0.41 / MR 7.64 / GMR 6.62 / J_c 8.20 / random 0.90 / random∘c 3.15.
`:` (Stage 1a), L40: whole 3.22 / J 5.76 / rest_J 0.30 / PCA 0.44 / MR 2.93 / GMR 3.70 / J_c 3.82 / random 0.52 / random∘c 1.32.

**Observations.**
- The Oracle's dependence on the J subspace is the same, and if anything stronger, at the cell where the model's answer
  barely depends on the activation (patch share 0.01). Its piece dependence does not follow observed patch leverage
  (design §16): the J-subspace signal the Oracle reads is present whether or not the model uses that cell.
- At `</think>` mean removal raises the Oracle's z signal at L40 (P4 `+`), unlike at `:`.
- Prompt-only reader floor at `</think>`: D_z 0.70 (start 0.93, answer 0.51).
