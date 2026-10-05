# 007 Stage 5 — status (chain; exploratory; 2026-10-05)

**What ran.** 001's gated dev chain start pairs restricted to single-digit intermediates and answers for both members:
**25 pairs over 7 items** (≥ 10, so the stage ran). The final cell at L56 (001's whole-cell S 0.66 there), the
Oracle L56 only (on-contract), the 21 v1 pieces with the J_raw_L56 basis (PCA-k within step-sequence group, k = 3,
Δ-matched to J's 0.19), digits 0-9 as candidates (" d" and "d"), no-activation row, positive control
on "Tell me about the number d". GPU time 2.8 min.

**Gates.** G-score1 max 0.074, G-fastpath max 0.160, G-score2 0.110 — pass.
Positive control (whole, digits): LOO-prior ID 1.00, pairwise double differences positive 45/45.

**D on the first intermediate (nats; per-item means, mean over 7 items, 95% item bootstrap CI).**
whole 12.65 [9.12, 16.92]; J 1.88 [0.78, 3.08];
rest of J 12.76; PCA-3 12.25; LL_gain 1.60; MR 8.04;
random (mean of 10) 5.30; rest of random (3) 10.88.

**Observation (hypothesis-generating only; 7 items).** At the chain's final cell at L56 the Oracle's digit signal for the
first intermediate is carried *outside* the J subspace: the J piece keeps little of it and removing J leaves it intact,
the opposite of brew at `:` L40. One possible reading is the paper's own caveat that small integers may not align with
the J-lens vectors of digit tokens; another is a layer difference (L56 vs L40). Not tested here.
