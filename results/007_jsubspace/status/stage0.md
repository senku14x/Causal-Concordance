# 007 Stage 0 — status (2026-10-05)

**What ran (CPU):** `pieces.py` (float32 bases with QR re-orthonormalisation, every Stage 1a/1b
piece of the 48 dev tables, positive-control pieces incl. the frozen brew operators (A1), PCA bases, role table with
006's collision census and the off-trajectory pair), `stage0_2b.py` (class-mean z directions and their J / non-J split, k = 25,
J_NP and J_CB), the J_raw_L56 basis, the chain count (25 single-digit gated dev pairs over 7 items: Stage 5 runs), the
template-lens download and sha256 pins (all four match), the 007 analysis unit tests (7/7 pass).

**Gates.**
- G-bases: worst |QᵀQ − I| after QR 2.4e-6 (≤ 1e-5) — pass.
- G-recon: worst relative reconstruction error 3.5e-8, worst |⟨MR, u⟩|/‖h‖ 5.6e-10 (≤ 1e-6) — pass.
- G-geom: **first execution failed** (one comparison: rest-of-random visible ratio at `:` L44, |Δ| 0.0052 vs 0.005).
  Cause: the gate compared the median of our 3 complement draws with report 22's 10-draw median (gate-code error, not
  construction). Corrected to a per-draw comparison against report 22's own construction of the same seeds, threshold
  unchanged; rerun worst |Δ| 5e-7 — pass. Recorded in amendment A1.

**Headline numbers (geometry only; no reader).**

P5 pieces (`:` and `</think>`): Δ-share / visible ratio; the Visible model's P5 prediction per A1 (≈ if the visible ratios
differ by ≤ 15%).

| cell | J∘centred | random∘centred (0-2) | visible-ratio rel. diff | Visible P5 |
|---|---|---|---|---|
| colon_L40 | 0.277 / 5.86 | 0.197 / 5.71 | 0.026 | ≈ |
| colon_L42 | 0.243 / 5.79 | 0.197 / 5.73 | 0.011 | ≈ |
| colon_L44 | 0.279 / 5.00 | 0.199 / 4.84 | 0.033 | ≈ |
| think_L40 | 0.295 / 9.12 | 0.198 / 8.96 | 0.018 | ≈ |
| think_L42 | 0.233 / 7.79 | 0.198 / 7.89 | 0.013 | ≈ |
| think_L44 | 0.308 / 5.77 | 0.199 / 5.56 | 0.037 | ≈ |

Stage 2b directions (class-mean z, `:`): J-part energy share and own colour atom in the k = 25 support.

| layer | lens | median J energy (all tables) | median J energy (leave-table-out) | own atom in support |
|---|---|---|---|---|
| L40 | J_NP | 0.044 | 0.044 | 1/12 |
| L40 | J_CB | 0.075 | 0.075 | 3/12 |
| L42 | J_NP | 0.039 | 0.039 | 1/12 |
| L42 | J_CB | 0.064 | 0.063 | 2/12 |
| L44 | J_NP | 0.039 | 0.038 | 0/12 |
| L44 | J_CB | 0.070 | 0.066 | 0/12 |

**Unexpected / disclosed.**
1. G-geom gate-code error (above).
2. The role table first marked collisions only when member and partner shared a colour; 006's census also drops pairs
   where a role colour fills another role. Fixed to 006's definition before any reader outcome; collision-free start
   and answer D on 006's own likelihoods then reproduce 006's post hoc values exactly (e.g. Oracle L40 start 0.186,
   answer 2.232; 44 tables).
3. Stage 1a run r1 was stopped before any reader-piece outcome to switch to an exact 2-rows-per-activation scorer
   (≈ 14× faster); r1's prompt-only reader floor had been printed and seen (D_z 1.08). r1 is kept, unanalysed.
4. Template lens: the O4 convention check (authors' code, `camilablank/template-lens` @ e5f7748, `template_readout_scores`)
   was done after the first scoring instead of before. It confirms the pre-registered convention (cosine on the raw
   residual, no centring; metadata `space: raw`), so nothing changed.
5. The analysis unit tests were written and run while Stage 1a collected data, before any analysis was run.
