# 008 — Track B: 007's model-side test, done with the paper's protocol (report)

**Run info.**
- **Design:** `docs/design_specs/008_009_trackb_brewcode.md` Part 1 and execution notes §E, committed before any swap.
- **Code:** `scripts/008_trackb/` (`sigma.py`, `trackb.py`, `trackb_analysis.py`).
- **Run:** `b1`, 2026-10-06, 33.3 GPU-min, plus 1 GPU-min for Σ and the 3.3 GPU-min smoke run.
- **Results:** `results/008_trackb/analysis/{sigma,directions_report,trackb_b1}.json`; raw per-pair rows in
  `outputs/008_trackb/b1/pairs/`.
- **Label: EXPLORATORY** (relaxed rules). The decision rules were fixed in the design before the run.
- **Scope:** Qwen3.6-27B; brew dev; 82 gated pairs from 33 tables; the three emit cells (assistant newline, `</think>`,
  `:`); L40–48.

## 1. Summary

- **The paper's §3.3 privilege does not replicate on Qwen brew under the paper's own protocol, now with an adequate
  lever.**
  - Swapping z's class-mean direction across the emit cells and L40–48 (per-layer clamps, natural coordinates) moves the
    answer by a median S of 0.21 [0.11, 0.26] and flips 16% of answers. That is about 4× 2b's one-cell lever.
  - Of that effect, the J part (k = 25 pursuit, about 4–9% of the direction's energy) carries S 0.02–0.03. The non-J
    part carries 0.21, the same as the full direction.
  - J − non-J is −0.17 [−0.24, −0.09] (J_NP) and −0.15 [−0.21, −0.08] (J_CB); J wins in only 8–10 of 82 pairs.
- **The paper's mediation result is reversed here.** Clamping the J-lens coordinates of the J part's support barely
  changes the non-J swap (0.21 → 0.20 / 0.18). On Claude, the paper's clamp took the non-J effect from 28% to 6%.
- **Whitening narrows the gap but never reverses it.**
  - In the real whitened swap (whiten with Σ, split and swap in whitened coordinates, map back), the J part holds only
    0.6–1.7% of the whitened energy and moves S 0.07 (J_NP) / 0.10 (J_CB), against 0.14 / 0.11 for the non-J part.
  - J_CB is a tie there (−0.01 [−0.05, +0.03]), so per unit whitened energy the J part is potent. But it never carries
    more than the non-J part.
- **Matched magnitudes give J no advantage either.**
  - At equal raw or whitened edit norm, J_CB ties the non-J part (raw-norm J − non-J −0.004 [−0.016, +0.007]).
  - J_NP stays below it (−0.045 [−0.087, −0.020]).
- **Reading.** 007's model-side headline now rests on a powered test that follows the paper's protocol:
  - **on brew's emit cells, L40–48, the J subspace gets no privilege on the answer beyond the non-J part;**
  - so the reader–model dissociation stands. The Oracle privileges J for reading z (007); the model's answer does not.
  - Supported empirical claim (dev; Qwen3.6-27B; brew; 82 pairs).
- **The true lens-coordinate swap (arm A) is not read: its positive control failed.**
  - Swapping the J-lens atom coordinates of the *answer* colours (a ↔ a′) moves the answer by only S 0.010 at these
    layers, far below the 0.15 power bar.
  - The z-atom swaps are near 0 too (0.001–0.006 raw).
  - Observation, not a reading: on Qwen brew at L40–48, single colour-token J-lens atoms are not a handle on the answer.

## 2. Gates (all passed; nothing quarantined)

| Gate | Result |
|---|---|
| G-Σ | Ledoit–Wolf λ = 0.0035–0.0043; minimum eigenvalue about 0.02; condition number about 1e4 (L40–48; 100 UltraChat conversations, 50,199 tokens) |
| G0 (zero clamp) | Δm = 0 exactly (2 pairs) |
| G-repro (whole-cell `:` L42 vs 001) | 82/82, max \|Δ\| 0.000 |
| G1 fidelity + G-clamp | 5,309 rows, 0 quarantined; maximum post-write coordinate error 1.0e-6 |
| G-RA | max ‖R A − I‖ 3.1e-14 |
| Power, arm B, emit3 | raw full 0.208 (passes); whitened full 0.165 (passes) |
| Power, arm B, colon only | raw full 0.132, whitened 0.118 (fail: colon-only results are not read) |
| Power, arm A | answer-atom swap 0.010 (fails: arm A is not read) |

## 3. Results (emit3; median S over 82 pairs [95% table-cluster CI]; flip = top-1 colour becomes a′)

| Condition | J_NP | J_CB | Flip (NP / CB) |
|---|---|---|---|
| full direction, raw | 0.208 [0.114, 0.258] | — | 0.16 |
| J part, raw natural | 0.023 [0.012, 0.028] | 0.034 [0.021, 0.043] | 0.00 / 0.00 |
| non-J part, raw natural | 0.210 [0.111, 0.250] | 0.205 [0.098, 0.249] | 0.16 / 0.16 |
| J part, raw-norm matched | 0.114 | 0.180 | 0.09 / 0.15 |
| non-J part, raw-norm matched | 0.210 | 0.208 | 0.17 / 0.17 |
| J part, whitened-norm matched | 0.108 | 0.158 | 0.06 / 0.13 |
| non-J part, whitened-norm matched | 0.207 | 0.202 | 0.16 / 0.16 |
| full direction, whitened swap | 0.165 [0.059, 0.275] | — | 0.15 |
| J part, whitened swap | 0.069 [0.031, 0.094] | 0.103 [0.069, 0.175] | 0.02 / 0.04 |
| non-J part, whitened swap | 0.139 [0.057, 0.244] | 0.113 [0.046, 0.209] | 0.11 / 0.10 |
| non-J swap with J coordinates clamped | 0.198 | 0.184 | 0.16 / 0.13 |
| J swap with non-J coordinates clamped | 0.008 | 0.013 | 0.00 / 0.00 |
| random direction pairs (norm-matched to full; 4 draws) | 0.007–0.017 | — | 0.00 |
| arm A: z atoms, raw α1 / α2 | 0.001 / 0.002 | 0.001 / 0.002 | 0.00 |
| arm A: answer atoms (positive control) | 0.010 | 0.010 | 0.00 |
| arm A: off-path atoms (negative control) | −0.000 | 0.000 | 0.00 |

**Paired J − non-J (median [CI]; J better / worse, out of 82):**

| Arm | J_NP | J_CB |
|---|---|---|
| raw natural | −0.169 [−0.240, −0.093] (8/74) | −0.145 [−0.214, −0.077] (10/72) |
| whitened swap | −0.046 [−0.110, −0.000] (30/52) | −0.008 [−0.054, +0.034] (39/43) |
| raw-norm matched | −0.045 [−0.087, −0.020] | −0.004 [−0.016, +0.007] |
| whitened-norm matched | −0.061 [−0.091, −0.027] | −0.009 [−0.029, −0.000] |

Every reading is **"not replicated"** by 2b's rule (S(non-J) ≥ S(J)).

**Clamps (paired change from the unclamped swap):**
- non-J swap with J clamped: −0.007 [−0.010, −0.003] (J_NP) and −0.014 [−0.018, −0.007] (J_CB);
- J swap with non-J clamped: −0.015 and −0.016. The J part's small effect partly runs through non-J coordinates.

**Geometry** (`directions_report.json`; median over the 306 class-mean directions per layer and lens):
- The J part holds 3.3–8.7% of a direction's raw energy (J_CB more than J_NP), and 0.5–2.0% of its whitened energy.
- The colour's own token atom is in the k = 25 support for 3–29% of directions, depending on layer and lens.

**The 33-pair subset (004's frozen pairs; one per table):**
- the same pattern under the natural raw swap (J 0.017 / 0.024 vs non-J 0.136 / 0.116);
- J_CB is "mixed" under norm matching (J 0.16 vs non-J 0.12);
- the whitened arm fails its power gate on the subset (0.12).

## 4. What it changes

- **007's model-side half no longer rests on a weak lever.** 2b's one-cell, one-layer adaptation moved S 0.04–0.06. The
  paper's protocol here moves 0.21, and the J part's share is the same small fraction.
  - The model gives the J subspace no privilege on brew's answer at these cells and layers, in raw or whitened
    coordinates.
  - The reader-side J dependence (the Oracle) is therefore a property of the reader.
- **Camila's whitening question is answered.**
  - Whitening raises the J part's effect (0.02–0.03 → 0.07–0.10) and makes J_CB tie the non-J part. It does not create
    a privilege.
  - The answer depends on the metric only in degree, not in direction.
- **The paper's §3.3 tool, the lens-coordinate swap of single-token atoms, has no purchase on Qwen brew at L40–48, even
  for the answer.** Any future J-type intervention on Qwen should use directions built from activations (class means,
  probes), not single-token lens atoms. This matches 005's Q-c, where the logit-lens direction matched the J atom.

## 5. Limits

- Dev only; one family; one model; the emit cells and L40–48 only (the readers' band).
- No later layers were swapped, where the paper's answer swaps act.
- Class-mean directions come from 230 renders, leaving the pair's table out. They are a probe-like handle, not the
  model's mechanism.
- The J part depends on the pursuit (k = 25, NNLS). Other sparsities were not tried.
- The whitening uses a chat-corpus Σ, not the Oracle Lens's own whitener, which is not public.
