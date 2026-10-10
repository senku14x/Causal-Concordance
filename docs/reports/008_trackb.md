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

- **The paper's §3.3 privilege is not reproduced on Qwen brew under a band-limited version of the paper's protocol, now
  with an adequate lever** (3 emit cells × L40–48; the paper swaps and clamps at every position over a layer band;
  corrected 2026-10-06, report 31: was "under the paper's own protocol").
  - Swapping z's class-mean direction across the emit cells and L40–48 (per-layer clamps, natural coordinates) moves the
    answer by a median S of 0.21 [0.11, 0.26] and flips 16% of answers. That is about 4× 2b's one-cell lever.
  - Of that effect, the J part (k = 25 pursuit, 3.3–8.7% of the direction's raw energy; corrected 2026-10-06, report 31:
    was "about 4–9%") carries S 0.02–0.03. The non-J part carries 0.21, the same as the full direction.
  - J − non-J is −0.17 [−0.24, −0.09] (J_NP) and −0.14 [−0.21, −0.08] (J_CB; corrected 2026-10-06, report 31: −0.14465 was
    written −0.15); J wins in only 8–10 of 82 pairs.
- **The paper's mediation result is not reproduced under a band-limited clamp** (3 cells × L40–48; the paper clamps at
  every position) (corrected 2026-10-06, report 31; was "reversed here"). Clamping the J-lens coordinates of the J part's support barely
  changes the non-J swap (J_NP 0.21 → 0.20; J_CB 0.20 → 0.18; corrected 2026-10-09: J_CB's unclamped median is 0.2048,
  `trackb_b1.json` `conditions.emit3|B_raw_nonJ_J_CB.median_S`; report 36 E13). On Claude, the paper's clamp took the non-J effect from 28% to 6%.
- **Whitening narrows the gap but never reverses it.**
  - In the real whitened swap (whiten with Σ, split and swap in whitened coordinates, map back), the J part holds only
    0.5–2.0% of the whitened energy (corrected 2026-10-06, report 31; was 0.6–1.7%) and moves S 0.07 (J_NP) / 0.10 (J_CB), against 0.14 / 0.11 for the non-J part.
  - J_CB is a tie there (−0.01 [−0.05, +0.03]), so per unit whitened energy the J part is potent. But it never carries
    more than the non-J part.
- **Matched magnitudes give J no advantage either.**
  - At equal raw edit norm, J_CB ties the non-J part (J − non-J −0.004 [−0.016, +0.007]). At equal whitened norm it
    is slightly below it (−0.009 [−0.029, −0.000]; the CI's upper end is −0.00009, so it excludes 0) (corrected 2026-10-06, report 31; was
    "ties" at both norms).
  - J_NP stays below it (−0.045 [−0.087, −0.020]).
- **Reading** (rewritten 2026-10-06, report 31. The earlier text called the sparse k = 25 J part "the J subspace" and
  set it against 007's reader result on the dense 1,024-dim J subspace: two different objects. It said "the reader–model
  dissociation stands. The Oracle privileges J for reading z (007); the model's answer does not.")
  - **008, on its own object** (the sparse k = 25 J part of a class-mean z direction; emit cells × L40–48): the J part
    has no more demonstrated influence on this answer than the non-J part. Supported empirical claim (dev;
    Qwen3.6-27B; brew; 82 pairs).
  - **The reader–model dissociation, on one object (007's dense 1,024-dim J subspace, brew `:`):**
    - reader side (007 Stage 1a dev, brew test; `results/007_jsubspace/analysis/{stage1a,brew_test}.json`): the
      Oracle L40's D_z from J alone is 5.76 vs 3.22 nats for the whole activation on dev and 5.03 vs 2.87 on test;
      rest of J keeps 0.30 / 0.29;
    - model side (007 Stage 2, 33 pairs, 32 for the L40 recovery contrasts (corrected 2026-10-09), `stage2.json`; S_full 0.28 / 0.43 / 0.49 at L40 / L42 / L44): dense-J swaps
      track the Δ-share at L42 (recovery J − Δ-matched random +0.02 [−0.00, +0.03]) and exceed Δ-matched random
      modestly at L40 / L44 (+0.07 [+0.04, +0.11] / +0.09 [+0.02, +0.16]), below the pre-registered 0.10 margin.
  - 008's sparse-J-part result is separate evidence on a different object, consistent with Stage 2.
- **The true lens-coordinate swap (arm A) is not read: its positive control failed.**
  - Swapping the J-lens atom coordinates of the *answer* colours (a ↔ a′) moves the answer by only S 0.010 at these
    layers, far below the 0.15 power bar.
  - The z-atom swaps are near 0 in raw coordinates (0.0012–0.0024, α1 and α2, both lenses). In whitened coordinates
    they are 0.002 / 0.006 at α1 (J_CB / J_NP) and 0.005 / 0.031 at α2; the J_NP α2 swap's CI [0.006, 0.064] excludes 0
    (corrected 2026-10-06, report 31: "0.001–0.006 raw" mixed the raw arms with `A_white_a1`).
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
- the whitened arm fails its power gate on the subset (0.09; corrected 2026-10-06, report 31: 0.090577 was written 0.12).

## 4. What it changes

- **007's Stage 2b test (the sparse J part) no longer rests on a weak lever** (corrected 2026-10-06, report 31: said
  "007's model-side half"; 007's dense-J Stage 2 already had an adequate lever, S_full 0.28 at L40). 2b's one-cell,
  one-layer adaptation moved S 0.04–0.06. The
  paper's protocol here moves 0.21, and the J part's share is the same small fraction.
  - The model gives the sparse J part no privilege on brew's answer at these cells and layers, in raw or whitened
    coordinates (object: the k = 25 J part, not 007's dense J subspace; wording corrected 2026-10-06, report 31).
  - ~~The reader-side J dependence (the Oracle) is therefore a property of the reader.~~ Struck 2026-10-06 (report 31):
    it drew a reader conclusion from a model result on a different J object. On the dense J subspace, 007 Stage 2
    points the same way (§1, "Reading"). "Reader-specific" holds at `:` only: at `</think>` the NLA also collapses
    without J (007 §8).
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
- The whitening uses a chat-corpus Σ, not the Oracle Lens's own whitener. ~~which is not public~~ That whitener is
  public (`whitening_iolens_pooled_L{L}`, `agu18dec/local-workspace`, `ckpts/ar/ptag/whiteners/`; found in 011) and was
  not used here (corrected 2026-10-06, report 31).
