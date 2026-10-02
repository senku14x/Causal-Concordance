# 001 — Leverage profile (brew, chain) and off-trajectory content (brew)

*Draft v2, 2026-10-02, for PI review. Not run. v1 → v2 follows the external review of commit c13fd43 and the PI's
response (changes listed at the bottom). Governs `scripts/001_leverage_profile/` (to be written after approval).
Context: `docs/context/SPEC.md` + `docs/context/AMENDMENT_01.md` (including its 2026-10-02 errata); engine, regime and
gates as built and tested in block 000 (`STATUS.md`).*

---

## 1. Why

Two premises of the label design are untested and cheap to test before any direction, probe-based label or reader is
built.

1. **Where does a single (position, layer) cell carry leverage over the answer?** Whole-cell patches show where the
   *natural pair difference* (start → z → answer) has influence, and whether that influence varies between pairs at a
   position. This is a diagnostic. W can still be estimated if the contrast comes from content inside one cell
   (premise 2).
2. **Does brew supply a not-needed class?** AMENDMENT_01 §2 makes brew's off-trajectory colours at the emission cells
   the candidate decodable-not-needed content. That requires two separate facts:
   - (a) the content is decodable there;
   - (b) changing it changes neither the model's answer behaviour nor, when its activation is transplanted, the answer.

   Task invariance alone is not (b): the mathematical answer being unchanged does not show the model's computation is.

The pilot's output is a small set of inspectable cases. For each: what changed in the prompt, what is decodable at the
cell, and what the edit changed downstream.

## 2. Materials

**Splits** (fixed here, before any outcome is seen).

| Family | Unit of split | Dev / test | Notes |
|---|---|---|---|
| Brew | table (each of the 100 items has its own table; partners and edits share their item's table) | 50 / 50 tables, stratified by z (seed 20261002) | `brew-s2-r10d2-0000`, `-0001` (touched in block 000) forced into dev |
| Chain | step sequence (28 usable items in 20 groups) | 10 / 10 groups, stratified by depth | items with the same steps produce identical prompts (15 cross-item collisions), so the split is by group and pairs are de-duplicated as unordered {prompt, prompt}; `chain-w2-d3p-0408` forced into dev |

**Pairs.**
- **Start pairs (the natural pair difference).**
  - Brew: same table, different start colour (361 candidates, all token-aligned, differing only at the start-colour
    token).
  - Chain: same steps, different start number (110 candidates before de-duplication).
  - Every candidate (dev and test) is gated in stage `gate` (greedy correct and ≥ 9/10 at T = 0.8, both members), and
    admission counts plus the difficulty distribution are reported for the whole bank. The gate is label-free and
    reader-free. Only dev pairs enter the measurements.
- **Off-trajectory substitution pairs (brew, new in v2).**
  - For each dev item, one colour c that is in the table but off the trajectory (c ∉ {start, z, answer}) is replaced
    by a colour c′ absent from the table, everywhere it occurs (its row key and any row outputs).
  - "orange" is excluded on both sides because its article ("An") would also change.
  - By construction the trajectory and the answer are unchanged, only colour tokens inside the rule table change, and
    the renders stay token-aligned.
  - The edit flips table membership: c goes present → absent, c′ goes absent → present.
  - Checked on the bank: all 100 items admit at least one such edit (1,138 valid (c, c′) choices). One (c, c′) per
    item is drawn by seed.

**Cells.**
- **Brew:** the 24 WSB pinned positions (stir ×6, start ×6, question ×9, emit_asst, emit_think, emit_colon),
  re-asserted per render with the pinned tokenizer. Substitution pairs add the substituted key token as a reference
  cell.
- **Chain:** the question span (" What" → final token; 25 positions at T = 78) including the final cell.
- **Layers:**
  - L40, L42, L44 (reader-matched: NLA at 42, Oracle on-contract at 40/44);
  - L56 and L60, where every lens first names colours on the smoke cell (green and black), so the profile can say
    whether late-layer naming coincides with late-layer leverage;
  - early layers for the controls only (§3.5).

**Regime (changed in v2).** SEC true-fp32 replay for every block output ≥ 40 (`regime.SecReplay`, l0 = 39, resident:
bf16 blocks 0–39 + fp32 blocks 40–63 + fp32 W_U ≈ 72 GB; noise floor ≈ 2e-4 nats vs 0.05–0.25 under RHUT). Early-layer
controls and the embedding anchor run in the bf16 lower stack, which is the stock forward. Every comparison is made at
one sequence length, with sources captured at the matching length. The gate stage runs on the stock model before the
fp32 stack is built.

## 3. Estimands and measurements

Notation: member x (answer a), partner x′ (answer a′); m(·) = LSE_{S(a′)} log P − LSE_{S(a)} log P, full-sequence teacher
forcing over every token of every accepted spelling; G = m(x′) − m(x).

### 3.1 Whole-cell patch share (start pairs)

For every dev pair, cell c and layer ℓ: replace x's block-ℓ output at c by x′'s; Δm_c; S_c = Δm_c / G. Direction m ← p
for every pair; p ← m on a fixed random 30% of dev pairs (symmetry check).

**Outcomes per (pair, cell, layer)**, kept separate and never merged:

| Outcome | Rule |
|---|---|
| toward | S ≥ τ (0.10) |
| away | S ≤ −τ |
| negligible | \|S\| < ε′ (0.05) |
| intermediate | otherwise |

Damage readouts (Δlog P(a), Δlog P(a′), top-1 retention after c, off-pair mass) are recorded with every patch.

**Group-level negligibility** per (family, position, layer): TOST equivalence of E[S] within ±ε′ (90% cluster-t
interval; `stats.equivalence`). A group-level equivalence result is never turned into item-level absence labels.

**Wording rule.** A negligible whole-cell share means "the natural pair difference at this cell has negligible
influence on this answer". It never means "nothing at this cell matters": two components of the patched difference
can cancel, so the whole-cell patch is a reference measurement, not a ceiling.

**Within-position variation (diagnostic only).** For each (family, position, layer) with n ≥ 20 dev pairs, report:
- the fractions toward / negligible / away;
- the between-pair SD of S with a cluster-bootstrap CI;
- the SD of S against the noise floor (§3.6).

### 3.2 Span patches (context only)

The same pairs, whole regions replaced at L40–60 jointly:
- brew: start, question, emission, and start + question + emission;
- chain: input digits, question span.

Reported next to the single-cell shares; never used for a label.

### 3.3 Off-trajectory content (core; new in v2)

**(i) Behaviour.**
- Gate the substituted prompt (greedy + ≥ 9/10).
- Measure Δlog P(a) between the clean original and substituted prompts.
- Measure the total-variation distance between their answer distributions over the 12 colours (LSE over spellings,
  renormalised over colours).

The answer is task-invariant, but the model may not be.

**(ii) Activation transplant.**
- Full-cell patch of the substituted render's state into the original at the 3 emission cells × 5 layers; p ← m on 30%.
- Measures: Δlog P(a), TV, damage.
- Reference: the substituted key token patched at L0. Its effect estimates the behavioural difference carried
  through that token (localization, not an engine check).

There is no G here (a = a′), so no share is formed.

**Negligible bound** (provisional; frozen with τ and ε′ in AMENDMENT_02 before any test label):
- |Δlog P(a)| < ε′ · median(G) over brew dev start pairs;
- TV < 0.05.

**(iii) Matched decodability.** The edit changes the table membership of c and c′, so the membership probes of §3.4
are the matched variable. The probe trained on natural dev renders is applied to each substitution pair. Item-level
test: the score for c′ rises and for c falls, original → substituted (sign test across items).

Row binding (which colour a given key maps to) is secondary and descriptive: about 25 dev items per key cannot power a
12-way probe.

### 3.4 Decodability (probes; brew emission cells; no directions used anywhere else)

**Activations:** block outputs at emit_asst, emit_think, emit_colon × L40/42/44/56/60, in the §2 regime, for every dev
render (members, all partner candidates, substitution renders; ≈ 280 renders over 50 tables).

**Targets:**
- z, start and answer (12-way each; the {input, z, answer} comparison of AMENDMENT_01 §3);
- membership of each colour c: present and off-trajectory vs absent from the table (renders with c on the trajectory
  excluded for that c), pooled over colours by mean AUC;
- secondary: the answer's successor T(a) when off-trajectory (look-ahead), and row bindings (descriptive).

**Probe:**
- L2-regularised logistic regression on standardised features, no PCA in the primary pipeline. A PCA sensitivity
  check, if run, caps the rank at min(64, n_train − 1) inside each training fold.
- C from {1e-3, 1e-2, 1e-1, 1} by inner grouped CV.
- Outer StratifiedGroupKFold(5), grouped by table.
- Every training and test fold must contain both classes (asserted).
- A target with fewer than 5 tables per class in dev is reported as "insufficient support", never as "not decodable".

**Controls:**
- Hewitt–Liang control task: a random label per table, consistent across its renders. It should sit at chance under
  grouped folds; above chance flags leakage.
- 200 table-level label permutations as the null.

**Decodable:** pooled membership AUC above its permutation null at BH-FDR q = 0.05 across the 15 (cell × layer) tests,
**and** selectivity ≥ 0.10, at one or more emission cells × layers. The same rule is reported for z, start and answer.

### 3.5 Controls

**Engine checks (implementation invariants).** Any failure means the engine is broken (rule 3).

| Check | Cell | Requirement |
|---|---|---|
| Embedding anchor | start-colour token, input embedding | partner logits reproduced bitwise |
| Self-patch | emit_colon, each layer | m identical bitwise |
| Stir cells | brew stir positions (before the only differing token in all 100 items) | patch is a bitwise no-op |

**Localization results (reported, never alarms).** Start-token block-output patches at L0, L1, L2, L4. Block 0 is a
GatedDeltaNet layer whose recurrent state already carries the start colour into later positions within the block, so
the share there can be well below 1. Chain uses the start digit.

### 3.6 Noise floor

On 20 random dev pairs per family:
- 8 random perturbations of relative size 1e-4 at emit_colon / the final cell at L42 (SEC regime);
- the same at the start token at L2 (bf16 lower stack).

Reported per pair as a fraction of G. Every "negligible" call needs the bound to exceed the noise floor by ≥ 10×.

### 3.7 Descriptive reader snapshot (optional; PI decision)

The PI's original 001 instruction was "no readers", and none of the measurements above uses one. An *explained
disagreement* (e.g. the NLA names green where J does not; what do §3.1–3.4 say about green at that cell?) needs reader
outputs on the same cells.

Proposed separate stage `readers_snapshot`:
- runs after every reader-free output of 001 is written and hashed;
- 10 dev pairs fixed by rule (the first 10 dev pairs in bank order), their emission cells;
- readers on their own contracts: J_NP / J_CB / R_CB / logit lens at L40–60, Oracle at L40/L44, NLA at L42, each with
  the no-activation arm;
- descriptive only, never an input to any 001 decision.

Include it in 001 or defer it to 002 — the PI's call.

## 4. Gates (before any outcome is read)

1. Span/rendering: every render passes the WSB position assertions; every pair is token-aligned with only the allowed
   tokens differing (start token; or the c → c′ colour tokens inside the table).
2. Engine checks of §3.5.
3. Fidelity: full-cell patches are exact writes (ρ = κ = 1 recorded).
4. Damage: per patch; emission-cell patches that raise the off-pair mass by > 0.10 in > 20% of a family's pairs are
   reported as damage-dominated.
5. Sequence-length discipline (sources per continuation) and subject-model integrity (`registry.assert_subject_model_intact`,
   called by every capture and labelling entry point) are asserted in code.

A failed gate is repaired and the stage re-run; nothing scored is deleted.

## 5. Predictions and priors

*v1 priors written 2026-10-02 before any 001 data.*

Already seen, and disclosed because they inform these priors:
- the block-000 regime check on `brew-s2-r10d2-0001__blue` (whole-cell share emit_colon L44 ≈ 0.42–0.45, L40 ≈ 0.21–0.23,
  start token L40 ≈ 0.05);
- the reader smoke readouts on `brew-s2-r10d2-0000`:
  - under WSB's layer grid, the J cosine readout names green at emit_colon L36 and emit_think L40;
  - every lens names black then green from L52–60;
  - the NLA names green 4/4 at L42;
  - the Oracle at L40 names green but also both off-path candidates;
- 2 of the first 3 brew gate attempts failed.

*v2 revisions (2026-10-02, still before any 001 data) are marked †.*

| # | Prediction | Prior |
|---|---|---|
| Q1 | Gate pass rate ≥ 40% of brew candidates and ≥ 40% of chain candidates | 55% |
| Q2† | Start-token block-output patch at L0: median S ≥ 0.5 across brew dev pairs (v1 asked for S ≥ 0.9 in ≥ 90% of pairs at 75%; revised because block 0 is a GatedDeltaNet layer that mixes the colour forward within the block) | 60% |
| Q3 | Start-token S at L40–44 has median \|S\| < ε′ (the colour has been moved on) | 70% |
| Q4 | emit_colon median S increases L40 < L42 < L44, with L44 median ≥ 0.3 | 65% |
| Q5† | Every brew question-span cell is negligible at the group level (TOST within ±ε′) at L40–44 | 45% |
| Q6 | Within-position variation present (≥ 10% toward and ≥ 10% negligible) at ≥ 1 brew emission cell × layer | 50% |
| Q7 | The same for chain (final cell or question span) | 40% |
| Q8 | z decodable at emit_colon (12-way, §3.4 rule) at L40, L42 and L44 | 70% |
| Q9 | Off-trajectory membership decodable at ≥ 1 emission cell × layer (§3.4 rule) | 55% |
| Q10 | The answer is more decodable than z at emit_colon L44 | 65% |
| Q11 | Damage-dominated emission-cell patches in < 10% of brew dev pairs | 75% |
| Q12† | Substitution edits are behaviourally negligible (§3.3 bound) in ≥ 90% of dev items | 70% |
| Q13† | Emission-cell transplants of substitution edits are negligible in ≥ 90% of dev items at every layer | 75% |
| Q14† | emit_colon median S at L56 and L60 exceeds the L44 median | 70% |

## 6. Decision rules

1. **(Diagnostic, not a stop rule; v2.)** If no position shows within-position variation in whole-cell leverage, record
   "whole-cell leverage does not supply the contrast". Whether W is estimable is then decided by the content-level
   contrast (rule 2).
2. **If off-trajectory membership is not decodable at the emission cells (§3.4), or the substitution edits are not
   negligible behaviourally and by transplant (§3.3), brew has no not-needed class.** Design the RHUT
   tagged-selection fallback next.
3. **If any engine check of §3.5 fails, the engine is broken.** Fix it first and re-run. A small block-0 patch is a
   localization result, not an engine failure (v2).
4. If fewer than 20 brew or 15 chain dev pairs pass the gate, report and stop for the PI (power).
5. τ, ε′ and the §3.3 bound may be revised once from this profile (dev only) and frozen in AMENDMENT_02 before any
   test-split label is computed.

## 7. Stop rules

Stop and report (no silent continuation) if:
- any gate in §4 fails twice after repair;
- the gate admits < 20 brew dev pairs;
- the SEC noise floor exceeds one tenth of the negligible bound on any noise-floor pair.

## 8. Budget

Sources: `timing_smoke.json` and the regime check. Stock forward ≈ 0.23 s; SEC fp32 replay of blocks 40–63 ≈ 0.09 s;
SEC lower-stack capture ≈ 0.15 s (estimate: 40 of 64 bf16 blocks); SEC build 8.5 s; gate ≈ 5.4 s (brew) / 4.4 s (chain)
per pair.

| Stage | Wall-clock |
|---|---|
| Gate, all candidates (361 brew + ≤ 110 chain) + 50 substitution prompts | ≈ 45 min (stock model) |
| Brew profile, ~90 dev pairs × (24 cells × 5 layers + controls + spans) | ≈ 20 min (one lower-stack capture per prompt, one 0.09 s replay per edit) |
| Brew reverse direction, 30% | ≈ 6 min |
| Chain profile, ~25 dev pairs × 25 cells × 5 layers × ≤ 3 continuations | ≈ 15 min |
| Off-trajectory arm, ~50 dev items (behaviour + 15 transplants + reverse 30%) | ≈ 3 min (+ gate above) |
| Noise floor, 40 pairs × 16 | ≈ 2 min |
| Decodability captures (~280 renders × 5 layers) | ≈ 2 min |
| Probes (CPU; 3 × 12-way + 12 membership targets × 15 cell-layers, 5 folds, inner CV, 200 permutations) | ≈ 30–60 min CPU |
| Optional reader snapshot (§3.7; 10 pairs) | ≈ 15 min |
| **Total** | **≈ 1.5 GPU-h (+ 15 min with §3.7) and ≤ 1 CPU-h** |

## 9. Open questions for the PI

1. **Regime.** v2 uses SEC for every L ≥ 40. AMENDMENT_01's 5% rule did not require the switch (3.3%), but
   negligible-effect calls at ε′ = 5% need a floor far below 5%. Confirm.
2. **Reader snapshot** (§3.7): include in 001, or defer to 002?
3. **Chain split:** 50/50 by step-sequence group leaves ≈ 10 groups per side. Keep it, or use 70/30?
4. **Reader-input bridge.** Reader inputs come from the stock forward (WSB convention); SEC activations at L ≥ 40 are
   fp32 replays, about 1% different. SEC's lower stack *is* the stock forward, so this bridge is exact only for layers
   ≤ 39. Before reader scoring (002+), check that reader readouts at the label layers are stable across the two capture
   regimes, or feed readers the SEC activations.

---

## Changes v1 → v2 (2026-10-02; external review of c13fd43 + PI response)

1. Signed small-effect test fixed. Negligible = |S| < ε′; toward / away / negligible / intermediate kept separate;
   group claims by TOST equivalence; never converted into item labels (§3.1).
2. Whole-cell share worded as the influence of the *natural pair difference*; the patch is a reference, not a ceiling
   (cancellation) (§3.1).
3. Off-trajectory arm made core: behaviour first, then activation transplant. The edit is a colour substitution so that
   its decoded variable (table membership of c and c′) is exactly what the edit changes; row binding is secondary
   (§2, §3.3, §3.4).
4. PCA-256 removed (rank ≤ n_train − 1 ≈ 183); stratified grouped folds, both classes asserted, insufficient support
   reported as such (§3.4).
5. Block-0 patch is a localization result; only the embedding anchor, self-patch and stir cells are engine checks
   (§3.5, rule 3); prior Q2 revised.
6. Decision rule 1 demoted to a diagnostic (§6).
7. SEC true-fp32 regime for all L ≥ 40 (§2), pending PI confirmation (§9.1).
8. Layers 56 and 60 added to the profile (§2). The reader layer scan is a separate optional stage (§3.7) because 001
   was specified reader-free.
9. Subject-model integrity: reader setup that merges weights in place marks the model; every capture/label entry point
   refuses a marked model; tests added (code, 2026-10-02).
