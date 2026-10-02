# 001 — Leverage profile (brew, chain) and off-trajectory content (brew)

*v3, 2026-10-02 — approved by the PI ("proceed") with the second external review's refinements (review of 56cb7ab).
Not yet run. v1 → v2 → v3 changes are listed at the bottom. Governs `scripts/001_leverage_profile/` (to be written after approval).
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
| Chain | step sequence (28 usable items in 20 groups) | 10 / 10 groups, stratified by depth; reports count surviving independent groups, not only pairs; chain may stay a secondary pilot if support is thin | items with the same steps produce identical prompts (15 cross-item collisions), so the split is by group and pairs are de-duplicated as unordered {prompt, prompt}; `chain-w2-d3p-0408` forced into dev |

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

**Regime (changed in v2; adopted by the PI in v3).** SEC true-fp32 replay for every block output ≥ 40 (`regime.SecReplay`, l0 = 39, resident:
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

**(iii) Matched decodability, cross-fitted (v3).** The edit changes the table membership of c and c′, so the
membership probes of §3.4 are the matched variable.
- Let f_c^(−t) be colour c's probe fitted on **natural** renders only, from the outer folds that exclude table t.
- Apply the same fitted probe to the original and the substituted render of held-out table t.
- Directional test per item:
  - f_c^(−t)(h_sub) − f_c^(−t)(h_orig) < 0: the removed colour becomes less detectable;
  - f_c′^(−t)(h_sub) − f_c′^(−t)(h_orig) > 0: the inserted colour becomes more detectable.
- Report the fraction of items in the predicted direction (sign test) and the mean shift with a cluster interval.
- Substitution renders are never used to fit a probe.

Row binding (which colour a given key maps to) is secondary and descriptive: about 25 dev items per key cannot power a
12-way probe.

### 3.4 Decodability (probes; brew emission cells; no directions used anywhere else)

**Activations:** block outputs at emit_asst, emit_think, emit_colon × L40/42/44/56/60, in the §2 regime, for every dev
render (members, all partner candidates, substitution renders; ≈ 280 renders over 50 tables). Probes are fitted on natural
renders only; substitution renders are only scored (§3.3 iii).

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
| Stir cells (start pairs only, v3) | brew stir positions, which precede the only differing token of a **start** pair | patch is a bitwise no-op |
| Pre-difference prefix (every pair type, v3) | every position before a pair's first differing token | donor and recipient activations bitwise identical at every recorded layer |

Substitution pairs differ inside the rule table, which comes before the stir sentence, so their stir activations may
legitimately differ. The stir no-op applies to start pairs only.

**Localization results (reported, never alarms).** Start-token block-output patches at L0, L1, L2, L4. Block 0 is a
GatedDeltaNet layer whose recurrent state already carries the start colour into later positions within the block, so
the share there can be well below 1. Chain uses the start digit.

### 3.6 Noise floor

On 20 random dev pairs per family:
- 8 random perturbations of relative size 1e-4 at emit_colon / the final cell at L42 (SEC regime);
- the same at the start token at L2 (bf16 lower stack).

Reported per pair as a fraction of G. Every "negligible" call needs the bound to exceed the noise floor by ≥ 10×.

### 3.7 Descriptive reader snapshot (included by the PI in v3)

This stage runs after every reader-free output of 001 is written and its sha256 recorded. Reader outputs never select,
change or threshold anything in 001.

**Material:**
- 10 **distinct dev tables**: the first 10 in bank order that have a gated start pair and a valid substitution;
- their three emission cells.

**Activations read, per table:**
- original member;
- its start partner;
- the substituted render.

**Arms per activation:**
- real activation;
- no activation (the verbalizer's marker left as is);
- shuffled activation: the same cell and layer from another snapshot table (a fixed derangement).

**Readers on their contracts:**
- J_NP (standard and WSB cosine), J_CB, R_CB, logit lens at L40/42/44/56/60;
- Oracle Lens at L40 and L44;
- NLA at L42 (+ AR FVE);
- verbalizers sample k = 4 with WSB's sampling.

**The question it answers:** when the activation changes from "table contains c" to "table contains c′", does the
reader's description change accordingly? It also gives the explained-disagreement cases: what each reader says where
§3.1–3.4 measured leverage and decodability.

**Reader-input bridge:** readers are fed the stock-forward activation (WSB convention) and, as a check, the SEC
activation of the same cell. Reported:
- cosine between the two;
- top-10 overlap per lens;
- colour mentions per verbalizer.

Reader scores are not pooled with SEC assay results unless these agree.

**Qualitative samples:** every readout of the snapshot is saved in full and shown in the report.

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
| Reader snapshot (§3.7; 10 tables × 3 activations × 3 arms; verbalizers k = 4) | ≈ 35 min |
| **Total** | **≈ 2 GPU-h and ≤ 1 CPU-h** |

## 9. Decisions (resolved 2026-10-02)

1. **Regime:** SEC true-fp32 for every L ≥ 40 (adopted). SEC results are reported as results under that replay regime.
2. **Reader snapshot:** included, separate and after hashing (§3.7).
3. **Chain split:** 50/50 by step-sequence group; independent surviving groups are reported.
4. **Reader-input bridge:** stock vs SEC stability is checked on the snapshot before any pooling (§3.7).

## 10. Report requirements

- Every table comes with its intervals and sign counts.
- A **qualitative section** shows:
  - a fixed-seed sample of 8 brew and 4 chain dev pairs (seed 20261003), with their prompts, per-cell shares and damage;
  - all 10 snapshot tables with full reader outputs;
  - every substitution case where behaviour or transplant exceeded the negligible bound.
- Gate outcomes (including every failed attempt) and limitations sit next to the claims they qualify.

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

## Changes v2 → v3 (2026-10-02; second external review of 56cb7ab, PI approval)

1. PI decisions recorded (§9): SEC adopted, reader snapshot included, chain split kept, bridge check required.
2. Substitution decodability made explicitly cross-fitted: probes fitted on natural renders of other tables only, both
   renders of the held-out table scored with the same probe, directional test (§3.3 iii, §3.4).
3. Stir-cell no-op scoped to start pairs; a pre-difference-prefix identity check added for every pair type (§3.5).
4. Snapshot refined: 10 distinct tables, original/partner/substituted activations, no-activation and shuffled same-cell
   controls (§3.7).
5. Report requirements, including qualitative samples always (§10).
6. Code (commit 79f835a): the subject-model integrity check runs at use time (`Engine.forward`,
   `SecReplay.build`/`capture`), with a lifecycle test (62/62 tests).

## Amendment A1 (2026-10-02, before the snapshot ran; PI request)

The descriptive reader snapshot (§3.7) also follows WorkspaceBench's own reader protocol, so its readouts are
comparable with WSB's published brew results. WSB's post: J/R/logit top-10 at layers 20–60 every 4; Oracle Lens one
sample across the same layers; NLA one sample at L42.
- Lenses (J_NP standard + cosine, J_CB, R_CB, logit lens): layers 20–60 every 4 plus L42, on all three emission cells;
  real and shuffled arms everywhere, SEC arm at the profile layers.
- Oracle Lens: one sample per emission cell at layers 20–60 every 4 (all on-contract: its training grid), real and
  shuffled arms, plus a no-activation run per layer.
- NLA: k = 4 at L42 on all three emission cells (real, shuffled; SEC at emit_colon), plus one no-activation batch.
- The v3 contract runs (Oracle k = 4 at emit_colon L40/L44 with real / none / shuffled / SEC arms) are unchanged.

Still descriptive only: the added layers have no causal measurement and feed no decision.

## Amendment A2 (2026-10-02, after the external audit; probe repair; PI go "move onto fixing the probes")

**Why.** The audit found that the membership control task (§3.4) sat far from chance: median AUC 0.58 over 165 fits,
maximum 0.84. A diagnostic on the saved captures (40 fresh random table-level labels each, on four of the worst targets)
found three faults in the implementation, none of them leakage:
1. **One control draw per fit, with one seed shared by every target.** With about 10 negative tables, a single
   random-label AUC has SD ≈ 0.12. The shared seed repeats the same table assignment across all 15 cell × layer fits
   of a colour (red: 0.80–0.84 everywhere). Fresh draws average 0.49–0.54.
2. **AUC pooled across folds.** Decision scores from five different fold models were ranked together. On pure-noise
   activations this biases the AUC to 0.45 under the null; per-fold AUC gives 0.49–0.50.
3. **The permutation null reused the C chosen per fold on the real labels.** The real and control fits tuned C by
   inner CV, so the null did not repeat the procedure.

Fault 1 matters for the decision rules. Membership selectivity was computed against an unlucky control around 0.57,
which pushed the L40–56 calls below the 0.10 line. Those are the layers where the Oracle and NLA read.

**Repair** (decided before any repaired result was seen):
- Binary targets: AUC is computed within each outer fold and averaged over folds with both classes. The pooled AUC
  is kept as a secondary field.
- Controls: K independent draws per target through the full procedure (C tuned), K = 20 for binary and 10 for 12-way.
  Selectivity = real − mean of the control draws; their SD is reported.
- Permutation null: every permutation repeats the full procedure, inner-CV C tuning included. N = 200 for binary and
  100 for 12-way, because 12-way full-procedure fits take about 26 s. Each target gets its own seed stream.
- Pooled membership per cell × layer: one joint permutation of table identities per draw, applied to every colour.
  The colour-averaged null is then the null of the colour-averaged statistic. Per-colour p-values are reported too.
- Unchanged: folds (StratifiedGroupKFold(5) by table, the same fold seed), C grid, features, the BH-FDR + selectivity
  ≥ 0.10 rule, and the cross-fitted directional test, which uses the real fits' fold models.
- Rows from the first run stay in git history (commit dc33f48, `analysis/probes.json`); nothing scored is deleted.
- Implementation: every fit in the repaired run (real, control, null) uses `_SpanLR`. It solves the same
  L2-logistic objective in the span of the standardised training rows, which contains the L2 optimum when
  n_train < d (representer theorem; the intercept is unpenalised). It is 3× faster single-threaded and avoids the
  memory contention that made 120 parallel full-dimensional 12-way fits about 10× slower.
  - Against the full fit (tests/test_probe_span.py): predictions match at C ≤ 1e-2.
  - At C = 1, where the problem is nearly separable and the solver stops early, 12-way predictions agree on 93.5% of
    held-out rows.
  - So real-label 12-way values can shift slightly from the first run. Within the repaired run, real, control and
    null all use one estimator.
