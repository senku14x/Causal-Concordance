# 001 — Leverage profile (brew, chain) and off-trajectory decodability (brew)

*Draft, 2026-10-02, for PI review. Not run. Governs: `scripts/001_leverage_profile/` (to be written after approval).
Context: `docs/context/SPEC.md` + `docs/context/AMENDMENT_01.md`; engine, regime and gates as built and tested in block
000 (`STATUS.md`).*

---

## 1. Why

Two premises of the label design are untested and cheap to test before any direction, probe-based label or reader is
built.

1. **Do single-cell labels vary within a position?** W (SPEC §8 A2) is estimated within family × position. If a
   single (position, layer) cell's full-cell patch share is high for every pair at the emission cells and ~0
   everywhere else, then "used" and "not needed" collapse onto position and W cannot be estimated however good the
   labels are (reading report §5.2).
2. **Does brew supply a not-needed class?** AMENDMENT_01 §2 makes brew's off-trajectory colours at the emission cells
   the candidate *decodable-not-needed* content. That candidate exists only if those colours are decodable there.

The outcome decides whether label construction (002) proceeds on brew + chain, and whether the RHUT tagged-selection
fallback has to be designed next.

## 2. Materials

**Families and splits (fixed here, before any outcome is seen).**

| Family | Unit of split | Dev / test | Notes |
|---|---|---|---|
| Brew | table (each of the 100 items has its own table; partners share their member's table) | 50 / 50 tables, stratified by z (12 colours; seed 20261002) | Items touched by block-000 engineering (`brew-s2-r10d2-0000`, `-0001`) are forced into dev and listed in the manifest |
| Chain | step sequence (the 28 single-digit-intermediate items form 20 step-sequence groups) | 10 / 10 groups, stratified by depth (d2 / d3) | Grouping is required: items with the same steps produce identical prompts (15 cross-item collisions, e.g. partner `chain-w2-d3p-0408__12` == member `chain-w2-d3p-0422`). Pairs are de-duplicated as unordered {prompt, prompt} sets. `chain-w2-d3p-0408` (touched in block 000) is forced into dev |

Halves A/B (AMENDMENT_01 §3) are not needed in 001: nothing is fitted except decodability probes, which use
item-grouped cross-validation (§3.3).

**Pairs.** Brew: same-table partners (361 candidates over 100 items, all token-aligned, differing only at the
start-colour token). Chain: same-steps partners (110 candidates over 25 items before de-duplication; all
token-aligned). Every candidate (dev **and** test) is gated in stage `gate` (greedy correct and ≥ 9/10 at T = 0.8, both
members), so admission counts and the difficulty distribution (per-item candidate counts, pass rate by z and by table)
are reported for the whole bank. Only dev pairs enter the measurements below. The gate is label-free and reader-free.

**Cells.**
- Brew: the 24 WSB pinned positions (stir ×6, start ×6, question ×9, emit_asst, emit_think, emit_colon), re-asserted
  per render with the pinned tokenizer.
- Chain: the question span (" What" → final token, 25 positions at T = 78) including the final cell ("\n\n" after
  `</think>`).
- Layers 40, 42, 44 (block outputs). Controls add early layers (§3.4).

**Regime.** RHUT float32-residual from block 0 (`engine.Engine(fp32_from=0)`), float32 logits, every comparison at a
fixed sequence length with partner sources captured at the matching length (block-000 findings). See §9 for the open
regime question.

## 3. Estimands and measurements

Notation: member x (answer a), partner x′ (answer a′), margin m(·) = LSE_{s∈S(a′)} log P(s|·) − LSE_{s∈S(a)} log P(s|·)
with full-sequence teacher forcing over every token of every accepted spelling (brew: " c", " C"; chain: str(n)).
G = m(x′) − m(x) (clean, both renders).

### 3.1 Full-cell patch share (primary)

For every dev pair, cell c and layer ℓ: run x with its block-ℓ output at position c replaced by x′'s (captured at the
same sequence length), Δm_c = m(x with patch) − m(x), share S_c = Δm_c / G. Direction m ← p for every pair; the reverse
direction (p ← m, share = −Δm/G) on a fixed random 30% of dev pairs as a symmetry check.

Recorded per (pair, cell, layer): Δm, S, the damage readouts (Δlog P(a), Δlog P(a′), top-1 retention at the
teacher-forced prompt positions after c, off-pair mass at the answer position), and whether the write fired once.

**Within-position variation** (decision rule 1). For each (family, position, layer) with n ≥ 20 dev pairs:
- the fraction of pairs with S ≥ τ (0.10) and the fraction with S < ε′ (0.05) (provisional AMENDMENT_01 thresholds);
- the between-pair SD of S with a pair-level cluster bootstrap CI (clusters = items);
- the numerical noise floor at that cell (§3.5), so that variation is not rounding.

A (position, layer) **shows within-position variation** if ≥ 10% of its pairs have S ≥ τ **and** ≥ 10% have
S < ε′, and the between-pair SD exceeds 3× the noise-floor SD.

### 3.2 Span patches (context only)

The same pairs, with whole regions replaced at all three layers jointly (`SpanWriter`): brew regions start, question,
emission (3 cells), and start + question + emission; chain: input digits, question span. Reported next to the
single-cell shares (how much of the region effect a single cell carries). Never used for a label.

### 3.3 Decodability (brew emission cells; probes only, no directions used anywhere else)

Activations: stock-model block outputs (the regime of §2) at emit_asst, emit_think, emit_colon × L40/42/44, for every
dev render (members and all partner candidates; ~230 renders over 50 tables).

Targets:
- **z** (12-way), and for comparison the **start** colour and the **answer** (12-way each; the {input, z, answer}
  confusion of AMENDMENT_01 §3).
- **Off-trajectory presence**, one binary target per colour c: positive = c is in this render's table and off its
  trajectory; negative = c is absent from the table (2 absent colours per table); renders where c is on the
  trajectory are excluded for that c. Pooled over the 12 colours by mean AUC.
- Secondary (descriptive): the answer's successor table[a] when it is off-trajectory (look-ahead content), 12-way.

Probe: L2-regularised logistic regression on standardised features after PCA to 256 dims (fitted inside each training
fold); C from {1e-3, 1e-2, 1e-1, 1} by inner grouped CV; outer 5-fold CV **grouped by table**.

Controls:
- **Control task (Hewitt–Liang).** Each table gets a random label (consistent across its renders), and the identical
  pipeline is fitted. Selectivity = real − control. Under table-grouped folds the control should sit at chance; a
  control above chance flags leakage.
- **Null.** 200 table-level label permutations (preserving the group structure) give the null of the pooled statistic.

**Decodable** (for decision rule 2): the pooled off-trajectory presence AUC exceeds its permutation null at
BH-FDR q = 0.05 across the 9 (cell × layer) tests **and** its selectivity is ≥ 0.10, at one or more emission cells ×
layers. The same rule is reported for z, start and answer.

### 3.4 Controls

| Control | Cell | Prediction | What failure means |
|---|---|---|---|
| Embedding patch (exact anchor) | start-colour token, input embedding | partner logits reproduced bitwise | engine broken |
| **Positive** (rule 3) | start-colour token (brew) / start digit (chain), block outputs L0, L1, L2, L4 | S ≥ 0.9 at L0 | engine broken (rule 3) if S < 0.5 at L0 on most pairs; 0.5–0.9 is a model fact (block 0 already mixes the colour into later positions) and is reported |
| **Negative** | brew stir cells (before the start sentence in the bank prompts) | S = 0 bitwise (identical prefix, identical length) | engine broken |
| Self-patch no-op | emit_colon, each layer | m identical bitwise | engine broken |

### 3.5 Noise floor

On a fixed random 20 dev pairs per family: 8 random perturbations of relative size 1e-4 at emit_colon (brew) / the final
cell (chain) at L42, giving |Δm| (block 000 measured up to 0.05–0.24 nats on two brew pairs, 0.5–2.7% of G). Reported per
pair and as a fraction of G. Feeds the within-position variation rule (§3.1) and the regime question (§9).

## 4. Gates (before any outcome is read)

1. Span/rendering: every render passes the WSB position assertions; every pair is token-aligned with the single allowed
   difference.
2. Same-source == clean bitwise: self-patch, stir-cell and embedding-patch controls (§3.4).
3. Fidelity: n/a for full-cell patches (exact writes; ρ = κ = 1 recorded).
4. Damage: per patch; a family whose emission-cell patches raise the off-pair mass by > 0.10 in > 20% of pairs has those
   cells reported as damage-dominated.
5. Sequence-length discipline: asserted in code (sources captured per continuation).

A failed gate is repaired and the stage re-run; nothing scored is deleted.

## 5. Predictions and priors

*Written 2026-10-02 before any 001 data. Already seen, and disclosed because they inform these priors: the block-000
regime check on one dev pair (`brew-s2-r10d2-0001__blue`, G ≈ 10.5 nats): full-cell shares emit_colon L44 ≈ 0.42–0.45,
emit_colon L40 ≈ 0.21–0.23, start token L40 ≈ 0.05; the reader smoke test on `brew-s2-r10d2-0000` emit_colon (no lens
names a colour in its top 10 at L40–44, colours appear from L56; the NLA at L42 named z in 4/4 samples; the Oracle at L40
named the start and z colours); gate failures for 2 of the first 3 brew candidates.*

| # | Prediction | Prior |
|---|---|---|
| Q1 | Gate pass rate ≥ 40% of brew candidates and ≥ 40% of chain candidates | 55% |
| Q2 | Positive control: brew start-token S ≥ 0.9 at L0 in ≥ 90% of dev pairs | 75% |
| Q3 | Brew start-token S at L40–44 has median ≤ 0.10 (the colour has been moved on) | 75% |
| Q4 | Brew emit_colon median S increases L40 < L42 < L44, with L44 median ≥ 0.3 | 65% |
| Q5 | Every brew question-span cell has median S ≤ 0.10 at L40–44 | 60% |
| Q6 | Decision rule 1 is met for brew (≥ 1 emission cell × layer with within-position variation) | 50% |
| Q7 | Decision rule 1 is met for chain (final cell or question span) | 40% |
| Q8 | z decodable at emit_colon (12-way, by §3.3's rule) at all three layers | 70% |
| Q9 | Off-trajectory presence decodable at ≥ 1 emission cell × layer (decision rule 2 met) | 55% |
| Q10 | The answer is more decodable than z at emit_colon L44 (higher selectivity) | 65% |
| Q11 | Damage: emission-cell patches raise the off-pair mass by > 0.10 in < 10% of brew pairs | 75% |

## 6. Decision rules

1. **If no family shows leverage variation within a position** (§3.1 criterion), stop and rethink before building
   labels. Report the profile; the next step is a redesign discussion, not 002.
2. **If the off-trajectory colours are not decodable at the emission cells** (§3.3 rule), brew has no not-needed class;
   the RHUT tagged-selection fallback is designed next.
3. **If the positive control fails** (§3.4: S < 0.5 at L0 on most pairs, or any exact-anchor/bitwise control fails), the
   engine is broken; fix it first and re-run.
4. *(added)* If fewer than 20 brew or 15 chain dev pairs pass the gate, report and stop for the PI (power).
5. *(added)* τ and ε′ may be revised once from this profile (dev only) and frozen in AMENDMENT_02 before any test-split
   label is computed.

## 7. Stop rules

Stop and report (no silent continuation) if: any gate in §4 fails twice after repair; the gate admits < 20 brew dev
pairs; the noise floor exceeds 0.05 of G in more than a quarter of the noise-floor pairs (then the regime question of §9
blocks 002).

## 8. Budget (from `results/000_setup/analysis/timing_smoke.json`)

Measured on the RTX PRO 6000 (96 GB), RHUT regime: one forward ≈ 0.23 s for both brew (T = 165) and chain (T = 78);
gate ≈ 5.4 s (brew) / 4.4 s (chain) per pair; probe capture ≈ 0.23 s per render.

| Stage | Forwards | Wall-clock |
|---|---|---|
| Gate, all candidates (361 brew + ≤ 110 chain) | — | ≈ 32 + 8 min |
| Brew profile, ~90 dev pairs × (72 cells + 14 controls + 6 spans + 3 clean/capture) | ≈ 8.5 k | ≈ 35 min |
| Brew reverse direction, 30% of pairs | ≈ 2 k | ≈ 8 min |
| Chain profile, ~25 dev pairs × (75 cells + 10 controls + 2 spans), × ≤ 3 for two-digit answers | ≈ 2–7 k | ≈ 10–27 min |
| Noise floor, 40 pairs × 8 | 320 | ≈ 1.5 min |
| Decodability captures (~230 brew dev renders) | 230 | ≈ 1 min |
| Probes (CPU; 12 colours + 3 12-way targets × 9 cells, 5 folds, 200 permutations) | — | ≈ 20–40 min CPU |
| **Total** | | **≈ 1.7–2.2 GPU-h + < 1 CPU-h** |

## 9. Open questions for the PI

1. **The not-needed status of off-trajectory colours.** The full-cell share at an emission cell measures the *pair*
   difference (start, z, answer), which is expected to be large there; it cannot certify that the off-trajectory
   *content* is not needed. Their not-needed status currently rests on task construction (the answer is invariant to
   off-trajectory rows). Option: add an **off-trajectory edit arm** (partner = same item with one off-trajectory row's
   result colour changed, single token, token-aligned; answer unchanged) and record |Δlog P(a)| and the answer-
   distribution KL of the emission-cell full-cell patch (G = 0, so no share). Cost ≈ 10 min. Needs your call because it
   implies a label criterion other than ε′ for content that is not the pair's latent (AMENDMENT_02).
2. **Regime.** By the 5% rule RHUT stays the default (max 3.3% of G). But its noise floor is absolute (0.05–0.24 nats on
   the two pairs measured, vs 0.0002 under SEC), so on pairs with G below ≈ 7 nats a 3.3%-sized discrepancy would exceed
   5% of G. SEC's upper stack fits on this GPU (72 GB resident, replay 0.09 s). Options: (a) keep RHUT and report the §3.5
   noise floor; (b) switch the label stack to SEC for layers ≥ 40 (early-layer controls stay in the bf16 lower stack).
   I recommend (b) for 002; 001 is unaffected either way.
3. **Dev/test sizes.** Chain has 20 step-sequence groups (≈ 10 per split). If too few chain dev pairs pass the gate,
   should chain use 70/30 dev/test instead?
