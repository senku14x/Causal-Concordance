# AMENDMENT 01 — decisions after the reading pass (2026-10-02)

*Records the PI's decisions of 2026-10-02 (taken on `docs/reports/000_reading_report.md`) and every conflict with
`docs/context/SPEC.md` (cited by line; SPEC.md is byte-identical to the spec the reading report cites). Where this file
and the spec disagree, this file wins. Nothing scientific had been run when it was written.*

Thresholds marked *provisional* may be revised **once**, on dev material, from the leverage profile (design 001), and
are then frozen in AMENDMENT_02 before any test-split label is computed.

---

## 1. Tasks (families)

**WorkspaceBench only** for now. Bank files are pinned by sha256 in `scripts/common/registry.json` (WSB commit 92d763e).

| Family | Status | Construction |
|---|---|---|
| **Brew** | core | Partner = same table, different start colour. Constraints: z′ ≠ z, a′ ≠ a, z′ ∉ {a, a′}, z ∉ {a′}. The start colour is a single token in " X."; both renders have identical token counts and differ only at the start-colour token. Gate both members: greedy correct and ≥ 9/10 correct at T = 0.8. Report candidates built, candidates passing, and the difficulty distribution (do not keep only easy tables). The 24 WSB pinned positions are re-verified with the pinned tokenizer. Stir cells are kept. The runner-up caveat is carried (WSB: brew cannot separate the intermediate from the runner-up answer; prompt-only scores 0.96). |
| **Chain** | core | Number tokenisation checked first (digits are single tokens, " 23" → [" ", "2", "3"]). Only items whose intermediates are single tokens (single digits) are kept. Partner = same steps, different start number with the same digit count. Same constraints and gate as brew, plus WSB's own chain constraints applied to the partner. Read cell = the final token, with the question span (" What" → end) used for the leverage profile. |
| **Relational** | core subset | Only the bank's token-aligned a/b pairs, gated. Demoted to secondary if fewer than ~30 pairs survive the gate. |
| **Multihop** | later | Cue-swap pairs, plain render; never pooled with RHUT two-hop. |
| **Role-bound** | agreement only | No labels. |
| **Arithmetic** | not now | Fraction variants dropped. |

## 2. The not-needed contrast

Brew's off-trajectory colours at the emission cells are the candidate *decodable-not-needed* content. Design 001
establishes whether they are decodable there (item-grouped linear probe + Hewitt–Liang control task). If they are not,
the fallback is RHUT's tagged-selection organism (pointed vs unpointed word at the question/decision cells). The
fallback is **not built** unless 001 says so.

## 3. Labels

- **Unit.** One token position at one layer. Span patches are context only and never label anything.
- **Used.** S ≥ τ **and** S above the norm-matched random-direction null after Benjamini–Hochberg FDR across the family's cells. τ = 0.10 (*provisional*).
- **Not needed.** Decodable **and** full-cell patch share < ε′, with ε′ < τ; ε′ = 0.05 (*provisional*). Cells in between are **unresolved**.
- **Illusion flag.** Every cell where the 1-D share exceeds the full-cell share is flagged (Makelov signature).
- **Halves (cross-fitting).** By items: brew by table, stratified by z; chain by start number, stratified by z; relational by name set. Cross-fit A→B and B→A so every item is labelled. Label reliability (agreement between labels computed from two disjoint fitting subsets for the same items) ≥ 0.7 before any reader is scored.
- **Naming.** The label is called **"pair difference used"**. Every used cell gets a four-way confusion of what the direction tracks: {input, z, answer, other}. A decorrelated direction (z varying while input and answer are decorrelated from it) is fitted where the family supports it.
- **Necessity.** Pooled mean = mean of û·h over the fitting half at that cell; a random-direction ablation control is run alongside; "used + necessary" requires N_i above that control's 95th percentile. INLP/LEACE-style removal later.
- **Decodability.** Regularised linear probe; folds grouped by item and table; Hewitt–Liang control task (selectivity). Binary z-vs-z′ probes are acceptable. The planted positive control must not use the probe's own direction.
- **Damage.** Δlog P(a), Δlog P(a′) and top-1 retention on non-answer positions after every edit (plus the off-pair mass at the answer position); broad damage → unresolved.
- **Precision.** Default: RHUT's float32-residual regime on transformers 5.15.0. Verify on ~5 edits against SEC's true-float32 replay; switch if Δm differs by > 5% of G. (Done 2026-10-02: max 3.3% of G → RHUT kept; see STATUS.md, `results/000_setup/analysis/regime_check.json`, and the noise-floor note there.)

## 4. Readers

- **Oracle Lens:** on-contract at **L40 and L44**, labelled at those layers; an off-contract L42 arm is reported separately and never pooled. AR analyses for the Oracle are limited (its trained whiteners are private).
- **NLA:** the step-matched `ceselder/qwen3.6-27b-nla-L42` pair (`av_rl_lora_step400` + `rl_critic_step400`).
- **Skip-lens:** dropped.
- **SAE:** *[to be filled in by the PI]*; scored with non-J auto-interpretation labels only (J projection kept as a diagnostic).
- **J_NP, J_CB, R_CB, logit lens:** as in the spec (J_NP in both the WSB cosine readout and the standard readout).
- **Baselines from day one:** prompt-only, no-activation, shuffled-activation. **Prompt-only = stock Qwen3.6-27B.**
- **Closed-set candidates:** gold z, the input, the answer, and same-category foils. **Token-reader hit** = z ranked first among the candidates **and** its score above that reader's no-activation run.

## 5. Also accepted

- Family × position fixed effects (A3 and pooling).
- A power calculation with a pre-registered minimum number of cells per stratum below which W is not reported.
- SAE with non-J labels (§4).
- A5: add the "Δ(c) > 0 for the hallucinated claim" branch (co-adapted codes); use the step-matched NLA; a **dated P5 prior update** (value to be set by the PI; the reading report §4.5.6 evidence points away from Δ(c) ≈ 0).
- Brew is **not** the label pipeline's positive control. The pipeline gets its own controls: the start-colour token at an early layer (full-cell share ≈ 1 by construction) and the stir cells (≈ 0).

## 6. Conflicts with SPEC.md

| # | SPEC | Says | Now |
|---|---|---|---|
| C1 | L107–115 (§5 table) | Bank = brew, RHUT two-hop + WSB multihop (~100), chain/arithmetic, role-bound/relational, a new commitment family (~60 pairs), RHUT organisms as the decodable-not-needed family, descriptive safety families | WorkspaceBench only (§1). RHUT two-hop, the commitment family, the RHUT organisms and the safety families are out for now; multihop later; arithmetic dropped; role-bound agreement only |
| C2 | L109 | Brew is the "positive-control family" with "token-aligned donors" | Brew is not the pipeline's positive control (§5). Token-aligned partners did not exist; they are constructed (§1) |
| C3 | L112 | Role-bound / relational "ab/ba pairs" | Relational: only token-aligned a/b pairs; role-bound agreement only |
| C4 | L114, L140–143 (matched contrasts), L366 (slice) | Decodable-not-needed content comes from RHUT organisms; the slice pairs brew with RHUT maintain-vs-ignore | Candidate = brew off-trajectory colours at the emission cells, established in 001; RHUT tagged selection only as a fallback (§2) |
| C5 | L151 | Cells = source, carriers, decision | Unit = one token position at one layer; span patches context only (§3) |
| C6 | L153–157 | Core layer L42 for every reader; J/R/logit also at 40, 48, 56 | Oracle on-contract at L40/L44 with labels there; off-contract L42 arm separate (§4). Design 001 profiles layers 40/42/44 |
| C7 | L159–161 | Template halves A/B: fit on A, label on B | Item halves (table / start number / name set, stratified by z), cross-fitted so every item is labelled (§3) |
| C8 | L172 | "run the rest of the model in fp32" | Default is RHUT's float32-residual regime (bf16 matmuls); SEC true-fp32 is the cross-check, switch rule 5% of G (§3) |
| C9 | L177 | Used = S above the 95th percentile of norm-matched random directions | Used = S ≥ τ **and** above that null after BH-FDR across the family's cells (§3) |
| C10 | L178 | Necessity: ablate to "the pooled mean" (undefined) | Pooled mean = fitting-half mean of û·h at the cell, plus a random-direction ablation control and a 95th-percentile rule (§3) |
| C11 | L179 | Decodable: probe folds split by template; shuffled-label null | Folds grouped by item and table; Hewitt–Liang control task; binary probes allowed (§3) |
| C12 | L182 | Not needed: full-cell share < ε (e.g. 0.1) | < ε′ = 0.05 (provisional) with ε′ < τ; the band between is unresolved (§3) |
| C13 | L183 | Absent: positive control plants α·û_c | The planted control must not use the probe's direction (§3) |
| C14 | §6 (no gate) | No damage gate | Damage readouts on every edit; broad damage → unresolved (§3) |
| C15 | §6 | Label = "z used" | Label = "pair difference used", with the four-way {input, z, answer, other} confusion and a decorrelated direction where possible (§3) |
| C16 | L89–91 | Skip-lens fed L42 is a reader arm | Dropped (§4) |
| C17 | L88, L383 | NLA "with its reconstructor if released" | Released; the step-matched nla-L42 pair is used (§4) |
| C18 | L96 | Prompt-only = "an LLM reads the prefix and guesses" | Prompt-only = stock Qwen3.6-27B (§4) |
| C19 | L209 | Closed-set candidates = gold z + 4 same-category distractors | Gold z + input + answer + same-category foils; token-reader hit rule defined (§4) |
| C20 | L277 | A3 ensemble with family fixed effects | Family × position fixed effects (§5) |
| C21 | L292–298, L314 | A5 outcomes assume Δ(c) ≈ 0 (P5, prior 55%) | Δ(c) > 0 branch added; P5 prior updated, dated (§5; value pending) |
| C22 | L362–366 (order of work) | Step 1 reproduces WSB J numbers and runs verbalizers on 3 items; step 3 is the brew + maintain-vs-ignore slice | Block 000 (setup + one-cell reader smoke) and design 001 (reader-free leverage profile + decodability) come first; the slice's not-needed arm is decided by 001 |
| C23 | L127 | Question-turn ladder figures (install vs removal) | Factual correction recorded in the reading report §4.1.4 / §5.10 (no design consequence yet) |

## 7. Implementation decisions taken in block 000 (not in the spec)

- **Regime origin.** The float32 residual starts at block 0 (`engine.Engine(fp32_from=0)`), not RHUT's block 35, so that early-layer controls (§5) and direction-level writes at any layer are carried in float32. Regime check: from-0 vs SEC 3.3% of G, from-35 vs SEC 2.3% (both < 5%).
- **Logits in float32.** Margins are computed from float32 logits (`final_norm(h) @ W_U^T` in float32), not from the bf16 lm_head output, whose spacing at |logit| ≈ 20 is 0.125.
- **Fixed-length comparisons.** Appending tokens changes prompt-position activations (rounding onset at block 3, the first full-attention block; not a causal leak). Every clean/edited comparison is made at one sequence length, and patch sources are captured at the matching length (`margin.score` builds writes per continuation).
- **Reader activations** are captured from the stock bf16 forward (WSB's `Backend.capture` convention), not from the label regime.
