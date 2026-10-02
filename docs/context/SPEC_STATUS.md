# SPEC status — what is done and what is not (2026-10-02)

The whole project spec (`docs/context/SPEC.md`), item by item, against what exists in the repo. Overrides from
`docs/context/AMENDMENT_01.md` (A01, conflicts C1–C23, errata E1–E4) and later PI decisions are applied, so "the spec
says" means the spec as amended. Status words:

- **done**: built and run, with evidence in the repo;
- **built**: code exists and is tested, not yet run for its purpose;
- **partial**: some of it exists;
- **not started**;
- **dropped / out / deferred**: by a recorded decision.

Nothing in the adjudication or agreement analyses (§6–§8 of the spec) has produced a result yet. **No label and no
reader score exists. Every run so far is on the dev split.**

## Spec §0–§2: aims, North Star, proxy

| Aim | Status |
|---|---|
| Agreement: measure reader agreement at four levels | **not started** (only descriptive mention rates, 001 §8) |
| Adjudication: score readers against reader-free causal labels | **not started**: no labels yet (design 002) |
| Decompose agreement into shared signal / shared error / shared prompt inference | **not started**; 001 §8 has descriptive hints only |
| Proxy: sensitivity to used content, hit rate on decodable-not-needed content, conditional agreement, matched cells and claim budgets | **not started** |

## Spec §3: questions Q1–Q10

All open. Descriptive, exploratory hints from 001 (10 brew tables; not answers):

| Q | Hint so far |
|---|---|
| Q3 (activation vs prior) | Verbalizer off-path mentions sit at shuffled rates; z mentions above them; strong per-reader colour preferences (001 §8) |
| Q8 (evidence tags for text-reader claims) | Some claims are activation-sensitive (within-table start-swap tracking), many name z in the wrong role (001 §8) |
| Q9 (NLA hallucination) | FVE is dominated by the shared template, with a small source preference (244/360); best-of-N untested |
| Q1, Q2, Q4–Q7, Q10 | nothing |

## Spec §4: model and readers

| Item | Status | Evidence / decision |
|---|---|---|
| Qwen3.6-27B @ 6a9e13bd, bf16, eager | **done** | `registry.json`, `results/000_setup/analysis/env_check.json` |
| Logit lens | **done** (smoke, 001 snapshot) | `readers.py`, `results/000_setup/analysis/reader_smoke.json` |
| J_NP n=1000, WSB cosine + standard readout | **done** | same |
| J_CB, R_CB (n=25) | **done** | same |
| Oracle Lens RL (`olens_s3d_rl600`) | **done** on-contract at L40/L44 (A01 §4) | 001 snapshot. **No L42 arm yet**; A01 allows an off-contract L42 arm reported separately. A matched-layer Oracle L42 arm was raised in the external audit and is a pending decision |
| Oracle AR reconstructor | **limited**: its trained whiteners are private (A01 §4) | not used |
| NLA RL L42 + reconstructor | **done**: step-matched `av_rl_lora_step400` + `rl_critic_step400` (A01 C17) | smoke + 001 snapshot; AR positive controls pass |
| Skip-lens | **dropped** (A01 C16) | trained at L62; reinstatable on-contract |
| SAE / BSF at L42 | **deferred** by the PI | candidate `ceselder/qwen36-27b-bsf-l42-1b`, no auto-interp labels |
| Optional: modulation lens, phrase-J | **dropped** (PI, 2026-10-02) | — |
| Optional: template lens | **deferred** (PI, 2026-10-02) | exists at the pinned `camilablank/workspace-lenses` @ d740106d; vocabulary lacks teal |
| Baseline: lucky guess | **not started** | — |
| Baseline: prompt-only (= stock Qwen3.6-27B, A01 C18) | **done (002)** as a task baseline: z 1.00, start 1.00, answer 0.89 on 231 dev renders | `results/002_claim_scoring/prompt_only/` |
| Baseline: no-activation | **done** for the verbalizers (001 snapshot) | `analysis/snapshot.json` |
| Baseline: shuffled-activation | **done** (001 snapshot, one derangement over 10 tables) | same |
| 8 samples per cell; self-consistency variable | **partial**: 4 samples (Oracle contract, NLA), 1 (Oracle WSB grid) | — |
| vLLM acceleration for readers | **parked** (PI: optional, slot in later) | engine fails on sm_120 (FlashInfer); `scripts/000_setup/vllm_nla_check.py` |

## Spec §5: the bank

| Family | Status | Evidence |
|---|---|---|
| Brew | **done**: token-aligned start partners built (361 candidates), gated: dev 82/183 pairs (33 tables), test 77/178 (35 tables) | `partners.py`, `analysis/gate_summary.json` |
| Chain | **done** (single-digit intermediates only): dev 48/53 pairs (9 step groups), test 37/46 | same |
| Relational | **partial**: 50 token-aligned a/b pairs built and aligned; **not gated** | `partners.py`, tests |
| Two-hop (RHUT + WSB multihop) | **out for now**; multihop later (A01 C1) | — |
| Arithmetic | **dropped** (A01 §1) | — |
| Role-bound | **agreement only**, nothing built | — |
| Commitment (new, ~60 pairs) | **out for now** (A01 C1) | — |
| Decodable-not-needed family | **partial**: replaced by brew off-trajectory substitutions (A01 C4): 100/100 items admit one, dev 43/50 gated, behaviour and transplant measured (001 §6). Decodability provisional (probe controls). RHUT tagged-selection fallback **not built** (only if needed) | `partners.brew_substitution`, 001 §6–§7 |
| Safety families (descriptive) | **out for now** (A01 C1) | — |
| Admission gate (≥ 0.9 both members) | **done**: greedy correct and ≥ 9/10 at T = 0.8 | `gates.py`, 001 §2 |
| Token alignment at labelled cells | **done**, asserted on every render | `wsb.py`, `partners.token_align` |
| Two-hop admission rule | n/a (two-hop out) | — |
| Cells | **done** as unit = token position × layer (A01 C5); brew 24 pinned cells, chain 25-position question span | `cells.py` |
| Layers | **partial**: label layers 40/42/44 (+56/60 in 001) per A01 C6; lens/Oracle grid L20–60 (amendment A1) | design 001 |
| Splits: dev/test | **done**: brew by table, chain by step group, relational by name set | `splits.py` |
| Splits: A/B halves for cross-fitting (A01 C7) | **built**, not used yet (no labels) | `splits.py` |
| Matched used / not-needed contrasts within family × position | **not started** (brew emission cells are the candidate, 001 §13) | — |

## Spec §6: causal labels (no reader involved)

| Step | Status | Evidence |
|---|---|---|
| Margin m: full-sequence, LSE over spellings, every token scored | **done** | `margin.py`, tests |
| Precision regime | **done**: SEC true-fp32 replay for L ≥ 40 (errata E4, design 001 v3); RHUT regime as the cross-check | `regime.py`, `engine.py`, `regime_check.json` |
| Full-cell patch share S (the reference, E2) | **done** for brew and chain dev (001 §4–§5) | `outputs/001_leverage_profile/run/profile_*.jsonl` |
| 1. Reader-blind mean-difference direction per contrast | **not started** (design 002) | — |
| 2. 1-D swap along û | **built** (`engine.Swap1D`, `regime.swap_fn`; write tested) | not run for labels |
| 3. Demonstrated influence: S ≥ τ and above the norm-matched random-direction null, BH-FDR (A01 C9) | **not started** | — |
| 4. Necessity: ablate to the fitting-half mean, random-direction ablation control (A01 C10) | **built** (`engine.AblateToMean1D`), not run | — |
| 5. Decodable: cross-fitted probe + Hewitt–Liang control (A01 C11) | **partial**: 001 probes run (z, start, answer, membership). **Needs repair:** the membership control sits far from chance (median AUC 0.58, max 0.84), and the permutation null reuses C from the real fit. No item-level labels yet | `analysis/probes.json`, 001 §7, §15 |
| 6. States (used / decodable-not-needed / absent / unresolved) | **not started** | — |
| 7. Per consumer | n/a (one consumer per WSB family) | — |
| "Pair difference used" naming + four-way {input, z, answer, other} confusion; decorrelated direction (A01 C15) | **not started** | — |
| DAS as a second label method | **not started** | — |
| Share vs flips | **partial**: shares recorded; top-1 retention recorded for damage; answer flips not tabulated | — |
| Gate: write fidelity ρ ∈ [0.9, 1.1], κ ≥ 0.99, fp32 hook | **built** (`gates.fidelity_gate`, tested); not exercised by a label run | — |
| Gate: label reliability ≥ 0.7 | **not started** (no labels) | — |
| Gate: unresolved fraction per family | **not started** | — |
| Damage readouts (A01 C14) | **partial**: off-pair mass, Δlog P(a/a′), top-1 retention recorded; the flag fires at `:` (46–67%). The full answer-colour distribution is not logged, so the flag cannot be decomposed (001 §4) | — |
| Thresholds τ = 0.10, ε′ = 0.05, substitution bound | **provisional**; freeze proposed for AMENDMENT_02 (001 §13) | — |
| Power calculation, minimum cells per stratum (A01 §5) | **not started** | — |
| Engine checks (self-patch, embedding anchor, prefix, stir no-op) | **done**: 100% bitwise on every pair | 001 §3 |

## Spec §7: scoring readers fairly

| Item | Status |
|---|---|
| Matched claim budget (k = 4; also 1, 8, 16) | **not started** |
| Token readers: closed-set reciprocal rank; top-25 → summariser | **not started** |
| Text readers: atomic claims ranked by recurrence; judge with "cannot tell" | **partial (002)**: role extraction per readout with none / cannot tell / conflicting; not atomic claims ranked by recurrence |
| Closed-set candidates = gold z + input + answer + foils; token-reader hit rule (A01 C19) | **done on dev (002)**: lens closed-set hits for all 48 tables, with within-table tracking (post hoc) |
| Open-set pairwise claim matching | **not started** |
| Planted-concept positive control (sensitivity curves) | **not started** |
| Judge models | **chosen:** Gemini 3.8 Flash via OpenRouter (WSB's judge), PI 2026-10-02 |
| Role-aware scoring (mention vs correct role vs contradicted, abstention) | **done on dev (002)**: judge = Gemini 3.8 Flash (WSB's), closed candidate set per table, validated against blind author codes (κ 0.67 / 0.81); `docs/reports/002_claim_scoring.md` |

## Spec §8: analyses

| Analysis | Status |
|---|---|
| A1 agreement: concept, open-set, vector and behavioural levels; excess agreement within label; shuffle-sensitive Δκ; test–retest; P(used \| k families); Dawid–Skene; atlas | **not started**; descriptive precursors in 001 §8 and the 002 overlap table (mention vs claim; shared wrong values) |
| A2 causal accuracy: Sens, Spec, W | **not started** |
| A3 ensemble, ΔAUC | **not started** |
| A4 evidence tags for reader-only claims | **not started** (activation-sensitivity hint in 001 §8) |
| A5 NLA mechanism: Δ(c), discriminability, geometric faithfulness, FVE selection experiment | **not started** (FVE source preference measured descriptively) |
| A6 commitment | **out** (commitment family out) |

## Spec §9: registered predictions P1–P10

None testable yet (they need labels and reader scores). The P5 prior update (A01 §5, C21) is **pending the PI's value**.
Design 001 registered its own predictions Q1–Q14; outcomes are in report 001 §10.

## Spec §12: order of work

| Step | Status |
|---|---|
| 0. Rotated-row null gate (optional) | **not started** |
| 1. Plumbing: reproduce WSB J numbers (brew ≈ .86) | **partial**: one item (0000) reproduces WSB's pass rule for the three lenses; the full brew reproduction was not run |
| 1. Plumbing: verbalizers end to end | **done** (smoke on one cell; 001 snapshot on 10 tables) |
| 1. Plumbing: reconstructor availability | **done**: the NLA AR is released and works; the Oracle AR's whiteners are private |
| 2. Planted-concept sensitivity curves | **not started** |
| 3. End-to-end slice: brew + matched not-needed at L42, labels, reliability, W | **not started**. 001 (leverage profile + decodability, reader-free) is its precursor per A01 C22 |
| 4. Open-set agreement atlas | **not started** |
| 5. Bank construction for remaining families | **partial** (relational built, ungated); the rest out |
| 6. Labels + reliability gate | **not started** |
| 7. Reader runs at matched cells with controls | **partial** (001 snapshot, descriptive, 10 tables) |
| 8. Analyses A1–A6 | **not started** |
| Stop-and-redesign checks (bench reproduction, planted curves, reliability, slice) | **not yet testable** |

## Spec §13: open questions

| Question | Answer so far |
|---|---|
| 1. Does the NLA ship its reconstructor? | Yes; step-matched pair in use |
| 2. SAE/BSF auto-interp labels? | The candidate BSF has none; SAE deferred |
| 3. Phrase-J checkpoint? | Dropped |
| 4. Judge models? | Gemini 3.8 Flash (as in WSB), via OpenRouter |

## Infrastructure (not in the spec, needed by all of it): done

- Pinned environment and rebuild script; 29/29 artifact hashes match.
- WSB renders, byte-identical to WSB's producer.
- Partners, substitution builder, gate, cells and splits.
- Margin; RHUT and SEC engines with the integrity guard; gates; stats; figures; reader wrappers with the import boundary.
- 78 tests.
- Results page (`scripts/000_setup/results_page/`, private artifact).
- Experiments: block 000 (`docs/reports/000_reading_report.md`, STATUS.md) and design/report 001.

## Open PI decisions (as of 2026-10-02)

1. Review report 001 (revised after the external audit, §15); AMENDMENT_02 contents (001 §13).
2. Not-needed reader scoring at L60 only, or also L40–44 via the directional test.
3. Oracle L42 arm (matched layer with the NLA), off-contract, reported separately (A01 §4).
4. HF dataset name for the raw outputs (`outputs/` is not persistent and not mirrored).
5. P5 prior value (A01 §5).
6. SAE (deferred); template lens (deferred).
7. Paper PDFs: gitignored while the repo was public; the repo is now private.
