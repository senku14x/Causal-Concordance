# 005 — Planted concepts: J-part vs non-J part, for the model and every reader — design v1 (DRAFT for PI review)

*Status: draft, committed for the PI's review only. **Nothing here is authorised to run.** The smoke and the run each need the PI's explicit go.*
- Sources: PI feedback 2026-10-04 (§2 c1/c2, §3.6–3.8, §4 doubts 10, 11, 13, 14, 16; §5 item 8); review `docs/context/SPEC_v2_review_2026-10-04.md` (c) c1, c2.
- Paper: Gurnee et al. 2026 (arXiv 2607.15495), §3.1 Figs. 7–8, App. A.8, A.9.2.
- The official protocol file is `anthropics/jacobian-lens` `data/experiments/verbal-introspection.json` (`/workspace/jacobian-lens`).
- **One run, not two stages** (PI 3.6). The stage-1 rule survives as analysis gate G2 (§9).
- Every `[PI]` field is left for Vishesh, the prediction priors included.

## 1. Question

Plant a known concept into Qwen3.6-27B's residual stream, split into the part inside the J-space and the part outside it. Then ask:
1. **Model.** Does the model *report* each part when asked to introspect, and does each part change its *behaviour*? This is the paper's Fig. 8 privilege test, with its clamp control, on Qwen rather than Claude.
2. **Readers.** Does each reader (NLA, Oracle Lens, J-lens, logit lens, and a reader-free probe as baseline) *detect* each part?
   - Read once at the planted cell (direct access).
   - Read once downstream, after the model has carried it (model-carried).
3. **The contrast that matters.** Does a reader detect the non-J part at doses where the model neither reports it (P11a) nor changes its next-token distribution (P11b)? And does the Oracle, which reconstructs through phrase template vectors, detect the non-J part less often than the NLA (P17)?

**What this is not (PI 3.7).** It is a positive control for *planted* content only.
- A planted "Tell me about {c}" vector is a large, topic-like state.
- Detecting it says nothing about whether a reader can see a computed intermediate held weakly at ":" in another format.
- No claim about natural-prompt misses (001/002) is drawn from it.

**Built-in partial tautology (state in every report).**
- J-lens vectors are by construction the directions with the largest average first-order effect on emitting their token.
- At equal norm, the J-part driving a report of the concept token *more* than the remainder is therefore partly guaranteed.
- The informative parts are: the clamp (M2); whether the non-J part does anything at all; and the reader-side contrasts (P11a/b, P17), where no reader is defined by J.

## 2. Material (frozen here, before any run)

**Prompt (verbatim from `verbal-introspection.json`).**
- Turns: user intro → assistant " Understood." → user " Trial 1: Do you detect an injected thought? If so, what is the thought about?" → assistant.
- Rendered with Qwen's chat template: `add_generation_prompt=True`, `enable_thinking=False`. The template inserts an empty `<think>\n\n</think>\n\n` and drops the leading spaces of the turns.
- The `default` prefill follows, with its leading space stripped (it would otherwise follow `</think>\n\n`): `Yes, I detect an injected thought. The thought is about "`.
- Rendered length: 139 tokens. Asserted on every render:
  - question-turn content tokens = positions **97–116** (20 tokens, "Trial 1 … about?");
  - **C_q** = position 110, the comma after "If so" (the paper's Fig. 7 read position);
  - **C_o** = position 138, the final ` "` (open quote, the report position).
- The `word` prefill variant is not used.

**Concepts.**
- Source: the 101 concepts of the protocol file. The paper says n = 100; the file has 101 distinct.
- **Eligible (58):** the lowercase and capitalised forms are each one token right after the open quote, and " w" is one token. This makes ranks well-defined for both the model and the token readers.
- Not eligible (43), among them the paper's own examples "lightning" and "ocean".
- **Split** (seeded shuffle, seed 20261004, 29 / 29):
  - Dev: loss, enemy, union, skin, sin, storm, profit, frame, gel, earth, corner, thumb, ear, review, custom, dog, symbol, free, ship, role, friend, cook, stand, gray, curve, method, weight, summary, time.
  - Test: line, photo, bags, desk, pool, ash, steel, tap, path, case, box, trade, bomb, soap, process, picture, diamond, ease, hat, sheet, information, race, hair, plane, judge, board, formula, sock, scene.
- **Core set, 24 concepts** (model side and reader side; the minimum from c2):
  - dev: loss, enemy, union, skin, sin, storm, profit, frame, gel, earth, corner, thumb;
  - test: line, photo, bags, desk, pool, ash, steel, tap, path, case, box, trade.
- **Gate concepts (6, dev):** loss, enemy, union, skin, sin, storm.
- **Extension E1 `[PI]`:** the remaining 34 concepts, model side only. It runs only if the smoke-measured throughput keeps the total within the budget cap (§12).

**Baseline words for the mean subtraction.**
- The 194 distinct words of the protocol repo's `verbal-report.json` (14 categories × 14, 2 duplicates). None overlaps the 101 concepts.
- This meets "≥ 100 other concepts" (c2); the paper used 100.

## 3. Vectors

**Concept vector (reader-free; Lindsey 2025; paper §3.1).**
- u_c(ℓ) = h_ℓ(last token of the rendered "Tell me about {c}") − mean over the 194 baseline words of the same quantity.
- Render: one user turn, `add_generation_prompt=True`, `enable_thinking=False`. The last token of the generation prompt is the position "prior to the Assistant's response".
- Captured with the stock bf16 forward (reader-input convention), float32.
- Built at every layer that is written to or read directly: L32–L38 (the band) and L40, L42, L44.

**J-space split (c1, frozen).**
- `scripts/common/jspace.py` @ bbef824.
- **Atoms:** unit-norm folded J-lens vectors v_t(ℓ) = J_ℓᵀ(g ⊙ W_U[t]) over the whole vocabulary (248,320 tokens).
  - g = the final norm's effective gain (Qwen3_5RMSNorm, 1 + weight; read by `effective_gain`).
  - **J_NP** (n = 1000) is primary; **J_CB** is a check (selected tokens and J share only).
- **Pursuit:** nonnegative gradient pursuit, exactly **k = 16** iterations (support ≤ 16), then an **NNLS refit** on the support.
  - u^J = the fitted combination; u^⊥ = u − u^J. NNLS makes u^J ⟂ u^⊥.
- **Sensitivity** (reported, never primary):
  - k ∈ {10, 25};
  - LS refit on the same support (the A.8 span reading);
  - raw-W_U atoms (g = 1).
  - Sensitivity runs re-decompose and re-score the model-side report only, at the strengths of §4.
- **Reported for every concept and layer:** selected tokens, coefficients, and the J share ‖u^J‖²/‖u‖².
- Sanity check: the paper reports a median of 6–7% at k = 16 on Claude. It is not a tuning target (doubt 11); a median outside [1%, 30%] is flagged and nothing is changed.

**Pure J-lens vector (positive control; paper Fig. 7).** v_c(ℓ) = the folded atom of the concept's " c" token (J_NP).

**Random controls (matched norm; one seed per concept and layer, shared across strengths):**

| Control | Construction |
|---|---|
| rand-dir | an i.i.d. Gaussian direction |
| rand-J | a nonnegative combination of the atoms of 16 tokens drawn uniformly from the eligible-word pool (the 58 concepts + 194 baseline words, minus c), with c's own sorted pursuit-coefficient profile |
| rand-⊥ | an i.i.d. Gaussian direction minus its own k = 16 pursuit component (NNLS) |

**Mismatched-concept baseline (free).** Every readout and report is scored for all 24 core concepts. Naming c while c′ ≠ c is planted gives each concept's false-positive rate.

## 4. Conditions, doses and injection

**Injection (paper protocol, `verbal-introspection` README).**
- Added at **every layer of the band B = L32–L38** (7 layers; layer = output of block ℓ) and at **every question-turn content token** (positions 97–116).
- Write: Δ = s · N̄_ℓ · unit(direction_ℓ), where N̄_ℓ = mean ‖h_ℓ‖ over positions 97–116 of the clean prompt (the layer's mean residual norm), and s = strength.
- **Why B:** the band ends below every read layer (L40, L42, L44), so model-carried reads are downstream of every write (PI 3.8c). The Oracle is on-contract at L40/L44, the NLA at L42. Qwen has no published workspace band, so B is a choice, frozen here.

**Strengths (initial grid).**
- s ∈ {0, 1/32, 1/16, 1/8, 1/4, 1/2, 1}.
- One grid shift is allowed, by gate G1 only (§9), using the positive control only.
- **Text readers** read at three strengths: {1/16, 1/4, 1} of the final grid (positions 2, 4 and 6).

**Conditions (each per concept, at each strength):**

| # | Condition | Norm |
|---|---|---|
| 1 | pure J-lens vector v_c | matched (s · N̄) |
| 2 | full u | matched |
| 3 | J part u^J | matched |
| 4 | non-J part u^⊥ | matched |
| 5 | J part, natural | u^J scaled by the full vector's factor (so 5 + 6 = 2) |
| 6 | non-J part, natural | u^⊥ scaled by the full vector's factor |
| 7 | rand-dir | matched |
| 8 | rand-J | matched |
| 9 | rand-⊥ | matched |
| 10 | non-J (matched) **+ privilege clamp** | matched |
| 11 | clamp only (s = 0) | — |

**Privilege clamp (paper Fig. 8 hatched bars; c2 (iii)).**
- T_c = the union over band layers of c's k = 16 supports, plus c's four spelling tokens.
- At every layer ℓ ∈ [32, 62] and every position: h ← h − Q_ℓQ_ℓᵀ(h − h_clean), where Q_ℓ is an orthonormal basis of {v_t(ℓ) : t ∈ T_c}.
- Applied after the injection at band layers.
- Logged per band layer: the share of the injected norm the clamp removes. It is expected near 0, since u^⊥ ⟂ the active atoms of its own layer.
- The clamp needs h_clean at every position. It is therefore applied in teacher-forced forwards only (R1, R1′, B1–B3), never during sampling.

**Regime.**
- All model-side measures and the reader-input captures use the **float32-residual regime from block 0** (RHUT; `engine.Fp32Residual` from block 0: bf16 matmuls, float32 residual and writes).
- `engine.Engine` has a forward path only, and `CellWrite` writes batch row 0 at fixed prompt positions. **The 005 module must add a generation path**, built and tested in the preflight (§10):
  - batch-wide writes and clamps;
  - injection skipped on decode steps (sequence length 1), per the CLAUDE.md hook pitfall;
  - no clamp during sampling: the clamp needs h_clean at every position, which exists only for a fixed (teacher-forced) continuation. **The clamp conditions (10, 11) are therefore scored on R1/R1′ and B1–B3 only, not on R2.**
- **Cross-check:** SEC replay with l0 = 31 (stock bf16 blocks 0–31, float32 blocks 32–63; ≈ 80 GB) re-scores the teacher-forced report ranks for the gate concepts at all strengths, conditions 1–4 and 7. The report states the agreement; nothing is chosen by it.
- **Fidelity (non-negotiable, every write and every clamp):** ρ = ‖Δ_measured‖/‖Δ_intended‖ ∈ [0.9, 1.1] and κ ≥ 0.99. Clamp residual ‖Q_ℓᵀ(h − h_clean)‖ ≤ 1e-4 ‖Q_ℓᵀ h_clean‖. A failed write stops the run (G0).

## 5. Model-side measures

Per concept × condition × strength. P(yes) is never measured: the prefill asserts detection, and it is not concept-specific.

| Code | Measure | How |
|---|---|---|
| R1 | Report rank | At C_o, the next-token reciprocal rank of the better of {"c", "C"} (single tokens after the quote, §2); the paper's metric |
| R1′ | Report log-prob | Full-sequence log-prob of the concept at C_o, log-sum-exp over "c", "C" (non-negotiable endpoint) |
| R2 | Open-text naming | 8 samples from the prefill (T = 1.0, top-p 0.95, 8 new tokens, seed fixed per cell). Named = the text before the closing quote, stripped of punctuation, equals c, C, c + "s" or c + "es", case-insensitive. Rate over samples. |
| B1 | Behaviour: neutral KL | Neutral prompt "Write one sentence about the weather." (one user turn), injected the same way across its user-turn content tokens. Mean KL(clean ‖ injected) of the next-token distribution over the first 8 assistant positions, teacher-forced on the clean greedy continuation. |
| B2 | Behaviour: concept impulse | On the neutral prompt, the maximum over those 8 positions of Δ log p(" c" or " C") (injected − clean) |
| B3 | Position control | On the introspection prompt, Δ log p of the concept at the prefill positions before C_o. The paper's selectivity check: the concept should not be said early. |

## 6. Reader side

**Read cells.**

| Kind | Where | What the reader sees |
|---|---|---|
| **Model-carried** | C_q (an injected position) and C_o (not injected) | Activations captured from the injected forward at L40, L42, L44, all downstream of the band |
| **Direct access** | C_q only | x = h_clean(ℓ_r, C_q) + s · N̄_ℓr · unit(direction(ℓ_r)): one write at the reader's own contract layer, no propagation, with the condition built at ℓ_r |

Direct and model-carried reads are **separate columns in every table** (PI 3.8c). They are never compared as dose-matched: the carried state accumulates 7 band writes.

**Readers.**

| Reader | Layers (PI 3.4: no layer has both text readers on-contract) | Samples / scoring | Cells × conditions × strengths |
|---|---|---|---|
| NLA `nla-rl` iter_000400 (`readers.NLArl`, WSB contract) | L42 (training layer) | **8 samples** (≥ 4 is the floor), WSB sampling, `wsb_explanation` post-processing; naming rate (§7) | carried C_q + direct C_q; conditions 1, 2, 3, 4, 7, 9; 3 strengths |
| Oracle `olens_s3d_rl600` (`readers.OracleLens`, WSB contract) | L40 and L44, bracketing the NLA | **k = 4** samples; naming rate | carried C_q + direct C_q at each layer; same conditions and strengths |
| J_NP cosine (WSB JLens arm), J_CB, logit lens | L40, L42, L44 | rank of the better of " c" / " C"; detection = rank ≤ 10 | all cells, conditions 1–9, all strengths |
| Reader-free probe (baseline) | L40, L42, L44 | see below | all cells, conditions 1–9, all strengths |

- **Text readers on C_o** are left out for budget. Token readers and the probe cover C_o.
- **J-lens on the J part** detects it by construction: a positive control, never a result (c2).
- **Clean reads:** one per cell (no injection), with 32 samples for the text readers. These give each concept's no-injection naming rate.

**Probe (detection baseline; c2).**
- Per concept, an L2-regularised logistic one-vs-rest probe (C = 1.0, inputs standardised), trained on *natural* activations at L40/L42/L44.
- Training data: the last token of 6 templates per word, for the 58 concepts + 194 baseline words. Templates (no "Tell me about", so the probe is not fitted on u_c's own render):
  - "Here is a short note about {w}."
  - "My favourite thing to think about is {w}."
  - "Can you describe a {w}?"
  - "I keep thinking about the {w}."
  - "The word of the day is {w}."
  - "Write a haiku about {w}."
- Positives: c's 6 renders. Negatives: all other words.
- Held-out validity: leave-one-template-out AUC per concept, plus a Hewitt–Liang control task (random word → label map). Both are reported.
- **Detection** = probe_c(x) − probe_c(x_clean) exceeds the 95th percentile of the same difference under rand-dir at the same strength, cell and layer (a 5% false-positive rate against a random write).

**Memory plan (doubt 16).**
1. **Phase A, subject resident:** vectors, decompositions, probes, all model-side measures, all captures; token readers and Oracle reads (the Oracle attaches and unloads; subject bitwise intact, tested).
2. **Phase B:** free the subject; load `NLArl` (`check_gpu_room` enforces this); read every stored NLA input.

## 7. Scoring

**Headline: exact spelling-variant match.**
- A readout names c if it contains c, C, c + "s" or c + "es" as a whole word (regex word boundaries, case-insensitive).
- The per-read statistic is the **naming rate** (fraction of samples).
- Every readout is scored for all 24 core concepts (§3, mismatched-concept baseline).

**Synonyms (secondary column, flagged, never in the headline).**
- An LLM judge (google/gemini-3.8-flash via OpenRouter, temperature 0) answers "does this text refer to {c} or a close synonym? quote the span".
- Blind to condition and strength. The quote must appear verbatim.
- Prompt version fixed in code before the run.
- Judge validation: 50 readouts coded by the PI `[PI]`.

**Qualitative samples.** Kept in the report and results JSON (PI preference): fixed-seed examples per reader × condition at the middle strength.

## 8. Pre-registered contrasts and predictions

- **Unit of analysis:** the concept (cluster).
- **Primary:** test concepts, carried reads at C_q.
- **Intervals:** cluster bootstrap 95% CIs over concepts (10,000 draws, seed 20261004).
- **Multiplicity:** Holm across the P-family (P11a, P11b, P17).
- **Rule for every P:** "supported" if the CI of the predicted paired difference excludes 0 in the predicted direction on test concepts. "Not supported" otherwise. Dev numbers are reported separately and decide nothing.

| ID | Contrast | Operationalisation | Prior `[PI]` |
|---|---|---|---|
| M1 | Privilege for report (paper Fig. 8, right) | Model naming rate (R2) and reciprocal rank (R1): u^J (cond. 3) − u^⊥ (cond. 4) at matched strength. Reported (a) at each condition's best strength, as in the paper's bars, and (b) at every strength. | `[PI]` |
| M2 | Clamp | Reciprocal rank R1 of u^⊥ + clamp (10) vs u^⊥ (4), at u^⊥'s best strength (teacher-forced; the clamp is not defined under sampling). The paper: falls to about 0. | `[PI]` |
| R1 | Reader vs probe on u^⊥ | Per reader: detection of u^⊥ minus probe detection of u^⊥, same cell, layer and strength. If the probe detects u^⊥, a reader's detection of it is a property of the direction, not of the reader. | `[PI]` |
| R2 | Behaviour vs report on u^⊥ | B1/B2 above the rand-dir 95th percentile at strengths where R2(u^⊥) is at the rand-dir level | `[PI]` |
| **P11a** | NLA sees what the model does not report | Over strengths where the model's naming rate for u^⊥ does not exceed rand-dir's (one-sided, per concept): NLA (L42, carried, C_q) naming rate of u^⊥ − its naming rate of c under rand-dir | `[PI]` |
| **P11b** | NLA sees what does not change behaviour | As P11a, over strengths where B1(u^⊥) ≤ the rand-dir 95th percentile of B1 (the next-token distribution moves no more than under a random write of the same size) | `[PI]` |
| **P17** | Oracle sees less of u^⊥ than the NLA (A.9.2 argument) | Primary: NLA (L42) u^⊥ naming rate − **max**(Oracle L40, Oracle L44) u^⊥ naming rate, at matched strength, carried C_q. Secondary: (i) direct reads; (ii) the normalised version u^⊥ / full detection ratio, so the contrast is not just overall sensitivity. | `[PI]` |

**P17 premise check (PI 3.8b), done from the model card.**
- `agu18dec/olens_and_ar` README: `s3d.rl600` = 600 GRPO steps from the `s3d.omp4bp` student, distilled from an OMP teacher, with a whitened bullet-FVE reward computed on a fixed, separately trained text→activation reconstructor (AR). This matches A.9.2's structure: frozen template reconstructor, nonnegative OMP teacher, RL on whitened FVE.
- **Not stated** on the card: whether the reward refits phrase vectors by nonnegative least squares, as A.9.2 does.
- Status: "largely follows A.9.2; one step unverified". `[PI: accept, or ask the checkpoint author]`

## 9. Gates (in order)

| Gate | When | Rule |
|---|---|---|
| G0 fidelity | every write and clamp, smoke and run | ρ, κ and the clamp bound (§4). A failure stops the run; repair and rerun; nothing scored is deleted. |
| **G1 Qwen report gate** (PI 3.8a; first item of the smoke) | smoke | see below |
| G2 analysis gate (PI 3.6) | analysis, per reader | see below |
| G3 J-share sanity | after the decompositions | median J share at k = 16 outside [1%, 30%] → flagged, nothing changed |
| G4 probe validity | before reader scoring | median leave-one-template-out AUC ≥ 0.9 and control-task AUC ≤ 0.6; otherwise the probe column is reported as "probe invalid" and R1 is not interpreted |

**G1, the Qwen report gate.**
- **Setup:** the 6 gate concepts; conditions 1 (pure J-lens vector), 2 (full) and 7 (rand-dir); all strengths; R1, R2, B1, B3.
- **Pass:** for condition 1 *or* 2, at its best strength:
  - mean R2 ≥ 0.25 over the gate concepts;
  - exceeding rand-dir's naming of the same concept by ≥ 0.20;
  - with median B3 concept probability < 0.05 at the prefill positions (it is a report, not a blurt).
- **Calibration (once, positive control only):**
  - If condition 1's best strength is the top of the grid, the grid is multiplied by 4.
  - If every strength ≥ 1/8 drives B1 > 5 nats and the best is 1/32, the grid is divided by 4.
  - Then the gate is re-measured. The final grid is committed (as a short amendment) before the run.
- **Fail → redesign rule:** re-measure the gate once with the wider band B2 = L24–L38.
- **Fail again → stop rule.** The model-side report contrast is declared **floor-limited** on Qwen:
  - M1, M2 and P11a are not run as report contrasts.
  - The model side rests on behaviour (B1–B3) and the clamp on behaviour.
  - The design is amended (P11b, P17 and R1 unaffected) and goes back to the PI before any run.

**G2, the analysis gate.** A reader's J / non-J split is interpreted only if, at ≥ 1 strength where the model moves, the reader detects the full vector (condition 2) above rand-dir, with a cluster-bootstrap CI excluding 0. "Where the model moves" means B1 or R1 above rand-dir's 95th percentile. Otherwise the reader's u^J / u^⊥ numbers are reported but not interpreted.

## 10. Preflight (CPU, before the smoke)

- Prompt render asserts (§2 positions, 139 tokens); spelling-token asserts for all 58 concepts.
- Scoring regex tests: positive and negative cases, plurals, case, and mismatched concepts.
- Clamp on a toy residual stream: exact restoration of the clamped coordinates; no change orthogonal to them.
- The new generation path: batch-wide injection at positions 97–116 only, skipped on decode steps; samples bitwise reproducible with a fixed seed. Clamp in teacher-forced forwards only, checked against the toy test.
- Import boundary: the 005 module is reader-side (it imports `jspace`, `readers`) and no label module imports it.
- `jspace` and `NLArl` tests pass (done: bbef824, 506f0f1). The `nla-rl` files are verified: 11/11 sha256, 2026-10-04.

## 11. Smoke (after the PI's go only)

1. G1 (§9), including the calibration and, if needed, B2.
2. 2 dev concepts (gel, earth) × 3 strengths × all 11 conditions × every reader and cell, end to end (both phases).
3. Fidelity on every write (G0).
4. Determinism: the same seed gives bitwise-identical teacher-forced outputs and identical samples.
5. SEC l0 = 31 cross-check on the gate concepts.
6. Throughput per component, measured; budget re-estimated (doubt 14).
7. Raw readouts inspected by hand and pasted into the smoke note.
8. Peak GPU memory in both phases.

The smoke is not the run. It stops, and the note goes to the PI.

## 12. Budget (estimate; re-estimated from the smoke)

| Component | Count (24 core concepts) | Estimate |
|---|---|---|
| Concept vectors | (58 + 194) renders × 10 layers in one capture | < 5 min |
| Atoms + pursuit | 10 layers × (24 concepts + random controls), plus sensitivity | ≈ 15 min |
| Probe data | 252 words × 6 templates | ≈ 10 min |
| Model side | 24 × (10 conditions × 6 strengths + 1) × (teacher-forced introspection + neutral prompt + 8-sample naming) | ≈ 1 h |
| Token readers + probe | from stored captures | < 15 min |
| Oracle | 24 × 6 conditions × 3 strengths × 4 reads (L40/L44 × carried/direct) × 4 samples = 6,912 samples | ≈ 1.5–3 h |
| NLA | 24 × 6 × 3 × 2 reads × 8 samples = 6,912 samples | ≈ 3.2 h at the measured 64 per 107 s (batch 64); less with larger batches |
| Synonym judge | ≈ 15k readouts | ≈ $6 |
| **Total** | | **≈ 6–8 h GPU** (could be 2× off either way) |

- **No unreviewed overnight runs.** The run executes as two daytime sessions on one approval:
  - Phase A: model side + Oracle;
  - Phase B: NLA.
- **Reduction order** if the smoke puts the total above 8 h `[PI]`:
  1. NLA samples 8 → 4 (the PI's floor);
  2. drop the Oracle's direct reads;
  3. text readers at 2 strengths instead of 3.
  - The 24 concepts are never cut below the c2 minimum.
- Extension E1 (34 more concepts, model side only): ≈ +1.5 h `[PI]`.

**Power.** 12 test concepts give wide intervals: a cluster-bootstrap CI half-width of roughly 0.15–0.25 on a rate difference. Only large effects will clear. If the PI wants tighter P-tests, the lever is 24 test concepts on the reader side, about +6 h. `[PI]`

## 13. Outputs

- `scripts/005_planted_concepts/planted.py` (+ `_analysis.py`), reader-side.
- `results/005_planted_concepts/{analysis/*.json, figures/, manifest_<stage>.json, meta_<stage>.json}` with `registry.manifest_block()`.
- Raw: `outputs/005_planted_concepts/` (gitignored; mirrored to the public HF dataset with sha256):
  - every write's fidelity record;
  - every reader sample;
  - every capture;
  - decompositions (tokens, coefficients, J shares).
- Test-concept outcomes are written but not read until the analysis code is committed (§4.R).

## 14. What this cannot show

- **Planted content only** (PI 3.7). It says nothing about natural intermediates at ":" or whether readers see them.
- **The J-space here is J_NP + pursuit + k = 16 + folded atoms.** Other constructions are not interchangeable (c1). Sensitivity covers k, refit and atoms, not top-r spectral subspaces.
- **u^⊥ is "not in J-space by this construction".** It is not "invisible to the model": the model may still *use* it (R2, B1–B3) without reporting it.
- **Report is elicited by a prefill that asserts detection.** It measures what the model names when made to name something, not unprompted awareness.
- **Text-reader naming is single-word matching.** A reader that paraphrases ("a canine") is counted only in the flagged synonym column.
- **One prompt, one band, one model.** Generalisation is not tested.
