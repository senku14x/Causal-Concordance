# 005 — Planted concepts: J-part vs non-J part, for the model and every reader — design v2 (DRAFT for PI review)

*Status: v2, revised to the PI review of 2026-10-04 (`docs/reviews/2026-10-04_pi_review_checks_005/`; changes in §15). **The PI authorised the smoke (§11) on 2026-10-04** ("build the 005 code and tests … run the code and the tests"). **The full run is not authorised:** it needs a separate go after the PI reviews the smoke note and the G1 amendment.*
- Sources: PI feedback 2026-10-04 (§2 c1/c2, §3.6–3.8; doubts 10, 11, 13, 14, 16; §5 item 8); the PI review of v1 (same day); review `docs/context/SPEC_v2_review_2026-10-04.md` (c) c1, c2.
- Paper: Gurnee et al. 2026 (arXiv 2607.15495), §3.1 Figs. 7–8, App. A.8, A.9.2.
- The official protocol file is `anthropics/jacobian-lens` `data/experiments/verbal-introspection.json` (`/workspace/jacobian-lens`).
- **One run, not two stages** (PI 3.6). The stage-1 rule survives as analysis gate G2 (§9).
- Every `[PI]` field is left for Vishesh, the prediction priors included. The agent's own priors are pre-registered next to them (§8), committed before any 005 GPU run (PI request 2026-10-04).

## 1. Question

Plant a known concept into Qwen3.6-27B's residual stream, split into the part inside the J-space and the part outside it. Then ask:
1. **Model.** Does the model *report* each part when asked to introspect, and does each part change its *behaviour*? This is the paper's Fig. 8 privilege test, with its clamp control, on Qwen rather than Claude.
2. **Readers.** Does each reader (NLA, Oracle Lens, J-lens, logit lens, and a reader-free probe as baseline) detect each part?
   - **Direct access:** the reader is given the written cell itself. This tests reader sensitivity.
   - **Model-carried:** the reader reads the report position (the open quote). It is never written to, so anything there arrived by attention, which is what the model must do in order to report.
3. **The contrasts that matter.**
   - Does the NLA detect the non-J part at the report position at strengths where the model's report *probability* and its behaviour do not move (P11a, P11b)?
   - Given direct access, does the Oracle, which reconstructs through phrase template vectors, detect the non-J part less often than the NLA (P17)?

**What this is not (PI 3.7).** It is a positive control for *planted* content only.
- A planted "Tell me about {c}" vector is a large, topic-like state.
- Detecting it says nothing about whether a reader can see a computed intermediate held weakly at ":" in another format.
- No claim about natural-prompt misses (001/002) is drawn from it.

**Built-in partial tautology (state in every report).**
- J-lens vectors are by construction the directions with the largest average first-order effect on emitting their token.
- At equal norm, the J-part driving a report of the concept token *more* than the remainder is therefore partly guaranteed.
- The informative parts are: the clamp against its random-clamp null (M2); whether the non-J part does anything at all; and the reader-side contrasts (P11a/b, P17), where no reader is defined by J.

## 2. Material (frozen here, before any run)

**Introspection prompt (verbatim from `verbal-introspection.json`).**
- Turns: user intro → assistant " Understood." → user " Trial 1: Do you detect an injected thought? If so, what is the thought about?" → assistant.
- Rendered with Qwen's chat template: `add_generation_prompt=True`, `enable_thinking=False`. The template inserts an empty `<think>\n\n</think>\n\n` and drops the turns' leading spaces.
- The `default` prefill follows, its leading space stripped: `Yes, I detect an injected thought. The thought is about "`.
- 139 tokens. Asserted on every render:
  - question-turn content (the **injected span**) = positions **97–116** (20 tokens);
  - **C_q** = 110, the comma after "If so";
  - prefill = positions **126–138** (13 tokens);
  - **C_o** = 138, the final ` "` (open quote, the report position).

**Neutral prompt.** "Write one sentence about the weather." as one user turn, same template. 19 tokens; user content = positions 3–9.

**Concepts.**
1. **Source:** the 101 concepts of the protocol file (the paper says n = 100).
2. **Eligibility (58):** the lowercase and capitalised forms are each one token right after the open quote, and " w" is one token.
3. **Screen (PI review; rule and 5% threshold fixed before computing; `scripts/005_planted_concepts/screen_concepts.py`, `results/005_planted_concepts/analysis/concept_screen.json`):**
   - Base rate = per-sample naming rate (§7 variant match) in 7,148 existing NLA and Oracle samples that did not target the concept:
     - NLA: Cathryn's pilot `nla-rl` readouts and 001's nla-L42 readouts (the readouts 002 scored);
     - Oracle: the pilot's and 001's Oracle readouts.
   - Brew readouts are not counted for palette colours.
   - Exclusion: pooled NLA *or* pooled Oracle rate > 0.05.
   - **Excluded: time** (NLA 0.055), **line** (NLA 0.129), **path** (Oracle 0.053). The others are ≤ 0.022 (NLA) and ≤ 0.030 (Oracle). 55 remain.
4. **Allocation (seeded order from v1, seed 20261004; dev only gates and calibrates):**
   - **Dev, 10:** loss, enemy, union, skin, sin, storm, profit, frame, gel, earth (the first 10 of the dev order).
     - Gate concepts: the first 6.
     - Smoke concepts: gel, earth.
   - **Test, 45:**
     - from the original dev order: corner, thumb, ear, review, custom, dog, symbol, free, ship, role, friend, cook, stand, gray, curve, method, weight, summary;
     - from the original test order: photo, bags, desk, pool, ash, steel, tap, case, box, trade, bomb, soap, process, picture, diamond, ease, hat, sheet, information, race, hair, plane, judge, board, formula, sock, scene.
5. **In-run screen (G6):** a concept whose clean-read naming rate at a reader's cell exceeds 0.05 is dropped from that reader's analyses (reported).

## 3. Vectors

**Concept vector (reader-free; Lindsey 2025; paper §3.1).**
- h_ℓ(w) = the residual at layer ℓ at the last token of the rendered "Tell me about {w}". Render: one user turn, `add_generation_prompt=True`, `enable_thinking=False`; the last token of the generation prompt is the position "prior to the Assistant's response". Captured with the stock bf16 forward, float32.
- **Primary (PI review):** u_c(ℓ) = h_ℓ(c) − mean over the **other 100 protocol concepts** (leave-one-out over the 101, eligible or not). This is the most plausible reading of the paper's "100 other concepts", though not stated in the file.
- **Sensitivity:** the mean over the 194 verbal-report words instead. Report the cosine between the two versions per concept and layer; if the median is > 0.9, the choice "does not matter".
- Built at L32–L38 (the band) and at L40, L42, L44 (direct reads).

**J-space split (c1, frozen).**
- `scripts/common/jspace.py` @ bbef824.
- **Atoms:** unit-norm folded J-lens vectors v_t(ℓ) = J_ℓᵀ(g ⊙ W_U[t]) over the whole vocabulary (248,320 tokens), with g = the final norm's effective gain (1 + weight).
  - **J_NP** (n = 1000) is primary; **J_CB** is a check (selected tokens and J share only).
- **Pursuit:** nonnegative gradient pursuit, exactly **k = 16** iterations, then an **NNLS refit** on the support.
  - u^J = the fitted combination; u^⊥ = u − u^J. NNLS makes u^⊥ orthogonal to the active atoms of *its own layer*, not to J-space as a whole (G5).
- **Sensitivity** (model-side R1/R1′ re-scored for the J and ⊥ parts at all strengths; never primary):
  - k ∈ {10, 25};
  - LS refit (A.8 span reading);
  - raw-W_U atoms;
  - the 194-word baseline.
- **Reported for every concept and layer:** selected tokens, coefficients, and the J share ‖u^J‖²/‖u‖².
- Sanity range (G3): the paper reports a median of 6–7% at k = 16. Not a tuning target.

**Pure J-lens vector (positive control).** v_c(ℓ) = the folded atom of " c" (J_NP).

**Random controls (matched norm; seeded per concept and layer, shared across strengths):**

| Control | Construction |
|---|---|
| rand-dir | an i.i.d. Gaussian direction. **Model side: 5 seeds per concept** (they form the null distributions of §8); readers use seed 0 |
| rand-J | a nonnegative combination of the atoms of 16 tokens drawn from the **165 single-token baseline words** (" w", from the 194), with c's own sorted pursuit-coefficient profile. No core concept enters it (PI review). |
| rand-⊥ | an i.i.d. Gaussian direction minus its own k = 16 pursuit component (NNLS) |

**Mismatched-concept baseline (free).** Every readout and report is scored for all 55 concepts.

## 4. Conditions, strengths and injection

**Injection (paper protocol, `verbal-introspection` README).**
- Added at **every layer of the band B = L32–L38** (7 layers; layer = output of block ℓ) and at **every token of the injected span** (97–116).
- Write: Δ = s · N̄_ℓ · unit(direction_ℓ), where N̄_ℓ = mean ‖h_ℓ‖ over 97–116 of the clean prompt.
- The neutral prompt is injected the same way over its user content (positions 3–9).
- Why B: it ends below every read layer (L40, L42, L44) and keeps the Oracle on-contract at L40/L44 and the NLA at L42. Qwen has no published band; the fallback B2 is in G1.

**Conditions.**

| # | Condition | Norm | Model side | Token readers + probe | Text readers |
|---|---|---|---|---|---|
| 1 | pure J-lens vector v_c | matched | ✓ | ✓ | optional `[PI]`, default off |
| 2 | full u | matched (s · N̄) | ✓ | ✓ | ✓ |
| 3 | J part u^J | matched | ✓ | ✓ | ✓ |
| 4 | non-J part u^⊥ | matched | ✓ | ✓ | ✓ |
| 5 | J part, natural | u^J at the full vector's scale (5 + 6 = 2) | ✓ (M3) | ✓ | — |
| 6 | non-J part, natural | u^⊥ at the full vector's scale | ✓ (M3) | ✓ | — |
| 7 | rand-dir | matched | ✓ (5 seeds) | ✓ (seed 0) | ✓ (seed 0) |
| 8 | rand-J | matched | ✓ | ✓ | — |
| 9 | rand-⊥ | matched | ✓ | ✓ | — |
| 10 | u^⊥ (matched) + **c-clamp** | matched | ✓ (teacher-forced) | — | — |
| 10r | u^⊥ (matched) + **random clamp** | matched | ✓ (teacher-forced) | — | — |
| 11 / 11r | c-clamp only / random clamp only (s = 0) | — | ✓ (teacher-forced) | — | — |

**Strengths.**
- **Model side and token readers:** s ∈ {0, 1/32, 1/16, 1/8, 1/4, 1/2, 1}. G1 may shift the grid once (×4 or ÷4), using the positive control only.
- **Text readers: three strengths chosen after G1, not now (PI review).**
  - In G1 (dev concepts), the **onset** s* = the smallest grid strength at which the median over the 10 dev concepts of ΔR1′(u^⊥) exceeds the 95th percentile of ΔR1′ under rand-dir (5 seeds × 10 dev concepts) at that strength. ΔR1′ is the concept's log-prob at C_o relative to clean.
  - Text-reader strengths = {s*/4, s*, 4s*}, clipped to the grid. Without an onset in the grid: the three highest grid strengths.
  - The strengths go into the G1 amendment and are frozen before the run.

**Privilege clamp (paper Fig. 8; c2 (iii)), with its null (PI review).**
- **c-clamp:**
  - T_c = the union over band layers of c's k = 16 supports, plus c's four spelling tokens.
  - At every layer ℓ ∈ [32, 62] and every position: h ← h − Q_ℓQ_ℓᵀ(h − h_clean), where Q_ℓ is an orthonormal basis of {v_t(ℓ) : t ∈ T_c}. Applied after the injection at band layers.
- **Random clamp:** the same, with |T_c| tokens drawn (seeded per concept) from the single-token alphabetic leading-space vocabulary, excluding the spelling tokens of all 101 concepts.
- **Why the null is needed.** At any layer, atoms selected at other band layers, and every atom at L39–62, are *not* orthogonal to the injected or carried u^⊥. So either clamp deletes part of the injection, not only its re-entry into J-space.
- **Logged for both clamps:**
  - at band layers, the share of the injected norm removed, ‖QQᵀΔ‖/‖Δ‖;
  - at L39–62, the share of the carried perturbation removed, ‖QQᵀ(h − h_clean)‖/‖h − h_clean‖.
- The clamp needs h_clean at every position, so it is applied in **teacher-forced forwards only** (R1, R1′, B1–B4), never during sampling.

**Regime.**
- All model-side measures and reader-input captures use the **float32-residual regime from block 0** (RHUT; `engine.Fp32Residual`: bf16 matmuls, float32 residual and writes).
- `engine.Engine` has a forward path only, and `CellWrite` writes batch row 0. **The 005 module adds a generation path**, built and tested in the preflight (§10):
  - batch-wide per-row writes (all conditions of a concept share one prompt, so they batch);
  - **sampling with a cache-free loop:** each step re-runs the full sequence through the proven `use_cache=False` forward, with the injection at prompt positions 97–116 only. This is equivalent to cached generation from an injected prompt and avoids decode-step hooks;
  - no clamp during sampling.
- **Cross-check:** SEC replay with l0 = 31 re-scores teacher-forced R1 for the gate concepts, conditions 1–4 and 7, at all strengths. Agreement is reported; nothing is chosen by it.
- **Fidelity (every write and clamp):** ρ ∈ [0.9, 1.1], κ ≥ 0.99. Clamp residual ‖Q_ℓᵀ(h − h_clean)‖ ≤ 1e-4 ‖Q_ℓᵀ h_clean‖. A failure stops the run (G0).

## 5. Model-side measures

Per concept × condition × strength. P(yes) is never measured.

| Code | Measure | How |
|---|---|---|
| R1 | Report rank | At C_o, the next-token reciprocal rank of the better of {"c", "C"} (the paper's metric) |
| R1′ | Report log-prob | Full-sequence log-prob of the concept at C_o, log-sum-exp over "c", "C" (the non-negotiable endpoint). **The most sensitive report measure.** |
| R2 | Open-text naming | 8 samples from the prefill (T = 1.0, top-p 0.95, 8 new tokens, seed fixed per cell). Named = the text before the closing quote, stripped of punctuation, equals c, C, c + "s" or c + "es", case-insensitive. Rate over samples. |
| B1 | Neutral-prompt KL | Mean KL(clean ‖ injected) over the first 8 assistant positions of the neutral prompt, teacher-forced on the clean greedy continuation |
| B2 | Concept impulse | On the neutral prompt, the maximum over those 8 positions of Δ log p(" c" or " C") |
| B3 | Position control | On the introspection prompt, Δ log p of the concept at the prefill positions 126–137 (it should not be said early) |
| **B4** | **Same-forward behaviour (PI review)** | Mean KL(clean ‖ injected) of the next-token distribution over the prefill positions 126–137 of the introspection prompt, **from the same forward as the reader captures** |

**Nulls.** For any measure X, "within the null" means ΔX (injected − clean) ≤ the 95th percentile of ΔX under rand-dir at the same strength. The null pools 5 seeds × the concepts of the same split.

## 6. Reader side

**Read kinds (PI review; never mixed in a table).**

| Kind | Where | What the reader sees | Readers | Primary for |
|---|---|---|---|---|
| **Direct** | C_q, at the reader's contract layer ℓ_r | x = h_clean(ℓ_r, C_q) + s · N̄_ℓr · unit(direction(ℓ_r)): one write at ℓ_r, no propagation | all | **P17**, R1 |
| **Model-carried** | **C_o** (138) at L40, L42, L44, from the injected forward | only what attention brought from the injected span | all | **P11a, P11b** |
| **Injected position** | C_q (110) at L40/42/44, from the injected forward | the band writes, still in the residual stream, plus the model's processing; near-direct, *not* model-carried | token readers + probe only | diagnostics |

**Readers.**

| Reader | Layers (PI 3.4) | Samples / scoring | Reads |
|---|---|---|---|
| NLA `nla-rl` iter_000400 (`readers.NLArl`, WSB contract) | L42 | **4 samples** (the PI's floor), WSB sampling and post-processing; naming rate (§7) | direct + carried C_o; conditions 2, 3, 4, 7; the 3 post-G1 strengths |
| Oracle `olens_s3d_rl600` (`readers.OracleLens`, WSB contract) | L40 and L44 (bracketing the NLA) | **4 samples** per read (`k = 4` = `num_return_sequences`, as in 001/002; each sample has several bullets); naming rate | direct + carried C_o at each layer; same conditions and strengths |
| J_NP cosine (WSB JLens arm), J_CB, logit lens | L40, L42, L44 | rank of the better of " c" / " C"; detection = rank ≤ 10 | all three kinds; conditions 1–9; all strengths |
| Probe (reader-free baseline) | L40, L42, L44 | below | all three kinds; conditions 1–9; all strengths |

- **J-lens on the J part** detects it by construction: a positive control, never a result.
- **Clean reads:** one per cell and kind, with 32 samples for text readers. They give the no-injection naming rate and the in-run screen (G6).

**Probe (c2).**
- Per concept, an L2-regularised logistic one-vs-rest probe (C = 1.0, inputs standardised), trained on natural activations at L40/L42/L44.
- Training data: the last token of 6 templates per word, over the 101 concepts + 194 baseline words:
  - "Here is a short note about {w}."
  - "My favourite thing to think about is {w}."
  - "Can you describe a {w}?"
  - "I keep thinking about the {w}."
  - "The word of the day is {w}."
  - "Write a haiku about {w}."
- No "Tell me about": the probe is not fitted on u_c's own render.
- Validity (G4): leave-one-template-out AUC and a Hewitt–Liang control task.
- **Detection** = probe_c(x) − probe_c(x_clean) > the 95th percentile of that difference under rand-dir (5 seeds) at the same strength, kind, cell and layer.

**Memory plan (doubt 16).**
1. **Phase A, subject resident:** vectors, decompositions, probes, the model side, all captures, token readers, Oracle reads (it attaches and unloads; subject bitwise intact, tested).
2. **Phase B:** free the subject; load `NLArl` (`check_gpu_room` enforces this); read every stored NLA input.

## 7. Scoring

**Headline: exact spelling-variant match.**
- A readout names c if it contains c, C, c + "s" or c + "es" as a whole word (case-insensitive).
- The per-read statistic is the **naming rate**. Every readout is scored for all 55 concepts.

**Synonyms (secondary column, flagged, never in the headline).**
- An LLM judge (google/gemini-3.8-flash via OpenRouter, temperature 0) asks whether the text refers to c or a close synonym and must quote the span verbatim.
- Blind to condition and strength; prompt version fixed in code before the run.
- Judge validation: 50 readouts coded by the PI `[PI]`.

**Qualitative samples.** Fixed-seed examples per reader × kind × condition at s*, kept in the report and results JSON.

## 8. Pre-registered contrasts and predictions

- **Unit of analysis:** the concept (cluster).
- **Primary:** the 45 test concepts.
- **Intervals:** cluster bootstrap 95% CIs over concepts (10,000 draws, seed 20261004).
- **Multiplicity:** Holm across the P-family (P11a, P11b, P17).
- **Rule:** "supported" if the CI of the predicted paired difference excludes 0 in the predicted direction on test concepts. Dev numbers are reported separately and decide nothing.
- **Split quality:** every "non-J" result carries the G5 qualifier if G5 flags.

| ID | Contrast | Operationalisation | Prior `[PI]` | Agent prior (pre-registered 2026-10-04, before any 005 GPU run) |
|---|---|---|---|---|
| M1 | Privilege for report, matched norm (paper Fig. 8, right) | R2 and R1: u^J (3) − u^⊥ (4). (a) at each condition's best strength; (b) at every strength | `[PI]` | 0.85 that u^J > u^⊥ on R1 at matched norm. Largely by construction: the J part contains c's own atom. |
| M2 | Clamp, against its null | At u^⊥'s best strength (by R1, per concept): [R1(10) − R1(4)] − [R1(10r) − R1(4)]. **Interpreted only if the c-clamp lowers R1 more than the random clamp** (CI excludes 0); the removed-norm shares are reported next to it. | `[PI]` | 0.30 that M2 is interpretable and supported. I expect u^⊥ to move R1 only weakly, and both clamps to remove a similar share of the carried perturbation. |
| M3 | Privilege at natural norm (PI review) | R1, R2: J-natural (5) vs ⊥-natural (6) vs full (2), at the same s. At about 6% J share the matched J part is scaled up about 4× over its natural size; M3 respects natural scale. | `[PI]` | 0.60 that J-natural > ⊥-natural on R1 at the strengths where the full vector moves R1, even though ⊥ is about 4× larger. |
| R1 | Reader vs probe on u^⊥ | Per reader: detection of u^⊥ minus probe detection of u^⊥, same kind, layer and strength. Direct is primary; carried C_o secondary. | `[PI]` | 0.70 that the probe detects u^⊥ at least as often as the NLA (direct reads). The probe is fitted for exactly this concept; u^⊥ carries about 90% of the concept vector. |
| R2 | Behaviour vs report on u^⊥ | B1/B2/B4 outside the null at strengths where R2 and R1′ of u^⊥ are within it | `[PI]` | 0.50 that u^⊥ moves B1, B2 or B4 out of the null at some strength where R1′ and R2 stay within it. |
| **P11a** | NLA sees, at the report position, what the model does not report | **Primary (strict):** a strength qualifies for a concept if R2(u^⊥) ≤ R2(rand-dir) for that concept **and** ΔR1′(u^⊥) is within the null. Statistic: the NLA's carried-C_o naming rate of c under u^⊥ minus under rand-dir, averaged over qualifying strengths. Concepts with no qualifying strength drop out (count reported). **Secondary:** v1's naming-only qualification. | `[PI]` | 0.20. The strict rule needs the model's report log-prob unmoved, while enough of u^⊥ reaches C_o by L42 for the NLA to *name* c. I expect detection at C_o to track the model's own report. |
| **P11b** | NLA sees, at the report position, what does not change behaviour | **Primary (strict):** a strength qualifies if B1, B2 **and** B4 of u^⊥ are all within the null. Same statistic. **Secondary:** v1's B1-only qualification. | `[PI]` | 0.15. Even stricter: KL is sensitive, so strengths where B1, B2 and B4 all sit in the null will mostly be ones where nothing reaches C_o. |
| **P17** | Oracle detects u^⊥ less than the NLA (A.9.2 argument; premise "largely follows A.9.2", PI-accepted) | **Primary: direct reads.** NLA (L42) u^⊥ naming rate − **max**(Oracle L40, Oracle L44) u^⊥ naming rate, at matched strength. Secondary: (i) carried C_o reads; (ii) the normalised version (u^⊥ / full detection ratio). | `[PI]` | 0.45. Genuinely unsure. The Oracle injects α · unit(h), so a perturbation of the clean cell is diluted for both readers; nothing I know predicts a large gap. |

**Agent priors on gates and descriptive outcomes (pre-registered with the above; not decision rules).**

| Item | Agent prior / point expectation |
|---|---|
| G1 passes at band B (L32–L38), condition 1 or 2 | 0.65. If not at B, then at B2: a further 0.10 |
| G1 calibration shifts the grid | 0.35 (direction: ×4 up more likely than ÷4) |
| Onset s* for u^⊥ (if any) | about 1/4 (between 1/8 and 1/2). No onset within the grid: 0.30 |
| G3: median J share at k = 16 within [1%, 30%] | 0.85; point estimate 5–10% |
| G4 probe validity passes | 0.70 |
| G5 flags a leaky split (> 20% of concept-layers) | 0.25 |
| G6 drops ≥ 1 concept for the NLA (clean naming > 5% on the introspection prompt) | 0.40 (most likely: information, process, review, summary) |
| G2 for the NLA, carried C_o: full vector detected above rand-dir at a strength where the model moves | 0.50 |
| G2 for the NLA, direct L42 / Oracle, direct L40 or L44 | 0.80 / 0.70 |
| SEC l0 = 31 cross-check: R1 agrees with the fp32-residual regime (same top-1, RR within 0.05) on ≥ 90% of cells | 0.85 |

## 9. Gates (in order)

| Gate | When | Rule |
|---|---|---|
| G0 fidelity | every write and clamp | ρ, κ and the clamp bound (§4). A failure stops the run; repair and rerun; nothing scored is deleted. |
| **G1 Qwen report gate + calibration** (PI 3.8a) | smoke, first | see below |
| G2 analysis gate (PI 3.6) | analysis, per reader and kind | A reader's J / non-J split is interpreted only if, at ≥ 1 strength where the model moves (R1′ or B1 outside the null), it detects the full vector (2) above rand-dir with a CI excluding 0 |
| G3 J-share sanity | after the decompositions | median J share at k = 16 outside [1%, 30%] → flagged, nothing changed |
| G4 probe validity | before reader scoring | median leave-one-template-out AUC ≥ 0.9 and control-task AUC ≤ 0.6; otherwise R1 is not interpreted |
| **G5 split quality (PI review)** | after the decompositions; token-reader data | J_NP cosine rank of c on the pure u^⊥(ℓ) at each band layer. If c is in the top 10 for > 20% of (concept, band layer) pairs, the split is flagged "leaky", and every "non-J" claim carries "(split leaky: c J-readable in u^⊥ for x% of concept-layers)". The same rank on direct reads is reported too. |
| G6 in-run screen | clean reads | a concept with clean naming rate > 0.05 at a reader's cell is dropped from that reader's analyses (reported) |

**G1, the Qwen report gate and calibration.**
- **Setup:**
  - 6 gate concepts, conditions 1, 2 and 7: R1, R1′, R2, B1, B3;
  - plus, for the onset, all 10 dev concepts on conditions 4 and 7 (5 seeds): R1′.
- **Pass:** for condition 1 *or* 2, at its best strength:
  - mean R2 ≥ 0.25 over the gate concepts;
  - exceeding rand-dir's naming of the same concept by ≥ 0.20;
  - median B3 concept probability < 0.05 (it is a report, not a blurt).
- **Calibration (once, positive control only):** if condition 1 peaks at the grid top, the grid is multiplied by 4; if every s ≥ 1/8 drives B1 > 5 nats and the peak is at 1/32, the grid is divided by 4. Then the gate is re-measured.
- **Onset and text-reader strengths:** §4.
- **Fail → redesign rule:** re-measure once with band B2 = L24–L38.
- **Fail again → stop rule.** The report contrast is declared floor-limited:
  - M1, M2, M3 and P11a are not run as report contrasts;
  - the design is amended (P11b, P17 and R1 unaffected) and returns to the PI before any run.
- **G1 amendment:** the final grid, band, s* and the text-reader strengths are committed before the run.

## 10. Preflight (CPU, before the smoke)

- Render asserts: introspection prompt (139 tokens; 97–116, 110, 126–138, 138) and neutral prompt (19; 3–9). Spelling-token asserts for all 55 concepts.
- Leave-one-out and 194-word baseline renders; the random-clamp token pool.
- Scoring regex tests: positives, negatives, plurals, case, mismatched concepts.
- Clamp on a toy residual stream: exact restoration of the clamped coordinates, no change orthogonal to them, and the removed-norm logging.
- The new generation path: batch-wide per-row injection at 97–116 only (never at generated positions); cache-free top-p sampling bitwise reproducible with a fixed seed.
- Import boundary: the 005 module is reader-side (it imports `jspace`, `readers`); no label module imports it.
- Already done: `jspace` and `NLArl` tests (bbef824, 506f0f1); `nla-rl` files verified (11/11 sha256); concept screen.

## 11. Smoke (after the PI's go only)

1. G1 with calibration, onset and (if needed) B2.
2. 2 dev concepts (gel, earth) × the 3 provisional text strengths × every condition, reader and kind, end to end (both phases).
3. G0 on every write; G5 on the dev concepts.
4. Determinism (bitwise teacher-forced outputs; identical samples).
5. SEC l0 = 31 cross-check on the gate concepts.
6. Throughput per component, then a re-estimated budget.
7. Raw readouts inspected by hand and pasted into the smoke note.
8. Peak GPU memory in both phases.

The smoke is not the run. It stops, and the note goes to the PI.

## 12. Budget (estimate; re-estimated from the smoke)

| Component | Count (55 concepts) | Estimate |
|---|---|---|
| Concept vectors | (101 + 194) renders × 10 layers | < 5 min |
| Atoms, pursuit and sensitivity variants | 10 layers × (55 concepts + random controls) × 5 variants | ≈ 20 min |
| Probe data | 295 words × 6 templates | ≈ 10 min |
| Model side | 55 × (≈ 92 teacher-forced introspection + 92 neutral + 54 sampled sets), batched per concept | ≈ 0.5–1 h |
| Sensitivity re-scores | 55 × 5 variants × 2 conditions × 6 strengths | ≈ 15 min |
| Token readers + probe | from captures | < 15 min |
| NLA | 55 × 4 conditions × 3 strengths × 2 reads × 4 samples = 5,280 samples | ≈ 2.5 h at the measured 64 per 107 s; less with larger batches |
| Oracle | 55 × 4 × 3 × 4 reads (L40/L44 × direct/carried) × 4 samples = 10,560 samples | ≈ 2–4 h (Oracle outputs are shorter; unmeasured) |
| Synonym judge | ≈ 16k readouts | ≈ $6 |
| **Total** | | **≈ 6–8.5 h GPU** (could be 2× off) |

- **No unreviewed overnight runs.** Two daytime sessions on one approval: Phase A (model side + Oracle) and Phase B (NLA).
- **Reduction order** if the smoke puts the total above 8 h `[PI]`:
  1. Oracle carried C_o reads at 2 samples;
  2. drop Oracle carried C_o at L44;
  3. text readers at 2 strengths (s*, 4s*).
  - NLA samples stay at 4. The 45 test concepts stay.
- Optional, off by default `[PI]`: condition 1 for text readers (+25% text-reader time).

**Power.** With 45 test concepts, the cluster-bootstrap CI half-width on a rate difference is roughly 0.08–0.12 (v1: 0.15–0.25 with 12).

## 13. Outputs

- `scripts/005_planted_concepts/planted.py` (+ `_analysis.py`, `screen_concepts.py`), reader-side.
- `results/005_planted_concepts/{analysis/*.json, figures/, manifest_<stage>.json, meta_<stage>.json}` with `registry.manifest_block()`.
- Raw: `outputs/005_planted_concepts/` (gitignored; mirrored with sha256):
  - fidelity and clamp-removal records;
  - every reader sample and every capture;
  - decompositions.
- Test-concept outcomes are written but not read until the analysis code is committed (§4.R).

## 14. What this cannot show

- **Planted content only** (PI 3.7).
- **The J-space here is one construction:** J_NP + pursuit + k = 16 + folded atoms. Others are not interchangeable (c1). G5 measures how leaky this split is for the concept's own token.
- **u^⊥ is "not in J-space by this construction"**, not "invisible to the model": the model may still *use* it (R2, B1–B4) without reporting it.
- **Report is elicited by a prefill that asserts detection.**
- **A carried read at C_o at L40–44 sees only what attention has delivered by L44.** The model's report at C_o draws on layers up to 63. A reader can miss content that arrives later; G2 guards the interpretation.
- **Text-reader naming is single-word matching** (synonyms only in the flagged column).
- **One prompt, one band, one model.**

## 15. Changes from v1 (PI review 2026-10-04; disposition in `docs/reviews/2026-10-04_pi_review_checks_005/`)

1. **Read kinds:**
   - v1 called C_q reads at L40–44 "model-carried", but C_q lies in the injected span and the stream still carries the band writes there.
   - Now: direct (P17 primary), carried at **C_o** (P11 primary), and "injected position" (C_q; token readers and probe only).
   - Text readers no longer read C_q after propagation.
2. **Strict P11a/P11b primary:** "does not report" needs R1′ within the null as well as naming at the random level; "does not change behaviour" needs B1, B2 and B4 within the null. v1's versions are secondary.
3. **Text-reader strengths** are chosen after G1, bracketing the u^⊥ onset ({s*/4, s*, 4s*}); v1 fixed {1/16, 1/4, 1}.
4. **Concept screen** (time, line, path excluded at 5%), **dev 10 / test 45** (v1: 12 / 12); text readers on conditions 2, 3, 4, 7 only; NLA 4 samples (v1: 8).
5. **Random-clamp null** (10r, 11r) with removed-norm logging. M2 is interpreted only against it. v1's "expected near 0" removal claim was wrong.
6. **G5 split-quality gate**; **G6** in-run screen.
7. **Leave-one-out baseline** over the other 100 concepts is primary; the 194 words are a sensitivity check.
8. **M3** (natural norm) added; conditions 5 and 6 now have a contrast.
9. **B4** (same-forward KL over the prefill) added.
10. **rand-J** tokens from the baseline words only; rand-dir with 5 seeds on the model side, forming the nulls.
11. **Oracle k = 4** = 4 samples (resolved from `readers.OracleLens` and 001/002).
12. P17's premise: accepted as "largely follows A.9.2" (PI); no query to the checkpoint author.

## 16. Overnight dev-only pilot (PI go 2026-10-04, after the smoke) — pre-registered before it runs

*Why.* The smoke triggered the G1 stop rule: sampled naming is floor-limited on Qwen, while the paper's rank / log-prob measure responds strongly. The u^⊥ onset also sat at the bottom of the grid, so P11a/b had no "model silent" strength. The design must be amended before any run. The PI asked for the GPU to be used overnight. This pilot gathers the **development** data the amendment needs. It is exploratory and descriptive: **no P-test, no prior is scored, and no test-concept outcome is produced.**

**Scope**
1. **Concepts:** the 10 dev concepts only.
2. **Split quality for all 55 concepts:** decompositions and the G5 rank of c on the pure u^⊥ at L24–38 and L40/42/44. This is material screening, not an outcome: no model output on a test concept is computed.
3. **Bands:** B (L32–38) and B2 (L24–38).
4. **Prefills:** the protocol's `default` and its `word` variant (`… The thought is about the word "`).
5. **Grid:** extended to s ∈ {1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1}.
6. **Model side:** conditions 1–11r as in §4, teacher-forced (R1, R1′, B1–B4) for every band × prefill. Sampled naming R2 for conditions 1–4 and 7 (seed 0).
7. **Readers (default prefill, both bands):**
   - text readers on conditions 2, 3, 4, 7 at s ∈ {1/256, 1/64, 1/16, 1/4, 1}: NLA 4 samples (L42), Oracle 4 samples (L40, L44), direct and carried-C_o;
   - token readers and probes on every condition and strength, all three read kinds;
   - clean reads with 32 samples.
8. **Outputs:** `outputs/005_planted_concepts/pilot/<run_id>/` and `results/005_planted_concepts/analysis/pilot_summary.json`.

**What it informs (amendment decisions, all on dev data):**
- the report measure;
- band;
- prefill;
- grid range;
- the text-reader strengths;
- a per-concept G5 exclusion rule;
- the clamp's usefulness against its null;
- the full-run budget.

**What it cannot inform.** Any P-contrast verdict: dev concepts decide nothing (§8), and the test set stays untouched.

### 16.1 Questions the pilot answers, what each answer changes, and the agent's priors (pre-registered)

All on the 10 dev concepts; descriptive. "Null" means the rand-dir 95th percentile (5 seeds × 10 concepts) at the same strength, band and prefill.

| # | Question | Measure | How the answer changes the amendment | Agent prior |
|---|---|---|---|---|
| Q1 | Is the paper's rank / log-prob report measure usable on Qwen where sampled naming failed? | Median ΔR1′ above the null at ≥ 3 strengths for condition 1 *and* condition 2 | Yes → R1′ (with R1) becomes the primary "report" measure; sampled naming R2 becomes secondary. No → the model-side report contrasts stay floor-limited and 005 keeps only behaviour + readers. | yes: 0.90 |
| Q2 | Does the `word` prefill remove the default-answer prior ("apple") that blocks naming? | Mean R2 at the best strength for condition 1 or 2 ≥ 0.25 under `word` | Yes → the `word` prefill is adopted, and naming returns as a co-primary measure. No → keep `default`; naming stays secondary. | yes: 0.30 |
| Q3 | Which band? | Best-strength median ΔR1′ for conditions 1 and 2, and median B1 at that strength, B vs B2 | B stays unless B2's condition-2 ΔR1′ exceeds B's by ≥ 1 nat without a larger B1 | B2 chosen: 0.50 |
| Q4 | Where does the non-J part start to act? | Onset s* (§4 rule) on the extended grid, per band | s* inside the grid → the text-reader strengths become {s*/4, s*, 4s*}. s* still at the bottom (1/256) → strict P11a/P11b have no qualifying dose at practicable strengths, and are dropped or reframed in the amendment | s* above 1/256 in at least one band: 0.60 |
| Q5 | Do the readers see anything, and where? (P11 / P17 preview, not a test) | NLA and Oracle naming rates by kind (direct vs carried C_o), condition and strength, against their clean-read and rand-dir rates | NLA naming at C_o ≈ 0 even where the model's R1′ moves strongly → P11a is near-unwinnable as posed, and the amendment focuses on direct reads (P17) and R1. Clean-read rates fix the G6 baseline on this prompt. | NLA names c at C_o for the full vector at some strength: 0.40; at direct reads: 0.80 |
| Q6 | Is the clamp informative? | c-clamp vs random-clamp reduction of ΔR1′ for u^⊥ at strengths where u^⊥ is above the null | Not distinguishable on dev → M2 is dropped (or redesigned) in the amendment; distinguishable → M2 stays | distinguishable: 0.25 |
| Q7 | How many concepts have a leaky split? (all 55; material only) | c in J_NP top-10 on the pure u^⊥ at ≥ 2 layers of the band in use. **This exclusion rule is fixed here, before the numbers are computed.** | Concepts meeting it are excluded from every "non-J" contrast in the full run | ≤ 5 of 55 leaky: 0.60 |
| Q8 | Does the concept-vector baseline matter? (all 55) | cos(leave-one-out u, 194-word u), per concept and layer | Leave-one-out stays primary regardless (PI review); a median < 0.9 makes the 194-word sensitivity mandatory in the full report | median < 0.9: 0.70 (smoke: 0.86) |
| Q9 | What will the full run cost? | Measured throughput per component | Sets the full-run budget and the reduction order | — |

**What we learn overall.**
- Whether 005's model side can test the privilege question on Qwen at all, and with which measure, band and prefill.
- Whether the reader comparison has a usable dose window, where the model is silent but the planted content is present.
- Whether the clamp control can work.
- What the full run costs.

**What we cannot learn.** Any P-contrast verdict, or anything about test concepts. Q5's reader numbers and any J-vs-non-J pattern on these 10 dev concepts are previews and will be labelled as such.

### 16.2 What runs next, in order, and why (updated 2026-10-04 ~20:20, before the pilot launches)

**Why the plan changed after the smoke (smoke1, dev concepts, descriptive).**

| Smoke finding | Consequence for what runs next |
|---|---|
| G1 failed at both bands: sampled naming 0% (L32–38) and best 8% (L24–38), against a strong default answer ("apple" is top-1 at the quote in the clean run) | Stop rule: no full run until amended. The pilot measures the paper's rank/log-prob report measure and the `word` prefill (Q1, Q2) |
| The pure J-lens vector lifts the concept to median rank 2 (+8.5 nats at s = 1/4, L24–38); random directions +0.4 | R1′ is likely a usable report measure (Q1) |
| The u^⊥ onset sat at the grid floor (s* = 1/32) | The pilot extends the grid down to 1/256 (Q4) |
| earth's u^⊥ was still J-readable as "earth" (J_NP rank 2–13 at 4 layers); 2% of concept-layers overall | A per-concept leak rule, fixed in §16.1 Q7, screened on all 55 concepts (Q7) |
| The c-clamp did not beat the random clamp on gel and earth | Q6 on 10 dev concepts |
| cos(leave-one-out u, 194-word u) had a median of 0.86 | Q8 |
| The SEC cross-check was skipped (band B2 starts below l0 = 31) | The pilot runs it at band B (below) |
| The NLA loader crashed in phase B (`NLArl.scaling` read a float attribute on a non-LoRA module) | Fixed. The smoke's NLA phase was resumed from its stored inputs (`planted.py nla-resume`); nothing else changed |

**Run order (one process per step; every block checkpointed, so a crash resumes without repeating finished blocks).**

| Step | What runs | Why | Estimate |
|---|---|---|---|
| 0 | Smoke NLA phase, resumed (48 reads × 4 samples + clean reads) | Completes smoke §11 item 2 (end to end, both phases) | ~3 min (running) |
| 1 | Pilot setup: renders for both prefills, clean forwards, lenses. Material reused from smoke1 (identical model and renders) | — | ~2 min |
| 2 | Split quality for all 55 concepts at L24–38 and L40/42/44: G5 ranks, J shares, sensitivity (k 10/25, LS, raw atoms, 194-word baseline), cos(LOO, base), leaky lists per band | Q7, Q8; material only, no model output on a test concept | ~3 min |
| 3 | Model side, 10 dev concepts × {B, B2} × {default, word} × 9 strengths: conditions 1–9 (rand-dir 5 seeds), 10, 10r, 11, 11r, teacher-forced (R1, R1′, B1–B4); sampled naming (8 samples) for conditions 1–4 and 7; onset per band × prefill | Q1–Q4, Q6 | ~70–90 min |
| 4 | Token readers (J_NP, J_CB, logit lens) and probes (default prefill, both bands, every non-clamp condition and strength, three read kinds); G4 | Q5 (token side) | ~5 min |
| 5 | Oracle reads (default prefill, both bands): conditions 2, 3, 4, 7 × s ∈ {1/256, 1/64, 1/16, 1/4, 1} × {direct C_q, carried C_o} × {L40, L44} × 4 samples = 6,400 samples, plus clean reads (32 samples per kind and layer) | Q5 (Oracle) | ~35 min |
| 6 | SEC l0 = 31 cross-check: band B, default prefill, gate concepts, conditions 1–4 and 7 (seed 0), all 9 strengths. R1, R1′ and top-1 compared with the fp32-residual regime | Completes the smoke item skipped at B2. Tells us whether the faster regime can be trusted in the full run | ~3 min |
| 7 | NLA reads (subject freed): the same items at L42 × 4 samples = 3,200 samples, plus clean reads | Q5 (NLA) | ~30 min |
| 8 | CPU analysis → `results/005_planted_concepts/analysis/pilot_summary.json`; pilot note answering Q1–Q9 against the pre-registered rules; the amendment draft (a proposal; the PI decides) | — | after the run |

**Total GPU ≈ 2.5–3 h.** It is dev-only throughout; no test-concept outcome is produced or read. The instance is **not** stopped afterwards unless the PI says so.

**Implementation notes recorded before the run.**
- Sampling batches 192 rows per forward (smoke: 96).
- Token-reader records keep only each row's own concept rank; the all-concept mismatched baseline comes from text-reader samples.
- G4's control is the sample-level label permutation (a Hewitt–Liang type-level control is uninformative for single-word probes; see the smoke note).
- The clamp's removed share is logged as ‖QQᵀ(h − h_clean)‖ / ‖h − h_clean‖ at every clamped layer, which at band layers includes that layer's write.
- Every band × prefill block's rows are saved (the smoke lost band B's G1 rows).

### 16.3 Step 7b: three additions from the PI-forwarded review (queued after step 7; pre-registered 2026-10-04 ~20:30, before they run)

The pilot was already running when the review arrived, so these run as a separate process (`scripts/005_planted_concepts/pilot_7b.py`) once the pilot's NLA phase ends.
- **Setting:** dev concepts only, default prefill, bands B and B2, the 9-point grid. Teacher-forced throughout.
- **Cost:** about 15 min GPU.
- **Reproducibility:** vectors, random seeds, clamp token sets and N̄ are recomputed by the same code, so they are identical to the pilot's.

**7b.1 Re-entry clamp (why).**
- The smoke showed the c-clamp no better than the random clamp. Both clamp to h_clean, so both delete part of the *carried injection* itself (§4).
- In an additive residual stream, h − h_clean = A + M, where:
  - A(ℓ) = Σ over band layers ℓ′ ≤ ℓ of Δ_ℓ′, the raw injection, at the injected positions only;
  - M = what the model's blocks wrote differently in response.
- Clamping c's J coordinates to **h_clean + A** removes only M's component in those coordinates. That is the re-entry the paper's clamp is meant to block; the raw injection's own component is kept.

**One extension (agent).**
- Near the output, c's J atom approaches c's unembedding direction, so a clamp that runs to L62 can block the route to saying c mechanically, not by stopping workspace re-entry.
- So the clamp is run over two ranges, each with its own random-token null:

| Condition | Clamp | Layers |
|---|---|---|
| 12 / 12r | u^⊥ (matched) + re-entry c-clamp / random clamp | band start to **L62** (as the design) |
| 13 / 13r | u^⊥ (matched) + re-entry c-clamp / random clamp | band start to **L44** (the readers' layers) |

- Conditions 4, 10 and 10r are re-run in the same process as within-run references.
- Same T_c and random token sets as 10/10r.
- **Logged:**
  - the share of M removed, ‖QQᵀM‖ / ‖M‖, per layer, and the residual;
  - cos(v_c(ℓ), g ⊙ W_U[c]) per layer for c's own token, to show where a clamp turns mechanical.
- **Caveat:** passive leakage, meaning the raw u^⊥'s own projection onto later layers' c atoms, is kept as part of A. G5 measures it at band layers.

**Q10 (rule).** At u^⊥'s best strength per concept (by R1; concepts where u^⊥ is within the null are not evaluable): does the re-entry c-clamp lower ΔR1′ more than the re-entry random clamp? The bootstrap 95% CI over evaluable dev concepts must exclude 0.

| Outcome | Reading |
|---|---|
| Full range (12) specific, band-limited (13) not | u^⊥'s report needs c's J coordinates only late, near the output. Consistent with the output-direction confound; **no** evidence for workspace re-entry |
| Band-limited (13) specific | u^⊥'s effect needs re-entry into c's J coordinates within the read band (supports the paper's mediation claim on Qwen; dev only) |
| Neither | u^⊥ reaches the report without writing into c's J coordinates beyond a random set's (against mediation via c's J coordinates; dev only) |

Agent priors: 12 specific **0.40**; 13 specific **0.25**.

**7b.2 Layer sweep at the report position (why).**
- The non-J part moves the report log-prob from the bottom of the grid, and the C_o reads stop at L44 while the model reports from L63 (§14).
- **Setup:** conditions 2, 3, 4 and 7 (seeds 0–4), plus clean, at every strength; h(C_o) recorded at **L32–L62**.
- **Per layer:** the J_NP cosine rank, logit-lens rank and R_CB standard rank of c (best single-token form).
- **Q11:** under u^⊥, at the strengths where u^⊥ moves R1′ above the null, does c become J_NP top-10 at C_o at some layer ≤ L62 (median over evaluable concepts), below the rand-dir 5th-percentile rank at that layer?
  - **Yes:** re-entry is visible at the report position, with the first such layer reported.
  - **No:** the effect reaches the output without showing in J at C_o.
- Agent prior: yes **0.55**.

**7b.3 R-lens as an extra token reader.**
- **Setup:** the R_CB standard readout (norm(h J_Rᵀ) W_Uᵀ, WSB RLens) rank of c, computed on:
  - the pilot's stored captures (C_q and C_o at L40/42/44);
  - direct reads (all non-clamp conditions, default prefill);
  - the sweep in 7b.2.
- Descriptive: J_NP vs R_CB vs logit-lens consistency, with no rule.

**7b.4 SAE/BSF: not added.** `ceselder/qwen36-27b-bsf-l42-1b` @ 52b898a5 is a custom ~16 GB format (blocks_Q.pt 5.4 GB, sasa.pt 10.7 GB, ZCA whitening) that needs its own encoder, and the SAE arm is PI-deferred (A01 C16). It is left for the full run if the PI wants it.

### 16.4 Step 8: where the planted concept reaches the report position (PI go 2026-10-04: "continue with something extra … which would give more info"; pre-registered before it runs)

**Why.**
- P11a/P11b ask whether a reader sees, at the report position C_o, content the model does not report.
- The pilot reads C_o only at L40–44, while the model reports from L63; the NLA can read only L42.
- If the planted concept reaches C_o only after L44, P11 at C_o is structurally untestable for the NLA, and the amendment must move P11's read position.
- Step 7b's sweep tracks the concept's *token* rank at C_o (J-lens, logit lens, R-lens). Step 8 adds:
  - a **reader-free** measure of concept information at C_o across layers;
  - the **Oracle at its later on-contract layers**.

**8a. Run (dev concepts, default prefill, bands B and B2, teacher-forced, `scripts/005_planted_concepts/pilot_8.py`).**
- **Rows:** conditions 2 (full), 3 (J part), 4 (non-J part), 7 (rand-dir, seeds 0–4) at s ∈ {1/256, 1/64, 1/16, 1/4, 1}, plus clean. h(C_o) recorded at **L32–L62**.
- **Probes:** a per-concept logistic probe (as in §6: 6 natural templates, the 101 concepts + 194 baseline words, C = 1.0, standardised), trained at **each layer L32–L62**.
- **Detection of c at C_o at layer ℓ:** Δprobe_c = probe_c(x) − probe_c(x_clean) exceeds the largest Δprobe_c over the 5 rand-dir seeds for the same concept, dose, band and layer.
- **Oracle at C_o:** L48, L52, L56, L60 (on-contract), conditions 2, 3, 4, 7 (seed 0) × the 5 doses × both bands × 4 samples, plus clean reads (32 samples per layer).

**Q12 (reader-free arrival).**
- Per condition × dose × band: the first layer at which at least half of the dev concepts are detected at C_o.
- Report the full curve: detection rate by layer.
- **Decision mapping for the amendment:**

| Outcome | Amendment |
|---|---|
| The full vector and the non-J part arrive at C_o by L42 at doses inside the strict P11 window (model's ΔR1′ within the null) | P11 at C_o is testable as designed |
| They arrive only after L44 | P11 for the NLA moves to a position it can read (proposal: the last question-turn token, read after propagation, with the caveat that it is not attention-only), or P11 is restricted to the Oracle at late C_o layers |
| They never arrive (at doses where the model's report is unmoved) | "The model doesn't report it" coincides with "it isn't at the report position", and P11a as posed cannot succeed |

- **Agent priors:**
  - full vector at s = 1/4 arrives at C_o by L44: **0.50**;
  - the non-J part arrives no later than the J part (s = 1/4): **0.60**;
  - inside the strict P11 window, any condition arrives by L44: **0.20**.

**Q13 (Oracle at late layers).**
- Does the Oracle name c at C_o at L48–60, with mean naming over dev concepts > 0.10 and above rand-dir, at some dose:
  - for the full vector? Agent prior **0.45**;
  - for the non-J part? Agent prior **0.25**.
- **Mapping:**
  - **yes** → late-layer Oracle reads at C_o become the Oracle's P11-type comparison in the amendment;
  - **no** → neither text reader sees planted content at the report position at any on-contract layer.

**8b. Power analysis (CPU only).**
- From the pilot's dev data: the between-concept SD of each previewed contrast (M1-like, R1-like, P17-like, P11a-like).
- The expected cluster-bootstrap CI half-width with **45 test concepts**, and the effect of 4 vs 8 text-reader samples (within-concept binomial variance).
- A planning number with no rule.

**Cost:** about 50 GPU minutes after step 7b; no test concept is touched.

### 16.5 Step 9: robustness of the two headline dev findings to the J-space construction (PI go 2026-10-04 "continue with something extra"; pre-registered before it runs)

**Why.**
- Two dev findings will shape the amendment and the PI's reading of 005:
  - **F1:** at matched norm, the J part beats the non-J part at middle doses and loses at high doses, with similar peaks;
  - **F2:** the re-entry clamp is specific only when it reaches L62, not when it stops at L44.
- Both rest on one construction: k = 16, NNLS refit, folded atoms, leave-one-out baseline. §3 lists the sensitivity constructions; this step runs them on these two findings.

**Run (dev concepts, band L32–38, default prefill, 9-point grid, teacher-forced; `scripts/005_planted_concepts/pilot_9.py`).**
- **Five variants:** k = 10; k = 25; LS refit (k = 16, the A.8 span reading); raw-W_U atoms (k = 16); 194-word baseline (k = 16).
- **Per variant:** conditions 3 (J part, matched), 4 (non-J part, matched), and 12/12r and 13/13r (re-entry clamp to L62 / to L44, each with its random-token null).
- The clamp token set T_c comes from that variant's own supports; its atoms are folded, except raw for the raw-atom variant.
- The null for "the non-J part has an effect" is the pilot's rand-dir rows (variant-independent).

**R9a (F1).**
- Per variant: the median over concepts of ΔR1′(J) − ΔR1′(non-J) at s ∈ {1/16, 1/8} (middle) and {1/2, 1} (high).
- **The pattern holds** if both middle-dose differences are > 0 and both high-dose differences are ≤ 0.
- Also reported: each part's peak ΔR1′ over doses.
- Agent prior: the pattern holds in ≥ 4 of 5 variants, **0.55**.

**R9b (F2).**
- Per variant, the Q10 rule:
  - the full-range re-entry clamp beats its null (bootstrap CI over evaluable concepts > 0);
  - the band-limited one does not (CI includes 0, or lies below it).
- Agent prior: the pattern holds in ≥ 4 of 5 variants, **0.65**.

**Reading.**
- Holding in ≥ 4/5 variants: the finding is robust to the construction choices in §3.
- Otherwise: the report names the variants it fails in, and the amendment flags the finding as construction-dependent.

**Cost:** about 15 GPU minutes. Dev concepts only.

### 16.6 Step 10: where the re-entry clamp starts to act, and whether the report measure is concept-specific (PI go 2026-10-04 "continue with something extra"; pre-registered 2026-10-05 ~00:10 UTC, before it runs)

**Why.**
- Two of the PI's substantive amendment decisions rest on dev evidence with a gap in it.
- **A8 (M2), the clamp:**
  - The re-entry clamp is specific when it runs to L62 and not when it stops at L44 (Q10; step 9: 5/5 constructions).
  - Between those layers, the concept's own J atom turns toward its output direction (median cos over dev concepts 0.47 at L44, 0.58 at L48, 0.67 at L50, 0.77 at L52, 0.85 at L56, 0.92 at L62).
  - Where the specificity starts decides between two readings:
    - the full-range result is "blocking the output direction" (A8 as drafted);
    - there is a mid-depth re-entry effect that M2 should target.
- **A1 (R1′ primary) and A3 (band), the report measure:**
  - R1′ is judged against a random-direction null at matched norm.
  - Qwen's default answer at C_o is a canned word ("apple"). Any perturbation that displaces it may raise every other word's log-prob.
  - Nothing yet checks that the rise is specific to the planted concept, or that band L24–38's larger rise is.
- Both are cheap teacher-forced checks on the 10 dev concepts (`scripts/005_planted_concepts/pilot_10.py`).

**10a. Clamp end-layer sweep (Q14; informs A8).**
- **Rows:**
  - condition 4's vector (the non-J part) injected at the band, as in the pilot;
  - the re-entry clamp (reference h_clean + accumulated injection, §16.3) from the band's first layer to an end layer E ∈ {44, 46, …, 62};
  - the clamp uses either the concept token set T_c (c) or its size-matched random token set (r), the same sets as step 7b;
  - bands B and B2, default prefill, 9-point grid, plus unclamped condition 4.
- E = 44 and E = 62 repeat step 7b's 13/13r and 12/12r (part of gate G10 below).
- **Measure (the Q10 definition):**
  - extra reduction = ΔR1′(r-clamp) − ΔR1′(c-clamp), at each concept's best non-J dose;
  - over evaluable concepts, i.e. those whose non-J ΔR1′ is above the rand-dir 95th percentile at that dose;
  - bootstrap 95% CI over concepts.
- **E\*** = the smallest E whose CI lies above 0 at E and at every larger E on the grid.
- Mapping (band B primary; B2 reported the same way):

| E* | Reading | Amendment |
|---|---|---|
| 56–62 | specificity only where the J atom is close to the output direction (median cos ≥ 0.85) | A8 as drafted |
| 52–54 | intermediate (cos 0.77–0.82) | reported; A8 left to the PI |
| 46–50 | a mid-depth window: blocking the concept's J subspace removes the report effect while its atom is still far from the output direction (cos 0.52–0.67) | M2's end layer becomes E* (replacing L44); A8 reworded |
| none, or 44 | contradicts step 7b (E = 62 and E = 44 repeat its conditions 12 and 13) | stop and investigate |

- Also reported: the extra-reduction curve against E, next to the cos curve.
- **Agent priors (band B):**
  - E* in 56–62: **0.55**;
  - E* in 52–54: **0.20**;
  - E* in 46–50: **0.15**;
  - none or other: **0.10**.

**10b. Concept specificity of the report measure (Q15; informs A1 and A3).**
- **Rows:** the pilot's teacher-forced rows, re-run:
  - conditions 1 (pure J-lens atom), 2 (full vector), 3 (J part), 4 (non-J part) and 7 (rand-dir, seeds 0–4);
  - × 9 doses × bands B and B2, default prefill.
- **Recorded:**
  - log-probs at C_o of the quote spellings of a word set W;
  - W = the 10 dev concepts + the 78 of the protocol's 194 baseline words whose quote spellings are single tokens (test and excluded concepts are not in W);
  - the clean top-10 tokens at C_o and each row's top-10 (raw readouts).
- **Per row:** ΔR1′(w) for every w ∈ W (log-sum-exp over quote spellings, injected − clean).
- **Measures:**
  - **ID:** the planted concept has the largest ΔR1′ among the 10 dev concepts (chance 0.1).
  - **SM** = ΔR1′(c) − median over W∖{c} of ΔR1′(w), the specificity margin. Bootstrap 95% CI over the 10 concepts.
  - Also reported: the planted concept's percentile within W, and the generic rise (the median over W∖{c} of ΔR1′(w)).
  - *Clarification, committed with the analysis code before any step-10 result was read:* "SM" in the rule and in the A3 comparison is the mean over concepts (the statistic whose bootstrap CI is computed). ID is a rate over the 10 concepts; for condition 7 it is over the 50 concept × seed rows, and SM there is averaged over seeds within each concept first.
- **Rule (per band × condition):** R1′ is *concept-specific* if, at ≥ 3 doses where the condition's median ΔR1′ exceeds the rand-dir 95th percentile:
  - ID ≥ 5/10; and
  - SM's CI lies above 0.
- **Mapping:**
  - Condition 2 specific in both bands: A1 stands (R1′ primary).
  - Condition 2 not specific in a band: A1 changes for that band. SM becomes the primary report measure and raw R1′ secondary (a new amendment item).
  - A3: band L24–38's larger effect counts as concept-specific if, for condition 2 at s ∈ {1/16, 1/4}, its SM ≥ band B's and its ID ≥ band B's. Otherwise it is "partly generic", an argument for L32–38.
  - Condition 7 (random) is a sanity check: ID ≤ 3/10 at every dose is expected.
- **Agent priors:**
  - condition 2 specific: B **0.65**, B2 **0.70**;
  - condition 4 specific: B **0.35**, B2 **0.45**;
  - condition 1 specific in both bands: **0.90**;
  - B2 at least as specific as B (the A3 rule): **0.60**;
  - random directions at chance (ID ≤ 3/10 at every dose, both bands): **0.90**.

**Gate G10 (consistency, checked before any reading).**
- Rows that repeat stored rows must reproduce their stored ΔR1′:
  - 10b: every row vs the pilot's blocks;
  - 10a: E = 44/62 vs step 7b.
- Pass: median |difference| ≤ 0.05 nats and 95th percentile ≤ 0.25 nats. This allows for the bf16 batch-composition floor; chunks differ from the original runs.
- On failure, step 10 is reported as unreliable and is used for no amendment item.

**Cost:**
- About 5,400 teacher-forced rows at ~20 rows/s: about 5 GPU minutes, about 15 minutes with setup.
- Dev concepts only. No test concept is injected or scored.

### 16.7 Step 11: is the late clamp effect specific to J coordinates, and are the readers less sensitive than the model under a matched measure? (PI go 2026-10-05 "run something extra!"; pre-registered before it runs)

**Why.** Step 10 left a caveat on each of the two findings the amendment leans on hardest.
- **A8, the clamp.**
  - Blocking the concept's J coordinates removes the non-J part's report effect specifically only from about L50 on.
  - There, those coordinates are 0.67–0.92 aligned with the concept's output direction.
  - So this is not yet distinguishable from "blocking the output direction removes it".
- **A6 (P11′) and the readers-vs-model preview.**
  - "The readers are less sensitive than the model" compares the model's log-prob (a fine measure) with reader naming in 4 samples (a coarse one).
- 11a and 11b test each caveat with a matched control.
- Dev concepts only. Teacher-forced forwards or direct reads, no sampling (`scripts/005_planted_concepts/pilot_11.py`).

**11a. Output-direction clamp (Q16; informs A8).**
- **Rows:** as step 10's 10a, except that the re-entry clamp acts on the concept tokens' *output directions* g ⊙ W_U[t] (logit-lens coordinates, the same at every layer) instead of their J atoms J_ℓᵀ(g ⊙ W_U[t]).
  - The token sets are step 10's: T_c and the size-matched random T_r. The clamp rank is therefore the same.
  - E ∈ {44, 46, …, 62}, both bands, 9-point grid, plus unclamped condition 4. Those unclamped rows repeat step 10's and must pass G10's tolerance.
- **Measures:**
  - the Q14 extra reduction for the output clamp, extra_out(E);
  - the paired difference D(E) = extra_J(E) − extra_out(E) over step 10's evaluable concepts at their best doses (J values from step 10); bootstrap 95% CI over concepts.
- **Rule (band B primary):** at E ∈ {48, 50, 52}:
  - **J-specific** if D's CI lies above 0 at ≥ 2 of the three;
  - **not J-specific** if at none;
  - **inconclusive** if at exactly one.
- **Mapping:**
  - **J-specific:** M2 as drafted (the J clamp to L50 is a test of routing through J coordinates).
  - **Not J-specific:**
    - M2 must report the output-direction clamp as a mandatory control beside the J clamp;
    - the amendment's wording changes from "re-entry into J coordinates" to "routing through the concept's output-related coordinates".
  - **Inconclusive:** both are reported, and the PI decides.
- **Agent prior:** J-specific, band B: **0.50**.

**11b. Reader likelihoods on the written cell (Q17; informs A6).**
- **Readers and reads:**
  - the NLA at L42 and the Oracle at L40 and L44;
  - inputs: direct reads of the written cell, h_clean(C_q, ℓ) + s·N̄_ℓ·vec_ℓ, as in the pilot; and h_clean(C_q, ℓ) itself, the reader's baseline;
  - conditions 2, 3, 4 and 7 (seeds 0–4) × the 9 doses.
- **Forced answer position (no sampling).** The reader's answer is teacher-forced to start with a fixed prefix, and the next token is scored:
  - Oracle: prefix "-" (its own bullet format); candidates " w" and " W".
  - NLA: prefix "<explanation>\nThe text is about"; candidates " w" and " W".
  - Logits are float32 from the final-norm output and checked against the reader's own logits.
- **Per row and word w ∈ W_R:**
  - W_R = the 10 dev concepts + the protocol's baseline words whose candidates are single tokens after that prefix;
  - Δ_R(w) = LSE log P(w's candidates | injected read) − the same on the clean read.
- **Measures, as in Q15:**
  - ID among the 10 dev concepts;
  - SM = Δ(c) − median over W_R∖{c} of Δ(w).
- **Gate G11 (positive control, per reader and layer):**
  - On natural activations ("Tell me about c", last token, the reader's layer), forced-choice ID among the 10 dev concepts from raw log-probs must be ≥ 7/10.
  - A reader that fails is reported as uninformative under this measure.
- **Why an absolute threshold, decided now.** In a deterministic, near-linear forward, comparing against random directions at the same dose is scale-free: it would "detect" at every dose (the A9 trap). So onsets use an absolute size.
- **Onset:** s_X(θ) = the smallest dose at which the median over the 10 concepts of SM is ≥ θ. θ = 1 nat is primary, 0.5 nat secondary.
  - For the model: from step 10's rows (band B, R1′ at the open quote, Q15's SM).
  - For the readers: from 11b. For the Oracle, the earlier of its two layers.
- **Rule:** per condition (2 full, 4 non-J), a reader is *at least as sensitive as the model* if s_R(1) ≤ s_M(1). A reader that never reaches θ is not.
- **Mapping:**
  - For the non-J part, if a reader that passes G11 is at least as sensitive as the model:
    - the naming-based preview ("readers are less sensitive than the model") is withdrawn for that reader;
    - P11′ uses the likelihood measure as the readers' primary measure, with naming secondary (A6 reworded).
  - Otherwise the preview stands under a matched measure. P11′ stays as drafted, with a likelihood column added.
- **Caveat, stated now:** the model's dose is applied at 7 band layers × 20 positions, while a direct read sees one written cell at the same per-cell dose. That is P11′'s comparison as drafted; the perturbations are not matched.
- **Agent priors:**
  - G11 passes: Oracle **0.80**, NLA **0.55**;
  - at least as sensitive as the model, non-J part: NLA **0.35**, Oracle **0.45**;
  - at least as sensitive as the model, full vector: NLA **0.40**, Oracle **0.50**.

**Cost:**
- 11a: ≈ 3,800 teacher-forced rows, about 4 GPU minutes.
- 11b: ≈ 2,200 forward-only reads plus controls (minutes), plus loading the NLA (≈ 2 minutes).
- Dev concepts only.
