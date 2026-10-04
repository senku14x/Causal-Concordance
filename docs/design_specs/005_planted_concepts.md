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
