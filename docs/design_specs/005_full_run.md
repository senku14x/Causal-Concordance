# 005 full run under amendment 01 — run spec (pre-registered 2026-10-05, before the run)

**Status.**
- The PI adopted amendment 01 on 2026-10-05, all items including the optional A18: "adopt but branch off to a new branch and run this there so I can reverse it".
- The adoption record, this spec, the code, the outputs and the results live on branch **`005-amended-full-run`** (off `spec-v2` @ 9709712). Dropping the branch reverses all of it.

**Inputs.**
- Design 005 v2: `005_planted_concepts.md` §1–§15.
- Amendment 01: `005_amendment_01.md`, A1–A18.
- The dev pilot (§16): calibration only.
- Where they differ, the amendment wins. Where this spec makes precise something the amendment left open, it is marked **[spec]**. Every such choice was fixed before any test-concept outcome existed.

## 1. Concepts

- **Primary:** the 45 test concepts (design §2). No test concept has been injected before this run.
- The 10 dev concepts are not re-run. Their numbers are in the pilot and decide nothing.
- **Rehearsal [spec].** The same code runs first on 2 dev concepts (gel, earth), with the word set W restricted to the dev concepts + baseline words. This catches crashes and measures time. The analysis code is developed on the rehearsal's outputs and committed **before** the test run's outputs are read. The test run's log prints progress, gates and timings only.

## 2. What runs

Setting: band L32–38 (A3), the default prefill (A2), the float32-residual regime (A13), and the 9-point grid s ∈ {1/256, 1/128, …, 1} (A4).

**Phase A (subject resident).**

| Block | Rows or reads | Measures |
|---|---|---|
| A1 model side | 45 × 9 doses × conditions 1, 2, 3, 4, 5, 6, 8, 9, 10, 10r and 7 (seeds 0–4); plus 11 and 11r at s = 0 | R1, R1′ (primary, A1), B1, B2, B3, B4, top-1 at C_o; ΔR1′ at C_o for every word of W (SM, A18) |
| A2 sampled naming R2 (secondary, A1) | 45 × {1/64, 1/4, 1} × conditions 1, 2, 3, 4, 7 (seed 0); 8 samples | R2 |
| A3 clamps (M2, A8) | 45 × 9 doses × end layer E ∈ {44, 46, …, 62} × {J-atom clamp, output-direction clamp} × {concept tokens T_c, random tokens T_r} | ΔR1′, R1 |
| A4 sensitivity (design §3, A12) | 45 × 9 doses × conditions 3 and 4 × the variants k10, k25, LS refit, raw-W_U atoms and 194-word baseline | ΔR1′, R1 |
| A5 token readers + probe (A7, A9, A11, A14) | conditions 1–9 (7: 5 seeds) × 9 doses × L40/L42/L44 × kinds direct, carried C_o and injected C_q | J_NP, J_CB, logit-lens and R_CB ranks; probe decision (absolute criterion, A9) |
| A6 Oracle naming (A5, A7) | direct reads at L40 and L44; conditions 2, 3, 4 and 7 (seed 0) × {1/64, 1/4, 1}; 4 samples; clean reads with 32 samples per layer | naming rate; G6 from the clean reads |
| A7 Oracle likelihoods (A6) | direct reads at L40 and L44; conditions 2, 3, 4 and 7 (seeds 0–4) × 9 doses; the clean read; natural "Tell me about c" activations (G11) | forced-prefix likelihoods, SM (§16.7) |

- In A3, the re-entry reference is that of §16.3, and the non-J part is injected.
- A3's unclamped reference rows are condition 4 from A1.

**Phase B (subject freed).**
- B1: NLA naming at L42, as A6.
- B2: NLA likelihoods at L42, as A7.

**W (A18) [spec].**
- W is the 55 concepts plus the 78 baseline words whose quote spellings are single tokens.
- ID is computed among the 55 concepts. SM = ΔR1′(c) − the median over W∖{c}.
- For readers, W_R is the 55 concepts plus the baseline words whose space spellings are single tokens after the reader's forced prefix.

**Not run, and why:**
- the SEC cross-check (A13; already 270/270);
- the `word` prefill (A2);
- band L24–38 (A3);
- condition 1 for text readers (design default: off);
- report-position text reads (A6, A7);
- the Oracle report-position comparison at L52–60 (A6 secondary, which applies only with L24–38);
- the synonym judge. Design §7 has it as a flagged secondary column; it is deferred, and naming uses exact spelling variants.

## 3. Gates (in order)

| Gate | Rule |
|---|---|
| G0 fidelity | every write: ρ ∈ [0.9, 1.1], κ ≥ 0.99; every clamp: residual ≤ 1e-4. A failure stops the run. |
| G3 J share | median J share at k = 16 outside [1%, 30%] → flagged |
| G4 probe validity (A11) | median leave-one-template-out AUC ≥ 0.9 and median permutation-control AUC ≤ 0.6; otherwise contrast R1 is not interpreted |
| G5 / leaky (A10) | the rule fixed in §16.1 Q7, recomputed on this run's vectors: c in the J_NP top 10 on the pure u^⊥ at ≥ 2 band layers. Leaky concepts are excluded from every non-J contrast. |
| G6 in-run screen | a concept whose clean-read naming rate at a reader's cell exceeds 0.05 is dropped from that reader's naming analyses |
| G2 analysis gate | a reader's J / non-J naming split is interpreted only if, at ≥ 1 reader dose where the model moves (median ΔR1′ or B1 above the rand-dir 95th percentile), its naming rate for the full vector exceeds rand-dir's with a CI excluding 0 |
| G11 likelihood control **[spec]** | per reader and layer: 55-way forced-choice ID on the natural activations of the 45 test concepts ≥ 0.5 (chance 0.018). A reader-layer that fails is excluded from the likelihood analyses. |

## 4. Contrasts, statistics and predictions

**Method.**
- All contrasts are on the test concepts.
- Each statistic is computed per concept, then averaged over concepts. CIs are cluster-bootstrap over concepts (10,000 draws, seed 20261004).
- **Supported** = the 95% CI excludes 0 in the predicted direction.
- **P-family** (Holm): P11′ (NLA) and P17. Every other contrast is reported with its CI but outside the family.

**Priors.**
- The registered agent priors are from df56646, before any 005 GPU run.
- The updated agent priors were written now, after the dev pilot and before any test outcome. They are for calibration only; the scorecard keeps both.

| ID | Statistic **[spec where marked]** | Predicted | Registered prior | Updated prior |
|---|---|---|---|---|
| M1 | Per concept: the peak over doses of ΔR1′ for the J part (condition 3) minus that for the non-J part (4). The R1 version and the per-dose curves are reported. | J > non-J | 0.85 (stated on R1) | 0.25 (dev: similar peaks, crossing order) |
| M2 (A8) | The J re-entry clamp to L50: ΔR1′(random-token clamp) − ΔR1′(concept clamp), at each evaluable concept's best non-J dose (by R1). Evaluable: the non-J ΔR1′ is above the rand-dir 95th percentile at that dose. **Control [spec]:** the output-direction clamp to L50; D = extra_J − extra_out. Both end-layer curves are reported. | > 0 | 0.30 | 0.65 (J clamp specific); 0.40 (D > 0) |
| M3 | Per concept: the mean, over the doses where the full vector's median ΔR1′ is above the null, of ΔR1′(5) − ΔR1′(6) | J-natural > non-J-natural | 0.60 | 0.10 (dev: the non-J part leads at natural scale) |
| R1 | Per concept, direct reads at L42, at {1/64, 1/4, 1}: probe detection of u^⊥ (absolute criterion) minus the NLA's naming rate, mean over doses | ≥ 0 (the probe detects at least as often) | 0.70 | 0.85 |
| R2 | Per concept: is there a dose where B1, B2 or B4 of u^⊥ is outside the null while its R2 and ΔR1′ are inside? The share of concepts. | descriptive | 0.50 | — |
| **P11′ (NLA)** (A6) | Per concept: the onset index (first grid dose at which SM ≥ 1 nat; "never" = 9) of the NLA's likelihood on direct reads of u^⊥ at L42, minus the model's onset index (SM at C_o); mean over concepts | < 0 (the NLA detects at a lower dose) | — (new in A6) | 0.55 |
| P11′ (Oracle) | The same, with the earlier of the Oracle's L40 and L44 onsets | < 0 | — | 0.85 |
| **P17** | Per concept: the mean over {1/64, 1/4, 1} of the NLA's naming rate (L42, direct, u^⊥) − max(Oracle L40, Oracle L44) | > 0 (the Oracle detects less) | 0.45 | 0.05 (dev: the Oracle names u^⊥ more) |

**Secondary, reported but not in any family:**
- P17 on the likelihood measure (SM at matched dose);
- M1 per dose;
- ID among the 55 concepts;
- the clamp curves;
- token-reader ranks by kind;
- the sensitivity variants;
- the rand-J and rand-⊥ controls.

## 5. Outputs

- **Code:**
  - `scripts/005_planted_concepts/full_run.py` (both stages);
  - `scripts/005_planted_concepts/full_analysis.py`;
  - the report and page generators.
- **Raw:** `outputs/005_planted_concepts/run/<run_id>/` (one file per block; resumable).
- **Results:** `results/005_planted_concepts/analysis/full_summary.json` and `results/005_planted_concepts/manifest_run.json`. A report goes to `docs/reports/005_full_run.md`, and a section is added to the report page.

## 6. Budget

Measured rates from the pilot:

| Component | Estimate |
|---|---|
| Model side | ≈ 6 min |
| Sampled naming | ≈ 35 min |
| Clamps | ≈ 12 min |
| Sensitivity variants | ≈ 3 min |
| Token readers + probe | ≈ 5 min |
| Oracle naming | ≈ 22 min |
| Likelihoods | ≈ 5 min |
| NLA load + naming | ≈ 20 min |
| **Total** | **≈ 1.75 GPU h**, plus the rehearsal (≈ 15 min) |

## 7. What this cannot show

Design §14 applies unchanged. In addition:
- The J-space privilege tests (M1, M3) are partly tautological for the J part (design §1).
- The clamp's late effect is mostly output-direction blocking on dev (§16.7 Q16); M2 is read with its control.
- P11′ compares one written cell with a report that integrates many cells. It is not a perturbation-matched comparison (§16.7).
