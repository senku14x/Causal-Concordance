# 005 amendment 01: DRAFT (proposal for the PI; nothing here is adopted until the PI decides)

*Why an amendment.* The 005 smoke (`docs/reports/005_smoke_note.md`) triggered the G1 stop rule: sampled naming is floor-limited on Qwen. It also showed:
- the u^⊥ onset at the bottom of the dose grid;
- a leaky split for one concept;
- a clamp no better than its random null.

The design (v2) cannot run its report contrasts as written.

*Evidence base.* The dev-only overnight pilot (design §16; questions Q1–Q9 in §16.1 and Q10–Q11 in §16.3, each with a pre-registered rule and agent prior). **Every choice below is made on dev concepts only; the 45 test concepts stay untouched until the amended run.**

*Status of this draft.* Written before the pilot finished. Each item names the pilot question that decides it; `[pilot: …]` marks values to be filled from `results/005_planted_concepts/analysis/pilot_summary.json`.

| # | Item | Options | Decided by | Proposal |
|---|---|---|---|---|
| A1 | Primary "report" measure | (a) R1′ (concept log-prob at the open quote, log-sum-exp over spellings) with R1 (reciprocal rank, the paper's metric) as primary and sampled naming R2 secondary; (b) keep R2 primary | Q1 (+ Q2) | `[pilot: Q1]` |
| A2 | Prefill | `default` vs the protocol's `word` variant | Q2 | `[pilot: Q2]` |
| A3 | Injection band | B (L32–38) vs B2 (L24–38) | Q3 (+ Q7 leak counts per band) | `[pilot: Q3]` |
| A4 | Dose grid and text-reader strengths | Grid 1/256 … 1 (9 points); text readers at {s*/4, s*, 4s*} around the u^⊥ onset s* on that grid | Q4 | `[pilot: Q4]` |
| A5 | P11a / P11b definitions | If A1(a): "model doesn't report" = ΔR1′(u^⊥) within the rand-dir null (the naming clause dropped). P11b unchanged (B1, B2, B4 within the null). If the onset is still at the grid floor: P11a/P11b reframed (no "silent" dose), or dropped | Q1, Q4, Q5 | `[pilot]` |
| A6 | Reader read kinds | Keep direct (P17 primary) and carried C_o (P11 primary). Add the C_o layer sweep (L32–62) as a descriptive arm | Q5, Q11 | `[pilot: Q5, Q11]` |
| A7 | Clamp (M2) | (a) M2 on the re-entry clamp (reference h_clean + raw injection) with its random null, full range and band-limited (≤ L44); (b) keep the v2 clamp; (c) drop M2 | Q6, Q10 | `[pilot: Q6, Q10]` |
| A8 | Leaky-concept exclusion | Exclude from every "non-J" contrast any concept with c in J_NP top-10 on the pure u^⊥ at ≥ 2 layers of the band in use (rule fixed in §16.1 Q7 before computing). Pilot: B none; B2 earth, soap | Q7 | adopt |
| A9 | G4 probe control | A sample-level label permutation replaces the Hewitt–Liang type-level control (uninformative for single-word identity probes) | smoke | adopt |
| A10 | Concept-vector baseline | Leave-one-out stays primary; the 194-word sensitivity is mandatory in the report (smoke cos median 0.86 < 0.9) | Q8 | adopt |
| A11 | Budget and reduction order | From measured throughput (smoke: NLA 2.0 samples/s, Oracle 3.1/s, sampled naming ~2.7 rows/s at chunk 96) | Q9 | `[pilot: Q9]` |
| A12 | Regime | fp32-residual primary if the SEC l0 = 31 cross-check agrees (top-1 equal, RR within 0.05) on ≥ 90% of cells; otherwise SEC for teacher-forced report measures | pilot step 6 | `[pilot]` |
| A13 | R-lens | Added as a token reader (R_CB standard), descriptive | 7b.3 | adopt |
| A14 | SAE / BSF | Stays deferred (PI decision A01 C16); `ceselder/qwen36-27b-bsf-l42-1b` needs its own 16 GB encoder | — | defer |
| A15 | Logging | Every block saved and state written after each; the clamp's removed share defined against h − ref | smoke mistakes | adopt |
| A16 | Probe detection criterion (R1, §6) | The v2 rule (Δprobe above the rand-dir 95th percentile at the same dose) is **scale-free in a deterministic, near-linear forward**: in the pilot it fired for 80–100% of dev concepts at the report position even at s = 1/256. Replace it with an absolute criterion: the probe classifies the read as the concept (decision > 0, the threshold calibrated on natural sentences). Pilot (band B, L42): present on direct reads and at the injected position at s ≥ 1/4; **never at the report position** | pilot token readers, Q12 | adopt (the absolute criterion was applied post hoc in the pilot and is labelled so) |

**Unchanged by this amendment.**
- The concept split: dev 10 / test 45, after the base-rate screen.
- The priors: the PI's columns and the agent's priors pre-registered in df56646.
- Holm across the P-family; test outcomes unread until the analysis code is committed.
- G0, G2, G3, G5 and G6 as in v2.

**What the PI decides.** A1–A7 and A11–A12, and the go for the amended full run.
