# Reading list for the build session

The project: score activation readers (J-lens, R-lens, logit lens, Oracle Lens, NLA, Skip-lens, SAE) on Qwen3.6-27B. Measure how much they agree, then check their claims against reader-blind causal labels: **used / decodable-not-needed / absent / unresolved**.

Read in tier order. Within a tier, read only the named sections. The "why" says what to extract; if a source does not change how you build something, skim it.

**Ground rules**
- The spec is the design. Papers are background on the instruments. If a paper suggests changing a spec decision, flag it; don't silently change it.
- Labels must never use any reader (no J swaps, no AR edits) for labelling or admission.

---

## Tier 0: project docs (read fully, before any paper)

| Doc | Why |
|---|---|
| `reader_adjudication_spec.md` | The design. Sections 0, 4–7 and 12 decide what gets built first. Section 6 (labels) is the critical path. |
| RHUT repo `senku14x/Readable-Here-Used-There`, **branch `extension`** | Read `STATUS.md` (top block and the 2026-09-24 update), `context/evidence_ledger.md`, and `H3/results/reports/two_hop_question_turn_ladder.md` §0 and §9. These are the current claims; older per-stage reports contain superseded wording. Reuse `common/scripts/` (registry, materials, rendering + span gates, readout, `hooks_fp32`, hooks, stats, instrument_gates) rather than rewriting. Organism materials: `H1/design_specs/natural_modulation.md` (maintain / mention / ignore conditions, i.e. the matched contrast), `tagged_selection.md`, `computed_sums.md`, `H3/design_specs/computed_sum_consumer.md`, `two_hop_organism.md` (Amendment 7: 42-item bank and admission), `H3/results/reports/query_local_workspace.md` (absent-source transplant). From its `CLAUDE.md`, read only *Environment*, *Instruments* and *Non-negotiables*. `context/paper/workspace_paper_notes.md` is a faster way into the paper; check numbers against the PDF. |
| `senku14x/jacobian-lens` branch `research_secant`: `README.md`, `STATUS.md`, `scripts/lib/engine.py`, `scripts/lib/mfl.py`, `docs/workspace_bench/README.md` | fp32 upper-stack replay engine. J_CB fit-corpus facts. bf16 finite differences give cos 0.80, so use true fp32. WorkspaceBench patches: CJK mojibake, OOM; the J arm is a cosine readout. Skip results 001/002 (closed branch). |

**Four traps from the RHUT repo**
1. RHUT's `CLAUDE.md` freezes the two-hop organism ("no new forward without the researcher's instruction"). That rule governs RHUT, not this project. Reuse the items and code; run new forwards here.
2. Use the float32 hooks (`hooks_fp32`) for every direction-level write. RHUT's bf16 coordinate writes realised only 0.53–0.59 of the requested norm and fail the ρ/κ gate.
3. RHUT admitted 35 of the 42 two-hop items with a J_NP coordinate swap. Re-admit with the reader-blind mean-difference swap before ranking J on them.
4. Clamp and score every answer token, not just the first (RHUT silly mistake 5). Multi-token answers like `R`+`ome` are common.

## Tier 1: the two sources the build depends on

**Gurnee et al. 2026, *Verbalizable Representations Form a Global Workspace in Language Models* (arXiv 2607.15495)**
- §2 (J-lens definition, readout, steering, and the lens-coordinate swap $h' = h + V(\sigma(c)-c)$): needed to implement the J arms.
- §3.1: the swap experiment, as an example of a behavioural check.
- §5–6: the workspace claims the project is careful *not* to over-claim.
- A.9: template lens and oracle lens.
- A.21: J vs NLA agreement (AUC .853), the closest prior result to this project.
- Skim A.12 and A.24.3.

**WorkspaceBench (Blank, Bhatia, Ong & Nanda 2026): blog post + repo `camilablank/workspace-bench` @92d763e**
- Item schemas for `brew_intermediates`, `chain_intermediates`, `arithmetic_intermediates`, `multihop`, `relational_multihop`, `role_bound_association`. You need fields like regions, eval_positions, donors and options.
- How each arm is scored, the any-cell rule (which we don't use), and the prompt-only baseline.
- The reproduction targets: J brew ≈ .86, basic multi-token ≈ .24.

## Tier 2: methods behind the causal labels

| Paper | Why |
|---|---|
| Zhang & Nanda 2023, *Towards Best Practices of Activation Patching* (arXiv 2309.16042) | Metric and corruption choices. Why margin/logit-difference metrics beat probabilities. |
| Heimersheim & Nanda 2024, *How to Use and Interpret Activation Patching* (arXiv 2404.15255) | Sufficiency vs necessity (denoising vs noising). This is the swap vs ablation tiers in spec §6. |
| Makelov, Lange & Nanda 2023, *An Interpretability Illusion for Subspace Activation Patching* (arXiv 2311.17030) | Why a 1-D swap that moves the answer ≠ natural use. Motivates the necessity tier and norm-matched random controls. |
| Elazar et al. 2021, *Amnesic Probing* (arXiv 2006.00995) | The original "decodable ≠ used" result. Removing a decodable property can leave behaviour unchanged. This is the decodable-not-needed label. |
| Hewitt & Liang 2019, *Designing and Interpreting Probes with Control Tasks* (arXiv 1909.03368) | Shuffled-label controls for the item-level decodability probe. |
| Geiger et al. 2023, *Distributed Alignment Search* (arXiv 2303.02536) | Second label method on a subset. Optimised to succeed, so it over-labels; read for implementation only. |

## Tier 3: the readers being scored

| Source | Why |
|---|---|
| Anthropic NLA paper (2026) | Verbalizer + reconstructor setup and the RL objective. Check whether it tests steering via reconstructor edits; we haven't confirmed either way. |
| Karvonen et al. 2025, *Activation Oracles* (arXiv 2512.15674) | How activations are injected into a verbalizer and queried. |
| Li et al. 2025, *Do Activation Verbalization Methods Convey Privileged Information?* (arXiv 2509.13316) | Verbalizers can answer from their own priors or the prompt. This is why prompt-only, no-activation and shuffled-activation arms are mandatory. |
| Bauer et al. 2026, *AObench* (arXiv 2606.02609) | Existing oracle-evaluation practice, to avoid duplicating or contradicting it. |
| R-lens (Blank et al. 2026) | LRP-rule lens, the R arm. Matched to J_CB at n=25. |
| `agu18dec/olens_and_ar` | Oracle Lens RL checkpoint `olens_s3d_rl600` + AR reconstructor. Loading and calling conventions. |
| `camilablank/workspace-lenses` (`qwen3.6-27b/j-lens/lens.pt`) | J_CB and R_CB checkpoints. J_NP comes from Neuronpedia (n=1000). |
| NLA at L42, Skip-lens (ceselder), SAE/BSF at L42 | Locations as in spec §4. Confirm each loads on one cell before anything else. Open question: is the NLA reconstructor released? |

The following came from the user's notes and have **not** been verified by us. Check that they say what we think before citing them: Dingeto 2026 (arXiv 2607.20379), Hong 2026 (fact-check reward), michaelzhang & TurnTrout 2026.

## Tier 4: only when you reach the analyses

- Dawid & Skene 1979, *Maximum likelihood estimation of observer error-rates using the EM algorithm*: the conditional-independence assumption behind majority voting (spec A1).
- Cohen 1960 (κ), Fleiss 1971, Krippendorff 2004 (α), Webber et al. 2010 (rank-biased overlap): agreement metrics.
- Feng & Steinhardt 2023, *binding IDs* (arXiv 2310.17191): only if the role-bound family gets labelled, to understand why binding may not be 1-D.

---

## First build target (spec §12, step 3)

Run the end-to-end label check on brew at L42 before wiring any reader:
1. Full-cell patch at the intermediate and decision positions. Does it move the answer?
2. Class-centroid direction fitted on template half A, swap on half B, compared with norm-matched random directions.
3. Split-half reliability and the fraction of unresolved cells.
4. One matched pair: RHUT maintain vs ignore at the same carrier position. Check first that the model still answers correctly under "ignore".

Write a short design doc (hypothesis, decision rule, stop condition) before running it, following the research_secant `designs/` convention.
