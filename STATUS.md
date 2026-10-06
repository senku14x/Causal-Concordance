# STATUS — 2026-10-06 (`main` = `007-jsubspace`: 007 run and reported; earlier work 005, 006, 006b included)

*New session? `main` holds everything through 007 (fast-forwarded from `007-jsubspace` on 2026-10-06). **First read
the Anthropic workspace paper** (Gurnee et al. 2026, `docs/papers/2607.15495.pdf`; text in `docs/papers/text/`) and the
relevant reading listed at the top of `CLAUDE.md`. That is the PI's rule. The newest results: `docs/reports/007_jsubspace.md`
and report 25 (`docs/claude_reports/25_nla_stops_and_open_issues_2026-10-06.md`). Then `CLAUDE.md`, this file (newest first),
`docs/reports/005_followups.md` and `docs/reports/006_reader_likelihood.md`.*

**2026-10-05, 007 — J-subspace split with controls (branch `007-jsubspace`; spec v2 2820e53 + A1 5cdafd9; report `docs/reports/007_jsubspace.md`; per-stage status `results/007_jsubspace/status/`; ~2.2 GPU-h, unattended under rule 12).**
- **Reader side, confirmed on the brew test:** the Oracle's z tracking at `:` L40 is carried by the J (token-readout) subspace — H_J supported on dev and on the 49 test tables (J 5.8 / 5.0 nats vs whole 3.2 / 2.9; rest of J 0.3); Raw and Visible contradicted at L40 and L44 on both splits; L44 unresolved by positive-control rules. Same pattern at `</think>` (patch share 0.01).
- **Model side:** J-subspace swaps track content share (L42) or exceed random modestly but not PCA-4 (L40/L44, below margin); in the adapted §3.3 test the J part of a class-mean z direction moves the answer less than its non-J part. → A reader–model dissociation.
- **The NLA is disabled for all of 007** (G-repro 0.41 > 0.25 after my 2-row scorer change; not repaired, rule 12). Post hoc NLA run on 006's 24-row path is the PI's call.
- Secondary: multihop confounded by restatement (CW-12 rates reproduced); relational stopped after 4a (no high-leverage read-layer position; 12/50 gated); chain (exploratory) signal outside J at L56. Brew test split used once under AMENDMENT_02 (opened this session).
- **2026-10-06, post hoc NLA rerun (amendment A2, PI: "rerun properly"; 006's exact 24-row scorer; ~4.8 GPU-h):** G-repro now passes (max 0.15, D_z 1.862 vs 1.862). The NLA's z signal is *not* J-concentrated (dev: J 0.95 vs whole 1.86, rest of J 0.63 ≈ rest of random 0.80; test similar); its categories are unresolved/uninformative and none replicates dev → test. → J dependence is reader-specific (Oracle yes, NLA no). Stage 1b NLA first stopped at G-score1 (0.280 > 0.25), diagnosed as bf16 batch-layout noise (report 25), then rerun under amendment A3 (median/p99 gate): at `</think>` the NLA's signal collapses without J (rest of J 0.21 vs 1.21 for rest of random) but J alone keeps only half, and mean removal doubles it; H_J unresolved (post hoc, dev). Multihop NLA: J ≈ whole (as the Oracle). Results page: https://claude.ai/artifact/8vnKTzcXczn4YGMiP1uJTW

**2026-10-05, end of session.**
- `006-reader-likelihood` was merged into `005-amended-full-run` (fast-forward), so 005 and 006 now live on one branch, pushed.
- **HF:** the public dataset `senku21x/causal-concordance-outputs` holds every output, 005 and 006 included, with a verified MANIFEST and a `results/` snapshot of this branch.
- **The PI is analysing the current set of results.** No experiment is queued.

**2026-10-05, 006b, the PI's "one final run" (pre-registered at 3d9f6d6, run m1, 15.6 GPU min; design `docs/design_specs/006b_model_output_baseline.md`; section in `docs/reports/006_reader_likelihood.md`).**
- **Question.** Is the readers' z tracking just the model's own next-token distribution at the same cells (z as the runner-up answer)?
- **Gate** passes: max 0.147, median 0.043 nats.
- **The model's own D_z** (48 tables): `:` +3.76 [3.27, 4.24], `</think>` +3.99 [3.39, 4.57], assistant newline +3.66 [3.10, 4.22].
  - At `:` the answer ranks first in 100% of tables, and z is the runner-up in 56% (median rank 2).
  - B1 holds, B4 holds, B3 fails (the model's output tracks z at `</think>` too).
- **B2 (readers beyond the model): not testable.** Only 9 pairs have model D_z ≤ 0 (the minimum is 15). The secondary OLS intercepts are positive but extrapolated.
- **Per-pair reader–model r:** 0.07 (NLA) to 0.30 (Oracle L40).
- **Reading.** 006's z tracking is consistent with the readers reading the model's output-facing distribution (z as the runner-up answer). No held intermediate beyond it is shown.
  - The Oracle's 2× at `</think>` is not mirrored by the model and stays unexplained.
  - D magnitudes are compared only within an instrument.
- **Also post hoc** (PI-forwarded review, `posthoc.json`, commit 70ba3e1): the readers' tracking does not follow 001's patch shares, and the Oracle's G-pc failure is a "blue" prior. The review's "`</think>` ≥ `:` for every reader" had the NLA's sign reversed (NLA −0.41 [−0.87, 0.04]).
- The page https://claude.ai/artifact/P35xhVkdzKjsdGhogp7PQY (v8) carries both.

**2026-10-05, 006 (now merged into `005-amended-full-run`; report `docs/reports/006_reader_likelihood.md`).**
- **Question.** At the brew `:` cell (saved 001 captures, 48 dev tables), do the verbalizers' likelihoods track z, as 002's lenses did?
- **NLA L42, likelihood:**
  - tracks z: +1.86 [1.41, 2.34] nats, 44 of 48 tables; supported (T1, Holm);
  - shuffled control ≈ 0;
  - it also tracks the start colour (+0.41) and the answer (+0.59), less than z.
- **Oracle L40/L44:** tracks z in 48 of 48 tables (+3.2 to +3.8). Its positive control failed at those layers (33% and 42% < 50%), so T2 is uninformative by the rule.
- **Bar:** the NLA passed its positive control exactly at the bar (6 of 12).
- **Reading.** z is decodable from the readers' likelihood at the cell where 002 found they rarely state it. **Qualified by 006b (above):** the model's own output at the same cells already favours z.

**2026-10-05: state at the end of the session (branch `005-amended-full-run`, pushed; not merged into `main`).**
- **005 is closed.** The PI ruled no further 005 variants.
- **Main conclusions, with the follow-up corrections applied:**
  - the model-side measure is lexical priming, not report;
  - no J-space privilege for it;
  - the clamp effect is output-direction blocking;
  - the logit-lens direction primes as well as the J atom;
  - on direct reads, probe > Oracle > NLA, and the token lenses detect nothing;
  - nothing readable arrives at the answer position at L40–44.
  - Details in CLAUDE.md, "Where things stand", and the reports listed there.
- **Mirrors (2026-10-05):**
  - all of `outputs/` (3.2 GB, incl. 005) is on the public HF dataset `senku21x/causal-concordance-outputs`, with MANIFEST.json and a `results/` snapshot;
  - the private dataset was not updated (the HF account's private storage is full); GitHub holds every branch, and
    `claude_reports` and the agent memory are in the repo (`docs/claude_reports/`, `docs/context/agent_notes/`);
  - script: `scripts/000_setup/mirror_outputs.py`;
  - third-party weights are not mirrored (they are pinned and re-downloadable).
- **Open:**
  - the PI merges this branch;
  - the next direction. Candidates are in the session's last report-back: back to the reader panel on WorkspaceBench cells with the likelihood readout from 005; the 004 §14 options.
- **Untracked:** `docs/design_specs/Untitled.ipynb` (the PI's notebook; left as is).

**2026-10-04 (branch `spec-v2`, now the working branch; repo private).**
- **SPEC_v2** is the unified spec; **errata 01** (V2-E1–E9) was accepted by the PI and wins over the text.
- **Checks C1–C3** (`docs/reports/checks_2026-10-04_no_gpu.md`):
  - CW-7 survives WSB's chain rule at a third of its size, and J-lens matches it;
  - ":" L42 keeps "influence" (a mixture of the two trajectories);
  - Cathryn's multihop removal control was ~2.4× smaller in energy.
- **NLA = `nla-rl` iter_000400** (`readers.NLArl`, sha256-pinned); **J-space pursuit** in `jspace.py`.
- **005 planted concepts**:
  - design v2 + the agent's pre-registered priors (`docs/design_specs/005_planted_concepts.md`);
  - smoke (`docs/reports/005_smoke_note.md`): G1 report gate failed (sampled naming floor) → stop rule;
  - dev-only pilot + extra steps 7b and 8–11 (`docs/reports/005_pilot_note.md`, design §16–§16.7; report page
    https://claude.ai/artifact/P35xhVkdzKjsdGhogp7PQY).
- **Pilot findings (dev, descriptive):**
  - the log-prob report measure works and is specific to the planted concept (step 10);
  - no clear J privilege at matched norm (holds under 5 J-space constructions, step 9);
  - the privilege clamp is specific only when it extends past about L48–50 (step 10: from L50 with band L32–38, L52
    with L24–38), where the concept's J direction is turning into its output direction; blocking the plain output
    directions does nearly as much (step 11), so it mostly blocks the answer itself;
  - under a likelihood measure the readers are not less sensitive than the model: the Oracle responds on the written
    cell at lower doses than the model's report, the NLA at about the same (step 11). The naming-based "less sensitive"
    preview is withdrawn;
  - readers name planted concepts on direct reads; at the report position only the Oracle at L52–60, with band
    L24–38, names it, for a few concepts (Q13). An earlier version of this line said nothing ever reaches the report
    position (written before Q13; corrected);
  - the relative probe criterion is degenerate (absolute criterion, A9).
- **2026-10-05:** the PI adopted amendment 01 (`docs/design_specs/005_amendment_01.md`, all items incl. A18); the amended full run on the 45 test concepts (`docs/design_specs/005_full_run.md`) runs on branch `005-amended-full-run`, so the adoption can be reverted by dropping the branch.
- **005 full run done (2026-10-05, run 20261005T015235Z, 110 min; report `docs/reports/005_full_run.md`; page https://claude.ai/artifact/5EGz2DxcAXXQs3wTtz1Lnu):** gates pass except G11 for the NLA (its likelihood measure fails the positive control, so P11′ for the NLA is untestable); P17 not supported (the Oracle names the non-J part more than the NLA); M1 not supported on R1′ (tiny positive on R1); M3 opposite of predicted; M2's J clamp to L50 is specific but not distinguishable from the output-direction control; P11′ (Oracle) supported (not perturbation-matched); R1 and R2 as predicted (R2 partly scale-free; post-hoc floor 29/45).
- **005 follow-ups (exploratory, `docs/reports/005_followups.md`):** the answer-slot log-prob is position-generic lexical priming (A2b), so 005's model-side "report" is renamed priming; the concept's own atom is in its J part for 1/45 concepts; the logit-lens direction is as potent as the J atom at s = 1/16–1/4 (Q-c); the per-concept probe is the best detector and the token lenses detect nothing on direct reads at L40–44 (A1). 005b's Q-a/Q-b not run (flawed as written).

**004, span assay at `:` L42: complete** (`docs/reports/004_span_assay.md`; 33 pairs, 55.8 min, commit 9da6b23 clean).
- Whole cell 0.43 of the gap. Recovery: 1-D z 0.06, z span 0.21, joint span 0.72.
- **Same-rank PCA does as well or better** (0.58 / 0.89; at matched norm the fitted spans are slightly behind).
- The z-span swap mostly raises the recipient's own intermediate (target-directed 4/33). Random controls are inert. The
  z-span clamp beats paired rotations in 31/33. The z span and its complement are non-additive.
- S2 and S6 hold; S1, S3, S4 and S5 do not. Descriptive geometry result; no intervention nominated. Next: report §14
  (default: back to the reader panel).


**003, reader-free causal labels: complete** (`docs/reports/003_labels.md`).
- Gates G1–G3 pass. **G4 (label reliability) fails** (κ 0.37, r(S) 0.41 between disjoint fits), so the labels are
  **descriptive only** and are not used to score readers.
- At `:` and the assistant newline, L40–44, the decorrelated z direction has a small, mostly positive effect: median
  S 0.03–0.05, positive in 85–90% of pairs, above the start and answer directions. Clamping it beats random clamps,
  but those clamps were not size-matched. It is "used" (≥ 0.10) in only 10–18% of pairs. No single tested direction
  reproduces the whole-cell effect.
- At the `</think>` newline, z is readable at the family level and the whole-cell patch has little net leverage (not an
  item-level label). At `:` L60 the answer direction carries the cell. (Corrected after the audit of 001–003.)
- Next options in report 003 §7: subspace labels, DAS, more fitting data for reliability, or a descriptive 004.

**Latest (2026-10-02, while the PI slept):**
- **002, reader claim scoring: complete.** `docs/reports/002_claim_scoring.md`; design v2.2; external review accepted.
  - Judge: Gemini 3.8 Flash, as in WSB. 3,930 readouts, 0 errors, $7.98. The gate passed (κ 0.67 / 0.81 against
    blind author codes).
  - **The verbalizers rarely claim z in its role:** NLA 5–15%, Oracle 0–3%. 85% of Oracle L40 samples contradict
    themselves. Role-based tracking is weak.
  - **The token lenses carry activation-sensitive z information inside the candidate set at L40–44:** positive
    start-swap tracking (a difference-in-differences of scores) in 47–48 of 48 tables, though actual rank reversals
    occur in about half of the partner comparisons (90/183 for J_NP cosine at `:` L40). No tracking of the start.
  - Descriptive; no causal label.
- **001 probe repair (amendment A2): done.** 41k fits, 2 h CPU.
  - Membership controls now average 0.48–0.52 (they were 0.58, up to 0.84).
  - **The call changed:** off-path membership is decodable at the `:` cell at every layer, narrowly at L40–44
    (selectivity 0.11–0.13) and clearly at L60.
  - Residual checks: the span solver matches or beats the full solver across the C grid; the grouped 12-way null
    is wider but changes no call.
  - Report 001 §7/§9/§10/§12/§13/§15 updated.

**New here?** Read `docs/COME_UP_TO_SPEED.md` first; `docs/context/SPEC_STATUS.md` maps the whole spec to done / not
done. **The repo is private and `main` carries everything** (merged from `setup/000-scaffold` on 2026-10-02).

**External audit (Codex, 2026-10-02; `docs/reviews/2026-10-02_codex_reader_audit/`).** Every number recomputed
independently (`scripts/001_leverage_profile/audit_followups.py`) and matched. The report was revised in place (§15 has
the point-by-point table). Still open:
- **probe controls:** the membership control task is far from chance and the permutation null reuses C, so
  membership calls are provisional;
- **damage:** the readout cannot be decomposed (top 5 saved only);
- **readers:** scoring is mention-only, and there is no prompt-only baseline yet.

**Design 001 v3 (+ amendment A1) is running on dev material.** Done:
- `gate`, `smoke` (every exact check passes);
- `run` (72 min, SEC regime);
- `probes` (25 min);
- `analysis`, `figures` (reader-free outputs).

`snapshot` (readers, descriptive, 32 min) is done too. The report is complete: `docs/reports/001_leverage_profile.md`.
It awaits PI review; its proposals for AMENDMENT_02 are in §13. No label and no reader score exists yet, and no
test-split measurement has been made.

| Block | State | Evidence |
|---|---|---|
| a. Repo | done | this file, `CLAUDE.md`, `README.md`, `docs/context/{SPEC,AMENDMENT_01,READING_LIST,reading_notes}.md`, `docs/reports/000_reading_report.md` |
| b. Environment | done | model loads (16 s, 50.1 GB); one brew forward gives the gated answer; 29/29 hashes match; `scripts/000_setup/rebuild_env.sh` |
| c. Reader smoke tests | done (5 work; SAE deferred, vLLM path parked) | table below; `results/000_setup/analysis/reader_smoke.json` |
| d. Shared code | done; **74/74 tests at `647dcb5`** (+4 added since, see Tests) | list below |
| e. Design 001 | **v3 + A1 approved**; all stages done; report complete | `docs/design_specs/001_leverage_profile.md`; `results/001_leverage_profile/{manifest_gate,manifest_smoke}.json`, `analysis/{gate_summary,gate_pairs,smoke_checks}` |

## At a glance (2026-10-02)

Results page (private artifact, generated by `scripts/000_setup/results_page/make_page.py`):
https://claude.ai/artifact/RMRX8R4iFnysbnHaNWTBeC

**Works (tested):**
- Pinned environment: the model loads, 29/29 hashes match, and the rebuild script recreates everything.
- WorkspaceBench rendering, byte-identical to WSB's producer, with position checks on every item.
- Partner builders (brew 361, chain 110, relational 50 candidates, all token-aligned), the off-trajectory substitution
  builder (100/100 items admit one; 1,138 valid (c, c′) choices), and the competence gate.
- Margin scorer, patching engine (RHUT float32-residual regime, the cross-check), **SEC true-fp32 replay (the 001
  label regime, L ≥ 40)**, gates, stats, figures, and the subject-model integrity guard.
- Readers on one smoke cell: logit lens, J_NP, J_CB, R_CB, Oracle Lens (L40), NLA AV + AR (L42). The NLA positive
  controls pass.
- The 001 pipeline `scripts/001_leverage_profile/leverage_profile.py` (stages gate → smoke → run → probes → snapshot →
  analysis → figures); every stage has run on dev.
- The results page, rebuilt 2026-10-02: the published copy had lost its middle (a template truncation, logged in
  CLAUDE.md). It now carries the 001 results, figures, reader grid and qualitative browser, and is checked in jsdom
  before publishing.

**Not working / blocked:**
- **SAE:** deferred by the PI (2026-10-02); no SAE arm until the PI decides what to do with it. Candidate located since:
  `ceselder/qwen36-27b-bsf-l42-1b` (block-sparse subspace featurizer, 8-dim blocks, k = 32, EV 0.51, no feature labels).
- **vLLM acceleration:** the engine fails to start on this GPU (sm_120 capability check inside FlashInfer; needs a
  CUDA ≥ 12.9 component). Parked by the PI; the HF route is used (64 NLA explanations ≈ 107 s).
- **Skip-lens:** dropped by the PI (AMENDMENT_01 §4). The ~22 `ceselder/skip-lens-qwen36-27b-*` checkpoints are
  trained and read at **layer 62**; "fed raw L42" would be off-distribution. Reinstatable on-contract at L62, or as a
  separate off-contract L42 arm, if wanted.

- **Optional readers (SPEC "Optional" line):** modulation lens and phrase-J dropped by the PI (2026-10-02). Template
  lens deferred: maybe later, not a priority. It exists for Qwen3.6-27B at the pinned
  `camilablank/workspace-lenses` @ d740106d (`qwen3.6-27b/template-lens/templates.safetensors`, 8.6 GB, 13,174 words,
  cosine-scored). Its vocabulary has 11 of the 12 brew colours (no teal).

**004 span assay (design v2, PI go 2026-10-03 "start with it, be careful") — implementation checklist:**
- [x] Raw-output mirror: public HF dataset with MANIFEST.json, retrieval verified.
- [x] Reader-free helpers factored into `scripts/common/brew_material.py` (identical 82 pairs and renders; boundary
  test extended).
- [x] `scripts/004_span_assay/span_assay.py` (reader-free; boundary test walks it): frozen 33-pair manifest (seed
  20261007, one per table); spans per fit with each fit's own mean; PCA per fit (within-table centred, 1/sqrt(n_t));
  arm manifest: 334 conditions, 476 scored prompts, ~956 forwards per pair (31,548 in all).
- [x] Preflight (CPU, `results/004_span_assay/preflight.json`): fit rows reproduce 003's n and means; basis identities
  float64/float32 (worst QᵀQ 3.6e-7 after the cast); zero writes bitwise no-ops; NEG_REL 1e-4 checked; the 24 accepted spelling
  sequences (12 colours × 2; only " Teal" is two tokens) are prefix-free; no reader or mixed modules loaded.
- [x] Smoke, 2 pairs (`smoke_checks.json`, `smoke_inspection.md`): all six gates pass (self-swap bitwise; identity
  projector vs direct assignment ≤ 3e-6 nats; S_full = 001 exactly; clean m and G = 003 exactly; fidelity on every
  write; M ≤ 1). 101 s per pair → ~56 min for 33. Outside-the-spellings mass (~14% clean) is formatting tokens
  (`<|im_end|>`, newlines, " **"; `smoke_outside_mass_tokens.json`).
- [x] External review of the smoke (`docs/reviews/2026-10-03_codex_004_smoke_review/`): S5 definition, G1 enforcement and
  fidelity resolution fixed before the run (design §13); smoke rerun (v2): all gates pass, 948/948 writes per pair
  fully checked; v1 smoke outputs kept as superseded.
- [x] PI go 2026-10-03 → run 33 pairs (55.8 min, every write passed fidelity) → analysis → report → page → mirror re-upload.

**Built and run since (see the top of this file and the reports):** label construction (003: directions, 1-D swaps,
necessity, random nulls), role-aware reader scoring (002: judge + closed set), prompt-only baseline (002), probe repair
(001 A2). **Not built yet:** content-span and complement patches, norm-matched ablation controls, per-item null
calibration, DAS, relational gating, any test-split measurement.

**Next (2026-10-02, after the audit of 001–003, `docs/reviews/2026-10-02_codex_audit_001_003/`):**
1. Done without the GPU: the audit's corrections (report 003 §9; the 002 tracking wording; this file).
2. Proposed, awaiting the PI's go: a small, separately designed **L42 assay check**:
   - 1-D z edit vs a z span vs the joint content span vs **their complements** vs the whole cell, with matched-rank
     and matched-norm controls;
   - two disjoint fits;
   - norm-matched ablations with damage logged;
   - the start-token positive control kept.
3. Still pending from earlier: PI review of reports 001–003; AMENDMENT_02 (τ, ε′, substitution bound) before any
   test-split label; the not-needed scoring-layer decision.

## 001 — gate and smoke (dev)

Gate (`manifest_gate.json`, run 20261002T074329Z at `5d5d1a3`, **dirty tree**; 621 distinct prompts, 24 min):

| Family | Split | Candidates | Passed | Items (groups) with a passing pair |
|---|---|---|---|---|
| brew start pairs | dev | 183 | **82** | 33 of 48 items |
| brew start pairs | test | 178 | 77 | 35 of 49 items |
| chain start pairs | dev | 53 | **48** | 12 of 13 items; **9 of 10 groups** |
| chain start pairs | test | 46 | 37 | 8 of 11 items; 7 of 10 groups |
| brew substitution | dev | 50 | **43** | 43 of 50 items |

- Design rule 4 (≥ 20 brew, ≥ 15 chain dev pairs) is satisfied; chain support is still thin and is reported by group.
- Pass rate differs by z (brew dev: green 4/24, gold 2/10 vs teal 12/19, gray 11/20): the gated set is not z-balanced,
  so the difficulty distribution must be reported next to any pooled number (AMENDMENT_01 §1).
- Chain dev passing pairs are dominated by two z classes ((5,) 22 and (9,) 12 of 48).
- 3 brew items have no partner candidate at all (97 items carry candidates).

Smoke (`manifest_smoke.json`, run 20261002T080920Z at `5d5d1a3`, dirty tree; 76 s): **all engine checks pass**
(self-patch, embedding anchor, pre-difference prefix and stir no-op bitwise) on 3 brew pairs (all three partners of
`brew-s2-r10d2-0009`), 1 chain pair and 2 substitution pairs. Observations, single pairs, not findings:
- brew emit_colon whole-cell share rises with depth: 0.11–0.51 at L40–44, 0.62–0.69 at L56–60; the start-colour
  token at L0–4 carries ≈ 1.00 (the anchor), the emission span 0.89–0.91, start + question + emission 1.00.
- chain final cell: ≈ −0.03 at L40–44 (negligible), 0.72–0.74 at L56–60; the input digits span and the question span
  each carry ≈ 1.00 jointly across L40–60.
- substitutions `0000 gray→teal`, `0001 blue→gold`: behaviour |Δlog P(a)| 0.009–0.023, TV ≤ 0.012; transplants
  ≤ 0.022; L0 key-token reference 0.002–0.064. All far inside the provisional bound (0.05 · median G ≈ 0.45).

Provenance note: both manifests record `git.dirty = true` at `5d5d1a3`. The uncommitted diff (committed as `647dcb5`)
touched only the `analysis` and `figures` stages and two CLAUDE.md log rows, not the gate or smoke code, so the code
that ran is the committed code; the manifest alone cannot show that. Run stages from a clean tree from here on.

## Environment

Full details in `CLAUDE.md` § Environment. RTX PRO 6000 Blackwell (sm_120, 96 GB); torch 2.11.0+cu128, transformers
5.15.0, peft 0.21.2, jlens @ 581d398; `Qwen/Qwen3.6-27B` @ 6a9e13bd (bf16, eager).

- **Hashes** (`results/000_setup/analysis/artifact_hashes.json`, 29/29 match):
  - J_NP, J_CB, R_CB lens files against the sha256 in RHUT's `instrument_registry.json` (which pins only these three
    files plus the model revision);
  - the 15 model shards, the Oracle adapter, the NLA adapters, critic and value head, and the NLA reference parquet
    against Hugging Face's LFS sha256 at the pinned revisions;
  - the 5 WSB bank files against the registry.
- **Env check** (`results/000_setup/analysis/env_check.json`): on `brew-s2-r10d2-0000` the greedy answer is " black"
  (gold: black). Top-5 next tokens: " black" −0.31, " Black" −2.31, " green" (= z) −2.68, `<|im_end|>` −4.06,
  " Green". The runner-up is the intermediate colour (WSB's runner-up caveat). One 165-token forward takes 0.23 s.

## Reader smoke tests (block c)

One cell: `brew-s2-r10d2-0000` (start blue → z green → answer black), emit_colon = the ":" after "Answer", position 164.
Activations come from the stock bf16 forward (WSB convention). The verbalizers sample 4 times with WSB's sampling
(T = 1.0, top_p 0.95, top_k 64, ≤ 256 tokens).

| Reader | Status | Load + run | GPU peak | Raw output (abridged) | Names colours? |
|---|---|---|---|---|---|
| Logit lens (L40/42/44) | **works** | < 1 s | 50 GB (model) | L42: ' ', ' (', ' either', '...', '\n\n' | no at L40–44. Positive control: at L63 top-1 is ' black' (= model greedy); colours appear from L56 |
| J_NP (standard + WSB cosine) | **works** | 16.9 s for all three J lenses (3 layers, sha-checked) | 56.4 GB | standard L42: '\n\n', '<\|im_end\|>', '<\|endoftext\|>'; cosine L42: ' END', ' FINAL', '입니다', '_final' | no at L40–44 at emit_colon (the cosine readout names green at L36 emit_colon and L40 emit_think); at L56: ' black' top-1; at L60–62: black, green, brown |
| J_CB (standard + cosine) | **works** | (above) | (above) | standard L42: '终点' (end point), '第二次' (2nd time), '不会再', '终结', ' downstream' | no at L40–44; yes from L56 |
| R_CB (standard + cosine) | **works** | (above) | (above) | standard L42: ' ...', ' either', '第二次', ' intermediate', ' second' | no at L40–44; yes from L56 |
| Oracle Lens, L40 (on-contract) | **works** | 32 s (adapter + 3 arms × 4 samples) | 51.3 GB | "- Green (Since after two steps from Blue, we reach Green, …)", "- blue (or green) …"; no activation: Korean company names (㈜ = the corporation sign); another cell (stir "You"): liquid-colour puzzle set-ups | **yes**: green, blue, red, purple, gray, black (start and z among them) |
| NLA-L42 AV | **works** | 156 s (SFT merge + RL adapter; 3 arms × 4 + 8 reference rows) | 52.5 GB | "Table structure … two successive applications of a lookup table … (which is green/…)"; 4/4 parsed, 1 injection per generate; no activation: generic web text (car audio, patents); other cell: colour-rule puzzle set-up | **yes**: z (green) in 4/4 samples, one also pink/blue/red |
| NLA-L42 AR | **works** | 23 s | 35.2 GB | FVE of the cell from its own explanations: **0.46, 0.55, 0.59, 0.50**. No-activation explanations: −0.15 to −0.48. Other-cell explanations vs this cell: −0.41 to −0.66 (vs their own cell: 0.56–0.65) | n/a |
| NLA checks | **work** | — | — | Our L42 capture re-extracts the NLA's stored reference activations at cos ≥ 0.99984. AV → AR FVE on 8 reference rows: 0.75–0.86 (card: 74.5% held-out) | — |
| SAE | **deferred** | — | — | candidate located (`ceselder/qwen36-27b-bsf-l42-1b`, no labels); PI decision pending | — |
| vLLM path (optional) | **parked** | — | — | engine init fails on sm_120: `Failed to get device capability: SM 12.x requires CUDA >= 12.9` → `RuntimeError: FlashInfer requires GPUs with sm75 or higher`. Likely cause: a CUDA-12.8 component in vLLM 0.21's stack misreads this card (vllm-metamodels was tested on B200 / sm_100 only). HF reference timing: 64 NLA explanations in 107 s | — |

Notes (smoke observations on one cell, not findings):
- No lens reads a colour at emit_colon L40–44 here, though the verbalizers name z; J does name green at L36
  (emit_colon) and L40 (emit_think), so the working hypothesis is "the methods surface the intermediate at different
  cells", not "J is blind".
- The FVE reference set is the NLA's own (FineWeb text), not this task's distribution.
- The Oracle adapter is the public copy `olens_s3d_rl600`. That it equals WSB's private `s3d.ddp600.s0/iter_000600`
  is assumed from the names and cannot be verified.

## Tests (block d)

`python -m pytest scripts/common/tests -v` (log `outputs/logs/pytest_full.log`): **74 passed, 0 failed at `647dcb5`**
(47 CPU + 27 GPU). Added since (not yet run on the GPU box): 2 CPU tests for the resident guard (pass) and 1 GPU test
for the SecReplay park/restore lifecycle; 1 CPU test for the T(a) substitution tag (pass). Collected total: 78.

| Module | Tests | Result at 647dcb5 |
|---|---|---|
| cells | round-trip (json, jsonl); validation (layer, half, split, family, position, empty ids; 6 cases); unknown keys rejected | 8/8 |
| wsb (rendering) | bank hashes and sizes; `brew-s2-r10d2-0000` regions (stir 131–136, start 138–143, question 145–153, emissions 158/162/164, " blue" at 142); all 100 brew items' 24 positions; chain " What" + final "\n\n" and relational "'s" on every item; byte-equality with WSB's own `_chat` renderer | 5/5 |
| partners + gate | brew trajectories reproduce the bank; hand-checked brew partner (0000: pink valid, gray excluded by z = a′; 165 tokens, diff only at 142); constraints hold for all 361 candidates; chain step functions reproduce all 120 items; hand-checked chain partner (0408, start 19 → 14: z (9,4) → (7,3), one digit differs); 50 relational pairs aligned (2 diffs); answer parsing (8 cases); hand-checked substitution (0000 gray → teal); every item has a substitution, deterministic by seed; **T(a) tag never filters and leaves the 001 seeded draw unchanged (new)** | 16/16 (+1) |
| margin | spelling tokenisation (" Teal" two tokens, "27" two digits); sign implied by greedy (brew and chain, member < 0 < partner); multi-token teacher forcing scores every token, with an off-by-one discriminator; shared first token ("21" vs "27") → margin from the second token; writes applied on every forward | 5/5 |
| engine | self-patch bitwise (5 cells incl. L2, L63, multi-position); 1-D swap ρ = κ = 1 in float32 with read-back < 1e-4 and exactly one row changed; swap along unit(h_p − h_m) equals the full patch at the write (rel 2.7e-8); last-layer last-position replacement reproduces the partner's logits bitwise; earlier layers and positions untouched bitwise; the stock bf16 regime distorts a small write (ρ 1.04, κ 0.91) | 10/10 |
| sec engine (SEC regime, hand-checked pair) | self-patch bitwise; stir cell patch bitwise no-op; prefix identical before the first difference; embedding anchor reproduces the partner; last-layer replacement reproduces partner logits; lower-stack write lands and moves the answer; SEC score matches stock sign and scores multi-token spellings; SEC noise floor < 1e-3 nats | 8/8 |
| splits | brew 50/50 by table stratified by z with forced-dev items; chain groups intact across the split | 2/2 |
| gates | fidelity thresholds; damage-gate logic; span gate on the hand-checked pairs; self-patch damage is exactly 0; sequence-length determinism reported (not bitwise, onset at block 3, no leak) | 5/5 |
| stats | cluster-t vs scipy; BH-FDR vs statsmodels (exact); Holm vs statsmodels; cluster bootstrap covers and is wider than the naive interval; ratio of means; empirical p; equivalence | 6/6 |
| figures | ICLR figure round-trip (pdf + png); distinct reader colours | 2/2 |
| import boundary | no label module (registry, cells, wsb, partners, margin, engine, gates, stats, regime, splits) reaches readers/jlens/peft/vllm, transitively and including imports inside functions; sanity: readers does reach jlens | 2/2 |
| model integrity | Engine, SecReplay and capture_stock refuse a model a reader merged weights into (3 cases); Oracle attach → capture (adapter disabled) → unload leaves the subject's activations and logits bitwise unchanged; mutation after construction refused at use time; **resident guard: parked block refused, `blocks_upto` honoured (new, CPU); SecReplay-parked model refused by Engine / gate / capture and restored by the context manager, also on an exception (new, GPU, unrun)** | 5/5 (+3) |

## Regime check (`results/000_setup/analysis/regime_check.json`)

The same 5 edits under RHUT's float32-residual regime and SEC's true-float32 replay (blocks 40–63 in float32, resident,
72 GB). Pair: the first gated pair in bank order, `brew-s2-r10d2-0001__blue` (purple→green→orange vs blue→black→pink).
The hand-checked `0000__pink` failed the gate (partner 5/10) and `0001__black` failed too (partner 8/10).

| Edit | RHUT from block 0: Δm | SEC: Δm | \|diff\| / G_SEC |
|---|---|---|---|
| full patch, emit_colon, L44 | +4.770 | +4.426 | **0.033** |
| full patch, emit_colon, L40 | +2.437 | +2.220 | 0.021 |
| full patch, start-colour token, L40 | +0.559 | +0.526 | 0.003 |
| 1-D swap at emit_colon L42 along the emit_think pair difference | +2.032 | +2.123 | 0.009 |
| 1-D swap at emit_colon L42 along a random direction | −0.015 | +0.001 | 0.002 |

- G: RHUT-from-0 10.63, RHUT-from-35 10.52, stock bf16 10.31, SEC 10.48 nats.
- Max |diff| / G against SEC: RHUT-from-0 **3.3%**, RHUT-from-35 2.3%, stock bf16 1.3%.
- Under the 5% rule RHUT passed, and AMENDMENT_01 §3 kept it as the default; **errata E4 and design 001 §9.1 then
  adopted SEC's true-fp32 replay for every label measurement at L ≥ 40**, because the RHUT absolute floor
  (0.05–0.25 nats) is too coarse for ε′ = 5% decisions. The RHUT regime (`engine.Engine`) remains the cross-check.
- RHUT writes are exact (ρ = κ = 1); stock-bf16 random-direction writes are not (ρ 1.04, κ 0.86).

**Noise floor**: |Δm| under 8 perturbations of 1e-4 relative norm at emit_colon L42.

| Regime | Max \|Δm\| (nats) | Fraction of G |
|---|---|---|
| RHUT, from block 0 | 0.053 | 0.50% |
| RHUT, from block 35 | 0.034 | 0.33% |
| Stock bf16 | 0.064 | 0.62% |
| SEC | **0.0002** | 0.002% |

- On `0000__pink` (G 8.6), the RHUT floor reached 0.24 nats (2.7% of G) at the start token at L2.
- Appending 3 tokens moved that pair's margin by 0.22 nats.
- So the RHUT floor is absolute, roughly 0.05–0.25 nats, and on pairs with G below about 7 nats a gap the size of the
  observed 3.3% would cross 5% of G.
- SEC's own STATUS (L45–46) made the same observation independently: bf16-matmul finite differences give cos 0.80.

## Other block-000 findings (engineering)

- **Sequence length.** Activations at prompt positions depend on the total length. Blocks 0–2 are bitwise identical;
  the onset is at block 3, the first full-attention block, at 6e-4 relative. That grows to ~1% median by block 63, and
  up to 40–99% at a few positions. It is rounding, not a leak. Consequence: comparisons are made at one length only,
  with partner sources captured at the matching length.
- **Bank.**
  - Brew: 361 partner candidates, all token-aligned; per-item counts 0–8; 3 items have none.
  - Chain: digits are single tokens, so only 28/120 items have single-token intermediates. Of those, 25 have partners
    (110 candidates, all aligned). There are 15 cross-item prompt collisions (identical prompts from items sharing step
    sequences), so chain splits by step sequence (20 groups, 10/10) and de-duplicates pairs.
  - Relational: 50/50 a/b pairs aligned (gating pending).
- **Timing** (`timing_smoke.json`), used for design 001's budget:
  - forward ≈ 0.23 s whether T = 78 or 165; SEC replay of blocks 40–63 ≈ 0.09 s; SEC build 8.5 s;
  - gate ≈ 5.4 s per brew pair, 4.4 s per chain pair (measured: 621 prompts in 24 min);
  - HF NLA generation: 64 explanations of ≤ 256 tokens in 107 s.

## Code guard added after `647dcb5` (this update)

`SecReplay.build()` parks the bf16 upper blocks on the CPU until `restore()`. Every stock-forward entry point
(`Engine.forward`, `partners.gate_member`, `readers.capture_stock`, the Oracle and NLA constructors) now calls
`registry.assert_subject_model_resident`, and `SecReplay` / `SecEngine` are context managers that restore on exit,
also on an exception. `SecReplay.capture` checks only blocks ≤ l0. The 001 pipeline already restores at the end of
`smoke` and `run`, so no run was affected; the guard turns a silent misuse into an error.

## Open questions for the PI

Resolved in design 001 v3 §9: regime (SEC for L ≥ 40), reader snapshot (included, after hashing), chain split (50/50 by
group), reader-input bridge (checked on the snapshot). Still open:

0. **(2026-10-03, most pressing)**
   - confirm the framing: native reader-configuration comparison is the primary line, causal work a validation branch;
   - the next step after 004 (report 004 §14);
   - review reports 001–004;
   - rotate the OpenRouter key.
1. ~~HF dataset name~~: done. The public `senku21x/causal-concordance-outputs` and the private
   `senku21x/causal-concordance-private` (CLAUDE.md, "Fresh instance").
2. ~~Paper PDFs and web sources~~: committed to the private repo (`docs/papers/`, `docs/sources/web/`; PI 2026-10-03).
3. **P5 prior**: the dated update accepted in AMENDMENT_01 §5 needs your number (suggested ≈ 35%, down from 55%, on the
   NLA literature in the reading report §4.5.6).
4. **SAE arm**: use `ceselder/qwen36-27b-bsf-l42-1b` (no labels: LLM-label ~2,000 features, or score decoders by
   non-J means) or keep the SAE deferred?
5. ~~Branch~~: `main` fast-forwarded to `setup/000-scaffold` on 2026-10-03 (PI).

## External reviews (2026-10-02)

Review 1 (of `c13fd43`): every concrete claim was checked against the repo, and all hold. Accepted: design 001 v2 (its
change list), AMENDMENT_01 errata E1–E4, the model-integrity guard and its tests. Two corrections to the review
discussion: J does name green at L36 (emit_colon) and L40 (emit_think) on the smoke item, so "no colours at L40–44"
held only for the emit_colon cell; SEC removes the reader-input bridge only for layers ≤ 39 (001 §9.4).

Review 2 (of `56cb7ab`): integrity check at use time; cross-fitted substitution scoring; stir no-op scoped to start
pairs plus a pre-difference-prefix identity check; snapshot on 10 distinct tables. Accepted → design 001 v3, approved.

## Next

001 `run` → `probes` → `snapshot` → `analysis` → `figures` → report 001 → AMENDMENT_02 (thresholds frozen, 002
critiques recorded) → design 002 (labels) → PI approval.
