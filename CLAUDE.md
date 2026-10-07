# CLAUDE.md — Causal-Concordance

Guide for any agent or collaborator working in this repo. Reading only this file should be enough to find everything.

## FIRST, before anything else: read the workspace paper, then the paper notes, then the relevant reading (PI rule, 2026-10-05; restated 2026-10-06)

**Every new session (and every subagent briefed on the science) must, in this order:**
1. **read the workspace paper in full** (item 1 below);
2. **then read the paper notes** (item 2 below);
3. then the relevant reading (item 3), then the project docs in the read order.

No code, run, review or answer about results before steps 1 and 2. The whole project is built on the paper's claims
and vocabulary: J-lens, workspace, J-space, privilege, verbalizable. A session that skips them misreads every result.

1. **The paper, in full:** Gurnee et al. 2026, *Verbalizable Representations Form a Global Workspace in Language
   Models* (Anthropic).
   - PDF: `docs/papers/2607.15495.pdf`; text: `docs/papers/text/2607.15495.txt` (from the PDF) and
     `docs/papers/text/2607.15495_html.md` (from the web version: cleaner headings and figure captions; cross-references
     show as "??").
   - A section-by-section digest made at the start of the 007 session (2026-10-05): `docs/papers/text/2607.15495_NOTES_digest.md`.
     It is a map, not a substitute: read the paper itself.
   - Read the main text and the appendices the project leans on: A.8 (J-space span), A.9 / A.9.2 (verbalizers and J-space), A.12, A.21, A.24.3, §3.1 with Figs. 7–8 (planted concepts and the privilege test).
   - The PDF is the only source for text inside figures; the .txt and .md extractions miss it.
2. **The paper notes, after the paper** (all three):
   - `docs/claude_reports/04_user_workspace_paper_notes_verbatim.md`: the PI's own notes;
   - `docs/claude_reports/04b_user_workspace_paper_notes_combined_2026-10-05.md`: the PI's combined notes (paper plus
     follow-up literature);
   - `docs/claude_reports/03_workspace_paper_notes_merged.md`: the agent's notes merged with the PI's, with corrections
     in §14 and disagreements in §15.
3. **The relevant reading** (`docs/context/READING_LIST.md` is the full list):
   - Readers:
     - Karvonen et al. 2025, *Activation Oracles* (`2512.15674`): the activation-oracle idea. **The Oracle Lens is
       not a Karvonen oracle.** It is WorkspaceBench's reproduction of the paper's A.9.2 oracle lens: an activation
       reconstructor (AR, text → activation, MSE in a whitened space), an inverter, and RL on FVE with the AR frozen
       (`docs/sources/web/wsb_post.md:341-349`; report 26). Its AR (`ar_ptag_pooled`) and whitener are public;
     - Bauer et al. 2026, AObench (`2606.02609`);
     - Li et al. 2025, *Do Activation Verbalization Methods Convey Privileged Information?* (`2509.13316`; prompt-only baselines);
     - Dingeto 2026 (`2607.20379`; NLA claim sensitivity);
     - the NLA, WorkspaceBench and R-lens pages in `docs/sources/web/`.
   - Notes: `docs/context/reading_notes.md` and `docs/context/reading_notes_v2.md` (the NLA paper, Viswanath 2026, AO, WorkspaceBench).
   - Method papers, when the task needs them:
     - Hewitt & Liang (probe controls);
     - Elazar et al. (amnesic probing);
     - Geiger et al. (DAS);
     - Makelov et al. (subspace illusions);
     - Zhang & Nanda and Heimersheim & Nanda (patching).
4. **Then the project docs**, in the read order below.

## Read order

(Updated 2026-10-04 for the unified spec v2, PI instruction.)

0. **The workspace paper, then the paper notes, then the relevant reading** (the section above). Always first.
1. **CLAUDE.md** (this file): rules, conventions, environment, pitfalls.
2. **STATUS.md**: where things stand, what works, open questions.
3. **docs/context/SPEC_v2.md**: the unified spec (2026-10-04). It contains the original SPEC.md verbatim, AMENDMENT_01
   (Appendix B), Cathryn Wayman's A4 amendment (Appendix A), results 000–004 and her pilot (§14), and tagged
   callouts ([A01], [CW], [R], [v2]). Where a callout and the original disagree, the callout wins. **[v2] items are
   proposals until the PI accepts or strikes them.** Start with the v2 header, §4.R, §6.0, the §7 planted-concepts
   callout, §8 A4/A7/A8/A9, §12, §14 and §15.
4. **docs/context/SPEC_v2_errata_01.md** (accepted 2026-10-04): V2-E1–E9 correct SPEC_v2 passages; an erratum wins over the text.
   **docs/context/AMENDMENT_01.md**: the PI's 2026-10-02 decisions (its §6 lists each conflict by SPEC.md line).
5. **The latest experiments (2026-10-06, branch `008-trackb-brewcode`, all exploratory):**
   - 008, Track B: `docs/reports/008_trackb.md` (design `docs/design_specs/008_009_trackb_brewcode.md` Part 1, §E);
   - 009, the brew-code bank: `docs/reports/009_brewcode_derisk.md` (same design note, Part 2, §E-009; reader rules in
     `docs/design_specs/009b_reader_phase.md`);
   - 010, reader agreement (CPU): `docs/claude_reports/29_reader_agreement_010_2026-10-06.md`;
   - 011, Oracle-space swaps: `docs/design_specs/011_olens_swaps.md` (loader accepted post hoc on scale-free criteria, not
     validated against the published AR; full run s1 done and analysed: partial; AR J-bias check: ambiguous).

   Before them, 007 (`docs/reports/007_jsubspace.md`, on `main`), and 005 and 006 (branch `005-amended-full-run`).
   For 006, read `docs/reports/006_reader_likelihood.md` and `docs/design_specs/006_reader_likelihood.md`. For 005,
   read, in order:
   - `docs/reports/005_followups.md`: the latest results and the corrections they force (read this first);
   - `docs/reports/005_full_run.md`: the full run on the 45 test concepts (pre-registered verdicts);
   - `docs/design_specs/005_full_run.md` (the run spec) and `docs/design_specs/005_amendment_01.md` (adopted 2026-10-05);
   - `docs/design_specs/005_planted_concepts.md` (design v2 + §16 pilot and steps 7b–11), `docs/reports/005_pilot_note.md`,
     `docs/reports/005_smoke_note.md`;
   - `docs/design_specs/005_followups.md` (the follow-ups' pre-registration, incl. the provenance of the PI's approvals).
6. Before 005: **docs/design_specs/004_span_assay.md** and **docs/reports/004_span_assay.md**;
   then the earlier ones as needed, in order (all four experiments are complete on the dev split and await PI review):
   - 001: leverage profile, probes (amendment A2) and the reader snapshot;
   - 002: role-aware reader claim scoring;
   - 003: reader-free 1-D labels (descriptive; G4 failed);
   - 004: span assay at `:` L42 (fitted spans ≈ PCA).
   Report 000 is the reading report.

**docs/context/SPEC.md is frozen:** do not edit it. Reports cite it by line, and AMENDMENT_01 §6 cites it line by line.

Supporting:
- `docs/COME_UP_TO_SPEED.md`: handoff (what exists, what 001 found, known problems, decisions log).
- `docs/context/SPEC_STATUS.md`: SPEC.md item by item (done / built / partial / not started / dropped), written
  before v2; SPEC_v2's status callouts supersede it where they differ.
- `docs/context/agent_notes/`: the PI's working preferences, as kept in the agent's memory (also summarized under
  "Working with the PI" below).
- `docs/context/PI_DIRECTIVE_2026-10-02.md`: the PI's block-000 directive, the source of the non-negotiables.
- `docs/reviews/`, every external review with its scripts and the disposition in each README:
  - two reviews of the scaffold and design 001;
  - Codex audits of 001 and of 001–003;
  - the design-002 review;
  - the design-004 review and the consolidated handoff that defines design 004 v2 and the project framing;
  - the 004 smoke review.
- `docs/context/READING_LIST.md`, `docs/context/reading_notes.md` (cited notes from the reading pass),
  `docs/papers/INDEX.md`.

## Project in one paragraph

Score activation readers on **Qwen3.6-27B** (logit lens, J_NP, J_CB, R_CB, Oracle Lens, NLA; SAE to be added) against
**reader-blind causal labels** — "pair difference used" / decodable-not-needed / absent / unresolved — computed per
(token position, layer) cell from full-cell patches, 1-D swaps along mean-difference directions, necessity ablations and
cross-fitted probes on **WorkspaceBench** families with token-aligned counterfactual partners (brew, chain, relational).
Then decompose reader agreement into shared signal, shared error and shared prompt inference. The labels measure
demonstrated influence on one specified answer, not workspace membership.

## Non-negotiables (verbatim from the PI, 2026-10-02)

- Labels never use any reader: no J swaps, AR edits or SAE directions in labels or item admission. Label code must
  not import reader code.
- Direction-level writes go through float32 hooks. Fidelity is measured after the write (rho in [0.9, 1.1],
  kappa >= 0.99).
- Every experiment has a design spec, committed before its run and shown to me. Launch only on my go.
- Gates before outcomes: span/rendering, same-source == clean bitwise, fidelity, damage, label reliability. A failed
  gate is repaired and re-run; nothing that has been scored is deleted.
- Full-sequence endpoint, log-sum-exp over accepted spellings; clamp and score every answer token, never just the
  first.
- Baselines from day one (prompt-only, no-activation, shuffled-activation).
- Wording: "demonstrated influence on this answer", not "uses"; "decodable", not "represents". Scope every claim to
  model, family, position, layer and n.
- Record every silly mistake in the commit message and the report.
- No subagents.

RHUT's freeze rules (its registered-label stop rules and frozen ledgers) do **not** apply to this project; RHUT is a
source of code and organisms only.

## Conventions

- **Layer** = output of decoder block l (0..63), the residual stream entering block l+1 (RHUT/WSB/EasyNLA convention).
- **Cell** = one token position at one layer (`scripts/common/cells.py`: item, pair_id, position, layer, family, half,
  split). Span patches are context only.
- **Renders** reproduce WSB's producer exactly (`scripts/common/wsb.py`): brew = chat template + "Answer:" prefill
  (24 pinned cells), chain = chat (read cells " What" → final "\n\n"), relational = bare cloze (final "'s").
  Thinking is off (`enable_thinking=False`). Position assertions run on every render.
- **Partners** (`partners.py`): brew = same table, different start; chain = same steps, different start (same digit
  count, single-digit intermediates); relational = the bank's a/b members. Constraints z′≠z, a′≠a, z′∉{a,a′}, z∉{a′};
  renders token-aligned. **Gate**: greedy correct and ≥ 9/10 correct at T = 0.8 (top_p 0.95, 16 tokens, EOS
  suppressed), both members.
- **Margin** (`margin.py`): m = LSE_{S(a′)} log P − LSE_{S(a)} log P, full-sequence teacher forcing, every token of every
  spelling (brew " c"/" C" — " Teal" is two tokens; chain digits; relational " hop1's hop2"). Logits computed in float32
  from the final-norm output.
- **Regime**: design 001 v3 (§2, §9.1) measures every label quantity at L ≥ 40 under **SEC's true-float32 replay**
  (`regime.SecEngine`, l0 = 39: stock bf16 blocks 0–39, float32 blocks 40–63 and W_U; noise floor ≈ 2e-4 nats). RHUT's
  float32-residual regime from block 0 (bf16 matmuls under autocast, float32 residual and writes; `engine.Engine(model,
  fp32_from=0)`) is the cross-check (AMENDMENT_01 §3, errata E4). Every clean/edited comparison is made at **one
  sequence length**; patch sources are captured at the **matching length** (`margin.score(writes_fn=...)` /
  `SecEngine.score(edit_fn=...)` build writes per continuation). Reader inputs use the stock bf16 forward
  (`readers.capture_stock`, WSB's convention), **except brew-family reads in 006, 007 and 009, which use saved SEC
  captures** (stock vs SEC cos ≥ 0.99985, 002's bridge; stated in each design). Natural-text reader inputs ("Tell me
  about {colour}") are stock. `SecReplay.build()` parks the bf16 upper blocks on the CPU until
  `restore()`: use `SecReplay` / `SecEngine` as context managers; every stock-forward entry point refuses a parked model
  (`registry.assert_subject_model_resident`).
- **Splits**: dev/test by table (brew), by step-sequence group (chain; identical prompts recur across chain items with
  the same steps), by name set (relational); defined in design 001. Halves A/B for cross-fitting by items
  (AMENDMENT_01 §3). Items touched by engineering checks go to dev.
- **Naming**: experiments `NNN_snake_case`; stages `smoke` / `pilot` / `run`; run IDs are UTC timestamps.
  Code in `scripts/NNN_<experiment>/<experiment>.py` (+ `_analysis.py`); results in
  `results/NNN_<experiment>/{analysis/*.json, figures/, manifest_<stage>.json, meta_<stage>.json}`; raw arrays and logs
  in `outputs/` (gitignored; mirrored to the **public** HF dataset `senku21x/causal-concordance-outputs` (PI 2026-10-03; `MANIFEST.json` there lists sizes and sha256)). Every manifest
  carries `registry.manifest_block()` (model/lens/reader/WSB pins, software versions, git commit + dirty flag).
- **Statistics** (`stats.py`): cluster-level t intervals, BH-FDR, cluster bootstrap (clusters = items/tables), Holm.
  Report intervals and sign counts, never a bare mean.
- **Figures** (`figures.py`): ICLR style (vendored `iclrplot`), fixed colour per reader, drawn from committed JSON only.

## Environment (verified 2026-10-02; still valid 2026-10-05)

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, sm_120, 96 GB (driver supports CUDA 13.2) |
| Host | 128 CPUs, 377 GB RAM, 500 GB disk; **/workspace is not a persistent volume** (rebuild with the script below) |
| Label + reader stack (`/venv/main`) | Python 3.12.14, torch 2.11.0+cu128, transformers 5.15.0, peft 0.21.2, accelerate 1.15.0, jlens @ 581d398 (`/workspace/jacobian-lens`, editable) |
| Reader-generation stack (`/venv/vllm`, optional, not yet adopted) | vLLM 0.21.0, torch 2.11.0+cu130, transformers 5.5.4, vllm-metamodels @ 6a0a0e5. Never used for labels |
| Model | `Qwen/Qwen3.6-27B` @ 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9, bf16, eager attention; 50.1 GB on GPU, loads in ~15 s |
| HF cache | `HF_HOME=/workspace/.hf_home` (~149 GB on 2026-10-05: Qwen 52 GB, nla-rl 52 GB, nla-L42 36 GB, lenses 9 GB, Oracle 0.4 GB). Third-party weights, re-downloaded pinned by `rebuild_env.sh`; not mirrored |
| Page checks (`/venv/page`) | playwright + headless Chromium for the 1280/400 px render check of report pages |
| Pins + hashes | `scripts/common/registry.json` (model, lenses with sha256, reader checkpoints, WSB bank sha256s, software); verification record `results/000_setup/analysis/artifact_hashes.json` (29/29 match) |

Lenses (sha256 in the registry, matching RHUT's `instrument_registry.json`): J_NP = neuronpedia/jacobian-lens @ 0731326e
(n=1000 wikitext), J_CB and R_CB = camilablank/workspace-lenses @ d740106d. Readers: Oracle Lens =
agu18dec/olens_and_ar @ c95215d5 `olens_s3d_rl600` (the public copy; WSB's default path is a private dataset repo);
NLA for 001/002 = ceselder/qwen3.6-27b-nla-L42 @ 54e3ce60 (`av_sft_lora` merged + `av_rl_lora_step400`; AR
`rl_critic_step400`). **NLA for 005 and onward = ceselder/qwen3.6-27b-nla-rl @ 5a13b7ec**, `av_base` + adapter
`iter_000400` (`readers.NLArl`, 11 sha256 pins in the registry; it is WSB's and Cathryn's NLA). av_base (53.8 GB) and
the subject (50.1 GB) do not fit on one GPU together: capture subject activations first, then free the subject.

Sources (read-only clones, pinned in `scripts/000_setup/rebuild_env.sh`): `/workspace/sources/repos/` — workspace-bench
@ 92d763e, RHUT (senku14x/Readable-Here-Used-There) @ d4197b0, SEC (senku14x/jacobian-lens, `research_secant/`)
@ a3720ab, EasyNLA @ 4d72847, vllm-metamodels @ 6a0a0e5; papers and web sources in `/workspace/sources/{papers,web}` (copies committed in `docs/papers/` and `docs/sources/web/`).

**Rebuild:** `bash scripts/000_setup/rebuild_env.sh` (add `--vllm` for the optional vLLM venv).
**Tests:** `python -m pytest scripts/common/tests -q` (`-m "not gpu"` for the CPU-only subset).
**Git:** commits as senku14x / visheshgupta14x@gmail.com. A stale `GITHUB_TOKEN` env var shadows the gh login: push with
`env -u GITHUB_TOKEN git -c credential.helper='!gh auth git-credential' push origin <branch>`. Do not push to `main`
without the PI's say-so.

## Where things stand (2026-10-06)

**Branches.** **`origin/main` holds everything through 007:** on 2026-10-06 the PI had `007-jsubspace` fast-forwarded
into `main` ("merge with main"). On this instance the local `main` ref was left stale (95ab0e5), so branch from
`origin/main`, not `main`. Never push to `main` without the PI's explicit say-so.
- **`008-trackb-brewcode`** (off `origin/main`, 2026-10-06, pushed): experiments 008-011, reports 26-29, the
  `claude_reports` cleanup. **The newest work.** Not merged.
- `007-jsubspace` (off `005-amended-full-run`): experiment 007 (design, amendments A1-A3, all code, results, status files,
  reports 22-25) plus everything below. Identical to `main` at the merge; kept for reference.
- `005-amended-full-run` (off `spec-v2`): amendment 01, the 005 full run and follow-ups, 006 and 006b.
- `006-reader-likelihood`: experiment 006, merged into `005-amended-full-run`; kept for reference.
- `spec-v2` @ 9709712: SPEC_v2 + errata 01, checks C1-C3, the NLA loader, and 005 up to the dev pilot.
- `main`: up to date with `007-jsubspace` since the 2026-10-06 fast-forward (before that @ 95ab0e5, the 000-004 scaffold).

- **012** (branch `012-reader-concordance`, exploratory, 2026-10-07): J-row swaps change Qwen's multihop answer (0.585 vs random 0.06 at L24-47); every reader's src→tgt shift tracks the change (Oracle ≈ NLA), mostly through output-facing content at the read cell. Spec `docs/design_specs/012_reader_concordance.md`, ledger `docs/ledger/012_ledger.md`, page https://claude.ai/artifact/NoevKvBkXNUTvtpnJPVsuq.

**008, Track B: the paper's §3.3 privilege test on brew, done the paper's way** (exploratory; dev, 82 gated pairs; all
gates pass; `docs/reports/008_trackb.md`; design `docs/design_specs/008_009_trackb_brewcode.md` Part 1 and §E):
- Class-mean z swaps clamped per layer over the three emit cells (assistant newline, `</think>`, `:`) × L40-48.
- **The lever is adequate:** the full direction moves the answer by median S 0.21 [0.11, 0.26] (flips 16%).
- **No model-side J privilege.** The J part (k = 25 pursuit) carries S 0.02-0.03, the non-J part 0.21; J − non-J
  −0.17 (J_NP) / −0.14 (J_CB; −0.15 before report 31's correction), CIs below 0. The real whitened swap (Ledoit-Wolf Σ, swap in whitened coordinates, map
  back) narrows the gap (J 0.07 / 0.10 vs non-J 0.14 / 0.11) but never reverses it. Matched raw norm: J_CB
  ties, J_NP below; matched whitened norm: both below (J_CB CI [−0.029, −0.000]). Clamping the J coordinates barely changes the non-J swap (J_NP 0.21 → 0.20; J_CB 0.20 → 0.18); the paper's clamp
  took Claude's from 28% to 6%.
- The true lens-coordinate swap (arm A) is not read: its answer-atom positive control moves S 0.01 (power bar 0.15).
- Not "reversed": the paper clamps at every position; 008 clamps 3 cells × L40-48, so the paper's mediation result is
  **not reproduced under a band-limited clamp**.
- **The reader-model dissociation, on matched objects** (report 31): with 007's dense J subspace on both sides, the
  Oracle's z reading depends on J (Stage 1a / brew_test; J beats Δ-matched PCA, P1) while the model's answer does not
  favour J over Δ-matched PCA-4 (Stage 2 recovery J − PCA4 +0.00 / −0.02 / +0.05 at L40 / L42 / L44, CIs include 0;
  J − Δ-matched random +0.07 [+0.04, +0.11] / +0.02 [−0.00, +0.03] / +0.09 [+0.02, +0.16]; lever S_full 0.28-0.49). 008's sparse J part (a different object) agrees. The PI's
  2026-10-06 numbers to the TA come from 008 (J_CB corrected to −0.14).

**009, the brew-code bank** (exploratory, reader-free de-risk; `docs/reports/009_brewcode_derisk.md`; design Part 2 and
§E-009; reader rules `docs/design_specs/009b_reader_phase.md`):
- The idea: answer with a two-digit code looked up from the final (or start) colour, so the hidden colour z is needed
  but not said.
- **Gate:** depth 2 is beyond Qwen without chain of thought (0/77 pairs after 30 tables); depth 1 passes 771 of 1,200
  pairs (241 of 300 tables).
- **The output-facing check fails, so readers do not run** (rule §2.5.1). With a code answer, z is still the top colour
  at the answer slot in 98% of renders (median rank 11 of 248,320 tokens; the model's own D +11.9 nats, 240/240
  tables), at only ~0.05% total colour mass. D is scale-free, so the tail still counts.
- Reader-free probes decode z under both questions (110 of 112 sites; but 77 sites precede the differing token and are
  identical under both questions, so 33 of 35 informative sites; z decodes at q1-q2 already). The patch profile reproduces brew's structure
  (`</think>` L40-44: z decodable, no single-cell leverage).
- The PI has called the new prompts weak and suggested returning to the old prompts and runs. 009 readers are on hold.

**010, reader agreement** (exploratory, CPU; `docs/claude_reports/29_reader_agreement_010_2026-10-06.md`): NLA-Oracle
per-pair agreement is weak (r 0.24-0.30). The mid-layer Oracle tracks what the J lens reads (r 0.63-0.80), not the model's
next-token output (0.21-0.30); the NLA is weaker on both (0.33-0.37 with the J lens, 0.07 with the output); only the late Oracle (L56/L60) follows the output.

**011, Oracle-space swaps** (the TA's tip: put a colour word through the Oracle Lens's own reconstructor (AR), get a
vector, do it for two colours, swap them in whitened space; `docs/design_specs/011_olens_swaps.md` with amendments 1-3;
`scripts/011_olens_swaps/`). **Everything from 011 is post hoc, exploratory.**
- **The AR loader is ours** (the official one is not public). It failed the pre-set validation twice (v1, v2), then
  passed under post hoc amendment 2. The AR's output is a raw activation, scored in the whitened metric, with the
  AR's own whitener (`whitening_iolens_pooled`). On 16-token spans its FVE after one fitted global
  scale is 0.24 (the published 0.22 is unscaled, on 1-32-token spans), and within-batch retrieval@1 is 0.86
  (published 0.93). **Not validated against the published AR** (unscaled FVE negative; outputs ~2× too long; 64 blocks
  vs a truncated backbone untested; 011 note "Loader caveats"), so 011 is not "the TA's reconstructor method" yet. A global output scale (0.3-0.55)
  is unexplained, so only colour vectors centred over the 12 colours are used (the scale cancels there).
  Validation: `results/011_olens_swaps/ar_validation_v3.json`, `ar_offline_d2.json`.
- **Smoke s0b** (2 pairs, one table; observations only; all gates pass; `olens_swaps_s0b.json`), median S:
  - reference (class-mean coordinate swap): 0.23;
  - **the TA's swap as given (AR vectors swapped in whitened coordinates): −0.00.** Its answer-colour positive
    control is −0.00 too, so it is not read. Its edit lies along the same raw direction as the additive one; it is null
    because the clean coordinates do not favour AR(z) over AR(z′) (report 31; "edits almost nothing" was wrong);
  - the same directions swapped in raw coordinates: 0.16, but its answer control is 0.08 and two size-matched random
    edits 0.09 / 0.00 (generic, not colour-specific, on this smoke);
  - **additive reading** ("subtract the AR vector for z, add z′'s", at the reference's edit size): 0.10; the AR's
    answer-colour difference 0.12; the model's own class-mean difference 0.23; random 0.01 / −0.02.
- **Full run s1 (82 pairs; resumed 10-06 after the bookkeeping fix; POST HOC)**: TA's whitened swap and raw-coordinate
  swap not read (answer controls 0.016 / 0.050, below 0.15); additive **partial**: AR colour difference S 0.09 vs
  class-mean 0.20 vs random 0.01 (D_ar − D_classmean −0.10 [−0.15, −0.05]; D_ar − D_rand +0.06 [+0.02, +0.11]).
  The smoke's raw-coordinate 0.16 did not hold (0.005). `results/011_olens_swaps/olens_swaps_analysis_s1.json`.
- **AR J-bias check** (CPU, exploratory; `ar_jbias.json`): ambiguous by its pre-set rule. The AR over-weights 007's J
  subspace 1.27× / 1.15× (L40 / L44) relative to the activations, above PCA and random controls; too small to explain
  007 alone. What the AR config's `jfve` / `loss_mix_lambda` mean is open (no training code public).
- **Likely limit of the tip:** the AR reconstructs 1-token inputs poorly (cosine ≈ 0.1 with the real activation,
  vs ≈ 0.5 for 16 tokens). Single-word colour vectors are noisy but colour-specific (cos 0.15-0.24 with the model's class-mean
  differences, ≈ 0 for mismatched pairs; report 31). The proposed next step is vectors built
  from several short phrases that start with the colour (a new post hoc amendment first).

**007, the J-subspace split** (`docs/reports/007_jsubspace.md`; design `docs/design_specs/007_jsubspace.md` with
amendments A1-A3; per-stage status `results/007_jsubspace/status/`; results page https://claude.ai/artifact/8vnKTzcXczn4YGMiP1uJTW;
why the NLA runs stopped and the other open issues: `docs/claude_reports/25_nla_stops_and_open_issues_2026-10-06.md`):
- **Question:** do the readers' signals depend on the J subspace (top-1,024 directions of the J_NP word directions), on
  how much prompt-specific content a piece holds, or on removing the shared mean direction? Does the model's answer
  depend on J beyond its content share?
- **Confirmed (pre-registered, dev + 49 held-out brew test tables, AMENDMENT_02):** the **Oracle**'s z tracking at brew
  `:` L40 is carried by the J subspace. Registered contrasts confirmed at L40: P1, P2, P3, P5 (J beats Δ-matched PCA,
  random subspaces, rest-of-J vs rest-of-random, random after mean removal); at L44 only P5. Descriptive: J alone 5.8 vs
  whole 3.2 nats (dev; test 5.0 vs 2.9; J − whole is not a registered contrast), rest-of-J ≈ 0.3. Secondary, dev
  only: the same pattern at `</think>`, a cell with lower observed patch leverage (0.01 vs 0.48 at `:`, L42).
- **Model side:** J subspace swaps track content share (L42) or exceed random modestly (L40/L44, below margin). In the
  paper's §3.3 test adapted to brew, the J part of a class-mean z direction moves the answer *less* than the non-J part
  at matched raw norm. Like-for-like (dense J on both sides) is Stage 1a vs Stage 2 (report 31).
- **NLA (post hoc, A2/A3):** first disabled by a G-repro stop gate (my 2-row fast scorer), then rerun on 006's exact
  scorer. Its z signal is **not** J-concentrated at `:` (J ≈ half of whole; removing J costs about as much as removing a
  random slice). At `</think>` it collapses without J but J alone is not enough. So "J dependence is reader-specific" holds at `:` only. Multihop: J ≈ whole for both
  readers; restatement shares (J piece) 0.28 / 0.24 / 0.15, whole piece 0.56 / 0.25 / 0.58. PI's hand-read (post hoc,
  10-06): from the whole activation each reader names the bridge on 36/40 test items; the NLA's random-slice hits are
  mostly weather boilerplate carrying the country (38 of 43); the Oracle's rest-of-J readouts often name a wrong country.
- **Secondary:** relational stopped after its profile (12/50 gated; no high-leverage read-layer position). Chain
  (exploratory, 7 items): the signal lies outside J at L56. The template lens fails its colour sanity check at L40-44.
- **The PI's position (2026-10-06):** trusts the results and does not want a confirmatory 007c rerun of the NLA. A TA
  (Camila) asked how the interventions are done, suggested whitened-space swaps for the Oracle, gating checks, and
  qualitative analysis of readout content. Those are candidate next steps.

**001–004** (dev split only; await PI review):
- **001:** the `:` cell carries about half the answer gap at L42–44, and z is decodable there.
- **002:** the verbalizers rarely claim z in its role.
- **003:** 1-D labels are descriptive (G4 failed).
- **004:** fitted spans ≈ PCA, so no intervention is nominated.
- Details in STATUS.md.

**005, planted concepts on Qwen3.6-27B:**
- The test: plant "Tell me about c" concept vectors, split into a J-lens part and the rest, and ask whether the model "reports" each part and whether the readers detect it.
- Stages so far: the smoke failed its report gate; a dev pilot and steps 7b–11; amendment 01 was adopted; the full run on 45 held-out test concepts; then the exploratory follow-ups.
- **Read the conclusions as corrected by the follow-ups:**
  - **The model-side measure is lexical priming, not report.** The concept's log-prob rises about equally at every prefill position (A2b) and more on an unrelated prompt. Every "report" result (M1, M3, M2, P11′) is about priming of the concept token. Qwen almost never *says* the planted concept.
  - **No J-space privilege for this priming at matched norm** (M1 not supported on R1′). At natural norm the non-J part (≈ 95% of the vector) does more (M3 reversed).
  - **The clamp.** The J-coordinate clamp is specific from about L50, but blocking the plain output direction does as much (M2 control). It blocks the answer; no re-entry into J-space is shown.
  - **The concept's own token atom is almost never in its J part** (1 of 45). The J part is related plus junk tokens.
  - **The logit-lens direction is as potent a write direction as the J-lens atom** at s = 1/16–1/4 (Q-c).
  - **Readers, on direct reads of the written cell:**
    - The per-concept probe is best.
    - The Oracle names the planted concept more than the NLA (P17 reversed). On a likelihood measure it responds at lower doses than the answer-slot priming (P11′ Oracle; not perturbation-matched).
    - The NLA failed the likelihood measure's positive control.
    - The token lenses detect nothing at L40–44.
  - **At the answer position, at L40–44, nothing readable arrives** for any reader. The Oracle at L52–60 names it for a few concepts, and only with the wider band.
- Pages:
  - pilot and everything up to the full run: https://claude.ai/artifact/P35xhVkdzKjsdGhogp7PQY
  - full run: https://claude.ai/artifact/5EGz2DxcAXXQs3wTtz1Lnu
  - Both carry the follow-ups' correction banner.

**006 and 006b, reader likelihoods at the brew cells** (dev, 48 tables, saved 001 captures; `docs/reports/006_reader_likelihood.md`):
- The NLA's and the Oracle's colour likelihoods track the cell's z: z is decodable through them where 002 found they rarely state it. The NLA's rule is supported. The Oracle's is uninformative by rule (its positive control failed on a "blue" prior).
- **006b:** the model's *own* final-layer distribution at the same cells already tracks z (+3.7–4.0 nats, z the runner-up behind the answer in 56% of tables). So the readers' tracking is consistent with reading the output-facing distribution. "Beyond the model" was not testable (9 pairs < 15).
- Their tracking does not follow 001's patch shares. The Oracle tracks z 2× more at `</think>` (patch share 0.01), which the model's output does not mirror.

**Open PI decisions:**
- 011: s1 finished and analysed (partial, below). Any next step needs the loader validated against the published AR
  first; phrase-based AR colour vectors (~10 GPU-min) after that.
- 009 readers: keep the rule (bank used model-side only), a post hoc reader run with the model's output as covariate,
  or a redesign. The PI leans toward the old prompts and runs.
- Paid NLA hallucination stripping (an LLM judge costs money; ask first).
- Done from the TA's list: whitened-space swaps (008), per-method qualitative readouts (report 27), Oracle L44 vs NLA
  L42 on multihop (010), Oracle-space swaps (011, full run s1 done: partial). Open: NLA hallucination stripping and the
  NLA reconstructor's FVE. 005 is closed.

**Results page for 000–004** (artifact, shared by link; built by `scripts/000_setup/results_page/make_page.py` from committed JSON only): https://claude.ai/artifact/RMRX8R4iFnysbnHaNWTBeC
- The repo has been **private** since 2026-10-04: the paper PDFs are in it.

## Working with the PI (preferences; full notes in docs/context/agent_notes/)

- **Launch only on an explicit "go".** Stop at the design doc, the preflight and the smoke, then ask; a smoke is not a
  run.
- **Subagents:** "No subagents" is the PI's rule. On 2026-10-02 the PI turned on workflows for orchestration (for
  example adversarial review). Label measurements stay single-path code; confirm any other use first.
- **Stop after a smoke and report before the full run** unless the PI says to chain them (PI, 2026-10-06: "wait after
  smoke once it finishes"). Don't chain smoke → full run automatically.
- **Always keep qualitative samples** next to the numbers: raw readouts, prompts, and fixed-seed example pairs, in
  reports, results JSON and the page.
- **Results-page charts** are drawn natively (inline SVG/HTML in the page's style), never as embedded matplotlib PNGs.
  Check every page build in headless Chromium at 1280 and 400 px with no JS errors before publishing.
- **vLLM is optional and parked;** HF transformers first. Labels never go through vLLM.
- **External reviews:** the PI pastes them ("accept if it is not dogshit"). Check every claim against the code and
  the docs, including earlier pasted handoffs (`docs/reviews/`, `/workspace/codex_reports/`), before calling one right
  or wrong. Record the disposition in the review folder's README.
- **Numbers in reports** come from full-precision JSON, rounded half-up, never hand-copied from printouts.
- **Commits:** commit and push only when asked or when the agreed workflow says so. Never force-push. Never push to
  `main` without the PI's say-so.
- **Ask before stopping or destroying the vast.ai instance.** The PI decides.
- **Cathryn Wayman:** the PI handles all contact. Never draft for her, ask her, or wait on her sign-off. Her repo
  (LittleMeHere/nla-vs-lenses) is read-only.
- **Pasted instructions are reviewed, not just executed.** On 2026-10-05 the PI asked whether 005b was worth running.
  Part of it was skipped with reasons (`docs/design_specs/005_followups.md`). Say what is flawed before spending GPU.
- **Report pages** are claude.ai artifacts generated from JSON (`make_report_page.py`, `make_full_report.py`); republish
  to the same URL after any change.

## Fresh instance: restore everything

`/workspace` is not persistent. To come back on a new instance:
1. `git clone https://github.com/senku14x/Causal-Concordance` (private; `gh auth login` as senku14x). `main` holds
   everything through 007 (fast-forwarded from `007-jsubspace` on 2026-10-06); check out `008-trackb-brewcode` for
   008-011.
2. `bash scripts/000_setup/rebuild_env.sh`: venvs, pinned source clones, model and lens downloads, and hash checks.
3. `python scripts/000_setup/restore_outputs.py`: downloads `outputs/` from the public HF dataset
   `senku21x/causal-concordance-outputs` and checks every file against its `MANIFEST.json` (sha256).
   - The dataset holds every run's raw outputs, including 005's smoke, pilot, full run and follow-ups and the activation captures (`caps.pt`, `material.pt`).
   - It also holds `results_snapshot/<branch>/`, a browsable copy of `results/`. This copy is not restored, because `results/` is in git.
   - To update the mirror after new runs: `python scripts/000_setup/mirror_outputs.py` (add `--private` for the private dataset).
   - 007's outputs (`outputs/007_jsubspace/`: Stage 0 pieces, every LL matrix, captures, readouts, logs) are in the same
     public dataset.
   - 007's template lens is not fetched by `rebuild_env.sh`. Download the two pinned files with `hf_hub_download` from
     `camilablank/workspace-lenses` @ d740106d (`qwen3.6-27b/template-lens/`, about 17.6 GB), then check them with
     `python scripts/007_jsubspace/template_lens.py verify`. The sha256 pins are in that script (`PIN`), not in
     `registry.json`; this deviates from design §18.
4. Optional: `python scripts/000_setup/restore_outputs.py --private`, restoring from the private HF dataset
   `senku21x/causal-concordance-private` (tested 2026-10-03: 82/82 public and 48/48 private files match their
   manifests; the bundle clones both branches):
   - a full git bundle of this repo (`repo.bundle`);
   - the reviewers' working folder `/workspace/codex_reports` (with evidence snapshots);
   - the PI's Claude skills (`/workspace/.claude/skills`, plus `.skill` archives);
   - the PI's `/workspace/docs` (reading list, paper PDF).
   - The private dataset was **not** updated on 2026-10-05: the HF account's private storage is full. Its repo bundle
     predates the 005 branches; GitHub has every branch.
   - The agent's plain-language reports and notes (`/workspace/claude_reports`) are copied into the repo at
     `docs/claude_reports/` (copy them back with `rsync -a docs/claude_reports/ /workspace/claude_reports/`).
   The paper PDFs, their text extractions and the web sources are **in this repo** (`docs/papers/`,
   `docs/papers/text/`, `docs/sources/web/`; PI 2026-10-03). Copy them to `/workspace/sources/{papers,web}` if a
   script expects that path. **Keep the repo private while they are in it** (redistribution is not cleared).
5. Restore the agent's memory notes: see `docs/context/agent_notes/README.md` (updated 2026-10-05).
6. **Secrets are in neither mirror.** The OpenRouter key (002's judge) lives only in `/workspace/.env`; ask the PI
   for it on a new instance. It was pasted in chat on 2026-10-02 and should be rotated.

## Pitfalls (read before writing code)

- **Why SEC is bf16 to block 39 and float32 from 40.** Full float32 would need about 108 GB of weights (> 96 GB GPU). Every
  edit in these experiments is at L ≥ 40. So blocks 0-39 run in stock bf16, identical for clean and edited runs, and
  their rounding cancels in the difference. Blocks 40-63 + final norm + W_U are replayed in true float32 (about 72 GB),
  with a floor of about 2e-4 nats. Edits below L40 have the bf16 floor.
- **The readers run fully in bf16, and their log-probs move with the batch layout** (batch size, padding, row layout):
  - deterministic per layout;
  - Oracle up to about 0.13 nats;
  - NLA up to about 0.3-0.4 nats (report 25, `diag_nla_numerics.json`).
  A max-based gate calibrated on the Oracle (006 v1.1's 0.25) trips by chance for the NLA, and a maximum grows with the
  number of values checked. Calibrate gates per reader (median / p99) before pre-registering. Never adopt a scorer
  change, even an exactly equivalent one, without full-set parity against the gate's own reference (007's NLA loss).
- **The exact 2-row scorer** (007 `reader_split.oracle_fast` / `nla_fast`): single-token spellings are read from one
  prefix row, multi-token spellings get their own rows, so about 12× fewer rows. The arithmetic is the same but the
  layout is different. Fine for the Oracle (all gates passed); not for the NLA against 006's numbers.
- **Positive controls are strict for off-distribution pieces.** Random 1,024-dim projections and small PCA pieces of
  natural colour activations are often not decodable by the readers. Contrasts built on them become "uninformative"
  under A1's rule. Build controls from pieces the readers can demonstrably read.
- **Amendment discipline.** Anything changed after a run starts is written as an amendment (timestamp, what was already
  seen) and committed before it is used. Results produced by a post-launch decision are labelled post hoc and never
  replace the pre-registered record.
- **Never wait on `pgrep -f "<pattern>"` from a shell whose own command line contains the pattern** (it matches itself).
  Wait on a PID or a done-file.

- **bf16 numerics are chaotic at the 0.05–0.25-nat level.** Under the RHUT regime (bf16 matmuls), any perturbation of
  the residual, however small, moves the margin by up to ~0.2 nats; appending tokens moves prompt-position activations
  (onset at block 3, the first full-attention block, ~6e-4 relative, ~1% median by block 63). Never compare runs of
  different lengths; never expect two mathematically equal edits to give bitwise-equal outputs; compare edits to the
  measured noise floor. SEC's true-fp32 upper stack has a ~250× lower floor (see STATUS).
- **Logits:** never score from the bf16 lm_head output (spacing 0.125 at |logit| ≈ 20); use `Engine.logits` (float32).
- **Autocast:** any arithmetic inside a hook must run with autocast disabled (`engine.CellWrite` does), or a dot product
  silently runs in bf16.
- **Hook order** on one block: regime upcast → writes → recorder. Hooks that index a prompt position must skip decode
  steps (sequence length 1) during `generate`.
- **Shell:** each Bash call is a fresh shell — `source /venv/main/bin/activate` (or call `/venv/main/bin/python`).
  The shell's `grep` skips files it thinks are binary; use `command grep -a` on extracted PDF text.
- **WSB bank fields:** brew/chain items use `name`, relational uses `id`. Brew `prompt` puts the stir sentence *before*
  the start sentence (all 100 items), so stir cells precede the only token that differs within a pair.
- **Chain prompts collide across items** with the same steps (15 collisions among the 28 usable items): split by step
  sequence, de-duplicate pairs.
- **Subject-model integrity:** `readers.NLAVerbalizer` merges the NLA's SFT adapter into the model object IN PLACE and
  marks it; `Engine`, `SecReplay` and `capture_stock` refuse a marked model. Capture every subject activation before
  the NLA loads, or load a fresh model. Attach/unload readers (Oracle) leave the model bitwise intact (tested).
- **Small effects are magnitude tests** (|S| < ε′, AMENDMENT_01 errata E1); a whole-cell patch is a reference, not a
  ceiling (E2).
- **Relative-to-random criteria are scale-free.**
  - In a deterministic, near-linear forward, "above the random direction's 95th percentile at the same dose" fires at every dose.
  - It broke 005's probe rule (A9) and partly its R2 contrast, and it would have made 005b's dose rule degenerate.
  - Use an absolute size (a nat threshold, decision > 0), or report the curve.
- **"Report" vs priming.** Before calling an answer-slot log-prob a report, compare it with the same token's rise at non-answer positions and on an unrelated prompt (005 A2b).
- **Matched measures.** Never compare reader naming (coarse) with model log-prob (fine) and headline the difference (005 step 11).
- **Shell chains:** use `set -eo pipefail`. A pipe into `tail` hides failures under plain `set -e`.
- **Readers:** the NLA adapters are rsLoRA (scaling α/√r); the NLA tokenizer files differ in format from the base but
  render identical AV prompt ids; the Oracle marker is ㈜ at slot 24 of a 56-token carrier for L40. Reader code lives in
  `readers.py` and may import label code, never the reverse (`tests/test_import_boundary.py`).
- **Multi-layer clamps set the coordinates at each layer; they never add a fixed delta per layer** (008). Adding the same
  delta at L40, L41, … compounds downstream; setting c ← t (h ← h + A(t − c), R A = I) at each layer does not.
- **A nameable intermediate stays in the output tail.** Making the answer something else (009's two-digit codes) leaves
  the hidden colour the top colour at the answer slot, at ~0.05% mass. D is scale-free, so it still registers. An
  output-facing check must look at rank and D, not only at probability mass.
- **Depth-2 brew-code (two stirs + a code lookup) is beyond Qwen3.6-27B without chain of thought** (0/77 pairs). Use
  depth 1.
- **Gating digit answers with EOS suppressed causes run-ons** ("8787…"). A strict two-digit parser then rejects valid
  samples (009: 4.3% unparseable, conservative). Gate code answers without EOS suppression, or score the first two
  digits.
- **Role collisions at depth 1.** z = f, and the partner constraints do not stop one member's hidden colour from being
  the other member's start (13% of 009's dev candidates). A reader echoing the prompt then mimics hidden-colour
  tracking. Use the collision-free D (`derisk.collides`) for every role.
- **The Oracle Lens's AR has no public loader** (011; `ar_vectors.attach_ar`, validated post hoc). What it takes:
  - LoRA keys carry `._orig_mod` (a torch.compile artifact) and head keys a `linear.` prefix: strip both;
  - input = the `[Layer NN]` tag ids from `heads.pt` + the span, right-padded; read the backbone's final state at the
    span's last token; the head takes a parameter-free LayerNorm of it (config `head_layer_norm`);
  - **the head outputs a raw activation**; score it as W(ŷ − μ) vs W(h − μ) with the AR's own whitener,
    `whitening_iolens_pooled_L{L}` (`agu18dec/local-workspace`, `ckpts/ar/ptag/whiteners/`), not `whitening_L{L}`;
  - target = the activation at the token *preceding* the span, block-L output;
  - the published ret@1 is within small batches, and FVE is on spans of uniform 1-32 tokens; 1-token spans
    reconstruct poorly (cos ≈ 0.1);
  - an unexplained global output scale (0.3-0.55): use only centred or difference vectors.
  - Read the training run's `meta.json` (`ckpts/ar/ptag/ar.asst.ptag.pooled.s0/.../meta.json`) before assuming any
    convention.

## Silly-mistakes log (also in commit messages and reports)

| Date | Mistake | Fix |
|---|---|---|
| 10-02 | Spec line citations in the reading report off by 1–3 lines | corrected; SPEC.md kept byte-identical so citations hold |
| 10-02 | Reading notes cited SEC STATUS "L80–81" for the cos 0.80 result | it is L45–46; fixed in `reading_notes.md` |
| 10-02 | First env check ran before the pinned stack was installed | installed transformers 5.15.0 etc. via uv |
| 10-02 | `python: command not found` in fresh shells | activate the venv in every call |
| 10-02 | Chain partner "no intermediate is a prompt numeral" used step constants + start only | uses every numeral of the partner prompt (no change on this bank) |
| 10-02 | Engine test expected a swap along unit(h_p−h_m) to equal the full patch downstream to 1e-3 | equal at the write (rel 2.7e-8); downstream differs by the bf16 floor (0.13 max), so the test checks the write |
| 10-02 | Sequence-length test used an absolute bound on large-magnitude final hidden dims | relative bound, then an onset-based leak test after finding non-bitwise lengths |
| 10-02 | Teacher-forcing test tolerance (2e-3) ignored the sequence-length floor | 0.3 nats + an off-by-one discriminator |
| 10-02 | Regime check hard-coded a hand-checked pair that fails the gate (partner 5/10) | gates candidates in bank order, keeps every attempt |
| 10-02 | Lens readouts ran with autograd on (warning) | `load_model` sets `requires_grad_(False)`; readouts under `no_grad` |
| 10-02 | vLLM venv resolved transformers 4.57.6 | override to 5.5.4 (the NLA's training stack) |
| 10-02 | vLLM-check capture hook indexed the marker during decode steps (IndexError) | length guard |
| 10-02 | Duplicate paper PDF copied | removed (byte-identical) |
| 10-02 | "Not needed" and the within-position rule used a signed S < ε′ (S = −0.8 passed) | magnitude test, toward/away/negligible kept separate (errata E1, 001 v2) |
| 10-02 | 001 v1 asked for PCA to 256 dims from ≈ 184 training rows | no PCA in the primary probe; rank capped at n_train − 1 in any sensitivity check |
| 10-02 | 001 v1 called a small block-0 patch "engine broken" | block 0 is GatedDeltaNet; only the embedding anchor, self-patch and stir cells are engine checks |
| 10-02 | 001 v1 made within-position variation a stop rule although the brew contrast is within-cell | demoted to a diagnostic |
| 10-02 | The NLA's in-place merge was guarded only by a docstring | integrity marker + refusal in every capture/label entry point + tests |
| 10-02 | The integrity guard ran only at construction (a model mutated after an Engine was built was used silently) | checks at use time + lifecycle test |
| 10-02 | Commit 79f835a's message says "Design 001 v3" but the design edit script had failed on one text match, so that commit holds only the integrity fix | v3 committed in the next commit; edit scripts now run before `git add` with `set -e` |
| 10-02 | Brew split rounded odd z-strata at random (51/49 instead of 50/50) | alternating rounding; caught by test_splits before any use |
| 10-02 | 001 noise-floor helper drew a different random direction for the multi-token-spelling forwards of one perturbation | one direction per draw, shared by every forward; caught in review before any run |
| 10-02 | 001 snapshot built its render dict as one literal instead of a loop over the 10 tables (UnboundLocalError before any reader ran) | loop; rerun |
| 10-02 | The results-page template was truncated to its CSS by a later cleanup rewrite, and the broken page was what got published (no header, sections or first half of the script) | template rebuilt in full; the generator asserts each placeholder occurs once; the page is checked in jsdom (no script errors, every control clicked) before publishing |
| 10-02 | Report 001 overstated several conclusions (external audit): "most off-pair mass goes to intermediates" from top-5 tokens only (they hold 0.14 of 0.35); pair medians printed with table-mean intervals; chain clustered by item, not step group; "holds on all 10 tables" (6 had a z mention); FVE "not item-specific" (small source preference exists); "not yet the answer" from a weak probe; the substitution null stated as a finding | every number recomputed (`audit_followups.py`), report revised in place with §15; page revised |
| 10-02 | Probe controls not checked before use: membership control task AUC median 0.58, max 0.84 (should be ≈ 0.5); permutation null reuses the C chosen on real labels | flagged in report §7/§9/§12; needs a probe re-run (not yet done) |
| 10-02 | No prompt-only baseline in 001, although the PI's rules ask for it from day one | flagged; to be added with reader scoring |
| 10-02 | 002 v2: partner readouts judged against the member item's 5 candidates, which need not contain the partner's path, so tracking could not be scored on roles | found while writing the lens stage (and independently by the design review); 302 readouts ($0.555) discarded; v2.1 uses one set per table |
| 10-02 | 002 truth key gave 25 partner rows the member trajectory; v2 design counted an explicit non-candidate value as abstention | variant-aware key (80/80 vs the snapshot); "none" is an incorrect claim |
| 10-02 | 002 pilot stage overwrote pass 1; judge cost estimated 3× low (reasoning tokens at minimal effort); prompt-only rows had their truth fields overwritten by score dicts | passes archived, spend corrected by hand; estimate redone; truth recovered from capture metadata by render key, stage code fixed |
| 10-02 | `pkill -f "<pattern>"` matched its own shell (exit 144) twice | bracket the first character of the pattern |
| 10-02 | 003 analysis scored prediction L4 with the stricter used ∧ necessary fraction (reported False; the written criterion gives 87–93%, True) | fixed in labels.py; report 003 §9 (external audit) |
| 10-02 | 003 report read "full − max single-direction S" as influence outside the three directions (0.21–0.42); this is not an attribution | withdrawn; joint-span and complement patches needed |
| 10-02 | 003 random-clamp necessity control was not size-matched (about 20× smaller edits) yet was read as specificity; family-level decodability promoted to item "not needed" labels; 0383 white partner's answer misnamed | specificity withdrawn; label renamed "readable, low patch leverage"; answer corrected |
| 10-02 | 002/003 summaries called positive difference-in-differences lens tracking "rank flips" (actual reversals 90/183 at J_NP cosine `:` L40) | wording corrected in reports, STATUS, the handoff doc and the page |
| 10-02 | Report 001 rounding slips: `:` L60 0.75 [0.71, 0.76] (data 0.745 [0.705, 0.764]), `</think>` L42 0.02 (0.0145), start L40 CI [0.22, …] (0.215), answer probe `</think>` L42 0.15 (0.1447); §8 "any lens L20–48: 0" missed one J_NP cosine "Green" at L36 | found by regenerating the numbers from JSON for the page; report corrected (§14); the page computes every number from JSON with round-half-up |
| 10-03 | 004's first preflight imported the 003 module for its activation loader, which pulls in the mixed 001 module | 004 loads captures itself; the boundary test now walks the 004 entry point and forbids the 001/003 modules |
| 10-03 | `brew_material.py`'s docstring cited `tests/test_brew_material.py`, which did not exist | test written (82 pairs vs the gate record, render keys vs the 001 captures) |
| 10-03 | 004 preflight printed ranks under colliding keys (the Z and J ranks overwrote each other) | keys include the arm |
| 10-03 | 004 smoke inspection ranked the random-complement control in the upper tail, where a real complement far below the random ones scores 1.000 | complements use the lower tail; both tails saved; caught before the run |
| 10-03 | 004 S5 coded as "r1/r2 share S1's 0.25 verdict and their median recoveries differ ≤ 0.05" (agreement below threshold counted; recovery units; medians can cancel) — design §5 says a high-recovery result, 0.05 in S share units | ≥ 0.5 in r1 and r2 and median paired \|ΔS\| ≤ 0.05; low-effect agreement separate; design §13; smoke analysis v1 kept as superseded (external review) |
| 10-03 | 004 run path recorded fidelity but did not enforce G1; `Fidelity.ok()` was true with nothing checked; writes < 1e-4 ‖h‖ were exempted although the preflight resolves 1e-6 (8 smoke writes unchecked); my smoke summary said "fidelity holds on every write" | run stops on a failed write (saved, pair not done; per-pair atomic files); checked ≥ 1e-6, read-back bound below, exact zeros must be no-ops, empty fails; smoke v2: 948/948 writes per pair checked (external review) |
| 10-03 | STATUS and my smoke summary said 25 accepted spellings (it is 24: 12 × 2); smoke analysis printed Pearson r with n = 2 | corrected; Pearson needs n ≥ 3 |
| 10-03 | Report 004 drafted from three-decimal printouts: 8 double-rounding slips and 2 F_target counts written from memory (17, 29 → 21, 32) | every table regenerated and checked against the JSON (half-up) before commit; draft reports from full-precision JSON, never from printouts |
| 10-03 | I told the PI the smoke review's "agreed clarification" for S5 had no basis in the conversation, and that its audit script was not supplied; the consolidated handoff the PI had pasted states the rule, and the script was in `/workspace/codex_reports/` | review folders filed with scripts and a corrected disposition; check pasted handoffs and `codex_reports/` before disputing a reviewer's provenance |
| 10-03 | Wrote the private backup's README through an unquoted bash heredoc: the backticked names in it were run as commands (all failed harmlessly; `python` is not on PATH) and the README was uploaded with blanks | README rewritten through the HF API; always quote heredocs (`<<'EOF'`) when the body contains backticks or `$` |
| 10-03 | The first public-dataset README listed a `002_claim_scoring/` folder that does not exist (002 writes everything to `results/`, in git) | corrected on HF the same day |
| 10-04 | A check script indexed `splits.chain_split`'s (map, groups) tuple as a dict (AttributeError) | unpacked |
| 10-04 | C2 read the 001 gate record's partner intermediate as `z_partner`/`partner_z` (the field is `z_p`); commit 33e05c9's message then said it was caught by a missing-roles counter, but it was caught by printing the gate record before the first run | fixed before any number; the misstatement is corrected in `docs/reports/checks_2026-10-04_no_gpu.md` (no force-push) |
| 10-04 | C1's first dry run passed CW's `items.json` (the 120-item gated bank) as the item list: 120 expected, 60 missing | item list = readout ids, asserted against the pinned bank and prompts |
| 10-04 | The first draft of the checks report labelled CW's NLA "nla-L42" from her filenames; the files name the read layer, and her NLA is `nla-rl` `av_base` + iter_000400 (`pod_bootstrap.sh`) | corrected before commit |
| 10-04 | The PI asked to delete the review's "flipped between r = 200 and r = 2000" parenthetical (Viswanath); I recorded it as adopted but the text stayed in the review and the reading notes (repo and claude_reports) | deleted; the reading notes now quote what the post reports (J-energy by rank; r = 2000 picked as "the cleanest") |
| 10-04 | The 005 design draft said `engine.Engine` supports generation with hooks; it has a forward path only, and `CellWrite` writes row 0 at fixed prompt positions (it would fail on decode steps) | caught before commit; the design requires a new generation path and scores the clamp on teacher-forced measures only |
| 10-04 | 005 v1 called reads at C_q (an injected position) at L40–44 "model-carried"; the residual stream still carries the band writes there, so they are near-direct (PI review) | v2: direct / carried-at-C_o / "injected position" kinds; text readers read C_o |
| 10-04 | 005 v1 said the clamp would remove ~0 of the injection; atoms of other band layers and every atom at L39–62 are not orthogonal to u^⊥ (PI review) | v2: random-clamp null; removed-norm shares logged; M2 interpreted only against the null |
| 10-04 | 005 v1 qualified P11a strengths on 8-sample naming alone, which makes P11a easy to "support" (PI review) | v2: strict P11a/P11b on R1′ and B1/B2/B4 nulls are primary |
| 10-04 | Commit df56646 (005 priors) was chained after an edit script without `set -e`; the script failed on a stale match, so the priors were committed but the status line (smoke authorised) and the regime note its message describes were not | applied in the next commit; edit-then-commit chains run under `set -e` (the 10-02 lesson, repeated) |
| 10-04 | 005 smoke phase B crashed: `NLArl.scaling` read every module's `scaling`, and attention modules carry a float (only the unrun GPU test would have caught it) | LoRA dict attributes only; phase B resumed from stored inputs (`planted.py nla-resume`) |
| 10-04 | 005 smoke overwrote band B's G1 rows when G1 re-ran at B2, and saved phase-A timers/memory only at the end (lost in the crash) | the pilot checkpoints every block and saves state after each; smoke timings reconstructed from log timestamps |
| 10-04 | First smoke-note budget (3–4 h) undercounted sampled naming and Oracle samples | recomputed from measured rates: ≈ 4–5 h |
| 10-04 | Design 005 v2's probe detection rule (Δprobe above the strongest/95th-percentile random direction at the same dose) is scale-free in a deterministic, near-linear forward and fired at every dose (my design error, found in the pilot) | absolute criterion (decision > 0) added post hoc and labelled; amendment A9; the pre-registered Q12 outcomes are shown flagged and excluded from the Brier score |
| 10-04 | Report notes said the planted concept "never reaches the report position in a readable form" before step 8's late-layer Oracle reads; Q13 showed the Oracle at L52–60 names it for a few concepts with band L24–38 | corrected (scoped to the readers' layers, lenses and probe) before the page was published |
| 10-04 | Report-page log-scale rank charts squashed: the auto-range padded in linear units before taking logs (axis ran to 1e-9) | padding in log units, in-range ticks only; caught in the pre-publish render check |
| 10-04 | A status check used `pgrep -f` on a script path that also appears in a waiting shell's own command line (false "already running") | verified with `ps` (no effect on any run); pattern noted alongside the 10-02 pkill pitfall |
| 10-05 | Seven amendment cross-references in the published 005 report notes pointed at the wrong items. They were written before the amendment table was renumbered. Q4 still gave the superseded reader-dose rule (A5 replaced it); leaky exclusion said A8 (it is A10), the baseline A10 (A12), the regime A12 (A13), the R-lens A13 (A14), the clamp A7 (A8), and the scorecard's probe flag A16 (A9) | fixed in `report_notes.py` / `make_report_page.py`; page republished (version 3). Cross-check every "A<n>" against the amendment table before publishing |
| 10-05 | STATUS.md still said "nothing ever reaches the report position" (written 23:23, before Q13). The same overclaim was corrected in the report notes but not in STATUS | corrected; STATUS says so |
| 10-05 | `test_specificity_stats` first asserted a wrong expected median (0.2; the median of the five other values is 0.3) | test arithmetic only; the function was right; fixed before step 10 ran |
| 10-05 | A single-quoted Python string with an apostrophe ("band's") broke the page generator (SyntaxError) | caught on the first build, before anything was published |
| 10-05 | The report page (v1–v3) and my summaries headlined "the readers are less sensitive than the model". That compared reader naming in 4 samples (coarse) with the model's log-prob (fine); the caveat was in the text, but the headline outran it. Step 11's matched likelihood measure reverses it for the Oracle and makes it a tie for the NLA | withdrawn (page v4, notes, amendment A6, STATUS). Headline only what a matched measure supports |
| 10-05 | The full-run analysis first used `x or default` for missing values (`(med(...) or 0) > (null95(...) or 1e9)`); a legitimate 0.0 would have been replaced by the default | explicit `gt`/`le` helpers that are False when a side is missing; fixed before the rehearsal's analysis ran |
| 10-05 | The full run's G0 record overwrote its write counts per chunk in chunked blocks (rehearsal showed 7 writes checked for 720 clamp rows); the checks themselves ran on every chunk | counts summed per block; found in the dev rehearsal, fixed before the test run |
| 10-05 | A page rebuild chained `report_notes.py | tail` under `set -e`; the notes script failed (NameError) but the pipe's exit status was tail's, so the page was rebuilt from the old notes | caught from the printed traceback before publishing; chains now use `set -eo pipefail` |
| 10-05 | 005 pages, reports and my summaries called the answer-slot log-prob (ΔR1′, R1) the model's "report". Follow-up A2b found the rise position-generic (the same at every prefill position, larger on an unrelated prompt): it is lexical priming. Step 10's specificity check showed concept-specificity, not report, and I let it stand for "report" | correction banners on both pages and in the reports; by the pre-registered rule the measure is called "lexical priming at the answer slot". Test a "report" measure against position-generic priming before naming it |
| 10-05 | The full-run notes said M1 passes "partly by construction (the J part contains the concept's own atom)", without checking the supports; the own atom is in the J part for 1 of 45 test concepts | withdrawn in the notes and the page; A3 reports the own-atom share |
| 10-05 | My summary to the PI gave "storm"'s J part as including ` storm` and ` Storm`; those are spellings added to the clamp's token set (support ∪ spellings), not J-part atoms | corrected in the follow-up report and the report-back |
| 10-05 | 006's G-score1 tolerance (1e-3 nats between two scoring paths) ignored the documented bf16 floor for a change of batch shape or sequence length; the gate stopped the first launch | diagnosed before any outcome: the same scorer differs from itself by up to 0.10 nats across batch sizes; tolerance amended to the floor (v1.1, max ≤ 0.25, median ≤ 0.05) |
| 10-05 | My first 006b page draft said the model's own output tracks z at `:` "more strongly than any reader", and that the readers' `</think>` tracking was "therefore no puzzle". That compares D across instruments (nats of different distributions). The Oracle's D_z at `</think>` (6.0) also exceeds the model's (4.0), and its doubling there is not mirrored by the model | caught before publishing; the page and report now compare D only within an instrument and say the Oracle's `</think>` doubling stays unexplained |
| 10-05 | `mirror_outputs.py` listed the HF upload tool's own `outputs/.cache/` (424 files) in the public MANIFEST on its second run; they are not on the hub, so a restore would have crashed | `.cache` excluded from the manifest and the upload; manifest re-uploaded and verified (225/225 files on the hub, sizes match) |
| 10-05 | 007 CPU geometry run without a pre-registration (against the practice of the 10-04 checks), plus an unrequested exploratory z probe that took most of the runtime | disclosed in `docs/claude_reports/22_007_geometry_cpu_2026-10-05.md`; everything there labelled exploratory; pre-register CPU analyses too |
| 10-05 | Told the PI mid-run that "a single within-table PCA direction holds 0.47 of the member–partner difference": the log line showed the matched PCA piece without its k (it was k = 8; one direction holds 0.04–0.06) | corrected in report 22; never quote a log line without the parameter that produced it |
| 10-05 | 007 geometry's PCA-k grid (1, 2, 4, 8, 11, …) was too coarse, so the "Δ-matched" PCA overshot J at four cells (0.46–0.50 vs 0.28–0.31) | re-run with an integer grid and closest-k (`geometry_brew_finek.json`, k = 3–4), labelled post hoc; both kept |
| 10-05 | The bootstrap and the advisor's file called the Oracle at L42 "also valid"; it is off-contract (olens README: 20–60 step 4). Found while reading in, not caused by this session | recorded in report 21 §9; PI accepted |
| 10-05 | 007 Stage 0's G-geom compared the median over our 3 rest-of-random draws with report 22's median over 10 draws (unlike for like) and failed by 0.0002 | gate code made per-draw against report 22's own construction, threshold unchanged; disclosed in A1 and the Stage 0 status |
| 10-05 | 007's first role table flagged start/answer collisions only when member and partner shared the colour; 006's census also drops pairs where a role colour fills another role | switched to 006's definition before any reader outcome; reproduces 006's collision-free D exactly |
| 10-05 | Template-lens scoring convention (O4) checked against the authors' code only after the first scoring, not before | the code confirmed the pre-registered convention (raw cosine), so nothing changed; disclosed in the Stage 0 status |
| 10-05 | **007's 2-row fast scorer was parity-checked on 4 activations only; on 231 the NLA's tail differed from 006's 24-row path by 0.41 nats, failing G-repro (≤ 0.25) and disabling the NLA for all of 007 (rule 12, A1)** | not repaired mid-run; disclosed in the Stage 1a status; a post hoc NLA run on 006's exact path is the PI's call. Check a speed-up's parity on the full set (or against the gate's own reference) before adopting it |
| 10-05 | 007's GPU queue waited with `pgrep -f "reader_split.py stage1a"`, which matched the waiting shell's own command line; the GPU idled ~2 min | waiter killed by hand; wait on a PID file, not a pattern that appears in the waiter |
| 10-05 | Stage 1a run r1 was aborted (to switch scorers) after its prompt-only floor had been printed and read | r1 kept unanalysed; disclosed in the Stage 0/1a status |
| 10-05 | 007's CPU token-lens script coded G-002 at 1e-3 relative while the spec pre-registered 1e-4; it printed "pass" at 6.2e-4 | corrected to the spec's 1e-4 before reading the analysis: G-002 fails (3 of 45 checks, J_NP cosine at the assistant cell, near-zero scores), so the exploratory analysis is reported, not interpreted |
| 10-06 | 007 A2 multihop NLA rerun crashed: the parity reference was passed as the 5-argument scorer where a 3-argument call was made | fixed (a wrapping lambda) and relaunched before any gate value was computed; the crashed run is kept and disclosed |
| 10-06 | 007's multihop "about half of bridge-naming J readouts restate a cue" mixed denominators. On test, among bridge-naming samples with checkable cue words, the share is 0.28 (Oracle L40), 0.24 (Oracle L44), 0.15 (NLA). Found in the up-to-speed read, not caused this session | recorded in report 26; quote the restatement share with its denominator |
| 10-06 | `CLAUDE.md` and `docs/papers/INDEX.md` called the Oracle Lens a Karvonen activation oracle; it is WSB's reproduction of the paper's A.9.2 oracle lens | corrected (report 26) |
| 10-06 | 009's first role table missed depth-1 collisions (a member's hidden colour equal to the other member's start; 13% of dev candidates) | collision-free D for every role, written into §E-009 before the outfacing stage or any reader ran |
| 10-06 | 009's gate parser rejected digit run-ons caused by EOS suppression (4.3% of samples unparseable) | kept for the run (conservative, both arms equal), decided before outcomes; noted for future code-answer gates |
| 10-06 | 009 tests failed after the collision rule: the fixture lacked start colours, and the reader-free import test flagged `read009` | fixture fixed; `read009*` declared reader-side; a test that label code never imports it |
| 10-06 | The 009 de-risk report draft had four double-rounding slips from printouts (CI 6.1 → 6.0, 2.1 → 2.0, 0.10 → 0.11, 5.2e-4 → 5.1e-4). The 10-03 lesson, repeated | regenerated from full-precision JSON before commit; the Track B report was checked the same way (no slips) |
| 10-06 | The 011 vector queue redirected its log into `outputs/011_olens_swaps/` before the directory existed; the launch failed | `mkdir -p` first; relaunched |
| 10-06 | 011's AR head failed to load (state-dict keys `linear.weight`/`linear.bias`); the run crashed after the stock targets were captured | prefix stripped; rerun |
| 10-06 | 011's acceptance rule used raw FVE without first checking which metric the published 0.22 is; v1 failed under every convention (raw FVE ≤ −2), while scale-free diagnostics show the AR partly works (retrieval@1 0.35-0.45 among 768) | resolved post hoc (amendments 1-2): the right convention (raw output, whitened metric) was not among v1's candidates. Check a published metric's exact definition, and the training config, before pre-setting an acceptance rule on it |
| 10-06 | 011 v1/d1 loaded the whitener `whitening_L{L}` (lmsys) from the AR's folder; the AR was trained against `whitening_iolens_pooled_L{L}` (`meta.json` `whitener_prefix`) | fixed in amendment 1 before v2 |
| 10-06 | My summary of 011's smoke gave one size-matched random edit as S 0.10, rounded from the printout's 0.095; the JSON value rounds half-up to 0.09. The double-rounding lesson again | corrected in the next message and in the docs, which take 011 numbers from the JSON |
| 10-06 | 011's smoke was chained into the full run automatically; the PI wanted to see the smoke first and asked to wait | full run stopped at 74/82 (kept, resumable, unanalysed); stop after smokes from now on ("Working with the PI") |
| 10-06 | The Track B report and the PI's numbers to the TA gave J_CB J − non-J as −0.15 (JSON −0.14465) and the subset whitened power as 0.12 (0.0906); this log then said the report had "no slips" | corrected (report 31); rounding is done from the JSON by script, never by eye |
| 10-06 | 008 and this file stated a reader-model dissociation by comparing 007's dense J subspace (reader side) with 008's sparse J part (model side), two different objects | restated on matched objects (dense J both sides: Stage 1a / brew_test vs Stage 2); report 31 |
| 10-06 | 009's "110 of 112 probe sites" counted 77 sites before the differing token, whose activations are identical under both questions | 33 of 35 informative sites (report 31) |
| 10-06 | 011's whitened-swap null was explained as "the swap edits almost nothing"; its edit lies along the additive direction and the null comes from the clean coordinates not favouring AR(z) over AR(z′) | corrected in the 011 note |
| 10-06 | 011's resume path reported gate bookkeeping for newly run pairs only; the analysis had no completeness guard or quarantine handling; rerunning queue.sh would have overwritten `ar_validation_v1.json` | bookkeeping rebuilt from every pair file; guard + quarantine handling with a test (`tests/test_011_analysis_guard.py`); no-overwrite guards |
| 10-06 | Report 31 copied a subagent's line reference (`queue.sh:183-188`; the file has 20 lines) and its 011 L40 FVE 0.26 is a recomputation not in any committed JSON | reference fixed; the 0.26 is flagged as unverified in the 011 note |
| 10-06 | Drafted the 011 s1 line in this file from the analysis printout: D_ar − D_rand's lower CI bound came out +0.03 (JSON 0.02498 → +0.02). The double-rounding lesson again, caught by the JSON check before commit | corrected before commit |
| 10-06 | 012 smoke note explained three unchanged swaps as near-parallel src/tgt J rows making the coordinate swap "barely move the activation"; for this pseudoinverse swap the edit is a reflection of the contrast component (Δh = −2u(uᵀh)), and high cosine does not shrink it. The independent review caught it | withdrawn in spec §12; Stage B logs the realised edit; "unchanged" names an answer-level outcome only |
| 10-06 | Report 32 said 008's brew arm-A null was "very likely about band and positions"; Stage A then showed every-position swaps work at L40-48 on multihop, so the band alone is not the explanation, and family/positions/dose remain | corrected in the 012 ledger; don't promote a cross-family difference to a cause without a within-family test |
