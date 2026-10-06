# Come up to speed — Causal-Concordance (state of 2026-10-03; see the update box)

> **Update 2026-10-06: read this first.**
> 1. **Check out `007-jsubspace`.** A fresh clone lands on `main`, which stops at 004; `007-jsubspace` holds everything
>    through 007.
> 2. **Read the workspace paper first** (Gurnee et al. 2026; PDF `docs/papers/2607.15495.pdf`; text
>    `docs/papers/text/2607.15495_html.md` or `.txt`; digest `docs/papers/text/2607.15495_NOTES_digest.md`), then the
>    reading at the top of `CLAUDE.md`.
> 3. Then `CLAUDE.md` ("Where things stand", "Pitfalls"), `STATUS.md`, `docs/reports/007_jsubspace.md` and
>    `docs/claude_reports/25_nla_stops_and_open_issues_2026-10-06.md`.
> 4. 007 in one line: the Oracle reads z through the J subspace (confirmed on held-out tables); the model's answer gives
>    J no privilege; the NLA (post hoc) does not share the Oracle's J dependence. Results page:
>    https://claude.ai/artifact/8vnKTzcXczn4YGMiP1uJTW
> 5. Restore outputs on a fresh instance with `scripts/000_setup/rebuild_env.sh` and then
>    `scripts/000_setup/restore_outputs.py` (`CLAUDE.md`, "Fresh instance").
>
> **Earlier update, 2026-10-05:**
> 1. **First read Anthropic's workspace paper** (Gurnee et al. 2026, `docs/papers/2607.15495.pdf`; text in
>    `docs/papers/text/`), then the relevant reading listed at the top of `CLAUDE.md`. This is the PI's rule; do it
>    before anything else.
> 2. This file describes the state of 2026-10-03 (experiments 001–004).
>    - Since then: SPEC_v2 + errata 01, the 005 planted-concepts experiment (smoke, dev pilot, full run, follow-ups), and 006 (reader likelihoods at the brew `:` cell).
>    - All of it is on branch `005-amended-full-run`.
>    - For the current state, read `CLAUDE.md` ("Where things stand") and `STATUS.md`. The newest results are in `docs/reports/005_followups.md` and `docs/reports/006_reader_likelihood.md`.

For a new session or collaborator taking over. It says where things are and why. Rules live in `CLAUDE.md`, the
live state in `STATUS.md`, and the item-by-item spec status in `docs/context/SPEC_STATUS.md`.

## 1. The project in five lines

- **Question:** do interpretability readers (logit lens, J_NP, J_CB, R_CB, Oracle Lens, NLA; SAE later) report
  content that Qwen3.6-27B's computation actually depends on?
- **Method:** judge them against **reader-free causal labels** computed per (token position, layer) cell from
  patches, 1-D swaps, necessity ablations and cross-fitted probes. Then split reader agreement into shared signal,
  shared error and shared prompt inference.
- **Motto:** *agreement is not truth.*
- **Material:** WorkspaceBench only (brew, chain; relational later).
- **The labels measure** demonstrated influence on one specified answer, not workspace membership.

## 2. Read order

0. The workspace paper and the relevant reading (top of `CLAUDE.md`).
1. `CLAUDE.md`: non-negotiables, conventions, environment, pitfalls, silly-mistakes log.
2. `STATUS.md`: current state, what works and what doesn't.
3. This file, then `docs/context/SPEC_STATUS.md`.
4. `docs/context/SPEC.md` (the original spec) + `docs/context/AMENDMENT_01.md` (wins where they disagree).
5. `docs/context/PI_DIRECTIVE_2026-10-02.md`: the PI's directive that set block 000 and the non-negotiables.
6. Designs and reports 001–004, in order. Each report opens with a summary and closes with its silly mistakes:
   - `docs/design_specs/001_leverage_profile.md` (v3 + A1, A2) → `docs/reports/001_leverage_profile.md` (§15 = audit
     response);
   - `002_claim_scoring` (v2.2) → report 002;
   - `003_labels` → report 003 (§9 = audit corrections);
   - `004_span_assay` (v2 + §13) → report 004 (§14 = options for the next step).
7. `docs/reviews/`: every external review, each folder with the review as delivered, the reviewer's scripts and JSON,
   and a README stating the disposition. Includes the **consolidated handoff of 2026-10-03**, which sets the project
   framing (still awaiting PI confirmation).
8. `docs/context/agent_notes/`: the PI's working preferences as kept in the agent's memory (copy them back into
   `~/.claude/projects/-workspace/memory/` on a new instance).
9. Background: `docs/reports/000_reading_report.md`, `docs/context/reading_notes.md`, `READING_LIST.md`,
   `docs/papers/INDEX.md`.

**On a fresh instance:** follow CLAUDE.md, "Fresh instance: restore everything": clone, `rebuild_env.sh`,
`restore_outputs.py` (public outputs, sha256-checked) and optionally `--private` (repo bundle, review evidence,
skills). Paper PDFs, text extractions and web sources are committed in `docs/papers/` and `docs/sources/web/`.

## 3. What has been done

| Block | What | Where |
|---|---|---|
| 000 | Reading pass and report; PI decisions → AMENDMENT_01 | `docs/reports/000_reading_report.md`, `docs/context/AMENDMENT_01.md` |
| 000 | Environment pinned, 29/29 hashes, rebuild script | `scripts/000_setup/`, `scripts/common/registry.json` |
| 000 | Shared code + 78 tests: renders, partners, gate, margin, RHUT and SEC engines, integrity guard, stats, splits, readers, figures | `scripts/common/` |
| 000 | Reader smoke tests on one brew cell (all six readers work; NLA positive controls pass) | `results/000_setup/analysis/reader_smoke.json` |
| 000 | Precision regime check (RHUT vs SEC) | `results/000_setup/analysis/regime_check.json` |
| 001 | Design v3 + A1, approved; gate, smoke, run (72 min), probes (25 min), reader snapshot (32 min), analysis, figures | `scripts/001_leverage_profile/`, `results/001_leverage_profile/` |
| 001 | Report, revised after an external audit; every audit number recomputed | `docs/reports/001_leverage_profile.md`, `scripts/001_leverage_profile/audit_followups.py` |
| 001 | Probe repair (amendment A2): per-fold AUC, multi-draw controls, full-procedure nulls | `scripts/001_leverage_profile/probe_residual_checks.py`, report 001 §7 |
| 002 | Role-aware reader claim scoring (Gemini 3.8 Flash judge via OpenRouter, validated κ 0.67/0.81), lens closed-set scores, prompt-only baseline; 3,930 readouts, $7.98 | `scripts/002_claim_scoring/`, `docs/reports/002_claim_scoring.md` |
| 003 | Reader-free 1-D labels (decorrelated LS directions, swaps, necessity), 82 pairs, 37k edits; G4 failed → descriptive; audited | `scripts/003_labels/`, `docs/reports/003_labels.md` |
| 004 | Span assay at `:` L42 (z / joint / start / answer spans, complements, PCA and random comparators, paired-rotation necessity), 33 pairs, 56 min | `scripts/004_span_assay/`, `docs/reports/004_span_assay.md` |
| — | Raw-output mirror (public HF) + private backup (HF) + restore script | `scripts/000_setup/restore_outputs.py`, CLAUDE.md "Fresh instance" |
| — | Results page (claude.ai artifact, shared by link) | https://claude.ai/artifact/RMRX8R4iFnysbnHaNWTBeC, `scripts/000_setup/results_page/` |

## 4. What 001 found (dev split; scope: Qwen3.6-27B, brew/chain, SEC regime)

- **Engine:** every exact check passes on every pair. The SEC noise floor is about 1e-5 of the answer gap.
- **Leverage (whole-cell patches):**
  - At brew's `:` cell, the combined pair difference moves the answer about halfway to the partner's at L42–44
    (median S 0.48–0.50), and 0.74 by L60.
  - The start-colour token still carries 0.23 at L40.
  - In chain no single cell matters at L40–44; the final cell carries about 0.66 at L56.
  - **Whole-cell shares cannot say which content matters.**
- **Probes:**
  - z and start decode well at the emission cells at L40–44; the answer decodes poorly until L56.
  - Off-path table-colour membership passes only at L60, **provisionally**: the probe controls need repair.
- **Substitution:** replacing an off-path colour barely moves the answer (39/43 items within the bound) but shifts its
  tail.
- **Readers (descriptive, 10 tables):**
  - The verbalizers mention z above chance, often in the wrong role.
  - A within-table start-swap contrast shows some activation-sensitive signal.
  - Strong colour preferences: the NLA names red in 62% of readouts, the Oracle blue in 93%.
  - The lenses list no palette colour at L40–48.
  - No substitution tracking was detected.
  - **No label or reader score exists yet.**

## 4b. What 002 found (claim scoring; dev; descriptive)

- A judge (Gemini 3.8 Flash, WSB's) read all 3,930 snapshot readouts against a closed candidate set per table. It
  agrees with blind author codes at κ 0.67 / 0.81.
- **The verbalizers rarely claim z in its role:** NLA 5–15% vs mention 42–60%; Oracle 0–3%, with 85% of Oracle L40
  samples self-contradictory. Role-based tracking under the start swap is weak (≤ 0.14).
- **The token lenses carry activation-sensitive z information:** positive start-swap tracking (a
  difference-in-differences of scores) in 47–48 of 48 tables at L40–44, and none for the start. Rank reversals occur in
  about half of the comparisons. This is invisible in top-10 lists.
- Prompt-only (stock model) gets z right in 100% of renders: z is fully inferable from the prompt.
- Next (PI): the label experiment 003, asking whether z at `:` L40–44 is causally used. See report 002 §12.

## 4c. What 003 found (reader-free labels; dev; descriptive because G4 failed)

- Decorrelated content directions: held-out AUC 1.00; fidelity perfect; 82 pairs, 37k edits.
- At `:` and the assistant newline, L40–44, **z has a small, mostly positive effect**: median S 0.03–0.05, positive
  in 85–90% of pairs, above the start and answer directions. It is "used" (≥ 0.10) in only 10–18% of pairs (a few
  tables, e.g. 0383 at 0.36). No single tested direction reproduces the whole-cell effect.
  - Clamping z beats random clamps, but those were not size-matched, so this is not specificity.
- At the `</think>` newline: family-level readable, little net patch leverage (not an item label). At `:` L60 the
  answer direction carries the cell.
- An external audit of 001–003 corrected several claims (report 003 §9). A stable content-specific causal reference
  does not yet exist.
- **Label reliability fails** (κ 0.37), so the labels are not used to score readers. Report 003 §7 lists the next
  options.

## 4d. What 004 found (span assay at `:` L42; dev, 33 pairs; descriptive)

- Whole cell moves a median 0.43 of the gap. Recovery of that: 1-D z 0.06, z-associated span (rank 11) 0.21, joint
  start/z/answer span (rank 33) 0.72.
- **PCA of the activations at the same rank does as well or better** (0.58 at rank 11, 0.89 at rank 33). At matched
  edit norm the fitted spans are still slightly behind (ΔS −0.02). Recovery tracks the share of the pair difference a
  subspace holds.
- The z-span swap mostly raises the recipient's own intermediate, not the donor's answer (target-directed in 4/33).
- Random controls are inert. The z-span clamp beats paired rotations in 31/33 pairs. The z span and its complement are
  far from additive (|I| ≥ 0.15 in 67%).
- Predictions: S2 and S6 hold; S1, S3, S4 and S5 do not. Reading: a descriptive geometry result; no intervention
  construction nominated. Report `docs/reports/004_span_assay.md`; §14 lists the options (default: back to the reader
  panel).

## 5. Known problems (not yet fixed)

1. **Probe controls:** fixed (amendment A2; re-run 2026-10-02). Membership is now decodable at `:` L40–60, but only
   narrowly at L40–44 (selectivity 0.11–0.13); the 12-way row-level null is somewhat narrower than a grouped one
   (no call changes).
2. **Damage readout:** only the top 5 tokens of edited distributions were saved, so most of the flagged off-pair mass
   is unaccounted for. Log the full answer-colour distribution.
3. **Prompt-only baseline:** done in 002 as a task baseline (z 1.00).
4. **Reader scoring:** role-aware scoring done in 002. Still missing: atomic-claim extraction that keeps out-of-set
   colour identities, and matched claim budgets beyond one claim per role.
5. **Oracle and NLA read different layers** (L40/44 vs L42); there is no matched-layer comparison.
6. **Raw outputs** (`outputs/`, gitignored) are mirrored to the public HF dataset `senku21x/causal-concordance-outputs`
   (`MANIFEST.json` there has sha256 per file; last upload at 004). Re-upload after each new run (the commands are in
   the upload step of report 004's commit history; `restore_outputs.py` restores). Some analyses need the raw
   captures, so restore before re-running anything.
7. **004:** fitted content spans recover no more than same-rank PCA. No content-specific causal reference exists yet,
   so causal labels are still not used to score readers.
8. **Branches:** `main` was fast-forwarded to `setup/000-scaffold` on 2026-10-03 (PI). Work on the branch; ask before
   updating `main` again.

## 6. Decisions log (PI; 2026-10-02 unless dated)

- WorkspaceBench only; brew and chain core, relational a core subset, multihop later, arithmetic dropped (A01 §1).
- Readers: Skip-lens dropped; SAE deferred; modulation lens and phrase-J dropped; template lens deferred.
- Oracle on-contract at L40/L44; any L42 arm off-contract and separate.
- Precision: SEC true-fp32 replay for every label quantity at L ≥ 40 (errata E4); RHUT's fp32-residual regime is the
  cross-check.
- Thresholds τ = 0.10, ε′ = 0.05 provisional; freeze in AMENDMENT_02 before any test-split label.
- Reader protocol for the snapshot follows the WorkspaceBench post: lenses top-10 at L20–60 every 4; Oracle 1 sample
  on the same grid; NLA at L42 (amendment A1 to design 001).
- vLLM is optional: go the HF route; slot vLLM in later (the engine fails on sm_120).
- Always keep qualitative samples (raw readouts, prompts, example cells) next to numbers.
- The results page draws charts natively (SVG/HTML), not matplotlib PNGs.
- Workflows (multi-agent orchestration) were allowed for orchestration in one stretch; label measurements stay
  single-path. The standing rule in `CLAUDE.md` is still "no subagents" unless the PI turns workflows on.
- 2026-10-02: branch `setup/000-scaffold` merged to `main`; repository made private.
- 2026-10-02: modulation lens and phrase-J dropped; template lens deferred.
- 2026-10-02: judge = Gemini 3.8 Flash via OpenRouter (key in `/workspace/.env`, never committed); fix the probes, then
  score reader claims properly (002); external reviews of 001 and design 002 accepted.
- 2026-10-02: "do 003"; the Codex audit of 001–003 accepted (report 003 §9).
- 2026-10-03: the consolidated handoff accepted as the basis of design 004 v2 ("i like the design spec start with it
  be careful with it"); the raw-output mirror made **public** ("make the mirror public dont worry").
- 2026-10-03: the 004 smoke review ("seems bullshit of a review") checked; its concrete points were right and were
  fixed before the run (design §13). 004 run on the PI's "go".
- 2026-10-03: back up everything of value to HF (public outputs; private backup: repo bundle, review evidence, skills).
  The papers and web sources did not fit in HF's private quota, so they were committed to this (private) repo instead.
  `main` fast-forwarded.

## 7. Open decisions for the PI

See `docs/context/SPEC_STATUS.md`, last section. Most pressing:
- **confirm the framing**: native reader-configuration comparison is the primary line, causal work a validation
  branch (consolidated handoff);
- **the next step after 004** (report 004 §14): back to the reader panel (recommended), or a bounded DAS-vs-PCA check;
- review reports 001–004 and AMENDMENT_02's contents (τ, ε′, substitution bound) before any test-split label;
- not-needed scoring layer (L60 only, or also L40–44); an Oracle L42 arm;
- rotate the OpenRouter key (it was pasted in chat).

## 8. Practicalities

- **Hardware:** RTX PRO 6000 Blackwell (sm_120, 96 GB); torch 2.11 cu128 in `/venv/main`. Every shell is fresh:
  `source /venv/main/bin/activate`.
- **Model:** the model loads in about 15 s (50 GB).
  - SEC replay parks the bf16 upper blocks on the CPU: use `SecEngine` as a context manager.
  - Never load the NLA before capturing subject activations: it merges in place, and the integrity guard will refuse.
- **Tests:** `python -m pytest scripts/common/tests -q` (`-m "not gpu"` for CPU only).
- **Stages:** `python scripts/NNN_<exp>/<exp>.py <stage>`.
  - 001: gate, smoke, run, probes, snapshot, analysis, figures.
  - 002: see its docstring; the judge needs `OPENROUTER_API_KEY` in `/workspace/.env`.
  - 003: capture, directions, smoke, run, analysis.
  - 004: preflight, smoke, `run --pi-go`, analysis, inspect.
  Each stage is resumable and writes a manifest.
- **Results page:** `python scripts/000_setup/results_page/make_page.py`. Check it in headless Chromium, then publish
  to the same artifact URL.
- **Push:** `env -u GITHUB_TOKEN git -c credential.helper='!gh auth git-credential' push origin <branch>` (a stale
  token shadows the gh login). Commit as senku14x / visheshgupta14x@gmail.com.
- **Rebuild on a fresh instance:** `bash scripts/000_setup/rebuild_env.sh`. `/workspace` is not a persistent volume.
- **Process rules (from the PI):**
  - design spec before every experiment, launched only on the PI's explicit go;
  - gates before outcomes;
  - every silly mistake recorded in the commit message, the report and the `CLAUDE.md` log.
