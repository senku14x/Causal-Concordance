# Come up to speed — Causal-Concordance (state of 2026-10-02)

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

1. `CLAUDE.md`: non-negotiables, conventions, environment, pitfalls, silly-mistakes log.
2. `STATUS.md`: current state, what works and what doesn't.
3. This file, then `docs/context/SPEC_STATUS.md`.
4. `docs/context/SPEC.md` (the original spec) + `docs/context/AMENDMENT_01.md` (wins where they disagree).
5. `docs/context/PI_DIRECTIVE_2026-10-02.md`: the PI's directive that set block 000 and the non-negotiables.
6. `docs/design_specs/001_leverage_profile.md` (v3 + A1) and `docs/reports/001_leverage_profile.md` (§15 = audit
   response).
7. `docs/reviews/`: two external reviews of the scaffold and design (answered by design 001 v2/v3 and errata E1–E4),
   and the external audit of the 001 results (answered in report 001 §15).
8. Background: `docs/reports/000_reading_report.md`, `docs/context/reading_notes.md`, `READING_LIST.md`,
   `docs/papers/INDEX.md`.

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
| — | Results page (private claude.ai artifact) | https://claude.ai/artifact/RMRX8R4iFnysbnHaNWTBeC, `scripts/000_setup/results_page/` |

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
- **The token lenses carry z specifically:** within the candidate set the z-vs-z′ ranking flips with the start swap
  in 47–48 of 48 tables at L40–44, and the start does not. This is invisible in top-10 lists.
- Prompt-only (stock model) gets z right in 100% of renders: z is fully inferable from the prompt.
- Next (PI): the label experiment 003, asking whether z at `:` L40–44 is causally used. See report 002 §12.

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
6. **Raw outputs** (`outputs/`, gitignored) exist only on the current instance, which has no persistent volume, and are
   not mirrored (HF dataset name pending). Committed `results/` and the code reproduce every reported number only
   while `outputs/001_leverage_profile/run/` exists.

## 6. Decisions log (PI, all 2026-10-02)

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

## 7. Open decisions for the PI

See `docs/context/SPEC_STATUS.md`, last section. Most pressing:
- review report 001 and AMENDMENT_02's contents;
- not-needed scoring layer (L60 only, or also L40–44);
- an Oracle L42 arm;
- HF dataset name.

## 8. Practicalities

- **Hardware:** RTX PRO 6000 Blackwell (sm_120, 96 GB); torch 2.11 cu128 in `/venv/main`. Every shell is fresh:
  `source /venv/main/bin/activate`.
- **Model:** the model loads in about 15 s (50 GB).
  - SEC replay parks the bf16 upper blocks on the CPU: use `SecEngine` as a context manager.
  - Never load the NLA before capturing subject activations: it merges in place, and the integrity guard will refuse.
- **Tests:** `python -m pytest scripts/common/tests -q` (`-m "not gpu"` for CPU only).
- **001 stages:** `python scripts/001_leverage_profile/leverage_profile.py <stage>`. Stages: gate, smoke, run,
  probes, snapshot, analysis, figures. Each is resumable and writes a manifest.
- **Push:** `env -u GITHUB_TOKEN git -c credential.helper='!gh auth git-credential' push origin <branch>` (a stale
  token shadows the gh login). Commit as senku14x / visheshgupta14x@gmail.com.
- **Rebuild on a fresh instance:** `bash scripts/000_setup/rebuild_env.sh`. `/workspace` is not a persistent volume.
- **Process rules (from the PI):**
  - design spec before every experiment, launched only on the PI's explicit go;
  - gates before outcomes;
  - every silly mistake recorded in the commit message, the report and the `CLAUDE.md` log.
