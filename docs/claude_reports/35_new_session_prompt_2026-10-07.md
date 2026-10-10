# 35. New-session prompt (2026-10-07, after 013 and 014)

Paste this as the first message of a new Claude Code session. It supersedes reports 30 and 33 as the entry point; their rules still hold where this file doesn't change them.

---

You're joining an ongoing mechanistic-interpretability research project as my research collaborator. I'm Vishesh, the PI (GitHub senku14x). You're on the same GPU instance the previous sessions used, and everything is installed and restored. **My usage limit is tight: do not use subagents unless I ask, and keep tool output lean.**

Your first job is to come fully up to speed with the whole project: the paper it builds on, my notes, and everything the project has done and currently believes. Then write a report I can use to check that you are. No experiments until I've read it.

## 0. The project in one paragraph

**Causal-Concordance:** do activation readers agree on, and track, what Qwen3.6-27B demonstrably uses?
- **Readers:**
  - J-lens (J_NP, J_CB), R-lens, logit lens;
  - the Oracle Lens (`olens_s3d_rl600`, on-contract L20–60 step 4; WorkspaceBench's reproduction of the paper's A.9.2 oracle lens, *not* a Karvonen oracle);
  - the NLA (`ceselder/qwen3.6-27b-nla-rl`, av_base + `iter_000400`, read at L42; its final `ar_reconstructor/` is the reconstructor).
- **Labels of "used"** come from reader-free interventions on the model, never from readers.
- **Motto:** agreement is not truth, and decodable is not used.

## 1. Read, in this order (my rule)

1. **The paper, in full:** Gurnee et al. 2026, *Verbalizable Representations Form a Global Workspace in Language Models*.
   - `docs/papers/2607.15495.pdf`; text in `docs/papers/text/`.
   - Read the main text and the appendices the project leans on (A.8, A.9 / A.9.2, A.12, A.21, A.24.3, §3.1 with Figs. 7–8, §3.3).
   - Text inside figures is only in the PDF: render pages with `pypdfium2` from `/venv/main`.
2. **My notes, after the paper:** `docs/claude_reports/04_user_workspace_paper_notes_verbatim.md`, `04b_*` (combined), and `03_*` (merged; §14 corrects my notes against their sources, §15 lists where the agent's reading differs from
   mine; corrected 2026-10-10).
3. **`CLAUDE.md`, all of it:** rules, non-negotiables, conventions, environment, "Where things stand", pitfalls, the mistakes log.
4. **What the project has done and what it concluded,** in time order:
   - **001–004:** `STATUS.md` and the reports in `docs/reports/` (dev split only).
   - **005, planted concepts:** `docs/reports/005_followups.md` first (it corrects the earlier 005 reports), then `005_full_run.md`.
   - **006/006b, reader likelihoods:** `docs/reports/006_reader_likelihood.md`.
   - **007, the J-subspace split:** `docs/reports/007_jsubspace.md`.
   - **008–011:**
     - `docs/reports/008_trackb.md` (the paper's §3.3 privilege test on brew);
     - `docs/reports/009_brewcode_derisk.md`;
     - report 29 (010, reader agreement);
     - `docs/design_specs/011_olens_swaps.md`.
   - **Report 31:** the onboarding review that fixed several of these.
   - **012, readers on swapped states:** `docs/ledger/012_ledger.md` (read it fully; Stages K and L are at its end), spec `docs/design_specs/012_reader_concordance.md` §12–§22, `docs/reviews/012_codex_audits/`, report 34.
   - **013, the NLA hallucination test:** `docs/ledger/013_ledger.md`, spec `docs/design_specs/013_nla_hallucination.md` (§1–10, A1 §11, A2 §12).
   - **014, typo:** `docs/ledger/014_ledger.md`, spec `docs/design_specs/014_typo.md`.
   - **Results page** (v13, public by link; covers 012, 013 and 014): https://claude.ai/artifact/NoevKvBkXNUTvtpnJPVsuq. It is built from committed JSON by `scripts/012_reader_concordance/make_page.py`.
   - **My plain-language notes** (outside the repo): `/workspace/claude_suggestions/`. Of these, `project_map_prompts_and_numbers_2026-10-07.md` maps which prompt set produced which number.
5. `STATUS.md` (newest first) and report 30 for the history of 001–011.

**Verify, don't trust.** Every summary (this one, `CLAUDE.md`, `STATUS.md`) can be stale. The full-precision JSON under `results/` and the ledgers are the record.

## 2. Environment

- **Hardware:** 1× RTX PRO 6000 Blackwell (96 GB). Python is `/venv/main/bin/python`; each Bash call is a fresh shell.
- **Memory:** the subject (50 GB) and the NLA av_base (54 GB) don't fit together, so capture first, then free.
- **`/workspace` is NOT persistent.** Every output through 014 is mirrored to the public HF dataset `senku21x/causal-concordance-outputs` (`scripts/000_setup/mirror_outputs.py`; restore with `restore_outputs.py`).
- **Page checks:** `/venv/page` (playwright + headless Chromium) for the 1280 / 400 px render check before publishing.
- **OpenRouter key:** `/workspace/.env` only. Never log it.
- **Git:**
  - Push with `env -u GITHUB_TOKEN git -c credential.helper='!gh auth git-credential' push origin <branch>`.
  - Commit as senku14x / visheshgupta14x@gmail.com (author and committer).
  - Never push to `main`; never force-push.
  - The newest branch is `014-typo` (on top of 013, which is on top of 012).
- **`CLAUDE.md` rules hunk:** my relaxed-rules change is uncommitted (mine to commit). Stage other `CLAUDE.md` edits via `git show HEAD:CLAUDE.md` → apply the edits → `git hash-object -w` → `git update-index --cacheinfo`, and show me the diff first.

## 3. How I want you to work

- Exploratory runs need no go from me. Label them, log every step in the experiment's ledger, and pre-register decision rules before running. **Stop after every smoke and report.**
- Numbers come from full-precision JSON, rounded half-up, never from printouts. Quote a range only from the full JSON.
- Always say which prompt set a number comes from. Keep qualitative samples next to the numbers.
- "Reader-specific" needs a **paired** contrast.
- **Privacy:** private messages I paste (Slack/TA threads) stay in the chat. Never put them in the repo, commits, reports, memory or `claude_suggestions`.
- **Cathryn Wayman:** I handle all contact. Her repo is read-only. Don't duplicate her runs, and keep her content off the public page.
- **Communication:** lead with conclusions in plain language, say what each section of a spec does, and give estimates in GPU time, never coding time.

## 4. First task (no GPU, no subagents)

1. Read §1 in order.
2. **Check the instance:**
   - `git status` on `014-typo`: only `CLAUDE.md` modified;
   - the GPU is free;
   - the CPU tests pass (`python -m pytest scripts/common/tests tests -q -m "not gpu"`; 140 passed on 2026-10-07).
3. **Write `docs/claude_reports/36_up_to_speed_<date>.md`.** Commit and push it, and add it to the reports README. It must contain:
   - **(a) The paper in your own words** (about one page): its main claims, its methods (J-lens, J-space, the privilege test, verbalizers), and what it does *not* show.
   - **(b) My notes:** the points I stress, where my notes correct or disagree with the paper (04 §2.3 and §9, 04b),
     and where the merged notes correct my notes (03 §14) or differ from me (03 §15) (corrected 2026-10-10: said
     03 §14–15 correct or disagree with the paper).
   - **(c) The project, one row per experiment (001–014):**
     - the question;
     - the prompt set;
     - the key numbers, from the JSON (cite the key);
     - the verdict;
     - anything later corrected or withdrawn.
   - **(d) What the project currently believes.** List the claims it stands behind, with their scope (model, family, position, layer, n), and how confident each is. Then list what it explicitly does *not* claim.
   - **(e) Inconsistencies:** anything stale, contradictory or wrong across the ledgers, specs, `CLAUDE.md`, `STATUS.md` and the page, with file:line.
   - **(f) Open threads:** what the docs say is unfinished or pending.
   - **(g) Your questions for me.**
4. **Paste the full report in your reply, then stop.** I'll check it before we decide what to work on next.
