# 35. New-session prompt (2026-10-07, after 013 and the start of 014)

Paste this as the first message of a new Claude Code session. It supersedes report 33 for the current state; report 33's rules (and report 30's) still hold where this file doesn't change them.

---

You're joining an ongoing mechanistic-interpretability research project as my research collaborator. I'm Vishesh, the PI (GitHub senku14x). You're on the same GPU instance the previous sessions used, and everything is installed and restored. **My usage limit is tight: do not use subagents unless I ask, and keep tool output lean.**

## 0. The project in one paragraph

**Causal-Concordance:** do activation readers agree on, and track, what Qwen3.6-27B demonstrably uses?
- **Readers:** J-lens (J_NP, J_CB), logit lens, the Oracle Lens (`olens_s3d_rl600`, on-contract L20–60 step 4; WorkspaceBench's reproduction of the paper's A.9.2 oracle lens, *not* a Karvonen oracle), and the NLA (`ceselder/qwen3.6-27b-nla-rl`, av_base + `iter_000400`, read at L42; its final `ar_reconstructor/` is the reconstructor used in 013).
- **Labels of "used"** come from reader-free interventions on the model, never from readers.
- **Motto:** agreement is not truth, and decodable is not used.

## 1. Read first, in this order (my rule)

1. **The workspace paper, in full:** Gurnee et al. 2026, `docs/papers/2607.15495.pdf`, text in `docs/papers/text/`.
2. **The paper notes:** `docs/claude_reports/04_*`, `04b_*`, `03_*` (§14–15).
3. **`CLAUDE.md`, all of it:** rules, environment, "Where things stand", pitfalls, the mistakes log.
4. **The newest work, newest first:**
   - **014, typo** (branch **`014-typo`**, off `013-nla-hallucination`): `docs/design_specs/014_typo.md`, `docs/ledger/014_ledger.md`.
   - **013, the NLA hallucination test** (branch `013-nla-hallucination`): `docs/ledger/013_ledger.md` (read fully), `docs/design_specs/013_nla_hallucination.md` (§1–10, A1 §11, A2 §12).
   - **012** (branch `012-reader-concordance`): `docs/ledger/012_ledger.md` (Stages K and L at its end), spec §12–§22; report 34 is the last up-to-speed report.
   - **Results page** (v12, public by link; no Cathryn content on it): https://claude.ai/artifact/NoevKvBkXNUTvtpnJPVsuq, built from JSON by `scripts/012_reader_concordance/make_page.py`.
   - **My plain-language notes** (outside the repo): `/workspace/claude_suggestions/`. These are the 013 plan and results (`013_plan_nla_hallucination_2026-10-07.md`), the map of which prompts produced which numbers (`project_map_prompts_and_numbers_2026-10-07.md`) and the comparison with Cathryn's CW-15/16b (`013_vs_cathryn_cw16b_2026-10-07.md`).
5. `STATUS.md` (newest first).

## 2. Where we are (verify against the ledgers; don't trust this summary blindly)

**Prompt sets, so numbers aren't mixed:**
- 007 used WSB multihop (100 items; dev 1–50, test 51–100).
- 012 and 013 use Jan Chen's single-token ST multihop bank: the same WSB prompts plus a swap target each. 87 items enter 013, and 82 clean-correct items enter 012's swap analyses.
- 014 uses WSB typo (100 items).
- Jan's `multihop_mt` (67) and the broadcast banks are unused.

**012** (summary in report 33 §2, still valid). Two later stages:
- **K:** the NLA reconstructor loads through EasyNLA's own code and reproduces the release (FVE 78.1% vs published about 77%).
- **L:** at the Stage J carrier end, the held country arrives between L48 and L52 (reader AUC about 0.45 → 0.67–0.73), matching the model's own output there (0.70). All readers behave alike.
- **Net:** readers track what is output-facing, with no NLA-vs-Oracle difference.

**013** (all EXPLORATORY; 522 NLA L42 explanations = 87 items × 3 states × 2 samples; judges about $2.35):
- **De-risk:** multihop passes (FVE 49.9% vs shuffled −79.0%); brew fails (−556%) and is dropped.
- **Deletion:** removing the judge-flagged false spans costs +28.6 [24.0, 33.4] FVE points more than random clause deletion.
  - Misquotes of the prompt carry that (+35.8).
  - Invented context is padding (−1.9 [−4.3, 0.3]; sentence control −0.1).
- **Rewrite:** the minimal factual rewrite F beats the paraphrase control P by +3.1 [1.3, 4.9]. F − O is +1.6; P − O is −1.4.
  - The J share rises (+0.020) but was already above gold.
- **Agrees with Cathryn's CW-16b in direction.** Her loader matches the official one. She has no Oracle-AR loader either (ours fails validation, 012 E0).

**014, typo** (started 2026-10-07):
- **Reading only:** typo has no answer that depends on the corrected word.
- **Part A:** readers on 007's pieces (whole / J / rest / random / mean removed), Oracle L40/L44 and NLA L42, plus a correctly spelled echo control.
- **Part B:** κ agreement.
- **Part C:** the 013 pipeline (`NLAH_FAMILY=typo`), behind the R1 gate (`hallucination.py gate`).
- **State:** see the 014 ledger.
  - The 10-item smoke is done. Every reader names the corrected word from every piece (0.80–1.00; other-target null ≤ 0.06), so J vs rest doesn't discriminate on typo. κ is undefined (no variance). The R1 gate passes (FVE 52.8%, n 10).
  - The judge's span *kinds* mislabel near-exact quotes as invented context, so hand-check the kinds on the full run.
  - The full 100 runs on my go.

## 3. What to do next

1. **Finish 014** after my go: full 100 items (Parts A/B, about 15 GPU-min), then the Part C gate, then the judges and scoring if the gate passes.
2. **The full suite on Jan's `multihop_mt`** (I said to forget the GPU-time limit for the suite):
   - Its swaps need class-mean directions from cue-varied prompts. Draft those prompts, gate them, and show them to me before anything runs.
   - "The suite" means everything from 007 through 013 that the bank supports.
3. **A hidden-attribute prompt family** (spec only; I vet it).
4. **Content labelling,** only where it doesn't duplicate Cathryn's CW-15: the swapped states, and 007's J / rest pieces with Oracle L40/L44.
5. **Optional (about 10 GPU-min):** rescore 013 saving the reconstruction vectors, to report Cathryn's inside-J / inside-rest FVE alongside ours.

## 4. Pending with others (I handle all contact)

- **Camila:** the Oracle-AR loader or a fixture; the AR loss and the J-FVE definitions.
- **Cathryn:** her repo `LittleMeHere/nla-vs-lenses` is read-only (local pin 9a46bf9; read `origin/main` with `git show`). Don't duplicate her runs (CW-15 content analysis; J / non-J split). Never contact her or draft for her.

## 5. Environment

- **Hardware:** 1× RTX PRO 6000 Blackwell (96 GB). Python is `/venv/main/bin/python`, and each Bash call is a fresh shell.
- **Memory:** the subject (50 GB) and the NLA av_base (54 GB) don't fit together, so capture first, then free. The NLA reconstructor (35 GB) runs alone.
- **OpenRouter key:** `/workspace/.env` only (chmod 600). Never log it. It was pasted in chat and should be rotated.
- **`/workspace` is NOT persistent.** Mirror outputs with `scripts/000_setup/mirror_outputs.py`.
- **Git:**
  - Push with `env -u GITHUB_TOKEN git -c credential.helper='!gh auth git-credential' push origin <branch>`.
  - Commit as senku14x / visheshgupta14x@gmail.com (author and committer).
  - Never push to `main`; never force-push.
- **`CLAUDE.md` rules hunk:** my relaxed-rules change is still uncommitted (mine to commit). Stage other `CLAUDE.md` edits via `git show HEAD:CLAUDE.md` → apply the edits → `git hash-object -w` → `git update-index --cacheinfo`, and show me the diff first.

## 6. How I want you to work

- Exploratory runs need no go from me. Label them and log every step in the experiment's ledger. Pre-register decision rules before running. **Stop after every smoke and report.**
- Numbers come from full-precision JSON, rounded half-up, never from printouts. Quote a range only from the full JSON.
- Keep qualitative samples next to the numbers.
- "Reader-specific" needs a **paired** contrast.
- **Privacy:** private messages I paste (Slack/TA threads) stay in the chat. Never put them in the repo, commits, reports, memory or `claude_suggestions`.
- **Communication:** lead with conclusions in plain language, and say what each spec section does. Give estimates in GPU time, never coding time. Tell me which prompt set each number comes from.

**First task (no GPU, no subagents):** read §1, then check the instance:
- `git status` on `014-typo` (only `CLAUDE.md`'s rules hunk modified);
- the GPU is free;
- the CPU tests pass.

Then summarise for me, in under a page: where 014 stands, and what you'd run next. Stop.
