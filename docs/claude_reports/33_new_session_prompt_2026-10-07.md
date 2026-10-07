# 33. New-session prompt (2026-10-07, after 012)

Paste this as the first message of a new Claude Code session. It supersedes report 30 for current state; report 30's rules still hold where this file doesn't change them.

---

You're joining an ongoing mechanistic-interpretability research project as my research collaborator. I'm Vishesh, the PI (GitHub senku14x). You're on the same GPU instance the previous sessions used, and everything is installed and restored. **My usage limit is tight: do not use subagents unless I ask, and keep tool output lean.**

## 0. The project in one paragraph

**Causal-Concordance:** do activation readers agree on, and track, what Qwen3.6-27B demonstrably uses?
- **Readers:** J-lens (J_NP, J_CB), R-lens, logit lens, the Oracle Lens (`olens_s3d_rl600`, on-contract L20–60 step 4; WorkspaceBench's reproduction of the paper's A.9.2 oracle lens, *not* a Karvonen oracle), and the NLA (`ceselder/qwen3.6-27b-nla-rl`, av_base + `av_rl_adapters/iter_000400`, read at L42).
- **Labels of "used"** come from reader-free interventions on the model, never from readers.
- **Motto:** agreement is not truth, and decodable is not used.
- **Neel's questions frame 012:** what do readers agree on; what does one catch that others miss; does the NLA see non-workspace content; is majority agreement a usable ground truth; does ensembling help?

## 1. Read first, in this order (my rule)

1. **The workspace paper, in full:** Gurnee et al. 2026, `docs/papers/2607.15495.pdf`, text in `docs/papers/text/`. Render figure pages with `pypdfium2` from `/venv/main` if you need text inside figures.
2. **The paper notes:** `docs/claude_reports/04_*`, `04b_*`, `03_*` (§14–15).
3. **`CLAUDE.md`, all of it:** rules, non-negotiables, environment, "Where things stand", pitfalls, the mistakes log.
4. **012, the newest work** (branch **`012-reader-concordance`**, off `008-trackb-brewcode`):
   - `docs/ledger/012_ledger.md`: the chronological record of every run, oddity and correction. **Read it fully.**
   - `docs/design_specs/012_reader_concordance.md`: §12–§20 are the live contract; §20 lists the corrections after the audit.
   - `docs/reviews/012_codex_audits/` (README first): the independent audits and how each was handled.
   - Results page: https://claude.ai/artifact/NoevKvBkXNUTvtpnJPVsuq (built from JSON by `scripts/012_reader_concordance/make_page.py`).
5. `STATUS.md` (newest first); report 30 for the history of 001–011.

## 2. Where we are (verify against the ledger; don't trust this summary blindly)

**012 bank:** Jan Chen's phrase-lens ST multihop bank, i.e. 96 WSB multihop items plus a swap target and answer each. Read-only clone at `/workspace/sources/ext/phrase-lens-src` @ ea85583; Jan cleared its use. 82 clean-correct items are analysed.

**Results** (all exploratory):
- **A, the lever:** J-row coordinate swaps (Jan's `SwapEdit`, every prompt position, prefill only) change the answer on 48/82 at L24–47 and 38/82 at L24–35; random-target controls about 0; logit rows 0/82. The swap is a *reflection* of the src−tgt contrast, so toward_src edits fail.
- **B/C/D, readers on swapped states:** all readers track working swaps (AUC 0.9–0.96). But the model's own output at the read token predicts them just as well (0.97), and readouts mostly name the new *answer* ("Tokyo"). So this is output-facing, and there is no paired Oracle/NLA difference.
- **F1, position split:** editing everything but the read token flips 35/82; editing only it flips 0/82. The change is built earlier and carried forward.
- **F2/R, early positions:** readers detect the write, not use. **The "NLA early-position hint" was a position-mask bug and is withdrawn.**
- **G:** whitened swaps flip fewer (26 vs 38/82).
- **H:** brew every-position control still below the power bar; the cause of 008's null is not established.
- **Q:** readers share one signal; no demonstrated ensemble gain.
- **I/J, delayed question** ("Fact: Consider <clue>. <carrier> The <relation> of that country is"):
  - swapping before the question flips 42/60 under a relaxed gate, clue-only swapping 36/60; random 0;
  - hold states are bitwise identical across the three questions;
  - readers at the carrier end see nothing (about 0.47);
  - at the computed clue end readers give about 0.8, equal to the logit lens, but the model's own output there gives 0.69, so the pre-set bar fails;
  - only 60/168 prompts pass the gate, and Japan is over-represented.
- **E0, Oracle-AR loader:** fails its validation (retrieval 8/12 layers). That the AR was trained with a J-lens term is a **hypothesis only**: candidate jfve = J(y−μ) matches at L32–60.

**Net:**
- No evidence that any reader reads the hidden step beyond output-facing content, and no NLA-vs-Oracle difference.
- **These prompts are a good causal lever but structurally can't separate bridge from answer:** the bridge is guessable from the prompt, it fixes the answer one-to-one, and the answer comes immediately.

## 3. What to do next (my decisions so far)

1. **Camila's NLA hallucination test, now unblocked:**
   1. Download `ceselder/qwen3.6-27b-nla-rl` `ar_reconstructor/` (35.4 GB: a 43-block truncated backbone plus `value_head.safetensors`; "final", `saved_at_step` 600).
   2. **The model card says to pair any AV adapter with this final AR**, so iter_000400 + this AR is fine. The old "not step-paired" blocker was an unchecked assumption (it's in the mistakes log).
   3. Load it following EasyNLA (`/workspace/sources/repos/EasyNLA` @ 4d72847: `nla/utils/critic.py` `critic_predict`, `scripts/show_nla_generations.py` `load_ar_critic`, `nla/schema.py` normalisation).
   4. **Validate first** on the repo's `data/example_activations.parquet` (64 L42 activations with source text): verbalize, reconstruct, compare FVE with the published ~77% (`fve_trajectory.json`). **Stop and report after this smoke.**
   5. Then, on our saved 012 NLA readouts (`outputs/012_reader_concordance/stageC/readouts_nla.jsonl`, with captures in `stageB/captures.pt`): have an LLM minimally rewrite each explanation to remove claims the prompt doesn't support. Compare original vs cleaned on reconstruction FVE and on the reconstruction's share in 007's dense J subspace at L42 (`outputs/007_jsubspace/bases/J_raw_L42.pt`). Paired by item; pre-register the rule first.
2. **Systematic readout content analysis** (Camila's ask 3, and ask 2: is the NLA's outside-J content mostly hallucination?).
   - A fixed category list: bridge, answer, restated cue, format boilerplate, wrong entity, contradicted claim, situational commentary.
   - Label a seeded sample per method × condition (Oracle vs NLA; whole / J part / rest from 007; clean vs swapped from 012) with the judge, and validate the judge by hand on about 30.
   - Stage J clue-end readouts are saved but not tallied yet.
3. **The paid judge is approved:** OpenRouter, key in `/workspace/.env`. Gemini 3.8 Flash, or GLM / Kimi. Keep the spend small and log it.
4. **A better prompt family for Neel's questions** (recommended: hidden user attributes, which the model infers and uses but never states). Spec only; no runs until I vet it.

## 4. Pending with others (I handle all contact)

**Camila:**
- the Oracle-AR loader `load_lc_reconstructor`, or an input/output fixture;
- separately: the AR training loss, the definitions of validation FVE and J-FVE, and whether the Oracle's "joint bullet-FVE" reward includes J.

**Draft message to her:** in the chat history (corrected numbers). Do not mention the withdrawn NLA hint.

**Cathryn (partner):** runs the WSB J/non-J reader split. Don't duplicate it.

## 5. Environment

- **Hardware:** 1× RTX PRO 6000 Blackwell (96 GB), 128 CPUs.
- **Python:** `/venv/main/bin/python`; each Bash call is a fresh shell.
- **GPU memory:** the subject (50 GB) and the NLA av_base (54 GB) don't fit together. Capture first, then free.
- `/workspace` is NOT persistent. All 012 outputs (273 files) are on the public HF dataset `senku21x/causal-concordance-outputs` (`scripts/000_setup/mirror_outputs.py`).
- **Git:**
  - push with `env -u GITHUB_TOKEN git -c credential.helper='!gh auth git-credential' push origin <branch>`;
  - commit as `GIT_AUTHOR_NAME=senku14x GIT_AUTHOR_EMAIL=visheshgupta14x@gmail.com` (and the same `GIT_COMMITTER_*`);
  - never push to `main` or force-push.
- **`CLAUDE.md` rules hunk:** my relaxed-rules change is still uncommitted (mine to commit). To commit other `CLAUDE.md` edits, stage a copy without it (`git show HEAD:CLAUDE.md` → apply edits → `git hash-object -w` → `git update-index --cacheinfo`).

## 6. How I want you to work

- **Process:**
  - Exploratory runs up to about 2 GPU-h need no go from me; label them, and log every step in the 012 ledger.
  - Pre-register each stage's decision rule in the spec before running.
  - **Stop after every smoke and report.**
- **Numbers and claims:**
  - numbers come from full-precision JSON, rounded half-up, never from printouts (the mistakes log has repeated slips);
  - keep qualitative samples next to the numbers, and commit seeded readout samples;
  - "reader-specific" requires a **paired** contrast (spec §20), not one reader passing and another failing;
  - the intervention labels mean "an answer effect under this J-derived intervention", not natural mediation;
  - check claims against the source (code, model cards, JSON) before treating them as constraints.
- **Privacy:** private messages I paste (Slack/TA threads) stay in the chat. Never put them in the repo, commits, reports or memory.
- **Communication:** lead with conclusions in plain language, put caveats next to the claims they qualify, give estimates in GPU time, and be concise.

**First task:**
1. Read §1.
2. Check the instance: `git status` on `012-reader-concordance` (only `CLAUDE.md`'s rules hunk modified); GPU free; CPU tests pass (`python -m pytest scripts/common/tests tests -q -m "not gpu"`, 140 passed on 2026-10-07).
3. Report back briefly, then propose the NLA-reconstructor smoke from §3.1 with a GPU estimate. Wait for my go before any GPU run.
