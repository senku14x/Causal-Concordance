You're joining an ongoing mechanistic-interpretability research project as my research collaborator. I'm Vishesh, the
PI (GitHub senku14x). You're on the same GPU instance the previous Claude sessions used: everything is installed,
downloaded and restored, and the previous session is idle (it holds no GPU memory). Your first job is to come fully up
to speed. Read in the order below, then report back.

## 0. What the project is

**Causal-Concordance.** Do activation readers agree on, and track, what Qwen3.6-27B demonstrably uses? A reader turns a
model's internal activations into words or token scores. Our labels of what the model uses come from reader-free causal
experiments and never use any reader.
- **Readers studied:**
  - the token lenses: the J-lens (J_NP from Neuronpedia, J_CB from camilablank/workspace-lenses), the R-lens (R_CB, an
    LRP variant of the J-lens, same repo) and the logit lens;
  - the **Oracle Lens** (`olens_s3d_rl600`). It is a PEFT adapter attached to the subject model itself; its on-contract
    read layers are 20–60 in steps of 4. It is **WorkspaceBench's reproduction of the paper's A.9.2 oracle lens, not a
    Karvonen activation oracle**. Its training has three parts: an activation reconstructor (AR: text → the activation
    preceding the text), an inverter, and RL on FVE with the AR frozen. The AR (`ar_ptag_pooled`) ships alongside it;
  - the **NLA** (`nla-rl`: a separate 54 GB verbalizer model, av_base + RL adapter iter_000400). It reads layer-42
    activations; it is WorkspaceBench's and Cathryn Wayman's NLA;
  - the paper's **template lens** (since 007). It failed its own sanity check at L40–44.
- **Benchmark: WorkspaceBench (WSB).** Families:
  - **brew**, the main testbed: a colour-mixing table. The model must compute a hidden intermediate colour z on the way
    to the answer without saying it. A pair is the same table with a different start colour;
  - **multihop**: two-hop facts with an unspoken bridge entity;
  - **relational** (a bare cloze) and **chain** (arithmetic).
- **Method:** compare what readers report about a cell (one token position at one layer) with reader-free causal
  evidence on token-aligned counterfactual prompt pairs. The evidence comes from whole-cell activation patching,
  subspace and direction swaps, and ablations.
- **Motto:** agreement is not truth, and decodable is not used.

## 1. Read first (my rule): the workspace paper, then the paper notes

**In this order, before anything else.** No code, no runs and no opinions on results until steps 1 and 2 are done.

**Step 1: the paper, in full.** Gurnee et al. 2026, *Verbalizable Representations Form a Global Workspace in Language
Models* (Anthropic). It is from July 2026, newer than your training data.
- **Files:**
  - PDF: `/workspace/causal-concordance/docs/papers/2607.15495.pdf`;
  - text from the web version: `docs/papers/text/2607.15495_html.md` (clean headings and captions; cross-references
    show as "??");
  - text from the PDF: `docs/papers/text/2607.15495.txt`.
- **Figures.** The text extractions miss text inside figures, and the Read tool can't render PDF pages on this instance
  (no poppler). Render the pages you need to PNG with `pypdfium2` from `/venv/main` into your scratchpad, then Read the
  PNG.
- **Sections that matter most:** the main text, plus the appendices the project leans on:
  - A.8 (J-space span) and A.9 / A.9.2 (verbalizers and J-space; A.9.2 is the oracle lens);
  - A.12, A.21 and A.24.3;
  - the appendix on multi-token concepts (template lens, Oracle lens);
  - §3.1 with Figs. 7–8 (verbal report: planted concepts and the privilege test);
  - §3.3 (internal reasoning: swapping an intermediate's J part against the rest). Experiment 008 reproduces this
    protocol on brew.

**Step 2: the paper notes, after the paper, not instead of it.** All three:
- `docs/claude_reports/04_user_workspace_paper_notes_verbatim.md`: my own notes;
- `docs/claude_reports/04b_user_workspace_paper_notes_combined_2026-10-05.md`: my combined notes (paper plus follow-up
  literature);
- `docs/claude_reports/03_workspace_paper_notes_merged.md`: an earlier agent's notes merged with mine. Corrections are
  in §14 and disagreements in §15.

The digest `docs/papers/text/2607.15495_NOTES_digest.md` is a map, not a substitute.

**Key concepts:**
- J_ℓ = E[∂h_final/∂h_ℓ], one d×d matrix per layer;
- the readout softmax(W_U · norm(J_ℓ h_ℓ));
- J-lens vectors, which are the rows of W_U J_ℓ;
- "J-space", the workspace and its layer band;
- "privileged" representations.

**The project has three different "J" objects. Never mix their numbers:**
1. the J-lens itself: J_ℓ and its per-token vectors (lenses J_NP, J_CB);
2. a sparse **J part** of a vector, found by nonnegative pursuit over J-lens vectors plus an NNLS refit. This is the
   paper's privilege-test construction: 005 used k = 16; 007 Stage 2b and 008 use k = 25;
3. 007's dense **J subspace**: the top-1,024 eigenvectors of Σ_t unit(W_U[t] J_ℓ), using raw W_U and the matching
   layer. It reproduces Cathryn's basis exactly at L42.

**Step 3: the related reading** listed at the top of `CLAUDE.md` (the full list is `docs/context/READING_LIST.md`; all
PDFs and text extractions are in `docs/papers/`):
- Activation Oracles (2512.15674): background only, see above;
- AObench (2606.02609);
- Li et al. on prompt-only baselines (2509.13316);
- Dingeto on NLA claim sensitivity (2607.20379);
- the NLA, WorkspaceBench and R-lens pages in `docs/sources/web/`. The WSB post's appendix describes the Oracle Lens
  and its AR (`docs/sources/web/wsb_post.md:341-349`).

## 2. Then the project docs

The repo is `/workspace/causal-concordance`. **Work is on branch `008-trackb-brewcode`** (pushed; head f86c512 or
later), which sits on top of `origin/main` = 007.
- The local `main` ref on this instance is stale (95ab0e5). Use `origin/main` if you need main.
- `CLAUDE.md` may show as modified, with an uncommitted "relaxed process rules" hunk. That hunk is mine to commit (see §7).
  Leave it as it is.

Read:
1. **`CLAUDE.md`, all of it:** read order, non-negotiables, conventions, environment, "Where things stand", "Working
   with the PI", pitfalls, and the silly-mistakes log, which is full of lessons. Some of its process rules are relaxed
   for you; see §7 below.
2. **Follow CLAUDE.md's read order:**
   - `STATUS.md` (newest entries first);
   - `docs/context/SPEC_v2.md` with its errata and `AMENDMENT_01.md` (together, the unified project spec);
   - `docs/COME_UP_TO_SPEED.md` (its top box is current);
   - the earlier reports in `docs/reports/` as needed: the 005 follow-ups and full run, 006, and 001–004.
3. **007:**
   - `docs/reports/007_jsubspace.md` (§8 covers the post hoc NLA rerun);
   - `docs/design_specs/007_jsubspace.md` (pre-registered v2; execution notes in §20; amendments A1–A3 in §21–23);
   - `docs/context/AMENDMENT_02.md`;
   - `results/007_jsubspace/status/*.md`.
4. **The newest work (2026-10-06, all exploratory):**
   - **008, Track B:** `docs/reports/008_trackb.md`. Design: `docs/design_specs/008_009_trackb_brewcode.md`, Part 1,
     with execution notes in §E;
   - **009, the brew-code bank:** `docs/reports/009_brewcode_derisk.md`. Design: Part 2 of the same note, with execution
     notes in §E-009; reader rules in `docs/design_specs/009b_reader_phase.md`;
   - **010, reader agreement (CPU):** `docs/claude_reports/29_reader_agreement_010_2026-10-06.md`;
   - **011, Oracle-space swaps:** `docs/design_specs/011_olens_swaps.md`, with amendments 1–3 and execution status at
     the end. Code in `scripts/011_olens_swaps/`, results in `results/011_olens_swaps/`.
5. **`docs/claude_reports/README.md`**, the index of agent reports. The most useful:
   - 25: why the NLA runs kept stopping (bf16 batch-layout noise), and the open issues;
   - 26: the up-to-speed report of 2026-10-06, with every inconsistency found;
   - 27: what the readers actually say (qualitative);
   - 28: the proposal behind 008/009 (partly superseded by my changes, recorded in the design note);
   - 29: 010.

## 3. Where we are (verify against the docs; don't trust this summary blindly)

**001–004 (dev split only):**
- Brew's `:` cell (after "Answer") carries about half the answer gap at L42–44 (whole-cell patch share 0.48–0.50).
  `</think>` carries almost nothing (0.01). z is probe-decodable at `:`.
- The verbalizers rarely state z in its role (002).
- Reader-free 1-D labels are only descriptive (003; its G4 gate failed).
- Fitted spans do no better than PCA (004).

**005 (planted concepts, 45 held-out test concepts):**
- The model-side "report" measure turned out to be lexical priming.
- No J-space privilege at matched norm.
- The Oracle names planted concepts more often than the NLA, and the NLA failed the likelihood positive control.
- The token lenses detect nothing at L40–44.

**006 and 006b (dev, 48 tables):**
- Both readers' colour likelihoods track z at brew `:`.
- The model's own final-layer distribution tracks z too (z is the runner-up answer in 56% of tables). So the readers'
  tracking is consistent with reading output-facing content.
- The Oracle's 2× stronger tracking at `</think>` is not mirrored by the model.

**007 (J-subspace split, pre-registered):**
- **Reader side, confirmed on 49 held-out brew test tables.** The Oracle's z tracking at `:` L40 is carried by the J
  subspace:
  - the registered contrasts confirmed on dev and test are P1, P2, P3 and P5 at L40, and only P5 at L44;
  - descriptively, J alone gives 5.76 nats vs 3.22 for the whole activation on dev (5.03 vs 2.87 on test), and removing
    J leaves 0.30 / 0.29 (not a registered contrast; corrected 2026-10-06, report 31).
- The same pattern holds at `</think>` (dev only, secondary), where 001's patch share is 0.01 at L42 (0.03 at L44): the
  reader's J dependence is as strong where observed patch leverage is low (corrected 2026-10-06, report 31: was "does not follow use").
- **NLA (post hoc):** not J-concentrated at `:`. ~~J dependence is reader-specific.~~ At `</think>` the NLA also
  collapses without J (rest of J 0.21 vs rest of random 1.21), so "reader-specific" holds at `:` only (corrected 2026-10-06, report 31).
- **Multihop:** J ≈ whole for both readers, confounded by restatement. Among bridge-naming samples with checkable cue
  words, the restatement share is 0.28 (Oracle L40), 0.24 (Oracle L44) and 0.15 (NLA) for the J piece (15/53, 13/54,
  8/53), and 0.56 / 0.25 / 0.58 for the whole activation (27/48, 12/48, 28/48). Not "about half": that figure mixed
  denominators.
- 007's own model-side test (Stage 2b) had a weak lever. 008 replaces it.

**008, Track B: the paper's §3.3 privilege test on brew, done the paper's way** (82 gated dev pairs; all gates pass):
- Class-mean z swaps, clamped per layer over the three emit cells (assistant newline, `</think>`, `:`) × L40–48.
- The lever is adequate: the full direction moves the answer by median S 0.21 [0.11, 0.26], and flips 16% of answers.
- **No model-side J privilege:**
  - the J part (k = 25) carries S 0.02–0.03 and the non-J part 0.21;
  - J − non-J is −0.17 [−0.24, −0.09] for J_NP and −0.14 [−0.21, −0.08] for J_CB (corrected 2026-10-06, report 31: −0.14465 was
    written −0.15; the −0.15 also went into my numbers to the TA).
- **The real whitened swap** (Ledoit–Wolf Σ from about 50k chat tokens; swap in whitened coordinates; map back) narrows
  the gap but never reverses it: J 0.07 / 0.10 vs non-J 0.14 / 0.11.
- Clamping the J coordinates barely changes the non-J swap (0.21 → 0.20 / 0.18). On Claude, the paper's clamp took the
  non-J effect from 28% to 6%.
- The lens-coordinate swap (arm A) is not read: its answer-atom positive control moves S 0.01, below the power bar of
  0.15.
- ~~→ **The reader–model dissociation stands on a powered lever:** the Oracle privileges J for reading z; the model's
  answer does not.~~ (corrected 2026-10-06, report 31: 008's J part is the sparse k = 25 part of a class-mean direction, not 007's dense
  1,024-dim J subspace, and the paper is "not reproduced under a band-limited clamp (3 cells × L40–48; the paper clamps
  at every position)", not reversed.) On one object, the dense J subspace: the Oracle L40 reads z from J at 5.76 vs
  3.22 nats for the whole activation (dev; 5.03 vs 2.87 on test), while 007 Stage 2's dense-J swaps track the Δ-share at
  L42 and exceed Δ-matched random only modestly at L40/L44 (+0.07 / +0.09 recovery, below the 0.10 margin; S_full 0.28
  at L40). 008's sparse-J-part result is separate, consistent evidence. My 2026-10-06 numbers to the TA come from 008.

**009, the brew-code bank** (reader-free de-risk):
- **The idea:** the answer is a two-digit code looked up from the final (or start) colour, so z is needed but never
  said.
- **Gate:**
  - depth 2 is beyond Qwen without chain of thought (0/77 pairs after 30 tables);
  - depth 1 passes 771 of 1,200 pairs (241 of 300 tables).
- **The output-facing check fails, so readers do not run (rule §2.5.1).**
  - With a code answer, z is still the top colour at the answer slot in 98% of renders: median rank 11 of 248,320
    tokens, model D +11.9 nats, 240/240 tables.
  - Yet total colour mass is only about 0.05%. D is scale-free, so the tail still counts.
- Reader-free probes decode z under both questions at 33 of the 35 post-divergence sites (corrected 2026-10-06, report 31: was "110 of
  112 sites"; 77 sites precede the differing token and are identical under both questions; z already decodes at
  q1–q2).
- The patch profile reproduces brew's structure: `</think>` at L40–44 has z decodable but no single-cell leverage.
- **My view:** the new prompts are weak, and we should probably go back to the old prompts and runs. 009 readers are on
  hold.

**010, reader agreement** (CPU):
- NLA–Oracle per-pair agreement is weak (r 0.24–0.30).
- The Oracle tracks what the J lens reads (r 0.63–0.80) more than the model's next-token output (0.21–0.30); the NLA
  tracks neither strongly (J lens 0.33–0.37, model 0.07) (corrected 2026-10-06, report 31: was "Mid-layer readers ... (r 0.63–0.80), not
  the model's next-token output (0.2–0.3)").
- Only the late Oracle (L56/L60) follows the output.

**011, Oracle-space swaps** (my TA's tip: put a colour word through the Oracle Lens's reconstructor, get a vector, do it
for two colours, swap them in whitened space). **Everything is post hoc and exploratory.**
- **AR loader.** The official loader is not public, so we wrote one. It failed the pre-set validation twice, then
  passed under post hoc amendment 2:
  - the AR's own whitener is `whitening_iolens_pooled` (v1 used the wrong one);
  - the head takes a parameter-free LayerNorm of the backbone's final pre-norm state and **outputs a raw activation**,
    scored in the whitened metric;
  - the target is the activation preceding the span.
  - On 16-token spans, FVE after one global scale is 0.24 (published 0.22), and within-batch retrieval@1 is 0.86
    (published 0.93).
  - **Unexplained:** a global output scale of 0.3–0.55. Only colour vectors centred over the 12 colours are used,
    because the scale cancels there.
- **Smoke s0b** (2 pairs, one table; observations only; all gates pass), median S:

  | Condition | S |
  |---|---|
  | class-mean reference | 0.23 |
  | **the TA's swap as given (AR vectors swapped in whitened coordinates)** | −0.00 |
  | its answer-colour positive control | −0.00 |
  | raw-coordinate swap of the same directions | 0.16 |
  | its answer control / a size-matched random edit | 0.08 / 0.09 |
  | **additive: minus AR(z), plus AR(z′), at the reference's edit size** | 0.10 |
  | the AR's answer-colour difference (same edit) | 0.12 |
  | the model's own class-mean difference (same edit) | 0.23 |
  | random (same edit) | 0.01 / −0.02 |

  - The TA's swap is therefore not read: its own positive control fails.
  - The raw-coordinate result looks generic: one random edit of the same size moves the answer almost as much.
- **The full run s1 was stopped at my request at 74 of 82 pairs.** It is resumable and has not been analysed.
- **Likely limit of the tip:** the AR reconstructs 1-token inputs poorly (cosine ≈ 0.1 with the real activation, vs
  ≈ 0.5 for 16 tokens). So single-word colour vectors are mostly noise.

**Held-out data already spent:**
- the brew test split: read by the Oracle (pre-registered, AMENDMENT_02), the template lens, the prompt-only floor, and
  the NLA (post hoc A2, 10-06) (corrected 2026-10-06, report 31: it is not "used once");
- multihop's test split, items 51–100 (007 Stage 3; Oracle, and the NLA post hoc);
- multihop dev, items 1–50, touched earlier.
- 009's 150 test tables per depth are generated and hashed but untouched.

Check this before planning anything confirmatory.

## 4. Machinery in brief (details are in the specs)

- **Numerics.** Label quantities use the SEC regime: bf16 blocks 0–39, a true-float32 replay of blocks 40–63, and a
  floor of about 2e-4 nats. All edits are at L ≥ 40.
  - Readers run fully in bf16. They are deterministic for a fixed batch layout but move when the layout changes: up to
    about 0.13 nats for the Oracle and 0.3–0.4 for the NLA.
  - Calibrate gates per reader.
- **Whole-cell patch share:** S = Δm/G.
  - m = LSE log P(a′) − LSE log P(a), scored over the full sequence and every spelling token.
  - G is the clean member–partner gap.
- **D statistic (006):** D = [LL_m(t_m) − LL_m(t_p)] − [LL_p(t_m) − LL_p(t_p)], averaged over partners per table.
  Uncertainty comes from a table-cluster bootstrap (10k draws, seed 20261004) and a null from 200 derangements. D is
  scale-free.
- **The 008 clamp primitive** (`scripts/008_trackb/trackb.py`, reused by 011): c = R(h − μ); h ← h + A(t − c), with
  R A = I.
  - The target t is computed from the clean run's coordinates and *set* at each layer (never a fixed delta added per
    layer).
  - Whitened versions use W = Σ^{-1/2}.
  - Power gate: the answer-colour control must reach median S ≥ 0.15 before the z arm is read.
- **007's pieces and contrasts:** see §4 of the 007 design and `scripts/007_jsubspace/pieces.py`. P1–P5; Holm over 15
  contrasts; δ = 0.5 nats; TOST for "≈"; the categories +, −, ≈, unresolved and uninformative.
- **009's collision rule:** at depth 1, z = f. For every role, skip pairs where one member's hidden colour is the other
  member's start colour.
- **Reader layers:** Oracle at L40/L44 (never L42, which is off-contract); NLA at L42 only.

## 5. Open threads and likely next steps (decisions pending with me)

1. **011, my call:**
   - finish the stopped full run (`olens_swaps.py --vec-run v3 --run-id s1` skips finished pairs; about 1 GPU-min), then
     `olens_swaps_analysis.py --run-id s1`, which applies amendment 3's reading rules;
   - build colour vectors from several short phrases that start with the colour, averaged, because 1-token inputs
     reconstruct poorly (about 10 GPU-min). Write it as a post hoc amendment first.
2. **009 readers, my call:**
   - keep the rule (bank used model-side only);
   - run the readers post hoc with the model's output as a covariate;
   - or redesign.

   I lean towards going back to the old prompts and runs.
3. **NLA hallucination stripping**, and the NLA reconstructor's FVE.
   - nla-rl ships an `ar_reconstructor/` saved at RL step 600. It is not step-paired with our AV (iter_000400), and
     `readers.py` has no wrapper for it.
   - Stripping needs an LLM judge, which costs money: ask me first.
4. **Cheap CPU work:**
   - a systematic content analysis of NLA readouts outside J (saved readouts in
     `outputs/007_jsubspace/stage1a/{r2,r2_nla_posthoc}/qualitative_*.jsonl` and
     `outputs/007_jsubspace/stage3/{m1,m1_nla_posthoc}/restatement_*.jsonl`);
   - reader-side gating checks on 007's multihop and brew.
5. **The structural confound.** On brew, z is a plausible answer; 009 showed that a nameable intermediate stays in the
   output tail even when the answer is a code. Any new family must pass an output-facing check on rank and D, not just
   on probability mass.
6. **Open puzzles:**
   - the AR's global output scale;
   - the Oracle's 2× tracking at `</think>`.
7. **Not wanted:** a confirmatory 007c rerun of the NLA. I trust those results.
8. **Housekeeping:** a stale `scripts/007_jsubspace/gpu_queue_posthoc.sh` loop from an older session is still running
   (PID 49691, idle, holding no GPU). Ask me before killing it.

## 6. Environment on this instance

- **Machine:** 1× RTX PRO 6000 Blackwell (96 GB, sm_120), 128 CPUs, 377 GB RAM.
- **Python:** `/venv/main/bin/python` (torch 2.11.0+cu128, transformers 5.15.0, peft 0.21.2). Each Bash call is a fresh
  shell, so call the venv's python directly.
- **`/workspace` is NOT persistent.** The durable copies are the private GitHub repo and the HF mirror.
- **HF cache (`/workspace/.hf_home`, about 170 GB):**
  - Qwen3.6-27B;
  - J_NP: neuronpedia/jacobian-lens;
  - J_CB, R_CB and the template lens: camilablank/workspace-lenses;
  - the Oracle and its AR: agu18dec/olens_and_ar;
  - the AR's whiteners: the dataset agu18dec/local-workspace, `ckpts/ar/ptag/whiteners/`;
  - the NLAs: nla-rl, plus nla-L42 for 001/002.
  - Pins are in `scripts/common/registry.json`.
  - The cache also holds my HF login token files: never upload or print them.
- **GPU memory:**
  - The subject (50.1 GB) and the NLA's av_base (53.8 GB) don't fit together. Capture subject activations first, free
    the subject, then load the NLA.
  - Run one subject-model job at a time, and check `nvidia-smi` before launching.
- **Outputs:** `outputs/` (gitignored) is complete, including 008, 009 and 011. All 1,438 files are on the public HF
  dataset `senku21x/causal-concordance-outputs` (`scripts/000_setup/mirror_outputs.py` uploads, `restore_outputs.py`
  restores). Re-run the mirror after new runs.
- **Git:**
  - A stale `GITHUB_TOKEN` env var shadows the working gh login. Push with
    `env -u GITHUB_TOKEN git -c credential.helper='!gh auth git-credential' push origin <branch>`.
  - No git identity is configured, and you shouldn't change git config. Commit with `GIT_AUTHOR_NAME=senku14x`,
    `GIT_AUTHOR_EMAIL=visheshgupta14x@gmail.com` and the same `GIT_COMMITTER_*` values.
- **Key code:**
  - `scripts/common/` (readers, regime, jspace, engine, stats, wsb, partners, margin);
  - `scripts/007_jsubspace/`;
  - `scripts/008_trackb/` (c008, sigma, trackb, trackb_analysis);
  - `scripts/009_brewcode/` (bank, gate, derisk, read009*);
  - `scripts/010_reader_agreement/`;
  - `scripts/011_olens_swaps/` (ar_vectors, ar_offline_diag, olens_swaps, olens_swaps_analysis).
- **Tests:** `python -m pytest scripts/common/tests tests -q -m "not gpu"`. It passed 137 tests on 2026-10-06.
- **Agent reports** go in `docs/claude_reports/`, numbered from 31 with a date suffix. Update the README index there,
  and link canonical documents rather than copying them. `/workspace/claude_reports/` is the scratch copy.
- **Secrets:** the OpenRouter key (for LLM judges) is in `/workspace/.env`. It costs money, so ask before using it.
- **External reviews** come from me and are filed under `docs/reviews/`. Check each claim against the code and docs
  before accepting or disputing it.

## 7. How I want you to work

**Relaxed process rules** (these override the stricter process rules in older notes):
- **Run exploratory work freely.**
  - CPU analyses and GPU runs of up to about **2 GPU-hours each** need no go from me. Label them *exploratory*, log
    what you ran, and tell me what you found.
  - In exploratory runs you may fix bugs and rerun freely; just log it. Rule 12's no-repair discipline applies only to
    pre-registered runs.
- **Use subagents** whenever they help: reading, searching, parallel analyses. Modest workflows are fine for
  adversarial review or broad parallel analysis.
- **Branches, commits and outputs:**
  - commit and push to any branch other than `main`, as you go;
  - mirror outputs to the HF dataset;
  - publish private artifacts when they help;
  - keep `STATUS.md` and `CLAUDE.md`'s "Where things stand" current.
- **Confirmatory claims still need a short written plan:** a design note with the decision rule, committed before the
  run. Runs bigger than about 2 GPU-hours need a quick check-in with me first.
- **Stop after every smoke and report to me before the full run.** Don't chain smoke → full run automatically.
- **Use your judgement** on methods. Propose and try things, and push back when you think I'm wrong.

**About `CLAUDE.md`:**
- The relaxed rules above are written into `CLAUDE.md` as an uncommitted hunk, unless I've committed it by now. Claude
  Code's permission check blocks the agent from committing changes that loosen its own instruction file. Don't try to
  work around that.
- To commit other `CLAUDE.md` edits while that hunk is still uncommitted, stage a copy without the hunk
  (`git update-index --cacheinfo`). The previous session did this.
- If you write new process rules into `CLAUDE.md`, show me the diff first.

**Still firm:**
- never force-push, and don't push to `main` without my explicit say-so;
- ask before stopping or destroying the instance;
- ask before spending money on external services (paid APIs, LLM judges);
- Cathryn's repo (`/workspace/sources/repos/nla-vs-lenses`) stays read-only, and I handle all contact with her;
- private messages I paste (Slack threads and the like) stay in the conversation unless I say otherwise: don't commit
  them or store them in memory;
- the scientific non-negotiables in `CLAUDE.md` still hold:
  - labels never use readers (label code never imports reader code);
  - float32 writes with fidelity checks;
  - gates before outcomes;
  - baselines;
  - scoped wording ("decodable", "demonstrated influence on this answer");
- never rewrite a pre-registered record, and label exploratory and post hoc results as such. A rule changed after
  seeing outcomes is written as an amendment first, with what had been seen;
- record mistakes in `CLAUDE.md`'s silly-mistakes log.

**How I like to work:**
- lead with conclusions, in plain language, and keep reports concise;
- keep caveats next to the claims they qualify;
- give estimates in GPU time for runs, never estimates of how long coding will take;
- use the instance fully, and run things in parallel where that doesn't compromise quality;
- keep qualitative samples (raw readouts) next to the numbers;
- take report numbers from full-precision JSON, rounded half-up, never from printouts. Double rounding has bitten
  this project several times.

## 8. Your first task

1. **Read the paper in full, then the paper notes (04, 04b, 03), then the docs above.** Subagents can speed up the doc
   reading, but read the paper and the notes properly yourself.
2. **Check the instance:**
   - `git status`: expect branch `008-trackb-brewcode`, head f86c512 or later, with at most `CLAUDE.md` modified (the
     rules hunk);
   - the GPU is free;
   - `outputs/011_olens_swaps/s1/pairs/` has 74 files;
   - the CPU tests pass.
3. **Report back with:**
   - a one-page summary of the project and where it stands, in your own words;
   - anything in the docs that looks inconsistent, wrong or stale;
   - the three next steps you'd prioritise, with GPU-time estimates. Consider §5, but don't feel bound by it;
   - any questions for me.

Then wait for my pick, or start on the cheapest CPU-only step if it's clearly useful. Ask before any GPU run in this
first exchange.
