---
name: causal-concordance-pi-decisions-2026-10
description: "PI decisions on Causal-Concordance after report 38 not yet in the repo (AR loader acceptance, CW-20, Oracle-vs-NLA J scope = brew/typo only, patch check status)"
metadata:
  node_type: memory
  type: project
  originSessionId: 93adeb25-7209-4a73-96ec-799b09ae8e90
  modified: 2026-10-11T01:16:19.086Z
---

Causal-Concordance (PI Vishesh, GitHub senku14x; repo private, newest branch `014-typo`). Decisions the PI gave on 2026-10-10 (most are now committed, see below):
- The TA confirmed the Oracle Lens reconstructor (AR, `agu18dec/olens_and_ar` `ar_ptag_pooled`) has a scale-invariant training loss, and said to use our loader (`scripts/011_olens_swaps/ar_vectors.py`) as is.
- The loader is accepted post hoc for L36–60 only (retrieval still misses at L20–32). Reconstructor outputs are treated as directions only.
- Cathryn's CW-20 (2026-10-09) used our loader. Correcting the wrong pieces of the Oracle's write-ups changed FVE by −0.3 points [−0.7, +0.2].
- The PI confirmed on 2026-10-10 that the Oracle-vs-NLA J-dependence difference holds on brew `:` and typo only, not on multihop (the NLA also needs J there: 007 test half 0.80 → 0.225, CW-14). Don't call it a general reader property.
- The PI also confirmed that 03 §14 corrects the PI's notes and §15 lists agent-vs-PI differences; neither is about the paper.
- On 2026-10-10, with the PI's go, these were committed on `014-typo` and pushed:
  - `d4b9860`: the 10-09 patch with P1–P8 fixed. The 012 lever is now worded "mainly where the bridge is a named place" (countries plus 7 geographic places, `GEO_PLACES` in `stageA_splits.py`, POST HOC).
  - `128bd12`: claude_ovarida reports 39 and 40, plus the doc fixes I1/I7/I11/I12/I6/L1/N1.
- The PI ruled that the 10-09 acceptance (L36–60, directions only) meets 011's precondition. The E0 JSON stays unedited; the ledger and CLAUDE.md carry the acceptance. `STATUS.md:31` "Page v12" (dated) is left as is.
- 2026-10-10: the next experiment, 012 Stage E / 011 amendment 4 (AR concept-vector swaps, brew then multihop), is approved and committed at `05061db` (`docs/design_specs/012E_ar_concept_swaps.md`). Multihop arms must run in Jan's ModelRunner bf16 + BandEditor regime, not `engine.Engine`. Step 2 (phrase generation) is blocked: the OpenRouter key (`/workspace/.env`) is missing on this instance.
- 2026-10-10 Stage E progress: step 2 report 41 (brew stopped by rule; multihop primary = pooled phrases); step 4 smoke report 42 at `a25b649` (parity exact; class-mean lever failed 10/51, so the primary uses J-row sizes). Waiting on the PI's go for the full multihop run with J-row-size nulls added.
- 2026-10-10 Stage E full run done (report 43, `5335721`). AR concept differences flip 37/82 at J-row size; J-removed 0/82 (weak handle, +0.29 nats); random-removed 41/82, so J-attribution. No non-J lever from the Oracle (S1 stands). The answer control flips equally (S2 binding). Named places / Japan only. Waiting on the PI for next direction.
- 2026-10-10 §17 (report 44, `b975986`): an energy-matched random removal keeps the AR edit effect (45/82), so the J reading holds (J-specific); PCA-k could not be energy-matched (rank about 618 captures only 0.32). Next design (PI): AR edits on the Stage I/J delayed-question prompts, clue positions only, two relations per held state. Draft it only on the PI's word.
- 2026-10-11: experiment 015 (Oracle hallucination test with cross-scoring) is on branch `015-oracle-hallucination`; spec `e6dd72f`. Step 2 R1 passed on all four cells (report 45, `f64d6fc`). Next: step 3 smoke with judges, on the PI's go. The NLA reconstructor (35 GB) was downloaded at `5a13b7ec`; rebuild_env.sh doesn't fetch it.
- 2026-10-11: 015 step 3 smoke done (report 46, `aad789f`, pushed). All gates pass; G-repro exact.
  - Oracle × Oracle AR: Δ_A +15.9, but the position-matched Δ_AP +5.3 is unresolved.
  - The Oracle's F is mostly deletion of follow-on text, so F − P is −7.6 (NLA +7.8).
  - Judge P echoed `<<<>>>` delimiters (stripped before scoring).
  - Under the Oracle AR the NLA texts sit at the floor.
  - Waiting on PI decisions 4a–4e (Fc arm, scale-free reader view, PA coverage, misquote tags, 013 empty-control note), then the full run on the PI's go.
- 2026-10-11: the PI said "okay start it", so 4a–4e were adopted as I recommended. 015 full run done: report 47, results page https://claude.ai/artifact/7JkKovrdEVXZsAoB7WY6jd (private until the PI shares it), pushed `b9b8c04`, outputs mirrored to HF.
  - Oracle × Oracle AR: Δ_A +14.5, Δ_AP +4.3, primary +9.0 (wrong answers).
  - NLA AR: nothing moves on Oracle texts.
  - The reader contrasts flip with the reconstructor, so there is no reader-specific claim.
  - Fc − Pc −1.5.
  - Judge spend $6.76 (my estimate was $1.8).
  - Fixed during the run: the answer-span locator (013's first-occurrence locate) and delimiter echoes.
  - Next is the PI's call; report 47 §9 proposes an answer-swap test.
- 2026-10-11 (later): the PI's red-team review was checked and corrections 1–3 applied. Report 48 (`29d9310`, mirrored to HF, page v4):
  - Swap test: any rearrangement of Oracle openings costs under the Oracle AR; true vs wrong is unresolved.
  - Runner-up test: the Oracle's wrong answers are Qwen's runner-ups (+6.6 nats, median rank 7). The gate missed on 6/261 near-tied cells.
  - The red-team's "report-44 clamp mechanism check" does not exist anywhere; it was not run.
  - The PI called the project at a stopping point. The Stage I/J design is not drafted.

**Why:** these change how the E0 "not validated" verdict and the 011 "unexplained scale" caveat should be read.
- The TA did NOT address the AR's J term (`loss_mix_lambda` 0.25, `jfve`) or the Oracle's reward. They said only that the loss has no scale term and that the loader works. The J-aware-AR hypothesis is still open.
- The PI wants the patch verified before anything in it is trusted ("don't take it to heart, verify it first").

**How to apply:** check whether the repo has recorded these before restating them. Never treat the J term as settled. Related: [[research-collaborator-stance]].
- 2026-10-11 (final wrap-up, PI): Part 1 inventory given in chat.
  - The clamp mechanism check was first defined that day and was run (012E §18, report 49). Per-layer clamps accumulate the persistent part to about k/cos (rrem 1.01, emrem 0.97 of 1/cos); AR_Jrem does not accumulate.
  - Caveats were added to reports 44 and 008; CLAUDE.md got a pitfall; STATUS.md got a closing note.
  - Spend since report 41 is $10.64 (results/spend_compute_2026-10-11.json).
  - The PI decides on merging; main is untouched. Still logged in to GitHub and HF: log out only when the PI says so.
