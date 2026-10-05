# My doubts and open questions (2026-10-04)

These are things I am unsure of, could not verify, or assumed. They are separate from the PI decision list in `01_SPEC_v2_review_2026-10-04.md` §e: these are the places where my own report could be wrong.

Each item says what I did, what remains uncertain, and how to settle it.

## A. Claims in my report that rest on indirect evidence

1. **Which NLA produced WorkspaceBench's published numbers.**
   - *What I did:* concluded "nla-rl + iter_000400" from the producer's code default (`methods.py:269–270`, unchanged since 494a980) and from the eval READMEs' arm name "nla-rl-L42".
   - *Doubt:* a run could have overridden the default on the command line. I did not check WSB's raw readout files.
   - *How to settle:* look at the arm metadata in WSB's published readout files.
2. **What WorkspaceBench's "NLA-SFT (L42)" bar is.**
   - *What I did:* presumed it is `av_base` with adapter `none`.
   - *Doubt:* this is not verified.
3. **My WorkspaceBench numbers.**
   - *What I did:* read the per-family numbers (±0.03) off bar heights in the results image.
   - *Doubt:* several of your notes' values were mis-assigned to the wrong family by the same method, so mine could be off too.
   - *Partial check:* the eval READMEs' "values of record" match my readings where they overlap (e.g. typo-mt NLA 0.75, multilingual typo 0.60, multilingual multihop 0.34).
4. **Which step the `nla-rl` reconstructor comes from.**
   - *What I found:* `saved_at_step.txt` says 600. The card says "checkpointed once at the end". But adapters exist up to `iter_000800`, and the card's own example calls 600 "latest".
   - *Doubt:* it is unclear whether the run continued past 600 after the AR was saved, or whether the card is stale.
   - *Why it matters:* "iter_000600 is the only step-matched pair" depends on this.
5. **The Oracle hash match.**
   - *What I verified:* the *current* `local-workspace` revision (e7804861, modified 2026-10-03) matches `olens_and_ar`.
   - *Doubt:* I cannot verify what Cathryn downloaded at the time of her pilot. The match is very likely but not proven for her files.
6. **Cathryn's multihop read position.**
   - *What I assumed:* "last word" (WSB, after 2026-09-23), based on her STATUS ("the bank's read token").
   - *Doubt:* I did not trace it in her code. If her runs predate 2026-09-23 or used the trailing space, 33/100 multihop items were read at a different token.
7. **The damage flag at ":" (contradiction B2).**
   - *What I did:* flagged it because the pre-registered rule and A01 say damage → unresolved.
   - *Doubt:* 001 §4 shows part of the off-pair mass moves to the intermediates (z, z′), which may be meaningful transfer rather than damage. The rule may be too blunt for whole-cell patches.
   - *Status:* I am flagging an inconsistency, not claiming the cell is damaged.
8. **004's span as a z-eraser (B12).**
   - *What I claimed:* it is not a validated z-eraser.
   - *Doubt:* this is an inference from 004 matching PCA and moving the recipient's own intermediate. 004 never tested erasure, so it might erase z fine. It is untested, not shown to fail.
9. **The damage share in Cathryn's removal control (B14).**
   - *What I claimed:* random vocabulary-J removal takes less of h than the bridge-neighbourhood removal.
   - *Doubt:* I reasoned this from the code. I did not compute ‖Δh‖ for her saved vectors.
   - *How to settle:* quick CPU check on `runs/abl3/resid_abl2.pt`, if it is in her repo.

## B. Design choices I proposed but am not sure about

10. **J-lens vector convention.**
    - The paper writes "rows of W_U J_ℓ", glossing over the final-norm gain γ.
    - I proposed the folded J_ℓᵀ(γ ⊙ W_U[t]) to match our readout. Anthropic's exact choice is unknown, and it changes the J / non-J split.
11. **Gradient pursuit details.**
    - The paper cites Blumensath & Davies with a non-negativity constraint, but gives no atom normalisation, stopping rule or step size.
    - Different choices change the selected set S and the J share.
    - The paper's "6–7% / 10–15%" figures are a sanity check, not a target to tune toward.
12. **Viswanath's "top-r Jacobian directions".**
    - From the post I could not tell whether these are singular vectors of J_ℓ, of W_U J_ℓ, or something else. I did not read his code (github PranavViswanath/nla-introspection).
13. **005 neutral cells.**
    - I argued 001's zeros do not transfer to injected content. I don't yet have a better candidate cell; the proposal is to measure cells with the random-direction control first.
14. **My GPU estimates.**
    - They scale from 000's nla-L42 timing (64 explanations / 107 s).
    - `nla-rl` is a separate full model with an unmerged r64 all-module LoRA (attention + MLP), so generation may be slower. The 1.3–1.7 h estimate could be off by about 2×.
15. **`NLAReconstructor` vs the `nla-rl` reconstructor.**
    - The layouts look the same (43 blocks + value head), but `nla_meta` says `norm: none` and our wrapper normalises to √d.
    - Untested. It could silently give wrong FVE.
16. **GPU memory plan.**
    - Capture first, then swap models, because `av_base` (53.8 GB) + the subject (50 GB) > 96 GB.
    - I assumed this; I haven't checked whether the PEFT-wrapped `av_base` plus the generation cache fits alongside anything else.

## C. Things I did not read or could not access

17. **Not read:**
    - design specs 001–004 in full (I read the reports);
    - `docs/COME_UP_TO_SPEED.md`;
    - `docs/reviews/*`;
    - Cathryn's scripts other than `ablate2_pass.py`;
    - the NLA paper's case-study transcripts and auditing appendix;
    - the AO paper's appendices.
18. **No access:** the shared "Workspace Work" doc. Ownership and joint-framing statements in SPEC_v2 §15 and §14.7 are taken on trust from the spec.
19. **Not read: `/workspace/codex_reports/`.**
    - The folder holds files written today (16:28–16:52): `workspace_paper_reading_notes.md`, `workspace_notes_reconciliation.md`, `workspace_paper_appendix_reference.md`, `sources/`.
    - Another agent (Codex?) seems to be working in this workspace. CLAUDE.md says to check `codex_reports/` before disputing a reviewer.
    - Was this folder meant for me to read?
20. **Inaccessible post:** "J-space auditing might be unreliable" (LessWrong 7Qi3LR9ZXpvptLp6m) returns 401. I only have a search-engine snippet.

## D. Questions about intent (I may have misread you)

21. **Repo visibility.** Is the repo public on purpose, e.g. so Cathryn can read it? I flagged it as urgent because three docs say it must be private while the PDFs are in it, but you may have decided otherwise after they were written.
22. **Which branch.** You said "branch: main", but CLAUDE.md names `setup/000-scaffold` as the working branch, and it is one commit ahead of main (the results page). I cut `spec-v2` from `main`. Was that right?
23. **What I committed.** I committed the review and reading notes to `spec-v2` and pushed (not main). You only explicitly asked for the spec commit. Tell me if you'd rather keep reports out of the repo; they are also here now.
24. **`nla-rl` iteration.** You named the repo, not the iteration. I am assuming the question is still open (400 for WSB/pilot comparability vs 600 for a matched reconstructor). If you meant 400 (what WSB and Cathryn used), say so and I'll drop the iteration check.
25. **Cathryn and the [v2] items.** Has she seen SPEC_v2's [v2] items? Several touch her arm: the A4 tag redefinition, quarantine of her chain test items, and her Oracle-at-L42 numbers being off-contract. Joint items need her sign-off under §15.
26. **Prompt-only baseline.** For reader floors, do you want the WSB-style prompt-only ("what would a model be thinking"), the 002 task baseline, or both? The spec currently treats them as one.
