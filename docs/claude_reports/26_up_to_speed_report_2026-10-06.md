# Up-to-speed report: Causal-Concordance (new session, 2026-10-06)

*For the PI. This is a reading and audit report: no experiment was run for it. Every number comes from the repo's
full-precision JSON or from the primary sources cited next to it. Two subagents helped: one digested the reader
literature and one audited the onboarding prompt's §3 numbers against the JSON. I re-checked their consequential claims
myself, as noted below.*

## 0. Short version

- **On brew, the readers' z signal tracks what the model is about to output, not what it demonstrably uses.**
  - The Oracle's z tracking is about 1.85× stronger at `</think>` (patch share ≈ 0.01) than at `:` (≈ 0.48).
  - The model's own next-token distribution already tracks z at all three cells.
- **007's model-side test is weaker than the docs suggest.** Stage 2b swaps at one cell and one layer, and the full z
  direction itself barely moves the answer. Fix that before building on the dissociation.
- **Recommended order:**
  1. redo the model-side §3.3 test the paper's way;
  2. give the Oracle J-test controls it can read;
  3. de-risk a family whose intermediate is never a plausible answer.

  Steps 1 and 2 together cost under 1 GPU-h.
- **Doc problems found:**
  - **One changes a reported conclusion:** the multihop restatement share is about 0.15–0.28, not "about half".
  - **Two change what we can build on:**
    - the Oracle Lens is WSB's oracle lens, not a Karvonen activation oracle;
    - nla-rl does ship an AR.
  - The rest are stale lines (§4).

## 1. What I read and checked

- **The workspace paper**, in full: main text and all appendices. I rendered figure pages from the PDF where the text
  extractions miss figure text (Figs 20, 22, 49–50).
- **Paper notes:**
  - the PI's notes `04_*` and `04b_*`;
  - `03_*` §4.5 and §13.5–15;
  - the parts of the digest `2607.15495_NOTES_digest.md` that matter here.
- **Project docs:**
  - all of `CLAUDE.md`;
  - `STATUS.md`;
  - the SPEC_v2 header and §2;
  - `AMENDMENT_02`;
  - `COME_UP_TO_SPEED` (update box);
  - the 005 follow-ups;
  - 006/006b;
  - the 007 report, design (§0–6, §20–23), status files and analysis JSON;
  - reports 21, 22 and 25;
  - the `claude_reports` index.
- **Reader literature:** Karvonen (AO), AObench, Li et al., Dingeto, and the NLA, nla_init, Hong, WSB and R-lens pages.
  I read these through a subagent's digest, plus my own check of the WSB Oracle Lens appendix.
- **Not read in full:**
  - the rest of `03`;
  - the SPEC_v2 body;
  - the 001–004 reports (I used their numbers through the JSON);
  - the review folders;
  - the results page (I read its source JSON instead).

## 2. Instance state

| Check | Result |
|---|---|
| git | `007-jsubspace` = `origin/main` = 97a63ee, clean tree. **The local `main` ref is stale at 95ab0e5**, 91 commits behind, so "branch off `main`" would start from the 000–004 scaffold. Branch from `origin/main`. |
| GPU | free (1 MiB used) |
| `outputs/007_jsubspace/` | present, 2.0 GB |
| CPU tests (`scripts/common/tests` + `tests/test_007_analysis.py`, `-m "not gpu"`) | 111 passed, 29 deselected |
| `scripts/common/readers.py` | imports |

## 3. The project and where it stands (my reading)

**Question.** Do activation readers report content that Qwen3.6-27B demonstrably depends on, judged against causal
labels that never use a reader? (SPEC_v2 §2)

**Labels.**
- **Only position-level labels are reliable.** On brew at L ≥ 40, the whole-cell patch share is 0 at every prompt
  position (median S over all rows of `outputs/001_leverage_profile/run/profile_brew.jsonl`, all 21 prompt positions).
  All the position-level variation sits in three cells (`results/001_leverage_profile/analysis/summary.json`, brew
  profile):

  | Cell | Patch share |
  |---|---|
  | `:` | 0.476 (L42), 0.503 (L44) |
  | assistant newline | about 0.30 |
  | `</think>` | 0.00–0.03 |

- **Content-level labels failed on brew:** 003's reliability gate (κ 0.37), and 004's fitted spans did no better than
  PCA.
- **So the North Star can currently only be answered at position level, on one family.** A 24-cell sweep would add
  little: the zero-share prompt positions are zero mostly through redundancy, so they can't tell "unused" from
  "redundant".

**Readers vs use: the most North-Star-relevant result so far.**
- The readers' z signal does not follow the cell's influence on the answer (006 post hoc).
  - Oracle L40: D_z 6.01 at `</think>` vs 3.22 at `:` (1.87×; L44 1.84×).
  - NLA: about equal across the two cells (−0.41, CI includes 0).
- The model's own next-token distribution tracks z at all three cells, D_z 3.66–3.99 (006b). z is the runner-up answer
  in 27 of 48 tables.
- **Reading:** reader z-tracking on brew is consistent with reading output-facing content; no held intermediate beyond
  that has been shown. Scope: one family, 3 cells, 48 dev tables.

**005.**
- On Qwen the planted-concept "report" is position-generic lexical priming. J gets no privilege at matched norm, and the
  token lenses detect nothing at L40–44.
- **This diverges from the paper**, whose Fig 7 includes a position control and found the report selective on Claude.

**007.**
- **Reader side.** The Oracle's z tracking at `:` L40 depends on the J subspace. H_J is confirmed on dev and on the 49
  test tables (J 5.76 / 5.03 vs whole 3.22 / 2.87; rest of J 0.30 / 0.29).
  - This is the J-transported token geometry specifically, not just unembedding-aligned directions: the LL_gain
    subspace gives 0.94 at L40 (`stage1a.json`).
  - The NLA's z signal is not J-concentrated (post hoc). So J dependence is a property of the reader.
- **Model side.** J-subspace swaps track content share (Stage 2), and the J part of the class-mean z direction moves the
  answer less than the non-J part (2b). Hence the reader–model dissociation.

**Weak links in 007 (what I'd fix before building on it).**
1. **Stage 2b is a weak test.**
   - It swaps at the member's `:` cell only, one layer at a time (design §6.3). The full z direction itself moves the
     answer only S 0.04–0.06, against 0.28–0.49 for the whole-cell patch.
   - The paper's §3.3 swaps are applied at every token position (§3.3: "we exchange (at all token positions) the lens
     coordinates"; Fig 13 caption: "a clamped lens-coordinate swap at every position"). The probe-split test of Fig 16
     repeats that swap protocol.
   - So "not demonstrated" is weak evidence of absence.
2. **H_J rests cleanly on P3 and P5 only.**
   - P1 and P2 are "+" against pieces the Oracle cannot read even on natural activations: PCA-4 LOO-prior ID is 0.25 at
     L40, and 5 of 10 random slices fail.
   - Amendment A1 only turns *nulls* into "uninformative", so these "+" results count.
   - The report says P3 and P5 are the clean form, but the "confirmed (P1, P2, P3, P5)" headline does not.
3. **The J piece also tracks the answer** (D_answer 3.22 at `:` L40). That is consistent with the output-facing
   reading, and brew cannot separate it.

## 4. Inconsistencies found (most consequential first)

### 4.1 Changes a reported conclusion

**Multihop restatement is overstated.**
- The figure "about half of bridge-naming J readouts restate a cue" comes from 1 − P(bridge without cue) / P(names
  bridge).
- In `analysis_secondary.py:139-150`, `bridge_without_cue` is counted only on items that have cue words, but both terms
  are divided by all samples.
- With a consistent denominator, among bridge-naming samples on items with cue words (test split), recomputed by me
  from `restatement_*.jsonl` with the same word-match rule:

  | Reader | J piece | whole | docs' figure (J) |
  |---|---|---|---|
  | Oracle L40 | 15/53 = 0.28 | 27/48 = 0.56 | 0.48 |
  | Oracle L44 | 13/54 = 0.24 | 12/48 = 0.25 | 0.45 |
  | NLA L42 | 8/53 = 0.15 | 28/48 = 0.58 | 0.37 |

- About 20 bridge-naming samples per reader come from items without cue words; the docs' formula counts them as
  restating.
- The "about half" claim appears in:
  - the 007 report (L30, L111);
  - `results/007_jsubspace/status/stage3.md` (L48);
  - report 25 (issue 6);
  - `CLAUDE.md` L211 and `STATUS.md` L13 ("confounded by restatement").
- Multihop still cannot separate reading from re-solving: re-solving needs no verbatim cue, and there are only 9 key
  clusters. But "confounded by restatement" is too strong for the J piece.

### 4.2 Change what we can build on

1. **The Oracle Lens is not a Karvonen activation oracle.**
   - It is WorkspaceBench's reproduction of the paper's A.9.2 oracle lens (`docs/sources/web/wsb_post.md:341-349`):
     - an AR (text → preceding activation, MSE in whitened space);
     - an inverter;
     - RL on bullet FVE with the AR frozen.
   - Its own AR ships in `agu18dec/olens_and_ar` (`ar_ptag_pooled`, val FVE 0.22 at L40/L44).
   - It reads the raw unit activation (`run.json`: transform "unit", α 16000). The whitener statistics are not public
     (its README: "Not included: the pooled whitener moments").
   - Misattributed in: `CLAUDE.md:23`, `docs/papers/INDEX.md:11` and the onboarding prompt.
2. **nla-rl ships an AR.**
   - `ceselder/qwen3.6-27b-nla-rl/ar_reconstructor/` is 35 GB bf16, saved at RL step 600.
   - It is not step-paired with our AV `iter_000400`; `av_rl_adapters/iter_000600` exists.
   - `fve_trajectory.json`: best FVE 77.7% at step 540.
   - `readers.py` has no wrapper for it. The onboarding prompt says there is no AR.
3. **More held-out data has been spent than the onboarding list says.** Beyond the brew test split (007) and multihop
   items 51–100:
   - the brew test split was read twice (the Oracle pre-registered, then the NLA post hoc);
   - relational Stage 4a used the whole bank, both name-set splits;
   - 005's 45 test concepts;
   - 001's gate already ran on the brew and chain test splits (design 007 §1.1).
4. **Reader inputs.** `CLAUDE.md` (Conventions) says reader inputs use the stock bf16 forward. The 006 and 007 brew
   reader inputs are saved SEC captures (007 design §4.1; 006 report, caveats).

### 4.3 Numbers sent to the TA

- **"44 held-out" planted concepts:** the correct number is **45**. 44 is only the n of the NLA contrasts (R1, P17),
  after G6 dropped "case" for the NLA (`results/005_planted_concepts/analysis/full_summary.json`).
- **"70 prompts" at L42:** Cathryn's numbers (PI, 2026-10-06). Dropped.

### 4.4 Stale or slightly off

- **"NLA disabled" with no mention of the post hoc rerun:**
  - the 007 report header (L7–8), §6 ("Not shown: anything about the NLA") and §7;
  - `STATUS` L12;
  - the `claude_reports` README row 24.
- **STATUS:**
  - L14 gives about 4.8 GPU-h for the post hoc NLA reruns. Including A3 it is about 7.0 GPU-h (134.9 + 75.9 + 0.7 +
    75.8 + 125.1 min + diagnostics).
  - L14 says mean removal "doubles" the NLA signal at `</think>`; it is 2.5× (3.58 vs 1.45).
  - L45 says 005 is "not merged into main"; L64 lists an untracked notebook that is gone; L134 ("read COME_UP_TO_SPEED
    first") conflicts with the paper-first rule.
- **CLAUDE.md:**
  - the read order (item 5) still calls 005/006 the latest experiments;
  - "private since 2026-10-04" conflicts with COME_UP_TO_SPEED (2026-10-02).
- **COME_UP_TO_SPEED:**
  - L94's membership statement contradicts its own L147 and report 001 §7;
  - L156–157 and L162 are stale.
- **`claude_reports` README:**
  - "Also in the repo? no" is wrong for 03–06, 10 and 12;
  - "Other locations" has the wrong path and branch;
  - the header is dated 10-04.
- **Over-rounded figures:**
  - "2×" for the Oracle at `</think>` vs `:` is 1.84–1.87×;
  - report 25's "2×" NLA noise is the median ratio, and the tails are 1–3×;
  - the 2b J-part S range starts at 0.013 (0.0125 rounded half-up), not 0.012.
- **The paper digest** (L204) repeats the paper's "load-bearing in base models". The paper has no base-model
  intervention: §6 is readouts only.
- **`docs/context/reading_notes.md`** (from the literature digest; line numbers are its):
  - L117: lists multihop prompt-only .90 as matching or beating every reader, but the Oracle is about .99;
  - L236: Li et al. Table 6 is not "target-only";
  - L242: the R-lens metric is a per-layer mean, not "pass@10 at any layer";
  - L249: co-adapted codes are not established for the released NLAs.

## 5. What the reader literature adds

- **No source tests a verbalizer on an intermediate that is never a plausible answer.**
  - Several training signals push readers toward next-token content:
    - Karvonen's next-token prediction half;
    - the NLA warm start on "features for this prediction";
    - nla_init's last-paragraph dominance.
  - That is our runner-up confound, unaddressed in the literature.
- **Prompt-only baselines often match verbalizers.**
  - Li et al.: zero-shot .64/.68 vs LIT .64/.61.
  - AObench footnote 5: a text-only variant performed "similar".
  - WSB brew: prompt-only .96 against every reader.
  - Li et al. Table 28: a verbalizer reading a shuffled-trained model reports the *original* labels (0.94 vs 0.03), i.e.
    the reader's prior.
- **NLA claims are rarely grounded.**
  - Dingeto: only 1.6–4.2% of claims grounded at r̃ 0.84.
  - Masking content words drops reconstruction to .04.
  - Hong: entity swaps cost 0.22–0.39 FVE.
- **Off-contract layers:** Karvonen finds untrained 33%/66% depth works for classification but 75% hurts secret-keeping.
  Nobody tests reads at an untrained layer between two trained ones.

## 6. Next steps (GPU time; the code is not the cost)

| # | Step | What gets built | GPU |
|---|---|---|---|
| 1 | **The model-side §3.3 test, done the paper's way.** Class-mean z swaps clamped over the three emit cells × L40–48. Conditions: full direction, J part (J_NP, J_CB), non-J part, random, at matched raw norm **and** matched whitened norm. Includes the paper's complementary-component clamp, fidelity checks and damage readouts. | A multi-cell, multi-layer swap on 007's `swap.Runner` / SEC `edit_fn`; one wikitext forward for a covariance per layer | ~35–40 min |
| 2 | **Readable controls for H_J** (report 25's top issue). Oracle L40/L44, dev tables. Pieces that pass the positive control: the top-1,024 PCs of generic activations; random 1,024-dim slices drawn inside those PCs; J built in whitened coordinates. | New pieces from step 1's covariance, scored through 007's `reader_split` path | ~10–15 min (Oracle); +60–70 min with the NLA |
| 3 | **De-risk a family that removes the runner-up confound.** About 60 counterfactual multihop pairs: a one-token cue change changes the bridge, and the bridge is never an answer candidate. Check: (a) the bridge ranks low in the model's own output at the read cell; (b) a whole-cell patch label exists; (c) class-mean bridge swaps move the answer as predicted. | Items, gate, patch profile and swaps, reusing 001/007 machinery | ~1–1.5 h |

- **Decision value:**
  - **Step 1.** If J ≥ non-J with a strong lever, 007's dissociation story changes. If not, "no model-side privilege" at
    brew's emit cells becomes a supported claim. Either way it answers Camila's whitening question.
  - **Step 2.** Tells whether the Oracle depends on J itself or on any natural-looking piece.
  - **Step 3.** A go/no-go for content-level labels on Qwen. Zeisler's 6–11% swap flips on Qwen make (c) the risk.
- **Started (exploratory, CPU only, scratchpad):** a qualitative analysis of the saved 007 readouts. It counts how often
  each reader names z, the answer or the start, how often it reconstructs the task, and what the NLA says from outside
  J. Results to follow.
- **Deprioritised: AR-based NLA hallucination stripping.**
  - The literature expects stripping false specifics to lower FVE, so FVE retention does not measure truth.
  - The AR is not step-matched to our adapter, and the rewrite needs a paid LLM.
  - Brew gives exact start, z and answer, so NLA claims can be scored against truth directly, with no AR.

## 7. Open questions for the PI

1. Which steps: 1 and 2 together, with 3's design in parallel (my recommendation)?
2. Near-term deliverable: a write-up of the reader–model dissociation (steps 1–2 first), or progress toward content-level
   labels (step 3 first)?
3. May I fix the stale docs on a branch, with the restatement numbers as a dated post hoc correction (pre-registered text
   untouched)? Do you want the relaxed rules in `CLAUDE.md`? I'll show the diff first.
