# 40. Problems found (2026-10-10)

This report lists in one place every problem I found while coming up to speed (report 39) and while checking the
2026-10-09 patch. Each item gives where the problem is, what is wrong, the evidence and its status:
- **fixed by the patch**;
- **fixed in `d4b9860`** (the patch with P1–P8 fixed) or **fixed in this commit** (the drafts plus I6, L1 and N1);
- **open**;
- **your call**.

**Status (2026-10-10, PI's go).** This report was first written uncommitted. The statuses below are after two commits on
`014-typo`:
- `d4b9860` applied the 10-09 patch with P1–P8 fixed;
- the commit that adds this report applies the I1/I7/I11/I12 drafts and fixes I6, L1 and N1.

Line numbers refer to `b8d458e` unless marked *patched*, which means `b8d458e` plus the 10-09 patch. Numbers are
rounded half-up from the JSON named.

**Summary.**

| Group | Items | Status |
|---|---|---|
| A0. The 10-09 patch: what is right | process, all numbers but P1 | checked against JSON |
| A. The 10-09 patch: problems | P1–P8 | fixed in `d4b9860` (one wrong number, P1; seven wording fixes, P2–P8) |
| B. Repo docs | I1, I7, I11, I12 | fixed in this commit (from the drafts) |
| | I3, I5, I8, I9; most of I2 | fixed by the patch |
| | I6, L1, N1 | fixed in this commit |
| | I4 | left as is (dated; PI) |
| | I10, I13, I14 | open / your call / note only |
| C. Science | S1–S5 | open; these limit what can be claimed |
| D. Instance and tooling | T1–T4 | open; none affects a number |
| E. My own mistakes | M1–M3 | corrected |

## A0. What the 10-09 patch gets right

I re-checked these on 2026-10-10, from the JSON, after the first check. Every number the patch writes was compared with
its source, rounded half-up. All match except P1 (section A).

**Process.**
- `--check` passes and writes nothing.
- Both scripts reproduce the bundle's JSON byte for byte. The sha256 values are `91c87806…8ec3dd` (`stageA_splits.json`)
  and `12c46fbe…213519` (`rewrite_by_state.json`), as the patch list says.
- The parity asserts pass. The cohort is Stage A/D's 82 clean-correct items, and the bootstrap is the one in
  `rewrite_all.json`.
- Ledgers are only appended to, no spec is touched, and the CPU tests pass (`pytest scripts/common/tests -m "not gpu"`:
  104 passed, 29 deselected).

**Numbers, by source.**

| Patch text (where) | Values | Source (key) |
|---|---|---|
| 012 lever, pooled (`CLAUDE.md`, `STATUS.md`) | 48/82, 38/82; random 5/82, 1/82; logit 0/82; 0.585 | `stageA_summary.json` `runs.j_swap_L24-47`, `j_swap_L24-35`, `logit_swap_L24-47` |
| 012 lever split (`CLAUDE.md`, 012 ledger table) | Japan 26/27, 27/27 (random 0, 2); other targets 12/55, 21/55 (1, 3); country bridges 33/42, 38/42 (0, 4); country not Japan 7/15, 11/15 (0, 2); other bridges 5/40, 10/40 (1, 1) | `stageA_splits.json` `bands.*.by_target`, `by_bridge` |
| Shared relations; direction | Japan 16/17, 17/17 vs others 5/10, 9/10; toward_src 2/18, 0/14 | `bands.*.shared_relations`, `by_direction` |
| Read-position edit size | J 0.0077 / 0.0074, random 0.041 / 0.067, random larger on 82/82; working = failed 0.0077 at L24-35 | `bands.*.edit_size_read_position` |
| Random arm | text changed 12/82 and 36/82 (J 40, 51); median Δm J +5.0 / +8.8, random +0.28 / +0.79 | `bands.*.arm_effects` |
| Model-output AUC, toward_tgt | 0.97 (64) / 0.91 (68); no Japan 0.94 (39, 12) / 0.84 (41, 21); country 0.95 / 0.85; other 0.96 (27, 5) / 0.88 (26, 10) | `bands.*.model_output_auc_toward_tgt` |
| Japan cohorts | 29/96, 28/87, 27/82 | spec `012_reader_concordance.md:54`; `012_ledger.md:148`; `stageA_splits.json` |
| 012 ledger correction to line 101 | J_NP / J_CB 0.92 at L40, 0.93 at L44 | `stageD_analysis.json` `cells.J_*|L4x|L24-35.auc_tt` |
| E0 (`CLAUDE.md`, 012 ledger) | scaled FVE within 0.04 at 12/12 layers; ret@1 within 0.05 at 8/12; L20-32 short by 0.05-0.11 | `stageE0_ar_profile.json` `verdicts.counts`, `layers.*` |
| | ret@1 0.885 / 0.901 / 0.914 vs published 0.92 / 0.92 / 0.93 (L36 / L40 / L44) | `layers.*.white.ret1_b24`, `published.ret1` |
| | scale 0.11 (L20) → 2.33 (L63), 0.38 / 0.42 (L40 / L44); L44 cosine 0.46 | `layers.*.white.global_scale`, `cos_mean` |
| CW-20 checks | capture cosine min 0.9997; bullets cosine 0.46, scale 0.43, ret@1 0.84 among 70 | Cathryn's `runs/cw20/gate.txt` (`orig_bullet`) |
| 014 results (`CLAUDE.md`) | naming 0.77-1.00 (min 0.765); other-target null ≤ 0.02 (max 0.024, which rounds to 0.02) | `parts_full.json` `naming` |
| | prompt-only 1.00 | WSB `evals/baselines/prompt_only.json` `typo.rate` |
| | J cost Oracle 0.14 / 0.225, NLA 0.025; Oracle − NLA +0.115 [0.05, 0.19] / +0.20 [0.13, 0.275] | `paired_full.json` `contrasts` |
| | PCA-k cost Oracle 0.035 / 0.02; D_PCA 0.105 [0.035, 0.18] / 0.205 [0.13, 0.28]; NLA 0.02 [−0.005, 0.055] | `claude_ovarida/results/typo_content_control.json` `readers` |
| | Part C: gate FVE 54.7%; deletion +39.3 [31.8, 47.3]; rewrite F − P +1.8 [−1.2, 4.6] | `nlah_gate_full.json`, `hallucination_all.json`, 014 `rewrite_all.json` |
| 013 by state (`CLAUDE.md`, `STATUS.md`, 013 ledger table) | all 15 cells (F − O, P − O, F − P for clean, J swap, random swap, items 31-100, items 1-30) and their n | `rewrite_by_state.json` `per_state`, `clean_items_*` |
| | pooled F − P +3.1 [1.3, 4.9], F − O +1.6, P − O −1.4 | 013 `rewrite_all.json` (matches `rewrite_by_state.json` `pooled`) |
| CW-16b reference | +5.4 [2.6, 8.0] | Cathryn's `runs/cw16b/analysis.txt` (corrected − orig +0.054 [+0.026, +0.080]) |
| | her 70 items are exactly WSB multihop items 31-100 | `runs/cw16b/texts.jsonl` keys vs `wsb.load_bank("multihop")` |
| 009 (`CLAUDE.md`, `STATUS.md`) | D +11.9 [11.7, 12.1], 240 tables / 658 pairs; all pairs +11.3 [11.1, 11.5] (771 pairs) | `outfacing.json` `D.final.emit_colon.z`, `D_all_pairs.final.emit_colon.z` |
| 010 (`CLAUDE.md`, `STATUS.md`) | mid-layer Oracle vs J lenses 0.33-0.63 (same layer 0.43-0.63), vs output 0.24-0.30 | `agreement.json` `agreement.colon|z`, pairs Pearson |
| | NLA 0.30-0.37 / 0.07; late Oracle 0.62 / 0.82; NLA-Oracle 0.24-0.30 | same |
| | across the three cells 0.33-0.75 and 0.12-0.30 | same, `colon|z`, `think|z`, `asst|z` |
| 006 (`CLAUDE.md`) | 1.87× / 1.84× | `posthoc.json` `cells.Oracle L4x`, ratio `emit_think.mean / emit_colon.mean` (see P8) |
| 008 (`008_trackb.md`) | non-J J_NP 0.21 → 0.20 clamped; J_CB 0.2048 → 0.18 | `trackb_b1.json` `conditions.emit3|B_raw_nonJ_*`, `…clampJ_*` `.median_S` |
| | 007 Stage 2: 33 pairs, 32 at L40; S_full 0.28 / 0.43 / 0.49 | `stage2.json` `layers.*.primary.*.n`, `S_full` |
| 003 (`STATUS.md`, `003_labels.md:164`) | low ends: median S 0.024, "used" 0.085 at assistant newline L42 | `003_labels/analysis/summary.json` `summary.emit_asst|L42|z` |

**Edits without numbers that are also right.**
- The `STATUS.md` header now points at 014 and reports 36-38.
- "Fresh instance" now says to check out `014-typo`.
- Stage L is marked descriptive.
- The 009 "98%" and the 011 single-token cosines are marked as having no JSON key.
- The 005 "≈ 95%" is marked as untraced.
- s1 and the 007 NLA are marked superseded, apart from the leftover sentence in P6.
- The 013 ledger says A2's F − P is a post hoc co-primary.

**Not checked.**
- The live page state in E2 (v13 has the typo sections but not 16b). I did not open the artifact.
- The "17-row head" note. It is labelled a hypothesis, and I did not check it against the AR's files.
- 012's random-target direction being unit norm (T4).

## A. Problems in the 10-09 patch

**How I checked it.** I applied the patch to a scratch clone at `b8d458e` (`/workspace/scratch/cc_check/repo`). There:
- `--check` passes;
- both scripts reproduce the bundle's JSON byte for byte, and their parity asserts pass;
- every other number the patch writes matches the JSON;
- the ledgers are only appended to;
- the CPU tests pass.

All eight are fixed in `d4b9860` as written below. P5 was your call; you answered yes.

Details are in report 39, §5. Nothing below blocks applying the patch, and each is a one-line fix.

- **P1. Wrong number.** The patch's 012 ledger append says CW-20 gives "+0.22 bullet by bullet".
  - Cathryn's `runs/cw20/analysis.txt` gives +0.209 for the unedited write-ups, and her `STATUS.md` (CW-20, Checks)
    gives +0.21. The 0.22 in her file is the published reconstructor value.
  - Fix: +0.21. Also worth saying: her "whole write-up as one text" +0.02 equals her other-prompt control (+0.02).
- **P2. Stale clause.** The *patched* `CLAUDE.md` results-page bullet says "covering 012, Stage L, 013 and 014", then
  still ends with the old "It covers 012, Stage L and 013,". Fix: delete the old clause.
- **P3. Dangling parenthetical.** In the *patched* `CLAUDE.md` 013 block, "(correction gains a few points;
  restatements carry the content; framing is padding)" described CW-16b's findings. It now follows "the FVE baselines
  differ", so it reads as explaining the baselines. Fix: move it after "Matches Cathryn's CW-16b in direction".
- **P4. "agree" is too strong.** The *patched* E0 acceptance says "Cathryn's CW-20 checks at L44 agree". The inputs
  differ:
  - CW-20 scores Oracle bullets among 70 targets; E0 scores assistant spans;
  - her FVE level uses a corpus-centred score that her `STATUS.md` says was chosen post hoc.
  So the matching cosine (0.46 vs 0.46) and scale (0.43 vs 0.42) are consistent, not a test of agreement. Fix:
  "are consistent with". The 012 ledger append words it correctly.
- **P5. 011 not reconciled with the 10-09 acceptance.** **Your call.** 011 read L40/L44, inside the accepted L36–60.
  - The *patched* `CLAUDE.md` read-order item 5 and the 011 block still say "Not validated against the published AR".
  - One of the reasons listed is "unscaled FVE negative", which the scale-invariant loss now explains.
  - "Open PI decisions" still says any 011 next step needs the loader validated first.
  These lines should at least cite the acceptance.
- **P6. Leftover sentence.** The *patched* `STATUS.md` 007 line gains "*(Superseded: …)*" but keeps "Post hoc NLA run on
  006's 24-row path is the PI's call." Fix: delete it.
- **P7. "Country" vs "place"** (observation; post hoc on top of a post hoc split). The lever text says the swaps work
  "mainly where the bridge is a country".
  - The 40 "other" bridges include 7 geographic places (Africa, Antarctica, Chicago, Everest, Nile, Olympus, Sydney).
    They flip 4/7 (L24-35) and 6/7 (L24-47), like country bridges (33/42 and 38/42, `stageA_splits.json`
    `bands.*.by_bridge`).
  - The other 33 flip 1/33 and 4/33.
  - These are my counts from `outputs/012_reader_concordance/stageB/items.jsonl` with the script's cohort. They are
    not in any committed JSON, and "place" is my grouping.
  - Fix (done in `d4b9860`): a named-place group in `stageA_splits.py`. It is now in `stageA_splits.json`
    `by_bridge.named_place`, `place_not_country` and `not_named_place`: named places flip 37/49 and 44/49 (random 1/49
    and 5/49), the rest 1/33 and 4/33 (random 0/33). With 7 items this is weak either
    way.
- **P8. Derived number cited as a JSON key.** The *patched* `CLAUDE.md` 006 line cites `posthoc.json` for "1.87× /
  1.84×".
  - These are ratios, `cells.Oracle L40|L44.emit_think.mean / emit_colon.mean`; they are correct half-up.
  - The stored key is the difference `think_minus_colon`: +2.79 / +2.73 nats.
  - Fix: "ratio of".
- **Not checkable here.** The patch says the random-target arm swaps toward "a random unit direction". That direction
  is generated in Jan Chen's phrase-lens code, which `rebuild_env.sh` does not clone and which is not on this
  instance (T4).

## B. Problems in the repo docs

### Fixed in this commit (drafted 2026-10-10, applied after `d4b9860`)

- **I1. 03 §14–§15 are described as corrections to the paper; they aren't.**
  - Where: `CLAUDE.md:28-29`, `docs/claude_reports/README.md:36`, `35_new_session_prompt_2026-10-07.md:27` and `:82`.
    Report 35 is where the misdescription starts.
  - What they are: §14 is "Corrections to the user's notes (checked against originals)" (`03:998`). §15 is "Where my
    reading and the user's differ" (`03:1022`).
  - The notes' own corrections of the paper are 04 §2.3 (A.8 spans vs cones, `04:80`) and 04b. Their disagreements
    with the paper are 04 §9, refined by 04b.
  - You confirmed this reading on 2026-10-10.
- **I7. 002 rounding slips.** In `docs/reports/002_claim_scoring.md:73-82`, 14 numbers (in 13 cells) were rounded
  down from values ending in 5 (rates are multiples of 1/40). Report 31 lists 4 of them.
  - Two bullets carry the same slips: `:88` (42% → 43%) and `:93` (0.00–0.07 → 0.00–0.08).
  - Every corrected value is checked against `results/002_claim_scoring/analysis/scores.json` `rates`.
  - No reading changes.
- **I11. Fig. 8's prose numbers are used as if they were the figure.** `04:393` and 03 §4.1 (`03:208-210`) give the
  prose values 88 / 59 / 5 / 0.
  - The rendered bars read about 87 / 55 / 9 / 1 for the swap (PDF p. 13, my reading, ±2 points).
  - For the injected thought they read 89 / 79 / 12 / 3, each at its best strength.
  - Fig. 16 (p. 19) has a 54% bar the prose omits, and gives the J share of probe variance as 16% (prose 10–15%).
  - 03 §14's "Checked and matching" doesn't say that figures were only inspected for A.9.
  - 04b:73 already notes the Fig. 8 discrepancy. 04 is verbatim, so the draft leaves it alone and records the
    discrepancy in 03 §14 (new item 7).
- **I12. Typo is counted as "a second family" for the Oracle-vs-NLA J difference, without the multihop counterexample.**
  - Where: report 38:56-58 and `claude_ovarida/README.md:10`.
  - On multihop, on the same J basis, the NLA loses the hidden step without J as the Oracle does. 007's `J_raw_L42`
    equals Cathryn's CW-12/14 basis (mean cos² 0.99999997, `007_jsubspace/analysis/geometry_overlaps.json`
    `ours_raw_L42_vs_cw`).
  - Naming, whole → rest of J on 007's test half (`012_reader_concordance/analysis/j_split_naming.json` `cells`): NLA
    L42 0.80 → 0.225; Oracle L40 0.85 → 0.425; Oracle L44 0.8625 → 0.125.
  - CW-14 J − rest: +0.65 (NLA) and +0.71 (Oracle).
  - You confirmed on 2026-10-10 that the difference holds on brew `:` and typo only.
  - The draft also adds a scope line to `CLAUDE.md`'s 014 block, `STATUS.md`, the 014 ledger (appended) and the page
    text in `make_page.py`. The page is not rebuilt. Report 36's R6 is not yet restated.
  - The fixes in report 38 and the 014 ledger cite report 39 (S3), which is committed with this report.

### Fixed by the patch

- **I2.** Reports 37–38 were not mentioned in `CLAUDE.md` or `STATUS.md`. The *patched* `STATUS.md` header points to
  `claude_ovarida/` reports 36–38, and the 014 block cites report 38. `CLAUDE.md`'s heading still says "Where things
  stand (2026-10-07)" (minor).
- **I3.** `STATUS.md:3-8` said the newest work was 012 and report 34. Fixed.
- **I5.** "Fresh instance" step 1 said to check out `008-trackb-brewcode`. Now `014-typo`.
- **I8.** `docs/ledger/012_ledger.md:101`, "J_NP and J_CB about 0.88–0.91". A correction is appended: at L40/L44 the
  values are 0.92–0.93; 0.88–0.91 is L52/L56.
- **I9.** `STATUS.md` and `003_labels.md:164`: the assistant newline L42 median S is 0.024 and "used" is 0.085. Ranges
  are corrected to 0.02–0.05 and 9–18%.

### Leftovers the patch misses

- **I4.** Page version.
  - *Patched* `STATUS.md:31` still says "Page v12."
  - v13 is published and v14 is built but unpublished, which the *patched* header itself says.
  - `CLAUDE.md:216` ("v9-v12" for the 013 branch) is history and fine.
  - **Left as is:** the line is dated, so you asked for it to stay.
- **I6.** "Unexplained global output scale".
  - The patch fixes `CLAUDE.md`, but *patched* `STATUS.md:144` still says "The global scale is unexplained".
  - The 011 spec (`011_olens_swaps.md:111`, `:126`, `:149`, `:189`) says the same, and so does `:186-187` on "the
    published 0.22 is unscaled". The spec is a pre-registration and must not be edited, so this belongs in a ledger or
    note annotation.
  - **Fixed in this commit:** `STATUS.md` now gives the per-layer scale, and `CLAUDE.md`'s 011 block points at the spec
    lines it supersedes (`:109`, `:126`, `:150`, `:186-189`). The spec itself is not edited.
  - The scale is per layer (0.11 at L20 → 2.33 at L63, `stageE0_ar_profile.json` `layers.*.white.global_scale`), not
    global. It is expected under the scale-invariant loss (A.9.2; the TA).
- **L1.** *Patched* `STATUS.md:182` still says "The Oracle's 2× at `</think>`". The patch corrects the same claim in
  `CLAUDE.md` to 1.87× / 1.84× (see P8 on how to cite it). **Fixed in this commit** (the ratio, cited as a ratio).

### Open, your call, or note only

- **I10.** Ratnaditya's control-AUC CI is [0.532, 0.659] in 03 (`:824`) but [0.523, 0.659] in 04b (`:243`). The source
  isn't in the repo, so I can't say which is right. **Open.**
- **I13.** Commits `1aa47a3` and `ec053d1` say your "rules hunk" in `CLAUDE.md` was left uncommitted. It is not in the
  clone, so it is lost unless you have it. Report 36 E7 summarised it as allowing subagents. **Your call.**
- **I14.** Cathryn's `STATUS.md:7` still says "Updated 2026-10-05 (after CW-14)", though it covers CW-15 to CW-20.
  That is her file; I'm only noting it.
- **N1** (new, minor). `002_claim_scoring.md:84-85` says the WSB-grid Oracle's role-correct rate is "0–0.10 at every
  layer".
  - That holds for the rows the page shows (member renders, `:`).
  - At `</think>` L60 the member rate is 0.20 (n = 10, `scores.json` `rates["oracle_wsb|member|cell1|L60|real"]`).
  - Partner and substituted renders reach 0.30.
  - **Fixed in this commit:** the sentence is scoped to `:`, with the `</think>` and partner exceptions.

## C. Scientific concerns (open)

These are limits on what can be claimed, not errors in the docs. Details are in report 39, §4 and §6.
- **S1. Circular labels.** 012's "used" labels come from J-row swaps. Token lenses are J-sensitive by construction, and
  the Oracle may be J-sensitive by training (the J-aware-AR hypothesis). So reader-vs-label results on multihop can't
  rank J-type readers independently.
- **S2. Output-facing content.** No family yet has a used intermediate that is absent from the output tail. 009 tried,
  and its check failed. A new family (hidden user attributes) needs that check before any reader runs.
- **S3. Typo is re-solvable.** The corrected word follows from the input. Rest-of-J keeps the input's form (cosine 0.92
  with the whole activation, CW-14), and the NLA's typo explanations are mostly misquotes of the prompt (014 Part C).
  So the NLA naming the word without J may be inversion followed by correction, not reading a held representation.
- **S4. Layer mismatch.** Every Oracle–NLA comparison pairs Oracle L40/L44 with NLA L42.
- **S5. A mostly post hoc decision trail.** 014's paired contrasts, report 38's D_PCA and 013's kind and rewrite arms
  were chosen after their data were seen. Confidence should stay at medium at most.
- **The AR's J term is still unanswered.** The TA confirmed only that the AR loss has no scale term and that the loader
  works. `loss_mix_lambda` 0.25 / `val_jfve` and the Oracle's reward remain unknown, so the J-aware-AR hypothesis
  stays open.

## D. Instance and tooling (open; none affects a number)

- **T1.** The `GITHUB_TOKEN` environment variable is stale and overrides the working `gh` login. Clone and push with
  `env -u GITHUB_TOKEN` (`CLAUDE.md` already says this for pushes).
- **T2.** `scripts/000_setup/rebuild_env.sh` writes its hash record into a tracked file
  (`results/000_setup/analysis/artifact_hashes.json`). So every rebuild dirties the tree. I ran a copy that writes to
  the scratchpad.
- **T3.** `restore_outputs.py` exits 1. It restored 7,575 of 7,576 files; the exception is
  `outputs/007_jsubspace/stage1a/r2/qualitative_oracle_L44.jsonl`, which is 94,111 bytes on HF against 111,305 in the
  manifest. The mirror holds a different version of that file. It holds qualitative samples only.
- **T4.** `rebuild_env.sh` does not clone Jan Chen's phrase-lens code (`/workspace/sources/ext/phrase-lens-src`). So
  012's random-target construction can't be checked on a fresh instance.

## E. My own mistakes (corrected)

- **M1.** Report 39's first version said I7 was "12 cells". It is 14 numbers in 13 cells, plus 2 bullets. Fixed in
  report 39.
- **M2.** The same version gave Python's `round` as the cause of the I7 slips. I had not verified that, so the draft
  correction states only what the values were.
- **M3.** In my report-back on the patch I gave the CPU tests as "140 passed". The run gives 104 passed and 29 GPU
  tests deselected, as report 39's header already said. Corrected in reports 39 and 40.

M1–M3 are also in the `CLAUDE.md` mistakes log.

## Decisions (PI, 2026-10-10) and what stays open

- **P5:** yes. The 10-09 acceptance (L36–60, directions only) meets 011's precondition, and the 011 lines now cite it.
- **Reports 39 and 40** are committed to `claude_ovarida/`.
- **The patch** is applied with P1–P8 fixed (`d4b9860`). The I1/I7/I11/I12 drafts, I6, L1 and N1 are in this commit.
  `STATUS.md:31` is left as is.
- **E0 JSON:** not edited. The ledger and `CLAUDE.md` carry the acceptance.
- **Still open:**
  - the rules hunk (I13);
  - whether dropping SAEs and the skip lens from scope is accepted;
  - the next step (my recommendation: a J-independent class-mean lever on multihop, S1);
  - the page change from the patch list (lever split in `make_page.py` section 1), which is not done, and no page
    rebuild has been run.
