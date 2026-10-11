# What changed on 2026-10-10, and what I did

This is a summary for you, kept outside the repo. The details are in `claude_ovarida/39_up_to_speed_2026-10-10.md` and
`claude_ovarida/40_problems_found_2026-10-10.md`. Every number here comes from the committed JSON, rounded half-up.

## In short

- Two commits on `014-typo`, both pushed. The remote branch went from `b8d458e` to `128bd12`.
  - **`d4b9860`**: your 10-09 patch, with the 8 problems I found in it fixed (P1–P8).
  - **`128bd12`**: my reports 39 and 40, plus 7 doc corrections (I1, I7, I11, I12, I6, L1, N1).
- **Tests:** `pytest scripts/common/tests -m "not gpu"` gives 104 passed, 29 deselected, after each commit.
- **Nothing else was run:** no GPU, no model loaded, no experiment. Cathryn's repo was only read.
- **The one change of substance:** the 012 lever is now described as working "mainly where the bridge is a named
  place" (a country or a geographic place), not "a country". This grouping is post hoc. Everything else is
  corrections to wording and rounding.

## What I did, in order

1. **Came up to speed** on a fresh instance.
   - I cloned the repo at `b8d458e` and Cathryn's repo read-only.
   - I rebuilt the environment from a copy of `rebuild_env.sh` that doesn't write into the repo. All 40 pinned hashes
     match.
   - I restored `outputs/` from HF: 7,575 of 7,576 files match. The exception is one qualitative 007 file whose HF
     copy differs from the manifest.
   - I wrote report 39: the paper, your notes, 001–014, current beliefs, and 14 inconsistencies with 5 scientific
     concerns.
2. **Checked the 10-09 patch** in a scratch clone (`/workspace/scratch/cc_check/`), not in the repo.
   - The scripts reproduce your bundle's JSON byte for byte, and `--check` passes.
   - Every number matched its JSON except one. The check also turned up 7 wording problems, P2–P8.
3. **Drafted corrections** for I1, I7, I11 and I12 in a second scratch clone, after you confirmed I1 and I12.
4. **Wrote report 40.**
   - It lists every problem in one place, with its location, evidence and status.
   - Section A0 is what the patch gets right: each number re-checked against its JSON source and key.
5. **With your go:** applied everything, ran the tests, made two commits and pushed them.

## What changed in the repo

### Commit `d4b9860`: the 10-09 patch with P1–P8 fixed (10 files)

**Your patch, as written:** 29 edits to `CLAUDE.md`, `STATUS.md`, `003_labels.md` and `008_trackb.md`, appends to the
012 and 013 ledgers, and 2 mistakes-log rows. It also adds two scripts and their JSON:
- `scripts/012_reader_concordance/stageA_splits.py` → `results/012_reader_concordance/analysis/stageA_splits.json`;
- `scripts/013_nla_hallucination/rewrite_by_state.py` → `results/013_nla_hallucination/analysis/rewrite_by_state.json`.
  This JSON is byte-identical to your bundle's.

**My fixes on top:**

| Fix | Where | Before | After |
|---|---|---|---|
| P1 | 012 ledger | CW-20 bullet by bullet "+0.22" | +0.21 (her STATUS; her `analysis.txt` gives +0.209; 0.22 is the published value). Also: her one-text +0.02 equals her other-prompt control |
| P2 | `CLAUDE.md` results-page line | ended with the old "It covers 012, Stage L and 013," | clause removed |
| P3 | `CLAUDE.md` 013 block | the CW-16b parenthetical sat after "the FVE baselines differ" | moved next to CW-16b |
| P4 | `CLAUDE.md` E0 | CW-20 checks "agree" | "are consistent with", plus why: different inputs, and a corpus-centred score she chose post hoc |
| P5 | `CLAUDE.md` read order, 011 block and open decisions; `STATUS.md` 011 | "not validated … any next step needs the loader validated first" | cites the 10-09 acceptance (L36–60, directions only), which you ruled meets 011's precondition |
| P6 | `STATUS.md` 007 line | kept "Post hoc NLA run … is the PI's call" after "(Superseded …)" | sentence removed |
| P7 | `stageA_splits.py`, its JSON, `CLAUDE.md`, `STATUS.md`, 012 ledger | "mainly where the bridge is a country" | "mainly where the bridge is a named place" (post hoc; numbers below) |
| P8 | `CLAUDE.md` 006 line | 1.87× / 1.84× cited as if it were a `posthoc.json` value | labelled a ratio; the stored key is `think_minus_colon`, +2.79 / +2.73 nats |

**The P7 numbers** (`stageA_splits.json` `by_bridge`; the J swap with the random arm in brackets):

| Bridge | L24-35 | L24-47 |
|---|---|---|
| named place (49) | 37/49 (1/49) | 44/49 (5/49) |
| of which a country (42) | 33/42 | 38/42 |
| of which a geographic place, not a country (7) | 4/7 | 6/7 |
| not a named place (33) | 1/33 (0/33) | 4/33 (0/33) |

- The 7 places are Africa, Antarctica, Chicago, Everest, Nile, Olympus and Sydney (`GEO_PLACES`). I chose them after
  seeing the outcomes, so the grouping is post hoc.
- With only 7 non-country places, "place" can't be separated from "country".
- The bundle's original values are unchanged; the JSON only gains keys. The parity asserts still pass.
- I also added a mistakes-log row for the patch's own errors.

### Commit `128bd12`: reports and corrections (13 files)

**New files:**
- `claude_ovarida/39_up_to_speed_2026-10-10.md`;
- `claude_ovarida/40_problems_found_2026-10-10.md`.

Both are listed in `claude_ovarida/README.md`.

**Corrections:**

| Item | What was wrong | Fixed in |
|---|---|---|
| I1 | 03 §14–§15 were described as corrections to and disagreements with the paper. §14 corrects your notes, and §15 is where my reading differs from yours | `CLAUDE.md`, `docs/claude_reports/README.md`, report 35 (where it started) |
| I7 | 002's §3 table had 14 numbers (in 13 cells) rounded down from values ending in 5, and 2 bullets had the same slips (42% → 43%, 0.00–0.07 → 0.00–0.08). Report 31 had listed 4 | `docs/reports/002_claim_scoring.md`, with a dated note |
| I11 | Fig. 8's prose numbers (88 / 59 / 5 / 0) were used as the figure. The bars read about 87 / 55 / 9 / 1 (my reading, ±2). Fig. 16 also has a 54% bar the prose omits | 03 §4.1 and a new §14 item 7, and report 36's summary. 04 is verbatim and left alone |
| I12 | Typo was counted as "a second family" for the Oracle-vs-NLA J difference. On multihop, on the same J basis, the NLA also loses the step without J: 007 test half, naming 0.80 → 0.225 vs the Oracle's 0.85 → 0.425 and 0.8625 → 0.125; CW-14 J − rest +0.65 vs +0.71. You confirmed it holds on brew `:` and typo only | report 38, the claude_ovarida README, `CLAUDE.md` 014 block, `STATUS.md`, 014 ledger (appended), page text in `make_page.py` |
| I6 | `STATUS.md` still called the AR output scale "global" and "unexplained". It is per layer (0.11 at L20 → 2.33 at L63) and expected under the scale-invariant loss | `STATUS.md`. `CLAUDE.md`'s 011 block points at the 011 spec lines it supersedes; the spec is not edited |
| L1 | `STATUS.md` still said "the Oracle's 2×" | 1.87× / 1.84×, cited as a ratio |
| N1 | 002 said the WSB-grid Oracle's role-correct rate is "0–0.10 at every layer". That holds at `:`, but `</think>` L60 is 0.20 (n = 10) | `docs/reports/002_claim_scoring.md`, scoped to `:` |

I also added a mistakes-log row for my own errors (M1–M3 below).

## What I deliberately did not change

- **The E0 JSON:** not edited, as you said. The ledger and `CLAUDE.md` carry the acceptance.
- **Specs:** not edited (they are pre-registrations). That includes the 011 spec's old scale wording, which
  `CLAUDE.md` now flags.
- **04** (your verbatim notes): not edited.
- **`STATUS.md:31`** ("Page v12", dated): left as is, as you said.
- **The results page:** not rebuilt. The `make_page.py` text changed (I12), but the built page and the live artifact
  didn't. Your patch list's page change (adding the lever split to section 1) is not done.
- **Cathryn's repo:** only read, never written.

## Mistakes

**In the patch, caught before commit:**
- the +0.22 (P1);
- "agree" (P4);
- the country split that hid 7 places (P7);
- a stale clause (P2), a misplaced parenthetical (P3), a leftover sentence (P6), and a derived ratio cited as a key
  (P8).

**Mine:**
- **M1.** Report 39's first draft said 12 cells for I7. It is 14 numbers in 13 cells.
- **M2.** The same draft named a cause for those slips (Python's `round`) that I hadn't checked.
- **M3.** I told you the tests gave "140 passed". They give 104 passed, 29 deselected.

All three are corrected and logged in `CLAUDE.md`.

## Still open

- **I13:** your uncommitted "rules hunk" for `CLAUDE.md`. It is lost unless you have it.
- **I10:** Ratnaditya's CI differs between 03 and 04b. The source isn't in the repo.
- **Scope:** whether dropping SAEs and the skip lens is accepted for the writeup.
- **Next step.** My recommendation is a J-independent class-mean lever on multihop (S1 in report 40: the J-row-swap
  labels can't rank J-type readers independently).
- **The AR's J term** (`loss_mix_lambda` 0.25, `val_jfve`) and the Oracle's reward are still unanswered.
- **The results page:** the lever-split change, a rebuild, and publishing are yours.
- **Instance issues:**
  - the stale `GITHUB_TOKEN`;
  - `rebuild_env.sh` writes a tracked file;
  - one restore file mismatches its manifest;
  - Jan Chen's phrase-lens code isn't cloned by the rebuild.

## Where everything is

| What | Where |
|---|---|
| The two reports | `claude_ovarida/39_up_to_speed_2026-10-10.md`, `claude_ovarida/40_problems_found_2026-10-10.md` |
| Full diffs | `git show d4b9860`, `git show 128bd12`; saved as `/workspace/scratch/cc_check/commit_a_d4b9860.diff` and `commit_b_128bd12.diff` |
| The edit scripts I used (each asserts that every find occurs once) | `/workspace/scratch/cc_check/fixups_a_2026-10-10.py` (P1–P8), `drafts_2026-10-10.py` (I1/I7/I11/I12), `fixups_b_2026-10-10.py` (I6/L1/N1) |
| Scratch clones used for checking | `/workspace/scratch/cc_check/repo` (patch applied), `/workspace/scratch/cc_check/drafts_repo` (patch plus drafts) |
