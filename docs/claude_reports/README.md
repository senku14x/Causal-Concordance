# claude_reports — the agents' own notes and reports

This folder holds what the Claude sessions wrote **for the PI**: reading notes, session reports, analysis notes and
proposals. The project's canonical documents live elsewhere and are not copied here:
- experiment reports: `docs/reports/`;
- designs: `docs/design_specs/`;
- spec and context: `docs/context/`;
- reviews: `docs/reviews/`.

Cleaned 2026-10-06:
- 13 byte-identical copies of canonical documents were removed (table at the end);
- superseded drafts were moved to `archive/`;
- file numbers were kept, so existing links still work.

## Start here

| File | What it is |
|---|---|
| `35_new_session_prompt_2026-10-07.md` | **The prompt for the next session** (after 013 and 014): read the paper, the notes and every experiment 001–014, then write up-to-speed report 36 (paper, notes, per-experiment table, current beliefs, inconsistencies, open threads, questions) for the PI to vet |
| `34_up_to_speed_2026-10-07.md` | **Up-to-speed report after 012** (new session, 2026-10-07): 012 summary, stage table with every number from JSON (half-up, keys cited), audit corrections, 19 stale/inconsistent items with file:line, next steps with GPU estimates, questions |
| `33_new_session_prompt_2026-10-07.md` | **The prompt for the next session** (after 012): read order (paper → notes → CLAUDE.md → 012 ledger/spec/audits), 012 state after the audit corrections, next steps (NLA reconstructor + hallucination test unblocked; content labelling; better prompt family), environment, working rules |
| `32_012_outlook_2026-10-06.md` | **012 outlook**: what phrase-lens contains (checked), whether PJ directions are needed, porting 008–011 to its multihop banks, fixes to the prompt set and the 012 draft, recommended order |
| `31_onboarding_review_2026-10-06.md` | **Onboarding review** (new session, 2026-10-06): 001–011 checked against JSON; the dissociation uses mismatched J objects; the AR config hints at a J-lens loss term; number slips (008 J_CB −0.14, not −0.15); ranked next steps |
| `30_new_session_prompt_2026-10-06.md` | **The prompt for the next session** (end of 2026-10-06): read order (paper → paper notes → docs), where 001–011 stand, pending decisions, environment, working rules |
| `26_up_to_speed_report_2026-10-06.md` | State of the project on 2026-10-06 (before 012; superseded by 34): instance check, what each experiment showed, weak links, every inconsistency found, next steps |
| `28_next_direction_proposal_2026-10-06.md` | The proposal behind tonight's work. Superseded in part by the PI's changes in `docs/design_specs/008_009_trackb_brewcode.md` |
| `docs/reports/008_trackb.md` (not here) | 008 result (before 012): 007's model-side test done the paper's way; no J privilege |
| `docs/reports/009_brewcode_derisk.md` (not here) | **009 de-risk:** the brew-code bank's gate passes at depth 1 (771 pairs) but the output-facing check fails: with a code answer the hidden colour is still the top colour (98%) and ranks ~11th at the answer slot, at ~0.05% total probability; readers blocked by the rule; options for the PI |
| `29_reader_agreement_010_2026-10-06.md` | 010 analysis: readers agree weakly; mid-layer readers track what the J lens reads, not the model's next-token output |
| `docs/design_specs/011_olens_swaps.md` (not here) | **011, Oracle-space swaps** (post hoc): loader validated after two fixes; smoke: the TA's whitened swap does nothing, the additive AR difference does about half of the class-mean's; full run stopped at 74/82 for the PI (status at the end of the note) |

## The paper and the reading

| File | What it is |
|---|---|
| `03_workspace_paper_notes_merged.md` | The agent's notes on the workspace (J-lens) paper merged with the PI's; §14 corrects the PI's notes against their sources, §15 lists where the agent's reading differs from the PI's (neither corrects the paper; corrected 2026-10-10) |
| `04_user_workspace_paper_notes_verbatim.md` | The PI's own notes on the paper, verbatim |
| `04b_user_workspace_paper_notes_combined_2026-10-05.md` | The PI's combined notes (paper + follow-up literature), 2026-10-05 |

Reading notes on the readers' papers: `docs/context/reading_notes.md` and `reading_notes_v2.md`.

## Session reports (newest first)

| File | Session | What it is |
|---|---|---|
| `../Causal-Concordance/claude_ovarida/45_015_r1_2026-10-11.md` (link) | 2026-10-11 | 015 step 2 R1 on four cells |
| `../Causal-Concordance/claude_ovarida/46_015_smoke_2026-10-11.md` (link) | 2026-10-11 | 015 step 3 smoke: gates, answer arm, rewrite arm, PI decisions |
| `../Causal-Concordance/claude_ovarida/47_015_full_2026-10-11.md` (link) | 2026-10-11 | 015 full run: answer arm, primary, rewrites, reader contrasts |
| `../Causal-Concordance/claude_ovarida/48_015_followups_2026-10-11.md` (link) | 2026-10-11 | 015 follow-ups: corrections, swap test, runner-up test |
| `../Causal-Concordance/claude_ovarida/49_clamp_check_2026-10-11.md` (link) | 2026-10-11 | 012E §18 clamp mechanism check |
| `../Causal-Concordance/claude_ovarida/44_stageE_energy_controls_2026-10-10.md` (link) | 2026-10-10 | 012 Stage E §17 energy controls and overlaps |
| `../Causal-Concordance/claude_ovarida/43_stageE_full_2026-10-10.md` (link) | 2026-10-10 | 012 Stage E full multihop run |
| `../Causal-Concordance/claude_ovarida/42_stageE_step4_smoke_2026-10-10.md` (link) | 2026-10-10 | 012 Stage E step 4: parity, floors, lever, smoke |
| `../Causal-Concordance/claude_ovarida/41_stageE_step2_2026-10-10.md` (link) | 2026-10-10 | 012 Stage E step 2 report: gate, pre-checks, decisions |
| `../Causal-Concordance/claude_ovarida/40_problems_found_2026-10-10.md` (link) | 2026-10-10 | Every problem found in report 39 and in the 10-09 patch, what the patch gets right, each item's status after `d4b9860` / `128bd12` |
| `../Causal-Concordance/claude_ovarida/39_up_to_speed_2026-10-10.md` (link) | 2026-10-10 | Up-to-speed report on a fresh instance (001–014, beliefs, patch check, 14 inconsistencies) |
| `/workspace/claude_suggestions/2026-10-10_what_changed.md` (link) | 2026-10-10 | Plain summary of what changed and what was done on 2026-10-10 |
| `26_up_to_speed_report_2026-10-06.md` | 2026-10-06 | Up-to-speed report; inconsistencies (multihop restatement 0.15–0.28, Oracle Lens provenance, nla-rl AR) |
| `21_up_to_speed_report_2026-10-05.md` | 2026-10-05 | Comprehension report on a fresh instance (001–006b, Cathryn's CW-12, problems found) |
| `12_what_was_done_and_why.md` | 2026-10-04 | Plain-language account of the day |
| `10_report_back_2026-10-04.md` | 2026-10-04 | Report back on the PI's feedback note |
| `05_doubts_and_open_questions.md` | 2026-10-04 | What could not be verified; questions about intent |

## Analysis notes by experiment

| File | Experiment | What it is |
|---|---|---|
| `29_reader_agreement_010_2026-10-06.md` | 010 (CPU, exploratory) | Reader agreement on 006/007; Oracle L44 vs NLA L42 on multihop |
| `27_qualitative_readouts_2026-10-06.md` | 007 (CPU, exploratory) | What the readers actually say: word-match analysis of saved readouts |
| `25_nla_stops_and_open_issues_2026-10-06.md` | 007 | Why the NLA runs kept stopping (bf16 batch-layout noise); every open issue |
| `22_007_geometry_cpu_2026-10-05.md` | 007 (CPU, exploratory) | J-subspace geometry: bases, overlaps, what the split does to brew activations |

**Experiment reports themselves:** `docs/reports/`:
- 001–004;
- 005 (smoke, pilot, full run, follow-ups);
- 006;
- 007 (`007_jsubspace.md`);
- 008 (`008_trackb.md`);
- the 10-04 checks.

## Other folders

- `advisor/JBASIS_CHECKS_2026-10-05.md`: the advisor's J-basis checks (input to 007).
- `archive/`: superseded drafts, kept for the record.

  | File | Superseded by |
  |---|---|
  | `06_next_steps_proposal.md` | the 005 design (one run) |
  | `09_design_005_planted_concepts_DRAFT.md` | `docs/design_specs/005_planted_concepts.md` (2 lines differ) |
  | `15_005_amendment_01_DRAFT.md` | `docs/design_specs/005_amendment_01.md` (adopted version) |
  | `20_006_reader_likelihood.md` | `docs/reports/006_reader_likelihood.md` (the full report, incl. 006b) |

## Removed copies (identical to the canonical file; still in git history)

| Old file | Canonical file |
|---|---|
| `01_SPEC_v2_review_2026-10-04.md` | `docs/context/SPEC_v2_review_2026-10-04.md` |
| `02_reading_notes_v2_NLA_Viswanath_AO_WSB.md` | `docs/context/reading_notes_v2.md` |
| `07_checks_C1_C2_C3_results.md` | `docs/reports/checks_2026-10-04_no_gpu.md` |
| `08_SPEC_v2_errata_01_ACCEPTED.md` | `docs/context/SPEC_v2_errata_01.md` |
| `11_PI_review_disposition.md` | `docs/reviews/2026-10-04_pi_review_checks_005/README.md` |
| `13_005_smoke_note.md` | `docs/reports/005_smoke_note.md` |
| `14_005_pilot_note.md` | `docs/reports/005_pilot_note.md` |
| `16_005_full_run.md` | `docs/reports/005_full_run.md` |
| `17_005_full_run_spec.md` | `docs/design_specs/005_full_run.md` |
| `18_005_followups.md` | `docs/reports/005_followups.md` |
| `19_005_followups_spec.md` | `docs/design_specs/005_followups.md` |
| `23_007_jsubspace_spec_v2.md` | `docs/design_specs/007_jsubspace.md` |
| `24_007_report_2026-10-05.md` | `docs/reports/007_jsubspace.md` |

## Conventions for new files

- Numbered filenames continue from 31, with a date suffix.
- Add a row to the right section above.
- **Don't copy canonical documents here; link them.** Copies drift.
- The scratch copy outside the repo is `/workspace/claude_reports/`. It also holds the PI's
  `NEW_SESSION_PROMPT_2026-10-06.md`.
