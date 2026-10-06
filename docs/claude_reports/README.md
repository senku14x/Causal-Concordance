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
| `26_up_to_speed_report_2026-10-06.md` | **Current state of the project** (2026-10-06): instance check, what each experiment showed, weak links, every inconsistency found, next steps |
| `28_next_direction_proposal_2026-10-06.md` | The proposal behind tonight's work. Superseded in part by the PI's changes in `docs/design_specs/008_009_trackb_brewcode.md` |
| `docs/reports/008_trackb.md` (not here) | **Newest result:** 007's model-side test done the paper's way; no J privilege |
| `docs/reports/009_brewcode_derisk.md` (not here) | **009 de-risk:** the brew-code bank's gate passes at depth 1 (771 pairs) but the output-facing check fails: with a code answer the hidden colour is still the top colour (98%) and ranks ~11th at the answer slot, at ~0.05% total probability; readers blocked by the rule; options for the PI |
| `29_reader_agreement_010_2026-10-06.md` | **Newest analysis:** readers agree weakly; mid-layer readers track what the J lens reads, not the model's next-token output |

## The paper and the reading

| File | What it is |
|---|---|
| `03_workspace_paper_notes_merged.md` | The agent's notes on the workspace (J-lens) paper merged with the PI's; corrections in §14, disagreements in §15 |
| `04_user_workspace_paper_notes_verbatim.md` | The PI's own notes on the paper, verbatim |
| `04b_user_workspace_paper_notes_combined_2026-10-05.md` | The PI's combined notes (paper + follow-up literature), 2026-10-05 |

Reading notes on the readers' papers: `docs/context/reading_notes.md` and `reading_notes_v2.md`.

## Session reports (newest first)

| File | Session | What it is |
|---|---|---|
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

- Numbered filenames continue from 30, with a date suffix.
- Add a row to the right section above.
- **Don't copy canonical documents here; link them.** Copies drift.
- The scratch copy outside the repo is `/workspace/claude_reports/`. It also holds the PI's
  `NEW_SESSION_PROMPT_2026-10-06.md`.
