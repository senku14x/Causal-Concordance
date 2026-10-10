# claude_ovarida: this session's reports and analyses

Everything from the "ovarida" sessions (2026-10-07 onward) lives here, so the canonical folders (`docs/`, `results/`,
`scripts/`) aren't touched.

| File | What it is |
|---|---|
| `36_up_to_speed_2026-10-07.md` | Up-to-speed report after 013/014: the paper in my own words, your notes, one row per experiment 001–014 with JSON keys, current beliefs, 17 inconsistencies (file:line), open threads, questions. Moved here from `docs/claude_reports/` |
| `37_typo_energy_check_2026-10-07.md` | Is 014's J contrast just energy? CPU check; pre-set rule says no GPU follow-up; J holds about 2× random of the item-varying variance |
| `38_typo_content_control_2026-10-07.md` | 014's J contrast vs a content-matched (PCA-k) and random control (GPU ~15 min): the Oracle's J cost survives (D_PCA +0.11 / +0.21); Oracle − NLA survives; NLA unresolved. Brew `:` and typo only, not multihop (corrected 2026-10-10) |
| `39_up_to_speed_2026-10-10.md` | Up-to-speed report on a fresh instance (2026-10-10): the paper, your notes, 001–014 with JSON keys, current beliefs and Neel's questions, the check of the 10-09 patch (§5), 14 inconsistencies and 5 scientific concerns (§6) |
| `40_problems_found_2026-10-10.md` | Every problem found in 39 and in the 10-09 patch, with what the patch gets right (A0, all numbers re-checked against JSON) and each item's status after `d4b9860` and the commit that adds it |
| `41_stageE_step2_2026-10-10.md` | 012 Stage E step 2 (design 012E): phrases, prompts, AR vectors, gate and pre-checks. Brew stop rule fires (phrases worse than single words); multihop partly passes; AR differences J-heavy (0.49–0.62); decisions for the PI |
| `scripts/typo_content_control.py`, `results/typo_content_control*.json` | Script (rules in header, committed before the run) and results for 38; raw readouts in `outputs/claude_ovarida/014_content_control/` (mirrored to HF) |
| `scripts/typo_energy_check.py` | The CPU script for 37 (reads `outputs/014_typo/captures.pt`) |
| `results/typo_energy_check.json` | Its full-precision output |

**Results page v14 (built, not published).** Section 16b (the content-matched control) was added to
`scripts/012_reader_concordance/make_page.py`; the built page is `results/012_reader_concordance/page/index.html`
(render-checked at 1280 / 400 px, no overflow or script errors). Publishing to
https://claude.ai/artifact/NoevKvBkXNUTvtpnJPVsuq was refused from this session (version check; the artifact is owned
outside this session's org), so the PI publishes it.
