# claude_ovarida: this session's reports and analyses

Everything from the "ovarida" sessions (2026-10-07 onward) lives here, so the canonical folders (`docs/`, `results/`,
`scripts/`) aren't touched.

| File | What it is |
|---|---|
| `36_up_to_speed_2026-10-07.md` | Up-to-speed report after 013/014: the paper in my own words, your notes, one row per experiment 001–014 with JSON keys, current beliefs, 17 inconsistencies (file:line), open threads, questions. Moved here from `docs/claude_reports/` |
| `37_typo_energy_check_2026-10-07.md` | Is 014's J contrast just energy? CPU check; pre-set rule says no GPU follow-up; J holds about 2× random of the item-varying variance |
| `38_typo_content_control_2026-10-07.md` | 014's J contrast vs a content-matched (PCA-k) and random control (GPU ~15 min): the Oracle's J cost survives (D_PCA +0.11 / +0.21); Oracle − NLA survives; NLA unresolved |
| `scripts/typo_content_control.py`, `results/typo_content_control*.json` | Script (rules in header, committed before the run) and results for 38; raw readouts in `outputs/claude_ovarida/014_content_control/` (mirrored to HF) |
| `scripts/typo_energy_check.py` | The CPU script for 37 (reads `outputs/014_typo/captures.pt`) |
| `results/typo_energy_check.json` | Its full-precision output |
