# External review of the 004 smoke (Codex, 2026-10-03, at 98e8edf)

- `review.md`: the review as delivered; `audit_004_smoke.py` + `2026-10-03_004_smoke_audit.json`: its independent
  recomputation from the saved smoke rows (CPU only; no model or API calls).
- PI: "seems bullshit of a review". Checked point by point: its concrete claims were correct.

**Disposition (all before the 33-pair run; design 004 §13, commit 9da6b23):**
- **Accepted and fixed:**
  - the S5 definition: ≥ 0.5 recovery in r1 and r2, and median paired |ΔS| ≤ 0.05 in S units;
  - G1 is now enforced in the run path;
  - the fidelity exemption was narrowed from 1e-4 to the measured resolution of 1e-6, with an absolute read-back bound
    below it, exact zeros required to be no-ops, and an empty record failing;
  - 24 (not 25) spelling sequences;
  - no Pearson r with n = 2;
  - complement rank labels, the examples selection name, and "J" naming.
- **Correction (2026-10-03, after the run):** at the time I said the review's "agreed clarification" for S5 had no
  basis in the PI conversation. That was wrong. The consolidated handoff the PI pasted
  (`../2026-10-03_codex_004_consolidated_handoff/handoff.md`, reproducibility bullet) states the rule. I had also said
  the audit script and JSON were not supplied; they were in `/workspace/codex_reports/` and are now filed here.
