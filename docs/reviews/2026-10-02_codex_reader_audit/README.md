# External audit of the 001 pilot (Codex, 2026-10-02)

- `audit.md`: the audit as delivered (copied from `/workspace/codex_reports/2026-10-02_reader_audit.md`).
- `reader_audit.py`, `reader_audit_results.json`, `reproduction.log`, `source_manifest.json`: its analysis script,
  output, run log and the sha256 of every file it read.
- Not copied: its `evidence/` folder (12 MB): copies of this repo's own files (snapshot.json, probes.json, run
  outputs, code, the report) as they stood when audited. `source_manifest.json` pins them by sha256. Relative links
  in `audit.md` point into that folder.

Response: report 001 §15 (point-by-point table) and `scripts/001_leverage_profile/audit_followups.py`
(`results/001_leverage_profile/analysis/audit_followups.json`), which recomputes every number independently. All of the
audit's numbers matched.
