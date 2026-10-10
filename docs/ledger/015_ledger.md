# 015 ledger: the hallucination test on the Oracle Lens, with cross-scoring

Append-only. Spec: `docs/design_specs/015_oracle_hallucination.md`. Branch `015-oracle-hallucination`. EXPLORATORY.

## 2026-10-11: spec approved and committed

- The PI approved the spec with six changes, applied before commit:
  - R1 on all four cells;
  - 013's exact metric, with B stated;
  - reader-specific claims only for the pre-named contrasts;
  - the per-share requirement;
  - F's composition logged;
  - the NLA P redo as a post hoc arm, with the sign convention stated.
- B computed from the gold targets alone before any text was scored:
  - Oracle AR: 0.7032, so ±2 points = ±0.0070 cosine;
  - NLA AR: 0.2818, which reproduces 013's `baseline_B`, so ±2 points = ±0.0028 cosine.
- No judge call or reconstructor forward has run.

## 2026-10-11: step 2, R1 on all four cells (O versions, clean first samples, no judge calls)

- **Stop gate** (Oracle texts × Oracle AR): +88.6 [81.0, 96.1], retrieval@1 0.78. Pass.
- **Other cells:**
  - Oracle texts × NLA AR: +75.6 [66.9, 84.8], retrieval 0.68;
  - NLA texts × NLA AR: +130.4 [118.0, 142.9], retrieval 0.75;
  - NLA texts × Oracle AR: +9.7 [7.4, 12.0], retrieval 0.26.
  - All CIs exclude 0. The last cell is weak (cosine 0.067).
- **Determinism:** a true rerun is bitwise identical. Batch-layout noise per text: 90th pct 0.11, max 0.35 points.
- **G-repro (NLA side):** 1.9e-7. **CW-20 consistency:** +0.215, cos 0.466, scale 0.439 (hers +0.21, 0.46, 0.434).
- **Material fetched:** the NLA reconstructor (`ar_reconstructor/`, `nla_meta.yaml` at `5a13b7ec`) was downloaded; it
  was not on this instance.
- **Correction before commit:** my first "determinism" check compared two batch layouts. It is now the identical call
  twice, with the layout comparison reported separately.
- Report: `claude_ovarida/45_015_r1_2026-10-11.md`. No judge call has run.

