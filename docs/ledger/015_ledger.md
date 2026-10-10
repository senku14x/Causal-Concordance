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
