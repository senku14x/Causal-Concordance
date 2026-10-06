# 012 research ledger

Chronological record of what was run, why, what was found, and anything odd. UTC times. Newest at the bottom.
Spec: `docs/design_specs/012_reader_concordance.md` (v1 → v1.1 §11 → v1.2 §12). Every result here is EXPLORATORY.

## 2026-10-06

**22:20 — phrase-lens read.** Cloned `chen-jan/phrase-lens` @ ea85583, read-only, to `/workspace/sources/ext/phrase-lens-src`; Jan cleared its use.
- The banks are package data.
- All 96 ST items are WSB multihop items. The 67 MT items are Jan's own; 16 are flagged.
- No fitted tensors and no results are shipped.
- Outlook: report 32. Jan's writeup was read the same evening (report 32 §8).

**22:46 — Stage A smoke** (4 items, J swap L24–47, Jan's code unmodified).
- Banks built; 10 s per 4 items.
- Odd: `chem-bones-Z`'s clean prompt already answers y_b (" 26"), and Jan's code counted it as a success → rates now count clean-correct items only.
- The other 3 continuations were unchanged.
- My note explained them geometrically (near-parallel rows). **That was wrong**; see 23:10.

**22:56–23:08 — Stage A full** (8 runs, about 80 s each; `outputs/012_reader_concordance/stageA/`).
- Summary: `results/012_reader_concordance/analysis/stageA_summary.json`.
- Seeded continuation samples: `results/012_reader_concordance/qualitative/stageA_samples.jsonl`.
- **Clean-correct, n = 82:**

  | Arm | Success | Random |
  |---|---|---|
  | J swap L24–47 | 0.585 | 0.061 |
  | J swap L24–39 | 0.549 | 0.012 |
  | J swap L24–35 | 0.463 | 0.012 |
  | logit-row swap | 0.00 | — |

  J ablation survival 0.72 (random 1.00).
- Reproduces Jan's ~59% figure. The early partial (2 of 18) was the alphabetically first, harder items.

**23:10 — the independent review (Codex) and the PI's go.**
- Adopted:
  - the contamination rule plus clean-target-absent;
  - the render contract (captures from the exact labelled runs);
  - the reflection identity (my geometric note withdrawn);
  - withdrawal of the two invalid Stage C rules;
  - the d_r statistic;
  - a fixed cohort that keeps failures and controls;
  - a common downstream band for both readers (L24–35 works, so it is the primary band);
  - narrowed label wording;
  - per-target and pooled reporting.
- Not adopted tonight: per-cell mediation transfers; Q3/Q4 endpoints.
- Spec v1.2 written and committed before any B/C code ran.
