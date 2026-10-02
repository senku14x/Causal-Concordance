# Causal-Concordance

Do interpretability readers agree on what the model actually uses?

We run activation readers — the logit lens, Jacobian lenses (J_NP, J_CB, R_CB), the Oracle Lens and a natural-language
autoencoder (NLA) — on the same cells of **Qwen3.6-27B**, and score each against **reader-blind causal labels**: does the
content a reader names have demonstrated influence on a specified answer at that cell, is it decodable there but not
needed, absent, or unresolved? The labels then split reader agreement into shared signal, shared error and shared prompt
inference. Prompts come from **WorkspaceBench** (brew, chain, relational) with token-aligned counterfactual partners.

Status: setup (block 000) done; design 001 (reader-free leverage profile) approved, its gate and smoke stages run on dev material. No label computed yet. See [`STATUS.md`](STATUS.md).

## Layout

```
CLAUDE.md                 agent/collaborator guide: read order, non-negotiables, conventions, environment, pitfalls
STATUS.md                 where things stand (environment, reader smoke tests, tests, regime check, open questions)
docs/context/             SPEC.md (project spec), AMENDMENT_01.md (decisions + every spec conflict), READING_LIST.md, notes
docs/design_specs/        NNN_<experiment>.md — written and approved before each run
docs/reports/             NNN_<experiment>.md — results; 000 = reading report
docs/papers/              paper index (PDFs local only)
scripts/common/           shared label code (registry, cells, wsb, partners, splits, margin, engine, regime, gates, stats),
                          reader code (readers.py; never imported by label code), figures; tests/ (pytest)
scripts/000_setup/        environment rebuild, artifact checks, env/reader/regime/timing smoke scripts
scripts/001_leverage_profile/  design 001 pipeline (gate, smoke, run, probes, snapshot, analysis, figures)
results/NNN_<experiment>/ analysis/*.json tables, figures/, manifests
outputs/                  raw arrays and logs (gitignored)
```

## Quickstart

```bash
bash scripts/000_setup/rebuild_env.sh            # pinned stack, source clones, model + lens downloads, hash checks
python -m pytest scripts/common/tests -q         # 78 tests (the GPU ones load the 27B model once; -m "not gpu" for the CPU subset)
```

Requires one GPU with ≥ 80 GB (developed on an RTX PRO 6000 Blackwell, 96 GB).
