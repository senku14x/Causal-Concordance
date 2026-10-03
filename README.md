# Causal-Concordance

Do interpretability readers agree on what the model actually uses?

We run activation readers — the logit lens, Jacobian lenses (J_NP, J_CB, R_CB), the Oracle Lens and a natural-language
autoencoder (NLA) — on the same cells of **Qwen3.6-27B**, and score each against **reader-blind causal labels**: does the
content a reader names have demonstrated influence on a specified answer at that cell, is it decodable there but not
needed, absent, or unresolved? The labels then split reader agreement into shared signal, shared error and shared prompt
inference. Prompts come from **WorkspaceBench** (brew, chain, relational) with token-aligned counterfactual partners.

Status (2026-10-03): experiments 001–004 are complete on the dev split and await PI review.
- 001: leverage profile, probes, reader snapshot.
- 002: role-aware reader claim scoring.
- 003: reader-free 1-D labels (descriptive).
- 004: span assay; the fitted spans recover no more than same-rank PCA.

No test-split measurement yet. New here? Start with [`CLAUDE.md`](CLAUDE.md) (rules, preferences, fresh-instance
restore), then [`docs/COME_UP_TO_SPEED.md`](docs/COME_UP_TO_SPEED.md), [`STATUS.md`](STATUS.md) and
[`docs/context/SPEC_STATUS.md`](docs/context/SPEC_STATUS.md). The working branch is `setup/000-scaffold`.

## Layout

```
CLAUDE.md                 agent/collaborator guide: read order, non-negotiables, conventions, environment, pitfalls
STATUS.md                 where things stand (environment, reader smoke tests, tests, regime check, open questions)
docs/COME_UP_TO_SPEED.md  handoff: what exists, what 001 found, known problems, decisions log, practicalities
docs/context/             SPEC.md (project spec), SPEC_STATUS.md (done / not done, item by item), AMENDMENT_01.md
                          (decisions + every spec conflict), PI_DIRECTIVE_2026-10-02.md, READING_LIST.md, notes
docs/reviews/             every external review (as delivered, with the reviewer's scripts/JSON and a disposition README)
docs/context/agent_notes/ the PI's working preferences as kept in the agent's memory
docs/design_specs/        NNN_<experiment>.md — written and approved before each run
docs/reports/             NNN_<experiment>.md — results; 000 = reading report, 001–004 = experiments
docs/papers/              paper index (PDFs local only)
scripts/common/           shared label code (registry, cells, wsb, partners, splits, margin, engine, regime, gates, stats),
                          reader code (readers.py; never imported by label code), figures; tests/ (pytest)
scripts/000_setup/        environment rebuild, artifact checks, env/reader/regime/timing smoke scripts
scripts/000_setup/restore_outputs.py  restore outputs/ (public HF, sha256-checked) and the private backup
scripts/001_leverage_profile/  design 001 pipeline (gate, smoke, run, probes, snapshot, analysis, figures) and
                          audit_followups.py (post-audit recomputations from saved outputs)
scripts/002_claim_scoring/ … scripts/004_span_assay/  one pipeline per experiment (stages in each docstring)
results/NNN_<experiment>/ analysis/*.json tables, figures/, manifests
outputs/                  raw arrays and logs (gitignored; mirrored to HF dataset senku21x/causal-concordance-outputs)
```

## Quickstart

```bash
bash scripts/000_setup/rebuild_env.sh            # pinned stack, source clones, model + lens downloads, hash checks
python scripts/000_setup/restore_outputs.py     # raw outputs from HF, checked against MANIFEST.json
python -m pytest scripts/common/tests -q         # GPU tests load the 27B model once; -m "not gpu" for the 65 CPU tests
```

Requires one GPU with ≥ 80 GB (developed on an RTX PRO 6000 Blackwell, 96 GB).
