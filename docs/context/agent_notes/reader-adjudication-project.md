---
name: reader-adjudication-project
description: "State of the Causal-Concordance project (readers vs reader-blind causal labels on Qwen3.6-27B, WorkspaceBench only) — where everything lives and what comes next"
metadata:
  node_type: memory
  type: project
  originSessionId: 0f5258a9-0567-4d36-8e23-ce24b09256c7
  modified: 2026-10-02T11:55:32.820Z
---

Project (started 2026-10-01): score activation readers (logit lens, J_NP, J_CB, R_CB, Oracle Lens, NLA; SAE deferred) on Qwen3.6-27B against reader-blind causal labels and decompose their agreement. WorkspaceBench only (brew, chain; relational later).

As of 2026-10-02 (overnight block done):
- Repo /workspace/Causal-Concordance (github senku14x/Causal-Concordance, private). Work on setup/000-scaffold (head 2fd96c0 as of 2026-10-03); main still at dc33f48 — ask before fast-forwarding main again.
- Handoff docs are the source of truth: docs/COME_UP_TO_SPEED.md, STATUS.md, docs/context/SPEC_STATUS.md, CLAUDE.md, docs/reviews/ (3 external reviews + the 001 audit).
- 001 done, audited, probes repaired (amendment A2): membership decodable at ':' L40–60 (L40–44 narrowly). 002 (claim scoring, judge Gemini 3.8 Flash via OpenRouter, key in /workspace/.env) done: verbalizers rarely claim z in role; J lenses track z within the candidate set (47–48/48 tables). No causal labels yet.
- 003 (reader-free labels) done 2026-10-02 (head 40a9393): z at ':' L40–44 has a small consistent influence (median S 0.03–0.05, used in 10–18% of pairs); z decodable-not-needed at </think>; label reliability gate G4 FAILED (κ 0.37) → labels descriptive. External audit of 001–003 accepted (head aa229ae): L4 actually true (87–93%), 'other' attribution and random-clamp specificity withdrawn, lens tracking ≠ rank flips (90/183). 2026-10-03: design 004 v2 (span assay) implemented, smoke reviewed externally (fixes: S5 rule, G1 enforcement, fidelity resolution; design §13), run on PI go: 33 pairs, head 2fd96c0. Result: joint span recovers 0.72 of the ':' L42 whole cell, z span 0.21, 1-D 0.06, but same-rank PCA does as well or better → descriptive geometry result, no intervention nominated; report docs/reports/004_span_assay.md §14 lists next options (default: back to the reader panel). Handoff framing (reader-configuration comparison primary, causal work a validation branch) still awaits PI confirmation. Raw outputs mirrored to the PUBLIC HF dataset senku21x/causal-concordance-outputs with MANIFEST.json (82 files, 264 MB at 2fd96c0). User should rotate the OpenRouter key (pasted in chat).

**Why:** the next session must start from the handoff docs, not redo setup, 001 or the audit.

**How to apply:** read COME_UP_TO_SPEED → STATUS → SPEC_STATUS; never launch a run without the user's explicit go. Related: [[research-workflow-non-negotiables]], [[github-commit-setup]], [[vllm-optional-later]], [[keep-qualitative-samples]], [[page-charts-native-style]].
