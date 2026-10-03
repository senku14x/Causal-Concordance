---
name: research-workflow-non-negotiables
description: "User's standing rules for interp research sessions — no subagents, design doc before any run, launch only on explicit go, reader-free labels, fp32 writes with post-write fidelity"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0f5258a9-0567-4d36-8e23-ce24b09256c7
  modified: 2026-10-02T03:56:54.547Z
---

For the reader-adjudication project (and the user's interp work generally):
- Never use subagents (Agent tool / workflows) — do the reading and work directly.
- Write a design doc (hypothesis, decision rule, stop condition; research_secant `designs/NNN-slug.md` convention) before any run, and launch only after the user explicitly says go.
- Causal labels must never use any reader (no J-lens swaps, no AR edits) for labelling or admission.
- Every direction-level write goes through float32 (hooks or true-fp32 replay), with realised-write fidelity (rho, kappa) measured after the write.
- When reading sources: primary sources only, record read status per source, say explicitly when something could not be accessed, separate what a source says from inference, cite section or file:line.

**Why:** earned in prior projects (RHUT bf16 writes realised only 0.53–0.59 of requested norm; J-based admission made two-hop rankings circular); subagents burn the user's usage limits.

**How to apply:** default to these without being asked; if a step seems to need a run, stop at the design doc and ask. Related: [[reader-adjudication-project]].

2026-10-02 update: the user turned on workflows/ultracode ("let me turn on workflows for you"), which lifts the no-subagents rule for orchestration (e.g. adversarial review of analyses and reports). Label measurements stay single-path code; still confirm if a use of subagents seems outside that intent.
