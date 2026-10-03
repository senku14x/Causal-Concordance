---
name: keep-qualitative-samples
description: "User wants qualitative samples (raw readouts, prompts, example cells) kept and shown in every result, report and results page — never aggregates alone"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0f5258a9-0567-4d36-8e23-ce24b09256c7
  modified: 2026-10-02T07:38:23.576Z
---

Always keep qualitative samples alongside quantitative results: raw reader outputs, the actual prompts, and a fixed-seed set of example cells/pairs read by hand, in reports, results JSON and results pages (user, 2026-10-02: "remember to keep qualitative samples always").

**Why:** the user inspects results by reading real examples; aggregate numbers alone hid what the readers actually said, and the project's planned first deliverable is a handful of inspectable, explained cases.

**How to apply:** every experiment stage saves raw samples (not just scores); every report and results page includes a qualitative section (fixed seed, stated selection rule) next to the tables. Related: [[reader-adjudication-project]].
