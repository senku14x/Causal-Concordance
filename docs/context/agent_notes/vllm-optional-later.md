---
name: vllm-optional-later
description: User guidance — vLLM/vllm-metamodels acceleration is optional; use the normal HF route and only slot vLLM in once things work
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0f5258a9-0567-4d36-8e23-ce24b09256c7
  modified: 2026-10-02T06:06:44.412Z
---

Don't force vLLM (vllm-metamodels) into the pipeline. If it doesn't work, go the normal HF route and slot it in later, once everything works and the initial tests/experiments are done (user, 2026-10-02).

**Why:** the user wants progress on the science, not time spent debugging an accelerator; acceleration only matters for reader-generation scale-up.

**How to apply:** when vLLM breaks, record the exact error in STATUS.md and move on with HF; revisit only after the label pipeline and first experiments run. Labels never go through vLLM regardless. Related: [[reader-adjudication-project]].
