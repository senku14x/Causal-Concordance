---
name: workspace-paper-notes
description: "Where my detailed notes on Anthropic's \"global workspace\" / Jacobian-lens paper (incl. template lens & oracle lens) live, plus code/lens resources"
metadata:
  node_type: memory
  type: reference
  originSessionId: afa9f8f2-949b-4d3f-871d-9476bd46a4d3
  modified: 2026-10-04T16:39:03.530Z
---

Paper: Gurnee, Sofroniew, … Lindsey, "Verbalizable Representations Form a Global Workspace in Language Models" (Anthropic, Transformer Circuits, July 2026; arXiv 2607.15495). The user calls it "the workspace paper".

- My full notes, written 2026-10-04 after reading the whole paper and appendix: `/workspace/notes/workspace_paper_notes.md`. They cover the J-lens method, the §3–§7 experiments with numbers, template lens (A.9.1), oracle lens (A.9.2), my critiques, and a replication checklist.
- The user's own notes are kept verbatim at `/workspace/notes/user_workspace_paper_notes.md`. They were merged into mine with attribution; mine §13 covers follow-up literature, §14 corrections to their transcriptions, §15 open disagreements. Their §11 metric catalogue is the reference checklist.
- Follow-up LW posts (Nanda review, R-lens, meta-tokens, WorkspaceBench, token-frequency z-scoring, GPT-2 replication, open-weight multihop replication) are saved as text in `/workspace/papers/lw/`. Key takeaway: Claude-level swap results have NOT replicated on Qwen/Gemma (6–11% flips).
- PDF + text: `/workspace/papers/workspace_arxiv.{pdf,txt}`. Official code: `/workspace/papers/jacobian-lens` (github anthropics/jacobian-lens; no template/oracle code).
- Pre-fitted J-lenses for ~40 open models: HF `neuronpedia/jacobian-lens`. Third-party template lenses for Qwen3.6-27B: HF `ec75hash/template-lens-qwen3.6-27b`, `camilablank/workspace-lenses` (unverified).

/workspace is not persistent, so re-download these if they're missing. Related: [[activation-readers-project]].
