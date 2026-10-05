---
name: activation-readers-project
description: "Activation readers" = Causal-Concordance (senku14x) — comparing J/R/logit lens, Oracle Lens, NLA on Qwen3.6-27B; joint with Cathryn Wayman's nla-vs-lenses pilot
metadata:
  type: project
---

The project (started 2026-10-04 in this workspace) is **Causal-Concordance** (github senku14x/Causal-Concordance, PI = the user, Vishesh), joint with Cathryn Wayman's pilot LittleMeHere/nla-vs-lenses (read-only, never push). It compares activation readers on Qwen3.6-27B (J_NP, J_CB, R_CB, logit lens, Oracle Lens, NLA) on WorkspaceBench families.

The unified spec is `docs/context/SPEC_v2.md`. [v2] items in it are proposals until the PI accepts them. Its main lines:
- reader agreement vs measured causal influence;
- reading vs re-solving (A4/A7);
- design 005, planted J / non-J concepts.

The repo's CLAUDE.md holds the rules, environment, pitfalls and silly-mistakes log. Read it first.

Status (2026-10-04):
- The PI chose NLA = `ceselder/qwen3.6-27b-nla-rl`, default iter_000400 [PI: confirm]; loader `readers.NLArl` + registry `readers.nla_rl` (sha256-pinned). nla-L42 (a different run) stays for 001/002 reproducibility only.
- My review and the PI decision list are in `docs/context/SPEC_v2_review_2026-10-04.md`; paper notes are in `docs/context/reading_notes_v2.md`. Both are on branch `spec-v2` (not main).
- The PI made the repo **private** on 2026-10-04 (it had been public with the paper PDFs in it).
- The OpenRouter judge key is in `/workspace/.env` (not in any mirror; the PI pasted it in chat 2026-10-04, so it should be rotated).
- The official introspection protocol (prompt, prefills, 101 concepts) is in `/workspace/jacobian-lens/data/experiments/verbal-introspection.json` (anthropics/jacobian-lens).

Status (2026-10-05, end of session):
- Branch `005-amended-full-run` holds everything: amendment 01, the 005 full run, the follow-ups, and 006 (merged from `006-reader-likelihood`, fast-forward). Not merged into `main` (the PI merges).
- Key corrections: the 005 model-side "report" measure is lexical priming (A2b); 006: the NLA's likelihood tracks z at the brew `:` cell (T1); the Oracle tracks it 48/48 but failed its positive control at L40/L44.
- HF public dataset senku21x/causal-concordance-outputs mirrors all outputs (verified MANIFEST). The private dataset is full (free-plan limit).
- Pages: https://claude.ai/artifact/P35xhVkdzKjsdGhogp7PQY (main, v7, incl. 006) and https://claude.ai/artifact/5EGz2DxcAXXQs3wTtz1Lnu (full run).
- The PI is analysing the results; no experiment queued. Never stop the vast.ai instance without the PI's yes.

**Why:** standing context for every session on this project.
**How to apply:**
- No GPU runs or experiments without a committed design spec and the PI's explicit go.
- No subagents.
- Never push to main.
- Push with `env -u GITHUB_TOKEN git -c credential.helper='!gh auth git-credential' push`.
- /workspace is not persistent; rebuild with `scripts/000_setup/rebuild_env.sh`.

Related: [[user-research-profile]], [[workspace-paper-notes]].
