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

Status (2026-10-05, ~04:30 UTC):
- The PI adopted amendment 01 in full (incl. A18); the 005 full run on the 45 test concepts is done on branch `005-amended-full-run` (off spec-v2 @ 9709712; dropping the branch reverses it). Report `docs/reports/005_full_run.md`.
- Pages: main https://claude.ai/artifact/P35xhVkdzKjsdGhogp7PQY (v5); full run https://claude.ai/artifact/5EGz2DxcAXXQs3wTtz1Lnu.
- Open PI questions: merge the branch into spec-v2?; mirror ~1 GB raw outputs to the public HF dataset?; stop the instance?
- The PI asked me never to stop the vast.ai instance without an explicit yes.

**Why:** standing context for every session on this project.
**How to apply:**
- No GPU runs or experiments without a committed design spec and the PI's explicit go.
- No subagents.
- Never push to main.
- Push with `env -u GITHUB_TOKEN git -c credential.helper='!gh auth git-credential' push`.
- /workspace is not persistent; rebuild with `scripts/000_setup/rebuild_env.sh`.

Related: [[user-research-profile]], [[workspace-paper-notes]].
