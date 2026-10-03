---
name: github-commit-setup
description: "How to commit and push for the user's repos on this Vast instance — commit email, account, and the invalid-GITHUB_TOKEN workaround"
metadata:
  node_type: memory
  type: reference
  originSessionId: 0f5258a9-0567-4d36-8e23-ce24b09256c7
  modified: 2026-10-02T04:33:23.072Z
---

- Commit email: visheshgupta14x@gmail.com (user stated 2026-10-02). GitHub account: senku14x. RHUT used author name "senku14x"; the Causal-Concordance initial commit (made on GitHub) shows "Vishesh Gupta". Repo-local config set to senku14x / visheshgupta14x@gmail.com.
- The image's GITHUB_TOKEN env var is invalid and shadows the working `gh` login (senku14x, scopes repo/gist/read:org). Use:
  `env -u GITHUB_TOKEN gh ...` and
  `env -u GITHUB_TOKEN git -c credential.helper='!gh auth git-credential' push origin main`
- Main project repo: https://github.com/senku14x/Causal-Concordance (private since 2026-10-02; cloned at /workspace/Causal-Concordance). Work happens on setup/000-scaffold; main was fast-forwarded to it on the user's request 2026-10-02 — still ask before pushing to main again. See [[reader-adjudication-project]].
- Commit/push only when the user asks (or the agreed workflow says so); end commit messages with the Claude co-author trailer.
