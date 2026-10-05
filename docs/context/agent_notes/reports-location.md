---
name: reports-location
description: "User wants every report, set of notes and my doubts saved under /workspace/claude_reports (with a README index), in addition to anywhere else"
metadata:
  node_type: memory
  type: feedback
  originSessionId: afa9f8f2-949b-4d3f-871d-9476bd46a4d3
  modified: 2026-10-04T17:53:31.480Z
---

Save every report and note I write, plus a doubts / open-questions file, in `/workspace/claude_reports/`. Keep its `README.md` index current. Use numbered filenames (01_, 02_, …). This is in addition to any repo copy, e.g. under docs/ on a branch.

**Why:** the user asked for it on 2026-10-04: "that way it will be easier for me to navigate".
**How to apply:** after writing any report or notes, copy it to `/workspace/claude_reports/` and update the README. Keep a doubts file listing what I could not verify, what I assumed, and questions about intent. /workspace is not persistent, so anything important should also go to git or an off-box mirror. Related: [[activation-readers-project]].
