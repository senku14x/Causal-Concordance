---
name: page-charts-native-style
description: "On results pages/artifacts, draw charts natively (SVG/HTML in the page's style), not embedded matplotlib PNGs"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0f5258a9-0567-4d36-8e23-ce24b09256c7
  modified: 2026-10-02T10:49:58.128Z
---

On the Causal-Concordance results page, draw plots natively in the page's own chart style (inline SVG / HTML tables
using the page's fonts and CSS palette), not as embedded matplotlib/ICLR-style PNGs.

**Why:** the user called the embedded PNGs (serif ICLR style, overlapping labels, half-empty heatmaps, white box in
dark mode) "ass" and said the page's own charts are "much cleaner" (2026-10-02).

**How to apply:** any new figure for the page goes into `scripts/000_setup/results_page/` as JS drawn from DATA,
built from committed JSON; check it with a headless-Chromium screenshot (light, dark, 400 px) before publishing.
Report figures in `results/*/figures/` still come from `figures.py` unless the user asks otherwise. Related:
[[keep-qualitative-samples]].
