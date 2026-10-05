# AMENDMENT_02 — test-split use (opened 2026-10-05)

**Status.**
- Opened to record a single PI-approved, reader-side use of the brew test split.
- τ, ε′ and the substitution bound (001 §13, SPEC_STATUS) are **not** frozen by this file.
- Freezing them, before any test-split *label*, remains open and is not touched here.

## Use 1 — 007 brew test replication (approved 2026-10-05)

- **PI, verbatim:** "I approve this use of the test split under AMENDMENT_02; record that."
  The approval came with the 007 spec v2 decisions; the spec v2 go was given the same day.
- **What:** design 007 §7. The pre-registered Stage 1a reader-side rules (P1–P5, δ = 0.5 nats, Holm/TOST, the
  three-model mapping) are applied unchanged to the 49 brew test tables:
  - SEC captures at `:` L40/42/44;
  - Oracle L40/L44 and NLA L42, plus the template lens (secondary);
  - PCA-k basis and k frozen from dev.
- **Material:** all 178 test start candidates (primary). The gated pairs from 001's existing, reader-free gate (77/178)
  are secondary.
- **No labels are computed.** Labels never use readers, so this use cannot leak into any label.
- **Consequence recorded:** after 007 the brew test tables have been read by the readers. A later reader evaluation on
  brew test is not on untouched tables.
