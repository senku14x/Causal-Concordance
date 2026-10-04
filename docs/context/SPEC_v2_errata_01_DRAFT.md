# SPEC_v2 errata 01: DRAFT for PI review (2026-10-04)

*Not applied. SPEC_v2.md is unchanged.*
- Errata-block format, in the style of AMENDMENT_01 §8 (the PI's default).
- Each erratum gives the place in SPEC_v2, the text it corrects, the replacement, and the evidence.
- Covers review items B2–B5 and B10–B13, with the wording changes from the PI's feedback of 2026-10-04 (§3.2, §3.3, §3.5).
- Where a no-GPU check (pre-registered in `docs/design_specs/checks_2026-10-04_no_gpu.md`) bears on the wording, its result is quoted with n.

---

**V2-E1 (B2). Tier-1 label at ":" and the damage flag.**
- *Places:* §6.0 tier 1; §14.1.
- *Was:* ":" carries position-level "influence (reliable)"; the 001 damage flag is not mentioned.
- *Replace with:*

  > Tier-1 labels carry the 001 damage flag where it applies.
  > - 001 (82 pairs) flagged 46–67% of ":" whole-cell patches at L40–60. The rule: off-pair next-token mass up > 0.10 in > 20% of pairs.
  > - The rule was written with 1-D edits in mind. In a whole-cell *partner* patch, mass moving to the intermediates is part of the transfer, not damage.
  > - Check C2 (004 raw rows, ":" L42, 33 pairs, exact 12-colour masses) measured where the off-pair increase goes. Median off-pair increase 0.151 [0.128, 0.187]. Median shares of the increase:

  | Destination | Median share [95% CI] |
  |---|---|
  | z (recipient's intermediate) | 0.20 [0.14, 0.31] |
  | z′ (partner's intermediate) | 0.22 [0.14, 0.34] |
  | other palette colours | 0.20 [0.07, 0.25] |
  | non-colour tokens | 0.21 [0.16, 0.29] |
  | **z + z′** | **0.54 [0.39, 0.74]** |

  > - **Label at ":" L42: "influence".** The pre-registered rule gives this (z + z′ ≥ 0.50 and non-colour ≤ 0.25). The CI on z + z′ crosses 0.50, so the margin is thin.
  > - **":" at L40/44/56/60: "influence (damage-flagged)".** Only top-5 masses were saved there, and most of the off-pair increase lies outside the top 5. Settling them needs a full-distribution measurement.
  > - The 001 damage rule is to be rewritten for whole-cell partner patches: count z / z′ mass as transfer and report other-colour and non-colour mass separately.

**V2-E2 (B3). No W statement from different positions.**
- *Places:* §8 A2 callout; §9 P2 status.
- *Was:* "a W ≈ 0 pattern"; "position-level pattern so far is W ≈ 0 for J and the NLA on Brew (descriptive)".
- *Replace with:*

  > ":" and the newline after `</think>` are different positions, so they are not a matched contrast (§5). No W statement is made. Report 003 §9 withdrew the "W ≈ 0" reading after the audit. The observation stands as written: J-lens and the NLA report z at the no-influence position at least as often as at the influential one (002).

**V2-E3 (B4). "Readable, low patch leverage", not "decodable, not needed".**
- *Place:* §14.4, last sentence.
- *Was:* "At the `</think>` newline, z is decodable and not needed in 70–100% of pairs."
- *Replace with:*

  > At the `</think>` newline, z is readable at the family level (probe 0.79–0.84) while the whole-cell patch has little net leverage in 70–100% of pairs ("readable, low patch leverage"; 003 §3). This is not an item-level "decodable, not needed" label: no per-item probe probability was used. A small net patch cannot exclude cancellation (A01 E2).

**V2-E4 (B5). Necessity specificity not established.**
- *Place:* §14.4.
- *Was:* "'Used' z in 11–18% of pairs at ':' L40–44, each also necessary."
- *Replace with:*

  > "Used" z in 11–18% of pairs at ":" L40–44. Each of these also passes the necessity test against random clamps, but those clamps were about 20× smaller edits, so necessity *specificity* is not established, and damage was not logged on the ablated runs (003 §3, §9).

**V2-E5 (B10). Two prompt-only baselines, always labelled.**
- *Places:* §1 point 2 callout; §4 Baselines and [A01 C18]; §10 added pitfalls ("Guessable hidden steps").
- *Was:* Brew prompt-only 1.00 "(stock Qwen3.6-27B, asked directly)" treated as WSB's prompt-only; WSB multihop prompt-only ≈ 0.90 cited beside it.
- *Replace with:*

  > Two baselines, labelled wherever they appear:
  > - **WSB prompt-only (the reader floor):** stock Qwen3.6-27B, given the prompt up to the read position, asked what a model would be thinking there, judged by the family's own instrument. WSB brew: 0.96; multihop: values of record from the eval READMEs.
  > - **Task baseline ("what the text gives away"):** stock Qwen3.6-27B given the full prompt and asked for the value. 002: intermediate 1.00, start 1.00, answer 0.89 (231 Brew dev renders). It is not a ceiling or a floor for readers.

**V2-E6 (B11). Denominators in the §1 point 2 callout.**
- *Place:* §1 point 2 callout.
- *Was:* "its candidate *scores* rank the intermediate first in 0.39–0.45 of 48 tables (shuffled ≈ 0.17)".
- *Replace with:*

  > Its candidate scores rank the intermediate first:
  > - in 0.39–0.45 of 231 renders (members and partners) at ":" L40–44 (`lens_closed_set`);
  > - on member renders only, in 0.38–0.46 of 48 tables, against a shuffled 0.15–0.17 (`lens_post_hoc`).

**V2-E7 (B12). Reader-free removal routes.**
- *Places:* §4 "New controls for reading vs re-solving"; §8 A4 "[v2] Adopted as follows".
- *Was:* "reader-free (LEACE-style erasure fitted on dev items; on Brew, 004's fitted content subspace)".
- *Replace with:*

  > Reader-free removal, per family:
  > - **Brew:** LEACE fitted on dev z labels at that cell, accepted only if a fresh cross-fitted probe falls to chance (with the Hewitt–Liang control task) after erasure. 004's fitted z span is **untested as a z eraser** (004 never tested erasure; it matched same-rank PCA on swaps); at most it is a secondary removal.
  > - **Multihop** (75 distinct bridges in 100 items, so no class-label eraser): a per-bridge reader-free direction built as in Gurnee et al. Fig. 16 — the mean activation over prompts that imply the bridge through different surface cues and questions, minus the mean over other bridges. The fallback is the "Tell me about {bridge}" concept vector. Project it out (1-D, or a small subspace from several prompt variants). Validate erasure with a held-out probe for that bridge plus the Hewitt–Liang control task.
  > - **Controls for every removal:** a count-matched random-vocabulary J control (as CW) and an energy-matched random-subspace control. In check C3, CW's count-matched control removed a median of 0.39% vs 1.16% of ‖h‖² on multihop (ratio 2.4 [1.6, 4.7]; n = 50) and 0.09% vs 0.16% on Brew (ratio 1.7 [1.5, 2.3]; n = 50).

**V2-E8 (B13). Necessity null.**
- *Place:* §6 step 4, [R, 004; v2] callout.
- *Was:* "Adopt 004's paired-rotation control … as the necessity null."
- *Replace with:*

  > The necessity null has two parts:
  > - 004's paired rotation (keeps both runs' edit norms and their inner product; randomises orientation);
  > - a same-rank PCA clamp (PCA fitted on the same activations, evaluated table held out), which 004 did not run.
  >
  > Clearing the paired rotation shows that orientation matters relative to random; clearing the PCA clamp is needed before any content-specificity claim. INLP/LEACE-style removal remains the stronger tier.
