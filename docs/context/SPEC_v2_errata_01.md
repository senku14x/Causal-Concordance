# SPEC_v2 errata 01: ACCEPTED (PI review, 2026-10-04)

*Status: accepted by the PI on 2026-10-04.*
- **As drafted:** E2–E6 and E8.
- **Reworded as instructed:** E1 (mixture description; damage-rule rewrite marked post-hoc and prospective) and E7 (C3's removal-size point in A4).
- **Added on the PI's instruction:** E9 (§14.7 CW-7 entry, and §1 point 6's counter-case).
- The review and its disposition: `docs/reviews/2026-10-04_pi_review_checks_005/`.
- Errata-block format, in the style of AMENDMENT_01 §8. **Where an erratum and the SPEC_v2 text disagree, the erratum wins.** SPEC_v2's body is not edited in place; its header points here.
- Each erratum gives the place in SPEC_v2, the text it corrects, the replacement and the evidence. Check results are quoted with n (pre-registration: `docs/design_specs/checks_2026-10-04_no_gpu.md`; report: `docs/reports/checks_2026-10-04_no_gpu.md`).

---

**V2-E1 (B2). Tier-1 label at ":" and the damage flag.**
- *Places:* §6.0 tier 1; §14.1.
- *Was:* ":" carries position-level "influence (reliable)"; the 001 damage flag is not mentioned.
- *Replace with:*

  > Tier-1 labels carry the 001 damage flag where it applies.
  > - 001 (82 pairs) flagged 46–67% of ":" whole-cell patches at L40–60. The rule: off-pair next-token mass up > 0.10 in > 20% of pairs.
  > - The rule was written with 1-D edits in mind. In a whole-cell *partner* patch the patched output becomes **a mixture of the two trajectories**:
  >   - the answer moves toward the partner's (a → a′);
  >   - both intermediates rise: z, the recipient's own intermediate and its runner-up answer in 23/33 pairs, as well as the partner's z′;
  >   - about 0.04 of probability (absolute) leaks to non-colour tokens.
  > - This is not transfer from the partner (z rises as much as z′), and it is not generic damage either.
  > - Check C2 (004 raw rows, ":" L42, 33 pairs, exact 12-colour masses) measured where the off-pair increase goes. Median off-pair increase 0.151 [0.128, 0.187]. Median shares of the increase:

  | Destination | Median share [95% CI] |
  |---|---|
  | z (recipient's intermediate) | 0.20 [0.14, 0.31] |
  | z′ (partner's intermediate) | 0.22 [0.14, 0.34] |
  | other palette colours | 0.20 [0.07, 0.25] |
  | non-colour tokens | 0.21 [0.16, 0.29] |
  | **z + z′** | **0.54 [0.39, 0.74]** |

  > - **Label at ":" L42: "influence" (PI-accepted), described as the mixture above.** The pre-registered rule gives this (z + z′ ≥ 0.50 and non-colour ≤ 0.25). The CI on z + z′ crosses 0.50, so the margin is thin.
  > - **":" at L40/44/56/60: "influence (damage-flagged)".** Only top-5 masses were saved there, and most of the off-pair increase lies outside the top 5. No GPU will be spent decomposing them, since no decision depends on it (PI).
  > - **Damage rule for whole-cell partner patches: post-hoc, prospective, future data only.** It was written after seeing C2 and is not re-applied to 001–004.
  >   - Damage = **non-colour** mass (1 − Σ palette colours, full-sequence) up by > 0.10 in > 20% of pairs. These are the 001 thresholds, applied to non-colour mass only, not tuned on C2.
  >   - Mass spreading across palette colours (z, z′, other colours) is reported separately as "mixture", never as damage.
  >   - 1-D and span edits keep the 001 rule.

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
  > - **What "survives removal" can show (add to A4).**
  >   - Both CW removals take only 0.1–1% of ‖h‖² (C3).
  >   - Removing that much takes J-lens hits 15 → 0 (CW-3, CW-4), as it must, since J-lens reads exactly those directions.
  >   - Prose readers meanwhile keep about 99% of the activation, including what they would need to re-solve.
  >   - So "survives J-removal" cannot separate reading outside J-space from re-solving. Only an eraser shown to remove the information can: a fresh held-out probe at chance, with the Hewitt–Liang control task.
  >   - Even then, a LEACE-type eraser removes *linear* information only. A claim that survives validated erasure means re-solving or nonlinear reading.
  >   - The survives-removal tag means little without validated erasure.

**V2-E8 (B13). Necessity null.**
- *Place:* §6 step 4, [R, 004; v2] callout.
- *Was:* "Adopt 004's paired-rotation control … as the necessity null."
- *Replace with:*

  > The necessity null has two parts:
  > - 004's paired rotation (keeps both runs' edit norms and their inner product; randomises orientation);
  > - a same-rank PCA clamp (PCA fitted on the same activations, evaluated table held out), which 004 did not run.
  >
  > Clearing the paired rotation shows that orientation matters relative to random; clearing the PCA clamp is needed before any content-specificity claim. INLP/LEACE-style removal remains the stronger tier.

**V2-E9 (B16; PI instruction 2026-10-04). CW-7 after the WSB re-score.**
- *Places:* §14.7, first bullet ("Read from the activation"); §1 point 6, counter-case sentence; §14 CW table, row CW-7 (annotate).
- *Was (§14.7):* "Chain at L56: the Oracle names the intermediate in 36/60 vs chance 12, with 31 hits lacking the start number (CW-7)."
- *Was (§1 point 6):* "Counter-case: on chain at L56 the Oracle names the intermediate in 36/60 (chance 12), 31 of them without restating the start number (CW-7)."
- *Replace §14.7's chain sentence with:*

  > **Chain, WSB rule (C1, 2026-10-04; 60 items, one readout per cell).**
  > - **The Oracle result survives at a third of its size.** The Oracle at L56 (on-contract) gives the intermediate as its top computed value in 11/60 (0.183 [0.10, 0.28]), against a magnitude-matched decoy null of 0.041. That is about a third of CW-7's any-number count (36/60).
  > - **It is not Oracle-specific.** J-lens passes 11/60 at L56 (null 0.024). J-lens is a linear readout and cannot re-solve, so its hits show the intermediate is linearly readable at the final cell at L56 in some items.
  > - **Which hits matter.** The Oracle hits J-lens does not share (7 of 11) are the ones that bear on reading vs re-solving. With one readout per cell they are at noise level.
  > - **Chain cannot test the NLA.**
  >   - At L42, its training layer, the final cell carries no leverage (S ≈ 0.001 at L40–44; 001) and J-lens passes 2/60. The content is probably not there yet (decodability at that cell is untested).
  >   - At L56 the NLA is off-layer.
  >   - Its 1/60 at both layers is therefore uninformative, not a failure.
  > - **A weak re-solving candidate.** At that same no-leverage L42 cell, the Oracle (off-contract) passes 8/60 against a null of 0.046 (paired CI on pass − null [0.001, 0.185]).
  > - **Not re-examined.** CW-7's restatement check (31 of 36 without the start number) was made under the any-number rule.

- *Replace §1 point 6's counter-case with:*

  > Counter-case: on chain at L56, under WSB's rule, the Oracle gives the intermediate as its top computed value in 11/60 (null 0.041; 36/60 under the any-number rule). J-lens does as well (11/60), so the case is not Oracle-specific (C1; E9).

- *§14 CW table, row CW-7:* annotate "any-number rule; under WSB's rule Oracle 11/60, J-lens 11/60, NLA 1/60 (C1)".

