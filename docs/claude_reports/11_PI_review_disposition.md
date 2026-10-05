# PI review of checks C1–C3, design 005 v1 and errata 01 (2026-10-04)

- `review.md`: the review as delivered (summary + pasted change list).
- The PI asked for acceptance "only if it makes sense". Every factual claim was checked against the data or the code before acceptance (evidence below).
- **Disposition: accepted, with four nuances and one addition.** Implemented in:
  - design 005 v2;
  - the checks report addendum;
  - errata 01, now accepted.

## Factual claims checked

| Claim | Check | Result |
|---|---|---|
| 001 found zero whole-cell leverage at the chain final cell at L40–44 | report 001 §5 | **Holds:** S ≈ 0.001 at L40/42/44 (96–100% of pairs negligible) |
| J-removal takes J-lens hits 15 → 0 | SPEC_v2 §14 CW table (CW-3, CW-4) | **Holds** |
| Brew's intermediate is the model's runner-up answer | C2 clean full-sequence masses, 33 pairs | **Holds in 23/33 pairs** (z is the most likely non-answer colour: median 0.026 vs a′ 0.008, z′ 0.004, other colours 0.0036 each). Scoped as "in most pairs" |
| The C_q read at L40–44 is near-direct access | design v1 §4/§6: C_q (110) lies in the injected span (97–116), and the residual stream carries the band writes forward | **Holds.** My v1 mislabelled it "model-carried" |
| The clamp partly deletes the injection | T_c = the union of supports over 7 band layers plus spellings; each layer's u^⊥ is orthogonal only to its own active atoms; at L39–62, atoms of other layers are not orthogonal to the carried u^⊥ | **Holds**, and is stronger than stated: at every layer above the band, the clamp removes part of the carried injection. My v1 "expected near 0" was wrong |
| Oracle "k = 4" | `readers.OracleLens.read(k=4)`; designs 001/002 | **Resolved: 4 samples** (`num_return_sequences`), each a few bullets |
| Common words will swamp detection | screen run (`scripts/005_planted_concepts/screen_concepts.py`; 7,148 existing NLA/Oracle samples, rule fixed before computing) | **Partly.** At 5%, only **time, line, path** are excluded. Information (1.2% NLA / 0.0% Oracle), process (1.2% / 1.2%), case, frame, method, summary, review, role and free pass. The screen is kept as a principled rule; in-run clean reads re-check on the introspection prompt |

## Nuances added (where the review is right, but stated more strongly than the evidence)

1. **"The content probably isn't there yet" at chain L42.** S ≈ 0.001 shows that the final cell carries no *causal* leverage at L40–44. Whether the intermediate is *decodable* there is untested: there is no chain probe at that cell, and 003 found "readable, low patch leverage" states on Brew. "Chain cannot test the NLA at L42" is still right in practice.
2. **Addition the review missed.** At that same no-leverage cell, the Oracle (off-contract at L42) passes 8/60, against J-lens 2/60 and a null of 0.046 (paired CI on pass − null [0.001, 0.185]). If the intermediate is not causally present there, those hits are a re-solving candidate, weak and off-contract. Recorded in E9.
3. **"Only a validated eraser can separate reading from re-solving."** A probe-validated eraser (LEACE) removes *linear* information only. A claim that survives validated erasure therefore means re-solving *or* nonlinear reading. That is still far stronger than J-removal. Recorded in E7.
4. **Power argument.** More concepts beat more samples here because between-concept variance dominates. Halving samples doubles the within-concept term. Accepted at NLA 4 samples (the PI's floor).

## What was implemented

- **Design 005 v2** (`docs/design_specs/005_planted_concepts.md`, §15 lists the changes):
  - read cells redefined (direct = P17 primary; carried C_o = P11 primary; C_q at L ≥ 40 = "injected position", token readers and probe only);
  - strict P11a/P11b primary;
  - text-reader strengths chosen after G1 to bracket the u^⊥ onset;
  - screen, plus dev 10 / test 45;
  - text readers at 4 conditions, NLA at 4 samples;
  - random-clamp null;
  - G5 split-quality gate;
  - leave-one-out baseline (194 words as sensitivity);
  - M3 natural norm;
  - B4 same-forward behaviour;
  - rand-J from the baseline words;
  - Oracle k = 4 samples.
- **Errata 01 accepted** (`docs/context/SPEC_v2_errata_01.md`): E1 and E7 reworded as asked; E9 (§14.7 CW-7) added on the PI's instruction; the damage-rule rewrite (post-hoc, prospective) is in E1.
- **Checks report addendum** with the sharper readings of C1–C3.
