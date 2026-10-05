# Report: three no-GPU checks (C1, C2, C3), 2026-10-04

*Pre-registration:* `docs/design_specs/checks_2026-10-04_no_gpu.md` (commit e50864d, committed before any analysis).
*Code / results:* `scripts/checks_2026-10-04/`, `results/checks_2026-10-04/` (commit 33e05c9).
- CPU only; no model forward.
- Read-only on `LittleMeHere/nla-vs-lenses` @ 58d34a8.
- Numbers are taken from the full-precision JSON and rounded half-up.
- Intervals are bootstrap 95% CIs of the median over pairs / items (seed 20261004).

## Summary

| Check | Result (n) | Pre-registered rule says | My prior |
|---|---|---|---|
| C1 chain re-score | Oracle L56 names an intermediate as its top computed value in **11/60 (0.183 [0.10, 0.28]) vs decoy null 0.041**; under CW's any-number rule it was 36/60. J-lens L56 also 11/60; NLA 1/60 at L42 and L56 (at or below null) | **CW-7 stands** as "read from the activation (WSB chain rule)", at about a third of its any-number size | 55% that Oracle L56 beats the null by ≥ 0.10 (it did: +0.14) |
| C2 off-pair mass at ":" L42 | z + z′ take **0.54 [0.39, 0.74]** of the off-pair increase; non-colour 0.21 [0.16, 0.29] (33 pairs) | **"influence"** at ":" L42 (CI crosses 0.5; thin) | 25% for z + z′ ≥ 0.5 (I was wrong) |
| C3 removal energy | multihop ratio **2.4 [1.6, 4.7]** (50 items); Brew **1.7 [1.5, 2.3]** (50 items) | multihop: **B14 stands** (borderline); Brew: **roughly matched** (borderline) | 80% for multihop > 2 |

## C1. Chain re-score with WSB's chain instrument (CW-6 / CW-7)

**Status: scored (2026-10-04, after the PI supplied the judge key).**
- WSB @ 92d763e, `chain_intermediates`, prompt `chain-free-2026-09-16`, `google/gemini-3.8-flash`, `cells=last`, unmodified; 360 judge calls, all 360 decided.
- Judge spend: $0.33.

**WSB rule, all 60 items (subset (i), exploratory, the same set as CW-6/7):**

| Reader, layer | Top computed value is an intermediate [95% CI] | Decoy null (±3, magnitude-matched) | Any of top 3 | Readout commits to a value | CW any-number (chance) |
|---|---|---|---|---|---|
| Oracle L42 (off-contract) | 8/60 = 0.133 [0.05, 0.233] | 0.046 | 0.233 | 1.000 | 27/60 (21.0) |
| NLA L42 (training layer) | 1/60 = 0.017 [0.0, 0.05] | 0.049 | 0.050 | 0.967 | 9/60 (12.4) |
| **Oracle L56 (on-contract)** | **11/60 = 0.183 [0.10, 0.283]** | **0.041** | 0.267 | 0.867 | 36/60 (12.1) |
| NLA L56 (off its training layer) | 1/60 = 0.017 [0.0, 0.05] | 0.020 | 0.017 | 0.650 | 13/60 (8.9) |
| J-lens L42 (secondary arm) | 2/60 = 0.033 [0.0, 0.083] | 0.000 | 0.033 | 0.150 | 6/60 (3.6) |
| J-lens L56 (secondary arm) | 11/60 = 0.183 [0.083, 0.283] | 0.024 | 0.217 | 0.583 | 25/60 (6.8) |

**Rule outcome.** Oracle L56: pass − null = 0.183 − 0.041 = 0.142 ≥ 0.10, and the pass rate is 4.5× the null. So **CW-7 stands as "read from the activation (WSB chain rule)"**. §14.7's entry is kept.

**What changes even so:**
- **Size.** 36/60 under the any-number rule becomes 11/60 under the top-computed-value rule. Most of the any-number credit came from intermediates that the readout mentions but does not present as the computed result.
- **Not specific to the Oracle.** The J-lens at L56 also passes 11/60. The Oracle's lead over the J-lens under the any-number rule (36 vs 25) disappears under WSB's rule.
- **Low per-item agreement.** The Oracle and the J-lens pass on largely different items (4 of 11 shared). With one readout per cell, per-item results are noisy.
- **The NLA** (`nla-rl` iter_000400) passes 1/60 at both layers, at or below its null. It does not present chain intermediates as computed values at the final cell, even at its training layer.
- **Reading vs re-solving.** "Read from the activation" here means passing WSB's rule against the decoy null. It does not separate reading from re-solving (A4 / A7); CW-7's "31 of 36 without restating the start" was measured under the any-number rule and has not been re-examined.

**Exploratory additions (not pre-registered):**
- Paired bootstrap of (pass − null) over items (seed 20261004): Oracle L56 0.143 [0.041, 0.249]; J-lens L56 0.159 [0.063, 0.267]; Oracle L42 0.087 [0.001, 0.185].
- Usable single-digit dev subset (subset (ii), n = 7, descriptive only): Oracle L42 2/7, Oracle L56 1/7, NLA 0/7 at both layers, J-lens 0/7 at both layers. Test-subset numbers are in `outputs/checks_2026-10-04/c1/usable_test_QUARANTINED.json`, unreported (§4.R).
- Contract status (PI note 3.4): the Oracle is off-contract at L42; the NLA is off-layer at L56.

**Dry run, before scoring (WSB @ 92d763e, `chain_intermediates`, prompt `chain-free-2026-09-16`, `google/gemini-3.8-flash`, `cells=last`), all 6 readout files:**
- Item match: 60 items each; all ids are in the pinned bank and every prompt is identical.
- Missing cells: 0. Rows not at the last token: 0.
- Layer 42 is accepted, although it is off WSB's grid.
- The J-lens files skip 660 rows each (the 11 other layers × 60), as they should.
- The rendered judge prompt is WSB's, unmodified. The Oracle readout (one sample of 4 bullets) is shown as one block.

**CW's any-number rule, recomputed with her `score_chain.py` imported unmodified (exploratory, the same set as CW-6/7):**

| Reader, layer | Names any intermediate | CW chance | Median distinct numbers per readout |
|---|---|---|---|
| Oracle L42 (off-contract) | 27/60 | 21.0 | 9 |
| NLA (nla-rl, `av_base` + iter_000400) L42 | 9/60 | 12.4 | 5 |
| **Oracle L56** | **36/60** | **12.1** | 6 |
| NLA (same) L56 (off its training layer, L42) | 13/60 | 8.9 | 4 |
| J-lens L42 | 6/60 | 3.6 | 1 |
| J-lens L56 | 25/60 | 6.8 | 3 |

- CW-7 (Oracle L56 36/60 vs chance 12) reproduces exactly.
- Checkpoints: CW's Oracle is WSB's `s3d.ddp600.s0/iter_000600` (hash-identical to our `olens_s3d_rl600`, verified earlier). CW's NLA is `ceselder/qwen3.6-27b-nla-rl` `av_base` + `av_rl_adapters/iter_000400` (her `scripts/pod_bootstrap.sh`), the same checkpoint our default plan uses. "nla_L42" in her filenames is the read layer.
- Two notes written before scoring:
  - The Oracle's chance at L42 (21.0 of 60) is high because its readouts carry a median of 9 distinct numbers. WSB's top-value rule exists to remove that advantage.
  - J-lens L56 at 25/60 vs 6.8 was not discussed in the pilot; J-lens is the secondary arm here.
- Subset (ii), the usable single-digit items: 14 of the 60 (7 dev, 7 test by our chain split). Test numbers will be written only to `outputs/` (quarantined, §4.R).

**Reproduce:** `python scripts/checks_2026-10-04/c1_chain_rescore.py /workspace/nla-vs-lenses --real` with `OPENROUTER_API_KEY` set.

## C2. Where the off-pair mass goes after the whole-cell partner patch at ":"

**(A) Exact, L42.**
- Data: 004 raw rows, arm `colon|full`, 33 pairs, one per table, full-sequence 12-colour masses.
- The off-pair mass rose by more than 0.01 in 32/33 pairs; the median increase was **0.151 [0.128, 0.187]**.

| Category | Median Δ mass [CI] | Pairs with Δ > 0 | Median share of off-pair increase [CI] |
|---|---|---|---|
| a (recipient's answer) | −0.366 [−0.444, −0.331] | 0/33 | – |
| a′ (partner's answer) | +0.176 [0.127, 0.235] | 33/33 | – |
| z (recipient's intermediate) | +0.025 [0.012, 0.062] | 29/33 | 0.20 [0.14, 0.31] |
| z′ (partner's intermediate) | +0.023 [0.016, 0.045] | 32/33 | 0.22 [0.14, 0.34] |
| other 8 palette colours | +0.016 [0.008, 0.036] | 28/33 | 0.20 [0.07, 0.25] |
| non-colour | +0.040 [0.024, 0.049] | 29/33 | 0.21 [0.16, 0.29] |
| z + z′ | – | – | **0.54 [0.39, 0.74]** |

- The share medians are separate medians, so they do not sum to 1.
- The next-token version agrees to the third decimal: 11 of 12 colours are single tokens, and " Teal" is the only two-token spelling.

**Rule outcome:** z + z′ ≥ 0.50 and non-colour ≤ 0.25 → proposed label **"influence"** at ":" L42, together with a rewrite of the 001 damage rule for whole-cell partner patches. The interval on z + z′ crosses 0.50, so the margin is thin. The PI decides.

**Caveat I did not anticipate when I wrote the rule (exploratory; affects the interpretation, not the rule's outcome):**
- The recipient's *own* intermediate z gains as much as z′ does (29/33 pairs).
- Counting z′'s gain as transfer is natural. Counting z's gain as transfer is not: patching in the partner's cell should not raise the recipient's own z.
- The better description is that the patched output spreads toward **both trajectories' intermediates**. Per colour, z and z′ each gain about 0.02–0.03, while the 8 other colours gain about 0.002 each (Δ median / 8).
- So the off-pair mass is structured, not generic damage, but "transfer" overstates it. If the PI wants a stricter reading, z′ alone takes 0.22 [0.14, 0.34].

**(B) Partial, all layers.**
- Data: 001 raw rows, `emit_colon`, p→m, 82 pairs.
- Only the top-5 tokens were saved, so this part is descriptive.

| Layer | Off-pair increase (median) | Pairs > 0.10 | Visible z + z′ Δ (median) | Unattributed (median) |
|---|---|---|---|---|
| L40 | 0.115 | 0.59 | 0.035 | 0.072 |
| L42 | 0.140 | 0.67 | 0.023 | 0.115 |
| L44 | 0.121 | 0.60 | 0.020 | 0.101 |
| L56 | 0.092 | 0.48 | 0.008 | 0.070 |
| L60 | 0.092 | 0.46 | 0.016 | 0.064 |

- Most of the off-pair increase lies outside the top 5 at every layer.
- So L40/44/56/60 keep "influence (damage-flagged)" until a full-distribution measurement exists there. That measurement needs the GPU and is proposed for a later design.
- L42 from (B), 0.140 over 82 pairs, agrees with (A), 0.151 over 33 pairs.

## C3. How much of the activation the CW removals take away

**Gate:** all 200 edits (50 items × 2 conditions × 2 sets) are exact orthogonal projections (|cos(Δh, h_edit)| < 1e-3; Pythagoras holds to < 1e-4 relative). No exclusions.

| Set | Content removal (share of ‖h‖²) | Random control (share of ‖h‖²) | Ratio R, median [CI]; quartiles | Content > control | Directions removed (min / q1 / median / q3 / max) |
|---|---|---|---|---|---|
| Multihop CW-4, L42, n = 50 | 1.16% [0.82, 1.45] | 0.39% [0.16, 0.57] | **2.4 [1.6, 4.7]**; 1.2 / 2.4 / 7.3 | 41/50 | 4 / 5.25 / 18.5 / 45.25 / 129 |
| Brew CW-9, ":" L42, n = 50 | 0.16% [0.15, 0.17] | 0.088% [0.071, 0.106] | **1.7 [1.5, 2.3]**; 1.3 / 1.7 / 3.0 | 42/50 | 4 / 4 / 4 / 4 / 8 |

**Reading by the rule:**
- Multihop: R > 2 → **B14 stands**. The control removes less of the activation, so CW-4's survival figures carry the annotation "control removes less of the activation (median R = 2.4)". The CI reaches 1.6, so this is borderline.
- Brew: R in 0.5–2 → **roughly matched**. B14 is withdrawn for CW-9, and the count-matched control is acceptable there as one of the two A4 controls. The CI reaches 2.3.

**What C3 does not change:**
- Both removals are small, about 0.1–1% of the activation's energy. Equal energy does not mean equal effect.
- My recommendation is unchanged: use an energy-matched random-subspace control in any rerun. It is cheap and removes the question.

## Errors in this work (also in the commit messages and CLAUDE.md)

- A check script indexed `splits.chain_split`'s (map, groups) tuple as a dict (AttributeError). Fixed by unpacking.
- C2 first read the 001 gate record's partner intermediate as `z_partner` / `partner_z`; the field is `z_p`.
  - **Correction to commit 33e05c9's message:** it says the bug was "caught by the missing-roles counter". In fact I caught it by printing the gate record before the first run. The commit is pushed, and I do not force-push.
- C1's first dry run took CW's `items.json` (the 120-item gated bank) as the item list. WSB reported 120 expected cells, 60 missing. The item list is now the readout ids, asserted against the pinned bank and its prompts.

## Addendum: PI review of these checks (2026-10-04)

Disposition: `docs/reviews/2026-10-04_pi_review_checks_005/`. The PI accepted C1–C3 with a sharper reading of each; every factual claim in the review was checked first.

- **C1.**
  - The Oracle's chain result is not Oracle-specific. J-lens, a linear readout that cannot re-solve, also passes 11/60 at L56, so the intermediate is linearly readable at the final cell at L56 in some items.
  - The Oracle hits J-lens does not share (7 of 11) are the ones relevant to reading vs re-solving. With one readout per cell, they are at noise level.
  - **Chain cannot test the NLA.** At L42 the final cell carries no leverage (S ≈ 0.001 at L40–44, report 001 §5; checked) and J-lens passes 2/60. At L56 the NLA is off-layer. Its 1/60 is uninformative, not a failure.
  - **Added in the disposition:** at that same no-leverage L42 cell, the Oracle (off-contract) passes 8/60 against a null of 0.046 (paired CI [0.001, 0.185]). That is a weak re-solving candidate.
  - Recorded as SPEC_v2 erratum E9.
- **C2.**
  - "Influence" at ":" L42 is accepted, described as a **mixture of the two trajectories**: a → a′; both intermediates rise (z is the recipient's runner-up answer in 23/33 pairs, checked); about 0.04 leaks to non-colour tokens.
  - The damage-rule rewrite for whole-cell partner patches is in erratum E1. It is post-hoc and applies to future data only.
  - No GPU will be spent on the other layers.
- **C3.**
  - Both removals take only 0.1–1% of ‖h‖², so J-lens hits fall 15 → 0 by construction, while prose readers keep about 99% of the activation.
  - "Survives J-removal" therefore cannot separate reading outside J-space from re-solving. Only a probe-validated eraser can, and since it removes linear information only, survival after it means re-solving or nonlinear reading.
  - Recorded in erratum E7 (A4).

