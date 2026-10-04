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
| C1 chain re-score | **Not scored**: the judge key is absent. Dry run passes on all 6 files (60/60 items each, all last-token, L42 accepted). CW's any-number counts reproduce exactly. | blocked | 55% that Oracle L56 beats the decoy null by ≥ 0.10 |
| C2 off-pair mass at ":" L42 | z + z′ take **0.54 [0.39, 0.74]** of the off-pair increase; non-colour 0.21 [0.16, 0.29] (33 pairs) | **"influence"** at ":" L42 (CI crosses 0.5; thin) | 25% for z + z′ ≥ 0.5 (I was wrong) |
| C3 removal energy | multihop ratio **2.4 [1.6, 4.7]** (50 items); Brew **1.7 [1.5, 2.3]** (50 items) | multihop: **B14 stands** (borderline); Brew: **roughly matched** (borderline) | 80% for multihop > 2 |

## C1. Chain re-score with WSB's chain instrument (CW-6 / CW-7)

**Status: blocked.**
- `OPENROUTER_API_KEY` is not on this instance; per CLAUDE.md it lives only in `/workspace/.env`, which is absent.
- No WSB numbers exist. Nothing below is a WSB pass rate.

**Dry run (WSB @ 92d763e, `chain_intermediates`, prompt `chain-free-2026-09-16`, `google/gemini-3.8-flash`, `cells=last`), all 6 readout files:**
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
- Two notes for when it is scored:
  - The Oracle's chance at L42 (21.0 of 60) is high because its readouts carry a median of 9 distinct numbers. WSB's top-value rule exists to remove that advantage.
  - J-lens L56 at 25/60 vs 6.8 was not discussed in the pilot; J-lens is the secondary arm here.
- Subset (ii), the usable single-digit items: 14 of the 60 (7 dev, 7 test by our chain split). Test numbers will be written only to `outputs/` (quarantined, §4.R).

**To finish:** with the key in the environment, run `python scripts/checks_2026-10-04/c1_chain_rescore.py /workspace/nla-vs-lenses --real` (360 judge calls, ≈ $1–3). The decision rule is in the pre-registration.

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
