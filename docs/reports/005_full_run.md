# 005 full run under amendment 01 — results (run `20261005T015235Z`, 45 test concepts)

*Spec:* `docs/design_specs/005_full_run.md` (pre-registered before the run). *Amendment:* `docs/design_specs/005_amendment_01.md` (adopted 2026-10-05). *Branch:* `005-amended-full-run`. *Summary JSON:* `results/005_planted_concepts/analysis/full_summary.json`.

## The short version

- The run is clean, with one gate failure that matters. G0 had 0 failures over every write and clamp; the J share median is 5.0%; the probe is valid (LOTO AUC 0.997, permutation control 0.48); no split is leaky; G6 drops case for the NLA; G2 passes for every reader; G11 passes for the Oracle and fails for the NLA.
- The P-family is not supported. P11′ for the NLA could not be tested (its likelihood measure failed the positive control). P17 goes the other way: the Oracle names the non-J part more often than the NLA (−0.14 [−0.18, −0.11]).
- No J-space privilege for report on the primary measure. At matched norm the J part's peak beats the non-J part's by +0.16 [−0.08, 0.40] nats (not supported). The registered R1 form clears the rule by a tiny margin that is partly by construction. At natural scale the non-J part moves the report more (−0.22 [−0.28, −0.16]).
- The clamp is specific, but not beyond blocking the output direction. The J re-entry clamp to L50 beats its null (+0.12 [0.03, 0.21]), and so does the output-direction clamp; the difference is +0.06 [−0.04, 0.15].
- On the written cell, the Oracle is more sensitive than the model's report. Its likelihood reaches a 1-nat specific rise −2.0 [−2.5, −1.5] dose doublings earlier; the model's report never does for 29 of 45 concepts. The probe also beats the NLA's naming (R1). Not perturbation-matched.
- Behaviour moves without report for 71% of concepts by the registered definition, 29 of 45 with a post-hoc size floor (mostly at s = 1).

## Gates

| Gate | Rule | Result | Pass |
|---|---|---|---|
| G0 fidelity | every write ρ ∈ [0.9, 1.1], κ ≥ 0.99; clamp residual ≤ 1e-4 | 7119 write records, 17100 clamp rows, 0 failures | yes |
| G3 J share | median J share at k = 16 within [1%, 30%] | 5.0% | yes |
| G4 probe validity | median LOTO AUC ≥ 0.9 and permutation-control AUC ≤ 0.6 | 0.997 / 0.48 | yes |
| G5 leaky splits | c in the J_NP top 10 on the pure non-J part at ≥ 2 band layers | none | yes |
| G6 oracle L40 | clean-read naming > 5% drops the concept for this reader | none dropped | yes |
| G6 oracle L44 | clean-read naming > 5% drops the concept for this reader | none dropped | yes |
| G6 nla L42 | clean-read naming > 5% drops the concept for this reader | case | no / flag |
| G2 nla L42 | full vector named above rand-dir at a dose where the model moves | s = 1/64: 0.00 [0.00, 0.00]; s = 1/4: +0.18 [0.09, 0.29]; s = 1: +0.58 [0.45, 0.71] | yes |
| G2 oracle L40 | full vector named above rand-dir at a dose where the model moves | s = 1/64: 0.00 [0.00, 0.00]; s = 1/4: +0.28 [0.17, 0.41]; s = 1: +0.69 [0.57, 0.81] | yes |
| G2 oracle L44 | full vector named above rand-dir at a dose where the model moves | s = 1/64: −0.01 [−0.02, 0.00]; s = 1/4: +0.26 [0.15, 0.37]; s = 1: +0.76 [0.64, 0.86] | yes |
| G11 nla L42 | natural activations identified among 55 concepts ≥ 50% | 44% of 45 (missed: ash, board, box, case, cook, corner…) | no / flag |
| G11 oracle L40 | natural activations identified among 55 concepts ≥ 50% | 67% of 45 (missed: board, case, cook, corner, ear, formula…) | yes |
| G11 oracle L44 | natural activations identified among 55 concepts ≥ 50% | 67% of 45 (missed: board, case, cook, corner, ear, hat…) | yes |

## Pre-registered contrasts

| Contrast | n | Mean [95% CI] | Predicted | Verdict | Registered prior | Updated prior |
|---|---|---|---|---|---|---|
| M1 on R1 (registered form): J part > non-J part at peak | 45 | +0.0027 [0.0004, 0.0063] | J > non-J | supported | 0.85 | — |
| M1 on R1′ (primary under A1): J part > non-J part at peak | 45 | +0.16 [−0.08, 0.40] | J > non-J | not supported | — | 0.25 |
| M2: J re-entry clamp to L50 beats its random-token null | 36 | +0.12 [0.03, 0.21] | > 0 | supported | 0.30 | 0.65 |
| M2 control: J clamp beats the output-direction clamp at L50 | 36 | +0.06 [−0.04, 0.15] | > 0 | not supported | — | 0.40 |
| M3: J-natural > non-J-natural where the full vector moves R1′ | 45 | −0.22 [−0.28, −0.16] | > 0 | not supported | 0.60 | 0.10 |
| R1: probe detects the non-J part at least as often as the NLA (point estimate ≥ 0) | 44 | +0.32 [0.27, 0.37] | ≥ 0 | yes (≥ 0) | 0.70 | 0.85 |
| R2: behaviour moves without report (share of concepts, CI > 0) | 45 | +0.71 [0.58, 0.84] | > 0 | supported | 0.50 | — |
| P11′ (NLA): reader onset earlier than the model's | 0 | — | < 0 | not tested (G11 failed) | — | 0.55 |
| P11′ (Oracle): reader onset earlier than the model's | 45 | −2.04 [−2.53, −1.53] | < 0 | supported | — | 0.85 |
| P17: the Oracle names the non-J part less than the NLA | 44 | −0.14 [−0.18, −0.11] | > 0 | not supported | 0.45 | 0.05 |

Holm (P-family): P17: p = 1.0000 vs 0.0250 → does not

Brier: registered priors 0.236 (n = 6); updated priors 0.058 (n = 7).

## Notes

**r2_floor_post_hoc.** {'share': 0.6444444444444445, 'n': 29, 'of': 45, 'qualifying_by_dose': {'1/64': 0, '1/4': 6, '1': 27}, 'floor': {'B1': 0.1, 'B2': 1.0, 'B4': 0.1}, 'label': 'post hoc (after the results were read)'}
**M1.** Not on the primary measure. At each part's peak, the J part beats the non-J part by +0.16 [−0.08, 0.40] nats of ΔR1′, an interval that includes 0, and none of the five alternative J-space constructions changes that. On R1, the form the prior was registered on, the J part's peak reciprocal rank is higher by +0.0027 [0.0004, 0.0063]: this clears the rule but is tiny, and it is partly by construction (the J part contains the concept's own atom). Per dose the J part leads from 1/128 to 1/4 (largest at 1/8, +0.34 nats) and trails at 1/2 (−0.24), the same crossing pattern as on the dev concepts.
**M3.** Opposite of the prediction. At natural scale the non-J part moves the report more than the J part: −0.22 [−0.28, −0.16] nats (J-natural minus non-J-natural) over the doses where the full vector moves the report. The non-J part is about 95% of the vector's norm, and at its natural size it carries most of the report effect.
**M2.** The J re-entry clamp to L50 removes more of the non-J part's report effect than its random-token null: +0.12 [0.03, 0.21] nats over 36 evaluable concepts, replicating the dev finding. But clamping the plain output directions of the same tokens does so too (+0.06 [0.00, 0.12]), and the difference is not distinguishable from 0 (+0.06 [−0.04, 0.15]); along the end-layer curve the J clamp is ahead only at L56. Under the adopted A8 wording, the clamp "blocks the concept's output-related coordinates"; routing through J coordinates specifically is not shown. The design's original clamp (to clean, L32–62) is strongly specific (+0.42 [0.29, 0.56]), as on dev, and is read next to the same caveat.
**P11.** P11′ for the NLA could not be tested: its likelihood measure failed the registered positive control (it identifies 44% of natural test-concept activations among the 55 concepts, against a 50% bar; chance is 2%). The Oracle passes (67% at L40 and at L44), and on the written cell its likelihood reaches a 1-nat specific rise for the non-J part −2.0 [−2.5, −1.5] grid steps (dose doublings) before the model's report does. The model's report never reaches it for 29 of 45 concepts. Caveat registered beforehand: this compares one written cell with a report that integrates the injection over 7 layers × 20 positions; it is not a perturbation-matched comparison.
**P17.** Not supported, and the interval lies on the other side: the Oracle names the non-J part more often than the NLA (−0.14 [−0.18, −0.11] in naming rate, NLA minus the better Oracle layer, direct reads), and the likelihood version agrees (−1.46 [−1.80, −1.13] nats). The A.9.2-based premise that the Oracle sees content outside J-space less does not hold on these concepts. P11′ (NLA) being untestable leaves P17 alone in the P-family; with the registered family of two its Holm threshold would be 0.0125, which changes nothing (p = 1).
**R.** R1: the probe (absolute criterion) detects the non-J part on direct reads at L42 more often than the NLA names it, by +0.32 [0.27, 0.37] (as predicted). R2: by the registered definition, 71% of concepts have a dose where the non-J part moves behaviour (B1, B2 or B4 above the random null) while the report stays within it. Part of this is the scale-free trap: at s = 1/64 the neutral-prompt KL is about 0.002 nats, above a random null near zero. With a post-hoc absolute floor (B1 ≥ 0.1, B2 ≥ 1 or B4 ≥ 0.1 nats) it holds for 29 of 45 concepts, mostly at s = 1, where random directions also move the report a lot (the report's random null there is 1.26 nats).
**caveats.** Planted content only (design §14). One prompt, one band, one model. The non-J part is "outside J-space by this construction" (J_NP, pursuit k = 16, folded atoms). Direct reads give the reader the written cell; the model's report integrates many cells. Naming is exact spelling-variant matching; the synonym judge was deferred.
