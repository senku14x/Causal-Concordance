# 007 Stage 3 — status (multihop Part A; secondary; 2026-10-05)

**What ran.** WSB multihop, 100 items, plain render, captured at the bank's read position (stock forward, L40/42/44);
relation-level pairs within split (dev = items 1-50, touched; test = 51-100): 28 dev / 104 test pairs over 27 / 40 items
in 9 / 9 key groups; the 21 v1 pieces; the Oracle L40/L44 (the NLA is disabled, A1) with each key group's bridge and
target strings as candidates; no-activation rows; positive control on 50 "Tell me about {bridge}" activations (10 piece
types); prompt-only floor; restatement samples (k = 2; 6 pieces). GPU time 31.0 min.

**Gates.** G-cap (read token = last word or opening quote, plain render) on all 100 — pass. G-cw against Cathryn's L42
residuals: 25 matched positions, cos ≥ 0.99986 — pass. Oracle L40 / L44: G-score1, G-fastpath, G-score2,
G-inject incl. scale check — pass.
**Disclosed limitation:** the within-key PCA on the 50 dev items has rank ≤ 7, holding only 0.024 of the dev pair
difference vs J's 0.27: the multihop PCA-k is not Δ-matched (k = 7 = kmax), so P1 has no Δ-matched meaning here.

**Bridge D (nats), test split (the reported number; dev touched), key-cluster bootstrap CI (9 clusters — wide).**

| reader | whole | J | rest of J | PCA-7 | MR | random (mean of 10) | rest random (3) | dev: whole / J / rest J |
|---|---|---|---|---|---|---|---|---|
| oracle_L40 | 14.7 [8.7, 18.0] | 12.8 [6.8, 16.6] | 6.8 [3.9, 8.8] | -0.3 [-1.1, 0.3] | 8.1 [2.3, 11.6] | 8.7 | 13.4 | 8.1 / 7.0 / 3.4 |
| oracle_L44 | 14.8 [8.7, 18.2] | 13.5 [7.8, 17.0] | 2.0 [0.6, 2.8] | 0.4 [-0.5, 1.1] | 8.7 [2.4, 12.6] | 7.8 | 13.8 | 8.0 / 8.4 / 0.9 |

**Restatement flags on sampled readouts (test; word match, no judge; cue shares over the items with a cue word).**

| reader | piece | P(names bridge) | P(restates cue) | P(bridge without cue) | P(names target) |
|---|---|---|---|---|---|
| oracle_L40 | whole | 0.85 | 0.55 | 0.38 | 0.94 |
| oracle_L40 | J | 0.91 | 0.27 | 0.68 | 0.93 |
| oracle_L40 | rest_J | 0.42 | 0.09 | 0.39 | 0.55 |
| oracle_L40 | R0 | 0.34 | 0.02 | 0.34 | 0.65 |
| oracle_L40 | PCA | 0.00 | 0.00 | 0.00 | 0.04 |
| oracle_L40 | MR | 0.45 | 0.36 | 0.27 | 0.76 |
| oracle_L44 | whole | 0.86 | 0.27 | 0.64 | 0.90 |
| oracle_L44 | J | 0.94 | 0.25 | 0.73 | 0.95 |
| oracle_L44 | rest_J | 0.12 | 0.04 | 0.12 | 0.14 |
| oracle_L44 | R0 | 0.17 | 0.05 | 0.14 | 0.51 |
| oracle_L44 | PCA | 0.00 | 0.00 | 0.00 | 0.03 |
| oracle_L44 | MR | 0.42 | 0.21 | 0.30 | 0.82 |

**Beside CW-12 (first 30 items, P(names bridge); hers: L42, Oracle off-contract, 2 samples, her matcher).**
whole: ours 0.45 (L40) / 0.47 (L44) vs hers 0.45 (Oracle) / 0.43 (NLA);
J: 0.42 / 0.42 vs 0.48 / 0.42; rest of J: 0.16 / 0.08 vs 0.20 / 0.12;
random: 0.32 / 0.24 vs 0.17 / 0.35. Close to hers on every piece.

**Observations (secondary; the design expected multihop to say little about J).**
- The whole activation's bridge D is large (about 15 nats) and J alone keeps most of it; random 1,024-dim slices keep
  half to two thirds, far more than on brew. Removing J costs much more than removing a random slice at L44
  (2.0 vs 13.8), less so at L40 (6.8 vs 13.4).
- J-piece readouts name the bridge in 91-94% of test samples, and about half of those also restate a cue word: as
  designed (flag 3), D here cannot separate reading the bridge from restating the question.
- M1 holds (whole-activation bridge D CI above 0); M2 (≥ 50% of bridge-naming J readouts restate the cue) is just under
  (about 0.46-0.48 of bridge-namers).

**Correction note (2026-10-06, report 31; appended, the text above is unchanged).** The restatement share should be
taken among bridge-naming test samples that have a checkable cue word (the table's P(restates cue) divides by all
samples with a cue word, and "about half" / "0.46-0.48 of bridge-namers" mixed denominators). Recomputed from
`outputs/007_jsubspace/stage3/{m1,m1_nla_posthoc}/restatement_*.jsonl` with this file's word match:

| reader | J piece | whole |
|---|---|---|
| oracle_L40 | 0.28 (15/53) | 0.56 (27/48) |
| oracle_L44 | 0.24 (13/54) | 0.25 (12/48) |
| nla_L42 (post hoc A2) | 0.15 (8/53) | 0.58 (28/48) |

So M2 (≥ 50% of bridge-naming J readouts restate the cue) is not met (0.28 / 0.24), and not "just under".
