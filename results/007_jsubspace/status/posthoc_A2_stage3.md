# 007 post hoc (A2) — NLA L42, Stage 3 (multihop) — status (2026-10-06)

**Label: post hoc** (A2). Secondary stage; the design expected multihop to say little about J.

**What ran.** The NLA on 006's exact scorer over the 21 v1 pieces of Stage 3's saved multihop pieces (the Oracle's inputs;
recompute deviation 0.0), each key group's bridge and target strings, the positive control on 50 bridge activations,
restatement samples. GPU time 75.8 min. The first launch crashed on a coding error before any gate (disclosed).

**Gates — pass.** G-score1 0.061, G-score2 0.028, G-inject incl. scale check.

**Bridge D (nats), test split.** whole 14.7 [11.4, 16.4]; J 14.5;
rest of J 6.1; PCA-7 -0.0; mean-removed 16.5; random (mean of 10) 10.6; rest of random 15.9.

**Restatement (test; share of samples).**

| piece | P(names bridge) | P(restates cue) | P(bridge without cue) |
|---|---|---|---|
| whole | 0.80 | 0.59 | 0.36 |
| J | 0.89 | 0.16 | 0.80 |
| rest_J | 0.23 | 0.04 | 0.18 |
| R0 | 0.54 | 0.04 | 0.57 |

CW-12 comparison (first 30 items, names bridge): whole 0.39 (hers, NLA: 0.43); J 0.42 (0.42);
rest of J 0.00 (0.12); random 0.26 (0.35).

**Observation.** On multihop the NLA behaves like the Oracle: J keeps essentially all of the bridge signal, random slices
keep about half, and removing J halves it. Unlike brew, where the NLA's z signal did not depend on J, here both readers
lean on J — but the bridge strings are cued by the question, so this does not separate reading the bridge from
restating the prompt.
