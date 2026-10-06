# 007 post hoc (A2) — NLA L42, brew test replication — status (2026-10-06)

**Label: post hoc** (A2). The pre-registered replication (Oracle only) is unchanged in `results/007_jsubspace/status/brew_test.md`.

**What ran.** The NLA on 006's exact scorer over the 15 test pieces × 227 renders (49 test tables), no-activation row,
positive controls with the test-stage operators. GPU time 75.9 min.

**Gates — all pass.** G-score1 max 0.2437 / median 0.0451; G-score2 0.065; G-inject incl. scale
check (exact); G-repro not applicable (new captures).

**Confirmation in the post hoc full family (Holm/TOST over 15 with real p-values; same category on dev and test).**

| reader | contrast | dev | test | confirmed |
|---|---|---|---|---|
| oracle_L40 | P1 | + (+5.32) | + (+4.28 [+3.46, +5.09]) | **yes** |
| oracle_L40 | P2 | + (+5.25) | + (+4.46 [+3.69, +5.21]) | **yes** |
| oracle_L40 | P3 | + (+1.81) | + (+1.62 [+1.28, +1.96]) | **yes** |
| oracle_L40 | P4 | unresolved (-0.30) | unresolved (+0.18 [-0.45, +0.82]) | no |
| oracle_L40 | P5 | + (+2.51) | + (+2.09 [+1.47, +2.67]) | **yes** |
| oracle_L44 | P1 | uninformative (+4.70) | + (+3.80 [+3.17, +4.41]) | no |
| oracle_L44 | P2 | uninformative (+4.51) | + (+3.88 [+3.27, +4.50]) | no |
| oracle_L44 | P3 | uninformative (+1.85) | uninformative (+1.81 [+1.47, +2.17]) | no |
| oracle_L44 | P4 | − (-0.90) | − (-0.72 [-1.23, -0.19]) | **yes** |
| oracle_L44 | P5 | + (+2.69) | + (+1.86 [+1.21, +2.48]) | **yes** |
| nla_L42 | P1 | uninformative (+0.96) | uninformative (+1.36 [+0.83, +1.86]) | no |
| nla_L42 | P2 | uninformative (+0.69) | uninformative (+0.77 [+0.38, +1.15]) | no |
| nla_L42 | P3 | unresolved (+0.17) | unresolved (-0.04 [-0.39, +0.30]) | no |
| nla_L42 | P4 | unresolved (+0.36) | unresolved (+0.18 [-0.45, +0.83]) | no |
| nla_L42 | P5 | unresolved (+0.52) | + (+0.85 [+0.23, +1.49]) | no |

- Oracle L40: H_J confirmed (as in the pre-registered record). Oracle L44: P4 (−) and P5 (+) confirmed in this family
  (in the pre-registered family P4 on test was "unresolved": with the NLA's real p-values ahead of it, Holm's later
  thresholds are slightly more lenient; both records are kept).
- **NLA L42: nothing confirmed.** On test P5 is `+` (J∘centred 1.92 vs random∘centred about 1.1), unresolved on dev;
  P1/P2 uninformative (PCA-4 and random pieces fail the NLA's positive control); P3 and P4 unresolved on both.
- NLA test D_z: whole 1.70; J 1.06; rest of J 0.78; random 0-2 0.29;
  rest of random 0.74; mean-removed 1.88. The same shape as dev: J keeps about 60% of
  the signal and removing it costs about as much as removing a random slice.

**Reading (post hoc).** The Oracle's J dependence replicates; the NLA's z signal is spread more evenly over the
activation and does not depend on J beyond a modest enrichment. J dependence is a property of the reader.
