# 007 brew test replication — status (confirmatory; AMENDMENT_02 use 1; 2026-10-05)

**What ran.** SEC captures of the 3 emission cells at L40/42/44 for every test render (49 tables: 49 members + all 178
token-aligned partner candidates, O1), the 15 test pieces with dev-frozen operators (J, random draws 0-2 and their
complements, PCA-4 fitted on dev, mean-removed with the leave-one-test-table-out mean, P5 pieces), the Oracle L40/L44
(the NLA is disabled since Stage 1a, A1), positive controls with the test-stage operators, the prompt-only floor.
GPU time 7.4 min (capture 1 min + scoring 6.3 min).

**Gates.** Cell-region assertion on every render; G-recon on the test pieces (≤ 3.4e-8 / 6.8e-10); Oracle L40: G-fastpath
0.085, G-score1 0.111, G-score2 0.076, G-inject incl. scale check;
Oracle L44: G-fastpath 0.111, G-score1 0.079, G-score2 0.081, G-inject — **all pass**
(G-repro not applicable: new captures).

**Confirmation (design §7.5: a dev category counts only if the test gives the same category; the A1 3-draw dev
comparator alongside).**

| reader | contrast | dev (10 draws) | dev (draws 0-2) | test | confirmed |
|---|---|---|---|---|---|
| oracle_L40 | P1 | + (+5.32) | + | + (+4.28 [+3.46, +5.09]) | **yes** |
| oracle_L40 | P2 | + (+5.25) | + | + (+4.46 [+3.69, +5.21]) | **yes** |
| oracle_L40 | P3 | + (+1.81) | + | + (+1.62 [+1.28, +1.96]) | **yes** |
| oracle_L40 | P4 | unresolved (-0.30) | unresolved | unresolved (+0.18 [-0.45, +0.82]) | no |
| oracle_L40 | P5 | + (+2.51) | + | + (+2.09 [+1.47, +2.67]) | **yes** |
| oracle_L44 | P1 | uninformative (+4.70) | uninformative | + (+3.80 [+3.17, +4.41]) | no |
| oracle_L44 | P2 | uninformative (+4.51) | + | + (+3.88 [+3.27, +4.50]) | no |
| oracle_L44 | P3 | uninformative (+1.85) | uninformative | uninformative (+1.81 [+1.47, +2.17]) | no |
| oracle_L44 | P4 | − (-0.90) | − | unresolved (-0.72 [-1.23, -0.19]) | no |
| oracle_L44 | P5 | + (+2.69) | + | + (+1.86 [+1.21, +2.48]) | **yes** |

- **Oracle L40: the J-content model (H_J) is supported on dev and on test — confirmed.** Raw and Visible are
  contradicted on both.
- Oracle L44: H_J unresolved on both splits (P3 uninformative on test: rest of J fails its positive control with a D_z
  CI that includes 0; on dev P1-P3 were uninformative and P4 was −). P5 confirmed.
- Gated pairs only (O1 secondary, 77 test pairs over 35 tables): the same direction for every contrast; L40 H_J supported.

**Test D_z (nats), Oracle L40 / L44.** whole 2.87 / 3.14; J 5.03 / 4.34;
rest of J 0.29 / 0.08; PCA-4 0.75 / 0.54; MR 3.05 / 2.42;
J∘c 3.63 / 2.96; random 0-2 0.57 / 0.46;
random∘c 0-2 1.54 / 1.10. J piece roles (L40): D_start -0.39, D_answer 2.96, D_off -0.12.

**What the confirmed result says (scoped).** On brew at `:`, the Oracle's likelihood-based tracking of the intermediate
colour is carried by the J (token-readout) subspace: J alone carries more z signal than the whole activation, removing
it leaves almost none, and J beats a Δ-matched data subspace, dimension-matched random subspaces, and random after the
same mean-direction removal. Decodability through one reader (the NLA is out), one layer confirmed (L40), one family.

**What it does not say.** Nothing about use: on the model side, J's influence on the answer tracks its content share
(Stage 2) and the J part of a class-mean z direction is not privileged at matched magnitude (Stage 2b) — a reader-model
dissociation. J is the token-readout subspace and z is the model's runner-up at `:` (006b), so "the Oracle reads
output-facing colour content" fits as well as "it reads a held intermediate". Random and PCA-4 pieces fail their
positive control (their low D may partly be reader insensitivity, flagged); P3 and P5, whose pieces pass, carry the
cleanest form of the result.
