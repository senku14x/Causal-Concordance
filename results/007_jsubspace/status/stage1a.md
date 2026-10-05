# 007 Stage 1a — status (brew dev, `:`; 2026-10-05)

**What ran.** Oracle `olens_s3d_rl600` at L40 and L44, every one of the 26 pieces × 231 renders (48 dev tables), the
no-activation row, the positive control (35 piece types incl. A1's frozen brew operators), the prompt-only reader floor,
the generic (wikitext) mean, qualitative samples (8 tables × 2 renders × 7 pieces × k = 2). GPU time 14.6 min
(run r2; the scorer uses 2 rows per activation instead of 24, exact arithmetic, parity-gated; execution note §20).

**Gates.**
- Oracle L40: G-score1 max 0.146 (fast path vs the 24-row scorer 0.085); G-score2 0.087;
  G-repro max 0.168 / median 0.020, D_z 3.222 vs 006 3.221; G-inject incl. the 0.5×/2× scale check — **all pass**.
- Oracle L44: G-score1 max 0.077 (fast path 0.093); G-score2 0.061;
  G-repro max 0.244 / median 0.026, D_z 3.244 vs 3.235; G-inject — **all pass**.
- **NLA L42: G-repro FAILED** — max |ΔLL| vs 006 = 0.413 nats (bound 0.25); median 0.038 (≤ 0.05), D_z 1.851 vs 006 1.862,
  G-score2 0.067, G-score1 0.227 and the fast-path check on 4 activations 0.178 passed. By rule 12 and A1 the NLA's path
  is **disabled for every later stage** (brew test, 1b, multihop, relational); no NLA piece was scored. Probable cause:
  my 2-row scorer changes the bf16 batch shape; its parity check covered only 4 activations and missed the tail
  (006's own NLA G-score1 already reached 0.247). Not repaired mid-run. A post hoc NLA run on 006's exact 24-row
  path is the PI's call (it would be labelled post hoc).

**Primary family (Holm over the pre-registered 15; the NLA's 5 enter as p = 1).** Contrast estimates in nats,
95% table bootstrap CI, category:

| reader | P1 J − PCA-4 | P2 J − random | P3 rest-random − rest-J | P4 mean-removed − whole | P5 J∘c − random∘c | mapping |
|---|---|---|---|---|---|---|
| Oracle L40 | +5.32 [+4.54, +6.17] **+** | +5.25 [+4.55, +6.01] **+** | +1.81 [+1.50, +2.12] **+** | -0.30 [-0.85, +0.25] **unresolved** | +2.51 [+1.88, +3.15] **+** | Raw: contradicted, Visible: contradicted, H_J: supported |
| Oracle L44 | +4.70 [+4.09, +5.35] **uninformative** | +4.51 [+3.97, +5.11] **uninformative** | +1.85 [+1.46, +2.30] **uninformative** | -0.90 [-1.40, -0.39] **−** | +2.69 [+2.11, +3.27] **+** | Raw: contradicted, Visible: contradicted, H_J: unresolved |

- Oracle L44's P1–P3 are uninformative under A1's rule: rest of J and PCA-4 fail their positive control with D_z CIs
  that include 0, and so does one random draw (R3). With random draws 0–2 only (A1's replication comparator),
  L44's P2 becomes `+`; L40 is unchanged.
- **Dev only; nothing is confirmed until the brew test replicates it.**

**Per-piece D_z (nats).**

| reader | whole | J | rest J | PCA-4 | rest PCA | LL_gain | rest LL | MR | GMR | J∘c | random (mean of 10) | rest random (3) | random∘c (3) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Oracle L40 | 3.22 | 5.76 | 0.30 | 0.44 | 2.23 | 0.94 | 2.34 | 2.93 | 3.70 | 3.82 | 0.52 | 2.11 | 1.32 |
| Oracle L44 | 3.24 | 4.96 | 0.08 | 0.26 | 2.37 | 0.29 | 2.22 | 2.35 | 3.90 | 3.64 | 0.45 | 1.93 | 0.95 |

**Observations (dev; decodability through the Oracle's likelihood, not use):**
- The J piece (9% of the energy, about a quarter of the member–partner difference) carries a *larger* z signal than the
  whole activation, and removing it leaves almost none (rest of J 0.30 / 0.08), although the rest holds 91% of the
  energy and 75% of the difference. Mean removal does not amplify the Oracle's z signal (P4 unresolved / −), against the
  Visible model; J beats random also after the same mean-direction removal (P5 `+` at both layers).
- Role pattern of the J piece: D_answer 3.2 / 3.0, D_start 0.1 / 0.2, off-trajectory D 0.13 / 0.16 — output-facing colours
  (z and the answer), not the start colour in the prompt.
- Positive control mirrors this: J projections of natural colour activations are decodable (LOO-prior ID 1.0 / 0.83),
  most random 1,024-dim projections and PCA-4 are not.
- Qualitative (`qualitative_oracle_L*.jsonl`): J-piece readouts name colours ("黑 (Black) → White"); rest-of-J readouts
  keep the two-step task structure without colours; random and PCA-4 pieces drift off-topic.
- **Reading constraint (design §15/§17):** J is the token-readout subspace and z is the model's own runner-up at `:`
  (006b). "The Oracle reads J-readable, output-facing colour content" fits all of this; it does not show a held
  intermediate or anything about model use (Stage 2 / 2b address the model side).
- Prompt-only reader floor D_z 1.07 (start 2.07, answer 1.73; off −0.02); 006b's task-level D_z 3.76.
- Template lens: its pre-registered colour sanity check fails at L40/42/44 (own colour first for 3–5 of 11) → uninformative
  at these layers by the frozen rule (descriptive scores in `template_lens_stage1a.json`).

**Unexpected.** The NLA gate failure above (my acceleration; logged as a silly mistake). The GPU queue's first wait
command matched its own command line and idled the GPU ~2 min before Stage 2 (killed by hand).
