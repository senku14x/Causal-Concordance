# 006 — Do the verbalizers' likelihoods track z at the brew `:` cell?

*Design:* `docs/design_specs/006_reader_likelihood.md` (v1 pre-registered a92dfa6; amendment v1.1 f5e6dff, before any outcome). *Branch:* `006-reader-likelihood`. *Run:* `run1`, 82.5 min. *Data:* the saved 001 captures, 48 dev brew tables, 231 member/partner renders. *Summary:* `results/006_reader_likelihood/analysis/summary.json`.

## Gates

- **G-score1** (full-sequence scorer vs the single-token scorer on single-token spellings): max |difference| Oracle L40 0.102, Oracle L44 0.089, Oracle L56 0.074, Oracle L60 0.100, NLA L42 0.247 nats (bar 0.25 after amendment v1.1; the NLA's 0.247 is close to it). Medians 0.023, 0.022, 0.018, 0.023, 0.041 (bar 0.05).
- **G-score2** (float32 vs the readers' own logits): max 0.106 nats (bar 1.0).
- **G-pc** (12-way colour identification on natural "Tell me about {colour}" activations, bar 0.5): NLA L42 p0 0.50; Oracle L40 p0 0.33; Oracle L44 p0 0.42; Oracle L56 p0 1.00; Oracle L60 p0 1.00. **The Oracle fails at L40 and L44**, so by the pre-registered rule its primary result (T2) is uninformative and shown descriptively only. The NLA passes exactly at the bar.

## Tracking at the `:` cell (mean D over 48 tables [95% CI] (tables + / −); prefix p0 primary)

| Reader, prefix | G-pc | z | start | answer | z, shuffled | forced choice z (member / no-activation) |
|---|---|---|---|---|---|---|
| NLA L42, p0 | 0.50 | +1.86 [1.41, 2.34] (44/4) | +0.41 [0.08, 0.74] (30/18) | +0.59 [0.16, 1.01] (34/14) | −0.20 [−0.73, 0.34] | 0.27 / 0.23 |
| NLA L42, p1 | 0.50 | +1.53 [1.20, 1.90] (46/2) | +0.41 [0.17, 0.65] (31/17) | +0.55 [0.26, 0.87] (35/13) | −0.02 [−0.32, 0.28] | 0.33 / 0.27 |
| Oracle L40, p0 | 0.33 (fails) | +3.22 [2.78, 3.72] (48/0) | −0.05 [−0.40, 0.32] (23/25) | +1.73 [1.17, 2.31] (40/8) | −0.19 [−0.61, 0.25] | 0.35 / 0.25 |
| Oracle L40, p1 | 0.25 (fails) | +3.83 [3.41, 4.27] (48/0) | +0.32 [−0.08, 0.70] (29/19) | +1.89 [1.33, 2.48] (41/7) | −0.23 [−0.67, 0.22] | 0.40 / 0.25 |
| Oracle L44, p0 | 0.42 (fails) | +3.24 [2.85, 3.65] (48/0) | +0.72 [0.28, 1.19] (33/15) | +1.50 [0.88, 2.15] (39/9) | −0.34 [−0.78, 0.15] | 0.44 / 0.25 |
| Oracle L44, p1 | 0.42 (fails) | +3.66 [3.28, 4.04] (48/0) | +0.91 [0.49, 1.34] (35/13) | +1.47 [0.89, 2.07] (40/8) | −0.40 [−0.91, 0.14] | 0.44 / 0.25 |
| Oracle L56, p0 | 1.00 | +6.40 [5.15, 7.59] (45/3) | −4.02 [−5.43, −2.69] (6/42) | +17.98 [16.76, 19.16] (48/0) | −0.16 [−1.89, 1.61] | 0.00 / 0.17 |
| Oracle L56, p1 | 1.00 | +6.04 [5.25, 6.82] (47/1) | −2.42 [−3.38, −1.45] (8/40) | +13.22 [12.37, 14.10] (48/0) | +0.01 [−1.27, 1.33] | 0.00 / 0.23 |
| Oracle L60, p0 | 1.00 | +6.41 [5.55, 7.26] (47/1) | −2.12 [−3.26, −1.01] (12/36) | +15.04 [13.92, 16.14] (48/0) | −0.17 [−1.66, 1.35] | 0.02 / 0.27 |
| Oracle L60, p1 | 1.00 | +5.80 [5.15, 6.46] (48/0) | −1.11 [−1.90, −0.32] (15/33) | +11.11 [10.20, 11.99] (48/0) | 0.00 [−1.15, 1.17] | 0.00 / 0.29 |

**002's token lenses at the same cell, same statistic (log-score units differ by reader):** J_NP cosine L40 z +0.58 (48/0), start −0.03 (21/27), answer +0.29 (42/6); J_CB L40 z +1.64 (48/0); logit lens L42 z +0.41 (38/10).

## Pre-registered rules

- **Holm (primary family, one-sided bootstrap p):** NLA L42 p = 0.0000 vs 0.0083 → rejects; Oracle L40 p = 0.0000 vs 0.0125 → rejects; Oracle L44 p = 0.0000 vs 0.0250 → rejects.
- **T1, NLA L42 tracks z:** supported (agent prior 0.55).
- **T2, Oracle tracks z (L40 or L44):** uninformative: both fail G-pc (prior 0.60). Descriptively the Oracle's likelihood tracks z in 48 of 48 tables at both layers.
- **T3, start not tracked:** does not hold (prior 0.55): the NLA's likelihood also tracks the start colour, though less than z. The token lenses do not.
- **T4, forced-choice z > 0.4:** not met (prior 0.25).
- **T5, G-pc passes:** NLA True (prior 0.65); Oracle L40 False, L44 False (prior 0.85).

## Reading

- **The NLA's likelihood registers z at the `:` cell.** Within tables, swapping the cell from the member to a partner moves the NLA's likelihood toward the cell's own intermediate colour in 44 of 48 tables. The shuffled control is about zero. This is the cell where 002 found the verbalizers almost never *state* z in its role. The gap is between what the reader's likelihood carries and what it says. Wording: z is decodable from the NLA's likelihood there; not "the NLA represents z", and not use by the model.
- **The Oracle's likelihood tracks z even more consistently (48/48 at L40 and L44),** but its positive control failed at those layers: it identifies natural colour activations only 33–42% of the time. By the rule, this is reported, not counted. At L56/L60 the Oracle passes the control (100%). There its likelihood is dominated by the answer colour (+11 to +18 nats), with z tracked as well and start anti-tracked.
- **Not as clean as the lenses on role.** The NLA and Oracle L44 likelihoods also follow the start colour and the answer. Forced choice among the 5 candidates picks z on member renders only 27–44% of the time (chance 20%; no-activation reference 23–27%). The readers carry z-specific information, but not as a dominant "this is the intermediate" signal.

## Caveats

- The NLA is `nla-rl` iter_000400; 002's verbalizer claims came from `nla-L42`, so "rarely states z" and "likelihood tracks z" are measured on different checkpoints.
- G-pc for the NLA is exactly at the bar (6 of 12). G-score1 for the NLA is close to its bar (0.247 vs 0.25).
- Saved SEC captures, not stock (cos ≥ 0.99985 per 002's bridge). Dev split only; one family; prefixes fixed in advance (p0 primary, p1 agrees).
- Likelihood tracking is decodability through the reader, not a claim and not use by the model.

## Mistakes (logged in CLAUDE.md)

- G-score1's first tolerance (1e-3) ignored the documented bf16 floor. The gate stopped the first launch; the diagnosis and amendment (v1.1) came before any outcome was read.
- The lens-comparison table was empty in the first analysis output (wrong key format for 002's scores.json); fixed (descriptive only).
- The run took 82.5 min against a 15–25 min estimate.
