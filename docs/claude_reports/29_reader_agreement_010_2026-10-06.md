# 010: reader agreement on 006/007, and Oracle L44 vs NLA L42 on multihop (2026-10-06)

**EXPLORATORY, post hoc, CPU only, on saved LL matrices.**
- No model, judge or API was used.
- A subagent wrote and ran the analysis. I re-derived the headline correlations from its per-pair D file:
  - Oracle L40 vs J_CB L40: 0.634 (it reported 0.63);
  - Oracle L40 vs the model's output: 0.302 (0.30);
  - 007's Oracle L40 J piece vs J_CB: 0.795 (0.80);
  - 007's Oracle L40 J piece vs the model's output: 0.205 (0.21);
  - Oracle L60 vs the model's output: 0.822 (0.82);
  - the Oracle L40 D_z table mean: 3.221 (as in 006).
- **Every earlier number it touched reproduces:** 125 brew checks (max |Δ| 1.4e-7), the multihop bridge D, and the
  consistent-denominator restatement shares of report 26.
- **Code:** `scripts/010_reader_agreement/`. **Outputs:** `results/010_reader_agreement/analysis/`
  (`agreement.json`, `per_pair_D.json`, `multihop_l44_vs_l42.json`).
- **Statistics:** 95% table-cluster bootstrap (10,000 draws) over brew dev (48 tables, 183 pairs); no multiplicity
  correction.

## Findings

1. **The NLA and the Oracle agree only weakly on which pairs carry z** (brew `:`).
   - Per-pair r: NLA–Oracle L40 0.24 [0.05, 0.43]; NLA–Oracle L44 0.30 [0.14, 0.47].
   - Kappa on signs is 0.05–0.10. The 86% same-sign rate is what each reader's base rate predicts.
   - The two Oracle layers agree with each other: 0.81 [0.72, 0.87].
   - The NLA is the less reliable instrument: its two prefixes agree at r 0.67, against 0.94 / 0.84 for the Oracle.
2. **The agreement depends on the NLA's prefix.** Post hoc: p1 is 006's pre-registered secondary, colour-forcing
   prefix.
   - Under p1, NLA–Oracle r rises to 0.55–0.60 at `:` and 0.71–0.75 at `</think>`; paired change +0.30 to +0.53.
   - It moves toward the J lens (J_CB L42: 0.33 → 0.56), not toward the model's output (0.07 → 0.19).
3. **The model's own output does not explain what the mid-layer readers share.**
   - Partialling out the model's per-pair D_z (006b) barely changes reader–reader r: NLA–Oracle 0.24 → 0.23 and
     0.30 → 0.29; Oracle L40–L44 0.81 → 0.80.
   - Only the late Oracle follows the model's output: L60 vs model r 0.82; L56 0.62.
4. **The Oracle tracks what the J lens reads more than the model's next-token output; the NLA tracks neither
   strongly** (corrected 2026-10-06, report 31: the heading said "The mid-layer readers track what the J lens reads, not the model's
   next-token output", which holds for the Oracle only).
   - Oracle L40 vs J_CB L40: 0.63 at `:` (0.75 at `</think>`). Vs the model's output: 0.30 (0.18).
   - 007's Oracle J piece vs J_CB: 0.80. Vs the model: 0.21.
   - The NLA vs J_NP-cosine L42: 0.37 (vs J_CB L42: 0.33). Vs the model: 0.07. So "0.63–0.80 with the J lens" is the
     Oracle's range only; the NLA's is 0.33–0.37.
5. **Reader D_z has no relation to the `:` cell's single-cell leverage across pairs; the model's own output has a modest
   one.**
   - Reader vs S: Oracle L40 0.02; NLA −0.08. Model vs S: 0.24–0.33 (L44 0.24, L42 0.27, L40 0.33; corrected 2026-10-06, report 31:
     was 0.27–0.33).
   - Caveat: 82 gated pairs, and S at `:` only ranges from 0.22 to 0.67, so the test has little power.
6. **Across cells, the mid-layer readers' change from `:` to `</think>` follows neither the model's output nor the
   change in leverage.**
   - Oracle L40 rises by +2.79 nats. The model changes by +0.23; S falls by 0.43.
   - Per-pair r with the model's change: 0.02. With the change in S: −0.04.
   - The late Oracle (L56/L60) does follow the model: r 0.57–0.68.
7. **007's J piece does not change Oracle–NLA agreement relative to the whole activation** (differences of +0.06 on dev
   and 0.00 / −0.12 on test, all CIs include 0). Between the two Oracle layers, J agrees as much as whole (0.81), while
   rest_J collapses (0.31).
8. **Multihop** (secondary; test; 40 items; 9 key clusters).

   | piece | bridge D, Oracle L44 | bridge D, NLA L42 |
   |---|---|---|
   | whole | 14.8 | 14.7 |
   | J | 13.5 | 14.5 |
   | rest_J | 2.0 | 6.1 |
   | random 1,024-dim (mean) | 7.8 | 10.6 |
   | mean-removed | 8.7 | 16.5 |

   - The two readers match on whole and J. ~~The NLA keeps much more of the bridge signal outside J~~ (qualified
     2026-10-06, report 31): as shares of each reader's whole-activation bridge D (`stage3.json`,
     `posthoc_A2_stage3.json`), the random slice keeps 0.73 for the NLA vs 0.52 for the Oracle L44 (0.59 at L40), and
     rest of J keeps 0.42 vs 0.13 (0.47 at L40); J keeps 0.99 vs 0.91 (0.88 at L40). The NLA's excess over the Oracle
     L44 is real in nats (random-slice difference −2.9 [−3.8, −2.3], negative in 40 of 40 items), but against the
     Oracle L40 the gap is small.
   - From the random slice the NLA names the bridge in 43/80 samples, against the Oracle L44's 14/80. **PI, post hoc
     (2026-10-06, hand-read of the mirrored readouts):** 64 of the NLA's 80 random-slice readouts are weather / airport
     boilerplate, and 38 of its 43 hits fall inside them, naming the right country. Read these hits as the country's
     identity surviving in the slice, not as the NLA reading the question.
   - The whole-activation NLA readouts restate a cue word more often than the Oracle L44's (28/48 vs 12/48), but about
     as often as the Oracle L40's (27/48) (corrected 2026-10-06, report 31: was "far more often", against L44 only). Its J readouts
     restate less (8/53, against 15/53 and 13/54).
   - **PI, post hoc (2026-10-06):** on the whole activation each reader names the bridge on 36 of 40 test items; the
     regex agrees with a hand-read of all 240 whole readouts. Failures: most-populous-country (all three readers);
     Kremlin and IKEA / Guinness (Oracle L40); Kremlin / LEGO / Golden Gate (Oracle L44); Golden Gate / Griffith / Zeus
     (NLA). The Oracle's rest-of-J readouts often name a wrong country (only-wrong vs right: L40 26 vs 34; L44 41 vs 10).
     Details in `docs/reports/007_jsubspace.md` §9.

## What this changes

- **006b's reading needs a refinement.**
  - "Consistent with reading the model's output-facing distribution" holds at the level of averages: z is the model's
    runner-up answer, and the readers track z.
  - But pair by pair, the mid-layer readers (Oracle L40/L44, NLA L42) do not track the model's actual next-token output
    (r 0.21–0.30 for the Oracle, 0.07 for the NLA). The Oracle tracks what the J lens reads from the same activation
    (r 0.63–0.80): content the activation is, on average across contexts, poised to make the model say. The NLA tracks
    it only weakly (0.33–0.37) (corrected 2026-10-06, report 31: said both mid-layer readers track the J lens).
  - Only the late Oracle tracks the model's output.
  - So the brew confound is "token-readable (J-lens) content, which on brew includes z as an answer candidate" more than
    "the model's own next-token distribution". 009's brew-code family is the test of this: there, no colour is ever an
    answer.
- **The NLA and the Oracle are different instruments.**
  - The Oracle is J-concentrated and layer-consistent.
  - The NLA is prefix-sensitive and spreads more of its signal outside J (partly weather boilerplate that keeps the
    country, item 8). Its whole readouts restate more than the Oracle L44's, not more than the Oracle L40's
    (corrected 2026-10-06, report 31: was "spreads its signal outside J, and restates more").
  - Their weak per-pair agreement at `:` (r ≈ 0.24–0.30) is mostly not shared output content.
- **Neither reader's signal follows single-cell leverage**, across pairs or across cells. This is the North-Star
  quantity, measured on brew with little power.

## Caveats

- Exploratory: about 350 instrument pairs per block, uncorrected. Read the patterns, not single p-values.
- The partial correlation controls only a linear term in the model's per-pair D.
- Agreement between a lens and a reader shows shared readable content, not use.
- Scorer noise attenuates every r. The NLA is noisier, so its correlations are biased down more than the Oracle's.
- Multihop: 9 key clusters, with capital dominating (66 of 104 pairs); restatement is word-matched; dev was touched.
