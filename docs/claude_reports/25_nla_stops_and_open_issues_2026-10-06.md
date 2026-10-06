# Why the NLA runs keep stopping, and the other open issues in 007 (2026-10-06)

*For the PI. Evidence: `results/007_jsubspace/analysis/diag_nla_numerics.json` (an engineering diagnostic run on
2026-10-06, about 5 GPU-min, post hoc; it computes no research outcome), the 007 status files and gate logs.*

## 1. Short answer

**The NLA is not scoring wrongly. Its numbers carry more bf16 rounding noise than the Oracle's, and our gates use a fixed
maximum bound (0.25 nats) that was calibrated on the Oracle in 006. For the NLA that bound sits inside its normal noise
tail, so any gate that compares two computations with a different batch layout over enough values will trip by chance.**

- **The scoring is deterministic.** The same call run twice gives bit-identical numbers for both readers.
- **Changing only how activations are batched moves the numbers.** The model runs in bf16, so any change of batch size,
  padding or row layout changes the order of floating-point sums inside the kernels. That shifts every colour log-prob by
  a few hundredths of a nat, and occasionally by a few tenths. For the NLA these shifts are about twice the Oracle's.
- **The research statistic is not affected.** D averages over 183 pairs and 48 tables. The NLA's D_z on 006's path
  equals 006's to four decimals (1.8621 vs 1.8621). Even the fast scorer that failed the gate moved it by only 0.011.
- **The stops are still binding,** because we committed to the gates before the run (rule 12).

## 2. What stopped, and why each time

| run | scorer | gate | value | bound | what the gate compared |
|---|---|---|---|---|---|
| Stage 1a, pre-registered | 2-row fast scorer (mine) | G-repro | max **0.413**, median 0.038 | max ≤ 0.25, median ≤ 0.05 | all 5,544 NLA log-probs against 006's, computed with a different row layout (2 rows per activation instead of 24) |
| Stage 1b, post hoc (A2) | 006's exact scorer | G-score1 | max **0.280**, median 0.046 | same | two scoring paths (`score_seq` vs `score_next`) on 4 `</think>` activations |
| Stage 1a / brew test / multihop, post hoc | 006's exact scorer | all gates | pass (G-repro max 0.152, median **0.0000**) | same | the same layout as 006, so almost everything is bit-identical |

- **The first stop was my mistake.** I replaced the 24-row scorer with a faster 2-row layout. The arithmetic was the
  same, but the batch layout was different, and I checked parity on only 4 activations. Over 4 activations the worst
  difference was 0.178. Over all 5,544 values it was 0.413 (7 values above 0.25, 0.13%).
- **The second stop was the gate itself.** G-score1 compares two different computation paths by design. On `</think>`
  activations the NLA's tail reaches the bound. 006 only ran this gate at `:`, where the NLA scored 0.247 and passed by
  0.003. The same gate on `:` activations still passes today (0.244-0.247).

## 3. The diagnostic

**Size of the differences (|Δ log-prob|, nats).**

| quantity | Oracle L40 | NLA L42 |
|---|---|---|
| same call run twice | 0 (identical) | 0 (identical) |
| batches of 1 vs 2 activations (same scorer, same inputs) | median 0.017, max 0.115 | median 0.037, max 0.136 |
| batches of 4 vs 2 activations | median 0.020, max 0.130 | median 0.041, **max 0.288** |
| G-score1 quantity, 32 activations at `:` | median 0.019, p99 0.093, max 0.132 | median 0.028, p99 0.130, max 0.211 |
| G-score1 quantity, 32 activations at `</think>` | median 0.023, p99 0.097, max 0.129 | median 0.037, p99 0.199, **max 0.255** |
| 2-row fast vs 24-row, all 231 `:` renders | — | median 0.038, p99 0.167, **max 0.413** |
| 24-row rerun vs 006's saved values | — | median 0.0, max 0.152 |

**What the numbers show.**
1. **The noise comes from the batch layout, not from randomness.** Identical calls are identical. Any change of layout
   moves values.
2. **The NLA's noise is about 2× the Oracle's** (median 0.03-0.04 vs 0.02; tail 0.2-0.4 vs about 0.13). The Oracle never
   came near 0.25 in any comparison; the NLA crosses it in three different comparisons.
3. **A maximum grows with the number of values checked.** At `:`, the NLA's worst G-score1 difference is 0.09 over 1
   activation, 0.13 over 4 and 0.21 over 16. A max-based gate's pass or fail therefore depends on how many values it
   looks at. 006's 0.25 bound came from a 4-activation Oracle check (max 0.10-0.12).
4. **The large NLA differences cluster on particular colour tokens:** "gray" at `:` (the fast-scorer worst case: render
   221, " Gray", −6.69 vs −7.10 nats) and "purple" and "blue" at `</think>`. They are not the least likely spellings.
   All the NLA's colour log-probs sit deep in its tail (−17 to −6 nats: after "The text is about" no colour is ever its
   preferred word), while the Oracle often puts one colour near the top (up to −0.5).
5. **Why the NLA is noisier is measured, not explained.** Candidate reasons, not tested here: its colour tokens are
   always tail tokens; its injection adds a vector scaled by the marker's own norm at block 1, computed in bf16; it is a
   separately fine-tuned model with a longer prompt.

## 4. What this means

- **The NLA's results are trustworthy where they exist.** The post hoc reruns (Stage 1a, brew test, multihop) use 006's
  exact layout and pass every gate. Their D statistics are stable to about 0.01 nats.
- **What is missing because of the stops:**
  - the NLA in the pre-registered family (it was rerun post hoc and is labelled so);
  - the NLA at `</think>` (Stage 1b).
- **Nothing in the Oracle's results depends on this.** The Oracle's noise stays below 0.13 nats in every comparison, and
  its decisive contrasts are several nats.
- **The gate design is flawed for the NLA.** A fixed maximum over many values, calibrated on a quieter reader, cannot tell
  a bookkeeping bug (wrong nats across many values) from rounding noise (rare tenths of a nat on a few values).

## 5. Options for the NLA (your call; each is post hoc)

1. **Rerun Stage 1b's NLA with G-score1 evaluated as 006 did** (on `:` activations, where it passes). About 2.2 GPU-h.
   Smallest change, but a change made after seeing the stop.
2. **Replace the max-based gates for the NLA with a noise-aware rule** and rerun 1b. For example: median ≤ 0.05 and
   p99 ≤ 0.25, or the D statistic reproducing within 0.05 nats. Today's diagnostic would be the calibration. About 2.2 GPU-h.
3. **Leave the NLA out of `</think>`.** The `</think>` question is secondary, and the Oracle answers it.
4. **For future experiments:** calibrate each reader's gate on a diagnostic like this one before pre-registering, and
   never compare scorers with different batch layouts against a max bound.

## 6. Other issues we are facing

| # | issue | effect on 007 | cause | what would fix it |
|---|---|---|---|---|
| 1 | **Positive-control failures make contrasts uninformative.** Random 1,024-dim projections and PCA-4 of natural colour activations are not decodable by the readers | Oracle L44 and the NLA end "unresolved" on P1-P3; only Oracle L40 reaches a decision | small or random projections are far from anything the readers were trained on, so the positive control is strict for exactly the pieces the controls need | control pieces that keep more of the natural geometry (larger random subspaces, or controls matched on decodability), or a positive control calibrated per piece type |
| 2 | **J dependence is reader-specific** (a finding, not a bug) | the confirmed result is the Oracle's; the NLA's z signal is spread over the activation | — | a third reader (the paper's J-lens itself, or an SAE readout) would show which reader is typical |
| 3 | **J is the token-readout subspace, and z is the model's runner-up answer** | "the Oracle reads a held intermediate" and "the Oracle reads output-facing content" fit equally well | brew's intermediate is also an answer candidate | a family whose intermediate is never a plausible answer (e.g. a non-colour bridge with a fixed answer vocabulary) |
| 4 | **Template lens uninformative** | it fails its own colour sanity check at L40-44 (own colour first for 3-5 of 11) | its templates are built at "about to say w" positions; our check uses "Tell me about {colour}" | a sanity check at positions where the model is about to say the colour |
| 5 | **CPU token-lens gate G-002 fails** at the pre-registered 1e-4 | the exploratory mean-removed lens analysis is not interpreted | my code used 1e-3; the 3 failing checks are near-zero J_NP cosine scores, where a relative error is ill-defined | a relative-or-absolute tolerance, fixed before running |
| 6 | **Multihop cannot test J cleanly** | its D is confounded by restating the question (half of the bridge-naming readouts restate a cue); PCA-k could not be Δ-matched (rank ≤ 7); 9 clusters | the bank has no counterfactual partners and few items per template | a counterfactual multihop bank (same question, swapped cue) |
| 7 | **Relational stopped after its profile** | only 12 of 50 scenes passed the strict gate, and no read-layer position had high patch leverage at L40-44 | the model is unreliable on the bare cloze; the composed answer reaches the final token only from about L56 | a lenient gate, or reading at L56-60 |
| 8 | **Process mistakes this session** | listed in CLAUDE.md's silly-mistakes log and the status files | — | — |

Issue 8 in detail:
- the fast scorer's parity was checked on 4 activations (the first NLA stop);
- Stage 0's geometry gate first compared unlike draw sets;
- the role-collision definition first differed from 006's;
- the template-lens convention was checked after scoring;
- the GPU queue's wait command matched itself and idled the GPU for about 2 minutes;
- the G-002 threshold was coded at 1e-3 instead of 1e-4;
- Stage 1a run r1 was aborted after its prompt-only floor had been printed;
- the multihop rerun crashed once on a wrong function call.

## 7. Recommended next step

If the NLA at `</think>` matters to you, take option 2 (a noise-aware gate, calibrated on today's diagnostic) rather than
option 1: it fixes the cause rather than this one instance. Otherwise take option 3, and carry the gate lesson (§5.4)
into the next design. In either case, the issue with the most leverage for the project is #1. The controls that would
discriminate the models are the pieces the readers cannot read, so the next J-type experiment should build its controls
from pieces the readers can demonstrably decode.
