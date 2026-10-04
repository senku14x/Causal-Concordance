# 005 dev-only pilot note (run `pilot1`: pilot 20:17–22:59 UTC, step 7b 22:59–23:03, step 8 from 23:04)

*Design:* `docs/design_specs/005_planted_concepts.md` §16–§16.4 (questions, rules and agent priors committed
before each run: 0b0ca0b, ba31daa, c7cde05, bf1da37). *Data:* `outputs/005_planted_concepts/pilot/pilot1/`;
summary `results/005_planted_concepts/analysis/pilot_summary.json`. **Dev concepts only (10); descriptive; no
test-concept outcome exists.** The full report with charts and raw readouts is the published report page.

**The short version**

- Qwen reports planted concepts only by the paper's own metric. Asked “the thought is about "…"”, Qwen almost never says the planted concept: it has a canned answer (“apple”), and neither prefill removes it. But the concept's log-probability at that position rises reliably above what random directions do (Q1 yes, 4/4 blocks). The report gate as written failed; the amendment makes log-prob the report measure.
- No clear J-space privilege for report on these dev concepts. At equal size the J part beats the non-J part at low and middle doses, the order reverses at high doses, and the two reach similar peaks. This is a preview on 10 dev concepts, not the test (M1 runs on the 45 test concepts).
- The privilege clamp works only by blocking the output end. Clamping the concept's J coordinates all the way to L62 removes much of the non-J part's effect (beating a random-token clamp), but stopping at L44 removes nothing beyond the random clamp. The concept's J direction turns into its output direction with depth, so the full-range clamp blocks the answer itself. On these concepts there is no sign that the non-J effect re-enters J-space within the workspace layers.
- The readers see the planted concept only when handed the written cell. On direct reads at the top dose the best reader names the full vector in 93% of samples and the non-J part in 90%; at the report position no text reader exceeds 8%, no lens puts the concept in its top 10 at any layer up to L62, and the probe never classifies it as present. The model's report shifts without a readable concept ever forming there.
- The readers are less sensitive than the model, not more. The model's report responds from s = 1/128 or 1/64; the first dose at which any reader names any planted condition in more than 10% of direct reads is s = 1/4. This compares a fine measure (the model's log-prob) with a coarse one (whether a reader names the concept), so it holds for these measures only; a reader-side likelihood could be more sensitive. On these concepts the NLA does not see what the model cannot use, and the Oracle reads the non-J part at least as well as the NLA (against the P17 premise).
- A methods trap, caught. Comparing a probe with random directions at the same dose is meaningless in a deterministic forward: it ‘detects’ any planted direction at any dose. The absolute criterion is used instead (amendment A9).
- Late layers don't rescue it. The Oracle at L48–60 at the report position names the full vector in 0 qualifying cells and the non-J part in 0.
- The machinery is sound. Every write exact (ρ within 1 ± 3e-07), runs bit-identical on repeat, the fast regime matches the exact one on 100% of 270 cells, the C1–C3 checks and errata are in, and the amendment's budget for the full run is about 1.5–3 GPU hours (measured throughput, A16).


**What you decide**

- The report measure (A1): log-prob at the open quote, with the paper's rank, instead of sampled naming.
- Band (A3): L32–38 as registered, or the stronger but more disruptive L24–38.
- P11 (A6): drop the report-position version (nothing arrives there) for a reader-vs-model sensitivity comparison.
- The clamp (A8): test re-entry with the band-limited clamp and its null.
- The go for the amended full run on the 45 test concepts (about 1.5–3 GPU hours).



Everything here is development data on 10 dev concepts. None of it is a test of the pre-registered predictions; those need the amended full run.

## Q1. Report measure

The paper's own measure works on Qwen where sampled naming does not. In every block the concept's log-probability at the report position rises above what random directions reach, at 9/9 doses for the pure J-lens vector and 8/9 for the full vector (L24–38/default), 9/9 doses for the pure J-lens vector and 8/9 for the full vector (L24–38/‘word’), 9/9 doses for the pure J-lens vector and 7/9 for the full vector (L32–38/default), 9/9 doses for the pure J-lens vector and 8/9 for the full vector (L32–38/‘word’). Proposal A1: make R1′ (with R1) the primary report measure and keep sampled naming as a secondary column.

## Q2. Prefill

No. The ‘word’ prefill does not remove Qwen's default answer: “apple” stays the top next word in the clean run and under weak injections, and sampled naming stays far below the 0.25 bar (best 0.15, pure J-lens vector, L24–38/‘word’). Qwen appears to have a canned answer for “the thought is about”. Proposal A2: keep the protocol's default prefill (the paper's), and treat naming as secondary.

## Q3. Band

The rule keeps band L32–38. The wide band L24–38 gives a much larger report effect for the full vector (3.80 vs 1.45 nats at its best dose, default prefill), but it also disturbs the neutral prompt more (KL 0.96 vs 0.28), and the pre-registered rule required a larger effect without a larger KL. This is a real trade-off: L24–38 buys a bigger report signal at the price of broader disruption. Proposal A3: L32–38 as registered; you may prefer L24–38 for signal, and the pilot gives you both.

## Q4. Onset

Yes. With the grid extended down to 1/256, every block has a dose range where the non-J part leaves the report untouched: onset s* = 1/64 (L24–38/default), 1/128 (L24–38/‘word’), 1/64 (L32–38/default), 1/128 (L32–38/‘word’). The reader doses by the rule would be 1/256, 1/64, 1/16 for L32–38/default. Proposal A4: the 9-point grid, with text readers at {s*/4, s*, 4s*}.

## Q5. Readers

Both text readers name the planted concept when they are handed the written cell, and neither sees it at the report position. On direct reads at s = 1 (band L32–38) the NLA names the full vector in 78% of samples and the non-J part in 60%; the Oracle names the non-J part in up to 90%. At the report position the highest naming rate for any planted condition, reader, dose and band is 8%. The reader-free probe agrees (it never classifies the report position as the concept at L42), so this is the concept not having arrived there, not the readers missing it. As posed, P11a and P11b compare the readers with the model at a position the content does not reach by the readers' layers; step 8 checks later layers.

## Q6. Clamp (design's)

Yes, unlike the 2-concept smoke: in all four blocks the design's clamp removes more of the non-J part's report effect than the random clamp does (L24–38/default +0.81 nats [0.39, 1.26], 9 concepts; L24–38/‘word’ +0.83 nats [0.34, 1.32], 9 concepts; L32–38/default +0.62 nats [0.26, 1.03], 9 concepts; L32–38/‘word’ +0.63 nats [0.29, 0.97], 9 concepts). Caveat: this clamp runs to L62, where a concept's J direction approaches its output direction, so part of the effect may be the clamp blocking the answer itself. Step 7b's band-limited re-entry clamp (Q10) tests that.

## Q7. Leaky splits

Few splits leak. With the rule fixed beforehand, band L32–38 has no leaky concepts and band L24–38 has earth, soap. Proposal A8: exclude these from every non-J contrast in the full run (earth is a dev concept; soap is a test concept and would be excluded before any test outcome exists).

## Q8. Concept-vector baseline

The baseline matters somewhat: the median cosine between the two concept-vector versions is 0.851 (lowest 0.43), below the 0.9 line. Leave-one-out stays primary and the 194-word version becomes a mandatory sensitivity check (A10).

## Q10. Re-entry clamp

The clamp's specificity lives at the output end. Run all the way to L62, the re-entry clamp removes much more of the non-J part's report effect than its random null (L32–38: +0.87 nats [0.41, 1.39]; L24–38: +1.19 nats [0.58, 1.83]). Stopped at L44 it removes nothing beyond its null (L32–38: −0.03 nats [−0.11, 0.07]; L24–38: −0.31 nats [−0.56, −0.03], the random clamp even removing slightly more). The concept's J direction turns into its output direction with depth (median cosine 0.38 at L40, 0.47 at L44, 0.77 at L52, 0.92 at L62), so a clamp reaching the late layers blocks the answer itself. By the pre-registered reading: no evidence on these dev concepts that the non-J part's effect on the report is mediated by re-entry into the concept's J coordinates within the workspace layers. A clamp that reaches the late layers could produce the paper's result (the non-J effect falling to about zero) without any workspace re-entry, so a full-range clamp alone cannot separate the two. Proposal A7: if M2 stays, run it as the band-limited re-entry clamp with its null, and report the full-range version only next to the cosine curve.

## Q11. Layer sweep at the report position

No, in either band. At every dose where the non-J part moves the model's report, the concept's median J-lens rank at the report position stays in the hundreds to thousands at every layer up to L62; the full vector and the J part behave the same, and so do the logit lens and the R-lens. Planting shifts the report distribution toward the concept without the concept ever becoming a top candidate at the report position, consistent with the readers (Q5) and the absolute probe criterion.

## Q12. Arrival at the report position (step 8)

Read with care. By the pre-registered rule (probe response above the strongest of five random directions), every condition “arrives” at the report position by L32–34 at every dose, even 1/256, so all three registered predictions technically came true. That rule is degenerate: the forward pass is deterministic and nearly linear at these doses, so the comparison with random directions is scale-free and fires at any dose. By the absolute criterion (the probe classifies the read as the concept), the concept never reaches the report position at any layer L32–62, for any condition, dose or band (highest rate anywhere 10%). Together with Q5 and Q11: planting nudges the model's report distribution without ever producing a readable concept state at the report position. P11a/P11b at the report position therefore compare the readers with the model where neither can find the concept; the amendment must re-pose P11 (proposal: compare reader sensitivity with model sensitivity on matched measures at the written cell, and drop the report-position version).

## Preview: J vs non-J at matched norm

Preview only (dev concepts, no test): at matched norm the J part beats the non-J part at low and middle doses (up to +0.55 nats at L32–38/default) and loses at high doses (down to −0.50). The two reach similar peaks; the J part gets there at a lower dose.

## Preview: Oracle vs NLA on the non-J part

Preview of P17 (dev only, not a test): on direct reads the Oracle names the non-J part at least as often as the NLA (s = 1: Oracle up to 90%, NLA 60%; s = 1/4: Oracle up to 15%, NLA 0%). That runs against the A.9.2-based expectation that the Oracle sees content outside J-space less. At s = 1/4 both readers name the J part far more often than the non-J part (Oracle up to 53% vs 15%), the same dose pattern as the model's report.

## R-lens

The R-lens and J-lens agree closely on these reads (Spearman 0.82, top-10 agreement 98%), so for planted concepts the two Jacobian lenses are interchangeable as token readers; adding R-lens to the full run costs little and answers the consistency question directly (A13).

## Regime and budget

The fast float32-residual regime matches the exact SEC replay on 100% of 270 cells (median log-prob difference 0.030 nats), so the full run can use it throughout (A12).

## Power

Planning numbers from the 10 dev concepts. For the J-vs-non-J report contrast at s = 1/4 the predicted 95% CI half-width with 45 test concepts is ±0.24 nats at band L32–38 and ±0.69 at L24–38 (concepts vary much more in the wide band, another point for L32–38). For reader naming the spread between concepts dominates: the NLA's naming of the non-J part at s = 1 gives ±0.139 with 4 samples and ±0.137 with 8, so doubling the samples buys almost nothing; more concepts is what helps (as your review argued).

## Amendment

Proposals filled from these results: `docs/design_specs/005_amendment_01_DRAFT.md`.
