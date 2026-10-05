# 005 follow-ups (A1–A3, A2b, Q-c) — EXPLORATORY

*Spec:* `docs/design_specs/005_followups.md` (pre-registered before the GPU part ran; commit 323362c). The test concepts' 005 outcomes had been seen, so nothing here is confirmatory. *Data:* full run `20261005T015235Z` and follow-up run `fu1`; summary `results/005_planted_concepts/analysis/followups_summary.json`.

## What ran

- CPU on the full run's outputs: A1 (reader overlap), A2 (ΔR1′ vs B2).
- GPU, 4 minutes: A2b (the concept's log-prob rise at every prefill position, 45 test concepts × 9 doses × conditions 1–4 and rand-dir), the A3 geometry (no model writes), and Q-c (30 concepts × 9 doses: J atom, logit-lens direction, full vector, rand-dir × 5).
- G0: 1400 write records, 0 failures. Rows that repeat the full run reproduce it (A2b max |Δ| 0.055 nats; Q-c condition 1 0.000).
- Not run: 005b's Q-a and Q-b as written (reasons in the spec).

## A2 and A2b: what the model-side measure measures

- **A2.** Spearman ρ over the 45 test concepts between ΔR1′ (answer slot) and B2 (the concept's rise on the neutral weather prompt): median 0.55 over conditions 1–4 × s = 1/16…1/2 (range 0.14 to 0.70). Below the 0.7 rule, so A2 alone does not force the rename.
- B2 is larger than ΔR1′ in every cell: for the full vector at s = 1/4 the neutral-prompt rise exceeds the answer-slot rise by +4.83 [4.11, 5.65] (n = 45) nats (median ΔR1′ 1.33, median B2 5.74). Caveat: B2 is a max over 8 positions.
- **A2b (the discriminative test).** Answer-slot rise minus the median rise at the 12 non-answer prefill positions, mean over s = 1/16, 1/8, 1/4:
  - pure J-lens atom: −1.46 [−1.66, −1.26] (n = 45) nats → position-generic
  - full vector: +0.13 [−0.02, 0.29] (n = 45) nats → position-generic
  - J part: −0.21 [−0.37, −0.04] (n = 45) nats → position-generic
  - non-J part: +0.08 [−0.05, 0.20] (n = 45) nats → position-generic
  - random direction: −0.02 [−0.06, 0.03] (n = 45) nats → position-generic
  - Profile, full vector at s = 1/4 (median Δ by position 125–138): 1.32, 0.96, 0.79, 2.17, 2.13, 1.90, 1.56, 0.91, 1.12, 1.94, 1.41, 0.88, 1.36, 1.33.
- **Verdict by the pre-registered naming rule: "lexical priming at the answer slot".** The concept's log-prob rises about as much at every prefill position as at the answer slot, and more on an unrelated prompt. Every 005 statement about the model's "report" (ΔR1′, R1; M1, M3, M2, P11′) is about lexical priming of the concept token, not a report.

## A3: does alignment with the output direction explain potency?

- Regression of ΔR1′ on cos(direction, g ⊙ W_U[answer token]), 45 concepts × 7 conditions: R² 0.29 at s = 1/8 and 0.53 at s = 1/4 (slopes 6.2, 16.1). Condition identity adds ΔR² 0.22 and 0.12 (F-test p < 0.001). Rule (R² ≥ 0.5 at both doses and ΔR² < 0.05): **not met**. Within a condition the cosine explains little (|r| ≤ 0.4); the between-condition difference is mostly the pure J atom (median cos 0.187) vs everything else (≤ 0.012).
- **The concept's own atom is in its J part for 1 of 45 test concepts** (median energy share 0.000; median cos²(u^J, own atom) 0.066). The J part is a mixture of related and junk tokens (soap: sodium, soda, bacterial, laundering, …; dog: Dogs, adorable, Bulldogs, …). The earlier statement that M1 passes "partly by construction because the J part contains the concept's own atom" is wrong and is withdrawn.

## Q-c: the logit-lens direction vs the pure J-lens atom as a write direction

- Per concept, ΔR1′(J atom) − ΔR1′(logit-lens direction of the same token), mean over s = 1/16, 1/8, 1/4: all +0.04 [−0.32, 0.40] (n = 30); dev −0.16 [−0.79, 0.49] (n = 10); test +0.14 [−0.28, 0.57] (n = 20).
- **By the PI's rule (the interval includes 0): J-lens adds nothing as a write direction on Qwen**, at these doses and for this measure (which A2b shows is priming).
- The curves differ outside that window: median ΔR1′ J atom vs logit direction s = 1/64: 0.18 vs 0.05; s = 1/32: 0.36 vs 0.14; s = 1/16: 0.61 vs 0.40; s = 1/8: 1.52 vs 1.58; s = 1/4: 3.18 vs 3.16; s = 1/2: 3.51 vs 1.31; s = 1: 1.42 vs 0.26. The J atom is ahead at the lowest and the highest doses. No dose broke the model (B1 ≤ 5 nats everywhere).

## A1: reader overlap and ensembles (direct reads, 45 test concepts; naming = ≥ 1 of 4 samples)

| Reader | non-J s = 1/4 | full s = 1/4 | non-J s = 1 | full s = 1 | random s = 1 |
|---|---|---|---|---|---|
| NLA_L42 | 13% | 29% | 62% | 73% | 2% |
| Oracle_L40 | 16% | 36% | 73% | 84% | 4% |
| Oracle_L44 | 9% | 38% | 73% | 91% | 2% |
| probe_L42 | 44% | 49% | 98% | 98% | 0% |
| J_NP_L42 | 0% | 0% | 0% | 2% | 0% |
| J_CB_L42 | 0% | 0% | 0% | 0% | 0% |
| R_CB_L42 | 0% | 0% | 0% | 0% | 0% |
| logit_L42 | 0% | 0% | 0% | 0% | 0% |

- Agreement on the non-J part at s = 1: κ(NLA, Oracle L40) 0.35 (both 24, NLA only 4, Oracle only 9).
- Unique catches on the non-J part at s = 1/4: probe 11, NLA 1, Oracle 0 and 0, every token lens 0.
- Union of the 8-reader panel vs the best single reader (the probe) at s = 1/4: full vector +0.16 [0.07, 0.27] (n = 45); non-J part +0.02 [0.00, 0.07] (n = 45); the union's false-positive rate on random directions 4%. Majority vote is worse than the probe (the lenses never fire).
- The token lenses (J_NP, J_CB, R_CB, logit lens) put the concept in their top 10 for ≤ 4% of direct reads at L40–44 even at s = 1. Carried reads at C_o: 0% for every token reader and the probe.
- Caveat: the probe is a supervised, per-concept classifier that knows the target; it is a ceiling, not a reader in the same sense as the text readers.

## What contradicts earlier claims

1. **"Report" → lexical priming.** Pages, reports and my summaries called ΔR1′ the model's report ("Qwen reports planted concepts by the paper's own metric"; "the metric is specific to the planted concept"). The concept-specificity stands, but A2b shows the rise is position-generic: it is priming of the concept token.
2. **"M1 passes partly by construction (the J part contains the concept's own atom)"** is wrong: own atom in 1 of 45.
3. **My last summary's storm example** listed ` storm` and ` Storm` as J-part atoms. They are spellings added to the clamp's token set, not part of the J part.
4. **P11′ (Oracle) compared a reader's likelihood with the model's "report".** It is a comparison with priming at the answer slot, which makes "the reader is more sensitive than the model's report" a statement about priming, not report.

## Doubts

- B2 is a max over 8 positions and ΔR1′ a single slot, so A2's magnitude comparison is biased toward B2; A2b avoids this.
- A2b's non-answer positions predict prefill tokens such as "Yes", "detect" and "thought"; a concept token is implausible there. Priming that rises equally where the token is implausible and where it is the answer is the point, but it is one prompt.
- Q-c's window (1/16–1/4) was fixed in advance. The J atom wins outside it, so "adds nothing" is window-dependent.
- A1's best-single-reader comparison picks the best reader on the same data (optimistic for the single reader only by selection; here the probe is best by a wide margin).
