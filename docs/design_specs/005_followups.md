# 005 follow-ups (PI instructions 2026-10-05): pre-registration — EXPLORATORY

**Status.**
- Exploratory. The 45 test concepts' 005 outcomes have been seen (full run 20261005T015235Z), so nothing here is a confirmatory test.
- The `[PI]` prior fields are left blank for the PI. The agent's priors were written before each analysis or run below.
- Branch `005-amended-full-run`.

**Disclosure.**
- While diagnosing contrast R2 in the full run, I looked at a few concepts' B1/B2/B4 values for the non-J part at s ∈ {1/64, 1/4, 1}.
- No ΔR1′ vs B2 correlation, position profile, cosine regression or logit-direction run had been computed.

**Provenance of the approval for amendment 01 and the full run (the PI asked for this first).**
- The approval was the PI's chat message: "adopt but branch office to a new branch and run this shit there I guess so i can reverse it when possible".
- It came after my 00:59 UTC step-11 summary (commit 9709712, 2026-10-05 00:59:12 UTC). A later message from the PI, "what are the ammendment decision?", came in between, and I answered it with the list of decisions.
- It came before I created the branch: git reflog, `checkout: moving from spec-v2 to 005-amended-full-run` at 2026-10-05 01:37:00 UTC.
- I do not have the message's own timestamp, so it lies in that window.
- I read "adopt" as all items including the optional A18, and "run this … there" as the go for the full run. Both are recorded in `005_amendment_01.md` and commit 30d5351.

## Part A (existing 005 outputs; A2b and A3 also need the GPU, as noted)

**A1. Reader overlap and ensembles on planted ground truth (descriptive).**
- **Readers:**
  - NLA L42; Oracle L40 and L44 (direct reads; naming: ≥ 1 of 4 samples, and ≥ 2 of 4);
  - probe at L40/42/44 (decision > 0);
  - J_NP, J_CB, R_CB and the logit lens at L40/42/44 (rank ≤ 10).
- **Units:** concept × condition × dose on the 45 test concepts, at the text-reader doses {1/64, 1/4, 1}, conditions 2, 3, 4 and 7 (seed 0).
  - Direct reads for every reader.
  - Carried C_o and injected C_q reads are added for token readers and the probe.
- **Per dose:**
  - each reader's hit rate on conditions 2/3/4 vs its condition-7 rate;
  - pairwise Cohen's κ with both-hit / one-only counts;
  - unique catches (a hit by this reader and by no other);
  - the union and majority-vote ensembles vs the best single reader.
- Cluster bootstrap over concepts. No rule.
- **Agent priors:**
  - the union ensemble's non-J hit rate at s = 1/4 exceeds the best single reader's by ≥ 0.05: **0.55**;
  - κ(NLA, Oracle L40) on direct non-J reads at s = 1 ≥ 0.4: **0.45**.
- `[PI]` priors: blank.

**A2. What does the model-side measure measure?**
- **A2 (PI):**
  - Spearman ρ over the 45 test concepts between ΔR1′ (answer-slot log-prob rise) and B2 (the concept's log-prob rise on the neutral weather prompt, max over 8 positions), at each dose, for conditions 1–4.
  - Dev concepts from the pilot (band B, default prefill) are reported separately.
  - **Rule (the PI's, made precise here):** "tracks closely" if the median ρ over conditions 1–4 × s ∈ {1/16, 1/8, 1/4, 1/2} is ≥ 0.7. Below 1/16 both measures sit at the numerical floor.
  - If it tracks closely, every 005 report calls the model-side measure "lexical priming at the answer slot", not "report".
  - Also reported: the paired magnitude ΔR1′ vs B2. B2 is a max over 8 positions, so it is biased upward against a single slot.
- **A2b (agent addition, discriminative; GPU, teacher-forced, introspection prompt only):**
  - Rationale: cross-concept correlation shows shared potency, not mechanism. Generic priming (a logit bias toward c) raises the concept's log-prob by roughly the same amount at every position. A report-like use should raise it specifically at the answer slot.
  - **Rows:** the 45 test concepts × 9 doses × conditions 1, 2, 3, 4 and 7 (seed 0), band L32–38. Recorded: Δ log p of the concept (LSE over all four spellings c, C, " c", " C") at every position 125–138, injected minus clean.
  - **Statistic:** per concept, the mean over s ∈ {1/16, 1/8, 1/4} of [Δ at C_o (138) − the median over the non-answer positions 126–137 of Δ].
  - **Rule:**
    - CI above 0: the answer slot gets a rise beyond the position-generic one ("slot-specific"; consistent with, not proof of, report-like use);
    - otherwise: the rise is position-generic priming.
    - Primary for the full vector (condition 2); conditions 1, 3 and 4 are reported.
  - **Naming:** A2's rule (the PI's) governs the name. A2b is reported next to it, and if it says position-generic, "lexical priming" is used regardless of ρ.
- **Agent priors:**
  - A2 median ρ ≥ 0.7: **0.50**;
  - A2b slot-specific for the full vector: **0.60**.
- `[PI]` priors: blank.

**A3. Does alignment with the output direction explain potency?**
- **Geometry (GPU for the matmuls only; no model writes):**
  - every injected direction of conditions 1, 2, 3, 4, 7 (seed 0), 8 and 9, at each band layer;
  - cos(direction, g ⊙ W_U[t_q]), with t_q = c's lowercase quote token (the answer token R1′ scores), averaged over L32–38;
  - sensitivity: t = " c".
- **Regression:**
  - ΔR1′ (full run, test concepts) on the cosine, across concept × condition, separately at s = 1/8 and 1/4;
  - reported: slope, R², and whether condition dummies add anything (ΔR², nested F-test);
  - within-condition slopes are also reported.
- **Per concept:**
  - the NNLS coefficient of c's own atom (" c") inside its J part (0 if not selected);
  - its energy share coef² / ‖u^J‖²;
  - cos²(u^J, atom).
- **Rule:** "alignment explains potency" if R² ≥ 0.5 at both doses and the condition dummies add ΔR² < 0.05 (or F-test p ≥ 0.05).
- **Agent priors:**
  - R² ≥ 0.5: **0.50**;
  - dummies add < 0.05: **0.35**.
- `[PI]` priors: blank.

## Part B (005b): disposition

**Not run as written.** Reasons, checked against the code and records:
1. **Premise (ii) is partly false.** The text readers *were* run on model-carried content:
   - the pilot read the answer position C_o with the NLA (L42) and the Oracle (L40, L44), naming ≤ 8% (§16.1 Q5);
   - the Oracle at L48–60 named the full vector there for a few concepts with band L24–38 (§16.4 Q13).
2. **Q-b's "carried C_q" is not model-carried.** C_q = 110 lies inside the injected span (97–116). The band writes persist in the residual stream, so a read at C_q at L40–44 is near-direct. This is the v1 mislabel the PI's own review corrected (design §15 item 1).
3. **The reader-dose rule is degenerate.** "The lowest dose where the full vector's ΔR1′ beats the random 95th percentile" is a relative criterion, and those are scale-free in a deterministic forward (A9).
   - In the full run it gives d0 = 1/128 for band B, so the reader doses would be 1/128, 1/32 and 1/8.
   - B_early would put them lower still.
   - The readers name nothing below 1/4 even on *direct* reads (full run: 0% at 1/64). The carried reads would be null by construction.
4. **B_early confounds band position with total injection.** At matched per-layer dose it writes 23 layers instead of 7 (≈ 3.3× the summed write), so any change in arrival cannot be attributed to the earlier start.
5. **Explanation (iii) is already mostly supported** by the pilot: the J-lens never puts c in the top 10 at C_o at any layer to L62; the absolute probe never fires at C_o; the Oracle names it at late layers only with the wider band, for a few concepts. A2/A2b test (iii) directly and cheaply.

A fixed version of Q-a, if the PI wants it later, would need:
- reader doses fixed at {1/16, 1/4, 1};
- a total-norm-matched B_early control (per-layer dose × 7/23);
- C_q dropped as a "carried" read.

**Q-c is well posed and cheap, and runs.**
- **Rows:** band L32–38, teacher-forced (introspection + neutral prompts).
- **Concepts:** 30 = the 10 dev concepts + 20 test concepts. The test concepts are drawn with `numpy.random.default_rng(20261004).choice(TEST, 20, replace=False)`.
- **Conditions × 9 doses:**
  - pure J-lens atom (condition 1: unit J_ℓᵀ(g ⊙ W_U[" c"]));
  - the **logit-lens direction** unit(g ⊙ W_U[" c"]) for the same token (new);
  - the full vector (2);
  - rand-dir (seeds 0–4).
- **Measures:** ΔR1′, R1, B1, B2, B3; G0 on every write; any dose with median B1 > 5 nats is flagged as broken.
- **Statistic:** per concept, the mean over s ∈ {1/16, 1/8, 1/4} of ΔR1′(J atom) − ΔR1′(logit direction); cluster bootstrap CI over the 30 concepts; dev and test reported separately.
- **Rule (the PI's):** if the CI includes 0, J-lens adds nothing as a write direction on Qwen.
- **Agent prior:** the J atom is more potent (CI above 0): **0.80**. At small doses the J atom is the first-order gradient direction for the same-position logit (by construction). The injection is read at another position, though, so this is not guaranteed.
- `[PI]` prior: blank.

**Cost:** about 10 GPU minutes (model load, vectors for 55 concepts, ≈ 2,000 rows for A2b and ≈ 2,200 for Q-c). CPU analysis takes minutes.

**Code:**
- `scripts/005_planted_concepts/followups.py` (GPU: A2b, A3 geometry, Q-c);
- `scripts/005_planted_concepts/followups_analysis.py` (A1, A2, A2b, A3, Q-c), with output in `results/005_planted_concepts/analysis/followups_summary.json`.
