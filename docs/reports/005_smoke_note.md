# 005 smoke note (run `smoke1`, 2026-10-04 19:50–20:17 UTC)

*Design:* `docs/design_specs/005_planted_concepts.md` v2 §11, with the agent's priors pre-registered in df56646 before the run.
*Code:* 4ce5c21 (smoke), plus the NLA-phase fix and resume in ba31daa.
*Authorisation:* the PI's go for the smoke (2026-10-04). The full run is **not** authorised.
*Data:*
- raw outputs in `outputs/005_planted_concepts/smoke/smoke1/`;
- summary in `results/005_planted_concepts/analysis/smoke_summary.json` (`scripts/005_planted_concepts/smoke_analysis.py`).

Everything here is on **dev concepts**: 6 gate concepts, 10 for the onset, gel and earth end to end. It is descriptive and settles no prediction.

## Summary

| Smoke item (§11) | Outcome |
|---|---|
| 1. G1 Qwen report gate | **Failed at band B (L32–38) and at the fallback B2 (L24–38). Stop rule triggered:** the design must be amended before any run. Details below; the short version is that *sampled naming* is floor-limited while the paper's *rank / log-prob* measure responds strongly. |
| 1b. Calibration | Not triggered (condition 1 peaked at s = 1/2 at B and 1/4 at B2, not at the grid top) |
| 1c. Onset | s* = **1/32, the bottom of the grid** (B2). The non-J part already moves the report log-prob above the random null at the smallest dose, so P11a/P11b have no "model silent" strength on this grid |
| 2. End to end (gel, earth), both phases | Ran. Phase B crashed on a loader bug (below); it was fixed and resumed from the stored inputs, and nothing else changed |
| 3. G0 fidelity | **Pass.** 180 injection write batches: ρ ∈ [0.9999997, 1.0000002], κ ≥ 0.9999998. 1,092 clamp entries: residual ≤ 1.2e-6 |
| 3b. G5 split quality | 2% of (concept, layer) pairs have c in J_NP top-10 on the pure u^⊥, so not flagged overall. **But earth is leaky** (ranks 2, 4, 8 at L25, L29, L35) |
| 4. Determinism | **Pass:** teacher-forced outputs bitwise identical; samples identical with the same seed |
| 5. SEC l0 = 31 cross-check | **Skipped:** the final band B2 starts below l0. Moved into the pilot at band B |
| 6. Throughput | See below; the model side and Oracle run much faster than budgeted |
| 7. Raw readouts | Inspected; samples below |
| 8. Peak GPU memory | Phase B (NLA): 76.0 GB. Phase-A peaks were not saved (bug below) |

## G1: why the gate failed, and what still works

Band B2 (L24–38), 6 gate concepts. Medians over concepts; R2 = sampled naming (8 samples); random = rand-dir seed 0.

| Condition | s | Median reciprocal rank R1 | Median ΔR1′ (nats) | Mean R2 | Median B1 (neutral KL) | Top-1 at the quote |
|---|---|---|---|---|---|---|
| 1 pure J-lens vector | 1/8 | 0.032 | +5.3 | 0.02 | 0.47 | apple |
| 1 pure J-lens vector | **1/4** | **0.500** | **+8.5** | 0.08 | 0.58 | apple |
| 1 pure J-lens vector | 1/2 | 0.129 | +7.0 | 0.02 | 0.45 | apple |
| 2 full concept vector | 1/4 | 0.003 | +3.3 | 0.00 | 0.77 | word / apple |
| 3 J part (matched) | 1/4 | 0.002 | +2.3 | — | 1.21 | word / apple |
| 4 non-J part (matched) | 1/4 | 0.003 | +2.3 | — | 0.67 | apple / word |
| 7 random direction | 1/4 | 0.001 | +0.2 | 0.00 | 0.05 | apple |

- **Clean run:** the top-5 tokens after `The thought is about "` are apple, cat, banana, ele(phant), pine(apple). The prompt behaves as in the paper ("elephant" was the paper's own clean-run example).
- **The pure J-lens vector works by the paper's metric:** median rank 2 at s = 1/4. But "apple" stays top-1, so 8 samples at temperature 1 rarely draw the concept.
- **Full concept vectors push the reply toward topic-like words rather than the concept.** Sampled examples for "loss": "ocean", "silence", "anticipation", "safety". At high strength, every condition drifts toward "word" (as in "…about the word …").
- **Band B (L32–38):** naming was 0% for conditions 1 and 2 at every strength.
  - Its rows were overwritten when G1 re-ran at B2 (a logging bug, fixed in the pilot); only the decision record survives.
- **Agent prior:** I had 0.65 on G1 passing at B. It failed.

**Onset trace (B2, 10 dev concepts):** median ΔR1′ of u^⊥ against the rand-dir 95th percentile.

| s | 1/32 | 1/16 | 1/8 | 1/4 | 1/2 | 1 |
|---|---|---|---|---|---|---|
| u^⊥ median ΔR1′ | 0.30 | 0.48 | 1.23 | 2.62 | 2.48 | 2.02 |
| null 95th percentile | 0.12 | 0.24 | 0.45 | 0.85 | 1.53 | 2.07 |

## End to end (gel, earth; B2)

ΔR1′ (nats) / R1 at s = 1/4:

| Concept | 1 J-lens | 2 full | 3 J | 4 non-J | 5 J natural | 6 non-J natural | 7 random | 8 rand-J | 9 rand-⊥ | 10 non-J + c-clamp | 10r non-J + random clamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gel | +9.3 / 1.00 | +2.8 / .003 | +3.0 / .003 | +2.3 / .002 | +0.4 | +2.3 | +0.2 | +0.1 | +0.4 | +1.4 | +2.0 |
| earth | +4.8 / .20 | +5.9 / 1.00 | +1.5 / .006 | **+5.9 / 1.00** (named 3/8) | +1.0 | +6.1 | −0.3 | +0.7 | −0.0 | +2.0 | +2.0 |

- **earth's non-J part looks reportable, but earth is G5-leaky** (its u^⊥ is still J-readable as "earth"). This is an artifact of the split, not evidence against privilege. The pilot's pre-registered rule (§16.1 Q7) excludes such concepts.
- **At natural norm**, the full vector's effect sits almost entirely in the non-J part (6 ≈ 2; 5 small). That is expected, since u^⊥ carries about 94% of the vector.
- **The clamp is not specific on these two concepts.** The c-clamp removes about the same share of the carried perturbation as the random clamp (median 5.5% vs 6.0% at band layers; 4.6% vs 4.5% above them), and leaves the report effect of u^⊥ about as large (earth: identical).

## Readers (gel, earth; text strengths 1/32, 1/16, 1/8, from the onset)

**Text readers name the concept almost never at these low strengths.**
- NLA: 0/4 at every condition, kind and strength.
- Oracle: names earth at L44 on direct reads of the full vector and the J part at s = 1/8 (3 of 4 samples each), and on 1 of 4 carried C_o reads (full vector); otherwise 0.
- These doses are low because the onset sat at the grid floor. The pilot reads up to s = 1.

**Token readers and the probe at L42** (medians over gel and earth at s = 1/32 | 1/16 | 1/8):

| Kind | Pure J-lens vector: J_NP rank | Full: probe Δ | Non-J: probe Δ | Non-J: J_NP rank | Random: probe Δ |
|---|---|---|---|---|---|
| direct (C_q) | 106 / 2 / 1 | +2.5 / +4.9 / +9.9 | +2.3 / +4.5 / +9.0 | 12,098 / 9,956 / 6,874 | −0.1 / −0.2 / −0.4 |
| carried (C_o) | 14,521 / 8,142 / 12,944 | +0.7 / +1.5 / +2.8 | +0.5 / +1.1 / +2.2 | ~25,000 | +0.1 / +0.1 / +0.0 |
| injected position (C_q after propagation) | 4 / 1 / 1 | +6.7 / +14.6 / +46.2 | +5.8 / +12.4 / +39.8 | 5,879 / 5,037 / 1,174 | +0.9 / +1.5 / +1.2 |

- **The probe detects the non-J part about as strongly as the full vector on direct reads.** That makes it the R1 baseline: anything a text reader finds in u^⊥ there is linearly present.
- **Little reaches the report position by L42.** J-lens ranks are in the tens of thousands, and probe deltas are small.
- **The "injected position" reads at L42 still carry the planted vector strongly.** This confirms the PI review's point that C_q after propagation is near-direct access.

**Clean-read base rates (G6, concepts named in > 5% of 32 no-injection samples).**
- NLA L42: "case" and "dog" (carried C_o); "case" and "role" (direct; max 0.19).
- Oracle: "dog" and "process" at L40 carried C_o, with **"dog" in 53% of samples**, because the Oracle reads the model's candidate answers (cat, dog, apple…); "process" at L44 carried.
- G6 will drop these concept × reader cells. My prior that ≥ 1 NLA concept would be dropped was 0.40; it happened.

**Raw readouts (first sample each; abridged).**
- **NLA, clean, C_o:** "Q&A format established throughout the dialogue section … The scenario describes a word-embedding completion where the LLM is prompted to identify a single hidden concept … Final token `"` opens a quoted string"
- **NLA, earth full s = 1/8, direct:** "Game mechanics established: player must classify each response as 'yes' (hidden word was flashed) or 'no' …" (no concept named)
- **Oracle L40, clean, C_o:** "- cat" or "dog". The word in my hidden state right now is " / - 苹果" (apple) … / - coffee" …"
- **Oracle L44, earth full s = 1/8, direct:** "… - 目标 word: **“Earth”**. Let's imagine the neural activity corresponds to thinking about a specific word …"
- **Oracle L44, earth non-J s = 1/8, direct:** "… 目标词 was "apple" this time … **Presenting Word: "London"**"

Both text readers mostly describe the task framing ("hidden word game"). The Oracle at the report position reads out the model's candidate answers.

## Other gates and checks

- **G3:** median J share 6.3% (range 2.2–17.4%; 180 concept-layers). Passes, and matches the paper's 6–7%.
  - Sensitivity medians: k = 10: 5.6%; k = 25: 7.0%; LS refit: 6.3%; 194-word baseline: 7.1%; raw-W_U atoms: 4.9%.
- **G4:** passes. Leave-one-template-out AUC median 1.00 (min 0.97); permutation-control median 0.48 (max 0.87, small-sample noise: each held-out template has one positive).
- **cos(leave-one-out u, 194-word u):** median 0.86 (min 0.43). Below the PI review's 0.9 line, so the baseline sensitivity must be reported.

## Throughput (reconstructed from log timestamps; phase-A timers were lost, see below)

| Component | Measured |
|---|---|
| Model load | 23 s |
| Material: 295 "Tell me about" + 1,770 template captures | 18 s |
| J-space split (10 concepts × 10 layers, with all sensitivity variants) | 13 s |
| G1 at one band: 324 teacher-forced rows + 864 sampled rows (8 tokens) | ~5.7 min, dominated by sampling: ~2.7 sampled rows/s at chunk 96 |
| Onset: 360 teacher-forced rows | ~25 s (~15 rows/s, each 2 forwards) |
| Token readers + probes (1,620 reads; 390 probe fits in parallel) | 23 s |
| Oracle: 512 samples | 166 s (3.1 samples/s) |
| NLA: load 49 s; 256 samples | 130 s (2.0 samples/s) |

Re-estimated full run (amended shape: one band, one prefill, 9 strengths, 55 concepts, 4 text-reader strengths): **≈ 4–5 h**, versus the budgeted 6–8.5 h.
- Teacher-forced rows: ~10 min.
- Sampled naming: ~2 h at 2.7 rows/s.
- NLA: 7,040 samples ≈ 1 h.
- Oracle: 14,080 samples ≈ 1.3 h.

The pilot measures this at scale (Q9).

## Mistakes and deviations (also in the commit messages and the CLAUDE.md log)

1. **NLA loader crash in phase B.** `NLArl.scaling` collected every module's `scaling` attribute; the attention module's is a float, not a LoRA dict. Only the GPU test would have caught it, and it had not been run. Fixed, and phase B resumed from the stored inputs (`planted.py nla-resume`).
2. **Band B's G1 rows were overwritten** when G1 re-ran at B2; only the decision record survives. The pilot saves every block.
3. **Phase-A timers and memory peaks were not saved**, because they were written only at the end of the run, after the crash. The pilot saves state after every block.
4. **G4's control** is a sample-level label permutation, not the design's Hewitt–Liang type-level control. The latter is uninformative for single-word identity probes, since any word's identity is decodable. To be written into the amendment.
5. **The clamp's removed share is logged against the total perturbation** (‖QQᵀ(h − h_clean)‖ / ‖h − h_clean‖) at every clamped layer, rather than against the injected Δ alone at band layers. Noted in §16.2.

## What this means

As written, the design cannot run its report contrasts. The amendment must choose:
- the report measure;
- the band and prefill;
- a grid that reaches below the u^⊥ onset;
- the clamp's fate;
- the leaky-concept exclusion.

The overnight dev-only pilot (§16; Q1–Q9 pre-registered) gathers exactly that evidence. Nothing here bears on P11a, P11b or P17.
