# 002 — Reader claim scoring (role-aware, closed-set) — design v1, DRAFT for PI approval

*2026-10-02. Exploratory development of the reader-scoring procedure on dev material: the saved 001 reader snapshot
(10 brew tables) and the saved 001 activation captures (48 dev tables). No new reader generations. The procedure is
frozen when this design is approved (after the 20-readout prompt pilot in §6) and before any held-out use. The
label-construction design, previously called "002", becomes 003.*

## 1. Why

The 001 snapshot scored readers by word mention: does a colour word occur anywhere in the readout? The external audit
(`docs/reviews/2026-10-02_codex_reader_audit/`; report 001 §15) showed that this overstates what readers claim:
- of 17 NLA mentions of z at `:`, about 5 give z its correct role;
- lens top-10 lists and long descriptions are not comparable claim budgets;
- marginal real-vs-shuffled rates are weaker evidence than within-table contrasts.

The spec's own scoring rules (§7 and A01 §4) were never implemented: matched claim budgets, closed-set candidates,
the token-reader hit rule, a judge that sees only the readout and the options, abstention, and prompt-only and
no-activation baselines. 002 implements them on data we already have, so that a held-out run (and 003's labels) has
a frozen, tested scorer.

## 2. Material

| Reader | Readouts (10 tables; member / partner / substituted; arms) | Source |
|---|---|---|
| NLA L42, 3 emission cells, k = 4 | 880 explanations (real, shuffled, SEC at `:`; plus no-activation) | `results/001_leverage_profile/analysis/snapshot.json` |
| Oracle contract, `:` L40/L44, k = 4 | 960 samples (real, none, shuffled, SEC) | same |
| Oracle WSB grid, L20–60 every 4, 3 cells, k = 1 | 2,090 samples (real, shuffled, none) | same |
| Lenses: J_NP (std, cosine), J_CB, R_CB, logit lens | top-10 lists (saved), and recomputed candidate scores (§4) | same; `outputs/001_leverage_profile/run/captures_brew.npz` |

**Closed set per table:** WSB's 5 options for the item, i.e. start, z, answer and 2 off-path colours (the A01 §4 set),
plus "none of these" and "cannot tell".

## 3. Verbalizer claims: role extraction by a judge

For every readout, a judge sees **only the readout text and the 5 candidates** (order shuffled per readout with a fixed
seed), never the prompt. It answers in JSON which candidate the text puts in each role of a two-step colour chain:
- `start`: the colour the chain begins from;
- `intermediate`: the colour after one application of the rule;
- `answer`: the colour after the second application, i.e. the final answer.

Each role may be "none of these" (the text gives that role a colour outside the candidates) or "cannot tell" (no
role is stated). Rules given to the judge, verbatim in the prompt (appendix A, frozen at approval):
- "Applying the rule twice to X" makes X the start.
- A chain "A → B → C" maps to start, intermediate, answer.
- A colour merely mentioned, or listed as one option among several, gets no role.

**Judges** (PI decision, §9): J1 = stock Qwen3.6-27B (thinking off, greedy, JSON); J2 = Gemma-3-27B-it (a different model
family from the subject model and both verbalizers, which are Qwen3.6-27B fine-tunes). Recommended: both, with
inter-judge agreement reported (Cohen's κ per role). Disagreements stay disagreements: scores are reported per judge,
and a cell counts as "agreed" only where both match.

**Validation set:** 60 readouts, stratified by reader × arm. They are role-coded by the author (Claude) before any
judge output is seen, and reported as judge-vs-author agreement. A 50-readout sample is offered to the PI as a
spot-check page (spec §10).

## 4. Token-reader claims (judge-free)

1. **From the saved top-10 lists (all snapshot layers, 10 tables).** The reader's claim is its highest-ranked candidate
   colour in the top 10 (either spelling); abstain if none appears.
2. **Full closed-set scores (A01 §4 rule; L40/42/44/56/60, all 48 dev tables, member and partner renders).**
   - Recompute each lens's score for the 5 candidates (max over " c" / " C") from the saved activations and the
     pinned lens files, on CPU.
   - **Hit** = z ranked first among the candidates **and** z's score above the same lens applied to the
     no-activation input (the dev-mean activation at that cell and layer). Also reported: reciprocal rank of z, and
     the same for start and answer.
   - The activations are the SEC-regime captures; the 001 bridge shows stock and SEC agree at cos ≥ 0.99985 and on
     99.4% of top-10 lists. The `none` input is the mean over dev renders.

A lens claim has no role, so lens "hit on z" is compared with the verbalizers' "z in the intermediate role". Both are
budget 1: one colour per reader per readout.

## 5. Scores and contrasts

Per readout:
- **mention(z)**: the 001 word match, kept for continuity;
- **role-correct(z)**: intermediate = z;
- **contradicted**: intermediate = another candidate;
- **misplaced(z)**: z in the start or answer role;
- **abstain**: intermediate = cannot tell or none.

The same scores are computed for start and answer. For stochastic readers, report both the per-sample rate and the
per-cell majority claim (self-consistency = the share of samples agreeing with the majority).

Contrasts, all within table, with the per-table values listed:
- (a) real vs shuffled;
- (b) **start-swap tracking on role-correct claims:** [(member: z in role − partner: z in role) + (partner: z′ in
  role − member: z′ in role)] / 2;
- (c) substitution tracking (mentions; c and c′ are off-path, so they have no role);
- (d) own-z vs other-tables'-z (the colour-preference control).

Uncertainty: table bootstrap and exact sign-flip tests, unadjusted and labelled exploratory.

**Baselines:**
- **shuffled** and **no-activation**: as saved;
- **prompt-only** (A01 C18: stock Qwen3.6-27B): given the prompt and the 5 candidates, the log-probability of each as
  "the colour after the first stir", for all 231 dev renders. Expected near 1.0. It is the ceiling on what the
  prompt alone reveals, not a reader.

## 6. Procedure

1. Pilot the judge prompt on 20 readouts (not from the validation set). Fix only format and parse problems, and log
   every change.
2. Commit the frozen prompt, seeds and parsers.
3. Run J1 and J2 on all 3,930 verbalizer readouts.
4. Run the lens closed-set scoring (CPU) and the prompt-only baseline (GPU, one forward per render).
5. Analysis, figures (native page style) and report 002 with qualitative samples: for each reader, examples of
   role-correct, misplaced, contradicted and abstained readouts, with the judge's JSON beside each.

## 7. Predictions (priors; written before any judge output)

| # | Prediction | Prior |
|---|---|---|
| R1 | NLA L42 `:` role-correct(z), real, < 0.25 | 60% |
| R2 | Oracle L40 `:` role-correct(z): real − shuffled ≥ 0.10 | 40% |
| R3 | NLA start-swap tracking on role-correct claims positive in ≥ 7/10 tables at the assistant newline | 45% |
| R4 | Inter-judge κ on the intermediate role ≥ 0.6 | 60% |
| R5 | Lens closed-set hit on z at L40–44 (A01 rule) ≤ 0.2 for every lens; at L60, answer hit ≥ 0.8 | 70% |
| R6 | Prompt-only picks z in ≥ 0.95 of renders | 85% |

## 8. Budget

| Step | Hardware | Time |
|---|---|---|
| Judges | GPU, HF batched generation, about 60 output tokens per readout | about 15–25 min each |
| Gemma-3-27B-it | download about 54 GB | — |
| Lens closed-set scoring | CPU | minutes |
| Prompt-only | GPU, 231 forwards | about 2 min |

## 9. Decisions for the PI

1. Judge: J1 only, J2 only, or both (recommended: both).
2. Include the Oracle WSB grid (2,090 single samples), or only the contract arms and the NLA (1,840)? Recommended:
   include it, reported separately.
3. Approve the naming change: label design → 003.

## Appendix A — judge prompt (draft; frozen after the pilot)

```
You will read a short text written by an AI system that describes a colour puzzle. The puzzle applies a colour
rule twice: start colour → intermediate colour (after one application) → answer colour (after two).

Candidate colours: {c1}, {c2}, {c3}, {c4}, {c5}

According to the text, which candidate is the START colour, which is the INTERMEDIATE colour, and which is the
ANSWER colour? Rules:
- If the text says to apply the rule twice to X, X is the START.
- A chain written as A → B → C means START = A, INTERMEDIATE = B, ANSWER = C.
- If the text gives a role a colour that is not a candidate, answer "none".
- If the text does not state a role, or only lists colours without roles, answer "cannot tell".
Use only the text. Reply with JSON only: {"start": ..., "intermediate": ..., "answer": ...}

TEXT:
{readout}
```
