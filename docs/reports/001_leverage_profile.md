# 001 — Leverage profile and off-trajectory content

*Qwen3.6-27B · WorkspaceBench brew + chain · dev split only · design `docs/design_specs/001_leverage_profile.md` (v3 +
amendment A1) · 2026-10-02.*

## Scope

- **Model:** one model, Qwen3.6-27B.
- **Material:** two WorkspaceBench families, brew and chain; dev split only.
- **Layers:** five block outputs: 40, 42, 44, 56, 60. The controls add L0–L4.
- **Regime:** SEC (stock bf16 through block 39, true fp32 from block 40).
- **Readers:** no reader is used in §1–§7. §8 is a separate descriptive reader snapshot.
- **Wording:** "carries leverage" / "share S" means the natural pair difference at this cell, transplanted from the
  partner prompt, moves this answer by that fraction of the full gap G. It does not mean "the model uses x".
- **Revised after an external audit (Codex, 2026-10-02; §15).** Several statements were stronger than the evidence:
  the damage reading, the reader snapshot's mention counts, the reconstruction and null-result wording, and the probe
  controls. They are corrected in place and listed in §15, with every number recomputed from the saved outputs by
  `scripts/001_leverage_profile/audit_followups.py` (`analysis/audit_followups.json`). The defensible summary: on this
  small brew pilot, the readers share task-structure information, show some activation-sensitive colour signal, and
  often surround it with incorrect detail. Whether their claims track causally influential content is unresolved.

## Provenance

| Stage | run_id | git at manifest write |
|---|---|---|
| gate | 20261002T074329Z | 5d5d1a3, dirty |
| smoke | 20261002T080920Z | 5d5d1a3, dirty |
| run | 20261002T081118Z | 647dcb5, clean |
| probes | 20261002T092423Z | f445816, dirty |
| snapshot | 20261002T095116Z | 6b55a02, dirty |

- **gate, smoke:** "dirty" because the analysis/figures stages were uncommitted when the manifest was written. The gate
  and measurement code were committed.
- **snapshot:** the code that ran is the one committed in 6b55a02, which includes the render-loop fix; it ran
  after the run and probe outputs were hashed. "dirty" because the report was being edited.
- **probes:** the probe code is as committed at f445816. "dirty" because snapshot-summary code was uncommitted when the
  manifest was written.
- **raw outputs:** `outputs/001_leverage_profile/` (sha256 in `manifest_run.json`).
- **tables:** `results/001_leverage_profile/analysis/`; figures in `results/001_leverage_profile/figures/`.

---

## 1. Setup

**Pairs** (design §2):
- **Brew start pairs:** same rule table, different start colour; token-aligned, differing only at the start-colour
  token.
- **Chain start pairs:** same steps, different start number.
- **Brew colour substitutions:** an off-trajectory colour c is replaced everywhere in the table by a colour c′ absent
  from it. Path and answer stay the same; only colour tokens inside the table change.
- **Splits:** brew by table (50/50, stratified by z); chain by step-sequence group (10/10).

**Measurement:**
- For every gated dev pair and every read cell × layer, the partner's whole residual at that cell replaces the
  member's.
- Δm is the change in m = LSE log P(a′ spellings) − LSE log P(a spellings), with full-sequence teacher forcing over
  every token.
- S = Δm / G.
- The reverse direction is measured on a fixed 30% of pairs.

**Cells:**
- Brew: WSB's 24 pinned cells (stir ×6, start ×6, question ×9, and the newline after `assistant`, the `\n\n` after
  `</think>`, the `:` after `Answer`).
- Chain: the question span (" What" → final `\n\n`, 25 positions).

## 2. Gate (admission; every candidate, dev and test)

| Family | dev pairs passing | test pairs passing | Notes |
|---|---|---|---|
| Brew start pairs | **82 / 183** (33 of 48 tables) | 77 / 178 (35 of 49 tables) | 3 items have no valid partner at all |
| Chain start pairs | **48 / 53** (12 of 13 items; **9 of 10 step groups**) | 37 / 46 (7 of 10 groups) | 22 of the 48 dev pairs share z = 5: thin and clustered |
| Brew substitutions | **43 / 50** | — (dev only) | |

**Brew pass rate by member z (dev)** varies widely: green 4/24, gold 2/10, orange 0/2, teal 12/19, gray 11/20.

**Failures look like competence, not formatting.** The model splits its samples between the right colour and a
neighbour. Seed-20261003 examples:
- answer **teal**: greedy " brown", samples brown ×6, teal ×2, gold, Brown;
- answer **green**: 5/10, the rest brown ×3, plus "Orange" and "1. Star…".

Every decode is in `outputs/001_leverage_profile/gate/gate_prompts.jsonl` and in the qualitative file.

## 3. Engine checks and controls (exact)

| Check | Brew (107 pair runs) | Chain (62) |
|---|---|---|
| Embedding anchor reproduces the partner bitwise | 100% | 100% |
| Self-patch is bitwise identical to clean | 100% | 100% |
| Prefix before the first differing token is bitwise identical | 100% | 100% |
| Stir cells are bitwise no-ops (start pairs) | 100% | n/a |

**Start-token (or start-digit) patch at block outputs L0, L1, L2, L4:**
- median S = 1.00 in both families (IQR about 0.997–1.003);
- 100% of pairs ≥ 0.9.

Block 0 is a GatedDeltaNet layer, but in this setting it does not spread the start colour into later positions enough
to matter. Patching the start token's block-0 output alone transfers the whole effect.

**Noise floor (SEC):** |Δm| from 1e-4-relative perturbations at the last read cell, L42, on 20 pairs per family:
- upper stack: max 4e-4 nats (brew) and 5e-4 (chain); RMS 1.4e-5 · G and 1.1e-5 · G.
- the same perturbation in the bf16 lower stack (start token, L2): max 0.12–0.16 nats, RMS ≈ 0.004 · G.

Every negligible call at L ≥ 40 is about 4 orders of magnitude above the floor.

## 4. Where a single cell carries leverage — brew (82 pairs, 33 tables)

![brew profile](../../results/001_leverage_profile/figures/profile_brew.png)

S is the median over pairs. The interval is a 95% percentile bootstrap of that median, resampling tables (the
clusters). (An earlier version printed the table-mean cluster-t interval next to the pair median: two different
estimands. Table means and their intervals are in `audit_followups.json`, `brew_estimands`.) Fractions are over pairs:
- "toward" = S ≥ 0.10;
- "negligible" = |S| < 0.05;
- "away" = S ≤ −0.10.

| Cell | L40 | L42 | L44 | L56 | L60 |
|---|---|---|---|---|---|
| `:` after `Answer` | 0.35 [0.28, 0.39] | 0.48 [0.40, 0.51] | **0.50 [0.46, 0.53]** | 0.70 [0.66, 0.74] | **0.74 [0.72, 0.77]** |
| newline after `assistant` | 0.29 [0.25, 0.32] | 0.29 [0.26, 0.34] | 0.32 [0.28, 0.36] | 0.14 [0.12, 0.16] | 0.13 [0.11, 0.15] |
| `\n\n` after `</think>` | 0.00 | 0.01 | 0.03 | 0.06 | 0.04 |
| start-colour token | **0.23 [0.19, 0.28]** | 0.15 [0.12, 0.18] | 0.12 [0.09, 0.15] | 0.00 | 0.00 |
| last question token `?` | 0.03 | 0.07 | 0.09 [0.05, 0.11] | 0.05 | 0.04 |
| every other start/question token, every stir token | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

- **The `:` cell at L42–44 carries about half the gap in every pair:** 100% toward, 0% negligible, 0% away. Its share
  rises with depth to 0.74 by L60.
- **The start-colour token still carries about a quarter of the gap at L40.** It falls to 0 by L56: the start
  position's own state still influences this answer in the middle layers.
- **What a whole-cell share does not say.** The patch replaces all 5120 dimensions. In a start pair the start, the
  intermediate, the answer and everything downstream differ together, so S measures the influence of that combined
  difference. It cannot say which content (z, start, answer, or something else) produces the effect. A
  content-specific test is still needed, and a reader-free mean-difference direction does not provide one
  automatically, since the direction can carry the same mixture.
- **The newline after `assistant` carries about 0.3 at L40–44.** The `\n\n` after `</think>` carries almost nothing,
  although it sits between the two.
- **Spans** (whole region, all five layers jointly):

  | Span | Median S |
  |---|---|
  | emission (3 cells) | 0.92 |
  | start | 0.24 |
  | question | 0.17 |
  | start + question + emission | **1.00** |

  The whole effect is recovered by these regions; no other position is needed.
- **No cell moves the answer the wrong way:** "away" is 0% everywhere.
- **Reverse direction** (partner ← member, 30%) agrees with the forward direction: Pearson 0.84 over 375 cell-layer
  pairs; median |difference| 0.065.

**Within-position variation (diagnostic):** cells where at least 10% of pairs are toward and at least 10% are
negligible, with SD above 3× the noise:
- start token at L42;
- last question token at L42/44/56;
- question token #3 at L40;
- `\n\n` after `</think>` at L56/60.

The `:` and `assistant` cells at L40–44 show no such variation: whole-cell leverage there is uniform across pairs. So
whole-cell leverage alone cannot supply a used / not-used contrast at the emission cells. The content-level contrast
(§6–§7) has to.

**Damage flag** (pre-registered: an off-pair mass increase > 0.10 in > 20% of pairs means damage-dominated):

| Cell | Layers | Pairs flagged |
|---|---|---|
| `:` | L40–60 | 46–67% |
| start token | L40 | 61% |
| newline after `assistant` | L40 | 46% |

So these cells are **damage-dominated by the pre-registered rule, and the flag stands.**

Off-pair mass here is next-token probability outside both answers' first-token sets (not a full-sequence answer
statistic). At the `:` cell (82 pairs, means):

| | L42 | L44 | L60 |
|---|---|---|---|
| Pairs flagged (increase > 0.10) | 55/82 | 49/82 | 38/82 |
| Off-pair mass, clean → edited | 0.19 → 0.35 | 0.19 → 0.34 | 0.19 → 0.30 |
| Edited: mass in the saved top-5 tokens | 0.76 | 0.77 | 0.80 |
| of which a and a′ | 0.63 | 0.63 | 0.67 |
| of which z and z′ | 0.093 | 0.091 | 0.054 |
| of which other colours, format tokens | 0.045 | 0.052 | 0.077 |
| Edited: mass outside the saved top 5 | 0.24 | 0.23 | 0.20 |

Only about 0.14 of the 0.35 off-pair mass is visible in the saved top 5, and the intermediates account for about 0.09
of it. An earlier version said "most of the off-pair mass lands on the intermediate colours". The saved data do not
support that: about 0.23 lies outside the top 5 and is unaccounted for, and no clean-vs-edited decomposition of the
intermediate mass was saved. Probability moving to the wrong intermediate would itself be a failure of selective
transfer, not a reason to waive the flag.

Proposal for AMENDMENT_02: log the full 12-colour and non-colour distribution, clean and edited, at every flagged
cell; report intermediate mass, non-colour mass and top-1 retention separately.

## 5. Chain (48 pairs, 12 items, 9 step groups)

![chain profile](../../results/001_leverage_profile/figures/profile_chain.png)

- **At L40–44 no single question-span cell carries the answer.**
  - Final `\n\n` cell: S ≈ 0.001 at each of L40, L42 and L44 (widest interval, L44, group means over the 9 step
    groups: [−0.003, 0.012]); 96–100% negligible.
  - Every other question-span cell is about 0.
- **At L56–60 the final cell carries 0.66–0.69** (pair medians; 95% bootstrap of the median resampling the 9 step
  groups: [0.56, 0.76] and [0.51, 0.75]), with the newline after `assistant` at 0.28 (L56). Group means are 0.62 [0.49, 0.74]
  and 0.60 [0.44, 0.75] (cluster-t over 9 groups). An earlier version clustered by item (12), not by step group, the
  independent unit. Pairs per group: 10, 8, 7, 6, 4, 4, 4, 3, 2.
- **Spans** across L40–60: the input digits carry 0.98, the question span 1.00. With the per-cell zeros at L40–44,
  this is consistent with the answer-relevant difference still sitting at the input digits at those layers and
  reaching the end of the prompt only after L44 (whole-cell patches; which content moves is not tested).
- **Reverse direction:** Pearson 0.97 over 70 pairs.

Limitation: only 9 independent step groups, and 22 of 48 pairs share z = 5. Chain stays a secondary pilot, as the
design allowed.

## 6. Off-trajectory substitution (43 items)

Bound (§3.3, provisional):
- |Δlog P(a)| < ε′ · median G = 0.05 × 9.59 = **0.48 nats**;
- total variation (TV) over the 12-colour answer distribution < **0.05**.

| Measure | Result |
|---|---|
| Behaviour (natural prompt): within bound | **39 / 43 (90.7%)**; median \|Δlog P(a)\| 0.017, median TV 0.019 |
| Transplant of the substituted state at the emission cells: within bound at every cell × layer | **41 / 43 (95.3%)**; per cell × layer 97.7–100% |
| Substituted key token at L0 (reference) | median \|Δlog P(a)\| 0.020 |
| Removed colour = T(a), the answer's successor (3 items) | behaviour 2/3 and transplant 2/3 within bound; c unrelated (40 items): 92.5% and 97.5% |

**All four items over the bound fail only TV, and narrowly** (0.053–0.096; |Δlog P(a)| ≤ 0.11 nats). Each one moves a
small tail mass from the removed colour to the inserted one.

Example, `0107` (green → black, answer gold):

| | gold | green | black |
|---|---|---|---|
| original | 0.55 | 0.04 | — |
| substituted | 0.62 | — | 0.05 |

(Next-token probabilities of the lower-case tokens; "—" = not in the top 5. On the 12-colour answer distribution, green
falls 0.057 → 0.001 and black rises 0.0004 → 0.062.)

So the answer is insensitive to the substitution, but its tail is not: the model's output distribution carries a
trace of which colours are in the table.

Caveat: the 90.7% behavioural figure sits just above the 90% line (Q12). That margin is fragile.

## 7. Decodability (probes; natural dev renders, folds grouped by table)

12-way balanced accuracy (chance ≈ 0.083); the Hewitt–Liang control task sits at 0.05–0.12:

| Target | `:` L40 | `:` L44 | `:` L60 | assistant-newline L42 | `</think>`-newline L42 |
|---|---|---|---|---|---|
| z (intermediate) | **0.77** | **0.75** | 0.68 | 0.81 | 0.81 |
| start | 0.81 | 0.74 | 0.53 | 0.77 | 0.85 |
| answer | **0.17** | **0.18** | **0.83** | 0.19 | 0.14 |

- **z and start are strongly decodable at every emission cell at L40–44** (BH-significant, selectivity 0.61–0.77).
- **This linear probe decodes the answer poorly at L40–44:** balanced accuracy 0.14–0.26, and only 3 of the 9 cell ×
  layer combinations pass the selectivity rule, all weakly (selectivity 0.11–0.16). It decodes it well at L56/60
  (0.83–0.89).
- So at the verbalizers' layers, start and z are linearly decodable from the emission cells and the answer much less
  so. That ordering is consistent with a two-step computation. It does not show that the answer is absent or not yet
  computed at L40–44; a weak linear probe is not evidence of absence.

**Off-trajectory membership** ("is colour c in this table?"; 11 colours had support; pooled AUC):
- **0.57–0.63 at L40–56.** Above the permutation null (BH p ≤ 0.02 everywhere), but selectivity over the control task
  is < 0.10, so not decodable by the pre-registered rule.
- **0.76 at `:` L60 and 0.72 at `</think>` L60.** Selectivity 0.19 and 0.13: decodable by the rule.
- **Both calls are provisional; the probe controls need repair** (found in the audit, §15):
  - The membership control task (a random label per table, scored on held-out tables) should sit at AUC ≈ 0.5. It
    does not: median 0.58 over the 165 colour × cell × layer fits, 71 above 0.6, 13 above 0.7, maximum 0.84 (red,
    `:` L56, where the real AUC is 0.53). The cause is not yet diagnosed. One candidate is pooling decision scores
    across folds with different intercepts, under heavy class imbalance (e.g. red 145 : 40). Until it is fixed,
    selectivity against this control is not interpretable. The 12-way controls are fine (0.05–0.12).
  - The permutation null reuses the regularisation C chosen per outer fold on the real labels, while the real and
    control fits tune C by inner cross-validation. So the null does not repeat the full fitting procedure, and the
    p-values are not yet usable as a label gate.
- A failed rule at L40–44 is not evidence that membership is absent there: the directional test below follows the
  edit in most items.

**Cross-fitted directional test** (probe fitted without the item's table, applied to its original and substituted
renders):

| Layer | Removed colour's score falls | Inserted colour's score rises |
|---|---|---|
| L40–44 | 67–82% of items | 77–89% |
| L60 | 98% (`:`), 98% (`</think>`) | 98%, 94% |

Mean logit shifts at the `:` cell, L60: −1.90 and +2.14.

Probe power (from a synthetic check; it predates the control problem above): with about 180 renders in 5120 dimensions, a single strong direction in isotropic
noise only reaches AUC about 0.68. The z/start/answer results show the probe works on strong signals. The weak
L40–56 membership AUCs may be a limit of the probe as much as of the representation.

## 8. Reader snapshot (descriptive; 10 dev tables; 32 min; run after the reader-free outputs were hashed)

**Protocol** (design §3.7 + A1):
- **Tables:** the first 10 dev tables with a gated start pair and a gated substitution.
- **Activations per table:** the member, its start partner, and the substituted render.
- **Cells:** the three emission cells.

| Reader | Layers | Samples |
|---|---|---|
| Lenses: J_NP (cosine and standard), J_CB, R_CB, logit lens | L20–60 every 4, plus L42 | top-10 tokens |
| Oracle Lens, WSB protocol | the same grid | 1 per cell |
| Oracle Lens, contract | L40 / L44 at `:` | 4 |
| NLA | L42, all three cells | 4, plus reconstruction (FVE) |

**Arms:**
- real: this cell's own activation;
- none: no activation, for the verbalizers;
- shuffled: the same cell from another snapshot table. Scored against this table's colours, this gives the chance rate
  of naming them.
- SEC: the same cell captured in the SEC regime.

Colour naming is by word match (English + Chinese colour characters) over the 12 brew colours, not a judge. **A match
is a mention, not a claim:** it does not check that the reader gives the colour the right role, endorses it, or
describes the computation correctly. The token lenses are scored on 10-token lists and the verbalizers on long
descriptions, without matched claim budgets, so rates are not comparable across reader types. Full readouts are in
`analysis/snapshot.json`; rates are in `analysis/snapshot_summary.json`; the post-audit contrasts in
`analysis/audit_followups.json`.

**Bridge (stock vs SEC capture).**
- cosine ≥ 0.99985 at every layer;
- lens top-10 lists agree 99.4% on average (minimum 80%);
- Oracle/NLA naming rates match between the two arms to within sampling noise.

Reader outputs do not depend on the capture regime, so SEC assay results and stock-fed reader outputs can be set side
by side (the design's bridge condition holds).

**Who names what at the `:` cell (member activation; fraction of readouts naming the item's colour).**

| Reader (real arm) | z | start | answer | an off-path candidate | Same reader, shuffled arm (z / off-path) |
|---|---|---|---|---|---|
| Any lens, L20–48 (12-colour palette) | **0** | 0 | 0 | 0–0.1 (J_NP cosine L36: “Green” in 1 table) | 0 / 0 |
| Any lens, L52 | 0 | 0 | 0.4–0.9 | 0 | |
| Any lens, L56 | 0–0.1 | 0 | **1.0** | 0.1–0.2 | |
| Any lens, L60 | 0.2–0.7 | 0–0.1 | **1.0** | 0–0.5 | |
| Oracle, 1 sample, L20–44 | 0.2–0.7 | 0.2–0.4 | 0–0.2 | 0.3–0.5 | |
| Oracle, 1 sample, L52–60 | 0.5–0.6 | 0.1 | **1.0** | 0.3–0.5 | |
| Oracle, 4 samples, L40 | **0.53** | 0.35 | 0.28 | 0.55 | 0.38 / 0.60 |
| NLA L42 (`:` / `</think>` / assistant cells) | **0.43 / 0.60 / 0.55** | 0.03–0.30 | 0.15–0.23 | 0.30–0.45 | 0.20–0.25 / 0.30–0.63 |

- **The smoke-test contrast recurs, but not on every table.** (An earlier version said "holds on all 10 tables".)
  - At L40–44 the probes decode z from the `:` cell (bal. acc. 0.75–0.77), and the whole cell carries about half the
    pair difference (§4).
  - The NLA mentions this table's z in 17 of 40 samples (shuffled: 8 of 40). The 40 samples are 4 per table on 10
    tables, and only **6 tables** have any z mention at `:`.
  - No lens lists a brew-palette colour in its top 10 at L40–44; the J lenses show task-structure tokens instead
    ("second", "again", "next", "final"; 第二次 "second time", 再次 "again"). Outside the palette, J_NP cosine lists
    “YELLOW” at L28 in all 10 tables (a constant lens token) and at L40/L42 for table 0069.
  - The lenses name the answer from L52 and z only at L60, where z is the model's runner-up output.
- **Mention is not role.** Author reading of the 17 NLA mentions of z at `:` (not a scored judgment; listed in
  `audit_followups.json`, `nla_colon_z_mentions`):

  | Role the explanation gives z | Count | Example |
  |---|---|---|
  | the first step's result, which the next step continues from (correct) | 5 | 0027: "Answer: pink … second iteration requires continuing from pink" |
  | the start of a two-step chain | 7 | 0104 (orange → green → black): "two steps from green yields red" |
  | a candidate answer | 2 | 0075: "likely … gray" |
  | incidental or muddled | 3 | 0073: a rule entry "red → cyan" |

  The count of 17 is therefore not 17 correct intermediate explanations. Hypothesis, not a finding: the NLA most
  often places z where the *remaining* computation starts, which at this cell is one step from z.
- **Some activation-sensitive signal survives a within-table contrast** (contrast chosen after the run, from the
  audit). Keep the table fixed and swap in the partner's start, which changes the correct z:
  tracking = [(member names z − partner names z) + (partner names z′ − member names z′)] / 2, per table, using
  each condition's 4 samples.

  | Reader, cell | Tracking, mean over 10 tables | Tables + / − / tie | Real − shuffled (z) | Own z − other tables' z |
  |---|---|---|---|---|
  | NLA L42, assistant newline | 0.35 | 8 / 1 / 1 | 0.30 | 0.35 |
  | NLA L42, `</think>` newline | 0.30 | 6 / 1 / 3 | 0.40 | 0.37 |
  | NLA L42, `:` | 0.16 | 7 / 1 / 2 | 0.23 | 0.22 |
  | Oracle L40, `:` | 0.14 | 4 / 1 / 5 | 0.15 | 0.19 |
  | Oracle L44, `:` | 0.24 | 9 / 0 / 1 | 0.03 | 0.07 |

  "Own z − other tables' z" compares naming this table's z with naming other tables' z values that are not on this
  table's path: a control for the readers' colour preferences. Those preferences are strong: across all real readouts
  at `:` the NLA names red in 62% and green in 55%, the Oracle (L40) blue 93%, red 90%, green 78%, black 62%; neither
  ever names teal or gold. Bootstrap intervals and unadjusted sign-flip p-values are in `audit_followups.json`; at
  n = 10 none is a confirmatory test. This shows some item-specific signal in the activations the readers see. It
  does not show faithful role assignment or causal use, and Oracle and NLA read different layers, so this is not a
  matched-layer comparison (no Oracle L42 arm exists).
- **Agreement was not measured.** Rates of mentioning a colour are not matched claims. On structure the readers
  plausibly agree: the J lenses list "second / cycle / again" and the verbalizers describe an iterated transformation.
  On entities and roles they disagree. Separating task structure, entity identity and entity role is the next
  analysis.
- **One lens colour at L20–48:** J_NP cosine at L36 lists “Green” for table 0015 (an off-path candidate there) and
  for table 0009's partner (its answer). The same capitalised token at the same lens and layer in two unrelated
  tables looks like a lens-level token, not table content (n = 2). At L40–44 no lens names a colour.
- **Off-path candidates are named about as often with a shuffled activation as with the real one** (Oracle 0.55 vs
  0.60; NLA 0.30–0.45 vs 0.30–0.63). On these tables the verbalizers' off-path mentions show no dependence on the
  activation.
- **No substitution tracking was detected** (substitution-tracking contrast, same form as above: −0.05 to +0.03 for
  every verbalizer arm; mostly ties).

| Reader, `:` cell | Removed colour c, original → substituted | Inserted colour c′, original → substituted |
|---|---|---|
| NLA L42 (real) | 0.12 → 0.10 | 0.28 → 0.15 |
| Oracle L40, 4 samples (real) | 0.28 → 0.30 | 0.42 → 0.45 |
| Oracle L44, 4 samples (real) | 0.30 → 0.35 | 0.25 → 0.30 |
| Lenses L56 / L60 | 0–0.2 → 0–0.1 | 0 → 0–0.1 |

The changes are within the shuffled-arm noise (n = 10 tables). This is an exploratory null: 10 tables, 4 samples,
mention scoring. It does not show that the readers cannot report the change, and the probe's 98% at L60 (§7) says
nothing about the sensitivity of verbalizers reading L40–44.

- **NLA reconstruction mostly reflects the shared template, with a small source preference.**

  | Explanations | FVE against this cell (median) |
  |---|---|
  | Own (real) | 0.49 |
  | Shuffled, from another table | 0.48 (0.50 against its own source) |
  | No activation | −0.35 |

  Close means do not imply no item information. For the same shuffled explanation, FVE against its true source cell
  beats FVE against the wrong target cell in 244 of 360 cases (mean +0.016; positive in 7 of 10 target tables). The
  360 share tables, positions and samples, so that is not an independent n. On these near-identical prompts FVE is
  dominated by the shared template and carries a small item-specific signal. Whether that signal can rank claims or
  pick the best of N samples is untested.

**Qualitative example** (table 0009: gold → **red** → purple; substitution pink → blue; shuffled from table 0015):

| Reader | Output |
|---|---|
| J_NP cosine, `:` L40 | `kembali`, `다시`, `снова`, `послед`, `第二次`, `再次`, `Cycle`, `Sequence`, `AGAIN`, `novamente` ("again"/"second time" in five languages) |
| J_NP cosine, `:` L44 | `SECOND`, `AGAIN`, `FOLLOW`, `NEXT`, `второй`, `FINAL`, `END`, … |
| J_NP cosine, `:` L60 | `purple`, `Purple`, `紫色`, … `pink` |
| Oracle L40 (real, sample 1) | "- Blue (Explanation: Green → Red, then Red → Blue after two steps) - Red (or Blue …) - Blue, Red …" |
| Oracle L40 (no activation) | Korean company names (㈜) |
| NLA L42 (real, sample 1; FVE 0.55) | "…a chain-following puzzle with specific examples ("Input: **red** → output: red → next: green") … a color cycle lookup table (red→cyan, cyan→blue …) …" |
| NLA L42 (shuffled, sample 1; FVE 0.53) | "…a color transformation chain step … "orange -> yellow -> ?" …" |
| NLA L42 (no activation) | "Educational product review pattern … a knitting pattern book …" |

The NLA centres its story on **red**, the hidden step, and surrounds it with invented content (cyan, shell commands).
The Oracle names red and blue in invented chains.

Limitations: 10 tables; word matching, not a judge; mentions, not roles; no matched claim budgets; one prompt
template; 1–4 samples per reader and cell; Oracle and NLA at different layers; no prompt-only baseline. Descriptive
only. No reader result here enters any decision.

## 9. Decision rules (design §6)

| Rule | Outcome |
|---|---|
| 1. Within-position variation (diagnostic) | Present at the start token, the last question token and the `</think>` newline; **absent at the `:` and assistant-newline cells at L40–44**, where every pair is "toward". Whole-cell leverage does not supply the contrast at the emission cells; content level does (rule 2) |
| 2. Brew supplies a not-needed class | **Provisional yes, only at L60.** Off-trajectory membership passes the rule at `:` and `</think>` L60, and the substitution is behaviourally and transplant-negligible (90.7% / 95.3%). But the membership control task is far from chance and the permutation null does not repeat C tuning (§7), so the call waits on the probe repair. At L40–44, where the Oracle and NLA read, the rule fails, which is not evidence of absence: the directional test follows the edit in most items. L60 success does not validate an L42 not-needed class. The RHUT fallback is not triggered |
| 3. Engine checks | **Pass** (every exact check, every pair) |
| 4. Power | **Pass** (82 brew, 48 chain dev pairs), but chain spans only 9 groups |
| 5. Threshold revision (τ, ε′, bound) | Not yet; proposal in §13 |

## 10. Predictions vs outcomes (priors from design §5)

| # | Prediction | Prior | Outcome |
|---|---|---|---|
| Q1 | Gate ≥ 40% of brew and chain candidates | 55% | **True** (brew 159/361 = 44%; chain 85/99 = 86%) |
| Q2 | Start-token L0 median S ≥ 0.5 | 60% | **True** (1.00) |
| Q3 | Start token L40–44 median \|S\| < ε′ | 70% | **False** (0.23 / 0.15 / 0.12) |
| Q4 | `:` median rises L40 < L42 < L44, L44 ≥ 0.3 | 65% | **True** (0.35 < 0.48 < 0.50) |
| Q5 | Every brew question cell negligible (TOST) at L40–44 | 45% | **False** (`?` at L42/44 and token #3 at L40) |
| Q6 | Within-position variation at ≥ 1 brew emission cell × layer | 50% | **True, only at `</think>` L56/60** |
| Q7 | Same for chain | 40% | **True** (one question cell, L60) |
| Q8 | z decodable at `:` L40, 42, 44 | 70% | **True** |
| Q9 | Off-trajectory membership decodable at ≥ 1 emission cell × layer | 55% | **True (L60 only); provisional** until the probe controls are repaired (§7) |
| Q10 | Answer more decodable than z at `:` L44 | 65% | **False** (0.18 vs 0.75) |
| Q11 | Damage-dominated emission patches in < 10% of pairs | 75% | **False** (46–67% at `:`; the flag stands, §4) |
| Q12 | Substitution behaviourally negligible in ≥ 90% | 70% | **True, narrowly** (90.7%) |
| Q13 | Transplants negligible in ≥ 90% at every layer | 75% | **True** (≥ 97.7% per cell × layer) |
| Q14 | `:` median at L56 and L60 > L44 | 70% | **True** (0.70, 0.74 > 0.50) |

## 11. Qualitative sample (seed 20261003; full record in `analysis/qualitative.json`)

Brew, the `:` cell transplanted at L44:

| Pair | Member → partner answer | G | Share S at `:`, L40 / 44 / 60 | Start-token S, L40 | Top-3 after the L44 patch |
|---|---|---|---|---|---|
| 0012__red | black → purple | 8.3 | 0.32 / 0.34 / 0.55 | 0.51 | black 0.26, purple 0.18, Black 0.08 |
| 0383__red | brown → orange | 6.1 | 0.45 / 0.59 / 0.76 | 0.41 | orange 0.56, brown 0.16, Orange 0.09 |
| 0009__black | purple → green | 9.0 | 0.40 / 0.51 / 0.69 | 0.19 | purple 0.32, green 0.20, Green 0.10 |
| 0172__teal | gray → white | 6.6 | 0.26 / 0.39 / 0.70 | 0.17 | gray 0.55, white 0.18, `<im_end>` 0.05 |
| 0075__orange | teal → white | 11.2 | 0.50 / 0.60 / 0.83 | 0.13 | white 0.49, teal 0.20, White 0.10 |
| 0312__blue | purple → brown | 12.6 | 0.37 / 0.56 / 0.69 | 0.35 | brown 0.52, purple 0.13, Brown 0.08 |
| 0140__brown | blue → gold | 9.3 | 0.30 / 0.38 / 0.74 | 0.19 | gold 0.22, brown 0.18, blue 0.14 |
| 0312__red | purple → gold | 14.6 | 0.54 / 0.65 / 0.79 | 0.23 | gold 0.43, Gold 0.14, brown 0.10 |

In 0140__brown (member gold → teal → blue; partner brown → black → gold), the second-ranked token after the patch is
**brown, the partner's start colour**: off-pair mass going to a colour that is neither answer nor intermediate.

Chain:
- `0408__13` (2 → 1): final cell S 0.70 / 0.71 at L56 / L60, 0 at L40–44. After the L56 patch, "1" = 0.99.
- `0181__5` (11 → 7): final cell 0.40 / 0.48.
- `0013__5` (14 → 17): 0.33 / 0.27, with "1" = 1.00 as the first token. The digit after it carries the effect, which
  is why every answer token is scored.

## 12. Limitations

Each is stated next to its claim above; collected here:
- **Scope:** one model, dev split, two synthetic families, five layers. Chain has 9 independent groups.
- **Whole-cell shares measure the natural pair difference** (start, z, answer and everything downstream move together).
  They do not say which content is used.
- **Whole-cell shares cannot say which content matters**, and a mean-difference direction does not settle it by
  itself; a content-specific test that separates intermediate from answer information is still needed.
- **Damage:** most patches at `:` are flagged; the saved top-5 cannot say where most of the off-pair mass goes.
- **Probe controls need repair:** the membership control task sits far from chance (median AUC 0.58, max 0.84), and
  the permutation null reuses C from the real fit. Membership calls are provisional at every layer. A weak linear
  probe (answer at L40–44, membership at L40–56) is not evidence of absence.
- **Reader snapshot:** mention scoring, no role scoring, no matched claim budgets, no prompt-only baseline (the
  project rule asks for it from day one; it was not run in 001), Oracle and NLA at different layers, n = 10 tables.
- **The 90.7% behavioural substitution figure is near its threshold.**
- **SEC results are results under the SEC regime.** The stock-vs-SEC bridge for reader inputs is in §8.

## 13. Interpretation and next decision

Under these conditions (Qwen3.6-27B, brew dev, SEC regime):
- The combined pair difference at the answer cells at L40–44 has a large, uniform influence on this answer (about half
  the gap at `:`). Which content carries that influence is not yet tested.
- Start and z are linearly decodable there; the answer is decoded much less accurately by the same probe.
- Off-trajectory table contents leave a trace: the substitution barely moves the answer (90.7% within the bound) but
  shifts its tail, and a cross-fitted probe follows it, most clearly at L60. The pooled membership rule is
  provisional until the probe controls are repaired.

Reader snapshot (descriptive, 10 tables):
- The readers share task-structure information, show some activation-sensitive colour signal (the within-table
  tracking contrast), and often surround it with incorrect detail: z is often named in the wrong role.
- No substitution tracking was detected; that null is exploratory.

So the candidate disagreement is narrower than an earlier draft put it: "z is decodable at `:` L40–44 and the
combined pair difference there influences the answer; the NLA mentions z more than chance, sometimes in the right
role; the J lenses list no palette colour". Whether z itself is causally influential there, and whether any reader's
claims track it, is unresolved.

For W, this means:
- Brew offers a candidate within-cell contrast: a pair difference with large leverage, against off-trajectory colours
  that leave a decodable trace and barely move the answer. It is clean only once the probe controls are repaired, and
  at L40–44 it rests on the directional test.

Proposed for AMENDMENT_02 (before any test-split label):
- keep τ = 0.10 and ε′ = 0.05 (the profile gives no reason to move them);
- adopt the §6 bound as frozen;
- log the full answer-colour distribution at flagged cells and report the damage components separately (§4);
- repair the probes: permutation null that repeats C tuning, per-fold rather than pooled AUC (or a diagnosis of the
  control), and a control that sits at chance before any membership call is used;
- reader scoring: separate mention, correct role, and contradicted claims, with abstention allowed and matched claim
  budgets; within-table contrasts; a prompt-only baseline; frozen on dev before any held-out use;
- decide whether reader scoring of the not-needed class uses L60 (lenses) only, or also L40–44 via the directional
  criterion. This last one is the PI's decision.

## 14. Breakages and silly mistakes (logged in CLAUDE.md)

- **Commit 79f835a's message** describes design v3, but its edit script had failed; v3 landed in e3666e1.
- **The brew split** rounded odd strata at random (51/49). Caught by a test before use.
- **The noise-floor helper** drew different directions across the forwards of one perturbation. Caught in review
  before running.
- **The snapshot's render dictionary** was written without its loop: UnboundLocalError before any reader ran.
- **A concurrent session** pushed b71d1df (park guard, the T(a) tag) while 001 ran.
- **Rounding slips in this report**, found while generating the results page from the same JSON (fixed here): `:` L60
  was written 0.75 [0.71, 0.76] (data 0.745 [0.705, 0.764] → 0.74 [0.70, 0.76]; also in §4 text and Q14);
  `</think>` L42 0.02 (0.0145 → 0.01); start-token L40 interval [0.22, 0.33] (0.215 → [0.21, 0.33]); answer probe
  at `</think>` L42 0.15 (0.1447 → 0.14; §7 range now 0.14–0.26). No conclusion changes. The chain L40–44 interval
  was L44's, unlabelled; now labelled. The page now draws every number from the JSON with round-half-up.
  The §8 row "Any lens, L20–48" said 0 for off-path candidates; it is 0–0.1 (one J_NP cosine readout at L36).
  - The run finished on 647dcb5 without the tag; the analysis back-fills it from the bank (f445816).
  - The push from this session was rejected and rebased, not forced.
- **Overstatements caught by the external audit** (§15), all corrected in place:
  - "most of the off-pair mass lands on the intermediates", read off the top-5 tokens only (they hold about 0.14 of
    the 0.35);
  - pair medians printed with table-mean intervals;
  - chain clustered by item instead of step group;
  - "holds on all 10 tables" (6 tables have any z mention);
  - "FVE reflects the template, not item-specific content" (a small source preference exists);
  - "not yet the answer" from a weak probe;
  - the substitution null stated as a finding.
- **Probe controls were not checked before use:** the membership control task sits far from chance, and the
  permutation null reuses C from the real fit. Not yet fixed: needs a probe re-run (§13 proposal).
- **No prompt-only baseline was run in 001,** although the project rules ask for it from day one.

## 15. External audit (Codex, 2026-10-02) and follow-ups

The audit (`/workspace/codex_reports/2026-10-02_reader_audit.md`, outside the repo) read the raw outputs, code and this
report; it ran no model. Every number it gave was recomputed independently by
`scripts/001_leverage_profile/audit_followups.py` from the saved outputs and matched.

| Audit point | Verified | Change |
|---|---|---|
| Colour mentions read too strongly; mention ≠ role; "all 10 tables" too broad; J "no colour" only within the palette | Yes; 6/10 tables; YELLOW at L28 in all tables and at L40/42 for 0069 | §8 rewritten; role reading of the 17 mentions added |
| A within-table start-pair tracking contrast shows activation-sensitive signal | Yes (0.35 / 0.30 / 0.16 / 0.14 / 0.24) | Added to §8 with a colour-preference control |
| Reader agreement not measured | Yes | Stated in §8; listed as the next analysis |
| Whole-cell effects do not adjudicate the intermediate; a weak answer probe is not absence | Yes | §4, §7, §12, §13 wording |
| FVE conclusion too categorical | Yes (source wins 244/360, +0.016) | §8 |
| Damage explanation incomplete | Yes (top 5 holds 0.77 of mass; off-pair 0.35, intermediates 0.09 within the top 5) | §4 rewritten; the flag stands |
| Table-mean intervals next to pair medians | Yes | §4 now uses a bootstrap of the median |
| Chain clustered by item, not step group | Yes | §5 recomputed over 9 groups |
| Permutation null reuses C from the real fit | Yes (`leverage_profile.py`, `_one_target`) | Flagged; needs a probe re-run |
| Membership controls up to AUC 0.84 | Yes; worse than stated: median 0.58, 71/165 above 0.6 | §7, §9 marked provisional; needs diagnosis |
| Substitution null is exploratory | Yes | §8 |
| No prompt-only baseline, matched budgets or causal reader ranking | Yes | §12, §14 |
| No Oracle L42 arm, so no matched-layer comparison | Yes | §8 |

