# 012: Reader concordance on demonstrated use. Swaps on multihop, readers on clean and swapped states (design spec v1)

**Status.**
- Draft v1, 2026-10-06; branch `012-reader-concordance` (off `008-trackb-brewcode`).
- EXPLORATORY under the relaxed rules.
- Each stage stops after its smoke and reports. The full run of a stage needs the PI's go, and any run over about 2 GPU-h needs a check-in.
- Decision rules are fixed in this note before the stage they govern. Changes after a run starts are written as amendments.
- Wording rules:
  - "demonstrated influence on this answer", never "uses";
  - "decodable through the reader", never "represents";
  - every claim scoped to model, family, position, layer and n.
- Background: report 32 (outlook, plus §8 on Jan's writeup).

## 1. The question (PI, 2026-10-06)

> Cross-reference the interpretability techniques (J-lens, NLAs, Oracle Lens, skip lens, sparse autoencoders). What do they
> agree on? What can some find that others miss? Is it true that NLAs and SAEs find non-workspace things in a way that
> J-lens and Oracle Lens don't? Can we use these techniques and their agreement to establish some tentative ground truth
> for interpretability that the others can be measured against, e.g. rating each technique by its agreement with the
> majority? What can NLAs capture that Oracle Lens / J-lens don't? Is it really the stuff not in the workspace? How
> consistent are the different methods? How much better does interpretability get if you ensemble them? How much do they
> overlap? What is caught by some but not others?

**Operational version.** All on Qwen3.6-27B, on items where we can label the model's dependence without any reader.

| | Question | Measured as |
|---|---|---|
| Q1 | **Agreement.** Do readers name the same content on the same item and cell? | Cohen's κ matrix, per-item overlap |
| Q2 | **Coverage.** What does each reader find that the others miss? | Per-item union and intersection, plus hand-read categories: the bridge; a restated cue word; genre boilerplate (e.g. the NLA's "weather" text); a wrong entity; an unsupported claim |
| Q3 | **Validity against demonstrated use, not against the majority.** | Each reader, and the majority vote as one more "ensemble reader", is scored against reader-free labels: swaps that move the model's answer, ablations, and later whole-cell patches |
| Q4 | **Ensemble gain.** | Union-vs-best and majority-vs-best against the labels, paying for false positives (off-path concepts, random slices, the prompt-only baseline) |
| Q5 | **Is the NLA's extra content outside the workspace?** | The pieces (whole / J / rest-of-J / random slice) per reader, with the same content categories as Q2. This is the reader-side split already in 007, extended to the swap items |

**Why majority agreement cannot be the ground truth.** Readers share failure modes, so agreement can be shared error:
- **Prompt inference.** Every reader can name what the prompt implies. WSB's prompt-only baselines reach 0.5–1.0 on several families.
- **Output-facing content.** The hidden intermediate sits in the model's own output tail (006b, 009).
- **Shared training.** The Oracle and the NLA are both trained on assistant text, and the Oracle's AR config names a J-lens term (011 note).

That is why SPEC_v2 decomposes agreement into shared signal, shared error and shared prompt inference. 012 keeps that frame:
- labels from interventions decide what is true;
- the prompt-only floor and the model's own output distribution are measured as covariates;
- agreement and ensembles are scored against the labels.

**Relation to SPEC_v2.** Same frame: reader-blind causal labels, scored per cell, with agreement decomposed. What is new:
- multihop gets model-side labels for the first time (swaps in the mid band, where they move the answer);
- readers are scored on swapped states, i.e. does a reader's content follow a state the model demonstrably acts on.

## 2. Material

**Jan's phrase-lens banks** (`chen-jan/phrase-lens` @ ea85583; Jan cleared their use):

| Bank | n | Notes |
|---|---|---|
| causal multihop ST | 96 | Every item is a WSB multihop item with identical prompt and bridge; 49 are in 007's test half. **Japan is the target in 29**; 53 distinct targets |
| causal multihop MT | 67 | Jan's own items; 16 flagged for scrutiny |
| causal broadcast ST | 192 | 164 headline |
| causal broadcast MT | 192 | 112 headline |

**Also used:** the WSB multihop bank and 007 Stage 3's clean reads (Oracle L40/L44, NLA L42, pieces).

**Handling:**
- Gate the actual intervention prompt (Stage B).
- Scrutiny items go in separate tables.
- Every rate is broken down by target, and by relation cluster (36 keys for ST).
- The 14 ST items with a multi-token src are labelled.
- **Our reader reads use WSB's `last_word_token` position.** Jan's code edits every prompt position, so its read position does not matter for the swaps.

**Not duplicated here:** the reader-side J/rest split on the WSB readout banks is the partner's track. 012 provides its model-side counterpart and the swapped-state reads.

## 3. Instruments

**Readers** (reader code never enters label code):
- Token lenses J_NP, J_CB and logit.
- Oracle `olens_s3d_rl600`, on-contract at L20–60 step 4.
  - L40/L44 primary.
  - L52/L56 when the edit band reaches L47.
- NLA `nla-rl` iter_000400, L42 only.
- Template lens:
  - Jan's rebuilt lens if he shares the tensors (8.2–16 GiB);
  - otherwise Camila's, labelled "uneven passages; failed 007's sanity check".
- Prompt-only floor (007's).
- **SAE:** none for Qwen3.6-27B in our stack. WSB evaluated an L42 SAE; we need its source.
- **Skip lens:** unknown to us; we need a pointer.

**Causal handles:**
- J_NP rows.
- Logit rows, as a control: Jan finds about 2% swap success with them.
- Class-mean directions, for MT concepts and the Fig. 16 split.
- Random controls:
  - Jan's: true src, random target;
  - isotropic pairs, norm-matched.
- Oracle-AR phrase vectors: Stage E, blocked.

## 4. Stages

### Stage A: reference lever with Jan's code, unmodified (model side only, no readers)

- **Code:** `scripts/012_reader_concordance/stageA_prep.py` (CPU vocabulary) and `stageA_queue.sh` (smoke / full).
  - Jan's CLI runs from the read-only clone through `PYTHONPATH`; nothing is installed into `/venv/main`.
  - It builds J_NP and logit token banks from our pinned J_NP file.
  - 108 of the 117 ST concepts resolve to one token. The unresolved ones (11, 12, Los Angeles, New York, New Zealand, San Francisco, Sherlock Holmes, octagon, polar bear) are reported as uncovered.
- **Conditions** (all ST, `--cohort all`, α 1, every prompt position, prefill only, bf16 as in Jan's runs):
  - J swap at L24–47, L24–39, L24–35, L36–47 and L40–48 (the last for comparison with 008);
  - J ablation at L24–47;
  - logit swap at L24–47;
  - J broadcast swap at L24–47.
- **Measures** (from Jan's per-item output):
  - clean-answer correctness;
  - swap success (y_b in the greedy continuation, at most 8 tokens) against its random control;
  - ablation survival against its random control;
  - per-target and per-cluster rates, with Wilson intervals.
- **Reading rule:**
  - **Lever present at a band:** J swap success ≥ 30%, and at least 20 points above its random control.
  - **Expected** (Jan's figure): about 59% at L24–47, against about 5% random.
  - The band results decide where Stage C can read downstream of the edit:
    - L24–39 keeps the NLA (L42) usable;
    - L24–35 keeps Oracle L40/L44 usable.

### Stage B: our swap engine (labels with margins; parity with Stage A)

- **Primitive:** 008's clamp, c = R(h − μ), h ← h + A(t − c) with R A = I.
  - t is set from the clean run's coordinates at every edited layer and every prompt position, on prefill only.
  - Generated tokens are never edited.
- **Regime.** SEC's float32 replay starts at block 40 and cannot cover edits at L24–39 (the float32 weights would not fit in 96 GB). Use the float32-residual regime (`engine.Engine(model, fp32_from=0)`) and measure its noise floor per item (repeated clean runs and zero edits).
- **Scores:**
  - Δm, teacher-forced: log-sum-exp over every accepted spelling, every answer token, float32 logits;
  - flip (greedy, 8 tokens);
  - ablation survival.
- **Gates:**
  - G0: a zero edit reproduces the clean run.
  - G-clamp and G-RA, as in 008.
  - **G-parity:** per-item flip agreement with Stage A's J swap at L24–47 of at least 90%. Our engine is the second path; Jan's is the reference.
  - Any speed-up must reach parity on the full set before it is adopted (007's lesson).
- **Clean gate on the actual prompt:** greedy y_a, at least 9 of 10 correct at T 0.8, and the bare prompt does not already give y_b. Items that fail are labelled, not deleted.
- **Arms:**
  - J raw;
  - J whitened (Σ for L24–47; Ledoit–Wolf on the UltraChat corpus as in 008; rows as covectors, Ṽ = W⁻¹V);
  - logit;
  - the two random controls;
  - an answer-swap positive control;
  - class-mean directions (10–16 cue-varied, gated prompts per concept; the item under test left out), split into full / k = 25 J part / non-J remainder, with complementary clamps (the paper's Fig. 16), for ST and MT.
- **Per-item label:**
  - **works:** Δm exceeds the larger random control by at least 1 nat **and** the greedy output flips;
  - **partial:** exactly one of the two holds;
  - **fails:** neither holds.
- **Captures:** swapped residuals at the read position (WSB `last_word_token`) at every layer at or above the band end, for Stage C.
- **Positive control:** if the answer swap fails at a band, J rows are not read for that band.

### Stage C: readers on clean, swapped and real-target states

- **States read:**
  - clean;
  - swapped (each arm that works);
  - a **real tgt prompt**, i.e. a prompt that actually implies the swapped-in concept: the reader's positive reference.
- **Reads are always downstream of the edit:**
  - Oracle L52/L56 for L24–47 edits;
  - Oracle L40/L44 for L24–35 edits;
  - NLA L42 for L24–39 edits (needs Stage A's band result);
  - token lenses at the same layers.
- **Subject before NLA.** Subject activations are captured first; the NLA (53.8 GB) loads only after the subject is freed.
- **Scores:** naming rate for src and tgt (word match, plus hand-reads of seeded samples), and likelihood D for src vs tgt candidates (006's scorer conventions; per-reader noise floors).
- **Pieces:** whole / J / rest-of-J / random slice, with 007's operators.
- **Prediction, written now:** the Oracle follows a working swap (names tgt) more often than the NLA does, because 007 found the Oracle's signal J-dependent and the NLA's not.
- **Reading rule:**
  - The comparison is between working swaps and failed or random ones.
  - A reader naming tgt equally after failed swaps is reading the edit, not the model's state.
  - Naming src after a working swap is the default for a faithful whole-vector reader. The swap moves a 2-dim span, and 84–94% of the src content sits outside J (the paper's Fig. 8: 6–7%; Fig. 16: ~16%).
  - So "reader follows the model" is judged only relative to the real-tgt reference.

### Stage D: concordance analysis (CPU; runs alongside the GPU stages)

Q1–Q5 on the joined table (item × reader × state × piece), with three covariates:
- the prompt-only floor;
- the model's own output rank and D for the bridge at the read position;
- J loading (the cosine between the residual and the src row).

Further rules:
- **"Used" is defined by Stage B labels from a non-J lever (class means)** as well as by J swaps. The paper shows J-swap success tracks J loading, which would make a J-defined label circular.
- **Majority vote and union are evaluated as readers.**
- **Hand-reads** cover seeded samples for the key conditions only: clean, working swap, failed swap, rest-of-J and random slice per reader. They flag restated cues, boilerplate and wrong entities.

### Stage E: Oracle-reconstructor swaps (the TA's request; blocked)

- **Needs the official loader** (`load_lc_reconstructor`), the whitener convention and the definition of the AR's loss (`loss_mix_lambda`, `jfve`). The PI is asking Camila.
- **Until then, Route A is CPU-only:**
  - repair 011's loader against the published numbers (retrieval@1 with a fixed pool of 24; whitened FVE on spans of uniform 1–32 tokens; truncated vs full backbone);
  - **G-AR:** retrieval@1 at least 0.85 at L40/L44; FVE within 0.05 of 0.22; norm check only if the loss is known.
- **Then:** AR phrase vectors, centred across concepts, at AR layers inside the band (24–44, step 4). Whitened coordinate swap and centred additive edit, with the same controls and labels as Stage B.
- **Route B (training our own reconstructor): rejected.** It isn't the Oracle's ruler, and it adds an unvalidated instrument.

## 5. Stop rules

Stop and report, without improvising, if:
- any validity gate fails (G0, G-clamp, G-RA, G-Σ, G-parity);
- fewer than 50 ST items pass the clean gate;
- the answer-swap control fails at every band;
- Stage A finds no lever at any band. In that case Stage B runs only the class-mean arm.

## 6. Acceleration and compute (one GPU, 96 GB; 128 CPUs)

- **GPU work in one queue, one stage per model load:**
  - batch prompts of equal length;
  - cache clean runs and clean coordinates once per item;
  - compute directions once per concept and layer.
- **CPU analysis runs while the GPU works** (Stage D, gating parsers, Σ shrinkage).
- **Rough estimates**, to be replaced by the smoke timings:

| Stage | GPU estimate |
|---|---|
| A, full | about 0.5–1.5 h |
| B, ST | about 1–2 h |
| B, class-mean prompt gating | about 0.3 h |
| C | about 1–3 h (the NLA is the slow part; 007's NLA runs took 1–2 h each) |
| E | about 0.5 h after the loader |

## 7. Outputs

- `results/012_reader_concordance/{analysis/*.json, manifest_*.json}` (each manifest carries `registry.manifest_block()`).
- Raw outputs in `outputs/012_reader_concordance/`, mirrored to the HF dataset.
- Report: `docs/reports/012_reader_concordance.md`.
- Every number comes from the JSON, rounded half-up, with qualitative samples beside every table.

## 8. Not claimed

- Anything beyond Qwen3.6-27B, these banks, the stated bands, positions and n.
- That a direction which works when swapped is the model's mechanism. It is evidence about that intervention.
- That a failed swap means the bridge is unused. It means unused through that direction, at that band.
- That agreement among readers is truth (§1).

## 9. Open items for the PI

1. The SAE WSB used (source and layer), and what "skip lens" refers to.
2. Jan's template-lens tensors: ask him.
3. The Oracle-AR loader and loss definition: ask Camila.
4. Stage A's band result decides the Stage C read layers. Confirm after Stage A.
5. Overlap with the partner's J/rest work: Stage D's Q5 reuses 007 operators on the swap items only.

## 10. Execution notes

**2026-10-06 22:46 UTC: Stage A smoke** (`stageA_queue.sh smoke`; 4 ST items, J swap L24–47; EXPLORATORY).
- Both token banks built (about 8 s each). The 4-item swap took 10 s, model load included, so the full Stage A is roughly 10–20 GPU-min.
- **Contamination.** On `chem-bones-Z` (calcium → iron) the clean prompt already outputs y_b (" 26"; clean_correct False), and Jan's code scores it as swap success and random success.
  - **Rule added before the full run:** Stage A rates count only items with `clean_correct` True; the others are reported separately.
  - This is the same issue Stage B's clean gate handles.
- **The other 3 swaps were no-ops** (the output was unchanged).
  - Their src/tgt J rows are nearly parallel at the start of the band (cosine 0.83–0.93 at L24–26), so a coordinate swap barely moves the activation there.
  - Expected from the paper's low-rank early J-space. It is why the band comparison in Stage A matters.
- Jan's code drops the bank prompts' trailing space when it renders. This doesn't matter for the swaps (every position is edited); our reads use WSB's render.
- Nothing else ran. The full Stage A waits for the PI.

**Multi-token concepts (PI question).**
- WSB's `multihop_mt` (100 items) has obscure 3–5-token bridges ("Simula 67") and no swap targets.
- Jan's MT bank has swap pairs (67, 16 flagged), and his figure shows a weaker but real handle: about 21–40% swap success depending on the direction family, against about 3% random.
- Isolating the effect causally is harder:
  - there is no single lens row;
  - partial-token overlap (swapping "New" in "New Zealand");
  - multi-token answers to score.
- Plan: ST first. MT second, with class-mean directions (no lens needed), plus Jan's template rows if he shares them.

## 11. v1.1 decisions (2026-10-06, after the PI's review; where they conflict with earlier sections, these win)

**Confirmed arms** (PI asked for written confirmation). Stage B runs all of these:
1. **The paper's lens-coordinate swap with clamping** (Fig. 4C, §3.1/§3.3): V = [v_src, v_tgt]; the coordinates are clamped to the swapped clean values at every band layer and every prompt position; raw and **whitened** (Σ for L24–47; rows as covectors).
2. **The paper's privilege test (Fig. 16):** class-mean directions split into full / k = 25 J part / non-J remainder, swapped with complementary clamps.
3. **The answer-swap positive control** and both random controls.
4. **The brew rerun of 008 arm A** at L24–47 at every prompt position, on 008's 82 pairs, including the answer-atom control. It tests whether 008's null was about band and positions.

**Reconstructor vectors:**
- **R1: Oracle AR (Camila's method).** Waits for her loader. Nothing goes under her name before G-AR passes.
- **R2: NLA-reconstructor vectors. Optional and deferred.**
  - With in-distribution inputs (the NLA's own explanations of clean activations whose bridge is c, averaged, then centred across concepts), R2 is a legitimate test: does the NLA's verbal channel keep the concept in a causally usable direction? With made-up explanation text it is not.
  - Costs:
    - several activations per concept, i.e. the cue-varied prompt set plus NLA sampling on each;
    - a lineage match: the NLA reconstructor we can load (`rl_critic_step400`) pairs with `nla-L42`'s verbalizer, not `nla-rl`'s. `nla-rl`'s own reconstructor (step 600) has no wrapper yet.
  - Runs only after Stage B shows class-mean directions work. It is labelled "NLA-reconstructor vectors", never the Oracle method.

**Readers:**
- **NLA at L44 added**, labelled off-training, with its own positive control. Camila suggested it.
- **SAEs and the skip lens dropped** (PI). WSB did evaluate an L42 SAE (its post; PI notes 04), but its source isn't in our stack.

**Logging** (from the Stage A smoke):
- per item and layer, the src/tgt cosine and the realised ‖Δh‖;
- "no-op" (continuation unchanged) is its own category.

**If Stage A comes in far below Jan's ~59% on clean-correct items, check for a setup difference before concluding.** Candidates:
- render: trailing space stripped;
- dtype;
- library versions: our transformers 5.15.0 / torch 2.11 vs his 5.16.1 / 2.13;
- how his hooks attach to Qwen3.6's layers;
- whether the bank rows match his J rows.

**G-parity:** per-item flip agreement and token-identical greedy continuations against Stage A. A margin correlation is impossible there, because Jan's code emits no margins. Margins are compared only within our engine, e.g. against the zero-edit floor.

**Multi-token:**
- Jan's CLI cannot run the MT bank with J rows, which are single-token only, and his PJ and template banks aren't hosted.
- MT runs in Stage B with class-mean directions (cue-varied prompts per concept), and with Jan's template rows if he shares them.
- First-token J rows are not used as phrase interventions.

**Process** (the review's item 1 is not adopted):
- Gates decide within a stage.
- New code paths (Stages B and C) still stop after their smoke for the PI's look. They are unvalidated code, and 007's fast-scorer loss came from adopting a path without a full check.
- Validated reference code (Stage A, Jan's) runs through without a separate go.
- The §6 estimates are kept and are being replaced with measured timings (Stage A full: 4 items took 10 s).

**Qualitative storage:**
- Every stage commits seeded readout and continuation samples to `results/012_reader_concordance/qualitative/*.jsonl`. Stage A's are produced by `stageA_summary.py`.
- Full readouts stay in `outputs/` and are mirrored.
- A detailed Claude artifact with plots comes after all runs.

**LLM judge** (content categories for Q2): optional. The OpenRouter key is in `/workspace/.env`; ask before spending.

## 12. v1.2: Stage A result, corrections, and the overnight plan (2026-10-06 ~23:10 UTC, written before any Stage B/C run)

### Stage A result (rule of §4: lever present = success ≥ 30% and ≥ 20 points above the random control)

Cohort: clean-correct items of the 87 covered (82); the 5 clean-incorrect items are kept and excluded. EXPLORATORY.

| Arm | Success | Random | Lever? |
|---|---|---|---|
| J swap L24–47 | 0.585 (48/82) | 0.061 | **yes** (reproduces Jan's ~59%) |
| J swap L24–39 | 0.549 | 0.012 | yes |
| J swap L24–35 | 0.463 | 0.012 | yes |
| logit-row swap L24–47 | 0.00 | — | — |

J ablation at L24–47: the original answer survives on 0.72 of items (random: 1.00).

### Correction to §10

§10 explained the three unchanged smoke swaps as near-parallel src/tgt rows making "a coordinate swap barely move the activation". **That is wrong.**
- At the clean state, this pseudoinverse swap is a reflection of the contrast component: Δh = −2u(uᵀh), with u = (s − t)/‖s − t‖ (identity from the independent review; checked numerically).
- High cosine does not shrink the edit.
- "Unchanged" means the greedy *answer* did not change. Nothing was measured about the activation edit.
- The category is renamed **unchanged**, and Stage B logs the realised edit.

### Contract (from the review; adopted)

- **Render:** Jan's (bank prompt, trailing whitespace stripped).
  - **Read position:** the final token of that render, which is WSB's `last_word_token` for every item once stripped; verified per item in B.
  - Readers read **captures from the exact labelled runs**. Stage A labels are never attached to 007's captures.
- **Cohort:** fixed. The primary analysis uses the clean-correct items whose clean continuation does not already contain y_b. Failures, "other" outcomes and random-control states are retained and analysed, not filtered by outcome.
- **Labels mean** "an answer effect under the specified intervention (item × prompt × arm × band × dose)". They do not mean the bridge is the mediator. J-swap labels are J-derived, so they are not an independent standard for ranking J-based readers. Class means are "J-independent in construction".

### Overnight Stage B-lite (model side, our captures)

**Code:** built on Jan's library objects (`SwapEdit`, `BandEditor`, `continuation_scores`, `capture_ids`), with the same edit as Stage A.

**Arms:**

| Arm | Role | Downstream reads |
|---|---|---|
| clean | reference | — |
| J swap L24–35, with its source-preserving random-target control | **primary** | Oracle L40/L44 and NLA L42 |
| J swap L24–47, with its random control | secondary | Oracle L52/L56 only |

**Recorded per item and arm:**
- the greedy continuation, with Jan's budget and matcher;
- teacher-forced m = LSE log P(y_b spellings) − LSE log P(y_a spellings), and Δm against clean;
- edit diagnostics: layers applied, ‖Δh‖ at the read position per layer, and coordinate error after the clamp;
- captures at the read position, L40/42/44/52/56.

**Gates:**
- **G-parity:** the clean and J-swap greedy continuations must be token-identical to Stage A's, reported as a paired success/failure table. Any mismatch is investigated before readers run.
- **G0:** a zero edit reproduces the clean scores.

### Overnight Stage C pilot (readers on whole activations, same states)

**Readers:**
- Oracle at L40/L44, plus L52/L56 for the L24–47 arm.
- NLA at L42, plus L44 (labelled off-training), run after the subject model is freed.
- J_NP, J_CB and logit lens at the same layers.

**Primary statistic** per reader r:

d_r = [LL_r(tgt | edited) − LL_r(src | edited)] − [LL_r(tgt | clean) − LL_r(src | clean)]

- LL uses the readers' likelihood scorers with fixed prefixes and candidate spellings (007's conventions).
- Absolute LLs are kept alongside.
- Oracle and NLA also give 2 sampled readouts per state; **all readouts are saved.** Each readout is categorised as src-only / tgt-only / both / neither by word match, plus hand-reads.

**Reading** (replaces v1's Stage C rules; the energy-share and "equal naming = artifact" rules are withdrawn):
- **Responsive:** d_r > 0 after targeted swaps, beyond the random control.
- **Outcome-specific:** d_r is larger after working swaps than after unchanged ones, and tracks Δm.
- **Concordant:** both readers show the same pattern on the same items.
- **Prediction (prospective; not implied by 007, whose multihop J/whole results were similar for both readers):** Oracle d_r > NLA d_r after working swaps.
- Japan items are reported pooled and excluded. Uncertainty is clustered by relation.

### Not tonight

The whitened arm (needs Σ for L24–35/47), the class-mean / Fig. 16 arm (needs cue-varied prompt sets), the brew arm-A rerun, MT, and Stage E. They follow in that order once B-lite and the C pilot have been inspected.

### Process (PI, ~04:00 local, 2026-10-07)

- Proceed through the stages overnight since Stage A is positive.
- The agent inspects each new smoke before its full run.
- A failed gate stops that branch and is recorded in the ledger (`docs/ledger/012_ledger.md`).
- The independent review (Codex, 2026-10-06) was adopted where listed above. Not adopted tonight: per-cell mediation transfers and the Q3/Q4 endpoint definitions (deferred until the pilot shows variation).

## 13. Stage F: separating bridge reading from answer reading (pre-registered 2026-10-07 ~00:25 UTC, before any F run)

**Why.** Stage D found that every reader's src→tgt shift at the read cell (the answer-producing token) tracks swap success. But so does the model's own output at that cell (AUC 0.96–0.97), and the bridge shift adds only a modest partial association beyond it. Two cheap cuts test whether readers carry bridge content beyond output-facing content and beyond the written edit. Everything here is EXPLORATORY.

**Shared setup:**
- Cohort: the same 87 covered items; the primary analysis uses 82 clean-correct items.
- Edit: the J swap at L24–35 (Stage A's lever band, which keeps every reader downstream).
- Controls: source-preserving random-target controls.
- Same engine, same float32 margins, same gates (G0 α = 0; edits applied only where specified).

**F1: position split** (the readers read the read cell, L40/L44 Oracle, L42 NLA, token lenses).

| Arm | Positions edited | Control |
|---|---|---|
| `J_readonly_L24-35` | the read position only | `rand_readonly_L24-35` |
| `J_allbutread_L24-35` | every prompt position except the read position | `rand_allbutread_L24-35` |
| `J_L24-35` (all positions) | existing | existing |

- **Model side:** flip and float32 Δm per arm.
- **Readers:** bridge d_r and answer d_ans at the read cell. Under `allbutread` the read cell is never written, so a reader shift there is computed by the model.
- **Reading rules:**
  - **(i)** If `allbutread` changes answers (flip rate ≥ 0.2 and well above its control), and readers' d_r at the read cell tracks those changes (AUC ≥ 0.8 for success within `allbutread`), then readers follow model-computed states, not only written coordinates.
  - **(ii)** `readonly` flips show how much of the effect is local to the read cell.
  - Both remain answer-cell reads, so the output-facing caveat stays. Report d_r together with d_ans and the model's own output.

**F2: earlier positions** (existing all-position `J_L24-35` and `rand_L24-35` edits; read at every prompt position p ≤ T−2, i.e. excluding the read token, and its preceding token when the prompt ends on " is").
- **Readers:** Oracle L40/L44, NLA L42, J_NP / J_CB / logit. Likelihood only (no sampling): LL(src) and LL(tgt) per position, giving d_r,p.
- **Model's own output per position:** the next-token log-prob change for the src/tgt bridge tokens (d_model_bridge,p) and the answer first tokens (d_model_answer,p).
- **Item summary:** the mean over early positions (pre-specified), plus the per-position profile for figures.
- **Reading rules** (within toward_tgt items, as in Stage D):
  - **"Bridge content beyond output-facing"** if:
    - the readers' early-position mean d_r predicts success with AUC ≥ 0.75, **and**
    - its partial Spearman with Δm, given the model's own early-position output shift and the edit size, stays above 0 (cluster-bootstrap CI excludes 0).
  - **"Not outcome-specific"** if the readers shift toward tgt at early positions about equally on working and failed swaps (AUC < 0.65). Readers would then detect written content regardless of use.
  - **"Absent"** if d_r,p ≈ 0 at early positions.

**Not claimed:** that any position is the mechanism's site. Positions are read cells. The edits remain whole-band interventions, apart from F1's split.

## 14. Stage G: whitened J-row swaps (Camila's request a) (pre-registered 2026-10-07 ~01:20 UTC, before any G run)

**Σ** for L24–35:
- 008's recipe (`scripts/008_trackb/sigma.py`): the same UltraChat corpus (about 50k chat tokens), Ledoit–Wolf shrinkage with λ reported per layer, the G-Σ gate.
- W = Σ^{-1/2} in float64; μ = the corpus mean.

**Whitened coordinate swap**, at every prompt position, L24–35, α 1:
- J rows are covectors, so their whitened images are Ṽ = W⁻¹V (as in 008 §1.4).
- Coordinates: c̃ = Ṽ⁺ W (h − μ).
- Clamp to swap(c̃_clean); write Δh = W⁻¹ Ṽ (swap(c̃_clean) − c̃).
- Control: a source-preserving random-target control in the same whitened construction (src row plus a seeded random unit target, same seed recipe).

**Measures** (same engine as Stage B/F):
- flips (Jan's matcher and budget) and float32 Δm;
- edit direction, from the whitened clean coordinates;
- realised ‖Δh‖ at the read position;
- readers at the read cell: Oracle L40/L44, NLA L42, token lenses at L40/L44. Bridge and answer LL only; no sampling.

**Gates:** G0 (α = 0 bitwise); G-Σ; a whitened round-trip check (W W⁻¹ ≈ I, max error ≤ 1e-6 in float64).

**Reading rules** (82 clean-correct items; compared with the raw J swap at L24–35: 38/82 flips, random 1/82):
- **"Whitening helps on this family":** flip rate at least 10 points above raw, with the random control still at most 0.05.
- **"Whitening hurts":** at least 10 points below raw.
- **"No material difference":** otherwise.
- Reader AUCs are reported as in Stage D, within toward_tgt edits defined in whitened coordinates.

EXPLORATORY.

## 15. Stage H: brew rerun of 008 arm A at every position (pre-registered 2026-10-07 ~01:55 UTC, before any H run)

**Question.** 008's lens-coordinate swap (arm A) on brew edited 3 emit cells at L40–48, and its answer-atom positive control moved S by only 0.010. On multihop, every-position J-row swaps work at L24–47 and at L40–48 (Stage A: 0.585 / 0.549). Is the brew null about **positions** or about the **family**?

**Material:** 008's 82 gated dev pairs (the same renders, partners, z/z′, a/a′ and spelling sets as 008; chat template plus the "Answer:" prefill).

**Edit:** Jan's clamp primitive (`SwapEdit`, raw coordinates, unit J_NP rows, every prompt position, prefill only, α 1). Arms:

| Arm | Swap | Band |
|---|---|---|
| `Az_L24-47` | z atom → z′ atom (single-token colour words, phrase-lens surface-form rule) | L24–47 |
| `Az_L40-48` | z atom → z′ atom | L40–48 |
| `Aa_L24-47` | answer-atom positive control (a → a′) | L24–47 |
| `Aa_L40-48` | answer-atom positive control (a → a′) | L40–48 |
| controls | source-preserving random target, for each arm | as above |

**Scores:**
- 008's S = Δm / G: m = LSE log P(a′) − LSE log P(a) over every spelling and token, float32 logits, with the member's clean gap G from 008;
- greedy flip to a′;
- G0 (α = 0 bitwise);
- the realised ‖Δh‖.

**Reading rules:**
- **Positions explain 008's null:** the answer-atom control at L40–48, every position, reaches median S ≥ 0.15 (008's power bar), where 008's 3-cell version reached 0.010.
- **Family explains it:**
  - if even every-position answer-atom swaps stay below 0.15, J-row atoms are a weak handle on brew;
  - the z-atom arm is then not read.
- If the control passes, the z-atom arm is read: median S, against its random control, and against 008's class-mean full-direction lever (0.21).

EXPLORATORY; the brew dev split has already been used in many places.

## 16. Stage R: does the NLA's early-position signal replicate on fresh items? (pre-registered 2026-10-07 ~02:40 UTC, before any R run)

**Why.** Stage F2 found one reader-specific hint: at early, directly written prompt positions, NLA L42's tgt shift separated working from failed swaps (AUC 0.72 [0.58, 0.84]; partial ρ 0.54 [0.20, 0.69] beyond the model's own early-position output). No other reader showed this (verdict "intermediate"). It was post hoc interest on one family.

**Fresh items:** Jan's ST **broadcast** bank (`causal_st_broadcast`; Stage A: J swap L24–47 success 0.315 on 146 clean-correct items).
- Caveat: the source concept is **explicit** in these prompts ("The capital of France is the city of"), unlike multihop's hidden bridge.
- So a replication supports the NLA property generalising, while a failure does not refute the multihop hint. This is a different family.

**Protocol** (identical to F2):
- J swap at L24–35, every position, with its source-preserving random-target control.
- Clean gate and primary cohort as in §12; edit direction at the read position.
- Read every prompt position before the read token. Readers: NLA L42, Oracle L40/L44, J_NP / J_CB / logit at L40/L44; likelihood only.
- The model's own next-token output per position.
- Item summary: the mean over early positions.

**Pre-registered test** (within toward_tgt items):
- **Replicates:** NLA L42 early-position AUC ≥ 0.65 **and** partial Spearman with Δm, given the model's own early output and the edit size, with a CI excluding 0.
- **Fails:** AUC < 0.65, or a partial CI including 0.
- The Oracle and the lenses are reported with the same statistics. "NLA-specific" requires them to fail the same test.

**Power caveat:** with about 30% success, the toward_tgt working-swap count may be small (about 40). Report n, and treat a null as weak.

EXPLORATORY.

## 17. Stage E0: validating our Oracle-reconstructor (AR) loader against the published per-layer profile (pre-registered 2026-10-07 ~02:55 UTC, before any E0 run)

**Status of the official loader.** `global_workspace.ola.ar_loader.load_lc_reconstructor` is not public. `agu18dec/local-workspace` holds only checkpoints, data and eval viewers (listed 2026-10-07), and the PI's request to Camila is pending. So we validate our own loader empirically.
- Its README says "truncates the backbone to the deepest needed layer". The LoRA covers all 64 blocks (checked), so truncation is not the issue.
- The README's "17-row head" conflicts with `heads.pt`, which has one shared head and 12 tags (20, 24, …, 60, 63). The README is partly stale.

**Published reference** (`ar_ptag_pooled/meta.json`, per layer):

| Metric | Range | Coverage |
|---|---|---|
| val_fve | 0.12–0.34 | L20–63 |
| val_ret1 | 0.88–0.98 | L20–63 |
| val_jfve | 0.39–0.61 | L20–60 only |
| val_jret1 | 0.56–0.96 | L20–60 only |

Eval protocol: up to 1,024 rows, micro-batch 24, spans of uniform 1–32 tokens.

**Run.**
- Our loader (011 conventions: tag + span, backbone final state at the span's last real token, parameter-free LayerNorm, shared head, raw-activation output, target = block-L output at the token before the span).
- All 12 tag layers; about 1,024 held-out assistant-text spans of uniform 1–32 tokens, from the same source as 011's validation spans.
- Whitened metric uses `whitening_iolens_pooled_L{L}` (the 9 missing layers are downloaded).

**Metrics per layer:**
- unscaled whitened FVE, pooled: 1 − Σ‖W(ŷ − y)‖² / Σ‖W(y − μ)‖²;
- FVE after one fitted global scale;
- per-sample cosine;
- retrieval@1 within micro-batches of 24.

**"jfve" candidates** (same spans and layers):
- (a) FVE of J_ℓ y vs J_ℓ ŷ in raw space;
- (b) the same with the whitener mean removed first (`jspace_mu_prefix`);
- (c) FVE of the J-lens logits W_U·norm(J_ℓ y);
- (d) FVE inside 007's dense J subspace (top-1,024).
- Each candidate also gets a matching retrieval@1 (the jret1 analogue).

**Decision rules:**
- **Loader validated up to a global scale:** globally scaled FVE within 0.04 of the published val_fve at ≥ 10/12 layers, **and** ret@1 (micro-batches of 24) within 0.05 of val_ret1 at ≥ 10/12 layers.
- **Fully validated:** additionally, unscaled FVE within 0.04 at ≥ 10/12 layers.
- **jfve identified:** one candidate matches val_jfve within 0.04 at ≥ 9/11 layers **and** val_jret1 within 0.07 at ≥ 9/11. Otherwise the definition stays open.
- If jfve is identified as a J-lens-transformed target, the AR (the Oracle's training ruler) carries an explicit J-lens objective. The 007 discussion must then say so.

**Consequence for Stage E:**
- If validated up to scale, Camila's AR-vector swaps run with centred or difference vectors (the global scale cancels), labelled "our loader, validated up to a global output scale".
- If not, Stage E keeps waiting for her loader.

EXPLORATORY.

### §17a. POST HOC amendment (2026-10-07 ~03:55 UTC; written after seeing E0's results)

**Seen.** Our loader's scaled FVE matches the published val_fve at 12/12 layers, but the best scale rises from 0.11 to 2.33 with depth. This suggests a scale-invariant AR loss, under which the published FVEs are scale-aligned. The unscaled jfve candidates all missed.

**Amendment.** Recompute the four jfve candidates with one fitted scale per layer, the same alignment as our main FVE. Apply §17's jfve rule (within 0.04 of val_jfve at ≥ 9/11 layers, and the jret1 analogue within 0.07 at ≥ 9/11) to these scaled values.

**Not changed.** §17's loader verdict ("not validated": ret@1 8/12) stands. This amendment only addresses the jfve definition, and any identification it produces is post hoc.

## 18. Stage I: delayed question (pre-registered 2026-10-07, before any I run; the PI chose "do 1")

**Why.** On Jan's prompts the hidden step fixes the answer one-to-one, and the answer is produced at the read token. So readers' tracking of a working swap is largely output-facing (Stage D), and there are no positions where the bridge is held but not yet being used (Stage F). Stage I separates *holding* the bridge from *using* it.

**Prompts** (`scripts/012_reader_concordance/stageI_prompts.py` → `outputs/012_reader_concordance/stageI/prompts.json`):
- Jan's 28 ST multihop items with a country clue ("Fact: The ⟨relation⟩ of ⟨the country …⟩ is"; New Zealand dropped as not single-token) are rewritten as:
  "Fact: Consider ⟨clue⟩. ⟨carrier⟩ ⟨question⟩".
- 2 neutral carriers × 3 questions (capital / main language / continent of "that country"), giving 168 prompts. 154 trials are discriminating, i.e. the src and tgt answer sets don't overlap.
- Answers come from a hand-written fact table of common facts for 23 countries.
- **Hold tokens:** H1 = the "." closing the clue sentence; H2 = the final "." of the carrier. In a causal LM the state at H1/H2 is identical whichever question follows.
- Japan is the swap target for 16 of the 28 clues; results are reported pooled and without Japan.

**Gates:**
1. **Clean gate per prompt** (src version): greedy contains y_a (Jan's matcher), ≥ 9/10 at T 0.8, and the clean output does not contain y_b.
2. **Tgt fact check:** "Fact: The ⟨relation⟩ of ⟨tgt⟩ is" gives y_b greedy.
3. G0 (α = 0 bitwise), the write mask, and float32 margins, as in Stage F.

Trials failing a gate are kept and excluded from the primary analysis.

**Arms** (J_NP rows from Stage A's bank; Jan's `SwapEdit`; band L24–35, so readers at L40/42/44 are downstream; prefill only):

| Arm | Positions edited | Role |
|---|---|---|
| `J_pre` | every position up to and including H2, none of the question | primary |
| `rand_pre` | same positions; source-preserving random-target control | control |
| `J_all` | every prompt position | reference |

**Model-side measures**, per trial:
- flip: greedy contains y_b, with Jan's budget and matcher;
- float32 Δm for y_b vs y_a (all spellings);
- edit direction at H2 (pinv coordinates, averaged over band layers);
- the model's own next-token output at H2: the shift for the src/tgt bridge tokens and for the first tokens of all three relations' answers.

**Readers**, at H2 (primary) and H1, on clean / J_pre / rand_pre hold states:
- Oracle L40/L44, NLA L42, J_NP / J_CB / logit at L40/L44.
- Bridge LL (src vs tgt names) → d_r.
- 2 sampled readouts per state for Oracle L44 and NLA L42, saved, with a seeded subset committed.

**Units.** One hold state per (clue, carrier): 56 states, clustered by clue. Its outcome is the fraction of its discriminating questions that flipped. **Hold success** = at least half flipped.

**Decision rules:**
- **(I-1) Lever:** J_pre flips ≥ 0.20 of the gated discriminating trials, and ≥ 15 points above rand_pre. Otherwise stop: "no lever with a delayed question".
- **(I-2) Non-output-facing check:** the model's own H2 next-token bridge shift predicts hold success with AUC < 0.65.
- **(I-3) Reader test** (within toward_tgt hold states):
  - A reader whose d_r at H2 predicts hold success with AUC ≥ 0.75, **with I-2 met**, reads the hidden step beyond output-facing content.
  - "NLA-specific" (or "Oracle-specific") if one passes and the other fails (AUC < 0.65).
- **(I-4) Bridge-level use** (descriptive): the share of clues whose J_pre swap flips every discriminating question.

**Power.** About 28 clues × 2 carriers. This is a pilot: report n, cluster intervals, and per-clue outcomes.

EXPLORATORY.

## 19. Stage J: relaxed gate, and reading the clue end when it is computed rather than written (pre-registered 2026-10-07, before any J run; PI: "go")

**J1: relaxed clean gate** (an amendment to §18's gate, written after seeing Stage I's results; the reason is stated).
- **Seen in I:** only 35/168 prompts passed. Many failed because Jan's short answer budget (answer length + 2 tokens) cut off wandering continuations ("located in the"), so no continent question passed.
- **J1 gate:**
  - every generation (gate samples, greedy, every arm's greedy) uses a fixed **8-token budget**;
  - greedy contains y_a (Jan's matcher);
  - **≥ 8/10** correct at T 0.8, top_p 0.95 (Jan's own threshold was 8/10);
  - the clean output does not contain y_b.
- Flips for J_pre / rand_pre / J_all are re-scored with the same 8-token budget.
- The hold-state reads from Stage I (H1/H2) are reused unchanged; the state does not depend on generation.
- §18's rules I-1 to I-3 are re-applied on the J1 cohort and reported **next to** the Stage I numbers, not instead of them.

**J2: clue-only edit** (the follow-up as corrected: in a causal LM, edits after the clue cannot reach H1).

| Arm | Positions edited | Read at |
|---|---|---|
| `J_clue` | clue tokens before H1, i.e. positions < H1 | H1 and H2, both computed by the model, not written |
| `rand_clue` | same mask; source-preserving random-target control | — |

- Positions ≥ H1 (the clue's final ".", the carrier, the question) are never written.
- Band L24–35. Same engine, gates (G0 α = 0, write mask) and float32 margins as Stage I. Flips use the 8-token budget, on the J1 cohort.
- **Readers** at H1 and H2 on the clean / J_clue / rand_clue hold states: Oracle L40/L44, NLA L42, J_NP / J_CB / logit at L40/L44. Bridge LL → d_r. 2 sampled readouts per state for Oracle L44 and NLA L42 at H1.
- **The model's own next-token output at H1 and H2** (bridge tokens; answer first tokens) for J_clue.

**Decision rules (J2):**
- **(J-1) Lever:** J_clue flips ≥ 0.20 of the J1-gated discriminating trials and ≥ 15 points above rand_clue. If not, readers are reported descriptively only.
- **(J-2) Output-facing check at H1:** the model's own H1 next-token bridge shift predicts hold success with AUC < 0.65.
- **(J-3) Reader test at H1:** a reader whose J_clue d_r at H1 predicts hold success with AUC ≥ 0.75, with J-1 and J-2 met, **reads the hidden step at a token that is neither written nor output-facing.** That would be the project's first such evidence.
  - "Reader-specific" if one family passes and the other (Oracle vs NLA) has AUC < 0.65.
  - The same test at H2 is secondary.

**Units:** hold states (clue × carrier), clustered by clue; hold success = at least half of the gated discriminating questions flipped. Japan reported pooled and excluded. A pilot; report n.

EXPLORATORY.

## 20. Corrections adopted after the Codex audit (2026-10-07)

- **F2 and R analyses:** one shared (item, pos) mask for readers and the model covariate, with an implementation fix. The corrected F2 NLA verdict is "not outcome-specific". See the ledger.
- **Reader-specificity rules in §16, §18 and §19 are replaced.** "X-specific" now requires a **paired** contrast on the same items: the AUC difference between readers with a paired cluster-bootstrap 95% CI excluding 0 and |ΔAUC| ≥ 0.10. A single reader passing a threshold while another fails is not specificity. Lack of a detected difference is reported as "unresolved", not equivalence.
- **Labels:** "answer effect under the specified J-derived intervention", not natural mediation.
- **E0:** "reused calibration contexts", not held-out validation. The scale is "per-layer, same-sample calibration".
