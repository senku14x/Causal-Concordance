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
