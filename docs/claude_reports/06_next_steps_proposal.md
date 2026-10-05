# What I'd do next (proposal, 2026-10-04)

Nothing here runs without your go.

## The single highest-value next step

**005, stage 1: planted-concept sensitivity curves for every reader and the model. Full concepts, no J split yet.**

**Why this one first**

- **It is the positive control the original spec put at step 2, and nobody has run it.**
  - Every reader "miss" we have is uninterpretable without it: the verbalisers rarely claim z in role (002), and J's top-10 shows no colour at ":" L40–44 (001).
  - A miss can mean the content is not there in readable form, or that the reader can't see that kind of content at that dose. Only a dose-response curve against known ground truth separates the two.
- **It measures both sides at once.** For each dose, it records whether each reader detects the concept, and whether the model's own output and report change.
  - That shows at which doses the content *matters* to the model, and whether each reader detects it there.
- **It is informative under either outcome.**
  - *Readers detect planted concepts at behaviour-relevant doses:* natural-prompt misses become meaningful.
  - *Readers don't:* v2's stop rule triggers, and we rethink the reader set before spending more on natural prompts.
- **It builds the harness stage 2 needs.** Stage 2 is the J / non-J split (the actual "outside the workspace" test), and it runs only if stage 1 shows detection.
- **It avoids the current bottleneck of the alternatives:**
  - *A7 with a reader-free eraser is under-powered right now:*
    - Multihop has 75 distinct bridges in 100 items, so no class-label eraser (LEACE) can be fitted per bridge.
    - On Brew the readers barely claim z.
    - Chain has 28 usable items.
  - *Planted concepts* give exact ground truth without any eraser.

## Order of work

| Step | What | GPU | Needs |
|---|---|---|---|
| 0 | Decide the NLA iteration (400 vs 600). Stage 1 doesn't need the J-space decisions; those wait for stage 2. | – | you |
| 1 | Engineering: `nla-rl` loader in `readers.py` (separate `av_base`, Karvonen hook at block 1, no in-place merge into the subject); registry pins + sha256 verify; AR wrapper for the `nla-rl` reconstructor (check the `norm: none` convention); CPU tests | none | — |
| 2 | Design spec `005_planted_concepts` stage 1 (below), committed for your review | none | your review + Cathryn's sign-off (joint) |
| 3 | Smoke: 2 concepts × 3 doses × every reader; fidelity gates; determinism; raw readouts inspected | ~15 min | your go |
| 4 | Stage-1 run | ~1.5–2 h (estimate; could be 2× off) | your go |
| 5 | Stage 2 (J / non-J split), only if stage 1 shows detection | similar | your go; J-space construction decided |

## Stage-1 design sketch (the design spec would freeze all of this)

**Concepts and vectors**
- ≥ 24 single-token concepts, split dev/test.
- Brew colours included, so the result links to our family.
- Concept vector = mean residual for "Tell me about {c}" minus the mean over ≥ 100 other concepts (reader-free).

**Doses and injection**
- Dose grid in units of the natural residual norm at the injection cell (e.g. 0.25 / 0.5 / 1 / 2 / 4×).
- Matched-norm random-direction control at every dose, plus a zero dose.
- Float32 writes (SEC replay for L ≥ 40), with ρ/κ fidelity measured on every write.
- Injection layer chosen so every reader is on-contract when reading downstream: e.g. inject at L40, read at L40 (direct) and at L42/L44 (model-carried).

**Readers**
- J_NP and logit lens: the concept's rank among candidates.
- Oracle at L40/L44: k = 4.
- NLA `nla-rl` at L42: 4 samples.
- A reader-free linear probe for each concept, trained on natural activations, as the detection baseline.

**Model side**
- Open-text report (introspection protocol; concept named or not).
- Next-token KL on a neutral continuation and the concept-token log-prob.
- The Brew task margin when injected at a Brew cell.

**Scoring and decision rule**
- Scoring: concept naming by exact spelling-variant match (no LLM judge needed for the headline), with a judge only for synonyms, flagged separately.
- Decision rule (proposed, set before the run):
  - Proceed to stage 2 if, for at least one text reader, detection at doses that move the model's output exceeds the random-direction control by a pre-set margin.
  - Otherwise stop and report the reader set as unable to see planted content.

**Budget (rough)**
- GPU: ~1.5–2 h.
- Judge: ~$5.
- Downloads: ~93 GB (`av_base`, adapters, AR).

## Cheaper things that can run in parallel

- **Human-coded judge validation set:** ~50 readouts per family, coded blind by you and Cathryn. It's a v2 gate and needs no GPU, so start it now.
- **WSB reproduction on Brew under WSB's exact rule.** This is the plumbing check that never finished.
  - Low information (we already know any-cell ≠ per-cell), but cheap: lens forwards plus ~10k judge calls ≈ $20.
  - Worth doing once the judge budget is agreed.
- **Cathryn's no-GPU items:** A7.5 (prompt recovery predicts re-solving) and A7.7 (J-any-layer cross-check).
- **Spec housekeeping:** apply the corrections you accept from the review, plus the repo-visibility fix.

## What I would not do yet

- **DAS:** it only pays off if a content-level label would change a decision, and nothing currently depends on one.
- **The agreement atlas and ensembles:** they need the NLA settled and the judge validated first.
