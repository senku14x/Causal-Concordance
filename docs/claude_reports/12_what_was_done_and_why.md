# What was done, and why it was done at all (2026-10-04)

A plain account of today's work: what each piece is for, and what it bought us. The smoke results are at the end.

## The point of the whole project, in one breath

Readers like the NLA, the Oracle Lens and the J-lens claim to tell us what Qwen is "thinking". Before trusting any of them we need to know two things:
- **Accuracy:** when a reader says "the model holds X here", is X actually there?
- **Kind of reading:** is the reader reading the activation, or re-solving the task from cues in the prompt? And does it see things the model itself cannot use or report?

Everything today serves those two questions, or removes an error that would have corrupted the answers.

## 1. Three cheap checks before any new GPU work (C1–C3)

**Why first.** These checks were cheap, needed no GPU, and each could overturn a claim the spec leaned on.

| Check | Why it was worth doing | What it changed |
|---|---|---|
| **C1:** re-score Cathryn's chain readouts with WorkspaceBench's stricter rule | CW-7 (the Oracle naming chain intermediates) was our only positive "the reader really reads" case, and her rule credited any number appearing anywhere in a readout | It survives at a third of its size (11/60 vs null 0.04), and **J-lens does exactly as well**. So it is not evidence that the Oracle reads anything special. The NLA can't be tested on chain at all. |
| **C2:** where probability goes after the ":" patch | 001 had flagged these patches as "damage"; if true, the main influence label at ":" was in doubt | Most of the shifted mass goes to the two intermediate colours, so the patch makes a mixture of both trajectories rather than breaking the model. "Influence" kept, with a rewritten damage rule for future data. |
| **C3:** how big Cathryn's "removal" edits were | Her "readers survive removal of the content" result compared her content removal against a random control with no size matching | Multihop's control removes 2.4× less; Brew's is roughly matched. Both remove only 0.1–1% of the activation, so "survives J-removal" can never separate reading from re-solving. That needs a validated eraser. |

## 2. Corrections to the spec (errata 01, accepted)

**Why.** The spec had wording that overstated results ("not needed", "W ≈ 0", wrong denominators) and a few claims the checks changed. Errata keep the spec honest without rewriting it, so earlier citations still hold.

## 3. Engineering: the NLA loader and the J-space tool

- **`nla-rl` loader.**
  - **Why:** you chose this NLA. The repo could only load an older, different one, which modified the subject model in place (a known hazard).
  - **What it does:** loads the NLA as a separate model, exactly as WorkspaceBench does, with every file hash-pinned.
  - It cannot be confused with Qwen: the code refuses to use it as the subject.
- **J-space pursuit.**
  - **Why:** design 005 needs to split each concept into the part inside the paper's "J-space" and the rest. Anthropic's released code doesn't include that step, so we built it.
  - The tests prove it recovers known sparse signals exactly.

## 4. Design 005, and why it is the next experiment

**The idea.** Plant a known concept into Qwen, split into its J-part and its non-J part, then ask:
- Does Qwen itself report each part?
- Does each reader detect each part?

**Why this experiment.** It is the only one where we know the ground truth exactly. That lets us test the paper's central claim on Qwen (only J-space content is reportable) and the NLA's selling point (it sees content outside J-space that the model can't report), with nothing left to judgement.

**Why the design changed twice.**
- Your review found that my v1 read the readers at a position where the planted vector was still sitting in the residual stream. That would have made the NLA's "advantage" close to guaranteed.
- v2 reads at the position where the model makes its report, which content can only reach by attention.
- It also:
  - tightens what "the model doesn't report it" means;
  - adds a null for the clamp;
  - screens out words readers say anyway;
  - quadruples the test concepts.

**Priors.** I pre-registered my own predictions before any GPU use, so the results can't quietly reshape what I "expected". The headline guesses:
- P11a (the NLA sees what the model doesn't report): **0.20**
- P11b (the NLA sees what doesn't change the model's behaviour): **0.15**
- P17 (the Oracle sees the non-J part less than the NLA): **0.45**

## 5. The 005 code and the smoke run

**What the smoke is for.** Not results. It checks that every part works on the real model before the expensive run:
- whether Qwen can report planted thoughts at all (the G1 gate; if not, the main contrast is floor-limited and the design must change);
- at which strengths the non-J part starts to matter (this sets the readers' strengths);
- that every write and clamp is exact;
- that runs are deterministic;
- that the cheap and the exact numerics agree;
- how long the full run will take;
- what the raw readouts actually look like.

**Smoke results (dev concepts; full note in `13_005_smoke_note.md`).**

| What happened | What it means |
|---|---|
| **The report gate failed at both bands.** Qwen almost never *says* the planted concept (0–8% of samples), because a strong default answer ("apple") holds the top slot | By the pre-registered stop rule, the design must be amended before any full run |
| **By the paper's own metric the planting works:** the pure J-lens vector pushes the concept to median rank 2 at the report position (+8.5 nats; random +0.4) | The amendment will likely make rank/log-prob the primary "report" measure |
| The non-J part already acts at the smallest dose | The dose grid must go lower before P11a/b ("readers see what the model doesn't report") can be tested |
| earth's "non-J" part still carried earth's own J-lens direction | A per-concept leak screen is needed; it showed 2 leaky concepts of 55 at the wide band and 0 at the main band |
| The concept-specific clamp did no better than a random clamp | The clamp as designed deletes part of the injection; step 7b tests a corrected "re-entry" clamp |
| Writes exact, runs deterministic, the NLA/Oracle pipelines work | The machinery is sound |

## 6. The overnight pilot (dev concepts only) and step 7b

**Why run it.** The smoke showed the design can't run as written. Rather than leave the GPU idle, or spend the held-out test concepts on a design we know must change, the pilot answers the amendment's open choices on the 10 dev concepts. Each question has a pre-registered rule (design §16.1):
- the report measure;
- band and prefill;
- how low the dose grid must go;
- whether the readers see anything, and where;
- whether the clamp can work;
- which concepts have leaky splits;
- what the full run costs.

**Step 7b** (from the review you forwarded; design §16.3):
- a *re-entry* clamp that blocks only what the model itself writes into the concept's J coordinates, plus a band-limited version, because near the output a clamp blocks the answer mechanically;
- a layer-by-layer readout at the report position (L32–62);
- the R-lens as an extra reader.

**What you get in the morning.** The pilot note answers Q1–Q11 against their rules, and an amendment draft proposes the choices those answers imply. The decision is yours; then the full run on the 45 test concepts goes blind and in one piece.
