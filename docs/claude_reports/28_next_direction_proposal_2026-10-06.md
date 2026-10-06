# What we should do next, and why (proposal, 2026-10-06)

*Draft for the PI. Nothing here has run. Costs are GPU time; the code is not the bottleneck. Where a claim rests on
earlier work, the report is named.*

## 0. The recommendation in one paragraph

Move the main line off brew. Build a small, fresh family where the hidden intermediate is **never a plausible answer**,
with token-aligned counterfactual pairs. Then ask the project's real question there: **does a reader's signal for the
intermediate follow where the model demonstrably uses it, or only where it is present?** Brew cannot answer that,
because its intermediate is also the model's runner-up answer. In parallel, run one cheap repair of 007's weakest link:
the model-side §3.3 test, redone with the paper's protocol. Together, about 2–2.5 GPU-h of exploratory work. Its
outcomes decide whether the next confirmatory experiment happens on the new family, and how 007 can be written up.

## 1. Where we stand, and what limits us

**What the project has shown so far** (Qwen3.6-27B, brew, dev unless noted):

| Result | Source | What it says |
|---|---|---|
| The readers' z signal does not follow use. The Oracle's D_z is about 1.85× higher at `</think>` (patch share ≈ 0.01) than at `:` (≈ 0.48); the NLA's is about equal | 006 post hoc | Readers report z where the model does not use it |
| The model's own next-token output tracks z at all three cells (D_z 3.66–3.99); z is the runner-up answer in 27/48 tables | 006b | z is output-facing content on brew |
| From the J piece, the readers name z for the right render but as often as the answer, mostly framed as a result | report 27 | Their content reads like "candidate answers", not "a held step" |
| The Oracle's z tracking depends on the J subspace (confirmed on test); the NLA's does not; the model's answer gives J no privilege | 007 | A reader–model dissociation, on a weak model-side test |
| Content-level labels fail on brew (κ 0.37; fitted spans ≈ PCA) | 003, 004 | Only position-level labels are reliable |

**The binding limitation.** Every reader result on brew has two explanations that brew cannot separate:
- **H-held:** the reader reads a held intermediate.
- **H-output:** the reader reads output-facing content, i.e. what the model is about to say, which includes z as a
  candidate answer.

006b, report 27 and the `</think>` result all fit H-output, and nothing yet requires H-held. Running more brew variants
(more pieces, more layers, more readers) does not change this, because the confound is built into the task.

**The North Star (SPEC_v2 §2)** is to know which readers report content that the computation depends on. The proxy asks
for each reader's sensitivity to *used* content and its hit rate on *decodable-but-not-needed* content. On brew both
quantities are contaminated by H-output. A family where the intermediate is never an answer candidate removes that
route. Then whatever reader signal remains must come from something other than the model's next-token disposition.

## 2. Track A (main line): a counterfactual family with a non-answer intermediate

### 2.1 The family

**Two-hop factual prompts in a few templates.** In each, the answer is a different *type* from the bridge, so the bridge
is not a plausible next token at the answer position. Candidates:

| Template | cue → bridge → answer | Why the bridge is not an answer candidate |
|---|---|---|
| T1 | landmark → country → capital ("The capital of the country of the Acropolis is" → Greece → Athens) | a city is asked for; the country is the wrong type |
| T2 | landmark → country → currency or language | as above |
| T3 | holiday → month → month number ("Christmas Day is in month number" → December → 12) | digits are asked for; a month name is the wrong type |
| T4 | atomic number → element → symbol ("The symbol of element 26 is" → iron → Fe) | a symbol is asked for; the element name is the wrong type |

**Pairs.** Each pair is a member and a partner from the same template, differing only in the cue (Acropolis ↔
Colosseum), with cues chosen to tokenize to the same length. Every position then lines up, which whole-cell patching
needs. Cues must satisfy z′ ≠ z and a′ ≠ a, as in our partner rules.

**Render.** Brew's convention: chat template, thinking off, and an "Answer:" prefill. This gives the same emit cells as
brew (assistant newline, `</think>`, `:`), so results are comparable across families. Brew's `</think>` cell was exactly
the "present but unused" cell, and we want to see whether that repeats.

**Gate.** The project standard: greedy correct, and ≥ 9/10 correct at T = 0.8 for both members.

**Split.** The cues are new, so the test split is clean from the start: dev and test are split by cue entity, and test is
not touched until a design note with decision rules is committed. This avoids the held-out problem 007 left (brew and
multihop test already spent).

**Size.** Roughly 3 templates × ~25 cues, aiming for about 60 gated dev pairs and a similar number held out. WSB's
multihop bank is not reusable here: it has 53 small template families, mostly 1–4 items each, no token-aligned partners,
and its test split is spent.

### 2.2 What gets measured (reader-free first)

Every label below is reader-free (non-negotiable 1).

1. **The output-facing check.** The model's own log-probs for the bridge strings at every candidate cell, scored with
   006b's D statistic, plus the bridge's rank in the model's next-token distribution. **This is the check the family
   exists to pass:** the bridge must get almost no output mass.
2. **Position-level labels.** Whole-cell patch share S (partner → member) at every position from the cue onward, at L40,
   42, 44, 48, 52, 56 and 60, under the SEC regime. This is 001's protocol: full-sequence answer margin, every token of
   every spelling.
3. **Decodability.** A cross-validated linear probe for the bridge at every cell (folds by cue entity). It gives the
   "present" map to compare against the "used" map from step 2.

**What we need to see.** Cells where the bridge is decodable **and** used (S high), next to cells where it is decodable
but **not** used (S ≈ 0). Brew had one such contrast (`:` vs `</think>`). If the new family gives several, the use test
has power.

### 2.3 Reader measurements (only after the reader-free side looks right)

- **Readers:** the Oracle at L40/L44 (on-contract; L48–60 as secondary), the NLA at L42, and the token lenses (J_NP,
  J_CB, R_CB, logit lens) as cheap comparators.
- **Statistic:** 006's likelihood D statistic for the bridge at every cell, with:
  - the prompt-only floor;
  - a no-activation row;
  - the 200-derangement null;
  - a positive control on natural "Tell me about {country}" activations (pass rule as in 007).
- **Qualitative samples saved at every cell, `</think>` included.** Report 27 found that `</think>` was never saved in
  007.

### 2.4 The test, and what each outcome means

**The core contrast, per reader and layer.** Compare the reader's D_bridge at cells where the bridge is used with its
D_bridge at cells where the bridge is decodable but not used. Compare that against the same contrast for the probe
(presence) and for the model's own output (output-facing content).
- The prompt-only floor is the same for every cell after the cue, so prompt inference cancels in the contrast.
- With H-output closed by design, the contrast directly measures whether the reader tracks use.

| Outcome | What we learn | What we do next |
|---|---|---|
| The reader's D follows S (high where used, low where only present), beyond the probe's profile | **The first evidence that a reader tracks use**, and evidence for H-held on a family where H-output is closed | Pre-register the confirmatory version on the held-out cues; this becomes the project's main positive result |
| The reader's D follows the probe (equally high where present but unused) | **The reader reports presence, not use**, now without the output-facing excuse. Brew's `</think>` result generalises | This is the project's main negative result, and it is well posed: confirm on held-out cues, then write up |
| The reader's D is flat or near the prompt-only floor everywhere | The reader is not reading the activation here | Check the positive control. If it passes, the reader re-solves from the prompt; Cathryn's re-solving story wins on this family |
| Readers disagree (e.g. the Oracle follows S, the NLA follows the probe) | **Reader-specific behaviour**, like 007's J dependence | The concordance analysis has real material: which reader to trust, for what |

### 2.5 De-risk first: three go/no-go checks (reader-free, about 1–1.5 GPU-h)

Before any reader runs, on dev pairs only:

1. **The bridge is not an answer candidate.** At the answer-side cells, the median rank of the bridge in the model's
   next-token output is far from the top (threshold to be fixed in the design note, e.g. > 50), and the model's D_bridge
   is near 0.
   - **If it fails:** drop that template or switch the answer type. If it fails everywhere, this route is closed and that
     is itself a finding about how output-facing Qwen's late cells are.
2. **The position labels vary.** Some cells carry a real share of the answer gap (S ≥ about 0.3 at L40–44), and some
   cells have the bridge decodable but S ≈ 0.
   - **If it fails** (e.g. all the leverage sits in one cell, or nothing is decodable where S is 0): use the paired-task
     variant below.
3. **Optional: content-level labels.** Class-mean bridge-direction swaps (the paper's §3.3 construction) move the answer
   to the partner's in a meaningful share of pairs.
   - On Qwen this may be weak; Zeisler saw 6–11% flips with J-lens swaps. **This check is not required for the main
     test**, which uses position-level labels. If it works, it upgrades the labels to content level.

**Paired-task variant (the stronger design if check 2 fails).** This is the paper's §3.5.1 / Fig. 11 idea. Keep the cue
text identical, and change only whether the question needs the bridge:
- "The capital of the country of the Acropolis is" needs the bridge;
- "The number of letters in 'Acropolis' is" does not.

Reading the same tokens under the two tasks tests use at matched position and token type. That removes the residual
worry that readers simply read some cell types better than others. It costs a second gate and a second patch profile,
about +0.5 GPU-h.

### 2.6 Known risks, stated before we start

- **Cell type.** Readers may read delimiter tokens (e.g. `</think>`) differently from content tokens; the paper's oracle
  lens does exactly this (A.9.2). A cross-cell contrast can confound use with cell type. The mitigations are:
  - compare the same position across layers where S changes;
  - use the paired-task variant, which holds position and tokens fixed.
- **Redundancy.** S ≈ 0 can mean "unused" or "redundant elsewhere". Brew's zero-share prompt positions are mostly
  redundancy. "Decodable-but-not-needed" must therefore be stated per cell and never read as "the model ignores it". The
  paired-task variant helps here too.
- **Prompt inference.** Bridges are inferable from the prompt (WSB multihop prompt-only ≈ .90). The cell contrast cancels
  a constant floor, but a reader that re-solves *better* at some cells would still confound it. The prompt-only floor is
  computed per cell to check this.
- **Small n.** About 60 pairs is enough for a within-pair contrast with a cue-cluster bootstrap, but not for many
  subgroups. Keep the primary contrast to one per reader-layer.

## 3. Track B (in parallel): repair 007's model-side test

### 3.1 Why

007's headline is a reader–model dissociation: the Oracle privileges J, and the model does not. The model-side half rests
on Stage 2b. That stage swapped the class-mean z direction at **one cell (`:`) and one layer at a time**, and the full
direction itself moved the answer by only S 0.04–0.06, against 0.28–0.49 for the whole-cell patch. The paper's §3.3
swaps are applied at **every token position** and clamped. "J part < non-J part" on such a weak lever is weak evidence
that J has no privilege. Track B also tests the paper's central causal claim (internal reasoning routes through the
J-space) on Qwen with the paper's own protocol, which matters regardless of readers.

### 3.2 What runs (about 35–40 GPU-min, exploratory)

- **Directions.** Class-mean z directions per layer (leave-table-out, reader-free), split by nonnegative pursuit into the
  J part (k = 25, J_NP and J_CB) and the non-J part, as in 2b.
- **Swap.** Clamped over the three emit cells (assistant newline, `</think>`, `:`) and the band L40–48. All writes are
  float32 under SEC, with fidelity checks (non-negotiable 2).
- **Conditions:**
  - full direction;
  - J part (two lenses);
  - non-J part;
  - 4 random directions;
  - the paper's complementary clamp: the non-J swap with the J coordinates clamped to clean values, and the reverse.
- **Magnitude matching, both ways:**
  - raw norm, as 2b did;
  - **whitened norm**, with Σ from one wikitext forward per layer, ridge-regularised. This answers Camila's question.
  - The Oracle Lens itself was trained on a whitened FVE (WSB appendix). Its whitener statistics are not public, so we
    build our own.
- **Material and metrics.** The 33 gated pairs of 004/007. Metrics: S (answer margin), the top-1 flip rate (the paper's
  metric), and damage readouts.
- **Power gate.** The full-direction swap must reach a median S ≥ 0.15 before J vs non-J is read. Otherwise the lever is
  still too weak, and we say so.

### 3.3 What each outcome means

| Outcome (under both metrics) | What we learn |
|---|---|
| J part ≥ non-J part, and S(J) ≥ 0.5 · S(full) | The paper's §3.3 privilege replicates on Qwen brew under its own protocol. **007's "no model-side privilege" was an artefact of the one-cell design, and the dissociation story is withdrawn** |
| Non-J part ≥ J part | "No model-side J privilege at brew's emit cells" becomes a supported claim, and the dissociation stands |
| Depends on the metric (raw vs whitened) | J privilege is metric-dependent. The write-up must state the metric, and Camila's point is confirmed |
| The full direction stays weak (power gate fails) | Class-mean z directions are a poor handle on Qwen brew; the model-side question needs another construction |

## 4. What I would not do now, and why

- **The Oracle's J-test controls it can read** (top generic PCs, whitened J pieces; about 15 GPU-min). Worth running
  **only if we write 007 up**. Whether the Oracle depends on J itself or on any natural-looking piece is a reader-mechanism
  detail; it does not move the North Star.
- **AR-based hallucination stripping for the NLA.**
  - The literature expects stripping false specifics to *lower* FVE (Dingeto; Hong; the NLA paper's coherence rewrite),
    so FVE retention is not a truth measure.
  - Our AR is from step 600 while our adapter is iter_400, and the rewrite needs a paid model.
  - On brew we can count invented content against exact truth directly; report 27 already found "yellow" in 8–19/32
    readouts from both readers.
- **The Oracle at L44 vs the NLA at L42, more brew pieces, more layers.** All inherit brew's confound.

## 5. Order of work

1. **Now, in parallel** (exploratory, logged as such):
   - Track B (about 35–40 GPU-min);
   - Track A's bank: write cues and templates, check token alignment on CPU, then gate on GPU.
2. **Track A de-risk** (about 1–1.5 GPU-h): the three go/no-go checks on dev pairs. A short report with raw numbers and
   samples.
3. **Decision point with you:**
   - if the checks pass, a committed design note for the confirmatory reader run on held-out cues, with decision rules
     and gates calibrated per reader (report 25's lesson);
   - if they fail, the paired-task variant or a different template.
4. **Confirmatory reader run** on the new family:
   - the Oracle: about 15–30 GPU-min;
   - the NLA: about 1–2 GPU-h (separate model; capture subject activations first);
   - check in before it if the total exceeds about 2 GPU-h.

Total before the decision point: about 2–2.5 GPU-h, all within the exploratory limit.

## 6. What we will know at the end, and what we won't

**We will know:**
- Whether any of our readers tracks *use* rather than *presence* on a task where output-facing content cannot explain
  the signal. This is the first clean answer to the project's question for one family on Qwen3.6-27B.
- Whether brew's `</think>` pattern (strong reader signal at an unused cell) generalises, or was the runner-up confound.
- Whether the paper's §3.3 J privilege holds on Qwen under the paper's protocol, and so whether 007's dissociation
  stands.
- Whether J privilege, if any, depends on the norm metric.
- Whether reader behaviour is reader-specific (Oracle vs NLA vs token lenses) on a clean family. That is the material
  for the concordance analysis.

**We will not know:**
- Anything beyond Qwen3.6-27B and these templates.
- Whether a reader is faithful in general.
- Whether a "not needed" cell is unused rather than redundant. Only the paired-task variant addresses that.
- Content-level use, unless check 3 succeeds.

## 7. Decisions needed from you

1. **Is the near-term goal a 007 write-up, or progress toward use-sensitive reader evaluation?** If it is the write-up,
   run Track B and the readable controls first. If it is progress, run as above.
2. **Templates.** Any you would add or veto from T1–T4? Domain variety vs depth per template is a trade-off: about 60
   pairs from 3 templates gives more pairs per cluster.
3. **Should the paired-task variant be part of the first de-risk** (+0.5 GPU-h), or only a fallback?
