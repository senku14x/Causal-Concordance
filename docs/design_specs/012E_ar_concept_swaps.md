# 012 Stage E: swaps along the Oracle reconstructor's concept vectors (011 amendment 4)

**Status:** APPROVED by the PI on 2026-10-10, with four fixes, applied below. Committed before anything ran: no phrase,
vector, prompt or edit exists yet. EXPLORATORY.
- **Source:** the PI's brief v3 (2026-10-10). Sections marked *deviation* differ from the brief, with the reason. The PI
  accepted every deviation.
- **Committed with:**
  - the `rebuild_env.sh` change below;
  - pointer sections in `012_reader_concordance.md` (§23) and `011_olens_swaps.md` (amendment 4).
- **Thresholds:** those in §6 and §9 are fixed. They may change only before the data they govern has been seen, by a
  dated amendment.
- **The PI's four fixes (2026-10-10), applied before commit:**
  1. **Engine (§2).** The draft said multihop would run in `engine.Engine(fp32_from=0)`. That was wrong: every 012
     model-side stage (B, F, G, I, J, L, R) ran on Jan's `ModelRunner` in bf16 with `BandEditor`, with float32 only for
     the final logits (`stage*_capture.py`; no 012 script calls `Engine`). I had copied 012 §4's plan rather than what
     ran.
  2. **Power for the size reference (§7, §9).**
  3. **Answer-concept phrases for every y_a / y_b (§3, §7).** Checked: 98 distinct answers in the 82-item cohort, 21 of
     them among the 89 concepts, and 12 items with both answers covered.
  4. **The multihop primary's score and the toward_tgt reporting (§9).**

**What this replaces.** 012 §4 Stage E was blocked on the AR loader. The 2026-10-09 post hoc acceptance (L36–60,
directions only; `CLAUDE.md` 012 E0, commit `d4b9860`) unblocks it. Brew Stage 1 is 011's harness with new vectors.

## 0. Prerequisites

| Item | Status (checked 2026-10-10) |
|---|---|
| Loader acceptance, the 10-09 patch, reports 39–40 | In the repo at `128bd12` |
| Jan Chen's phrase-lens at `ea85583` | Cloned to `/workspace/sources/ext/phrase-lens-src` (read-only). `rebuild_env.sh` now clones it (uncommitted; commits with this file) |
| Jan's random target is a unit direction | Verified: `interventions.py:136-145`, `unit(torch.randn(width))` per layer, sha1-seeded. This closes report 40's "not checkable" item |
| OpenRouter key | **Missing.** `/workspace/.env` doesn't exist on this instance (`CLAUDE.md` restore step 6). Step 2 is blocked until the PI provides it. It should also be rotated |
| AR whitener (`agu18dec/local-workspace`) | Not cached. Downloaded at first use, as in 011 |
| 007 dense J bases | Exist at L40, L42, L44 and L56 only (`outputs/007_jsubspace/bases/`). New layers are built with 007's own `geometry.stage_bases` (about 18 CPU-seconds per layer; `geometry_bases.json`) |

## 1. Question

Do the Oracle Lens's own concept directions (its reconstructor, the AR) move Qwen's answer? Above all: do they move it
when they have **no component in 007's dense J subspace at the edit layer**?

A yes on multihop gives a lever not built from J rows. That is the test S1 asks for (report 40).

**What a yes still would not show:**
- that the effect avoids J downstream, since the edit can re-enter J at later layers;
- that any reader reads a hidden step.

**Why "starts with" vs "mentions" matters.** The AR's target is the activation at the token *preceding* its input span
(011 amendment 2). So a phrase that starts with c gives, by construction, the state just before the model says c: an
output-facing direction. A phrase that mentions c later gives the state before a text about c.

## 2. Material and regimes

### Brew (Stage 1)

011's harness, unchanged apart from the vectors:
- the 82 gated Track B dev pairs;
- the three emit cells (assistant newline, `</think>`, `:`), at L40, L44 and L48 (AR tags);
- Track B's clamp primitive, SEC regime and gates (G0, G1, G-clamp, G-RA, G-repro);
- score S;
- the class-mean reference.

Brew is output-facing (006b, 009). A brew success shows that the Oracle's directions move the answer, not that they
reach a hidden step.

### Multihop (Stage 2)

- **Items:** 012's 82-item cohort (clean-correct, clean output lacks y_b).
- **Concepts:** 59 distinct bridges (src) and 43 distinct targets (tgt), 89 in all; 71 distinct (src, tgt) pairs.
- **Band:** L36–47, every prompt position, prefill only. This is inside the accepted loader range.
- **Engine (PI fix 1).** Every multihop arm runs in the regime of every other 012 model-side stage:
  - Jan's `ModelRunner` in bf16 with `BandEditor` (phrase-lens `ea85583`, read-only, via `PYTHONPATH`);
  - the band edit applied once per band layer to the prefill's prompt positions, computed in float32 and written back in
    bf16;
  - float32 only for the final logits (final-norm output recomputed in float32 × float32 W_U, as `stageB_capture.py`).

  Edit objects:
  - the reconstructor, class-mean and random arms are new `BandEditor` edit objects, built on Stage G's whitened clamp
    (`stageG_capture.py`) as the precedent;
  - each logs its coordinate error in float32 and after the bf16 cast (Stage B's fidelity record);
  - `engine.Engine` is not used.

  Noise floor and gates:
  - **noise floor:** the identity clamp (each arm's clamp written back to its own clean coordinates) gives |Δm| per item
    in this regime, as Stage B's `floor_identity_clamp`;
  - **G0:** the α = 0 edit reproduces clean bitwise.
- **Reference levers:**
  - **J-row swap.** Our scripts have never run L36–47 (Stage B ran L24-35 and L24-47), so G-parity against Jan's Stage A
    at L36–47 (46/82, random 3/82, `stageA_summary.json` `runs.j_swap_L36-47`) runs first, at ≥ 90% per-item flip
    agreement;
  - **class-mean swap** (§4).
- **Layers without an AR tag (*deviation*).** The AR has tags only at L36, L40 and L44 in this band. Each layer uses the
  nearest tag at or below it: L36 for L36–39, L40 for L40–43, L44 for L44–47.
  - Why: 012 §4 Stage E planned AR layers only. But the J-row lever's power (46/82) is known for the full band only.
  - A pre-check in §6 measures whether this approximation is reasonable.
- **Reporting:** by bridge type (named place vs other, with countries as a sub-group; `stageA_splits.py` `GEO_PLACES`)
  and by Japan target. All of these groupings are POST HOC. With 7 non-country places, "place" can't be separated from
  "country".

## 3. Phrase generation (OpenRouter)

### Settings (fixed before any call)

- **Generator:** `google/gemini-3.8-flash`, the model 002 and 013 pinned.
- **Parameters:** temperature 1.0, top_p 1.0, reasoning effort minimal, max_tokens 2000, JSON response.
- **Seed:** `seed = 20261010 + concept_index`. OpenRouter passes the seed on only where the provider supports it, so
  exact reproduction is not guaranteed; the raw responses are the record.
- **Requests:** one per (concept, set). Up to 3 retries on a parse failure, each logged.
- **Logging:** every request and response is saved raw under `outputs/012_reader_concordance/stageE/gen/`.

### The prompt (exact; `{…}` filled per concept)

> Write 40 short English phrases (3–12 words each) of the kind that appear in an assistant's reply. Every phrase must
> {begin with the word "{c}" | contain the word "{c}", but not as its first word}. Vary the topic and the sentence form.
> Do not use any of these words: {forbidden}. No numbering, quotes or lists. Return JSON: {"phrases": [...]}.

### Sets per concept

- **"starts":** the phrase begins with c.
- **"mentions":** the phrase contains c, but not first.
- **"template"** (no LLM): `"{c}"`, `"{c} is"`, `"the {c}"`, `"about {c}"`, `"{c} was"`, `"and {c}"`, `"like {c}"`,
  `"{c} has"`, with a leading space as in 011.
- **"word":** 011's single-word input `" {c}"`, rebuilt for multihop.

### Forbidden words (mechanical, built from the bank before any call)

**Multihop.** For concept c, the union of:
- the content words (≥ 4 letters, not a stopword, other than c) of every evaluation prompt whose src is c;
- y_a and its alternatives for items whose src is c;
- **y_b and its alternatives for items whose tgt is c** (*deviation*). Without this, "Japan, whose capital is Tokyo"
  would put the swap's answer into the vector, and the edit would become an answer handle.

**Brew.** The other 11 brew colour words, plus the content words of the brew prompt template.

**Answer concepts (PI fix 3).** Phrases are also generated for every y_a and y_b in the multihop cohort. That is 98
distinct answers; the 21 already among the 89 concepts reuse their vectors. Every item then gets the answer-concept
control. Rules:
- **Forbidden words:** for an answer concept, the src and tgt words of every item it answers, plus the content words of
  those items' prompts. This keeps the control an answer direction, not a bridge direction.
- **Sets:** all four (starts, mentions, template, word), checked and kept as for concepts.
- **Weak inputs:** answers that are single letters, chemical symbols or numerals ("A", "C", "Ag", "1") are flagged. A
  phrase "starting with A" is mostly a sentence starting with the article, so those vectors are expected to be weak. The
  answer control is reported by answer type (word / letter-symbol / numeral).

### Code checks (before anything reaches the AR)

- c is present, in the required position (word-boundary, case-insensitive);
- 3–12 words;
- not a duplicate;
- no 5-gram shared with any evaluation prompt;
- no forbidden word.

The first 32 passing phrases are kept, in generator order. A concept with fewer than 16 is flagged and dropped from the
phrase arms. Every drop is reported.

**Hand check:** 10 random phrases per set for 5 concepts per family (seed 11), read and coded (fluent; on-concept;
leaks a cue or answer).

## 4. Class-mean material for multihop

**Generation.** For each of the 89 concepts, the generator writes 24 two-hop prompts in the bank's format ("Fact: The
<relation> of <clue> is"), with the concept as the bridge. JSON gives `prompt`, `clue` and `answer`. Constraints:
- the concept word does not appear in the prompt;
- the clues differ from the bank's (the forbidden list applies);
- **the relation varies** (*deviation*): at most 4 prompts share a relation, and at most 1/3 share an answer. Otherwise
  the class mean at the final token would mostly encode the answer (for France with relation "capital": Paris), and the
  class-mean swap would be an answer swap.

**Gating:** as the bank (012 §4 Stage B):
- greedy correct;
- ≥ 9/10 correct at T = 0.8 (top_p 0.95, 16 tokens);
- the concept word absent from the prompt.

The first 16 passing prompts are kept. A concept with fewer than 10 is dropped from the class-mean arm. Items whose src
or tgt is dropped have no class-mean arm (§7 edit-size fallback).

**Direction.** d(c, L) is the mean, over c's kept prompts, of the block-L output at the prompt's final token, centred by
subtracting the mean over all kept concepts' prompts. Two further rules:
- **leave one out:** an evaluation item's own prompt is never among them (they are generated, so this holds by
  construction; checked by the 5-gram rule);
- **every-position edit (PI decision, 2026-10-10):** the primary direction is fit at the final token and applied at
  every prompt position, like the J rows. F1 showed the J-row effect comes from earlier positions.
- **sensitivity, reported alongside:** the same direction fit at the clue's last token. The position is found by
  matching the generator's `clue` string in the tokenized prompt; prompts where the match fails are dropped from this
  variant, with a count.

**J split (*deviation from 012 §4*):** the class-mean reference is split with **007's dense J subspace**, the same object
as the AR's J removal, not §4's sparse k = 25 pursuit part. Mixing the two objects is the 10-06 logged mistake. §4's
sparse Fig. 16 split is not run here.

## 5. Vectors

**AR forward.** 011 amendment 2's convention: `[Layer NN]` tag plus the phrase, read `ln_pre` at the phrase's last
token, block-L target, raw head output.

**Per concept and layer:**
1. mean of the phrase outputs;
2. centred across the family's concepts (12 colours; 89 multihop concepts);
3. unit-normalised.

Only directions are used.

**Differences:** u = unit(AR(z′) − AR(z)), from the swapped-in and swapped-out concepts (multihop: tgt and src).

**Variants:**

| Variant | Definition |
|---|---|
| full | u |
| J-removed | unit((I − Q_L Q_Lᵀ) u), with Q_L 007's dense J basis at layer L (top 1,024 of `J_raw_L{L}`) |
| random-removed (*deviation*) | unit((I − P_r P_rᵀ) u), P_r a random 1,024-dim orthonormal basis (seeded per item and layer). The control for "removing J" vs "removing any 1,024 dimensions" (007's rest-of-J vs rest-of-random logic) |
| single word | the same steps from 011's `" {c}"` input |
| starts / mentions / template | per set |

The class-mean references get the same full and J-removed treatment.

**New J bases.** Built with 007's `stage_bases` at L36–39, L41, L43, L45–48. A sanity gate first: the rebuilt L40, L42
and L44 bases must reproduce the committed ones (mean cos² ≥ 0.9999, as `geometry_overlaps.json` measures).

## 6. Pre-checks and the vector-quality gate (no edits to the model)

**(a) J share.** For each difference vector, the share of its energy inside the dense J basis at the edit layer. Reported
for AR and class-mean differences, per family and layer, against the 0.20 that a random 1,024-dim subspace holds.
- Prior from 007: dense J holds 0.28 of the brew member–partner difference at `:` L40 and L44 (0.23–0.31 across the 6
  cells; `geometry_brew.json` `cells.*.J_raw.delta_share.median`). That is only about 1.4× the random share.
- **Proposed threshold:** the J-attribution reading in §9 is used only where the median J share of the edited AR
  differences is ≥ 0.30.

**(b) Nearest-tag check (multihop).** The cosine between the centred concept differences at adjacent tags (L36 vs L40,
L40 vs L44).
- **Proposed:** median ≥ 0.7, or the nearest-tag rule is replaced by AR-layer-only edits (L36/40/44), with the
  references also run on those three layers, before any edit.

**(c) Gate** (thresholds from the brief, made explicit):

| Check | Rule |
|---|---|
| Stability | Median split-half cosine (phrases split 16/16, seeded) across concepts ≥ 0.8 |
| Specificity, 1 | cos(AR difference, class-mean difference) is higher for matched than for mismatched pairs: paired over pairs, table- or relation-cluster bootstrap 95% CI > 0 |
| Specificity, 2 | The phrase vectors' median matched cosine exceeds the single-word vectors' by ≥ 0.05. Brew's single-word baseline (011's unverified "0.15–0.24") is recomputed here |
| Decodability, brew | Nearest-concept accuracy on 007's 49 held-out brew test tables (`:` captures, label z; centred cosine) ≥ 1/6 (2× chance) and above the single-word vectors |
| Decodability, multihop | Nearest-concept accuracy over the 89 concepts on the gated class-mean prompts' final-token activations ≥ 2/89 and above the single-word vectors |
| Stop rule | If the phrase vectors don't beat the single-word vectors on both specificity and decodability for a family, stop and report. Nothing is edited for that family |

## 7. Edits and arms

**Common to both families:**
- every arm is a per-layer clamp along its direction, coordinate = clean + k (011 amendment 3);
- nothing is added per layer, so nothing compounds;
- edit sizes are logged at every edited (position, layer), not only the read cell.

**Arms per family:**

| Arm | Brew | Multihop |
|---|---|---|
| AR full / J-removed / random-removed | yes | yes |
| AR starts, mentions, template, single word (full and J-removed) | yes | yes |
| Answer-concept AR difference (power control) | AR(a′) − AR(a) | AR(y_b) − AR(y_a), for every item (answer phrases, §3) |
| Mismatched-concept AR difference (*deviation*: the structured null) | two colours not in the pair's trajectory | two concepts from another relation cluster |
| Random, full space | unit Gaussian, per pair, cell and layer | per item and layer |
| **Random, J-complement** (the null for J-removed arms) | uniform in the complement of Q_L | same |
| TA's whitened coordinate swap | yes (011's `C_white`) | yes |
| References | class-mean swap (011's `Bref_full_3L`) | J-row swap; class-mean swap, full and J-removed |

**Edit size:** every additive arm is matched per (position, layer) to **one** reference.
- **Brew:** the class-mean swap, as in 011.
- **Multihop:** the class-mean swap. Items without a class-mean arm (§4) use the J-row swap's size and are reported
  separately. Matching to the J-row swap for all items is a sensitivity analysis.
- **Power for the size reference (PI fix 2; fixed now).** A size taken from a lever that doesn't move the answer would be
  arbitrary. So the multihop class-mean **full** swap must pass 012's lever rule, measured in the same run before any
  primary contrast is read:
  - ≥ 30% flips;
  - ≥ 20 points above its random control. That control keeps d(src) and swaps toward a random unit direction (Jan's
    "true source / random target" construction).

  If it fails, the primary uses **J-row-matched sizes** for every item, and the class-mean-matched results are reported
  as secondary. Brew's class-mean reference already has power (011 s1: 0.20 [0.11, 0.24] at this size).

**Dose (*deviation*, brew only, secondary):** the J-removed AR arm and its J-complement null at 0.5×, 1× and 2× the
reference size. CLAUDE.md's pitfall: relative-to-random criteria are scale-free, so report the curve.

## 8. Scores and statistics

- **Brew:** S (011), median over 82 pairs. Table-cluster bootstrap, 10k draws.
- **Multihop:**
  - Δm toward y_b (teacher-forced; LSE over accepted spellings, every answer token, float32 logits);
  - flip (greedy, at most 8 tokens);
  - 012 Stage B's per-item works / partial / fails labels.
  - Relation-cluster bootstrap, 10k draws.
- **Paired differences** are per item, between arms on the same item.
- **Qualitative samples:** greedy continuations under every arm for 8 fixed items per family (seed 11), saved and shown
  next to the numbers.

## 9. Decision rules

### Power (per stage)

- **Brew:** answer-concept AR arm median S ≥ 0.15.
- **Multihop:**
  - the J-row reference ≥ 30% flips and ≥ 20 points above its random control (012's lever rule);
  - G-parity passes.
- A stage that fails power is reported as uninformative.
- The multihop class-mean full swap's lever rule decides the size reference (§7, PI fix 2). It does not decide
  whether the stage is read.

### Primary: J-removed AR vs the J-complement random null

Both conditions must hold (the absolute bar is a *deviation*, needed because a relative rule alone is scale-free):
- **relative:** the paired difference (J-removed AR − mean of the J-complement random draws) has a 95% CI above 0:
  - brew: on S, table-cluster bootstrap;
  - **multihop (PI fix 4): on Δm**, paired per item, relation-cluster bootstrap;
- **absolute:**
  - brew: median S ≥ 0.05;
  - **multihop: on flips**, ≥ 30% and ≥ 20 points above the J-complement null (012's own lever rule).

**Relative passes, absolute fails → "weak handle"** (PI fix 4). The direction moves the margin specifically, but not
enough to change the answer at this size. This is reported as such, not as a lever.

**Reporting within toward_tgt (PI fix 4).** The additive arms always push toward the target. The reference swaps are
reflections whose direction is set by the clean state, so on some items they go toward the source (012: toward_src J
swaps fail, 2/18 at L24-35). So every arm is also reported within each reference's toward_tgt items:
- J-row swap: `edit_direction` from the G-parity run at L36–47;
- class-mean swap: the same diagnostic on its own coordinates.

Both subsets are reported separately, next to the all-item numbers.

### Readings

- **Primary passes:** a direction with no component in 007's dense J subspace *at the edit layer* moves the answer.
  That is a lever not built from J rows, for re-testing readers (S1). It is scoped to the edit layer: re-entry into J
  downstream is not ruled out.
- **Only the full AR vector passes:** the effect is "consistent with needing the J component" **only if**:
  - the J share ≥ 0.30 (§6a);
  - and the full − J-removed difference is larger than the full − random-removed difference (paired CI excludes 0).

  Otherwise: "removing any 1,024 dimensions of this size weakens it", which is not attributed to J.
- **The J-removed class-mean passes and the J-removed AR doesn't:** the gap is the Oracle's.

### Secondary (all paired, labelled exploratory)

- full AR − reference;
- phrases − single word (was 011's gap noise?);
- starts − mentions (if only "starts" works, the handle is output-facing; see §1);
- LLM phrases − templates;
- mismatched-concept null vs random null;
- brew dose curve.

## 10. What each outcome would change

As the brief, with two additions:
- Primary passes on brew but not multihop: an output-facing family only; no hidden-step lever.
- Every AR arm ≈ the mismatched-concept null: AR directions perturb the answer generically. Not concept handles.

## 11. Order of work, stops and compute

1. **This draft.** PI approval, then commit (with `rebuild_env.sh`).
2. **CPU and API work, then the gate; stop and report.** Needs the OpenRouter key.
   - J bases;
   - phrases and code checks;
   - cue-varied prompts;
   - hand checks.

   Then on the GPU:
   - gating the prompts (about 89 × 24 × 11 short generations; estimated 30–60 GPU-min);
   - AR forwards (about 180 concepts: 89 multihop, 77 further answers, 12 colours; × about 80 inputs × 3–4 tags;
     about 20 GPU-min);
   - class-mean captures.

   Then pre-checks (a)–(c).
3. **Brew smoke** (2 pairs: gates and timing; stop), then **the brew full run** (82 pairs; about 20–30 GPU-min,
   scaled from 011 s1; stop).
4. **Multihop:** G-parity and a smoke (5 items; stop), then the full run on the PI's go. Estimate: 82 items × about
   25 arms × about 0.7 s ≈ 25–40 GPU-min. Stage B took a median 3.6 s per item for 5 conditions
   (`outputs/012_reader_concordance/stageB/items.jsonl` `sec`).

Everything is EXPLORATORY. Anything decided after a result is seen goes in as a dated amendment, and ledgers are
append-only (012 ledger).

## 12. Not settled by this stage

- Whether any reader reads a hidden step beyond what the model is about to say: the read cells stay answer-facing.
- The AR's loss and J term, and the Oracle's reward.
- Anything beyond brew and multihop, these layers, and Qwen3.6-27B.

## 13. Decisions (PI, 2026-10-10)

1. **Placement:** this file, with pointer sections in 012 (§23) and 011 (amendment 4).
2. **Edits:** the nearest-tag rule across L36–47, with the 0.7 pre-check (§6b) as the fallback trigger.
3. **The deviations:** all accepted:
   - random-removed control;
   - mismatched-concept null;
   - absolute bars on the primary;
   - answers in the forbidden list;
   - relation variety in the class-mean prompts;
   - dense J for the class-mean split;
   - brew dose curve.
4. **The class-mean direction:** fit at the final token as the primary, and at the clue's last token as a sensitivity
   check, both reported (§4).
5. **Thresholds:** as proposed:
   - J share ≥ 0.30 for the J-attribution reading;
   - nearest-tag median cosine ≥ 0.7;
   - brew absolute bar S ≥ 0.05.
6. **Still needed before step 2:** the OpenRouter key in `/workspace/.env`, ideally rotated.

## 14. Execution notes, step 2 (2026-10-10; dated, written after step 2's data had been seen)

None of these changes a threshold or a decision rule. Each changes how the material was produced, and each was found
in a smoke run or by a check.

**Changes made during generation:**
1. **Generator `max_tokens` 2000 → 16000.** In the smoke run, the France prompts and the brew "red" phrases were cut
   off (`finish_reason: length`), because the model spent about 1,920 tokens reasoning even at effort "minimal".
2. **Cue-varied prompts: 3 requests of 24 per concept**, merged in request order (§4 said 24). In the France gate
   smoke, 7 of 21 prompts passed. Request 0 keeps its seed. The text, checks, gate and caps are unchanged.
3. **Fallback parser.** One response (brew "green", "starts") listed its phrases without quotes, taking the prompt's
   "no quotes" literally. Phrases are split at a comma followed by a capital letter, and the code checks then apply as
   usual. Flagged in the record.

**Change made during gating:**

4. **Answer matcher v2.** It folds accents, maps number words 0–20 to digits, and matches token prefixes. v1 kept only
   ASCII letters, so "Brasília" ≠ "Brasilia" and "five" ≠ "5", and it matched "Lyon" inside "Lyonnais".
   - v2 changes the gate rows of 42 concepts, mostly numerals and elements. Those 42 were re-gated; every other
     concept's rows are identical under v1 and v2.
   - The v1 records are kept (`outputs/.../stageE/gate/gate_shard*.json`). The re-gates are `gate_v2a`/`gate_v2b`.

**My interpretation (PI to confirm):**

5. **The "phrase vector"** used for the gate, and proposed as the primary AR vector, pools the two LLM sets (starts +
   mentions). Each set is also reported alone.

**What the checks rejected:**

6. **The 5-gram rule** rejected 696 prompt candidates, mostly because the bank's own frames ("fact the capital of the")
   are 5-grams shared with the evaluation prompts. The rule is kept as written.
7. **Single-letter answers** C, F, J, M and O got no valid "starts" phrases (all 40 failed the position check). They
   keep their other sets (flagged weak inputs, §3).

**Results:** `stageE_bases.json`, `stageE_generation_*.json`, `stageE_ar.json`, `stageE_gate.json`,
`stageE_phrases.json` and `stageE_precheck.json` (`results/012_reader_concordance/analysis/`).

## 15. Amendment after report 41 (PI, 2026-10-10; POST HOC: written after step 2's gate results had been seen, before any edit)

**Decisions.**
- **Brew:** the stop stands. No brew edits, post hoc or otherwise.
- **Multihop primary AR vector:** pooled phrases (starts + mentions; §14 item 5, confirmed).
  - Single words are secondary.
  - Starts, mentions and template are reported alone.
- **Report 41 additions:**
  - all sets are near-equivalent on multihop;
  - the decodability margin is thin;
  - a paired bootstrap CI for phrases − word decodability (concept-cluster bootstrap, descriptive).

**Coverage rescue (POST HOC, labelled as such everywhere it is used).**

Easier cue-varied prompts are generated **only** for the 9 dropped named-place concepts: Brazil, Canada, China, Denmark,
Fuji, Miami, Nile, Olympus, Sydney. Months, numbers and people are not regenerated.

What stays the same as §4 and §14:
- forbidden words, code checks, gate, caps (≤ 4 per relation, ≤ 5 per answer);
- keep 16, minimum 10;
- 3 requests × 24, matcher v2.

Only the request text changes. Exact text:

> Write 24 short factual two-hop prompts about well-known facts that most adults know, in exactly this format: "Fact:
> The <relation> of <clue> is". The <clue> must describe "{c}" unambiguously, using a famous and unmistakable fact,
> without naming it, and the next word after "is" must be the answer. Prefer relations whose answer is one common
> word. Example, for a different concept (Kenya): "Fact: The capital of the country where the Maasai Mara reserve lies
> is" (answer: Nairobi). Rules: never write the word "{c}"; use a different clue in every prompt; use at least 8
> different relations and at most 4 prompts per relation; every answer must be one to three words, and no answer may be
> used by more than 8 prompts; do not use any of these words: {forbidden}. Return JSON: {"prompts": [...]} with
> "prompt", "clue", "relation", "answer" per item.

**Reporting the rescue:**
- final coverage by bridge type (named place / other; countries as a sub-group) and by Japan share;
- every class-mean result scoped to the covered set;
- uncovered items use the J-row-size fallback (§7) and are reported separately.

**Style control (descriptive; my addition, not a rule).**

The rescued concepts' class means come from a different prompt style than every other concept's. A swap between a
rescued and an original concept could therefore carry a style difference.
- **Check:** the same easy-prompt request is run for the 3 most frequent covered named places in the cohort (Japan,
  Egypt, Italy). Their class-mean material is **not** replaced.
- **Reported:** cos(easy-prompt class mean, original class mean) for the same concept at L36–47, against the
  cross-concept cosines.

**Step 4 now (the PI's go).**
1. **G-parity:** the J-row swap and its random control at L36–47 on all 82 items, against Stage A's
   `j_swap_L36-47` (46/82, random 3/82). Bar: ≥ 90% per-item flip agreement.
2. **Noise floors:** identity-clamp floors in this regime for:
   - Jan's `SwapEdit` (as Stage B);
   - the additive clamp (target = own clean coordinate).
3. **Class-mean lever rule (fix 2):** the class-mean full swap and its "true source / random target" control on every
   item whose src and tgt are both covered. This is estimated on the full covered set, because 5 smoke items can't
   estimate it. It runs after the rescue's gate, so the covered set is final.
4. **5-item smoke** with every §7 arm. The items are drawn at random (seed 11) from those with both class means.
   - Logged: edit sizes (realised ‖Δh‖/‖h‖) at every edited position and layer, plus coordinate errors after the bf16
     write.
5. **Stop** and report: parity, floors, edit sizes, and the class-mean lever-rule status.

## 16. Full-run additions (PI, 2026-10-10; POST HOC: written after the 5-item smoke had been seen, before any full-run edit)

Everything else stays as fixed in §9 and §15.

**Material and sizes.**
- **Items:** all 82 cohort items, split across both GPUs. Each item's clean run is recomputed in the same process.
- **References (not re-run):** the J-row swap and its random control come from the parity run. The class-mean swap and
  its random control come from the lever run. Both are deterministic in this regime: parity reproduced Stage A's
  generated text on every item. They are joined by item name.
- **Two edit sizes** for every additive arm, per (layer, position):
  - **J size:** the J-row swap's clean-state edit size. This is the primary (fix 2, since the class-mean lever failed).
  - **C size:** the class-mean swap's clean-state edit size on the 51 covered items. On the 31 uncovered items it is a
    fixed profile: per layer, the median over the 51 covered items and their prompt positions of (class-mean clean-state
    edit size / ‖h‖), applied per position as profile[L] × ‖h_clean(L, p)‖. A one-pass GPU pre-run measures the
    profile on clean activations only, before the full run.

**Arms.**

| Size | Arms |
|---|---|
| J | AR_full, AR_Jrem, AR_rrem, AR_answer, AR_mismatched, AR_mismatched_Jrem, rand_full × 2, rand_Jcomp × 2, AR_{starts, mentions, template, word} full and Jrem |
| C | AR_full, AR_Jrem, AR_rrem, AR_answer, rand_full × 2, rand_Jcomp × 2 |
| — | C_white (all items); CM_swap_Jrem (covered items) |

**Rules (fixed now).**

1. **Power at J size.**
   - AR_full@J and AR_answer@J are each compared with the mean of rand_full@J on Δm, paired per item, with a
     relation-cluster bootstrap (10k).
   - If neither has a 95% CI above 0, the J-size primary is reported as **"no power at this size"**. It is not then a
     negative result for the J-removed arm.
2. **Primary (§9, at J size).** Both conditions must hold:
   - **relative:** AR_Jrem@J against the mean of rand_Jcomp@J on Δm, paired, relation-cluster CI above 0;
   - **absolute:** AR_Jrem@J flips ≥ 30% and ≥ 20 points above the mean flip rate of rand_Jcomp@J.

   "Relative passes, absolute fails" reads as a weak handle.
3. **The J reading on the margin.**
   - The contrast is (AR_full − AR_Jrem) against (AR_full − AR_rrem), paired on Δm, at both sizes.
   - It is read under §9's rule (J share ≥ 0.30 holds: report 41), **whether or not AR_full clears the absolute flip
     bar**.
   - If AR_full is a weak handle, the result is called **"J-attribution on the margin"**, not a lever.
4. **Like-for-like lever (CPU, from the parity run).** J-row L36–47 flips and median Δm on the 51 covered items, and
   within the class-mean swap's 28 toward_tgt items, next to the class-mean swap's 10/51 and 9/28.

**Subsets reported.**
- all 82;
- covered (51) and uncovered (31), for the C-size arms;
- the 13 rescued-concept items, separately;
- within each reference's toward_tgt items;
- by bridge type (named place / other) and Japan target.

**Secondary contrasts (paired Δm, descriptive):**
- AR_full − AR_word;
- AR_starts − AR_mentions;
- (AR_starts + AR_mentions)/2 − AR_template;
- AR_mismatched − rand_full.

## 17. Energy-matched removal controls (PI, 2026-10-10; POST HOC: written after report 43, before any §17 run)

**Why.** Report 43's J reading compared removing 007's dense J subspace with removing a random 1,024-dim subspace. The
two remove very different amounts of the AR difference u:
- J removes about 0.53–0.62 of u's energy (report 41);
- a random 1,024-dim subspace removes 1,024 / 5,120 = 0.20 in expectation, about a third of what J removes.

So "J removal kills the effect, random removal doesn't" could just mean "a larger removal kills it". Two new arms match
the removed energy.

**Arms.** Run on the same 82 items, the same regime and both sizes (J and C). Every other setting is as §16.
- **AR_emrem.** Remove a random subspace of dimension d = round(s_J × 5120). Here s_J = ‖Q_L Q_Lᵀ u‖² / ‖u‖² is the J
  share of u at that (item, layer). The subspace is orthonormal, seeded per (item, layer), and the same at both sizes.
  Then renormalise. Logged: the realised removed share and cos(new vector, u).
- **AR_pcarem.** Remove a Δ-matched PCA-k subspace of clean activations at that layer, then renormalise. 007's
  construction is adapted to multihop:
  - rows are the clean block-L outputs at every prompt position of the cohort's items, centred within item, each row
    weighted 1/√n_item;
  - cross-fitted over 4 folds of relation clusters (clusters sorted, assigned to folds round-robin), so an item's basis
    never uses its own cluster;
  - k is the smallest k whose top-k share of u reaches s_J. If no k reaches it, the maximum rank is used and the case is
    flagged "unmatched";
  - logged: k, the realised removed share, cos(new vector, u), and the PCA-k basis's overlap with J (mean J share of its
    basis vectors; random expectation 0.20).
  - The clean activations come from a capture pre-pass (forward only, no edits) over all 82 items before the run.
- **Determinism check:** AR_full@J is re-run in the same process. Its Δm must equal the full run's exactly.

**Pre-stated reading** (PI). Each removal X ∈ {emrem, pcarem}, at each size:

| Condition | Reading |
|---|---|
| AR_X − AR_Jrem, paired Δm, relation-cluster CI > 0; **and** median Δm of AR_X ≥ 0.5 × median Δm of AR_full | **J-specific** |
| Only the CI condition holds | **partly energy** |
| Neither holds | withdraw "runs through J"; report **"the handle breaks under any removal of this size"** |

Until §17 has been read, report 43 states the J result as "removing J removes the effect; removing a random subspace
of the same dimension (about a fifth of u's energy, a third of what J removes) does not".

