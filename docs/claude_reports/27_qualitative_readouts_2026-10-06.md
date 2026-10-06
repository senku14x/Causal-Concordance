# What the readers actually say: a qualitative analysis of 007's saved readouts (2026-10-06)

**EXPLORATORY. Post hoc, CPU only, on saved readouts. Word match, no judge.**
- No model, GPU or paid API was used.
- A subagent wrote and ran the analysis; I re-counted the headline brew numbers with independent code.
  - The J-piece counts match exactly.
  - The rest_J "no colour" counts differ only by CJK colour words, which my English-only recount does not match.
- Code, tables and quotes are in `results/007_jsubspace/posthoc_qualitative/` (`qual_analysis.py` reproduces every file
  byte-for-byte):
  - `qual_summary.json` holds every number with Wilson intervals;
  - `qual_appendix.md` holds 63 fixed-seed quotes with hand codes;
  - `qual_tables.md` holds the full tables.

**Scope and limits.**
- **Brew:** only the `:` cell. 8 tables give 16 renders × 2 samples = **32 samples per reader-layer and piece** (the 95%
  intervals are about ±0.17). The pieces are whole, J, rest_J, PCA-4, MR (mean-removed), GMR (generic-mean-removed) and
  R0 (one random 1,024-dim slice).
- **No `</think>` readouts were saved** (Stage 1b and the brew test kept likelihood matrices only).
- **Multihop:** test split, 40 items × 2 samples = 80 per piece.
- Word matches count mentions, not roles.
- The hand codes are one coder's, not blind to the path, and unvalidated.

## Findings

1. **Outside J the readers lose the colours but keep the task's shape.**
   - rest_J readouts name no colour in 26/32 (Oracle L40), 32/32 (Oracle L44) and 27/32 (NLA). For J the figures are 1,
     2 and 3/32.
   - rest_J still describes a two-step lookup ending in an answer slot, but in another domain (URL jumps, compiler
     passes). That holds for all 9 hand-coded rest_J samples.
   - The prompt's own words (potion, stir, brew) appear in **0 of 672** samples, so the readers never literally re-tell the
     problem.
   - Task-structure words are common in every piece, random R0 included (21–26/32), so they do not mark reading.
   - Fits the likelihood result: the colour signal is J-concentrated; the abstract task frame is everywhere.
2. **In J, the readers name z for the right render, but no more often than the answer, and mostly as a result.**
   - The control is the pair-mate's z: the same table's colours, so the expected count is identical. Own z vs mate's z in
     J: 16 vs 8 (Oracle L40), 13 vs 5 (Oracle L44), 11 vs 6 (NLA).
   - J names the answer as often: 18, 16 and 13/32. It names z without the answer in only 6, 6 and 9/32.
   - Of 9 hand-coded J samples, 4 frame z as a final result and only 1 as a step.
   - This fits the output-facing reading (z as a candidate answer) better than a held intermediate.
3. **Colour-naming readouts are full of invented specifics, from both readers.**
   - "Yellow" is not a brew colour; it appears in 8–19/32 whole and J samples.
   - The first colour named is off-path in 15–25/32 of whole and J samples.
   - Oracle L44 J over-names the colour one step *past* the answer: 19 observed vs 12.0 expected from colour priors. That
     looks like chaining through the table rather than reading one held colour.
   - In brew, hallucination is not NLA-specific.
4. **A lead, not a result: mean removal surfaces z as a theme, without the task.**
   - MR names z in 6, 9 and 6/32 samples (expected from colour priors: 0.7, 0.7, 0.3), mostly without the answer.
   - z appears as a connotation inside unrelated text ("The pink collar is a permanent mark…", "4 gold bars").
   - It rests on the same 3–5 of 16 renders across all three reader-layers.
   - Report 22 saw the same thing with the J-lens on mean-removed pieces.
5. **Multihop.**
   - The consistent-denominator restatement shares reproduce report 26: among bridge-naming J samples on items with cue
     words, 15/53, 13/54 and 8/53 also restate a cue.
   - J names the bridge in 71–75/80 samples, nearly always together with the target.
   - Outside J the NLA names the bridge *without* the target: 17/18 bridge-naming rest_J samples and 28/43 R0 samples. The
     R0 samples read were weather or travel filler with the country inserted, rarely restating the cue.
   - `stage3.json`'s `P(restates_cue)` divides by all 80 samples, so it is not this share.

## Key table (brew `:`; k/32)

| reader | piece | z | answer | no colour | own z vs mate's z |
|---|---|---|---|---|---|
| Oracle L40 | whole | 11 | 15 | 0 | 11 vs 8 |
| | J | 16 | 18 | 1 | 16 vs 8 |
| | rest_J | 2 | 0 | 26 | 2 vs 1 |
| | MR | 6 | 5 | 19 | 6 vs 0 |
| | R0 | 1 | 4 | 22 | 1 vs 1 |
| Oracle L44 | whole | 13 | 11 | 5 | 13 vs 5 |
| | J | 13 | 16 | 2 | 13 vs 5 |
| | rest_J | 0 | 0 | 32 | 0 vs 0 |
| | MR | 9 | 1 | 18 | 9 vs 1 |
| | R0 | 1 | 0 | 30 | 1 vs 0 |
| NLA L42 | whole | 12 | 10 | 5 | 12 vs 4 |
| | J | 11 | 13 | 3 | 11 vs 6 |
| | rest_J | 1 | 0 | 27 | 1 vs 0 |
| | MR | 6 | 3 | 22 | 6 vs 0 |
| | R0 | 0 | 0 | 32 | 0 vs 0 |

PCA-4 and GMR rows are in `qual_tables.md`.

## What this changes

- **It supports the output-facing interpretation of the reader-side J result.** In J the readers name z about as often
  as the answer, and frame it as a result.
- **Score hallucination on brew against exact truth**, for both readers, rather than through an AR. Yellow and other
  invented colours are directly countable.
- **The MR lead is worth one targeted check** in the next reader-side run: z as content without the answer frame. It is
  only 3–5 renders today.
- **Save `</think>` readouts in future runs.** The cell where the Oracle's z signal is strongest but unused has no saved
  text.
