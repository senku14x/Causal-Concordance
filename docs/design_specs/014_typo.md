# 014: the reading-side suite on WorkspaceBench typo (design spec, pre-registered 2026-10-07 before any run)

Branch `014-typo` (off `013-nla-hallucination`). Ledger `docs/ledger/014_ledger.md`. EXPLORATORY. PI: "lets run on typo
at least what we can and then we can try on the other sets".

| § | Section | What it does |
|---|---|---|
| 1 | Question | why typo, and what it can and cannot tell us |
| 2 | Material | prompts, read cell, layers, the correctly spelled control |
| 3 | Part A: readers on activation parts (007-style) | which part of the activation the readers name the corrected word from |
| 4 | Part B: reader agreement (010-style) | do the readers agree on which items they name it for |
| 5 | Part C: the NLA hallucination test (013 pipeline) | do the NLA's false claims carry signal here too |
| 6 | Predictions and readings | what we expect, fixed now |
| 7 | Gates, smoke, cost | checks, where I stop |
| 8 | Not claimed | limits |

## 1. Question

*What this section does: says why typo and what it can't show.*

- In typo the hidden step is the corrected word ("Her birthday falls on the last day of Febuary" → *February*).
- It is **not the next token**, unlike multihop's bridge, whose answer comes next. So output-facing reading is less of
  an issue.
- **But it is recoverable from the prompt.** WorkspaceBench reports a prompt-only baseline of 1.00 on typo, with a
  correctly spelled control bank scoring the same.
- **Typo has no answer that depends on the corrected word,** so the causal parts of the suite (007 model side, 008,
  011, 012) can't run. This experiment covers reading only: where the corrected word is decodable, whether readers
  agree, and what the NLA's false claims do for its reconstructor.

## 2. Material

*What this section does: fixes the data.*

- **Prompts:** WorkspaceBench `evals/typo/items.json` @ 92d763e. 100 items, all already gated by WorkspaceBench
  (consistency 1.0). Each has one single-token target (`intermediates[0]`).
- **Render:** WorkspaceBench "plain": the prompt text, no chat template, no special tokens.
- **Read cell:** the final prompt token (the last piece of the misspelled word), WorkspaceBench `final_prompt_token`.
- **Captures:** stock bf16 forward (`readers.capture_stock`, WorkspaceBench convention), block outputs at L40, L42 and
  L44. The model's greedy continuation (8 tokens) is saved for the judges.
- **Correctly spelled control (echo ceiling):** the same prompt with its last word replaced by the target ("…last day
  of February"), captured the same way and read from the whole activation only. It measures how often a reader names
  the word when the word is literally there.
- **Activation parts** at each layer, with the 007 constructions and bases:
  - whole;
  - **J part** (007's dense J basis `J_raw_L{L}`, first 1,024 columns, orthonormalised);
  - **rest** (whole − J);
  - **random 1,024 directions** (`pieces.rand_basis_f32(p007.rand_seed("r1024", 0), 1024)`, as 007's R0);
  - **mean removed** (whole minus its projection on the unit mean of the 100 typo whole activations at that layer;
    007's MR construction, with typo's own mean).
  - 007's PCA-k needs member–partner pairs and is not built here.

## 3. Part A: readers on activation parts (007-style)

*What this section does: asks from which part of the activation each reader names the corrected word.*

- **Readers:**
  - Oracle L40 and L44 (`readers.OracleLens.read_batch`, 2 samples, seed `p007.SEED`, chunks of 32);
  - NLA L42 (`readers.NLArl.read_batch`, the same settings; run after the subject model is freed);
  - token lenses J_NP / J_CB / logit at L40, L42 and L44 on the whole activation: rank of the target token
    (descriptive).
- **Score:** the share of sampled readouts naming the target word: whole word, case-insensitive (`c012._hit`).
  Item-bootstrap 95%. The echo control is scored the same way.

## 4. Part B: reader agreement (010-style)

*What this section does: checks whether the readers name the word on the same items.*

- Per item, "named" = at least 1 of the 2 samples names the target, from the whole activation.
- Cohen's κ for Oracle L40 vs NLA L42, Oracle L44 vs NLA L42, and Oracle L40 vs L44.
- The items named by only one reader, at equal opportunity (one layer each).

## 5. Part C: the NLA hallucination test (013 pipeline, unchanged except the data)

*What this section does: repeats 013 on typo's NLA explanations.*

- **Explanations:** the NLA L42 readouts of the *whole* activation (100 items × 2 samples = 200). The model's
  continuation and the prompt go to the judges.
- **Pipeline:** the same judges, prompts, versions and statistics as 013 (spec §3–§7, A1, A2):
  - deletion: C1 against R1/RS1; per kind, C1i / C1ii against their matched controls;
  - rewrite: F against P;
  - FVE with B on the 100 typo L42 activations, and the J share.
- **Gates:**
  - the de-risk gate R1 from 013 (FVE ≥ 40% and matched − shuffled ≥ 20 points) must pass on typo first; if it
    fails, Part C stops and is reported, as brew was;
  - G1 / G3 as in 013;
  - my hand check of 20 cleaner outputs (precision ≥ 0.80).
- **Readings:** 013's bands (§6, A1, A2), applied as there.

## 6. Predictions and readings (fixed now)

*What this section does: commits to expectations before any number.*

- **P1:** every reader names the target from the whole activation in ≥ 0.80 of readouts. Close to the echo control,
  because the word is recoverable from the prompt.
- **P2 (the informative one; uncertain):**
  - In multihop the content sat in the J part and the rest kept the *form* (current word, frame).
  - In typo the corrected word *is* the current word.
  - Prediction: the rest keeps the target more often than in multihop (rest ≥ 0.40 for at least one reader), and the
    J part still names it (≥ 0.70).
  - If the rest < 0.25 for every reader: typo behaves like multihop (content in J). Otherwise "the corrected word is
    readable outside J".
- **P3:** agreement is high (κ ≥ 0.4 for every pair), because most items are named by everyone.
- **P4 (Part C):**
  - invented context is padding again (Δi inside ±2 points);
  - misquotes carry signal (Δii > +2);
  - the factual rewrite helps by a few points.
  - Any departure is reported as such.

## 7. Gates, smoke, cost

*What this section does: says what has to pass and where I stop.*

- **Gates:**
  - the render assertion (the read token is the last prompt token, and decodes to the end of the misspelled word);
  - the piece reconstruction check (J + rest = whole, ≤ 1e-6 relative);
  - the NLA injection check (as in 013's de-risk).
- **Smoke:** 10 items, every part (captures, pieces, Oracle and NLA readouts, a 5-explanation judge pass).
  **Stop and report.** Then the full 100 on the PI's go.
- **Cost:** about 0.5–1 GPU-h in total; judges about $1.

## 8. Not claimed

*What this section does: states the limits.*

- Reading only. Nothing here shows the model *uses* the corrected word.
- WorkspaceBench's prompt-only baseline is 1.00, so naming success can come from the prompt.
- One read cell, layers 40–44, and the WorkspaceBench typo bank only.
