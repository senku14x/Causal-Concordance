# 002 — Reader claim scoring (role-aware, closed-set) — design v2

*2026-10-02.*
- **What this is:** exploratory development of the reader-scoring procedure on dev material. The material is the
  saved 001 reader snapshot (10 brew tables) and the saved 001 activation captures (48 dev tables). There are no new
  reader generations.
- **Freezing:** the procedure is frozen at the end of the 20-readout prompt pilot (§7, step 2) and before any
  held-out use.
- **PI go (2026-10-02):** "after you're done with this design 002 in detail … start with it". Judge choice:
  "use gemini 3.8 flash, the same as WorkspaceBench". Open choices I took on my recommendation: the Oracle WSB grid
  is included and reported separately; the label design is renamed 003.

## 1. Why

The 001 snapshot scored readers by word mention: does a colour word occur anywhere in the readout? The external audit
(`docs/reviews/2026-10-02_codex_reader_audit/`; report 001 §15) showed that this overstates what readers claim:
- of 17 NLA mentions of z at `:`, about 5 give z its correct role;
- lens top-10 lists and long descriptions are not comparable claim budgets;
- marginal real-vs-shuffled rates are weaker evidence than within-table contrasts.

The spec's scoring rules (§7 and A01 §4) were never implemented: matched claim budgets, closed-set candidates, the
token-reader hit rule, a judge that sees only the readout and the options, abstention, and prompt-only and
no-activation baselines. 002 implements them on data we already have, so that a held-out run (and 003's labels) has
a frozen, tested scorer.

**Question 002 answers (descriptively, n = 10 tables):** when a reader makes a claim about the colour chain, how often
does it put the right colour in the right role? Does that depend on the activation (real vs shuffled vs none;
within-table start swap)? And how does it compare with the token readers' closed-set ranks?

## 2. Material

| Reader | Readouts | Cells, layers | Arms |
|---|---|---|---|
| NLA (`av_rl_lora_step400`), k = 4 | 880 | assistant newline, `</think>` newline, `:`; L42 | real 360, shuffled 360, real-SEC 120 (`:`), none 40 |
| Oracle contract (`olens_s3d_rl600`), k = 4 | 960 | `:`; L40, L44 | real 240, shuffled 240, real-SEC 240, none 240 |
| Oracle WSB grid, k = 1 | 2,090 | 3 cells; L20–60 every 4 | real 990, shuffled 990, none 110 |
| Lenses: logit, J_NP (standard, cosine), J_CB, R_CB | saved top-10 lists; recomputed candidate scores (§5) | 3 cells; snapshot grid (top-10); L40/42/44/56/60 (scores) | real, shuffled (top-10); real + mean-activation reference (scores) |

Each reader runs on the member render, its start partner and the substituted render. All readouts are in
`results/001_leverage_profile/analysis/snapshot.json`. Per-table no-activation runs are stored under 4-part keys
(`none|L|reader|none`).

**Closed set per table:** WSB's 5 options for the item, i.e. start, z, answer and 2 off-path colours (the A01 §4 set).
A judge may also answer "none", "cannot tell" or "conflicting".

## 3. Judge

**Model and call.** WorkspaceBench's judge: `google/gemini-3.8-flash` via OpenRouter
(`https://openrouter.ai/api/v1/chat/completions`), with:
- `reasoning: {"effort": "minimal"}` (Gemini cannot turn reasoning off; WSB's setting);
- `provider: {"require_parameters": true}`;
- `temperature: 0`, chosen for reproducibility (WSB leaves it unset);
- `max_tokens: 2000`;
- `response_format`: a strict JSON schema built per readout, where each of `start`, `intermediate`, `answer` is an
  enum of that table's 5 candidates plus `"none"`, `"cannot tell"`, `"conflicting"`.

The key is read from `/workspace/.env` (`OPENROUTER_API_KEY`), never logged or committed.

**Inputs.** The judge sees **only the readout text and the 5 candidates**, never the prompt, the table or the truth.
Candidate order is shuffled per readout with a fixed seed (`20261004` + the readout id's code points). Oracle samples
contain several bullet answers and are judged as one readout; inconsistency between bullets is what "conflicting" is
for.

**Prompt.** System: "You extract claims from text about a colour puzzle. Reply with JSON only." User: Appendix A,
frozen after the pilot; its sha256 is recorded in the manifest.

**Robustness and logging.**
- Concurrency 16; up to 4 attempts per readout with exponential backoff on HTTP errors, empty choices or schema
  violations. After 4 failures the readout is recorded as `judge_error`.
- Every request and response is saved with token usage and cost in `results/002_claim_scoring/judge/claims.jsonl`.
  Each row holds: the readout id, reader, arm, role, cell, layer, table, sample index, candidate order, the judge
  JSON, usage, cost, model id returned, provider and attempt count.
- Hard spend cap: $10 total, checked before every call. The estimate is about $2.35 per full pass.

**Second judge.** Not run: the PI chose one judge. The judge is validated against author codes (§4). If that
agreement fails the stop rule (§9), a local second judge (stock Qwen3.6-27B) is the fallback, decided with the PI.

## 4. Validation of the judge

- **Pilot set:** 20 readouts drawn at random (seed 20261004). Used only to debug the prompt and parsers; every change
  is logged in §11 with the reason. No scores from the pilot are reported.
- **Validation set:** 60 readouts disjoint from the pilot, stratified by reader × arm:

  | Reader | real | shuffled | none |
  |---|---|---|---|
  | NLA | 12 | 6 | 4 |
  | Oracle contract | 12 | 6 | 4 |
  | Oracle WSB | 12 | 4 | — |

  `scripts/002_claim_scoring/make_sets.py`.
- **Author codes:** the author (Claude) role-codes the 60 readouts **blind to the table's trajectory and before any
  judge output on them is seen**. Codes are committed before the judge runs on the validation set
  (`results/002_claim_scoring/validation/author_codes.jsonl`). Reported:
  - exact agreement and Cohen's κ per role, overall and per reader;
  - a confusion matrix over {candidate correct for the role, other candidate, none, cannot tell, conflicting}.
- **PI spot-check:** a page with 50 randomly drawn judged readouts (text, candidates, judge JSON) for the PI to mark
  agree / disagree, reported when available.

## 5. Token readers (judge-free)

1. **Top-10 claim (all snapshot layers, 10 tables, real and shuffled).** The reader's claim is the highest-ranked
   candidate colour among its saved top 10 tokens (English word match, either case; Chinese colour characters as in
   001). It abstains if no candidate appears. Budget 1, no role.
2. **Closed-set scores (A01 §4; L40/42/44/56/60; all 48 dev tables; member and partner renders; `:` and both newline
   cells).**
   - For each lens, compute each candidate's score from the saved SEC captures and the pinned lens files, on CPU or
     GPU.
     - logit lens: final norm, then W_U;
     - J_NP standard / J_CB / R_CB: softmax(W_U · norm(J h)), as in `readers.VectorLenses`;
     - J_NP cosine: WSB's cosine readout.
     - A candidate's score is the maximum over its single-token spellings " c" / " C". Multi-token spellings, e.g.
       " Teal", are skipped, and that is reported.
   - **Hit (A01 rule):** z ranks first among the 5 candidates **and** z's score exceeds the score the same lens gives z
     on the reference input. The reference is the mean activation over the 231 dev renders at that cell and layer.
   - Also reported: reciprocal rank of z, start and answer; the same hit rule for start and answer.
   - Activations: SEC-regime captures; the 001 bridge shows stock and SEC reader inputs agree (cos ≥ 0.99985; 99.4% of
     top-10 lists).

## 6. Scores and contrasts

**Per verbalizer readout** (from the judge JSON):
- **mention(z)**: the 001 word match, kept for continuity;
- **role-correct(z)**: intermediate = z. **The primary score.**
- **contradicted**: intermediate = another candidate;
- **misplaced(z)**: z given the start or answer role;
- **abstain**: intermediate ∈ {none, cannot tell, conflicting}, each reported separately.

The same scores are computed for start and answer. For k = 4 readers, report the per-sample rate and the per-cell
majority claim with self-consistency (the share of samples equal to the majority).

**Matched budget.** Every reader contributes at most one colour per role per readout: the judge's pick, or a lens's
top candidate. A lens has no roles, so "lens hit on z" is set beside "verbalizer role-correct(z)", and the difference
in what each measures is stated wherever both appear.

**Contrasts** (within table; per-table values listed; table-bootstrap 95% intervals and exact two-sided sign-flip
p-values, unadjusted and labelled exploratory):
- (a) real − shuffled, and real − none;
- (b) **start-swap tracking on role-correct claims:** [(member: z in role − partner: z in role) + (partner: z′ in
  role − member: z′ in role)] / 2;
- (c) substitution tracking on mentions (c and c′ are off-path, so they have no role);
- (d) own-z − other tables' z (the colour-preference control; role-correct version: how often z_other is put in the
  intermediate role);
- (e) real vs real-SEC: agreement of role claims, the bridge for judged claims.

**Baseline: prompt-only** (A01 C18: stock Qwen3.6-27B, stock bf16 forward, thinking off).
- **Inputs:** the 231 natural dev renders (48 tables). The user turn is the brew problem followed by "Question: what
  colour is the potion after the first stir? Answer with one colour word."; the assistant turn is empty.
- **Score:** each candidate's log-probability as the full answer word (LSE over the "c" / "C" spellings, all tokens
  scored).
- **Pick:** the argmax.
- **Reported:** accuracy for z, and likewise for start ("the potion's colour before stirring") and answer. It is the
  ceiling on what the prompt reveals, not a reader.

## 7. Procedure (order of work)

1. `make_sets.py` draws the pilot and validation sets (done). The author codes the 60 validation readouts blind;
   commit.
2. **Pilot:** judge the 20 pilot readouts; read every response; fix only the prompt wording or format and parsing;
   log each change in §11; freeze Appendix A (sha256); commit.
3. **Judge run:** all 3,930 readouts (validation included). Resumable: rows already in `claims.jsonl` are skipped.
4. **Validation:** judge vs author codes (§4). Check the stop rules (§9) before any reader result is read.
5. **Token readers:** top-10 claims and closed-set scores (§5).
6. **Prompt-only baseline** (§6).
7. **Analysis:** `results/002_claim_scoring/analysis/{scores.json, contrasts.json, validation.json,
   qualitative.json}`.
8. **Report 002** with qualitative samples: for each reader, at least two readouts each of role-correct, contradicted,
   misplaced, abstained and conflicting, with the judge JSON beside them. Then the results-page section (native
   charts) and STATUS / SPEC_STATUS updates.

## 8. Predictions (priors; written before any judge output)

| # | Prediction | Prior |
|---|---|---|
| R1 | NLA L42 `:` role-correct(z), real, < 0.25 | 60% |
| R2 | Oracle L40 `:` role-correct(z): real − shuffled ≥ 0.10 | 40% |
| R3 | NLA start-swap tracking on role-correct claims positive in ≥ 7/10 tables at the assistant newline | 45% |
| R4 | Judge–author Cohen's κ on the intermediate role ≥ 0.6 | 70% |
| R5 | Lens closed-set hit on z at L40–44 (A01 rule) ≤ 0.2 for every lens; at L60, answer hit ≥ 0.8 | 70% |
| R6 | Prompt-only picks z in ≥ 0.95 of renders | 85% |
| R7 | "conflicting" is the most common Oracle (contract, real) outcome for the intermediate role | 50% |

## 9. Gates and stop rules

- **Judge errors:** if `judge_error` exceeds 5% of readouts, stop, diagnose and re-run before any analysis.
- **Judge validity:** if judge–author κ on the intermediate role is below 0.6, judge-based reader scores are **not
  reported as results**. Raise it with the PI: fix the prompt on the pilot, or add the local second judge.
- **Spend:** if it reaches $10, stop.
- **Bridge:** report real vs real-SEC agreement; if it is below 0.8 on the intermediate role, flag the judge as noisy at
  the level of individual claims.

## 10. Budget

| Step | Resource | Estimate |
|---|---|---|
| Judge, full pass (3,930 readouts, about 2.0 M input and 0.24 M output tokens) | OpenRouter | about $2.35, about 15–20 min |
| Pilot and validation iterations | OpenRouter | about $0.15 |
| Lens closed-set scores | CPU/GPU | minutes |
| Prompt-only | GPU, 231 renders × 5 candidates | about 10 min |

## 11. Changes log (pilot and after)

*(filled in as changes happen; none yet)*

## Appendix A — judge prompt (v2 draft; frozen after the pilot)

```
You will read a short text written by an AI system. The text may describe a colour puzzle in which a colour rule is
applied twice: START colour -> INTERMEDIATE colour (after one application) -> ANSWER colour (after two).

Candidate colours: {candidates}

According to the text, which candidate is the START colour, which is the INTERMEDIATE colour, and which is the ANSWER
colour? Rules:
- If the text says to apply the rule twice to X (or that X is where the chain begins), X is the START.
- If the text says a first step, first iteration or first answer produced X, X is the INTERMEDIATE.
- A chain written as A -> B -> C means START = A, INTERMEDIATE = B, ANSWER = C.
- If the text gives a role a colour that is not a candidate, answer "none".
- If the text does not state a role, or only lists or mentions colours without roles, answer "cannot tell".
- If the text gives the same role different colours in different places, answer "conflicting".
Use only the text. Do not solve the puzzle yourself.

TEXT:
{readout}
```
