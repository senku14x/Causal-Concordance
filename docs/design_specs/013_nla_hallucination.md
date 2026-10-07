# 013: Do the NLA's false claims carry activation signal? (design spec, pre-registered 2026-10-07)

Branch `013-nla-hallucination`. Ledger `docs/ledger/013_ledger.md`. Plain-language plan and de-risk:
`/workspace/claude_suggestions/013_plan_nla_hallucination_2026-10-07.md`. EXPLORATORY. Written and committed before
any judge call.

**How to read this spec.** Each section opens with a line saying what it does.

| § | Section | What it does |
|---|---|---|
| 1 | Question | the one question this experiment answers, and why it matters |
| 2 | Material | which prompts, activations and NLA readouts are used (all already saved) |
| 3 | Judges | the two LLMs that find false claims, and exactly what they are asked |
| 4 | Versions | the text variants built from each explanation (original, cleaned, random-deleted, …) |
| 5 | Measures | how each variant is scored with the NLA's own reconstructor |
| 6 | Decision rules | the thresholds, fixed now, that turn the numbers into a verdict |
| 7 | Gates | the checks that must pass before any result is read (judge quality, reproducibility) |
| 8 | Smoke and run | what runs first (the smoke), where I stop, and what runs after your go |
| 9 | Cost | GPU time and judge spend |
| 10 | Not claimed | what the result cannot show |

## 1. Question

*What this section does: states the one question and the competing answers.*

NLA explanations contain many false claims. On the 012 multihop readouts, many describe a quiz or worksheet framing
that the prompt doesn't have. **Do those false claims carry information about the activation that the NLA's
reconstructor uses?** Two competing answers:
- **(H-signal)** false claims carry activation information. Deleting them costs more reconstruction than deleting the
  same amount of text at random. Then the NLA encodes real signal in confabulated text, and its claims can't be read
  at face value.
- **(H-padding)** false claims carry less information than average text. Deleting them costs less than random
  deletion.

Secondary (Camila's "outside-J content"): does what the false claims contribute lie in 007's dense J subspace at L42,
or outside it?

## 2. Material

*What this section does: fixes the data. Nothing new is generated from the subject model.*

- **Prompts:** the WorkspaceBench multihop prompts. They are the same text in 007 and in Jan Chen's bank; 87 items
  are covered in 012.
- **Read cell:** the final token, layer 42. The 012 states are clean, J swap L24–35 and random-target swap L24–35.
- **Activations:** `outputs/012_reader_concordance/stageB/captures.pt` `X[item, arm, L42]`: 87 × 3 = 261.
- **Explanations:** the saved 012 NLA readouts, `outputs/012_reader_concordance/stageC/readouts_nla.jsonl`.
  - nla-rl av_base + iter_000400, L42, on-training, WSB sampling;
  - 2 samples per state: **522 explanations**, all extracted;
  - the explanation text is EasyNLA's `extract_explanation`, as in 012 Stage K.
- **The model's own continuation** in each state: the greedy text recorded in 012 Stage B
  (`stageB/items.jsonl` `arms[state].text`), e.g. " Tokyo.\n" after a working Italy → Japan swap.
- **Brew is excluded:** the de-risk showed reconstructor FVE −556% on brew (ledger 013).
- **Unit and cluster:** one explanation is the unit (522). Bootstrap clusters are items (87).

## 3. Judges

*What this section does: says which two LLMs find the false claims, what each is told, and why there are two.*

Both are reached through OpenRouter, with the key in `/workspace/.env`. Temperature 0. Reasoning effort "low". Every
request and response is cached in `outputs/013_nla_hallucination/judge/`. Token usage and cost are logged.

**Judge A, the "cleaner": `google/gemini-3.8-flash`.** It returns JSON
`{"contradicted": [spans], "unsupported": [spans]}`:
- each span is an exact, verbatim substring of the explanation covering one whole claim (a sentence or clause);
- spans don't overlap; anything not flagged is left alone.

**Judge B, the "labeller": `z-ai/glm-5.3-flash`.** A different model family, as the PI decided. It labels every
sentence of the explanation independently, as `consistent` / `contradicted` / `unsupported` / `format`. Sentences come
from a fixed splitter: sentence-final punctuation followed by whitespace, plus newlines. B's labels check A (§7, G3)
and feed the later content labelling. They don't change any version.

**What both judges are told** (the same definitions for both; exact prompt texts are saved in the results JSON):
- **Inputs:** the subject prompt; the model's own continuation in this state; the explanation.
- **For J and random states:** "the model's internal state was edited; its continuation is what it actually computed.
  For claims about which entity or answer the model is working with, the continuation is the truth, not the prompt."
  This keeps the cleaning from deleting exactly what a swap put there (e.g. "Japan" after an IKEA → Japan swap that
  made the model say "Tokyo").
- **CONTRADICTED:** the claim is false given the inputs:
  - (a) it asserts text or structure in the input that isn't there: invented quotes, headers, answer options,
    earlier questions, misquotes;
  - (b) it names an entity or answer that conflicts with the model's continuation;
  - (c) it is a factually false statement about the world.
- **UNSUPPORTED:** not false, but neither stated in nor reasonably inferable from the inputs (speculative context,
  guessed intent). A reasonable genre inference (e.g. "a factual statement to complete" for a "Fact:" prompt) is
  **not** unsupported.
- **Otherwise** the claim is left alone (A) or labelled consistent or format (B).

## 4. Versions

*What this section does: defines the text variants that go through the reconstructor, built deterministically (no
LLM writes new text).*

For each explanation O:

| Version | Built how | Role |
|---|---|---|
| **O** | the original explanation | reference |
| **C1** | O with A's *contradicted* spans deleted | **primary cleaning** |
| **C2** | O with *contradicted + unsupported* spans deleted | secondary cleaning |
| **R1** | O with random clause units deleted, matched in tokens to what C1 removed | **primary control** (the PI's length-matched random deletion) |
| **R2** | the same, matched to C2 | secondary control |
| **K1** | O with random *unflagged* clause units deleted, matched to C1 | secondary: false text against an equal amount of not-flagged text |

- **Clause units:** O is split at sentence ends, newlines and `,` `;` `:`.
- **Random matching:** units are shuffled with a fixed seed and added while the deleted token count stays ≤ the
  target; then the single unit that brings the total closest to the target is added if it helps. Five seeds per
  explanation; the MSE is averaged over them. Tokens are counted with the AV tokenizer.
- **Clean-up:** whitespace is normalised after any deletion; nothing else is changed.
- **Degenerate cases, excluded from the primary analysis and counted:**
  - C1 removes nothing (no contradicted span): "nothing to clean";
  - C1 leaves fewer than 10 tokens: "cleaned away";
  - K1 is not computable when unflagged text is shorter than the target.

## 5. Measures

*What this section does: says how each version is scored. This is the reconstructor path validated in 012 Stage K.*

- **Reconstructor:** the final nla-rl AR, through EasyNLA's `NLACriticModel` + `critic_predict`, with the critic
  template and mse_scale √5120; the same code as 012 Stage K and the 013 de-risk.
- **Per version:** MSE = mean (n(pred) − n(gold))², with n rescaling to norm √5120; FVE_v = 1 − MSE_v / B, where B is
  the predict-the-mean baseline of the 261 activations (one B for the set, as in the de-risk).
- **Primary statistic:** **Δ = FVE_R1 − FVE_C1** in FVE points, per explanation.
  - Positive means deleting the false claims costs more than deleting random text: the false claims carry above-average
    signal.
  - Reported as the mean over eligible explanations, with a cluster bootstrap by item (2,000 draws, seed 0).
- **Also reported:**
  - FVE_O − FVE_C1 (absolute cost of cleaning);
  - Δ2 = FVE_R2 − FVE_C2;
  - FVE_K1 − FVE_C1;
  - each of the above per state (clean / J / random);
  - the share of each explanation's tokens flagged contradicted or unsupported.
- **J share (secondary):** share_J(v) = ‖Q_Jᵀ(n(p_v) − μ)‖² / ‖n(p_v) − μ‖².
  - Q_J is 007's dense J basis at L42 (`outputs/007_jsubspace/bases/J_raw_L42.pt`, first 1,024 columns,
    orthonormalised, as `geometry.load_basis`); μ is the mean of n(gold) over the 261 activations.
  - Reported: ΔJ_C = share_J(C1) − share_J(O) against ΔJ_R = share_J(R1) − share_J(O), paired, with a bootstrap CI;
    plus share_J(gold) for reference. Descriptive.

## 6. Decision rules

*What this section does: fixes, before any judge call, how the primary number becomes a verdict.*

On eligible explanations (≥ 1 contradicted span; C1 not degenerate), pooled over the three states:

| Verdict | Condition on mean Δ (FVE points) and its 95% cluster-bootstrap CI |
|---|---|
| **False claims carry extra signal** (H-signal) | mean ≥ +2 and CI lower bound > 0 |
| **False claims carry less signal than average text** (H-padding) | mean ≤ −2 and CI upper bound < 0 |
| **No difference beyond ±2 points** | CI entirely inside [−2, +2] |
| **Unresolved** | anything else |

- The same rule applied to Δ2 is secondary. The per-state results and the J share are descriptive.
- A per-state difference is not a "state-specific" claim unless a paired contrast between states has a CI excluding 0
  (the §20 lesson from 012).

## 7. Gates

*What this section does: lists the checks that must pass before any result is read; a failed gate stops the run.*

- **G-repro:** O's reconstructor MSE on the 87 clean first samples reproduces the de-risk scores
  (`outputs/013_nla_hallucination/derisk/multihop_ar_scores.jsonl`) to ≤ 1e-6. Same code path.
- **G1, judge parse:** at least 90% of A's responses are valid JSON whose spans are verbatim substrings (after
  whitespace normalisation). One retry per failure; failures are excluded and counted.
- **G2, hand check** (by me, before any Δ is computed; 20 smoke explanations, then 10 more before the full run):
  - the precision of A's contradicted spans must be ≥ 0.80, meaning I agree the span is contradicted under §3;
  - missed contradicted claims are listed (descriptive recall);
  - if G2 fails, A's prompt is revised in a written amendment and the smoke is re-run.
- **G3, A–B agreement** (descriptive): Cohen's κ, per sentence, between "contains an A-contradicted span" and B's
  "contradicted". κ < 0.4 is reported as weak agreement and noted next to every result.

## 8. Smoke and run

*What this section does: says what runs now, where I stop, and what needs the PI's go.*

- **Smoke (now):** 20 explanations, seeded (`random.Random(13)`), stratified 7 clean / 7 J / 6 random.
  - All versions, the reconstructor scoring and G-repro; G1–G3 on these 20.
  - **I hand-check before looking at Δ, then stop and report.**
- **Full run (on the PI's go):** all 522 explanations, plus the 10 more hand-checks first.

## 9. Cost

*What this section does: says what it costs.*

- **GPU:** about 5 min for the smoke; about 0.3 GPU-h for the full run (522 × 1 + 5 × 2 + 4 versions ≈ 7,000
  reconstructor forwards, 2 min of model load).
- **Judges:** about 1.5 k input and 0.4 k output tokens per call, two calls per explanation.
  - Smoke: under $0.10.
  - Full run: about $1–3.
  - Reasoning tokens can multiply the output (002's lesson), so actual usage is logged per call. The full run aborts
    if its projected spend passes $10.

## 10. Not claimed

*What this section does: states the limits of the result.*

- Scope: one NLA (nla-rl iter_000400), one layer (L42), one prompt family (multihop, final token), 87 items, and
  explanations judged by two particular LLMs.
- "Contradicted" means contradicted by the prompt, the model's own continuation or world knowledge, as these judges
  apply it. It does not mean every unflagged claim is true of the activation.
- The reconstructor is the NLA's own co-trained critic. A signal in false claims shows what *this* reconstructor
  uses; it doesn't show what the subject model computes.
- On these prompts the bridge is guessable from the prompt, so "unsupported but true" claims are rare. The C2 arm is
  weak here, and a family with hidden attributes would test it properly.

## 11. Amendment A1 (POST HOC; written 2026-10-07 after the smoke and before any full-run judge call; PI: "yes")

*What this section does: adds analyses that separate the two kinds of false claim the smoke revealed. The
pre-registered primary (§5–6) is unchanged and is reported as recorded.*

**Seen in the smoke** (ledger 013):
- Δ = +29.1 [9.1, 54.1], n 19, which reads "H-signal" by §6.
- But many contradicted spans are misquotes of the prompt that carry the bridge or answer (e.g. "the country that has
  the Tokyo tower [Japan]"), and deleting them removes true content.
- Pure confabulation (invented headers and framings) is a different kind.

**Additions:**
1. **Span kinds.** Judge B (`z-ai/glm-5.3-flash`) tags each of A's contradicted spans with exactly one kind. It is
   told to use the first kind that applies:
   - **misquote:** quotes or paraphrases the prompt's own content in altered wording, including versions that swap in
     an entity or answer;
   - **invented_context:** asserts text, structure or framing that isn't in the input (headers, instructions, quiz
     names, answer options, earlier questions, source documents);
   - **wrong_entity:** names an entity or answer that conflicts with the model's continuation, without misquoting;
   - **false_fact:** a false statement about the world.

   Cached and cost-logged like the other calls.
2. **Versions per kind:**
   - **C1i** = O with only the *invented_context* spans deleted; **C1ii** = O with only the *misquote* spans deleted;
   - each gets a clause-level random control matched in tokens (**Ri**, **Rii**; 5 seeds, built as R1 in §4).
3. **A sentence-level random control** for C1, C1i and C1ii (**RS1**, **RSi**, **RSii**): whole sentences from the §3
   splitter are deleted at random, matched in tokens; 5 seeds.
4. **Statistics,** each as in §5 (mean over eligible explanations, cluster bootstrap by item), with the §6 bands applied
   *descriptively*, labelled post hoc:
   - Δi = FVE(Ri) − FVE(C1i) and ΔSi = FVE(RSi) − FVE(C1i). **The answer to Camila's question is Δi:** does deleting
     pure confabulation cost more than deleting random text?
   - Δii and ΔSii likewise;
   - ΔS1 = FVE(RS1) − FVE(C1), next to the primary Δ.
   - Eligible means the explanation has ≥ 1 span of that kind and the cleaned version keeps ≥ 10 tokens.
5. **Hand checks before the full scoring:**
   - G2 continued: 10 more seeded explanations outside the smoke, for A's precision;
   - G2k: the kind tags of the smoke's 63 spans, checked by me. I report agreement; if it is below 0.80 the per-kind
     results are reported as unreliable.
6. **Spend cap:** the full run's judges stop at $10.

## 12. Amendment A2: factual rewrite against paraphrase (POST HOC; written 2026-10-07 before any rewrite call; PI: "yeah lets do that next")

*What this section does: adds the arm that matches the TA's request as written in report 33 §3.1. An LLM minimally
rewrites each explanation so it contains only factual content, the reconstructor is run again, and we check whether
the reconstruction projects more onto the J subspace and whether FVE rises. The deletion analyses (§5–6, A1) are
unchanged.*

**Why a separate arm.** Deleting false spans also deletes the true content that misquotes carry (smoke: "the country
that has the Tokyo tower [Japan]"), so deletion can't test whether cleaner text reconstructs *better*. A rewrite
corrects a misquote instead of removing it. A rewrite also rewords, so it needs a rewrite control.

**Material:** the same 522 explanations (§2). Judge A (`google/gemini-3.8-flash`) writes both rewrites; judge B
(`z-ai/glm-5.3-flash`, another family) checks them. Calls are cached and cost-logged, with temperature 0 and reasoning
"low".

**Versions:**
- **F, factual rewrite.** A rewrites O minimally so that every claim is true given the inputs (§3 definitions):
  - misquotes are corrected to the prompt's exact wording;
  - invented context and unsupported claims are removed;
  - wrong entities and answers are corrected to agree with the model's continuation;
  - true claims are kept with their wording unchanged, and nothing new is added beyond what a correction needs.
- **P, paraphrase control.** A rewrites the *same* O, keeping every claim's meaning (true or false) and correcting
  nothing. It changes about as many words as F did: the target is the word-level edit distance between O and F,
  passed in the prompt.
- B labels the sentences of F and P with the §3 labeller. Expected: F has close to no contradicted sentences, and P
  keeps about the rate O has.

**Measures** (the §5 reconstructor path):
- FVE and J share (§5 definition) of O, F and P.
- **Co-primary for this arm,** paired by explanation, cluster bootstrap by item:
  - (a) **ΔFVE = FVE(F) − FVE(P);**
  - (b) **ΔJ = share_J(F) − share_J(P).**
- Also: F − O and P − O for both measures; the edit size (words changed) of F and P; and B's contradicted-sentence
  rate for O, F and P. Everything is reported per state too.

**Readings** (bands fixed now; post hoc relative to the original spec):
- **ΔFVE:**
  - ≥ +2 with the CI above 0: "removing hallucinations improves reconstruction";
  - ≤ −2 with the CI below 0: "the reconstructor relies on hallucinated content";
  - CI inside [−2, +2]: "no difference beyond ±2 points";
  - otherwise unresolved.
- **ΔJ:** a CI excluding 0 gives "more J" or "less J"; otherwise no detected change. Reported next to the gold
  activations' own J share.

**Gates:**
- B's contradicted-sentence rate on F is ≤ 0.10, and on P is at least 0.7 × O's rate. Otherwise the rewrite didn't do
  its job, and this is reported.
- I hand-check 20 F (factual, minimal, no new information) and 10 P (claims preserved).
- F and P must parse and not be empty.

**Smoke first:** the same 20 seeded explanations as the §8 smoke. Stop and report. Then the full 522, unless the PI
says otherwise.
