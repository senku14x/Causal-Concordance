# Do interpretability readers track what the model uses? Reader agreement versus measured causal influence on Qwen3.6-27B

*Draft project spec, 2026-10-01 (revised after an external red-team review). Priors are gut estimates written before any run. Nothing here has been run yet.*

> **v2 · Unified spec, 2026-10-04.** "Nothing here has been run yet" is superseded: see *Status at a glance* below and §14. This file merges, without cutting any of the original:
> 1. the original draft (2026-10-01, Vishesh), kept verbatim;
> 2. **AMENDMENT_01** (PI decisions of 2026-10-02: conflicts C1–C23, errata E1–E4; verbatim in Appendix B);
> 3. **Cathryn Wayman's A4 spec amendment** (2026-10-03; verbatim in Appendix A) and her pilot `LittleMeHere/nla-vs-lenses` (numbers re-checked 2026-10-04);
> 4. the results of Causal-Concordance experiments 000–004 (dev split only; §14);
> 5. additions proposed in this merge.
>
> **v2 working title:** *Do interpretability readers track what the model uses, and do they read it or re-solve it? Reader agreement, reading vs re-solving, and measured causal influence on Qwen3.6-27B.*

### How to read this file (v2)

- **Original text is kept verbatim.** Statements that are now superseded stay in place; a callout directly after them says what replaces them. Where a callout and the original disagree, the callout wins. Where AMENDMENT_01 and a later callout disagree, the later callout wins.
- **Callout tags:**
  - **[A01]** AMENDMENT_01, PI decision of 2026-10-02 (conflict C*n* or erratum E*n*);
  - **[CW]** Cathryn Wayman's amendment or pilot (2026-10-03/04);
  - **[R]** forced by a result; the experiment is named (001–004, or CW-*n* for Cathryn's pilot rows);
  - **[v2]** added in this merge; a proposal until the PI accepts or strikes it.
- **Status words** on sections, questions, families and analyses:
  - **active**: in the current plan;
  - **conditional**: runs only if its stated precondition holds;
  - **deferred**: kept, not scheduled;
  - **out for now**: kept for the record by a recorded decision; may return;
  - **done** / **failed**: run; a failed gate's result is kept, never deleted.
- **Numbers** in callouts are dev-split, Qwen3.6-27B, and scoped to the family, position, layer and n given with them.
- **Experiment IDs.** Causal-Concordance experiments are `NNN` (001–004 run; 005 proposed). Cathryn's pilot rows are `CW-1` … `CW-9`, numbered as in her STATUS.md (§14.6).

### What changed in v2 (summary)

1. **Two collaborators, one spec.** Cathryn's pilot joins as the *reading vs re-solving* arm (A4 amended, new A7 and A9), with multihop and chain at L56. Vishesh keeps labels, Brew and the codebook contrast. Ownership and rules: §15.
2. **Content-level labels are no longer a gate for reader scoring.** On Brew they failed reliability (003: κ = 0.37), and fitted subspaces did no better than PCA subspaces of the same size (004). Labels become two tiers (§6.0): position-level (reliable now) and content-level (conditional).
3. **Re-solving is a fourth source of reader output**, next to shared signal, shared error and prompt priors. The NLA and the Oracle Lens are Qwen3.6-27B with a LoRA; they can rebuild the prompt from an activation and solve it [CW]. New tests: the A4 tag and the A7 battery.
4. **Cathryn's "re-derived" tag is adopted as "survives removal"**; "re-derived" now needs positive evidence (§8 A4).
5. **Planted concepts are promoted** from a positive control to a main experiment, **005**: a concept split into its J-space part and the rest, measured on every reader *and* on the model's output (§7, §8 A8).
6. **Bank updated:** multihop in for A4/A7 without labels [CW]; chain read at L56 [R, CW]; the codebook (RHUT tagged selection with the codebook consumer) is the planned matched not-needed contrast [A01 fallback]; Brew is no longer a positive control [A01 C2, R].
7. **Reader pins:** the Oracle Lens adapters match by hash; the NLA adapters do not, and one must be chosen before any joint number (§4.R) [CW].
8. **Judge validation becomes a gate**, with one shared rubric for both arms (§7).
9. **Order of work and stop rules rewritten** (§12); results to date (§14); collaboration rules (§15).

### Status at a glance (2026-10-04; dev split only; nothing on test)

| Experiment | Owner | What | Outcome |
|---|---|---|---|
| 000 setup | V | Environment, pinned hashes (29/29), reader smoke tests, shared code (78 tests), precision-regime check | done |
| 001 leverage profile | V | Whole-cell patches at every Brew cell (L40/42/44/56/60) and the chain question span; probes; off-path substitution; 10-table reader snapshot | done; revised after an external audit; probe controls repaired (A2) |
| 002 claim scoring | V | Role-aware judge on 3,930 verbalizer readouts; token-lens closed-set scores on 48 tables; prompt-only baseline | done (descriptive) |
| 003 1-D labels | V | Decorrelated content directions; swaps, nulls, necessity; 82 pairs, 37,299 edits | **reliability gate failed** (κ 0.37); labels descriptive only |
| 004 span assay | V | Fitted subspaces vs PCA and random at ":" L42, 33 pairs | done; descriptive geometry, no content-specific handle |
| CW pilot | C | Multihop (100), layer sweep, removal, swap, chain at L42 and L56, Brew (50), NLA text decomposition | done; exploratory |
| Not run | — | WSB reproduction, planted concepts (005), codebook, relational gate, DAS, agreement atlas, ensembles, A5, any test-split result | — |

---

## 0. One paragraph

Run every available activation reader (J, R, logit lens, Oracle Lens, NLA, Skip-lens, SAE, plus optional specialists) on the **same cells** of Qwen3.6-27B. Do two things with the outputs.

> **[A01 C16] Skip-lens dropped** (trained at L62; reinstatable on-contract). **SAE / BSF deferred**: no labelled SAE exists for Qwen3.6-27B; the candidate `ceselder/qwen36-27b-bsf-l42-1b` has no auto-interpretation labels [A01, CW]. Optional specialists: modulation lens and phrase-J dropped, template lens deferred (PI, 2026-10-02). **Same cells** stays the rule for every comparison; Cathryn's off-layer runs (L16/28/36/56) are exploratory and reported separately (§5 Layers).

- **Agreement.** Measure how much the readers agree, at four levels (concept, open-set claims, activation-space direction, behavioural effect).
- **Adjudication.** Score each reader against a reference that uses **no reader**: whether the content it names has *demonstrated causal influence* on a specified answer at that cell, is *decodable but not needed* there, is *absent*, or *cannot be resolved* by these assays.

> **[R, 003–004] Status.** No content-level reference has passed reliability on Brew: 1-D labels gave κ = 0.37 between disjoint fits, and fitted subspaces recovered no more than PCA subspaces of the same rank and edit size. Adjudication now runs at two tiers (§6.0): **position-level** (does a whole-cell patch at this position move the answer; reliable) and **content-level** (conditional). Reader comparison against task truth and baselines no longer waits for content labels.

The causal labels then split agreement into three parts: shared signal (both see real content), shared error (both fail the same way), and shared prompt inference (both guess from the text). This turns "how much do the readers agree" into "which readers' claims match content with measured causal influence on the answer, and what is their agreement evidence of". It also tests Nanda's two hypotheses directly: that NLAs and SAEs pick up non-workspace content, and that majority agreement can serve as ground truth.

> **[CW] A fourth source: shared re-solving.** The NLA and the Oracle Lens are Qwen3.6-27B with a LoRA. Both can recover the prompt from the activation and solve the task, so their agreement can come from making the same re-solving move rather than from reading the same content (CW-1: on 100 multihop items at L42 the two prose readers name the bridge without J-lens in 36). Agreement is decomposed into shared signal, shared error, shared prompt inference **and shared re-solving**; A4 and A7 separate re-solving from reading.
>
> **[v2]** Nanda's first hypothesis (NLAs and SAEs pick up non-workspace content) is tested most directly by the planted J / non-J split (005, A8) and the removal tests (A4, A7). W (A2) only says whether a reader prefers content the current answer depends on.

**What this does not give:** a lens-free definition of "what is in the workspace". A variable can be available for flexible use and still not drive the one answer measured here. The labels measure current-task causal influence, not workspace membership. The pattern of which readers see that content still bears on whether J-space is an output interface or a working store.

> **[CW]** Cathryn's pilot excludes "whether the model uses the content" (this project's labels cover that). It tests whether a reader *read* the content or *re-solved* it, which the labels cannot. The two are complementary: content can be used without any reader reading it, and read without being used.

**Terminology.** Throughout, "used" is shorthand for *a reader-blind edit at this cell demonstrably moves this specified answer* (section 6). It does not mean the unmodified model naturally routes through that direction; a necessity check gives a stronger tier. "Decodable, not needed" means decodable here, while patching the whole cell barely moves the answer.

> **[v2] Added terms.**
> - **Read**: the reader's claim depends on the content being present in the activation it was given.
> - **Re-solved** (re-derived): the reader rebuilt enough of the prompt to compute the content itself; the claim does not depend on the content's own presence.
> - **Survives removal**: the claim still appears after the content's directions are removed from the reader's input. Operational; does not by itself decide read vs re-solved (§8 A4).
> - **Position-level influence**: a whole-cell patch at this position moves this answer (S ≥ τ), or does not (|S| < ε′) [A01 E1].
> - **Content-level influence**: an edit restricted to a content direction or subspace moves this answer, beyond random and PCA controls of the same rank and edit size (§6 step 8).
> - **Mention vs claim**: a mention is the word appearing anywhere in a readout; a claim puts it in a role (002).

---

## 1. Why do this

**1. Agreement is not truth.** Nanda proposes rating techniques by agreement with the majority. That is only valid if readers fail independently, and ours don't:
- J and R share an estimator.
- Oracle and NLA share a verbalizer + reconstructor lineage.
- J, R and the logit lens share the unembedding. SAEs and the text readers do not, though they can still have correlated errors.
- Most can see prompt-inferable content.

Readers with shared errors agree whether or not they are right. Nobody has checked how much of the observed agreement is signal.

> **[CW]** The Oracle Lens and the NLA share more than lineage: both are the subject model plus a LoRA, so both can re-solve. CW-1 agreement on 100 multihop items at L42: all three readers name the bridge in 36; Oracle + NLA without J in 36; none in 16; Oracle only 6; NLA only 3; J + Oracle 2; J only 1. When J names it, the Oracle does in 38/39 and the NLA in 36/39.
>
> **[R, 002]** On 10 Brew tables, the NLA (L42) and the Oracle (L40) both mention the intermediate in 6, both state it in its role in 0 (1 with the Oracle at L44), and assert the same wrong intermediate in 3–4.

**2. WorkspaceBench measures readout, not use.** On many families the prompt-only baseline matches or beats every reader. An item passes if any (layer, position) cell passes. And a 250-token NLA readout gets more chances to be right *and* wrong than a top-10 token list. Its authors state it has no ground truth for workspace contents.

> **[R, 001–002]** On Brew, prompt-only (stock Qwen3.6-27B, asked directly) names the intermediate for 231/231 dev renders. WSB's J ≈ .86 on Brew uses the any-cell rule; per cell, J_NP's top-10 at ":" L40–44 rarely lists any palette colour, while its candidate *scores* rank the intermediate first in 0.39–0.45 of 48 tables (shuffled ≈ 0.17).
>
> **[CW]** Same pattern on multihop: J's top-10 names the bridge in 39/100 at L42, but in 76/100 at one or more of 11 layers.

**3. Readable and used are different variables.** Prior work (RHUT, below) found concepts readable at copy positions with ~0 causal leverage. At the decision cell, the causally used part of J was mostly answer geometry. So a reader can fire on readable-but-unused content without hallucinating and without reading the workspace. No current evaluation separates these cases.

> **[R, 001–003] Replicated on WorkspaceBench Brew.** At L40–44 the intermediate is decodable (probe balanced accuracy ≈ 0.8, chance 0.08) both at the ":" cell (whole-cell share 0.35–0.50) and at the newline after `</think>` (share ≈ 0). J-lens and the NLA report the intermediate at least as often at the no-influence cell as at the influential one (002). Scope: Brew dev; 82 pairs (patching), 48 tables (lenses), 10 tables (verbalizers).

**4. It bears on the workspace, within limits.** The labels measure causal influence on a specified answer, with no lens involved. That is not workspace membership: content can be available for flexible use and still not drive the measured answer. Within that limit, each reader's hit rate on used vs decodable-not-needed content shows what that reader reads. For J, this is evidence on the two competing readings:
- **Store:** high recall of used content, low firing on decodable-not-needed content.
- **Interface:** tracks answer geometry and reportable labels, misses used content that isn't about to be said.

**5. It returns a result under every outcome.** Every branch produces a map: which reader, on which content type, at which cell, agrees with which other reader, and whether it sees used content.

> **[R]** So far it has: 001–004 produced a position × layer map of influence and decodability and a reader snapshot, although content labels failed.

**6. [CW] Text readers can re-solve, so activation-sensitivity is not reading.** A reader that recovers the prompt from the activation can solve the task and name a hidden step that is not there. Evidence (multihop, word match): the prose readers name the bridge at L28 in 26–27/50 items where J-lens finds it once (CW-2), and 20–23 of 33–34 mentions survive removal of the bridge's J-neighbourhood (CW-4). Counter-case: on chain at L56 the Oracle names the intermediate in 36/60 (chance 12), 31 of them without restating the start number (CW-7).

**7. [v2] What a token reader displays is not all its scores contain.** Comparisons on top-k displays undercount token readers by construction (point 2 callout). Token readers are always scored on candidate scores or ranks as well (§7).

---

## 2. North Star, proxy, validity, divergence

- **North Star.** Know which interpretability readers can be trusted to report content the model's computation depends on, measured independently of any reader.
- **Proxy.** On families with token-aligned counterfactuals, at matched cells and matched claim budgets, measure each reader's:
  - sensitivity to used content;
  - hit rate on decodable-but-not-needed content;
  - agreement structure after conditioning on the causal label.

> **[R, v2]** Proxy targets, in order of availability: task truth (always); position-level influence (reliable now); content-level labels (conditional, §6.0). Added proxies: **survival of each reader's claims under content removal** (A4, A7) and **detection of planted J / non-J parts** (005, A8).

- **Validity argument.** A reader that fires on used and not on not-needed content is reporting what the answer depends on, not just what is present. Agreement that survives conditioning on truth is shared error. Agreement that survives shuffled activations is prior-driven.
- **Divergence risks and mitigations.**
  - "Used" is relative to the consumers we measure. *Mitigation:* multiple consumers where the organism has them; label per consumer.
  - The label estimator (a mean-difference direction) could miss content in other formats. *Mitigation:* "not needed" requires a small full-cell patch, not a failed 1-D swap; cells that fit neither label are *unresolved*; DAS labels on a subset, with agreement between the two label methods reported.
  - A 1-D edit that moves the answer may open a route the unmodified model rarely uses (the subspace-patching illusion). *Mitigation:* a necessity (ablation) check alongside the swap; "used" is reported as demonstrated influence, with necessity as a stronger tier.
  - Used and not-needed cases could differ by family and position rather than by use. *Mitigation:* matched contrasts within family × position (section 5), reported before any pooling.
  - Labels exist only for families with counterfactuals. *Mitigation:* the safety families get the agreement analyses but stay descriptive.

> **[CW]** *Text readers re-solve.* A reader can depend on the activation without reading the content. *Mitigation:* removal and swap tests, restatement checks, layer sweeps with a probe, prompt-recovery scores (A4, A7).
>
> **[v2]** *Readers off their training layer.* Inputs from other layers are off-distribution and may push text readers toward prompt recovery. *Mitigation:* on-contract layers are primary; off-layer runs are exploratory, reported separately, with their own baselines.
>
> **[v2]** *Display vs scores.* *Mitigation:* score token readers on candidate ranks as well as displays (§7).
>
> **[R, 003–004]** *The label method can fail outright.* It did on Brew. *Mitigation:* two tiers (§6.0); the DAS decision (§6 step 9); failure reported as a finding.

---

## 3. Questions

**Agreement**
- **Q1.** How much do readers agree, at the concept, open-set, vector and behavioural levels?
- **Q2.** How does agreement vary with family, content type, layer, position and prompt-inferability?
- **Q3.** How much agreement comes from the activation, and how much from priors or the prompt?
- **Q4.** Which reader pairs share errors beyond what their accuracies predict?
- **Q5.** Does agreement predict causal influence? Is P(used | k reader families agree) steep or flat in k?

> **[v2] Status.** Q1 active (concept and open-set levels; vector and behavioural levels deferred). Q2 active. Q3 active, extended to re-solving [CW]. Q4 active against task truth and position-level labels; within-content-label excess is conditional. Q5 conditional on content labels; its position-level version is active.

**Adjudication**
- **Q6.** For each reader, what is P(fires | used) vs P(fires | decodable, not needed), within matched family × position? Is NLA/SAE-only content concentrated in the second?
- **Q7.** Does an ensemble predict causal use better than the best single reader? Which readers add information given the others?
- **Q8.** When a text reader surfaces something J misses, what evidence attaches to it: prompt-predictable, activation-sensitive, behaviour-linked, contradicted by context? (Overlapping tags, not a partition.)
- **Q9.** Why do NLAs hallucinate? Is it reconstructor blindness to true-vs-false minimal pairs, or verbalizer priors? Does best-of-N selection by reconstruction actually reduce hallucination?
- **Q10.** Can any reader tell the option the model endorses from one it merely considers?

> **[v2] Status.** Q6 conditional at content level; position-level descriptive version active (partly answered on Brew, §1 point 3 callout); within-cell matched contrasts via off-path colours and the codebook. Q7 conditional; retargeted to task truth and position-level influence (A3). Q8 active, extended by the survives-removal tag (A4) [CW]. Q9 deferred (low priority; the NLA reconstructor is available). Q10 out for now (commitment family out, A01 C1).
>
> **[CW, v2] New questions.**
> - **Q11.** When a text reader names a hidden step that J-lens does not, did it **read** the step from the activation or **re-solve** it from the prompt it decodes? [CW]
> - **Q12.** Does NLA-only content lie **outside J-space**? When a concept is split into its J-space part and the rest and planted, which readers report which part, and which part changes the model's output? [CW, v2]
> - **Q13.** Per item, does how much of the prompt a readout recovers predict re-solving? [CW]
> - **Q14.** Do readers report content at positions where the answer does not depend on it (position-level W)? [v2; partial answer in §1 point 3 callout]
>
> **[CW] Cathryn's hypotheses** (pilot spec, 2026-10-03), mapped to the questions above:
> - **H1.** Prose readers name hidden steps the activation does not hold, by re-solving (Q11). *Status:* supported on multihop; swap test weak.
> - **H2.** Where the hidden step is linearly present, the Oracle Lens reads it (Q11). *Status:* supported on chain at L56.
> - **H3.** Most NLA text beyond the hidden step is prompt paraphrase, format description or false (A9). *Status:* supported, with an unvalidated judge.
>
> **[CW] Joint framing (shared doc, 2026-10-04).** When the NLA or the Oracle says something J-lens doesn't, where does it come from? (a) real signal the model uses; (b) readable but not used; (c) worked out from the prompt; (d) made up.
>
> **[v2]** (a) needs two tests, not one: evidence that a reader *read* content is a reading test (A4, A7); "the model uses it" additionally needs position- or content-level influence (§6). §14.7 sorts current results under this framing.

---

## 4. Model and readers

**Model.** Qwen3.6-27B. Every meta-model exists for it, and an fp32 intervention harness already runs on it. Labels are residual-cell interventions, so no attention cuts are needed and the GDN state doesn't break them.

> **[A01 C8, E4; R, 000] Precision regime.** Every label quantity at L ≥ 40 uses SEC's true-float32 replay (stock bf16 blocks 0–39; float32 blocks 40–63 and W_U; noise floor ≈ 2e-4 nats, ≈ 1e-5 of the typical gap). RHUT's float32-residual regime is the cross-check (3.3% of G vs SEC). In plain bf16, any perturbation moves the margin by up to ~0.2 nats. Margins come from float32 logits; every clean/edited comparison is made at one sequence length; reader inputs come from the stock bf16 forward (WSB convention).
>
> **[CW]** The pilot ran bf16 throughout, which matches the reader-input convention; its lens scores are bf16 (§7 differences).

**Reader arms.**
- **Token readers:**
  - logit lens;
  - J_NP (n=1000), in both WorkspaceBench's cosine readout and the paper's softmax(W_U·norm(Jh));
  - J_CB and R_CB (n=25, matched to each other).
- **Oracle Lens RL** (`agu18dec/olens_and_ar`, `olens_s3d_rl600`) with its AR reconstructor.
- **NLA RL** (Celeste's L42 model), with its reconstructor if released.
- **Skip-lens** (ceselder; fed raw L42).
- **SAE / BSF at L42.** Top features by activation, labelled by auto-interpretation. Their decoders are also read through J as a check.
- **Optional:** modulation lens (specialist), template lens, phrase-J (if a checkpoint exists).

> **[A01, PI 2026-10-02] Reader status.**
> - **Token readers: active, done.** J_NP in both readouts [A01 §4]. **[v2]** Never compare token readers on their top-k display alone; always also on candidate scores/ranks (§7).
> - **Oracle Lens RL: active.** On-contract at **L40 and L44**, labelled there; an off-contract L42 arm is reported separately and never pooled [A01 C6]. Its AR is limited: the trained whiteners are private [A01 §4].
>   **[CW]** The pilot ran the WSB default `agu18dec/local-workspace` `s3d.ddp600.s0/iter_000600` at L42. Its adapter file hash equals the registry's `agu18dec/olens_and_ar` `olens_s3d_rl600`: same weights, nothing to reconcile.
> - **NLA RL: active.** Step-matched `ceselder/qwen3.6-27b-nla-L42` (`av_sft_lora` merged + `av_rl_lora_step400`; reconstructor `rl_critic_step400`) [A01 C17].
>   **[CW]** The pilot used `ceselder/qwen3.6-27b-nla-rl`, adapter `iter_000400`, loaded through WSB's NLA class. Adapter sizes 1.27 GB (registry) vs 1.87 GB (pilot); file hashes differ. **Reconcile before any joint number (§4.R).**
> - **Skip-lens: dropped** [A01 C16].
> - **SAE / BSF at L42: deferred** [PI]. **[CW]** No labelled SAE exists for Qwen3.6-27B. Gemma 3 has Gemma Scope 2 plus public NLAs and J-lenses: a candidate second model if the SAE arm is ever needed.
> - **Optional:** modulation lens and phrase-J **dropped**; template lens **deferred** (its vocabulary lacks "teal").

**Baselines.**
- Lucky guess.
- Prompt-only: an LLM reads the prefix and guesses.
- For every learned reader: a **no-activation** run (mean activation injected) and a **shuffled-activation** run (activation from another item in the same family).

> **[A01 C18]** Prompt-only = stock Qwen3.6-27B, asked directly (as in WSB). **[R, 002]** Done on Brew: intermediate 1.00, start 1.00, answer 0.89 (231 dev renders). Lucky guess: not started.
>
> **[CW]** Pilot chance = the same readout scored against other items' hidden steps (in expectation, a shuffled-activation arm). The pilot did not run no-activation or prompt-only arms; they are added when its kept results are rerun under §7.
>
> **[CW, v2] New controls for reading vs re-solving:**
> - **Content removal** from the reader's input, with a control removing the same number of random directions. Two removals: J-based (the content tokens' J directions and their cos > 0.8 neighbourhood) **and** reader-free (LEACE-style erasure fitted on dev items; on Brew, 004's fitted content subspace).
> - **Swap**: remove own content, add another item's, at a strength J-lens can see.
> - **Restatement check**: does the readout restate the question or input before naming the content?
> - **Prompt-recovery score**: per readout, how much of the prompt it reconstructs (Q13).
> - **Real shuffled activations** alongside other-item scoring.

**Stochastic readers.** Take 8 samples per cell. Self-consistency (the fraction of samples naming z) is kept as its own variable.

> **[R, 001–002]** Run so far: 4 samples per cell (Oracle contract, NLA); 1 for the Oracle WSB grid. Two NLA readouts from nearly identical activations (stock vs SEC capture, cos ≥ 0.99985) agree on the intermediate only 48% of the time, so per-claim tags from one sample are noisy.
>
> **[CW]** Pilot: 1 NLA sample per cell; the Oracle one readout of 4 bullets.
>
> **[v2]** Budget for joint numbers: NLA ≥ 4 samples per cell (target 8), Oracle k = 4 per its contract; self-consistency reported; every survives-removal comparison is a rate over samples, not a single readout.

### 4.R Reconciliation before any joint number (v2; items from [CW])

| Item | Finding | Action |
|---|---|---|
| NLA checkpoint | Registry `nla-L42 av_rl_lora_step400` (1.27 GB) vs pilot `nla-rl iter_000400` (1.87 GB); hashes differ [CW] | **[v2]** Run both on the same 10 items with fixed seeds at L42; compare readouts and reconstruction quality with the step-matched reconstructor; check which one WSB's published NLA numbers used; pick one and record it in the registry with its hash. No NLA numbers are pooled until then. |
| Oracle Lens | Same adapter hash in both repos [CW] | None. |
| Split | The pilot's Brew run used the first 50 bank items: 25 dev and 25 test by the seeded split; readouts only, no labels, no tuning [CW] | **[v2]** The 25 test-item readouts are quarantined: they inform no design choice (judge rubric, thresholds, claim extraction). Joint numbers are dev-only until the test split is opened. |
| Chain items | The pilot's 60 items include multi-digit intermediates outside the label bank [CW] | **[v2]** Joint chain numbers use the single-token (single-digit) subset, per A01 §1. The full 60 stay a separately reported exploratory set. |
| Lens precision | Pilot lens scores are bf16 (after a memory patch to WSB's J/R code) [CW] | **[v2]** Rerun joint lens scores in float32, or show on a subset that rankings agree. |
| Judges | 002: Gemini 3.8 Flash, closed candidates. Pilot: prompt-blind role judge, Claude and Gemini (agree on 34/50) | **[v2]** One rubric, one validated judge (§7 gate). |

---

## 5. The bank

**Why counterfactual families.** A label requires an intervention whose correct outcome is known in advance. Token-aligned pairs give that: same template, one latent changed, known answer on both sides.

| Family | Source | Latent z | Role |
|---|---|---|---|
| Brew | WorkspaceBench, 100 items, token-aligned donors | intermediate colour | J reads it well (.86); positive-control family |
| Two-hop | 42 RHUT items over 7 relation types (12 released probe-swap + 30 constructed; RHUT admitted 35 of them **using a J_NP coordinate swap**, so they must be re-admitted with the reader-blind swap before J is ranked on them) + filtered WorkspaceBench multihop, ~100 total | bridging entity | classic intermediate; answer-geometry risk controlled by admission |
| Chain / arithmetic | WorkspaceBench chain (120), computable counterfactual answers | numeric value | value content; J expected weak |
| Role-bound / relational | WorkspaceBench ab/ba pairs | who did what to whom | binding; tests whether single-cell labels exist at all |
| Commitment | new, ~60 pairs | endorsed vs rejected option | readers expected to disagree; direct audit relevance |
| Decodable, not needed | RHUT organisms: tagged selection (A.2), computed sums with report + parity consumers (A.4–A.5), plus absent-source variants as a flip | held copy of z | the only family built to contain decodable-not-needed content (Q6). Seeds: sum carrier reaches 1.04 of source readout at 0.08 share; word carriers ≤ 0.04. Two-hop carrier (0.27) is mid-range, so it is labelled by threshold like any other cell and not assumed unused |
| Descriptive only | WorkspaceBench agentic misalignment, jailbreak, user modeling | none | agreement analyses only; no labels |

> **[A01 C1–C4; R; CW] Family status (2026-10-04).** The table above is the original plan; this is the current one.
>
> | Family | Status | Notes |
> |---|---|---|
> | Brew | **core; done (dev)** | Partners built (361 candidates) and gated: dev 82/183 pairs (33 tables), test 77/178 (35 tables). **Not** the pipeline's positive control [A01 C2]; that is the start-colour token at early layers and the stir cells. The intermediate is guessable from the prompt (prompt-only 1.00) and is often the model's runner-up answer (22 of 41 wrong gate samples answered z). Content labels failed (003–004). Role now: position-level contrasts, reader scoring, the off-path within-cell contrast. |
> | Two-hop (RHUT) | **out for now** [A01 C1] | Kept for the record; needs reader-blind re-admission if it returns. |
> | Multihop (WSB) | **active for A4 / A7 only** [CW] | 100 items, plain render, read at the last word; no partner prompts, so no labels; never pooled with RHUT two-hop [A01]. Cue-swap pairs would be needed for labels. **[v2]** Needs a dev/test split before any reported number. |
> | Chain | **core; done (dev)** | Single-digit intermediates only [A01]: dev 48/53 pairs, test 37/46. **[R, 001]** No single cell carries leverage at L40–44; the final cell reaches 0.66 at L56. **[CW]** J-lens finds an intermediate only around L56 (0 at L20–36, 6–8 at L40–52, 25 at L56, 6 at L60). Chain work is read at **L56**. |
> | Relational | **core subset; built, not gated** | 50 token-aligned a/b pairs. **[v2]** Gate them or record a decision to drop; demote to secondary if fewer than ~30 survive [A01]. |
> | Role-bound | **agreement only** | Nothing built. |
> | Commitment | **out for now** [A01 C1] | Q10 and A6 wait on it. |
> | Decodable-not-needed (RHUT organisms) | **out for now as a family** [A01 C1] | Replaced by Brew off-trajectory colours [A01 C4]: weakly decodable at ":" L40–44 (membership AUC 0.61–0.64), better at L60 (0.79); swapping one out keeps the answer within bound in 39/43. The **codebook** (next row) is the planned matched contrast. |
> | **Codebook (new)** | **planned; owner V; design not written** | RHUT tagged selection with the codebook consumer: pointed vs unpointed word at the same question/decision cells [A01 §2 fallback]. A simplified version of the PI's application project. This is the matched not-needed contrast §5 requires. |
> | Safety families | **out for now** [A01 C1] | Descriptive agreement only, if they return. |

**Commitment family construction.**
- Paired diagnosis/debugging cases that mention the same two options A and B.
- One token-aligned piece of evidence flips which option is correct.
- Forced identical neutral text ("Let me look at this.") before the choice.
- Readouts are taken on the neutral text.
- Mention-only control: A and B named with no evidence.

**What RHUT is.** RHUT ("Readable Here, Used There") is a prior project on Qwen3.6-27B, with Qwen3-32B as a dense control. Its prompts ask the model to keep something in mind (a word, a sum, a two-hop answer) while copying text, then ask a question about it. What it found, scoped to those tasks:
- **Readability and causal leverage come apart by position.** The same variable is J-readable at the source, carrier and question positions, but patching those positions has very different effects. On sums, the carrier reaches 1.04 of the source's J readout but carries 0.08 of the source patch's answer effect. In the paper-style swap, clue / carrier / question positions give 0.09 / 0.07 / 0.20 of the ceiling, so a global swap working does not say which readable occurrence did the work.
- **Low leverage is not "unused".** With the source hidden, the same carrier states move the answer (sums 28/32 vs 4/32; two-hop 28–32/40 vs a 20/40 floor). The carrier is usable. What is not shown is whether it is on the model's natural route.
- **Even at one position, the J-readable part is incomplete** (frozen question-turn ladder, `extension` branch, 35 admitted items, six relation types, float32 writes). Writing the donor's question-turn state reaches 0.46 of the natural clue-swap effect. Of that, the best 25 vocabulary-J directions install about 0.39, against 0.04 for rank- and norm-matched random directions and 0.13 for 25 fitted functional-Jacobian directions. But the answer's own two-direction plane alone installs 0.37 (the intermediate's 0.19), so a privilege for non-answer vocabulary directions is not established. With the tested J-derived span held clean at every scored answer position, 0.41 still arrives, almost all through the last layer's read of the question turn. The tested span is a pursuit-selected dictionary, so none of this shows the remaining state is unreadable to other bases. (The earlier 12-item stage gave the intermediate plane 0.20 and complement 0.78; removing the plane cut flips from 18/48 to 1/48.)
- **The instruction selects; the fitted maintain-minus-mention direction does not.** Tagged selection is real (+4.10, 8/8), but the fitted direction behaves as a gain on readout that is already there (r up to 0.999), not as a controller.
- **Limits.** Small sets (8 sums, 32 words; two-hop 12 items for the carrier results and 35 for the question-turn ladder), teacher-forced carriers, thinking disabled, one question wording. The registered label holds on Qwen3.6-27B only; Qwen3-32B is directionally consistent but unlabelled under the stop rule.

> **[A01 C23]** A factual correction to the question-turn ladder figures above is recorded in the reading report (§4.1.4 / §5.10); no design consequence yet.

**RHUT organisms: what they add and their limits.**
- Maintain-then-ask prompts *do* have a downstream consumer (the later question turn), unlike a plain copy-task instruction. The carrier cell is the cleanest known "readable here, low leverage here" state.
- "Unused" here means *low leverage while the source is visible*. The source route may simply override a donor carrier; RHUT's own next-experiment 1 flags this conflict. The label is honest only as "decodable, not needed given the source".
- Tagged selection: all three words are in the prompt and only the tag token differs. At the carrier, the pointed word is the target and the two unpointed words are present-but-unselected. This is a direct test of whether a reader names the selected item or just whatever is in context.
- Sums: the answer word never appears in the prompt. The span is token-aligned (5 tokens), and two consumers read the same carrier. Only 8 rows exist; extend over the single-token number words two…twenty before use.
- Categories (A.3): no category member appears in the prompt. Use it as an open-set check that readers name members, not the category token.

> **[A01 C1, C4]** RHUT organisms are out for now as a family. Tagged selection returns only as the **codebook** contrast (§5 status table).

**Same activation, different label (a ceiling on activation-only readers).** Install the source-visible run's carrier states into the source-hidden prompt, where they become used (sums 28/32 vs 4/32). The activation at the carrier is identical in both contexts, so any reader that sees only that activation gives the same output in both. Report this pair separately. It shows that "used" is a property of activation-in-context, and it bounds how high W can go at carrier cells. It is not a reader failure.

> **[v2] Status: conditional** on the codebook or the RHUT sum organism being built.

**Matched used / not-needed contrasts.** Without these, W could measure "does this reader like decision cells" instead of "does it track use". Attention only looks backwards, so the activation at a cell cannot depend on which question comes later. A matched pair at the same position must therefore differ in something *before* the cell.
- RHUT maintain / mention / ignore conditions: same word, same carrier position, different instruction, same question consumer.
- Tagged selection: pointed vs unpointed word, both present at the same carrier position.
- Report W within each family × position first; pool only through the fixed-effects model.

> **[R, v2]** On Brew, ":" vs the newline after `</think>` compares different positions, so by the argument above it is **descriptive, not a W**. Within-cell contrasts available now: the intermediate vs off-path colours at the same cell (weakly decodable at L40–44; §5 status table). Later: the codebook's pointed vs unpointed word.
>
> **[A01 §5]** A power calculation with a pre-registered minimum number of cells per stratum, below which W is not reported.

**Admission rules.**
- The model is correct ≥ 0.9 on both members of a pair, without chain of thought.
- Family-specific copy check: the *scored distinction* cannot be obtained by copying from the prompt. Brew, tagged selection and commitment deliberately name the candidates; what must not be copyable is which candidate is the answer. The model's own output before the read cell never states the latent.
- Tokens are aligned at the labelled cells.
- Two-hop only: the intermediate swap must beat the answer swap at mid-depth. The swap used here is the reader-blind mean-difference swap, never a J-coordinate swap, because J is later ranked on this bank. The rule is applied on dev items only, the number rejected is reported, and two-hop results are scoped to this assay-positive set.

> **[A01; R, 001]** Gate as run: both members greedy-correct and ≥ 9/10 correct at T = 0.8 (top-p 0.95, 16 tokens, EOS suppressed). Report candidates built, candidates passing, and the difficulty of the survivors. Token alignment is asserted on every render.

**Cells.** Source (where z becomes available), carriers, decision (last prompt token).

> **[A01 C5]** Unit = one token position at one layer; span patches are context only and never label anything. Brew's read cells are the three emission cells: the newline after `assistant`, the `\n\n` after `</think>`, and the ":" after "Answer" (24 pinned WSB positions in all). Chain: the final token plus the question span.
>
> **[CW]** The pilot reads the bank's read token (Brew ":" after "Answer"; multihop the last word; chain the last token).

**Layers.**
- Core matched layer is **L42**: the NLA and SAE exist only there, and Skip-lens is fed L42.
- Labels are computed at L42.
- J, R and logit lens are also read at 40, 48 and 56, each against labels at that layer.
- Report per layer. Never use the any-cell rule.

> **[A01 C6]** The Oracle is on-contract at L40/L44 with labels there; its off-contract L42 arm is separate.
>
> **[R]** Patches and labels were computed at L40/42/44 (+56/60 in 001). Chain belongs at **L56** (§5 status table).
>
> **[CW]** Text readers were also run at L16/28/36/56. These are **exploratory**: off the NLA's training layer (L42), reported separately, with their own baselines (§2).

**Splits.**
- Templates A/B: label directions are fitted on A, labels computed on B.
- Items dev/test: every scoring choice (thresholds, summariser prompt, claim extraction) is tuned on dev and reported on test.

> **[A01 C7]** WSB families have one template, so "halves" are item halves, cross-fitted A→B and B→A: Brew by table (stratified by z), chain by start number, relational by name set.
>
> **[R, 003]** 003 fitted directions on 3 table folds, labelling each pair with the two folds that exclude its table; reliability compared single folds. Dev/test: Brew by table, chain by step-sequence group (identical prompts recur across items), relational by name set.
>
> **[CW, v2]** Quarantine rule for collaborator readouts on test items (§4.R); multihop needs its own split.

---

## 6. Causal labels (no reader involved)

For item i with counterfactual partner i′ (latent z vs z′), cell c, layer ℓ:

> **§6.0 [v2; R, 003–004] Two tiers of labels.**
> - **Tier 1, position-level (active; reliable).** Whole-cell patch share S at (position, layer): *influence* if S ≥ τ and above the null; *no influence* if |S| < ε′ [A01 E1]; otherwise intermediate. This labels a position, not a content; the whole-cell patch is a reference, not a ceiling [A01 E2]. It was stable on Brew (001: toward the partner in every pair at ":" L42–44; reverse vs forward direction r = 0.84).
> - **Tier 2, content-level (conditional).** Steps 1–9 below. A content label is reported only if it passes reliability (≥ 0.7 between disjoint fits) **and** beats a generic subspace of the same rank and edit size (step 8). Reader scoring never waits on tier 2.

1. **Reader-blind direction (fixed contrast).** Class centroids on template half A:
   $\mu_z=\mathbb E_{x\in A,\;z(x)=z}\,[h_{\ell,c}(x)]$, and one direction per contrast, $\hat u_{zz'}=\operatorname{unit}(\mu_{z'}-\mu_z)$. Contrasts are never averaged across different (z, z′) pairs, which would cancel.

> **[A01 C15; R, 003]** The label is renamed **"pair difference used"**, with a four-way {input, z, answer, other} attribution. 003 fitted decorrelated directions by least squares, H = b₀ + Σ onehot(colour in role)·B_role (role ∈ {start, z, answer}), on two of three table folds. The naive centroid direction was nearly identical (cos 0.94–0.95) and gave the same S. Held-out AUC for z contrasts: 1.00 at the emission cells.

2. **Swap.** On B items, set the coordinate along $\hat u$ to the partner's value, with magnitude set by the donor:
   $h'=h+\big(\hat u\!\cdot\!h^{(i')}-\hat u\!\cdot\!h^{(i)}\big)\,\hat u$, then run the rest of the model in fp32.

> **[A01 C8, E4]** "The rest of the model in fp32" = SEC's true-float32 replay for L ≥ 40. Patch sources are captured at the matching sequence length, per scored continuation.

3. **Effect.** Let m be the full-sequence log-prob margin, log-sum-exp over accepted spellings (never first-token scoring; in RHUT, first-token scoring counted prefixes like "R", "Tok" as answers).
   - Primary: the signed margin change $\Delta m_i=m(h')-m(h)$, in nats.
   - Normalised share: $S_i=\Delta m_i/G_i$, where $G_i=m(x_{i'})-m(x_i)$ is the whole-run counterfactual gap. Admission makes $G_i$ large and stable, because both pair members are answered correctly. It is never a cell-level patch, which is small by design at carrier cells.
   - Cell ceiling: the full-cell patch effect $\Delta m^{\text{cell}}_i$ (the partner's entire residual at c), recorded for every cell.
   - The edit has *demonstrated influence* if $S_i$ exceeds the 95th percentile of shares from norm-matched random directions.

> **[A01 C9, E1]** Used = S ≥ τ (0.10, provisional) **and** above the norm-matched random-direction null after BH-FDR. Small effects are magnitude tests (|S| < ε′); toward / away / negligible / intermediate are kept separate.
>
> **[R, 003]** At ":" L40–44, median S_z 0.03–0.05 (whole cell 0.35–0.50); "used" in 11–18% of pairs.

4. **Necessity (stronger tier).** Ablate $\hat u$ at c in both x and x′, setting the coordinate to the pooled mean, and measure how much the counterfactual gap shrinks: $N_i=1-G_i^{\text{abl}}/G_i$. Necessity can fail because other cells carry z (redundancy), so it is reported as a tier, not required for "used".

> **[A01 C10]** Pooled mean = the fitting-half mean of û·h at the cell, plus a random-direction ablation control; "used + necessary" requires N above that control's 95th percentile.
>
> **[R, 004; v2]** Adopt 004's **paired-rotation control** (keeps both runs' edit norms and their inner product, randomizes only the orientation) as the necessity null. INLP/LEACE-style removal is the stronger tier.

5. **Decodable (item level).** Cross-fitted probe for z at c, with folds split by template. Keep each item's held-out probability $p_i$ for the true z. The item is decodable if $p_i$ exceeds the 95th percentile under shuffled-label probes. The family-level held-out accuracy is a gate, not an item label.

> **[A01 C11; R, 001 (A2 repair)]** Regularised probe; folds grouped by item and table; Hewitt–Liang control task (selectivity); full-procedure permutation nulls; per-target seeds. Binary z-vs-z′ probes allowed. **Item-level** decodability is required for item labels; the dataset-level gate is not an item label (003 assigned "decodable, not needed" from dataset-level decodability, an audit-caught error).

6. **State.** Each (item, cell) gets exactly one of:
   - **used**: demonstrated influence (step 3); tier "used + necessary" if step 4 also passes;
   - **decodable, not needed**: decodable, and the *full-cell* patch share is below a pre-set ε (e.g. 0.1). A missed direction cannot explain a small full-cell patch; redundancy still can, so this means "not needed given the other routes";
   - **absent**: not decodable, the probe detects a planted copy of z at this cell at a behaviour-relevant dose (positive control), and the full-cell patch share is below ε;
   - **unresolved**: anything else, e.g. a large full-cell patch with a failed 1-D swap (wrong direction or non-1-D format, expected for role binding), or a probe failure without a passing positive control. Unresolved cells are counted and reported per family, and never scored as used or unused.

> **[A01 C12–C14, E1–E3]** Not needed: decodable and |S_cell| < ε′ = 0.05 (provisional), with ε′ < τ; the band between is unresolved. Off-path content (no pair gap) uses the behavioural + transplant bound instead (E3). Broad damage → unresolved. The planted control for "absent" must not use the probe's own direction.
>
> **[v2]** "Absent" is not assigned until planted-concept detection curves exist (005).

7. **Per consumer.** Where an organism has two consumers (e.g. report and parity on the sum organism), label each separately. "Used by one consumer, ignored by another" is itself interface-type evidence.

> **[v2]** WSB families have one consumer each; per-consumer labels apply to the codebook / RHUT organisms if built.

8. **[v2; R, 004] Subspace labels must beat generic subspaces.** Any multi-dimensional content label is compared with PCA subspaces (fitted on the same activations, evaluated table held out) and random subspaces at the **same rank and the same edit size**. Without an advantage, the result is reported as geometry, not as a content label. 004: the z span recovered 0.21 of the whole cell vs PCA 0.58 (0.29 at matched edit size); the joint span 0.72 vs 0.89 (0.80).
9. **[v2] DAS decision.** DAS is the remaining content-label method. Run it bounded (dev subset; matched rank and edit size; PCA and random comparators) **only if** its answer would change the plan. Otherwise record the decision not to run it and treat content labels on Brew as unresolved.

**Why mean-difference.**
- A full patch labels the cell, not the content. So it serves only as the ceiling and to rule influence out ("not needed", "absent"), never to label content as used.
- DAS is optimised to succeed, so it would over-label.
- Mean-difference is the simplest reader-blind estimator. DAS runs as a second label method on a subset, and agreement between the two methods is reported.

> **[R, 003]** On Brew, one mean-difference/decorrelated direction carried 3–5% of the gap where the whole cell carried 35–50%; the start-direction positive control recovered 0.076 of the start token's 0.23. The single-direction estimator is too narrow here (see steps 8–9).

**Share vs flips.** Record both. They can disagree sharply: in RHUT's 12-item question-turn stage, the intermediate's J plane carried 0.20 of the continuous effect, but removing it cut flips from 18/48 to 1/48. Primary label = share; flips reported alongside, and any cell whose label changes between the two is listed.

> **[R]** Shares recorded; flips not tabulated yet. In 002, lens "flips" were true rank reversals in only about half the cases (the difference-in-differences was positive in nearly all).

**Gates.**
- Write fidelity: every swap is measured after it is applied. ρ = ‖δ_realised‖/‖δ_requested‖ ∈ [0.9, 1.1], κ = cos(δ_realised, δ_requested) ≥ 0.99, fp32 hook. (RHUT's bf16 coordinate write realised only 0.53–0.59 of requested norm.)
- Label reliability (split-half agreement across the A/B template halves) must be ≥ 0.7 before any reader is scored.
- If more than a pre-set fraction (e.g. 40%) of a family's cells are unresolved, that family is excluded from W and reported descriptively.
- If mean-difference and DAS labels disagree badly, the labels aren't robust yet. Fix them first.

> **[R] Gate outcomes so far.**
> - Write fidelity: passed (003: 0/37,299 failures, ρ 0.9996–1.0024, κ ≥ 0.998; 004: all writes passed).
> - Label reliability: **failed** (003: κ 0.37, r 0.41 between disjoint fits at ":" L42–44; 004's joint span missed its reproducibility tolerance by 0.002).
> - Unresolved fraction: not yet computed. DAS: not run.
> - Engine checks (self-patch, embedding anchor, prefix, stir no-op): 100% bitwise.
>
> **[A01 C14]** Damage readouts on every edit (Δlog P(a), Δlog P(a′), top-1 retention, off-pair mass); 004 logs all 24 accepted colour spellings.
>
> **[v2] New gates.** (i) Judge validation before any judged number is reported (§7). (ii) Checkpoint reconciliation before any joint number (§4.R). (iii) A within-cell matched contrast exists before any W is reported.

---

## 7. Scoring readers fairly

**Matched claim budget.** Every reader emits at most k claims per cell. Primary k = 4; also report 1, 8 and 16.
- Token readers: two scorings, reported separately. (i) Raw token ranks, judge-free (the closed-set reciprocal rank below). (ii) Top-25 → summariser (frozen prompt and model, abstain option) → top k concepts. The summariser can infer and select, so it counts as part of the method: it also gets its own prompt-only arm.
- Text readers: atomic claims ranked by how often they recur across the 8 samples (ties broken by first appearance), top k. Not the first k, which can drop the most salient claims.
- SAE: top-k features.

> **[CW]** The pilot scored whole readouts (no k). **[v2]** Report whole-readout and k = 4 side by side; recurrence-ranked atomic claims need ≥ 4 samples per cell.

**Closed-set scoring (labelled families).** Each item has a candidate set: gold z plus 4 same-category distractors.
- Token readers are judge-free: reciprocal rank of z among the candidates under the reader's own scores.
- Text readers: a judge that sees only the readout and the options, never the prompt, with "cannot tell" allowed.
- Outputs per reader m: a continuous score $s_{im}$ and a binary hit $R_{im}$.

> **[A01 C19]** Candidates = gold z, the input, the answer and same-category foils. Token-reader hit = z ranks first among the candidates **and** its score exceeds the same reader's no-activation run. **[R, 002]** Done on 48 Brew dev tables; within-table start-swap tracking added post hoc.
>
> **[R, 002] Text-reader judge as run:** Gemini 3.8 Flash (WSB's judge), role extraction per readout with none / cannot tell / conflicting. Validated against blind codes from another model (κ 0.67 / 0.81), **not** human labels; the PI spot-check is open.
>
> **[CW]** The pilot used word match on multihop and chain. On Brew it used a prompt-blind role judge (readout + candidates only), run with Claude and with Gemini; the two agree on 34/50 labels per reader.
>
> **[v2] Judge gate.** One shared rubric for both arms:
> - roles: start / intermediate / answer (Brew); bridge / answer (multihop); intermediate numbers (chain);
> - plus no claim / cannot tell / conflicting;
> - the judge never sees the prompt.
>
> Validate on a human-coded set (~50 readouts per family, coded blind by both collaborators; report κ) before any judged number is reported.

**Open-set scoring (all cells).** Pairwise claim matching by a judge that sees only the two readouts, labelling each pair same / different / unclear. Used for agreement level B.

> **[v2]** Active; needs no labels (§12).

**Positive control: planted concepts.** Before touching the bank, inject $\alpha\hat u_c$ at a neutral cell and sweep α. This gives each reader's sensitivity curve against exact ground truth. A reader that cannot detect a planted concept at doses that change behaviour cannot be blamed for missing natural content.

> **[CW] Promoted to a main experiment, design 005.** Add a concept at a neutral cell, sweep the dose, every reader; split the concept into its J-space part and the rest.
>
> **[v2] Design requirements for 005** (design spec written and approved before any run, §15):
> - **Define "J-space" concretely.** The J directions of the whole vocabulary span all 5,120 dimensions, so "the J-space part" needs a stated construction: e.g. the projection onto the span of the concept's token J-directions plus their cos > 0.8 neighbourhood, or the top-k right singular subspace of J at that layer. Report results under at least two constructions.
> - **Concept vector built without any reader** [non-negotiable; A01 C13]: e.g. mean activation on dev prompts containing the concept minus matched prompts without it; never a probe's own direction.
> - **Neutral cells with measured zero leverage**: Brew stir cells, or the newline after `</think>` (001).
> - **Dose** in units of the natural activation norm at that cell; float32 writes with fidelity checks.
> - **Both sides measured:** each reader's detection rate (≥ 4 samples per dose for text readers; candidate ranks for token readers) **and** the model's output change (KL of the next-token distribution, and the task margin). Then "the NLA reports the non-J part the model ignores" can be tested directly.
> - **Layers:** each reader on its training/contract layer first; off-layer runs separately.
> - **Controls:** zero dose; a random direction of matched norm; the J part and the non-J part each at matched norm.
>
> **[CW] Where the pilot differs from §7** (from Cathryn's amendment; she will rerun what is kept under §7):
>
> | Spec | Pilot |
> |---|---|
> | 8 samples per cell | NLA 1 sample. Oracle lens 1 readout of 4 bullets. |
> | Claim budget k = 4 | Whole readout scored |
> | Closed-set judge | Word match on multihop and chain. Her own role judge on Brew. |
> | No-activation and prompt-only arms | Not run |
> | fp32 lens scores | bf16 |
>
> **[v2]** Causal-Concordance's own runs also differ from §7: 002 used 4 samples per cell, not 8, and no recurrence-ranked atomic claims. The budget for joint numbers is the one in §4 (Stochastic readers callout), applied to both arms.

---

## 8. Analyses

### A1. Agreement: levels, decomposition, predictiveness

Plain words: two readers can agree because they both see real content, because they share a flaw, or because they both infer from the prompt. Measure agreement several ways, then split it into those parts.

**Four levels of agreement.**

- **A. Concept level (labelled families).** Hits $R_{im}$. Pairwise Cohen's κ; Fleiss' κ or Krippendorff's α across all readers. The causal labels attach to this level.
- **B. Open-set level (every cell, including the safety families).** k claims per reader, matched by the pairwise judge. Also: Jaccard and rank-biased overlap of top-k token lists, and embedding similarity of claim sets. This is the no-ground-truth atlas.
- **C. Vector level.** Each reader with a write map (J/R rows, Oracle AR reconstructor, NLA reconstructor, SAE decoders) gives a reconstruction direction $\hat r_m$ from its own claims at a cell. Compare readers by $\cos(\hat r_m,\hat r_{m'})$. At the dictionary level, compare principal angles between each reader's atom subspaces on a family. This catches readers that say the same thing in different words, and readers that use the same word for different directions.
- **D. Behavioural level.** Swap along each reader's direction for z→z′ and compare which output moves. Two readers "agree causally" if their edits have the same effect.

> **[v2] Status.** Levels A and B active. Level C deferred: needs reconstructors (the NLA's is available; the Oracle's whiteners are private). Level D deferred.
>
> **[CW]** Pilot agreement table (multihop, 100 items, L42): §1 point 1 callout.

**Shared signal vs shared error (needs labels).** With causal state $y\in\{\text{used},\text{decodable-not-needed},\text{absent}\}$ (unresolved cells excluded and counted), conditional independence (the Dawid–Skene assumption behind majority voting) predicts, within each y,

$$\hat P(R_m=R_{m'}=1\mid y)=P(R_m=1\mid y)\,P(R_{m'}=1\mid y).$$

Excess agreement within y is

$$E^{(y)}_{mm'}=P_{\text{obs}}(R_m=R_{m'}=1\mid y)-P(R_m=1\mid y)\,P(R_{m'}=1\mid y).$$

Positive excess can also come from shared item difficulty or from labels that are incomplete, not only from a shared flaw. Check whether it survives adding difficulty proxies (prompt-only correctness, token length, label margin), and read it as "shared dependence" unless it is within *absent* and survives.
- Positive excess within **absent** means a shared false positive: the pair hallucinates or infers together.
- Positive excess within **used** means shared sensitivity to the same content types.

> **[v2]** Until content labels exist, y = task truth (correct / incorrect claim) and position-level influence (influence / none) stand in. Within-"absent" excess waits for planted-concept curves (005).

**Shuffle-sensitive agreement (a diagnostic).**

$$\Delta\kappa_{mm'}=\kappa_{mm'}(\text{real})-\kappa_{mm'}(\text{shuffled activations}).$$

Agreement that survives shuffling comes from family priors, not from reading the activation. κ differences are not a clean decomposition, so this is reported as a diagnostic, next to each reader's κ with prompt-only.

> **[CW]** Add: agreement that survives **content removal** is a candidate for shared re-solving (A4, A7).

**Test–retest.** Report each stochastic reader's self-agreement across its 8 samples directly, next to its agreement with other readers. (No disattenuation: the correction is not a valid general ceiling for κ.)

> **[R, 002]** NLA test–retest on near-identical activations: 48% agreement on the intermediate.

**Does agreement predict causal influence?** Count the independent reader *families* that name z. J/R count as one family; Oracle/NLA/Skip count as one; SAE is one; logit lens is one. Plot $P(\text{used}\mid k\text{ families agree})$ for k = 0…3, with prompt-only agreement as a separate curve.
- Steep and monotone: majority agreement is a usable proxy, and its usability can be quantified.
- Flat, or rising only with prompt-only agreement: agreement tracks inferability, not use.

Also fit Dawid–Skene without labels. Compare its inferred truth to the causal labels, and compare reader rankings (Kendall τ).

> **[v2]** Count the Oracle and the NLA as one family (shared base model; both can re-solve) and the token lenses J/R as one. P(used | k families agree) is conditional on content labels; its position-level and task-truth versions are active. Dawid–Skene needs no labels: active.

**What drives agreement: the atlas.**

$$\text{logit}\,P(R_m=R_{m'})=\beta_0+\beta_{\text{family}}+\beta_{\text{content type}}+\beta_{\text{layer}}+\beta_{\text{position}}+\beta_{\text{token len}}+\beta_{\text{inferable}}.$$

Expected: high agreement on entities at decision cells, collapsing on numbers and bindings; agreement rising with depth; a large prompt-inferability effect.

### A2. Causal accuracy per reader

- Sensitivity: $\text{Sens}_m=P(R_m=1\mid\text{used})$
- Specificity: $\text{Spec}_m=P(R_m=0\mid\text{absent})$
- **Current-task-use selectivity:** $W_m=P(R_m=1\mid\text{used})-P(R_m=1\mid\text{decodable-not-needed})$, computed within family × position on the matched contrasts first, and pooled only through the fixed-effects model.

High W: the reader reports content the measured answer depends on. W ≈ 0: it reports decodable content whether or not this answer needs it. Low W does not by itself mean the reader reads "outside the workspace". Nanda's hypothesis predicts $W_{\text{NLA}},W_{\text{SAE}}\ll W_J,W_{\text{Oracle}}$; a confirmed gap supports "NLA reports content the current task doesn't use", not "NLA reads outside the workspace".

> **[v2] Status: conditional.** A content-level W needs labels and a within-cell matched contrast.
>
> **[R, 002–003] Descriptive position-level pattern on Brew:** J-lens tracks the intermediate at ":" and at the newline after `</think>` about equally (47–48/48 tables at both). The NLA mentions it at 0.42 (":") vs 0.60 (`</think>`), role-correct 0.05 vs 0.15. At these cells, z-reporting does not separate influential from non-influential positions (a W ≈ 0 pattern; an observation, not a score).

### A3. Ensemble and conditional contribution

Held-out model:

$$\operatorname{logit}P(\text{used}_i)=\beta_0+\sum_m\beta_m s_{im}+\text{family FE}.$$

Per reader, **incremental predictive utility** $\Delta\text{AUC}_m=\text{AUC(all)}-\text{AUC(all}\setminus m)$. This is the added discrimination of this fitted ensemble at this sample size, not conditional mutual information. Cross-validation folds are grouped by template and source problem, so near-identical counterfactual pairs never straddle train and test. Fit the same model for the *decodable* label. If NLA adds information about decodability but not about use, that answers Nanda's question in one line.

> **[A01 C20]** Family × position fixed effects.
>
> **[v2] Status: conditional; retargeted.** Until content labels exist, fit the ensemble for (i) task truth (the reader names the correct hidden step in its role) and (ii) position-level influence, held out by table / item group.
>
> **[CW]** On no task so far does the NLA find a hidden step the Oracle misses: a first hint that the NLA's increment over the Oracle on hidden steps is small.

### A4. Where reader-only claims come from

For each claim a text reader makes and J doesn't, attach overlapping evidence tags. These are not a partition: a real activation signal can also be prompt-inferable.
- **Prompt-predictable.** Prompt-only, or the same reader on shuffled activations from the same family, produces it.
- **Activation-sensitive.** It appears with the real activation at a higher rate than under shuffled activations.
- **Behaviour-linked.** The claimed content is labelled *used* at that cell (section 6).
- **Contradicted by context.** The counterfactual or the model's output contradicts it. This may be a considered-but-rejected option rather than a confabulation; cross-check against the commitment family.
- **J-adjacent (descriptive only).** Its constituent tokens rank highly under J at the same cell. A reconstructed vector's J alignment is not used, since it does not show the natural activation expressed the claim that way.

Claims with no decisive tag stay **unresolved**. Output: tag frequencies and co-occurrences per reader and family.

> **[CW] Amendment to A4 (the one change Cathryn proposed).** "The spec counts a claim as activation-sensitive if it appears with the real activation and not with a shuffled one. That is not enough to say the content was read. A text reader is the model itself with a LoRA. It can recover the prompt from the activation and then solve the task. The claim then depends on the activation and is still not a readout of the hidden step." Proposed tag: **re-derived** = the claim still appears after the content's own directions are removed from the activation. Her removal uses J-lens directions; "a removal built from the §6 labels should replace it when one exists."
>
> **[v2] Adopted as follows.**
> - **Survives removal** (operational tag): the claim's rate over samples, after the content's directions are removed from the reader's input, stays above the matched random-removal control's lower bound. Run with **both** removals: J-based (as in the pilot) and **reader-free** (LEACE-style erasure fitted on dev items; on Brew, 004's fitted content subspace). J-lens's own drop under J-based removal (CW-3, CW-4: 15 → 0) is a check that the removal worked, not a result.
> - **Re-derived** (inference tag) needs survival **plus** positive evidence of re-solving: (i) after reader-free removal a fresh cross-fitted probe cannot decode the content (nothing left to read), **or** (ii) the readout restates the question/input before naming the content (as in CW-7's restatement check).
> - **Read outside J-space** (inference tag): survives J-based removal, does **not** survive reader-free removal, and the content stays probe-decodable after J-based removal. This is the reading Q12 asks about.
> - **Behaviour-linked**, until content labels exist: the content sits at a position with influence (tier 1), or the planted version of it changes the model's output (005).

### A5. NLA mechanism

- **Reconstructor blindness.** For hallucinated claims, on minimal pairs:
  - reward difference $\Delta(c)=r(s_c)-r(s_{\bar c})$;
  - discriminability $d(c,\bar c)=\|\mathcal D(s_c)-\mathcal D(s_{\bar c})\|/\|\mathcal D(s_c)-\mu\|$.
- **Geometric faithfulness.** $\cos(\delta_{\text{NLA}},\delta_{\text{true}})$, where $\delta_{\text{NLA}}=\mathcal D(C[c\to c'])-\mathcal D(C)$ and $\delta_{\text{true}}=h(z')-h(z)$.
- **Selection experiment.** Using the 8 samples per cell, compare the hallucination rate on held-out verifiable claims for three choices: a random sample, the highest-FVE sample, and the sample with the highest claim-grounding score. Blindness on single claims (Δ(c) ≈ 0) does not by itself show FVE selection fails, because whole-sample FVE could still correlate with accuracy through other claims. This experiment answers the scaffold question directly.

> **[v2] Status: deferred (low priority).**
>
> **[A01 §5, C21]** Add the "Δ(c) > 0 for the hallucinated claim" branch (co-adapted codes); use the step-matched NLA; record a dated P5 prior update (value pending, PI).
>
> **[R, 001]** FVE is dominated by the shared template, with a small source preference (244/360); best-of-N untested.
>
> **[CW]** Cathryn's next step "whether reconstruction distinguishes true and false claims, and whether choosing samples by reconstruction quality reduces unsupported explanations" maps here.

### A6. Commitment

Compare the J margin $\kappa_J=\langle v_A-v_B,h\rangle$ and the other readers' endorsement claims against which option is causally used (lens-free commitment direction). Swap J rows vs swap the lens-free direction. Question: which reader can tell "considering" from "decided"?

> **[A01 C1] Status: out for now** (commitment family out).

### A7. Reading vs re-solving battery (new; [CW], extended [v2])

Plain words: when a text reader names a hidden step, find out whether it read the step or worked it out from the prompt it decoded. Owners: C (multihop, chain), V (Brew, model side). Status: **active** (pilot done, exploratory).

1. **Layer sweep** [CW]: read at layers before the step is computed (L16/28/36). Prose readers naming the step where J-lens **and a probe** find nothing is evidence of re-solving. The probe is required (J-lens alone is weak early), and off-layer inputs are flagged (§2).
2. **Removal** [CW]: as in A4 (both removals, matched random control, rates over samples).
3. **Swap** [CW]: remove own content, add another item's at a strength J-lens shows. A reader that follows the swap read; one that keeps naming its own step re-solved. The planted step must be visible to J-lens (CW-5 was weak: J showed it in 14/50 even at 16× scale).
4. **Restatement check** [CW]: whether each hit's readout restates the question or input (CW-7: 31 of the Oracle's 36 hits at chain L56 do not contain the start number).
5. **Prompt recovery predicts re-solving** [CW]: per item, how much of the prompt a readout recovers; does it predict naming the step after removal (Q13)? No GPU needed.
6. **Probe-based presence per layer** [CW].
7. **J-any-layer cross-check** [v2]: for items named only by the prose readers at L42, does J-lens name the bridge at any of its 11 layers? Prose-only items that J never sees anywhere are the strongest re-solving candidates. No GPU needed.
8. **Model-side removal** [v2]: apply the same removal in the model's own forward at that cell and measure the answer margin. If the model still answers, the needed information persists outside the removed directions (or is recomputed downstream). This is the causal half the reader-input test lacks.

### A8. Planted J / non-J concepts (new; [CW] + [v2]; design 005)

Plain words: put a known concept into the activation, split into the part J-lens can see and the rest, and ask which readers report which part, and which part the model itself responds to. This is the direct test of "NLAs read outside the workspace" (Q12). Status: **proposed**; design requirements in §7 (planted concepts callout). Owner: joint (§15). Prior work: Viswanath (2026) on Llama-3.3-70B, injection only, no natural prompts and no re-solving test.

### A9. What the NLA's text consists of (new; [CW])

Concept split of NLA readouts: claims extracted by an LLM, bucketed by word match, then judged against the prompt. CW pilot, multihop, 100 readouts, 1,696 concepts:
- hidden step or answer 12%; also in J-lens top-10 15%; prompt words 22%; format description 22%; next token 1%; other 27%;
- of the 461 "other": form 45%, unstated and not ruled out 27%, contradicted by the prompt 22%, restates 6%;
- 66/98 readouts contain at least one contradicted concept;
- NLA readouts average 127 words and ~7 quoted strings; Oracle readouts 73 words and < 1.

Judge unvalidated (the §7 gate applies). Status: done on multihop (exploratory). **[v2]** Repeat on Brew with the validated judge.

---

## 9. Registered predictions (priors)

| # | Prediction | Prior |
|---|---|---|
| P1 | Majority-vote reader ranking differs from causal ranking (Kendall τ < 0.5) | 60% |
| P2 | $W_{\text{NLA}}<W_J$ and $W_{\text{NLA}}<W_{\text{Oracle}}$, CIs clear | 55% |
| P3 | J–R and Oracle–NLA are conditionally dependent given the causal label | 70% |
| P4 | Ensemble beats the best single reader by ΔAUC ≥ 0.05 on used-labels | 50% |
| P5 | For hallucinated NLA claims, Δ(c) ≈ 0 and hallucination rate decreases in d | 55% |
| P6 | On commitment, J does not separate endorsed from rejected; the causal label does | 55% |
| P7 | J sensitivity high on brew and two-hop, near floor on chain and role-bound at L42 | 75% |
| P8 | $E_{\text{J,R}}$ and $E_{\text{Oracle,NLA}}$ within *absent* are positive, CIs exclude 0 | 70% |
| P9 | $P(\text{used}\mid k)$ rises by < 0.2 from k=1 to k=3 once prompt-inferability is controlled | 45% |
| P10 | Highest-FVE NLA sample does not reduce held-out hallucination vs a random sample (difference < 0.05) | 50% |

> **[v2] Status of P1–P10.** None is testable yet as written (each needs content labels and/or reader scores).
> - P1, P3, P8: conditional on labels.
> - P9: conditional; has a position-level version.
> - P2: conditional; position-level pattern so far is W ≈ 0 for J and the NLA on Brew (descriptive).
> - P4: conditional; retargeted to task truth (A3).
> - P5: prior update pending, dated (A01 C21).
> - P6: out for now (commitment).
> - P7: partly informative already: J's per-cell sensitivity on Brew at L42 is low on displays and moderate on candidate scores; chain at L42 is near floor.
> - P10: deferred with A5.
>
> **[CW] Pilot hypotheses H1–H3:** §3 callout (H1 supported on multihop; H2 supported on chain at L56; H3 supported, judge unvalidated).
>
> **[v2] New predictions** (written 2026-10-04, before the runs they govern; **priors to be set by the PI before each run**):
>
> | # | Prediction | Prior |
> |---|---|---|
> | P11 | In 005, the NLA detects the non-J part of a planted concept at doses where the model's next-token distribution does not change (KL within the noise floor) | PI to set |
> | P12 | In 005, the model's output responds to the J part at a lower dose than to the non-J part | PI to set |
> | P13 | On multihop at L42, the share of prose-reader mentions that survive removal is lower under reader-free removal than under J-based removal | PI to set |
> | P14 | Per item, the prompt-recovery score predicts naming the hidden step after removal (AUC ≥ 0.65) | PI to set |
> | P15 | On chain at L56, the Oracle's hits fall to chance after reader-free removal of the intermediate | PI to set |
> | P16 | Of the multihop items named only by the prose readers at L42, J-lens names the bridge at some other layer in fewer than half | PI to set |

---

## 10. Pitfalls and handling

| Pitfall | Handling |
|---|---|
| Prompt-inferable content | Increment over prompt-only for every reader; shuffled-activation controls; inferability as a regressor in the atlas |
| Verbosity | Matched claim budgets, reported at several k |
| Any-cell inflation | Matched cells, per-layer reporting |
| Judge and summariser noise | Closed-set scoring; abstain option; dev/test split; judge only text readers; hand spot-check of 50 items |
| ~100 items per family | Paired statistics (McNemar on hits, paired bootstrap on scores); pool only through the fixed-effects model |
| Circular labels | Labels never use a reader. J atoms appear only in A4, never in labels |
| Answer geometry | Two-hop admission rule (reader-blind); binary-answer consumers where possible |
| Stochastic readers | 8 samples; self-consistency variable; test–retest reported directly |
| Shared lineage inflating majority | Count reader *families*, not readers, in the "does agreement predict causal influence" curve |
| Label method | Mean-difference vs DAS agreement reported; disagreement is itself a result |
| Subspace-patching illusion | Necessity check alongside the swap; "used" reported as demonstrated influence |
| Failed swap read as "unused" | "Not needed" requires a small full-cell patch; unresolved category |
| Bank design confounds use with position | Matched contrasts within family × position before pooling |
| Admission circularity | Reader-blind admission swaps on dev only; rejection counts reported |
| Item-level labels from dataset-level probes | Cross-fitted per-item probe probabilities |

> **[v2] Added pitfalls.**
>
> | Pitfall | Handling |
> |---|---|
> | Text readers re-solve from a decoded prompt [CW] | Removal (J-based and reader-free), swap, restatement, prompt recovery, layer sweep with a probe (A4, A7) |
> | Token-reader display vs scores [R, 002; CW] | Score candidate ranks as well as displays; never compare on displays alone |
> | Readers off their training layer [CW, v2] | On-contract layers primary; off-layer runs exploratory, with separate baselines |
> | Checkpoint mismatch between collaborators [CW] | One registry with hashes; reconcile before joint numbers (§4.R) |
> | Reader sampling variance [R, 002] | ≥ 4 samples per cell; rates, not single-readout tags; test–retest reported |
> | Intermediate = runner-up answer on Brew [A01; WSB] | Carried into every Brew claim; four-way {input, z, answer, other} attribution |
> | Guessable hidden steps (Brew prompt-only 1.00; WSB multihop prompt-only ≈ 0.90) [R; WSB] | Prompt-only arm; removal and restatement tests; prefer families where the step is not guessable |
> | Collaborator readouts on test items [CW] | Quarantine (§4.R); dev-only joint numbers |
> | bf16 noise (~0.2 nats per edit) [R, 000] | SEC float32 replay for L ≥ 40; effects compared to the measured noise floor |
> | Sequence-length drift [A01 §7] | One sequence length per clean/edited comparison; sources captured at the matching length |
> | Content labels fail reliability [R, 003] | Two tiers (§6.0); DAS decision; failure reported as a finding |
> | Subspace labels no better than generic subspaces [R, 004] | PCA and random comparators at matched rank and edit size (§6 step 8) |
> | Mentions read as claims; score shifts read as rank flips [R, audits] | Role-aware judge; rank reversals reported separately from score shifts |
> | Unvalidated LLM judges [R; CW] | Judge gate with a human-coded set (§7) |
> | Runs started before their design was settled [R] | Design spec committed and approved before every run, whoever runs it (§15) |

---

## 11. What each outcome means

- **P1 and P3/P8 hold.** Majority-vote ground truth is unsafe for interpretability evaluation, and the replacement exists (causal labels plus the excess-agreement decomposition). This is the methodological headline.
- **P9 holds.** Agreement mostly tracks prompt-inferability. Any claim that "technique X corroborates technique Y" needs a shuffled-activation control.
- **P2 holds.** NLAs report content the measured answer does not depend on: good for exploration, risky for audit claims about what drives behaviour. Nanda's hypothesis gets a number, scoped to current-task use.
- **P2 fails.** NLAs track current-task use as well as J does, and their errors are hallucinations rather than scope. The fix is grounding (A5).
- **J's W is high but its sensitivity is narrow.** J is a faithful but narrow window, and an output interface for some content types.
- **J fires on decodable-not-needed content (low W).** J reads readability, not current-task use. This generalises the readable-here-used-there finding across families.
- **P4 holds.** Readers carry complementary evidence; the ensemble weights show how to combine them.
- **P6 holds.** No J-based audit readout should be interpreted as intent without a commitment check.

> **[v2] Added outcomes.**
> - **P11 (and P12) hold.** The NLA reports planted content the model does not respond to: direct support for "NLAs read outside the workspace", scoped to planted states, model, layer and dose.
> - **P11 fails; no reader reports the non-J part.** NLA-only content on natural prompts is more likely re-solving or confabulation than non-workspace reading.
> - **A7: claims survive reader-free removal and readouts restate the prompt** (H1 holds). Text-reader agreement on hidden steps is weak evidence about activation content; any "NLA sees X" claim needs a removal control.
> - **A7: claims vanish under reader-free removal but survive J-based removal, and the content stays probe-decodable.** Reading outside J-space on natural prompts (Q12).
> - **Content labels stay unresolved after the DAS decision.** Report it: on these tasks, a reader-free content-level causal reference was not reached with linear edits; reader adjudication rests on position-level influence and planted concepts.

---

## 12. Order of work

0. **Gate (optional context).** Rotated-row null on kurtosis, occupancy and install share. The project does not depend on it.
1. **Plumbing.**
   - Reproduce WorkspaceBench J numbers (brew ≈ .86, basic multi-token ≈ .24) with the patched producer.
   - Run Oracle, NLA and Skip-lens end to end on 3 items.
   - Confirm reconstructor availability.
2. **Planted-concept sensitivity curves** for every reader.
3. **End-to-end slice (critical path).** Brew plus one matched not-needed condition (RHUT maintain vs ignore at the same carrier position), L42, 3–4 diverse readers (J, Oracle, NLA, SAE), held-out templates, shuffled-activation controls. Labels, reliability gate and unresolved fraction first, then W on this slice. Whether the labels work at all is the riskiest assumption, so it is tested before the bank is extended.
4. **Open-set agreement atlas (A1 level B)**, in parallel. Needs no labels and comes nearly free from the reader runs, but does not block step 3.
5. **Bank construction and admission** for the remaining families, including commitment.
6. **Labels, then the reliability gate.** Stop and fix if reliability < 0.7.
7. **Reader runs at matched cells**, including shuffled and no-activation controls.
8. **Analyses A1 (full) → A2 → A3 → A4 → A5 (including the selection experiment) → A6.**

> **[A01 C22]** Block 000 (setup, reader smoke) and design 001 (reader-free leverage profile + decodability) came first; the slice's not-needed arm was to be decided by 001.
>
> **[R] Status of the original steps.**
> - 0: not started.
> - 1: partial. One item reproduces WSB's pass rule for the three lenses; the full Brew reproduction was not run. Verbalizers end to end: done. Reconstructor availability: done.
> - 2: not started.
> - 3: precursor done (001); labels run (003); reliability failed; W not computed.
> - 4: not started (descriptive precursors in 001 and 002).
> - 5: partial (relational built, not gated; the other families out).
> - 6: done on dev (003); the gate failed.
> - 7: partial (001 snapshot; 002; CW pilot).
> - 8: not started.
>
> **[v2] Revised order of work.**
> 1. **Reconcile and validate (both).** NLA checkpoint (§4.R); one judge rubric with a human-coded validation set; a shared scoring protocol; the test-split quarantine. Reproduce WSB's J numbers on Brew under the any-cell rule and per cell (the plumbing check never finished).
> 2. **005, planted J / non-J concepts (joint).** Design spec first; readers and model side (A8).
> 3. **Reading vs re-solving (A4 amended, A7).** Multihop and chain at L56 (C); Brew (V); both removals; model-side removal (V). Rerun the pilot rows that are kept, under the §7 budget.
> 4. **Open-set agreement atlas (A1 level B)** and Dawid–Skene: no labels needed.
> 5. **Position-level labels wherever readers are scored** (tier 1); the Brew off-path within-cell contrast; codebook design (V) if time.
> 6. **DAS decision** (§6 step 9). Content labels only if they pass reliability and beat PCA.
> 7. **A2 / A3 against whatever truth exists** (task truth, position-level, content-level if any); then A9 on Brew; A5 if time.

**Stop and redesign if:**
- the bench reproduction fails;
- planted-concept curves show a reader detects nothing;
- label reliability stays < 0.7 after adding DAS labels;
- on the slice, most cells are unresolved or the matched contrast shows no used / not-needed difference in the labels themselves.

> **[R] Status of the original stop rules.**
> - Bench reproduction: not run in full.
> - Planted curves: not run.
> - "Label reliability stays < 0.7 after adding DAS labels": reliability is < 0.7 and DAS has not been added, so this rule waits on the DAS decision.
> - Slice: not run as specified (001–004 replaced it on Brew).
>
> **[v2] Replacement rule.** If the DAS decision leaves content-level reliability < 0.7, the project does not stop: reader analyses use tier-1 labels, task truth and planted concepts, and the label failure is reported as a result.
>
> **[v2] New stop rules.** Stop and fix if:
> - the NLA checkpoint cannot be reconciled;
> - the judge's κ against the human-coded set is < 0.6;
> - in 005, no reader detects planted content at any dose that changes the model's output (readers' misses on natural prompts would then be uninterpretable).

---

## 13. Open questions

1. Does Celeste's NLA ship its reconstructor? A4, A5 and vector-level agreement for the NLA need it. Without it, those analyses fall back to the Oracle only.
2. Are there auto-interpretation labels for the L42 SAE/BSF? If not: use J-projected decoders, or LLM-label ~2,000 features.
3. Is there a phrase-J checkpoint to run as an arm?
4. Which judge models are available for text readers and open-set matching (Gemini, Sonnet)?

> **[R; A01] Answers so far.**
> 1. Yes: the step-matched reconstructor `rl_critic_step400` is released and works. The Oracle's AR whiteners are private.
> 2. No labels for the candidate BSF; SAE deferred.
> 3. Phrase-J dropped.
> 4. Gemini 3.8 Flash (WSB's judge) via OpenRouter; the pilot also used Claude.
>
> **[v2] New open questions.**
> 5. Which NLA checkpoint (§4.R)?
> 6. How exactly is "J-space" defined for 005?
> 7. Does the codebook need a new bank, and who builds it?
> 8. DAS: run it, or record the decision not to?
> 9. Multihop dev/test split: by what grouping?
> 10. Oracle L42 arm (matched layer with the NLA): run off-contract and report separately? (pending since 2026-10-02)
> 11. Gate the 50 relational pairs, or drop them?
> 12. Who owns design 005, and who approves designs the other collaborator runs?
> 13. HF dataset name for raw outputs (both repos).
> 14. P5 prior value (A01 §5).

---

## 14. Results to date (v2; dev split only)

*All numbers: Qwen3.6-27B, WorkspaceBench, dev items. Brew/chain patching under SEC float32 replay; reader inputs from the stock bf16 forward. Nothing here is a label-based reader score.*

### 14.1 Where the answer depends on one cell (001; Brew, 82 pairs / 33 tables; median share S)

| Cell | L40 | L42 | L44 | L56 | L60 |
|---|---|---|---|---|---|
| ":" after "Answer" | 0.35 | 0.48 | 0.50 | 0.70 | 0.74 |
| newline after "assistant" | 0.29 | 0.29 | 0.32 | 0.14 | 0.13 |
| `\n\n` after `</think>` | −0.00 | 0.01 | 0.03 | 0.06 | 0.04 |
| start-colour token | 0.23 | 0.15 | 0.12 | −0.00 | −0.00 |
| every other prompt token | ≤ 0.04 | | | | |

Engine checks 100% bitwise; noise floor 1.4e-5 of the gap; reverse vs forward r = 0.84. Chain (48 pairs): the final cell is 0.00 at L40–44, 0.66 at L56, 0.69 at L60.

### 14.2 What is decodable at ":" (001, repaired probes; 12-way balanced accuracy, chance 0.083)

| Target | L40 | L42 | L44 | L56 | L60 |
|---|---|---|---|---|---|
| intermediate z | 0.79 | 0.78 | 0.77 | 0.64 | 0.70 |
| start colour | 0.84 | 0.85 | 0.75 | 0.50 | 0.54 |
| answer | 0.18 | 0.18 | 0.19 | 0.84 | 0.84 |
| off-path membership (AUC) | 0.64 | 0.63 | 0.61 | 0.64 | 0.79 |

z at the other emission cells, L40–44: assistant newline 0.80–0.82; `</think>` newline 0.79–0.84. Off-path substitution keeps the answer within bound in 39/43 items.

### 14.3 What readers report (002; verbalizers 10 tables × 4 samples; lenses 48 tables)

| Reader, cell | z in its role | mentions z (real / shuffled) | conflicting |
|---|---|---|---|
| NLA L42, assistant newline | 0.15 | 0.55 / 0.25 | 0.20 |
| NLA L42, `</think>` newline | 0.15 | 0.60 / 0.20 | 0.12 |
| NLA L42, ":" | 0.05 | 0.42 / 0.20 | 0.38 |
| Oracle L40, ":" (k = 4) | 0.00 | 0.53 / 0.38 | 0.85 |
| Oracle L44, ":" (k = 4) | 0.03 | 0.30 / — | 0.57 |

- **Token lenses** (J_NP cosine): z ranked first among the candidates at L42 in 0.46 (assistant newline), 0.46 (`</think>`) and 0.42 (":") of tables; shuffled 0.15–0.19. Start-swap tracking positive in 47–48/48 tables at L40–44; the start colour is not tracked.
- **Prompt-only:** z 1.00, start 1.00, answer 0.89 (231 renders).
- **Overlap** (10 tables): the NLA and the Oracle both mention z in 6; both state it in its role in 0 (Oracle L40) or 1 (L44); they give the same wrong intermediate in 3–4.

### 14.4 One content direction (003; 82 pairs; median S)

| Cell | z direction | start | answer | whole cell |
|---|---|---|---|---|
| ":" L40 / L42 / L44 | 0.031 / 0.034 / 0.050 | ~0.01 | ~0.00 | 0.35 / 0.48 / 0.50 |
| ":" L60 | 0.004 | 0.006 | 0.62 | 0.74 |
| start token L40 (positive control) | 0.007 | 0.076 | 0.001 | 0.23 |

"Used" z in 11–18% of pairs at ":" L40–44, each also necessary. Reliability κ 0.37 (r 0.41), so the labels are descriptive. At the `</think>` newline, z is decodable and not needed in 70–100% of pairs.

### 14.5 Subspaces (004; ":" L42; 33 pairs; whole cell 0.43)

| Edit | Recovery of the whole-cell effect |
|---|---|
| 1-D z | 0.06 |
| z subspace (11 dims) | 0.21 |
| PCA, 11 dims (same edit size) | 0.58 (0.29) |
| start + z + answer subspace (33 dims) | 0.72 |
| PCA, 33 dims (same edit size) | 0.89 (0.80) |
| random | ≈ 0 |

The z-subspace swap mostly raises the recipient's own intermediate (18/33) and is aimed at the donor's answer in only 4/33. The z span plus its complement falls 0.18 short of the whole cell. Clamping the z span removes 0.12 of the gap, above paired rotations in 31/33.

### 14.6 Cathryn's pilot (CW; word match unless stated; numbers re-checked 2026-10-04)

| # | Experiment | Items, layer | J-lens | Oracle | NLA |
|---|---|---|---|---|---|
| CW-1 | Main comparison | 100 multihop, L42 | 39 | 80 | 75 |
| | chance | same | 0.7 | 1.7 | 1.2 |
| CW-2 | Earlier layers | 50 multihop, L16 / 28 / 36 | 0 / 1 / 9 | 9 / 26 / 33 | 11 / 27 / 28 |
| CW-3 | Delete the step's J directions (2–6 per item) | 50 multihop, L42 | 15 → 0 | 30 → 17 | 36 → 25 |
| CW-4 | Delete the step's J-neighbourhood (cos > 0.8, ~31 dirs); control = same number of random dirs | 50 multihop, L42 | 15 → 0 | 33 → 20 | 34 → 23 |
| CW-5 | Swap: delete own step, add another item's | 50 multihop, L42 | planted 14 | own 20, planted 7 | own 20, planted 6 |
| CW-6 | Chain | 60 chain, L42 | 6 (chance 4) | 27 (21) | 9 (12) |
| CW-7 | Chain | 60 chain, L56 | 25 (7) | 36 (12) | 13 (9) |
| CW-8 | Brew, z in its role (Claude / Gemini judge) | 50 Brew, ":" L42 | 0 in top-10 | 2 / 5 | 5 / 3 |
| CW-9 | Brew, z in any role: random control → z deleted (Gemini) | 50 Brew, ":" L42 | — | 7 → 0 | 12 → 2 |

Her reading: multihop is mostly re-solving; chain at L56 is reading; Brew is neither (mentions are rare and vanish with deletion). Her untested explanation: readers re-solve when they can rebuild the prompt (a one-line question, not a rule table).

### 14.7 Current results under the joint framing (shared doc, corrected in v2)

- **Read from the activation.** Chain at L56: the Oracle names the intermediate in 36/60 vs chance 12, with 31 hits lacking the start number (CW-7). Brew: mentions vanish when z's directions are deleted (CW-9). *Whether the model uses that content is a separate claim:* chain's final cell gains leverage at L56 (001), position-level only.
- **(b) Readable, not used (position-level).** z is equally decodable at ":" and after `</think>`; patching moves the answer ~0.44–0.48 at ":" L42 and ~0.01 after `</think>` (001); J-lens and the NLA report z equally at both (002).
- **(c) Worked out from the prompt (candidate).** Multihop L28: prose readers 26–27/50 vs J 1/50 (CW-2); 20–23 of 33–34 mentions survive J-neighbourhood deletion (CW-4). *Not yet separated from reading outside J-space* (A4, A7).
- **(d) Made up or wrong role.** Multihop: 66/98 NLA readouts contain a claim the prompt rules out (A9). Brew: z in its correct role in 5–15% (NLA) and 0–3% (Oracle) of samples (002).

---

## 15. Collaboration and working rules (v2)

**People.** Vishesh (PI of Causal-Concordance; repo `senku14x/Causal-Concordance`). Cathryn Wayman (MATS 12 mini project; pilot repo `LittleMeHere/nla-vs-lenses`). Shared doc: "Workspace Work".

**Ownership** (from the shared doc, 2026-10-04; [v2] proposals marked):

| Workstream | Owner | Status |
|---|---|---|
| Brew reader scoring and position-level labels | V (both run Brew) | active |
| Multihop, chain at L56, reading vs re-solving (A7 items 1–7) | C | active |
| Codebook contrast | V | planned |
| Labels for z: DAS decision; off-path within-cell contrast | V (shared doc: "labels for z (DAS, or score off-path colours)") | active |
| Prompt recovery predicts re-solving (Q13) | C | active |
| 005 planted J / non-J concepts | Unassigned in the shared doc. **[v2] joint**: C proposed it; V's float32 write harness runs it | proposed |
| Model-side removal (A7 item 8) | **[v2] V** | proposed |
| Judge validation set | **[v2] both** | proposed |
| NLA checkpoint reconciliation | **[v2] both** | proposed |

**Non-negotiables** (Causal-Concordance PI directive, 2026-10-02). They apply to every joint number, whoever runs it.
- Labels never use any reader: no J swaps, AR edits or SAE directions in labels or item admission. Label code must not import reader code. (J-based removal is allowed in A4/A7 as a reader-side analysis, never in labels or admission.)
- Direction-level writes go through float32 hooks. Fidelity is measured after the write (ρ in [0.9, 1.1], κ ≥ 0.99).
- Every experiment has a design spec, committed before its run and shown to the PI. Launch only on the PI's go. **[v2]** For joint experiments, both collaborators sign off on the design.
- Gates before outcomes: span/rendering, same-source == clean bitwise, fidelity, damage, label reliability. A failed gate is repaired and re-run; nothing that has been scored is deleted.
- Full-sequence endpoint, log-sum-exp over accepted spellings; clamp and score every answer token, never just the first.
- Baselines from day one (prompt-only, no-activation, shuffled-activation).
- Wording: "demonstrated influence on this answer", not "uses"; "decodable", not "represents". Scope every claim to model, family, position, layer and n.
- Record every silly mistake in the commit message and the report.
- No subagents (Causal-Concordance working rule).

**Joint numbers** come only from the shared protocol: reconciled checkpoints (§4.R), the validated judge (§7), the §7 sample budget (§4 callout), the dev split, float32 lens scores, and the single-token chain subset. Pilot numbers stay labelled CW-*n* and exploratory.

**Registry.** One table of checkpoints and file hashes for both repos: model, lenses, Oracle, NLA, reconstructor, WSB bank files.

**Numbering.** Causal-Concordance experiments keep `NNN` (005 = planted concepts). Cathryn's pilot rows are `CW-n`. New joint experiments get `NNN`, with both names in the design spec.

**Compute.** Cathryn's pilot: 10–20 hours of her time, under $10 of GPU per experiment, one 80 GB GPU on RunPod. Causal-Concordance: one RTX PRO 6000 Blackwell (96 GB).

---

## References

- Gurnee et al. 2026, *Verbalizable Representations Form a Global Workspace in Language Models*: §3.1–3.5, §5, §6, A.9 (template and oracle lens), A.12, A.21 (J vs NLA agreement at AUC .853), A.24.3.
- Blank, Bhatia, Ong & Nanda 2026, *WorkspaceBench*.
- Neel Nanda, review of the workspace paper; updated project ideas doc (cross-referencing techniques, majority rating, scaffolds).
- Blank et al. 2026, R-lens.
- Anthropic NLA 2026; Dingeto 2026 (arXiv 2607.20379); Hong 2026 (fact-check reward); michaelzhang & TurnTrout 2026.
- Li et al. 2025, *Do Activation Verbalization Methods Convey Privileged Information?* (arXiv 2509.13316).
- Karvonen et al. 2025, Activation Oracles (arXiv 2512.15674); Bauer et al. 2026, AObench (arXiv 2606.02609).
- Dawid & Skene 1979, *Maximum likelihood estimation of observer error-rates using the EM algorithm*.
- Cohen 1960 (κ); Fleiss 1971; Krippendorff 2004 (α); Webber et al. 2010 (rank-biased overlap).
- Geiger et al. 2023, DAS (arXiv 2303.02536); Makelov et al. 2023, interpretability illusion (arXiv 2311.17030).
- Feng & Steinhardt 2023, binding IDs (arXiv 2310.17191).
- Prior work: RHUT (readable-here-used-there), phrase_J 003a/004, ekko-lens, research_secant 000–002, the WorkspaceBench producer patches.

**[CW, v2] Added references.**
- Viswanath 2026, *Models are blind outside the J-space. NLAs aren't.* LessWrong, 2026-07-08 (https://www.lesswrong.com/posts/LhDJdccLszLEAqgZ9/models-are-blind-outside-the-j-space-nlas-aren-t). Llama-3.3-70B: an injected concept's non-J part is invisible to the model and read by the NLA. Injection only; no natural prompts or re-solving. [CW]
- Bhatia, Blank & Nanda 2026, *Towards surfacing model algorithms with meta-tokens in the J-space.* LessWrong, 2026-07-20 (https://www.lesswrong.com/posts/6ek6n7yZ5DzfarJHy/towards-surfacing-model-algorithms-with-meta-tokens-in-the-j). J-lens readouts on Qwen3.6-27B carry process tokens. [CW]
- Prabhu 2026, *Can you hide from a natural language autoencoder?* LessWrong, 2026-06-24 (https://www.lesswrong.com/posts/3oRQxpxn4LiWKwbeQ/can-you-hide-from-a-natural-language-autoencoder). NLA explanations on Qwen2.5-7B can be flipped while behaviour is unchanged. [CW]
- *WorkspaceBench: Evaluating Interpretability Methods for the Global Workspace*, LessWrong (https://www.lesswrong.com/posts/Zeg2JztbdhguL48uH/workspacebench-evaluating-interpretability-methods-for-the). [CW]
- Wayman 2026, *NLA vs lenses pilot*, https://github.com/LittleMeHere/nla-vs-lenses. [CW]
- Belrose et al. 2023, *LEACE: Perfect linear concept erasure in closed form* (arXiv 2306.03819). [v2: reader-free removal]
- Ravfogel et al. 2020, *Null It Out* / INLP (arXiv 2004.07667). [v2]
- Hewitt & Liang 2019, *Designing and Interpreting Probes with Control Tasks* (arXiv 1909.03368). [A01: probe control task]
- Benjamini & Hochberg 1995, false discovery rate. [A01: BH-FDR]

---

## Appendix A. Cathryn Wayman's A4 spec amendment (verbatim, 2026-10-03)

*A4 Spec Amendment · 3 Oct 2026 · proposed changes to SPEC.md in Causal-Concordance · section numbers are from that spec*

**Summary**
- I take A4 / Q8 (where text-reader-only claims come from) and the planted-concept test (§7, step 2).
- No change to the question, the model, the labels (§6) or the other analyses.
- One change to A4, below.

**The one change**

The spec counts a claim as activation-sensitive if it appears with the real activation and not with a shuffled one.

That is not enough to say the content was read. A text reader is the model itself with a LoRA. It can recover the prompt from the activation and then solve the task. The claim then depends on the activation and is still not a readout of the hidden step.

- Add a tag to A4: re-derived.
- A claim is re-derived if it still appears after the content's own directions are removed from the activation.

Pilot evidence, multihop, 50–100 items, word match:

| Check | J-Lens | Oracle lens | NLA |
|---|---|---|---|
| Names the hidden step, L42 (of 100) | 39 | 80 | 75 |
| Same, from another item's activation | 0.7 | 1.7 | 1.2 |
| Names it at L28 (of 50) | 1 | 26 | 27 |
| After removing the step: control → removed (of 50) | 15 → 0 | 33 → 20 | 34 → 23 |

The removal uses J-Lens directions. A removal built from the §6 labels should replace it when one exists.

**What I add**

| Item | What | Status |
|---|---|---|
| Multihop | Used for A4 only. No partner prompts, so no labels. | Run, 100 items |
| Removal test | Remove the content's directions at one cell. Control removes the same number of random directions. | Run on multihop and brew |
| Swap test | After removal, add another item's content. | Run. Weak on multihop. |
| Other layers | Text readers at L16, 28, 36, 56. The spec fixes them to L42. | Run. Exploratory. |
| Planted concepts | Add a concept at a neutral cell, sweep the dose, every reader. Split the concept into its J-space part and the rest. | Not run. Proposed as 005. |

**Where the pilot differs from §7**

| Spec | Pilot |
|---|---|
| 8 samples per cell | NLA 1 sample. Oracle lens 1 readout of 4 bullets. |
| Claim budget k = 4 | Whole readout scored |
| Closed-set judge | Word match on multihop and chain. My own role judge on brew. |
| No-activation and prompt-only arms | Not run |
| fp32 lens scores | bf16 |

I will rerun what we keep under §7.

**To reconcile before any joint number**
- NLA: different models. Registry: ceselder/qwen3.6-27b-nla-L42, av_rl_lora_step400 (1.27 GB adapter). Pilot: ceselder/qwen3.6-27b-nla-rl, iter_000400 (1.87 GB adapter). The file hashes differ. We need to pick one.
- Oracle lens: same weights. Registry agu18dec/olens_and_ar/olens_s3d_rl600 and pilot agu18dec/local-workspace/…/iter_000600 have the same adapter file hash. Nothing to do.
- Split. My brew run used the first 50 bank items. By the seeded split, 25 are dev and 25 are test. I produced reader readouts on them, no labels and no tuning.
- Chain. My 60 items include multi-digit intermediates, which are outside the label bank.

**Next**
- Design 005: planted concepts, all readers.
- Rescore the multihop readouts with the 002 judge.

Pilot results and raw files: github.com/LittleMeHere/nla-vs-lenses

## Appendix B. AMENDMENT_01 (verbatim, 2026-10-02)

### AMENDMENT 01 — decisions after the reading pass (2026-10-02)

*Records the PI's decisions of 2026-10-02 (taken on `docs/reports/000_reading_report.md`) and every conflict with
`docs/context/SPEC.md` (cited by line; SPEC.md is byte-identical to the spec the reading report cites). Where this file
and the spec disagree, this file wins. Nothing scientific had been run when it was written.*

Thresholds marked *provisional* may be revised **once**, on dev material, from the leverage profile (design 001), and
are then frozen in AMENDMENT_02 before any test-split label is computed.

---

#### 1. Tasks (families)

**WorkspaceBench only** for now. Bank files are pinned by sha256 in `scripts/common/registry.json` (WSB commit 92d763e).

| Family | Status | Construction |
|---|---|---|
| **Brew** | core | Partner = same table, different start colour. Constraints: z′ ≠ z, a′ ≠ a, z′ ∉ {a, a′}, z ∉ {a′}. The start colour is a single token in " X."; both renders have identical token counts and differ only at the start-colour token. Gate both members: greedy correct and ≥ 9/10 correct at T = 0.8. Report candidates built, candidates passing, and the difficulty distribution (do not keep only easy tables). The 24 WSB pinned positions are re-verified with the pinned tokenizer. Stir cells are kept. The runner-up caveat is carried (WSB: brew cannot separate the intermediate from the runner-up answer; prompt-only scores 0.96). |
| **Chain** | core | Number tokenisation checked first (digits are single tokens, " 23" → [" ", "2", "3"]). Only items whose intermediates are single tokens (single digits) are kept. Partner = same steps, different start number with the same digit count. Same constraints and gate as brew, plus WSB's own chain constraints applied to the partner. Read cell = the final token, with the question span (" What" → end) used for the leverage profile. |
| **Relational** | core subset | Only the bank's token-aligned a/b pairs, gated. Demoted to secondary if fewer than ~30 pairs survive the gate. |
| **Multihop** | later | Cue-swap pairs, plain render; never pooled with RHUT two-hop. |
| **Role-bound** | agreement only | No labels. |
| **Arithmetic** | not now | Fraction variants dropped. |

#### 2. The not-needed contrast

Brew's off-trajectory colours at the emission cells are the candidate *decodable-not-needed* content. Design 001
establishes whether they are decodable there (item-grouped linear probe + Hewitt–Liang control task). If they are not,
the fallback is RHUT's tagged-selection organism (pointed vs unpointed word at the question/decision cells). The
fallback is **not built** unless 001 says so.

#### 3. Labels

- **Unit.** One token position at one layer. Span patches are context only and never label anything.
- **Used.** S ≥ τ **and** S above the norm-matched random-direction null after Benjamini–Hochberg FDR across the family's cells. τ = 0.10 (*provisional*).
- **Not needed.** Decodable **and** full-cell patch share < ε′, with ε′ < τ; ε′ = 0.05 (*provisional*). Cells in between are **unresolved**. *(Read |S| < ε′: see errata E1–E3, §8.)*
- **Illusion flag.** Every cell where the 1-D share exceeds the full-cell share is flagged (Makelov signature). *(Renamed and reinterpreted: errata E2, §8.)*
- **Halves (cross-fitting).** By items: brew by table, stratified by z; chain by start number, stratified by z; relational by name set. Cross-fit A→B and B→A so every item is labelled. Label reliability (agreement between labels computed from two disjoint fitting subsets for the same items) ≥ 0.7 before any reader is scored.
- **Naming.** The label is called **"pair difference used"**. Every used cell gets a four-way confusion of what the direction tracks: {input, z, answer, other}. A decorrelated direction (z varying while input and answer are decorrelated from it) is fitted where the family supports it.
- **Necessity.** Pooled mean = mean of û·h over the fitting half at that cell; a random-direction ablation control is run alongside; "used + necessary" requires N_i above that control's 95th percentile. INLP/LEACE-style removal later.
- **Decodability.** Regularised linear probe; folds grouped by item and table; Hewitt–Liang control task (selectivity). Binary z-vs-z′ probes are acceptable. The planted positive control must not use the probe's own direction.
- **Damage.** Δlog P(a), Δlog P(a′) and top-1 retention on non-answer positions after every edit (plus the off-pair mass at the answer position); broad damage → unresolved.
- **Precision.** Default: RHUT's float32-residual regime on transformers 5.15.0. Verify on ~5 edits against SEC's true-float32 replay; switch if Δm differs by > 5% of G. (Done 2026-10-02: max 3.3% of G → RHUT kept; see STATUS.md, `results/000_setup/analysis/regime_check.json`, and the noise-floor note there.)

#### 4. Readers

- **Oracle Lens:** on-contract at **L40 and L44**, labelled at those layers; an off-contract L42 arm is reported separately and never pooled. AR analyses for the Oracle are limited (its trained whiteners are private).
- **NLA:** the step-matched `ceselder/qwen3.6-27b-nla-L42` pair (`av_rl_lora_step400` + `rl_critic_step400`).
- **Skip-lens:** dropped.
- **SAE:** *[to be filled in by the PI]*; scored with non-J auto-interpretation labels only (J projection kept as a diagnostic).
- **J_NP, J_CB, R_CB, logit lens:** as in the spec (J_NP in both the WSB cosine readout and the standard readout).
- **Baselines from day one:** prompt-only, no-activation, shuffled-activation. **Prompt-only = stock Qwen3.6-27B.**
- **Closed-set candidates:** gold z, the input, the answer, and same-category foils. **Token-reader hit** = z ranked first among the candidates **and** its score above that reader's no-activation run.

#### 5. Also accepted

- Family × position fixed effects (A3 and pooling).
- A power calculation with a pre-registered minimum number of cells per stratum below which W is not reported.
- SAE with non-J labels (§4).
- A5: add the "Δ(c) > 0 for the hallucinated claim" branch (co-adapted codes); use the step-matched NLA; a **dated P5 prior update** (value to be set by the PI; the reading report §4.5.6 evidence points away from Δ(c) ≈ 0).
- Brew is **not** the label pipeline's positive control. The pipeline gets its own controls: the start-colour token at an early layer (full-cell share ≈ 1 by construction) and the stir cells (≈ 0).

#### 6. Conflicts with SPEC.md

| # | SPEC | Says | Now |
|---|---|---|---|
| C1 | L107–115 (§5 table) | Bank = brew, RHUT two-hop + WSB multihop (~100), chain/arithmetic, role-bound/relational, a new commitment family (~60 pairs), RHUT organisms as the decodable-not-needed family, descriptive safety families | WorkspaceBench only (§1). RHUT two-hop, the commitment family, the RHUT organisms and the safety families are out for now; multihop later; arithmetic dropped; role-bound agreement only |
| C2 | L109 | Brew is the "positive-control family" with "token-aligned donors" | Brew is not the pipeline's positive control (§5). Token-aligned partners did not exist; they are constructed (§1) |
| C3 | L112 | Role-bound / relational "ab/ba pairs" | Relational: only token-aligned a/b pairs; role-bound agreement only |
| C4 | L114, L140–143 (matched contrasts), L366 (slice) | Decodable-not-needed content comes from RHUT organisms; the slice pairs brew with RHUT maintain-vs-ignore | Candidate = brew off-trajectory colours at the emission cells, established in 001; RHUT tagged selection only as a fallback (§2) |
| C5 | L151 | Cells = source, carriers, decision | Unit = one token position at one layer; span patches context only (§3) |
| C6 | L153–157 | Core layer L42 for every reader; J/R/logit also at 40, 48, 56 | Oracle on-contract at L40/L44 with labels there; off-contract L42 arm separate (§4). Design 001 profiles layers 40/42/44 |
| C7 | L159–161 | Template halves A/B: fit on A, label on B | Item halves (table / start number / name set, stratified by z), cross-fitted so every item is labelled (§3) |
| C8 | L172 | "run the rest of the model in fp32" | Default is RHUT's float32-residual regime (bf16 matmuls); SEC true-fp32 is the cross-check, switch rule 5% of G (§3) |
| C9 | L177 | Used = S above the 95th percentile of norm-matched random directions | Used = S ≥ τ **and** above that null after BH-FDR across the family's cells (§3) |
| C10 | L178 | Necessity: ablate to "the pooled mean" (undefined) | Pooled mean = fitting-half mean of û·h at the cell, plus a random-direction ablation control and a 95th-percentile rule (§3) |
| C11 | L179 | Decodable: probe folds split by template; shuffled-label null | Folds grouped by item and table; Hewitt–Liang control task; binary probes allowed (§3) |
| C12 | L182 | Not needed: full-cell share < ε (e.g. 0.1) | < ε′ = 0.05 (provisional) with ε′ < τ; the band between is unresolved (§3) |
| C13 | L183 | Absent: positive control plants α·û_c | The planted control must not use the probe's direction (§3) |
| C14 | §6 (no gate) | No damage gate | Damage readouts on every edit; broad damage → unresolved (§3) |
| C15 | §6 | Label = "z used" | Label = "pair difference used", with the four-way {input, z, answer, other} confusion and a decorrelated direction where possible (§3) |
| C16 | L89–91 | Skip-lens fed L42 is a reader arm | Dropped (§4) |
| C17 | L88, L383 | NLA "with its reconstructor if released" | Released; the step-matched nla-L42 pair is used (§4) |
| C18 | L96 | Prompt-only = "an LLM reads the prefix and guesses" | Prompt-only = stock Qwen3.6-27B (§4) |
| C19 | L209 | Closed-set candidates = gold z + 4 same-category distractors | Gold z + input + answer + same-category foils; token-reader hit rule defined (§4) |
| C20 | L277 | A3 ensemble with family fixed effects | Family × position fixed effects (§5) |
| C21 | L292–298, L314 | A5 outcomes assume Δ(c) ≈ 0 (P5, prior 55%) | Δ(c) > 0 branch added; P5 prior updated, dated (§5; value pending) |
| C22 | L362–366 (order of work) | Step 1 reproduces WSB J numbers and runs verbalizers on 3 items; step 3 is the brew + maintain-vs-ignore slice | Block 000 (setup + one-cell reader smoke) and design 001 (reader-free leverage profile + decodability) come first; the slice's not-needed arm is decided by 001 |
| C23 | L127 | Question-turn ladder figures (install vs removal) | Factual correction recorded in the reading report §4.1.4 / §5.10 (no design consequence yet) |

#### 7. Implementation decisions taken in block 000 (not in the spec)

- **Regime origin.** The float32 residual starts at block 0 (`engine.Engine(fp32_from=0)`), not RHUT's block 35, so that early-layer controls (§5) and direction-level writes at any layer are carried in float32. Regime check: from-0 vs SEC 3.3% of G, from-35 vs SEC 2.3% (both < 5%).
- **Logits in float32.** Margins are computed from float32 logits (`final_norm(h) @ W_U^T` in float32), not from the bf16 lm_head output, whose spacing at |logit| ≈ 20 is 0.125.
- **Fixed-length comparisons.** Appending tokens changes prompt-position activations (rounding onset at block 3, the first full-attention block; not a causal leak). Every clean/edited comparison is made at one sequence length, and patch sources are captured at the matching length (`margin.score` builds writes per continuation).
- **Reader activations** are captured from the stock bf16 forward (WSB's `Backend.capture` convention), not from the label regime.

#### 8. Errata (2026-10-02, after the external review of commit c13fd43; accepted by the PI)

These correct §3 in place of silently rewriting it.

- **E1. Small effects are magnitude tests.** Wherever §3 says "full-cell patch share < ε′", read **|S| < ε′**. A patch
  with S = −0.8 moves the answer strongly the other way and is not small. Patch outcomes are kept separate:
  - toward (S ≥ τ);
  - away (S ≤ −τ);
  - negligible (|S| < ε′);
  - intermediate (otherwise).

  Group-level claims of negligibility use an equivalence test (90% CI of E[S] inside ±ε′, `stats.equivalence`) and
  are never converted into item-level labels.
- **E2. The full-cell patch is a reference, not a ceiling.** Two components of the patched difference can cancel. A
  negligible full-cell share therefore means "the natural pair difference at this cell has negligible influence on
  this answer", not "nothing at this cell matters". The "Illusion flag" is renamed the **1-D > full-cell flag**. It has
  two readings, and the flag alone does not choose between them:
  - a subspace-patching illusion;
  - cancellation inside the full patch.
- **E3. Not-needed content that is not the pair's latent** (e.g. brew off-trajectory colours) has no counterfactual
  answer gap (G = 0). Its criterion is a behavioural plus transplant bound on |Δlog P(a)| and the answer-distribution
  shift (design 001 §3.3), to be frozen with τ and ε′ in AMENDMENT_02.
- **E4. Precision for small-effect decisions.** The RHUT regime passed the 5% rule (3.3% of G). Its absolute noise
  floor (0.05–0.25 nats) is still too coarse for ε′ = 5% decisions. Design 001 v2 uses SEC's true-fp32 replay for all
  L ≥ 40 (pending PI confirmation, 001 §9.1).
