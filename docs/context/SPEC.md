# Do interpretability readers track what the model uses? Reader agreement versus measured causal influence on Qwen3.6-27B

*Draft project spec, 2026-10-01 (revised after an external red-team review). Priors are gut estimates written before any run. Nothing here has been run yet.*

---

## 0. One paragraph

Run every available activation reader (J, R, logit lens, Oracle Lens, NLA, Skip-lens, SAE, plus optional specialists) on the **same cells** of Qwen3.6-27B. Do two things with the outputs.

- **Agreement.** Measure how much the readers agree, at four levels (concept, open-set claims, activation-space direction, behavioural effect).
- **Adjudication.** Score each reader against a reference that uses **no reader**: whether the content it names has *demonstrated causal influence* on a specified answer at that cell, is *decodable but not needed* there, is *absent*, or *cannot be resolved* by these assays.

The causal labels then split agreement into three parts: shared signal (both see real content), shared error (both fail the same way), and shared prompt inference (both guess from the text). This turns "how much do the readers agree" into "which readers' claims match content with measured causal influence on the answer, and what is their agreement evidence of". It also tests Nanda's two hypotheses directly: that NLAs and SAEs pick up non-workspace content, and that majority agreement can serve as ground truth.

**What this does not give:** a lens-free definition of "what is in the workspace". A variable can be available for flexible use and still not drive the one answer measured here. The labels measure current-task causal influence, not workspace membership. The pattern of which readers see that content still bears on whether J-space is an output interface or a working store.

**Terminology.** Throughout, "used" is shorthand for *a reader-blind edit at this cell demonstrably moves this specified answer* (section 6). It does not mean the unmodified model naturally routes through that direction; a necessity check gives a stronger tier. "Decodable, not needed" means decodable here, while patching the whole cell barely moves the answer.

---

## 1. Why do this

**1. Agreement is not truth.** Nanda proposes rating techniques by agreement with the majority. That is only valid if readers fail independently, and ours don't:
- J and R share an estimator.
- Oracle and NLA share a verbalizer + reconstructor lineage.
- J, R and the logit lens share the unembedding. SAEs and the text readers do not, though they can still have correlated errors.
- Most can see prompt-inferable content.

Readers with shared errors agree whether or not they are right. Nobody has checked how much of the observed agreement is signal.

**2. WorkspaceBench measures readout, not use.** On many families the prompt-only baseline matches or beats every reader. An item passes if any (layer, position) cell passes. And a 250-token NLA readout gets more chances to be right *and* wrong than a top-10 token list. Its authors state it has no ground truth for workspace contents.

**3. Readable and used are different variables.** Prior work (RHUT, below) found concepts readable at copy positions with ~0 causal leverage. At the decision cell, the causally used part of J was mostly answer geometry. So a reader can fire on readable-but-unused content without hallucinating and without reading the workspace. No current evaluation separates these cases.

**4. It bears on the workspace, within limits.** The labels measure causal influence on a specified answer, with no lens involved. That is not workspace membership: content can be available for flexible use and still not drive the measured answer. Within that limit, each reader's hit rate on used vs decodable-not-needed content shows what that reader reads. For J, this is evidence on the two competing readings:
- **Store:** high recall of used content, low firing on decodable-not-needed content.
- **Interface:** tracks answer geometry and reportable labels, misses used content that isn't about to be said.

**5. It returns a result under every outcome.** Every branch produces a map: which reader, on which content type, at which cell, agrees with which other reader, and whether it sees used content.

---

## 2. North Star, proxy, validity, divergence

- **North Star.** Know which interpretability readers can be trusted to report content the model's computation depends on, measured independently of any reader.
- **Proxy.** On families with token-aligned counterfactuals, at matched cells and matched claim budgets, measure each reader's:
  - sensitivity to used content;
  - hit rate on decodable-but-not-needed content;
  - agreement structure after conditioning on the causal label.
- **Validity argument.** A reader that fires on used and not on not-needed content is reporting what the answer depends on, not just what is present. Agreement that survives conditioning on truth is shared error. Agreement that survives shuffled activations is prior-driven.
- **Divergence risks and mitigations.**
  - "Used" is relative to the consumers we measure. *Mitigation:* multiple consumers where the organism has them; label per consumer.
  - The label estimator (a mean-difference direction) could miss content in other formats. *Mitigation:* "not needed" requires a small full-cell patch, not a failed 1-D swap; cells that fit neither label are *unresolved*; DAS labels on a subset, with agreement between the two label methods reported.
  - A 1-D edit that moves the answer may open a route the unmodified model rarely uses (the subspace-patching illusion). *Mitigation:* a necessity (ablation) check alongside the swap; "used" is reported as demonstrated influence, with necessity as a stronger tier.
  - Used and not-needed cases could differ by family and position rather than by use. *Mitigation:* matched contrasts within family × position (section 5), reported before any pooling.
  - Labels exist only for families with counterfactuals. *Mitigation:* the safety families get the agreement analyses but stay descriptive.

---

## 3. Questions

**Agreement**
- **Q1.** How much do readers agree, at the concept, open-set, vector and behavioural levels?
- **Q2.** How does agreement vary with family, content type, layer, position and prompt-inferability?
- **Q3.** How much agreement comes from the activation, and how much from priors or the prompt?
- **Q4.** Which reader pairs share errors beyond what their accuracies predict?
- **Q5.** Does agreement predict causal influence? Is P(used | k reader families agree) steep or flat in k?

**Adjudication**
- **Q6.** For each reader, what is P(fires | used) vs P(fires | decodable, not needed), within matched family × position? Is NLA/SAE-only content concentrated in the second?
- **Q7.** Does an ensemble predict causal use better than the best single reader? Which readers add information given the others?
- **Q8.** When a text reader surfaces something J misses, what evidence attaches to it: prompt-predictable, activation-sensitive, behaviour-linked, contradicted by context? (Overlapping tags, not a partition.)
- **Q9.** Why do NLAs hallucinate? Is it reconstructor blindness to true-vs-false minimal pairs, or verbalizer priors? Does best-of-N selection by reconstruction actually reduce hallucination?
- **Q10.** Can any reader tell the option the model endorses from one it merely considers?

---

## 4. Model and readers

**Model.** Qwen3.6-27B. Every meta-model exists for it, and an fp32 intervention harness already runs on it. Labels are residual-cell interventions, so no attention cuts are needed and the GDN state doesn't break them.

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

**Baselines.**
- Lucky guess.
- Prompt-only: an LLM reads the prefix and guesses.
- For every learned reader: a **no-activation** run (mean activation injected) and a **shuffled-activation** run (activation from another item in the same family).

**Stochastic readers.** Take 8 samples per cell. Self-consistency (the fraction of samples naming z) is kept as its own variable.

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

**RHUT organisms: what they add and their limits.**
- Maintain-then-ask prompts *do* have a downstream consumer (the later question turn), unlike a plain copy-task instruction. The carrier cell is the cleanest known "readable here, low leverage here" state.
- "Unused" here means *low leverage while the source is visible*. The source route may simply override a donor carrier; RHUT's own next-experiment 1 flags this conflict. The label is honest only as "decodable, not needed given the source".
- Tagged selection: all three words are in the prompt and only the tag token differs. At the carrier, the pointed word is the target and the two unpointed words are present-but-unselected. This is a direct test of whether a reader names the selected item or just whatever is in context.
- Sums: the answer word never appears in the prompt. The span is token-aligned (5 tokens), and two consumers read the same carrier. Only 8 rows exist; extend over the single-token number words two…twenty before use.
- Categories (A.3): no category member appears in the prompt. Use it as an open-set check that readers name members, not the category token.

**Same activation, different label (a ceiling on activation-only readers).** Install the source-visible run's carrier states into the source-hidden prompt, where they become used (sums 28/32 vs 4/32). The activation at the carrier is identical in both contexts, so any reader that sees only that activation gives the same output in both. Report this pair separately. It shows that "used" is a property of activation-in-context, and it bounds how high W can go at carrier cells. It is not a reader failure.

**Matched used / not-needed contrasts.** Without these, W could measure "does this reader like decision cells" instead of "does it track use". Attention only looks backwards, so the activation at a cell cannot depend on which question comes later. A matched pair at the same position must therefore differ in something *before* the cell.
- RHUT maintain / mention / ignore conditions: same word, same carrier position, different instruction, same question consumer.
- Tagged selection: pointed vs unpointed word, both present at the same carrier position.
- Report W within each family × position first; pool only through the fixed-effects model.

**Admission rules.**
- The model is correct ≥ 0.9 on both members of a pair, without chain of thought.
- Family-specific copy check: the *scored distinction* cannot be obtained by copying from the prompt. Brew, tagged selection and commitment deliberately name the candidates; what must not be copyable is which candidate is the answer. The model's own output before the read cell never states the latent.
- Tokens are aligned at the labelled cells.
- Two-hop only: the intermediate swap must beat the answer swap at mid-depth. The swap used here is the reader-blind mean-difference swap, never a J-coordinate swap, because J is later ranked on this bank. The rule is applied on dev items only, the number rejected is reported, and two-hop results are scoped to this assay-positive set.

**Cells.** Source (where z becomes available), carriers, decision (last prompt token).

**Layers.**
- Core matched layer is **L42**: the NLA and SAE exist only there, and Skip-lens is fed L42.
- Labels are computed at L42.
- J, R and logit lens are also read at 40, 48 and 56, each against labels at that layer.
- Report per layer. Never use the any-cell rule.

**Splits.**
- Templates A/B: label directions are fitted on A, labels computed on B.
- Items dev/test: every scoring choice (thresholds, summariser prompt, claim extraction) is tuned on dev and reported on test.

---

## 6. Causal labels (no reader involved)

For item i with counterfactual partner i′ (latent z vs z′), cell c, layer ℓ:

1. **Reader-blind direction (fixed contrast).** Class centroids on template half A:
   $\mu_z=\mathbb E_{x\in A,\;z(x)=z}\,[h_{\ell,c}(x)]$, and one direction per contrast, $\hat u_{zz'}=\operatorname{unit}(\mu_{z'}-\mu_z)$. Contrasts are never averaged across different (z, z′) pairs, which would cancel.
2. **Swap.** On B items, set the coordinate along $\hat u$ to the partner's value, with magnitude set by the donor:
   $h'=h+\big(\hat u\!\cdot\!h^{(i')}-\hat u\!\cdot\!h^{(i)}\big)\,\hat u$, then run the rest of the model in fp32.
3. **Effect.** Let m be the full-sequence log-prob margin, log-sum-exp over accepted spellings (never first-token scoring; in RHUT, first-token scoring counted prefixes like "R", "Tok" as answers).
   - Primary: the signed margin change $\Delta m_i=m(h')-m(h)$, in nats.
   - Normalised share: $S_i=\Delta m_i/G_i$, where $G_i=m(x_{i'})-m(x_i)$ is the whole-run counterfactual gap. Admission makes $G_i$ large and stable, because both pair members are answered correctly. It is never a cell-level patch, which is small by design at carrier cells.
   - Cell ceiling: the full-cell patch effect $\Delta m^{\text{cell}}_i$ (the partner's entire residual at c), recorded for every cell.
   - The edit has *demonstrated influence* if $S_i$ exceeds the 95th percentile of shares from norm-matched random directions.
4. **Necessity (stronger tier).** Ablate $\hat u$ at c in both x and x′, setting the coordinate to the pooled mean, and measure how much the counterfactual gap shrinks: $N_i=1-G_i^{\text{abl}}/G_i$. Necessity can fail because other cells carry z (redundancy), so it is reported as a tier, not required for "used".
5. **Decodable (item level).** Cross-fitted probe for z at c, with folds split by template. Keep each item's held-out probability $p_i$ for the true z. The item is decodable if $p_i$ exceeds the 95th percentile under shuffled-label probes. The family-level held-out accuracy is a gate, not an item label.
6. **State.** Each (item, cell) gets exactly one of:
   - **used**: demonstrated influence (step 3); tier "used + necessary" if step 4 also passes;
   - **decodable, not needed**: decodable, and the *full-cell* patch share is below a pre-set ε (e.g. 0.1). A missed direction cannot explain a small full-cell patch; redundancy still can, so this means "not needed given the other routes";
   - **absent**: not decodable, the probe detects a planted copy of z at this cell at a behaviour-relevant dose (positive control), and the full-cell patch share is below ε;
   - **unresolved**: anything else, e.g. a large full-cell patch with a failed 1-D swap (wrong direction or non-1-D format, expected for role binding), or a probe failure without a passing positive control. Unresolved cells are counted and reported per family, and never scored as used or unused.
7. **Per consumer.** Where an organism has two consumers (e.g. report and parity on the sum organism), label each separately. "Used by one consumer, ignored by another" is itself interface-type evidence.

**Why mean-difference.**
- A full patch labels the cell, not the content. So it serves only as the ceiling and to rule influence out ("not needed", "absent"), never to label content as used.
- DAS is optimised to succeed, so it would over-label.
- Mean-difference is the simplest reader-blind estimator. DAS runs as a second label method on a subset, and agreement between the two methods is reported.

**Share vs flips.** Record both. They can disagree sharply: in RHUT's 12-item question-turn stage, the intermediate's J plane carried 0.20 of the continuous effect, but removing it cut flips from 18/48 to 1/48. Primary label = share; flips reported alongside, and any cell whose label changes between the two is listed.

**Gates.**
- Write fidelity: every swap is measured after it is applied. ρ = ‖δ_realised‖/‖δ_requested‖ ∈ [0.9, 1.1], κ = cos(δ_realised, δ_requested) ≥ 0.99, fp32 hook. (RHUT's bf16 coordinate write realised only 0.53–0.59 of requested norm.)
- Label reliability (split-half agreement across the A/B template halves) must be ≥ 0.7 before any reader is scored.
- If more than a pre-set fraction (e.g. 40%) of a family's cells are unresolved, that family is excluded from W and reported descriptively.
- If mean-difference and DAS labels disagree badly, the labels aren't robust yet. Fix them first.

---

## 7. Scoring readers fairly

**Matched claim budget.** Every reader emits at most k claims per cell. Primary k = 4; also report 1, 8 and 16.
- Token readers: two scorings, reported separately. (i) Raw token ranks, judge-free (the closed-set reciprocal rank below). (ii) Top-25 → summariser (frozen prompt and model, abstain option) → top k concepts. The summariser can infer and select, so it counts as part of the method: it also gets its own prompt-only arm.
- Text readers: atomic claims ranked by how often they recur across the 8 samples (ties broken by first appearance), top k. Not the first k, which can drop the most salient claims.
- SAE: top-k features.

**Closed-set scoring (labelled families).** Each item has a candidate set: gold z plus 4 same-category distractors.
- Token readers are judge-free: reciprocal rank of z among the candidates under the reader's own scores.
- Text readers: a judge that sees only the readout and the options, never the prompt, with "cannot tell" allowed.
- Outputs per reader m: a continuous score $s_{im}$ and a binary hit $R_{im}$.

**Open-set scoring (all cells).** Pairwise claim matching by a judge that sees only the two readouts, labelling each pair same / different / unclear. Used for agreement level B.

**Positive control: planted concepts.** Before touching the bank, inject $\alpha\hat u_c$ at a neutral cell and sweep α. This gives each reader's sensitivity curve against exact ground truth. A reader that cannot detect a planted concept at doses that change behaviour cannot be blamed for missing natural content.

---

## 8. Analyses

### A1. Agreement: levels, decomposition, predictiveness

Plain words: two readers can agree because they both see real content, because they share a flaw, or because they both infer from the prompt. Measure agreement several ways, then split it into those parts.

**Four levels of agreement.**

- **A. Concept level (labelled families).** Hits $R_{im}$. Pairwise Cohen's κ; Fleiss' κ or Krippendorff's α across all readers. The causal labels attach to this level.
- **B. Open-set level (every cell, including the safety families).** k claims per reader, matched by the pairwise judge. Also: Jaccard and rank-biased overlap of top-k token lists, and embedding similarity of claim sets. This is the no-ground-truth atlas.
- **C. Vector level.** Each reader with a write map (J/R rows, Oracle AR reconstructor, NLA reconstructor, SAE decoders) gives a reconstruction direction $\hat r_m$ from its own claims at a cell. Compare readers by $\cos(\hat r_m,\hat r_{m'})$. At the dictionary level, compare principal angles between each reader's atom subspaces on a family. This catches readers that say the same thing in different words, and readers that use the same word for different directions.
- **D. Behavioural level.** Swap along each reader's direction for z→z′ and compare which output moves. Two readers "agree causally" if their edits have the same effect.

**Shared signal vs shared error (needs labels).** With causal state $y\in\{\text{used},\text{decodable-not-needed},\text{absent}\}$ (unresolved cells excluded and counted), conditional independence (the Dawid–Skene assumption behind majority voting) predicts, within each y,

$$\hat P(R_m=R_{m'}=1\mid y)=P(R_m=1\mid y)\,P(R_{m'}=1\mid y).$$

Excess agreement within y is

$$E^{(y)}_{mm'}=P_{\text{obs}}(R_m=R_{m'}=1\mid y)-P(R_m=1\mid y)\,P(R_{m'}=1\mid y).$$

Positive excess can also come from shared item difficulty or from labels that are incomplete, not only from a shared flaw. Check whether it survives adding difficulty proxies (prompt-only correctness, token length, label margin), and read it as "shared dependence" unless it is within *absent* and survives.
- Positive excess within **absent** means a shared false positive: the pair hallucinates or infers together.
- Positive excess within **used** means shared sensitivity to the same content types.

**Shuffle-sensitive agreement (a diagnostic).**

$$\Delta\kappa_{mm'}=\kappa_{mm'}(\text{real})-\kappa_{mm'}(\text{shuffled activations}).$$

Agreement that survives shuffling comes from family priors, not from reading the activation. κ differences are not a clean decomposition, so this is reported as a diagnostic, next to each reader's κ with prompt-only.

**Test–retest.** Report each stochastic reader's self-agreement across its 8 samples directly, next to its agreement with other readers. (No disattenuation: the correction is not a valid general ceiling for κ.)

**Does agreement predict causal influence?** Count the independent reader *families* that name z. J/R count as one family; Oracle/NLA/Skip count as one; SAE is one; logit lens is one. Plot $P(\text{used}\mid k\text{ families agree})$ for k = 0…3, with prompt-only agreement as a separate curve.
- Steep and monotone: majority agreement is a usable proxy, and its usability can be quantified.
- Flat, or rising only with prompt-only agreement: agreement tracks inferability, not use.

Also fit Dawid–Skene without labels. Compare its inferred truth to the causal labels, and compare reader rankings (Kendall τ).

**What drives agreement: the atlas.**

$$\text{logit}\,P(R_m=R_{m'})=\beta_0+\beta_{\text{family}}+\beta_{\text{content type}}+\beta_{\text{layer}}+\beta_{\text{position}}+\beta_{\text{token len}}+\beta_{\text{inferable}}.$$

Expected: high agreement on entities at decision cells, collapsing on numbers and bindings; agreement rising with depth; a large prompt-inferability effect.

### A2. Causal accuracy per reader

- Sensitivity: $\text{Sens}_m=P(R_m=1\mid\text{used})$
- Specificity: $\text{Spec}_m=P(R_m=0\mid\text{absent})$
- **Current-task-use selectivity:** $W_m=P(R_m=1\mid\text{used})-P(R_m=1\mid\text{decodable-not-needed})$, computed within family × position on the matched contrasts first, and pooled only through the fixed-effects model.

High W: the reader reports content the measured answer depends on. W ≈ 0: it reports decodable content whether or not this answer needs it. Low W does not by itself mean the reader reads "outside the workspace". Nanda's hypothesis predicts $W_{\text{NLA}},W_{\text{SAE}}\ll W_J,W_{\text{Oracle}}$; a confirmed gap supports "NLA reports content the current task doesn't use", not "NLA reads outside the workspace".

### A3. Ensemble and conditional contribution

Held-out model:

$$\operatorname{logit}P(\text{used}_i)=\beta_0+\sum_m\beta_m s_{im}+\text{family FE}.$$

Per reader, **incremental predictive utility** $\Delta\text{AUC}_m=\text{AUC(all)}-\text{AUC(all}\setminus m)$. This is the added discrimination of this fitted ensemble at this sample size, not conditional mutual information. Cross-validation folds are grouped by template and source problem, so near-identical counterfactual pairs never straddle train and test. Fit the same model for the *decodable* label. If NLA adds information about decodability but not about use, that answers Nanda's question in one line.

### A4. Where reader-only claims come from

For each claim a text reader makes and J doesn't, attach overlapping evidence tags. These are not a partition: a real activation signal can also be prompt-inferable.
- **Prompt-predictable.** Prompt-only, or the same reader on shuffled activations from the same family, produces it.
- **Activation-sensitive.** It appears with the real activation at a higher rate than under shuffled activations.
- **Behaviour-linked.** The claimed content is labelled *used* at that cell (section 6).
- **Contradicted by context.** The counterfactual or the model's output contradicts it. This may be a considered-but-rejected option rather than a confabulation; cross-check against the commitment family.
- **J-adjacent (descriptive only).** Its constituent tokens rank highly under J at the same cell. A reconstructed vector's J alignment is not used, since it does not show the natural activation expressed the claim that way.

Claims with no decisive tag stay **unresolved**. Output: tag frequencies and co-occurrences per reader and family.

### A5. NLA mechanism

- **Reconstructor blindness.** For hallucinated claims, on minimal pairs:
  - reward difference $\Delta(c)=r(s_c)-r(s_{\bar c})$;
  - discriminability $d(c,\bar c)=\|\mathcal D(s_c)-\mathcal D(s_{\bar c})\|/\|\mathcal D(s_c)-\mu\|$.
- **Geometric faithfulness.** $\cos(\delta_{\text{NLA}},\delta_{\text{true}})$, where $\delta_{\text{NLA}}=\mathcal D(C[c\to c'])-\mathcal D(C)$ and $\delta_{\text{true}}=h(z')-h(z)$.
- **Selection experiment.** Using the 8 samples per cell, compare the hallucination rate on held-out verifiable claims for three choices: a random sample, the highest-FVE sample, and the sample with the highest claim-grounding score. Blindness on single claims (Δ(c) ≈ 0) does not by itself show FVE selection fails, because whole-sample FVE could still correlate with accuracy through other claims. This experiment answers the scaffold question directly.

### A6. Commitment

Compare the J margin $\kappa_J=\langle v_A-v_B,h\rangle$ and the other readers' endorsement claims against which option is causally used (lens-free commitment direction). Swap J rows vs swap the lens-free direction. Question: which reader can tell "considering" from "decided"?

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

**Stop and redesign if:**
- the bench reproduction fails;
- planted-concept curves show a reader detects nothing;
- label reliability stays < 0.7 after adding DAS labels;
- on the slice, most cells are unresolved or the matched contrast shows no used / not-needed difference in the labels themselves.

---

## 13. Open questions

1. Does Celeste's NLA ship its reconstructor? A4, A5 and vector-level agreement for the NLA need it. Without it, those analyses fall back to the Oracle only.
2. Are there auto-interpretation labels for the L42 SAE/BSF? If not: use J-projected decoders, or LLM-label ~2,000 features.
3. Is there a phrase-J checkpoint to run as an arm?
4. Which judge models are available for text readers and open-set matching (Gemini, Sonnet)?

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
