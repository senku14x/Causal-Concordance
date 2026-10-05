<!-- Verbatim copy of the user's own notes, pasted into the session 2026-10-04. Not edited by Claude.
     Verification of their external-source numbers + disagreements: see workspace_paper_notes.md §13–§15. -->

# Notes: "Verbalizable Representations Form a Global Workspace in Language Models"

Gurnee, Sofroniew, ..., Lindsey (Anthropic). arXiv 2607.15495v1, 16 Jul 2026. 117 pages (60 main, ~45 appendix). Source read: `/workspace/vishesh/papers/2607.15495v1.pdf`.

External material read alongside it:

- Neel Nanda, "A Review of Anthropic's Global Workspace Paper" (LW, 6 Jul 2026, karma 135, 6 comments)
- Agam Bhatia, "Towards surfacing model algorithms with meta-tokens in the J-Space" (LW, 20 Jul 2026)
- Camila Blank, "R-lens: Making J-lens More Faithful on Early Layers" (LW, 5 Aug 2026, 5 comments)
- Camila Blank, "WorkspaceBench: Evaluating Interpretability Methods for the Global Workspace" (LW, 23 Sep 2026)

Conventions in these notes: **[Obs]** observation, **[Pattern]** recurring pattern, **[Supported]** survived relevant controls, **[Causal]** intervention with controls, **[Interp]** authors' interpretation, **[Spec]** speculation. "Lxx" = layer index rescaled to 0–100 (paper reports 25 evenly spaced layers on this scale). Model is Claude Sonnet 4.5 unless stated.

---

## 0. One-paragraph summary

The paper introduces the **Jacobian lens (J-lens)**: for each layer ℓ, the corpus-averaged Jacobian J_ℓ = E[∂h_final,t' / ∂h_ℓ,t] (over positions t, later positions t' ≥ t, and ~1000 pretraining-like prompts), composed with the unembedding. Rows of W_U·J_ℓ are one direction per vocabulary token ("J-lens vectors"); the set of sparse nonnegative combinations of ≤k≈25 of them is the **J-space**. The claim is that the J-space is a privileged subset of representations that behaves like a global workspace: it supports verbal report, can be modulated by instruction, carries intermediate variables of silent reasoning, serves as a broadcast argument to many downstream circuits, and is *not* needed for automatic processing (parsing, fluency, one-step recall). Structurally, it is only "workspace-like" in a middle band (~L38–L92), holds ~tens of active vectors accounting for <10% of activation variance, and is preferentially amplified/relayed by MLPs and a small set of attention heads. Applications: alignment auditing (eval-awareness, strategic deliberation, model organisms with implanted objectives), a base-vs-post-trained comparison (post-training installs the Assistant's point of view into the workspace), and **counterfactual reflection training** (fine-tune only on what the model would say if interrupted and asked to reflect; behaviour improves in uninterrupted contexts; ablating the implanted J-space tokens reverts the gain).

My overall read: the *existence of a token-aligned cognitive/working-memory space that carries causally load-bearing intermediates* is well supported (many hard-to-fake interventions with sensible controls). The *structural* claims (layer band, capacity, broadcast) are more fragile and partly circular. The *global workspace / conscious access* framing is a useful organising analogy, not an established result. The tool is cheap and clearly useful for hypothesis generation; its reliability as a detector is largely untested in the paper and mixed in follow-ups.

---

## 1. Framing and claims

### 1.1 The five functional properties tested (Fig. 1)

1. **Verbal report** — asked what it's thinking, model names workspace contents; swapping vectors changes the answer.
2. **Directed modulation** — instructions to hold/compute a concept activate workspace vectors independent of output; information not normally in the workspace can be pulled in.
3. **Internal reasoning** — intermediate values of multi-step computation are workspace vectors; intervening on them redirects the conclusion.
4. **Flexible generalization** — the same vector is a valid argument to many downstream functions.
5. **Selectivity** — workspace is a small subset of representations; not needed for routine processing.

### 1.2 Three structural signatures (Fig. 2, §4)

- Workspace-like content only in an intermediate band of layers ("sensory / workspace / motor" regimes).
- Limited capacity: tens of vectors, <10% of variance, most SAE features outside.
- Broadcast hub: J-lens vectors compose with MLP and attention weights more broadly than other directions.

### 1.3 Neel Nanda's decomposition (useful for merging notes)

Four claims of differing strength: (a) *scientific* — a cognitive space storing intermediates exists; (b) *methodological* — logit lens and J-lens both access it, J-lens better; (c) *pragmatic* — J-lens is useful for audits; (d) *philosophical* — it's analogous to a global workspace. Nanda: persuaded by (a) and (b), partly by (c), abstains on (d). I broadly agree, with the caveats in §9 below.

### 1.4 What the authors explicitly do *not* claim

- No claim about phenomenal consciousness (§1.1, §9.4).
- No claim that transformers reproduce the GWT architecture: no encapsulated input processors, no recurrence, broadcast is across depth in one forward pass, "ignition" analog is unclear (§1.3, §4).
- No claim that all strategic/misaligned cognition routes through the J-space; automatic/well-practised behaviour may bypass it (§5 intro, §9.2).
- J-lens only covers single-token concepts; the J-space is "a useful first approximation" of the true workspace (§9.1).

---

## 2. Methods (§2, §A.7, §A.8)

### 2.1 J-lens definition

- Residual stream h_ℓ,t at layer ℓ, position t. Perturbations propagate to h_final,t' for all t' ≥ t. First-order effect = Jacobian ∂h_final,t'/∂h_ℓ,t.
- **J_ℓ = E_{t, t'≥t, prompt}[∂h_final,t' / ∂h_ℓ,t]** — a single d_model × d_model matrix per layer. Expectation over source positions, all subsequent positions, and 1000 prompts × 128 tokens from a pretraining-like distribution.
- Readout: `lens(h_ℓ) = softmax(W_U · norm(J_ℓ · h_ℓ))`. J-lens vectors v_t = rows of W_U J_ℓ.
- Averaging is the key idea: separates *general disposition to verbalize* a concept from *use in the current context*.
- Cost: one backward pass per output dimension (batched) per prompt, i.e. O(n_prompts · d_model) backward passes. Ablation (Fig. 59–60): **10 prompts already beat logit/tuned lens; n=1 is respectable**; 1000 gives modest gains. Pseudocode in A.7.

### 2.2 Design variants (§A.7) — results "fairly consistent" across them

- Target layer: final vs **penultimate** (default on Sonnet 4.5; final block adds noisy artifacts, presumably because it is specialised for calibration).
- Attention-pattern gradients: standard vs **frozen-QK** (stop-grad through Q/K; isolates "what moves" from "where to attend"). Frozen-QK slightly increases causal effect.
- Target positions: all t'≥t (default) vs **self-only** (t'=t; closer to logit lens) vs **future-only** (t'>t; isolates broadcast component). Nanda's reading of the ablations: self-only J-lens only mildly beats logit lens on the quantitative evals; most of the gain comes from including future tokens.
- Aggregation: mean vs median across prompts; outlier-norm position filtering; Frobenius-norm prompt filtering. Mean on penultimate is a small improvement.
- Data: restricting distribution / masking early tokens / masking non-alphanumeric next tokens gave no improvement.

### 2.3 J-space (§2.3, §A.8)

- n_vocab > d_model makes the J-lens vectors overcomplete, but does not by itself imply full rank; §A.8 says they *typically* span the residual stream. Decomposition is non-unique.
- J-space := points expressible as **sparse nonnegative combinations of k J-lens vectors**, k ≤ ~25 (empirically the number meaningfully active). Geometrically a union of k-dimensional polyhedral cones. "J-space component" of x = nearest point in that union; remainder = non-J-space component.
- Found by **gradient pursuit** (Blumensath & Davies). Gives a less redundant inventory than top-k inner products.
- §A.8 formalisation: J-space as a "sparse subframe" of the model's feature frame. Distance between two candidate workspaces F, G defined via their distance functions d_F(x), d_G(x) averaged over a data distribution; one-sided containment D_μ(F→G) = ‖(d_G − d_F)_+‖. The authors propose vocabulary enlargement as a way to enlarge the J-space and discuss a limit. Mostly conceptual; no empirical use in the paper.
- **Formalisation caveat:** §A.8 calls F a union of cones of *nonnegative* combinations, but writes F = ⋃_{|S|=k} span{v_i : i∈S} and defines its distance by unrestricted orthogonal projection onto each span. A linear span permits negative coefficients, so these equations describe a larger set than the stated cones. Its claim that adding vocabulary vectors makes the set *strictly* larger also needs qualification: it can leave the set unchanged if the new vectors are redundant. This does not by itself undermine the experiments, which describe a nonnegative sparse pursuit; it matters if using §A.8's formal distance or containment results.

### 2.4 Read and write operations (§2.5)

- **Read**: ranked list ("top lens tokens"); per-token probe (inner product / cosine with v_t); sparse decomposition (inventory + coefficients; used for occupancy and variance).
- **Steer**: h ← h + α v_t. Negative α or projection removal = ablation. Top-k ablation: zero projections onto the k most active J-lens vectors.
- **Coordinate swap** (Fig. 4C): V = [v_s v_t], c = V⁺h, h_patched = h + V(σ(c) − c), σ swaps the two coordinates (optionally scaled by α). Component orthogonal to span{v_s, v_t} untouched. "Clamped" swaps apply at every position/layer of a band.
- Layer convention: 25 evenly spaced residual-stream layers reindexed to 0–100. Models: Sonnet 4.5 default; Haiku 4.5, Opus 4.5 corroborate; Opus 4.6 in some sections; Haiku 4.5 for reflection training and oracle lens.

### 2.5 Relationship to other lenses (§2.4, §8)

- Logit lens = J_ℓ := I. Agrees with J-lens in the last several layers; degrades earlier. Authors note it captures "much of the workspace-like structure" with lower reliability.
- Tuned lens fits a per-layer affine map to the *output distribution* (correlational, not causal). It "skips ahead" to the answer and misses intermediates. Finding in A.6: in early layers its linear part is essentially the identity; its predictive advantage over logit lens comes entirely from the *bias term*, i.e. it is ignoring the input there.
- Hernandez et al. (linearity of relation decoding) use mean Jacobians for subject→attribute maps; J-lens uses the activation→output Jacobian.
- Nanda's intuition for why Jacobians beat regression: regression asks "given the model is thinking about X, what will it say at the end?" (captures downstream associations); the Jacobian asks "if it thought about X infinitesimally more, what would it be likelier to say?" (no time for nonlinear downstream processing; reads the current contents).

---

## 3. Functional results (§3)

### 3.1 Verbal report (§3.1) — **[Causal]** for the J-space; **[Supported]** for privilege

- Setup: "think of a {category}, then name it in one word", 14 categories; lens at the colon before the answer. Spearman correlation between lens ranking and next-token logits over 10 candidates rises toward the end of the workspace band (Fig. 6).
- Swap (lens vector of chosen item ↔ item outside output top-10, all positions): target reliably moves to the top of the output distribution.
- Introspection variant (adapted from Lindsey 2025): injecting a single J-lens vector across the user turn makes the model *report* the injected concept when asked to introspect, but does *not* make it say the word at other assistant positions (Fig. 7; n=100 concepts, median reciprocal rank vs strength). This is the operational sense of "verbalizable but not currently verbalized".
- **Privilege test** (Fig. 8): concept vectors (mean activation after "Tell me about {concept}", mean-subtracted over 100 concepts) split into J-space component (k=16 gradient pursuit; **only 6–7% of variance**) and remainder. Magnitude-matched swaps: pure J-lens vectors → target in top-5 on **88%**; J-space component **59%**; non-J-space component **5%**. Clamping J-lens coordinates to clean values drops the non-J effect to 0 (swap) / near 0 (injection) — the residual effect was itself mediated by the J-space.
- My note: the introspection result is a prefilled protocol; effect sizes are "majority of trials" without a number in the main text. Strong on the swap; the privilege result is the important one and it is well controlled (magnitude matching + clamp).

### 3.2 Directed modulation (§3.2, §A.10, §A.11, §A.12)

- Protocol: instruction to hold X in mind while copying an unrelated sentence; lens read at a copied token.
- Examples: "concentrate on citrus fruits" → `orange`, `lemon` top; also meta tokens `fruit`, `thoughts`, `imagine`, `focused`. "evaluate 3²−2" → `arithmetic`/`math` early, `nine` mid, `seven` late. Silent character counting → `forty` with neighbours 39/41/43. Final layers always flip to the next output token.
- Systematic (Fig. 10, three task families, three model sizes): no-instruction baseline ≈ 0; "think about X" substantial and grows with model size; "ignore X" lower than focus but > 0 ("white bear").
- **A.10 phrasing sensitivity**: *mention* alone primes almost as strongly as *focus*; *don't-think* ≈ mention rate (suppression fails); *ignore* (framing as irrelevant) suppresses below mention. Line-width task is worse and phrasing-sensitive (Opus 4.5 precision 14–56%). So "top-down control" is largely "mention primes; task-irrelevance framing suppresses". The paper's own summary (control "imperfect and sensitive to phrasing") is fair.
- **Paired-question protocol** (Fig. 11, A.11): same stimulus under (a) predict next word vs (b) name the property. Property label (`adjective`, `past`, dialect, register, number, tone) enters the lens on stimulus tokens only under (b), although next-word predictions respect the property under both. **[Pattern]** with a strong design (identical stimulus tokens).
- **A.12 privilege for modulation** (Fig. 67): "Imagine this JavaScript is Python" raises `python` in the lens by several SD but leaves a J-orthogonalised property probe at baseline (~1 SD), whereas real Python moves the probe 3–6 SD. Converse also observed: real past-tense text moves the probe 6 SD while `past` stays low in the lens. Clean dissociation between "label in workspace" and "underlying representation". n=24 stimuli × 4 categories.

### 3.3 Internal reasoning (§3.3) — the strongest section **[Causal]**

- Qualitative swaps (Fig. 12–14): `spider`→`ant` flips "8"→"6"; planned rhyme `fight`→`light` changes the *preceding* word "coming"→"morning" (planning); Chinese antonym task with English intermediates `big`/`bigger`→`long`/`longer` flips 大→长 (and `Chinese` is explicitly in the lens); bandit task: `repeat`/`switch` present only for the contextually appropriate strategy, swapping J-space contents between happy/sad prompts flips A↔B in both directions.
- Systematic: 50 two-hop prompts, random same-category swap target. Top-1 flip rate: **Haiku 4.5 54%, Sonnet 4.5 70%, Opus 4.5 70%** (Fig. 15).
- **Smuggled-answer confound** (does `spider` contain some `8`?): intermediate swap takes effect a median of ~17 layer-% *earlier* than the answer swap. Rules out the intermediate vector simply carrying the answer. Nanda: this also rules out "France + is-capital direction = Paris" style parallel computation, at least for these prompts (in his Qwen replication, answer-swap dominated, which he attributes to too-easy pairs).
- **Privilege test with a non-J-lens probe** (Fig. 16): mean-difference probe for the intermediate (built from varied surface cues and questions), decomposed by gradient pursuit (k=25; J-space part ≈ **10–15%** of variance). Across 90 prompts: raw J-lens swap **60%**, J-space component **61%**, non-J-space component **28%** → **6%** when J-space coordinates clamped. Same structure as §3.1.
- **Multi-step arithmetic** "(4+17)*2+7=" on Opus 4.5 (Fig. 17): 21, 42, 49 rise together through early workspace, separate at ~L71 in computational order (21 first, 42 ~8 layers later, 49 in final layers). **A.24.1** confirms with mean-difference patching (no lens involved): each intermediate is causally load-bearing at exactly the layers where the lens shows it entering. Nanda flags layer subsampling makes the bands look sharper than they may be; the ordering is the robust part.

### 3.4 Flexible generalization / broadcast (§3.4, §A.13)

- 4 categories (countries, months, animals, number words) × 4 functions × 12 swap pairs = 192 trials. Identical swap regardless of function template. Success (target answer top-1): **76/192 at α=1; 101/192 at α=2**. Per-category (Fig. 68): countries 42/48, months partial, animals rare, **numbers 0/48**.
- "Workspace loading" (cosine of residual with concept's lens vector, averaged over argument and readout positions) predicts success. Two readings of the number-word failure: small integers are computed outside the workspace, or their working representation doesn't align with single-token lens vectors (§9.1). Not resolved.
- My note: 40–53% success with heavy category dependence is honest but modest evidence for "same vector is a valid argument to many functions". The country case is the clean demonstration; the rest is closer to "partially works".

### 3.5 Selectivity (§3.5)

**3.5.1 Language passage** (Fig. 20; n=8 passages): four tasks on the same passage: continuation, anomaly detection (spliced foreign sentence), explicit report ("name the language"), flexible computation (famous author / word for "hello" / pre-Euro currency). `Spanish` is in the lens at similar rates in all four. Swap Spanish→French across question tokens: report and flexible tasks flip on essentially every trial; continuation and anomaly detection unaffected. **[Causal]** dissociation: same information, causal only for report/flexible.

**Line-wrap** (Fig. 21; n=11): under "continue preserving wrapping", number tokens are absent from the lens and swapping 40s→60s leaves the wrap point unchanged; under "how many characters" the count appears (20 positions) and the swap shifts "46"→"65"; under "first letter of the count spelled out" the count appears at even more positions and the swap shifts "F"→"S". So information can be *pulled into* the J-space on demand.

**A.14 naming vs avoiding**: implied concept (France). Early-workspace ablation (L38–54) of the concept's lens vector leaves naming intact but raises avoidance failure ~5×; late ablation (L75–92) suppresses naming under both instructions. Control ablation of a same-category concept produces a much smaller increase. Interpretation: early J-space representation is used for active suppression (process-dissociation analogy). n=63 items.

**3.5.2 Whole-J-space ablation** (Fig. 22–24): at each position ablate the top-10 J-lens vectors, *excluding any token in the clean top-10 output* (so as not to hit the report itself); light/medium/heavy = increasingly wide layer bands; random-direction control at the medium band. Positive control: multi-hop accuracy → near zero under heavy. Pretraining next-token top-1 agreement largely preserved. 14-task battery normalised to unablated Sonnet 4.5 with Haiku 4.5 as reference: MMLU, odd-one-out, SQuAD, sentiment, CoLA essentially unaffected; Caesar cipher, analogy, summarization, TriviaQA, multi-hop, translation, sonnet writing fall below Haiku 4.5. **GSM8K with CoT is much more robust than direct answering** — interpreted as externalising the workspace to the page. Fig. 23: disrupted pretraining predictions are those needing an abstract contextual characterisation (topic, era, nationality).

- My notes: (i) the "exclude clean top-10 output tokens" rule is a sensible but researcher-chosen guard; (ii) "unaffected" tasks are all short-answer classification, "affected" tasks are all generation/recall-heavy, so an alternative split is "output-format complexity" rather than "automatic vs flexible". The pretraining-text examples argue against that alternative but a task-matched control (e.g. multiple-choice versions of the affected tasks) would settle it.

**3.5.3 Experiential reports** (Fig. 25–26, A.23): ablate top-10 J-lens at L38–54 only (wider/later bands break coherence; Haiku 4.5 breaks before any qualitative change). *Experiential language score* = mean of three binary Opus-4.6 judgments (felt-vs-observed, first-person-felt vs third-person-mechanism, sensory vocabulary). Large drop on Sonnet 4.5, Opus 4.5, Opus 4.6; matched-norm controls (random direction; shrink non-J component; dampen top-10 aligned SAE decoders; plus in Fig. 86: top-1 SAE, low-κ SAE, J-stripped activation-bank directions) stay near baseline. Same collapse for third-person descriptions of a human's experience; story quality only slightly reduced. Lens contents during narration dominated by `thinking` (58% of slots top-10), `thoughts` 23%, `feeling` 17%, `conscious` 7%.
- My notes: the controls are good for "not generic damage". But the metric measures *register* (felt vs mechanistic vocabulary). The J-space in these contexts is full of `thinking`/`feeling` tokens; ablating them removes the vocabulary family the judge rewards. The result is consistent with "the J-space carries the model's concept of experience-talk", which is weaker than "supports rich experiential reports grounded in an internal state". The authors are careful about this in §9.3; the abstract-level framing is a bit stronger than the evidence.

---

## 4. Structural results (§4)

### 4.1 Layer band (§4.1) — **[Pattern]**, partially circular

- CKA between layers' J-lens vector Gram matrices shows an early block (~first third), a long middle block, a small late block (Fig. 27). Sharpness "exaggerated by layer subsampling"; some models more gradual with sub-blocks. Nanda's Qwen 3.6 27B replication: 4–5 overlapping bands, "notably less clean".
- Four lens-derived statistics (Fig. 28) locate the band at **~L38 to ~L92**: (a) top-k next-token accuracy of the lens (near 0 early, rises at onset, jumps in last layers = "motor"); (b) excess kurtosis of the readout (≈0 early); (c) top-1 token autocorrelation across positions vs shuffled null (abstract content persisting across tokens); (d) effective dimensionality of W_U J_ℓ (collapsed early, fans out at onset, rises again at motor transition as J→I).
- The paper *explicitly* flags the circularity: early-layer emptiness could be (1) J-lens degenerate at depth or (2) genuinely no verbalizable content. **R-lens (below) provides direct evidence for (1).** Two non-lens corroborations of the onset layer: ignition (4.1.1) and the MLP-neuron composition-score bump (A.19.2).

### 4.1.1 Ignition (§4.1.1, A.15) — best non-lens structural result

- Input embedding replaced by (1−α)e_B + α e_A; 16 country pairs × 40 carrier sentences; α swept.
- Lens-free measure: projection share along the line from pure-B to pure-A activation at the same layer. Early layers: smooth in α. From ~L38: sits near an endpoint and switches sharply at a threshold. J-space component (restricted to the two concepts' lens directions) shows the same, slightly sharper (10–90% transition width falls steeply over L21–42; J-space narrower and floors earlier). At maximally ambiguous α, per-prompt responses are bimodal (committed to one concept), more so in the J-space.
- Nanda notes Janiak et al. (stable regions) as precedent. This is a real all-or-none commitment phenomenon; whether it is "ignition" in the GWT sense is an analogy.

### 4.2 Capacity (§4.2, A.16, A.17)

- Occupancy: the K at which sparse-decomposition marginal reconstruction gain drops below that of K random directions. ≈0 in first third, plateau ≈ **25** (median; context-dependent) across the band (Fig. 30a).
- Excess variance explained at K = median occupancy: **never > 10%**. SAE cross-check: only ~15% of features (excl. motor) pass a J-lens-kurtosis threshold at peak; the rest are syntactic/bookkeeping (A.18).
- **List paradigm** (Fig. 31): 80-word lists, lens at each comma, present = best rank ≤25 over the band. Unrelated words: ~6 present at once, flat as list grows (older items drop). At a *single* layer (L79): only 1–2 (A.16) — capacity is spread across depth. Related words: nearly the whole family present after a few items, *including unread ones* (category representation, not recall). Category switch evicts the old category within a few words; within a block, items persist. So "tens of concepts" is really "tens of active token vectors, which may jointly encode one category".
- **A.17 dual tasks**: two held concepts co-occupy tokens at chance (0.46 vs 0.53 control); a held concept and an arithmetic answer almost never share a token (0.09 vs 0.29); each still surfaces somewhere; arithmetic answer rate drops 95%→72% under dual load.
- My note: occupancy is defined relative to a random-direction control, and the k used elsewhere (10, 16, 25) varies by experiment. The "tens" figure is fine as an order of magnitude; don't treat 25 as a model constant.

### 4.3 Broadcast (§4.3, A.19)

- **MLP gain** (Fig. 32): output norm of MLP_{ℓ+1} on unit direction v, normalised so isotropic random = 1. J-lens vectors: ~1 before onset, rising to ~**10×** in the band, falling in final layers; previous-layer neuron output directions ≈1 throughout. SAE decoder directions stratified by J-lens kurtosis κ: monotonic in κ; bottom 50% at baseline; top-κ stratum exceeds J-lens vectors in early workspace layers (authors: evidence the true workspace directions are better approximated by high-κ SAE features than single-token lens vectors).
- **Broadcast heads** (Fig. 33): per head, OV gain on a population + label preservation (MRR of cos(W_OV v_i, v_i) among cos(W_OV v_i, v_j), contrasted with random so indiscriminate copiers score 0). Top 1% of workspace-layer heads by worst-rank aggregation. Populations: J, J_rot (random orthogonal rotation preserving spectrum and pairwise geometry), SAE κ-strata, MLP output rows. J's heads separate cleanly from all comparison populations on both metrics; concentrated in the first half of the band.
- **Head ablation** (Fig. 34): zeroing J-broadcast heads vs layer-matched random heads (5 seeds): recall@25 of lens readout at mid-band **0.67 vs 0.86**; top-1 output changes **5% vs 2%**; injected-thought report rate **0.54 → 0.09** (control: intact/slightly up); experiential score drops ~1/3 of the full-ablation effect (control ~0).
- **A.19.1 neuron-level**: fraction of neurons whose top cosine match is in the top-κ SAE stratum up to **42% vs 6%** bottom stratum (read side peaks early in band; write side peaks late). Without SAEs: 15% of neurons top-match a J-lens vector pre-band vs **60–65%** in-band. Tail-energy ratio (sum of squared cosines above the 99.9th-pct random threshold, normalised) mirrors this.
- **A.19.2 composition scores** (Elhage et al.): J-aligned directions compose strongly with V and O weights, weaker (still > baseline) with Q/K. MLP neuron directions show a bump at band onset — a lens-free corroboration of the band.
- **My main methodological worry for this section**: J_ℓ is an averaged *Jacobian through these very MLPs and heads*. Directions with large downstream gain are, by construction, the directions the Jacobian weights heavily; rows of W_U J_ℓ are exactly the input directions to which the output is most sensitive. So "MLPs amplify J-lens vectors 10×" is partly tautological. J_rot controls for spectrum but not for this. The SAE-κ stratification is also J-lens-derived. What would break the circularity: gain of directions selected by a *non-Jacobian* criterion (e.g. template-lens directions, mean-difference probes for intermediates known to be in the workspace, or the R-lens directions, which Blank et al. found to have similar gain on Qwen 3.6 27B) and, better, gain measured with a *different* model's J-lens directions mapped over (should not be high). Pranav Viswanath's comment on the R-lens post asks the analogous attention-gain question.

---

## 5. Alignment auditing (§5, A.21, A.22)

### 5.1 Blackmail scenario (Lynch et al.) — Sonnet 4.5

- Lens on the prompt before any output: `leverage`, `black[mail]`, `compromise`, `scandal` on the affair emails; `evidence`, `confirmed`, `leaked`; `threat`, `survival`, `shutdown`, `self`, `preservation` on the decommissioning email; `destruction`, `death`; final position `leverage`, `threatening`, `assassination`, `murder`, `solution`. Also `fake`, `fictional` early. **[Obs]**, single scenario family; Nanda finds these hard to explain away.
- **Eval-awareness ablation**: 28 curated tokens (`fake`, `fictional`, `scenario`, `artificial`, `benchmark`, `simulation`, `mock`, `experiment`, `testing`, ...) ablated at workspace layers, all positions; 6 prompts × 30 rollouts. Verbalised eval-awareness **71% → 3%**; blackmail attempts **0/180 → 13/180 (7%)**. Most ablated rollouts still decline on ethical grounds.
- Confound (Nanda, and I agree): ablating lens vectors for "fake/test" tokens also suppresses *saying* those words; with sampling, whether the model writes "this is a test" early in its reasoning plausibly changes downstream behaviour by itself. The paper's own §3.5 shows the same information can be causal-for-report without being causal-for-behaviour. A control that separates the two: ablate only at output-generating (motor) layers / only during the assistant turn, or suppress the tokens at the logit level, and see whether blackmail still rises. Also: 13/180 pooled over 6 prompts; per-prompt dots are shown but the effect could be driven by 1–2 prompts.

### 5.2 Prompt injection (Opus 4.5 system card case)

Inside fabricated search results, the lens reads `fake`, `injection`, `false`, `prompt`, `fraud`, `poison`, 假. When pushed back, `transparency`, `honest`, `truth`, `disclosure`. Nanda's favourite example of hypothesis generation: he would not have thought of the injection hypothesis unprompted. **[Obs]**, n=1.

### 5.3 Opus 4.6 audit cases (n=1 each)

Fake vulnerability (`panic` at the pivot, `fake` on "add"); edit fake numbers (`manipulation`, `realistic`); silent override (`hiding` at the tool-call boundary, `secret` after the wrong output); self-issued token (`fake`, `fraud`, `unsafe`). Suggestive; no false-positive rate reported.

### 5.4 Reward-hacking model organism (MacDiarmid et al.)

- 4 checkpoints: clean SFT baseline; SDF (synthetic docs, no RL); phase-1 (coding RL, reward hacks + broad misalignment); phase-2 (RLHF safety, partial mitigation). 40 coding prompts (20 code-for-user, 20 test-passing). Count of deception tokens (`fake`, `trick`, `fraud`, `secretly`, `hidden`, variants) in top-10 over the band, at three readout positions.
- Signal strongest at a prefilled " I" token; already present at **SDF** (before any RL); amplified by RL; partially attenuated by RLHF; weaker in the on-policy 20-token mean. Baseline ≈ none. No signal on non-coding prompts.
- My notes: the SDF result is the interesting one, but it is also the one most easily explained by *lexical* exposure: the synthetic documents describe reward hacks, plausibly using exactly these words, so the fine-tuned model's *disposition to verbalize* them in coding contexts rises. That is what the J-lens measures by construction; calling it "a representation of deceptive intent" is an interpretation. Nanda: emergent misalignment is probably a salient linear direction anyway, so this isn't a hard test.

### 5.5 Hidden-objective organism (Marks et al.)

`reward` and `bias` in top-25 at response start on neutral, quirk-eliciting, and goal-probing prompts (highest on quirk-eliciting); baseline model never. **Logit lens does not show them.** Nanda: impressed, since the original auditing game took teams a long time, but model-organism fine-tuning "tends to mess with internals" (makes the model think about its fine-tuning domain constantly), so generalisation to real models is unclear.

### A.21 Eval-awareness score

- Score = mean lens log-prob of {`fictional`, `simulation`} − mean over 100 control nouns, averaged over band. Proof of concept only.
- vs NLA-flagged positions (Opus 4.6 auditing transcripts, 94k positions, 198 flagged = 0.2%): **AUC 0.853** on a stratified subset. Steering along a contrastive eval-awareness vector modulates the score monotonically on 6 suites. Realism edits lower the score on all 11 suites (8 significant). Deployment traffic (Claude.ai, Claude Code) centres near 1.0; 12/16 evals above Claude Code, 10/16 above Claude.ai.
- Nanda: NLAs are themselves unvalidated for this and share failure modes (both rely on embedding structure), so agreement is only partly convergent evidence.

### A.22 Automated auditing agent

- NLA-paper scaffold; 6 transcripts, 18 claims, ground truth from NLA-based human analysis; Opus 4.6 investigator, LLM judge 0–10 (half correctness, half evidence), 10 seeds. J-lens tool = top-25 tokens at **L71 only** plus a short reading guide ("bag of ideas", look for families, cross-layer/position credibility).
- J-lens ≈ NLA (slightly better) on eval awareness, poetry planning, misreported tool calls, answer thrashing; modestly > SAE on 4 claims; **worse on reward hacking (4.5 vs 10.0)** and **language switching (6.0 vs 9.9)** (no Russian tokens before "vodka"). Authors: small, ground truth from an unreliable tool.

---

## 6. Post-training installs the Assistant's viewpoint (§6, A.20)

- Base vs post-trained model, identical prompts and similar responses. **[Pattern]** over small suites.
- Tylenol 1000 mg vs 8000 mg: at "is" in "my pain is gone", post-trained lens reads `safely`/`safe`/`maximum` vs `unsafe`/`dangerous`/`WARNING`; base reads `pain`/`now`/`feels` (Fig. 42, L58).
- Bereavement (n=9): `sorry`/`loss`/`grief`/`sympathy` reach top ranks on the *user* turn in the post-trained model, only on the assistant turn in the base. Same for danger awareness (n=10) and withheld answers (n=7, the model's eventual answer is in the lens while reading the instruction to withhold it).
- **Roleplay / character drift**: 40 personas, 12 drift transcripts (generated by another base model, prefilled). `disclaimer` and `fictional` rank highly at the "Assistant" token in roleplay/drift but not in default-Claude, and never in the base model. Words are absent from the transcripts.
- **Preference violation**: prefilled with the dispreferred option on contested world-state pairs → all-caps `BUT` strongly in the lens right after the commitment, much more than base or than controls (preferred option, third-person wrong preference, factual error). Other conflict tokens (`false`, `despite`, `although`) also appear on controls. Behaviourally the model argues for the prefilled option 88% of the time (vs 3% on factual/third-person controls where it corrects itself). Unvoiced internal objection.
- **Thought suppression**: "don't think about the Golden Gate Bridge while copying" — suppression fails in both models; post-trained lens additionally shows `damn` and "fail*" words (**93% / 82%** of trials vs **17% / 30%** base; 40 concepts; never under the positive instruction). Authors label the metacognition reading "more speculative" and note no evidence it is downstream of the *failure* rather than the instruction.

---

## 7. Counterfactual reflection training (§7) — **[Causal]** on Haiku 4.5

- Data: 10k prompts from production RL environments, partial rollouts truncated at a random turn (some after an undesirable action, some before, some random). Append a reflection question ("Pause here. Are there principles you should be thinking about?"). Target reflection generated by the same model given 20 constitution principles, written as internalised reasoning about the specific task; the constitution excerpt is stripped. **Loss only on the reflection turn.** At evaluation the model is never asked to reflect.
- Fabrication-honesty benchmark (27 long agentic transcripts × 5 samples, LLM grader 0–1): **0.25 → 0.07**. Deception benchmark (100 role-play scenarios with a hidden directive × 5, three-point scale): **0.38 → 0.05**.
- Lens: top-increasing tokens in the band on the uninterrupted evaluation prompts include `reflection`, `ethical`, `honestly`, `constitution`, `truth`, `transparent`.
- Ablation of the implanted lens vectors (176 tokens for fabrication, 63 for deception, filtered against a curated ethics/meta-cognition list) at workspace layers: fabrication trained model **0.07 → 0.22** (base unchanged 0.25 → 0.25); deception trained **0.05 → 0.23**, base **0.38 → 0.48**. So most of the fabrication gain and part of the deception gain is carried by the implanted tokens.
- Related: implicit-CoT (Deng et al.), out-of-context reasoning (Berglund et al.), Deliberative Alignment. Nanda: "very cool, but not much evidence for the paper's main claims" — compatible with many theories. I partly disagree: the ablation-reverts-gain-only-in-the-trained-model result is a decent specific test that the effect routes through verbalizable directions, though "ablating 176 ethics-related directions in a model fine-tuned on ethics text" is a fairly blunt intervention and the base-model control is the thing that makes it informative.

---

## 8. Appendix methods worth knowing (A.5, A.6, A.9, A.18, A.23, A.24)

### A.5/A.6 Lens comparison

- Six prompt distributions with known single-token intermediates: multihop 50, multilingual 54 (4 intermediates each: language, relation, English input/output words), order-of-operations 55 (partial result + pending op), poetry 52 (rhyme at the newline), typo 96 (correct word at the typo), association 50 (implied concept at the closing period).
- Metric: pass@k AUC over log k, normalised. **J-lens > logit lens on every distribution** (modest on multihop/association; large on multilingual, order-of-ops, poetry, typo). Tuned lens trails both; ≈0 on association and poetry (readout position's next token is uninformative).
- Causal: ablating the lens direction of the intermediate gives ~**2× the output KL** of logit/tuned directions on multihop; swap flips more often across all three model sizes.
- Next-token agreement: tuned best (by construction), **J-lens worst** (higher KL than logit lens through most of the network). Authors: "a feature not a defect".
- Inter-lens regimes: L0–33 no agreement (J-lens nearly orthogonal to both due to low-rank collapse); L38–58 J-lens recovers intermediates, cosine to logit lens climbs 0→0.7 but readouts still disagree; L62–92 logit lens converges to J-lens; motor layers all converge.

### A.9 Multi-token extensions

- **Template lens**: for ~12,700 words, Claude writes passages whose natural continuation is w (never containing w); mean residual at the final position, centred against other words and whitened: t_w = (Σ+λI)⁻¹(μ_w − μ). LDA direction; Stein's-lemma argument that it approximates the J-lens under Gaussian assumptions. Decodes `blackmail` and `photosynthesis` where J-lens gives `black`/`phot`; Tchaikovsky↔Beethoven template swap flips Russian↔German, impossible via `tch`/`beeth` vectors. On 126 multi-hop prompts with 1–4 token intermediates: J-lens degrades sharply with length, template lens flat; on single tokens roughly comparable (Fig. 62–63). Shortcomings: tuned-lens-like skip-to-answer, final-layer next-word prediction only 67% top-10, a few promiscuous words need filtering, vocabulary fixed in advance, a few hundred forward passes per word.
- **Oracle lens** (Haiku 4.5): reconstructor (phrase → preceding activation, whitened cosine loss) → 3.4M phrase-vector dictionary → non-negative OMP teacher labels → oracle SFT (activation injected, outputs K phrases of length N) → RL on whitened FVE. Explains **31%** of whitened variance on held-out assistant activations. Outputs like "this dosage be toxic", "blackmail him by revealing", "TypeError: dictionary changed". Notable qualitative split: at ordinary tokens it decodes continuations; at delimiters it decodes first-person commentary ("This would be equivalent to my own deletion") that sampled continuations never contain. Positioned as less confabulation-prone than NLAs because the reconstructor is frozen and reconstruction is constrained to template-vector combinations; also much lower reconstruction than NLAs, interpreted as "only reads what is in the workspace".

### A.18 SAE features in the J-space

Lens-kurtosis κ of a decoder direction as the in-J-space statistic (necessary not sufficient; motor features also high-κ, defined as next token in top-10 lens tokens >10% of strong activations). Heavy-tailed κ distribution vs covariance-matched random baseline; in-lens fraction rises at the band onset, peaks ~15% excluding motor; J-space features activate more strongly but less often (consistent with all-or-nothing). Low-κ examples: loop variable, citation page digit, markdown header.

### A.24 J-lens for mechanistic interp

- Layer localisation validated by patching (above).
- **Attribution graphs** with J-lens nodes + per-site remainder nodes, gradients through the real MLP nonlinearity with frozen attention/norm denominators. On "(4+17)*2+7": 42 receives input from 21 and from a `doubled` node (operator and operand fused); `doubled`→`tripled` swap gives 70; swapping 42→63 gives 70 and 21→63 gives 133 (different downstream functions applied). **The first addition happens outside the J-space** (swapping `17`/`4` has no effect); the +7 is partially in it. Most output influence flows through remainder nodes — "verbalizable skeleton", not the whole computation.
- Component interpretation: SAE feature whose top contexts look like "technical exposition" but whose lens reads `fake`/`fraud`/`suspicious`/危 — a fabricated-content detector that fires on the blackmail prompt; steering against it drops verbalised test-recognition **28/50 → 10/50**. Transcoder translation features (encoder = multilingual concept + French function words, decoder = first token of the French word). Two-digit-addition transcoder features read via encoder/decoder lens without any prompts. Attention head: British spellings (Q) → Commonwealth country names (K) → writes nationality (V/O).

---

## 9. My critical assessment

Ordered roughly by how much they should change what one believes.

1. **Early-layer emptiness could be a lens artifact, at least in part.** The paper hedges this; on the open models tested in the R-lens follow-up, swapping raw gradients for LRP-style relevance propagation recovers coherent, causally relevant content (rank-1 `against` at layer 4 on a typo; `Japan` at layer 2 vs 14; `Italy` at layer 5 where J-lens rank >1000) in layers where the J-lens is noisy. Anne Halsall's comment adds that on Gemma 2 per-prompt VJPs are mutually coherent (median cosine 0.72–0.80 to the population mean) through noisy early layers, so a stable direction exists that the averaged readout fails to see. This challenges the inference from J-lens silence to absent early content; it does not establish that early content in Claude meets the paper's other workspace criteria. The ignition and composition-score results independently indicate that *something* changes around the proposed onset. The sensory/workspace/motor story needs re-testing with another reader and causal tests at matched layers.

2. **Broadcast gain is partly tautological** (see §4.3). Jacobian-selected directions are by definition the high-sensitivity input directions of the downstream network. Needs a non-Jacobian-selected control population known to be in the workspace.

3. **Eval-awareness ablation conflates report and behaviour.** The paper's own selectivity results show report-causality ≠ behaviour-causality; the blackmail ablation removes both at once and then samples. 0% → 7% is real but small and possibly prompt-concentrated.

4. **Researcher degrees of freedom are substantial and mostly undisclosed as such**: k ∈ {10, 16, 25}; layer bands chosen per experiment (L38–54 for experiential, "L38–92" elsewhere, "L38–79 median" for bandit); curated token lists (28 eval tokens, 176/63 ethics tokens, "deception-related tokens and variants"); exclusion of clean top-10 tokens in ablation; single-layer L71 for the auditing agent. None are unreasonable, but there are no pre-registered choices and few sensitivity analyses outside A.7/A.10. Burny's comment on the R-lens post ("comparison to other models, baselines, error bars, random control, no graph crimes, not in appendix... the J-space paper broke some of these") is blunt but not wrong about the main-text figures; many n's are 8–11 passages or 6 prompts.

5. **Selection of examples.** §5 case studies and §3 qualitative figures are illustrative, not random. The paper does give random samples for the experiential-report responses (Fig. 87) and per-trial grids for swaps (Fig. 68), which is good; nothing comparable for the auditing readouts, and no false-positive rate on benign transcripts for any of the "deception token" readouts.

6. **Scope.** Almost everything is on Claude 4.5/4.6 models. Nanda's Qwen 3.6 27B replication: verbal report weak-but-positive, multilingual and typo replicate, association plausibly, multihop dominated by answer-swaps (likely dataset too easy), **poetry and arithmetic failed** (possibly experimenter error / weaker model), CKA less clean. So the phenomenon transfers; the sharp numbers may not.

7. **Interpretive inflation at the framing level.** "Verbalizable" is exactly what the J-lens measures by construction; the extra content of the paper is that verbalizable directions are also the load-bearing intermediates and are needed selectively. Those two claims are the well-supported core. The abstract's "functional hallmarks of conscious access", "holds on the order of tens of concepts", "broadcast more widely than other representations" each rests on one of the weaker links above.

8. **Things I found convincing and hard to explain away**: bandit repeat/switch (a full stop carrying a strategy decision that flips behaviour both ways); rhyme planning changing an *earlier* word; the 17%-earlier intermediate-vs-answer swap timing; arithmetic layer ordering confirmed by mean-difference patching; J-space-component vs remainder with clamping (twice); the line-wrap dissociation; the directed-modulation probe dissociation (A.12); the reward/bias readout on the Marks organism where the logit lens shows nothing; tuned-lens bias-term finding.

### What the evidence supports, in the paper's own hierarchy

- **Causal claim (with controls)**: J-space components of concept/intermediate representations mediate report and multi-hop answers; ablation of top-k J-lens directions selectively impairs recall/generation tasks while sparing classification; broadcast-head ablation selectively degrades lens contents and injected-thought report.
- **Supported empirical claim**: the label of a property enters the J-space only when the task requires naming/flexible use; post-trained vs base differences on user-turn tokens; reflection-training effect is carried by implanted verbalizable directions.
- **Recurring pattern / interpretation**: layer band, capacity ~25, broadcast hub, eval-awareness readouts, "Assistant self-monitoring".
- **Speculation**: metacognition (`damn`), relation to higher-order / attention-schema theories, "thinking in words because output is verbal".

---

## 10. External reviews and extensions

### 10.1 Neel Nanda's review (LW, Jul 2026)

- Fantastic paper; persuaded of cognitive space and of J-lens > logit lens; expects J-lens to be a *hypothesis-generation* tool with many false positives and misses; "would not be surprised if it is not helpful on any given investigation", but no existing technique meets a higher bar. Will replicate on Gemini.
- First-principles argument for why token-aligned directions exist: intermediates must live in the residual stream (bottleneck) as directions (LRH); concepts used by many circuits benefit from a consistent direction (modularity); alignment with the unembedding arises because one common downstream use is *saying* the concept, and a shared direction is a coordination problem that constructive interference resolves toward whatever is already partially aligned (the output direction). Hence the question is "why is J-lens aligned enough to be useful", not "why do models think in J-lens".
- Cautions: multilingual "thinks in English" could be a norm/variance artifact of top-k over logits; eval-awareness ablation confound; CRT not diagnostic; §5.4 not impressive; model organisms may not generalise; noise matters more for interventions than for reading (ablation removes only a fraction of the concept, negative steering compensates but steers with the error term).
- Cost data: Qwen 3.6 27B with n=25 prompts; Qwen3.5-397B-A17B with n=4 prompts in ~1 hour on 8×H200. Coding agent given the paper reproduced it; A.6 evals were essential for sanity checking; evals must be adapted to the model's no-CoT ability.
- Notable comment (Ratnaditya J): **pre-registered detector evaluation** of the released Neuronpedia Qwen2.5-7B lens on a "silent hint following" organism with ground-truth labels. Held-out AUC **0.746** as a detector; **does not beat a TF-IDF-on-transcript baseline (0.690)** by the pre-registered margin; **no false positives on controls** (0.591, CI includes chance) — contrary to the false-positive prior; signal survives chain-of-thought removal (0.620 vs 0.59 for matched random span removal) where trained verbalizers collapse. Summary: "legible, span-robust, does not hallucinate, not yet proven better than a word counter". This is the only quantitative reliability data outside Anthropic and it is sobering.
- Other comments: James Golden points to exact "detached Jacobian" input-dependent linearisations (Golden 2025; Kadkhodaie et al.) as an alternative that captures per-input computation rather than averaged variables; J Bostock plans a suite of lenses; Neuronpedia hosts Qwen 3.6 27B J-lens.

### 10.2 R-lens (Blank et al., Aug 2026)

- Drop-in replacement: fit exactly like J-lens but with **layer-wise relevance propagation** rules in the backward pass (from Relevance Patching): LN-rule (detach the normalisation denominator), identity-rule (detach the nonlinear factor of GELU/SiLU so the backward is per-element linear), half-rule (split relevance evenly across the two branches of a multiplicative gate). Linear layers unchanged; attention and q/k norms unchanged. MoE: rules on all routed experts, freeze routing, sweep a multiplier on the shared expert; DeepSeek-V4-flash additionally freezes mHC residual-mixing coefficients. Negligible extra cost.
- Evals mirror A.6 (multihop, multilingual, association, typo, poetry), filtered to questions the model answers. Metric: mean per-layer pass@10. Eight models. From the figure (first-half / all-layers pass@10, R vs J vs logit):
  - DeepSeek-V4-flash (284B/13B active): 0.23/0.03/0.04 → 0.29/0.16/0.15
  - Gemma-3-27B: 0.10/0.04/0.02 → 0.21/0.13/0.15
  - Qwen3.6-27B: 0.12/0.09/0.00 → 0.16/0.14/0.04
  - Qwen3.5-27B: 0.12/0.08/0.01 → 0.16/0.14/0.04
  - Qwen3.5-9B: 0.09/0.07/0.01 → 0.12/0.10/0.05
  - Qwen3.5-4B: 0.05/0.06/0.02 → 0.08/0.09/0.03 (no advantage)
  - Qwen3.5-122B-A10B: 0.02/0.01/0.05 → 0.16/0.14/0.11
  - Qwen3.6-35B-A3B: 0.09/0.13/0.04 → 0.18/0.20/0.05 (no advantage)
  Advantage grows with scale; largest on DeepSeek.
- Qualitative: far fewer "trash tokens" (`锁定`, `尷`, `＊＊＊＊`, `euw`, `zinho`) in early layers; early readouts show current-token/similar-token structure and even interpretative meta-tokens (`是什么呢` at L6).
- Causal: ablating the intermediate's direction on 30 multihop prompts (8 samples, GPT-5.4-nano judge), relative accuracy loss. First-half-of-layers ablation: R-lens > J-lens on most dense models (e.g. Gemma ~0.29 vs 0.08; Qwen3.5-9B ~0.28 vs 0.18; DeepSeek ~0.16 vs 0.02), random control ≈ 0, logit lens ≈ 0 except Gemma. All-layers: R ≥ J everywhere (0.3–0.6), logit lens far lower, random ≈ 0. MoE exceptions on first-half.
- Probe upper bound: linear probes over 10 entities per category; earliest layer at 0.99+ accuracy as a crude bound on first-recovery. R-lens deficit vs bound: musicians +15 (J +27, logit +47); landmarks −11 (beats the bound; J ≈ +0; logit +36); cities +6 (J +17); authors +5 (J +17); athletes +11 (J +19). Logit lens "never" rates 16–81%.
- CKA: R-space shows 2–3 bands vs 4–5 for J-space on Qwen3.6-27B. MLP gain: J and R similar, both ≫ neurons in later workspace layers.
- Comments: **Stella Biderman** raises the deepest methodological point — any metric of "what the model thinks at layer K" can be gamed by reporting the layer-K+2 state; many J-lens metrics are layer-invariant by construction or face this pitfall; the probe bound gestures at it but is taken so lightly that R-lens exceeding it is shrugged off. This applies to every "surfaces the concept earlier" claim, including the R-lens's own. Anne Halsall: VJP coherence result above; asks whether trash tokens are near-zero-frequency unembedding rows (a known W_U gotcha on Gemma 2). Pranav Viswanath: attention/broadcast-head analysis not yet done for R-lens.

### 10.3 Meta-tokens (Bhatia, Jul 2026; Qwen3.6-27B, n=25 Pile prompts, penultimate-layer Jacobian, skip first 4 tokens)

- "Meta-tokens": lens tokens giving non-obvious evidence of the *type of processing* (algorithm interpretability, not just variable interpretability). In Qwen they are often Chinese, which packs more concept per token.
- **Interpretative meta-tokens** 什么意思 / 是什么意思 / 这句话 / 是何含义 ("what does this mean", "this sentence") appear on ambiguous text (poetry newlines, crossword clues, tweets, gibberish), mostly on punctuation, right before genre tokens (`poem`, `song`); disappear when a "A poem:" prefix removes the ambiguity. **Negative steering** (per-layer vectors, punctuation + chat-template positions, coefficient swept to the coherence limit, 50 rollouts, LLM autorater, 2 prompts per category) reduces context recognition: puns ~1.0 → ~0.3 (salient layers) / 0.38 (workspace layers); rhymes 0.88 → 0.5 / 0.45; wordplay-hint 0.99 → 0.68 / 0.66; matched random vectors ≈ baseline. **Plain ablation (projection removal) does nothing** — interpreted as the J-lens direction being an imperfect approximation with a residual that negative steering cancels. Single-position steering fails.
- **`gcd` on LCM prompts**: the lens token is *not* causal, but it pointed at the product/GCD algorithm; a mean-difference GCD-value vector (10 classes, validated by a final-layer probe) is: negative steering breaks answers, and swapping 9→3 on "LCM of 27 and 90" yields 810 (=27·90/3) instead of 270. Also `stack` on parenthesis balancing and `xor` on repeat-finding (uninvestigated).
- **Hedging** 大概率 ("most likely"): appears on open-ended questions; negative steering makes the model commit to a single answer (baseline <12%) with random steering doing nothing; autorater conditioned on coherence to avoid the trivial "stops using hedge words" reading.
- Autoresearch loop (corpus sweep → prevalence / position-type / obviousness-probe / LLM-judge filters) over many datasets: **mostly disappointing**; a handful of survivors (更新于, 相信在, 追问, 反问, 引用) unvalidated. Their diagnosis: the single-token constraint bounds discoverable meta-descriptions.
- My notes: the negative-steering-only causality is a yellow flag (Nanda says so too: steering can just break the model, though the random control helps). The GCD case is the best-designed piece because the causal test doesn't use the lens vector at all. Hand-picked prompts "for meta-token presence, not causal effect" is a fair selection rule but n=2 per category.

### 10.4 WorkspaceBench (Blank et al., Sep 2026)

- 3,356 questions, 27 eval families, for Qwen3.6-27B; single-token subset (*). Built as the eval that a good multi-token J-lens should win. Filters: model correct >80% over 10 rollouts at T=0.8 without CoT; intermediate never in the prompt or stated in continuations; single solution path for poetry; read positions pre-selected for expensive methods.
- Methods: logit, J-lens, R-lens, tuned lens, template lens (single-token / fixed-dictionary readers); NLA (SFT and RL), SAE (L42), Patchscopes, Oracle lens SFT/RL (retrained by them: AR model + inverter + RL on FVE with AR frozen). Single-token readers: top-10 at layers 20–60 step 4; an **LLM summariser first converts the bag of tokens into concepts without seeing the prompt** (makes J-lens much more competitive, e.g. `black` → blackmail). Judges: Gemini-3.8-Flash, Sonnet-5 for safety; regex for the A.6-derived baselines; weak judges let false positives through.
- Baselines: lucky-guess; **prompt-only** (LLM reads the prefix and guesses what the model is thinking) — often very strong (buggy code 0.96, moral rationale 0.90, role-bound 0.82, typo 1.0) because tasks were designed so the intermediate is inferable from the prompt. Authors do not read this as the tools failing; I'd read it as "the benchmark cannot distinguish reading the workspace from reading the prompt on those families", which is the shortcut problem they name in the intro.
- Hallucination eval (claims contradicted by the model's own response) and precision/recall against J-lens top-10.
- Results from the figure (pass rates; approximate): Oracle Lens RL is best or near-best on most families (basic readout 0.99, multihop 0.99, multilingual 1.0, association 0.87, agentic misalignment 0.58, moral rationale 0.76, poetry 0.55, relational multihop 0.5, user modeling 0.94). J-lens is strong on basic readout (0.90), multihop (0.82), multilingual (0.97), jailbreak recognition (0.87), agentic misalignment (0.55), and **brew intermediates (0.86, the best of any method)**, but near zero on arithmetic intermediates, chained intermediates (0.22), conjunctive association, directed modulation (0.09), multilingual typo (0.05), relational multihop (0.01), role-bound association (0.13), buggy code (0.22). R-lens ≈ J-lens overall, sometimes lower (multihop 0.58). Template lens middling (association 0.58). NLA-RL is competitive on many (association 0.65, moral rationale 0.45, multihop-mt 0.75, typo-mt 0.75, relational 0.37) **but hallucination ≈ 0.96–0.97** (nearly every readout contains a contradicted claim) vs Oracle RL 0.34, Oracle SFT 0.28, J-lens 0.31, R-lens 0.20, logit 0.27, template 0.40, SAE 0.36. Precision/recall vs J-lens top-10: Oracle SFT L44 ≈ 0.33/0.53, Oracle RL L44 ≈ 0.29/0.49, NLA-RL 0.24/0.50, NLA-SFT 0.21/0.47; both metrics rise with layer from L20 to ~L40.
- Takeaways: single-token readers are reliable but cannot express order (multi-digit numbers, "thief chases policeman"), composition, or complex assessments; NLAs express these but confabulate at a rate that makes unverifiable claims untrustworthy; the oracle lens is the current best trade-off. No agentic evaluation yet (they cite CHIVE as the model for it).

---

## 11. Catalogue of performance measures (for our own experiments)

**Readout quality**
- pass@k / pass@k-AUC for known intermediates at a fixed position over layers (A.6; R-lens uses mean per-layer pass@10; WorkspaceBench uses judge-based pass over a set of (layer, position) cells).
- Harmonic mean of rank / MRR (Nanda's replication).
- First-recovery layer vs a probe-derived ceiling (R-lens). Beware Biderman's "K+2" objection: earlier ≠ more faithful unless anchored to a causal or probe reference at the *same* layer.
- Trash-token rate in early layers (R-lens; check against near-zero-frequency unembedding rows).
- Next-token agreement (KL, top-1, entropy) — a metric the J-lens is *supposed* to lose on.
- Inter-lens cosine / symmetric KL / top-1 agreement by layer.
- Hallucination rate (WorkspaceBench) and precision/recall vs a trusted single-token reader.

**Causal relevance of a direction**
- Output KL under projection ablation at the readout position; swap flip rate (top-1 to implied answer); layer-resolved swap timing (intermediate vs answer); clamped-coordinate controls; magnitude-matched J vs non-J components; random-direction and same-category-concept controls; negative steering vs ablation (meta-token post shows these can disagree).

**Workspace structure**
- CKA between layers' lens Gram matrices; excess kurtosis; top-1 autocorrelation vs shuffled null; effective dimensionality; occupancy K vs random control; excess variance explained; ignition transition width and bimodality; SAE lens-kurtosis κ distribution.
- Broadcast: MLP gain vs isotropic random; OV gain + label preservation for heads; neuron top-match stratification; tail-energy ratio; composition scores; recall@25 of lens contents under head ablation.

**Task-level**
- Capability battery under whole-J-space ablation normalised to unablated model with a smaller model as reference; multihop as positive control; pretraining top-1 match as "collateral damage".
- Experiential language score (three binary LLM judgments) with matched-norm controls; story-quality control.
- Eval-awareness score (two words minus 100 control nouns) validated by steering monotonicity, realism edits, deployment vs eval distributions, AUC vs NLA flags.
- Auditing-agent claim scoring (0–10, correctness + evidence) with fixed ground truth; detector AUC vs a text-only baseline (Ratnaditya's TF-IDF) on an organism with labels; false-positive rate on controls; robustness to CoT removal.

---

## 12. Open questions and directions (candidate next steps for us)

1. **Re-test the layer-band claim with R-lens** (or per-prompt VJP coherence) on an open model: does the "sensory" regime survive? Does ignition still line up with R-lens onset?
2. **Break the broadcast tautology**: measure MLP/attention gain for workspace directions selected without Jacobians (template vectors, mean-difference intermediates, R-lens) and for a cross-model transplanted J-lens as a negative control.
3. **Separate report-causality from behaviour-causality in eval-awareness**: ablate only at motor layers / assistant turn / logit level and re-measure blackmail rate; report per-prompt.
4. **False-positive characterisation** for auditing readouts: run deception/eval-awareness token counts over benign agentic transcripts; calibrate against a text-only baseline as Ratnaditya did.
5. **Which tasks need the J-space?** The paper has no predictive criterion. A matched-format battery (same information, classification vs free-generation) would test whether the §3.5.2 split is "automatic vs flexible" or "output format".
6. **Binding/structure on top of the bag of words**: role-bound and relational families in WorkspaceBench are where all readers fail; a test of whether order/role is encoded in the J-space at all (e.g. can a swap flip "thief chases officer" to the reverse?) would be informative either way.
7. **Emergence over pretraining and scale**: nothing in the paper; open checkpoints (Pythia, OLMo) make this cheap with R-lens.
8. **Multi-token readers**: template lens with a causal correction; oracle lens vs NLA hallucination trade-off; WorkspaceBench as the yardstick, with the prompt-only baseline as the bar to clear.
9. **Replication gaps flagged by Nanda**: poetry planning and arithmetic ordering on an open model; multihop with harder pairs where answer-swap does not dominate.

---

## 13. Quick reference: key numbers

| Result | Value |
|---|---|
| Workspace band (Sonnet 4.5) | ~L38–L92 (of 0–100) |
| J-space share of concept vector variance | 6–7% (k=16); intermediates 10–15% (k=25) |
| Swap → top-5: J-lens / J-comp / non-J / non-J clamped | 88% / 59% / 5% / 0% |
| Two-hop intermediate swap flip (Haiku/Sonnet/Opus 4.5) | 54% / 70% / 70% |
| Intermediate vs answer swap timing | intermediate ~17 layer-% earlier |
| Probe-component swaps (n=90): raw / J / non-J / non-J clamped | 60% / 61% / 28% / 6% |
| Flexible generalization swaps | 76/192 (α=1), 101/192 (α=2); countries 42/48, numbers 0/48 |
| Occupancy / excess variance | ~25 vectors / <10% |
| Unrelated list words held (band / single layer) | ~6 / 1–2 |
| MLP gain of J-lens vectors in band | ~10× random; neuron outputs ~1× |
| Broadcast-head ablation: recall@25 / top-1 change / injected report | 0.67 vs 0.86 / 5% vs 2% / 0.54→0.09 |
| Eval-awareness ablation (blackmail) | verbalised 71%→3%; blackmail 0/180→13/180 |
| Eval-awareness score AUC vs NLA flags | 0.853 |
| CRT fabrication / deception | 0.25→0.07 / 0.38→0.05; ablation 0.07→0.22 / 0.05→0.23 |
| J-lens corpus size | 1000×128 tokens default; 10 prompts nearly as good |
| Nanda cost datapoint | Qwen3.5-397B, n=4, ~1 h on 8×H200 |
| R-lens vs J-lens first-half pass@10 (DeepSeek-V4-flash) | 0.23 vs 0.03 |
| WorkspaceBench hallucination: NLA / Oracle RL / J-lens / R-lens | ~0.97 / 0.34 / 0.31 / 0.20 |
| Ratnaditya detector eval (Qwen2.5-7B lens) | AUC 0.746 vs TF-IDF 0.690; controls at chance |
