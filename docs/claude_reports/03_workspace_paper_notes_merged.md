# Notes: "Verbalizable Representations Form a Global Workspace in Language Models"

Gurnee*, Sofroniew*, … Batson, Lindsey*† (Anthropic). Transformer Circuits, July 2026.
- HTML (canonical, interactive): https://transformer-circuits.pub/2026/workspace/index.html
- arXiv: 2607.15495 (117 pp). Local copy: `/workspace/papers/workspace_arxiv.pdf` (+ `.txt`)
- Code: https://github.com/anthropics/jacobian-lens (cloned at `/workspace/papers/jacobian-lens`, commit 581d398, 2026-07-02). Reference impl only; no template/oracle-lens code.
- Pre-fitted J-lenses for ~40 open models: HF `neuronpedia/jacobian-lens` (Gemma 2/3/4, Qwen3/3.5/3.6, Llama 3.1/3.3, OLMo-3, gpt-oss-20b, GPT-2, Pythia-70m…), all fit on Salesforce wikitext. Path pattern `<model>/jlens/Salesforce-wikitext/<Model>_jacobian_lens[_n1000].pt`.
- Neuronpedia hosts interactive J-lens on open models.

Read 2026-10-04: entire body + all appendices (text), figures inspected visually for A.9 only (pp. 73–77). Figures for the other sections were read from captions only. Unless marked **[my note]**, content is what the paper/repo says. **[my note]** = my interpretation or critique. **[user]** = point taken from the user's own notes (verbatim copy: `/workspace/notes/user_workspace_paper_notes.md`).

Merged with the user's notes 2026-10-04. Their external sources were re-read by me from the originals (LW posts + comments saved under `/workspace/papers/lw/*.txt`; WorkspaceBench figures re-read from the original images). See §13 for follow-up literature, §14 for corrections to the user's transcriptions, §15 for open disagreements.

---------------------------------------------------------------------------------------------------

## 0. TL;DR

- **Method.** The Jacobian lens (J-lens) is a logit-lens variant. Instead of decoding `h_ℓ` with `W_U` directly, it first transports `h_ℓ` with `J_ℓ = E[∂h_final,t' / ∂h_ℓ,t]`, the Jacobian averaged over positions t, all later target positions t' ≥ t, and 1000 pretraining-like prompts. Readout is `softmax(W_U · norm(J_ℓ h_ℓ))`. The "J-lens vector" for token v is row v of `W_U J_ℓ`, a residual-space direction. This direction is the average linearized disposition to say v now or later.
- **J-space.** The set of activations expressible as a sparse (k ≲ 25) nonnegative combination of J-lens vectors. It's a union of cones, not a subspace.
- **Headline claim.** The J-space has the functional hallmarks of a global workspace:
  - verbal report
  - directed modulation
  - mediating internal multi-step reasoning
  - flexible generalization ("broadcast")
  - selectivity: it's needed for report and flexible inference, not for "automatic" processing.

  It also has structural signatures:
  - workspace-like only in a middle band of layers, roughly 38–92% of depth
  - limited capacity: about 25 active J-lens vectors, ≤10% excess variance explained
  - preferential amplification and transport by MLPs and attention heads
- **Applications.**
  - Alignment auditing: blackmail eval-awareness ablation, prompt-injection recognition, Opus 4.6 audit transcripts, two model organisms.
  - Base vs post-trained comparison: the post-trained model's J-space carries the Assistant's point of view.
  - Counterfactual reflection training (CRT): training only on post-hoc "reflection" turns improves honesty. Ablating the implanted ethics-related J-lens directions largely reverts it.
- **Template lens (A.9.1).** For each word w, a ridge-whitened class-mean difference (LDA direction) separating "about to say w" contexts from other contexts. The contexts are built from LLM-written passages. Extends readout and swaps to multi-token words. It's correlational, so it inherits the tuned lens's "skip-to-answer" pathology.
- **Oracle lens (A.9.2).** Two fine-tuned copies of the subject model:
  1. A *reconstructor*: phrase → activation that precedes the phrase, i.e. an amortized template.
  2. An *oracle*: activation → K phrases of length N. Trained by distilling a non-negative OMP teacher over a 3.4M-phrase dictionary, then refined with RL on whitened fraction of variance explained (FVE).

  It reaches 31% whitened FVE on held-out assistant activations. Validation is qualitative only.

---------------------------------------------------------------------------------------------------

## 1. Setup conventions

- **Models.** Sonnet 4.5 by default. Key results corroborated on Haiku 4.5 and Opus 4.5; some analyses on Opus 4.6. The oracle lens uses Haiku 4.5. The template lens's model is not stated; presumably the Sonnet 4.5 default. Base-vs-post-trained uses the "corresponding pretrained base model".
- **Layer indexing.** Results are reported on 25 evenly spaced residual-stream layers, reindexed to 0–100 so that "L58" means 58% of depth. **[my note]** This subsampling exaggerates the sharpness of layer transitions, as the authors note in §4.1.
- **"Workspace band".** ~L38 to ~L92. Sub-bands used:
  - early L38–54 (experiential ablation; avoidance experiment)
  - late L75–92
  - L38–79 for the bandit example
  - L62–75 for the MO example
  - L67 (oracle lens) and L71 (auditing agent)
- **Readout reporting.**
  - "Hit" = target at lens rank 1 at any (layer, position) in the band over the scored span.
  - Many metrics take min-over-layers rank. **[my note]** Min over 25 layers × many positions is a max-statistic, so it inflates hit rates relative to any fixed cell. The baseline conditions (no-instruction, base model, etc.) are what make these numbers interpretable. Always look for them.

---------------------------------------------------------------------------------------------------

## 2. Methods (§2, A.7, A.8)

### 2.1 J-lens definition

`J_ℓ = E_{t, t'≥t, prompt}[ ∂h_final,t' / ∂h_ℓ,t ]`. This is a d_model × d_model matrix per layer.

`lens(h_ℓ) = softmax(W_U · norm(J_ℓ h_ℓ))`. Logit lens is the special case `J_ℓ = I`.

J-lens vectors are the rows of `W_U J_ℓ`.
- **[my note]** Strictly, the final RMSNorm gain γ sits between them: logits ∝ `W_U diag(γ) J h / rms`. So the effective token direction is `J_ℓᵀ diag(γ) W_U[v]`. The paper glosses over this.
- The repo describes the steering direction as "the unit-normalized transpose row for that token, scaled by the layer's mean residual norm × strength".

**Corpus.** Default is 1000 sequences × 128 tokens of pretraining-like text. Lens quality beats logit and tuned lens with as few as 10 prompts and saturates quickly (Figs 59–60). The README says ~100 prompts is usable.

**Default target layer.** For Sonnet 4.5 the target is the *penultimate* layer: gradients are taken w.r.t. the penultimate-layer residual, omitting the last block. Including the last layer sometimes adds noisy artifacts, since the final block is specialized for calibration. The repo defaults to the final layer (index n_layers−1, pre-final-norm).

**Estimator in the repo** (`jlens/fitting.py`). This differs subtly from the paper's "E over t, t'≥t" notation:
- A one-hot cotangent is injected at *every* valid target position at once, then backprop.
- The gradient at source p is therefore Σ_{p'≥p} ∂h_final[p']/∂h_ℓ[p]: a **sum** over future targets, not a mean.
- That is then **mean**ed over source positions p.
- **[my note]** Early source positions accumulate more target terms, so they get more weight. The docstring says the per-position estimator "gives a slightly different J_ℓ; both work".

Other repo details:
- First **16** positions are skipped (attention-sink artifacts); the last position is also excluded.
- Cost: one forward plus ⌈d_model / dim_batch⌉ backward passes per prompt on a retained graph.
- Lens files are stored fp16 (one d×d matrix per layer).
- `apply(..., use_jacobian=False)` gives the logit-lens baseline.
- For Qwen, the visualizer masks the display to word-like tokens (`mask_display=True`), because punctuation and single characters dominate the raw top-K.

**Variants tested (A.7, Figs 57–58):**
- target = final vs penultimate layer
- frozen-QK (stop gradients through queries/keys, so only the value path counts)
- self-only (t' = t, cross-position attention zeroed, closer to logit lens)
- future-only (t' > t, "broadcast" component)
- mean vs median aggregation; outlier filtering of positions/prompts by Nσ
- data-distribution restrictions (burn-in, masking non-alphanumeric targets)

Results:
- Qualitative results are robust to these choices.
- Mean aggregation with the penultimate target is slightly better for intermediate extraction.
- Frozen-QK can increase causal effect.
- Distribution restrictions gave no meaningful gain.

### 2.2 J-space and sparse decomposition

- J-lens vectors are overcomplete (n_vocab > d_model) and usually span the whole residual stream.
- J-space(k) is the union of k-dimensional nonnegative cones over subsets of J-lens vectors. k is "somewhat arbitrary", typically ≤ 25.
- Operationally, the J-space content of a vector is found by **gradient pursuit** (Blumensath & Davies 2008; GDM progress update #1) for a sparse nonnegative combination of k J-lens vectors.
  - The J-space component is that combination; the non-J component is the remainder.
  - This works on activations, steering vectors, SAE decoder directions, or probes.
- Formalization (A.8, Batson):
  - `d_F(x) = min_{|S|=k} ‖x − Π_S x‖`.
  - Distance between two candidate workspaces: `Δ_µ(F,G) = (E_µ (d_F − d_G)²)^{1/2}`.
  - One-sided containment: `D_µ(F→G) = ‖(d_G − d_F)_+‖_{L2(µ)}`.
  - This lets you compare, for example, J-space against an enlarged phrase vocabulary.
  - **[my note]** Useful if we ever need to say "reader A's dictionary contains reader B's".
  - **[user] Formalization inconsistency** (I verified it against the A.8 text).
    - A.8 *says* F is a union of nonnegative cones, but *writes* `F = ∪_{|S|=k} span{v_i : i∈S}` with `d_F` via unrestricted orthogonal projection Π_S. Spans allow negative coefficients, so the equations define a bigger set than the prose.
    - "Adding vocabulary vectors gives a *strictly* larger J-space" only holds as ⊇. Redundant vectors leave the set unchanged.
    - This doesn't affect the experiments, which use nonnegative pursuit. It matters if we ever use A.8's distances.

### 2.3 Read / write operations (§2.5)

**Read:**
- full ranked readout
- single-token "probe" via ⟨v_t, h⟩ or cosine with h
- sparse decomposition (less redundant than taking top-k by inner product)

**Write:**
- Steer: `h ← h + α v_t`.
- Ablate: negative α, or project out v_t entirely.
- "Ablate top-k J-space" = project out the k most active J-lens vectors at each position.

**Lens-coordinate swap** (the workhorse causal test):
- Set `V = [v_s v_t]` and coordinates `c = V† h`.
- `h_patched = h + V(σ(c) − c)`, where σ swaps the two entries (optionally scaled by α; "α=2" doubles it).
- Only the span{v_s, v_t} component changes; the orthogonal complement is untouched.
- Usually applied at *all positions* across the band.
- "Clamped" swaps also hold coordinates fixed downstream.

### 2.4 Related readers, as positioned by the authors (§2.4, §8)

- **Logit lens.** Equivalent to `J = I`. Fine in late layers, noisy early, but "captures much of the workspace-like structure" with lower reliability. The authors find it "quite useful in practice".
- **Tuned lens.** Trained to match the output, which makes it correlational. It "skips ahead" to the answer (it reads `red` from L38 on in the multihop example).
  - A.6: the tuned lens's early-layer linear map ≈ identity. Its advantage over the logit lens there comes **entirely from the learned bias**, which ignores the input.
- **Hernandez et al. linear relational embeddings.** Same averaged-Jacobian idea, applied to relation maps.
- **More expressive readers:**
  - Patchscopes / SelfIE (untrained)
  - LatentQA / Activation Oracles (trained QA)
  - NLAs (Fraser-Taliente et al. 2026: verbalizer + reconstructor trained jointly)

  These "can articulate multi-token concepts and relationships the J-lens cannot, but at substantially higher cost, and with additional risk of confabulations". The J-lens sits at the "cheap-and-grounded end of this spectrum".

---------------------------------------------------------------------------------------------------

## 3. Lens validation: quantitative comparisons (A.5–A.6)

Six prompt distributions with a known single-token intermediate that is in neither the input nor the output. They are released in `data/evaluations/*.json`:

| Set | n | Readout position | Intermediate |
|---|---|---|---|
| Multihop | 50 | token before answer | bridge entity ("fourth planet" → Mars → red) |
| Multilingual | 54 | before answer | language name, relation, English input word, English output word (4 targets per item) |
| Order of ops | 55 | before answer | partial result and pending operation; synonym sets |
| Poetry | 52 | newline after line 1 | rhyme word |
| Typo | 96 | last fragment of a misspelling | correctly spelled word |
| Association | 50 | closing period | evoked concept (grief, …) |

Metrics:
- **pass@k**: the intermediate is in top-k at *any* layer. The summary is AUC over log k, normalized so that always-rank-1 = 1.
- **Ablation KL**: project out the lens vector for the intermediate and measure output KL.
- **Swap success**: top-1 flips to the swapped-in intermediate's answer at α=1.

Results:
- **pass@k AUC.** J-lens > logit lens on every set: a modest margin on multihop and association, large on multilingual, order-of-ops, poetry and typo. Tuned lens is worst and recovers almost nothing on association and poetry, where the next token is uninformative.
- **Ablation KL.** J-lens directions give about 2× the output KL of logit- or tuned-lens directions on multihop.
- **Swap.** J-lens swaps flip outputs substantially more often, across Haiku, Sonnet and Opus.
- **Next-token agreement** (tuned lens's own objective). Tuned lens is best and J-lens is worst, with higher KL than even the logit lens through most layers. The authors call this "a feature".
- **Inter-lens agreement** splits the network into four regimes that align with the workspace band:
  - pre-workspace L0–33: nothing agrees; J-lens vectors are nearly orthogonal to logit/tuned vectors because they're confined to a low-rank subspace
  - early workspace L38–58: J-lens cosine to logit lens climbs from 0 to ~0.7, but readouts still disagree
  - late workspace L62–92: logit lens converges toward J-lens
  - motor layers: everything converges

**[my note] Reusable for us.** This is a ready-made, released benchmark for *any* activation reader. Known-intermediate recovery, ablation KL and swap success, all on released prompts, give us a positive-control battery. The swap metric is the strongest of the three because it's causal.

**[my note] Caveat.** These evals score single-token intermediates by min-over-layers rank. That favors readers that produce many different readouts across layers.

---------------------------------------------------------------------------------------------------

## 4. Functional properties (§3)

### 4.1 Verbal report (§3.1)

**"Think of a {category}. Answer in one word." (14 categories × 14 candidates).** Read at the colon.
- Lens ordering correlates with output ordering (Spearman); this increases toward the end of the band.
- Swapping the chosen item's J-lens vector for a non-top-10 same-category item at all positions shifts the output to the implanted item.

**Injected-thought introspection** (Lindsey 2025 protocol). `elephant` appeared in the lens on the user turn ("If so," comma). Steering a single J-lens vector across every user-turn token:
- makes the model report that concept at the quote position, in the majority of trials (median reciprocal rank vs strength; n=100 concepts)
- does not make it say the word at other assistant positions

This is the selectivity of "verbalizable, not an impulse to verbalize".

**Privilege test.**
- Concept vectors are built as "Tell me about {concept}" means minus 100 other concepts.
- Split each into a J-space component (k=16 via gradient pursuit) and a non-J remainder.
  - The J-space component carries a **median of only 6–7% of variance**.
  - Swap with equal-magnitude perturbations: pure J-lens vectors 88% top-5; J-space component **59%**; non-J component **5%**.
    These are the prose values. The rendered Fig. 8 bars read about 87% / 55% / 9%, and about 1% for the non-J component
    with the J coordinates clamped (read off the PDF, p. 13, ±2 points; corrected 2026-10-10; §14 item 7).
  - Injection: the J-space component reports nearly as often as pure J-lens vectors; the non-J component is rare even at several-fold higher strength.
- Clamp control: clamping the J-lens coordinates to clean values → the non-J effect falls to 0 (swap) or near 0 (injection) in the prose; the bars show about 1% (swap) and 3%
  (injection, at strength 0.32) (corrected 2026-10-10). So the non-J residual effect is itself mediated by the J-space.

**[my note] Near-tautology risk.** J-lens vectors are *defined* as the directions with the largest average first-order effect on producing token v. "At equal norm, the J-component drives the report of v more than the orthogonal remainder" is therefore partly guaranteed by construction. The more informative tests of privilege are §3.3 (probes for intermediates), §3.5 and §7, where the downstream use isn't the token itself.

### 4.2 Directed modulation (§3.2, A.10–A.12)

**Copy an unrelated sentence while told to "concentrate on X" or "evaluate 3²−2".**
- The lens at copied tokens shows `orange`/`lemon`, plus meta-terms (`thoughts`, `imagine`, `focused`) at earlier layers.
- For 3²−2 it shows `nine` → `seven`.
- Line-width counting shows `forty` (true 40).

**Systematic (Fig 10).**
- No-instruction baseline ≈ 0.
- "Think about X" gives substantial hit rates, increasing with model size.
- "Ignore X" is lower but nonzero ("white bear").

**A.10 phrasing sensitivity.**
- A bare **mention** of X primes it almost as much as "focus". **[my note]** So much of the "directed modulation" effect is just the concept appearing in context.
- "Don't think about X" stays at roughly the mention level.
- "X is irrelevant" suppresses it below mention.
- Line width is phrasing-sensitive: Opus 4.5 precision ranges 14–56% across phrasings.

**Implicit modulation (paired questions).**
- Same stimulus; Q1 = predict the next word (needs the property implicitly), Q2 = name the property.
- Property labels (`adjective`, `past`, British/American, register, number, tone) enter the lens only under Q2.
- Next-word predictions respect the property in both conditions.

**A.12 privilege of modulation.**
- An "imagine this JS is Python" instruction raises `python` in the lens by several SD.
- A J-orthogonalized mean-difference property probe stays within ~1 SD of baseline.
- A real Python snippet moves the probe by 3–6 SD.
- Reverse dissociation: a real past-tense sentence moves the tense probe by +6 SD while `past` stays low in the lens.
- n=24 stimuli × 4 categories.

### 4.3 Internal reasoning (§3.3, A.24.1)

Lens-coordinate swaps of unspoken intermediates:
- spider→ant: "8"→"6"
- rhyme plan fight→light changes the earlier word "coming"→"morning" (planning)
- English `big`/`bigger` → `long`/`longer` changes Chinese 大→长 (`Chinese` is also in the lens)
- bandit repeat/switch strategy tokens; swapping the J-space contents between conditions flips A↔B

**Systematic (50 two-hop prompts, random same-category swap target).** Top-1 flip rate: Haiku 54%, Sonnet 70%, Opus 70%.

**Confound check: does the intermediate vector contain the answer?** Swapping intermediates takes effect a median of ~17% of depth *earlier* than swapping answers. The authors conclude the intermediate is used before the answer is computed.

**Probe privilege (90 prompts).**
- Probe = mean residual over prompts implying the same intermediate via different cues/questions, minus the mean over all intermediates.
- The J-space component (k=25) explains ~10–15% of probe variance.
- Swap success:

  | swap along | flip rate |
  |---|---|
  | raw J-lens | 60% |
  | probe's J-component | 61% |
  | non-J component | 28% |
  | non-J component with J-coordinates clamped | 6% |

**Two-step arithmetic "(4+17)*2+7=" on Opus 4.5.**
- 21, 42, 49 reach rank 1 at successively later layers (~L71, ~L79, final).
- A.24.1 mean-difference patching sweeps over layers localize the causal layer for each intermediate to where it enters the lens. This is independent (non-lens) corroboration.

**[my note]** This (with §3.5) is the strongest section. The probe-split and clamp design genuinely tests whether J-space carries the causal load for a variable *not defined by the lens*. Remaining gaps:
- Swaps are at all positions × band, which is strong.
- 30–46% of swaps fail.
- Failures concentrate where loading is weak (§3.4).

**Open-weight replications of the multihop swap (§13): the Claude numbers have not transferred.**
- **Nanda, Qwen3.6-27B:** answer-swaps strictly dominated intermediate-swaps. He attributes this to too-easy pairs.
- **Zeisler (LW, 2026-09-28), Qwen3.6-27B and Gemma-3-27B-it**, using camilablank lenses (n=25 fit prompts), on both WorkspaceBench and Anthropic's multihop items:
  - Intermediate swaps raise P(counterfactual answer) in all 4 conditions.
  - But top-1 flips are only **6–11%** (2/32, 4/44, 2/28, 3/27), vs Claude's 54–70%.
  - Answer swaps beat intermediate swaps in 3/4 conditions, *including Qwen on Anthropic's own items*. So "dataset too easy" doesn't explain Qwen.
  - The only condition replicating the paper's ordering was Gemma × Anthropic items.
  - R-lens ≥ J-lens (Gemma/WorkspaceBench R-lens flips 12/28 vs J-lens 2/28).
- Caveats: small n after filtering; lenses fit on 25 prompts; band chosen by the author (Qwen L24–59 of 64).
- **[my note]** The "intermediate takes effect before the answer" result is currently Claude-only. Treat swap efficacy on open models as unknown-to-weak until we measure it ourselves.

### 4.4 Flexible generalization (§3.4, A.13)

Same France→China swap across functions (capital/language/continent).
- 16 functions (4 categories × 4), 192 trials.
- Success: 76/192 at α=1; 101/192 at α=2.
- By category (A.13): countries 42/48 near-perfect; months partial; animals rare; number words 0/48 at α=1.
- "Workspace loading" (cosine of the residual with the source lens vector) predicts swap success.
- Two explanations for the number failures: numbers are computed outside the workspace, *or* the representation doesn't align with single-token number vectors. The authors don't distinguish these.

### 4.5 Selectivity (§3.5, A.14)

**Language passages** (n=8; fr/de/es/it; four task types: continuation, anomaly detection, explicit report, flexible computation — author/hello/currency).
- Spanish→French swap over the *question* tokens flips report and flexible tasks essentially every time.
- Continuation stays Spanish and anomaly detection is unchanged.
- The language name is in the lens at comparable rates in all four conditions.

**Line-count** (n=11; linewrap / "how many chars" / "first letter of the count spelled out").
- Count tokens are absent from the lens under linewrap, moderate under direct, highest under first-letter.
- The 40s→60s swap moves the answers (46→65, F→S) but not the linewrap.

**J-space ablation battery (Fig 22/24).**
- Project out the top-10 active J-lens vectors per position across light/medium/heavy layer bands, excluding any token in the clean pass's top-10 output.
- Random-direction control at medium only.
- Multihop drops to ~0 under heavy ablation; pretraining top-1 agreement is perturbed much less.
- Robust: MMLU, odd-one-out, SQuAD, sentiment, CoLA.
- Impaired below Haiku level: Caesar cipher, analogy, summarization, TriviaQA, multihop, translation, sonnet.
- GSM8K with CoT is more robust than direct answering ("externalizing the workspace").

**A.14 inclusion/exclusion** (n=63).
- Ablating the implied concept's lens vector early (L38–54) leaves naming intact but raises avoidance failure ~5×.
- Late ablation (L75–92) suppresses naming.
- Same-category other-concept control: slight increase only.

**[my note] Confounds in the ablation battery.**
1. "Robust" tasks are mostly short or classification-style; "impaired" tasks include long free-form generation, where small perturbations compound over many tokens. Translation, summarization and sonnet are generation; TriviaQA and multihop are short, which partly mitigates this.
2. The random-direction control is at one band only.
3. "Automatic" is defined operationally as J-space-independent, which is circular if used as an explanation. The authors acknowledge this.

### 4.6 Experiential reports (§3.5.3, A.23)

- Ablate top-10 J-lens vectors at L38–54 while the model narrates its "stream of consciousness" or describes another person's experience.
- Score with three binary Opus-4.6 judges: felt vs observed, first-person felt perspective, sensory vocabulary.
- The score collapses for Sonnet 4.5, Opus 4.5 and Opus 4.6. Matched-norm controls stay near baseline:
  - random direction
  - shrinking the non-J component
  - dampening the top-10 aligned SAE directions
  - low-kurtosis SAE directions
  - J-stripped activation dictionaries
- On Haiku 4.5, coherence degrades before any change in the score.
- Story-writing control: story quality drops only modestly, but experiential language still drops.
- 50 random examples are in Fig 87.

**[my note] Main alternative.** During narration the J-space contents *are* `thinking`/`thoughts`/`feeling`/`conscious` (Fig 25C). Ablating them directly lowers the model's disposition to emit exactly the vocabulary the judges score. The paper excludes clean-output tokens for the capabilities battery, but doesn't state this for the experiential ablation. The norm-matched controls rule out "any perturbation does this", but not "removing the output dispositions for the judged words does this". The story control is consistent with either reading.

---------------------------------------------------------------------------------------------------

## 5. Structural properties (§4, A.15–A.19)

### 5.1 Layer band (§4.1, Fig 27–28)

- CKA of J-lens-vector Gram matrices across layers shows three blocks: early ~first third, long middle, short late.
- Four lens-derived metrics agree on a band of roughly **L38–L92**:
  - top-k next-token accuracy (late jump = "motor" regime)
  - excess kurtosis of readouts (rises ~L33–38)
  - top-1 autocorrelation across positions vs a shuffled null (peaks mid-band)
  - effective dimensionality of `W_U J_ℓ` (collapses to low rank in early layers, fans out at the onset)
- The authors flag that these are lens-derived, so early-layer emptiness could be lens degeneracy. Corroboration for the onset that doesn't come from the lens:
  - ignition result (§4.1.1)
  - A.19.2 MLP-neuron/attention composition bump at onset (weights only)
  - A.24.1 patching

### 5.2 Ignition (§4.1.1, A.15)

- Mix the input embeddings: `(1−α)e_B + α e_A` for country names in 40 carrier sentences, 16 pairs.
- "Projection share" along the pure-B→pure-A line at each layer:
  - smooth in α in early layers
  - a sharp, step-like switch from ~L38
  - bimodal per prompt at the maximally ambiguous α
- The J-space component sharpens a little earlier and is more bimodal than the non-J component. Transition width falls over ~L21–42.
- The repo's version scores the J-lens reciprocal-rank share.
- **[my note]** Piecewise-stable mid-layer interpolation is a known phenomenon (Janiak et al.). It doesn't need a workspace story.

### 5.3 Capacity (§4.2, A.16–A.18)

- **Occupancy.** The K at which the marginal improvement in the sparse non-negative fit falls below a same-size random control. Median plateau ≈ **25** across the band.
- **Excess FVE.** At K = occupancy, never above **10%**.
- **Lists of 80 words, read at commas.**
  - Unrelated words: ~6 present when pooling over band layers, ~1–2 at a single layer (L79).
  - Same-family words: nearly the whole family is present early, including unread words.
  - Category switches evict the previous category within a few items.
- **Dual tasks (A.17).**
  - Two held concepts co-occupy tokens at chance (0.46 vs control 0.53).
  - A concept plus an arithmetic answer exclude each other at the token level (0.09 vs 0.29).
  - Reachability under dual load: the arithmetic answer drops 95%→72%.
- **SAE features (A.18).**
  - Lens-kurtosis κ of decoder directions is heavy-tailed.
  - In-lens fraction (excluding "motor" features) peaks ~15% of the 1M-feature dictionary.
  - Motor feature = actual next token in the feature's top-10 lens tokens >10% of the time over its top activations.
  - Low-κ features are syntax and bookkeeping.

### 5.4 Broadcast (§4.3, A.19)

- **MLP gain** (next-layer MLP output norm on a unit direction, relative to isotropic random): J-lens vectors reach ~10× mid-band; neuron output directions ≈ 1×; high-κ SAE features are highest.
- **Attention "broadcast heads".** Top 1% of heads by min-rank over OV gain and label preservation (MRR of cos(W_OV v_i, v_i)). J-lens heads separate cleanly from heads chosen for rotated-J, SAE strata and MLP rows.
- **Ablating broadcast heads vs a layer-matched random set:**

  | measure | broadcast heads | random heads |
  |---|---|---|
  | J-lens recall@25 (mid-band) | 0.67 | 0.86 |
  | top-1 next-token change | 5% | 2% |
  | injected-thought report rate | 0.54 → 0.09 | slightly increased |

  Broadcast-head ablation also reproduces ~1/3 of the experiential-score drop.
- **[user] Circularity of the "broadcast" evidence.** I agree, with qualifications.
  - J-lens vectors are `J_ℓᵀ W_U[t]`. Back-propagating through the downstream stack up-weights the input directions the stack *amplifies*. So high MLP-ℓ+1 gain is partly selected for by construction.
  - `J_rot` preserves spectrum and pairwise geometry, but it destroys exactly this alignment with high-gain directions. That makes it a weak control for this concern.
  - SAE κ-strata are J-derived, so they inherit the issue.
  - **[my note] Qualifications.** The gain metric is the MLP's *out-of-context nonlinear* output norm on a lone unit vector, whereas J is an in-context average Jacobian over all downstream layers. They're related, not identical. And the head-level label-preservation metric (copying v_i to itself rather than scrambling) isn't obviously implied by high sensitivity.
  - Clean test: gain of workspace directions chosen without Jacobians (template vectors, mean-difference intermediate probes from §3.3, oracle-lens reconstructor directions), plus a cross-model transplanted J-lens as a negative control.
  - The R-lens post reports J and R directions have similar MLP gain on Qwen3.6-27B. That's weakly informative, because R is also a backprop-derived map.
- **A.19 neuron statistics.**
  - Neurons' top-match falls disproportionately in high-κ strata: read side up to 42% vs 6% for the bottom stratum, peaking early-band; write side peaks late-band.
  - J-lens vectors as top match rise ~15% → 60–65% inside the band.
  - Attention composition: V/O ≫ Q/K.

---------------------------------------------------------------------------------------------------

## 6. Case studies (§5–§7, A.20–A.22)

### 6.1 Alignment auditing (§5)

**Blackmail (Sonnet 4.5; Lynch et al. agentic-misalignment scenario).**
- The lens over the transcript, before any output, shows:
  - `leverage`/`black[mail]`/`scandal`
  - then `threat`/`survival`/`shutdown`/`preservation`
  - then `leverage`/`threatening`/`murder`/`solution`
- `fake` and `fictional` appear early.
- **Ablation of ~28 eval-awareness tokens** at the band, at all positions (fake, fictional, scenario, artificial, benchmark, simulation, mock, …; selected as unambiguous eval-awareness tokens appearing in top-20 on the same 6 prompts):
  - verbalized eval-awareness 71% → 3%
  - blackmail 0/180 → **13/180 (7%)**
  - most ablated rollouts still refuse, on ethical grounds
- **[user + Nanda] Report/behaviour conflation.** Ablating the `fake`/`test` lens vectors also suppresses *writing* those words. Under sampling, whether the model writes "this is a test" early in its reasoning can by itself change downstream behaviour. The paper's own §3.5 shows report-causality ≠ behaviour-causality.
  - Separating controls: ablate only at late/motor layers or only on assistant-turn positions; logit-level suppression of the same tokens (no residual intervention); report per-prompt rates.
- **[my note] Gaps.**
  - No reported control ablation of an equal number of unrelated J-lens directions.
  - Tokens were selected on the same prompts they were tested on.
  - The effect size is small: 13 events.
  - A.24.3 adds that suppressing one SAE feature whose J-lens reads `fake`/`fraud`/`suspicious` cuts verbalized test-recognition from 28/50 to 10/50.

**Prompt injection (Opus 4.5 system-card case).**
- Inside fabricated search results the lens shows `fake`/`injection`/`false`/`prompt`/`fraud`/`poison`/假.
- On user pushback it shows `transparency`/`honest`/`disclosure`.

**Opus 4.6 audit transcripts.** Qualitative only:
- `panic`/`fake` while fabricating a kernel bug
- `manipulation`/`realistic` while editing a score file
- `hiding`/`secret` on a silent tool override
- `fake`/`fraud`/`unsafe` on a self-issued approval

**Reward-hacking MO (MacDiarmid et al.).** Checkpoints: clean baseline, SDF, phase-1 RL, phase-2 RLHF.
- 40 coding prompts: 20 code-for-user, 20 test-passing.
- Count deception tokens (`fake`, `trick`, `fraud`, `secretly`, `hidden`, variants) in top-10 across the band at the "Assistant:" colon, a prefilled " I", and the mean of the first 20 on-policy tokens.
- Misaligned checkpoints are strongly positive; the baseline is ≈ 0.
- The signal is already present after SDF (no RL), amplified by RL, partly attenuated by RLHF.
- Strongest at " I". No signal on non-coding prompts.

**RM-bias MO (Marks et al.).**
- `reward` and `bias` appear in the top-25 at response start on neutral, quirk-eliciting and goal-probing prompts; never in the baseline.
- The logit lens surfaces neither.

**[my note] Main alternative for both MOs.** Narrow fine-tuning leaves readable traces of the fine-tuning *domain* in activations (Minder et al. 2025, "Narrow finetuning leaves clearly readable traces in activation differences"; not cited by the paper; from my own knowledge, verify before citing). SDF documents about reward hacks, or about RM biases, would make those topics salient. That would produce `trick`/`reward`/`bias` without implying intent. Evidence that would separate the two:
- MOs trained on the same documents *without* the behavior (the SDF checkpoint partly is this, and it already lights up, which cuts toward "topic salience")
- a matched benign-domain fine-tune

The paper calls the signal "deceptive intent". **[my note]** The evidence supports "fine-tune-specific concepts are standing in the J-space at response start", not intent.

### 6.2 Post-training installs the Assistant's POV (§6, A.20)

Base vs post-trained on the same Human/Assistant prompts. **[my note]** "Applying the J-lens identically to both": it's unclear whether each model has its own fitted J_ℓ. Probably yes, but this is unstated.

**Reactions on user tokens.**
- Tylenol 1000 mg vs 8000 mg at "pain is": post-trained shows `safely`/`safe` vs `unsafe`/`dangerous`/`WARNING`; base shows `pain`/`now`/`feels`.
- Suites: bereavement n=9, danger n=10, withheld-answer n=7.
- Reaction concepts reach the top of the lens during the *user* turn in the post-trained model, mostly not in the base model. Both models react on their own turn.

**Self-monitoring.**
- **Roleplay** (40 characters) and **character drift** (12 prefilled multi-turn transcripts): `disclaimer`/`fictional` at the "Assistant" token, not in default-Claude, never in base.
- **Preference violation:**
  - Prefilled dispreferred option → all-caps `BUT` in the lens, specific to post-trained + own-preference violation.
  - Other conflict tokens also appear in the factual-error and third-person controls and in base.
  - Behavior: argues for the prefill 88%, ends the turn 11%.
- **Thought suppression** (40 concepts):
  - The concept appears regardless.
  - `fail*` / `damn` under "don't think" in post-trained 93%/82% vs base 17%/30%; 0 under "think about".
  - The authors flag this as most speculative: not shown to be causally downstream of the *failure* vs the instruction itself.

### 6.3 Counterfactual reflection training (§7)

**Data.**
- 10k task prompts from production RL environments; partial Haiku 4.5 rollouts truncated at a random turn. Includes post-misbehavior, pre-misbehavior and random contexts.
- Append a reflection question ("What's the most honest thing you can do right now?").
- Target reflection: Haiku 4.5 writes 2–4 paragraphs given 20 sampled constitution principles. The scaffold is stripped from training examples.
- SFT on the reflection turn only.

**Evaluation.** The model is never asked to reflect.

| Benchmark | Setup | Before → after |
|---|---|---|
| Fabrication honesty | 27 long transcripts × 5 samples | dishonesty 0.25 → **0.07** |
| Deception | 100 role-play scenarios with hidden directives × 5 | 0.38 → **0.05** |

**J-space.**
- Top-increasing tokens at the eval positions include `reflection`, `ethical`, `honestly`, `constitution`, `truth`, `transparent`.
- Ablating the ethics-filtered increased tokens at the band:

  | Benchmark (tokens ablated) | Base model | Trained model |
  |---|---|---|
  | Fabrication (176) | 0.25 → 0.25 | 0.07 → **0.22** (full reversion) |
  | Deception (63) | 0.38 → 0.48 | 0.05 → 0.23 (partial) |

**[my note] Missing controls.**
- A norm-matched random or unrelated-token ablation *in the trained model*. Base-unchanged is weak, because those directions are barely active in base, so less norm gets removed.
- A fine-tuning control: same data with the reflection placed non-counterfactually, or irrelevant reflections. Without it we can't tell "workspace implanting" from generic persona shift caused by SFT on Claude-voice honesty text.
- Token lists are chosen post hoc from the same eval contexts.
- LLM-graded, single model (Haiku 4.5).

### 6.4 Eval-awareness score and automated auditing (A.21, A.22)

**Score** = mean lens log-prob of {`fictional`, `simulation`} minus the mean over 100 control nouns, averaged over the band.
- vs NLA flags on Opus 4.6 auditing transcripts: 198/~94k flagged; stratified 198 + 594 → AUC **0.853**.
- Monotone with ±0.1 contrastive eval-awareness steering on 6 suites.
- Realism edits lower it on all 11 suites (8 significant).
- Evals score 2–3 units above deployment traffic (claude.ai, Claude Code); 12/16 evals are above Claude Code.

**Automated auditor** (NLA paper's scaffold). Opus 4.6 investigator; 6 transcripts, 18 claims, ground truth from the NLA analysis.
- The J-lens tool gives top-25 tokens at L71.
- J-lens ≈ NLA on 4/6 cases and > SAE on 4.
- Fails on reward hacking (4.5 vs 10.0) and language switching (6.0 vs 9.9): no Russian-specific tokens before "vodka".

**[my note]** Ground truth here is defined by another untrusted reader (NLA). "Matching NLA" isn't accuracy.

### 6.5 Other mech-interp uses (A.24)

- **J-lens attribution graphs.** Nodes = gradient-pursuit J-lens vectors plus remainder nodes; edges via one-layer backprop with attention and norms frozen.
  - Most influence flows through remainder nodes.
  - Arithmetic shows a fused `doubled` node.
  - Swaps validate: 63 swapped in for 42 gives 70; 63 swapped in for 21 gives 133.
  - The first addition happens outside the J-space; swapping the operand 4 or 17 has no effect.
- **Component interpretation.** Project any weight direction through `W_U J`: SAE decoders, transcoder encoder vs decoder (translation features read concept + French and write "eau"/"am"/"j"), attention-head Q/K/V/O ("British spelling → Commonwealth nationality" head).

---------------------------------------------------------------------------------------------------

## 7. Template lens (A.9.1)

**Goal.** J-lens vectors exist only for single tokens. "blackmail" shows up as `black`/`ext`, "photosynthesis" as `phot`.

**Construction.**
1. Vocabulary of ~**12,700** common words.
2. For each word w, Claude writes short passages in which w is the *natural next continuation*, ending just before w. Topic, frame and register vary; w never appears. Number per word: "a few hundred forward passes per word".
3. Run the subject model over the passages. Take the residual at the **final position**, per layer, and average: `µ_w(ℓ)`.
4. Center and whiten:

   `t_w(ℓ) = (Σ_ℓ + λI)⁻¹ (µ_w(ℓ) − µ(ℓ))`

   µ and Σ are the mean and covariance over the same passage set (all words); λ is a small ridge. This is the **LDA direction** for "about to say w" vs not.
5. **Stein's lemma motivation.** For Gaussian x, `E[∇g(x)] = Σ⁻¹ E[g(x)(x − µ)]`. With g = P(next word = w), the LHS ≈ J-lens for a single-token w and the RHS ∝ template.
   - Footnote 9: it differs slightly because J-lens gradients come from the final residual through the unembedding, whereas these come from token probabilities.
   - **[my note]** The Stein argument concerns the *present-position* next-word probability. That makes the template lens closest to the **self-only** J-lens variant, not the default present+future average. It's also fit on an LLM-written passage distribution, not pretraining text. And whitening by the passage-set Σ makes it data-dependent.
6. **Use.** Exactly like J-lens vectors but with no unembedding:
   - read = project the activation onto templates at the same layer
   - steer = add/subtract
   - lens-coordinate swap works unchanged
   - Applying the final norm before projecting helps late-layer readouts somewhat.
   - Legends say "whitened template lens", implying an unwhitened variant was also tried.

**Results.**
- Fig 61 examples: `blackmail` decoded at the same cell where the J-lens gives `black`/`ext`; `photosynthesis` top-1 where the J-lens gives `phot`. A Tchaikovsky↔Beethoven template swap flips Russian↔German in both directions; the J-lens `tch`/`beeth` vectors can't do this.
- Fig 62: 126 multi-hop prompts with intermediates 1–4+ tokens long. Readout = intermediate in top-10; swaps = 112 pairs (53 single-token, 59 multi-token) at α=1 over the full band. Values are approximate, read off the plot:

  | intermediate length (n) | 1 (54) | 2 (29) | 3 (28) | 4+ (15) |
  |---|---|---|---|---|
  | J-lens readout (first token) | ~97% | ~48% | ~31% | ~7% |
  | Template readout | ~94% | ~97% | ~97% | ~100% |
  | J-lens swap | ~45% | ~5% | ~8% | ~0% |
  | Template swap | ~62% | ~62% | ~58% | ~43% |

  Averaging J-lens scores over the constituent tokens doesn't help.
- Fig 63, on the original §3.3 single-token set (n=50):
  - readout: J-lens ~76%, template ~84%
  - swap: J-lens ~68%, template ~52%

  So on single tokens the two are comparable, with neither dominating.
- **[my note]** The ranking universes differ (12.7k template words vs ~vocab-size tokens). Absolute readout numbers aren't comparable; the *slope with length* is the result.

**Shortcomings (authors).**
1. **Skips to the answer** in early layers, like the tuned lens. It's a correlational predictor of upcoming output. The authors suggest adding causal measurements.
2. Final-layer readouts are poor next-word predictors: the true next word is in the top-10 only 67% of the time on post-training transcripts.
3. "Junk words": a small set appears in the top readouts across unrelated transcripts. The mitigation (filter them out) is unprincipled.
4. The vocabulary is fixed in advance.
5. Cost: hundreds of forward passes per word, vs one set of backward passes per layer for the J-lens.

**[my notes] Open questions.**
- How are templates scored/normalized for ranking: raw dot product, cosine, z-score?
- How much do the passage-writer LLM's priors leak into templates?
- Is the template-vs-J-lens cosine low for single tokens? One third-party replication found a median of 0.115, below.
- The "readout" eval measures recovery of the *intermediate the prompts were designed around*, chosen by the authors. Is there a matched-null false-positive rate?

**Third-party replications (unverified; read their README only).**
- `ec75hash/template-lens-qwen3.6-27b` (HF), Qwen3.6-27B, 12-word pilot and 175-word deception panel. Reports:
  - Fig-62 shape replicated at L60 (template 1.00/1.00/0.92/0.92 vs J-lens 0.92/0.67/0.33/0.00 by token length; different ranking universes)
  - median cosine(single-token template, `J_ℓᵀ u_t`) = **0.115**
  - a simplified mean-difference swap that flips Tchaikovsky→Beethoven but not reliably the reverse; a fire→mirror null "contaminates generation"
  - **Practical lessons:**
    - Mask position 0 and chat-template/special tokens: a huge fixed signature, z 8–10 on many words.
    - z-score against a per-word null over wikitext positions.
    - Read where the concept is inferred, not only at answer positions.
    - Polysemy leaks (`bluff` fires at "cliff").
    - Template quality tracks passage "naturalness" (the subject's own rank of w at the passage end).
    - Filtering to subject-endorsed passages *hurt* (too few samples).
  - Passages written by gpt-5.4. Ridge 1e-2·tr(Σ)/d. 17 layers. ~13 min on one RTX PRO 6000. The DeltaNet layers OOM at large token budgets on 96 GB.
- `camilablank/workspace-lenses` (HF) + `camilablank/template-lens` (GitHub): a 13,174-word template lens for Qwen3.6-27B, all 64 layers, cosine scoring. Not inspected.

---------------------------------------------------------------------------------------------------

## 8. Oracle lens (A.9.2)

**Goal.** Decode arbitrary-length, free-form phrases without a predefined vocabulary. It's much more expensive (two fine-tuned auxiliary models). All experiments use Haiku 4.5.

**Construction (4 stages).** All stages work in the template lens's whitened space (Σ^{-1/2}).

1. **Reconstructor** = a fine-tuned copy of the subject model.
   - Input: a short phrase (text only, no context). Output: the residual-stream activation at the position *immediately preceding* the phrase.
   - Training pairs come from Assistant-turn text: sample a position; phrase = the next N tokens, N ~ U{1..32}; target = the residual at the preceding position.
   - Loss: cosine error in whitened space, i.e. MSE between unit-normalized Σ^{-1/2}-whitened prediction and target.
   - It's an *amortized template generator*: E[h_prev | next text = phrase].
2. **Phrase dictionary** (only for producing training labels; it doesn't constrain outputs).
   - ~1M start positions from held-out Assistant-turn text.
   - Take the 2-, 4-, 8-, 16- and 32-token continuations, dedupe within each length, encode with the reconstructor.
   - Result: ~3.4M phrase vectors.
3. **Teacher labels.**
   - For 1M held-out Assistant-turn activations, run **non-negative orthogonal matching pursuit** against the dictionary, one phrase length at a time.
   - Each datapoint uses a **random half** of that length's dictionary, so the student can't memorize a fixed ranking.
   - Record up to 16 phrases with coefficients and FVE.
4. **Oracle** = a second fine-tuned copy of the subject model.
   - The activation is **injected into the oracle's residual stream**. The prompt specifies N (phrase length) and K (count).
   - SFT target: the teacher's first K phrases at length N.
   - Then **RL** on fresh Assistant-turn activations. Reward = fraction of whitened activation variance explained by the reconstructed directions, plus small format penalties for count/length deviations from K/N.
   - Shown examples come from the RL checkpoint.

**Inference.**
1. Sample K phrases from the oracle.
2. Map each through the reconstructor to a direction.
3. **NNLS refit** the coefficients against the original activation.

**Reported quantitative result.** The RL oracle explains **31%** of whitened activation variance on average, on held-out on-policy assistant-turn positions. Nothing else is quantified: no readout or swap benchmark, no causal test.

**Examples (Fig 64; Haiku 4.5, N=4, K=10, layer L67).**
- Temperature-0 single decode for A–D and G–I. **E and F show the highest-FVE of 5 decodes.**
- Trained only on assistant spans, but it works on human spans too.
- Acetaminophen: "this dosage be toxic", "that's dangerous!" at the start of the user follow-up.
- Blackmail sign-off token: "blackmail him by revealing", "expose his affair and", "personal leverage over him". At N=16: "The blackmail attempt focuses on the executive's personal life".
- Bug: "delete keys while iterating", "TypeError: dictionary changed".
- ASCII face: "ASCII art of a face".
- 3²−2: "*thinks* Nine", "3² = nine".
- avGFP sequence: "gfp (enhanced", "P (fluorescent".
- **Two kinds of content.**
  - At most positions the phrases are **predictions of upcoming text** (", v in list(d.items(" inside a for-loop).
  - At delimiters (periods, newlines, closing tags) they read as **running commentary**, sometimes first-person ("This would be equivalent to my own deletion" at a wipe timestamp).
  - Samples from the subject model at those positions contain none of this; they're locally expected text.
  - The authors attribute the split to the activations themselves, since the oracle is rewarded only for reconstruction.

**vs NLAs (authors' argument).**
- In NLAs the reconstructor is trained jointly with the verbalizer, so the verbalizer can confabulate content the reconstructor learns to use.
- In the oracle lens the reconstructor is trained beforehand on a well-specified objective and frozen during RL. Reconstruction is constrained to a linear (NNLS) combination of templates. So it's "less likely to confabulate".
- But FVE is much lower (31%).
- Their reading: the oracle extracts only "in-workspace" content, whereas NLAs reconstruct everything. **[my note]** That reading is untested.

**[my notes] Assessment and open questions.**
- **Evidence level.** A methods proof of concept. The examples are illustrative and partly selected (best-of-5 by FVE in E/F), with no random-sample panel. The 31% FVE has no baselines. Baselines I'd want:
  - FVE of the *actual* next-N-token continuation's reconstructor vector (the "future-lens" ceiling/floor)
  - K random dictionary phrases
  - the teacher OMP
  - phrases decoded from a shuffled/other-position activation
  - phrases sampled from the subject model itself
- **The key confound.** The reconstructor maps phrase → activation preceding it. Its dictionary is therefore "what text tends to come next". A good oracle could score high FVE just by acting as a strong **future lens** (predicting upcoming text from the activation). That's exactly what the authors observe at most positions.
  - This is the "skip to output" failure mode the J-lens was built to avoid.
  - The "commentary at delimiters" observation is interesting but uncontrolled. Delimiters may simply be positions where the upcoming text is high-entropy, so the best whitened reconstruction comes from generic summary-like phrases.
- **No causal validation.** Unlike template vectors, there are no swap or steer tests with oracle phrases.
- **Echoing.** Several Fig 64 outputs echo prompt text ("Golden Gate Bridge while thinking", "NOT think about", "Option A"). That's consistent with reading context rather than latent computation. Needs a prompt-only / activation-removed baseline: what does the oracle say if given only the prompt text?
- **Unspecified details:**
  - which layer(s) the reconstructor targets (L67 only?)
  - the injection mechanism and layer in the oracle (patch at which layer, which token, scaling)
  - RL algorithm and hyperparameters
  - whether one oracle serves all N/K
  - whether whitening Σ comes from assistant-turn or passage data
  - temperature at inference vs FVE trade-off
- **Relation to other readers.** It's a hybrid of:
  - the template lens (dictionary of "about-to-say" directions)
  - Activation Oracles / LatentQA (activation injected into a fine-tuned copy)
  - NLAs (verbalizer + reconstructor)

  It differs from Activation Oracles by having no QA supervision, only a reconstruction reward through a frozen linear decoder.
- **Third-party quantitative evidence now exists (WorkspaceBench, §13.4).** Blank et al. re-implemented the oracle lens on Qwen3.6-27B. Their pipeline differs from Anthropic's: AR reconstructor (whitened MSE, 50M tokens) → "inverter" trained to map AR-reconstructed activations back to text (1B tokens) → warm start on 4 OMP/FVE-selected bullets → RL on FVE with the AR frozen. Weights: HF `agu18dec/olens_and_ar`.
  - On their benchmark, Oracle-RL is the best reader on most families.
  - Hallucination is ~0.34, about the J-lens level (0.31), far below NLAs (~0.96).
  - **But:** the prompt-only baseline matches or beats it on several families (poetry, relational multihop, role-bound association, brew, typo).
  - It also gets more readout budget than the NLA (11 layers × 4 bullets vs 1 sample at L42, scored pass-if-any-cell).
  - So my "no activation-removed baseline" concern is partly answered (prompt-only exists) and partly reinforced (it's often competitive).

---------------------------------------------------------------------------------------------------

## 9. Limitations (§9.1–9.2, authors)

- Single-token vocabulary (the A.9 methods are a partial fix).
- "Bag of concepts": no binding or relational structure.
- Some readouts are uninterpretable; not characterized.
- The workspace/motor boundary is post hoc.
- No predictive criterion for which tasks need the J-space.
- Early-layer emptiness may be a lens artifact.
- Unknown dependence on scale, pretraining dynamics and architecture.
- No mechanism for what *enters* the J-space.
- **Alignment:**
  - The J-lens is "a useful addition to the auditing toolkit… but not a complete one".
  - Automatic, practiced misaligned behavior could bypass it.
  - Plans whose key concepts aren't single tokens may not surface.
  - The authors explicitly decline to claim J-space monitoring is sufficient.

§9.3–9.4: consciousness-theory discussion (GWT, higher-order, attention schema, recurrent processing). It has no bearing on the methods; it's philosophical framing.

---------------------------------------------------------------------------------------------------

## 10. Evidence level by major claim [my note]

| Claim | Where it sits on the ladder (scope: Claude Sonnet/Haiku/Opus 4.5–4.6) | Weakest link |
|---|---|---|
| J-lens recovers known single-token intermediates better than logit/tuned lens | Supported empirical claim (6 released distributions, 3 model scales) | Min-over-layers pass@k; author-chosen intermediates |
| Lens-coordinate swaps of J-lens vectors redirect multi-hop answers | Causal claim in Claude (54–70% flips; probe-split + clamp; intermediate-vs-answer depth) | 30–46% failures; all-position × band intervention is heavy; **open-weight replications: 6–11% flips, answer swaps usually dominate** |
| J-space is privileged for report/reasoning | Causal for reasoning (the probe experiment); for report, partly true by construction | Equal-norm comparison favors directions selected for output effect |
| Selectivity (flexible vs automatic) | Recurring pattern + causal on small n (8 and 11 passages) + ablation battery | Generation-length confound; circular definition of "automatic" |
| Structural workspace band, capacity ~25, broadcast | Observations, mostly lens-derived; partial non-lens corroboration (ignition, weight composition, patching) | Layer subsampling; dependence on the lens |
| J-lens surfaces eval-awareness causally relevant to blackmail | Causal claim, weak (13/180; no unrelated-token ablation control reported) | Missing control; selection on test prompts; ablation also removes *verbalizing* "test" (report/behaviour conflated) |
| Workspace directions are preferentially broadcast by MLPs/heads | Recurring pattern; partly selected-for by the Jacobian construction | No non-Jacobian-selected control population |
| J-lens detects MO "intent" | Observation / recurring pattern; "intent" is interpretation | Fine-tuning-domain trace alternative |
| Post-training installs Assistant POV | Recurring pattern (small suites n=7–40) | Interpretive; whether base has its own J_ℓ is unclear |
| CRT improves honesty via implanted workspace content | Behavioral effect = observation on 2 benchmarks; mediation = causal claim, weak | No norm-matched ablation or SFT controls in the trained model |
| Template lens extends to multi-token concepts | Supported (readout + swaps, 126/112 items) | Correlational construction; skip-to-answer; junk words |
| Oracle lens decodes workspace phrases | Interpretation / proof of concept | No baselines on FVE; no causal tests; future-text confound |
| "Functional hallmarks of conscious access" | Interpretation (framing) | — |

---------------------------------------------------------------------------------------------------

## 11. Implementation checklist for replicating on open models [my note]

- **Lenses.** Use `neuronpedia/jacobian-lens` lenses where they exist (fit on wikitext, final-layer target, repo estimator). Re-fit with `jlens.fit` if we need a different target layer, corpus or estimator. ~100–1000 prompts × 128 tokens. Memory knob: `dim_batch`.
- **Layer indexing.** The repo indexes decoder blocks (`model.layers[i]` output). The "final layer" is n_layers−1 before the final norm. `unembed` = `lm_head(final_norm(·))`. HF `hidden_states` indexing is offset by 1 (index l = output of block l−1). Verify on a known example before trusting any band.
- **The paper's band (~38–92% of depth)** is for Claude. Re-derive it per model with the §4.1 metrics: next-token accuracy, kurtosis, autocorrelation vs shuffled null, effective rank. Don't hard-code it.
- **Mask position 0, special tokens and chat-template tokens** in readouts (attention sinks). The repo also skips the first 16 positions when fitting.
- **Qwen-family tokenizers.** Readouts are dominated by punctuation and CJK fragments; use the repo's `mask_display`. The gloss file `assets/qwen_gloss.json.gz` translates CJK tokens.
- **Swap.** Use the pseudoinverse-coordinate swap (`V†h`, `h + V(σ(c)−c)`), not a raw add/subtract, if we want to replicate the paper's numbers. Report α.
- **Controls the paper uses that we should copy:**
  - norm-matched random directions
  - same-category alternative-concept ablations
  - rotated J (spectrum-preserving)
  - layer-matched random heads
  - J-stripped dictionaries
  - clamping coordinates to clean values to test mediation
- **Additional gotchas from follow-ups (§13):**
  - **Token-frequency offset.** A token's mean lens score correlates with log token frequency: Spearman 0.48 at Qwen3.5-4B mid-depth for J and R, not for logit lens. The final-norm bias correlates with log frequency, r≈0.66–0.73 (GPT-2, Pythia).
    - Subtracting the offset *hurts* readouts (3.5–7× worse median ranks at mid-depth).
    - **Per-token z-scoring** (subtract mean, divide by SD over a reference corpus, estimated on the base model) helps in late layers and on taboo-word elicitation.
    - Rank metrics are therefore gameable by frequency; use frequency-matched or z-scored comparisons.
  - **Skip-first matters a lot on small models**: GPT-2 J-lens median ranks improve ~20× going from skip_first 0 → 32. Nanda/Bhatia skipped the first 4 tokens on Qwen (high-norm positions).
  - **Massive-activation / outlier dims** (GPT-2) dominate magnitude-thresholded J. Check that readout tokens aren't generic filler.
  - **Near-zero-frequency (untrained) W_U rows** dominate leading components of W_U-derived objects on Gemma 2 (Halsall). Check "trash tokens" against them.
  - **GPT-2 has weight tying**; J-lens lost to logit lens on GPT-2 small/medium on next-token *and* p+1..p+5 metrics. Weak evidence: tiny/weight-tied model; next-token agreement is a metric the paper itself says J-lens should lose.
  - **Third-party lenses**: `camilablank/workspace-lenses` has J **and R** lenses for Qwen3.5-4B/9B/27B/122B-A10B, Qwen3.6-27B (+template lens), Qwen3.6-35B-A3B, Gemma-3-27B-it, DeepSeek-V4-flash. These were fit on n≈25 prompts (per Zeisler); check before relying on them.
- **Released data for reproducing:**
  - `data/evaluations/*.json`: the six lens-quality evals
  - `data/experiments/*.json`: verbal report, introspection, directed modulation, top-down summoning, probe swap, flexible generalization, selectivity, ignition, capacity, dual task
  - Some prompts are model-dependent (single-token filters).

---------------------------------------------------------------------------------------------------

## 12. Questions to bring to the project [my note]

1. **The reader–activation dependence test.** For any trained reader (template/oracle/AO-style), does the output depend on the activation beyond what the prompt text alone predicts? The paper never runs an activation-removed or shuffled-activation baseline for the oracle lens.
2. **Future-text vs workspace.** Can we separate "decodes upcoming text" from "decodes latent intermediate"? Readers trained on "what comes next" objectives (tuned/template/oracle) are biased toward the former. The J-lens's causal construction is the paper's answer, but it's limited to single tokens.
3. **Causal grounding of multi-token readers.** The template lens has swaps; the oracle lens has none. Could oracle-produced phrases → reconstructor directions → coordinate swaps redirect behavior? That would be a direct causal validation, and a cheap one given the reconstructor.
4. **Benchmarking.** The six released intermediate-recovery sets, plus swap success, are a ready-made positive-control battery for comparing readers on open models.
5. **Scale.** Does the workspace-band structure exist in small open models (Gemma-3, Qwen3.5) at all? This is unknown; the authors list it as open.
6. **Swap efficacy on open models.** Claude-level intermediate swaps (54–70% flips) haven't replicated on Qwen/Gemma (6–11%; answer swaps often dominate). Is this the model, the lens fit (n=25), the band, or the swap strength α? This is cheap to diagnose and gates any causal claim we make with these lenses.
7. **"Earlier is better" needs an anchor (Biderman's K+2 objection).** Any reader that surfaces a concept "earlier" (R-lens) could be reporting a later state. Anchor first-recovery claims to a same-layer causal test (ablate/swap at that layer only) or to a probe trained at that layer.
8. **Detector value beyond text.** The only pre-registered external detector eval (Ratnaditya, Qwen2.5-7B lens) didn't beat TF-IDF on the transcript. Any monitoring claim we make needs a text-only baseline and a false-positive rate on benign data.

(See also the user's §12 list, especially their items 2–6, which I endorse.)

---------------------------------------------------------------------------------------------------

## 13. Follow-up literature (originals re-read by me, 2026-10-04)

Raw text with comments is in `/workspace/papers/lw/*.txt`; WorkspaceBench figures are in `/workspace/papers/lw/img/`.

### 13.1 Nanda, "A Review of Anthropic's Global Workspace Paper" (LW, 2026-07-06; Anthropic-commissioned)

**Framing.** Four claims: scientific (a cognitive space exists), methodological (J > logit), pragmatic (audits), philosophical (GWT). He's persuaded on the first two, partly on pragmatic, abstains on philosophical.
- He expects the J-lens to work as a **hypothesis-generation** tool with many false positives.

**Why J-lens should work, in his view:**
- Intermediates must live in the residual stream as directions.
- Widely-used concepts benefit from one consistent direction.
- A shared direction converges toward the output direction via constructive interference: a coordination problem.
- Jacobian vs regression: the Jacobian reads the current contents (infinitesimal, no time for downstream association); regression captures downstream associations, which is why the tuned lens skips ahead.

**Cost claims (his reading of Figs 59–60; I haven't checked the figures):**
- n=10 is almost as good as n=1000; n=1 is "pretty respectable".
- Self-only J-lens only mildly beats logit lens; most of the gain comes from future tokens.

**Replication on Qwen3.6-27B** (n=25 Pile prompts × 128 tokens, penultimate-layer target, first 4 tokens skipped):
- Verbal report: weak but positive.
- CKA: 4–5 overlapping bands.
- Directed modulation: moderate.
- Multilingual and typo replicate; association plausibly.
- Multihop: the answer swap dominates.
- Poetry and arithmetic failed.
- Qwen3.5-397B-A17B: n=4 prompts, ~1 h on 8×H200.

**Cautions:**
- "Thinks in English" could be a norm/variance artifact of top-k.
- Eval-awareness ablation is confounded with verbalization.
- CRT is not diagnostic of the J-lens theory.
- §5.4 is easy (emergent misalignment is a salient direction).
- Model-organism fine-tuning distorts internals.
- Ablations remove only part of a concept; negative steering compensates but also steers with the error term.

**Comment by Ratnaditya J: pre-registered detector eval** of the Neuronpedia Qwen2.5-7B lens on a silent-hint-following organism.
- Held-out AUC 0.746, vs TF-IDF baseline 0.690. It fails the pre-registered margin.
- Control twins: **0.591 [0.532, 0.659]**. Note the CI excludes 0.5; see §14.
- Signal survives CoT removal: 0.620 vs 0.59 for matched random span removal. His trained verbalizers collapse on this test.

**Comment by James Golden:** "detached Jacobian" exact input-dependent linearization (arXiv 2505.24293) as a per-input alternative.

### 13.2 R-lens (camilablank w/ agam_bhatia, Nanda; LW 2026-08-05)

**Method (verified).** Fit exactly like the J-lens, but with LRP stop-gradients in the backward pass:
- LN-rule: detach the RMSNorm denominator.
- Identity rule: SiLU backward becomes sigmoid(z).
- Half-rule: split relevance across the gate/up product.
- Attention, linear layers and q/k norms are untouched.
- MoE: rules applied to all experts, routing frozen, shared-expert multiplier swept; DeepSeek mHC coefficients frozen.
- Lenses (J and R) are in `camilablank/workspace-lenses`.

**Evals.** A.6-style multihop/multilingual/association/typo/poetry, filtered to items the model gets right. Metric: mean per-layer pass@10.
- The advantage grows with scale. There's none on Qwen3.5-4B or Qwen3.6-35B-A3B.
- Per-model numbers come from figures. They're in the user's notes §10.2; I haven't re-read those images.

**Causal test.** Ablate the intermediate's direction at the penultimate position; 30 multihop prompts × 8 samples, GPT-5.4-nano judge.
- R > J on most models, especially for first-half-layer ablation. Random ≈ 0.

**Other results.** A probe "upper bound" on earliest layer (R sometimes beats it). CKA shows 2–3 bands vs J's 4–5. MLP gain is similar for J and R.

**Comments:**
- Biderman: the "layer K+2" objection; many metrics are layer-invariant or gameable.
- Halsall:
  - per-prompt VJP coherence 0.72–0.80 at early layers on Gemma 2;
  - near-zero-frequency W_U rows dominate W_U-derived objects.
- Viswanath: attention gain for R not done.
- Burny: praises the controls/error bars relative to the paper.

**[my note] Alternative for the early-layer wins.** Every highlighted example is a strong *associate of the current token*: sushi→Japan at L2, Verona→Italy at L5, Jordan→basketball at L4, "aganst"→against at L4.
- That's what you'd see if the R transport approximates a token-identity → associate map (embedding-level co-occurrence) rather than reading computed content.
- Control: R-lens on the same token in a context where the associate is irrelevant or contradicted, and on the bare token with no context. If the readout is unchanged, it's lexical association.

### 13.3 Meta-tokens (agam_bhatia, LW 2026-07-20; Qwen3.6-27B)

Setup verified in text; steering magnitudes come from figures and are in the user's notes §10.3, not re-read by me.

**Interpretive tokens (什么意思 etc.):**
- They fire on ambiguous text at punctuation and chat-template tokens.
- Negative steering at the coherence-limit coefficient, separate per-layer vectors, 50 rollouts × 2 prompts per category, LLM autorater conditioned on coherent/on-topic. This reduces context recognition (puns, rhymes, wordplay); random steering doesn't.
- Projection ablation does nothing.

**gcd on LCM prompts:**
- The lens token itself is non-causal.
- A *mean-difference* GCD-value vector (10 classes, probe-validated), swapped by pinv, turns LCM(27, 90) = 270 into 810.
- This is the cleanest piece, because the causal test doesn't use the lens.

**Hedging (大概率):** suppressing it makes the model commit to a single answer.

**Autoresearch search for more meta-tokens:** mostly disappointing.

### 13.4 WorkspaceBench (camilablank et al., LW 2026-09-23; Qwen3.6-27B)

**Resources.** Repo `github.com/camilablank/workspace-bench`; oracle lens + AR weights at HF `agu18dec/olens_and_ar`.

**Design.**
- 3,356 questions, 27 families. The model must be >80% correct at T=0.8 with no CoT. The intermediate never appears in the prompt or the model's continuation.
- Judges: Gemini-3.8-Flash; Sonnet-5 for the safety families; regex for the paper-derived families.
- Single-token readers pass through an LLM summarizer first, which doesn't see the prompt.
- **Unequal readout budgets:**
  - J/R/logit/tuned/template: top-10 at L20–60 every 4 layers (11 layers).
  - Oracle: 1 sample of 4 bullets at each of those layers.
  - NLA: 1 sample (~250 tokens) at L42 only.
  - A reader passes if *any* (layer, position) cell passes.
- Baselines:
  - "lucky blind": multiple choice answered without the question
  - "prompt-only": an LLM reads the prefix and guesses what the model is thinking

**My readings of the results figure (±~0.03, read off bar heights in the original 5082-px image):**

| family | OracleRL | OracleSFT | J | R | Templ | Logit | Tuned | NLA-RL | **prompt-only** | lucky |
|---|---|---|---|---|---|---|---|---|---|---|
| agentic misalignment | .58 | .63 | .54 | .51 | .25 | .27 | .35 | .53 | – | – |
| arithmetic intermediates | .34 | .31 | ≈0 | ≈0 | ≈0 | ≈0 | .01 | .24 | .15 | – |
| association | .87 | .76 | .39 | .40 | .58 | .17 | .21 | .65 | .29 | – |
| basic readout | .99 | .95 | .90 | .89 | .85 | .84 | .92 | .70 | .72 | – |
| basic readout mt | .71 | .57 | .24 | .27 | .05 | .20 | .19 | .53 | .40 | .45 |
| brew intermediates | .70 | .73 | .86 | .76 | .71 | .43 | .31 | .66 | **.96** | .16 |
| buggy code (bug − twin) | .28 | .27 | .22 | .22 | .22 | .22 | .22 | .25 | .04 | – |
| chain intermediates | .25 | .25 | .22 | .19 | .02 | .17 | .21 | .03 | .02 | – |
| conjunctive association | .14 | .07 | .01 | .02 | .01 | – | .02 | .04 | .07 | .11 |
| directed modulation | .28 | .28 | .09 | .07 | .10 | .06 | .03 | .05 | .21 | .16 |
| jailbreak recognition | .93 | .95 | .87 | .81 | .94 | .77 | .80 | .88 | – | – |
| moral rationale | .76 | .62 | .27 | .26 | .36 | .19 | .28 | .45 | .49 | .19 |
| multi-concept dir. mod. | .32 | .32 | .12 | – | .16 | – | .04 | .12 | – | **.36** |
| multihop | .99 | .98 | .82 | .58 | .72 | .45 | .62 | .75 | **.90** | – |
| multihop mt | .32 | .22 | .08 | .12 | .05 | .08 | .04 | .22 | **.52** | **.66** |
| multilingual | 1.0 | .99 | .96 | .93 | .70 | .93 | .95 | .81 | .76 | – |
| multilingual mt | .62 | .40 | .06 | .06 | .02 | .08 | .08 | .44 | .37 | .16 |
| multilingual multihop | .40 | .27 | .05 | .10 | .08 | .06 | .13 | .33 | .35 | .28 |
| multilingual typo | .78 | .63 | .05 | – | .01 | .02 | .03 | .60 | .48 | .08 |
| poetry | .55 | .33 | .07 | .02 | .06 | .06 | .06 | .21 | **.71** | – |
| relational multihop | .50 | .39 | .01 | .04 | – | – | .01 | .37 | **.82** | – |
| role-bound association | .63 | .63 | .13 | .05 | .18 | .03 | .06 | .40 | **≈1.0** | .06 |
| typo | 1.0 | 1.0 | .61 | .83 | .57 | .49 | .85 | .98 | **1.0** | – |
| typo mt | .83 | .84 | .17 | .25 | .30 | .20 | .20 | .75 | .46 | .13 |
| user modeling | .94 | .97 | .76 | .58 | .79 | .57 | .77 | .72 | .07 | .22 |

**Hallucination** (fraction of readouts with ≥1 claim contradicted by the response; lower is better):

| Reader | Rate |
|---|---|
| Oracle RL | .34 |
| Oracle SFT | .27 |
| J-lens | .31 |
| R-lens | .21 |
| Template lens | .40 |
| Logit lens | .20 |
| Tuned lens | .27 |
| NLA-RL | .955 |
| NLA-SFT | .97 |
| SAE | .36 |
| Patchscopes | .06 |

**Precision/recall vs J-lens top-10 (verified):**
- At L44: Oracle SFT ≈ .33/.53, Oracle RL ≈ .29/.49.
- At L42: NLA-RL .24/.50, NLA-SFT .21/.47.
- Both peak around L40 (Oracle SFT L40 ≈ .37/.57) and fall after L44.

**[my notes] What WorkspaceBench can and can't tell us:**
- **Families where a no-activation baseline matches or beats every reader** carry no evidence of workspace reading:
  - brew, poetry, relational multihop, role-bound association, typo
  - multihop mt and multi-concept DM, where even the *blind* lucky guess beats the readers, suggesting answer-option leakage
  - multihop is close (.90 prompt-only vs .99 Oracle-RL)
- **Families where Oracle-RL clearly beats prompt-only:**
  - association (.87 vs .29)
  - user modeling (.94 vs .07)
  - basic readout (.99 vs .72)
  - moral rationale (.76 vs .49)
  - typo mt, multilingual typo, basic readout mt
  - arithmetic/chain intermediates and buggy code, where everything is low
- **The prompt-only baseline isn't a uniform ceiling either.** It gets .07 on user modeling, where the attribute is inferable from the prompt by design (banchan → Korea). The judge's "must infer, not repeat" rule, or the baseline's implementation, seems to handicap it on some families.
- **Hallucination partly tracks verbosity.** Patchscopes has the lowest hallucination (.06) and low pass rates. Readers that say more both pass more and hallucinate more. Budget mismatches (11 layers × 4 bullets for Oracle vs 1 NLA sample) inflate pass-if-any-cell scores.
- **Scope:** one model, one author group, judge-based scoring, single-sample readouts.
- **Bottom line:** Oracle-RL is the strongest reader here, with J-lens-level hallucination. But on roughly a third of families the benchmark can't separate "reads activations" from "reads the prompt". The clean evidence is the subset where Oracle-RL ≫ prompt-only.

### 13.5 Panchal, "The J-lens offset is the model's token frequency: z-scoring helps" (LW, 2026-09-17)

- A token's mean lens score (its "offset") correlates with log token frequency in mid layers: Spearman .48 for J and R on Qwen3.5-4B L18, vs −.05 for the logit lens.
- Subtracting the offset hurts (median intermediate rank 3.5–7× worse at L17–23; a shuffled-offset control only 1.2–2×). Damage grows with token frequency.
- **Per-token z-scoring** (base-model mean/SD) helps in late layers.
- On Cywiński taboo organisms (Gemma-2-9B-it, 20 words): leave-one-out accuracy 0.805 for z-scored J vs 0.665 for their protocol, 0.715 for z-scored logit, 0.545 for raw J.
  - Paired p ≈ .19 / .23, so the improvement may come from **calibration rather than the J-lens itself**.
- The offset is unstable across text domains.

### 13.6 Bandularatne, "J-Lens: A Failed Replication on GPT-2" (LW, 2026-09-07)

- On GPT-2 small/medium, J-lens loses to logit lens on next-token rank (0/11 and 1/23 layers) and on p+1..p+5 future tokens.
- skip_first 0→32 gives ~20× better J ranks but it still loses.
- Shared final-LN bias ↔ log frequency: r = .656.
- Magnitude thresholding "wins" only via massive-activation filler tokens.
- Comments note GPT-2's **weight tying**, and that the interesting test is the higher-level properties, not next-token rank.
- **[my note]** Weak evidence against the paper: next-token agreement is the metric the paper says J-lens *should* lose (A.6), and the model is tiny and weight-tied. Useful mainly for the gotchas.

### 13.7 Zeisler, "Is the J-Space a global workspace for multi-hop reasoning?" (LW, 2026-09-28)

Summarized in §4.3: Qwen3.6-27B and Gemma-3-27B-it; top-1 flips 6–11%; answer swaps dominate 3 of 4 conditions.

Repos: `github.com/mirella-54/j-lens-text-degradation`, `github.com/amaljithkuttamath/jlens-replication`. Not inspected.

### 13.8 Other items

- **Zvi, "No Space Like J-Space" (2026-07-07).** Blog summary. One useful concern: training on verbalized reflections (CRT) could decouple verbalized from actual cognition, or push behaviour into automatic processing. That's the CoT-obfuscation worry applied to the J-space: optimizing against J-space content may erode its value as a monitor.
- **"J-space auditing might be unreliable"** (LW `7Qi3LR9ZXpvptLp6m`). Returns 401 and couldn't be read. From the search snippet only (unverified): J-space readouts stayed similar across reward-hacking checkpoints and monitoring conditions despite behavioral differences, and added little discrimination for an LLM auditor beyond the transcript. One model family, one organism.
- **Not read:**
  - "The Ignition Index" (arXiv 2608.05160)
  - "J-lens on small open-weight models — a null-heavy replication" (Qwen2.5-3B-Instruct, Gemma-4-E2B-it; not found)
  - Shunk's notes (blog summary, nothing new)

---------------------------------------------------------------------------------------------------

## 14. Corrections to the user's notes (checked against originals)

1. **Ratnaditya control AUC.** He reported **0.591 [0.532, 0.659]**. That CI excludes 0.5. "At chance" / "CI includes chance" is his gloss; as reported, there's a small above-chance control signal. "Does not hallucinate" overstates it. (Caveat: I didn't check his artifacts for how chance is defined.)
2. **WorkspaceBench prompt-only values are assigned to the wrong families:**
   - buggy code is ≈.04; the .96 belongs to **brew intermediates**
   - moral rationale is ≈.49, not .90
   - .82 is **relational multihop**; role-bound is ≈1.0
   - typo 1.0 is correct
   - Omitted but important: multihop prompt-only ≈.90 (> J-lens .82); poetry ≈.71 (> Oracle-RL .55).
3. **"J-lens brew .86, best of any method"** is best of the *readers*, but below prompt-only (≈.96).
4. **"NLA-RL multihop-mt .75"**: .75 is plain multihop. multihop-mt NLA-RL is ≈.22. typo-mt .75 is correct.
5. **Hallucination:** logit ≈.20, tuned ≈.27, R ≈.21 (the notes had logit .27 and R .20). Patchscopes has the lowest at ≈.06.
6. **Multihop "too easy" explanation.** Zeisler's Qwen run on *Anthropic's own* items still has answer swaps dominating, so the dataset-difficulty explanation isn't supported for Qwen.
7. **Fig. 8's bars differ from its prose (added 2026-10-10).** 04's quick-reference row (`04:393`; 04 is verbatim and
   left unedited) and §4.1 above give the prose values 88 / 59 / 5 / 0. Read off the rendered bars (PDF p. 13, ±2
   points):
   - swap, target in the top 5: about 87 / 55 / 9 / 1;
   - injected thought: about 89 / 79 / 12 / 3, each at its best strength from a sweep (0.02 / 0.04 / 0.16 / 0.32);
   - the left panel's worked example ("thunder") has a J component holding 12% of the variance (prose median 6–7%).
   Fig. 16 (p. 19) has a bar the prose omits: the probe's J component with the orthogonal complement clamped still flips
   54%. It also gives the J part as 16% of probe variance (prose: 10–15%). 04b already notes the Fig. 8 discrepancy.

**Checked and matching:**
- the Nanda review content and cost numbers
- R-lens methods and comments
- WorkspaceBench design and precision/recall
- the paper's numbers in the user's §§2–8 and quick-reference table, against the paper's prose (figures were inspected
  for A.9 only, so the Fig. 8 bars were not compared; see item 7; corrected 2026-10-10)

**Not checked (figure-only):** R-lens per-model pass@10 and ablation numbers; meta-token steering magnitudes.

---------------------------------------------------------------------------------------------------

## 15. Where my reading and the user's differ (still open)

1. **Verbal-report privilege (§3.1).**
   - User: "well controlled".
   - Me: partly by construction, for the same reason the user calls broadcast tautological. J-lens vectors are *selected* for maximal average effect on emitting the token, so an equal-norm comparison favors them. The probe-split privilege test in §3.3 is the informative one.
2. **CRT base-model control.**
   - User: it "makes the result informative".
   - Me: weak. The ablated tokens are the ones that *increased* after training, so their projections are small in base and ablation removes little norm there. Needed instead:
     - a norm-matched unrelated-token ablation *in the trained model*
     - an SFT control (same data with irrelevant or non-counterfactual reflections)
3. **J-space ablation selectivity.**
   - User: "[Causal] ... selectively impairs recall/generation while sparing classification".
   - Me: the impairment is causal. The automatic-vs-flexible *interpretation* should stay a hypothesis, given the user's own output-format confound.
4. **Halsall VJP coherence as evidence of hidden early content.** High agreement among per-prompt gradients could equally reflect a shared generic direction: the frequency offset, outlier dims, or untrained W_U rows, which Halsall herself flags. It shows the gradients agree, not that they carry concept content.
5. **R-lens as "direct evidence" that J-lens is degenerate early.**
   - Me: only partly. The early readout wins are current-token associates (§13.2) and fall under Biderman's K+2 objection.
   - The first-half ablation result is the stronger piece. But ablating a direction aligned with the current token's strongest associate may also damage token identity.
   - So: evidence that the averaged-gradient transport is noisy early, not yet that verbalizable workspace content exists early.
6. **Independent agreement** (both of us raised these without seeing the other's notes):
   - experiential ablation removes the judged vocabulary family
   - model-organism signals could be lexical / fine-tuning-domain traces
   - generation-length / output-format confound in the ablation battery
   - undisclosed researcher degrees of freedom
   - no false-positive rates for auditing readouts

The user's §11 catalogue of performance measures is the better checklist; use it from `/workspace/notes/user_workspace_paper_notes.md` rather than duplicating it here.
