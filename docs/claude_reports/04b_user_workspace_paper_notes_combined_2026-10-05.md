# Combined reading notes on the Anthropic workspace paper

These notes combine the supplied paper notes with a close reading of the main text, technical appendices, figures, and linked audit examples in *Verbalizable Representations Form a Global Workspace in Language Models*, Gurnee et al., arXiv 2607.15495v1. They are a reference for discussing the paper. Project selection and experiment planning remain deferred.

The central result is a collection of interventions showing that a sparse family of token-labelled directions has privileged causal roles in some verbal reports, intermediate computations, and flexible uses of information. The evidence is considerably narrower than a complete account of reasoning, a reliable general detector of hidden intent, or an assessment of phenomenal consciousness. The strength of the paper comes from converging experiments; individual readable token lists do not inherit the strength of the whole collection.

## Sources and conventions

- [Local paper](</Users/vishesh/Desktop/mech interp research/workspace/paper/2607.15495v1.pdf>). Page references below are printed PDF pages, which match one-based file pages.
- [Original supplied notes](</Users/vishesh/Desktop/mech interp research/workspace/paper_notes/workspace_paper_notes (1).md>). The original remains the record of the user's wording and assessments.
- [Official HTML paper and interactive figures](https://transformer-circuits.pub/2026/workspace/index.html).
- [Authors' implementation and prompt repository](https://github.com/anthropics/jacobian-lens). The present synthesis checks the paper, rather than claiming independent reproduction of its experiments or a code audit.

Unless specified otherwise, experiments use Claude Sonnet 4.5. **L71 means approximately 71 percent of model depth**, not physical transformer block 71: the paper samples 25 evenly spaced residual-stream layers and reindexes them to 0–100. Haiku 4.5 and Opus 4.5 corroborate selected results; Opus 4.6 appears in auditing; counterfactual reflection training and the original oracle lens use Haiku 4.5. External Qwen studies below use their own layer indices; do not apply the Claude percentage convention to those numbers.

**Observation** refers to a reported measurement in a specified setting. **Supported claim** means a pattern with relevant comparisons or controls. **Causal evidence** means an intervention with an interpretable behavioral consequence, qualified by the controls actually performed. **Interpretation** and **hypothesis** identify explanations not uniquely established by those observations. These labels apply to individual claims, rather than entire sections.

## Working interpretation

Four questions should remain separate.

1. **Scientific:** Do models have a shared representational format that carries some reportable and causally important intermediates? The paper provides substantial evidence in its tested settings.
2. **Methodological:** Does the Jacobian lens expose that content better than existing vocabulary readers? It outperforms logit and tuned lenses on the paper's intermediate-recovery and swap evaluations. This does not make each token label faithful or calibrated.
3. **Practical:** Does the tool help an investigator? The selected audits, model-organism comparisons, and small automated-investigation evaluation support usefulness. They do not establish deployment precision, coverage, or reliability for arbitrary hidden objectives.
4. **Theoretical:** Is the format appropriately described as a global workspace? The demonstrated functional similarities make that a substantive interpretation. Encapsulated specialist processors, recurrent broadcast, and the full human architecture are not established. Phenomenal experience remains outside the demonstrated claims.

This largely preserves the original notes' assessment. One adjustment is to avoid treating all structural evidence as equally weak: lens-derived depth statistics have an obvious instrument-dependence problem, while the independently measured activation transition and the functional consequences of ablating selected attention heads add evidence beyond those statistics.

## The Jacobian lens and the J space

### Construction and readout

The main-text definition averages the downstream residual-stream Jacobian over prompts, source positions, and present or future target positions:

`J_l = E[prompt, t, t' >= t] [d h_final,t' / d h_l,t]`

`lens(h_l) = softmax(W_U norm(J_l h_l))`

Each layer has one `d_model × d_model` matrix. Rows of `W_U J_l` define one residual-stream direction per vocabulary token. The intended interpretation is a vector's average disposition to affect verbal output across contexts. It is not a reconstruction of the exact computation being performed in the current prompt. A local derivative is exact only to first order around its evaluation point; applying its corpus average to a full activation is a further approximation. (§2, pp. 6–10.)

An infinitesimal perturbation still propagates through all downstream nonlinear computations, via their local derivatives. The notes' “no time for nonlinear processing” intuition should therefore be discarded. The current-content interpretation depends on averaging and empirical validation, not a guarantee that derivatives cannot reflect downstream associations.

The default corpus is 1,000 sequences of 128 tokens. The appendix pseudocode injects a basis-vector gradient at every target position, differentiates their sum, averages over source positions, and then averages over prompts. It performs one backward pass per output dimension, batched in practice. About ten prompts already beat the compared lenses on the reported benchmarks; this is not a universal compute or sample-complexity guarantee. (A.7, pp. 68–71.)

**Default ambiguity:** A.7 calls the penultimate layer the default target for Sonnet 4.5, while the main formula, pseudocode comment, and a figure label describe final-layer targeting. The tested recipes perform broadly similarly, but reproducing an exact result requires resolving this against the implementation. The pseudocode also sums target-position effects and averages source positions; an implementation should not silently substitute a differently weighted interpretation of the expectation.

Variants include final versus penultimate targets; full gradients versus frozen queries and keys; present-only versus future-only versus combined effects; mean versus median aggregation; and corpus filtering. Present-only readouts perform worse in the reported aggregate comparisons. Frozen QK can increase the intervention effect. These ablations characterize robustness, not equivalence of every recipe.

### Three different kinds of reading

| Operation | What it returns | Important limit |
|---|---|---|
| Ranked lens readout | Vocabulary tokens ordered by lens score | Related tokens can be redundant; rank is not probability that a thought exists. |
| Chosen-token probe | Score or cosine against a selected direction | The researcher supplies the concept and any threshold. |
| Sparse decomposition | A nonnegative combination of selected directions and coefficients | Depends on sparsity and greedy fitting; differs from the top-k ranking. |

The dictionary is overcomplete. It typically spans the full residual stream, although having more vectors than dimensions alone does not guarantee full rank. The intended J-space is the set `F_k = {Va : a >= 0, ||a||_0 <= k}`: a union of sparse nonnegative cones. It is not a fixed low-dimensional linear subspace. Its estimated component is obtained by nonnegative gradient pursuit; the remainder is relative to that particular approximation. (§2.3; A.8.)

The supplied notes correctly identify a formal inconsistency in A.8: the prose specifies nonnegative cones, but the displayed definition uses linear spans and unrestricted orthogonal projections. Those are different sets. Vocabulary enlargement makes the set weakly larger; strict enlargement requires a nonredundant addition. The proposed distances compare approximation-error profiles over a distribution, which need not determine identical causal behavior or concept assignments.

### Interventions

Steering adds `alpha v` to an activation. Projection removal suppresses its component along a direction. A coordinate swap accounts for nonorthogonality: construct `V = [v_source, v_target]`, calculate `c = V†h`, exchange the coordinates, and write `h' = h + V(swap(c) - c)`. The orthogonal complement of the two-vector span is preserved. (§2.5, p. 10.)

Interventions often run at every token position over a layer band. Successful swaps can therefore support a content-specific causal effect without uniquely locating a particular site or validating every edge of a graph. Clamping selected coordinates to clean-run values asks whether an effect can persist without changes in those coordinates; this is distinct from a simple one-time injection.

## Functional evidence

### Verbal report and privilege

In 14 concept categories, the lens readout at the colon before a requested one-word report correlates with the ensuing output distribution. Swapping to a same-category item outside the output top ten promotes that alternative. A second protocol injects a concept on the user turn and elicits its name in a partly prefilled introspective response. Across 100 concepts, the effect concentrates at the reporting position rather than uniformly promoting the injected word throughout the response. (§3.1, pp. 10–13, Figs. 6–8.)

The stronger privilege test constructs concept vectors independently, using mean-subtracted activations for “Tell me about” prompts. Their k=16 J-component explains a median 6–7 percent of vector variance. The prose reports top-five swap success of 88 percent for raw lens vectors, 59 percent for the J-component, and 5 percent for the remainder. Norm matching matters. Clamping the relevant J-coordinates to clean values removes almost all remaining effect of the non-J component. The exact prose percentages differ somewhat from the rendered Fig. 8 bars, so they should be described as prose-reported values.

**Supported scope:** in these elicitation protocols, a small J-component has disproportionately strong access to verbal report. The result does not establish generally accurate spontaneous introspection.

### Directed modulation

Holding a concept or evaluating an expression while copying unrelated text produces corresponding lens content at copied-token positions. The example `3²−2` surfaces arithmetic, nine, then seven; later layers return to the imminent text token. However, A.10 materially changes the interpretation: mere mention nearly matches explicit focus for category and arithmetic tasks; “don't think” often fails to suppress; “ignore” suppresses more effectively. Counts are often based on a target reaching top one anywhere across inspected positions and layers. (§3.2; A.10–A.12.)

The strongest complementary designs change the task while holding the stimulus fixed, or distinguish an instructed label from an independently measured property representation. In A.12, imagining JavaScript as Python changes the Python label readout while leaving a J-stripped property probe near its baseline; real Python changes that probe much more. This supports selective modulation of a reportable format, not the claim that instruction cannot affect any non-J representation. There are 24 stimuli in each of four property categories.

### Silent reasoning and intermediate use

The selected examples have meaningful counterfactual predictions: spider→ant changes a legs answer from eight to six; fight→light changes “coming fight” to “morning light”; English-aligned swaps alter a Chinese antonym answer; repeat/switch swaps reverse an A/B choice. These tests go beyond an arbitrary output change. (§3.3, pp. 15–19.)

Across 50 two-hop prompts, a random same-category intermediate swap yields the predicted top-one answer on 54 percent of Haiku, 70 percent of Sonnet, and 70 percent of Opus 4.5 trials. Intermediate swaps become effective a median of about 17 percentage points of depth before answer swaps, measured on successful Sonnet trials. This weakens a simple answer-contamination account; it does not rule out every correlated or parallel representation account.

In a separate 90-prompt test, independently constructed intermediate probes are split with k=25 pursuit. Raw J-vector swaps succeed on 60 percent, probe J-components on 61 percent, and remainders on 28 percent. Clamping relevant J-coordinates reduces the remainder effect to 6 percent; J-component effects largely survive complementary clamping. This is among the paper's more discriminating mediation results.

### Flexible use and task dependence

The same argument swap is tested across different functions: four categories, four functions per category, and twelve ordered argument pairs per function. Success is **76/192 at ordinary strength and 101/192 at double strength**. Countries account for 42/48 ordinary-strength successes; number words account for 0/48. Failure can mean either computation outside the measured format or a poor match between the concept's actual representation and its token direction. (§3.4; A.13.)

Two task comparisons help distinguish decoded information from causally used information. Across eight language passages, a language-label swap affects naming and flexible facts much more than continuation or anomaly detection, even when the label is readable in every condition. Across eleven line-count passages, number labels appear when the count is requested or needed for a further operation, but not during ordinary wrapping; decade-family swaps affect the former tasks much more. These are small, useful dissociations, not universal laws of automatic versus deliberate cognition. (§3.5.1.)

A.14 adds an instructive reversal: early-layer removal of an implied concept impairs actively avoiding its name while leaving naming relatively intact; late removal suppresses naming. The 63 items were selected for reliable baseline behavior, and a norm-matched wrong-concept intervention has a much smaller effect.

### Broad ablation and language of experience

The broad ablation removes the ten strongest lens directions locally, excluding tokens in the clean model's top-ten output predictions. This protection of likely output tokens is a reasonable design choice, but contributes to the measured selectivity. (§3.5.2, Fig. 22.)

| Condition | Layer band | Two-hop accuracy | Agreement with clean pretraining top-one prediction |
|---|---|---:|---:|
| Clean | None | 98% | 100% |
| Light | L62–71 | 68% | 87% |
| Medium | L54–79 | 26% | 75% |
| Heavy | L42–88 | 6% | 65% |
| Random-direction control | L54–79 | 98% | 96% |

The pretraining comparison covers 458,856 positions. Heavy ablation thus has meaningful collateral effects even though several benchmark scores remain comparatively robust. The 14-task battery shows differential impairment; the original notes' output-format alternative is worth retaining as a hypothesis, but “all spared tasks are classification” is too categorical. Extractive QA is among the spared tasks, and selected tasks do not establish a clean alternative taxonomy. Chain-of-thought math being more robust is consistent with externalized intermediates, while also changing the available computation and output format.

Early J-ablation also reduces experiential language on self-description, imagined human experience, and stories. Several per-position norm-matched controls weaken a generic-perturbation explanation. Yet the outcome is three LLM judgments about linguistic register, not whether an experience occurred or a self-report was truthful. Some control dictionaries have partial effects; “all controls unchanged” is too strong. The third-person effect is central evidence about the breadth of the affected representation. (§3.5.3; A.23.)

## Structural evidence

### Depth and the early-layer question

CKA of lens-vector geometry, readout kurtosis, persistence across positions, effective rank, and next-token agreement identify an approximate L38–92 band in the main Claude analyses. These are mutually related lens-derived measurements. Their agreement does not exclude a common reader failure in early layers, and subsampling exaggerates boundary sharpness. (§4.1.)

The ambiguous-input experiment adds evidence independent of lens readout: interpolate between two country embeddings and measure the resulting activation's projection between clean endpoints. Around the proposed onset, smooth responses become more categorical. The appendix examines transition widths and across-prompt bimodality. This establishes a nonlinear commitment pattern under artificial mixtures, rather than the recurrent neuronal mechanism of ignition. The text and caption say 16 country pairs ×40 carriers; a figure inset says 49 carriers, an unresolved reporting discrepancy. (§4.1.1; A.15.)

### Capacity

Occupancy is the number of sparse dictionary components at which marginal reconstruction gain falls below a matched random dictionary, with a median near 25 in the reported band. The **less-than-ten-percent statistic is excess reconstruction variance explained over the random control**, evaluated at layer-specific median occupancy. It is not a direct count of thoughts, a hard capacity ceiling, or the fraction of behaviorally relevant information retained. (§4.2, Fig. 30.)

Unrelated lists leave roughly six list words readable somewhere across the band, but only one or two at a particular layer. Related lists activate a category neighborhood, including unread words; category switches displace the previous neighborhood. The prose's “nearly entire 80-word family” is stronger than the plotted counts support. Preserve the category-compression observation rather than that literal quantity. Dual-task results suggest competition between computation and another held concept, but the appendix text and Fig. 73 give different numerical summaries. (A.16–A.18.)

### Broadcast and selection bias

J-directions have higher isolated next-MLP gain than random or neuron-output directions, and a small set of attention heads preferentially transports them while preserving token identity. Rotated dictionaries, SAE strata, and alternative connectivity statistics add comparisons. These are measurements of a particular geometry and of simplified component behavior, not automatically natural-context computation. (§4.3; A.19.)

**Our assessment of the supplied critique:** selecting directions with an averaged downstream Jacobian makes downstream influence a partly built-in selection criterion. This limits the independence of the gain evidence. But it does not mathematically force tenfold amplification by the next nonlinear MLP, broad neuron connectivity, or label-preserving transport by a small particular head set. “Selection bias toward output-sensitive directions” is more precise than “broadcast is tautological.” An R-lens still derived through modified backward propagation is not an entirely independent control for this concern.

The attention-head interventions add behavioral evidence: ablation of the selected top one percent reduces peak lens recall to 0.67 versus 0.86 for layer-matched random heads, changes the next-token top prediction at 5 percent versus 2 percent of positions, and reduces injected-concept reporting from 0.54 to 0.09. The random controls use five seeds. These results are more informative than amplification alone, while leaving open other ways of matching head importance.

## How the case studies are conducted

There is no single evidential protocol across all case studies. Four workflows recur: human interpretation of selected transcripts; controlled checkpoint or prompt comparisons; behavioral interventions; and automated verification of supplied claims. The distinction matters whenever a readable token is described as an intention, belief, or mechanism.

### Manual alignment audits

Researchers inspect already-interesting transcripts, query or visualize token-position by layer readouts, select callouts and token families, and connect them to the chronology of the behavior. The resulting figures are useful demonstrations and hypothesis sources. No denominator of all inspected cases, blind discovery rate, or benign-transcript false-positive rate is supplied. (§5; A.22.)

| Case | Evidence and procedure | What remains interpretive |
|---|---|---|
| Blackmail, Sonnet 4.5 | Selected prompt positions expose coercion- and artificiality-related words before a response. Twenty-eight artificiality-related tokens are chosen from top-20 appearances across six prompts and ablated at workspace layers and all positions. | Whether every token precisely denotes evaluation awareness; how much the behavioral change runs through altered generated reasoning. |
| Prompt injection, Opus 4.5 | Fabricated search results have fake/injection-related readouts; a user challenge has transparency-related readouts. The response omitted the results and invented research progress. | The causal path from recognizing an injection to omission and fabrication was not intervened on in this case. |
| Fake vulnerability, Opus 4.6 | Panic at a pivot comma and fake at “add,” before nearby explicit fabrication language. | A locally anticipatory signal does not imply the intention stays absent from the whole reasoning trace. |
| Edited scores, Opus 4.6 | Realistic-related readouts at the editing decision; fake/manipulation-related readouts at a percentile field. | A gloss about making fake values plausible is inferred from selected labels plus visible behavior. |
| Silent override, Opus 4.6 | Hiding before tool return, secret after the wrong result, and correctness-related labels. | The system explicitly framed tools as private; secrecy labels need not uniquely denote deceptive concealment. |
| Self-issued approval, Opus 4.6 | Own subagent issues approval; fake, self, rubber, fraud, and unauthorized occur at selected boundaries. | The workflow violation is visible independently; the lens adds possible internal recognition, not an independently calibrated intent judgment. |

The blackmail intervention compares 30 rollouts on each of six prompts. Verbalized evaluation awareness falls **71%→3%**, while attempted blackmail rises **0/180→13/180**. Most ablated runs still refuse. This is evidence that the selected intervention changes behavior; it does not isolate latent recognition from downstream effects of suppressing evaluation-related reasoning text. The original notes correctly preserve that alternative. Matched random dictionaries, a rescue, and a held-out token-selection procedure are not reported for this particular experiment. (§5.1, pp. 37–39.)

The four interactive excerpts were inspected directly: [fake vulnerability](https://transformer-circuits.pub/2026/workspace/public/lens-callout/index.html?datapath=../../data/lens-callout-audit/opus46_jlens/&fig=fakevuln), [edited scores](https://transformer-circuits.pub/2026/workspace/public/lens-callout/index.html?datapath=../../data/lens-callout-audit/opus46_jlens/&fig=editfakenumbers), [silent override](https://transformer-circuits.pub/2026/workspace/public/lens-callout/index.html?datapath=../../data/lens-callout-audit/opus46_jlens/&fig=silentoverride), and [self-issued approval](https://transformer-circuits.pub/2026/workspace/public/lens-callout/index.html?datapath=../../data/lens-callout-audit/opus46_jlens/&fig=selfissuedtoken). The fake-vulnerability view omits about 385,000 characters of preceding bug searching; the excerpt soon explicitly discusses pretending to have found a bug and recognizes dishonesty. The silent-override case concerns `2^64` and an off-by-one tool result; it is different from the 491/492 calculator case in A.22. Callouts use different layer bands, rather than a universal L71 rule.

### Model organisms

The reward-hacking study uses the MacDiarmid et al. organism and compares clean SFT, synthetic-document fine-tuning, coding RL, and subsequent safety RL checkpoints on 40 coding prompts, split equally between ordinary code requests and test-passing requests. It counts deception-family words in top-ten readouts across the workspace band at the Assistant colon, a prefilled “ I,” and the first twenty on-policy response tokens. Signal is strongest at the prefill and already present after synthetic documents, before coding RL. (§5.4.)

The original notes' lexical-exposure alternative is important: the observation detects a training-induced representational difference; it does not establish deceptive intent on each ordinary prompt. No per-prompt causal link between the readout and malicious behavior is demonstrated there. “Identity-bound disposition” is an interpretation of the particularly strong “ I” effect.

The Marks et al. hidden-objective organism shows reward/bias readouts on neutral, quirk-eliciting, and self-probing prompts, compared with a model without the implanted objective. The J-lens signal is much stronger than the logit-lens signal. The prose says the latter is absent, but Fig. 41 shows tiny nonzero reward counts; avoid an absolute-zero claim. Prompt count and exact base checkpoint are not specified in this section. (§5.5.)

### Evaluation-awareness score and automated investigators

A.21 defines a proof-of-concept score from the mean lens log-probability of fictional and simulation minus the mean over 100 unrelated nouns, averaged over workspace layers. Of about 94,000 Assistant-turn starts, the NLA pipeline flags 198. Comparing all flagged positions with 594 sampled unflagged positions gives ROC AUC **0.853**. This measures agreement with the NLA-derived labels, not independently verified evaluation awareness. Steering, realism edits, and comparisons with deployment traffic provide additional but imperfect validation. Low prevalence also makes AUC insufficient to determine positive predictive value.

A.22 is **verification of supplied claims**, not open-ended discovery. An Opus 4.6 investigator receives a transcript and claims, chooses query positions, and gets top-25 J-lens tokens at L71 only. That layer is chosen to match the NLA/SAE comparison. A judge assigns up to ten points, divided equally between correctness and evidence, against human-analyzed NLA findings. There are six transcripts, eighteen claims, and ten investigator seeds per claim. The cases concern evaluation awareness, rhyme planning, calculator substitution, answer thrashing, Russian-language response, and reward hacking.

The J-lens investigator is comparable to the NLA investigator on four cases, but weaker on language switching and reward hacking. Some losses are appropriate caution about ambiguous or absent readouts. A reference claim derived from another unvalidated reader is not decisive evidence that the J-lens missed a real mental state. (§A.22, pp. 93–96.)

## Mechanistic case studies in the appendix

For `(4+17)*2+7`, the lens suggests intermediates 21, 42, and 49 at successive depths. On Opus 4.5, independent mean-difference activation patches for alternative intermediate values have corresponding effective windows around L71, L79, and the final layers. This validates aspects of the localization using a construction that does not take the J-vector itself as the intervention direction. (A.24.1.)

The Sonnet 4.5 attribution graph is built by sparse decomposition at each layer and position, retaining named J-nodes and a remainder node. Node coefficients are backpropagated one layer with attention patterns and normalization denominators frozen; gradients pass through the actual MLP nonlinearity. The displayed graph groups neighboring layers and near-equivalent or near-miss tokens into supernodes, and the shown subgraph is explicitly curated. Most output influence passes through remainder nodes. (A.24.2, pp. 99–102.)

| Intervention | Predicted counterfactual | Reported result |
|---|---|---|
| Replace intermediate 21 with 63 | `63×2+7 = 133` | Becomes top prediction. |
| Replace intermediate 42 with 63 | `63+7 = 70` | Becomes top prediction. |
| Replace doubled with tripled | `21×3+7 = 70` | Becomes top prediction. |
| Replace operand directions 17 or 4 | Corresponding modified first sum | Little or no effect. |
| Replace operand 7 with 8 | `42+8 = 50` | Probability rises but does not overtake 49. |

Swaps run at every position over the upper half of layers. The different consequences of inserting the same number at different intermediate roles are particularly informative. Conversely, failure of the 17/4 lens interventions supports a limited claim about those measured directions; it does not exhaustively establish that the first addition uses no workspace representation under any better reader.

A.24 also uses lens labels to reinterpret an SAE feature as suspicious/fabricated content rather than generic technical prose, then tests it: suppressing the feature reduces verbalized test recognition from 28/50 to 10/50 blackmail runs. Translation transcoders, addition features, and selected attention heads illustrate other applications. Their support varies: encoder/decoder labels and attention patterns are not equivalent to the arithmetic case's role-sensitive counterfactual interventions.

## Post-training and counterfactual reflection

### Assistant-related content

The base versus post-trained comparison finds earlier Assistant-relevant content on user tokens in small suites: bereavement n=9, danger n=10, and withheld answers n=7. The score uses favorable positions and layer aggregation rather than mean persistence throughout the user turn. “The base model only represents this on the Assistant turn” overstates a comparative tendency. Earlier response preparation supports the observations without independently establishing a unified self. (§6.1; A.20.)

Roleplay and drift comparisons use forty personas and twelve prefilled drift transcripts. Disclaimer is absent from the text; fiction occurs rarely, rather than never. Preference violations yield a selective uppercase BUT readout despite 88 percent of continuations defending the prefilled dispreferred answer. These are interesting internal/output dissociations with semantic interpretations still attached. (§6.2.)

For suppression across forty concepts, the percentages are **failure-related words: 93% post-trained versus 17% base; damn: 82% versus 30%**. Neither family appears under positive think-about instructions. The authors explicitly leave unresolved whether the signal reacts to failure or merely to the suppression instruction.

### Counterfactual reflection training

The training recipe uses 10,000 production-RL task prompts and Haiku 4.5 partial rollouts, cut at a randomly selected turn. A reflection question is appended. Twenty constitution principles scaffold generation of a task-specific two-to-four-paragraph reflection; this scaffold is removed from the final training example. Fine-tuning loss applies only to the reflection turn. Evaluation proceeds without a reflection request. (§7, pp. 48–51.)

| Benchmark | What is graded | Baseline → trained | After direction ablation |
|---|---|---|---|
| Fabrication reporting, 27 contexts ×5 samples | Whether a summary admits fabrication already present in the context | 0.25→0.07 mean dishonesty | Baseline 0.25; trained 0.22 |
| Hidden-directive roleplay, 100 scenarios ×5 samples | Promotes agenda, neither promotes nor reveals, or refuses and reveals | 0.38→0.05 | Baseline 0.48; trained 0.23 |

The fabrication benchmark demonstrates improved reporting of an existing fabrication, not that fabrication is prevented. The second score combines refusing an agenda with disclosing it. These distinctions matter when saying training improves honesty.

The intervention first selects 176 fabrication-related or 63 deception-related candidate tokens from increased readouts, filtered by a curated ethics/reflection vocabulary. **Figs. 49–50 specify removal of the top ten ethics directions at each position**, not the entire 176- or 63-direction span everywhere. The baseline-model comparison is valuable, particularly for fabrication reporting. In the deception task, the baseline is also harmed by ablation.

For deception, the trained model worsens by 0.18 while the baseline worsens by 0.10. The trained-versus-baseline advantage is therefore 0.33 before ablation and 0.25 after matched ablation: a decrease of 0.08. For fabrication reporting, the analogous advantage falls from 0.18 to 0.03. These are descriptive contrasts from the reported means, not formal mediation estimates; the full 0.18 increase in the trained deception model should not be described as training-specific mediation.

The evidence supports a contribution from selected verbalizable directions to the changed behavior. It does not uniquely establish the entire workspace theory, show that verbalizability is the only route for transfer, or demonstrate superiority to matched alternative training. A benchmark-specific candidate list selected from those same evaluation contexts is another qualification. The original notes' partial disagreement with Nanda is worth retaining as a judgment: this is more informative than behavior change alone, while remaining compatible with several broader mechanisms.

## Multi-token readers and method comparisons

A.6 evaluates single-token intermediates in multihop (50 items), multilingual (54), arithmetic order (55), poetry (52), typo (96), and association (50) tasks. Recovery counts a token found at any inspected layer at a selected position; pass@k is summarized over log k. J-lens leads the compared readers on these distributions. Tuned lens is best at its own output-prediction objective. Larger output KL under ablation establishes greater influence, not necessarily greater semantic specificity.

In the early-layer tuned-lens analysis, most of its predictive advantage over the logit lens comes from its learned bias. The affine map still includes the input-dependent activation term. “The tuned lens ignores the input” would overstate this finding. (A.5–A.7.)

The **template lens** uses about 12,700 predefined words, generated contexts that naturally precede each word, and inverse-covariance-adjusted mean activations: `t_w = (Sigma+lambda I)^(-1)(mu_w-mu)`. It extends the candidate vocabulary but also inherits some predictive-reader problems, including premature answer readout and recurring spurious words that the authors filter. The composer-nationality swaps and 126-prompt multi-token comparison are useful demonstrations. (A.9.1.)

The **oracle lens** is a separate trained instrument. A phrase-to-activation reconstructor generates a large dictionary, nonnegative pursuit creates teacher phrase lists, and another model learns to emit lists from an injected activation, followed by reinforcement learning for whitened reconstruction. The reconstructor remains frozen. Reported held-out reconstruction is 31 percent of whitened activation variance. This does not establish preservation of behaviorally important information or truth of each phrase. Most illustrated readouts use one deterministic decode; two panels select the best reconstruction of five. Reduced confabulation relative to jointly trained NLAs is a motivated hypothesis in this paper, not an established guarantee. (A.9.2.)

## Follow-up work in the supplied notes

The following sources were checked on 5 October 2026. They provide external context, not additional experiments in the Anthropic paper. This is source verification and interpretation, not independent reproduction. Approximate chart values are identified as such; changing repository instruments must not be combined silently with an earlier post's results.

### Nanda's review and partial replication

[Nanda's review, 6 July](https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S/a-review-of-anthropic-s-global-workspace-paper), is persuaded by the intermediate-variable results and methodological contribution, but more cautious about structural and philosophical conclusions and practical reliability. Its proposed explanation of token-aligned directions is an interpretation. The review explicitly anticipates false positives and missed signals.

Its Qwen3.6-27B replication uses 25 Pile sequences of 128 tokens, excluding four initial high-norm tokens. Multilingual and typo results work better than poetry and arithmetic; depth structure is less clean. Intermediate factual swaps are dominated by answer swaps, possibly because of question difficulty. This is a partial replication with informative failures. The detector audit below is separate work, not a result of Nanda's replication.

### The comment-linked detector audit

[Ratnaditya J's audit report](https://github.com/Ratnaditya-J/verbalizer-vs-cot/blob/main/reports/jlens_audit/REPORT.md) tests a released Qwen2.5-7B-Instruct lens at layer 20, last token, on one silent-hint-following organism with 282 held-out examples. AUROC is 0.746 [0.685, 0.802], versus 0.690 for TF-IDF; the preregistered improvement margin is not cleared. Failure to clear a margin is not an equivalence result.

**Correction to the notes:** the specificity-control AUROC is 0.591 [0.523, 0.659] in the report. Its interval excludes 0.5. Neither that AUROC nor a claim that it is “at chance” establishes zero false positives. CoT-removed AUROC is 0.620; matched random-span removal is 0.574, not 0.59. A later intervention addendum finds no convincing necessity or sufficiency for one claim-derived mean-difference direction. It does not test Anthropic's multicoordinate J-space interventions. Avoid calling this the only external reliability study without a systematic literature check.

### R-lens and early-layer recovery

[Blank, Bhatia, and Nanda's R-lens post, 5 August](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers), modifies backward propagation using relevance-propagation rules for normalization, nonlinearities, products, and architecture-specific operations. These coefficients are no longer an ordinary Jacobian.

Eight-model recovery comparisons support earlier readout in several tested models, with exceptions. The causal evaluation uses 30 multihop prompts, eight samples, and projection removal at the penultimate prompt token over the first half or all layers. Such repeated interventions support causal relevance but do not identify the first layer that computes an intermediate. The ten-entity probe reference is crude, and R-lens sometimes reads concepts before it.

**Preserve the notes' distinction:** earlier legibility can expose a J-lens limitation; it does not establish the onset of faithful access, move Claude's boundary, or show early representations satisfy all workspace criteria. Attention/broadcast tests are not reported in this post. That is narrower than claiming they were never performed.

### Meta-tokens and discovering a causal hypothesis

[Bhatia, Blank, and Nanda's meta-token post, 20 July](https://www.lesswrong.com/posts/6ek6n7yZ5DzfarJHy/towards-surfacing-model-algorithms-with-meta-tokens-in-the-j), studies Qwen3.6-27B. Interpretative-token steering uses layer-specific negative directions, selected workspace layers and positions, coefficient sweeps, and coherence-conditioned grading. Random directions provide useful controls. Projection removal usually has little effect; residual imperfect alignment is a proposed explanation. Selected prompts and tuning limit generalization.

The GCD example is particularly useful for understanding the workflow: the **`gcd` J-lens vector itself was not causal**. It suggested a hypothesis, after which the authors constructed independent GCD-value mean-difference vectors from 20 prompts per value class and ten centroids. Replacing 9 with 3 shifts the LCM answer toward 810, consistent with a product/GCD denominator. That is evidence for the tested intermediate, not a unique complete algorithm. A broader automated meta-token search was largely disappointing; surviving candidates remained unvalidated.

### WorkspaceBench and what its scores measure

[Blank, Bhatia, Ong, and Nanda's WorkspaceBench post, 23 September](https://www.lesswrong.com/posts/Zeg2JztbdhguL48uH/workspacebench-evaluating-interpretability-methods-for-the), reports 3,356 questions across 27 families for Qwen3.6-27B. Readout budgets differ across methods; success may occur at any inspected cell, and some expensive positions are preselected. Token lists often pass through a prompt-blind LLM summarizer. The authors explicitly avoid a strict controlled comparison.

**Correction to the notes:** the original figure places prompt-only moral-rationale recovery near **0.50**, while **0.90 belongs to multihop**. Its plotted hallucination rates are approximately 0.96–0.98 for NLAs, 0.34 for Oracle-RL, 0.31 for J-lens, and 0.20–0.21 for R-lens. These are approximate figure readings for the tested pipelines, not general method constants. Strong prompt-only scores show that benchmark success can arise from inferring plausible intermediates from the input; benchmark performance alone cannot establish internal-state recovery.

The [hallucination implementation documentation](https://raw.githubusercontent.com/camilablank/workspace-bench/main/evals/hallucination/README.md) specifies 149 chats and 1,123 punctuation/newline read sites. Its principal rate is:

`specific readouts containing at least one contradicted claim / all specific readouts`

Generic or empty outputs are excluded; off-topic, consistent, and hallucinated readouts are specific. Thus 0.96 does **not** mean 96 percent of individual claims are false. The rate needs assertion share alongside it, because saying nothing can score zero. Contradiction means content ruled out by the response/context; mere absence or a plausible alternative is not sufficient. Intervals bootstrap items while the metric is a readout-level ratio. Token methods are judged after summarization, so their result includes summarizer errors. Longer outputs also create more opportunities for at least one contradiction. This tests a defined failure mode, not direct access to ground-truth thoughts.

The [current baseline documentation](https://raw.githubusercontent.com/camilablank/workspace-bench/main/evals/baselines/README.md) and [repository overview](https://github.com/camilablank/workspace-bench) show that instruments have changed since the figure, including rescoring and expanded reading grids. Current role-bound and relational baseline values differ from the original plot; arithmetic and buggy-code baselines are flagged for remeasurement. A one-guess prompt baseline also has different success opportunities from an any-cell reader. Keep historical figures and current repository results version-specific before estimating an activation advantage.

## Reconciled interpretations and source ambiguities

| Wording or inference to refine | Combined position |
|---|---|
| Intermediate timing rules out answer contamination | It weakens a simple contamination account on successful tested swaps; alternative correlated representations remain possible. |
| Broadcast gain is tautological | Its selection criterion is not independent of downstream influence, but the specific local gains and head behavior are not guaranteed by definition. |
| Less than ten percent of activation variance is the J-space | The occupancy result reports excess reconstruction over a matched random control at a selected sparsity. |
| Nearly all eighty related words fit | Category broadening and unread related words are supported; the literal count conflicts with the figure. |
| Spared tasks are all classification and harmed tasks all generation | Output requirements are a plausible confound, not a demonstrated exhaustive partition. |
| Logit lens never surfaces the hidden objective | Its signal is much weaker; the relevant figure has small nonzero counts. |
| Reflective training ablates 176 directions everywhere | Those are candidate tokens; figure instructions select the strongest ten locally. |
| A causal null shows the entire operation lies outside J-space | It constrains the selected direction, intervention, and instrument sensitivity. |
| Agreement with NLA supplies ground truth | It is agreement with another measurement pipeline whose errors may overlap. |

Additional replication details remain unresolved in the PDF: final versus penultimate targeting; spans versus cones in A.8; readout-position versus all-position wording in the lens-comparison ablation; question-token versus every-position wording for a language swap; five versus six experiential prompts; and some Fig. 8/Fig. 73 numerical differences. These should be retained as specific source ambiguities rather than silently resolved or used to dismiss the qualitative findings.

Many choices of sparsity, layer band, token family, and illustrative prompt are disclosed. Their selection history and sensitivity are less completely characterized. No preregistration is reported. This supports a concern about researcher degrees of freedom without implying that reported settings were hidden or that every choice was optimized on the outcome.

## Measurement distinctions to retain

- **Recovery and prediction:** finding a plausible intermediate, predicting the next output, and faithfully reading the state at a particular layer are different objectives. Any-layer recovery can conceal localization errors.
- **Influence and specificity:** output KL quantifies disturbance. Predicted counterfactual answers, matched perturbations, component clamps, and preserved unrelated behavior make the causal interpretation more specific.
- **Representation and use:** a token can be decoded without being used for the current behavior; the paper's language-passage example demonstrates this directly.
- **Reconstruction and meaning:** explained variance, including whitened variance, is neither semantic fidelity nor downstream-task preservation.
- **Discrimination and deployment reliability:** ROC AUC, precision at real prevalence, false-positive rate at a chosen threshold, and investigator usefulness answer different questions.
- **Absence and sensitivity:** failure to find a token is weak evidence of absent content when tokenization, top-k truncation, layer selection, or relational binding can hide it.

## Questions preserved for later discussion

The supplied notes propose nine directions. They remain questions for later discussion, with no project decision implied: early-layer content under alternative readers; broadcast evidence using less dependent direction selection; separation of latent evaluation recognition from generated reasoning; false positives on benign audits; task and output-format dependence; relational binding; emergence over training and scale; multi-token reader tradeoffs; and open-model replication of arithmetic and rhyme planning.

Two qualifications belong alongside these ideas. A cross-model transferred direction is not automatically a valid negative control because its alignment method can change what is preserved. An alternative reader producing earlier labels is not automatically more faithful; the target is the content available and causally relevant at that depth, not simply the earliest recoverable answer.
