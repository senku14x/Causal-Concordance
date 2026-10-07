# 012 statistical and label review — 2026-10-06

Independent read-only review of `docs/design_specs/012_reader_concordance.md` v1 and the scientific rationale in Claude report 32. No project files, experiments or dependencies were changed. Sources also include the frozen `phrase-lens-src` causal ST JSON bank. This is a design review, not validation of unrun results.

## Judgment

Stage A is a useful cheap feasibility check. Before scaling B/C/D, the crucial repair is to define an estimand the interventions can actually identify. The proposed labels characterize intervention outcomes, not concept truth or absence. Readers on matched clean, edited and real-target states can test responsiveness to interventions with demonstrated answer influence. They cannot yet support precision/recall against naturally used concepts or an exhaustive interpretability ground truth.

## Highest-priority changes

1. **Separate intervention success from semantic ground truth.** Lines 142–145 define success by a margin increase relative to random and a greedy flip. This is an operational answer-influence event for an item × arm × band. It is not a cell-level demonstration that the edited bridge mediated the effect: every prompt position across a band is edited, while one downstream cell is read. Failure is not a negative label for bridge presence or use. The wording at lines 227 ("unused through that direction") still overstates a null. Say "no qualifying answer effect through this intervention." The line "labels from interventions decide what is true" in §1 also needs narrowing.

   Choose between two distinct tasks: (a) describing source/target content after interventions, with target-prompt calibration; (b) predicting the predeclared intervention-success label from reader measurements. Both are informative. A reader naming retained source content after a successful intervention can be correct on (a) and poor on (b). Current Q3/Q4 combine these tasks. Neither licenses scoring all unmentioned or untested concepts as absent.

2. **Capture and read failures, partial effects and controls.** Lines 151–154 say swapped states for each arm that works; lines 165 and 180 require working-versus-failed/random comparisons. Resolve this before capture. A fixed matched cohort including all outcomes is best when affordable; a predeclared stratified subset is adequate for exploration if sampling weights and denominators are retained. Reading only successful swaps gives no estimate of specificity. Selecting different successful items per arm also confounds reader/arm comparisons with item difficulty.

3. **Use the same intervention states for head-to-head reader comparisons.** The listed L24–35 Oracle and L24–39 NLA routes are different interventions. If both can read after L24–35, that is a useful common comparison. Different read layers remain a reader-contract limitation, but different edit bands need not be added as another confound. A later band with no downstream NLA site cannot adjudicate Oracle-versus-NLA fidelity.

4. **Apply power gates to the interpretation they protect.** An answer-row control failure blocks interpreting a bridge-row null as absence. It does not invalidate a demonstrated bridge-row positive, and it should not veto a separately powered class-mean arm. Lines 147 and 196 currently risk discarding useful positives or stopping the whole study when only one intervention family lacks a positive control. The full class-mean arm is the direct power check for its J/remainder decomposition. Full class means are reader-independent in construction, but are not intrinsically non-J or an independent semantic ground truth.

## Labels, agreement and ensembles

The 1-nat-plus-flip rule is a usable exploratory threshold, not an inferential confidence statement for each item. Two random conditions with one draw each do not define the full null distribution; noisy control realizations near the threshold will change labels. Save continuous margin changes, control changes, answer probabilities and generated text, retaining the discrete categories as a convenience. Inspect whether a positive margin is driven by target-answer increase or source-answer collapse. The negative controls threaten generic disturbance; they do not fully exclude lexical or answer-level steering.

Cohen's kappa needs a fixed outcome space and denominator. For the first pass, each reader can be coded as naming source only / target only / both / neither, with other content separately annotated. Token readers require a fixed naming threshold or top-k budget; free-text readers require fixed generation and mention-matching rules. Pooling arbitrary unmentioned concepts as negatives can manufacture apparent agreement. Report marginal naming rates and positive agreement alongside kappa; correlated readers and skewed prevalence limit interpretation.

An ensemble can be evaluated on the same specific prediction or naming endpoint, but not on an unspecified universe of true concepts. Union recall improves mechanically when more content is emitted; compare at an explicit false-positive or output-budget cost. Random slices and prompt-only baselines are controls, not ground-truth absent-concept labels. J_NP and J_CB count as two votes under naive majority despite closely shared methodology; report the vote definition and a family-balanced sensitivity analysis if conclusions depend on majority. Do not use agreement itself to validate labels.

## Cohort and uncertainty

Read-only parsing of the frozen ST JSON gives 96 items and 53 distinct targets. Target Japan occurs 29 times. Applying the spec's nine uncovered concepts leaves 87 potentially covered items before any clean gate: Japan accounts for 28 of 87, and 37 of the remaining 47 distinct targets occur only once. These counts are based on the declared unresolved concept list, not an independent tokenizer validation.

Thus target-specific tables are diagnostic rather than precise estimators for most targets. Report pooled and no-Japan results, plus relation/target composition after every gate. Paired contrasts on the same items are preferable for reader differences. Wilson intervals describe binomial sampling under independence; related prompts and repeated targets make them optimistic for claims beyond the fixed bank. Relation-cluster bootstrap and a no-Japan sensitivity are a reasonable modest first pass; target clustering is cross-cutting and should be acknowledged rather than silently treating all items as independent. There is no need for a large mixed-model infrastructure at this stage.

The 50-item clean-gate threshold is an operational minimum, not statistical power for every reader × band × piece × target comparison. A 30% intervention success rate leaves about 15 positives among 50 clean items before further splits. Null reader contrasts can remain wide. Start with a small number of primary comparisons and treat the rest descriptively.

The 49 overlapping 007 test items are already spent for this program. New interventions on them are useful exploratory measurements, not fresh confirmatory data. Band selection using Stage A should be stated as selection; downstream reader evidence is exploratory on this chosen population.

G-parity's 90% agreement must use a common cohort and identical prompt rendering. Report the success/failure confusion matrix as well as total agreement: a predominantly negative cohort can hide disagreement on most positive cases. Different clean gates between A and B otherwise create a misleading denominator.

## Covariates and class means

Output rank/D and J loading are useful measurements and alternative predictive baselines. Pre-intervention J loading can stratify success propensity. Post-intervention output scores and loading are effects of the intervention; treating them as ordinary confounder adjustments may remove precisely the propagated signal being studied or induce selection bias. A residualized association beyond one scalar D is not proof of information absent from the full output distribution. Use unadjusted matched changes as the primary descriptive result; call any adjusted model an incremental-prediction analysis and validate on held-out item/target or relation groups as far as sample size permits.

Class-mean directions need genuinely varied source evidence. Leaving out the exact item does not prevent shared cue/template leakage or answer correlation. For the smoke, inspect the drafted prompts and held-out target-prompt references directly. A successful class-mean intervention is additional evidence that influence is not restricted to a J-row handle; it is not automatically evidence for the bridge rather than an associated answer or lexical category.

## Recommended minimal next step

Proceed with the clean-conditioned Stage A feasibility result, preserving all exclusions and generated outputs. Before B/C, agree on one common-band comparison: fixed items, clean / swap / matched random / real-target states, full-vector readers first, continuous answer effects and source/target reader changes. Interpret this as reader responsiveness to answer-influencing edits. Expand pieces, ensembling and multi-token families only after that table demonstrates useful variation and the basic comparison is interpretable.
