# Causal-Concordance: independent onboarding assessment

2026-10-06. Independent scientific auditor for Vishesh. Repository: `/workspace/causal-concordance`. Observed branch `008-trackb-brewcode`, HEAD `deb2e41eff63793a069ac46e9295d4b64577d653`; active Claude changes were present. This is an assessment of the available evidence during a moving working session, not certification of a frozen release. No repository file, experiment output, configuration, dependency, branch, or commit was changed by this audit. No model experiment was launched or resumed. Durable audit documents are confined to `/workspace/codex_reports/`; disposable analysis and paper renders used `/tmp`.

## Assessment

The project has established substantial differences between **reader sensitivity**, **decodability**, and **the effects of particular subject-model interventions**. It has not yet produced the stable, content-specific reader-free causal reference needed to rank readers by whether they recover what the model naturally uses. That is the central unfinished inference, not an incidental qualification.

The strongest replicated reader finding is Oracle L40's dependence on the dense J-derived projector in 007. The strongest local sparse-privilege comparison is 008: under a powered class-mean swap, the sparse J component is much less effective than its remainder at natural coordinates. These are compatible findings about different constructions and different consumers. Neither establishes that the Oracle reads a causally irrelevant representation or that the subject reasons outside J in general.

I am least convinced by the remaining headline **“the Oracle privileges J; the model does not.”** The dense comparison is mixed at the Oracle's on-contract layers, not an established absence of privilege. I find **009's failed output-facing check** the most decisive research-decision result: the proposed code-answer workaround did not remove the confound it was designed to remove. Its failure is large and independently reproducible, without relying on an activation reader.

The handoff's 011 status has changed: the current saved s1 run is complete, 82/82 pairs, and analyzed. Additive AR differences beat random controls but underperform class-mean differences. Both coordinate-swap versions fail their answer-control power gates. The new AR/J-bias diagnostic has a concrete PCA-rank control defect.

## 1. What the Workspace paper actually argues

Source: [local Workspace paper text](/workspace/causal-concordance/docs/papers/text/2607.15495_html.md), especially §§2, 3.1, 3.3, 3.5, A.8, A.9, A.12, A.21, A.24; [PDF](/workspace/causal-concordance/docs/papers/2607.15495.pdf). I directly inspected PDF pages 12, 13 and 19, containing Figs.7–8 and 16, to check what captions and plotted quantities actually show. The substantive notes 04, 04b and 03 were read after the paper; their corrections/disagreements are historical arguments, not independent experimental evidence.

The paper proposes that representations accessible to verbal report constitute an approximate global workspace. The average Jacobian transports intermediate activations toward the output residual space; vocabulary vectors derived through that map provide labels and intervention directions. It supports its broader proposal with reportability, directed modulation, reasoning and flexible-task interventions, broadcast/capacity measurements, and auditing examples. These are different evidential strands, not one demonstrated universal mechanism.

For this project, the relevant strong claim is **privileged causal influence of a sparse J component**, rather than the mere ability to read or steer concepts. In §3.1 a sparse component of planted concept vectors drives reporting more effectively than its remainder under tested controls. Fig.7 includes a report-position comparison; 005's Qwen measure failed an analogous selectivity test. In §3.3/Fig.16, independently obtained intermediate vectors are decomposed with sparse pursuit; the plotted target-answer top-one rates are 61% for the J component, 28% for the remainder, and 6% for the remainder with the relevant J coordinates clamped. These interventions cover every token position over a layer band. The Fig.16 variance label is 16%, while nearby prose describes roughly 10–15%; I do not rely on that small numerical mismatch.

The arithmetic interventions in A.24 are more discriminating than generic answer steering: replacing 42 by 63 favors 70, while replacing 21 by the same 63 favors 133. The downstream operation depends on the intermediate's role. This is useful evidence for an intermediate account in those examples; it is not proof of general completeness. The paper itself describes incomplete attribution graphs dominated by remainder nodes, ambiguous sparse decompositions, and computations outside the named J features.

Crucially, §3.5 explicitly shows readable content that affects some tasks but not others. Therefore “readable here, little effect on this task” does not by itself refute the paper. Our programme tests how reliably readers indicate demonstrated task-specific influence in Qwen, plus whether a particular privilege pattern transfers from Claude. It does not yet adjudicate global workspace theory in general.

Three distinctions matter mathematically:

- **J-lens:** the layer-dependent average-Jacobian readout and its token vectors. The actual readout includes normalization; raw unembedding vectors and vectors with the final normalization gain folded in are different conventions.
- **Sparse J component:** a k-limited nonnegative pursuit/NNLS approximation of a particular vector. k=16 in 005, k=25 in 007 Stage2b/008. Its remainder is not orthogonal to the entire vocabulary dictionary or equivalent to the complement of a single global J subspace.
- **Dense J projector:** 007's top 1,024 eigenvectors of the unit-normalized raw `W_U[t] J_l` collection. It is a fixed linear subspace, not the paper's union of sparse cones. The full token dictionary may span all 5,120 residual dimensions. Appendix A.8 itself mixes cone language with span/projection equations; that ambiguity should not be imported as a theorem.

The template lens uses an inverse-covariance discriminant direction, whereas whitened displacement coordinates use the inverse square root. These are different transformations. A.9.2's Oracle reconstructs the activation **preceding** a phrase, trains an inverter, and refines it with a frozen reconstructor. Its reward measures reconstruction, not claim truth or causal use. WorkspaceBench's `olens_s3d_rl600` follows that lineage but has its own training recipe and artifacts; it is not a Karvonen activation oracle. The NLA is a separate model/checkpoint and must not inherit Oracle conclusions.

A.21's agreement with NLA labels is reader agreement. The auditing-agent examples also use partly NLA-derived reference judgments. Neither is reader-independent ground truth. Adjacent primary sources reinforce the need for input-only and text-inversion controls, but do not establish that this project's Oracle or NLA is always re-solving. Dingeto's low detected claim-grounding rate is sensitivity-limited and not a measured fraction of false claims in our checkpoint.

## 2. Why the experiments followed one another

| Experiment | Question or failure it addressed | What survives scientifically |
|---|---|---|
| 001 | Find aligned cells where natural counterfactual differences affect the answer; independently measure decodability | `:` L42–44 whole-cell S approximately 0.48–0.50; the newline after `</think>` much lower. z decodes at both. Whole patches change start, z, answer and other information together. |
| 002 | Turn mentions into role-aware claims; check hidden signal below top-k displays | Verbalizers rarely state the correct intermediate role in this small pilot; token-score contrasts track z despite absent top-ten palette words. Claim and likelihood tasks differ. |
| 003 | Obtain reader-free one-dimensional content labels | Small mostly positive z-associated effects; reliability failed (kappa 0.37). Necessity controls were not size-matched. No valid causal reader ranking followed. |
| 004 | Test whether a broader fitted span rescues the label assay | Larger effects, but PCA performs as well or better at comparable rank/dose; many changes are not donor-answer-directed. Fitted span effects are not semantic isolation. |
| 005 | Test planted concepts and sparse J/report privilege | The supposed report metric is position-generic lexical priming; no demonstrated matched-norm J advantage on its primary measure. Oracle naming exceeds NLA in tested conditions; NLA likelihood positive control fails. |
| 006/006b | Check verbalizer likelihood tracking, then the subject's own output | Reader likelihoods track z; Oracle's original raw identification gate fails because of a strong colour prior. Subject output also tracks z; planned low-output-signal stratum is too small. |
| 007 | Discriminate dense-J sensitivity from content share and mean removal, with test replication | Oracle L40 dense-J dependence replicated; NLA's colon pattern differs post hoc. Dense model-side result is mixed at L40/L44, baseline-equivalent at L42. Sparse Stage2b lever weak. |
| 008 | Strengthen the sparse model-side intervention | Multi-site/multi-layer class-mean edits provide leverage. Sparse J gives less natural-coordinate effect than remainder. This repairs Stage2b power, not all causal specificity or the dense/sparse mismatch. |
| 009 | Remove explicit colour answers and contrast task dependence | Depth1 gates; depth2 fails this no-CoT format. The output-facing colour signal persists very strongly, so reader phase is held. |
| 010 | Quantify agreement from saved scores | Oracle–J score association is strong; NLA–Oracle agreement is weaker and prefix-sensitive. No causal ground truth or powered reader–leverage null. |
| 011 | Test directions obtained from the Oracle's reconstructor | Local-loader additive differences show limited directional influence beyond random; coordinate swaps fail power controls. Loader and J-bias interpretation remain provisional. |

Sources: canonical `docs/reports/001_...` through `009_...`, design 011, report29 for 010, plus [SPEC errata](/workspace/causal-concordance/docs/context/SPEC_v2_errata_01.md). The errata qualify old labels rather than merely updating terminology. In particular, original whole-cell damage flags should not disappear: the L42 follow-up indicates a mixture of trajectories, not a pure z transfer. A negligible full-cell patch can conceal cancellation or redundancy. It is not a proof that nothing at that cell matters.

## 3. The central J comparison, with the strongest defensible wording

### Reader side is strong, but instrument-specific

Verified from `results/007_jsubspace/analysis/{stage1a,brew_test}.json`, Oracle L40:

| Split | Whole D_z | Dense J D_z | Remainder D_z |
|---|---:|---:|---:|
| Development, 48 tables | 3.2222897438 | 5.7638180389 | 0.2990016070 |
| Brew test, 49 tables | 2.8701533411 | 5.0301057972 | 0.2935346590 |

P1, P2, P3 and P5 replicate under the registered categories at L40; only P5 does so at L44 in the primary analysis. P3 and P5 are the cleaner discrimination because several PCA/random pieces fail sensitivity checks. The three model predictions in the design are not exhaustive alternatives.

This establishes a dependence of this Oracle likelihood measurement on the projector under the tested input transformations. Oracle receives a norm-normalized piece; D is not additive over pieces. J-only D exceeding whole D can reflect removal of interfering context or altered input geometry. It does not mean the piece contains more information than the whole, and is not itself a statement about spontaneous truthful prose. Randomly selected saved whole readouts for table0312/0132 and other pieces retain task structure amid invented colour chains. The likelihood/prose distinction remains real.

The NLA at colon retains material signal outside J and does not reproduce the Oracle pattern. At `</think>`, however, NLA signal also collapses after J removal while J alone remains insufficient. “Reader-specific J dependence” must therefore be scoped to the colon comparison. Different native read layers remain another qualification; a universal NLA-versus-Oracle architectural conclusion is not available.

### The dense model-side null is overstated

Root verified `stage2.json -> layers -> primary` against design §5.3. Recovery is R=S/S_full, not S. The registered “tracks delta-share” verdict requires both 90% intervals within ±0.10. At L40/L44:

| Layer | Paired J minus delta-matched random recovery | 95% interval | Saved verdict |
|---|---:|---:|---|
| 40 | 0.0665644920 | [0.0407128870, 0.1102894276] | mixed |
| 44 | 0.0907645635 | [0.0242922201, 0.1626416618] | mixed |

J has a positive advantage over random under these exploratory comparisons. L40's 90% upper limit is 0.1003661583, narrowly outside equivalence; L44's is 0.1570308230. The secondary content-share regression residuals are positive too. J does not demonstrate the prespecified advantage over both random and PCA, but that is not absence of advantage. Equivalence is obtained at L42, which is not an on-contract Oracle layer.

The full-cell lever is adequate (S_full 0.2768432393/0.4285886932/0.4907383207 at L40/42/44). Thus this is not the same problem as weak Stage2b power. Also, reading `unit(P_J h)` in an auxiliary model differs from patching `h_m + P_J(h_p-h_m)` in the subject even when P_J is identical.

**Recommended wording:** “Oracle colour-likelihood tracking is strongly concentrated in the dense J projector. Subject-model component swaps do not demonstrate the registered strong privilege over both matched baselines; L40/L44 are mixed, and L42 is equivalent within the specified margin.” A qualitative mismatch in sensitivity is supported. A clean same-layer causal dissociation between Oracle dependence and zero model privilege is not.

### What 008 genuinely adds

82 gated development pairs, 33 tables; three emit cells, every block output L40–48; SEC; class means exclude the evaluated table. Raw full-direction median S is 0.2075949250. Natural sparse J gives 0.0227290579/0.0335668931 for NP/CB, versus remainders 0.2104256495/0.2047735135. Paired contrasts are −0.1691125528 [−0.2403514853,−0.0930547630] and −0.1446534558 [−0.2135260837,−0.0769388942]. This is substantial evidence against natural-scale privilege for these sparse pieces under this intervention.

Whitening narrows the contrast; CB's true-whitened difference includes zero. A colloquial “tie” is not a formal equivalence result. Arm A's answer-atom control is approximately 0.01, failing the 0.15 bar; its weak z result is not an informative absence result. Full raw and true-whitened direction gates pass by their point estimates, which establishes an operational lever, not formal statistical power for every contrast.

Additional qualifications found by narrow code inspection and saved-output checks:

- The complementary clamp preserves coefficients in the **joint dictionary** `[n_z,n_zprime,J-support]`. It does not necessarily preserve isolated J projections or lens scores; cross-support nonorthogonality exists. Passing RA=I verifies the chosen coordinates. Whether the paper intended the identical convention remains unresolved. This is an interpretation ambiguity, not a diagnosed coding bug.
- “Matched norm” means matching target displacements computed from the clean state. Later layers clamp against already-perturbed states. Actual summed raw write norms for the J arms are approximately 1.65/1.83 times the full reference, whereas the residual arms are near 1. This precludes literal equal-realized-dose claims; nonmonotonic dose response remains possible.
- The reported 13/82 flips are argmax changes among accepted colours, not verified full-vocabulary top-one flips. Median partner-answer log probability rises approximately 1.30 nats, so simple uniform degradation does not explain the full-direction result. Off-pair mass increases substantially, but “off-pair” includes valid other colours and is not a general damage measurement.
- A multi-site clamp supplies causal influence of a constructed edit policy. It does not independently validate z-specific natural use, exclude finite-sample start/answer contamination in class means, or show that every route through J was blocked.

Detailed evidence and exact source paths: [007–008 audit](/workspace/codex_reports/02_007_008_audit.md).

## 4. Output accessibility and agreement

009's failure is particularly clear. Raw outputs independently reproduce final-code D_z=11.8943137957 [11.7225175672,12.0659840996], 658 collision-free pairs from 240 tables. z is the top colour on 237/241 member renders, median full-vocabulary rank11. Total colour mass has median0.0005146791. These populations differ slightly; they should not be merged into a single n.

D cancels a common multiplicative colour-probability factor per render. Small total mass therefore does not remove discrimination; “scale-free” does not imply invariance to temperature. Forced post-answer continuations are conditional on teacher forcing and EOS suppression, not natural answer behavior. The rank/D failure stands without that continuation result.

009 shows **this format change did not work**. It does not show every nameable intermediate must remain output-accessible. Returning to original brew is reasonable if the next question can be answered there; it is not a confound cure. Starting the held reader phase merely to obtain another positive D would add little. At depth1 z=f, and 77 of the 112 probe sites precede the token distinguishing the tasks. Only 33/35 post-divergence sites support the informative both-task decodability comparison. The task's mathematical dependency is not the subject model's demonstrated internal algorithm.

010's main correlations reproduce. Oracle L40–J_CB is 0.6341, Oracle L40–output0.3025; NLA–Oracle0.2398/0.3025; NLA–output0.0702. High 0.63–0.80 J association belongs to the Oracle, not both verbalizers. Late Oracle L60–output0.8217.

A new, explicitly post hoc check subtracts each table's mean D before correlating. Oracle–J becomes **0.7978 [0.6929,0.8720]**, versus Oracle–output **0.1524 [−0.0150,0.3026]**. The specialist independently bootstrapped tables; root separately reproduced both point estimates from the saved per-pair data. This rejects a purely additive between-table difficulty explanation of Oracle–J agreement. Colour-pair geometry, shared training and other within-table factors remain.

Linear adjustment for one output D does not control the full output distribution or nonlinear relationships. Report31's suggestion that adding this covariate resolves the confound is incorrect. Likewise, 010's no-association reader–S result has broad intervals (NLA roughly −0.371 to0.249); it is not a powered equivalence finding. Prefix agreement measures sensitivity to wording, not repeated-measure reliability of a fixed instrument. Batch-layout noise need not attenuate every correlation unless additional independence assumptions hold.

Oracle's approximately 1.85-fold stronger D at the low-patch-leverage `</think>` newline remains a useful mismatch. A fixed colour prior cancels from D; cell-dependent gain and reader behavior do not. No Stage1b prose readouts were saved, so existing text cannot settle what Oracle says at that site. Available likelihood matrices can test broad colour amplification versus role-selective amplification.

Detailed recomputation, sources and limits: [009–010 audit](/workspace/codex_reports/03_009_010_audit.md).

## 5. 011: current evidence and defects

The new complete s1 results replace the stopped74/82 handoff and two-pair smoke. On 82 pairs/33 tables, L40/44/48 only, both coordinate controls fail: whitened answer controlS=0.0156219, raw answer controlS=0.0504021. Their weak z effects do not establish irrelevant AR directions.

Additive AR S=0.0893576; class-mean S=0.1964916. Paired AR minus random=0.0580564 [0.0249817,0.1089637]; AR minus class mean=−0.0983995 [−0.1515721,−0.0486703]. The specialist independently reproduced the key medians and cluster intervals from all82 raw pair files, and root checked the saved statistics. “Partial” is justified. This is directional influence, not natural use, and is never reader-free label evidence.

The design says the last interval is wholly below−0.05. It is not: its upper endpoint is−0.0486703. It is below zero; equivalence was not established. Rounding must not decide the threshold. Clean-reference norm matching also does not equal cumulative write matching in 011.

The AR loader passed a post hoc rule after two failures. A fitted global scale on selected16-token crops and unmatched retrieval batches is not reproduction of published unscaled FVE on the original span distribution. There are768 crops from12 responses, not768 independent contexts. Common positive scale/offset really do cancel in centered coordinate swaps and unit-normalized differences; the unresolved risk is other input-dependent or anisotropic error.

Low one-token cosine is not evidence that colour differences are mostly noise. A short phrase underdetermines its preceding activation; an MSE conditional mean can have low individual-target cosine and still preserve useful colour contrasts. The additive result is evidence against treating current differences as entirely uninformative. Longer phrases could improve a contrast or simply introduce extra contextual content.

Metadata referencing a J lens, loss_mix_lambda0.25 and val_jloss/jfve makes training-induced Oracle/J dependence plausible and worth clarifying. It does not establish which loss ran. “Joint bullet FVE” does not itself mean J-space. Earlier suspicions about full64 layers and ridge0.1 are not established loader bugs: heads include layer63, and loss_ridge_c is0 even though ridge_c is0.1.

**Definite control defect:** `ar_jbias.py` calls its PCA control1,024-dimensional, but root verified the fitting tensor is384×5120 and the code uses reduced SVD. It supplies384 columns, meaningful centered rank at most383. J/random have1,024. The reported J ratios1.2653/1.1507 remain descriptive; the purported dimension-matched PCA comparison does not. Its current ambiguous threshold verdict would not automatically change, because J alone misses the both-layer bias cutoff. Claude should record actual ranks and choose a feasible common-rank comparison or adequate independent fitting data.

Neither modest global energy bias nor FVE discrepancy can quantitatively explain or rule out a much larger nonlinear colour-likelihood preference. The design's “too small to explain007” and “therefore jfve is a different quantity” readings exceed what these mismatched statistics establish.

Detailed evidence: [011 audit](/workspace/codex_reports/04_011_audit.md).

## 6. Claims to retain, claims to retire

Retain: replicated Oracle projector sensitivity at L40; reader/checkpoint/site-specific differences; the powered natural-coordinate sparse-J disadvantage in008; 009's failed confound-removal design; weak and prefix-dependent NLA–Oracle agreement; exploratory additive AR influence beyond random; recurring decodability with low net single-cell leverage.

Retire or narrow: “model gives J no special purchase” across Oracle layers; “reader reads unused information”; “all readers privilege J”; “output covariate removes the confound”; “z necessarily computed for the task”; “AR colour directions are mostly noise”; “loader matches published validation”; “norm-matched” without clean-reference qualification; “clamp prevents J re-entry”; formal equivalence inferred from an interval crossing zero.

The project has not established a complete mechanism, a causal reader ranking, a universal failure of J privilege, natural necessity of the tested directions, or an output-independent reader signal. These are unresolved questions, not reasons to discard the observations.

## 7. Three scientific questions worth resolving next

1. **How much of Oracle's J preference comes from the measurement instrument?** The cheapest first action is to resolve the shipped AR/Oracle forward and training-loss contract using an authoritative implementation or trusted reference input/output. This could materially change the interpretation of007 and whether further011 phrase experiments test the intended instrument. A J-weighted loss would make Oracle/J agreement less independent corroboration; it would not prove that all observed preference is explained. After contract validation, colour-contrast geometry and readable, dimension-correct controls are more relevant than an arbitrary whole-energy threshold. No new paid service or contact is authorized by this recommendation.

2. **Can we isolate an intermediate-specific causal effect from general colour/output geometry?** First use saved008/011 endpoint changes, direction metadata and clean captures to quantify intended donor movement, role imbalance, and the actual clamp constraints. If a fresh test is needed, Claude should use a small design where the same intermediate replacement predicts different answers under different downstream mappings, with answer-direction, random/dose and damage controls. Correct mapping-dependent answers would weaken direct answer steering; fixed colour effects or broad degradation would weaken the intermediate account. This is the binding uncertainty for the project's causal adjudication objective. Strong overall leverage alone has already been established and need not be rediscovered.

3. **Does any reader add useful prediction of a reader-free outcome beyond accessible baselines?** Existing006/007/010 scores and matched model-side outputs permit a bounded table-held-out analysis before new reader runs. Compare simple output-colour-vector, J-score and metadata baselines with and without each reader; evaluate both raw margin change and normalizedS. Use common sites/pairs and state whether the outcome is whole-cell or content-specific. A stable held-out gain would support incremental predictive usefulness under the assay, not natural-use decoding. Failure with broad uncertainty means the saved set cannot settle it. If there is little variation or no credible endpoint after question2, stop instead of scaling regression complexity.

The first action is inexpensive instrument clarification; the second question is most central to the North Star. The third can reveal whether existing reader measurements add practical value without prematurely claiming mechanism. None requires a confirmatory007c NLA rerun, and none is an instruction for this auditor to implement changes.

## 8. Provenance, coverage and context preservation

Root reviewed the original paper, required substantive notes, project instructions/specification/amendments, canonical experiment reports, and historical reports25–29/31. Related AO/AObench/Li/Dingeto literature was read selectively for measurement distinctions; this is not a comprehensive literature review. Three non-nested reviewers covered007/008,009/010 and011. Root independently checked their consequential statistics, the Stage2 thresholds, the PCA input shape and SVD construction, clamp definitions, Oracle summary numbers, and the within-table correlation point estimates. Raw-data bootstrap replication details remain in the specialist reports; no claim of independently rerunning every experiment is made.

Living decisions and held-out status are preserved in [00_project_context.md](/workspace/codex_reports/00_project_context.md). The brew test and multihop test have already influenced analyses; 009 test banks were not opened by this audit and remain documented as untouched. Existing project summaries disagree about011 status; newest artifacts/design completion govern this report. All future substantive updates should record what new evidence changes, rather than silently upgrading older observations.
