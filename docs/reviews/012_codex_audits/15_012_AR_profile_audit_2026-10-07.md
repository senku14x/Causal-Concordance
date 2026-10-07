# 012 E0/E0a: independent Oracle-AR profile audit — 2026-10-07

Read-only review of the E0/E0a ledger entries, design §§17–17a, saved profile JSON, `stageE0_ar_profile.py`, `stageE0a_jfve_scaled.py`, the narrowly relevant 011 loader functions, pinned public AR/Oracle metadata, and saved validation spans. No model experiments, canonical workflows, repository edits or dependency changes were performed. This report contains no private correspondence.

## Bottom line

The run provides useful evidence that the local reconstructor extracts meaningful directions and that its scaled reconstruction profile resembles the published profile. It **does not validate the loader under the project's rule**. It also makes a mean-centered J-transform metric a plausible leading interpretation of `jfve`, but does not identify that metric conclusively, establish that a J loss contributed gradients during AR training, or show that the Oracle's reinforcement-learning reward explicitly included J-FVE.

The chain must remain separated:

1. The checkpoint is associated with a J-lens configuration and J evaluation metrics — directly documented.
2. `jfve` is precisely a particular centered, scaled J reconstruction metric — supported as a candidate, still unresolved.
3. That metric was used as an AR training objective with a particular mixing weight — plausible metadata-based hypothesis, not established by matching validation metrics.
4. The Oracle's reward explicitly mixes that J term into bullet FVE — presently unverified.
5. Such a reward explains 007's Oracle dependence quantitatively — not tested.

## What the new measurements actually show

Independently recalculated all threshold counts from full-precision `results/012_reader_concordance/analysis/stageE0_ar_profile.json`:

| Check | Count | Written requirement |
|---|---:|---:|
| Scaled whitened FVE within 0.04 of publication | 12/12 | at least 10/12 |
| Retrieval@1 within 0.05 | 8/12 | at least 10/12 |
| Unscaled whitened FVE within 0.04 | 2/12 | at least 10/12 for full validation |

The retrieval misses are L20, L24, L28 and L32. At L40, the scaled FVE is 0.233459 against 0.224 and retrieval is 0.900794 against 0.92. At L44, these are 0.212609 against 0.2202 and 0.913690 against 0.93. Thus the mid-layer agreement is encouraging, while the all-layer loader gate is correctly recorded as failed.

The gate is a chosen operational acceptance criterion, not a test with calibrated type-I/type-II error. Failing it says that the instrument has not met the agreed validation bar; it does not distinguish wrong loader details from evaluation-distribution or retrieval-pool differences. Passing it would likewise not uniquely identify the official implementation.

The phrase “one global scale” is misleading without qualification. The script fits **one scalar separately at each layer**, using the same examples on which FVE is then evaluated: 12 scalars for the main profile. It is global across examples within one layer, not a single calibration for the whole AR. Values rise monotonically from 0.1127607423 at L20 to 2.3344610489 at L63, with 0.3800409132 at L40 and 0.4180703779 at L44.

The main calculation is:

`T = W(y − μ); P = W(ŷ − μ); alpha = <P,T>/<P,P>`

`scaled FVE = 1 − ||T − alpha P||²/||T||²`.

This is the squared cosine between the flattened prediction and target arrays. It measures aggregate directional alignment after fitting amplitude. It does not demonstrate that the native AR output has the correct norm or that this is the published evaluation formula. In raw coordinates the calibration is `μ + alpha(ŷ − μ)`, an affine rescaling about the supplied mean, not simply multiplication of ŷ.

A scale-invariant training loss is consistent with the depth-dependent output calibration. It is not identified by it: target norm growth, fixed head-side normalization, an omitted output transformation, and evaluation convention differences can also contribute. The ledger statement that this “implies the published FVEs are scale-aligned” is too strong. The actual published metric code remains needed.

A common positive scalar per layer would cancel from normalized within-layer concept contrasts and suitably defined coordinate swaps. Thus the scale pattern alone does not rule out those uses. The unresolved issue is whether directional and input-dependent errors are small enough for the proposed comparison, not merely whether a fitted scalar exists.

## These are reused calibration texts, not fresh held-out loader validation

E0 uses 1,024 crops drawn from **the same 12 deterministic assistant responses used to choose 011's loader conventions**. I verified this from saved artifacts: reconstructing 011's crop selection on E0's `texts.json` reproduces **768/768** spans in `outputs/011_olens_swaps/d2/diag_tensors.pt` exactly.

E0 changes the crop-length distribution to uniform 1–32 tokens, and adds layers. Those are useful extensions. They do not make the underlying text distribution independent of the prior instrument selection.

Independent crop reconstruction found:

- 12 responses; eleven have 160 tokens and one has 102.
- 86 sampled crops per response except the final response, truncated to 78.
- 1,017 distinct response/start/length triples among 1,024 crops.
- 1,012 distinct token-sequence spans.
- 792 distinct preceding-token target positions.
- 34 new crops have exact text matching an old 011 crop; 25 match within the same response.

The crops also overlap extensively within each response. The unit count “1,024 spans” should not be read as 1,024 independent validation texts. The current summaries contain no uncertainty across response texts. Their layer agreement is also correlated because all layers are evaluated on the same crops.

The term held-out could mean absent from the AR's original optimization, but the local analysis has not checked the AR's original training corpus for overlap. More importantly for this project's inference, the texts are not held out from **local loader convention selection**.

The public config records `max_eval_rows=1024`, `micro_batch=24`, uniform span lengths 1–32 and the original `ml_pairs_iolens_chat` corpus. Matching these few settings does not establish identical source distribution, target sampling, batch assembly or retrieval protocol. The local reconstructor forward uses batches of 32 with variable right padding; retrieval uses a separate fixed random permutation into groups of 24. A published training micro-batch setting does not by itself establish that the published retrieval pools were constructed in the same way.

The local retrieval implementation drops the last partial batch: it evaluates 42 groups of 24, **1,008 rows**, not all 1,024. This is a small, explicit protocol detail rather than an explanation of the main discrepancy. Repeated targets and closely related crops can affect retrieval ties and difficulty, so retrieval@1 is not portable independently of its distractor construction.

## What candidate (b) means, and what was matched

E0a fits candidate (b) separately at each layer:

`T = J_L(y − μ_L)`

`P = J_L(ŷ − μ_L)`

`alpha_L = <P,T>/<P,P>`

`FVE_b = 1 − ||T − alpha_L P||²/||T||²`.

Here μ is the AR whitener's supplied mean. This is a full J-transform error geometry, **not** 007's orthogonal dense J-subspace projection and not a sparse J part. Singular directions of J are weighted by their singular values, so the distinction is scientifically consequential.

Recomputed E0a counts:

| Candidate | FVE within 0.04 | Retrieval within 0.07 |
|---|---:|---:|
| (a) sample-centered raw J transform | 0/11 | 9/11 |
| (b) J transform around supplied mean | 8/11 | 10/11 |
| (c) J-lens logits | 0/11 | 8/11 |
| (d) dense J projection | 0/3 | 3/3 |

Candidate (b) passes the FVE tolerance at every tested layer L32–L60. Its early-layer misses are substantial: L20 0.257356 versus 0.3854, L24 0.300298 versus 0.4234, L28 0.367191 versus 0.4819. L44 is 0.563738 versus 0.5887, and L48 is 0.621554 versus 0.6075. The prespecified/post-hoc-amended requirement was at least 9/11 FVE matches, so `jfve` correctly remains open.

There is a small stale statement in the E0 ledger: unscaled candidate FVEs did not miss everywhere. Candidate (b) already matched at two layers, L52 and L56, as the saved verdict records. It still failed the identification gate.

Candidate (b) is a reasonable leading hypothesis because the metadata independently names the J checkpoint, excludes L63 from J evaluation, and records `jspace_mu_prefix=whitening_iolens_pooled`. Nevertheless, finite aggregate profiles from a different evaluation sample cannot identify a metric uniquely. The candidate search is also not exhaustive: pooling, normalization, centering and scaling choices remain degrees of freedom.

### A comparison asymmetry in E0a

Candidates (a), (c) and (d) subtract **separate sample means** from their targets and predictions before fitting scale. This implicitly removes a vector mean error, equivalent to allowing an intercept, rather than only fitting one scalar to the original predictions. Candidate (b) instead subtracts the same fixed external mean from both.

Furthermore, E0a reuses E0 retrieval values because cosine is invariant to positive global scaling. That justification does not extend to the changed centering of (a), (c) and (d): cosine retrieval is not invariant to subtracting different sample means. Their FVE and retrieval entries therefore do not evaluate the same transformed vectors. Candidate (b)'s retrieval is consistent with its FVE geometry apart from positive scaling, so this issue does **not** directly invalidate its 8/11 and 10/11 result. It does weaken claims that the candidate comparison has comprehensively eliminated the alternatives.

The dense candidate has only three available layers and cannot reach the fixed nine-layer threshold by construction. Its FVE fits are poor on those three layers, which is evidence about those conditions; its “not identified” Boolean is not a comparable eleven-layer test.

## The J-training and Oracle-reward chain remains incomplete

The pinned AR `meta.json` directly records `val_jloss`, `val_jfve`, `val_jret1`, J checkpoint provenance, `loss_mix_lambda=0.25`, `loss_space=whiten`, and a J-mean prefix. This is substantial evidence that the training/evaluation setup had a J-aware pathway. But config names do not reveal whether the J term actually contributed gradients in this run, which branch `loss_space=whiten` selected, the weighting convention, or the exact loss formula. Matching a validation metric cannot establish those facts: a metric can be evaluation-only.

The pinned Oracle `run.json` says it used a joint bullet-FVE reward on this AR. It does not specify a J term. “Joint” can describe reconstructing the activation from several bullets together; it must not be interpreted as “joint raw plus J loss” without the reward definition.

Even if AR training used a J term, the Oracle reward could use ordinary whitened reconstruction error on that AR's outputs. That would allow an **indirect** measurement bias through the trained reconstructor, without an explicit J-weighted reward. These are different explanations and require different evidence.

Finally, an explicit J-based objective would make Oracle–J agreement less independent as evidence for privileged model computation. It would not alone explain the magnitude or precise geometry of 007's dense-projection effect, nor make every Oracle result circular. Demonstrating that explanation would require a relevant training/reward ablation or predictions that distinguish objective-induced sensitivity from alternatives.

## Highest-value next evidence

The cheapest decisive evidence is the exact official forward contract and the implementations/formulas for AR training loss, validation FVE/J-FVE, and Oracle bullet reward. A small official reference input/output fixture would validate the local forward independently of matching summary scores. If official code is unavailable, the questions should distinguish those three metrics explicitly rather than asking only whether the AR “uses J.”

Do not continue nearby metric variants until one happens to cross the profile threshold. Freeze the chosen forward and metric convention, calibrate scales on one text set, and evaluate on fresh response-level texts if independent empirical validation is required. This would assess generalization of the local instrument; it would still not recover the historical training objective.

The appropriate present wording is: **“The local AR reproduces the published scaled FVE profile reasonably well but fails the agreed retrieval gate. A mean-centered J-transform metric is the leading tested candidate for J-FVE. The AR training loss and the Oracle reward's dependence on J remain unverified.”**
