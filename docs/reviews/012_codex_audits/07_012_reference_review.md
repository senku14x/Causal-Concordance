# 012 reference evidence review — 2026-10-06

Scope: independent read-only review of report 32, Jan's original 16-page writeup, the portable phrase-lens reference paths needed to interpret Stage A, and the saved four-item smoke. No experiments, model loading, dependency changes or repository edits were performed. Relevant paper figures were rendered only under `/tmp`. This note does not reproduce private correspondence.

## Assessment

Jan's reported ST multihop result makes a short reproduction attempt worthwhile. It does not yet make the portable reference implementation a validated ground truth, does not identify why 008's brew intervention was weak, and does not support the current explanation of the smoke's unchanged outputs. The most concrete correction before interpretation is mathematical: near-parallel unit source and target rows do **not** necessarily yield a small pseudoinverse coordinate swap.

## Original evidence and prior

Read `/workspace/Jan Chen - A Turn of Phrase_ Multi-Token Lenses from Gradients and Templates.pdf` in full text and directly inspected the causal figure on page 12. It supports report 32's approximate J-lens ST multihop success rate of 59%, compared with a small random-control rate. ST broadcast is substantially weaker, around 26%. The stronger result therefore concerns one task, not a universal band-specific intervention lever. The writeup reports heterogeneous method denominators and a lenient accepted-phrase-in-continuation endpoint. It explicitly acknowledges tuning on the banks, repeated target concepts, unresolved MT relation issues and lack of new-concept validation.

The local portable clone is `/workspace/sources/ext/phrase-lens-src`, recorded by the project at `ea85583`. Its `docs/experiments.md` and `docs/evaluation.md` explicitly state that the release's CPU and toy-transformer tests do not imply a full Qwen end-to-end numerical reproduction. It uses a different tested package environment from this instance. This is not evidence of an implementation failure; it means Stage A is validating a released reproduction path, not replaying authenticated historical outputs. The original writeup and the portable code are distinct sources of evidence.

Report 32 §8 overstates what follows when it says the 008 arm-A null was very likely about band and positions. Jan's result differs from 008 in task, source/target concepts, prompt rendering, numerical regime, readout endpoint and spatial extent, as well as layer band. 012's band sweep can establish band dependence within these multihop prompts. It cannot by itself attribute the historical brew difference to band and positions.

## Smoke: observed result versus proposed explanation

The saved artifact is `outputs/012_reader_concordance/stageA/runs/smoke_j_swap_L24-47.jsonl` (four rows).

- `atomic-26-symbol`: clean/intervention/random all ` Fe.\n`.
- `atomic-79-symbol`: all ` Au.\n`.
- `birthstone-emerald-month`: all ` 5.\n`.
- `chem-bones-Z`: all ` 26. What`; the clean output already contains the target answer and is not source-correct.

Thus there are zero observed new target-answer transitions in the four-item smoke, and zero target successes among the three source-correct items. The unfiltered 1/4 treatment and 1/4 random rates are contaminated by pre-existing target output. The pre-full-run amendment to condition Stage A on clean correctness addresses this observed failure mode. It should also explicitly exclude clean outputs containing the target answer: source correctness alone does not rule out mentioning both answers under the actual matcher.

All four items record **zero collision layers**. The edit class only explicitly skips a layer if row cosine exceeds 0.999. The first three source-target cosines are high at the beginning of the band but fall appreciably later; for example iron/copper reaches about 0.556 at L47.

The spec says these were no-ops because row cosines of 0.83–0.93 imply a tiny coordinate swap. That inference is false for the actual primitive.

For unit, distinct, non-collinear directions s and t, let V=[s,t], let c=V†h, and swap the two coordinates. At a clean activation and alpha=1,

`delta = V(swap(c) - c) = -2 u (u^T h)`,

where `u=(s-t)/||s-t||`.

To see the cancellation, put rho=s^T t. The coefficient difference is `(s-t)^T h/(1-rho)`, while `||s-t||^2=2(1-rho)`. The swap reflects the activation's normalized contrast component. High similarity reduces the raw row difference but increases the pseudoinverse coefficient difference; similarity alone does not force a small edit. Near exact singularity can create conditioning concerns and the code has an explicit collision threshold, but neither explains these rows by the stated argument.

At later layers, after preceding edits, the current activation differs from the clean reference and the general clamp is `h'=(I-VV†)h + V swap(V†h_clean)`. It is then not simply a reflection of current h. This strengthens the need to inspect actual applied edits rather than infer their size from cosine.

The saved outputs show no change in greedy completions. They do not show unchanged activations or unchanged answer probabilities. No edit norms, achieved coordinates or logit margins are saved in these rows. Those omissions prevent the existing smoke from discriminating a genuinely small edit, a nonzero ineffective edit, a margin shift too small to flip, or a runtime application problem.

## Narrow implementation check

Inspected only `SwapEdit`, `BandEditor`, `ModelRunner.generate`, and outcome/rendering paths needed to resolve the claims above.

- `interventions.py:26–75`: targets are swapped pseudoinverse coordinates of separately captured clean residuals; alpha=1; directions are checked as finite unit vectors. This matches the stated algebra.
- `interventions.py:92–133`: hooks register on each band block, replace every prompt-position residual once, cast back to model hidden dtype, and skip later decode forwards using a per-layer `done` set.
- `modeling.py:124–143`: the editor context surrounds `model.generate`, so hooks are installed during generation's prefill.

I found no concrete hook bug in this path. But there is no persisted assertion that all expected edited layers ran, nor evidence of nonzero post-cast edit magnitudes. Therefore code plausibility is not a runtime verification. Claude could add those diagnostics to an authorized smoke rather than infer correctness from unchanged text or launch a broad source audit.

A useful small smoke would record, for at least one source-correct pair, exact token IDs, called layer count, pre/post-cast edit norms, clamp residual error and target/source answer margins. One known successful historical item would be particularly informative if available. No such additional experiment was run by this reviewer.

## Prompt and scoring contracts matter for parity

`run_intervention` strips the prompt with `rstrip()` before tokenization (`interventions.py:178`). It also stores the stripped prompt. Stage C's proposed WSB render retains the space on affected items. The claim that this does not matter because all positions are edited is incorrect. All-position editing is a spatial choice; changing tokens changes model computation, clean reference coordinates, last token, answer boundary, and potentially the answer.

In a causal decoder, activations at the last word before an appended space may be identical if tokenization preserves the entire prefix, but the subsequent space state and answer distribution need not be. That restricted prefix observation cannot establish equality of the intervention outcome. Byte/token equality must be checked, not inferred from prose.

Stage A can keep Jan's render to reproduce its reference. Stage B's parity comparison must then use exactly that render and answer budget. If the WSB render is used for the substantive Stage B/C comparison, it is a separate condition that needs its own clean gate and causal label. Readers should consume captures from the exact run supplying their label. A 90% aggregate flip agreement criterion cannot distinguish numerical-engine disagreement from a deliberately changed prompt contract.

The actual `strict_match` in `metrics.py:108–130` normalizes Unicode, case and punctuation and searches for an accepted answer as contiguous words anywhere in the continuation. It is not exact whole-answer equality, despite the name. Outputs naming both answers, correcting themselves, or negating the target can pass. Hand auditing successful outputs and rescoring exclusions is possible directly from the saved text.

The generator does not always generate eight tokens: its budget is accepted-answer length plus two, capped at eight. The four smoke rows use budgets 3, 3, 4 and 5. Evaluation uses source- or target-specific prefixes (`evaluation.py:147–162`). Stage B's fixed eight-token success endpoint can disagree with Stage A even when their underlying model trajectories are identical. Match the reference endpoint for parity, then report any more semantically restrictive answer endpoint separately.

The random control is genuine: it retains the true source direction and replaces the target direction with a seeded isotropic unit vector (`interventions.py:207–221`). This is useful for checking target specificity. It is not a norm-matched control on the actual applied displacement: pseudoinverse coordinate geometry and clean projection determine dose. Stage B's additional matched random control addresses a distinct question and needs a precise matching definition. One random draw per item is an observed comparator, not a calibrated per-item null distribution.

## What existing evidence can settle now

1. Recompute Stage A success only on clean-source-correct, clean-target-absent, covered items; retain the original cohort rates separately. The smoke then clearly has no demonstrated successful swap.
2. Inspect generated text for both-answer matches and accepted-alias ambiguities before treating reference success as an answer flip. This requires no model run.
3. Compare exact source prompt strings/tokenization and generation budgets across the planned paths before asserting parity or linking labels to WSB-render captures.
4. Correct the near-parallel-row explanation algebraically. Existing artifacts cannot determine actual edit size or probability movement; that requires runtime diagnostics in a future authorized smoke.
5. Treat Jan's figure as a reason to attempt reproduction, with historical denominator/scoring uncertainty. No saved result in the portable clone currently establishes that it reproduces the historical figure.

## Recommendation

The multihop direction is scientifically worthwhile and a short, instrumented reference check is a high-information first action. The complete 012 pipeline should not yet inherit the assumptions that the reference lever has reproduced, that unchanged text means no activation edit, that prompt stripping is harmless, or that 008's earlier failure is now explained. Resolve these narrow contract and measurement questions first; they could prevent the expensive reader stage from answering a different question than intended.
