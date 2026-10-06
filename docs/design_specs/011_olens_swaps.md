# 011 — Oracle-space swaps: is the Oracle Lens's own colour vector a handle on the model? (exploratory note)

**Status.**
- Written before any 011 run.
- **Origin:** Camila's tip via the PI (2026-10-06): put a word through the Oracle Lens reconstructor (AR), get a vector,
  do that for two words, and swap them in whitened space.
- The PI decided we build the vectors ourselves.
- Experiments about reader geometry; never labels. Code: `scripts/011_olens_swaps/`.

## Vectors (`ar_vectors.py`)

- **AR:** `agu18dec/olens_and_ar` @ c95215d, `ar_ptag_pooled` (README):
  - Qwen3.6-27B text model plus a LoRA (r16, α32, every linear layer), with the checkpoint's `_orig_mod.` keys stripped;
  - one shared Linear(5120, 5120) head;
  - input = the "[Layer NN]" tag (6 tokens, from `heads.pt`) + the phrase; read at the phrase's last token.
- **Whitener:** the AR's pooled whitener (`agu18dec/local-workspace` `ckpts/ar/ptag/whiteners/whitening_L{40,44,48}`;
  cov + mu; lmsys-chat-1m).
- **Validation first.** The undocumented conventions are settled by fraction of variance explained (FVE) on 768 held-out
  stock assistant-text spans:
  - the layer index (block L vs block L−1 output);
  - the output space (ZCA-whitened, PCA-whitened or raw);
  - the read point (post- or pre-final-norm).

  **Acceptance:** at L44 the best read/space convention reaches FVE ≥ 0.10 (published value 0.220) and beats every other
  read/space convention by ≥ 0.05. **Otherwise no vectors are written and no swap runs.**
- **Colour vectors:** " c" (leading space; the no-space form is kept as a sensitivity) for the 12 brew colours at L40,
  L44 and L48.

## Swaps (`olens_swaps.py`)

- **Material and site:** Track B's 82 gated brew dev pairs and clamp primitive, at the three emit cells, L40/44/48 (the
  AR's layers inside Track B's band).
- **Primary `C_white` (Camila's swap):**
  - V~ = the z and z′ vectors, each centred over the 12 colours (template-style), in the AR's whitened space;
  - coordinates c~ = V~⁺ W (h − μ); σ swaps them; map back with A = W⁻¹ V~;
  - clamped per layer to the clean run's swapped coordinates.
- **Variants:** α = 2; uncentred vectors; raw coordinates along the same directions.
- **Controls:**
  - answer colours a ↔ a′ (positive);
  - two off-trajectory colours (negative);
  - 4 random whitened direction pairs, edit norm matched to the primary's;
  - Track B's class-mean full-direction swap at the same three layers (the lever reference).
- **Gates:** G0 (zero clamp exact), G1, G-clamp, G-RA, G-repro (Track B's).

## Reading

| Outcome | Reading |
|---|---|
| Power gate fails (answer-colour swap median S < 0.15) | The z-swap is not read: AR colour vectors are not a handle on the answer at these cells and layers. Same as arm A's J-lens atoms |
| Power passes and `C_white` (z) ≈ the class-mean reference | The Oracle's colour vector is as good a handle as the model's own class-mean direction |
| Power passes and `C_white` ≪ the reference | The Oracle's notion of the colour is decodable but a weak handle on the answer. Extends the reader–model dissociation |
| The negative control moves S as much as the primary | The edit is generic, not colour-specific |

- Each reading is scoped to brew dev, Qwen3.6-27B, the emit cells and L40/44/48.
- On brew the AR vectors are "about to say colour X" directions, so the runner-up confound applies. The same swap on the
  009 code family, where no colour is said, follows if 009 passes its checks.

## Execution status (2026-10-06, 20:00 UTC; factual record, no rule changed)

- **Validation v1 failed the acceptance rule** (`results/011_olens_swaps/ar_validation_v1.json`). Raw FVE is negative
  under all 36 convention cells (best at L44: −2.72, block L, pre-norm, ZCA). No vectors were written and no swap ran.
- **Diagnostics run d1** (`--diag-only`, `results/011_olens_swaps/ar_diagnostics_d1.json`; scale-free, added after v1
  failed; exploratory):
  - retrieval@1 among the 768 spans: 0.35-0.37 at L44 and 0.42-0.45 at L48 under the ZCA and raw spaces (chance
    0.0013); the PCA space is at chance;
  - mean cosine ≈ 0.2 in ZCA space;
  - predicted norms 1.6-3× the targets';
  - FVE after a per-sample least-squares scale at most 0.053.
  - Reading: the loader is partly right (the predictions point roughly at their targets), but far from the published
    fit. Candidates: a scale or normalisation convention in the head, the target definition, or a metric mismatch (for
    example, retrieval within small batches, or FVE in another space). Not yet distinguished.
- **Next:** find the published metric's exact definition and the head's output convention; then, if the acceptance
  rule must change, write the change here as an amendment before any vector is used.

## Amendment 1 (2026-10-06, written before run v2; the v1 and d1 outcomes above had been seen)

- **Bug fix: the wrong whitener.** The AR was trained against `whitening_iolens_pooled_L{L}` (2.09M samples, chat +
  pretraining; `meta.json` `config.whitener_prefix`). v1 and d1 loaded `whitening_L{L}` (lmsys-chat-1m, 28.7M samples).
  Fixed. The two have similar moments at L44 (trace/d 0.766 vs 0.779, |μ| 74.6 vs 77.6), so this alone cannot explain
  v1, and raw-space FVE does not depend on it.
- **More read conventions.** The AR's config says `head_layer_norm: true`, but `heads.pt` holds only the Linear's
  weight and bias, so any head-side norm is parameter-free. The candidates now add three head-side normalisations:
  parameter-free LayerNorm of the pre-norm and post-norm reads (`ln_pre`, `ln_post`), and RMS normalisation of the
  pre-norm read without the gain (`rms_pre`).
- **Target unchanged:** the activation preceding the span (WSB post, Oracle Lens appendix: "predict the activation
  preceding it"). The HF README says "the span that produced it", so the span's last token is reported as a
  sensitivity. It is never a candidate.
- **Acceptance unchanged** (FVE ≥ 0.10 at L44 for the chosen convention; margin ≥ 0.05 over every other read/space
  convention at the chosen layer index), with one clarification: a rival read whose head outputs agree with the
  winner's at median per-span cosine ≥ 0.99 (L44) counts as the same convention, because its vectors would be
  interchangeable. It is therefore not a rival in the same space. Different spaces are always rivals.
- **Records:** results are now per run (`ar_validation_<run>.json`, `ar_diagnostics_<run>.json`); v1's and d1's files
  were renamed with contents unchanged. The swap script reads the validation file of the vector run it uses.
- **If v2 fails too:** no swaps; the diagnostics are reported.

## Amendment 2 (2026-10-06, committed 20:37:36 UTC per git; header time ~20:50 was wrong, report 31; POST HOC: written after v2, d2 and the offline sweep had been seen)

**What was seen.**
- v2 (amendment 1) failed the pre-set rule again. The scale-free diagnostics barely moved (ZCA cosine ≈ 0.2).
- d2 saved the head outputs and targets. The offline sweep (`ar_offline_diag.py`, `results/011_olens_swaps/ar_offline_d2.json`)
  scored a convention v1 and v2 never scored: **the head outputs a raw activation**, scored in the whitened metric
  W(ŷ − μ) vs W(h − μ). This fits the AR config (`loss_space: whiten`), with the head computed on the backbone's final
  pre-norm state passed through a parameter-free LayerNorm (`head_layer_norm: true`).
- On 16-token spans (training crops are uniform on 1–32 tokens; 40% of our spans are 1–2 tokens and carry little):
  - after one global least-squares scale, whitened FVE is 0.27 [not in a committed JSON; a recomputation for the chosen convention (`ln_pre`, no ridge) gives 0.26 (report 31); 0.27 matches the `pre_norm` read or ridge 0.1] / 0.24 / 0.24 at L40 / L44 / L48 (published val FVE
    0.22 / 0.22 / 0.23);
  - retrieval@1 within random batches of 24 (all spans) is 0.83 / 0.85 / 0.89 (corrected from 0.90: `ar_offline_d2.json` `L48|ridge0.0|block_L|ln_pre|raw` ret_b24 0.8945; report 31) (published 0.92 / 0.93 / 0.96);
  - block L beats block L−1.
- **Unexplained:** the prediction scale. The best global factor is 0.3-0.55, where an MSE-trained head should need ≈ 1.
- **What the swap needs is unaffected.** The swap uses the colour vectors centred over the 12 colours, in whitened
  coordinates. A global output scale, an input scale through the affine head, and a constant offset (raw vs centred
  raw) all cancel in the centred directions. Centred predictions from the five reads agree at median cosine ≥ 0.995
  (L44).

**Acceptance rule used from now on (post hoc).**
- Convention: raw output scored in the whitened metric; read `ln_pre`; block L.
- Accepted if, at L44:
  - on 16-token spans, the whitened FVE after one global scale is ≥ 0.10;
  - within-batch-24 retrieval@1 is ≥ 0.80;
  - and on 16-token scaled FVE, the convention beats the ZCA- and PCA-output conventions (which change the centred
    direction) by ≥ 0.05.
- The pre-set rule's verdict (fail) is kept in every record (`accepted`); this rule's verdict is recorded separately
  (`accepted_posthoc_a2`).

**Changes to the swap arm.**
- Only centred colour vectors are used. `C_white_unc` is dropped: it depends on the unexplained scale and offset.
- Every 011 swap result is labelled **post hoc, exploratory**.
- Run order: a 2-pair smoke (gates and timing) is inspected before the 82-pair run.

## Amendment 3 (2026-10-06, committed 20:43:06 UTC per git, before s0b and s1 ran; header time ~21:05 was wrong, report 31; POST HOC: written after the 2-pair smoke s0, before any other pair ran)

**Seen in s0** (2 pairs, one table; vectors run v3 under amendment 2):
- all gates pass;
- the reference moves S 0.23;
- `C_white` −0.01 / +0.00, its answer-colour control −0.01 / +0.00, with small edits (summed norm 15–21; the clean
  activations' whitened coordinates on the AR colour vectors barely differ);
  - *Correction (2026-10-06, report 31):* "the swap edits almost nothing" is the wrong explanation. The `C_white` edit is
    (c_z′ − c_z)·W⁻¹(v_z − v_z′), i.e. the same raw direction as `D_ar`, signed by which AR vector the clean activation
    projects on more. At a similar total edit `C_white_a2` gives S −0.022 / +0.011 while `D_ar` gives +0.104 / +0.094:
    the null is because the clean coordinates do not favour AR(z) over AR(z′). `C_white` also depends on W (no ridge);
    `D_ar` does not.
- `C_rawcoord` +0.21 / +0.10 (corrected from +0.22: per-pair S 0.214978, 0.096656; report 31), but with edits 2–3× the reference's and no matched random or answer control.

**Added arms** (the existing ones are unchanged):
- `C_raw_answer`: the answer colours swapped in raw coordinates (power control for `C_rawcoord`);
- `C_raw_rand_0/1`: random raw direction pairs, swap edit rescaled to `C_rawcoord`'s per site;
- **additive reading of the tip** ("subtract z's AR vector, add z′'s"):
  - `D_ar`: per site, add the raw AR difference AR(z′) − AR(z), at the class-mean reference's swap-edit norm for that
    site (only the direction is used, so the AR's unexplained scale does not enter);
  - `D_ar_answer`: the same with AR(a′) − AR(a) (positive control);
  - `D_classmean`: the class-mean difference d(z′) − d(z) (the model's own colour difference; lever reference);
  - `D_rand_0/1`: random directions;
  - all at the same norm, set per layer as clean coordinate + k.

**Reading** (median S, table-cluster bootstrap 10k; paired differences per pair):
- `C_white`: as pre-set. It is read only if `Cctrl_answer` ≥ 0.15.
- `C_rawcoord`:
  - read only if `C_raw_answer` ≥ 0.15;
  - "colour-specific" only if the paired difference against the mean of the two `C_raw_rand` draws has a CI above 0.
- Additive:
  - not read if `D_classmean` < 0.15 (the additive lever is too weak at this norm).
  - Otherwise, D_ar − D_classmean with CI inside ±0.05: the AR's colour difference is as good a handle as the model's
    own.
  - D_ar − D_rand CI including 0: not a handle.
  - Anything else: partial; report the numbers.
  - `D_ar_answer` is reported as the AR's positive control.
- Every number from 011 is **post hoc, exploratory**.

## Execution status after amendment 3 (2026-10-06, committed 21:07:59 UTC per git)

- **Vectors v3** (`--posthoc-a2`; `results/011_olens_swaps/ar_validation_v3.json`): pre-set rule `accepted: false`;
  amendment 2 accepted. Scaled FVE on 16-token spans: raw 0.24, ZCA 0.07, PCA −0.02 (margin 0.17); within-batch
  retrieval@1 0.86. The 12 centred colour vectors are near-equiangular (mean off-diagonal cosine −0.09 ≈ −1/11; *any* 12 centred vectors of similar norm give this, so it is not evidence about the vectors, report 31).
- **Smoke s0** (2 pairs, before amendment 3) and **smoke s0b** (2 pairs, with amendment 3's arms;
  `olens_swaps_s0b.json`): all gates pass (G0 Δm = 0, R·A ≈ 5e-15, repro = 001, nothing quarantined). Medians in the
  "Where things stand" section of `CLAUDE.md`.
- **Full run s1 stopped by the PI's request at 74 of 82 pairs.** The smoke→full chain had started it before the
  request arrived. The pair files are kept; rerunning the same command finishes the remaining 8. Nothing from s1 has
  been analysed.

## Loader caveats (2026-10-06, report 31; PI)

**011 is not "the TA's reconstructor method" until the loader is validated.** Amendment 2 accepted our AR loader on
scale-free criteria only. Open against the published AR (numbers recomputed by report 31's check from
`outputs/011_olens_swaps/d2/diag_tensors.pt`; not in a committed JSON):
- **Unscaled FVE is negative** on 16-token spans (−0.29 / −0.11 / +0.06 at L40 / L44 / L48); the published 0.22 is
  unscaled, on spans of uniform 1–32 tokens, which we never ran.
- **Outputs are about 2× too long**: at L44, 16-token spans, norm ratio 1.20 at cosine 0.51, where an MSE-trained head
  would shrink its predictions. The global scale (0.42 / 0.46 / 0.55) is unexplained.
- **64 blocks vs a truncated backbone:** the official loader truncates the backbone to the deepest needed layer;
  `ar_forward` runs all 64 blocks. Untested.
- `ridge_c` 0.1 and the config's J-space loss term (`loss_mix_lambda` 0.25; see below) were not used as checks.
- Not caveats: the AR colour differences are colour-specific (cos 0.15–0.24 with the model's class-mean differences on
  the first 10 tables, ≈ 0 for mismatched colour pairs), so "mostly noise" (earlier text) is too strong.

## Full run s1 completed and analysed (2026-10-06; POST HOC, exploratory)

- **Resume** (PI go, report 31 item 5): the remaining 8 pairs ran with `--vec-run v3 --run-id s1` after the
  resume-bookkeeping fix (1.05 GPU-min). Bookkeeping now covers all pairs: 82/82 pairs,
  0 quarantined, 0 repro failures, R·A max 7.2e-15
  (74 pairs from the run log at 1 significant figure, 8 from the rows), G0 Δm = 0 on both checked pairs.
- **Analysis** (`olens_swaps_analysis.py --run-id s1`, amendment 3's rules; `results/011_olens_swaps/olens_swaps_analysis_s1.json`;
  median S [table-cluster bootstrap 95%], n = 82):

  | Condition | S |
  |---|---|
  | class-mean coordinate reference `Bref_full_3L` | 0.20 [0.11, 0.24] |
  | TA's whitened swap `C_white` | 0.004 [0.001, 0.009] |
  | its answer-colour control `Cctrl_answer` | 0.016 [0.007, 0.031] |
  | raw-coordinate swap `C_rawcoord` | 0.005 [0.000, 0.022] |
  | its answer control `C_raw_answer` | 0.050 [0.018, 0.071] |
  | additive AR difference `D_ar` | 0.09 [0.03, 0.14] |
  | AR answer difference `D_ar_answer` | 0.17 [0.09, 0.24] |
  | model's class-mean difference `D_classmean` | 0.20 [0.11, 0.24] |
  | random `D_rand_0` / `D_rand_1` | 0.007 / 0.007 |
  | D_ar − D_classmean (paired) | -0.10 [-0.15, -0.05] |
  | D_ar − D_rand (paired) | 0.06 [0.02, 0.11] |

- **Verdicts (amendment 3):** `C_white` and `C_rawcoord` not read (their answer-colour controls, 0.016 and
  0.050, are below the 0.15 bar). Additive: **partial**: the AR's colour difference beats random at matched
  norm but moves the answer about half as much as the model's own class-mean difference (CI of the difference below −0.05).
- The smoke's raw-coordinate 0.16 did not hold at n = 82 (0.005).
- Scope: brew dev, 82 gated pairs, 3 emit cells × L40–48, SEC; v3 vectors from single colour words, under a loader not yet
  validated against the published AR (above).

## AR J-bias check (2026-10-06; EXPLORATORY, CPU; `ar_jbias.py`, `results/011_olens_swaps/ar_jbias.json`)

- **What the config says, and what stays open.** `ar_ptag_pooled/meta.json` names the Neuronpedia J-lens
  (`jspace_repo`/`jspace_file`, layers 20–60, `jspace_mu_prefix whitening_iolens_pooled`), `loss_mix_lambda` 0.25 (empty
  in `config.extra`), and reports `val_jloss` / `val_jfve` (0.58 at L40, 0.59 at L44, vs `val_fve` 0.22). No training
  code is public or in any local clone, so **whether a J term entered the AR's loss, and what `jfve` measures, are open.**
  The Oracle's reward is a "joint bullet-FVE … on the ptag AR" (`run.json`): "joint" most plausibly means the bullets'
  reconstructions fitted jointly, not J-space; nothing in `run.json` mentions J. Also open.
- **Decision rule** (in the script, before computing): J-biased if (AR's J-share of centred variance) / (targets' J-share)
  ≥ 1.5 at L40 and L44 and above PCA-1,024 and random-1,024 controls; not J-biased if ≤ 1.2 at both; else ambiguous.
- **Result: ambiguous.** J ratio 1.27 (L40) / 1.15 (L44), above PCA (0.77 / 0.85) and random (≈ 1.00).
  Within-subspace scaled FVE: J 0.22 / 0.24, PCA 0.28 / 0.31, random 0.16 / 0.18
  (384 held-out assistant spans, odd half; PCA fit on the even half).
- **Reading:** the AR over-weights 007's J subspace modestly (1.15–1.27×) and reconstructs J directions better than
  random but worse than the top-variance directions. A bias this size does not by itself account for 007's Oracle result
  (J alone 5.8 vs whole 3.2 nats, rest ≈ 0.3), but it does not rule out a J term in the Oracle's training either. Our
  `val_jfve`-style number (J-subspace FVE ≈ 0.22) is far from the config's 0.58, so the config's `jfve` is a different
  quantity.
