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

## Amendment 2 (2026-10-06 ~20:50 UTC; POST HOC: written after v2, d2 and the offline sweep had been seen)

**What was seen.**
- v2 (amendment 1) failed the pre-set rule again. The scale-free diagnostics barely moved (ZCA cosine ≈ 0.2).
- d2 saved the head outputs and targets. The offline sweep (`ar_offline_diag.py`, `results/011_olens_swaps/ar_offline_d2.json`)
  scored a convention v1 and v2 never scored: **the head outputs a raw activation**, scored in the whitened metric
  W(ŷ − μ) vs W(h − μ). This fits the AR config (`loss_space: whiten`), with the head computed on the backbone's final
  pre-norm state passed through a parameter-free LayerNorm (`head_layer_norm: true`).
- On 16-token spans (training crops are uniform on 1–32 tokens; 40% of our spans are 1–2 tokens and carry little):
  - after one global least-squares scale, whitened FVE is 0.27 / 0.24 / 0.24 at L40 / L44 / L48 (published val FVE
    0.22 / 0.22 / 0.23);
  - retrieval@1 within random batches of 24 (all spans) is 0.83 / 0.85 / 0.90 (published 0.92 / 0.93 / 0.96);
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

## Amendment 3 (2026-10-06 ~21:05 UTC; POST HOC: written after the 2-pair smoke s0, before any other pair ran)

**Seen in s0** (2 pairs, one table; vectors run v3 under amendment 2):
- all gates pass;
- the reference moves S 0.23;
- `C_white` −0.01 / +0.00, its answer-colour control −0.01 / +0.00, with small edits (summed norm 15–21; the clean
  activations' whitened coordinates on the AR colour vectors barely differ);
- `C_rawcoord` +0.22 / +0.10, but with edits 2–3× the reference's and no matched random or answer control.

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

## Execution status after amendment 3 (2026-10-06 ~21:30 UTC)

- **Vectors v3** (`--posthoc-a2`; `results/011_olens_swaps/ar_validation_v3.json`): pre-set rule `accepted: false`;
  amendment 2 accepted. Scaled FVE on 16-token spans: raw 0.24, ZCA 0.07, PCA −0.02 (margin 0.17); within-batch
  retrieval@1 0.86. The 12 centred colour vectors are near-equiangular (mean off-diagonal cosine −0.09 ≈ −1/11).
- **Smoke s0** (2 pairs, before amendment 3) and **smoke s0b** (2 pairs, with amendment 3's arms;
  `olens_swaps_s0b.json`): all gates pass (G0 Δm = 0, R·A ≈ 5e-15, repro = 001, nothing quarantined). Medians in the
  "Where things stand" section of `CLAUDE.md`.
- **Full run s1 stopped by the PI's request at 74 of 82 pairs.** The smoke→full chain had started it before the
  request arrived. The pair files are kept; rerunning the same command finishes the remaining 8. Nothing from s1 has
  been analysed.
