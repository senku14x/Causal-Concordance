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
