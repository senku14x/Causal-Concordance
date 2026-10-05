# Cathryn's J basis: construction and geometry checks (5 October 2026, evening)

*Updates `claude/NEW_RESULTS_2026-10-05.md`:*
- *§3.1: the construction is now known;*
- *§5: the "how chosen / norm or energy" bullet is answered;*
- *§6: item 2 is done; item 4's controls are revised below.*

*All numbers come from `jbasis_checks.py` (advisor chat, CPU, no model forward passes), written out as full-precision JSON (`jbasis_checks.json`) and rounded half-up.*

## 1. Construction (from her push, `LittleMeHere/nla-vs-lenses` @ 9a46bf9, `jbasis/`)

- **Readout directions:** for every vocabulary token t, d_t = W_U[t] @ J, where J is the J_NP Jacobian at L42 (neuronpedia, n1000 wikitext).
- **Basis:** the eigenvectors of G = Σ_t unit(d_t) unit(d_t)ᵀ over the whole vocabulary (248,320 tokens), largest eigenvalue first. The file keeps the top 2,560 in float16. Her "J part" is the projection onto the top 1,024.
- **What was fitted on what:** one fixed basis, built from J_NP and the unembedding only. It was **not fitted on any multihop item**, so there is no selection on the evaluated items.
- **"16%" is an energy share** (squared norm), per her docstring. It reproduces on her saved multihop residuals: median 15%.
- **What the basis is:**
  - The top 1,024 directions hold 93% of the J-readout directions' variance. The same construction on the raw unembedding holds only 38–39% in 1,024 directions. So the Jacobian squeezes token readouts into a subspace of about 1,000 dimensions, and **the basis is determined mainly by J, not by W_U**.
  - Its overlap with the logit-lens subspace (the same construction with J = identity, using the final-norm gain 1 + w) is 0.30 (mean squared canonical correlation). A random pair of 1,024-dim subspaces gives 0.20.
  - The top eigenvector alone holds 32% of the readout variance. It is orthogonal to the mean activation (cos −0.01).
- **Naming:** call it the **J subspace**. It is not 005's 16-atom J part, and not the paper's J-space (a union of sparse cones).

## 2. Geometry checks on her saved residuals

**Data:** her 50 saved multihop residuals (`runs/abl50/resid_L42.pt`: the first 50 bank items, L42, the " is" token at position −2). Her split used 29 two-step items, probably a subset; their ids aren't in the repo.

**Measures:**
- **Item variance:** each activation minus the leave-one-out mean over items.
- **Item fraction:** the share of the piece's own energy that is item-specific. This is what the Oracle sees after it renormalises the piece.

All values are medians over items.

| Piece | Energy share | Item variance held | Item fraction of the piece | Oracle names bridge (Cathryn, 29 items) |
|---|---|---|---|---|
| Whole activation | 100% | 100% | 32% | 43% |
| J piece (top 1,024) | 15% | 32% | 62% | 47% |
| Rest of J | 84% | 68% | 25% | 17% |
| Random 1,024 (10 draws) | 20% | 20% | 31% | 14% |
| All but random | 80% | 80% | 31% | 43% |
| Logit-lens subspace (top 1,024) | 10% | 18% | 50% | not run |
| Rest of logit-lens subspace | 90% | 82% | 29% | not run |

**Context:**
- The cross-item mean is 69% of a multihop activation's energy. Only 8.5% of it lies in the J subspace (a random subspace takes 20%).
- On Brew ":" L42 (her 50 residuals), the mean is 97% of the energy. The J subspace holds 23% of the item variance there (random 20%).
- **Dimensionality:** PCA cross-fitted on 25 multihop items and scored on the other 25 needs 24 components to hold 30% of the held-out item variance (J 32%). So item variance at this position is spread over many dimensions.

**Reading (observations on 50 items; reader behaviour not re-measured):**
1. **The J subspace avoids the shared direction and is enriched for item-specific variance,** but only 1.6× over random: it holds 32% of the item variance vs 20%. After the Oracle renormalises, the J piece is 62% item-specific, vs 32% for the whole activation.
2. **Generic item variance does not explain the "rest" result.**
   - The rest of J still holds 68% of the item variance, at an item fraction (25%) close to the whole activation's (32%). Yet the Oracle names the bridge from it in 17%.
   - "All but random" holds 80% at 31% and is named in 43%.
   - A 12-point difference in item variance goes with a 26-point drop in naming.
   - So the content the Oracle uses is concentrated in the J subspace beyond its share of generic item variance.
   - Caveat: the reader is nonlinear, and a threshold could exaggerate small input differences. This argues against "any item-content subspace would do" without ruling it out.
3. **The J subspace is where the Jacobian sends token readouts, so it is a token-identity subspace.** Prompt tokens live there too: the rest of J rebuilds 12% of prompt words vs 51% for "all but random".
   - So J-dependence predicts that bridge naming and prompt rebuilding drop together, which is what she sees.
   - **The J-subspace result cannot separate reading the bridge from rebuilding the question.** Re-solving is now the binding alternative. Generic item variance is not.
4. **The logit-lens subspace is a useful control.** Like J, it avoids the shared mean (item fraction 50%), but it holds item variance only at random level (18%). It separates "J just avoids the shared directions" from "J holds the content".

## 3. Controls run (proposal: design spec first, launch on Vishesh's go)

**No replication of Cathryn's settings** (PI, 5 Oct). The run uses the PI's own prompts and our pipeline. Our Oracle is `olens_s3d_rl600`.
- **In place of a replication gate:** the whole activation must give a clearly positive bridge contrast (positive control).
- **Before choosing the read layer,** rebuild the L42 basis with our pinned J_NP and confirm its overlap with her file is ≈ 1.

**Conditions** (each piece and its complement, Oracle, ≥ 4 samples per item):

| Control | What it tests |
|---|---|
| J subspace, top 1,024 | the effect |
| Random 1,024 (≥ 10 draws) | same-size null |
| Logit-lens subspace, top 1,024 (W_U ⊙ (1 + w), same construction) | is J's advantage just avoiding the shared directions? |
| Content-matched PCA-k: PCA cross-fitted on held-out items' read-position activations, k chosen so held-out item variance ≈ the J subspace's | does any subspace holding the same item variance do it? |
| Dims-matched PCA-1,024 of centred L42 token activations on wikitext (J_NP's own fitting data) | is the J construction better than plain variance on the same data? |

**Scores:**
- primary: the Oracle's likelihood of the bridge word on each piece, as in 006 (every token, LSE over spellings, cross-item contrast). It also shows whether the bridge is in "the rest" but unnamed;
- secondary: bridge named (word match) and prompt words rebuilt.

**Predictions, fixed before any run (5 Oct, evening):**

| Hypothesis | J piece | Logit-lens piece | PCA-k piece | Rest of PCA-k |
|---|---|---|---|---|
| H_J: J's token-readout geometry carries what the Oracle uses | ≈ whole | ≈ random | clearly < J | ≫ rest of J |
| H_content: any subspace with the item variance | ≈ whole | ≈ random | ≈ J | ≈ rest of J |
| H_shared: J only avoids the shared directions | ≈ whole | ≈ J | — | — |

- **The advisor's prior from §2:** H_J over H_content. The rest-of-J vs all-but-random gap is the evidence. The prior is weak, because the reader is nonlinear.
- **Power:** 29 items × 4 samples resolves differences of about 20 points. Use all 100 multihop items before reading smaller ones.

**Not answered by this run:** reading vs re-solving (§2, reading 3). The decisive tests:
- reader-free removal of the bridge, confirmed gone by a probe, then re-read;
- items where an upstream patch makes the held bridge disagree with the visible prompt.

**NLA arm:** prediction already fixed. The Oracle's rest/whole ratio is 17/43 ≈ 0.40. The NLA reading non-workspace content predicts ≥ 0.6. A ratio near 0.4 means it reads the same subspace.
