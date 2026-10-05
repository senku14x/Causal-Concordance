# 007 — Do the readers' signals depend on the J subspace, on content share, or on removing the shared direction? (design v2, approved with changes; pre-registered before any 007 GPU run)

**Status.**
- **v1** drafted 2026-10-05.
- **v2:** the PI's approval with decisions, additions and fixes (2026-10-05), folded in below. §17 records each one.
- **Branch:** `007-jsubspace`, off `005-amended-full-run` @ 82137e0.
- **Committed together:** this spec, report 22, `scripts/007_jsubspace/geometry*.py` and their results, before any run.
- **Brew:** the dev split (48 tables) is the discovery set. The brew test split is used once, for the confirmatory
  replication (§7). The PI approved this under AMENDMENT_02 (§7.6).
- **Multihop** has its own dev/test split (§8).
- **Before any outcome:** the CPU geometry (report 22, exploratory) fixed the bases, the piece statistics and the
  Δ-matched PCA-k values. The three-model table (§2.2) was written after that geometry but before any reader read a
  piece.
- **Priors:** the agent priors in §14 are fixed. Rows added in v2 carry priors written at v2, still before any 007
  GPU run. `[PI]` fields are left blank (PI decision).
- **Execution is unattended under §12:** gates decide, nothing is repaired mid-run, and anything changed after
  outcomes is labelled post hoc.

## 0. Summary

- **Reader side (Stage 1).** The readers' likelihoods track z at the brew `:` cell (006: NLA L42 D_z 1.86, Oracle
  L40/L44 3.22/3.24).
  - Each activation is split into pieces and D is scored on each.
  - A 15-contrast primary family (P1–P5, §2.2) decides between three models:
    - the J subspace specifically (H_J);
    - raw content share;
    - visibility after the shared direction is removed.
  - The same rules are re-applied on the brew test split (§7). **Only effects that replicate there count as
    confirmed.**
- **Model side (Stages 2 and 2b).**
  - Stage 2 swaps the J subspace vs controls between member and partner. Does the answer move beyond what J's content
    share predicts?
  - Stage 2b is the paper's §3.3 privilege test on class-mean z directions.
  - Both are experiments about J geometry, never labels.
- **Secondary:**
  - Stage 1b (`</think>`, the unused cell);
  - multihop (Stage 3: relation-level pairs; expected to say little about J);
  - relational (Stages 4a profile, 4b Part A; exploratory);
  - chain (Stage 5; first to be dropped);
  - the template lens as a non-J linear reader on every piece (§3.7);
  - CPU analyses (§11).
- **Budget:** cap about 15 GPU-h.
  - Projected 15.0–16.1 GPU-h with everything (§13). **Expect Stage 5 to be dropped**; Stage 4b runs only if the
    projection still fits.
- **Run order:** 0 → 1a → 2 (+2b) → 1b → brew test → 3 → 4a → 4b → 5. The template lens and the CPU analyses run
  whenever their inputs exist.

## 1. Open points I found while folding in v2 (decided here as stated; the PI can override before the go)

1. **Brew test material.** Dev Stage 1 (like 006) uses **all** start candidates of the 48 dev tables (183 partners),
   not only the 82 that pass 001's gate. 001's gate has **already run on the test tables** (reader-free, behavioural):
   77 of 178 candidates pass, across 35 of 49 tables (`results/001_leverage_profile/analysis/gate_summary.json`), so
   no new gating forward is needed.
   - If test used only gated partners, the replication would change the population (only reliably solved pairs) as
     well as the tables, and lose about half the power.
   - **Chosen:**
     - **primary** = all 178 test candidates, matching dev's material rule;
     - **secondary** = gated pairs only, on test **and** dev, so the gate's effect can be seen.
   - This departs from the literal "gate the test-table partners" instruction. It is flagged here for the PI's
     confirmation.
2. **Test needs the rest-of-random pieces.** P3 is D(rest of random) − D(rest of J). Applying the rules unchanged on
   test therefore needs rest of random 0–2, which v2's test list omits. They are added (3 pieces).
   - On test, D(random) is the mean over 3 draws (10 on dev). That is noted, not changed.
3. **AMENDMENT_02 does not exist as a file yet.** Its planned content (freezing τ, ε′ and the substitution bound)
   governs test-split **labels**, and 007 computes no labels.
   - The PI's approval of this reader-side test use is recorded in §7.6. A new `docs/context/AMENDMENT_02.md`,
     committed with this spec, records it and states that τ, ε′ and the bound remain unfrozen.
   - **Consequence:** the brew test tables will have been read by the readers. Later reader evaluations there are not
     on untouched tables. Labels are unaffected, since labels never use readers.
4. **Template-lens scoring.**
   - The authors' README says rows are "scored by cosine of the per-layer residual against each template direction".
     The safetensors metadata says `space: raw` (also `ridge_c 0.001`, `global_n 1,610,682`).
   - Neither documents whether h is centred before the cosine.
   - **Chosen:** raw cosine, cos(h, T_w). Stage 0 looks for the authors' `global_workspace.template_lens` source. If
     it centres h, that convention becomes primary, decided in Stage 0 before any piece is scored, and recorded.
5. **Earlier flags still standing** (v1 §1):
   - multihop D is confounded by the cue (restatement), so Stage 3 cannot separate reading the bridge from re-solving;
   - no-activation and prompt-only cannot discriminate pieces under D;
   - relational reads at the answer position;
   - J is the token-readout subspace, so any H_J support includes output-facing content: 006b found z is the model's
     runner-up at `:`;
   - mean-removed is the most off-distribution piece.

## 2. Question, models and wording

### 2.1 Questions

- **Q-A (reader side):** at a cell, do the Oracle's and the NLA's likelihood-based z signals (006's D) depend on:
  - the J subspace specifically (H_J);
  - how much of the member–partner difference a piece holds (raw content);
  - how visible that difference becomes once shared energy is removed (visibility)?
- **Q-B (model side):** does the answer margin depend on the J subspace beyond the content share it carries (Stage
  2), and is the J part of a class-mean z direction privileged at matched magnitude (Stage 2b)?

### 2.2 The three-model table (primary decision rule, Stages 1a and the brew test)

Categories `+`, `−` and `≈` are defined in §4.4. The predictions use report 22 T4 at `:`:

| Contrast at `:` (target z) | Raw difference | Visible difference | J content (H_J) |
|---|---|---|---|
| **P1** D(J) − D(PCA-k) | ≈ (both hold 0.24–0.28 of Δ) | − (PCA-k visible 2.3–3.1× vs J 1.7–1.8×) | + |
| **P2** D(J) − D(random) | ≈ (0.24–0.28 vs 0.20; predicted gap below δ) | + (1.7–1.8× vs 1.0×) | + |
| **P3** D(rest of random) − D(rest of J) | ≈ (0.80 vs 0.72–0.76) | ≈ (1.00× vs 0.89–0.91×) | + |
| **P4** D(mean-removed) − D(whole) | ≈ (both hold 1.00) | + (4.8–5.8× vs 1×) | no prediction |
| **P5** D(J∘centred) − D(random∘centred) | ≈ | ≈ | + |

- **P4 under H_J:** the removed direction is largely the shared task readout, which J itself reads, so H_J makes no
  clean prediction.
- **P5 (added in v2 by the PI):** J vs random on the centred (mean-removed) activation. Both pieces are free of the
  shared direction, so visibility is equalised and only J's construction differs.
  - Stage 0 reports the Δ-share and visible ratio of the P5 pieces, for context only. The P5 predictions stand as
    given.
- **The patterns are mutually exclusive:**

  | Model | P1 | P2 | P3 | P4 | P5 |
  |---|---|---|---|---|---|
  | Raw | ≈ | ≈ | ≈ | ≈ | ≈ |
  | Visible | − | + | ≈ | + | ≈ |
  | H_J | + | + | + | · | + |

### 2.3 Wording

- Reader results: "decodable through the reader's likelihood (D)". Never "the reader represents z" and never "the
  reader reads the workspace".
- Model-side results: "demonstrated influence on this answer margin". Stages 2 and 2b are experiments about J
  geometry. Their results never become labels and never score readers.
- Labels never use readers (non-negotiable). No stage assigns labels.

## 3. Shared definitions

### 3.1 Pieces

**Construction.**
- **Notation:** h is a float32 capture. P_X = Q_X Q_Xᵀ for an orthonormal Q_X.
- **Precision (fix 11):** every basis is rebuilt in float32 and re-orthonormalised by QR. The orthonormality error
  max|QᵀQ − I| is logged per basis.
- **Precompute:** pieces and complements are computed and saved as float32 before any GPU work. G-recon (§4.6) is
  unchanged.
- **Layer matching:** each reader reads its own layer, with bases for that layer:
  - Oracle L40 reads L40 captures with the L40 bases;
  - Oracle L44 reads L44 captures with the L44 bases;
  - the NLA reads L42 captures with the L42 bases.

| Piece | Definition |
|---|---|
| whole | h |
| J | P_J h. Q_J = top-1,024 eigenvectors of Σ_t unit(W_U[t] J_ℓ); raw W_U (Cathryn's convention; reproduced exactly at L42, report 22); J_NP @0731326e |
| rest of J | h − P_J h |
| random r (r = 0…9) | P_R h, Q_R = QR of a 5120 × 1024 Gaussian, `geometry.rand_basis(20261005 + r)` |
| rest of random r (r = 0, 1, 2) | h − P_R h |
| PCA-k | 004-style within-table PCA (rows centred within table, scaled by 1/√n_t). Cross-fitted over `geometry.table_folds(48, 4, seed 20261005)`. k frozen at the fine-k values: `:` L40/42/44 = 4/4/4; `</think>` L40/42/44 = 4/3/4. Chosen outcome-blind, post hoc (report 22 §7) |
| rest of PCA-k | h − P_PCA h |
| LL_gain | top-1,024 eigenvectors of Σ_t unit(W_U[t] ⊙ g), g = 1 + `model.language_model.norm.weight` (J = I; the advisor's convention) |
| rest of LL_gain | h − P_LL h |
| mean-removed (template mean) | c = h − (h·u)u, u = unit(mean of that cell and layer's captures over the renders of the other 47 tables) |
| **J∘centred** (v2, P5) | P_J c |
| **random r∘centred** (v2, P5; r = 0, 1, 2) | P_R c |
| **generic-mean-removed** (v2, secondary) | h − (h·u_gen)u_gen, u_gen = unit(mean of L-ℓ activations over wikitext tokens; §4.7). The same operator as mean-removed; only the mean differs |

- **Count:** 26 pieces per reader-layer and cell in Stages 1a and 1b.
- **Not run:** J_gain (overlap 0.98 with J_raw) and LL_raw (it captures the shared direction, report 22).

### 3.2 Readers and scoring (006 conventions, unchanged)

- **Oracle `olens_s3d_rl600`.** It replaces the carrier slot embedding with 16000 · unit(piece). L40 and L44 are
  on-contract. L42 is off-contract and is not used.
- **NLA `nla-rl` iter_000400 (`readers.NLArl`).** A separate av_base model with the adapter attached, never merged.
  The Karvonen add happens at block 1. It reads at L42.
- **Prefixes:** the 006 primary prefix only: Oracle `-`; NLA `<explanation>\nThe text is about`.
- **Score:** LL_x(c) = log-sum-exp over the spellings of c of the full-sequence log-probability, every token scored
  (`score_seq`), from float32 logits on the final-norm output.
- **Batching:** `acts_per_batch = 2`, as in 006.
- **Brew spellings:** " c" and " C" for the 12 colours (24 rows; " Teal" is two tokens, both scored).

### 3.3 The D statistic and role specificity

- **Brew, per table and role t:**
  - D_t = [LL_m(t_m) − LL_m(t_p)] − [LL_p(t_m) − LL_p(t_p)], averaged over the table's partners;
  - the mean over tables, with a 95% table-cluster bootstrap CI (10,000 draws, seed 20261004) and sign counts;
  - two-sided bootstrap p = 2 · min(share ≤ 0, share ≥ 0), floored at 1e-4;
  - contrasts between pieces are formed per table, then bootstrapped (paired).
- **Roles, reported side by side on every piece (v2):**

  | Role | Pairs used |
  |---|---|
  | z | all |
  | start | collision-free (006 census) |
  | answer | collision-free (006 census) |
  | **off-trajectory** | see below |

- **Off-trajectory D.** Per pair, two distinct colours o₁ and o₂, each in none of {start, z, answer} of either member.
  - Taken in order from the item's WSB off-path options (`t["candidates"]`, as in 002's `table_candidates`), then
    filled by a seeded draw from the rest of the palette.
  - D_off = [LL_m(o₁) − LL_m(o₂)] − [LL_p(o₁) − LL_p(o₂)], with the o₁/o₂ assignment seeded (20261005). It is expected
    to be about 0.
  - Pairs without two such colours are dropped and counted.
- **Reading rule, stated before outcomes and applied to every piece, explicitly to both centring arms:**
  - A piece's role D values are read only if its D_off CI includes 0 or |D_off| < δ/2. Otherwise the piece is flagged
    as carrying a generic, role-independent bias.
  - A piece is flagged **role-nonspecific** if D_z, D_start and D_answer all have CIs above 0 and lie within δ of
    each other. Such a piece's D_z is read as evidence about how this prompt differs from average (start sits in the
    prompt), **not about z**.
  - The flags are reported next to the P1–P5 categories. They do not change the categories; they qualify what a
    category means.

### 3.4 Colour-prior report (v2)

For every piece × reader-layer × cell:
- the distribution over renders of each render's top colour (argmax LL);
- the mean LL per colour;
- the total-variation distance of the top-colour distribution from the whole activation's.

This makes wholesale prior shifts visible, such as 006's Oracle "blue" prior.

### 3.5 Baselines, for every family and reader

| Baseline | Definition | Role |
|---|---|---|
| No-activation | Reader with no injection. Oracle: carrier slot left as its ideograph embedding (new `None` row in `score_seq`). NLA: an all-zero row (`karvonen_add_` skips it). One read per reader-layer and cell | D ≡ 0 by construction. Used for the reader prior (positive control) and the engagement check |
| Shuffled activation | 200 table derangements (006 post hoc), CPU from the saved LL matrices | Per-piece null for every role's D: mean and [2.5, 97.5] percentiles |
| Prompt-only, reader floor | Stock Qwen3.6-27B. User turn = WSB's `prompt_only_summary` template (`evals/baselines/prompt_only.json`) with the render up to and including the read token. Assistant turn (thinking off) teacher-forced with `The model is thinking about`, then the candidate. Full-sequence LL, then D | One row per cell and family, piece-independent |
| Prompt-only, task | The model's own colour log-probs at the cell (006b model-output D; reused for brew dev; measured for the brew test and the other families) | As above, labelled separately (V2-E5) |

### 3.6 Per-piece positive control and engagement

- **Activations:** 006's natural colour activations ("Tell me about {colour}", stock forward, last token;
  `outputs/006_reader_likelihood/run1/positive_acts.pt`, [12 colours, layers 40/42/44/56/60, 5120]).
- **Splitting** uses the same operators:
  - fixed projectors for J, random, LL and PCA-k;
  - template-mean centring and its P5 compositions use the leave-one-out mean of the other 11 colour activations;
  - generic-mean-removed uses u_gen.
- **Statistics:**
  - **LOO-prior ID:** argmax over c of LL_i(c) − mean_{j≠i} LL_j(c). Chance is 1/12.
  - **Pairwise double difference:** positive in how many of 66 pairs.
- **Pass rule (per piece and reader-layer):** LOO-prior ID ≥ 0.5 **and** ≥ 50/66 positive.
- **A failing piece makes every contrast involving it uninformative.** The failure is reported as a result.
- **Engagement (descriptive):** per piece, the median over renders of max_c |LL_piece(c) − LL_none(c)|. Below 0.1
  nats is flagged "reader effectively ignores this piece".

### 3.7 Template lens (v2; secondary reader, outside the Holm family)

- **Artifact:** `camilablank/workspace-lenses` @ d740106d, `qwen3.6-27b/template-lens/`. Pinned in
  `scripts/common/registry.json` by sha256:

  | Variant | File | sha256 |
  |---|---|---|
  | primary | `templates.safetensors` (8,633,819,288 bytes) | 2368c2ed10537b1e2b07af72c7973b93807484abfcf193c150c83206ce93ea98 |
  | primary | `template_words.txt` | 5b247e71def1a12689cf6f92f5838b24ae39f2275c66992ffe028c51be63508d |
  | sensitivity | `templates+phrases_v3.safetensors` (8,998,859,320 bytes) | c8529864c78b081867450bbad274bc4f2863f22fa8d196ae23bcb8962158cf91 |
  | sensitivity | `template_words+phrases_v3.txt` | 77e81a0374ee20916d65f467b4db186d0026cd681184c91a6b6679f04cd3dc12 |

  The safetensors hashes are the HF LFS sha256. The `.txt` hashes were computed on the files downloaded on
  2026-10-05.
- **Format:** `templates` [64, 13,174 (or 13,731), 5120] bf16, plus `word_ids` and `layers`; metadata `space: raw`.
  Only the needed layer slices are loaded on CPU (`safe_open(...).get_slice`), cast to float32.
- **Score:** cos(h, T_w) (§1 point 4). It is linear up to the per-render ‖h‖.
  - Scored per piece with the same D (on cosine scores), the same derangement null and 006-style closed-set ranks.
- **Vocabulary coverage:**

  | Material | Coverage | Handling |
  |---|---|---|
  | brew colours | 11 of 12 (no `teal`) | A pair is dropped from a role's D when that role's colour is teal on either side; counts reported |
  | multihop bridges | 57 of 75 | case-insensitive exact match; pairs with an out-of-vocabulary string are dropped for this reader and counted |
  | multihop targets | 59 of 82 | as above |
  | relational words | 15 of 16 (no `plumber`) | as above |
  | chain digits | 0 | scored through their number words (one…nine, zero if present), labelled |

- **Sanity check before use, per layer:** on the natural colour activations, each of the 11 colours ranks its own
  template first among the 11 colour templates for ≥ 9 of 11. A failing layer makes the lens uninformative at that
  layer.
- **Where it is scored:** every Stage 1a/1b piece, the brew test pieces, the multihop and relational pieces, and whole
  vs mean-removed in the CPU token-lens analysis (§11).
- **Question:** it is the non-J linear comparator. Does it also lose z when J is removed (whole vs rest of J), and
  where does it fall on P1–P5? Descriptive.

## 4. Stage 1a — Brew Part A at `:` (primary)

### 4.1 Material

- **Captures:** the 001 brew captures `outputs/001_leverage_profile/run/captures_brew.npz`, SEC regime, as in 006.
- **Renders:** 231 (48 members and all 183 candidate partners of the 48 dev tables).
- **Cell and layers:** `emit_colon`, L40/L42/L44, one per reader.

### 4.2 Reads

- **Per reader-layer:** 26 pieces × 231 renders, plus one no-activation read.
- **Positive control:** 12 colour activations × 26 pieces, per reader-layer (once; it does not depend on the cell).
- **Prompt-only:** 231 renders.
- **Generic-mean capture:** wikitext (§4.7).
- **Qualitative samples:** §4.8.

### 4.3 Outputs per piece × reader-layer

- D_z, D_start, D_answer and D_off with CIs, sign counts and derangement nulls;
- the role-specificity flags;
- the colour-prior report;
- positive control status and engagement;
- Δ-share and visible ratio;
- template-lens D and ranks.

### 4.4 Primary family and decision rule

**The family.**
- P1–P5 for each of the three reader-layers: Oracle L40, Oracle L44, NLA L42.
- That is **15 contrasts**.
- D(random) is the per-table mean over draws 0–9. D(rest of random) and D(random∘centred) are means over draws 0–2.

**Margin.** δ = **0.5 nats** (absolute):
- about 1.3× the half-width of 006's NLA derangement band (±0.37);
- 27% of the NLA's whole-activation D_z, and 16% of the Oracle's.

**Categories per contrast.**

| Category | Rule |
|---|---|
| `+` | Holm rejects across the 15 two-sided difference tests (α = 0.05) and estimate ≥ +δ |
| `−` | Holm rejects and estimate ≤ −δ |
| `≈` | TOST equivalence within ±δ: both one-sided bootstrap p < 0.05 after Holm across the 15 equivalence tests |
| unresolved | anything else |
| uninformative | either piece fails its positive control |

**Model mapping, per reader-layer.**
- A model is **supported** when every predicted entry (§2.2) is observed and none is uninformative. "No prediction"
  entries are ignored.
- A model is **contradicted** when any predicted entry is observed in another category.
- Otherwise the reading is **unresolved**.
- Holm is applied only within this family.
- **Confirmation:** a category is **confirmed** only if the brew test (§7) gives the same category for the same
  contrast and reader-layer.

**Secondary (descriptive, no multiplicity correction):**
- **S1, content curve:** piecewise-linear D(Δ-share) through (0, 0), (0.20, D(random)), (0.80, D(rest of random)) and
  (1, D(whole)). Each piece's excess over it is reported.
- **S2:** Spearman correlation of piece-level D_z with the visible ratio, and separately with the Δ-share.
- **S3:** role specificity (§3.3) and the colour prior (§3.4), for every piece.
- **S4:** the spread between random draws, and J against the draws' range.
- **S6, generic vs template centring:** D(generic-mean-removed) vs D(mean-removed) vs D(whole) for each role, with
  the role-specificity reading applied to both arms.
- **S7:** the template lens on every piece.

### 4.5 Run order inside the stage (decisive pieces first)

- **Phase A:**
  1. load the subject;
  2. capture wikitext;
  3. prompt-only;
  4. load the Oracle;
  5. G-score1/2;
  6. Oracle L40 whole (G-repro), then G-inject with the scale check;
  7. positive control for all pieces;
  8. pieces in the order J, rest of J, PCA-k, J∘centred, random 0–2∘centred, mean-removed, rest of PCA-k,
     generic-mean-removed, LL_gain, rest of LL_gain, random 0–9, rest of random 0–2;
  9. Oracle L44, same order;
  10. qualitative samples (Oracle);
  11. teardown.
- **Phase B:** the NLA L42 in the same order, then its qualitative samples.

### 4.6 Gates

| Gate | Rule (stop on failure) |
|---|---|
| G-bases (CPU) | Every basis float32 with max\|QᵀQ − I\| ≤ 1e-5 after QR; error logged |
| G-geom (CPU) | Each v1 piece's Δ-share and visible ratio reproduce report 22 T4 within 0.005 |
| G-recon (CPU) | ‖piece + complement − h‖ ≤ 1e-6‖h‖ (computed in float64 from the float32 pieces) for every pair; \|⟨mean-removed, u⟩\| and \|⟨generic-mean-removed, u_gen⟩\| ≤ 1e-6‖h‖ |
| G-score1/2 | 006 v1.1: `score_seq` vs `score_next` median ≤ 0.05 and max ≤ 0.25 nats; own-logits max ≤ 1.0 |
| G-repro | Whole-piece LLs vs 006's `ll_<reader>_L<ℓ>_p0.npz`: median \|Δ\| ≤ 0.05 and max ≤ 0.25 nats; whole D_z within 0.1 of 006 |
| G-inject | Oracle slot norm = α for every row. NLA `last_write`: rel ∈ [0.99, 1.01], cos ≥ 0.99. The no-activation row writes nothing. **Scale check (fix 10):** 0.5× and 2× copies of whole, J and mean-removed on 8 seeded renders give LLs matching 1× (median ≤ 0.05, max ≤ 0.25 nats) |
| G-mem | `check_gpu_room` passes before the NLA loads |
| Positive control | Per piece (§3.6). Not a stop gate: it decides informativeness |

### 4.7 Generic mean (v2)

- **Text:** about 200 sequences from wikitext-103-raw-v1, test split: the first 200 paragraphs of ≥ 128 tokens,
  truncated to 128 tokens, plain text. Never bank items.
- **Capture:** stock forward in Phase A, at L40/L42/L44.
- **u_gen** = unit(mean over all positions ≥ 16 of every sequence). The first 16 positions are excluded because
  early-position activations carry outlier norms.
- **Logged:** cos(u_gen, u_template) per layer.
- **Literature precedent for centring** (asked for by the PI):
  - **No precedent found for centring a verbalizer's input.** Activation Oracles / LatentQA and the NLA inject raw,
    norm-matched activations ([Activation Oracles, arXiv 2512.15674](https://arxiv.org/pdf/2512.15674)).
  - **Removing a shared mean direction has precedent elsewhere:**
    - steering vectors ([Jorgensen et al. 2023, mean-centring](https://arxiv.org/pdf/2312.03813): activations "are
      not centred around the origin, but … offset in some consistent direction");
    - word embeddings ([Mu & Viswanath 2018, All-but-the-Top](https://arxiv.org/abs/1702.01417): remove the common
      mean and top directions);
    - similarity analysis (Timkey & van Schijndel 2021, rogue dimensions);
    - SAE pre-encoder bias subtraction (Bricken et al. 2023).
  - So centred inputs are a novel, off-distribution manipulation for these readers. That is why every centred piece
    carries the per-piece positive control.

### 4.8 Qualitative samples (v2)

- **Tables:** 8 of the 48, drawn with `numpy.random.default_rng(20261005)`. Per table, the member and one seeded
  partner.
- **Pieces:** {whole, J, rest of J, PCA-k, mean-removed, generic-mean-removed, random 0}.
- **Readers:** 3 reader-layers, k = 2, 006 sampling settings (`SAMPLING_WSB`: T 1.0, top-p 0.95, top-k 64, 256 new
  tokens), fixed seed.
- **Total:** 672 readouts.
- **Saved** as JSONL beside the LL matrices, with a markdown digest.

## 5. Stage 2 — Brew Part B (model side; J geometry, never a label)

### 5.1 Setup

- **Pairs:** 004's 33 frozen pairs (`results/004_span_assay/manifest_pairs.json`).
- **Cell and layers:** the final `:` at L40, L42 and L44.
- **Regime:** SEC (bf16 blocks 0–39, fp32 from block 40). Writes are block outputs ≥ 40, inside the fp32 region.
- **Code:** 004's `make_delta`, `PairRunner`, fidelity enforcement and damage readouts.
- **Swap:** h_m ← h_m + P(h_p − h_m).

### 5.2 Conditions per layer

| Condition | Count |
|---|---|
| whole cell (S_full) | 1 |
| J(ℓ) | 1 |
| PCA-k (k = 4; Stage 1's table-held-out basis) | 1 |
| **PCA-k at k ∈ {1, 2, 8, 16}** (v2, regression points) | 4 |
| random 1,024 × 10 (Stage 1 seeds) | 10 |
| random Δ-matched (dimension 1,418 / 1,245 / 1,429) × 10 | 10 |
| **random 512 × 3 and random 2,560 × 3** (v2; seeds 20261005 + 100 + r) | 6 |
| rest of J | 1 |
| **Total** | **34** |

### 5.3 Score and decision rule

- **Score:** m = LSE log P(a′ spellings) − LSE log P(a spellings), full-sequence over 24 spellings. S = Δm / G.
  Recovery = S / S_full(ℓ) on the recovery subset (S_full ≥ 0.10).
- **Δ-share** of each edit per pair, from the run's own donor capture.
- **Primary (per layer; no Holm):** median over pairs of R_J − R_PCA-k(k=4) and of R_J − mean R_randΔ, each with a 95%
  table-bootstrap CI (10,000 draws, seed 20261004). Margin 0.10:

  | Reading | Rule |
  |---|---|
  | **J beyond content share** | both ≥ +0.10 with CI above 0 |
  | **Tracks Δ-share** | both 90% CIs inside ±0.10 |
  | **J below** | both ≤ −0.10 with CI below 0 |
  | mixed | otherwise |

- **Prediction to beat (004):** recovery tracks Δ-share, not construction. In 004, the z span (Δ-share 0.35) recovered
  0.21, PCA-11 (0.59) 0.58 and the joint span (0.66) 0.72.
- **Secondary (v2, regression):**
  - Per layer, fit an isotonic (monotone) regression of per-pair recovery on per-pair Δ-share across all non-J
    conditions.
  - J's residual = median over pairs of R_J − f(Δ_J), with a 95% table-bootstrap CI that refits f on each draw.
  - Also: F_target; the interaction I = S(J) + S(rest of J) − S_full; S_full against 001's medians (L40 0.35, L42
    0.48, L44 0.50).

### 5.4 Gates (004's)

| Gate | Rule |
|---|---|
| Smoke | 2 pairs, all gates, before the 33 |
| G0 | A zero edit reproduces the clean margin within 2e-4 nats |
| G1 | Every write: ρ ∈ [0.9, 1.1], κ ≥ 0.99. Enforced; failing rows quarantined |
| G-repro | Whole-cell S at L42 on the 33 pairs vs 004: max \|ΔS\| ≤ 0.01 |
| Damage | Rows flagged when outside-spelling mass rises > 0.05, or both endpoints fall > 0.05 nats. Flagged rows stay in every median, with counts. Reported, not a gate |

## 6. Stage 2b — The paper's §3.3 privilege test, model side (v2)

### 6.1 Directions (reader-free)

- **Class-mean z directions** at `:` L40/42/44 from the dev SEC captures:
  - d_c = (mean of h over renders with z = c) − (grand mean);
  - both means leave out the evaluated pair's table.
- The labels z come from the bank, not from readers.

### 6.2 Split

- **J part:** nonnegative gradient pursuit with an NNLS refit (`scripts/common/jspace.py` `decompose`), **k = 25**
  (the paper's intermediate-reasoning test; its report test used k = 16).
- **Atoms:** folded atoms v_t = unit(J_ℓᵀ(g ⊙ W_U[t])), primary gain convention (`jspace.atoms`).
- **Lenses:** run separately with **J_NP** and **J_CB**. The remainder is d − d_J.
- **Reported** per colour, layer and lens:
  - the J part's energy share ‖d_J‖² / ‖d‖² (the paper reports about 10–15% for its probes);
  - whether the colour's own token atom (" c" or " C") is in the support (005 A3 found the own atom in the J part
    for only 1/45 concepts).
- **Compute:** CPU in Stage 0 (float32, one (layer, lens) dictionary at a time).
- `jspace` is reader-side code. `swap.py` is not label code, so `tests/test_import_boundary.py` still holds.

### 6.3 Swap at matched magnitude (the paper's protocol)

- For pair (m, p) and component X ∈ {full, J, non-J}: Δh_X = s_X (v_p^X − v_m^X), where v^X is the X-part of
  d_{z_m} or d_{z_p}.
- **Scaling:** s_full = 1 and s_X = ‖v_p − v_m‖ / ‖v_p^X − v_m^X‖, so every edit has the full direction's
  magnitude. This removes the member's z component and adds the partner's.
- **Applied** at the member's `:` residual, one layer at a time.
- **Conditions:**
  - full (lens-independent);
  - J and non-J for each of the 2 lenses;
  - **4 norm-matched random directions** (004's control; added by me, cheap).
- **Score:** answer margin S as in Stage 2, plus the share of pairs whose top-1 answer flips to a′ (the paper's
  metric). 004's gates and damage readouts apply.

### 6.4 Power check first (on Stage 2's 2 smoke pairs × 3 layers)

- Proceed to the split only if the full-direction swap's median S ≥ 0.02, which is 4× 004's norm-matched random
  control median of ≤ 0.005.
- Otherwise report that and skip the J and non-J conditions.

### 6.5 Reading (descriptive; no Holm)

| Reading | Rule |
|---|---|
| **Privilege replicated** for a lens | median S(J) − S(non-J) > 0 with CI above 0, and S(J) ≥ 0.5 · S(full) |
| **Not replicated** | S(non-J) ≥ S(J) |
| mixed | otherwise |

## 7. Brew test replication (confirmatory; after Stage 1b, before Stage 3)

### 7.1 Material

- **Tables:** the 49 brew test tables (001's `brew_split`).
- **Primary:** all 178 start candidates (§1 point 1).
- **Secondary:** 001's gate result is already computed (77/178 pass, 35 tables), so the gated subset runs on test and
  dev.

### 7.2 Captures

- `:` at L40/42/44, SEC, with 001's `capture_brew` path, plus the cell-region assertion.

### 7.3 Pieces (15)

- whole, J, rest of J;
- PCA-k (basis refitted on all 48 dev tables, k frozen at the dev values 4/4/4, applied unchanged to test);
- mean-removed (leave-one-test-table-out mean; the same operator);
- random 0–2 and rest of random 0–2 (added, §1 point 2);
- J∘centred and random 0–2∘centred.

### 7.4 Readers, controls and template lens

- Oracle L40/L44 and NLA L42, plus the no-activation and prompt-only baselines.
- Positive controls with the test-stage operators.
- The template lens on every piece.

### 7.5 Rules

- §4.4 unchanged: the 15 contrasts, δ, Holm and TOST, the categories and the mapping.
- **Confirmed** = the same category on dev and on test.

### 7.6 Test-split use (AMENDMENT_02)

- **PI, 2026-10-05:** "I approve this use of the test split under AMENDMENT_02; record that."
- **Recorded here** and in `docs/context/AMENDMENT_02.md` (new; it states that τ, ε′ and the substitution bound are
  not frozen by this, and that 007 computes no labels).
- **This is the only planned use of the brew test split in 007.**

## 8. Stage 3 — Multihop Part A (secondary; expected to say little about J)

### 8.1 Material and split

- **Bank:** WSB `evals/multihop/items.json` (100 items; bank gate accepted as is).
- **Render:** plain text, no chat template.
- **Read position:** the bank's `readout.offsets` (−1 for 67 items, −2 for 33).
- **Split:**
  - **dev = items 1–50** (touched);
  - **test = items 51–100.**
  - Pairs never cross splits.
  - **Test is the reported number; dev is labelled touched.**

### 8.2 Pairs

- **Relation-level key:** the item name's final field. Pairs lie within key and split, with distinct bridges and
  targets:

  | Split | Pairs | Items | Key groups |
  |---|---|---|---|
  | dev | 28 | 27 | 9 |
  | test | 104 | 40 | 9 |

### 8.3 Captures and pieces

- **Captures:** stock forward at L40/42/44.
- **G-cw** against Cathryn's `runs/abl50/resid_L42.pt` where ids and positions coincide: cos ≥ 0.999, with mismatches
  counted.
- **Pieces:** the 21 v1 pieces:
  - PCA-k fitted on dev, centred within key, cross-fitted over key folds on dev, k Δ-matched and frozen on CPU;
  - mean-removed with the leave-one-key-out mean.

### 8.4 Targets and statistics

- **Targets:** the bridge (primary) and the target (secondary).
- **Spellings:** {" " + s, " " + swapcase(s[0]) + s[1:]}, deduplicated.
- **Candidates per read:** the key group's bridges and targets within the split (mean about 19 rows).
- **D:** item-level D (averaged over partners), then the mean over items.
  - Inference: a key-cluster bootstrap (flagged: 9 clusters), an exact sign-flip test over the 9 key groups (512
    permutations) and sign counts.
- **Derangements:** within key group, 200.

### 8.5 Positive control and restatement flag

- **Positive control:** "Tell me about {bridge}" activations for about 60 bridges, 10 piece types.
  - Pass: LOO-prior ID ≥ 0.5 and ≥ 70% of within-group double differences positive.
- **Restatement flag:**
  - **Sampling:** WSB settings, k = 2.
  - **Coverage:** the 67 paired items × {whole, J, rest of J, random 0, PCA-k, mean-removed} × 3 reader-layers.
  - **Flags** (deterministic word-boundary match; no judge):
    - `names_bridge`;
    - `restates_cue`: the capitalised words and numerals of the prompt other than Fact / The / In and the bridge or
      target;
    - `names_target`.
  - **Report:** P(names_bridge), P(names_bridge ∧ ¬restates_cue) and P(restates_cue). Items with no cue are reported
    as undeterminable.

### 8.6 Other readouts

- The template lens on every piece (in-vocabulary pairs only).
- **Reported beside CW-12:**
  - her numbers: 30 items, L42, word match;
  - her Oracle reads were at L42, off-contract;
  - her PCA-1,024 was fitted on 16k benchmark activations;
  - her matcher was not re-derived.

### 8.7 Gates

- G-cap, G-split, G-cw (report), and G-score1/2, G-inject (with the scale check) and G-mem as in Stage 1a.

## 9. Stage 4 — Relational multihop (secondary; exploratory; whole bank)

### 9.1 Material and gate

- **Bank:** WSB `evals/relational_multihop/items.json` (50 scenes × a/b). Bare render.
- **Gate (both members):** greedy correct and ≥ 9/10 at T = 0.8. Correct means the completion, stripped and
  lowercased, begins with `gold` or `gold_alt`. At most 8 new tokens.
- **Alignment:** equal token counts, with the renders differing only at the two relation-word positions.
- If fewer than 10 scenes survive, Stage 4 stops after the gate.

### 9.2 Stage 4a, profile (reader-free; 001-style)

- **Patch:** whole cell, partner → member, with the reverse direction on a fixed 30% of pairs.
- **Positions:** 5 positions shared by a and b:
  - the final `'s` (the bank's read cell);
  - " Avery" (the possessor);
  - " is";
  - the cloze subject name;
  - the period ending sentence 2.
- **Layers:** 40, 42, …, 60.
- **Score:** m = full-sequence LSE log P(" " + gold of the partner) − log P(" " + gold of the member). S = Δm / G.
- **Controls:** 001's noise floor and off-pair damage flag.
- **Used and unused positions** (at the reader layers):

  | Kind | Rule |
  |---|---|
  | used | median S ≥ 0.25 at L42 and ≥ 0.15 at L40 and L44, not damage-dominated |
  | unused | median S ≤ 0.05 at L40, L42 and L44, with CI upper ≤ 0.10 |

  - Ties go to the used position with the highest L42 S, and the unused position nearest to it.
  - If either kind is missing, stop after 4a.

### 9.3 Stage 4b, Part A at the two positions

- **Readers:** Oracle L40/L44 and NLA L42, with the 21 v1 pieces:
  - within-scene PCA-k;
  - leave-one-scene-out mean;
  - a pooled relation-word candidate set (one spelling).
- **D:** on X, the item's outer relation word. Flagged as answer readout (X is the next token).
- **Also:** the template lens, and a positive control on "Tell me about {relation word}".
- **Comparison:** the J-split pattern at the used vs the unused position.

## 10. Stage 5 — Chain (exploratory; dropped first if over budget)

- **Pairs:** 001's chain start pairs (`outputs/001_leverage_profile/run/pairs_chain.jsonl`, dev), restricted to pairs
  whose intermediates and answers are all single digits. Counted in Stage 0; **dropped if under 10 pairs.**
- **Cell and reader:** 001's `final` cell at L56 (001 S = 0.66; ≤ 0.0013 at L40–44). Oracle L56 only, on-contract.
- **Pieces:** the 21 v1 pieces, with the J_raw_L56 basis built in Stage 0.
- **D:** on the first intermediate, with spellings {" d", "d"}.
- **Also:** the template lens (via number words), and a positive control on "Tell me about the number {d}".

## 11. CPU-only analyses (run whenever their inputs exist)

1. **Oracle–NLA per-pair agreement by piece** (Stage 1a and test LL matrices).
   - Per piece: the same-sign share and Pearson r of pair-level D_z, Oracle L40/L44 vs NLA L42, with table-cluster
     bootstrap CIs.
   - Question: does agreement live in J? 006's whole-activation r is 0.24–0.30.
2. **Mean-removed token-lens readout** on all 48 brew dev tables.
   - **Lenses:** 002's five (logit; J_NP standard and cosine; J_CB; R_CB), **plus the template lens (v2)**, at
     L40/42/44 and the three cells, on whole and mean-removed.
   - **Single-position convention:** as in 002 (" Teal" skipped for the token lenses; teal absent from the template
     lens).
   - **Statistics:** 002's closed-set reciprocal rank of z; the share of member renders where z ranks first of 12 (11
     for the template lens); within-table tracking.
   - **Gate G-002:** the whole-activation token-lens scores reproduce 002's `colour_scores.json` within 1e-4
     relative.
   - **Prediction stated now:** these readouts are linear in h up to a per-render scale, so a rank-1 removal changes
     tracking mostly through scale. Expect sign counts within ±3 tables of whole; the gain shows in ranks.
3. **Stage 2b's pursuit summaries** (energy shares, own-atom presence). Computed in Stage 0.

## 12. Run plan, memory, unattended gating, persistence

### 12.1 Memory

- **Device:** RTX PRO 6000 Blackwell, 97,887 MiB (≈ 102.6 GB).
- **Footprints:** subject Qwen3.6-27B bf16, 50.1 GB; NLA av_base, 53.8 GB plus the adapter. They cannot co-reside.
- **Every reader stage runs Phase A, then teardown, then Phase B.**
  - **Phase A (subject resident):**
    - all subject forwards: captures, positive-control activations, the wikitext mean, prompt-only and gating;
    - then the Oracle (`PeftModel` on the resident subject).
  - **Teardown:** `OracleLens.unload()`, `del`, `gc.collect()`, `torch.cuda.empty_cache()`, then
    `check_gpu_room(53.8 GB + 4 GiB)`.
  - **Phase B:** `NLArl()`, the NLA reads, then `close()`.
- **Pieces** are precomputed on CPU, so Phase B never needs the subject.
- **SEC model:** Stages 2, 2b, the test captures and 4a use the SEC-built model, which cannot host the Oracle. They
  run in their own invocations.
- **Template lens:** CPU only. One layer slice is 135 MB in bf16. The two variants take 17.6 GB on disk; 339 GB are
  free.

### 12.2 Checkpoints

- **Unit:** (stage, reader-layer, cell, piece). Each writes `ll_<reader>_L<ℓ>_<cell>_<piece>.npz` and appends to
  `blocks_done.jsonl`.
- Model-side stages checkpoint per pair. A restart skips completed units.

### 12.3 Unattended gating (fix 12)

- A stage proceeds if its gates pass.
- **On any stop-gate failure:**
  - write up what failed;
  - skip the rest of that stage;
  - move to the next independent stage.
- **No repairs and no threshold changes mid-run.** Anything changed after outcomes is labelled post hoc.
- **Dependencies:**
  - 4b needs 4a;
  - the brew test needs Stage 0's frozen PCA-k;
  - every other stage depends only on Stage 0.

### 12.4 Reproducibility (fix 13)

- Every LL matrix, readout, capture, gate log and manifest (with git sha) is saved, along with `analysis.py`.
- An independent reviewer can recompute every headline number from the raw outputs. Each stage's analysis has a
  single entry point.

### 12.5 Run order (fix 14)

- 0 → 1a → 2 (+2b) → 1b → brew test → 3 → 4a → 4b → 5.
- The template lens and the CPU analyses run whenever their inputs exist.

### 12.6 After each stage

1. Write `results/007_jsubspace/status/<stage>.md`: what ran, the gates, headline numbers, anything unexpected.
2. Mirror `outputs/007_jsubspace/<stage>/` and `results/` to the **public** HF dataset
   `senku21x/causal-concordance-outputs` with `scripts/000_setup/mirror_outputs.py` (MANIFEST.json updated).
3. Commit the status file and results, and push the `007-jsubspace` branch (never `main`, never force).

`/workspace` is not a persistent volume, so this happens automatically per the PI.

## 13. GPU-hour estimate and budget

### 13.1 Rates (logged, same GPU model)

| Rate | Value | Source |
|---|---|---|
| Oracle read (24 rows) | 0.56 s | 006 run1 |
| NLA read | 1.17 s | 006 run1 |
| SEC forward | 0.106 s | 004 (2.86 forwards per condition) |
| Sampled readout | 0.5–1.0 s | 001 snapshot |

Read time is assumed to scale with candidate rows relative to 24.

### 13.2 Per-stage estimates

| Stage | What | GPU-h |
|---|---|---|
| 0 | CPU: bases, pieces, gates, PCA-k, 2b pursuits, J_L56, chain count, template-lens download and pin | 0 |
| 1a | 26 pieces: reads 3.82 · positive control 0.20 · qualitative 0.15 · prompt-only, wikitext, scale check, loads 0.18 | **4.35** |
| 2 | 34 conditions × 3 layers × 33 pairs, plus smoke | **0.4** |
| 2b | power check, plus 7 conditions × 3 layers × 33 pairs (if powered) | **0.1** |
| 1b | 26 pieces at `</think>` | **3.9** |
| brew test | captures 0.05 · 15 pieces × 227 renders 2.17 · positive control 0.11 · overhead 0.1 | **2.4** |
| 3 | multihop | **1.4–1.8** |
| 4a | relational gate and profile | **0.5** |
| 4b | relational Part A | **1.6–2.3** |
| 5 | chain, Oracle L56 | **0.3** |
| | **Total** | **≈ 15.0–16.1** |

### 13.3 Budget rule (PI: cap about 15 GPU-h)

- GPU-hours are the wall-clock time the GPU is held, loads included.
- **Before each stage:** projected = spent + the remaining stages' estimates, each scaled by the observed/estimated
  ratio so far.
- **If the projection exceeds 15 h,** drop Stage 5. If it still exceeds 15 h, drop Stage 4b.
- Dropped stages are recorded in the status file.
- On these estimates Stage 5 will likely be dropped, and 4b decided at that point.

## 14. Pre-registered rules and priors

Agent priors are fixed. `[PI]` is left blank by decision. R1–R4 were written under the four-contrast mapping, before
P5 was added, and are left unchanged.

| ID | Rule | Agent prior | `[PI]` |
|---|---|---|---|
| R1 | Visible model supported (§4.4), per reader-layer: Oracle L40 · Oracle L44 · NLA L42 | 0.30 · 0.30 · 0.25 | |
| R2 | H_J supported, per reader-layer | 0.10 · 0.10 · 0.10 | |
| R3 | Raw model supported, per reader-layer | 0.10 · 0.10 · 0.15 | |
| R4 | Unresolved, per reader-layer | 0.50 · 0.50 · 0.50 | |
| R5 | Positive control passes, per piece: whole · J · rest of J · random · PCA-k · mean-removed | 0.95 · 0.80 · 0.85 · 0.60 · 0.40 · 0.45 | |
| R6 | G-repro passes (Stages 1a and 2) | 0.95 | |
| R7 (v2) | P5 is `+` for ≥ 1 reader-layer on dev | 0.20 | |
| R8 (v2) | Mean-removed is flagged role-nonspecific for ≥ 1 reader-layer (dev) | 0.40 | |
| R9 (v2) | Generic-mean-removed D_z is within δ of template-mean-removed D_z, for each reader-layer | 0.35 | |
| R10 (v2) | Every dev category of P1–P5 that is `+` or `−` replicates on test | 0.55 | |
| B1 | Part B: J tracks Δ-share at ≥ 2 of 3 layers | 0.60 | |
| B2 | Part B: J beyond content share at ≥ 2 of 3 layers | 0.10 | |
| B3 (v2) | Part B regression: J's residual CI includes 0 at ≥ 2 of 3 layers | 0.65 | |
| B4 (v2) | 2b power check passes | 0.45 | |
| B5 (v2) | 2b privilege replicated with J_NP (given the power check passes) | 0.25 | |
| B6 (v2) | 2b J-part energy share (median over colours and layers, J_NP) ≤ 0.15 | 0.70 | |
| M1 | Multihop test: whole-activation D_bridge CI above 0 for ≥ 1 reader-layer | 0.60 | |
| M2 | Multihop: ≥ 50% of J-piece readouts that name the bridge also restate the cue | 0.60 | |
| Q1 | Relational: ≥ 20 scenes pass the gate and alignment | 0.55 | |
| Q2 | Relational profile yields a used and an unused position | 0.45 | |
| T1 (v2) | Template lens passes its colour sanity check at L40, L42 and L44 | 0.75 | |
| T2 (v2) | Template lens: D_z(rest of J) ≥ 0.5 · D_z(whole) at `:` L42 | 0.60 | |
| C1 | CPU lens: z-first share (J_NP cosine, `:` L40/L42) higher on mean-removed than on whole | 0.75 | |
| C2 | CPU agreement: r(J) exceeds r(rest of J) by ≥ 0.1 | 0.30 | |

## 15. Gates (summary)

| Stage | Before | During (stop) | Interpretation |
|---|---|---|---|
| 0 | G-bases, G-geom, G-recon, G-002; PCA-k frozen; CPU unit tests (D incl. D_off, derangements, Holm/TOST, mapping on synthetic outcomes incl. P5 and uninformative); template-lens sha256 and centring check; chain count; 2b pursuits | — | — |
| 1a / 1b | Stage 0 passed | G-score1/2, G-repro, G-inject (+ scale), G-mem | positive control, engagement, role flags, colour prior |
| 2 / 2b | Stage 0 passed | smoke, G0, G1, G-repro; 2b power check | damage flags |
| brew test | Stage 0 passed; captures pass the cell assertion | as 1a (G-repro not applicable: new captures) | as 1a |
| 3 | G-cap, G-split, G-cw (report) | G-score, G-inject, G-mem | positive control, restatement |
| 4a | ≥ 10 scenes after the gate | G0, G1, noise floor | damage |
| 4b | 4a yields both positions; budget | as 1a | positive control |
| 5 | ≥ 10 pairs; budget | as 1a | positive control |
| template lens | sha256; sanity check per layer | — | layer informativeness |

## 16. What each outcome would change

**Stage 1a, and confirmed on test.**
- **Visible supported.**
  - The readers' D tracks how strongly the item signal stands out from the shared direction, not J geometry. Brew
    activations are 96–99% shared.
  - Consequences:
    - CW-12-style J-part sufficiency is explained by shared-energy removal, and the project does not cite the J split
      as evidence that the verbalizers read the paper's workspace;
    - reader comparisons must account for the item fraction;
    - mean-removed scoring becomes a standard sensitivity.
  - If the centred pieces are role-nonspecific, the amplification concerns the prompt difference, not z.
- **H_J supported (P5 `+` included).**
  - The readers rely on J-readable directions even with visibility equalised. Stage 2 and 2b then decide whether that
    content is model-privileged.
  - "H_J reader side, tracks-Δ model side" would be a reader–use dissociation.
  - The runner-up caveat applies.
- **Raw supported.** The readers track item content share. CW-12's J ≈ whole would then point to a family or
  instrument difference; Stage 3 is the check.
- **Unresolved, uninformative, or not replicated on test.** Report the profile and stop pursuing the reader-side J
  question on brew.
- **Generic vs template centring (S6).**
  - If the generic mean does as well as the template mean, the effect is about removing the model's global offset.
  - If only the template mean works, it is about this task's shared component.
- **Template lens.** If a non-J linear reader also loses z without J, "J-dependence" is not specific to J-built
  readers.

**Stage 1b.** The same pattern at `</think>` (share 0.01) as at `:` (0.48) means the readers' piece dependence is
unrelated to use.

**Stage 2.**
- **Tracks Δ-share:** 004 extends to a fixed basis that was not fitted on the data. There is no model-side evidence
  that J is privileged for this answer at `:` L40–44.
- **J beyond content share:** a bounded follow-up. Still no label.
- **J below:** the answer-relevant difference sits mostly outside the token-readout subspace.

**Stage 2b.**
- **Privilege replicated:** the paper's §3.3 result holds on Qwen for brew z directions at matched magnitude, despite
  the small J share. That would sit against Stage 2 if Stage 2 tracks Δ-share; both are reported.
- **Not replicated:** the paper's privilege claim does not transfer to this model and task.
- **Power check fails:** class-mean directions at natural magnitude cannot test it here (the result is "not
  testable", not "no").

**Stage 3.** Same pattern → two families agree. Restatement dominates → multihop is retired as a J-split testbed.

**Stage 4.** A different J-split pattern at the used vs the unused position is evidence that the readers' piece
dependence carries use information. The same pattern is evidence against it.

**Stage 5.** Hypothesis-generating only.

## 17. What 007 cannot show

- **J specialness:** J is the token-readout subspace. H_J support cannot separate a held intermediate from
  output-facing content: z is the model's runner-up at `:` in ≥ 50% of tables (006b).
- **Reader D is decodability through a reader,** not the reader's claim and not use.
- **Distribution shift:** pieces, especially centred ones, are off the readers' training distribution (no precedent
  for centred verbalizer inputs, §4.7). The positive control bounds this, it does not remove it.
- **Scope:**
  - one model;
  - one prefix per reader;
  - SEC captures for brew;
  - brew dev for discovery and brew test for confirmation (once);
  - multihop clusters are few.
- **Model side:** single-cell swaps. "Influence on this answer margin" only. No natural-route necessity.

## 18. Code, outputs, tests

- **Code** (new files under `scripts/007_jsubspace/`):
  - `pieces.py` (CPU): every operator, with the G-bases, G-geom and G-recon checks;
  - `reader_split.py` (GPU; stages `stage1a`, `stage1b`, `brew_test`, `multihop`, `relational_a`, `chain`). It reuses
    006's `material()`, `spellings()` and `score_block`;
  - `swap.py` (GPU): Stages 2 and 2b, reusing 004's `make_delta`, `PairRunner`, fidelity and damage code;
  - `brew_test_capture.py` (SEC captures via 001's `capture_brew`);
  - `relational_profile.py` (001's `measure_pair` and `noise_floor`);
  - `template_lens.py` (CPU);
  - `analysis.py` (CPU): D for all roles, bootstrap, derangements, Holm/TOST, the mapping, role flags, colour prior,
    agreement, token lens, the Part B regression;
  - `status.py`: the per-stage status file plus mirror and push.
- **Changes to shared code:**
  - `readers.score_seq` gains a `None` row (Oracle: no injection; NLA: a zero row). A test asserts it equals a
    bare-carrier run;
  - `registry.json` gains the template-lens pins.
- **Tests:** `tests/test_007_*.py`, CPU:
  - D on synthetic LLs (incl. D_off);
  - piece reconstruction;
  - derangement validity;
  - Holm/TOST;
  - the five-contrast mapping on synthetic outcomes (every category, incl. uninformative);
  - template-lens scoring on a toy tensor.
- **Outputs:**
  - `outputs/007_jsubspace/<stage>/<run_id>/`;
  - `results/007_jsubspace/analysis/<stage>.json`;
  - `results/007_jsubspace/status/<stage>.md`;
  - one report per stage at the end.

## 19. Decisions record

| # | Decision (PI, 2026-10-05) | Where |
|---|---|---|
| 1 | Random complements: 3 | §3.1 |
| 2 | `</think>`: Stage 1b, after Stage 2 | §12.5 |
| 3 | Part B additions (Δ-matched random, rest-of-J swap): yes | §5.2 |
| 4 | Multihop: relation-level key, secondary; expect little about J | §8 |
| 5 | Relational: profile at 5 shared positions, whole bank, exploratory | §9 |
| 6 | Branch `007-jsubspace`; commit the spec, report 22 and the geometry code before any run | Status |
| 7 | Mirror outputs to the public HF dataset automatically after each stage | §12.6 |
| 8 | Budget about 15 GPU-h; drop Stage 5 first, then 4b | §13.3 |
| 9 | `[PI]` priors blank; agent priors fixed | §14 |
| A1 | P5 in the primary family (15 contrasts) | §2.2, §4.4 |
| A2 | Generic-mean centring (secondary), with a note on literature precedent | §3.1, §4.7 |
| A3 | Role specificity on every piece, incl. off-trajectory D; explicit reading for both centring arms | §3.3 |
| A4 | Colour-prior report | §3.4 |
| A5 | Qualitative samples, Stage 1a | §4.8 |
| A6 | Part B regression points and J's residual | §5.2, §5.3 |
| A7 | Stage 2b privilege test, J_NP and J_CB, power check first | §6 |
| A8 | Brew test replication, confirmatory; test use approved under AMENDMENT_02 | §7 |
| A9 | Template lens as a secondary reader, both variants pinned | §3.7 |
| F10 | Scale-invariance check in G-inject | §4.6 |
| F11 | Bases float32, re-orthonormalised by QR, errors logged | §3.1, §4.6 |
| F12 | Unattended gating | §12.3 |
| F13 | Save everything needed to recompute | §12.4 |
| F14 | Run order | §12.5 |
| O1–O4 | My open points (test material, rest-of-random on test, AMENDMENT_02 file, template-lens centring) | §1 |

## 20. Execution notes (added at the go, 2026-10-05, before any run)

- **PI at the go:** "Approved: spec v2 as written, including your open points O1–O4. Go. … utilise the instance to
  the fullest and try to minimise the time it takes."
- **Throughput.** Batch sizes are engineering choices, not analysis choices. The affected settings are
  `acts_per_batch` in `score_seq`, the forward batching in the SEC stages, and the generation batch. To minimise
  wall-clock, each GPU stage sets them with a short timing probe at its start, before any outcome is computed, and
  records the setting in its manifest.
  - Every gate applies unchanged at the chosen setting. G-score1/2 and G-repro already tolerate bf16 batch-shape
    differences (006 v1.1: max ≤ 0.25, median ≤ 0.05 nats).
  - This replaces §3.2's "`acts_per_batch = 2`, as in 006".
- **CPU-labelled heavy linear algebra** in Stage 0 (the W_U·J products for bases and atoms, the pursuits) may run on
  the idle GPU in float32. It is the same arithmetic, and no model forward is involved.
