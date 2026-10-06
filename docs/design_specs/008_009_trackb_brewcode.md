# 008 (Track B) and 009 (brew-code bank): design note

**Status.**
- Written and committed before any GPU run of either experiment (PI, 2026-10-06: "Design note first").
- Everything here is **exploratory** under the relaxed rules. Nothing is pre-registered as confirmatory, and the
  decision rules below are fixed in advance anyway.
- Branch `008-trackb-brewcode`.
- **Readers stay off the 009 bank until its dev gate and output-facing check pass (§2.6).**
- Wording: at a cell, "no single-cell leverage" (S ≈ 0), never "not used".

**What the PI changed from proposal 28 (verbatim intent):**
- the primary family is brew with fresh random tables and a colour→code key, not factual templates;
- the output-facing check scores the bridge after a teacher-forced answer and continuation, not just at the next token;
- the paired task (final-code vs start-code) is part of the first de-risk;
- scale: hundreds of pairs, folds by table, 007's pieces, and the NLA at L44 as well as L42;
- Track B gets the true coordinate swap, a real whitened swap, and per-layer clamps.

---

## Part 1 — 008: Track B, 007's model-side test done the paper's way

### 1.1 Question

On brew dev, does the J part of z's representation have special purchase on the answer? Track B tests this under the
paper's §3.3 protocol, rather than 2b's one-cell, one-layer adaptation:
- coordinate swaps;
- applied at every cell of a band and clamped per layer;
- in raw and in whitened coordinates.

### 1.2 Material

- **Pairs:** the 82 gated dev brew pairs from 33 tables (`brew_material.gated_dev_brew`).
  - 004's 33 frozen pairs (one per table) are reported as a subset, for comparability with Stage 2/2b.
  - Statistics cluster by table.
- **Recipient and donor:** member and partner, direction p → m as in 003, 004 and 007.
- **Sites:**
  - **emit3** (primary): the three emit cells (assistant newline, `</think>` newline, `:`), all clamped together;
  - **colon** (secondary, for 2b continuity): `:` only.
- **Band:** L40–48, every layer.
  - All writes happen in SEC's float32 region (blocks ≥ 40).
  - Layer = output of block l.

### 1.3 Two families of swap

All swaps use one primitive: a **per-layer coordinate clamp**.
- At each layer ℓ in the band and each site position, the edit sets coordinates c(h) = R(h − μ) to a target t_ℓ, writing
  along A, with R A = I:
  - h ← h + A (t_ℓ − c(h)).
- The target comes from the **clean** run's coordinates at that layer and position, t_ℓ = σ(c_clean,ℓ). σ exchanges the
  z and z′ coordinates; α = 2 means c + 2(σ(c) − c).
- So each layer is *set* to a target. The edit is not added again at every layer (PI).
- The clean coordinates come from a clean SEC forward of the same sequence: the same continuation, the same length.

**(A) The paper's true lens-coordinate swap.**
- V = [v_z, v_z′], the folded J-lens atoms of the lowercase colour tokens " z" and " z′" (all 12 are single tokens).
  Lenses J_NP and J_CB.
- Raw: A = V and R = V⁺ (pseudo-inverse), μ = 0.
- Conditions:
  - α = 1 and α = 2;
  - a sensitivity with both spellings (" z", " Z" against " z′", " Z′"; σ swaps like with like; J_NP, raw);
  - **positive control:** swap the answer atoms (a ↔ a′), which should move the answer;
  - **negative control:** swap two off-trajectory colours (in none of {start, z, a} of either member, seeded), which
    should not move it.

**(B) Class-mean z-direction swap (paper Fig. 16 style).**
- d_c^{cell, ℓ} = mean(h | z = c) − mean(h), computed at that cell and layer from new SEC dev captures (231 renders),
  leaving the pair's table out.
- J part: nonnegative gradient pursuit, k = 25, NNLS refit, folded atoms (`jspace.decompose`), separately for J_NP and
  J_CB. non-J part = d − d_J.
- Conditions X with V_X = [d_z^X, d_z′^X]:
  - full;
  - J part (×2 lenses);
  - non-J part (×2 lenses);
  - 4 norm-matched random direction pairs.
- **Complementary clamps** (Fig. 16 hatched bars):
  - swap the non-J parts while clamping the J-lens coordinates of the pursuit supports of d_z and d_z′ (plus the own
    colour atoms) to clean values;
  - and the reverse: swap the J parts while clamping the non-J parts' coordinates to clean values.
- **Magnitude:**
  - *natural* (primary: the coordinate swap as the paper does it);
  - *raw-norm matched* (each layer's edit rescaled to the full condition's raw edit norm; 2b's convention);
  - *whitened-norm matched* (rescaled to the full condition's ‖Σ^{-1/2} Δh‖; reported separately, and **not** called
    the whitened swap).

### 1.4 The whitened swap

**Σ.**
- Per layer L40–48, from about 50k tokens of chat-style text: UltraChat-200k `test_sft`, Qwen chat template, thinking
  off, sequences ≤ 512 tokens, the first 4 positions of each sequence dropped.
- Captured under SEC, so it matches the label regime.
- Ledoit–Wolf shrinkage toward (tr S / d) I. **The shrinkage intensity λ is reported per layer.**
- W = Σ_λ^{-1/2} and W⁻¹ = Σ_λ^{1/2} by eigendecomposition, in float64. μ is the corpus mean.

**Whitened coordinates.** h̃ = W(h − μ). The swap happens in whitened coordinates and is mapped back:
- **(A-w), lens atoms.** J-lens vectors are gradients (covectors), so their whitened images are Ṽ = W⁻¹V. Then
  c̃ = Ṽ⁺ W (h − μ), A = W⁻¹ Ṽ = Σ_λ V, R = Ṽ⁺ W.
- **(B-w), class-mean directions.** Directions are displacements: d̃ = W d.
  - The J/non-J split is redone in whitened space, by pursuit over whitened atoms ã_t = unit(W⁻¹ v_t).
  - Ṽ_X = [d̃_z^X, d̃_z′^X]; c̃ = Ṽ_X⁺ W (h − μ); A = W⁻¹ Ṽ_X.
- The same clamp primitive applies, with R A = I asserted.

**Reported separately:** raw-space directions with edits matched in the whitened norm. That is §1.3's
whitened-norm-matched arm, not the whitened swap.

### 1.5 Score, gates, decision rules

**Score (004/007).**
- m = LSE log P(a′) − LSE log P(a), full sequence, all 24 spellings.
- S = Δm / G, where G is the gap (donor − clean).
- Top-1 flip rate: the share of pairs whose top-1 colour becomes a′ (the paper's metric).
- Damage readouts as in 004.

**Gates** (before outcomes; a failure stops that arm, and the result is kept):

| Gate | Requirement |
|---|---|
| G0 | Zero edit (t = c_clean at every layer): \|Δm\| ≤ 2e-4 |
| G1 | Fidelity on every write (ρ ∈ [0.9, 1.1], κ ≥ 0.99, 004's rule) |
| G-clamp | After each write, max\|c(h′) − t\| ≤ 1e-3 · max(1, \|t\|) |
| G-repro | The whole-cell patch at `:` L42 reproduces 001's profile S within 0.01 (all 82 pairs) |
| G-RA | ‖R A − I‖_max ≤ 1e-5 for every condition |
| G-Σ | min eigenvalue of Σ_λ > 0; λ reported |

**Power gate.**
- Arm (B): the full-direction natural swap at emit3 must reach a median S ≥ 0.15. Otherwise J vs non-J in (B) is not
  read ("lever too weak"). Raw and whitened are gated separately.
- Arm (A): the answer-atom positive control must reach a median S ≥ 0.15. Otherwise (A)'s z-atom results are not read.

**Reading rule** (per lens × metric × magnitude; 2b's rule):

| Reading | Rule |
|---|---|
| Privilege replicated | median S(J) − S(non-J) > 0 with CI > 0, **and** S(J) ≥ 0.5 · S(full) |
| Not replicated | S(non-J) ≥ S(J) (median) |
| Mixed | otherwise |

The complementary clamps are read as mediation: does the non-J effect fall when the J coordinates are clamped?

**Statistics.** Medians and means over pairs, 95% table-cluster bootstrap (10,000 draws, seed 20261004), sign counts.

**What changes what:**
- Privilege replicated under the natural swap, raw or whitened: 007's "no model-side J privilege" was an artefact of 2b's
  adaptation, and the dissociation headline is withdrawn.
- Not replicated with the power gate passed: "no model-side J privilege at brew's emit cells, L40–48" becomes a supported
  claim (dev; Qwen3.6-27B; brew).
- Metric-dependent: the write-up states the metric.

---

## Part 2 — 009: the brew-code bank (de-risk; reader-free until §2.6 passes)

### 2.1 Why this family

- On brew the intermediate z is also a candidate answer: the model's runner-up in 27/48 tables (006b). So readers' z
  signal cannot be separated from output-facing content.
- Here the answer is a **two-digit code**, so no colour is ever a candidate output.
- Factual templates are out as primary (PI): the bridge is a fixed function of the cue (so a probe decodes the cue),
  readers can re-solve from the cue, and the model can skip the bridge.
- Random tables make z a table-specific function of the start. Folds by table stop a z probe from learning the start.

### 2.2 Items

- **Generator** (`scripts/009_brewcode/bank.py`, seed 20261006): fresh random tables in WSB's brew format:
  - 10 rules over a random 10 of the 12 palette colours, destinations within the table's colours;
  - "You stir the potion twice.", then "The potion starts out {s}.", then the question;
  - plus a code key line, "Each colour has a code: …", giving all 12 colours distinct two-digit codes (10–99, random per
    table, listed in random order).
- **Two tasks with identical prompts except one token in the question:**
  - **final-code:** "What is the code of the potion's colour at the end?" Needs z.
  - **start-code:** "What is the code of the potion's colour at the start?" Does not need z.
- The instruction line says "Answer with a single two-digit code and nothing else." Render: brew's chat_prefill
  (`Answer:`, thinking off). Answers are " NN" (3 tokens: space and two digits).
- **Trajectory:** start s → z → f (final colour); answer = code(f). Constraints:
  - s, z and f are pairwise distinct;
  - no fixed points;
  - **depth 1 is the fallback** (§2.4).
- **Pairs.** Per table, a seeded member start; the partners are every other start s′ in the table with:
  - z′ ≠ z, f′ ≠ f, z′ ∉ {f, f′}, z ∉ {f′};
  - s′ ∉ {z, f} (not on the member's trajectory);
  - token-aligned renders (`wsb.assert_aligned`: only the start token differs).
- **Split by table:**
  - dev = 300 tables, used now;
  - test = 150 tables, generated, saved and hashed, then **untouched** (not even gated) until a confirmatory design note
    is committed.
- Folds: 5 by table within dev.

### 2.3 Gate (both tasks, both members)

- WSB/project gate: greedy correct and ≥ 9/10 correct at T = 0.8 (top_p 0.95, 16 new tokens, EOS suppressed).
- Answer parse: the first two-digit number in the decode.
- Run on every dev render under both tasks; renders of one table and task are batched (equal lengths).
- A pair is **in** if both members pass both tasks.

### 2.4 Gate pass-rate check (decides depth)

- If at depth 2 fewer than **25%** of candidate dev pairs pass both tasks, or fewer than **200** dev pairs pass in
  total, regenerate at **depth 1**: "You stir the potion once."; z is the colour after the single stir and is then the
  final colour; answer = code(z).
- The pass rates per task and per depth are reported either way.

### 2.5 Reader-free de-risk measurements (dev, gated pairs)

1. **Output-facing check** (must pass before any reader). The model's own log-probs for the 12 colours (every spelling,
   full-sequence LSE):
   - at each emit cell's next-token position (006b: the assistant newline, `</think>`, `:` = the answer slot);
   - **after the teacher-forced answer**, at the position following " NN";
   - **along a continuation**: the answer teacher-forced, then 24 greedy tokens with EOS suppressed, scoring z and f
     at every continuation position (max and mean over positions), and counting continuations that spell z or f.

   006b's D statistic is computed for z and for f, plus the rank of z and f in the model's next-token distribution.

   **Pass rule (final-code task):**
   - at the answer slot, the 95% CIs of model D_z and D_f lie inside [−0.5, +0.5] nats;
   - the median rank of z and of f is > 100;
   - after the teacher-forced answer and along the continuation, z or f is spelled in ≤ 20% of member renders.

   A failure is reported and blocks readers on this bank.
2. **Captures.** SEC captures of every gated dev render (both tasks) at the question cells and the three emit cells,
   L40–60 (L40, 42, 44, 48, 52, 56, 60), float32.
3. **z decodability.** A 12-way logistic probe for z at every cell × layer × task:
   - cross-validated over the 5 table folds;
   - control: permuted z labels within folds;
   - "decodes" = balanced accuracy CI above 1/12 + 0.15, with the control at chance.

   The probe for f and for s are reported alongside.
4. **Position-level leverage.** The whole-cell patch share S (p → m), with 001's protocol on the code-answer margin,
   at the 3 emit cells × the 7 layers, both tasks. Reported as "single-cell leverage", never as "use".

**Readers are compared only at cells where the z probe decodes under both tasks** (PI).

### 2.6 Go/no-go for readers on 009

Readers start only when **all** of the following hold on dev:
- the gate yields at least 200 pairs (at whichever depth);
- the output-facing check passes;
- at least one emit cell × layer within the readers' layers (L40/42/44) has z decodable under both tasks.

### 2.7 Reader phase (written now; runs after the §2.6 go)

- **Readers:**
  - the Oracle at L40/L44 (on-contract);
  - the NLA at **L42 and L44**, each with its positive control (006's "Tell me about {colour}" activations at that layer;
    LOO-prior ID ≥ 0.5 and ≥ 50/66 positive);
  - the token lenses (J_NP, J_CB, R_CB, logit) as comparators.
- **Pieces:** at the key cells, 007's whole, J, rest-of-J and random (×3), with 007's operators and bases.
- **Statistic:** 006's likelihood D for z (and f and s, collision-free) per cell, with:
  - the 200-derangement null;
  - a no-activation row;
  - a per-cell prompt-only floor;
  - saved qualitative samples at every cell, `</think>` included.
- **Primary contrast:** at cells where z decodes under both tasks, the reader's D_z under final-code minus under
  start-code, against the same contrast for the probe and for the model's own output.
- **Gates:** calibrated per reader before use (report 25: median and p99, never a fixed max).
- A separate note fixes the reader-phase decision rules before the first reader score is read.

### 2.8 GPU plan (one subject-model job at a time)

1. Σ capture (L40–48): about 10 min.
2. Track B, including the new captures and GPU pursuit: about 40–60 min.
3. 009 gate on dev, both tasks: about 30–45 min.
4. 009 output-facing check, captures, patch profile: about 30 min.
5. Readers, only if §2.6 passes: Oracle about 30 min; the NLA L42 and L44 about 2 h. The NLA run needs a check-in if it
   exceeds 2 GPU-h.

---

## §E. Execution notes for 008 (added before any 008 swap; Σ was captured first)

- **Σ capture done** (`scripts/008_trackb/sigma.py`, `results/008_trackb/analysis/sigma.json`).
  - Corpus: 100 UltraChat conversations, 50,199 tokens after dropping 4 per sequence.
  - Ledoit–Wolf λ = 0.0035–0.0043 across L40–48; minimum eigenvalue about 0.02; condition number about 1e4. **G-Σ
    passes.**
- **Centring.**
  - Lens-coordinate arms: raw uses μ = 0, as in the paper; the whitened arm uses μ = the corpus mean, per the design.
  - Added diagnostic arm `A_rawc`: raw metric centred on the corpus mean. It separates the effect of centring from the
    effect of whitening.
  - Class-mean arms (raw and whitened) centre on the dev grand mean, leaving the pair's table out (μ_dev). The class-mean
    directions are defined relative to that mean, so this makes the coordinates read as "how much of class z vs z′".
- **Condition list per pair.**
  - **emit3** (43 conditions):
    - lens swaps, raw α1 and α2;
    - lens swaps, whitened α1 and α2;
    - lens swap, raw-centred α1;
    - answer-atom positive control;
    - off-path negative control, raw and whitened;
    - both-spellings lens swap (when " Z" is one token);
    - class-mean full, raw and whitened;
    - J and non-J parts × 2 lenses × {raw natural, whitened natural, raw-norm matched, whitened-norm matched};
    - complementary clamps × 2 lenses;
    - 4 random direction pairs, norm-matched.
  - **colon** (22 conditions): the same without α2, the controls, the whitened-norm arms and the clamps.
  - **Plus:** G-repro (whole-cell patch at `:` L42), and G0 on the first two pairs.
- **Atoms.** Unit-normalised folded J-lens atoms (`jspace.atoms`, gain = 1 + final-norm weight), J_NP and J_CB.
  - The coordinate swap with unit atoms is the paper's "subtract the projection on v_s, add an equal-magnitude
    projection on v_t", corrected for non-orthogonality by the pseudo-inverse.
- **Smoke first.** 2 pairs (G0, G1, G-clamp, G-RA, G-repro, timing), then all 82 pairs.

## §E-009. Execution notes for 009 (added 2026-10-06 at 17:17 UTC, after 6 depth-2 gate tables and before any later table was read)

- **Gate smoke** (3 tables; reported only): the final-code task passed 0/8 renders (greedy 0.625, mean 3.6/10 correct
  samples); the start-code task passed 8/8.
- **Running depth-2 gate g2**, partial at 6 tables: 1/15 final renders pass; pairs 0/9.
- **Early-stop rule** (fixed now, before reading further tables):
  - Once ≥ 30 depth-2 tables are gated, if the Wilson 95% upper bound of the pair pass share is < 0.25, the depth-2 gate
    is stopped, §2.4's outcome is taken as "fall back to depth 1", and the depth-1 gate runs (g1, `--max-partners 4`).
  - Otherwise g2 runs to completion.
  - Reason: the full depth-2 gate costs about 45 GPU-min, and its outcome would then already be determined.
- The rest of the reader-free queue (outfacing, captures, probes, profile) runs on whichever gate run passes §2.4.
- **Collision rule (added 17:59 UTC, before the outfacing stage or any reader ran on 009).**
  - The depth-1 partner constraints do not exclude a partner whose hidden colour equals the member's start, or the
    reverse: 271 of 2,134 dev candidate pairs (13%).
  - A reader echoing the prompt's start colour would then mimic hidden-colour tracking.
  - **Primary D for z and f (z = f at depth 1) is now collision-free:** pairs are skipped where either member's hidden
    colour equals either member's start colour. For start, the same rule applies with the roles reversed.
  - The all-pairs D is reported alongside (`D_all_pairs`).
  - The outfacing pass rule uses the collision-free D.
