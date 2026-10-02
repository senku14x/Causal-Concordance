# 003 — Reader-free causal labels at the brew answer cells — design v1

*2026-10-02. PI go: "do 003". Dev split only; no test-split label (AMENDMENT_02 must freeze τ, ε′ and the substitution
bound first). Labels never use any reader: no J/R/lens/SAE direction, no reader output; this code imports no reader
code. The label-construction design called "002" in older documents is this one (renumbered when 002 became claim
scoring).*

## 1. Question

At brew's emission cells, L40–44, where the Oracle and NLA read, does the **intermediate z itself** influence the
answer, and how does it compare with the start and the answer as content?

- Report 001: the whole cell carries about half the pair difference (S ≈ 0.5 at `:` L42–44).
- But a whole-cell patch moves start, z, answer and everything downstream together, so it cannot say which content
  matters. The 001 audit and report §4 make this point.
- Report 002: the token lenses carry z specifically (start-swap tracking in 47–48 of 48 tables), and the verbalizers
  rarely claim it.

Whether z at that cell is used decides how to read both results. 003 builds the label: "pair difference used along a
content direction", per SPEC §6 as amended by A01 §3 and errata E1–E4.

## 2. Material

- **Pairs labelled:** the 82 gated brew dev start pairs (33 tables) from 001. Recipient = member, donor = partner
  (the forward direction, as in 001).
- **Directions fitted on:** SEC-regime activations of all 231 natural dev renders (48 tables; members and every
  partner candidate, gated or not). Activations do not depend on the gate.
  - Emission cells: the saved 001 captures (`captures_brew.npz`; 3 cells × L40/42/44/56/60).
  - Start-colour token (position 142; positive control): a new capture stage on the same 231 renders, SEC regime,
    same layers.
- **Cells:**
  - the three emission cells (newline after `assistant`, `\n\n` after `</think>`, `:`);
  - the start-colour token, at L40 only, as a positive control.
- **Layers:** 40, 42, 44 (primary), and 60 (reference, `:` only).
- **Chain is out of 003.** No chain cell carries single-cell leverage at L40–44 (report 001 §5), so content labels
  there would be uninformative.

## 3. Directions (reader-free)

**Folds.** The 48 dev tables are split into 3 folds, seeded and balanced on render count. A pair is labelled with
directions fitted on the **other two folds**; it never sees its own table.

**Decorrelated content directions (primary, A01 C15).** At each cell × layer, on the fitting renders, fit by least
squares:

H = b₀ + Σ_role onehot(colour in role) · B_role,  role ∈ {start, z, answer}

This gives per-colour coefficient vectors B_start[c], B_z[c] and B_answer[c] in ℝ⁵¹²⁰. Contrasts within a role are
identified. For a pair whose member and partner values of content k are c and c′:

û_k = unit(B_k[c′] − B_k[c])

Because each colour plays z with different starts and answers across tables, the z coefficient is the z-specific part
of the state with start and answer regressed out. That is the spec's "decorrelated direction".
- Contrasts are never averaged across colour pairs.
- If a colour never takes that role in the fitting set, the direction does not exist and the cell-content is
  "unresolved: no direction".

**Naive z direction (secondary, the spec's original).** unit(μ_{z=c′} − μ_{z=c}), class centroids on the fitting set.
Reported beside the primary to show how much decorrelation changes.

**Reliability fits.** The same directions fitted on **each of the two other folds separately**: two disjoint fitting
subsets, each about a third of the tables.

**Direction validity (gate G3).** In the held-out fold, the projection on û_z separates renders with z = c from
z = c′. Held-out AUC per contrast; gate: median ≥ 0.75 at `:` L42.

## 4. Edits and measurements (SEC regime; float32 hooks)

For pair i at cell c, layer ℓ, content k:

1. **Swap.** In the member's run:
   - h′ = h + (û·h_donor − û·h) û at (c, ℓ).
   - h_donor is the partner's activation at the same cell and layer, captured **at the matching sequence length**
     (per scored continuation, as in 001).
   - Effect Δm = m(edited) − m(clean): full-sequence margin, LSE over the accepted spellings, every token scored.
   - S = Δm / G.
2. **Norm-matched random null.** Eight random unit directions r per (pair, cell, layer, content), with the same signed
   edit size: h′ = h + δ r, where δ = û·(h_donor − h).
   - The null is pooled over pairs per (cell, layer, content); empirical p per test.
   - BH-FDR (q = 0.05) per pair across its cell × layer × content tests (A01 C9).
3. **Necessity** (content z; the three emission cells at L40/42/44):
   - Set û_z·h to the fitting-set mean of û_z·h at (c, ℓ), in the member's run and in the partner's run.
   - G_abl = m(partner, ablated) − m(member, ablated); N = 1 − G_abl / G.
   - Control: the same ablation along 8 random unit directions (each clamped to its own fitting-set mean).
4. **Reliability swaps:** the swap of step 1 with each reliability direction (§3), for every content at every primary
   cell.
5. **Whole-cell reference S_full** (E2): taken from the 001 run for the same pairs and regime. It is a reference, not
   a ceiling.
6. **Logged on every edit:**
   - Δlog P(a) and Δlog P(a′);
   - the full 12-colour answer distribution, clean and edited (all spellings);
   - off-pair mass;
   - top-1 retention on the non-answer positions after the edit;
   - realized-write fidelity (ρ, κ) from the recorded post-edit activation.

## 5. Labels (per pair × cell × layer × content)

Thresholds τ = 0.10 and ε′ = 0.05 (provisional, A01; errata E1 magnitude tests).

| State | Rule |
|---|---|
| **used** (pair difference used along û_k) | S ≥ τ **and** BH-significant against the random null |
| **used + necessary** (content z only) | used **and** N above the 95th percentile of the random-ablation N at that (cell, layer) |
| **decodable, not needed** (content z) | \|S_full\| < ε′ at the cell **and** z decodable there (001 probes, repaired: decodable at every emission cell × layer) |
| **unresolved** | anything else: S below τ while S_full is large; no direction; broad damage; fidelity failure |
| absent | not applicable: z is decodable at every emission cell |

**Flags, reported but not labels:**
- **1-D > full-cell** (E2: illusion or cancellation): S_k > S_full + ε′.
- **Broad damage:** off-pair mass up by more than 0.10 **and** non-colour mass up by more than 0.05, or top-1
  retention below 0.95. The edit's label is then unresolved.

**Content attribution** (A01 C15, the four-way {input, z, answer, other}): for every cell × layer, report S_z,
S_start and S_answer side by side, S_full − max_k S_k as "other", and the cosines between the three content
directions of the pair.

## 6. Gates (before outcomes)

| Gate | Rule | Where |
|---|---|---|
| G1 fidelity | ρ ∈ [0.9, 1.1], κ ≥ 0.99 on every swap and ablation; stop if more than 1% fail | every edit |
| G2 engine | a self-swap (target = own coordinate) is bitwise identical to clean; the clean margin matches 001's m_clean for the pair to 1e-4 | smoke, 3 pairs |
| G3 direction validity | median held-out AUC ≥ 0.75 for z contrasts at `:` L42 | before the run (CPU) |
| G4 label reliability | the used / not-used call agrees between the two disjoint-fit labels at κ ≥ 0.7 (per content at `:` L42–44) | after the run. If it fails, the labels are **descriptive only** and are not used to score readers (SPEC §6 gate) |

## 7. Predictions (priors; written before any 003 result)

| # | Prediction | Prior |
|---|---|---|
| L1 | z-direction swap at `:` L42/L44 is **used** in ≥ 50% of pairs | 55% |
| L2 | At `:` L40–44, median S_z > median S_start | 65% |
| L3 | At `:` L42–44, median S_z > median S_answer | 55% |
| L4 | z necessity at `:` L40–44 exceeds the random-ablation q95 in ≥ 30% of pairs | 35% |
| L5 | G4 reliability κ ≥ 0.7 for z at `:` L42–44 | 50% |
| L6 | Positive control: start-direction swap at the start token, L40, is used in ≥ 70% of pairs | 75% |
| L7 | At `:`, median S_z at L60 > at L44 | 50% |
| L8 | The decorrelated z direction gives a smaller median S_z than the naive centroid direction at `:` L42–44 (the naive one carries start and answer too) | 60% |

## 8. Stop rules

- G1 failures above 1%: stop and fix.
- G3 fails: stop, report, redesign the directions.
- G4 fails: finish and report, but the labels stay descriptive.
- Most cells unresolved: report as a negative result (SPEC §12).

## 9. Budget

Per pair:
- primary swaps: 3 cells × 3 layers × 3 contents = 27, plus `:` L60 × 3 = 3, plus the start token L40 × 3 = 3;
- reliability swaps: × 2 = 66;
- naive z swaps: 9 + 1;
- random null: 8 per real primary swap, 264;
- necessity: 9 × (1 + 8) × 2 = 162.

That is about 540 edited forwards × 0.2 s ≈ 1.8 min per pair, about **2.5 h for 82 pairs** on the GPU. The capture
stage takes about 2 min.

## 10. Outputs

- `outputs/003_labels/{captures, run}/`: raw, gitignored.
- `results/003_labels/analysis/{directions_validity.json, labels.json, summary.json, qualitative.json}` and the
  manifests.
- `docs/reports/003_labels.md`.

**Reader scoring against these labels is 004.** 003 does not score readers.

## 11. Changes log

*(none yet)*
