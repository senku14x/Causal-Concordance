# Pre-registration: three no-GPU checks (2026-10-04)

*Status: pre-registered. Committed before any of these analyses is run. No GPU is used; no model forward is run.*
- Source of the request: the PI's feedback note of 2026-10-04, §5 items 1–3.
- Inputs: saved outputs only:
  - this repo's `outputs/`, restored from the public HF mirror (82/82 files match the manifest);
  - read-only files in `LittleMeHere/nla-vs-lenses` @ 58d34a8.
- Priors are mine (the agent's), written before running; the PI may overwrite them.

**Settled while preparing (no analysis needed): the multihop read position in the CW pilot (doubt 6).**
- In `runs/multihop100/nla.multihop.jsonl`, the 33 bank prompts ending in a space are read at position −2, and the other 67 at the final token.
- This is exactly WSB's last-word rule: the bank's `readout = {last_word_token, offsets [-2] | [-1]}`, applied by `readplan._final_token_families`.
- So CW-1 and the reruns compare at the same token.

---

## C1. Re-score CW-6 / CW-7 (chain) with WorkspaceBench's chain instrument

**Question.**
- CW-7 is the only positive "reading" case in v2: the Oracle names a chain intermediate in 36/60 readouts at L56, against a chance of 12.
- That count scores *any* number in the readout. Oracle readouts carry 6–9 distinct numbers.
- Does the result survive WSB's stricter rule? The rule: the judge names the number(s) the readout presents as *computed*, and the top one must be an intermediate; it is compared with a magnitude-matched decoy null.

**Why it matters.** The v2 decision to read chain at L56, and §14.7's "read from the activation" entry, both lean partly on CW-7.

**Data (read-only).**
- Readouts: `runs/chain60/{olens_L42,nla_L42}.chain.jsonl` and `runs/chain60_L56/{olens_L56,nla_L56}.chain.jsonl`. Each holds 60 items; the Oracle gives 1 readout of 4 bullets, the NLA 1 sample.
- Items: `runs/chain60*/items.json`, which must match the first 60 bank items by name; checked before scoring.
- J-lens: `jlens.chain.jsonl`, scored as a secondary arm. WSB scores token lenses as a bag, without a summariser.

**Instrument.**
- WSB `chain_intermediates` judge at the pinned commit 92d763e, unmodified: prompt version `chain-free-2026-09-16`, `google/gemini-3.8-flash`, temperature 0, `cells=last`.
- Credit rule: the top value counts only if it appears as a whole number, or the judge's quote is verbatim.
- Decoy null: `extras.null_top1_near`, ±3, magnitude-matched, weighted per WSB.
- Called through WSB's own code (`wsbench judge family=chain_intermediates ...`). Readouts are passed through unchanged; the row format already matches WSB's contract.

**Outputs, per reader (Oracle, NLA; J-lens secondary) × layer (L42, L56)**
- Pass rate (top value ∈ intermediates), `null_top1_near`, `any_of_3_rate`, `committed_rate`.
- Each next to CW's any-number count, as computed by her `scripts/score_chain.py`.

**Subsets**
- (i) All 60 items. This is exploratory, the same set as CW-6/7, and comparable with them.
- (ii) The single-digit (usable) dev subset, n = 7. Our chain split puts 7 of the 14 usable pilot items in test.
- Test-subset numbers are computed and stored but **not reported**, and inform no decision (SPEC_v2 §4.R quarantine).
- n = 7 is too small for anything but description; (i) is the main read.

**Prediction (agent prior).** At L56, the Oracle's WSB pass rate exceeds its decoy null by ≥ 0.10 absolute on all 60 items: **55%**.

**Decision rule (applied to subset (i)).**

| Outcome | Reading |
|---|---|
| Oracle L56 pass − null ≥ 0.10, and the pass rate is ≥ 2× the null | CW-7 stands as "read from the activation (WSB chain rule)" |
| Difference < 0.05, or the pass rate ≤ the null | CW-7 is reclassified "not established". The L56 choice for chain rests on 001 only (final cell S = 0.66 at L56, dev). §14.7's entry is withdrawn. |
| In between | Reported as "weak", with the numbers |

**Cost.** About 360 judge calls (60 items × 2 layers × 3 readers), ≈ $1–3.

**Blocker.** The OpenRouter key is not on this instance (`/workspace/.env` is absent, and CLAUDE.md says secrets are in neither mirror). Until the PI supplies it, this check is limited to a dry run (`dry_run=True`), which validates the row format and item match. No numbers are produced.

---

## C2. What the off-pair mass is made of after whole-cell patches at ":"

**Question.**
- Report 001 flagged ":" whole-cell patches as damage-dominated by its pre-registered rule: off-pair next-token mass up > 0.10 in > 20% of pairs, observed in 46–67%.
- That rule was written with 1-D edits in mind.
- In a whole-cell *partner* patch, mass moving to z′ (the partner's intermediate, often the runner-up answer) or to z is arguably the patch working, not damage.
- Where does the off-pair mass actually go?

**Why it matters.** It decides the tier-1 label at ":": "influence", "influence (damage-flagged)", or "unresolved" (B2, and the feedback's §3.2).

**Data**
- **(A) Exact, L42 only.** 004 raw rows (`outputs/004_span_assay/run/pairs/*.jsonl`), arm `colon|full`, primary fit, 33 pairs (one per table). For clean and edited:
  - the full-sequence log-probability of each of the 12 palette colours, as a log-sum-exp over all accepted spellings (`logq`);
  - next-token mass per colour (`next_token_mass`).
- **(B) Partial, all layers.** 001 raw rows (`outputs/001_leverage_profile/run/profile_brew.jsonl`), `region = emit_colon`, direction p→m, L40/42/44/56/60, 82 pairs:
  - next-token off-pair mass, clean and edited;
  - the edited top-5 tokens with log-probs;
  - the clean top-5 from `pairs_brew.jsonl`.

**Method (A).**
- Category masses: a, a′, z, z′, other palette colours (the remaining colours), non-colour (= 1 − Σ of the 12 colours).
- Ties are assigned by priority a > a′ > z > z′; only z = a can occur under the partner constraints.
- Δ = edited − clean, for both the full-sequence and the next-token version.
- Off-pair increase = Δz + Δz′ + Δother + Δnon-colour.
- Report median Δ per category and the **share** of the off-pair increase per category, both over pairs with off-pair increase > 0.01. Give bootstrap 95% CIs over pairs (one pair per table) and sign counts.

**Method (B).**
- Map each top-5 token to a category: case-insensitive match after stripping spaces; colour words and their capitalised forms count as that colour.
- Visible Δ per category = edited top-5 mass − clean top-5 mass in that category.
- Unattributed = (off-pair Δ) − (visible off-pair Δ).
- Report per layer with the same summaries.
- (B) is descriptive only: most off-pair mass lies outside the top 5 (001 §4).

**Prediction (agent prior).** At L42 (A, full-sequence), z + z′ take ≥ 50% of the off-pair increase: **25%**. I expect most of it to go to non-colour tokens and other colours.

**Decision rule.** Applied to (A) at L42. This only proposes a label; the PI decides.

| (A) median shares of the off-pair increase | Proposed tier-1 label at ":" L42 |
|---|---|
| z + z′ ≥ 0.50 **and** non-colour ≤ 0.25 | "influence". Recommend that the 001 damage rule be rewritten for whole-cell partner patches (count z′ / z as transfer). |
| non-colour ≥ 0.50 | "influence (damage-flagged)" stands; damage is real. Unresolved per A01 stays on the table for the PI. |
| otherwise | "influence (damage-flagged)", with the decomposition reported next to it |

- Other layers keep "influence (damage-flagged)" until a full-distribution measurement exists there. (B) cannot settle them.
- If (B) shows z / z′ dominating the *visible* part, I will propose that measurement (it needs the GPU) as an item for a later design.

**Cost.** CPU only, seconds.

---

## C3. How much of the activation the CW removals take away (removal energy)

**Question.**
- CW-3/4 (multihop) and CW-9 (Brew) compare readers' naming after removing the content's J-direction neighbourhood with naming after removing "the same number of random directions".
- In `ablate2_pass.py` the control removes the J directions of the same number of random *vocabulary tokens*, with no matching of the removed norm.
- Are the two removals comparable in size?

**Why it matters.** If the content removal takes away much more of h, then "drops vs control" compares unequal edits (B14). The A4 "survives removal" tag would then need an energy-matched control (c4 item 9).

**Data (read-only)**
- Multihop:
  - originals `runs/abl50/resid_L42.pt` (50 items, L42);
  - edited `runs/abl3/resid_abl2.pt` (conditions `strong` / `swap` / `rand`, with `n_dirs`; the cos > 0.8 run = CW-4).
- Brew:
  - originals `runs/brew50/resid_L42.pt` (50 items, ":" L42);
  - edited `runs/brew50_abl/resid_abl2.pt` (CW-9).
- **Consistency gate before any number:**
  - originals and edits are matched by `id` and layer;
  - for `strong` and `rand`, h − h_edit must be orthogonal to h_edit (|cos| < 1e-3);
  - ‖h‖² must equal ‖h_edit‖² + ‖h − h_edit‖² to 1e-4 relative.
  - These identities hold if the edit is the orthogonal projection the code claims. Items failing the gate are reported and excluded.

**Measures, per item and condition (strong, rand)**
- Removed share of energy, r = ‖h − h_edit‖² / ‖h‖².
- ‖Δh‖ / ‖h‖.
- `n_dirs`.
- Ratio R = r_strong / r_rand per item.
- Report the median [bootstrap 95% CI over items], sign counts (strong > rand), and the n_dirs distribution, for multihop and Brew separately.

**Prediction (agent prior).** Median R > 2 on multihop: **80%**.

**Decision rule.**

| Median R | Reading |
|---|---|
| > 2 | B14 stands: the CW-3/4/9 controls are not energy-matched. §14.6 survival figures are kept but annotated "control removes less of the activation (median R = …)". Energy-matched controls are required for any rerun (c4 item 9). |
| 0.5–2 | The controls are roughly matched in energy. B14 is withdrawn, and the count-matched random-vocabulary control is acceptable as one of the two A4 controls. |
| < 0.5 | Reported as unexpected; investigated before any reading |

**Cost.** CPU only, seconds.

---

## What none of these can show

- C1 re-scores existing single readouts; it does not add samples. One readout per cell means per-item rates are noisy (002: two NLA readouts of near-identical activations agree on z 48% of the time).
- C2 at L42 is exact but covers 33 pairs at one layer. It describes where probability goes; it does not say *why* (contamination vs intended transfer vs runner-up structure).
- C3 measures edit size, not effect. Equal energy does not make two removals equally disruptive. It only removes one obvious confound.
