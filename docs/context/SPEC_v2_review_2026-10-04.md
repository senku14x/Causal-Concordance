# SPEC_v2 review: setup, contradictions, proposed edits, NLA plan, decisions (2026-10-04)

*Agent report.*
- No GPU run and no experiment. Setup and reading only.
- Paper notes are in `docs/context/reading_notes_v2.md`.
- Everything here proposes; nothing in SPEC_v2.md has been edited beyond saving it.

## 0. Setup

| Step | Result |
|---|---|
| Clone | `senku14x/Causal-Concordance` @ main 95ab0e5; `LittleMeHere/nla-vs-lenses` @ 58d34a8 (read-only) |
| `rebuild_env.sh` | exit 0 (24 min) |
| Artifact hashes | **29/29 match** (record re-written with new timestamps only; reverted, not committed) |
| CPU tests (`-m "not gpu"`) | **64 passed, 1 skipped, 0 failed** |
| Spec | `docs/context/SPEC_v2.md` saved on branch `spec-v2` (da1fe5b, pushed). Every non-blank SPEC.md line is present verbatim (403/403); Appendix B = AMENDMENT_01.md except heading levels. SPEC.md untouched. CLAUDE.md read order updated. |

- The skipped test is `test_brew_material`. It needs `outputs/` from the HF mirror (`scripts/000_setup/restore_outputs.py`), which was not run.
- GPU tests were not run (no-GPU instruction).
- The suite now collects 93 tests (65 CPU, 28 GPU), not 78.
- `spec-v2` was cut from `main`. `origin/setup/000-scaffold` (named in CLAUDE.md as the working branch) has one extra commit, ae18eee (the results-page narrative), which is not on `main` or `spec-v2`.

## a. The project as SPEC_v2 defines it (in my words)

**Question.**
- Several activation readers on Qwen3.6-27B are in scope:
  - token lenses: logit lens, J_NP (n = 1000), J_CB / R_CB (n = 25);
  - the Oracle Lens (an RL-trained phrase verbaliser with a frozen reconstructor);
  - an NLA (verbaliser + co-trained reconstructor).
- Do these readers report content that a specified answer depends on, or content that is merely decodable, guessable from the prompt, or re-derived?
- What does their agreement mean?
- What do the NLA's extra claims consist of, and does that content lie outside J-space?

**Why agreement is not truth.** Readers can agree for four reasons:
1. reading the same activation content (shared signal);
2. failing the same way (shared error; J/R share an estimator, J/R/logit share W_U);
3. guessing from the prompt (prompt inference);
4. re-solving: the Oracle and NLA are the subject model plus a LoRA, so they can rebuild the prompt from an activation and compute the hidden step themselves.

Majority vote is valid only if errors are independent, and here they are not.

**What the original plan was and what happened.**
- The plan: score readers against reader-free causal labels per (position, layer) cell — "used", "decodable, not needed", "absent", "unresolved" — computed from patches, mean-difference swaps, necessity ablations and probes on WorkspaceBench counterfactual pairs.
- On Brew (dev):
  - **Whole-cell patching** gives a reliable *position*-level map: the ":" cell carries 0.35–0.50 of the answer gap at L40–44 and ~0.74 at L60; the newline after `</think>` carries ≈ 0, though z is equally decodable there.
  - **Content-level labels failed:** 1-D labels κ 0.37 between disjoint fits (003); fitted subspaces no better than same-rank PCA (004).
  - **Reader scoring (002):** verbalisers mention z often but rarely in its role; token lenses carry z in candidate scores below their top-10.
- The CW pilot adds:
  - multihop: the prose readers name the bridge well beyond J and partly survive J-neighbourhood removal, which looks like re-solving;
  - chain at L56: the Oracle names intermediates above chance without restating the start;
  - Brew: mentions are rare and vanish under removal.

**What v2 does about it.**
- Labels become two tiers: position-level (active) and content-level (conditional, only if reliable and better than PCA). Reader scoring no longer waits for content labels; it is scored against task truth, position-level influence, and baselines (prompt-only, no-activation, shuffled).
- Reading vs re-solving becomes a main line:
  - A4: the "survives removal" tag, plus "re-derived" and "read outside J-space" inference tags, using J-based and reader-free removal;
  - A7: layer sweep with probes, swaps, restatement, prompt recovery, a J-any-layer cross-check, model-side removal.
- **005, the planted-concepts experiment**, becomes the direct test of "NLAs read outside the workspace": a known concept split into a J-space part and a remainder, measured on every reader and on the model's own behaviour.
- Supporting changes:
  - judge validation becomes a gate (one rubric, human-coded κ);
  - checkpoints must be reconciled before joint numbers;
  - joint numbers are dev-only, with the test split quarantined;
  - collaboration rules: design spec before every run, both sign off on joint designs, the non-negotiables.

**North Star / proxy.**
- North Star: know which readers report what the model's computation depends on, measured independently of any reader.
- Proxy (now):
  - per-reader hit rates on hidden steps against task truth and position-level influence, beyond prompt-only and shuffled baselines;
  - survival under content removal;
  - detection of planted J / non-J parts at doses that do or do not move the model.
- Main divergence risks:
  - position-level influence labels positions, not content;
  - planted states are off-distribution;
  - off-layer reads.

## b. Contradictions found

Severity: **H** = changes a number, a claim or a rule; **M** = misattributed or stale but no conclusion changes; **L** = cosmetic.

| # | Where | SPEC_v2 says | Evidence | Sev |
|---|---|---|---|---|
| B1 | CLAUDE.md "Fresh instance" §4, L170; STATUS.md L45; `docs/papers/INDEX.md` | Repo is private; keep it private while the paper PDFs are in it | GitHub API, 2026-10-04: `private: false, visibility: public` (unauthenticated clone works). The PDFs, text extractions and `docs/sources/web/nla.html` are publicly redistributed. | **H (urgent)** |
| B2 | §6.0 tier 1; §14.1 | ":" has position-level "influence" (reliable) | 001 §4 and Q11: 46–67% of ":" whole-cell patches at L40–60 are **damage-dominated by the pre-registered rule; "the flag stands"**. A01 §3: broad damage → unresolved. SPEC_v2 never mentions the flag. | **H** |
| B3 | §8 A2 callout; §9 P2 status | "a W ≈ 0 pattern" for J and the NLA on Brew | 003 §7 and §9: "an earlier 'W ≈ 0' reading was withdrawn after the audit". SPEC_v2 §5 itself says the ":" vs `</think>` comparison is "descriptive, not a W". | **H** |
| B4 | §14.4 | at `</think>`, "z is decodable and not needed in 70–100% of pairs" | 003 §3 and §9: renamed **"readable, low patch leverage"**; "not an item-level 'decodable, not needed' label" (family-level probe). | **H** |
| B5 | §14.4 | "'Used' z in 11–18% … each also necessary" | 003 §3: the random clamps were ≈ 20× smaller edits; "specificity is **not established**"; damage not logged on ablated runs. | **H** |
| B6 | §4, §4.R, §13 answer 1, A01 C17, CLAUDE.md, `registry.json`, `readers.py` | NLA = `nla-L42`, step-matched AR; "hashes differ … pick one"; check reconstruction "with the step-matched reconstructor" | These are **two different training runs** (reading_notes_v2 §5): `nla-L42` = r128 attention/GDN-only on an `av_sft` merge; `nla-rl` = r64 all-module on a separate full `av_base`. `nla-rl` has **no step-400 AR** (AR saved at step 600). PI decision 2026-10-04: we work with `nla-rl`. So every NLA number in 001/002, §1 callouts and §14.3 (role 5–15%, test–retest 48%, FVE 244/360) comes from a reader we are not continuing with. | **H** |
| B7 | §14.6, §14.7, §1 point 1 callout | CW Oracle numbers used beside on-contract readers | Per the `olens_and_ar` README the Oracle contract is layers 20–60 step 4, so **L42 is off-contract**. CW-1, CW-3/4/5, CW-6 and CW-8/9 ran it at L42. A01 C6: off-contract L42 arm "reported separately and never pooled". §14.7 uses CW-9 (Oracle L42) as "read from the activation". Conversely, CW STATUS says "readers are off their training layer at L56": true for the NLA, **false for the Oracle** (L56 is on-contract). | **H** |
| B8 | §4.R split row; §5 chain/multihop | Only the 25 Brew test items are quarantined | `splits.chain_split`: CW's first-60 chain items include **14 usable (single-digit) items, 7 of them in our chain test split**. The v2 choice "chain work is read at L56" cites CW-7, run on all 60 items. Multihop: CW used all 100 items, so a dev/test split made now has no unseen test items. | **H** |
| B9 | §5 status table (codebook row) | Codebook is the planned matched not-needed contrast, tagged [A01 §2 fallback] | A01 §2: fallback "not built unless 001 says so". 001 §9 rule 2: "**The RHUT fallback is not triggered**". Adopting the codebook is a new decision and should carry [v2] / PI, not A01. | M |
| B10 | §1 point 2 callout; §4 [A01 C18]; §10 | Brew prompt-only 1.00 "as in WSB"; WSB multihop prompt-only ≈ 0.90 | Two different baselines. 002 asked the stock model **for each colour** directly (002 §6: "a task baseline, not a ceiling on readers"). WSB asks what a model "would be thinking" at the read position and judges with the family instrument (brew 0.96). | M |
| B11 | §1 point 2 callout | J candidate scores "rank the intermediate first in 0.39–0.45 of 48 tables (shuffled ≈ 0.17)" | 0.39–0.45 = `lens_closed_set` hit rate over **231 renders** (member + partners), ":" L40–44. 0.17 = member-only shuffled rate over 48 tables (`lens_post_hoc`). The member-only real rate is 0.38–0.46. | L |
| B12 | §4 controls; §8 A4 | Reader-free removal = LEACE, "on Brew, 004's fitted content subspace" | 004 §4 and §11: the fitted z span is no better than same-rank PCA and mostly raises the recipient's own intermediate. It is **not a validated z eraser**. | M |
| B13 | §6 step 4 [v2] | Paired rotation as *the* necessity null | 004 §6: "004 ran no PCA clamp to compare against", while 004 §4 shows PCA matches the fitted spans for swaps. A paired rotation shows orientation matters, not content specificity. | M |
| B14 | §4 controls; A4; §14.6 CW-4 | Control = "same number of random directions" | `ablate2_pass.py` L91: the J directions of the same number of random **vocabulary tokens**, with no matching of removed norm ‖Δh‖. The bridge neighbourhood is aligned with active content, so it removes more of h than the control. | M |
| B15 | A7.3; CW-5 | Swap at "a strength J-lens can see" | `ablate2_pass.py` L94–96: the scale is chosen **per item until J-lens shows the partner** (up to 16×). J's "planted 14/50" is partly by construction, and strength varies by item. | M |
| B16 | §14.7 (CW-7 under "read from the activation") | Oracle reads chain intermediates at L56 | CW-6/7 score **any number named**: Oracle readouts carry 6–9 distinct numbers, hence chance 12/60. The WSB chain rule (top *computed* value, decoy null) was not applied; the restatement check covers the start number only. | M |
| B17 | Status table "78 tests"; STATUS.md (Oracle hash "cannot be verified"); CLAUDE.md (working branch) | — | 93 tests collected. Oracle: `local-workspace` iter_000600 adapter sha256 7d4b301e… = `olens_s3d_rl600`, configs and `run.json` identical (**verified**). Branch: see §0. | L |
| B18 | §7 005 callout | Neutral cells = Brew stir cells or the `</think>` newline, "measured zero leverage (001)" | 001's zeros are for the **natural start-pair difference**. Stir cells precede the only differing token, so they are bitwise no-ops by construction (001 §3); `</think>` ≈ 0 net, with the E2 cancellation caveat. Neither shows insensitivity to **injected** content. | M |
| B19 | §9 P11 | NLA detects the non-J part "at doses where the model's next-token distribution does not change" | Viswanath 2026: the non-J part moved the concept token by **+13.7 to +21.1 nats** but was never reported. The observed dissociation is report vs output, not detection vs no output change. | M |
| B20 | §15 rules; A7 item 8; 005 | "J-based removal allowed only as a reader-side analysis in A4/A7" | A7.8 (removal in the model's own forward) and 005 (J / non-J parts written into the model) are **J-defined model edits**. SPEC_v2 does not say which side of the import boundary (`tests/test_import_boundary.py`) this code lives on, or that its outputs may never enter labels or admission. | M |

## c. Proposed edits to SPEC_v2 (each as a [v2] callout; nothing applied)

### c1. J-space for 005 (replaces the "Define J-space concretely" bullet in §7)

> **[v2] J-space (005).** Operational definition from Gurnee et al. §2.3/§2.5, as used in their Fig. 8 and Fig. 16.
> - **J-lens vectors.** At layer ℓ, v_t = J_ℓᵀ(γ ⊙ W_U[t]) for every vocabulary token t. This is the folded readout direction (γ = 1 + final-norm weight), as in our `J standard` readout and RHUT `readout.py`. The WSB cosine readout's raw-W_U variant is a sensitivity.
> - **J-space component** of a vector x: the sparse non-negative combination Σ_{t∈S} c_t v_t (|S| ≤ k, c_t ≥ 0), found by gradient pursuit over the whole vocabulary. The non-J component is x minus the J component.
> - **k.** Primary k = 16, as for the paper's concept vectors (Fig. 8). Sensitivity k ∈ {10, 25}; the paper used 25 for probes (Fig. 16).
> - **Lens.** J_NP primary (n = 1000); J_CB as a check.
> - **Report** for every concept: the selected tokens, their coefficients, and the J share of ‖x‖². Sanity check against the paper: median 6–7% at k = 16 for concept vectors, 10–15% at k = 25 for probes.
> - **§A.8 caveat.** The paper's formal set F = ∪_{|S|=k} span{v_i}, with unrestricted orthogonal projection, admits negative coefficients. It therefore describes a larger set than the non-negative cones in the prose. Primary = the non-negative pursuit (what the paper's experiments use). Sensitivity = least-squares projection onto span{v_t : t ∈ S} for the same S.
> - **Not interchangeable with the top-r singular-subspace split** (Viswanath 2026). His conclusion flipped between r = 200 and r = 2000, and r was chosen after seeing results. If run at all, it is a secondary construction with r fixed in the design.
> - **Frozen in the 005 design before any run:** construction, k, lens, layer and γ convention.

### c2. 005 = the paper's privilege test on the model side, with a reader side added (extends the §7 / A8 callouts)

> **[v2] 005 design requirements.**
>
> **Concepts and vectors**
> - At least 24 single-token concepts, split dev/test by concept, frozen before any run.
> - u_c = mean last-token residual at layer ℓ for "Tell me about {c}", minus the mean over at least 100 other concepts. This is reader-free (Fig. 8; Lindsey 2025).
> - Split u_c = u_c^J + u_c^⊥ as in c1.
>
> **Conditions**
> - Matched norm, each rescaled to the same ‖·‖ × dose: full, J part, non-J part, random direction, random J-span vector (non-negative combination of k random v_t), random non-J vector (a random direction with its pursued J component removed).
> - The J and non-J parts also at their natural (unrescaled) norms.
>
> **Model side (Fig. 7 / Fig. 8 protocol)**
> - (i) **Report:** inject across the user turn, prefill to the open quote, score the concept's reciprocal rank and its open-text naming rate. Do not use P(yes): Viswanath found it fires for random in-J directions.
> - (ii) **Behaviour:** next-token KL on a neutral continuation, the concept token's log-prob uplift, and the task margin at Brew cells.
> - (iii) **Privilege clamp (Fig. 8 / Fig. 16 hatched bars):** inject the non-J part while clamping the J coordinates of the concept's selected tokens to their clean values, at every position and layer downstream. Report how much of the non-J report effect survives. The paper found it falls to 0 for swaps and about 0 for injection.
> - Writes in float32 (SEC replay for L ≥ 40), with ρ/κ fidelity measured on every write.
>
> **Reader side**
> - Every reader reads (a) the injection cell at the injection layer (direct access) and (b) a downstream cell or layer after the model's own propagation (model-carried). Only (b) is a fair comparison with the model's report. Viswanath's NLA had direct access only.
> - Text readers: at least 4 samples per dose (target 8). Token readers: candidate ranks.
> - A reader-free linear probe for each concept, trained on natural activations, as a detection baseline. If the probe detects u^⊥, then NLA detection of it is a property of the direction, not of the NLA.
> - J-lens detecting u^J is true by construction: a positive control only.
> - Contract layers: NLA at L42; Oracle at L40/L44. Choose the injection layer and downstream reads so that each reader is on-contract, or run them separately.
>
> **Neutral cells**
> - Choose them for injected content.
> - The zeros in 001 are for the natural pair difference: stir cells are no-ops for start pairs by construction.
> - Measure each candidate cell's response to the random-direction control at each dose before using it.
>
> **Pre-registered contrasts**
> - Report rate (J) − report rate (⊥) at matched norm (model).
> - Detection (⊥) per reader vs the probe.
> - Behaviour (⊥) vs report (⊥).
> - Split P11:
>   - **P11a:** the NLA detects ⊥ at doses where the model does not *report* it;
>   - **P11b:** … at doses where the next-token distribution does not change. Viswanath's data predict P11b fails.

### c3. §10 pitfalls, new row (plus a pointer in A7 item 1)

> | "Surfaces earlier" may be "reports a later state" (Biderman's layer-K+2 objection, R-lens post comment 2026-08-10) | Any claim that a reader shows content at layer ℓ — earlier than another reader, or before the model computes it — is anchored to a same-layer reference: (a) a probe fitted on layer-ℓ activations, held out, with the Hewitt–Liang control task; and/or (b) a causal edit at layer ℓ only (whole-cell or content edit at ℓ, scored at the output). Without the anchor, write "reader R's readout at ℓ names X", never "X is present at ℓ". Applies to A7.1 (L16/28/36), R- vs J-lens comparisons, and every per-layer reader curve. |

### c4. Other changes the papers and checks call for

**Readers and checkpoints**
1. **§4 / §4.R NLA row (PI, 2026-10-04):**
   - NLA = `ceselder/qwen3.6-27b-nla-rl` (`av_base` + `av_rl_adapters/iter_000XXX`; AR = `ar_reconstructor`, saved at step 600). The iteration is open (§d).
   - `nla-L42` results in 001/002/§14.3 are kept, labelled as a different NLA, and never pooled.
   - Update A01 §4 / C17, the registry, `readers.py` (new loader: separate `av_base`, no in-place merge into the subject) and CLAUDE.md accordingly.

**Labels and spec text (B2–B5, B11–B13, B18)**
2. **§6.0 tier 1:** "influence" requires the A01 damage gate. Damage-flagged cells (001 §4: ":" L40–60 at 46–67%) are reported as "influence (damage-flagged)" or "unresolved", per the PI decision (§e).
3. **§8 A2 and §9 P2:** replace "a W ≈ 0 pattern" with "not a matched contrast; no W statement (003 §9 withdrew the W ≈ 0 reading)".
4. **§14.4:**
   - "readable, low patch leverage (family-level probe; not an item-level label; 003 §3)";
   - "'used' 11–18%; necessity specificity not established (unmatched clamp size; 003 §3)".
5. **§1 point 2 callout:** fix the denominators (B11).
6. **§6 step 4:** necessity null = paired rotation **and** a same-rank PCA clamp (B13).

**Baselines and controls (B10, B12, B14, B15)**
7. **§4 baselines:** define two baselines and use the WSB one as the reader floor:
   - the *task baseline* (002 style: the stock model asked for the value);
   - the *WSB prompt-only* (stock model, prompt up to the read position, "what would a model be thinking", judged by the family instrument).
8. **A4 reader-free removal:** LEACE fitted on dev z labels at that cell. Accept it only if a fresh cross-fitted probe falls to chance (with the control task) after erasure. 004's span becomes a secondary, labelled "not validated as z-specific".
9. **A4 "matched random-removal control":** two controls:
   - count-matched random vocabulary J directions (as CW);
   - energy-matched: a random subspace removing the same ‖Δh‖.
10. **A7.3 swap:** fix the planted scale per family before reading, not per item by J outcome. Report J's follow rate as a construction check, not a result.

**Splits and scoring (B7, B8, B16)**
11. **§4.R split row:**
    - extend the quarantine to chain (7 of 14 usable CW chain items are test);
    - state that the L56 choice rests on 001 dev data (final cell 0.66 at L56) independently of CW-7;
    - multihop: no unseen items remain. Either declare multihop dev-only/exploratory, or build a new held-out set (gated the WSB way) before any reported multihop number.
12. **§14.6 / §14.7:** mark the Oracle at L42 as off-contract (A01 C6; not pooled). Mark CW-7's Oracle at L56 as on-contract and the NLA at L56 as off-layer.
13. **§7 text-reader scoring:**
    - add claim recurrence across adjacent positions (NLA paper) as a reliability covariate;
    - state the single-cell handicap for verbalisers (AO paper; AObench);
    - use open prompts only; never questions that name candidates (NLA paper on question-cued oracles).
14. **§12 step 1 WSB reproduction under WSB's exact rule:**
    - Brew: K·G > ΣO, mention-based, judge prompt `brew-2026-09-16`, J_NP cosine top-10, layers 20–60 step 4, with WSB's `null` / `baseline` floors;
    - joint chain numbers: WSB's top-computed-value rule with the decoy null;
    - multihop: read at the last word (WSB, post-2026-09-23).

**Rules and housekeeping (B9, B20)**
15. **§15 new rule:** reader-derived model edits (A7.8, 005):
    - live in reader-side modules (extend `test_import_boundary.py`);
    - their outputs never enter labels or admission.
16. **§5 codebook row:** retag as a [v2] / PI decision (B9).
17. **Status table:** 93 tests.

## d. §12 step 1: NLA checkpoint plan, updated for the PI's choice of `nla-rl`

**Settled**
- The checkpoint repo is `ceselder/qwen3.6-27b-nla-rl` (PI).
- WSB's published NLA numbers used `nla-rl` + `av_rl_adapters/iter_000400`. Evidence:
  - the `NLA` class default (`src/wsbench/produce/methods.py:269–270`), unchanged since the producer's first commit (494a980, 2026-09-23);
  - the eval READMEs' arm name "nla-rl-L42".
- WSB's "NLA-SFT (L42)" is presumably `av_base` with adapter `none` (the same class).
- The CW pilot used the same pair.
- The injection (Karvonen add at block 1, marker ㈜) and the prompt are identical to our `readers.py` path.

**Still open**
- **Which iteration.** `iter_000400` matches WSB and the pilot, but the only reconstructor is the end-of-run AR saved at step 600. `iter_000600` + AR@600 is the only step-matched pair.
- **Whether 001/002's `nla-L42` findings transfer to `nla-rl`.**

**Proposed check** (design spec first; nothing runs without your go)

- **Items**
  - 10 Brew dev tables: the 001/002 snapshot tables, at the three emission cells.
  - 10 multihop items, from CW's 100 (exploratory: no unseen items exist), at the last word.
  - The 64 rows of `nla-rl/data/example_activations.parquet` (in-distribution positive control and FVE reference).
  - The 8 `nla-L42` reference rows already used in 000.
- **Layer and inputs.** L42 only (both NLAs' training layer). Activations from the stock bf16 forward (WSB convention), captured once and saved; every reader reads the identical tensors.
- **Readers**
  - `nla-rl` iter_000400;
  - `nla-rl` iter_000600 (real arm only, to save GPU);
  - `nla-L42` step400 (reuse 002's Brew readouts; new multihop and reference readouts only).
- **Sampling.** T = 1.0, top-p 0.95, top-k 64, max 256 tokens (WSB and 002). 8 samples per cell. The seed is fixed per (item, cell, arm, sample). One cell is re-run to confirm determinism.
- **Arms.** Real, shuffled (fixed derangement within family), no-activation (dev-mean activation, as in 002).
- **Metrics**
  1. *Readout:*
     - Brew: z in role, z mentioned, conflicting, scored with 002's frozen judge;
     - multihop: bridge named, scored with the WSB strict judge (verbatim quote);
     - for each: rates per reader, real − shuffled, item-clustered bootstrap CIs, and the paired difference nla-rl − nla-L42 per cell.
  2. *Text statistics:* length, quoted strings, prompt-restatement rate (A9 style).
  3. *Reconstruction:* paper-definition FVE (`NLAReconstructor.fve`):
     - grid: AV {L42-400, rl-400, rl-600} × AR {`rl_critic_step400`, `nla-rl ar_reconstructor@600`};
     - on the 64 in-distribution rows and on our cells;
     - matched pairs: (L42-400, critic400) and (rl-600, AR600). The cross terms show how much the ruler matters.
- **Code to write first** (design → preflight → smoke):
  - an `nla-rl` loader in `readers.py`: separate `av_base`, PEFT adapter, block-1 hook, no merge into the subject;
  - registry entries with sha256 (av_base shards, adapters 400/600, AR);
  - `artifacts.py` download and verify;
  - an AR wrapper for the `nla-rl` reconstructor (same 43-block + value-head layout; check `norm: none` against `NLAReconstructor`'s normalisation).
- **Memory.** `av_base` (53.8 GB) cannot sit beside the 50 GB subject on 96 GB. Sequence: capture → unload subject → reader → unload → each AR (35 GB) alone.
- **Budget**
  - Downloads: ≈ 93 GB (`av_base` 53.8 + adapters 2 × 1.87 + AR 35.4), about 12 min at today's ~135 MB/s.
  - GPU: roughly
    - rl-400: 40 cells × 3 arms × 8 + 64 × 4 ≈ 1,220 explanations ≈ 34 min (at 000's 64 explanations / 107 s);
    - rl-600: about 16 min;
    - L42 new readouts: about 13 min;
    - AR scoring: about 10 min;
    - loads: about 5 min;
    - **total ≈ 1.3–1.7 h.**
  - Judge: ≈ 1,700 readouts ≈ $3–4 (002's rate of $0.0019 per readout).
- **Decision rules (proposed; you set them)**
  - Iteration:
    - if FVE(rl-400 | AR600) ≥ FVE(rl-600 | AR600) − 0.03 on the in-distribution rows, use iter_000400 everywhere (WSB and pilot comparability);
    - otherwise use iter_000400 for readout comparisons and iter_000600 + AR600 for every AR-based analysis (A5, level C, best-of-N).
  - 001/002 NLA rows:
    - if the Brew role-correct and real − shuffled rates of `nla-rl` and `nla-L42` agree within CIs, cite 001/002's NLA findings as transferring (still labelled);
    - otherwise re-run 002's NLA arm with `nla-rl` before any joint use.
- **Cheaper alternative:** skip the bridge check. Re-run 002's NLA arm with `nla-rl` iter_000400 (3,930 readouts, about 1 h GPU, about $8 of judge), and treat 001/002's NLA rows as a different reader from then on.

## e. Decisions only you can make

1. **Repo visibility (urgent):** the repo is public with the paper PDFs and web copies in it, contrary to CLAUDE.md, STATUS and INDEX. Make it private, or remove the files from the tree *and* history.
2. **NLA iteration:** 400 (comparability) vs 600 (matched AR). Run the §d check (≈ 1.5 h GPU, ≈ $4) or the cheaper re-run of 002's NLA arm? Cathryn must sign off (joint numbers).
3. **The 001/002 `nla-L42` results:** keep them labelled as a different reader, or re-run them.
4. **Accept or strike the [v2] items.** The ones that change the most:
   - 005 as a main experiment;
   - the two tier labels;
   - the A4 tags (survives removal / re-derived / read outside J-space);
   - the judge gate (κ ≥ 0.6 stop rule; who codes the human set);
   - the dev-only joint rule and test quarantine;
   - the NLA ≥ 4 / target 8 sample budget.
5. **Tier 1 at ":" with the damage flag:** "influence (flagged)" or "unresolved" (A01 rule).
6. **Codebook:** does it override A01 §2 and 001 rule 2 (fallback not triggered)? Who builds it?
7. **DAS (§6 step 9):** run it bounded, or record the decision not to.
8. **005:**
   - ownership (proposed joint) and the sign-off procedure;
   - J-space construction (c1: pursuit, k = 16 primary; J_NP vs J_CB; folded γ vs raw W_U);
   - whether a Viswanath-style rank split is a secondary arm.
9. **P11–P16 priors,** and whether P11 splits into P11a (report) / P11b (output).
10. **Test quarantine scope:**
    - chain (7 of the pilot's usable items are test);
    - multihop (no unseen items: dev-only, or build a new held-out set?).
11. **CW Oracle numbers at L42 (off-contract):**
    - keep them as a separate arm, or re-run at L40/L44;
    - also the Oracle-L42 arm question pending since 2026-10-02.
12. **J-defined model edits (A7.8, 005):** allowed? On which side of the import boundary?
13. **Branch:** keep `spec-v2` based on `main`, or rebase it onto `setup/000-scaffold` (one results-page commit ahead)? Which branch is the working branch from now on?
14. **AMENDMENT_02** (freeze τ, ε′, the substitution bound) before any test-split number.
15. **Rotate the OpenRouter key** (open since 2026-10-02).
16. **Applying the corrections B2–B5 and B10–B13 to SPEC_v2:** as [v2] callouts in place, or as an errata block like A01 §8?
