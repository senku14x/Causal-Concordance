# 31. Onboarding review: state of 001–011, doc inconsistencies, next steps (2026-10-06)

Written by a new session after onboarding (prompt: report 30). Nothing was run on the GPU and no results changed.

**Reading done:**
- the paper in full (web text, plus PDF renders of Figs. 8, 16 and 20);
- notes 04 and 04b in full;
- note 03 only in part: §8 (oracle lens) and §13.5–15 (corrections and disagreements);
- not done: the related reading (step 3).

**Checks:** four subagents checked the docs for 001–011 against the full-precision JSON. All numbers below are rounded half-up from that JSON.

**Instance:**
- branch `008-trackb-brewcode` at `fe4688b`, with only `CLAUDE.md` modified;
- GPU free;
- `outputs/011_olens_swaps/s1/pairs/` holds 74 files;
- 137 CPU tests pass;
- PID 49691 (`gpu_queue_posthoc.sh`) is still running, idle.

## Bottom line

The headline claim is that the Oracle reads z through J while the model's answer does not depend on J. Two points make it weaker than the docs suggest:

1. its two halves use different J objects;
2. the Oracle's reconstructor (AR) may have been trained partly on a J-lens objective. If it was, the Oracle's J dependence is partly built into the instrument.

## 1. Where the project stands

**Reader side.**
- **007 (pre-registered; confirmed on 49 held-out brew test tables):** the Oracle's z signal at `:` L40 depends on 007's dense 1,024-dim J subspace.
  - The J piece beats PCA and random pieces.
  - The rest of the activation, without J, carries almost nothing.
- **NLA (post hoc):** not J-concentrated at `:`.
- **010:** the readers follow what the J lens reads, not the model's output. Only the late Oracle layers (L56/L60) follow the output.

**Model side.**
- **008, the paper's §3.3 test on brew:** the lever is adequate (full direction S 0.21).
  - The sparse k=25 J part moves the answer far less than the non-J part (0.02–0.03 vs 0.21).
  - Clamping the J coordinates barely changes the non-J effect.
- **007 Stage 2:** swaps of the dense J subspace move the answer roughly in proportion to J's share of the content.
- The Qwen replications in the PI's notes also found weak J-swap effects (Nanda; Zeisler, 6–11% top-1 flips), so the null fits.

**Structural confound (006b, 009).** z stays the top colour in the model's own output tail, even when the answer is a code. Every reader result at these cells can be read as reading output-facing content.

**011 (post hoc).**
- The TA's whitened swap has no effect.
- The additive AR-vector edit gives S 0.10, against 0.23 for the model's class-mean direction.
- Run s1 stopped at 74 of 82 pairs and has not been analysed.

## 2. Inconsistencies, stale and wrong statements

### Problems with the claims

1. **The dissociation compares two different J objects.**
   - The reader side uses 007's dense J subspace; the model side uses 008's sparse J part.
   - The like-for-like comparison already exists, both dense:
     - reader side: Oracle J 5.8 vs whole 3.2 on dev (`stage1a.json`; 5.0 vs 2.9 on test);
     - model side: 007 Stage 2, whose lever was adequate (S_full 0.28, `stage2.json`).
   - Affects: `docs/reports/008_trackb.md:34`, `CLAUDE.md:229`. The 008 report also calls the sparse J part a "J subspace" (`008_trackb.md:33`, `:104-105`).

2. **The AR may have been trained partly on J (open question).**
   - In `agu18dec/olens_and_ar` @ c95215d5, `ar_ptag_pooled/meta.json` includes:
     - `jspace_repo` / `jspace_file`, which name the Neuronpedia J-lens (`Qwen3.6-27B_jacobian_lens_n1000.pt`);
     - `jspace_mu_prefix`, `jspace_layers 20–60`, `loss_mix_lambda 0.25`, `val_jloss`;
     - `val_jfve_L40` 0.5799 and `val_jfve_L44` 0.5887, against `val_fve_L40` 0.224 and `val_fve_L44` 0.2202.
   - The Oracle's RL reward is a "joint bullet-FVE … computed on this exact AR" (README).
   - The WSB post (`docs/sources/web/wsb_post.md:346`) says only "MSE in whitened space". No training code is in any local clone.
   - If a J term entered the AR loss or the RL reward:
     - 007's Oracle result is partly a design property of the Oracle;
     - that would explain why the NLA, which has no such term, differs.

3. **"J dependence is reader-specific" holds only at `:`.**
   - At `</think>` the NLA also collapses without J: rest-of-J 0.21 vs rest-random 1.21 (`posthoc_A2_stage1b.json`).
   - Report 007 §8 (`007_jsubspace.md:165-168`) was not updated after the A3 run.

4. **008 does not "reverse" the paper.**
   - The paper swaps and clamps at every position over a layer band (Fig. 16). 008 edits 3 cells × L40–48.
   - Defensible wording: "not reproduced under a band-limited clamp" (`008_trackb.md:22`).

5. **007's "Confirmed" bullet mixes dev and secondary numbers** (`CLAUDE.md:281-284`).
   - 5.8 vs 3.2 are dev descriptives; J − whole is not a registered contrast.
   - What was confirmed:
     - L40: P1, P2, P3 and P5;
     - L44: only P5.
   - The `</think>` result is dev-only and secondary.
   - "Does not follow use" conflicts with A1 item 8, which asks for "higher/lower observed patch leverage" wording.

6. **The restatement shares are for the J piece only.**
   - J piece: 0.28 / 0.24 / 0.15. Whole piece: 0.56 / 0.25 / 0.58 (recomputed from `restatement_*.jsonl`).
   - "About half" is still in `007_jsubspace.md:32-33` and `:113-114`, and in `stage3.md:48-51`.

7. **009's "110 of 112 probe sites" is inflated.**
   - 77 sites (q0–q10) come before the one token that differs, so their activations are identical under both questions.
   - Only 33 of the 35 later sites are informative.
   - z already decodes at q1–q2 (BA 0.94–0.98).
   - Affects: `009_brewcode_derisk.md:33,135`, `STATUS.md:35`, `CLAUDE.md:243`.

8. **011's null is explained the wrong way** (`011_olens_swaps.md:136`, `CLAUDE.md:264`).
   - The `C_white` edit equals (c_z′ − c_z)·W⁻¹(v_z − v_z′): the same raw direction as `D_ar`.
   - It is null because the clean coordinates do not favour AR(z) over AR(z′), not because "the swap edits almost nothing". At a similar total edit, `C_white_a2` gives S −0.022 / +0.011, while `D_ar` gives +0.104 / +0.094.
   - AR colour differences align with the model's class-mean differences (cos 0.15–0.24 over the first 10 tables, ≈0 for mismatched pairs). So "mostly noise" is too strong.
   - "Matches the published fit" (`CLAUDE.md:256`) overstates the loader validation:
     - unscaled 16-token FVE is −0.29 / −0.11 / +0.06;
     - the outputs are about 2× too long (norm ratio 1.20 at cosine 0.51, L44);
     - running all 64 blocks, where the official loader truncates the backbone, was never tested;
     - `ridge_c` 0.1 and the J-loss term were not used as checks.

9. **The held-out ledger understates use.**
   - Brew test was read by:
     - the Oracle (pre-registered);
     - the template lens;
     - the prompt-only floor;
     - the NLA (post hoc A2, 10-06).
   - AMENDMENT_02, the spec (`:547`) and `STATUS.md:57` say "used once".
   - `SPEC_v2.md:281` says multihop has no split; 007 made one (items 1–50 dev, 51–100 test).

### Number slips (full precision → half-up)

| Where | Docs | JSON | Correct |
|---|---|---|---|
| 008 J_CB J − non-J (`008_trackb.md:21`, `CLAUDE.md:226`, report 30:160; **in the PI's numbers to the TA**) | −0.15 | −0.14465 | −0.14 |
| 008 subset whitened power (`008_trackb.md:98`) | 0.12 | 0.090577 | 0.09 |
| 008 J energy, raw / whitened (`008_trackb.md:19,26`) | 4–9% / 0.6–1.7% | 3.3–8.7% / 0.5–2.0% | as JSON |
| 008 z-atom swaps (`008_trackb.md:39`) | "0.001–0.006 raw" | raw 0.0012–0.0024; 0.0064 is `A_white_a1`; `A_white_a2_J_NP` 0.031 (CI excludes 0) | relabel |
| 008 J_CB "ties" at whitened norm (`008_trackb.md:30`) | tie | CI [−0.029, −0.000] | below 0 |
| 010 readers vs model output (`CLAUDE.md:248`) | 0.2–0.3 | NLA 0.0702 | Oracle 0.21–0.30, NLA 0.07 |
| 010 readers vs J lens (`CLAUDE.md:248`) | 0.63–0.80 | NLA 0.330 / 0.365 | Oracle only |
| report 29:40 model vs S | 0.27–0.33 | L44 0.2417 | 0.24–0.33 |
| 011 smoke random (`CLAUDE.md:265`) | 0.09 | `C_raw_rand_0` 0.0948, `C_raw_rand_1` 0.0042 | 0.09 / 0.00 |
| 011 `C_rawcoord` s0 (`011_olens_swaps.md:137`) | +0.22 | 0.214978 | 0.21 |
| 011 FVE L40 / ret@1 L48 (`011_olens_swaps.md:106-107`) | 0.27 / 0.90 | 0.2619 / 0.8945 | 0.26 / 0.89 |
| 001 `</think>` share | 0.01 | L42 0.0145, L44 0.0260 | L42 only |
| 003 z positive (`003:71`) | 85–90% | 0.841463 | 84–90% |
| 002 NLA `:` mention (`002:77`) | 0.42 | 0.425 | 0.43 |

### Stale statements and process

- **011 code:**
  - The resume bookkeeping counts G0, quarantine and RA only for newly run pairs (`olens_swaps.py:226-259`).
  - The analysis raises KeyError on any quarantined row and has no completeness guard (`olens_swaps_analysis.py:38-44`).
  - `queue.sh` (lines 13-18; the subagent's `:183-188` was wrong) still uses v1; a rerun would overwrite `ar_validation_v1.json` (fixed 2026-10-06: the script now refuses).
  - Docstrings at `ar_vectors.py:10,13-15` and `olens_swaps.py:4,11` describe the setup before the amendments.
- **011 amendment header times** (`~20:50 / ~21:05 / ~21:30`) disagree with git (20:37:36 / 20:43:06 / 21:07:59). Git shows amendment 3 was committed before s1 started (20:43:53), so it predates the s1 outcomes.
- **007:**
  - The A3 rerun maximum (0.2549, `posthoc_A3_stage1b.md:8`) would have failed the original gate (≤ 0.25); this is not disclosed.
  - The 2-row scorer and its G-fastpath gate were adopted after r1's prompt-only floor had been seen, without an amendment.
  - The report header's "2.2 GPU-h" omits about 6.9 h of post hoc NLA runs.
  - §6 still says "Not shown: anything about the NLA".
  - `007_jsubspace.md:25` says "dimension-matched random"; the decision used Δ-matched.
- **009:**
  - `009_brewcode_derisk.md:122` says the Oracle whitener is not public (stale since 011).
  - `--max-partners 4` is missing from the report's §8.
- **006:**
  - `006:67` says the NLA's positive-control failure is a "blue" prior; the NLA's dominant answer is green (6/12). The blue prior is the Oracle's.
- **Mistakes log:** the `CLAUDE.md` entry says the Track B report had "no slips"; it had the two above.
- **Older stale text:**
  - `COME_UP_TO_SPEED.md:35` gives the branch as `005-amended-full-run`, and `:107` says membership "passes only at L60".
  - `SPEC_v2.md:45` says "nothing on test".

## 3. Next steps, ranked by information gained per GPU hour

| # | Step | GPU | Decision it informs |
|---|---|---|---|
| 1 | **AR J-bias check:** compare the share of AR-output energy that lies in 007's J subspace with the share for the true targets, using the saved 011 tensors (`outputs/011_olens_swaps/d2/diag_tensors.pt`) | 0 (CPU) | If AR outputs are markedly more J-concentrated than the activations they reconstruct, the Oracle's J dependence is plausibly a training property. 007 then needs reframing, and the PI can ask WSB about the loss. If not, 007 is strengthened. |
| 2 | **Like-for-like dissociation:** restate it with the dense numbers (docs only). Optionally, read the Oracle on the sparse k=25 J part vs the remainder at L40/L44 on 48 dev tables, to match 008. | 0, then about 0.5–1 GPU-h (rough) | Whether the claim to the TA holds on matched objects |
| 3 | **Finish 011 s1:** fix the resume bookkeeping first, then analyse under amendment 3 | about 1 GPU-min | Whether the AR colour edit beats random at n=82 (post hoc; s0b already shows the pattern) |

**009, a view to put to the PI.** The confound (z stays in the output tail) is a property of any nameable intermediate, so the old brew prompts have it too (006b). Going back to them does not remove it. Comparing reader D against the model's own output D in the same tables does.

## 4. Questions for the PI

1. Correct −0.15 to −0.14 for the TA, and restate the dissociation on matched objects?
2. Do you or the TA know what the AR's `loss_mix_lambda` / `val_jloss` are?
3. May PID 49691 be killed?
4. May I fix the docs and the 011 resume bug on the branch (CPU only, no rule changes to `CLAUDE.md`)?
