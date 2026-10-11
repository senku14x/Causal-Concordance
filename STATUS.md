# STATUS — 2026-10-11 (branch `015-oracle-hallucination`: 015 on top of `014-typo`, which holds 012 Stage E and the 10-09 patch on top of 014 → 013 → 012 → 008 → `origin/main` = 007)

*New session? **First read the Anthropic workspace paper** (Gurnee et al. 2026, `docs/papers/2607.15495.pdf`), then the paper
notes, then `CLAUDE.md`, as the PI's rule says. The newest work is 015 and 012 Stage E:
- 015: `docs/ledger/015_ledger.md`, `docs/design_specs/015_oracle_hallucination.md`, `claude_ovarida/` reports 45–48;
- 012 Stage E: `docs/design_specs/012E_ar_concept_swaps.md` §14–§18, reports 41–44 and 49.

The project is at a stopping point (PI, 2026-10-11). Earlier dated sections below are kept as written.*

## Closing note (2026-10-11)

Scope throughout: Qwen3.6-27B; WorkspaceBench brew (001–009) and multihop via Jan Chen's bank (012–015); the cells,
layers and samples named. Everything after 007 is EXPLORATORY unless pre-registered.

**What the project found**
- **A multihop lever.**
  - J-row coordinate swaps change Qwen's answer on 48/82 items at L24–47 and 38/82 at L24–35. Random targets: 5/82 and 1/82; logit-row swaps 0/82.
    `results/012_reader_concordance/analysis/stageA_summary.json` → `runs["j_swap_L24-47"].success`, `.random_control`.
  - It works mainly where the bridge is a named place: 44/49 vs 4/33 (POST HOC).
    `stageA_splits.json` → `bands["L24-47"].by_bridge.named_place`.
- **Readers follow working swaps, and so does the model's own output.**
  - AUC of the bridge shift for a working swap: Oracle L44 0.96, NLA L42 0.94.
  - The model's own read-cell output gives 0.97.
  - `stageD_answers.json` → `cells["oracle|L44|L24-35"].auc_bridge`, `model_output["L24-35"].auc_d_model_bridge`.
- **007 (pre-registered, confirmed on held-out brew tables):** the Oracle's z tracking at brew `:` L40 depends on 007's
  J subspace. P1: J beats Δ-matched PCA, test 4.28 [3.46, 5.09] nats.
  `results/007_jsubspace/analysis/brew_test_confirmation.json` → `oracle_L40.contrasts.P1`.
- **008 (brew):** the J part of a class-mean z swap moves the answer less than the non-J part: J − non-J −0.17 [−0.24, −0.09].
  `results/008_trackb/analysis/trackb_b1.json` → `readings.B["emit3|J_NP|raw|natural"].J_minus_nonJ`. Caveat 4 applies.
- **013:** the NLA's false claims help the NLA's own reconstructor: deletion costs +28.6 [24.0, 33.4] FVE
  points over random deletion, carried by misquotes (+35.8).
  `results/013_nla_hallucination/analysis/hallucination_all.json` → `primary_delta_R1_minus_C1`, `A1_kind_misquote.delta_clause`.
- **015, under the Oracle's own reconstructor:**
  - Deleting the Oracle's answer costs +14.5 [11.3, 17.9] over random clauses, and +4.3 [2.0, 6.7] over
    a position-matched cut. `oh_all.json` → `cells["oracle_texts|oracle_ar"].dA_RA_minus_CA`, `.dAP_PA_minus_CA`.
  - The Oracle's "wrong answers" are mostly Qwen's runner-ups at the read cell: +6.6 [5.0, 8.3] nats over other items' cells,
    median rank 7. `runnerup.json` → `by_state.wrong.all`.
  - Rearranging the Oracle's answer openings costs points whichever way they are moved: SWAP +3.6.
    `swap.json` → `cells["oracle_texts|oracle_ar"]`.
- **012 Stage E:** the Oracle reconstructor's concept differences act as edits at J-row size.
  - AR_full@J 37/82, J-removed 0/82, random-removed 41/82. `stageE_step4.json` → `full.J_size.all.arm_table`.
  - Per-layer clamps accumulate the persistent part of their direction to about k / cos(u′, u): ratio ÷ (1/cos) 1.01 (rrem) and
    0.97 (emrem). The persistent part of u lies in J (descriptive, 10 items). `stageE_clampcheck.json` → `summary`.

**What it did not find**
- **No evidence that any reader reads the hidden step beyond output-facing content.** At the answer token the model's own
  output tracks swaps as well as the readers (above). At the Stage I/J carrier end, readers saw nothing.
- **No reader-specific difference between the Oracle Lens and the NLA.**
  - 012: Oracle L44 − NLA L42 AUC +0.016 [−0.017, 0.063]. `stageD_analysis.json` → `oracle_vs_nla_auc.oracle_L44_minus_nla_L42`.
  - 015: no pre-named contrast met §7's rule. `oh_all.json` → `reader_contrasts_nla_minus_oracle.*.claim_holds`, all false.
- **No model-side J privilege on brew under a band-limited clamp (008).** The paper's §3.3 mediation result was not
  reproduced in that setting. Caveat 4 applies.
- **No "report" of planted concepts (005).** The measure was lexical priming at the answer slot.
- **No non-J lever from the Oracle's own vectors (Stage E S1).** J-removed arms flip 0/82 at both sizes. Caveats 2–3 apply.
- **No usable depth-2 brew-code task (009):** it is beyond the model without chain of thought.

**Open caveats**
1. The Oracle reconstructor (AR) loader is accepted post hoc for L36–60, directions only; the L20–32 retrieval gap is unexplained, and the exact loss and any J term are unknown. (`CLAUDE.md:342`)
2. Per-layer clamps accumulate the persistent part of their direction (report 49), so clamp arms matched in k are not matched in effective push; this qualifies 012E §17's "J-specific" reading and 008's J vs non-J clamps. (`CLAUDE.md:651`)
3. Report 44's J-specific reading compares arms at unequal effective push (emrem about 1.6×, Jrem about 0.7× AR_full along u); a J-removed clamp at matched effective push is untested. (`claude_ovarida/44_stageE_energy_controls_2026-10-10.md:122`)
4. 008's J vs non-J clamp comparison carries the same unquantified caveat. (`docs/reports/008_trackb.md:149`)
5. Readers' tracking of working swaps is matched by the model's own read-cell output, so it does not separate reading the bridge from reading what the model is about to say. (`CLAUDE.md:331`)
6. The multihop lever works mainly for named-place bridges, and bridge type and target (Japan) cannot be separated. (`CLAUDE.md:318`)
7. Stage E: the bridge edit and the answer edit flip largely the same items, so the bridge-vs-answer confound binds. (`claude_ovarida/43_stageE_full_2026-10-10.md:228`)
8. 015's reader comparison can only be made under the NLA AR, because the Oracle AR barely reads NLA texts. (`claude_ovarida/47_015_full_2026-10-11.md:64`)
9. 015's wrong-answer effect (+10.4) is not position-controlled. (`claude_ovarida/47_015_full_2026-10-11.md:41`)
10. 015's runner-up gate failed on 6 of 261 near-tied cells (result unchanged without them). (`claude_ovarida/48_015_followups_2026-10-11.md:98`)
11. The clamp check's reproduction gate failed as stated for AR_rrem and AR_Jrem (numerical, max 0.0019). (`claude_ovarida/49_clamp_check_2026-10-11.md:52`)
12. 013's random-deletion controls can delete a whole explanation (R1 12/474, RS1 75/474), biasing 013's Δ toward 0. (`docs/ledger/013_ledger.md:152`)
13. The reconstructors are each reader's co-trained critic; 013/015 measure what each critic rewards, not what Qwen computes. (`docs/design_specs/015_oracle_hallucination.md:299`)
14. A PI notes file cited in CLAUDE.md (013 vs CW-16b) was lost with an earlier instance and never backed up. (`CLAUDE.md:306`)
15. The OpenRouter key was pasted in chat and should be rotated; Stage E has no results-page section and 012 page v14 was never published. (`CLAUDE.md:506`)

**2026-10-11, 012E §18 clamp mechanism check (POST HOC, descriptive; report 49).**
- Per-layer clamps accumulate the persistent part of their direction. Median over 10 items of (L47 ratio to AR_full) ÷ (1/cos):
  AR_rrem 1.01, AR_emrem 0.97.
- AR_Jrem does not accumulate (0.71 of AR_full along u).
- Caveats added to report 44 and report 008; CLAUDE.md pitfall added.
- The reproduction gate failed as stated for rrem/Jrem (numerical; diagnosed in `gate_diagnosis`).

**2026-10-10/11, 015, the Oracle hallucination test with cross-scoring (branch `015-oracle-hallucination`; reports 45–48).**
- 013's pipeline on the 522 Oracle L44 readouts; every version of both readers' texts scored by both reconstructors.
- **Answer arm, Oracle texts × Oracle AR:** Δ_A +14.5, Δ_AP +4.3.
- **Primary Δ:** Oracle × Oracle AR +9.0 (wrong answers, not position-controlled); Oracle × NLA AR −0.8; NLA × NLA AR +28.6 (= 013).
- **Reader contrasts:** no reader-specific claim; the Oracle AR barely reads NLA texts.
- **Follow-ups (report 48):** answer swaps cost whichever way they go; the wrong answers are Qwen's runner-ups. Red-team corrections 1–3 applied.
- **Results page:** https://claude.ai/artifact/7JkKovrdEVXZsAoB7WY6jd. Judge spend $6.76.

**2026-10-10, 012 Stage E, the Oracle reconstructor's concept differences as edits (on `014-typo`; reports 41–44).**
- Brew was stopped by rule at step 2 (report 41). On multihop the class-mean lever failed (10/51), so J-row sizes are primary (report 42).
- **Full run (report 43):** AR_full@J 37/82, AR_Jrem@J 0/82, AR_rrem@J 41/82, AR_answer@J 37/82.
  The bridge-vs-answer confound binds.
- **§17 (report 44):** AR_emrem@J 45/82 (median Δm 8.43 vs AR_Jrem 0.42): "J-specific" by the pre-stated rule, qualified by §18.
- Generation spend $3.88 (`stageE_generation_phrases.json` → `total_api_spend_usd`).

**2026-10-07, 014 typo (branch `014-typo`; EXPLORATORY; `docs/ledger/014_ledger.md`).** Reading-only suite on WSB typo
(100 items; hidden step = the corrected word; no answer depends on it, so no causal parts).
- **Readers:** every reader names the word from almost every activation part (0.77-1.00; other-target null ≤ 0.02); κ is undefined.
- **J dependence:** removing 007's J subspace costs the Oracle 0.14 / 0.225 (L40 / L44) and the NLA 0.025. The paired Oracle − NLA difference is +0.115 [0.05, 0.19] / +0.20 [0.13, 0.275] (POST HOC). Same direction as 007 on brew `:`; not on multihop, where the NLA also loses the step without J (007
  test half 0.80 → 0.225; Cathryn's CW-14) (added 2026-10-10).
- **NLA hallucination test (013 pipeline):** reconstructor gate FVE 54.7%.
  - Deletion: +39.3 [31.8, 47.3], carried by misquotes (+42.2); invented context ≈ 0 (CI wide).
  - Rewrite: F − P +1.8 [−1.2, 4.6], unresolved (multihop +3.1).

**2026-10-07, 013 NLA hallucination test (branch `013-nla-hallucination`; EXPLORATORY; `docs/ledger/013_ledger.md`).**
- **Material:** 522 NLA L42 explanations of 012's multihop states, scored by nla-rl's final reconstructor (validated in Stage K: FVE 78.1% vs published about 77%).
- **Deletion:** deleting judge-flagged false spans costs 28.6 [24.0, 33.4] FVE points more than random deletion.
  - Misquotes of the prompt carry it (+35.8).
  - Invented context is padding.
- **Rewrite:** a factual rewrite beats a paraphrase by +3.1 [1.3, 4.9]. F − O on clean states +4.1 [1.8, 6.3];
  on Cathryn's items 31-100 +4.6 [1.6, 7.5] vs her CW-16b +5.4 [2.6, 8.0] (added 2026-10-09, POST HOC,
  `rewrite_by_state.json`).
- **Brew** failed the de-risk and was dropped.
- **012 Stage L:** the held country arrives at the carrier end between L48 and L52, matching the model's own output there.
- Page v12.

**2026-10-07, after report 34 (housekeeping; PI-approved).** Stale numbers corrected and missing ones added to the 012
analysis JSON (CPU reruns; existing values unchanged): J1 re-scored flips, R's 39/146, Q equal-opportunity naming and the
output-increment CI, a §20 paired NLA − Oracle contrast for R, and audit notes on the H and E0 JSON. Page v8. Details in the
ledger's correction block ("Additions after report 34").

**2026-10-07, 012 later (EXPLORATORY).**
- **Stages I/J (delayed question):** a swap before the question changes later answers (42/60; clue-only 36/60; random 0). Readers at the carrier end see nothing. At the computed clue end they track the swap (about 0.8, equal to the logit lens), but the model's own output there does too (0.69, above the pre-set bar).
- **Codex audit:** accepted and corrected. The F2/R position-mask bug is fixed and the NLA hint withdrawn; the specificity rule is now a paired contrast; E0 wording corrected.
- **NLA repo read in full:** pairing step 400 with the final reconstructor is fine (model card).
- Ledger `docs/ledger/012_ledger.md`; session prompt `docs/claude_reports/33_new_session_prompt_2026-10-07.md`.

**2026-10-07, 012 overnight (branch `012-reader-concordance`; EXPLORATORY).** J-row coordinate swaps on Jan Chen's ST multihop bank (all WSB items) change Qwen's answer on 0.585 of 82 clean-correct items at L24-47 (random 0.06; logit rows 0), mostly where the bridge
is a named place, a country or a geographic place (44/49 vs 4/33 other bridges; all 27 Japan targets flip; POST HOC,
`stageA_splits.json`; added 2026-10-09, named-place group 2026-10-10). The swap operator's direction is set by the clean state (toward_src edits all fail at L24-47). Readers on the swapped states (Oracle L40/L44/L52/L56, NLA L42, token lenses) shift toward the swapped-in concept far more when the answer changed (AUC 0.88-0.97; Oracle ≈ NLA), but the model's own read-cell output does too (AUC 0.96-0.97 at L24-35); beyond it, the readers' bridge shift keeps only a modest partial association (ρ 0.31-0.57 across readers and layers; corrected 2026-10-07, first written as ~0.3-0.5 from a printed subset). Details: `docs/ledger/012_ledger.md`; page https://claude.ai/artifact/NoevKvBkXNUTvtpnJPVsuq.

**2026-10-06, late: 011 s1 finished; AR J-bias check; 011 code fixes (POST HOC / EXPLORATORY; PI go on report 31).**
- **011 s1** (82/82 pairs; the last 8 ran in 1.05 GPU-min after the resume fix; gates pass: repro 82/82, R·A ≤ 7.2e-15,
  nothing quarantined). Amendment 3 verdicts (`results/011_olens_swaps/olens_swaps_analysis_s1.json`): the TA's
  whitened swap and the raw-coordinate swap are **not read** (answer-colour controls 0.016 / 0.050, below 0.15); the
  additive reading is **partial**: AR colour difference S 0.09 vs the model's class-mean difference 0.20 vs random 0.01
  (D_ar − D_classmean −0.10 [−0.15, −0.05]; D_ar − D_rand +0.06 [+0.02, +0.11]). The smoke's raw-coordinate 0.16 did
  not hold (0.005). The loader is not validated against the published AR (011 note, "Loader caveats"), but 012 E0's
  2026-10-09 acceptance (L36-60, directions only) covers L40/L44 (PI, 2026-10-10).
- **AR J-bias check** (CPU; `scripts/011_olens_swaps/ar_jbias.py`, rule pre-set in the script): **ambiguous**. The
  AR's outputs put 1.27× / 1.15× (L40 / L44) as much centred variance in 007's J subspace as the true activations do,
  above PCA-1,024 (0.77 / 0.85) and random (≈ 1.00). Too small to account for 007's Oracle result alone. What the AR
  config's `jfve` / `jloss` / `loss_mix_lambda` mean is **open** (no training code public; the Oracle's "joint
  bullet-FVE" reward most plausibly means bullets fitted jointly).
- **Code:** 011 resume bookkeeping covers every pair; the analysis has a completeness guard and quarantine handling
  (`tests/test_011_analysis_guard.py`); `ar_vectors.py` and `queue.sh` refuse to overwrite validation records.
- **Housekeeping:** the stale 007 queue loop (PID 49691, `gpu_queue_posthoc.sh`) was killed. Its 4 queued jobs had all
  finished by 05:39 UTC (the last, the multihop NLA run, exited rc=1); it was only polling its queue file.
- **009 patch profile** (`profile_p1.json`) is complete (n = 48 at every site); §5 of the 009 report matches it.

**2026-10-06, doc corrections from report 31 (`docs/claude_reports/31_onboarding_review_2026-10-06.md` §2).**
- **Three J objects; never mix them:** (a) the J-lens; (b) 008's sparse k = 25 "J part" of a class-mean direction;
  (c) 007's dense 1,024-dim "J subspace". The reader–model dissociation is now stated on (c) on both sides
  (`docs/reports/008_trackb.md` §1 "Reading"):
  - reader side: Oracle L40 D_z from J vs whole 5.76 vs 3.22 nats (dev, Stage 1a) and 5.03 vs 2.87 (brew test); rest
    of J 0.30 / 0.29;
  - model side: 007 Stage 2 dense-J swaps track the Δ-share at L42 and exceed Δ-matched random modestly at L40 / L44
    (+0.07 / +0.09 recovery), below the 0.10 margin; S_full 0.28 at L40;
  - 008's result on (b) is separate, consistent evidence. 008 vs the paper: "not reproduced under a band-limited clamp
    (3 cells × L40–48; the paper clamps at every position)", not "reversed".
- **Number slips fixed:** 008 J_CB J − non-J −0.14 (was −0.15); subset whitened power 0.09 (was 0.12); J energy
  3.3–8.7% raw / 0.5–2.0% whitened; J_CB at whitened norm −0.009 [−0.029, −0.000], below 0 (was "ties"); report 29
  model vs S 0.24–0.33; 002 NLA `:` mention 0.43; 003 z positive in 84–90%.
- **Held-out ledger (corrected; was "brew test split used once"):**
  - brew test (49 tables) has been read by the Oracle (pre-registered, AMENDMENT_02 Use 1), the template lens, the
    prompt-only floor, and the NLA (post hoc A2, 2026-10-06; recorded in AMENDMENT_02's ledger note);
  - 007 made a multihop split (items 1–50 dev, touched; 51–100 test), and its test half has been read by the Oracle
    (Stage 3) and the NLA (post hoc A2);
  - 009's 150 test tables per depth are generated, hashed and untouched.
- **007 scoping:** confirmed registered contrasts are P1, P2, P3, P5 at L40 and only P5 at L44; the 5.8 vs 3.2 headline
  is a dev descriptive; `</think>` is dev-only and secondary. "Reader-specific J dependence" holds at `:` only: at
  `</think>` the NLA also collapses without J (rest of J 0.21 vs rest of random 1.21). Disclosed in the 007 report: the
  A3 rerun max 0.2549 would have failed the original ≤ 0.25 rule; the 2-row scorer and G-fastpath gate were adopted
  without an amendment after r1's floor had been seen; the post hoc NLA runs took about 6.9 GPU-h beyond the 2.2.
- **009:** "110 of 112 probe sites" overstated: 77 sites (q0–q10) precede the differing token and are identical under
  both questions; z decodes under both at 33 of the 35 post-divergence sites, and already at q1–q2.
- **PI's post hoc facts (2026-10-06; CPU, hand-read of the mirrored readouts; `docs/reports/007_jsubspace.md` §9):**
  - multihop test, whole activation: each reader names the bridge on 36/40 items (regex agrees with a hand-read of all
    240 whole readouts); failures: most-populous-country (all three), Kremlin and IKEA / Guinness (Oracle L40),
    Kremlin / LEGO / Golden Gate (Oracle L44), Golden Gate / Griffith / Zeus (NLA). Agent spot-check from the jsonl:
    36/40 each, same items;
  - the NLA's random-slice bridge hits are mostly weather / airport boilerplate naming the right country (64/80 such
    readouts; 38 of its 43 hits inside them): the country's identity survives in the slice; this is not reading the
    question;
  - the Oracle's rest-of-J readouts often name a wrong country (only-wrong vs right: L40 26 vs 34; L44 41 vs 10);
  - brew `:` (8 tables): z is named in 0.34 / 0.41 / 0.38 of readouts (Oracle L40 / L44 / NLA) vs an off-path colour
    in 0.31 / 0.21 / 0.19, and is the first colour named in only 1–6 of 32. Brew results are likelihood-only.

**2026-10-06 (new session; branch `008-trackb-brewcode`; relaxed rules per the PI's onboarding prompt §7).**
- **Reports:**
  - 26: up-to-speed report and inconsistencies (multihop restatement share is 0.15–0.28, not "about half"; the Oracle
    Lens is WSB's A.9.2 oracle lens, not Karvonen; nla-rl ships an AR at step 600);
  - 27: qualitative readouts;
  - 28: proposal;
  - 29: 010 reader agreement.
- **008, Track B** (exploratory; 82 gated dev pairs; all gates pass; `docs/reports/008_trackb.md`): the paper's §3.3
  protocol on brew (class-mean z swaps clamped per layer over the emit cells × L40–48).
  - Full-direction lever S 0.21, flips 16%.
  - The J part carries S 0.02–0.03, the non-J part 0.21; the whitened swap gives J 0.07–0.10 vs non-J 0.11–0.14.
  - Clamping J coordinates barely changes the non-J effect.
  - → **no model-side privilege for the sparse k = 25 J part** at these cells and layers. ~~007's dissociation
    stands.~~ (corrected 2026-10-06, report 31: that paired two different J objects; the dissociation is restated on 007's dense J
    subspace above.)
  - The lens-coordinate swap (arm A) failed its positive control: answer-atom swaps give S 0.01.
- **010** (exploratory, CPU; report 29):
  - NLA–Oracle per-pair agreement is weak (r 0.24–0.30);
  - the Oracle tracks what the J lens reads (at `:`, r 0.33–0.63 for L40/L44; same layer 0.43–0.63) more than the
    model's next-token output (0.24–0.30); the NLA tracks neither strongly (J lens 0.30–0.37, model output 0.07)
    (corrected 2026-10-09: the ranges were from a subset of `agreement.json`; report 36 E10) (corrected 2026-10-06, report 31: was "mid-layer readers ...
    (r 0.63–0.80), not the model's next-token output (0.2–0.3)");
  - only the late Oracle (L56/L60) follows the model's output.
- **009, brew-code bank** (exploratory, reader-free; `docs/reports/009_brewcode_derisk.md`; design note
  `docs/design_specs/008_009_trackb_brewcode.md` Part 2 and §E-009; reader rules in `009b`):
  - **Gate:** depth 2 stopped early by the §E-009 rule (0/77 pairs after 30 tables); depth 1 passes 771 of 1,200
    pairs (241 of 300 tables).
  - **The output-facing check fails, so no readers run** (§2.5.1). With a code answer, z is still the top colour at
    the answer slot in 98% of renders (median rank 11; the model's own D +11.9 nats [11.7, 12.1], 240/240 tables;
    2026-10-09: "98%" and the mass figure have no committed JSON key; D is the `:` cell, 240 tables / 658 pairs; all 771
    pairs +11.3 [11.1, 11.5]), at
    ~0.05% total colour mass.
  - Probes decode z under both questions at 33 of the 35 post-divergence sites (corrected 2026-10-06, report 31: was "110 of 112 sites";
    77 sites precede the differing token and are identical under both questions; z already decodes at q1–q2). The patch profile reproduces brew's structure:
    `</think>` L40-44 has z decodable but no single-cell leverage (S 0.00-0.01).
  - **The PI's call:** keep the rule, run readers post hoc with the model's output as covariate, or redesign. The PI
    thinks the new prompts are weak and leans toward the old prompts and runs.
- **011, Oracle-space swaps** (the TA's tip via the PI; design note `docs/design_specs/011_olens_swaps.md` with
  amendments 1-3; **post hoc, exploratory**):
  - **AR loader:** failed the pre-set validation twice (v1, v2), passed under post hoc amendment 2 (v3). The head
    outputs a raw activation scored in the AR's whitened metric. On 16-token spans the FVE after one global scale is
    0.24 (published 0.22), and within-batch retrieval@1 is 0.86 (published 0.93). The output scale is per layer and
    expected under the scale-invariant loss (E0: 0.11 at L20 → 2.33 at L63, 0.38 / 0.42 at L40 / L44,
    `stageE0_ar_profile.json` `layers.*.white.global_scale`), so only centred colour vectors are used (corrected
    2026-10-10: was "the global scale is unexplained").
  - **Smoke s0b (2 pairs):** the TA's swap as given (AR vectors swapped in whitened coordinates) S −0.00, with a
    failed answer-colour control, so it is not read. Raw-coordinate swap 0.16 vs a size-matched random 0.09 (generic).
    Additive "minus AR(z), plus AR(z′)" at the reference's edit size: 0.10, vs the model's class-mean difference 0.23
    and random 0.01 / −0.02.
  - **Full run s1 stopped at 74/82 pairs at the PI's request** (resumable; unanalysed). *(Superseded: s1 finished
    82/82 and was analysed, see the 2026-10-06 late entry above.)*
  - **Next, pending the PI:** finish s1 (~1 GPU-min); phrase-based AR vectors (~10 GPU-min), since 1-token inputs
    reconstruct poorly (cos ≈ 0.1).

**2026-10-05, 007 — J-subspace split with controls (branch `007-jsubspace`; spec v2 2820e53 + A1 5cdafd9; report `docs/reports/007_jsubspace.md`; per-stage status `results/007_jsubspace/status/`; ~2.2 GPU-h, unattended under rule 12).**
- **Reader side, confirmed on the brew test:** the Oracle's z tracking at `:` L40 is carried by the J (token-readout) subspace — H_J supported on dev and on the 49 test tables (J 5.8 / 5.0 nats vs whole 3.2 / 2.9; rest of J 0.3); Raw and Visible contradicted at L40 and L44 on both splits; L44 unresolved by positive-control rules. Same pattern at `</think>` (001 patch share 0.01 at L42, 0.03 at L44; dev only, secondary). (corrected 2026-10-06, report 31: the
confirmed registered contrasts are P1, P2, P3, P5 at L40 and only P5 at L44; 5.8 vs 3.2 is a dev descriptive; the
`</think>` share was quoted without its layer.)
- **Model side:** J-subspace swaps track content share (L42) or exceed random modestly but not PCA-4 (L40/L44, below margin); in the adapted §3.3 test the J part of a class-mean z direction moves the answer less than its non-J part. → A reader–model dissociation.
- **The NLA is disabled for all of 007** (G-repro 0.41 > 0.25 after my 2-row scorer change; not repaired, rule 12).
  *(Superseded: 007 A2/A3 reran the NLA post hoc on 006's exact scorer; `docs/reports/007_jsubspace.md`.)*
- Secondary: multihop confounded by restatement (CW-12 rates reproduced); relational stopped after 4a (no high-leverage read-layer position; 12/50 gated); chain (exploratory) signal outside J at L56. ~~Brew test split used once under AMENDMENT_02 (opened this session).~~ (corrected 2026-10-06, report 31: brew test was read by the
Oracle, the template lens and the prompt-only floor under AMENDMENT_02, and later by the NLA post hoc; see the ledger at
the top.)
- **2026-10-06, post hoc NLA rerun (amendment A2, PI: "rerun properly"; 006's exact 24-row scorer; ~4.8 GPU-h):** G-repro now passes (max 0.15, D_z 1.862 vs 1.862). The NLA's z signal is *not* J-concentrated (dev: J 0.95 vs whole 1.86, rest of J 0.63 ≈ rest of random 0.80; test similar); its categories are unresolved/uninformative and none replicates dev → test. → J dependence is reader-specific (Oracle yes, NLA no) at `:` only (corrected 2026-10-06, report 31: at `</think>` the NLA also
collapses without J, as the A3 sentence below says). Stage 1b NLA first stopped at G-score1 (0.280 > 0.25), diagnosed as bf16 batch-layout noise (report 25), then rerun under amendment A3 (median/p99 gate): at `</think>` the NLA's signal collapses without J (rest of J 0.21 vs 1.21 for rest of random) but J alone keeps only half, and mean removal doubles it; H_J unresolved (post hoc, dev). Multihop NLA: J ≈ whole (as the Oracle). Results page: https://claude.ai/artifact/8vnKTzcXczn4YGMiP1uJTW

**2026-10-05, end of session.**
- `006-reader-likelihood` was merged into `005-amended-full-run` (fast-forward), so 005 and 006 now live on one branch, pushed.
- **HF:** the public dataset `senku21x/causal-concordance-outputs` holds every output, 005 and 006 included, with a verified MANIFEST and a `results/` snapshot of this branch.
- **The PI is analysing the current set of results.** No experiment is queued.

**2026-10-05, 006b, the PI's "one final run" (pre-registered at 3d9f6d6, run m1, 15.6 GPU min; design `docs/design_specs/006b_model_output_baseline.md`; section in `docs/reports/006_reader_likelihood.md`).**
- **Question.** Is the readers' z tracking just the model's own next-token distribution at the same cells (z as the runner-up answer)?
- **Gate** passes: max 0.147, median 0.043 nats.
- **The model's own D_z** (48 tables): `:` +3.76 [3.27, 4.24], `</think>` +3.99 [3.39, 4.57], assistant newline +3.66 [3.10, 4.22].
  - At `:` the answer ranks first in 100% of tables, and z is the runner-up in 56% (median rank 2).
  - B1 holds, B4 holds, B3 fails (the model's output tracks z at `</think>` too).
- **B2 (readers beyond the model): not testable.** Only 9 pairs have model D_z ≤ 0 (the minimum is 15). The secondary OLS intercepts are positive but extrapolated.
- **Per-pair reader–model r:** 0.07 (NLA) to 0.30 (Oracle L40).
- **Reading.** 006's z tracking is consistent with the readers reading the model's output-facing distribution (z as the runner-up answer). No held intermediate beyond it is shown.
  - The Oracle's 1.87× / 1.84× at `</think>` (L40 / L44; ratio of `posthoc.json` `emit_think.mean` to `emit_colon.mean`;
    corrected 2026-10-10: was "2×") is not mirrored by the model and stays unexplained.
  - D magnitudes are compared only within an instrument.
- **Also post hoc** (PI-forwarded review, `posthoc.json`, commit 70ba3e1): the readers' tracking does not follow 001's patch shares, and the Oracle's G-pc failure is a "blue" prior. The review's "`</think>` ≥ `:` for every reader" had the NLA's sign reversed (NLA −0.41 [−0.87, 0.04]).
- The page https://claude.ai/artifact/P35xhVkdzKjsdGhogp7PQY (v8) carries both.

**2026-10-05, 006 (now merged into `005-amended-full-run`; report `docs/reports/006_reader_likelihood.md`).**
- **Question.** At the brew `:` cell (saved 001 captures, 48 dev tables), do the verbalizers' likelihoods track z, as 002's lenses did?
- **NLA L42, likelihood:**
  - tracks z: +1.86 [1.41, 2.34] nats, 44 of 48 tables; supported (T1, Holm);
  - shuffled control ≈ 0;
  - it also tracks the start colour (+0.41) and the answer (+0.59), less than z.
- **Oracle L40/L44:** tracks z in 48 of 48 tables (+3.2 to +3.8). Its positive control failed at those layers (33% and 42% < 50%), so T2 is uninformative by the rule.
- **Bar:** the NLA passed its positive control exactly at the bar (6 of 12).
- **Reading.** z is decodable from the readers' likelihood at the cell where 002 found they rarely state it. **Qualified by 006b (above):** the model's own output at the same cells already favours z.

**2026-10-05: state at the end of the session (branch `005-amended-full-run`, pushed; not merged into `main`).**
- **005 is closed.** The PI ruled no further 005 variants.
- **Main conclusions, with the follow-up corrections applied:**
  - the model-side measure is lexical priming, not report;
  - no J-space privilege for it;
  - the clamp effect is output-direction blocking;
  - the logit-lens direction primes as well as the J atom;
  - on direct reads, probe > Oracle > NLA, and the token lenses detect nothing;
  - nothing readable arrives at the answer position at L40–44.
  - Details in CLAUDE.md, "Where things stand", and the reports listed there.
- **Mirrors (2026-10-05):**
  - all of `outputs/` (3.2 GB, incl. 005) is on the public HF dataset `senku21x/causal-concordance-outputs`, with MANIFEST.json and a `results/` snapshot;
  - the private dataset was not updated (the HF account's private storage is full); GitHub holds every branch, and
    `claude_reports` and the agent memory are in the repo (`docs/claude_reports/`, `docs/context/agent_notes/`);
  - script: `scripts/000_setup/mirror_outputs.py`;
  - third-party weights are not mirrored (they are pinned and re-downloadable).
- **Open:**
  - the PI merges this branch;
  - the next direction. Candidates are in the session's last report-back: back to the reader panel on WorkspaceBench cells with the likelihood readout from 005; the 004 §14 options.
- **Untracked:** `docs/design_specs/Untitled.ipynb` (the PI's notebook; left as is).

**2026-10-04 (branch `spec-v2`, now the working branch; repo private).**
- **SPEC_v2** is the unified spec; **errata 01** (V2-E1–E9) was accepted by the PI and wins over the text.
- **Checks C1–C3** (`docs/reports/checks_2026-10-04_no_gpu.md`):
  - CW-7 survives WSB's chain rule at a third of its size, and J-lens matches it;
  - ":" L42 keeps "influence" (a mixture of the two trajectories);
  - Cathryn's multihop removal control was ~2.4× smaller in energy.
- **NLA = `nla-rl` iter_000400** (`readers.NLArl`, sha256-pinned); **J-space pursuit** in `jspace.py`.
- **005 planted concepts**:
  - design v2 + the agent's pre-registered priors (`docs/design_specs/005_planted_concepts.md`);
  - smoke (`docs/reports/005_smoke_note.md`): G1 report gate failed (sampled naming floor) → stop rule;
  - dev-only pilot + extra steps 7b and 8–11 (`docs/reports/005_pilot_note.md`, design §16–§16.7; report page
    https://claude.ai/artifact/P35xhVkdzKjsdGhogp7PQY).
- **Pilot findings (dev, descriptive):**
  - the log-prob report measure works and is specific to the planted concept (step 10);
  - no clear J privilege at matched norm (holds under 5 J-space constructions, step 9);
  - the privilege clamp is specific only when it extends past about L48–50 (step 10: from L50 with band L32–38, L52
    with L24–38), where the concept's J direction is turning into its output direction; blocking the plain output
    directions does nearly as much (step 11), so it mostly blocks the answer itself;
  - under a likelihood measure the readers are not less sensitive than the model: the Oracle responds on the written
    cell at lower doses than the model's report, the NLA at about the same (step 11). The naming-based "less sensitive"
    preview is withdrawn;
  - readers name planted concepts on direct reads; at the report position only the Oracle at L52–60, with band
    L24–38, names it, for a few concepts (Q13). An earlier version of this line said nothing ever reaches the report
    position (written before Q13; corrected);
  - the relative probe criterion is degenerate (absolute criterion, A9).
- **2026-10-05:** the PI adopted amendment 01 (`docs/design_specs/005_amendment_01.md`, all items incl. A18); the amended full run on the 45 test concepts (`docs/design_specs/005_full_run.md`) runs on branch `005-amended-full-run`, so the adoption can be reverted by dropping the branch.
- **005 full run done (2026-10-05, run 20261005T015235Z, 110 min; report `docs/reports/005_full_run.md`; page https://claude.ai/artifact/5EGz2DxcAXXQs3wTtz1Lnu):** gates pass except G11 for the NLA (its likelihood measure fails the positive control, so P11′ for the NLA is untestable); P17 not supported (the Oracle names the non-J part more than the NLA); M1 not supported on R1′ (tiny positive on R1); M3 opposite of predicted; M2's J clamp to L50 is specific but not distinguishable from the output-direction control; P11′ (Oracle) supported (not perturbation-matched); R1 and R2 as predicted (R2 partly scale-free; post-hoc floor 29/45).
- **005 follow-ups (exploratory, `docs/reports/005_followups.md`):** the answer-slot log-prob is position-generic lexical priming (A2b), so 005's model-side "report" is renamed priming; the concept's own atom is in its J part for 1/45 concepts; the logit-lens direction is as potent as the J atom at s = 1/16–1/4 (Q-c); the per-concept probe is the best detector and the token lenses detect nothing on direct reads at L40–44 (A1). 005b's Q-a/Q-b not run (flawed as written).

**004, span assay at `:` L42: complete** (`docs/reports/004_span_assay.md`; 33 pairs, 55.8 min, commit 9da6b23 clean).
- Whole cell 0.43 of the gap. Recovery: 1-D z 0.06, z span 0.21, joint span 0.72.
- **Same-rank PCA does as well or better** (0.58 / 0.89; at matched norm the fitted spans are slightly behind).
- The z-span swap mostly raises the recipient's own intermediate (target-directed 4/33). Random controls are inert. The
  z-span clamp beats paired rotations in 31/33. The z span and its complement are non-additive.
- S2 and S6 hold; S1, S3, S4 and S5 do not. Descriptive geometry result; no intervention nominated. Next: report §14
  (default: back to the reader panel).


**003, reader-free causal labels: complete** (`docs/reports/003_labels.md`).
- Gates G1–G3 pass. **G4 (label reliability) fails** (κ 0.37, r(S) 0.41 between disjoint fits), so the labels are
  **descriptive only** and are not used to score readers.
- At `:` and the assistant newline, L40–44, the decorrelated z direction has a small, mostly positive effect: median
  S 0.02–0.05 (corrected 2026-10-09: was 0.03–0.05; the assistant newline L42 is 0.024, `003_labels/analysis/summary.json`), positive in 84–90% of pairs (corrected 2026-10-06, report 31: was 85–90%), above the start and answer directions. Clamping it beats random clamps,
  but those clamps were not size-matched. It is "used" (≥ 0.10) in only 9–18% of pairs (corrected 2026-10-09: was 10–18%; assistant newline L42 0.085). No single tested direction
  reproduces the whole-cell effect.
- At the `</think>` newline, z is readable at the family level and the whole-cell patch has little net leverage (not an
  item-level label). At `:` L60 the answer direction carries the cell. (Corrected after the audit of 001–003.)
- Next options in report 003 §7: subspace labels, DAS, more fitting data for reliability, or a descriptive 004.

**Latest (2026-10-02, while the PI slept):**
- **002, reader claim scoring: complete.** `docs/reports/002_claim_scoring.md`; design v2.2; external review accepted.
  - Judge: Gemini 3.8 Flash, as in WSB. 3,930 readouts, 0 errors, $7.98. The gate passed (κ 0.67 / 0.81 against
    blind author codes).
  - **The verbalizers rarely claim z in its role:** NLA 5–15%, Oracle 0–3%. 85% of Oracle L40 samples contradict
    themselves. Role-based tracking is weak.
  - **The token lenses carry activation-sensitive z information inside the candidate set at L40–44:** positive
    start-swap tracking (a difference-in-differences of scores) in 47–48 of 48 tables, though actual rank reversals
    occur in about half of the partner comparisons (90/183 for J_NP cosine at `:` L40). No tracking of the start.
  - Descriptive; no causal label.
- **001 probe repair (amendment A2): done.** 41k fits, 2 h CPU.
  - Membership controls now average 0.48–0.52 (they were 0.58, up to 0.84).
  - **The call changed:** off-path membership is decodable at the `:` cell at every layer, narrowly at L40–44
    (selectivity 0.11–0.13) and clearly at L60.
  - Residual checks: the span solver matches or beats the full solver across the C grid; the grouped 12-way null
    is wider but changes no call.
  - Report 001 §7/§9/§10/§12/§13/§15 updated.

**New here?** Read `docs/COME_UP_TO_SPEED.md` first; `docs/context/SPEC_STATUS.md` maps the whole spec to done / not
done. **The repo is private and `main` carries everything** (merged from `setup/000-scaffold` on 2026-10-02).

**External audit (Codex, 2026-10-02; `docs/reviews/2026-10-02_codex_reader_audit/`).** Every number recomputed
independently (`scripts/001_leverage_profile/audit_followups.py`) and matched. The report was revised in place (§15 has
the point-by-point table). Still open:
- **probe controls:** the membership control task is far from chance and the permutation null reuses C, so
  membership calls are provisional;
- **damage:** the readout cannot be decomposed (top 5 saved only);
- **readers:** scoring is mention-only, and there is no prompt-only baseline yet.

**Design 001 v3 (+ amendment A1) is running on dev material.** Done:
- `gate`, `smoke` (every exact check passes);
- `run` (72 min, SEC regime);
- `probes` (25 min);
- `analysis`, `figures` (reader-free outputs).

`snapshot` (readers, descriptive, 32 min) is done too. The report is complete: `docs/reports/001_leverage_profile.md`.
It awaits PI review; its proposals for AMENDMENT_02 are in §13. No label and no reader score exists yet, and no
test-split measurement has been made (true as of 2026-10-02; since then 007 read the brew and multihop test splits, see
the held-out ledger at the top; noted 2026-10-06, report 31).

| Block | State | Evidence |
|---|---|---|
| a. Repo | done | this file, `CLAUDE.md`, `README.md`, `docs/context/{SPEC,AMENDMENT_01,READING_LIST,reading_notes}.md`, `docs/reports/000_reading_report.md` |
| b. Environment | done | model loads (16 s, 50.1 GB); one brew forward gives the gated answer; 29/29 hashes match; `scripts/000_setup/rebuild_env.sh` |
| c. Reader smoke tests | done (5 work; SAE deferred, vLLM path parked) | table below; `results/000_setup/analysis/reader_smoke.json` |
| d. Shared code | done; **74/74 tests at `647dcb5`** (+4 added since, see Tests) | list below |
| e. Design 001 | **v3 + A1 approved**; all stages done; report complete | `docs/design_specs/001_leverage_profile.md`; `results/001_leverage_profile/{manifest_gate,manifest_smoke}.json`, `analysis/{gate_summary,gate_pairs,smoke_checks}` |

## At a glance (2026-10-02)

Results page (private artifact, generated by `scripts/000_setup/results_page/make_page.py`):
https://claude.ai/artifact/RMRX8R4iFnysbnHaNWTBeC

**Works (tested):**
- Pinned environment: the model loads, 29/29 hashes match, and the rebuild script recreates everything.
- WorkspaceBench rendering, byte-identical to WSB's producer, with position checks on every item.
- Partner builders (brew 361, chain 110, relational 50 candidates, all token-aligned), the off-trajectory substitution
  builder (100/100 items admit one; 1,138 valid (c, c′) choices), and the competence gate.
- Margin scorer, patching engine (RHUT float32-residual regime, the cross-check), **SEC true-fp32 replay (the 001
  label regime, L ≥ 40)**, gates, stats, figures, and the subject-model integrity guard.
- Readers on one smoke cell: logit lens, J_NP, J_CB, R_CB, Oracle Lens (L40), NLA AV + AR (L42). The NLA positive
  controls pass.
- The 001 pipeline `scripts/001_leverage_profile/leverage_profile.py` (stages gate → smoke → run → probes → snapshot →
  analysis → figures); every stage has run on dev.
- The results page, rebuilt 2026-10-02: the published copy had lost its middle (a template truncation, logged in
  CLAUDE.md). It now carries the 001 results, figures, reader grid and qualitative browser, and is checked in jsdom
  before publishing.

**Not working / blocked:**
- **SAE:** deferred by the PI (2026-10-02); no SAE arm until the PI decides what to do with it. Candidate located since:
  `ceselder/qwen36-27b-bsf-l42-1b` (block-sparse subspace featurizer, 8-dim blocks, k = 32, EV 0.51, no feature labels).
- **vLLM acceleration:** the engine fails to start on this GPU (sm_120 capability check inside FlashInfer; needs a
  CUDA ≥ 12.9 component). Parked by the PI; the HF route is used (64 NLA explanations ≈ 107 s).
- **Skip-lens:** dropped by the PI (AMENDMENT_01 §4). The ~22 `ceselder/skip-lens-qwen36-27b-*` checkpoints are
  trained and read at **layer 62**; "fed raw L42" would be off-distribution. Reinstatable on-contract at L62, or as a
  separate off-contract L42 arm, if wanted.

- **Optional readers (SPEC "Optional" line):** modulation lens and phrase-J dropped by the PI (2026-10-02). Template
  lens deferred: maybe later, not a priority. It exists for Qwen3.6-27B at the pinned
  `camilablank/workspace-lenses` @ d740106d (`qwen3.6-27b/template-lens/templates.safetensors`, 8.6 GB, 13,174 words,
  cosine-scored). Its vocabulary has 11 of the 12 brew colours (no teal).

**004 span assay (design v2, PI go 2026-10-03 "start with it, be careful") — implementation checklist:**
- [x] Raw-output mirror: public HF dataset with MANIFEST.json, retrieval verified.
- [x] Reader-free helpers factored into `scripts/common/brew_material.py` (identical 82 pairs and renders; boundary
  test extended).
- [x] `scripts/004_span_assay/span_assay.py` (reader-free; boundary test walks it): frozen 33-pair manifest (seed
  20261007, one per table); spans per fit with each fit's own mean; PCA per fit (within-table centred, 1/sqrt(n_t));
  arm manifest: 334 conditions, 476 scored prompts, ~956 forwards per pair (31,548 in all).
- [x] Preflight (CPU, `results/004_span_assay/preflight.json`): fit rows reproduce 003's n and means; basis identities
  float64/float32 (worst QᵀQ 3.6e-7 after the cast); zero writes bitwise no-ops; NEG_REL 1e-4 checked; the 24 accepted spelling
  sequences (12 colours × 2; only " Teal" is two tokens) are prefix-free; no reader or mixed modules loaded.
- [x] Smoke, 2 pairs (`smoke_checks.json`, `smoke_inspection.md`): all six gates pass (self-swap bitwise; identity
  projector vs direct assignment ≤ 3e-6 nats; S_full = 001 exactly; clean m and G = 003 exactly; fidelity on every
  write; M ≤ 1). 101 s per pair → ~56 min for 33. Outside-the-spellings mass (~14% clean) is formatting tokens
  (`<|im_end|>`, newlines, " **"; `smoke_outside_mass_tokens.json`).
- [x] External review of the smoke (`docs/reviews/2026-10-03_codex_004_smoke_review/`): S5 definition, G1 enforcement and
  fidelity resolution fixed before the run (design §13); smoke rerun (v2): all gates pass, 948/948 writes per pair
  fully checked; v1 smoke outputs kept as superseded.
- [x] PI go 2026-10-03 → run 33 pairs (55.8 min, every write passed fidelity) → analysis → report → page → mirror re-upload.

**Built and run since (see the top of this file and the reports):** label construction (003: directions, 1-D swaps,
necessity, random nulls), role-aware reader scoring (002: judge + closed set), prompt-only baseline (002), probe repair
(001 A2). **Not built yet:** content-span and complement patches, norm-matched ablation controls, per-item null
calibration, DAS, relational gating, any test-split measurement (as of 2026-10-02; superseded by 007, see the held-out ledger at the
top; noted 2026-10-06, report 31).

**Next (2026-10-02, after the audit of 001–003, `docs/reviews/2026-10-02_codex_audit_001_003/`):**
1. Done without the GPU: the audit's corrections (report 003 §9; the 002 tracking wording; this file).
2. Proposed, awaiting the PI's go: a small, separately designed **L42 assay check**:
   - 1-D z edit vs a z span vs the joint content span vs **their complements** vs the whole cell, with matched-rank
     and matched-norm controls;
   - two disjoint fits;
   - norm-matched ablations with damage logged;
   - the start-token positive control kept.
3. Still pending from earlier: PI review of reports 001–003; AMENDMENT_02 (τ, ε′, substitution bound) before any
   test-split label; the not-needed scoring-layer decision.

## 001 — gate and smoke (dev)

Gate (`manifest_gate.json`, run 20261002T074329Z at `5d5d1a3`, **dirty tree**; 621 distinct prompts, 24 min):

| Family | Split | Candidates | Passed | Items (groups) with a passing pair |
|---|---|---|---|---|
| brew start pairs | dev | 183 | **82** | 33 of 48 items |
| brew start pairs | test | 178 | 77 | 35 of 49 items |
| chain start pairs | dev | 53 | **48** | 12 of 13 items; **9 of 10 groups** |
| chain start pairs | test | 46 | 37 | 8 of 11 items; 7 of 10 groups |
| brew substitution | dev | 50 | **43** | 43 of 50 items |

- Design rule 4 (≥ 20 brew, ≥ 15 chain dev pairs) is satisfied; chain support is still thin and is reported by group.
- Pass rate differs by z (brew dev: green 4/24, gold 2/10 vs teal 12/19, gray 11/20): the gated set is not z-balanced,
  so the difficulty distribution must be reported next to any pooled number (AMENDMENT_01 §1).
- Chain dev passing pairs are dominated by two z classes ((5,) 22 and (9,) 12 of 48).
- 3 brew items have no partner candidate at all (97 items carry candidates).

Smoke (`manifest_smoke.json`, run 20261002T080920Z at `5d5d1a3`, dirty tree; 76 s): **all engine checks pass**
(self-patch, embedding anchor, pre-difference prefix and stir no-op bitwise) on 3 brew pairs (all three partners of
`brew-s2-r10d2-0009`), 1 chain pair and 2 substitution pairs. Observations, single pairs, not findings:
- brew emit_colon whole-cell share rises with depth: 0.11–0.51 at L40–44, 0.62–0.69 at L56–60; the start-colour
  token at L0–4 carries ≈ 1.00 (the anchor), the emission span 0.89–0.91, start + question + emission 1.00.
- chain final cell: ≈ −0.03 at L40–44 (negligible), 0.72–0.74 at L56–60; the input digits span and the question span
  each carry ≈ 1.00 jointly across L40–60.
- substitutions `0000 gray→teal`, `0001 blue→gold`: behaviour |Δlog P(a)| 0.009–0.023, TV ≤ 0.012; transplants
  ≤ 0.022; L0 key-token reference 0.002–0.064. All far inside the provisional bound (0.05 · median G ≈ 0.45).

Provenance note: both manifests record `git.dirty = true` at `5d5d1a3`. The uncommitted diff (committed as `647dcb5`)
touched only the `analysis` and `figures` stages and two CLAUDE.md log rows, not the gate or smoke code, so the code
that ran is the committed code; the manifest alone cannot show that. Run stages from a clean tree from here on.

## Environment

Full details in `CLAUDE.md` § Environment. RTX PRO 6000 Blackwell (sm_120, 96 GB); torch 2.11.0+cu128, transformers
5.15.0, peft 0.21.2, jlens @ 581d398; `Qwen/Qwen3.6-27B` @ 6a9e13bd (bf16, eager).

- **Hashes** (`results/000_setup/analysis/artifact_hashes.json`, 29/29 match):
  - J_NP, J_CB, R_CB lens files against the sha256 in RHUT's `instrument_registry.json` (which pins only these three
    files plus the model revision);
  - the 15 model shards, the Oracle adapter, the NLA adapters, critic and value head, and the NLA reference parquet
    against Hugging Face's LFS sha256 at the pinned revisions;
  - the 5 WSB bank files against the registry.
- **Env check** (`results/000_setup/analysis/env_check.json`): on `brew-s2-r10d2-0000` the greedy answer is " black"
  (gold: black). Top-5 next tokens: " black" −0.31, " Black" −2.31, " green" (= z) −2.68, `<|im_end|>` −4.06,
  " Green". The runner-up is the intermediate colour (WSB's runner-up caveat). One 165-token forward takes 0.23 s.

## Reader smoke tests (block c)

One cell: `brew-s2-r10d2-0000` (start blue → z green → answer black), emit_colon = the ":" after "Answer", position 164.
Activations come from the stock bf16 forward (WSB convention). The verbalizers sample 4 times with WSB's sampling
(T = 1.0, top_p 0.95, top_k 64, ≤ 256 tokens).

| Reader | Status | Load + run | GPU peak | Raw output (abridged) | Names colours? |
|---|---|---|---|---|---|
| Logit lens (L40/42/44) | **works** | < 1 s | 50 GB (model) | L42: ' ', ' (', ' either', '...', '\n\n' | no at L40–44. Positive control: at L63 top-1 is ' black' (= model greedy); colours appear from L56 |
| J_NP (standard + WSB cosine) | **works** | 16.9 s for all three J lenses (3 layers, sha-checked) | 56.4 GB | standard L42: '\n\n', '<\|im_end\|>', '<\|endoftext\|>'; cosine L42: ' END', ' FINAL', '입니다', '_final' | no at L40–44 at emit_colon (the cosine readout names green at L36 emit_colon and L40 emit_think); at L56: ' black' top-1; at L60–62: black, green, brown |
| J_CB (standard + cosine) | **works** | (above) | (above) | standard L42: '终点' (end point), '第二次' (2nd time), '不会再', '终结', ' downstream' | no at L40–44; yes from L56 |
| R_CB (standard + cosine) | **works** | (above) | (above) | standard L42: ' ...', ' either', '第二次', ' intermediate', ' second' | no at L40–44; yes from L56 |
| Oracle Lens, L40 (on-contract) | **works** | 32 s (adapter + 3 arms × 4 samples) | 51.3 GB | "- Green (Since after two steps from Blue, we reach Green, …)", "- blue (or green) …"; no activation: Korean company names (㈜ = the corporation sign); another cell (stir "You"): liquid-colour puzzle set-ups | **yes**: green, blue, red, purple, gray, black (start and z among them) |
| NLA-L42 AV | **works** | 156 s (SFT merge + RL adapter; 3 arms × 4 + 8 reference rows) | 52.5 GB | "Table structure … two successive applications of a lookup table … (which is green/…)"; 4/4 parsed, 1 injection per generate; no activation: generic web text (car audio, patents); other cell: colour-rule puzzle set-up | **yes**: z (green) in 4/4 samples, one also pink/blue/red |
| NLA-L42 AR | **works** | 23 s | 35.2 GB | FVE of the cell from its own explanations: **0.46, 0.55, 0.59, 0.50**. No-activation explanations: −0.15 to −0.48. Other-cell explanations vs this cell: −0.41 to −0.66 (vs their own cell: 0.56–0.65) | n/a |
| NLA checks | **work** | — | — | Our L42 capture re-extracts the NLA's stored reference activations at cos ≥ 0.99984. AV → AR FVE on 8 reference rows: 0.75–0.86 (card: 74.5% held-out) | — |
| SAE | **deferred** | — | — | candidate located (`ceselder/qwen36-27b-bsf-l42-1b`, no labels); PI decision pending | — |
| vLLM path (optional) | **parked** | — | — | engine init fails on sm_120: `Failed to get device capability: SM 12.x requires CUDA >= 12.9` → `RuntimeError: FlashInfer requires GPUs with sm75 or higher`. Likely cause: a CUDA-12.8 component in vLLM 0.21's stack misreads this card (vllm-metamodels was tested on B200 / sm_100 only). HF reference timing: 64 NLA explanations in 107 s | — |

Notes (smoke observations on one cell, not findings):
- No lens reads a colour at emit_colon L40–44 here, though the verbalizers name z; J does name green at L36
  (emit_colon) and L40 (emit_think), so the working hypothesis is "the methods surface the intermediate at different
  cells", not "J is blind".
- The FVE reference set is the NLA's own (FineWeb text), not this task's distribution.
- The Oracle adapter is the public copy `olens_s3d_rl600`. That it equals WSB's private `s3d.ddp600.s0/iter_000600`
  is assumed from the names and cannot be verified.

## Tests (block d)

`python -m pytest scripts/common/tests -v` (log `outputs/logs/pytest_full.log`): **74 passed, 0 failed at `647dcb5`**
(47 CPU + 27 GPU). Added since (not yet run on the GPU box): 2 CPU tests for the resident guard (pass) and 1 GPU test
for the SecReplay park/restore lifecycle; 1 CPU test for the T(a) substitution tag (pass). Collected total: 78.

| Module | Tests | Result at 647dcb5 |
|---|---|---|
| cells | round-trip (json, jsonl); validation (layer, half, split, family, position, empty ids; 6 cases); unknown keys rejected | 8/8 |
| wsb (rendering) | bank hashes and sizes; `brew-s2-r10d2-0000` regions (stir 131–136, start 138–143, question 145–153, emissions 158/162/164, " blue" at 142); all 100 brew items' 24 positions; chain " What" + final "\n\n" and relational "'s" on every item; byte-equality with WSB's own `_chat` renderer | 5/5 |
| partners + gate | brew trajectories reproduce the bank; hand-checked brew partner (0000: pink valid, gray excluded by z = a′; 165 tokens, diff only at 142); constraints hold for all 361 candidates; chain step functions reproduce all 120 items; hand-checked chain partner (0408, start 19 → 14: z (9,4) → (7,3), one digit differs); 50 relational pairs aligned (2 diffs); answer parsing (8 cases); hand-checked substitution (0000 gray → teal); every item has a substitution, deterministic by seed; **T(a) tag never filters and leaves the 001 seeded draw unchanged (new)** | 16/16 (+1) |
| margin | spelling tokenisation (" Teal" two tokens, "27" two digits); sign implied by greedy (brew and chain, member < 0 < partner); multi-token teacher forcing scores every token, with an off-by-one discriminator; shared first token ("21" vs "27") → margin from the second token; writes applied on every forward | 5/5 |
| engine | self-patch bitwise (5 cells incl. L2, L63, multi-position); 1-D swap ρ = κ = 1 in float32 with read-back < 1e-4 and exactly one row changed; swap along unit(h_p − h_m) equals the full patch at the write (rel 2.7e-8); last-layer last-position replacement reproduces the partner's logits bitwise; earlier layers and positions untouched bitwise; the stock bf16 regime distorts a small write (ρ 1.04, κ 0.91) | 10/10 |
| sec engine (SEC regime, hand-checked pair) | self-patch bitwise; stir cell patch bitwise no-op; prefix identical before the first difference; embedding anchor reproduces the partner; last-layer replacement reproduces partner logits; lower-stack write lands and moves the answer; SEC score matches stock sign and scores multi-token spellings; SEC noise floor < 1e-3 nats | 8/8 |
| splits | brew 50/50 by table stratified by z with forced-dev items; chain groups intact across the split | 2/2 |
| gates | fidelity thresholds; damage-gate logic; span gate on the hand-checked pairs; self-patch damage is exactly 0; sequence-length determinism reported (not bitwise, onset at block 3, no leak) | 5/5 |
| stats | cluster-t vs scipy; BH-FDR vs statsmodels (exact); Holm vs statsmodels; cluster bootstrap covers and is wider than the naive interval; ratio of means; empirical p; equivalence | 6/6 |
| figures | ICLR figure round-trip (pdf + png); distinct reader colours | 2/2 |
| import boundary | no label module (registry, cells, wsb, partners, margin, engine, gates, stats, regime, splits) reaches readers/jlens/peft/vllm, transitively and including imports inside functions; sanity: readers does reach jlens | 2/2 |
| model integrity | Engine, SecReplay and capture_stock refuse a model a reader merged weights into (3 cases); Oracle attach → capture (adapter disabled) → unload leaves the subject's activations and logits bitwise unchanged; mutation after construction refused at use time; **resident guard: parked block refused, `blocks_upto` honoured (new, CPU); SecReplay-parked model refused by Engine / gate / capture and restored by the context manager, also on an exception (new, GPU, unrun)** | 5/5 (+3) |

## Regime check (`results/000_setup/analysis/regime_check.json`)

The same 5 edits under RHUT's float32-residual regime and SEC's true-float32 replay (blocks 40–63 in float32, resident,
72 GB). Pair: the first gated pair in bank order, `brew-s2-r10d2-0001__blue` (purple→green→orange vs blue→black→pink).
The hand-checked `0000__pink` failed the gate (partner 5/10) and `0001__black` failed too (partner 8/10).

| Edit | RHUT from block 0: Δm | SEC: Δm | \|diff\| / G_SEC |
|---|---|---|---|
| full patch, emit_colon, L44 | +4.770 | +4.426 | **0.033** |
| full patch, emit_colon, L40 | +2.437 | +2.220 | 0.021 |
| full patch, start-colour token, L40 | +0.559 | +0.526 | 0.003 |
| 1-D swap at emit_colon L42 along the emit_think pair difference | +2.032 | +2.123 | 0.009 |
| 1-D swap at emit_colon L42 along a random direction | −0.015 | +0.001 | 0.002 |

- G: RHUT-from-0 10.63, RHUT-from-35 10.52, stock bf16 10.31, SEC 10.48 nats.
- Max |diff| / G against SEC: RHUT-from-0 **3.3%**, RHUT-from-35 2.3%, stock bf16 1.3%.
- Under the 5% rule RHUT passed, and AMENDMENT_01 §3 kept it as the default; **errata E4 and design 001 §9.1 then
  adopted SEC's true-fp32 replay for every label measurement at L ≥ 40**, because the RHUT absolute floor
  (0.05–0.25 nats) is too coarse for ε′ = 5% decisions. The RHUT regime (`engine.Engine`) remains the cross-check.
- RHUT writes are exact (ρ = κ = 1); stock-bf16 random-direction writes are not (ρ 1.04, κ 0.86).

**Noise floor**: |Δm| under 8 perturbations of 1e-4 relative norm at emit_colon L42.

| Regime | Max \|Δm\| (nats) | Fraction of G |
|---|---|---|
| RHUT, from block 0 | 0.053 | 0.50% |
| RHUT, from block 35 | 0.034 | 0.33% |
| Stock bf16 | 0.064 | 0.62% |
| SEC | **0.0002** | 0.002% |

- On `0000__pink` (G 8.6), the RHUT floor reached 0.24 nats (2.7% of G) at the start token at L2.
- Appending 3 tokens moved that pair's margin by 0.22 nats.
- So the RHUT floor is absolute, roughly 0.05–0.25 nats, and on pairs with G below about 7 nats a gap the size of the
  observed 3.3% would cross 5% of G.
- SEC's own STATUS (L45–46) made the same observation independently: bf16-matmul finite differences give cos 0.80.

## Other block-000 findings (engineering)

- **Sequence length.** Activations at prompt positions depend on the total length. Blocks 0–2 are bitwise identical;
  the onset is at block 3, the first full-attention block, at 6e-4 relative. That grows to ~1% median by block 63, and
  up to 40–99% at a few positions. It is rounding, not a leak. Consequence: comparisons are made at one length only,
  with partner sources captured at the matching length.
- **Bank.**
  - Brew: 361 partner candidates, all token-aligned; per-item counts 0–8; 3 items have none.
  - Chain: digits are single tokens, so only 28/120 items have single-token intermediates. Of those, 25 have partners
    (110 candidates, all aligned). There are 15 cross-item prompt collisions (identical prompts from items sharing step
    sequences), so chain splits by step sequence (20 groups, 10/10) and de-duplicates pairs.
  - Relational: 50/50 a/b pairs aligned (gating pending).
- **Timing** (`timing_smoke.json`), used for design 001's budget:
  - forward ≈ 0.23 s whether T = 78 or 165; SEC replay of blocks 40–63 ≈ 0.09 s; SEC build 8.5 s;
  - gate ≈ 5.4 s per brew pair, 4.4 s per chain pair (measured: 621 prompts in 24 min);
  - HF NLA generation: 64 explanations of ≤ 256 tokens in 107 s.

## Code guard added after `647dcb5` (this update)

`SecReplay.build()` parks the bf16 upper blocks on the CPU until `restore()`. Every stock-forward entry point
(`Engine.forward`, `partners.gate_member`, `readers.capture_stock`, the Oracle and NLA constructors) now calls
`registry.assert_subject_model_resident`, and `SecReplay` / `SecEngine` are context managers that restore on exit,
also on an exception. `SecReplay.capture` checks only blocks ≤ l0. The 001 pipeline already restores at the end of
`smoke` and `run`, so no run was affected; the guard turns a silent misuse into an error.

## Open questions for the PI

*(Dated 2026-10-02/03 and kept for the record; the live list is `CLAUDE.md` "Open PI decisions".)*

Resolved in design 001 v3 §9: regime (SEC for L ≥ 40), reader snapshot (included, after hashing), chain split (50/50 by
group), reader-input bridge (checked on the snapshot). Still open:

0. **(2026-10-03, most pressing)**
   - confirm the framing: native reader-configuration comparison is the primary line, causal work a validation branch;
   - the next step after 004 (report 004 §14);
   - review reports 001–004;
   - rotate the OpenRouter key.
1. ~~HF dataset name~~: done. The public `senku21x/causal-concordance-outputs` and the private
   `senku21x/causal-concordance-private` (CLAUDE.md, "Fresh instance").
2. ~~Paper PDFs and web sources~~: committed to the private repo (`docs/papers/`, `docs/sources/web/`; PI 2026-10-03).
3. **P5 prior**: the dated update accepted in AMENDMENT_01 §5 needs your number (suggested ≈ 35%, down from 55%, on the
   NLA literature in the reading report §4.5.6).
4. **SAE arm**: use `ceselder/qwen36-27b-bsf-l42-1b` (no labels: LLM-label ~2,000 features, or score decoders by
   non-J means) or keep the SAE deferred?
5. ~~Branch~~: `main` fast-forwarded to `setup/000-scaffold` on 2026-10-03 (PI).

## External reviews (2026-10-02)

Review 1 (of `c13fd43`): every concrete claim was checked against the repo, and all hold. Accepted: design 001 v2 (its
change list), AMENDMENT_01 errata E1–E4, the model-integrity guard and its tests. Two corrections to the review
discussion: J does name green at L36 (emit_colon) and L40 (emit_think) on the smoke item, so "no colours at L40–44"
held only for the emit_colon cell; SEC removes the reader-input bridge only for layers ≤ 39 (001 §9.4).

Review 2 (of `56cb7ab`): integrity check at use time; cross-fitted substitution scoring; stir no-op scoped to start
pairs plus a pre-difference-prefix identity check; snapshot on 10 distinct tables. Accepted → design 001 v3, approved.

## Next

*(Dated 2026-10-02/03 and kept for the record; the live list is `CLAUDE.md` "Open PI decisions".)*

001 `run` → `probes` → `snapshot` → `analysis` → `figures` → report 001 → AMENDMENT_02 (thresholds frozen, 002
critiques recorded) → design 002 (labels) → PI approval.
