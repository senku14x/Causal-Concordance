<!-- Saved verbatim from the PI's message of 2026-10-02T04:54:06.308Z (UTC). PI directive after the reading report (records the decisions on report 000 and sets block 000 a–e; source of the non-negotiables in CLAUDE.md). -->

This message records the decisions from your reading report and sets out the next block of work: scaffolding the
repo and the shared code, smoke-testing the readers, and drafting the first design spec. Nothing scientific runs
yet. Where this conflicts with anything earlier, including the spec, this wins; record the conflicts in
docs/context/AMENDMENT_01.md (section 6).

Your reading report was good. Most of its §5 is accepted below. Where I've modified a recommendation, the reason is
given.

==================================================================================================================
1. WHICH TASKS: WorkspaceBench only
==================================================================================================================

The project's core question is how consistent the readers are, what each catches that the others miss, and whether
ensembling helps. That only needs "used" vs "absent" labels on tasks with known counterfactuals, and WorkspaceBench
families provide those. RHUT organisms are not the base of this project. They come back only as a fallback (see 2).

Per family:

- BREW (core). Build partners: same table, different start colour, so z, the intermediate after the first stir,
  changes.
  - Constraints: z' != z, a' != a, z' not in {a, a'}, z not in {a'}; both start colours single-token in context
    (" X."); identical token counts, so every position aligns.
  - Gate both members of every pair: greedy correct plus >= 9/10 correct at T = 0.8. An item enters only if both
    members pass.
  - Report how many candidates were built, how many passed, and the difficulty distribution of the survivors
    (e.g. WSB gate margins, table depth patterns). We must not end up keeping only the easy tables.
  - Re-verify WSB's 24 pinned positions against our render with the pinned tokenizer (6a9e13bd). WSB's backend did
    not pin a revision.
  - Keep the stir cells. The start colour hasn't been read there, so they are a label-free negative control.
  - Carry WSB's own caveat into every brew claim: the intermediate is often the model's runner-up answer colour.

- CHAIN (core). Check how this tokenizer splits numbers before anything else.
  - Keep only items whose intermediates are single tokens. Otherwise token readers lose by construction and the
    result says nothing about the workspace.
  - Build partners by changing the start number while keeping the steps fixed, with the same constraints as brew
    (distinct z, a; identical token counts) and the same gate.
  - Read cell: the final token of the chat render, as in WSB, plus the question-span cells for the leverage profile.

- RELATIONAL MULTIHOP (core, subset). Use only the a/b pairs whose two relation words have identical token counts,
  so the pair is token-aligned. Gate as above. Report how many pairs survive. If fewer than ~30, demote it to
  secondary.

- MULTIHOP (later). It has no donors; cue-swap pairs must be built. Plain render, no chat template, read at the last
  word. Never pool it with RHUT's two-hop, which is chat-rendered inside a copy task.

- ROLE-BOUND (agreement only). The ab/ba members are different stories, so no token-aligned partner exists and no
  label can be computed. Readers still run on it for the agreement atlas.

- ARITHMETIC (not now). If added later, drop the fraction variants, where the intermediate is ill-defined
  (e.g. "4.333333").

==================================================================================================================
2. WHERE THE "NOT NEEDED" CONTRAST COMES FROM
==================================================================================================================

The per-reader selectivity W = P(hit | used) - P(hit | not needed) only means something if used and not-needed
content are compared in the SAME activation. Otherwise W measures position or family (your §5.2 and §5.4).

Brew gives this natively. Each prompt lists about ten table colours, but only start -> intermediate -> answer lie on
the path. The off-trajectory colours (the item's `off_trajectory` field) are in the prompt but play no part in the
computation. At a brew emission cell, the gold intermediate is the candidate "used" content, and the off-trajectory
colours are candidate "decodable, not needed" content, if they are decodable there.

So the first thing to establish (in design 001) is whether the off-trajectory colours are decodable at the emission
cells:
- use an item-grouped linear probe with a Hewitt-Liang control task;
- if they are decodable, brew carries W;
- if they are not, they are "absent", brew has no not-needed class, and we fall back to RHUT's tagged-selection
  organism with the codebook consumer: pointed word vs unpointed words at the same question/decision cells.

Don't build the fallback unless 001 says we need it.

==================================================================================================================
3. LABEL DEFINITIONS AND YOUR OPEN QUESTIONS
==================================================================================================================

- LABEL UNIT. One token position at one layer. Every reader reads exactly that, so the label must refer to the same
  object. Span-level patches (whole question line, all emission cells) are useful for understanding where leverage
  lives, but they are context only, never labels.

- USED vs NOT NEEDED vs UNRESOLVED (your §5.1, accepted with numbers).
  - used = S >= tau AND S above the norm-matched random-direction null, after BH-FDR across all cells in the family.
  - not needed = decodable AND full-cell share < eps', with eps' < tau, so the two cannot overlap.
  - Everything in between, and every case your §6 table already lists, is unresolved.
  - Flag any cell where the 1-D share exceeds the full-cell share (the subspace-patching illusion signature).
  - Provisional tau = 0.10, eps' = 0.05. You may revise both once, on the dev split, using the leverage profile.
    Freeze them in an amendment before any test-split label is computed.

- TEMPLATE HALVES (your §5.6). WSB families have one template, so "halves" means halves of items, chosen so the
  direction is fitted on different content from what it labels:
  - brew: halves by table, stratified by z;
  - chain: halves by start number, stratified by z;
  - relational: halves by name set.
  Cross-fit (A->B and B->A) so every item is labelled. Reliability = agreement between labels computed from two
  disjoint fitting subsets for the same items. Gate: >= 0.7.

- WHAT THE DIRECTION CERTIFIES (your §5.3). Accepted. For 001 and the first label run, call the label
  "pair difference used", not "z used". Score readers on a four-way confusion, {input, z, answer, other}, so a
  reader naming the answer at a decision cell is recorded as naming the answer, not as missing z. Where a family
  allows decorrelation, also fit a decorrelated direction and report the share kept after projecting out the answer
  and input directions:
  - brew: z occurs across many tables with different starts and answers;
  - chain: different starts can reach the same intermediate.

- NECESSITY TIER (your §5.8). Pooled mean = the mean of u.h over the family's fitting half at that cell. Add a
  random-direction ablation control. "Used + necessary" requires N_i above that control's 95th percentile. An
  INLP/LEACE-style full removal is optional, later.

- DECODABILITY (your §5.7). Regularised linear probe; folds grouped by item and table (never by template alone);
  Hewitt-Liang control task; decodable = the item's held-out probability for the true class beats the control-task
  distribution. Binary z-vs-z' probes are fine if 12-way is underpowered. The planted-concept positive control must
  not use the probe's own direction.

- DAMAGE (your §5.9). For every label edit, record the change in log P(a) and log P(a') separately, plus top-1
  retention on non-answer positions. An edit whose margin change comes with broad damage is unresolved.

- PRECISION REGIME. Default: RHUT's fp32-residual regime on transformers 5.15.0. It is validated on this model
  (realised writes rho = kappa = 1.000) and runs in one phase. SEC's true-fp32 replay was needed for finite
  differences at eps = 1e-3, where bf16 noise dominates. Our swaps have natural magnitudes, so the bf16 matmul noise
  should be small relative to the effect. Verify that: on ~5 cells, compare Delta m in both regimes. If they differ
  by more than 5% of G, switch to SEC's engine. Clean margins are always computed in the same regime as the edits.

- READERS (not needed for 001; settles the reader set):
  - Oracle Lens: read on-contract at L40 and L44, against labels computed at those layers. An off-contract L42 arm
    gets its own planted-concept curve and is reported separately, never pooled. Its AR analyses depend on the
    whitener moments, which are not public; note this as a limitation.
  - NLA: the step-matched pair ceselder/qwen3.6-27b-nla-L42 (av_rl_lora_step400 + rl_critic_step400) for both
    readouts and reconstructor analyses.
  - Skip-lens: dropped from the core set (trained at L62).
  - SAE: [fill in].
  - J_NP, J_CB, R_CB, logit lens: as in the spec.
  - Every learned reader gets prompt-only, no-activation and shuffled-activation arms from day one. Prompt-only =
    stock Qwen3.6-27B, as in WSB.

- CLOSED-SET SCORING (your §5.11). Candidates = gold z, the input, the answer, and same-category foils. Token-reader
  hit = z ranks first among the candidates AND its score exceeds the same reader's no-activation run.

- ALSO ACCEPTED AS WRITTEN: family x position fixed effects in the ensemble (§5.12); a power calculation with a
  pre-registered minimum number of cells per stratum below which W is not reported (§5.13); SAE scored with non-J
  labels only (§5.15); A5 gets the "Delta(c) > 0 for hallucinated claims" branch, uses the step-matched NLA pair,
  and records a dated update to the P5 prior with the NLA literature that motivated it (§5.16); brew is not the
  label pipeline's positive control, which gets its own (§5.17, and 001 below).

==================================================================================================================
4. REPO STRUCTURE
==================================================================================================================

Model it on github.com/senku14x/Readable-Here-Used-There (branch extension): organised by experiment, every number
traceable to a table, and every table traceable to a raw array with a hash.

README.md
  What the project asks, in one paragraph. The layout. How to reproduce: environment, gates, then each experiment's
  battery and analysis commands. An index of experiments, each with a one-line result and links to its design spec
  and report. Keep the README to things that are true now; history goes in STATUS and the reports.

CLAUDE.md
  Written so a fresh session can pick up with no other context:
  - read order: CLAUDE.md -> STATUS.md -> docs/context/SPEC.md + AMENDMENT_01.md -> the latest design spec and
    report;
  - what the project is;
  - the non-negotiables (below);
  - conventions: naming, cell format, layer indexing (block outputs), endpoint definition, units;
  - environment: GPU, pins, hashes, paths, the rebuild recipe;
  - where things stand;
  - known pitfalls and the silly-mistakes log.
  Model it on RHUT's CLAUDE.md, but note that RHUT's freeze rules do not apply here.

STATUS.md
  Short. Current state at the top: what has run, what is running, what is next, and open issues with their owners.
  Dated update blocks below. The canonical one-paragraph statement of current findings lives here once there are
  findings.

docs/
  context/       SPEC.md (the spec), READING_LIST.md, AMENDMENT_01.md (this message's decisions and every conflict
                 with the spec), and any notes on papers or tools worth keeping.
  design_specs/  NNN_<experiment>.md, one per experiment, written and committed before any forward it governs.
                 Sections:
                 - why: the question and what it decides;
                 - materials: families, items, cells, layers;
                 - estimands, with formulas;
                 - interventions and controls;
                 - gates;
                 - registered predictions with priors;
                 - decision rule: what each outcome means for the next step;
                 - stop rules;
                 - budget: forwards and wall-clock.
                 Changes after registration are dated amendments at the bottom, registered before the forwards
                 they govern.
  reports/       NNN_<experiment>.md after each run:
                 - setup;
                 - main findings with intervals and sign counts;
                 - complete tables (every condition and control);
                 - gate outcomes;
                 - randomly selected cells read by hand, with a fixed seed;
                 - limitations next to the claims they qualify;
                 - interpretation;
                 - next decision.
                 Negatives stay in. Your reading report becomes 000_reading_report.md.
  papers/        PDFs of the papers we cite.

scripts/
  common/        Shared code, with a test for each module:
                 - registry and pins;
                 - cell format;
                 - WSB loading and rendering, with position assertions;
                 - partner constructors and gates;
                 - margin scorer;
                 - intervention engine/hooks: reuse RHUT's hooks_fp32 / SpanWriter / realised-write stats, and SEC's
                   engine for the regime cross-check;
                 - gates;
                 - stats: reuse RHUT's stats.py (cluster-t, ratio of means, Holm) and add BH-FDR and the cluster
                   bootstrap;
                 - figure helper (ICLR style).
  NNN_<experiment>/  <experiment>.py (battery, with stages such as smoke / run) and <experiment>_analysis.py (a
                 single scoring pass that reads raw outputs and writes the tables, the report's numbers and the
                 figures).

results/
  NNN_<experiment>/
    analysis/    Tables as JSON (every number in the report or a figure comes from here), plus any analysis outputs.
    figures/     Regenerated from the tables only, by the figure helper.
    (in scope)   manifest_<stage>.json: model revision, lens hashes, software versions, git commit, raw-array
                 sha256, run id. meta_<stage>.json: rendered cells, token ids, positions.

outputs/         Gitignored. Raw arrays (npz) and logs, mirrored to a private HF dataset ([fill in name]) at the same
                 relative paths. Every manifest records the sha256 of its raw arrays.

Naming: experiments are NNN_snake_case with scientific names (001_leverage_profile, not 001_test_v2_final). Stages
are smoke / pilot / run. Run IDs are timestamps.

Non-negotiables (put these in CLAUDE.md):
- Labels never use any reader: no J swaps, AR edits or SAE directions in labels or item admission. Label code must
  not import reader code.
- Direction-level writes go through float32 hooks. Fidelity is measured after the write (rho in [0.9, 1.1],
  kappa >= 0.99).
- Every experiment has a design spec, committed before its run and shown to me. Launch only on my go.
- Gates before outcomes: span/rendering, same-source == clean bitwise, fidelity, damage, label reliability. A failed
  gate is repaired and re-run; nothing that has been scored is deleted.
- Full-sequence endpoint, log-sum-exp over accepted spellings; clamp and score every answer token, never just the
  first.
- Baselines from day one (prompt-only, no-activation, shuffled-activation).
- Wording: "demonstrated influence on this answer", not "uses"; "decodable", not "represents". Scope every claim to
  model, family, position, layer and n.
- Record every silly mistake in the commit message and the report.
- No subagents.

==================================================================================================================
5. WHAT TO DO NOW (scaffolding only, no science runs)
==================================================================================================================

Work through these in order. Each has a "done when".

a. REPO. Create the structure above. Write README.md, CLAUDE.md and STATUS.md for real (no placeholders except
   fields that genuinely can't be known yet). Write docs/context/AMENDMENT_01.md recording sections 1-3 of this
   message and listing every place it changes the spec. Move your reading report to
   docs/reports/000_reading_report.md.
   Done when: a fresh session reading only CLAUDE.md could find everything.

b. ENVIRONMENT. Install the pinned stack (transformers 5.15.0 and RHUT's other pins). Download Qwen3.6-27B @ 6a9e13bd
   and the J_NP, J_CB and R_CB lens files. Verify all hashes against RHUT's instrument_registry.json. Record GPU,
   versions, paths and hashes in CLAUDE.md and STATUS.md, and write a rebuild script, since /workspace is not
   persistent.
   Done when: the model loads, one forward on a brew prompt gives the gated answer, and every hash matches.

c. READER SMOKE TESTS. For each of J_NP, J_CB, R_CB, the logit lens, the Oracle Lens (via WSB's reimplementation in
   methods.py, on-contract at L40) and the NLA (nla-L42 AV readout, plus one AR reconstruction): produce one readout
   on one brew emission cell. Record:
   - load time and memory;
   - the raw output;
   - whether it looks sane (does it name colours at all?);
   - for the AR, the FVE of reconstructing the cell from the AV's own explanation;
   - for any failure, the exact error and the likely cause.
   Done when: a table in STATUS.md says works / fails / blocked for each reader, with the reason.

d. SHARED CODE (scripts/common/). Each module gets a test against a known example:
   - CELL FORMAT. (item, pair_id, position, layer, family, half, split). One dataclass, serialisable.
   - RENDERING. Load the WSB banks; render with the pinned tokenizer; assert that region tokens and eval positions
     match the bank. Test: brew-s2-r10d2-0000 renders with the expected regions.
   - PARTNERS. Brew and chain constructors with the constraints in section 1, and the gate (greedy + 10 samples at
     T = 0.8). Test: a hand-checked partner for one brew item and one chain item, token counts equal, every position
     aligned.
   - MARGIN SCORER. Full-sequence teacher-forced log-probs for every accepted spelling of a and a', log-sum-exp per
     answer, m = log P(a') - log P(a). Test: on a brew item, the clean margin has the sign implied by the model's
     greedy answer, and multi-token spellings are scored at every token.
   - ENGINE AND HOOKS. Reuse RHUT's fp32-residual hooks. Full-cell replacement and 1-D coordinate swap at (layer,
     position); realised-write stats; no-op self-patch. Tests:
     - self-patch is bitwise identical to clean;
     - a swap's realised rho and kappa pass;
     - a full-cell replacement with the partner's residual at the last position of the last layer reproduces the
       partner's logits.
   - REGIME CHECK. The same 5 edits under RHUT's regime and SEC's true-fp32 replay; report the Delta m agreement.
   - GATES. Span, bitwise, fidelity, damage (top-1 retention, Delta log P(a), Delta log P(a')), and
     sequence-length determinism.
   - STATS. Port RHUT's stats.py; add BH-FDR and the cluster bootstrap; test against scipy/statsmodels on toy data.
   Done when: all tests pass, and STATUS.md lists them.

e. DESIGN SPEC 001. Draft docs/design_specs/001_leverage_profile.md. Do not run it.
   - Question: where does a single (position, layer) cell carry leverage over the answer in brew and chain, and are
     brew's off-trajectory colours decodable at the emission cells? This decides whether single-cell labels can vary
     within a position (needed for W) and whether brew supplies the not-needed class.
   - Materials: gated brew and chain pairs (dev split only); all 24 brew pinned cells; chain's final cell and
     question span; layers 40, 42 and 44.
   - Measurements:
     - full-cell patch share (partner's whole residual at the cell / whole-run gap G) for every cell x layer, with
       damage recorded;
     - span-level patches as context;
     - decodability of gold z and of each off-trajectory colour at the brew emission cells, with an item-grouped
       probe and a control task.
     No directions, no 1-D swaps, no readers.
   - Controls:
     - positive: the start-colour token at an early layer, where the full-cell share should approach 1 because it
       carries the input;
     - negative: the stir cells, where the share should be about 0 because the start colour has not been read;
     - self-patch no-op.
   - Predictions, with priors: write your own, before seeing data.
   - Decision rule, at minimum:
     (1) If no family shows leverage variation within a position, stop and rethink before building labels.
     (2) If the off-trajectory colours are not decodable at the emission cells, brew has no not-needed class, and the
         RHUT tagged-selection fallback is designed next.
     (3) If the positive control fails, the engine is broken; fix it first.
   - Budget: forwards and wall-clock, from a timed smoke run.

==================================================================================================================
6. WHAT TO REPORT BACK
==================================================================================================================

When a-e are done, send me:
- a short summary;
- the reader smoke-test table;
- the test list with pass/fail;
- the regime-check numbers;
- anything that broke and how you fixed it (silly mistakes included);
- the 001 design spec for review.

If you get blocked on something only I can resolve (access, credentials, a choice the spec doesn't settle), stop and
ask rather than picking silently.
