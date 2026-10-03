# CLAUDE.md — Causal-Concordance

Guide for any agent or collaborator working in this repo. Reading only this file should be enough to find everything.

## Read order

1. **CLAUDE.md** (this file): rules, conventions, environment, pitfalls.
2. **docs/COME_UP_TO_SPEED.md**: handoff (what exists, what 001 found, known problems, decisions log), then
   **STATUS.md** (where things stand, what works, open questions) and **docs/context/SPEC_STATUS.md** (the whole spec,
   item by item: done / built / partial / not started / dropped).
3. **docs/context/SPEC.md** + **docs/context/AMENDMENT_01.md**: the project spec and every decision that overrides it
   (the amendment wins where they disagree; its §6 lists each conflict by spec line).
4. The latest **docs/design_specs/NNN_*.md** and **docs/reports/NNN_*.md** (currently: design 001 v3 + A1, run in
   full on dev; report 001 = its results, awaiting PI review; report 000 = the reading report).

Supporting: `docs/context/PI_DIRECTIVE_2026-10-02.md` (the PI's block-000 directive, source of the non-negotiables),
`docs/reviews/` (two external reviews of the scaffold/design, the external audit of 001),
`docs/context/READING_LIST.md`, `docs/context/reading_notes.md` (cited notes from the reading pass),
`docs/papers/INDEX.md`.

## Project in one paragraph

Score activation readers on **Qwen3.6-27B** (logit lens, J_NP, J_CB, R_CB, Oracle Lens, NLA; SAE to be added) against
**reader-blind causal labels** — "pair difference used" / decodable-not-needed / absent / unresolved — computed per
(token position, layer) cell from full-cell patches, 1-D swaps along mean-difference directions, necessity ablations and
cross-fitted probes on **WorkspaceBench** families with token-aligned counterfactual partners (brew, chain, relational).
Then decompose reader agreement into shared signal, shared error and shared prompt inference. The labels measure
demonstrated influence on one specified answer, not workspace membership.

## Non-negotiables (verbatim from the PI, 2026-10-02)

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

RHUT's freeze rules (its registered-label stop rules and frozen ledgers) do **not** apply to this project; RHUT is a
source of code and organisms only.

## Conventions

- **Layer** = output of decoder block l (0..63), the residual stream entering block l+1 (RHUT/WSB/EasyNLA convention).
- **Cell** = one token position at one layer (`scripts/common/cells.py`: item, pair_id, position, layer, family, half,
  split). Span patches are context only.
- **Renders** reproduce WSB's producer exactly (`scripts/common/wsb.py`): brew = chat template + "Answer:" prefill
  (24 pinned cells), chain = chat (read cells " What" → final "\n\n"), relational = bare cloze (final "'s").
  Thinking is off (`enable_thinking=False`). Position assertions run on every render.
- **Partners** (`partners.py`): brew = same table, different start; chain = same steps, different start (same digit
  count, single-digit intermediates); relational = the bank's a/b members. Constraints z′≠z, a′≠a, z′∉{a,a′}, z∉{a′};
  renders token-aligned. **Gate**: greedy correct and ≥ 9/10 correct at T = 0.8 (top_p 0.95, 16 tokens, EOS
  suppressed), both members.
- **Margin** (`margin.py`): m = LSE_{S(a′)} log P − LSE_{S(a)} log P, full-sequence teacher forcing, every token of every
  spelling (brew " c"/" C" — " Teal" is two tokens; chain digits; relational " hop1's hop2"). Logits computed in float32
  from the final-norm output.
- **Regime**: design 001 v3 (§2, §9.1) measures every label quantity at L ≥ 40 under **SEC's true-float32 replay**
  (`regime.SecEngine`, l0 = 39: stock bf16 blocks 0–39, float32 blocks 40–63 and W_U; noise floor ≈ 2e-4 nats). RHUT's
  float32-residual regime from block 0 (bf16 matmuls under autocast, float32 residual and writes; `engine.Engine(model,
  fp32_from=0)`) is the cross-check (AMENDMENT_01 §3, errata E4). Every clean/edited comparison is made at **one
  sequence length**; patch sources are captured at the **matching length** (`margin.score(writes_fn=...)` /
  `SecEngine.score(edit_fn=...)` build writes per continuation). Reader inputs use the stock bf16 forward
  (`readers.capture_stock`, WSB's convention). `SecReplay.build()` parks the bf16 upper blocks on the CPU until
  `restore()`: use `SecReplay` / `SecEngine` as context managers; every stock-forward entry point refuses a parked model
  (`registry.assert_subject_model_resident`).
- **Splits**: dev/test by table (brew), by step-sequence group (chain; identical prompts recur across chain items with
  the same steps), by name set (relational); defined in design 001. Halves A/B for cross-fitting by items
  (AMENDMENT_01 §3). Items touched by engineering checks go to dev.
- **Naming**: experiments `NNN_snake_case`; stages `smoke` / `pilot` / `run`; run IDs are UTC timestamps.
  Code in `scripts/NNN_<experiment>/<experiment>.py` (+ `_analysis.py`); results in
  `results/NNN_<experiment>/{analysis/*.json, figures/, manifest_<stage>.json, meta_<stage>.json}`; raw arrays and logs
  in `outputs/` (gitignored; mirrored to the **public** HF dataset `senku21x/causal-concordance-outputs` (PI 2026-10-03; `MANIFEST.json` there lists sizes and sha256)). Every manifest
  carries `registry.manifest_block()` (model/lens/reader/WSB pins, software versions, git commit + dirty flag).
- **Statistics** (`stats.py`): cluster-level t intervals, BH-FDR, cluster bootstrap (clusters = items/tables), Holm.
  Report intervals and sign counts, never a bare mean.
- **Figures** (`figures.py`): ICLR style (vendored `iclrplot`), fixed colour per reader, drawn from committed JSON only.

## Environment (verified 2026-10-02)

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, sm_120, 96 GB (driver supports CUDA 13.2) |
| Host | 128 CPUs, 377 GB RAM, 500 GB disk; **/workspace is not a persistent volume** (rebuild with the script below) |
| Label + reader stack (`/venv/main`) | Python 3.12.14, torch 2.11.0+cu128, transformers 5.15.0, peft 0.21.2, accelerate 1.15.0, jlens @ 581d398 (`/workspace/jacobian-lens`, editable) |
| Reader-generation stack (`/venv/vllm`, optional, not yet adopted) | vLLM 0.21.0, torch 2.11.0+cu130, transformers 5.5.4, vllm-metamodels @ 6a0a0e5. Never used for labels |
| Model | `Qwen/Qwen3.6-27B` @ 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9, bf16, eager attention; 50.1 GB on GPU, loads in ~15 s |
| HF cache | `HF_HOME=/workspace/.hf_home` (~97 GB) |
| Pins + hashes | `scripts/common/registry.json` (model, lenses with sha256, reader checkpoints, WSB bank sha256s, software); verification record `results/000_setup/analysis/artifact_hashes.json` (29/29 match) |

Lenses (sha256 in the registry, matching RHUT's `instrument_registry.json`): J_NP = neuronpedia/jacobian-lens @ 0731326e
(n=1000 wikitext), J_CB and R_CB = camilablank/workspace-lenses @ d740106d. Readers: Oracle Lens =
agu18dec/olens_and_ar @ c95215d5 `olens_s3d_rl600` (the public copy; WSB's default path is a private dataset repo);
NLA = ceselder/qwen3.6-27b-nla-L42 @ 54e3ce60 (`av_sft_lora` merged + `av_rl_lora_step400`; AR `rl_critic_step400`).

Sources (read-only clones, pinned in `scripts/000_setup/rebuild_env.sh`): `/workspace/sources/repos/` — workspace-bench
@ 92d763e, RHUT (senku14x/Readable-Here-Used-There) @ d4197b0, SEC (senku14x/jacobian-lens, `research_secant/`)
@ a3720ab, EasyNLA @ 4d72847, vllm-metamodels @ 6a0a0e5; papers and web sources in `/workspace/sources/{papers,web}`.

**Rebuild:** `bash scripts/000_setup/rebuild_env.sh` (add `--vllm` for the optional vLLM venv).
**Tests:** `python -m pytest scripts/common/tests -q` (`-m "not gpu"` for the CPU-only subset).
**Git:** commits as senku14x / visheshgupta14x@gmail.com. A stale `GITHUB_TOKEN` env var shadows the gh login: push with
`env -u GITHUB_TOKEN git -c credential.helper='!gh auth git-credential' push origin <branch>`. Do not push to `main`
without the PI's say-so.

## Where things stand

See **STATUS.md** (block 000 done; 78 tests; design 001 v3 + A1 run in full on dev and reported in
`docs/reports/001_leverage_profile.md`, revised after an external audit (§15), awaiting PI review; no label, no reader
score and no test-split measurement yet). Known open problems: the probe controls (membership control far from chance,
permutation null reuses C), the damage readout (top 5 only), no prompt-only baseline yet.
Results page (private artifact, built by `scripts/000_setup/results_page/make_page.py` from committed JSON only):
https://claude.ai/artifact/RMRX8R4iFnysbnHaNWTBeC

## Pitfalls (read before writing code)

- **bf16 numerics are chaotic at the 0.05–0.25-nat level.** Under the RHUT regime (bf16 matmuls), any perturbation of
  the residual, however small, moves the margin by up to ~0.2 nats; appending tokens moves prompt-position activations
  (onset at block 3, the first full-attention block, ~6e-4 relative, ~1% median by block 63). Never compare runs of
  different lengths; never expect two mathematically equal edits to give bitwise-equal outputs; compare edits to the
  measured noise floor. SEC's true-fp32 upper stack has a ~250× lower floor (see STATUS).
- **Logits:** never score from the bf16 lm_head output (spacing 0.125 at |logit| ≈ 20); use `Engine.logits` (float32).
- **Autocast:** any arithmetic inside a hook must run with autocast disabled (`engine.CellWrite` does), or a dot product
  silently runs in bf16.
- **Hook order** on one block: regime upcast → writes → recorder. Hooks that index a prompt position must skip decode
  steps (sequence length 1) during `generate`.
- **Shell:** each Bash call is a fresh shell — `source /venv/main/bin/activate` (or call `/venv/main/bin/python`).
  The shell's `grep` skips files it thinks are binary; use `command grep -a` on extracted PDF text.
- **WSB bank fields:** brew/chain items use `name`, relational uses `id`. Brew `prompt` puts the stir sentence *before*
  the start sentence (all 100 items), so stir cells precede the only token that differs within a pair.
- **Chain prompts collide across items** with the same steps (15 collisions among the 28 usable items): split by step
  sequence, de-duplicate pairs.
- **Subject-model integrity:** `readers.NLAVerbalizer` merges the NLA's SFT adapter into the model object IN PLACE and
  marks it; `Engine`, `SecReplay` and `capture_stock` refuse a marked model. Capture every subject activation before
  the NLA loads, or load a fresh model. Attach/unload readers (Oracle) leave the model bitwise intact (tested).
- **Small effects are magnitude tests** (|S| < ε′, AMENDMENT_01 errata E1); a whole-cell patch is a reference, not a
  ceiling (E2).
- **Readers:** the NLA adapters are rsLoRA (scaling α/√r); the NLA tokenizer files differ in format from the base but
  render identical AV prompt ids; the Oracle marker is ㈜ at slot 24 of a 56-token carrier for L40. Reader code lives in
  `readers.py` and may import label code, never the reverse (`tests/test_import_boundary.py`).

## Silly-mistakes log (also in commit messages and reports)

| Date | Mistake | Fix |
|---|---|---|
| 10-02 | Spec line citations in the reading report off by 1–3 lines | corrected; SPEC.md kept byte-identical so citations hold |
| 10-02 | Reading notes cited SEC STATUS "L80–81" for the cos 0.80 result | it is L45–46; fixed in `reading_notes.md` |
| 10-02 | First env check ran before the pinned stack was installed | installed transformers 5.15.0 etc. via uv |
| 10-02 | `python: command not found` in fresh shells | activate the venv in every call |
| 10-02 | Chain partner "no intermediate is a prompt numeral" used step constants + start only | uses every numeral of the partner prompt (no change on this bank) |
| 10-02 | Engine test expected a swap along unit(h_p−h_m) to equal the full patch downstream to 1e-3 | equal at the write (rel 2.7e-8); downstream differs by the bf16 floor (0.13 max), so the test checks the write |
| 10-02 | Sequence-length test used an absolute bound on large-magnitude final hidden dims | relative bound, then an onset-based leak test after finding non-bitwise lengths |
| 10-02 | Teacher-forcing test tolerance (2e-3) ignored the sequence-length floor | 0.3 nats + an off-by-one discriminator |
| 10-02 | Regime check hard-coded a hand-checked pair that fails the gate (partner 5/10) | gates candidates in bank order, keeps every attempt |
| 10-02 | Lens readouts ran with autograd on (warning) | `load_model` sets `requires_grad_(False)`; readouts under `no_grad` |
| 10-02 | vLLM venv resolved transformers 4.57.6 | override to 5.5.4 (the NLA's training stack) |
| 10-02 | vLLM-check capture hook indexed the marker during decode steps (IndexError) | length guard |
| 10-02 | Duplicate paper PDF copied | removed (byte-identical) |
| 10-02 | "Not needed" and the within-position rule used a signed S < ε′ (S = −0.8 passed) | magnitude test, toward/away/negligible kept separate (errata E1, 001 v2) |
| 10-02 | 001 v1 asked for PCA to 256 dims from ≈ 184 training rows | no PCA in the primary probe; rank capped at n_train − 1 in any sensitivity check |
| 10-02 | 001 v1 called a small block-0 patch "engine broken" | block 0 is GatedDeltaNet; only the embedding anchor, self-patch and stir cells are engine checks |
| 10-02 | 001 v1 made within-position variation a stop rule although the brew contrast is within-cell | demoted to a diagnostic |
| 10-02 | The NLA's in-place merge was guarded only by a docstring | integrity marker + refusal in every capture/label entry point + tests |
| 10-02 | The integrity guard ran only at construction (a model mutated after an Engine was built was used silently) | checks at use time + lifecycle test |
| 10-02 | Commit 79f835a's message says "Design 001 v3" but the design edit script had failed on one text match, so that commit holds only the integrity fix | v3 committed in the next commit; edit scripts now run before `git add` with `set -e` |
| 10-02 | Brew split rounded odd z-strata at random (51/49 instead of 50/50) | alternating rounding; caught by test_splits before any use |
| 10-02 | 001 noise-floor helper drew a different random direction for the multi-token-spelling forwards of one perturbation | one direction per draw, shared by every forward; caught in review before any run |
| 10-02 | 001 snapshot built its render dict as one literal instead of a loop over the 10 tables (UnboundLocalError before any reader ran) | loop; rerun |
| 10-02 | The results-page template was truncated to its CSS by a later cleanup rewrite, and the broken page was what got published (no header, sections or first half of the script) | template rebuilt in full; the generator asserts each placeholder occurs once; the page is checked in jsdom (no script errors, every control clicked) before publishing |
| 10-02 | Report 001 overstated several conclusions (external audit): "most off-pair mass goes to intermediates" from top-5 tokens only (they hold 0.14 of 0.35); pair medians printed with table-mean intervals; chain clustered by item, not step group; "holds on all 10 tables" (6 had a z mention); FVE "not item-specific" (small source preference exists); "not yet the answer" from a weak probe; the substitution null stated as a finding | every number recomputed (`audit_followups.py`), report revised in place with §15; page revised |
| 10-02 | Probe controls not checked before use: membership control task AUC median 0.58, max 0.84 (should be ≈ 0.5); permutation null reuses the C chosen on real labels | flagged in report §7/§9/§12; needs a probe re-run (not yet done) |
| 10-02 | No prompt-only baseline in 001, although the PI's rules ask for it from day one | flagged; to be added with reader scoring |
| 10-02 | 002 v2: partner readouts judged against the member item's 5 candidates, which need not contain the partner's path, so tracking could not be scored on roles | found while writing the lens stage (and independently by the design review); 302 readouts ($0.555) discarded; v2.1 uses one set per table |
| 10-02 | 002 truth key gave 25 partner rows the member trajectory; v2 design counted an explicit non-candidate value as abstention | variant-aware key (80/80 vs the snapshot); "none" is an incorrect claim |
| 10-02 | 002 pilot stage overwrote pass 1; judge cost estimated 3× low (reasoning tokens at minimal effort); prompt-only rows had their truth fields overwritten by score dicts | passes archived, spend corrected by hand; estimate redone; truth recovered from capture metadata by render key, stage code fixed |
| 10-02 | `pkill -f "<pattern>"` matched its own shell (exit 144) twice | bracket the first character of the pattern |
| 10-02 | 003 analysis scored prediction L4 with the stricter used ∧ necessary fraction (reported False; the written criterion gives 87–93%, True) | fixed in labels.py; report 003 §9 (external audit) |
| 10-02 | 003 report read "full − max single-direction S" as influence outside the three directions (0.21–0.42); this is not an attribution | withdrawn; joint-span and complement patches needed |
| 10-02 | 003 random-clamp necessity control was not size-matched (about 20× smaller edits) yet was read as specificity; family-level decodability promoted to item "not needed" labels; 0383 white partner's answer misnamed | specificity withdrawn; label renamed "readable, low patch leverage"; answer corrected |
| 10-02 | 002/003 summaries called positive difference-in-differences lens tracking "rank flips" (actual reversals 90/183 at J_NP cosine `:` L40) | wording corrected in reports, STATUS, the handoff doc and the page |
| 10-02 | Report 001 rounding slips: `:` L60 0.75 [0.71, 0.76] (data 0.745 [0.705, 0.764]), `</think>` L42 0.02 (0.0145), start L40 CI [0.22, …] (0.215), answer probe `</think>` L42 0.15 (0.1447); §8 "any lens L20–48: 0" missed one J_NP cosine "Green" at L36 | found by regenerating the numbers from JSON for the page; report corrected (§14); the page computes every number from JSON with round-half-up |
| 10-03 | 004's first preflight imported the 003 module for its activation loader, which pulls in the mixed 001 module | 004 loads captures itself; the boundary test now walks the 004 entry point and forbids the 001/003 modules |
| 10-03 | `brew_material.py`'s docstring cited `tests/test_brew_material.py`, which did not exist | test written (82 pairs vs the gate record, render keys vs the 001 captures) |
| 10-03 | 004 preflight printed ranks under colliding keys (the Z and J ranks overwrote each other) | keys include the arm |
| 10-03 | 004 smoke inspection ranked the random-complement control in the upper tail, where a real complement far below the random ones scores 1.000 | complements use the lower tail; both tails saved; caught before the run |
| 10-03 | 004 S5 coded as "r1/r2 share S1's 0.25 verdict and their median recoveries differ ≤ 0.05" (agreement below threshold counted; recovery units; medians can cancel) — design §5 says a high-recovery result, 0.05 in S share units | ≥ 0.5 in r1 and r2 and median paired \|ΔS\| ≤ 0.05; low-effect agreement separate; design §13; smoke analysis v1 kept as superseded (external review) |
| 10-03 | 004 run path recorded fidelity but did not enforce G1; `Fidelity.ok()` was true with nothing checked; writes < 1e-4 ‖h‖ were exempted although the preflight resolves 1e-6 (8 smoke writes unchecked); my smoke summary said "fidelity holds on every write" | run stops on a failed write (saved, pair not done; per-pair atomic files); checked ≥ 1e-6, read-back bound below, exact zeros must be no-ops, empty fails; smoke v2: 948/948 writes per pair checked (external review) |
| 10-03 | STATUS and my smoke summary said 25 accepted spellings (it is 24: 12 × 2); smoke analysis printed Pearson r with n = 2 | corrected; Pearson needs n ≥ 3 |
| 10-03 | Report 004 drafted from three-decimal printouts: 8 double-rounding slips and 2 F_target counts written from memory (17, 29 → 21, 32) | every table regenerated and checked against the JSON (half-up) before commit; draft reports from full-precision JSON, never from printouts |
