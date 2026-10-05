# Up-to-speed report: Causal-Concordance, fresh instance (2026-10-05)

Written for the PI to vet before any new work. It covers:
- what I read and what I didn't (§1);
- the state of the instance (§2);
- the project as I understand it (§3–§7);
- open decisions (§8);
- problems I found in the bootstrap and the repo (§9);
- my assessment (§10), my doubts (§11), raw examples (§12) and quick ways to check me (§13).

**Tags on each claim.**
- **[V]**: I verified it this session against code, raw files or the PDF (the command or file is named).
- **[D]**: taken from a repo document, cited by path. I did not re-derive it from raw data.
- **[M]**: my own inference or assessment.

Numbers are copied from the cited reports or JSON. I did not recompute any experiment's statistics from raw outputs this session, except where marked [V].

---

## 0. Verdict

- **Up to speed on:**
  - the paper and the parts the project leans on;
  - the project's question, rules, conventions and reader panel;
  - what 001–006b found and every correction that applies;
  - the current state of Cathryn's J-subspace work, the advisor's checks, and what 007 would have to decide.
- **Partly up to speed on:**
  - the code (I grepped and read the parts behind the bootstrap's claims, not whole modules);
  - the 001–004 design specs (I read their reports, not the specs);
  - the 005 pilot steps (I read the outcomes via the amendment and the reports, not each §16 step);
  - the external-review folders (only the consolidated handoff's README).
- **Not up to speed on:**
  - anything that needs a run: I have run no experiment;
  - the reviewers' working folder `/workspace/codex_reports` (not restored; the private dataset wasn't pulled).

---

## 1. What I read (so you can see the gaps)

| Source | Read | Notes |
|---|---|---|
| Workspace paper (Gurnee et al. 2026) | **Full**, main text and all appendices (HTML version) | The arXiv appendix numbering matches A.1–A.24.3 [V] (`docs/papers/text/2607.15495.txt`). Figures viewed as images in the PDF: 22, 29, 49, 50 only; all others from captions |
| `04_user_workspace_paper_notes_verbatim.md` | Full | — |
| `04b_…combined_2026-10-05.md` (your combined notes, filed this session) | Full | Filed as a copy of `/workspace/anthropic_workspace_combined_notes.md` |
| `03_workspace_paper_notes_merged.md` | Full, incl. §14 corrections and §15 disagreements | — |
| `docs/context/reading_notes.md`, `reading_notes_v2.md` | Full | — |
| Li et al. 2509.13316 | Main text §1–8 | Appendices not read |
| Karvonen et al. 2512.15674 (Activation Oracles) | Main text §1–9 | Appendices not read |
| Bauer et al. 2606.02609 (AObench) | Full | — |
| Dingeto 2607.20379 | Main text §1–9 | Appendices not read |
| NLA paper page (`docs/sources/web/nla.txt`) | Intro, limitations, method, confabulation section | Case studies, evaluations, auditing and appendices not read |
| WorkspaceBench post, R-lens post, Hong post | Full | — |
| michaelzhang & TurnTrout NLA-initialisation post | First ~40% | — |
| `CLAUDE.md`, `STATUS.md` | Full | — |
| `SPEC_v2.md` with appendices A (CW) and B (AMENDMENT_01) | Full | — |
| `SPEC_v2_errata_01.md` | Full | — |
| `docs/COME_UP_TO_SPEED.md` | First 150 lines (§1–§5) | — |
| `docs/context/agent_notes/*` | All 14 files | — |
| 005 | Reports: `005_followups`, `005_full_run` full. Specs: `005_full_run`, `005_amendment_01`, `005_followups` full. Design `005_planted_concepts` §1–§15 full, §16 headings only. `005_pilot_note` headings only. `005_smoke_note` summary and G1 | — |
| 006 / 006b | Report and both designs, full | — |
| Reports 004, 003, 002 | Full | — |
| Report 001 | Scope, §4, §5, §13, §15 | §6–§12 not read in full |
| `checks_2026-10-04_no_gpu.md` | Full | — |
| Design specs 001–004, `000_reading_report.md`, original `SPEC.md`, `PI_DIRECTIVE`, `READING_LIST`, `SPEC_STATUS` | **Not read** | SPEC.md's text is embedded in SPEC_v2. The directive's non-negotiables are in CLAUDE.md |
| `docs/reviews/` | Only the consolidated-handoff README | — |
| Code | Grepped and read the relevant parts of `scripts/common/wsb.py`, `readers.py`, `splits.py`, WSB `backend.py` and the olens README; Cathryn's `jbasis/make_jbasis.py` in full | **Not read line by line:** `jspace.py`, `engine.py`, `regime.py`, `margin.py`, `partners.py`, experiment scripts |
| Cathryn's repo | `make_jbasis.py`; `STATUS.md` at origin/main (CW-10–CW-13 sections); raw CW-12 Oracle readouts for 2 random items (§12) | — |

---

## 2. Instance state [V]

| Item | State |
|---|---|
| Clone | `/workspace/causal-concordance`, branch `005-amended-full-run` @ 82137e0 |
| GitHub auth | gh is logged in as senku14x, but the stale `GITHUB_TOKEN` env var overrides it, so every git command uses `env -u GITHUB_TOKEN` |
| Rebuild | `rebuild_env.sh` exit 0. `artifact_hashes.json`: 40 files, 0 mismatches. HF cache 152 GB |
| Outputs | `restore_outputs.py`: 227/227 files match `MANIFEST.json` (manifest commit 82137e0) |
| Cathryn's repo | Read-only at 9a46bf9. Her `main` is 2 commits ahead (04484b2, 7b62673; §6) |
| Tests | CPU 104 passed. GPU 29 passed. `test_nla_rl` fails inside the full GPU run because the GPU-room guard sees the subject model still resident (42.3 GB free vs 55.7 + 4 needed); it passes alone |
| GPU | Free |
| Uncommitted | `results/000_setup/analysis/artifact_hashes.json` (rewritten by the hash check), `docs/claude_reports/advisor/JBASIS_CHECKS_2026-10-05.md`, `docs/claude_reports/04b_…`, this report. Nothing committed or pushed |
| Not attached by you | `jbasis_checks.py`, `jbasis_checks.json`, `NEW_RESULTS_2026-10-05.md` |

---

## 3. The project

**Question.** Do activation readers report content that Qwen3.6-27B's computation demonstrably depends on, judged against references that use no reader? Then: how much of their agreement is shared signal, shared error, shared prompt inference, or shared re-solving? [D] SPEC_v2 §0–§2.

**North Star.** Know which readers can be trusted to report content the computation depends on, measured independently of any reader.
- **Proxy:** on token-aligned counterfactual families, at matched cells, measure:
  - sensitivity to used content;
  - firing on decodable-but-not-needed content;
  - agreement after conditioning on the label.

  [D] SPEC_v2 §2.
- **Framing awaiting your confirmation:** "native reader-configuration comparison is primary, causal work a validation branch" [D] `docs/reviews/2026-10-03_codex_004_consolidated_handoff/README.md`; STATUS "Open questions" item 0.

**Why the labels are reader-blind.**
- The point is to grade readers against something they did not define.
- A label built from J swaps, AR edits or SAE directions would grade J against its own geometry. RHUT showed this concretely: its two-hop items were admitted with a J_NP coordinate swap, which made J's ranking on them circular.
- So: label code may not import reader code (enforced by `tests/test_import_boundary.py`), and J directions appear only in reader-side analyses (A4/A7). [D] CLAUDE.md non-negotiables; SPEC_v2 §5 table; reading_notes.md (RHUT two_hop Amd 7).

**What a label measures.** "Demonstrated influence on this answer":
- a reader-blind edit at one cell (one position, one layer) moves the full-sequence log-sum-exp answer margin;
- measured under SEC true-float32 replay at L ≥ 40;
- it is not workspace membership and not the model's natural route.

**Two tiers of labels** [D] SPEC_v2 §6.0:
- position-level (whole-cell patch share S; reliable);
- content-level (conditional; on brew it failed reliability in 003 and did no better than PCA in 004).

**Reader panel** (pins in `scripts/common/registry.json`; [D] CLAUDE.md "Environment" unless marked):

| Reader | Checkpoint | Layers | Input handling |
|---|---|---|---|
| Logit lens | — | any | W_U·final_norm(h) |
| J_NP | neuronpedia/jacobian-lens @ 0731326e, n = 1000 wikitext; the file stores `source_layers` 0–62 and no target field, which implies a final-block (63) target [V] | any 0–62 | Two readouts: WSB's cosine arm with **raw W_U** [V] (`workspace-bench/.../backend.py:58-59`: `unembed` = `get_output_embeddings().weight`), and the paper's softmax(W_U·norm(Jh)) |
| J_CB, R_CB | camilablank/workspace-lenses @ d740106d; n = 25 pile-10k, target 62, skip 4; R adds LRP rules | any | [D] reading_notes.md |
| Oracle Lens | agu18dec/olens_and_ar `olens_s3d_rl600` | **contract 20–60 step 4** [V] (olens `README.md:49,96`); project uses L40/L44, and L42 is an off-contract arm reported separately (A01 C6) | The marker embedding is replaced by 16000·unit(h), so the reader is scale-invariant |
| NLA (005 onward) | ceselder/qwen3.6-27b-nla-rl, `av_base` + `iter_000400` | L42 | A separate 53.8 GB model; adapter attached, never merged; Karvonen norm-matched add at block 1; scale-invariant [V] `readers.py:15-17,539-563` |
| NLA (001/002 only) | nla-L42 (`av_sft_lora` merged in place + `av_rl_lora_step400`) | L42 | This one modifies the subject in place, hence the integrity guard [V] `readers.py:329-341` |
| SAE / BSF | deferred | — | — |
| Skip-lens | dropped | — | — |
| Template lens | deferred | — | It lacks "teal" |

**Families** [D] SPEC_v2 §5 status table:
- **Brew (core).** 100 items, 24 pinned cells per item. Partner = same table, different start colour. Gate: greedy correct and ≥ 9/10 at T = 0.8. Dev 82/183 pairs (33 tables).
- **Chain (core).** Single-digit intermediates only; read at L56.
- **Multihop.** A4/A7 only; no partners, so no labels; needs its own dev/test split.
- **Relational.** Built, not gated.
- **Codebook.** Planned, no design.
- **Out for now:** RHUT organisms, commitment, safety families.

**Conventions** [D] CLAUDE.md:
- layer = output of block l (0–63);
- labels under SEC (stock bf16 for blocks 0–39, float32 for 40–63 and W_U; floor ≈ 2e-4 nats);
- reader inputs come from the stock bf16 forward;
- one sequence length per comparison;
- margins from float32 logits, every token of every spelling;
- dev/test by table (brew) or by step group (chain);
- cluster-bootstrap intervals with sign counts.

**Two prompt-only baselines; never conflate them** [D] V2-E5:
- **WSB reader floor:** "what would a model be thinking here", given the prefix. Brew 0.96.
- **Task baseline:** "what is the value", given the full prompt. Brew z 1.00.

**Shuffle control:** many random derangements, not one fixed shift [D] 006 post hoc §3.

---

## 4. The paper, and how the project's objects map to it

**J-lens** [D]+[V] paper §2.1 and A.7:
- J_ℓ = E[∂h_final,t′/∂h_ℓ,t], over 1000 × 128 tokens; readout softmax(W_U·norm(J_ℓ h)).
- The estimator injects at every target position, so it **sums** over t′ ≥ t and then averages over t.
- The appendix says the default target for Sonnet 4.5 is the penultimate layer, while the main text and pseudocode say final. J_NP appears to use final: its sources run 0–62 [V] (lens-file metadata). J_CB/R_CB use target 62, the penultimate [D].
- The repo skips 16 positions.

**J-space** [D] paper §2.3, A.8:
- The set of sparse nonnegative combinations of ≤ k J-lens vectors: a union of cones, not a subspace.
- A.8 writes spans and orthogonal projections, which describe a bigger set than the prose (your 04 §2.3 caveat; checked in 03 §2.2).

**Three different "J" objects in this project. Never mix their numbers.** [V for the definitions]
1. **005's J part:** NNLS refit on a 16-atom nonnegative pursuit of the folded atoms J_ℓᵀ(g ⊙ W_U[t]), with g = 1 + w. It holds about 5% of the concept vector's energy [D] `005_full_run.md` G3.
2. **Cathryn's J subspace:** the top 1,024 eigenvectors of Σ_t unit(W_U[t] J) over the whole vocabulary. It uses **raw W_U, no final-norm gain** [V] `make_jbasis.py`, WSB `backend.py:58`, and holds about 15–16% of the activation's energy.
3. **The paper's J-space:** sparse cones.

Also distinct: CW-4's "J-neighbourhood" removal (cos > 0.8, about 31 directions).

**Privilege test (§3.1, Fig. 8).**
- *Paper, on Claude:* 88% / 59% / 5% (raw J-lens / J component / non-J component); clamping J coordinates sends the non-J effect to about 0.
- *On Qwen (005):* no privilege at matched norm, and the non-J part does more at natural norm.
- [M] Earlier in this session I argued the 5% is close to a first-order consequence, because the NNLS residual is orthogonal to the concept's own atom. 005 found the own atom in the J part for only 1/45 concepts [D] `005_followups.md` A3, so that premise fails on Qwen. The argument stays a possible issue for the paper's Claude numbers only.

**A.9.2 (oracle lens)** [D].
- *Paper's interpretation:* the oracle lens reads only in-workspace content, the NLA everything.
- *005 P17 found the reverse on planted concepts:* the Oracle names the non-J part more than the NLA does (−0.14 [−0.18, −0.11]).

**§3.5 selectivity** (language passage: `Spanish` readable in all four tasks, causal only for report and flexible inference). This is the paper's own "decodable, not needed" case, i.e. the concept behind the project's label [D] reading_notes.md §3.5.1.

**A.21 and A.22** use the NLA as the reference, so they measure agreement between two readers, not ground truth. [D] 03 §6.4.

---

## 5. Experiments 001–006b

### 001: leverage profile (brew and chain, dev; SEC)
[D] `docs/reports/001_leverage_profile.md` §4–§5, §13, §15; erratum V2-E1.

- **Brew, `:` cell.** A whole-cell partner patch moves median S 0.35 / 0.48 / 0.50 / 0.70 / 0.74 at L40 / 42 / 44 / 56 / 60. At L42–44 it moves toward the partner in 100% of pairs.
- **Other brew cells.** Assistant newline about 0.29–0.32. `</think>` newline 0.00–0.03. Start-colour token 0.23 at L40, falling to 0 by L56.
- **Decodability.** z decodes at about 0.8 (12-way balanced accuracy, chance 0.083) at all three emission cells at L40–44, including `</think>` [D] SPEC_v2 §14.2.
- **Chain.** The final cell is about 0.001 at L40–44 and 0.66 / 0.69 at L56 / 60 (9 step groups).
- **Corrections:**
  - a whole-cell share cannot say which content moves the answer;
  - at `:` L42 the patched output is a **mixture of the two trajectories** (z + z′ take 0.54 [0.39, 0.74] of the off-pair increase), so "influence" holds by a thin margin; other layers are "influence (damage-flagged)" (V2-E1, check C2);
  - probe controls repaired (A2): off-path membership is decodable at `:` L40–60, narrowly at L40–44.

### 002: reader claim scoring (dev; descriptive)
[D] `docs/reports/002_claim_scoring.md`.

- **Verbalizers.** Gemini 3.8 Flash judge, κ 0.67 / 0.81 against blind codes from another model (not human). The verbalizers rarely put z in its role: NLA (nla-L42) 0.05 at `:`, 0.15 at the newlines; Oracle L40 0.00, with 85% of its samples self-contradictory. Mentions are 0.42–0.60.
- **Token lenses.** z ranks first among the candidates in about 0.4 of tables. Within-table tracking (a difference-in-differences) is positive in 47–48/48 tables at L40–44, but true rank reversals occur in only 90/183 comparisons (J_NP cosine, `:` L40). The start colour is not tracked.
- **Task baseline:** z 1.00.
- **Corrections:**
  - "flips" means difference-in-differences, not reversals;
  - no W statement can be made, because `:` and `</think>` are different positions (V2-E2);
  - 002's NLA is nla-L42, a different checkpoint from 006's nla-rl.

### 003: reader-free 1-D labels (dev; G4 failed → descriptive)
[D] `docs/reports/003_labels.md`.

- **Effect size.** The decorrelated z direction at `:` L40 / 42 / 44 has median S 0.031 / 0.034 / 0.050, positive in 85–90% of pairs, against whole-cell 0.35–0.50. "Used" in 11 / 15 / 18% of pairs, concentrated in a few tables (0383).
- **G4 failed:** κ 0.37, r 0.41 between disjoint fits.
- **Corrections:**
  - necessity specificity is not established, because the random clamps were about 20× smaller edits (V2-E4);
  - "decodable, not needed" at `</think>` becomes "readable, low patch leverage", since it was a family-level reading, not an item label (V2-E3);
  - "full − max" is withdrawn as an attribution.

### 004: span assay at `:` L42 (33 pairs; descriptive)
[D] `docs/reports/004_span_assay.md`.

- **Recovery** of the whole-cell effect (whole cell 0.43): 1-D z 0.06; z span (rank 11) 0.21; joint span (rank 33) 0.72.
- **PCA does as well or better:** PCA-11 0.58 (0.29 at the z span's edit norm); PCA-33 0.89 (0.80). Recovery tracks the share of the pair difference a subspace holds.
- **Direction of the effect.** The z-span swap is aimed at the donor's answer in only 4/33 pairs; its largest gain is the recipient's own z in 18/33.
- **Additivity.** z span and complement fall 0.18 short of the whole cell.
- **Reproducibility.** The joint span misses the frozen rule by 0.002.
- **Reading.** A geometry result; no intervention nominated.

### 005: planted concepts (45 test concepts; amendment 01; follow-ups exploratory)
[D] `005_full_run.md`, `005_followups.md`, `005_full_run` spec.

- **Setup.**
  - "Tell me about c" concept vectors (leave-one-out baseline over 100 concepts) are injected at L32–38 over the question span of the paper's introspection prompt.
  - Each vector is split into the J part (5.0% median energy) and the non-J remainder.
  - Readers make direct reads at their contract layers.
- **The model-side measure is lexical priming, not report.** Per A2b, the answer-slot rise minus the median rise at non-answer positions is about 0 for the full vector (+0.13 [−0.02, 0.29]) and negative for the pure J atom (−1.46). The rise is larger still on an unrelated prompt. Every "report" claim (M1, M2, M3, P11′) is therefore about priming.
- **No J privilege.**
  - M1 on R1′: +0.16 [−0.08, 0.40] (not supported); R1's registered form +0.0027 (tiny).
  - M3: −0.22 [−0.28, −0.16], so the non-J part does more at natural norm.
- **The clamp.**
  - The J re-entry clamp to L50 beats its random null (+0.12 [0.03, 0.21]), but the output-direction clamp does just as well (D +0.06 [−0.04, 0.15]).
  - Wording: "blocks output-related coordinates"; no re-entry into J coordinates is shown.
- **Own atom** in the J part for 1/45 concepts.
- **Q-c:** the logit-lens direction is as potent as the J atom (+0.04 [−0.32, 0.40], n = 30) within the pre-fixed window 1/16–1/4. The J atom leads outside that window.
- **Readers on direct reads** (non-J part, s = 1): probe 98%, Oracle L40/L44 73%, NLA 62%; every token lens about 0%.
  - P17 is reversed (the Oracle names the non-J part more).
  - P11′ (Oracle) is supported but not perturbation-matched.
  - The NLA failed G11 (44% vs the 50% bar), so its likelihood results are untestable.
- **R2:** behaviour moves without "report" for 71% of concepts by the registered rule, but only 29/45 with a post-hoc absolute floor (the relative criterion is scale-free).
- **005 is closed;** you ruled no further variants [D] STATUS.

### 006: reader likelihoods at the brew `:` cell (48 dev tables, saved 001 captures)
[D] `006_reader_likelihood.md`.

- **NLA (nla-rl) L42.** D_z +1.86 [1.41, 2.34], 44/48 tables. Its positive control (G-pc) passed exactly at the bar (6/12). It also tracks start (+0.41) and answer (+0.59).
- **Oracle L40/L44.** D_z +3.22 / +3.24, 48/48 tables. But G-pc failed (0.33 / 0.42), so T2 is uninformative by the rule.
  - Post hoc: the Oracle answers "blue" for 9 of 12 colours; with a leave-one-out prior, identification is 1.00.
- **Oracle L56/L60.** The answer colour dominates (+11 to +18).
- **Post hoc:**
  - 200 derangements give a null of about 0 for every role;
  - collision-free start tracking survives for the NLA and Oracle L44, answer tracking everywhere;
  - the Oracle's z tracking is about 2× larger at `</think>` (patch share 0.01) than at `:` (0.48);
  - per-pair NLA–Oracle r is 0.24–0.30.

### 006b: the model's own output at the same cells
[D] same report, 006b section.

- **Model D_z:** +3.76 at `:`, +3.99 at `</think>`, +3.66 at the assistant newline. At `:` z is the runner-up in 56% of tables. Model D_answer (collision-free) is +8.35 at `:`.
- **B2 (readers beyond the model):** not testable (9 pairs < 15).
- **Pending correction (your call; not edited):**
  - The repo reads 006b as "consistent with the readers reading the output-facing distribution" (`STATUS.md:20`, CLAUDE.md's 006b bullet, `docs/reports/006_reader_likelihood.md:104`) [V: those lines say so].
  - At L40–44 the numbers point the other way [V against the report's tables]:
    - **Model:** answer ≫ z (+8.35 vs +3.76).
    - **Readers:** z > answer. NLA +1.86 vs +0.59 (+0.84 collision-free); Oracle L40 +3.22 vs +1.73 (+2.23); Oracle L44 +3.24 vs +1.50 (+2.01).
    - **Reader–model r** is 0.07 / 0.30 / 0.24, against the readers' own cross-prefix r of 0.67 / 0.94 / 0.84.
  - [M] The output-facing reading fits Oracle L56/L60 (answer dominant) but not L40–44. The defensible line: 006b blocks a "held intermediate" claim, but it does not say what the readers read at L40–44.

### Checks C1–C3 (no GPU; 2026-10-04)
[D] `checks_2026-10-04_no_gpu.md`.

- **C1.** Under WSB's rule, CW-7 shrinks to Oracle L56 11/60 (null 0.041). J-lens L56 also scores 11/60, so the result is not Oracle-specific. The NLA scores 1/60, which is uninformative at a cell with no leverage.
- **C2.** The ":" L42 off-pair mass is the mixture described under 001.
- **C3.** CW's removals take 0.1–1% of ‖h‖², so "survives J-removal" cannot separate reading from re-solving; only a probe-validated eraser can (V2-E7).

---

## 6. Cathryn's J-subspace work and the advisor's checks

**Her repo has moved past the pinned commit** [V] `git log 9a46bf9..origin/main`:
- 04484b2, "cue removal (CW-11), J split with random and PCA controls (CW-12), bridge rank and template lens (CW-13)";
- 7b62673 (summary table; claim corrections).

**CW-12 as it stands now** [V: read from `git show origin/main:STATUS.md`; numbers not re-derived].
- **Setup.** 30 multihop items (the first 30 of CW-4's 50), L42 for both readers (**off-contract for the Oracle**), 2 samples per cell, word match.

| Piece | Energy share | Oracle | NLA |
|---|---|---|---|
| Whole activation | 1.00 | 45% | 43% |
| J part | 0.16 | 48% | 42% |
| Rest | 0.84 | 20% | 12% |
| Random 1,024 | 0.20 | 17% | 35% |
| All but random | — | 45% | 38% |
| PCA top 1,024 | 0.91 | 58% | 45% |
| All but PCA | 0.10 | 13% | 5% |

- **Contrasts:**
  - J − rest: Oracle +0.28 [0.12, 0.47], NLA +0.30 [0.12, 0.48];
  - J − whole: about 0;
  - J − PCA: −0.10 / −0.03.
- **Probe:** coarse prompt facts decode at AUC 0.93–1.00 from every piece.
- **Her own reading:** "sufficiency is not specific to J-Lens" and "the Oracle needs the J part intact; the NLA copes with a thinned copy".

**The bootstrap and advisor files quote older numbers** (29 items: 43 / 47 / 17 / 14 / 43) and don't mention the NLA arm or the PCA control.

**Advisor's checks** [D] `docs/claude_reports/advisor/JBASIS_CHECKS_2026-10-05.md`; numbers not re-derived, since the script and JSON weren't attached.
- **Construction:** the basis is fixed, built from J_NP and W_U only, not fitted on any multihop item, and set mainly by J.
- **Geometry:** the J subspace holds 32% of item variance vs 20% for a random subspace. Its item fraction after renormalisation is 62%. The logit-lens subspace holds item variance at random level (18%).
- **Reading:** the "rest" result is not explained by generic item variance, and the J subspace is a token-identity subspace. So **this split cannot separate reading the bridge from re-solving the question.**

[M] I agree with the advisor's main reading. The raw readouts in §12 make the re-solving point concrete: the J-part readouts restate the question.

---

## 7. The literature, as it bears on the project

- **Li et al.** On feature-extraction tasks a zero-shot prompt baseline matches verbalizers. Inversion followed by answering is competitive. Verbalizers lean on their own parametric knowledge.
  → Every reader claim needs a prompt-only arm. Our verbalizers are fine-tunes of the subject, so knowledge mismatch matters less than prompt inference. [V] main text §3–§5.
- **AO (Karvonen).** Norm-matched add after layer 2. Context-prediction training teaches input recovery. Prompting baselines are strong on some tasks.
  → Our Oracle is not a Karvonen AO; only the injection idea and the text-inversion concern carry over. [V] main text §3.
- **AObench.** Text inversion is the core evaluation confound. Use AUC on yes/no margins. Single-token reads underestimate AOs (1.26 vs 2.10 out of 5 with 50 tokens). Consensus sampling reduces hallucination. Narrow AO training matches probes but never beats them.
  → Our single-cell design handicaps verbalizers, so state this beside every comparison. [V] full.
- **Dingeto.** The reconstruction test is insensitive to individual false claims: only about 2% of specific claims are grounded on a released Qwen-2.5-7B NLA. Co-trained pairs develop private codes. Decodability is not verbalizability.
  → AR/FVE scores are not truth signals. [V] main text.
- **NLA paper.** Warm-started as a context summariser. "Excessive expressivity" is a named limitation. The AR is only a weak per-claim verifier. Claims that recur across positions are more often true.
  → Re-solving is a documented failure mode, not a hypothesis. [V] intro, method, confabulation sections.
- **WorkspaceBench.** Pass-if-any-cell scoring, unequal readout budgets, a prompt-blind summariser for token readers, and a prompt-only "upper floor" that often matches readers. The multihop read position is the last word.
  → Benchmark success does not imply internal-state recovery. [V] full.
- **R-lens.** LRP rules in the backward pass give earlier readouts and stronger intermediate-ablation effects on most larger models. MLP gain is similar for J and R.
  → Early-layer wins fall under Biderman's K+2 objection (03 §13.2). [V] full.
- **Hong.** The NLA AR depends mostly on (mostly false) specifics. About 95% of the reconstruction gains fall outside J-space. [V] full.
- **Init-robustness post.** An implausibly initialised NLA reaches near-control FVE while emitting 99.3% implausible statements. [V] first ~40%.

---

## 8. Open items waiting on you [D] STATUS, CLAUDE.md, SPEC_v2 §13

1. **Merge.** `005-amended-full-run` into `main` (you do this).
2. **Next direction.** 005 is closed. 007 (J-subspace split with controls) is proposed in the bootstrap; no design is written.
3. **The 006b correction** (§5): whether and how to edit `STATUS.md:20`, the CLAUDE.md 006b bullet and `report 006:104`.
4. **Framing confirmation** (consolidated handoff, 2026-10-03).
5. **Still open from earlier:**
   - P5 prior value (A01 §5);
   - SAE arm (BSF candidate with no labels);
   - gating the relational pairs;
   - a judge validation set (human-coded) before any further judged number;
   - multihop dev/test grouping (SPEC_v2 §13 item 9);
   - rotating the OpenRouter key (pasted in chat on 10-02).
6. **Files you meant to attach:** `jbasis_checks.py` and `.json`.

---

## 9. Problems I found (bootstrap, advisor file, repo)

1. **The Oracle at L42 is off-contract** [V]. olens `README.md:49` lists layers (20, 24, …, 60), and `:96` says "Layers outside 20–60 step 4 are off-contract. The student never saw them". A01 C6: the L42 arm is reported separately and never pooled.
   - The bootstrap's "L42 is also valid for it" is wrong.
   - All CW-12 Oracle numbers are off-contract.
2. **Stale CW-12 numbers; arms already run** (§6) [V].
   - The NLA arm's "prediction fixed now" (rest/whole ≥ 0.6 vs about 0.4) is already answered in her data (12/43 ≈ 0.28), so it isn't blind.
   - A dims-matched PCA-1,024 control already exists (hers was built on benchmark activations, not wikitext).
3. **Read position** [V].
   - Her residuals follow WSB's per-item `readout.offsets` (kind `last_word_token`): −2 for the 33/100 prompts ending in a bare space, −1 for the other 67 (tokens ' is', ' number', ' "', ' the', …).
   - "Position −2 (' is')" is an oversimplification. Our `wsb.py` sets no multihop `read_positions` (`render_item` handles brew, chain and relational only).
4. **Gain convention** [V]. Her J subspace uses raw W_U. The advisor's logit-lens control uses W_U ⊙ (1 + w), and our `jspace.py` atoms fold in g.
   - So the H_shared control differs from the J subspace in two ways (gain and J).
   - The overlap-≈1 check must use her convention or it can fail for a trivial reason.
5. **"`readers.NLArl` merges adapters in place" is wrong** [V] `readers.py:15-17,539-563`. The in-place merge is the nla-L42 class (`:329-341`). The NLA arm's constraint is memory only (`check_gpu_room`).
6. **Split tension** [M]. The first 50 multihop items have been used throughout CW-1 to CW-13 and informed the 007 hypothesis, so by SPEC_v2 §4.R they belong in dev. That leaves at most 50 test items, which conflicts with "use all 100". Using your own prompts removes the problem.
7. **Minor staleness** [V].
   - CLAUDE.md and COME_UP_TO_SPEED say "29/29 hashes" and "78 tests"; this rebuild verified 40 artifact files and collected 133 tests.
   - The `docs/claude_reports/README.md` index stops at 19, although `20_006_reader_likelihood.md` exists (I'm adding rows 04b, 20 and 21).
   - `test_nla_rl` cannot pass in a full GPU test run (§2).

---

## 10. My assessment [M]

**Weakest point in the current evidence.**
1. **The project's target does not exist yet at content level.**
   - Every reader result so far is scored against decodability (001 probes, 006 likelihood tracking), task truth (002) or the model's own output (006b).
   - None is scored against content-level demonstrated influence.
   - The reliable labels are position-level, and those cannot adjudicate a content claim.
   - So "agree on what the model demonstrably uses" is currently answered for positions only.
2. **Brew cannot separate z from answer geometry.** z is the runner-up answer in 56% of tables and fully recoverable from the prompt. Every brew reader result therefore carries the output-facing alternative (006b).
3. **The reader instruments sit at their positive-control bars.** NLA G-pc passed exactly at the bar, NLA G11 failed, and Oracle G-pc failed with a post-hoc rescue. The likelihood readout is the project's newest and least validated instrument.
4. **Small n and dev only.** Verbalizers on 10 tables in 002; CW-12 on 30 items × 2 samples. There is no test-split result anywhere.

**What 007 can and can't show.**
- **Can:** whether the Oracle's bridge signal depends on the J subspace beyond what a content-matched generic subspace holds (H_J vs H_content), and whether J's edge is just avoiding the shared directions (H_shared).
- **Can't:**
  - reading vs re-solving (the J subspace is a token-identity subspace; §12);
  - anything about what Qwen uses: it is reader-side only.
- **A cheap model-side arm would tie it to the question:** project out the J subspace vs a content-matched PCA-k (reader-free) at the read cell and measure the multihop answer margin.
  - Multihop has no partners, so this gives a necessity-style "does removing the piece cost the answer", not a swap.
  - That's your call.
- **Decisions the design must make explicitly** (beyond the bootstrap's list):
  - contract layer (L40/L44 basis, or an off-contract L42 arm);
  - gain convention;
  - per-item offsets;
  - dev/test (see §9.6);
  - Oracle likelihood positive control on multihop entities (006's colour G-pc failed on a prior, so use leave-one-out-prior identification as a pre-registered secondary);
  - many derangements for the shuffle.

**The single highest-value next step** (if you want 007): rebuild the L42 basis from our pinned J_NP under her convention (raw W_U) and check overlap with her file. That takes minutes on CPU, before any design choice depends on it. Then decide the layer question.

---

## 11. Doubts: where I could be wrong

- I didn't read the 001–004 design specs, the review folders, the 005 pilot steps in detail, or most of the code. A convention defined only in those places could be missing from this report.
- **CW-12 numbers** are from her STATUS text, not recomputed. The advisor's numbers aren't re-derived either (script not attached).
- **"Raw W_U"**: WSB `backend.unembed` = `get_output_embeddings().weight`, and Qwen3.6-27B has `tie_word_embeddings: false` [V] (model `config.json`). So her basis uses the untied `lm_head` without the final-norm gain. How much the gain changes the subspace is untested; it needs the overlap computation.
- **"First 50 items are touched"** assumes her and the advisor's 50 residuals are the first 50 bank items (advisor's file; her STATUS says CW-12 used "the first 30 of the 50 multihop items of CW-4").
- **My first-order critique** of the paper's Fig. 8 is untested on Claude and its premise fails on Qwen (005 A3).
- **The 006b correction:** I checked the numbers but not the D statistic's code (`model_output_analysis.py`).

---

## 12. Qualitative anchors (raw readouts)

Oracle `olens_s3d_rl600` at L42 (off-contract), CW-12 `runs/cw12/olens_split.jsonl` @ origin/main, 2 items drawn with `random.seed(20261005)`, first 2 samples each, truncated. Conditions: `full`, `J1024`, `N1024` (the rest), `R1024` (random), `RN1024` (all but random). [V]

**`inv-antarctica-opposite`**: "Fact: The cardinal direction opposite to where Antarctica lies from the equator is "; bridge `south`, answer `north`.
- `full`: "…the opposite direction from the Southern Hemisphere. Which direction is that opposite direction - 北 (North)…"
- `J1024`: "…: Another Hemisphere. What is the direction opposite to South? - 上方 (the opposite direction) is also [Direction]…"
  - "South" is named, but **inside a restated question**. Word match counts this as naming the bridge.
- `N1024`: "…**downward**. The - opposite to up. We traveled to a mining site…"
  - The generic "opposite" relation survives; the bridge and the geography don't.
- `R1024`: "…a quality of the sun… someone is moving from a bright area to a dark area…"
- `RN1024`: "…[opposite direction]. If the equator lies in that opposite direction, - 北 (North). Since the equator lies south of it…"

**`etym-caesar-monthnum`**: "Fact: The month named after Julius Caesar is month number "; bridge `July`, answer `7`.
- `full`: "- 6 in the year. If March is month number - 3 (or month number 9…)…"
- `J1024`: "- 4 (fourth month) of the year? What is the numerical number that comes after October…"
- `N1024`: "- 2 in the calendar year, and its full name is "June". It is the sixth…"
- **No condition names July.** Every piece keeps the "month number" frame, i.e. the prompt's task structure.

[M] Two items are anecdotes, not evidence. They show the failure mode that word-match "names the bridge" can't distinguish: question restatement followed by solving. 007's scoring should at least flag restatement (A7 item 4).

---

## 13. How to check this report quickly

- **Contract layers:**
  ```
  grep -n "20–60\|off-contract" /workspace/.hf_home/hub/models--agu18dec--olens_and_ar/snapshots/*/README.md
  ```
- **Raw W_U in WSB:**
  ```
  sed -n 56,60p /workspace/sources/repos/workspace-bench/src/wsbench/produce/backend.py
  ```
- **No multihop read positions in `wsb.py`:**
  ```
  sed -n 100,116p scripts/common/wsb.py
  ```
- **Per-item offsets:** in `workspace-bench/evals/multihop/items.json`, the `readout` field is −1 for 67 items and −2 for 33. The positions in her `runs/abl50/resid_L42.pt` follow them.
- **Her newer commits:**
  ```
  git -C /workspace/sources/repos/nla-vs-lenses log --oneline 9a46bf9..origin/main
  git -C /workspace/sources/repos/nla-vs-lenses show origin/main:STATUS.md | sed -n 159,195p
  ```
- **NLArl is not merged:**
  ```
  sed -n 539,566p scripts/common/readers.py
  ```
- **The 006b numbers:** `docs/reports/006_reader_likelihood.md`, lines 15–24 (reader D by role), 56–60 (cells), 79–81 (collision-free), 86 (per-pair r) and 94–103 (006b).
- **Instance state:** `results/000_setup/analysis/artifact_hashes.json` (`all_ok`, 40 files); the GPU-room failure reproduces with `pytest scripts/common/tests -m gpu`.
