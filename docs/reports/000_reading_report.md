# 000 — Reading report: reader adjudication on Qwen3.6-27B

*2026-10-02. Reading pass only. No code written, nothing run on the GPU, no model or lens downloaded. The spec is `docs/context/SPEC.md` (a verbatim copy of the 2026-10-01 spec, byte-identical to the `docs/reader_adjudication_spec.md` this report was written against, so the line citations hold). The reading list is saved verbatim as `docs/context/READING_LIST.md`. Sources are cloned or downloaded under `/workspace/sources/`. On this instance `/workspace` is not a persistent volume.*

**Citation keys.**
- SPEC = `docs/context/SPEC.md`, cited by line.
- RHUT = `senku14x/Readable-Here-Used-There`, branch `extension` @ `d4197b0` (2026-09-24).
- SEC = `senku14x/jacobian-lens`, branch `research_secant` @ `a3720ab`. The reading list's paths omit the `research_secant/` prefix the files actually sit under.
- WSB = `camilablank/workspace-bench` @ `92d763e`.
- WP = Gurnee et al., arXiv 2607.15495v1, cited by PDF page (p.N).
- Other papers are cited by arXiv id and PDF page.

---

## 1. Reading log

| Source | Access | Status | Coverage and notes |
|---|---|---|---|
| Spec | local | Full | §0–§13 and references. §6 re-read for section 3 below. |
| Reading list | pasted in chat; no file existed | Full | Saved verbatim to `docs/context/READING_LIST.md`. |
| RHUT `STATUS.md` | clone | Full | Top block, the 2026-09-24 update, and the older blocks. |
| RHUT `context/evidence_ledger.md` | clone | Full | §A–§E, including the freeze row. |
| RHUT `H3/results/reports/two_hop_question_turn_ladder.md` | clone | §0–§3 and §9 | §4–§8 not read. |
| RHUT `CLAUDE.md` | clone | Full (91 lines) | You asked for three sections; I read all of it. |
| RHUT `common/scripts/` | clone | Line by line | `registry`, `materials`, `rendering`, `readout`, `hooks`, `hooks_fp32`, `stats`, `instrument_gates`. Also `common/configs/instrument_registry.json` and the GPU half of `instrument_gates_result.json`. `make_figures.py` not read (not on the list). |
| RHUT organism specs | clone | Full | `H1/design_specs/natural_modulation.md`, `tagged_selection.md` (with Amd 2–3), `computed_sums.md`; `H3/design_specs/computed_sum_consumer.md` (with Amd 1–3). |
| RHUT `H3/design_specs/two_hop_organism.md` | clone | §0–§4 and Amendment 7 in full | Amd 1–6 and 8–10 headings only. The `two_hop_items*.json` banks inspected programmatically. |
| RHUT `H3/results/reports/query_local_workspace.md` | clone | `nosource`, `nosource2` and the interpretation (L82–186) | `queryread`, `querycausal` and `clampsplit` tables not read. |
| RHUT other | clone | Targeted | `rebuild_env.sh` read. `context/full_project_writeup_2026-09-11.md` read at L455 only, to source a seed number. `context/paper/workspace_paper_notes.md` not read; I used the PDF. |
| SEC `README.md`, `STATUS.md` | clone | Full | |
| SEC `scripts/lib/engine.py`, `scripts/lib/mfl.py` | clone | Line by line | |
| SEC `docs/workspace_bench/README.md` | clone | Full | Also `designs/000-matrix-free-lens.md` in full, for the design-doc convention, and `requirements.txt`. |
| WP | arXiv PDF | Full | Text extraction of the main text (pp.1–60) and appendix (pp.61–105). Read closely: §2, §3.1, §5, §6, A.9, A.12, A.21, A.24.3. References looked up for cited items only. Figures seen as captions only, except Fig. 82 (p.93), which I viewed as an image. |
| WSB repo | clone | Partial, listed here | `README.md`, `CLAUDE.md`, `docs/producing_readouts.md` (L1–140), `evals/baselines/README.md` (L1–80). `src/wsbench/produce/{backend,render,methods}.py` line by line. Family READMEs: brew (full), chain (full), arithmetic (L1–140), multihop (L1–60), relational (L1–45), role-bound (L1–60). All six `items.json` banks inspected programmatically. Judge code, `readplan.py` and `producer.py` not read. |
| WSB blog post (LessWrong `Zeg2JztbdhguL48uH`) | GraphQL markdown | Full | Results figure viewed as an image. |
| Zhang & Nanda, 2309.16042 | arXiv | Main text pp.1–10 | Appendix not read. |
| Heimersheim & Nanda, 2404.15255 | arXiv | Full | |
| Makelov, Lange & Nanda, 2311.17030 | arXiv | Main text pp.1–27 | Appendices pp.28–58 not read. |
| Elazar et al., 2006.00995 | arXiv | pp.1–7 (§1–§5) | Rest of §6, §7–§8 and appendix not read. |
| Hewitt & Liang, 1909.03368 | arXiv | pp.1–4 | Results sections only via the abstract and introduction. |
| Geiger et al. (DAS), 2303.02536 | arXiv | pp.1–9 (§1–§3, method and setup) | Experiments and appendices not read; the list says implementation only. |
| Anthropic NLA paper (transformer-circuits.pub/2026/nla) | HTML to text | Sections relevant to loading and calling | Read: Introduction; Method, Architecture, Init, Training; "Reasoning about Rewards"; "Characterizing confabulations"; Discussion and Limitations. Appendix: AV scaling factor, AR prompt, log reward, layer sensitivity. Other case studies, evaluation-awareness and auditing sections not read. |
| Karvonen et al. (Activation Oracles), 2512.15674 | arXiv | pp.1–5 | Injection method and training data. |
| Li et al., 2509.13316 | arXiv | pp.1–7 (§1–§5.2) | |
| Bauer et al., 2606.02609 | arXiv | pp.1–10 | Main text plus App. A.1–A.2. The paper is titled *Building Better Activation Oracles*; it introduces AObench. |
| R-lens post (LessWrong `nv8oedrnLXKRzNEL9`) | GraphQL | Full | |
| `agu18dec/olens_and_ar` | Hugging Face | README in full, file list | |
| `ceselder/qwen3.6-27b-nla-rl` | Hugging Face | README, `nla_meta.yaml`, `run_config.yaml` in full | |
| `ceselder/qwen3.6-27b-nla-L42` | Hugging Face | README | A second L42 NLA by the same author (see §4.5). |
| `camilablank/workspace-lenses` | Hugging Face | README, `qwen3.6-27b/` file list | |
| `neuronpedia/jacobian-lens` | Hugging Face | `qwen3.6-27b/` file list | |
| Skip-lens | Hugging Face | Cards for `-cnla-rl` and `-futurelens-rl`, plus a listing of about 22 `ceselder/skip-lens-qwen36-27b-*` repos | |
| SAE / BSF at L42 | — | **Not located** | No Hugging Face hit. WSB reports an `sae-l42` arm but gives no path. |
| `agu18dec/local-workspace` | — | **Not accessible** | Private per SEC; I did not re-test. The oracle's AR whitener moments live only there (per the `olens_and_ar` README). |
| Dingeto 2026, 2607.20379 | arXiv | pp.1–2 | Exists; claims in §4.5. |
| "Hong 2026 (fact-check reward)" | LessWrong `DFgt8fi3Wzwwe2Sib` | Full | Seong Pyo Hong, *Fixing rewards for NLA to reduce confabulation*, 2026-07-24. |
| "michaelzhang & TurnTrout 2026" | LessWrong `LQXWiF8PyJ5ojNsEv` | Introduction, setup, results, conclusions | *How robust are natural language autoencoders to initialization?*, 2026-07-10. |
| Tier 4 | — | Skipped | As instructed. |

**Environment, read-only check:**
- Present: torch 2.11.0+cu128 (matches RHUT's pin), on an RTX PRO 6000 Blackwell (sm_120, 96 GB) with driver CUDA 13.2.
- Absent: transformers, peft, accelerate, safetensors, scipy, sklearn, jlens.
- No model weights on disk.

## 2. The project in my own words

WorkspaceBench and the workspace paper grade readers by whether they name a chosen variable. Nanda proposes ranking readers by agreement with the majority. Neither has a reference that says whether the named content matters to the model. The readers also share failure modes:
- J and R share an estimator.
- J, R and the logit lens share the unembedding.
- The Oracle and the NLA share a verbalizer-plus-reconstructor lineage.
- Most can see what the prompt implies.

The project builds that reference without any reader. Each (item, cell, layer) of a token-aligned counterfactual pair on Qwen3.6-27B gets one of four labels from a mean-difference direction fitted on held-out templates:
- **Used:** a 1-D swap moves the specified answer beyond norm-matched random directions.
- **Decodable-not-needed:** z is decodable, but replacing the whole cell barely moves the answer.
- **Absent.**
- **Unresolved.**

Every reader is then read at the same cells with a matched claim budget and scored closed-set against z. Its hits are compared to the labels: sensitivity, specificity, and the selectivity W = P(hit | used) − P(hit | decodable-not-needed) within family × position. Pairwise agreement is split by label into shared signal, shared error and shared prompt inference. The result is a map of which readers report content that one specified answer depends on, and what their agreement is evidence of. It is explicitly not a definition of the workspace.

## 3. Spec §6, restated

**Notation.**
- Item i has a token-aligned partner i′. The latent is z vs z′; the answers are a vs a′.
- c is a token position and ℓ a layer; $h_{\ell,c}(x)$ is the output of block ℓ at position c.
- Each family has template halves A and B, and items are split dev/test.
- Every step works at a single (c, ℓ).

**Step 1 — Reader-blind direction** (SPEC L169–170).

$\mu_z=\mathbb E_{x\in A,\,z(x)=z}[h_{\ell,c}(x)]$, $\hat u_{zz'}=\operatorname{unit}(\mu_{z'}-\mu_z)$, with one direction per ordered contrast, never averaged across contrasts.

Why:
- No lens and no trained objective touch it, so the labels cannot inherit any reader's errors (the project's central non-negotiable).
- It is fitted on half A and applied on half B, so it does not absorb the noise of the item being labelled.
- One direction per contrast, because averaging contrasts with different targets cancels.
- Mean-difference rather than DAS because DAS is optimised to produce the effect and would over-label (SPEC L187–190).

**Step 2 — Swap** (L171–172).

On items in half B: $h'=h+(\hat u\cdot h^{(i')}-\hat u\cdot h^{(i)})\,\hat u$, then the rest of the model runs in fp32.

Why:
- It installs the partner's own coordinate, so the dose is the natural one, not a free α.
- The orthogonal complement is untouched. This is exactly Makelov et al.'s Eq. 1 (2311.17030 p.7).
- fp32 because RHUT's bf16 coordinate writes realised only 0.53–0.59 of the requested norm (RHUT `STATUS.md` L34).

**Step 3 — Effect, share and ceiling** (L173–177).

m is the full-sequence log-prob margin between the two answers, with log-sum-exp over accepted spellings. My reading of the sign is $m(x)=\log\sum_{s\in S(a')}P(s\mid x)-\log\sum_{s\in S(a)}P(s\mid x)$, so that $G_i>0$.
- $\Delta m_i=m(h')-m(h)$.
- $G_i=m(x_{i'})-m(x_i)$, the whole-run counterfactual gap.
- $S_i=\Delta m_i/G_i$.
- The cell ceiling $\Delta m^{\text{cell}}_i$ is the effect of replacing the entire residual at (ℓ, c) with the partner's.
- "Demonstrated influence" means $S_i$ exceeds the 95th percentile of shares from norm-matched random directions.

Why:
- A margin, i.e. a logit/log-prob difference, is close to linear in the logits and ignores effects common to both answers. Both patching papers recommend it (2309.16042 p.8; 2404.15255 pp.8–9).
- Whole spellings, because first-token scoring credited prefixes in RHUT (silly mistake 5).
- $G_i$ as the denominator makes shares comparable across cells. A cell-level denominator would be tiny at carrier cells by design.
- Random directions guard against generic perturbation.
- The full-cell patch bounds anything a 1-D edit at that cell could legitimately do.

**Step 4 — Necessity tier** (L178).

Set the û-coordinate to the pooled mean in both x and x′; $N_i=1-G^{\text{abl}}_i/G_i$. This is reported as a tier, not required for "used".

Why:
- A 1-D swap can work by activating a pathway the unmodified model does not use (Makelov et al.).
- Removing the coordinate from both natural runs asks whether the natural difference passes through it. This is noising, i.e. necessity, in Heimersheim & Nanda's terms (2404.15255 §2.3).
- It is not required because other cells can carry z redundantly.

**Step 5 — Item-level decodability** (L179).

- A probe for z at (ℓ, c) is cross-fitted with folds split by template.
- $p_i$ is the held-out probability of the true z.
- The item is "decodable" if $p_i$ exceeds the 95th percentile under shuffled-label probes.
- Family-level accuracy is only a gate.

Why:
- The labels are per item, so the decodability call must be too.
- The shuffled-label null calibrates chance.
- Decodable does not mean used (Elazar et al.; WP §3.5.1).

**Step 6 — States** (L180–184). Each (item, cell) gets exactly one state:

| State | Condition | Why this definition |
|---|---|---|
| Used | Step 3 passes. "Used + necessary" if step 4 also passes; the spec gives no pass criterion for step 4. | — |
| Decodable-not-needed | Decodable, and full-cell share $\Delta m^{\text{cell}}_i/G_i<\varepsilon$ (e.g. 0.1). | Uses the full-cell patch, so a badly chosen direction cannot manufacture a "not needed". Only redundancy can. |
| Absent | Not decodable, **and** the probe detects a planted copy of z at a behaviour-relevant dose, **and** full-cell share < ε. | Requires a positive control, so an insensitive probe cannot manufacture an "absent". |
| Unresolved | Everything else, e.g. a large full-cell effect with a failed 1-D swap, or a probe failure without a passing positive control. Counted per family and never scored. | Keeps the label method's own failures out of reader scoring. |

**Step 7 — Per consumer** (L185).

Organisms with two consumers are labelled per consumer, e.g. report and parity reading the same sum carrier.

Why: "used by one consumer, ignored by another" is itself evidence for interface-type use.

**Gates and conventions** (L192–198).
- Write fidelity: ρ ∈ [0.9, 1.1] and κ ≥ 0.99, measured after the write, through an fp32 hook.
- Split-half label reliability ≥ 0.7 across the A/B halves before any reader is scored.
- A family with more than ~40% unresolved cells is excluded from W.
- If mean-difference and DAS labels disagree badly, the labels get fixed first.
- Share is the primary measure; flips are reported beside it, with any cell whose label changes between the two listed. RHUT's 12-item question-turn stage is the reason: the intermediate's plane carried 0.20 of the continuous effect, yet removing it cut flips from 18/48 to 1/48.

## 4. Where the sources and the spec disagree or leave gaps

### 4.1 RHUT (code and records) vs what the spec assumes

1. **The maintain/mention/ignore organism has no question consumer.**
   - `natural_modulation.md` is readout-only: conditions L19, measurements L22–30, and LOGITS is "secondary … the primary consumer sites come in H3" (L29).
   - RHUT's consumers exist only for the tagged organism (codebook letter, competence 1.00; `STATUS.md` L83), the sum organism (parity and report; `computed_sum_consumer.md` L153–154) and two-hop (its question turn).
   - SPEC L141 ("same question consumer") and the slice (L366) therefore need a new consumer turn. They also need a competence gate under "ignore", which the reading list's step 4 already anticipates.
2. **"Same carrier position" across conditions is not the same absolute position.**
   - The three tails differ in wording and token length (`rendering.py` L15–21): maintain is `. Keep that word in mind while you copy…`, ignore is `. That word is irrelevant here. Now copy…`.
   - Matching therefore has to be by relative carrier index, with partners geometry-matched within each condition.
3. **The RHUT magnitudes quoted as seeds come from whole-span, multi-block interventions, not single-cell edits.**

   | Seed in the spec | What RHUT actually did |
   |---|---|
   | Sum carrier at "1.04 of source readout at 0.08 share" | C_full / O_donor: the entire carrier span overwritten at every block ≥ 36. Behavioural 0.083 for report and 0.181 for parity; J_NP 1.04 (`full_project_writeup_2026-09-11.md` L455). |
   | Two-hop carrier 0.27 | The same whole-carrier, blocks ≥ 36 donor, on 12 released items in bf16 (ledger, `two_hop_organism` row). |
   | Absent-source 28/32 | Transplants the whole carrier at all 64 blocks (`query_local_workspace.md`, `nosource2` and interpretation §5). |

   §6 labels a single (cell, layer) edit. None of these numbers predicts single-cell L42 effects. I expect those to be much smaller wherever z is held redundantly across positions (inference, untested).
4. **The spec's RHUT summary repeats an error RHUT itself corrected.**
   - SPEC L127 says the best 25 vocabulary-J directions "install about 0.39, against 0.04 for rank- and norm-matched random directions".
   - RHUT's freeze record §9 item 4 (ladder L173) says this pairing compared an installed share with a removal cost. Removing 25 random directions costs 0.04; removing the best 25 costs 0.39; the best 25 also install 0.39.
5. **The two-hop bank does not have the structure the label recipe assumes.**
   - Each item is one ordered pair, a recipient cue plus one donor cue (`two_hop_organism.md` Amd 7 §1).
   - Each intermediate appears as recipient in 1–3 of the 42 items (counted from the bank: Norway 3, Italy 3, the rest ≤ 2).
   - Each relation family has a single template.
   - So "class centroids on template half A" reduce to one item's pair difference, and a family has no second template to split on.
   - Admission criterion (b) did use `CoordSwap` along J_NP folded directions at blocks 36–50 on the tail positions (L300). This confirms the spec's re-admission requirement.
6. **Precision regime.**
   - RHUT's float32 regime (`hooks_fp32.py` L8–31) upcasts the residual from one block onward but keeps the forward under bf16 autocast. Matmuls, attention and the GatedDeltaNet kernels still see bf16 operands.
   - SEC's engine (`engine.py` L14–16, L44–55, L125–138) replays blocks ℓ+1..63 in true fp32 with TF32 off. SEC `STATUS.md` L45–46 shows bf16-regime finite differences reach only cos 0.80.
   - SPEC L171 asks for "the rest of the model in fp32", which matches SEC, not RHUT.
   - Whichever regime is used, the clean margin m(h) must be computed in the same regime as the edits.
7. **Sequence-length sensitivity.** Appending one token changes earlier positions' activations: |Δz| up to 0.33 at L48 (`instrument_gates_result.json`, section B; `instrument_gates.py` L141–146). Partners must match length exactly, which token alignment already requires.
8. **L42 is a GatedDeltaNet block.** Full attention sits only at blocks i ≡ 3 (mod 4) (`instrument_registry.json`). Residual patching is unaffected. Attention-read cuts would not apply at 42, and RHUT found masks are not complete cuts anyway (`STATUS.md` L35).
9. **Sums.**
   - These are the only RHUT materials where z varies independently of its inputs: eight sums, each with a same-sum sibling operand pair (`materials.py` L99–102).
   - `thirteen` and `fourteen` may be multi-token (`computed_sums.md` L22).
   - The spec's extension to the number words two…twenty needs a single-token check.
10. **No damage gate.** RHUT gates every intervened forward on damage: top-1 retention ≥ 0.95 and ΔNLL (`CLAUDE.md` L71). §6 has no damage gate for label edits.

### 4.2 WorkspaceBench item schemas vs the cell format

1. **Brew has no donors.**
   - Its item fields are `name, mode, depth, n_rows, table, start, stirs, intermediates, untaken, answer, off_trajectory, options_adjacent, options_random, prompt_first, prompt, gate, regions, eval_positions`.
   - The "token-aligned donors" in SPEC L109 must be constructed: same table, different start colour. Each needs gating.
   - WSB gated only the given start (`"orders": ["last"]`); 139 of 400 candidates passed 10/10.
2. **Brew has one prompt template.** `prompt_first` exists but was never gated. Brew has no template halves; A/B would have to be halves of items (tables).
3. **Brew positions are absolute indices from the source repo's renderer.**
   - The 24 pinned cells are labelled stir (6), start (6), question (9), `emit_asst`, `emit_think` and `emit_colon`.
   - The stir line comes before the start colour, so z cannot be computed there. WSB uses those cells as its negative control (brew README table). This gives a label-free "absent" class.
   - WSB's backend loads Qwen without a revision pin (`backend.py` L29–31), so the indices must be re-verified against the pinned `6a9e13bd` tokenizer.
4. **WSB's own caveat about brew:** "the intermediate is usually the model's runner-up answer colour, so this bank cannot separate a held step from a runner-up output" (brew README L60–61). The prompt-only baseline scores 0.96 there (L58).
   - Scoring the margin between a and a′ (z is not an answer) partly protects the labels.
   - But brew's .86 is a positive control for *readability*, not for intermediate content.
5. **What the reproduction target actually is.**
   - Brew .86 is the J_NP *cosine* readout: raw W_U, no final norm (`methods.py` L89–125).
   - It is read at 11 layers (20–60 in steps of 4) × 3 emission cells, and an item passes iff K·G > ΣO over that grid (brew README, Scoring), with a Gemini 3.8 Flash judge.
   - L42 is not on WSB's grid.
   - Reproducing it needs OpenRouter access. The basic multi-token .24 also uses the Gemini summariser (README §Judges).
6. **Chain.**
   - The bank has no `regions`, `eval_positions` or donors. The start number comes last, and the read cell is the final token of the chat render.
   - Qwen-family tokenizers split numbers into digits; I have not checked this tokenizer. WSB's write-up treats multi-digit numbers as unordered bags for token readers (blog L322).
   - So a token reader cannot name a two-digit z as one token. P7's "J near floor on chain" is then partly guaranteed by the tokenizer.
7. **Multihop.** Plain render with no chat template, read at the last word, 100 items, no donors. RHUT's two-hop is chat-rendered inside the copy organism. Pooling both into one "two-hop" family (SPEC L110) mixes renders and decision-cell definitions.
8. **Relational multihop.** The a/b items swap the two relation words, e.g. "Avery's landlord is Sam. Sam's sibling is Riley…" vs "Avery's sibling is Sam. Sam's landlord is Riley…". Pairs are token-aligned only where both relation words have equal token counts.
9. **Role-bound association: the ab and ba members are different generated stories with different names.**
   - Checked on `oa-police-thief-chase`: Marcus/Dylan vs Marcus/Dmitri, entirely different text.
   - No token-aligned partner exists, so no §6 label can be computed from this bank as it stands (SPEC L112).
   - Its read regime appends "Summarize the preceding story in one sentence." and reads 19 suffix and chat-tail sites.
10. **Prompt-only.**
    - WSB's prompt-only is stock Qwen3.6-27B given the prompt up to the read token and asked what a model would be thinking (`evals/baselines/README.md`; blog L127).
    - SPEC L96 says only "an LLM". Li et al. argue the zero-shot control should share the verbalizer's knowledge, and our verbalizers are Qwen3.6-27B fine-tunes.
11. **WSB's headline rule is any-cell everywhere** (blog L144), and so are WP's own evaluations: A.6 uses any-layer pass@k (p.64); Fig. 10 uses any-(layer, position) top-1 (p.14). SPEC L157 rightly drops this. The reproduction step must still use WSB's rule, and its numbers cannot be compared with per-layer numbers.

### 4.3 The workspace paper vs the spec

1. **The paper's own reader-free intermediate probe decorrelates z from cues and answers by construction.** It averages over "prompts that imply the same intermediate through different surface cues and ask different questions about it, minus the mean over all intermediates" (§3.3, p.18). §6 step 1 has no such requirement (see §5.3).
2. **Precedent for §6, with differences.**
   - A.24.1 (pp.98–101) localises arithmetic intermediates with mean-difference patching to the layers where J first shows them. But it adds the full mean-difference vector at a single layer rather than swapping a 1-D projection.
   - A.12 (pp.79–81) uses a mean-difference probe with its top-25 J-space component projected out.
3. **The paper itself shows readable-not-needed and used-not-J-readable content.**
   - §3.5.1 (pp.22–24): "Spanish" is in the J readout in all four tasks, but swapping it is causal only for report and flexible computation.
   - The character count is absent from J under linewrap, yet the wrap point depends on it.
   - A.11 and A.12 repeat the dissociation.
   - So J misses on used content are expected in some families; J's sensitivity is bounded by design, not only by noise.
4. **What A.21's .853 actually measures** (pp.92–93):
   - A continuous J score (mean J log-prob of *fictional*/*simulation* minus 100 control words, averaged over workspace layers).
   - It separates 198 NLA-flagged from 594 sampled unflagged Assistant-turn starts on Opus 4.6.
   - The NLA supplies the labels, there is one concept, the prevalence is stratified to 25%, and the model is Claude.
   - It is convergent evidence on one concept, not an agreement statistic comparable to κ.
5. **The paper's oracle lens is a different artifact.**
   - WP's oracle lens is built on Haiku 4.5 (A.9.2, pp.75–78). `olens_s3d_rl600` is an independent Qwen implementation with a different injection and a different AR.
   - "The oracle lens extracts only those representations that are 'in the workspace', while NLAs attempt to reconstruct the entire activation vector" (p.78) is offered as "one interpretation". It is the hypothesis P2 tests, not a prior result.
6. **J reads future positions too.**
   - J sums the effect on the present position and every later one (§2.1, p.7; SEC `mfl.py` L3–5). A J readout at a carrier can therefore reflect what the model will say later, including the answer.
   - WP's default lens targets the penultimate layer (A.7, p.68). This matches J_CB and R_CB (target 62).
7. **All Qwen lenses are third-party.** The paper's results are on Claude models. The Qwen lenses are Neuronpedia's (n=1000, wikitext) and Camila's (n=25, pile-10k, penultimate target).

### 4.4 Patching and probing literature vs §6

1. **Zhang & Nanda** support §6's token-aligned pairs (STR) and its margin metric (pp.3, 8). They recommend single-layer patching before sliding windows (pp.8–9). §6 complies; RHUT's sustained 36–62 writes are the sliding-window regime, another reason not to port RHUT's magnitudes.
2. **Heimersheim & Nanda.**
   - Single-component edits are blind to redundant, OR-like storage (§2.4 p.4; backup behaviour, §3.2 p.7). The spec acknowledges this for necessity, but it applies equally to the single-cell "used" swap and to the single-cell "not needed" test.
   - A logit difference can be inflated by an edit that "indiscriminately damages the model" (p.9). The remedy is to inspect each answer's log-prob separately and add a damage metric; §6 has neither.
3. **Makelov et al.**
   - §6's swap is their Eq. 1. Their signature of an illusory subspace is a subspace patch that beats the full-component patch: vMLP 46.7% FLDD vs −8% for the full MLP (Table 1, pp.14–15).
   - Residual-stream directions survived their tests (pp.17–18), and they recommend working in bottlenecks (p.27). But non-final residual positions are not a full bottleneck for the answer.
   - The risk is low, not zero. A cheap guard is to flag any cell whose 1-D share exceeds its full-cell share.
4. **Elazar et al.**
   - Their removal deletes *all* linearly decodable information (INLP) and is compared with removing as many random directions (the Rand control, pp.3–5).
   - §6 step 4 removes one direction with no random-ablation control.
   - In RHUT's question-turn ladder, 0.78 of the effect remained in the complement after removing the intermediate's plane. A 1-D necessity test will often fail even for used content.
5. **Hewitt & Liang.**
   - A probe's null should be a control task: a random label per input type, held consistent across tokens (pp.1–3).
   - Per-row label shuffles do not catch a probe that memorises item identity.
   - Folds "split by template" put the same item in both train and test.
6. **DAS.**
   - It needs (base, source, counterfactual-label) training pairs and held-out evaluation (pp.7–9), and it overfits with few items (Makelov p.16, App. B.7).
   - Its own step 6 tests whether a found representation decomposes into input-identity representations (p.9). That is the check §5.3 asks for.

### 4.5 Reader artifacts vs spec §4

1. **Oracle Lens.**
   - Its read contract is layers 20–60 in steps of 4: "Layers outside 20–60 step 4 are off-contract. The student never saw them" (`olens_and_ar` README).
   - Its AR has layer tags for 20–60 step 4 and 63, but not 42.
   - The whitener moments needed for its trained (whitened) FVE are not public.
   - The AR loader and sampler come from a `global_workspace` package I have not found publicly. WSB reimplements the AO read in `methods.py` L172–257.
   - Injection: the ㈜ input-embedding row is replaced with 16000·unit(h), and the source layer is written into the prompt.
2. **NLA: two candidate repos by the same author.**

   | Repo | Verbalizer | Reconstructor | Notes |
   |---|---|---|---|
   | `ceselder/qwen3.6-27b-nla-rl` | `av_rl_adapters/iter_000400` is the one WSB uses (`methods.py` L269–270) | "Latest only": "pair any AV adapter with the provided (final) AR" | — |
   | `ceselder/qwen3.6-27b-nla-L42` | `av_rl_lora_step400` | `rl_critic_step400` | A step-matched pair. |

   - Both inject Karvonen-style: a norm-matched add at the output of block 1, at the ㈜ token.
   - Both were warm-started from explanations of *texts* written for a Qwen3-8B L24 NLA, with the activations re-extracted at L42.
   - So the spec's open question has an answer: the reconstructor is released.
3. **Skip-lens.** The cards I read are trained and read at layer 62. "Fed raw L42" is off-distribution, and the spec names no variant among about 22.
4. **SAE/BSF at L42:** not located.
5. **Every verbalizer discards ‖h‖.** The Oracle injects 16000·unit(h); the NLA and Skip-lens inject ‖h_marker‖·unit(h). A planted-concept "dose" sweep therefore changes only the direction they see.
6. **Priors for A5 point away from P5's "Δ(c) ≈ 0".**
   - Anthropic's NLA paper calls the AR "only a weak per-claim verifier": deleting true claims hurts reconstruction more than deleting false ones.
   - Dingeto finds that flipping a claim changes the reconstruction for only ~2% of salient specific claims (Qwen-2.5-7B NLA; stated as a lower bound).
   - Hong, on his own Qwen3.6-27B NLA, finds the AR depends mostly on specifics that are ~90% unsupported. Entity swaps cost −0.219 FVE vs −0.072 for abstract swaps, and −0.394 vs −0.103 on Anthropic's checkpoints.
   - michaelzhang & TurnTrout find that NLAs initialised on implausible statements reach near-control FVE while ~99% of their claims stay implausible. RL lowered the plausibility of normally initialised NLAs from 21% to 7.6%.
   - So "Δ(c) > 0 for the hallucinated claim", meaning co-adapted codes, is a live outcome that P5 does not list.
7. **AObench practice** (Bauer et al., App. A.1, p.10).
   - A single final-token activation understates AOs: 1.26 vs 2.10 out of 5 with 50 tokens of context.
   - Yes/no answers should be scored by logit-margin AUC.
   - Consensus over k samples cuts hallucination. The spec's 8-sample self-consistency is the same idea.
   - Single-cell reads are right for cell-matched labels, but they put verbalizers at their weakest operating point. That caveat belongs next to every W comparison.
8. **R-lens.** The published RelP rules cover RMSNorm and the gated MLP; attention and q/k norms are left unmodified (R-lens post, Methods). Nothing says how the 48 GatedDeltaNet layers are handled, and the `workspace-lenses` README calls the model "27B dense".

## 5. Ambiguities, contradictions, and things I think are wrong

### 5.1 "Used" and "decodable-not-needed" overlap, so "exactly one state" is ill-defined

**The problem:**
- "Used" is a significance test against random directions (SPEC L177, L181). "Not needed" is an effect-size test: full-cell share < ε (L182).
- I expect a mean-difference direction aligned with the pair difference to beat random directions at almost any cell with a nonzero full-cell effect. A cell with full-cell share 0.05 and $S_i$ = 0.04 would then pass both tests.
- Multiple comparisons make it worse: a per-cell 95th-percentile threshold labels about 5% of truly null cells as "used".

**What I'd change:**
- "Used" requires $S_i\ge\tau$ (absolute and pre-registered; e.g. τ = 0.1) *and* $S_i$ above the random-direction null after an FDR correction across cells.
- "Not needed" requires a full-cell share < ε′, with ε′ < τ (e.g. 0.05).
- Cells in between are unresolved.
- Separately flag cells where $S_i$ exceeds the full-cell share (the Makelov signature).

### 5.2 The unit of labelling is undefined, and the seed magnitudes are the wrong scale

**The problem:**
- SPEC L151 lists cells as source, carriers and decision, edited at one layer.
- The RHUT magnitudes quoted alongside (§4.1.3) are whole-span and multi-block.
- My untested expectation: at a single L42 cell, effects will be large only at the final positions and near zero at most carriers. "Used" would then collapse onto decision cells and "not needed" onto everything else — the position confound the matched contrasts were meant to avoid.

**What I'd change:**
- Define the cell (token or span; one layer or a band) in design 001.
- Make the reading list's step 1 a reader-free leverage profile run before any direction or probe: full-cell patch share for every pinned brew cell at L42 and two neighbouring layers, plus the same for the RHUT pair.
- If no family shows label variation *within* a position, W cannot be estimated however good the labels are. That is cheap to learn now.

### 5.3 The label direction certifies that the pair difference matters, not that z matters

**The problem:**
- $\mu_{z'}-\mu_z$ carries whatever co-varies with z in the fitting set: the input token that sets z, z itself, the answer, and anything downstream of them.
- With one to three items per intermediate (two-hop), it is effectively the pair difference.
- At source cells it will be dominated by the input token's identity. At decision cells it will be dominated by the answer's geometry — RHUT's own question-turn finding (answer plane 0.37 vs intermediate plane 0.19; `STATUS.md` L11).
- Readers are scored on naming z. A reader that names the answer at a "used" decision cell scores a miss, and a cell can be labelled "z used" with no representation of z involved.

**What I'd change:**
- Fit û on variation where z changes while inputs and answers are decorrelated from it, as WP's §3.3 probe does:
  - Brew: many (start, table) combinations share a z, and answers vary by table. Usable if the ~8 items per z are split carefully.
  - Sums: same-sum siblings.
  - Two-hop: needs new items with several cues and several relations per intermediate.
- Report the share kept after projecting out the answer plane and the input-token direction.
- Where a family cannot support this, call the label "pair difference used", and score readers on {input, z, answer} rather than on z alone.

### 5.4 The slice's matched contrast probably has no "used" arm, and its consumer does not exist yet

**The problem:**
- Even replaced as a whole span, RHUT's word carriers move a competent consumer by ≤ 0.04 of the ceiling (ledger §E, "Word organism").
- Maintain vs ignore changes readability. RHUT never showed that it changes use.
- With the source visible, I expect both arms at the carrier to come out decodable-not-needed or absent.
- The only clearly used carriers in the spec are the source-hidden installs, which §5 rightly excludes from W.
- The stop rule (L376–377) would then fire on the slice.

**What I'd change (your call):** either
- (i) keep maintain-vs-ignore as a decodability contrast inside "not needed", and get the used/not-needed contrast elsewhere; or
- (ii) use the tagged-selection organism at the question and decision cells.

  Option (ii) has three advantages:
  - Its consumer already exists (codebook letter, competence 1.00).
  - The three words give a *within-cell* contrast: the pointed word should be used, and the two unpointed words decodable-not-needed.
  - The answer is a letter, not z, which also removes the answer-geometry confound of 5.3.

  This is the cleanest same-cell used/not-needed contrast I can find in the existing material.

### 5.5 No single layer is on-contract for every reader

**The problem:** SPEC L154 makes L42 the core layer for all readers.
- The Oracle is off-contract there: trained on 20–60 in steps of 4, with no L42 AR tag.
- Skip-lens is trained at L62.
- The NLA, and per WSB the SAE, exist only at L42.

**What I'd change:**
- Core L42 for J, R, logit lens, NLA and SAE.
- Read the Oracle on-contract at L40 and L44, against labels computed at those layers for the same positions, plus an explicitly off-contract L42 arm with its own planted-concept curve.
- Report Skip-lens as an off-distribution arm.
- Never pool on- and off-contract arms.

### 5.6 Template halves are undefined for most families, and split-half reliability could be trivially high

**The problem:**
- Brew has one template, and two-hop has one per relation.
- If "template" means carrier, halves A and B share the clue or start text. Split-half agreement would then measure robustness to the carrier, not generalisation of the direction.
- Labelling only half B also halves the labelled data.

**What I'd change:**
- Define A/B per family in design 001:
  - Brew: table halves stratified by z.
  - RHUT: word-bank × carrier halves.
  - Two-hop: cue halves.
- Cross-fit (A→B and B→A) so every item is labelled.
- Measure reliability as agreement between labels computed from two disjoint fitting subsets for the *same* items.

### 5.7 The decodability probe can memorise items, and the "absent" positive control is weak

**The problem:**
- With folds split by template, each item sits in training under its other templates, so the probe can decode z through item identity. Per-row label shuffles do not catch this.
- The planted-copy control injects $\alpha\hat u_c$, essentially the direction a mean-difference-like probe learns, so detection is close to guaranteed.

**What I'd change:**
- Group folds by item and source problem, as A3 already does (SPEC L279).
- Use a regularised linear probe and a Hewitt–Liang control task (one random z per item, consistent across that item's templates), and define decodability via selectivity.
- Use causal-precedence cells as a label-free absent class: brew's stir cells, and any position before the token that sets z.
- Keep probe-negative "absent" labels only where the planted direction was not used to fit the probe.

### 5.8 The necessity tier is ambiguous and weak

**The problem:**
- "Pooled mean" (L178) could mean the mean over {x, x′} or over the family.
- A 1-D ablation leaves z decodable from the rest of the cell.

**What I'd change:**
- Define it as the mean of û·h over the family's fitting half at that cell.
- Add a random-direction ablation control (Elazar's Rand).
- Optionally add a stronger tier that removes all linearly decodable z information at the cell (LEACE/INLP-style).

### 5.9 Label edits have no damage or specificity gate

**The problem:** Heimersheim & Nanda (p.9): damage pushes the output toward uniform and can raise the margin toward a′. Norm-matched random directions control for generic perturbations of the same size, not for damage specific to the edited direction.

**What I'd change:** record the change in log P(a) and log P(a′) separately, plus KL or top-1 retention on non-answer tokens after the edit, mirroring RHUT's damage gate.

### 5.10 Factual corrections to the spec text

1. L127: install-vs-removal mix-up (§4.1.4).
2. L109: brew's "token-aligned donors" do not exist yet (§4.2.1).
3. L112: role-bound "ab/ba pairs" are not token-aligned (§4.2.9).
4. L89–91: "with its reconstructor if released" is answered (released); the Skip-lens variant and the SAE/BSF location are unresolved.
5. L154: "Skip-lens is fed L42" is a model trained at L62 being read off-distribution.

### 5.11 Closed-set scoring needs confound foils and a hit rule for token readers

**The problem:**
- "Gold z plus 4 same-category distractors" (L209) makes an answer-reader or an input-reader look like a plain miss.
- WSB's brew candidates are gold, start, answer and two off-trajectory colours.
- There is no binary-hit rule for token readers, which cannot abstain, while judged text readers can.

**What I'd change:**
- Include the input (start or cue) and the answer as foils, and report the confusion: z, input, answer or other. This separates answer geometry from absent content.
- Define the token-reader hit, for example: z ranked first among the candidates *and* its score above that reader's no-activation run.

### 5.12 The A3 ensemble lacks position effects

**The problem:** the model at L277 has family fixed effects only, and reader scores vary by position for reasons unrelated to use.

**What I'd change:** use family × position fixed effects, or stratified AUC.

### 5.13 Statistical power has not been checked

**The problem:** about 100 items per family, labels on half B only, a dev/test split for scoring choices, then strata by family × position × label. Several strata will have fewer than 20 cells, and P2's "CIs clear" may not be reachable.

**What I'd change:** add a power calculation to design 001, cross-fit, and pre-register a minimum number of cells per stratum below which W is not reported.

### 5.14 Chain z is multi-token

**The problem:** if this tokenizer splits numbers into digits, token readers cannot score z = 11 as a single token. P7 would then partly restate the vocabulary limitation (WP §9.1) rather than test the workspace claim.

**What I'd change:** prefer chains whose intermediates are single-token number words or single digits, or score digit sequences explicitly.

### 5.15 The SAE's independence as a reader family

**The problem:** the fallback in open question 2 (J-projected decoder labels) would make SAE claims J-derived. The SAE would then no longer count as an independent family in A1's k-families curve.

**What I'd change:** score the SAE with non-J auto-interpretation labels only, and keep the J projection as a diagnostic.

### 5.16 A5's outcome table and inputs

- Add a "Δ(c) > 0 for the hallucinated claim" branch (co-adapted codes; §4.5.6).
- Use the step-matched NLA pair (`nla-L42`: `av_rl_lora_step400` + `rl_critic_step400`) for any reconstructor analysis.
- Note that the Oracle's trained, whitened FVE cannot be reproduced from public files.

### 5.17 Brew is not a positive control for the label pipeline

**The problem:** brew was chosen because J reads it well under WSB's any-cell rule. WSB itself says brew cannot separate the intermediate from the runner-up answer, and prompt-only scores 0.96 there.

**What I'd change:** give the label pipeline its own controls.
- Positive: a cell where use is known by construction, e.g. the start-colour token at a layer before its content has been moved on, where the full-cell share should approach 1.
- Negative: the stir cells.

## 6. Reuse plan for the first label check (brew at L42 + one maintain-vs-ignore pair)

The check loads no reader. Under the non-negotiables:
- No lens is loaded and no reader is used for admission or labels.
- Every direction-level write is in fp32, with ρ and κ recorded after the write.
- design 001 comes first; nothing is launched without your go.

**Environment and pins**
- Reuse:
  - RHUT `registry.py` pins: model `Qwen/Qwen3.6-27B@6a9e13bd`, 64 blocks, d_model 5120, sha-verified downloads following `rebuild_env.sh`.
  - torch 2.11.0+cu128, already installed and matching RHUT.
- Write: install transformers, safetensors, accelerate, scipy and sklearn; download the model (~54 GB).
- Riskiest assumption: RHUT used transformers 5.15.0 and SEC used 5.17.0. I am assuming SEC's engine runs unchanged on the RHUT pin, and that RHUT's gate numbers reproduce on this box.

**Forward and edit engine**
- Reuse SEC `engine.py`: `Engine.capture` (bf16 capture of block outputs plus each block's kwargs), `build_fp32_upper`, `run_from` and `logits_from`.
- Write extensions to:
  - capture L42 for each prompt-plus-answer-spelling sequence;
  - apply edits as fp32 tensor operations at (42, c);
  - replay blocks 43–63 to the logits;
  - record requested vs realised deltas, i.e. ρ and κ;
  - run a self-patch no-op gate.
- Riskiest assumption: that the fp32 replay from L42, through GatedDeltaNet layers, to the final logits reproduces the bf16 model's answers. SEC's 000 run only validated replays from L40/48/56 to block 62, on pile text.
  - This needs its own gate: replayed clean margins vs bf16 margins, and competence in the replay regime.
  - Memory: the bf16 model has to leave the GPU before the fp32 upper stack (~21 blocks, ~35 GB) is built. Capture and replay run as two phases.
- Fallback: RHUT `hooks.StateMove` (L56–90) inside `hooks_fp32.Fp32Residual` (L8–31). It is hook-based, but its matmuls run in bf16.

**Brew rendering and positions**
- Reuse: the WSB bank (`evals/brew_intermediates/items.json`), the WSB `produce/render.py` chat render with the `Answer:` prefill, and the `eval_positions` / `regions` fields.
- Write: an assertion that rendering with the pinned tokenizer reproduces each item's indices and region tokens.
- Riskiest assumption: that the source repo's indices match our render. WSB's backend did not pin a revision.

**Brew partners**
- Reuse: nothing.
- Write: a constructor and gate.
  - Candidates: other starts in the same table, single-token in context (" X.").
  - Constraints: z′ ≠ z, a′ ≠ a, z′ ∉ {a, a′}, z ∉ {a′}, equal token count.
  - Gate both members: greedy plus 10 samples at T = 0.8, the WSB rule.
- Riskiest assumption: that enough items survive (35% of WSB's candidates passed 10/10) without keeping only the easiest tables.

**Margin scorer**
- Reuse: the convention only — sequence endpoint, every answer token, log-sum-exp over spellings (ladder §3 L39). The implementation lives inside RHUT's H3 scripts, which I did not read.
- Write: teacher-forced log-probs for each accepted spelling of a and a′ after `Answer:`, then m and G.
- Riskiest assumption: that the spelling set is complete (e.g. " black", " Black"), and that the answer's first token after `Answer:` carries a leading space under this tokenizer.

**Directions, swaps, nulls, necessity, full-cell patch**
- Reuse: the logic of RHUT `hooks.realized_write_stats` (L140–147) for ρ and κ, and the `SpanWriter` semantics (L32–53) for full-cell replacement.
- Write a label module:
  - per-contrast centroids on the fitting half;
  - swap magnitudes from the donor's projection;
  - a pre-registered number of norm-matched random directions per cell;
  - the full-cell patch;
  - the necessity ablation;
  - state assignment with pre-registered thresholds (§5.1).
- Riskiest assumption: that brew has enough items per (z, z′) contrast in the fitting half. There are ~8 items per z overall, and far fewer per ordered contrast.

**Decodability probe**
- Reuse: nothing. RHUT's RESID_P is a presence axis, not a cross-fitted probe.
- Write: a linear, regularised probe, group-split by item, with a Hewitt–Liang control task.
- Riskiest assumption: that z is decodable at single cells with ~100 items over 12 classes. Binary z-vs-z′ probes may be needed instead of a 12-way probe.

**Statistics**
- Reuse RHUT `stats.py`:
  - `cluster_t` (L6)
  - `equivalence` (L18)
  - `ratio_of_means` (L26)
  - `holm` (L44)
- Write: cluster bootstrap for shares, and FDR across cells.
- Riskiest assumption: that the item (table) is the right clustering unit.

**Gates**
- Reuse the RHUT `instrument_gates.py` pattern:
  - rendering gates;
  - repeat determinism;
  - empty-patch and same-source bitwise checks;
  - sequence-length nondeterminism (L141–146);
  - realised-write check (L149–152).
- Write: port these to the engine, and add the damage gate (§5.9).
- Riskiest assumption: that the engine is deterministic across batch composition. It broadcasts the captured kwargs from row 0, which requires identical lengths (`engine.py` L109–114).

**Maintain-vs-ignore pair**
- Reuse RHUT:
  - `rendering.user_text_controlled` (L54–64);
  - `render` (L127–181), with `extra_turns` for a consumer;
  - `render_conversation` (L90–112);
  - `materials` carriers, word banks and decoys (L8–39).
- Write:
  - a consumer question;
  - a competence gate under maintain, mention and ignore;
  - partner words geometry-matched within each condition.
- Riskiest assumption: that a neutral question exists which the model answers correctly ≥ 0.9 under ignore and which does not restate the word (RHUT two-hop Amd 2's lesson).

**Cost driver.**
- Replays ≈ items × cells × (1 swap + 1 full patch + random directions + 2 necessity) × spellings × 2 answers.
- With 24 brew cells and 20 random directions per cell, that is on the order of 10⁵ replays for 50 items.
- design 001 should therefore:
  - cut the cell set (e.g. 1 stir, 2 start, 2 question, 3 emission cells);
  - cut the number of random directions;
  - run sequentially, starting with the full-cell profile of §5.2.

## 7. Open questions I need answered before drafting designs/001

1. **Cell definition:** one token at one layer, or a span and/or a layer band?
2. **Thresholds:** τ for "used", ε for "not needed", the precedence rule, and FDR across cells (§5.1).
3. **Template halves:** what are A/B for each family? Is cross-fitting acceptable (§5.6)?
4. **The slice's contrast:** which consumer question for maintain vs ignore, or should the slice's not-needed condition move to tagged selection at the question/decision cells (§5.4)?
5. **Direction content:** require decorrelated fitting for "z used" (§5.3), or accept "pair difference used" labels for the slice and score readers on input, z and answer?
6. **Precision regime:** SEC's true-fp32 replay engine or RHUT's fp32-residual regime? transformers 5.15 or 5.17?
7. **Reader choices:** these are not needed for 001 but block the reader stage.
   - Oracle layer policy (§5.5).
   - NLA checkpoint: WSB's `nla-rl` iter_000400, or the step-matched `nla-L42` pair?
   - Skip-lens variant.
   - SAE/BSF location and its auto-interpretation labels.
8. **Judge access:** OpenRouter (Gemini 3.8 Flash) for the WSB reproduction, and which judge for our text readers?
9. **Prompt-only arm:** stock Qwen3.6-27B, as in WSB?
10. **Brew donors:** construct and gate them, dropping items that fail?
11. **Storage:** where should code and outputs live (a new repo, or a branch of SEC)? Where should outputs be mirrored, given `/workspace` is not persistent?
12. **Leverage profile first:** should 001 open with the reader-free leverage profile (§5.2) before fitting directions and probes?

## 8. What I'm least sure I understood

- **Template halves:** what "template halves" are meant to be for brew and for the RHUT organisms.
- **Cells:** whether a "cell" is a token or a span, and whether edits are at one layer.
- **Necessity and the planted control:** what "pooled mean" refers to, the pass criterion for the "used + necessary" tier, and what "behaviour-relevant dose" means for the planted control.
- **Gate 0:** the optional gate 0 in §12 ("rotated-row null on kurtosis, occupancy and install share"). I have not read the document it comes from, probably SEC `docs/00_user_spec…`.
- **Undefined terms:** what "BSF" stands for, and whether "Skip-lens fed raw L42" names a specific variant meant as an off-distribution arm.
- **Same activation, different label:** how the pair is to be computed at cell level, given RHUT's transplant was whole-span at all 64 blocks.
- **Summariser:** whether the summariser's prompt-only arm means the summariser sees the prompt in place of the token bag.
- **"Intermediate positions" in the reading list's step 1:** for brew, I take them to be the start and question cells, after the start colour has been read and before emission.
