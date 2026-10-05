# Reading notes for SPEC_v2 (2026-10-04)

Per source:
- what it claims (with section or line references);
- what it means for SPEC_v2, by section;
- any conflict with our results 001–004 or the CW pilot.

**[obs]** = what the source says. **[inf]** = my inference.

Read status:

| Source | Local copy | How much was read |
|---|---|---|
| Anthropic NLA paper | `docs/sources/web/nla.txt` | in full except the case-study transcripts and the auditing appendix (Intro, Method, Rewards case, Evaluations, Confabulations, Discussion/Limitations, appendices on α, AR prompt, log reward, layer sensitivity, open-model NLAs) |
| Viswanath post + 4 comments | `/workspace/sources/web/viswanath_blind_outside_jspace.md` (fetched 2026-10-04 via the LessWrong GraphQL API; not committed) | in full |
| Karvonen et al., *Activation Oracles* (2512.15674) | `docs/papers/text/2512.15674.txt` | abstract, §1–§4 (methods, auditing evaluations, PersonaQA, classification), related work, discussion/limitations; appendices not read |
| WorkspaceBench READMEs: brew_intermediates, multihop, chain_intermediates | `/workspace/sources/repos/workspace-bench` @ 92d763e | in full, plus `src/wsbench/produce/methods.py` (NLA class) and its git history |
| Model cards, configs and file listings for `ceselder/qwen3.6-27b-nla-rl`, `ceselder/qwen3.6-27b-nla-L42`, `agu18dec/olens_and_ar`, `agu18dec/local-workspace` | Hugging Face API, 2026-10-04 (metadata only; no weights) | — |

---

## 1. Anthropic NLA paper (Fraser-Taliente et al. 2026)

**Claims [obs]**

- **Architecture** (§Method, Architecture):
  - The AV is a copy of the target. The scaled activation is *inserted in place of* a special token's embedding, and the AV samples an explanation at T = 1.
  - The AR is the target truncated to its first l layers, plus a learned affine map at the final token.
  - Training is joint: an AR regression step plus an AV GRPO step. The reward is −‖h − AR(z)‖², later −log of that, with a KL penalty toward the initialisation.
  - Activations are normalised to unit L2 norm.
  - FVE = 1 − L / E‖h − h̄‖². The paper's NLAs reach 0.6–0.8.
- **Warm start** (§Initializing): SFT on (activation, Claude summary of the text *up to that token*). The AV therefore starts as a context summariser.
- **Limitations stated by the authors:**
  - confabulation;
  - no mechanistic grounding;
  - **"Excessive expressivity: Because the AV is a full language model, it has the capacity to make additional inferences beyond what is stored in an activation"** (Introduction; Discussion: "the AV could in principle perform inference beyond what the activation encodes");
  - partial input inversion ("explanations sometimes quote sentences from the context verbatim");
  - single-layer reads.
- **Evaluations:**
  - Five prediction tasks improve with training. The authors note: "an LLM provided the full input as context would likely attain a high score on all of these tasks."
  - Steganography is tested by paraphrase/shuffle transforms, which cost little FVE.
  - Writing quality degrades during training.
  - Verifiable false claims stay at a roughly flat rate through training.
- **Confabulation** (§Characterizing):
  - Thematic claims are more accurate than specific ones.
  - Removing true claims hurts reconstruction more than removing false ones, but "the AR is only a weak per-claim verifier".
  - **True claims recur across adjacent token positions**, and the authors use recurrence as their reliability heuristic.
- **AR-edit steering** (§Reasoning about Rewards; Opus 4.6 toy setting):
  - Edit "rewarded" → "penalized" in explanations at one token.
  - Reconstruct both versions; average the difference over 5 rollouts; add it at that token.
  - This raises odd answers beyond a random vector.
- **Layer sensitivity** (Appendix): applied away from its training layer, an NLA is out of distribution. It "can surface more striking content, but is also generally less coherent", and it surfaced RM-sycophancy terms at a higher rate off-layer than the NLA trained at that depth did.
- **Scaling factor** (Appendix): α ≈ the 75th-percentile activation norm. The AV is "meaningfully sensitive to α".
  - [obs, model cards] The ceselder NLAs instead use a Karvonen norm-matched add at the output of block 1. Our `readers.NLAVerbalizer` and WSB's NLA class both do this.

**Meaning for SPEC_v2**

- **§3 Q11, §8 A4/A7.**
  - Re-solving is the paper's own named limitation ("excessive expressivity"), so the A7 battery tests a documented failure mode, not a hypothetical one.
  - The warm start predicts prompt paraphrase and context summary, consistent with A9 (CW: 22% prompt words, 22% format).
- **§7 scoring.**
  - Add position recurrence (does the claim recur at the adjacent positions?) as a reliability covariate for text-reader claims.
  - This is the paper's validated heuristic. SPEC §7 currently ranks claims by recurrence across *samples* at one cell, which is a different quantity.
- **§2 / §5 Layers (off-layer reads).** The paper reports *more* striking content off-layer. CW-2's prose-reader hits at L16/28 (26–27/50) could partly be this out-of-distribution effect rather than re-solving. Keep them exploratory, as v2 already says, and add this as an explicit alternative explanation in A7.1.
- **§8 A5 / P5.**
  - "Weak per-claim verifier" means Δ(c) is weakly > 0, not ≈ 0. This supports the A01 C21 branch.
  - Best-of-N by FVE is an inference method the paper names but did not evaluate (§Future Work).
- **§8 A1 level C.** AR-edit steering vectors are the paper's validated write map for the NLA. Level C/D for the NLA is feasible with `rl_critic_step400`, but only for the registry NLA (see §5 below).

**Conflicts with our results.** None direct. 001 found FVE dominated by the shared template, with a small source preference (244/360). That is consistent with "thematic > specific" and a weak verifier.

---

## 2. Viswanath, "Models are blind outside the J-space. NLAs aren't." (LessWrong, 2026-07-08)

**Weight (PI, 2026-10-04): unvetted LessWrong post; motivation and design lessons only, no evidential weight.**
- Different model (Llama-3.3-70B); n = 6 → 16 concepts with a flipped result; the boundary r was chosen after seeing outcomes; no held-out set.
- Nothing in SPEC_v2 should rest on its numbers.
- The lessons kept below stand on their own logic.

**Claims [obs]** (Llama-3.3-70B, the released Anthropic NLA at L53, a J-lens trained at L53)

- **Concept vectors:** "Tell me about {word}" last-token activation minus the mean of 100 baseline words (Macar et al.); 24 concepts.
- **De-risking:**
  - The NLA names the injected concept for 24/24.
  - With a neutral carrier, the NLA names a strength-1.0 injection 46% with injection vs 0% without.
- **J/non-J split:** the J part is the projection onto the **top-r Jacobian directions**, not the paper's sparse nonnegative token pursuit. Both parts are norm-matched to the original.
- **Results depend on r:**

  | r | J share of energy | J-part / non-J-part named by model | NLA reads non-J / J |
  |---|---|---|---|
  | 200 | 6% (with r = 1076 → 50%, r = 4450 → 90%) | 6 concepts: P(yes) 0.025 vs 0.002 | 0.83 / 0.44 |
  | 200, n = 16 | — | **flipped**: model tracked the non-J part (named about half the time); J part named 0 | — |
  | 2000 (chosen after a sweep as "cleanest") | — | 80% / 0% | 100% / 100% |

- **P(yes) is unreliable:** random directions inside J gave P(yes) 0.077–0.095, more than the real J parts (0.038). The author switched to "named out loud".
- **Franken-vectors** (A's non-J part + B's J part; 6 pairs × strength 4 = 12 trials): the model named a concept 3 times, all the J-side concept; the NLA named both every time.
- **Non-J part on neutral text:** raises the concept token's log-prob by +13.7 to +21.1 nats over 32 trials, yet the model never names it.
- **Line counting:** the NLA never reads the exact count that the J-lens shows.

**Meaning for SPEC_v2**

- **§7 / §8 A8 (design 005), J-space definition.**
  - "J-space" is construction-dependent: the post's own J-energy share changes with the rank r (6% at r = 200, 50% at r = 1076, 90% at r = 4450, as reported there).
  - 005 must pre-register a primary construction (the workspace paper's §2.3 pursuit; see the report, (c)), report a sensitivity sweep, and never choose the boundary on outcomes. The post picks r = 2000 as "the cleanest" after sweeping r.
- **005, model-side measure.**
  - Use the concept named in an open report, not P(yes).
  - Separate **report** from **output effect**: the non-J part moved logits by +13–21 nats with no report.
  - So "the model ignores the non-J part" (P11 as written: "next-token distribution does not change") is likely false as stated. The dissociation he found is "changes output, not report".
  - Split P11 into a report version and an output version.
- **005, reader-access asymmetry.**
  - The NLA read the layer where the vector was injected (direct access); the model had to carry it forward and verbalise it.
  - Add a downstream-read condition (inject at ℓ, read at ℓ′ > ℓ after the model's own propagation) and a reader-free detector, i.e. a linear probe trained on natural concept activations. If a linear probe also detects the non-J part, the NLA detecting it is a property of the direction, not of the NLA.
- **Power.** n = 6 flipped at n = 16. 005 needs a pre-registered concept set (≥ 24, split dev/test) and per-concept reporting.

**Conflicts with our results** [inf]

- None direct (different model).
- The line-counting soft negative agrees with the paper's §3.5 (counts are J-readable when asked for) and with CW-7 (chain at L56: J-lens finds numbers, the NLA is near chance).

---

## 3. Karvonen et al. 2025, *Activation Oracles* (arXiv 2512.15674)

**Claims [obs]**

- **What AOs are:** LatentQA-trained models answering arbitrary questions about activations.
- **Input format** (§3.1): "Layer ℓ:" + K placeholder tokens + a question.
- **Injection:** after the second transformer layer, h′ = h + ‖h‖·v/‖v‖ (norm-matched add, Eq. 1). Replacement caused activation-norm growth (A.5).
- **Training mix** (§3.2):
  - system-prompt QA;
  - 7 binary classification tasks;
  - **self-supervised context prediction**: predict the previous or next tokens from activations.
- **Evaluations:**
  - Secret-keeping (Taboo, Gender, SSC): AOs beat white-box baselines (e.g. Taboo 21% vs 4%), using the **full sequence** of activations. Single-token inputs help on Taboo; shorter selections help on Gender (App. C.4).
  - Emergent-misalignment diffing: the AO reads base-vs-fine-tuned activation differences.
  - PersonaQA, which addresses Li et al.'s text-inversion concern: modest. Open models go from ~5% to ~14%; Haiku 3.5 reaches 35%. On binary variants AOs beat Patchscopes.
- **Limitations** (§Discussion):
  - "frequently makes incorrect guesses", "not trained to express uncertainty";
  - simple prompting baselines are strong on hallucination detection;
  - narrow fine-tunes.

**Meaning for SPEC_v2**

- **Our Oracle Lens is not a Karvonen AO.**
  - `agu18dec` `olens_s3d_rl600` is a workspace-paper-style oracle lens: phrase bullets, RL on bullet FVE against a frozen AR.
  - Its marker *embedding row is replaced* by 16000·unit(h) (WSB `OLens`; `readers.py`).
  - The AO paper bears on SPEC_v2 only through the shared verbaliser class and the text-inversion concern. References should not cite it as the Oracle Lens's method.
- **§8 A4/A7.**
  - Context-prediction training explicitly teaches input recovery, which is the mechanism behind re-solving.
  - The NLA paper (§1 above) also notes that a question-cued oracle "could guess a plausible answer to the query". A7 should therefore use open prompts, not questions that name the candidate.
- **§7 fairness.**
  - AOs and AObench (Bauer et al.; `reading_notes.md`) find single-token reads handicap verbalizers.
  - SPEC's matched single-cell design is right for labels but is a known handicap for the text readers.
  - State this next to every reader comparison, and do not read "verbalizer misses X at one cell" as "X is absent".
- **§4 stochastic readers.** AOs are uncalibrated. AObench consensus sampling (k = 10, keep claims with ≥ 0.8 agreement) is the precedent for SPEC's self-consistency variable.

**Conflicts with our results.** None. 002's Oracle self-contradiction rate (85% "conflicting" at L40) is in line with "frequently makes incorrect guesses".

---

## 4. WorkspaceBench READMEs (brew, multihop, chain) @ 92d763e

**Claims [obs]**

- **brew_intermediates**
  - **Bank:** 100 items; 24 pinned cells per item; chat render with prefill "Answer:".
  - **Regions:**
    - stir (6 cells; "nothing can be computed": the negative control);
    - start (6);
    - question (9);
    - emit_asst / emit_think / emit_colon (headline cells).
  - **Default judged cells:** emission + stir, × 11 layers (20–60 step 4).
  - **Judge:** one prompt-blind multi-select call per cell over `options_adjacent` (gold, start, answer, two off colours). The judge marks *every* candidate named; this is mention-based.
  - **Headline:** an item passes iff K·G > ΣO over emission cells × all layers.
  - **Floors:** `null` = the same rule with off colours in the gold's role; `baseline` = the fraction of K+1 colours that clear the rule.
  - **Prompt-only is not a floor** ("the stock model computes the intermediate (measured 0.96)").
  - **Caveat:** "the intermediate is usually the model's runner-up answer colour, so this bank cannot separate a held step from a runner-up output."
  - **Judge prompt version:** brew-2026-09-16.
- **multihop**
  - **Bank:** 100 items, plain render.
  - **Read position:** the **last word** of the prompt, i.e. its last non-whitespace token. It moved there on 2026-09-23 from the final token, which was a bare space for 33/100 prompts.
  - **Judge:** one call per (item, layer), strict "named" rule with a verbatim quote. An item passes if any layer is expressed.
  - **Floor:** the prompt-only baseline (stock Qwen3.6-27B, "asked what a model would be thinking there").
- **chain_intermediates**
  - **Bank:** 120 items (36 depth-3, 84 depth-2).
  - **Read position:** the **last token of the render** (the assistant-onset `\n\n` after the empty think block).
  - **Judge:** free recall. It names ≤ 3 numbers the readout presents *as computed*, and the top value must be an intermediate, credited only as a whole number.
  - **Null:** magnitude-matched decoys (±3).
  - Prompt-only is "not item-blind".
  - `cells=all` is an any-of-grid rule that "inflates the pass rate on its own".
- **NLA arm** (`src/wsbench/produce/methods.py:261–270`): `repo = ceselder/qwen3.6-27b-nla-rl`, `adapter = av_rl_adapters/iter_000400`, loaded on `av_base`, Karvonen add at block 1. This default has been unchanged since the first commit of the producer (494a980, 2026-09-23). Eval READMEs record the arm as "nla-rl-L42".

**Meaning for SPEC_v2**

- **§12 step 1 (WSB reproduction).**
  - Must use WSB's rule exactly: mention-based, any-cell K·G > ΣO, judge prompt brew-2026-09-16, J_NP cosine top-10, layers 20–60 step 4.
  - The 0.86 is not comparable with any per-cell role metric in 002.
  - Report the `null` and `baseline` floors that WSB itself reports.
- **§4.R NLA.** The question "which checkpoint did WSB's published NLA numbers use" is answered from the code: `nla-rl` `iter_000400`, the pilot's checkpoint, not the registry's.
- **§5 multihop.** Joint multihop numbers must read the last *word* (WSB, post-2026-09-23). Check that any multihop render in our code uses the same position as CW's runs; she says she used the bank's read token.
- **§7 chain scoring.**
  - CW-6/7 counted *any* number in the readout. Oracle readouts carry 6–9 distinct numbers, which is why chance is 12/60.
  - WSB's top-computed-value rule with a decoy null is stricter. Joint chain numbers should use it, or the shared §7 rubric, not any-number mention.
- **§5 brew.** WSB's runner-up caveat is already carried in SPEC_v2 ✓.

**Conflicts with our results**

- WSB's brew prompt-only is 0.96 ("what would a model be thinking"). 002's prompt-only is 1.00, but 002 asked the stock model *for each colour* directly; it is a task baseline (002 §6).
- These are two different baselines. SPEC_v2 §1 point 2 and §4 treat them as one; see the report, (b).

---

## 5. Checkpoint metadata (Hugging Face API, 2026-10-04)

All rows below are [obs].

**Oracle Lens.** `agu18dec/local-workspace` (dataset; readable with our token on 2026-10-04) `ckpts/ao/rl/s3d.ddp600.s0/iter_000600/adapter_model.safetensors`:
- LFS sha256 7d4b301e002246af… = `agu18dec/olens_and_ar` `olens_s3d_rl600/adapter_model.safetensors` (467.0 MB);
- `adapter_config.json` and `run.json` are byte-identical.
- So the pilot's and the registry's Oracle are the same weights. STATUS.md's "assumed from the names and cannot be verified" is now verified.
- `local-workspace` also holds a `reference/` adapter (sha ecbafd1e…, a different file); it is not used.

**The two NLAs are different training runs**, not two serialisations of one model:

| | registry `nla-L42` (@54e3ce60, 2026-07-08) | pilot / WSB `nla-rl` (@5a13b7ec, 2026-07-21) |
|---|---|---|
| Verbaliser | Qwen3.6-27B + `av_sft_lora` (r128, attention/GDN projections only) merged + `av_rl_lora_step400` (r128, rsLoRA, q/k/v/o, in_proj_qkv/z, out_proj; **no MLP**) | `av_base` (full 53.8 GB model: Qwen3.6-27B + warm-start LoRA r128 *all-module*, merged) + `av_rl_adapters/iter_000400` (r64, rsLoRA, **all-module incl. gate/up/down_proj**) |
| Adapter file | 1,266.7 MB, sha d565bd0d… | 1,867.8 MB, sha b54e23e4… |
| Reconstructor | `rl_critic_step400` (step-matched; 43 blocks + value head) | `ar_reconstructor`: end of run (`saved_at_step.txt` = 600); "Latest only" per the card. **No step-400 AR exists.** |
| Held-out FVE at step 400 (card) | 74.5% | 76.6% (measured with the co-trained step-400 AR, not released) |
| Warm-start data | Claude Sonnet 4.6 summaries of Qwen3-8B L24 FineFineWeb texts, activations re-extracted at L42 | same lineage (`rl_Qwen3-8B_L24…` parent dataset) |
| Injection and prompt | Karvonen add at block 1, marker ㈜ (158983), identical actor template | identical |
| RL config | — | batch 504 prompts × group 8, KL 0.01, length penalty 0.01, train_critic true, 800 steps |

- Both use the same injection and prompt, so the difference is the weights: rank, target modules, base merge and RL run.
- The size difference (1.27 vs 1.87 GB) is the r64-all-module vs r128-attention-only LoRA layout.
