# 38. 014's J contrast survives a content-matched control (2026-10-07, EXPLORATORY, POST HOC)

**Answer.** Yes. On WorkspaceBench typo, the Oracle's loss of the corrected word when 007's J subspace is removed is
not explained by removing item-varying content. Removing the same amount of item-varying variance along the top
principal components barely hurts it, and removing a random 1,024-dim slice doesn't hurt it at all. The NLA's small
cost is unresolved. The paired Oracle − NLA contrast survives at both Oracle layers.

**Setup.**
- Script: `claude_ovarida/scripts/typo_content_control.py`. The rules were fixed in its header and committed (4102c15)
  before any read.
- Results: `claude_ovarida/results/typo_content_control.json`. Raw readouts: `outputs/claude_ovarida/014_content_control/`.
- **Material:** the exact activations 014 read (`outputs/014_typo/captures.pt`): 100 typo items, the last token of the
  misspelled word.
- **New pieces:**
  - **rest_PCA:** the whole activation minus its top-k principal components (k 23 / 20 / 17 at L40 / 42 / 44), chosen
    to match J's share of the centred variance (J 0.46 / 0.43 / 0.40 vs PCA-k 0.47 / 0.44 / 0.41);
  - **rest_R0, rest_R1:** the whole activation minus a random 1,024-dim subspace (007's seeds 0 and 1).
- **Readers and scoring:** exactly as in 014 (Oracle L40 / L44, NLA L42; 2 samples; seed `p007.SEED`; whole-word naming).
  The `whole` and `rest_J` readouts are 014's own.
- **Gates:**
  - rest_J rebuilt here equals 014's saved piece exactly (max rel. difference 0.0);
  - the NLA injection was healthy (write norm 0.99–1.00 of the residual, cos ≥ 0.99998).
- **Note:** rest_PCA removes *more* total energy than J (0.30–0.42 vs 0.23–0.27; `gates.json`). That makes it a strict
  control.
- **GPU:** about 8 min Oracle + 6.5 min NLA.
- **Bug, logged:** the first run freed the subject without `gc.collect()`, and the NLA refused to load (47 GB free). The
  Oracle reads had already been saved. The NLA reads were rerun in their own process (`nla` phase), unchanged otherwise.

**Results.** Naming share and the cost of each removal (whole − piece); item bootstrap, n = 100:

| Reader | whole | rest of J | rest of PCA-k | rest of random (R0 / R1) | cost J | cost PCA-k | **D_PCA = cost J − cost PCA** | verdict |
|---|---|---|---|---|---|---|---|---|
| Oracle L40 | 0.99 | 0.85 | 0.95 | 1.00 / 1.00 | 0.14 [0.08, 0.21] | 0.04 [0.00, 0.08] | **+0.11 [0.04, 0.18]** | J-specific beyond content |
| Oracle L44 | 0.99 | 0.77 | 0.97 | 0.98 / 0.99 | 0.23 [0.16, 0.30] | 0.02 [−0.01, 0.05] | **+0.21 [0.13, 0.28]** | J-specific beyond content |
| NLA L42 | 1.00 | 0.97 | 0.99 | 0.99 / 0.99 | 0.03 [0.00, 0.06] | 0.01 [−0.01, 0.02] | +0.02 [−0.01, 0.06] | unresolved |

(Shares and costs are rounded half-up from the JSON. Exact values: Oracle L40 D_PCA 0.105, L44 0.205; L44 cost J 0.225.)

**The paired contrast (the pre-set test):**
- Oracle L40 − NLA L42, on D_PCA: +0.085 [0.01, 0.165];
- Oracle L44 − NLA L42: +0.185 [0.115, 0.255].

**Verdict: "survives content matching".**

**What the Oracle says without J** (Oracle L44, first sample, seed-7 pick among the 23 items where rest-of-J misses the
word but rest-of-PCA names it; `typo_content_control_samples.json`):
- *present*: rest of J gives "…**established** at the new site. (Note: The original had a typo…" and "**journey**"; rest
  of PCA gives "e (present) under the tree. (…'Christmas present'…".
- *football*: rest of J gives "on Monday, Tuesday and Wednesday. (Note:…"; rest of PCA gives "…watches football".
- *training*: rest of J gives "…('prcess' is a typo for 'process')…"; rest of PCA gives "…completed their training…".

Without J, the Oracle keeps the *frame*: there is a typo, a correction is being made, and the sentence type. It loses
*which word* it is, and often substitutes another one.

**What this does and doesn't show.**
- On typo, at L40/L44, the Oracle's naming of the corrected word depends on 007's J subspace beyond its content share
  and beyond random subspaces. The NLA's L42 naming barely depends on it. This matches 007 on brew `:` (Oracle J-dependent,
  NLA not), so 014 now counts as **a second family** for report 36's R6. Confidence moves from low-medium to medium.
- **It doesn't show why.** The open hypothesis is still that the Oracle's reconstructor was trained with a J-lens term
  (report 36, (g)5). This result is what that hypothesis predicts, but it doesn't test it.
- **Caveats:**
  - post hoc;
  - near-ceiling naming;
  - 2 samples per cell;
  - different layers for the two readers (Oracle L40/44 vs NLA L42);
  - PCA is fitted on the same 100 activations;
  - reading only: the model's use of the word is not tested (typo has no answer that depends on it).
