# 37. Typo energy check: is 014's J contrast just energy? (2026-10-07, EXPLORATORY, CPU only)

This answers report 36's question 1, step 1. No model was run. The script is `claude_ovarida/scripts/typo_energy_check.py`;
the decision rule was written in its header before it ran. The result is `claude_ovarida/results/typo_energy_check.json`.

**Data.** `outputs/014_typo/captures.pt`: the exact activations 014's readers read (100 typo items, last token of the
misspelled word, L40/42/44, stock bf16).

**What it measures.** For each layer, the share of each activation's energy that lies in 007's dense J subspace
(`J_raw_L{L}`, first 1,024 columns), averaged over items. It compares this with 20 random 1,024-dim subspaces (007's
r1024 seeds 0–19). Two versions:
- **uncentred**, which is what the readers actually read (primary);
- **centred** on the 100-item mean, which gives the item-varying part (secondary).

**Result** (the mean share of energy in the subspace; a random 1,024 of 5,120 dims gives about 0.20):

| Layer | J, uncentred | random, uncentred (min–max) | J / random max | J, centred | random, centred | J / random max | PCA-k with J's centred share |
|---|---|---|---|---|---|---|---|
| L40 | 0.27 | 0.19–0.21 | 1.29 | 0.46 | 0.20 | 2.28 | 23 |
| L42 | 0.23 | 0.19–0.21 | 1.09 | 0.43 | 0.20 | 2.14 | 20 |
| L44 | 0.23 | 0.19–0.21 | 1.10 | 0.40 | 0.20 | 1.99 | 17 |

**Verdict by the pre-set rule:** step 2 is **not warranted**. On the activations the readers read, J's share is only
1.1–1.3× the largest random share, below the 1.5 bar. Removing J removes about 23–27% of the total energy, against
about 20% for a random slice.

**But the rule doesn't settle it** (post hoc reading, labelled):
- Most of the uncentred energy is the shared mean direction (one PCA component already holds more than J's share).
- On the **item-varying** part, which is what can carry *which* word it is, J holds about **2×** a random slice (0.40–0.46 vs
  0.20), as much as the top 17–23 principal components of the 100 activations.
- So 014's "removing J costs the Oracle 0.14 / 0.225" removes about twice the item-specific variance that a random
  1,024-dim removal would.
- The Oracle − NLA contrast is paired (both readers see the same pieces), so this doesn't explain why the *Oracle* drops
  more than the NLA. But it does mean "the Oracle depends on J" isn't yet separated from "the Oracle depends on the
  item-varying variance, which J happens to concentrate".

**What it changes.**
- The crude energy confound (J removes far more total energy) is **ruled out**.
- A content-matched control is still missing. That would be removing a matched PCA-k (k ≈ 17–23) or a Δ-matched random
  subspace, as 007 did. I didn't run it, because the pre-set rule doesn't trigger step 2. Cost if wanted: about 10–15 GPU-min of
  Oracle + NLA reads on the 100 items.
- For now, 014's R7 stays "post hoc, low-medium confidence": consistent with 007's brew result, not an independent
  replication of J specificity.
