# Decision record: DAS not run (2026-10-04)

*Requested by the PI's feedback of 2026-10-04 (§5, "Not now": "DAS: write a one-paragraph decision record: not run, and why").*

Distributed alignment search (DAS) is **not run**, and nothing currently planned depends on it.
- **The PI's rule.** "Do not reopen the content-label question (DAS etc.) unless I decide to", and the PI has not decided to.
- **No decision rests on it.** 004 found that at ":" L42 a fitted z span recovers 0.21 of the whole-cell effect and the joint span 0.72, with same-rank PCA doing as well or better. No current decision rests on whether a better-than-PCA content subspace exists:
  - 005 needs no content label: its ground truth is planted;
  - the reading-vs-re-solving line uses reader-free erasure (LEACE on Brew, per-bridge concept directions on multihop; PI note 3.5);
  - the chain re-score (C1) needs none.
- **Known risk of illusion.** DAS has a documented risk: a trained subspace can patch an effect through directions the model does not use for that variable (Makelov, Lange & Nanda 2023, arXiv 2311.17030, in `docs/papers/`). So a DAS result would itself need the PCA, random-subspace and held-out controls that 004 already supplies.

**What would reopen it.** A decision that turns on whether the z span at ":" is content-specific beyond PCA. In that case the bounded DAS-vs-PCA check of report 004 §14 is the form to propose, as its own design spec, and only on the PI's go.
