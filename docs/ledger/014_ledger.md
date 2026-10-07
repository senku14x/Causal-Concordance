# 014 ledger: the reading-side suite on WorkspaceBench typo (EXPLORATORY)

Branch `014-typo` (off `013-nla-hallucination`). Spec `docs/design_specs/014_typo.md`. Code `scripts/014_typo/typo.py`
(Parts A/B) and the 013 pipeline under `NLAH_FAMILY=typo` (Part C). Outputs `outputs/014_typo/`, results
`results/014_typo/`. Append only.

## 2026-10-07

**Why.** PI: "lets run on typo at least what we can and then we can try on the other sets". Typo has no answer that
depends on the corrected word, so only the reading parts of the suite run (spec §1).

**Code.**
- `typo.py` reuses 007's J basis (`J_raw_L{L}`, first 1,024 columns), 007's R0 seed, `c012._hit` for naming, and the
  012 reader calls (Oracle and NLA, 2 samples, seed `p007.SEED`).
- `hallucination.py` / `rewrite.py` gained `NLAH_FAMILY=typo` (material, gold vectors, paths) and a `gate` phase (013's
  de-risk R1 on typo). The multihop path is unchanged (still 522 explanations).

**Process slip.** The spec was written before the smoke but committed (0cb4ece) a few minutes after the smoke launched.
Nothing was read from the smoke before the commit.

**Smoke, 10 items (first 10 of the bank).** Subject phase gates: render 10/10 (the read token is the end of the
misspelled word); piece reconstruction max 1.9e-17 relative. Wall 138 s for captures, lens ranks and Oracle reads.

**Smoke, Part A/B results** (`results/014_typo/analysis/parts_smoke.json`; 10 items × 2 samples per cell; naming = whole-word
target in the readout).
- **Naming is at ceiling in every piece.**
  - Whole, J, MR and echo: 1.00 for all three readers.
  - Rest-of-J: NLA 0.95, Oracle L40 0.85, L44 0.80.
  - Random 1,024 (R0): NLA 1.00, Oracle L40 0.85, L44 0.80.
- **Is it incidental?** Typo targets are common words (account, across, again, …), so a readout could name them by chance.
  - POST HOC measure added after the smoke, before the full run: the *other-target null*, i.e. the share of other items' targets a readout names.
  - It is 0.00–0.06 in every cell. So the naming is item-specific, not incidental.
- **Reading:** on typo the corrected word is readable from every activation part, including a random fifth of the
  dimensions. That is unlike 007's multihop bridge (NLA rest-of-J 0.23, test half). P2's "readable outside J" holds, but
  random does as well, so the J-vs-rest contrast does not discriminate here.
- **κ (P3):** undefined. All 10 items are named by every reader (no variance).
- **Token lenses** (rank of the target, median over 10): J_CB 3.5–8; J_NP 43–113; logit 54.5–96. The echo control (the word
  literally present) is close to the misspelled version for J_CB. Descriptive only.
- **Qualitative** (Oracle L40, "Febuary"): whole / J / MR name February; rest-of-J drifts ("July", "August"); echo restates
  the corrected sentence. Readouts: `outputs/014_typo/smoke/readouts_{oracle,nla}.jsonl`.

**Smoke, Part C** (NLAH_FAMILY=typo NLAH_TYPO_TAG=smoke).
- **Gate R1** (`results/014_typo/analysis/nlah_gate_smoke.json`, first sample of 10 items): FVE 52.8% vs shuffled
  −110.3%. **Passes** (n 10, indicative only). The reconstructor reads typo explanations about as well as multihop (49.9%).
- **Judges on 5 explanations** ($0.010): G1 5/5, B parse 5/5, K 5/5. Kinds: misquote 7, invented context 9, wrong entity 1.
- **Hand check (G2, before scoring):** contradicted spans 16/17 correct by the definitions (action C3 is a hedged prediction).
  - **Kind problem:** the 6 `address` spans are tagged invented_context, but they quote the prompt minus one word ("your"). By the definitions they are trivial misquotes.
  - Typo explanations quote the prompt a lot, so the kind split needs a closer hand check on the full run before any per-kind reading.
- **Scoring** runs end to end (`results/014_typo/analysis/hallucination_smoke.json`). G-repro is not applicable to typo (no
  multihop reference; the R1 gate stands in), so the JSON's `G_repro_pass: false` is expected.
  - n = 5: the Δs are pipeline checks, not results (primary +49.4, CI [−6.3, 119.8]).
  - Flagged-token share 0.63, as on multihop.

**STOP after the smoke** (the PI's rule). Full run on the PI's go:
- Parts A/B on 100 items: about 15 GPU-min.
- The Part C gate on 100, then judges on 200 explanations (about $0.4) and scoring (about 10 GPU-min).
