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

**PI go for the full run (2026-10-07):** "lets just start it". `scripts/014_typo/run_full.sh` chains Parts A/B, then the
Part C gate, judges and scoring (deletion and rewrite arms). Part C analysis waits for the hand check (G2).
- The estimate was about 15 GPU-min for Parts A/B; it took about 25 (the Oracle reads alone took 16 min).
- The PI's message also asked for a new-session prompt aimed at coming up to speed on the whole project. Report 35 was
  rewritten that way.

**Full run, Parts A/B (100 items; `results/014_typo/analysis/parts_full.json`).** Gates: render 100/100; piece
reconstruction 2.0e-17.
- **Naming share (2 samples × 100; other-target null 0.01–0.02 everywhere):**

| reader | whole | J | rest of J | random 1,024 (R0) | mean removed | echo (correct spelling) |
|---|---|---|---|---|---|---|
| Oracle L40 | 0.98 | 0.98 | 0.84 | 0.93 | 1.00 | 1.00 |
| Oracle L44 | 0.99 | 0.98 | 0.77 | 0.92 | 0.98 | 0.99 |
| NLA L42 | 0.99 | 0.99 | 0.97 | 0.96 | 1.00 | 1.00 |

- **κ (P3):** undefined. Every reader names the word on all 100 items from the whole activation.
- **Token lenses** (median rank of the target, whole / echo): J_CB 4 / 2.5–3; J_NP 38.5–52 / 4.5–8; logit 45–128.5 / 13–38.
- **Paired contrasts (POST HOC, written after seeing the table; `scripts/014_typo/paired.py` → `paired_full.json`).**
  Per item, the share of 2 samples naming the target; item bootstrap.

  | reader | whole − rest of J (cost of removing J) | J − random 1,024 |
  |---|---|---|
  | Oracle L40 | +0.14 [0.08, 0.21] | +0.055 [0.015, 0.10] |
  | Oracle L44 | +0.225 [0.155, 0.30] | +0.065 [0.02, 0.115] |
  | NLA L42 | +0.025 [0.00, 0.055] | +0.03 [−0.005, 0.065] |

  - **Oracle − NLA on the cost of removing J:** L40 +0.115 [0.05, 0.19]; L44 +0.20 [0.13, 0.275].
- **Reading** (exploratory, post hoc contrasts):
  - The corrected word is readable almost everywhere. But removing the J subspace costs the Oracle 14–23 points and the NLA about 2.5, and the paired Oracle − NLA difference excludes 0.
  - This is the same direction as 007 on brew `:` (the Oracle's z reading depends on J; the NLA's does not).
  - Caveats:
    - near-ceiling naming;
    - 2 samples per cell;
    - different layers (Oracle L40/L44 vs NLA L42);
    - the contrast was chosen after seeing the table.
- **Predictions:**
  - P1 holds (every reader ≥ 0.80 from the whole; close to echo).
  - P2: rest ≥ 0.40 for every reader and J ≥ 0.70, so "the corrected word is readable outside J". J − R0 is small but positive for the Oracle.
  - P3 can't be tested (κ undefined, no variance).
- **Part C gate R1** (`nlah_gate_full.json`, 100 first samples): FVE 54.7% [50.7, 58.1] vs shuffled −100.8%. **Passes.**
  Judges and scoring are running.
