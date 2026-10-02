# STATUS — 2026-10-02

**Block 000 (setup) is done. Design 001 is at draft v2 (revised after the external review of c13fd43) and awaits PI
review. No scientific run has been made, and none may start without the PI's go.**

| Block | State | Evidence |
|---|---|---|
| a. Repo | done | this file, `CLAUDE.md`, `README.md`, `docs/context/{SPEC,AMENDMENT_01,READING_LIST,reading_notes}.md`, `docs/reports/000_reading_report.md` |
| b. Environment | done | model loads (16 s, 50.1 GB); one brew forward gives the gated answer; 29/29 hashes match; `scripts/000_setup/rebuild_env.sh` |
| c. Reader smoke tests | done (5 work; SAE and the optional vLLM path blocked) | table below; `results/000_setup/analysis/reader_smoke.json` |
| d. Shared code | done, 61/61 tests pass | list below |
| e. Design 001 | draft v2, not run | `docs/design_specs/001_leverage_profile.md` (v1 → v2 changes at its bottom) |

## Environment

Full details in `CLAUDE.md` § Environment. RTX PRO 6000 Blackwell (sm_120, 96 GB); torch 2.11.0+cu128, transformers
5.15.0, peft 0.21.2, jlens @ 581d398; `Qwen/Qwen3.6-27B` @ 6a9e13bd (bf16, eager).

- **Hashes** (`results/000_setup/analysis/artifact_hashes.json`, 29/29 match):
  - J_NP, J_CB, R_CB lens files against the sha256 in RHUT's `instrument_registry.json` (which pins only these three
    files plus the model revision);
  - the 15 model shards, the Oracle adapter, the NLA adapters, critic and value head, and the NLA reference parquet
    against Hugging Face's LFS sha256 at the pinned revisions;
  - the 5 WSB bank files against the registry.
- **Env check** (`results/000_setup/analysis/env_check.json`): on `brew-s2-r10d2-0000` the greedy answer is " black"
  (gold: black). Top-5 next tokens: " black" −0.31, " Black" −2.31, " green" (= z) −2.68, `<|im_end|>` −4.06,
  " Green". The runner-up is the intermediate colour (WSB's runner-up caveat). One 165-token forward takes 0.23 s.

## Reader smoke tests (block c)

One cell: `brew-s2-r10d2-0000` (start blue → z green → answer black), emit_colon = the ":" after "Answer", position 164.
Activations come from the stock bf16 forward (WSB convention). The verbalizers sample 4 times with WSB's sampling
(T = 1.0, top_p 0.95, top_k 64, ≤ 256 tokens).

| Reader | Status | Load + run | GPU peak | Raw output (abridged) | Names colours? |
|---|---|---|---|---|---|
| Logit lens (L40/42/44) | **works** | < 1 s | 50 GB (model) | L42: ' ', ' (', ' either', '...', '\n\n' | no at L40–44. Positive control: at L63 top-1 is ' black' (= model greedy); colours appear from L56 |
| J_NP (standard + WSB cosine) | **works** | 16.9 s for all three J lenses (3 layers, sha-checked) | 56.4 GB | standard L42: '\n\n', '<\|im_end\|>', '<\|endoftext\|>'; cosine L42: ' END', ' FINAL', '입니다', '_final' | no at L40–44; at L56: ' black' top-1; at L60–62: black, green, brown |
| J_CB (standard + cosine) | **works** | (above) | (above) | standard L42: '终点' (end point), '第二次' (2nd time), '不会再', '终结', ' downstream' | no at L40–44; yes from L56 |
| R_CB (standard + cosine) | **works** | (above) | (above) | standard L42: ' ...', ' either', '第二次', ' intermediate', ' second' | no at L40–44; yes from L56 |
| Oracle Lens, L40 (on-contract) | **works** | 32 s (adapter + 3 arms × 4 samples) | 51.3 GB | "- Green (Since after two steps from Blue, we reach Green, …)", "- blue (or green) …"; no activation: Korean company names (㈜ = the corporation sign); another cell (stir "You"): liquid-colour puzzle set-ups | **yes**: green, blue, red, purple, gray, black (start and z among them) |
| NLA-L42 AV | **works** | 156 s (SFT merge + RL adapter; 3 arms × 4 + 8 reference rows) | 52.5 GB | "Table structure … two successive applications of a lookup table … (which is green/…)"; 4/4 parsed, 1 injection per generate; no activation: generic web text (car audio, patents); other cell: colour-rule puzzle set-up | **yes**: z (green) in 4/4 samples, one also pink/blue/red |
| NLA-L42 AR | **works** | 23 s | 35.2 GB | FVE of the cell from its own explanations: **0.46, 0.55, 0.59, 0.50**. No-activation explanations: −0.15 to −0.48. Other-cell explanations vs this cell: −0.41 to −0.66 (vs their own cell: 0.56–0.65) | n/a |
| NLA checks | **work** | — | — | Our L42 capture re-extracts the NLA's stored reference activations at cos ≥ 0.99984. AV → AR FVE on 8 reference rows: 0.75–0.86 (card: 74.5% held-out) | — |
| SAE | **blocked** | — | — | location and labels not provided (PI) | — |
| vLLM path (optional) | **blocked (not pursued)** | — | — | engine init fails on sm_120: `Failed to get device capability: SM 12.x requires CUDA >= 12.9` → `RuntimeError: FlashInfer requires GPUs with sm75 or higher`. Likely cause: a CUDA-12.8 component in vLLM 0.21's stack misreads this card (vllm-metamodels was tested on B200 / sm_100 only). Parked per the PI (HF first). HF reference timing for later: 64 NLA explanations in 107 s | — |

Notes (smoke observations on one cell, not findings):
- No lens reads a colour at L40–44 here, though the verbalizers name z.
- The FVE reference set is the NLA's own (FineWeb text), not this task's distribution.
- The Oracle adapter is the public copy `olens_s3d_rl600`. That it equals WSB's private `s3d.ddp600.s0/iter_000600`
  is assumed from the names and cannot be verified.

## Tests (block d) — 61 passed, 0 failed (`python -m pytest scripts/common/tests -v`, log `outputs/logs/pytest_full.log`)

| Module | Tests | Result |
|---|---|---|
| cells | round-trip (json, jsonl); validation (layer, half, split, family, position, empty ids; 6 cases); unknown keys rejected | 8/8 |
| wsb (rendering) | bank hashes and sizes; `brew-s2-r10d2-0000` regions (stir 131–136, start 138–143, question 145–153, emissions 158/162/164, " blue" at 142); all 100 brew items' 24 positions; chain " What" + final "\n\n" and relational "'s" on every item; byte-equality with WSB's own `_chat` renderer | 5/5 |
| partners + gate | brew trajectories reproduce the bank; hand-checked brew partner (0000: pink valid, gray excluded by z = a′; 165 tokens, diff only at 142); constraints hold for all 361 candidates; chain step functions reproduce all 120 items; hand-checked chain partner (0408, start 19 → 14: z (9,4) → (7,3), one digit differs); 50 relational pairs aligned (2 diffs); answer parsing (8 cases) | 14/14 |
| margin | spelling tokenisation (" Teal" two tokens, "27" two digits); sign implied by greedy (brew and chain, member < 0 < partner); multi-token teacher forcing scores every token, with an off-by-one discriminator; shared first token ("21" vs "27") → margin from the second token; writes applied on every forward | 5/5 |
| engine | self-patch bitwise (5 cells incl. L2, L63, multi-position); 1-D swap ρ = κ = 1 in float32 with read-back < 1e-4 and exactly one row changed; swap along unit(h_p − h_m) equals the full patch at the write (rel 2.7e-8); last-layer last-position replacement reproduces the partner's logits bitwise; earlier layers and positions untouched bitwise; the stock bf16 regime distorts a small write (ρ 1.04, κ 0.91) | 10/10 |
| gates | fidelity thresholds; damage-gate logic; span gate on the hand-checked pairs; self-patch damage is exactly 0; sequence-length determinism reported (not bitwise, onset at block 3, no leak) | 5/5 |
| stats | cluster-t vs scipy; BH-FDR vs statsmodels (exact); Holm vs statsmodels; cluster bootstrap covers and is wider than the naive interval; ratio of means; empirical p; equivalence | 6/6 |
| figures | ICLR figure round-trip (pdf + png); distinct reader colours | 2/2 |
| import boundary | no label module (registry, cells, wsb, partners, margin, engine, gates, stats, regime) reaches readers/jlens/peft/vllm, transitively and including imports inside functions; sanity: readers does reach jlens | 2/2 |
| model integrity | Engine, SecReplay and capture_stock refuse a model a reader merged weights into (3 cases); Oracle attach → capture (adapter disabled) → unload leaves the subject's activations and logits bitwise unchanged | 4/4 |

## Regime check (`results/000_setup/analysis/regime_check.json`)

The same 5 edits under RHUT's float32-residual regime and SEC's true-float32 replay (blocks 40–63 in float32, resident,
72 GB). Pair: the first gated pair in bank order, `brew-s2-r10d2-0001__blue` (purple→green→orange vs blue→black→pink).
The hand-checked `0000__pink` failed the gate (partner 5/10) and `0001__black` failed too (partner 8/10).

| Edit | RHUT from block 0: Δm | SEC: Δm | \|diff\| / G_SEC |
|---|---|---|---|
| full patch, emit_colon, L44 | +4.770 | +4.426 | **0.033** |
| full patch, emit_colon, L40 | +2.437 | +2.220 | 0.021 |
| full patch, start-colour token, L40 | +0.559 | +0.526 | 0.003 |
| 1-D swap at emit_colon L42 along the emit_think pair difference | +2.032 | +2.123 | 0.009 |
| 1-D swap at emit_colon L42 along a random direction | −0.015 | +0.001 | 0.002 |

- G: RHUT-from-0 10.63, RHUT-from-35 10.52, stock bf16 10.31, SEC 10.48 nats.
- Max |diff| / G against SEC: RHUT-from-0 **3.3%**, RHUT-from-35 2.3%, stock bf16 1.3%.
- **Decision under the 5% rule: keep RHUT's float32-residual regime as the default** (`AMENDMENT_01` §7).
- RHUT writes are exact (ρ = κ = 1); stock-bf16 random-direction writes are not (ρ 1.04, κ 0.86).

**Noise floor**: |Δm| under 8 perturbations of 1e-4 relative norm at emit_colon L42.

| Regime | Max \|Δm\| (nats) | Fraction of G |
|---|---|---|
| RHUT, from block 0 | 0.053 | 0.50% |
| RHUT, from block 35 | 0.034 | 0.33% |
| Stock bf16 | 0.064 | 0.62% |
| SEC | **0.0002** | 0.002% |

- On `0000__pink` (G 8.6), the RHUT floor reached 0.24 nats (2.7% of G) at the start token at L2.
- Appending 3 tokens moved that pair's margin by 0.22 nats.
- So the RHUT floor is absolute, roughly 0.05–0.25 nats, and on pairs with G below about 7 nats a gap the size of the
  observed 3.3% would cross 5% of G.
- SEC's own STATUS (L45–46) made the same observation independently: bf16-matmul finite differences give cos 0.80.
- Open question 2 below.

## Other block-000 findings (engineering)

- **Sequence length.** Activations at prompt positions depend on the total length. Blocks 0–2 are bitwise identical;
  the onset is at block 3, the first full-attention block, at 6e-4 relative. That grows to ~1% median by block 63, and
  up to 40–99% at a few positions. It is rounding, not a leak. Consequence: comparisons are made at one length only,
  with partner sources captured at the matching length.
- **Bank.**
  - Brew: 361 partner candidates, all token-aligned; per-item counts 0–8; 3 items have none.
  - Chain: digits are single tokens, so only 28/120 items have single-token intermediates. Of those, 25 have partners
    (110 candidates, all aligned). There are 15 cross-item prompt collisions (identical prompts from items sharing step
    sequences), so chain must split by step sequence and de-duplicate pairs.
  - Relational: 50/50 a/b pairs aligned (gating pending).
- **Gate** (3 brew pairs tried so far): `0000__pink` fail (partner 5/10: wrong samples name the intermediate or another
  colour), `0001__black` fail (partner 8/10), `0001__blue` pass (member 10/10, partner 9/10). One chain pair tried,
  `chain-w2-d3p-0408__12`, passes.
- **Timing** (`timing_smoke.json`), used for design 001's budget:
  - forward ≈ 0.23 s whether T = 78 or 165;
  - gate ≈ 5.4 s per brew pair, 4.4 s per chain pair;
  - HF NLA generation: 64 explanations of ≤ 256 tokens in 107 s.

## Open questions for the PI

1. **SAE**: location and auto-interpretation labels (AMENDMENT_01 §4).
2. **Regime**: design 001 v2 uses SEC's true-fp32 replay for every L ≥ 40 (errata E4). Confirm (001 §9.1).
3. **Reader snapshot**: include the optional descriptive reader stage in 001, or defer it to 002 (001 §3.7)?
4. **Chain split**: 50/50 by step-sequence group (≈ 10 groups per side), or 70/30 (001 §9.3)?
5. **HF dataset name** for the private `outputs/` mirror.
6. **Paper PDFs** in this public repo: committed or local only? (Currently gitignored; `docs/papers/INDEX.md` lists them.)
7. **P5 prior**: the dated update accepted in AMENDMENT_01 §5 needs your number.
8. **Branch**: merge `setup/000-scaffold` to `main` on your say-so.

## External review (2026-10-02)

Every concrete claim was checked against the repo, and all hold. Accepted changes:
- design 001 v2 (its change list);
- AMENDMENT_01 errata E1–E4;
- the model-integrity guard and its tests.

Two corrections to the review discussion:
- J does name green at L36 (emit_colon) and L40 (emit_think) on the smoke item. "No colours at L40–44" held only for
  the emit_colon cell.
- SEC removes the reader-input bridge only for layers ≤ 39 (001 §9.4).

## Next

PI review of design 001 → amendments → on your go, `scripts/001_leverage_profile/` (stages smoke → run) → report 001.
