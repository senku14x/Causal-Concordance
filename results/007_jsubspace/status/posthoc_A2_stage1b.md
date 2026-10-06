# 007 post hoc (A2) — NLA L42, Stage 1b (brew dev, `</think>`) — status (2026-10-06)

**Label: post hoc** (A2). **Stopped at a gate; no NLA result at `</think>`.**

**What happened.** On 006's exact scorer, the first gate, G-score1 (`score_seq` vs `score_next` on the single-token spellings
of the first 4 `</think>` activations), gave max 0.280 nats against the 0.25 bound (median 0.046, bound 0.05). By A2 and rule
12 the stage stopped; nothing was repaired and no threshold was changed. GPU time 0.7 min.

**Why this is not a scorer fault.** The scorer is 006's own path; at `:` the same comparison gives 0.244-0.247 (006 and this
rerun). 006 ran G-score1 only on `:` activations, once per reader-layer; 007 runs it per stage on that stage's own
activations. The NLA's bf16 difference between the two computation paths sits near the 0.25 bound (set in 006 v1.1 from
Oracle diagnostics), and at `</think>` these 4 activations land above it.

**Options for the PI (post hoc, not taken).**
- Run Stage 1b's NLA with G-score1 evaluated as in 006 (on `:` activations, where it passes); about 2.2 GPU-h.
- Leave the NLA out of `</think>`.

The Oracle's Stage 1b result (H_J supported at both layers) is unaffected.

**Also disclosed: the multihop post hoc run crashed once** (my coding error: the parity reference was passed with the wrong
call signature) before any gate value was computed. It was fixed and relaunched; the crashed run is kept as
`outputs/007_jsubspace/stage3/m1_nla_posthoc_crashed/`.
