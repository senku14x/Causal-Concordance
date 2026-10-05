# 007 Stage 4 — status (relational multihop; secondary; exploratory; whole bank; 2026-10-05)

**What ran (4a, reader-free).** WSB relational multihop: 50 scenes × a/b. Token alignment (equal length, exactly the two
relation-word tokens differ): 50/50. Gate (both members greedy-correct and ≥ 9/10 at T 0.8, top-p 0.95;
`partners.GATE`): **12 scenes pass** (≥ 10, so the profile ran). SEC whole-cell patch partner → member at 5
shared positions × L40-60 step 2, reverse on a fixed 30%; noise floor (1e-4-relative perturbations at the final position,
L42). GPU time 6.6 min.

**Checks.** Noise floor |dm|/|G| median 8.4e-06, max 2.8e-05; forward vs reverse S, r = 0.96.

**Profile: median S over the 12 scenes († = damage-dominated by 001's rule).**

| position | L40 | L42 | L44 | L46 | L48 | L50 | L52 | L54 | L56 | L58 | L60 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| final_s | 0.131 | 0.151 | 0.161† | 0.194† | 0.190† | 0.203† | 0.478† | 0.467† | 0.512† | 0.522† | 0.571 |
| possessor | 0.096 | 0.068 | 0.054 | 0.061† | 0.047 | 0.025 | 0.042 | 0.033 | 0.029 | 0.027 | 0.024 |
| is | 0.022 | 0.029 | 0.039 | 0.026 | 0.021 | 0.006 | -0.004 | -0.005 | -0.006 | -0.006 | -0.005 |
| cloze_subject | 0.064 | 0.051 | 0.043 | 0.025 | 0.021 | 0.002 | -0.006 | -0.003 | -0.002 | -0.002 | -0.002 |
| period_s2 | 0.020 | 0.016 | 0.013 | 0.008 | 0.003 | 0.004 | 0.001 | 0.000 | 0.000 | 0.000 | -0.000 |

**Pre-registered selection (design §9.2).** Higher leverage requires median S ≥ 0.25 at L42 and ≥ 0.15 at L40 and L44, not
damage-dominated: **no position qualifies** (the final 's carries 0.13-0.16 at L40-44 and rises only from L56). Lower
leverage (≤ 0.05 at L40/42/44, CI upper ≤ 0.10): " is" and the sentence-2 period. With no higher-leverage position,
**Stage 4 stops after 4a; 4b (Part A) did not run**, by rule.

**Observations.** At the readers' layers the relational answer's composition is not yet concentrated at the read cell:
whole-cell leverage at the final token is low at L40-44 and becomes high only at L56-60. Q1 (≥ 20 scenes) and Q2 (both
positions found) do not hold. Only 12 of 50 scenes pass the strict gate.
