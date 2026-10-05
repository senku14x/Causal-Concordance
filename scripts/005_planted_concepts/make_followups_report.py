"""Write docs/reports/005_followups.md from results/005_planted_concepts/analysis/followups_summary.json (numbers from
JSON, rounded half-up). EXPLORATORY: the test concepts' 005 outcomes had been seen."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_report_page import RESA, ROOT, SNAME, pct, r  # noqa: E402

S = json.loads((RESA / "followups_summary.json").read_text())


def ci(b, nd=2):
    return f"{r(b['mean'], nd, True)} [{r(b['ci'][0], nd)}, {r(b['ci'][1], nd)}] (n = {b['n']})"


A1, A2, A2b, A3, QC = S["A1"], S["A2"], S["A2b"], S["A3"], S["Qc"]
g0 = S["G0"]
out = ["# 005 follow-ups (A1–A3, A2b, Q-c) — EXPLORATORY", "",
       "*Spec:* `docs/design_specs/005_followups.md` (pre-registered before the GPU part ran; commit 323362c). The test "
       "concepts' 005 outcomes had been seen, so nothing here is confirmatory. *Data:* full run `20261005T015235Z` and "
       f"follow-up run `{S['followups']}`; summary `results/005_planted_concepts/analysis/followups_summary.json`.", "",
       "## What ran", "",
       "- CPU on the full run's outputs: A1 (reader overlap), A2 (ΔR1′ vs B2).",
       "- GPU, 4 minutes: A2b (the concept's log-prob rise at every prefill position, 45 test concepts × 9 doses × "
       "conditions 1–4 and rand-dir), the A3 geometry (no model writes), and Q-c (30 concepts × 9 doses: J atom, "
       "logit-lens direction, full vector, rand-dir × 5).",
       f"- G0: {sum(v['writes_checked'] for v in g0.values())} write records, "
       f"{sum(v['failures'] for v in g0.values())} failures. Rows that repeat the full run reproduce it (A2b max |Δ| "
       f"{r(A2b['consistency_vs_full_run_dR1p_maxabs'], 3)} nats; Q-c condition 1 {r(QC['consistency_cond1_vs_full_run_maxabs'], 3)}).",
       "- Not run: 005b's Q-a and Q-b as written (reasons in the spec).", ""]

# A2 / A2b
cells = {k: v for k, v in A2["test"].items() if float(k.split("|")[1]) in (1 / 16, 1 / 8, 1 / 4, 1 / 2)}
out += ["## A2 and A2b: what the model-side measure measures", "",
        f"- **A2.** Spearman ρ over the 45 test concepts between ΔR1′ (answer slot) and B2 (the concept's rise on the "
        f"neutral weather prompt): median {r(A2['median_rho_rule_cells'], 2)} over conditions 1–4 × s = 1/16…1/2 "
        f"(range {r(min(v['rho'] for v in cells.values()), 2)} to {r(max(v['rho'] for v in cells.values()), 2)}). Below the "
        "0.7 rule, so A2 alone does not force the rename.",
        f"- B2 is larger than ΔR1′ in every cell: for the full vector at s = 1/4 the neutral-prompt rise exceeds the "
        f"answer-slot rise by {ci(A2['test']['2|0.25']['B2_minus_dR1p'])} nats (median ΔR1′ "
        f"{r(A2['test']['2|0.25']['median_dR1p'], 2)}, median B2 {r(A2['test']['2|0.25']['median_B2'], 2)}). Caveat: B2 is a "
        "max over 8 positions.",
        "- **A2b (the discriminative test).** Answer-slot rise minus the median rise at the 12 non-answer prefill "
        "positions, mean over s = 1/16, 1/8, 1/4:"]
names = {"1": "pure J-lens atom", "2": "full vector", "3": "J part", "4": "non-J part", "7": "random direction"}
for c, v in A2b["by_cond"].items():
    out.append(f"  - {names[c]}: {ci(v)} nats → {'slot-specific' if v['slot_specific'] else 'position-generic'}")
prof = A2b["profile_median"]["2"]["0.25"]
out += [f"  - Profile, full vector at s = 1/4 (median Δ by position 125–138): " + ", ".join(r(v, 2) for v in prof.values()) + ".",
        f"- **Verdict by the pre-registered naming rule: \"{S['A2_naming']}\".** The concept's log-prob rises about as much "
        "at every prefill position as at the answer slot, and more on an unrelated prompt. Every 005 statement about the "
        "model's \"report\" (ΔR1′, R1; M1, M3, M2, P11′) is about lexical priming of the concept token, not a report.", ""]

# A3
p8, p4 = A3["cos_quote|0.125"], A3["cos_quote|0.25"]
own = A3["own_atom"]
n_sel = sum(1 for o in own["per_concept"] if o["selected_layers"] > 0)
out += ["## A3: does alignment with the output direction explain potency?", "",
        f"- Regression of ΔR1′ on cos(direction, g ⊙ W_U[answer token]), 45 concepts × 7 conditions: R² "
        f"{r(p8['R2'], 2)} at s = 1/8 and {r(p4['R2'], 2)} at s = 1/4 (slopes {r(p8['slope'], 1)}, {r(p4['slope'], 1)}). "
        f"Condition identity adds ΔR² {r(p8['delta_R2'], 2)} and {r(p4['delta_R2'], 2)} (F-test p < 0.001). Rule "
        f"(R² ≥ 0.5 at both doses and ΔR² < 0.05): **{'met' if A3['rule_met_primary'] else 'not met'}**. Within a condition "
        "the cosine explains little (|r| ≤ 0.4); the between-condition difference is mostly the pure J atom "
        f"(median cos {r(p4['within_condition']['1']['median_cos'], 3)}) vs everything else (≤ "
        f"{r(max(v['median_cos'] for k, v in p4['within_condition'].items() if k != '1'), 3)}).",
        f"- **The concept's own atom is in its J part for {n_sel} of {len(own['per_concept'])} test concepts** (median energy "
        f"share {r(own['median_energy_share'], 3)}; median cos²(u^J, own atom) {r(own['median_cos2'], 3)}). The J part is a "
        "mixture of related and junk tokens (soap: sodium, soda, bacterial, laundering, …; dog: Dogs, adorable, Bulldogs, "
        "…). The earlier statement that M1 passes \"partly by construction because the J part contains the concept's own "
        "atom\" is wrong and is withdrawn.", ""]

# Q-c
cv = QC["curves"]
out += ["## Q-c: the logit-lens direction vs the pure J-lens atom as a write direction", "",
        f"- Per concept, ΔR1′(J atom) − ΔR1′(logit-lens direction of the same token), mean over s = 1/16, 1/8, 1/4: all "
        f"{ci(QC['all'])}; dev {ci(QC['dev'])}; test {ci(QC['test'])}.",
        f"- **By the PI's rule (the interval includes 0): J-lens adds nothing as a write direction on Qwen**, at these doses "
        "and for this measure (which A2b shows is priming).",
        "- The curves differ outside that window: median ΔR1′ J atom vs logit direction " +
        "; ".join(f"s = {SNAME[float(s)]}: {r(cv['1'][s]['dR1p'], 2)} vs {r(cv['L'][s]['dR1p'], 2)}"
                  for s in ("0.015625", "0.03125", "0.0625", "0.125", "0.25", "0.5", "1.0")) +
        ". The J atom is ahead at the lowest and the highest doses. No dose broke the model (B1 ≤ 5 nats everywhere).", ""]

# A1
d25, d1 = A1["by_dose"]["k1|0.25"], A1["by_dose"]["k1|1.0"]


def hr(d, rd, c):
    v = d["hit_rate"][rd][c]
    return pct(v) if v is not None else "—"
out += ["## A1: reader overlap and ensembles (direct reads, 45 test concepts; naming = ≥ 1 of 4 samples)", "",
        "| Reader | non-J s = 1/4 | full s = 1/4 | non-J s = 1 | full s = 1 | random s = 1 |", "|---|---|---|---|---|---|"]
for rd in ("NLA_L42", "Oracle_L40", "Oracle_L44", "probe_L42", "J_NP_L42", "J_CB_L42", "R_CB_L42", "logit_L42"):
    out.append(f"| {rd} | {hr(d25, rd, '4')} | {hr(d25, rd, '2')} | {hr(d1, rd, '4')} | {hr(d1, rd, '2')} | {hr(d1, rd, '7')} |")
k1 = d1["kappa_direct_nonJ"]["NLA_L42|Oracle_L40"]
e25 = d25["ensembles"]
out += ["",
        f"- Agreement on the non-J part at s = 1: κ(NLA, Oracle L40) {r(k1['kappa'], 2)} (both {k1['both']}, NLA only "
        f"{k1['x_only']}, Oracle only {k1['y_only']}).",
        f"- Unique catches on the non-J part at s = 1/4: probe {d25['unique_catches_nonJ']['probe_L42']}, NLA "
        f"{d25['unique_catches_nonJ']['NLA_L42']}, Oracle {d25['unique_catches_nonJ']['Oracle_L40']} and "
        f"{d25['unique_catches_nonJ']['Oracle_L44']}, every token lens 0.",
        f"- Union of the 8-reader panel vs the best single reader (the probe) at s = 1/4: full vector "
        f"{ci(e25['2']['union_minus_best'])}; non-J part {ci(e25['4']['union_minus_best'])}; the union's false-positive "
        f"rate on random directions {pct(e25['7']['union'])}. Majority vote is worse than the probe (the lenses never fire).",
        "- The token lenses (J_NP, J_CB, R_CB, logit lens) put the concept in their top 10 for ≤ 4% of direct reads at "
        "L40–44 even at s = 1. Carried reads at C_o: 0% for every token reader and the probe.",
        "- Caveat: the probe is a supervised, per-concept classifier that knows the target; it is a ceiling, not a "
        "reader in the same sense as the text readers.", "",
        "## What contradicts earlier claims", "",
        "1. **\"Report\" → lexical priming.** Pages, reports and my summaries called ΔR1′ the model's report (\"Qwen "
        "reports planted concepts by the paper's own metric\"; \"the metric is specific to the planted concept\"). The "
        "concept-specificity stands, but A2b shows the rise is position-generic: it is priming of the concept token.",
        "2. **\"M1 passes partly by construction (the J part contains the concept's own atom)\"** is wrong: own atom in 1 of 45.",
        "3. **My last summary's storm example** listed ` storm` and ` Storm` as J-part atoms. They are spellings added to "
        "the clamp's token set, not part of the J part.",
        "4. **P11′ (Oracle) compared a reader's likelihood with the model's \"report\".** It is a comparison with priming "
        "at the answer slot, which makes \"the reader is more sensitive than the model's report\" a statement about "
        "priming, not report.", "",
        "## Doubts", "",
        "- B2 is a max over 8 positions and ΔR1′ a single slot, so A2's magnitude comparison is biased toward B2; A2b avoids this.",
        "- A2b's non-answer positions predict prefill tokens such as \"Yes\", \"detect\" and \"thought\"; a concept token is "
        "implausible there. Priming that rises equally where the token is implausible and where it is the answer is "
        "the point, but it is one prompt.",
        "- Q-c's window (1/16–1/4) was fixed in advance. The J atom wins outside it, so \"adds nothing\" is window-dependent.",
        "- A1's best-single-reader comparison picks the best reader on the same data (optimistic for the single reader "
        "only by selection; here the probe is best by a wide margin).", ""]
(ROOT / "docs/reports/005_followups.md").write_text("\n".join(out))
print("wrote docs/reports/005_followups.md", len(out), "lines")
