"""Write results/005_planted_concepts/analysis/report_notes.json: the interpretive sentences for the report page.
Numbers are pulled from pilot_summary.json and smoke_summary.json and rounded half-up here; the wording is the
agent's interpretation and is labelled as such on the page (dev concepts only)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_report_page import RESA, SNAME, load, med, pct, r  # noqa: E402

S = load(RESA / "pilot_summary.json") or {}
N = {}


def blk(k):
    return {"B_default": "L32–38/default", "B_word": "L32–38/‘word’", "B2_default": "L24–38/default",
            "B2_word": "L24–38/‘word’"}.get(k, k)


# Q1
if S.get("Q1"):
    q = S["Q1"]["by_block"]
    N["Q1"] = ("The paper's own measure works on Qwen where sampled naming does not. In every block the concept's "
               "log-probability at the report position rises above what random directions reach, at "
               + ", ".join(f"{len(v['cond1_strengths_above_null'])}/9 doses for the pure J-lens vector and "
                           f"{len(v['cond2_strengths_above_null'])}/9 for the full vector ({blk(k)})" for k, v in q.items())
               + ". Proposal A1: make R1′ (with R1) the primary report measure and keep sampled naming as a secondary column.")
# Q2
if S.get("Q2"):
    q = S["Q2"]["by_block"]
    best = max([(v["cond1_best_R2"], k, "pure J-lens vector") for k, v in q.items()] +
               [(v["cond2_best_R2"], k, "full vector") for k, v in q.items()])
    N["Q2"] = ("No. The ‘word’ prefill does not remove Qwen's default answer: “apple” stays the top next word in the clean "
               "run and under weak injections, and sampled naming stays far below the 0.25 bar (best "
               f"{r(best[0], 2)}, {best[2]}, {blk(best[1])}). Qwen appears to have a canned answer for “the thought is about”. "
               "Proposal A2: keep the protocol's default prefill (the paper's), and treat naming as secondary.")
# Q3
if S.get("Q3", {}).get("by_prefill"):
    q = S["Q3"]["by_prefill"]
    d = q.get("default") or next(iter(q.values()))
    N["Q3"] = ("The rule keeps band L32–38. The wide band L24–38 gives a much larger report effect for the full vector "
               f"({r(d['B2']['cond2_best_dR1p'], 2)} vs {r(d['B']['cond2_best_dR1p'], 2)} nats at its best dose, default "
               f"prefill), but it also disturbs the neutral prompt more (KL {r(d['B2']['B1_at_best'], 2)} vs "
               f"{r(d['B']['B1_at_best'], 2)}), and the pre-registered rule required a larger effect without a larger KL. "
               "This is a real trade-off: L24–38 buys a bigger report signal at the price of broader disruption. "
               "Proposal A3: L32–38 as registered; you may prefer L24–38 for signal, and the pilot gives you both.")
# Q4
if S.get("Q4"):
    q = S["Q4"]["by_block"]
    N["Q4"] = ("Yes. With the grid extended down to 1/256, every block has a dose range where the non-J part leaves the "
               "report untouched: onset s* = " + ", ".join(f"{SNAME.get(v['s_star'])} ({blk(k)})" for k, v in q.items()) +
               ". The reader doses by the rule would be " + ", ".join(SNAME[s] for s in q["B_default"]["text_strengths_rule"]) +
               " for L32–38/default. Proposal A4: the 9-point grid, with text readers at {s*/4, s*, 4s*}.")
# Q6
if S.get("Q6"):
    q = S["Q6"]["by_block"]
    N["Q6"] = ("Yes, unlike the 2-concept smoke: in all four blocks the design's clamp removes more of the non-J part's "
               "report effect than the random clamp does (" + "; ".join(
                   f"{blk(k)} {r(v['extra_reduction_dR1p_mean'], 2, True)} nats [{r(v['ci'][0], 2)}, {r(v['ci'][1], 2)}], "
                   f"{v['n_evaluable']} concepts" for k, v in q.items()) +
               "). Caveat: this clamp runs to L62, where a concept's J direction approaches its output direction, so part of "
               "the effect may be the clamp blocking the answer itself. Step 7b's band-limited re-entry clamp (Q10) tests that.")
# Q7 / Q8
if S.get("Q7", {}).get("leaky"):
    lk = S["Q7"]["leaky"]
    N["Q7"] = ("Few splits leak. With the rule fixed beforehand, band L32–38 has " +
               (", ".join(lk.get("B", [])) or "no leaky concepts") + " and band L24–38 has " +
               (", ".join(lk.get("B2", [])) or "none") + ". Proposal A8: exclude these from every non-J contrast in the "
               "full run (earth is a dev concept; soap is a test concept and would be excluded before any test outcome exists).")
if S.get("Q8"):
    N["Q8"] = (f"The baseline matters somewhat: the median cosine between the two concept-vector versions is "
               f"{r(S['Q8']['median'], 3)} (lowest {r(S['Q8']['min'], 2)}), below the 0.9 line. Leave-one-out stays primary "
               "and the 194-word version becomes a mandatory sensitivity check (A10).")
# previews and budget
pv = S.get("previews_not_tests", {})
if pv:
    d = pv.get("B_default", {}).get("M1_like_J_minus_perp_dR1p", {})
    if d:
        lo = [d[k] for k in d if float(k) <= 0.125]
        hi = [d[k] for k in d if float(k) >= 0.25]
        N["M1_preview"] = (f"Preview only (dev concepts, no test): at matched norm the J part beats the non-J part at low and "
                           f"middle doses (up to {r(max(lo), 2, True)} nats at L32–38/default) and loses at high doses "
                           f"(down to {r(min(hi), 2, True)}). The two reach similar peaks; the J part gets there at a lower dose.")
# Q5 readers
q5 = S.get("Q5", {})
if q5.get("nla") and q5.get("oracle"):
    n, o = q5["nla"]["mean_naming"], q5["oracle"]["mean_naming"]

    def g(d, k):
        return d.get(k)
    nla_full_1 = g(n, "B|42|direct_cq|2|1.0")
    nla_perp_1 = g(n, "B|42|direct_cq|4|1.0")
    nla_perp_q = g(n, "B|42|direct_cq|4|0.25")
    ora_perp_1 = max(x for x in (g(o, "B|40|direct_cq|4|1.0"), g(o, "B|44|direct_cq|4|1.0")) if x is not None)
    ora_perp_q = max(x for x in (g(o, "B|40|direct_cq|4|0.25"), g(o, "B|44|direct_cq|4|0.25")) if x is not None)
    ora_j_q = max(x for x in (g(o, "B|40|direct_cq|3|0.25"), g(o, "B|44|direct_cq|3|0.25")) if x is not None)
    carried = [v for k, v in list(n.items()) + list(o.items()) if "|carried_co|" in k and k.split("|")[3] in ("2", "3", "4")]
    N["Q5"] = ("Both text readers name the planted concept when they are handed the written cell, and neither sees it at "
               "the report position. On direct reads at s = 1 (band L32–38) the NLA names the full vector in "
               f"{pct(nla_full_1)} of samples and the non-J part in {pct(nla_perp_1)}; the Oracle names the non-J part in up "
               f"to {pct(ora_perp_1)}. At the report position the highest naming rate for any planted condition, reader, "
               f"dose and band is {pct(max(carried))}. The reader-free probe agrees (it never classifies the report position "
               "as the concept at L42), so this is the concept not having arrived there, not the readers missing it. "
               "As posed, P11a and P11b compare the readers with the model at a position the content does not reach by the "
               "readers' layers; step 8 checks later layers.")
    N["P17_preview"] = (f"Preview of P17 (dev only, not a test): on direct reads the Oracle names the non-J part at least as "
                        f"often as the NLA (s = 1: Oracle up to {pct(ora_perp_1)}, NLA {pct(nla_perp_1)}; s = 1/4: Oracle "
                        f"up to {pct(ora_perp_q)}, NLA {pct(nla_perp_q)}). That runs against the A.9.2-based expectation that "
                        "the Oracle sees content outside J-space less. At s = 1/4 both readers name the J part far more often than "
                        f"the non-J part (Oracle up to {pct(ora_j_q)} vs {pct(ora_perp_q)}), the same dose pattern as the model's report.")
sec = S.get("SEC_crosscheck")
if sec:
    N["budget"] = (f"The fast float32-residual regime matches the exact SEC replay on {pct(sec['agree_frac'])} of "
                   f"{sec['n']} cells (median log-prob difference {r(sec['median_abs_dR1p'], 3)} nats), so the full run can "
                   "use it throughout (A12).")
out = RESA / "report_notes.json"
prev = load(out) or {}
prev.update(N)
out.write_text(json.dumps(prev, indent=1, ensure_ascii=False))
print("notes:", sorted(prev))
