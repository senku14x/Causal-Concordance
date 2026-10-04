"""Write results/005_planted_concepts/analysis/report_notes.json: the interpretive sentences for the report page.
Numbers are pulled from pilot_summary.json and smoke_summary.json and rounded half-up here; the wording is the
agent's interpretation and is labelled as such on the page (dev concepts only)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_report_page import RESA, SNAME, load, med, pct, r  # noqa: E402,F401

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
# Q10 / Q11 / R-lens (step 7b)
q10 = S.get("Q10", {}).get("by_band", {})
if q10:
    def pr(b, k):
        x = q10[b]["pairs"][k]
        return f"{r(x['extra_reduction_mean'], 2, True)} nats [{r(x['ci'][0], 2)}, {r(x['ci'][1], 2)}]"
    cos = S.get("cos_atom_vs_output") or {}
    cs = {l: med([cos[c].get(str(l)) for c in cos]) for l in (40, 44, 52, 62)} if cos else {}
    N["Q10"] = ("The clamp's specificity lives at the output end. Run all the way to L62, the re-entry clamp removes much more "
                f"of the non-J part's report effect than its random null (L32–38: {pr('B', '12_vs_12r')}; L24–38: "
                f"{pr('B2', '12_vs_12r')}). Stopped at L44 it removes nothing beyond its null (L32–38: {pr('B', '13_vs_13r')}; "
                f"L24–38: {pr('B2', '13_vs_13r')}, the random clamp even removing slightly more). The concept's J direction "
                f"turns into its output direction with depth (median cosine {r(cs.get(40), 2)} at L40, {r(cs.get(44), 2)} at "
                f"L44, {r(cs.get(52), 2)} at L52, {r(cs.get(62), 2)} at L62), so a clamp reaching the late layers blocks the "
                "answer itself. By the pre-registered reading: no evidence on these dev concepts that the non-J part's effect "
                "on the report is mediated by re-entry into the concept's J coordinates within the workspace layers. A clamp "
                "that reaches the late layers could produce the paper's result (the non-J effect falling to about zero) "
                "without any workspace re-entry, so a full-range clamp alone cannot separate the two. "
                "Proposal A7: if M2 stays, run it as the band-limited re-entry clamp with its null, and report the full-range "
                "version only next to the cosine curve.")
q11 = S.get("Q11", {}).get("by_band", {})
if q11:
    N["Q11"] = ("No, in either band. At every dose where the non-J part moves the model's report, the concept's median J-lens "
                "rank at the report position stays in the hundreds to thousands at every layer up to L62; the full vector and "
                "the J part behave the same, and so do the logit lens and the R-lens. Planting shifts the report distribution "
                "toward the concept without the concept ever becoming a top candidate at the report position, consistent with "
                "the readers (Q5) and the absolute probe criterion.")
rc = S.get("rlens_consistency")
if rc:
    N["rlens"] = (f"The R-lens and J-lens agree closely on these reads (Spearman {r(rc['spearman_JNP_RCB'], 2)}, top-10 "
                  f"agreement {pct(rc['top10_agree_JNP_RCB'])}), so for planted concepts the two Jacobian lenses are "
                  "interchangeable as token readers; adding R-lens to the full run costs little and answers the consistency "
                  "question directly (A13).")
# Q12 (step 8)
q12 = S.get("Q12", {}).get("by_band", {})
if q12:
    mx = max((x for v in q12.values() for k, cr in v.get("curves_absolute", {}).items() if not k.startswith("7")
              for x in cr.values() if x is not None), default=0)
    N["Q12"] = ("Read with care. By the pre-registered rule (probe response above the strongest of five random directions), "
                "every condition “arrives” at the report position by L32–34 at every dose, even 1/256, so all three registered "
                "predictions technically came true. That rule is degenerate: the forward pass is deterministic and nearly "
                "linear at these doses, so the comparison with random directions is scale-free and fires at any dose. By the "
                "absolute criterion (the probe classifies the read as the concept), the concept <b>never</b> reaches the "
                "report position at any layer L32–62, for any condition, dose or band (highest rate anywhere "
                f"{pct(mx)}). Together with Q5 and Q11: planting nudges the model's report distribution without ever "
                "producing a readable concept state at the report position. P11a/P11b at the report position therefore "
                "compare the readers with the model where neither can find the concept; the amendment must re-pose P11 "
                "(proposal: compare reader sensitivity with model sensitivity on matched measures at the written cell, "
                "and drop the report-position version).")
# power (8b)
pw = S.get("power_8b") or {}
if pw:
    m1b, m1b2 = pw.get("M1_like_dR1p_B"), pw.get("M1_like_dR1p_B2")
    nl = pw.get("nla_perp_direct_B_1.0")
    N["power"] = ("Planning numbers from the 10 dev concepts. For the J-vs-non-J report contrast at s = 1/4 the predicted 95% CI "
                  f"half-width with 45 test concepts is ±{r(m1b['ci_halfwidth_n45'], 2)} nats at band L32–38 and "
                  f"±{r(m1b2['ci_halfwidth_n45'], 2)} at L24–38 (concepts vary much more in the wide band, another point for "
                  "L32–38). For reader naming the spread between concepts dominates: the NLA's naming of the non-J part at s = 1 "
                  f"gives ±{r(nl['ci_halfwidth_n45_k4'], 3)} with 4 samples and ±{r(nl['ci_halfwidth_n45_k8'], 3)} with 8, so "
                  "doubling the samples buys almost nothing; more concepts is what helps (as your review argued).") if (m1b and m1b2 and nl) else ""

# the short version (top of the page)
sm = load(RESA / "smoke_summary.json") or {}
q13 = S.get("Q13")
bullets = []
if S.get("Q1") and S.get("Q2"):
    bullets.append("<b>Qwen reports planted concepts only by the paper's own metric.</b> Asked “the thought is about \"…\"”, "
                   "Qwen almost never <em>says</em> the planted concept: it has a canned answer (“apple”), and neither prefill "
                   "removes it. But the concept's log-probability at that position rises reliably above what random directions "
                   "do (Q1 yes, 4/4 blocks). The report gate as written failed; the amendment makes log-prob the report measure.")
if pv:
    bullets.append("<b>No clear J-space privilege for report on these dev concepts.</b> At equal size the J part beats the "
                   "non-J part at low and middle doses, the order reverses at high doses, and the two reach similar peaks. "
                   "This is a preview on 10 dev concepts, not the test (M1 runs on the 45 test concepts).")
if S.get("Q10"):
    bullets.append("<b>The privilege clamp works only by blocking the output end.</b> Clamping the concept's J coordinates all "
                   "the way to L62 removes much of the non-J part's effect (beating a random-token clamp), but stopping at L44 "
                   "removes nothing beyond the random clamp. The concept's J direction turns into its output direction with "
                   "depth, so the full-range clamp blocks the answer itself. On these concepts there is no sign that the "
                   "non-J effect re-enters J-space within the workspace layers.")
if S.get("Q5") and S.get("Q11"):
    nm = {**{"nla|" + k: v for k, v in S["Q5"]["nla"]["mean_naming"].items()},
          **{"ora|" + k: v for k, v in S["Q5"]["oracle"]["mean_naming"].items()}}
    top_full = max(v for k, v in nm.items() if "|direct_cq|2|1.0" in k)
    top_perp = max(v for k, v in nm.items() if "|direct_cq|4|1.0" in k)
    carried = max(v for k, v in nm.items() if "|carried_co|" in k and k.split("|")[4] in ("2", "3", "4"))
    bullets.append("<b>The readers see the planted concept only when handed the written cell.</b> On direct reads at the top "
                   f"dose the best reader names the full vector in {pct(top_full)} of samples and the non-J part in "
                   f"{pct(top_perp)}; at the report position no text reader exceeds {pct(carried)}, no lens puts the concept "
                   "in its top 10 at any layer up to L62, and the probe never classifies it as present. The model's report "
                   "shifts without a readable concept ever forming there.")
if S.get("Q5") and S.get("Q4"):
    onsets = sorted({v["s_star"] for v in S["Q4"]["by_block"].values() if v["s_star"]})
    first_read = None
    for sv in (1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0):
        if any(v > 0.10 for k, v in nm.items() if "|direct_cq|" in k and k.split("|")[4] in ("2", "3", "4")
               and abs(float(k.split("|")[5]) - sv) < 1e-12):
            first_read = sv
            break
    bullets.append("<b>The readers are less sensitive than the model, not more.</b> The model's report responds from s = "
                   + " or ".join(SNAME.get(o, str(o)) for o in onsets) + "; the first dose at which any reader names any "
                   f"planted condition in more than 10% of direct reads is s = {SNAME.get(first_read, '—')}. This compares "
                   "a fine measure (the model's log-prob) with a coarse one (whether a reader names the concept), so it holds "
                   "for these measures only; a reader-side likelihood could be more sensitive. On these concepts the NLA does "
                   "not see what the model cannot use, and the Oracle reads the non-J part at least as well as the NLA "
                   "(against the P17 premise).")
if S.get("Q12"):
    bullets.append("<b>A methods trap, caught.</b> Comparing a probe with random directions at the same dose is meaningless in "
                   "a deterministic forward: it ‘detects’ any planted direction at any dose. The absolute criterion is used "
                   "instead (amendment A9).")
if q13:
    bullets.append(f"<b>Late layers don't rescue it.</b> The Oracle at L48–60 at the report position names the full vector in "
                   f"{len(q13['full']['hits'])} qualifying cells and the non-J part in {len(q13['perp']['hits'])}.")
g0 = sm.get("G0_inject", {})
secx = S.get("SEC_crosscheck") or {}
dev = max(abs(1 - g0.get("rho_min", 1)), abs(g0.get("rho_max", 1) - 1)) if g0 else None
bullets.append("<b>The machinery is sound.</b> Every write exact (ρ within 1 ± " + (f"{dev:.0e}" if dev else "—") + "), runs "
               "bit-identical on repeat, the fast regime matches the exact one on " +
               f"{pct(secx.get('agree_frac'))} of {secx.get('n', '—')} cells, the C1–C3 checks and errata are in, and the "
               "amendment's budget for the full run is about 1.5–3 GPU hours (measured throughput, A16).")
N["short_html"] = ('<h2 id="short">The short version</h2><ul>' + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"
                   '<div class="callout"><h3>What you decide</h3><ul>'
                   "<li>The report measure (A1): log-prob at the open quote, with the paper's rank, instead of sampled naming.</li>"
                   "<li>Band (A3): L32–38 as registered, or the stronger but more disruptive L24–38.</li>"
                   "<li>P11 (A6): drop the report-position version (nothing arrives there) for a reader-vs-model sensitivity comparison.</li>"
                   "<li>The clamp (A8): test re-entry with the band-limited clamp and its null.</li>"
                   "<li>The go for the amended full run on the 45 test concepts (about 1.5–3 GPU hours).</li></ul></div>"
                   "<p class='muted'>Everything here is development data on 10 dev concepts. None of it is a test of the "
                   "pre-registered predictions; those need the amended full run.</p>")
out = RESA / "report_notes.json"
prev = load(out) or {}
prev.update(N)
out.write_text(json.dumps(prev, indent=1, ensure_ascii=False))
print("notes:", sorted(prev))
