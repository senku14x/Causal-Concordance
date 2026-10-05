"""005 full run: write docs/reports/005_full_run.md and the results page (one HTML file, inline SVG) from
results/005_planted_concepts/analysis/full_summary.json (+ full_notes.json for the interpretive sentences).

  python make_full_report.py OUT.html [--stage run|rehearse]

Every number is read from JSON and rounded half-up here. The interpretive sentences live in full_notes.json, written
after the results were read and labelled as interpretation on the page.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_report_page import (COND, CSS, GRID9, RESA, ROOT, SNAME, chip, esc, figure, legend,  # noqa: E402
                              line_chart, load, pct, r, table)

TEXT_S = [1 / 64, 1 / 4, 1.0]
ENDS = list(range(44, 63, 2))

# pre-registered priors: (contrast key, label, predicted, registered prior or None, updated prior or None)
PRIORS = [("M1_R1", "M1 on R1 (registered form): J part > non-J part at peak", "J > non-J", 0.85, None),
          ("M1_R1p", "M1 on R1′ (primary under A1): J part > non-J part at peak", "J > non-J", None, 0.25),
          ("M2", "M2: J re-entry clamp to L50 beats its random-token null", "> 0", 0.30, 0.65),
          ("M2_control_D_at_L50", "M2 control: J clamp beats the output-direction clamp at L50", "> 0", None, 0.40),
          ("M3", "M3: J-natural > non-J-natural where the full vector moves R1′", "> 0", 0.60, 0.10),
          ("R1_probe_minus_NLA", "R1: probe detects the non-J part at least as often as the NLA (point estimate ≥ 0)", "≥ 0", 0.70, 0.85),
          ("R2_behaviour_without_report", "R2: behaviour moves without report (share of concepts, CI > 0)", "> 0", 0.50, None),
          ("P11prime_NLA", "P11′ (NLA): reader onset earlier than the model's", "< 0", None, 0.55),
          ("P11prime_Oracle", "P11′ (Oracle): reader onset earlier than the model's", "< 0", None, 0.85),
          ("P17", "P17: the Oracle names the non-J part less than the NLA", "> 0", 0.45, 0.05)]


def outcome(key, C):
    c = C.get(key) or {}
    if key == "R1_probe_minus_NLA":
        return None if c.get("mean") is None else c["mean"] >= 0
    return c.get("supported")


def fmt_ci(c, nd=2):
    if not c or c.get("mean") is None:
        return "—"
    lo, hi = c.get("ci") or [None, None]
    return f"{r(c['mean'], nd, True)} [{r(lo, nd) if lo is not None else '—'}, {r(hi, nd) if hi is not None else '—'}]"


def brier(pairs):
    pairs = [(p, o) for p, o in pairs if p is not None and o is not None]
    return (sum((p - (1.0 if o else 0.0)) ** 2 for p, o in pairs) / len(pairs), len(pairs)) if pairs else (None, 0)


def build(S, N):
    G, C, T = S["gates"], S["contrasts"], S["tables"]
    n = S["n_concepts"]
    rehearse = S["stage"] == "rehearse"
    out, md = [], []

    # ------------------------------------------------------------------------------------------------ header
    gate_ok = G["G0"]["pass"] and G["G4"]["pass"] and not G["G3"]["flag"]
    hol = C.get("holm_P_family", {})
    chips = [chip(f"{n} {'dev (rehearsal)' if rehearse else 'test'} concepts", "good"),
             chip("gates G0/G3/G4 pass" if gate_ok else "a gate fails", "good" if gate_ok else "bad"),
             chip("amendment 01 adopted", "pend")]
    head = f"""<header class="top"><div class="eyebrow">Causal-Concordance · design 005 · full run under amendment 01 · 5 October 2026</div>
<h1>Planted Concepts Full Run</h1>
<p class="lede">The amended 005 run on the {n} {'rehearsal (dev) ' if rehearse else 'held-out test '}concepts: what ran, the gates,
every pre-registered contrast with its verdict, and the tables and raw readouts behind them. Every number is generated from
the saved result files. Run <code>{esc(S['run_id'])}</code>, branch <code>005-amended-full-run</code>.</p>
<div class="chips">{"".join(chips)}</div></header>"""
    out.append(head)
    md += [f"# 005 full run under amendment 01 — results (run `{S['run_id']}`, {n} {'dev rehearsal' if rehearse else 'test'} concepts)", "",
           "*Spec:* `docs/design_specs/005_full_run.md` (pre-registered before the run). *Amendment:* "
           "`docs/design_specs/005_amendment_01.md` (adopted 2026-10-05). *Branch:* `005-amended-full-run`. "
           f"*Summary JSON:* `results/005_planted_concepts/analysis/full_summary{'_rehearse' if rehearse else ''}.json`.", ""]
    if N.get("short_html"):
        out.append(N["short_html"])
        md += [N.get("short_md", ""), ""]

    # ------------------------------------------------------------------------------------------------ gates
    g6 = G.get("G6", {})
    g11 = G.get("G11", {})
    g2 = G.get("G2", {}).get("by_reader", {})
    rows = [["G0 fidelity", "every write ρ ∈ [0.9, 1.1], κ ≥ 0.99; clamp residual ≤ 1e-4",
             f"{sum(v.get('writes_checked', 0) for v in G['G0']['blocks'].values())} write records, "
             f"{sum(v.get('clamp_rows_checked', 0) for v in G['G0']['blocks'].values())} clamp rows, "
             f"{sum(v.get('failures', 0) for v in G['G0']['blocks'].values())} failures", chip("pass", "good") if G["G0"]["pass"] else chip("fail", "bad")],
            ["G3 J share", "median J share at k = 16 within [1%, 30%]", pct(G["G3"]["J_share_median"], 1),
             chip("flag", "bad") if G["G3"]["flag"] else chip("pass", "good")],
            ["G4 probe validity", "median LOTO AUC ≥ 0.9 and permutation-control AUC ≤ 0.6",
             f"{r(G['G4']['loto_auc_median'], 3)} / {r(G['G4']['perm_ctrl_auc_median'], 2)}", chip("pass", "good") if G["G4"]["pass"] else chip("fail", "bad")],
            ["G5 leaky splits", "c in the J_NP top 10 on the pure non-J part at ≥ 2 band layers", ", ".join(G["G5_leaky"]["leaky"]) or "none",
             chip("excluded from non-J contrasts", "warn") if G["G5_leaky"]["leaky"] else chip("none", "good")]]
    for k, v in g6.items():
        rows.append([f"G6 {k.replace('_', ' ')}", "clean-read naming > 5% drops the concept for this reader",
                     ", ".join(v["dropped"]) or "none dropped", chip("dropped " + str(len(v["dropped"])), "warn") if v["dropped"] else chip("pass", "good")])
    for k, v in g2.items():
        rows.append([f"G2 {k.replace('_', ' ')}", "full vector named above rand-dir at a dose where the model moves",
                     "; ".join(f"s = {SNAME.get(float(s), s)}: {fmt_ci(x)}" for s, x in v["by_dose"].items()),
                     chip("pass", "good") if v["pass"] else chip("fail", "bad")])
    for k, v in g11.items():
        rows.append([f"G11 {k.replace('_', ' ')}", f"natural activations identified among {10 if rehearse else 55} concepts ≥ 50%",
                     f"{pct(v['ID_natural'])} of {v['n']}" + (f" (missed: {', '.join(v['missed'][:6])}{'…' if len(v['missed']) > 6 else ''})" if v["missed"] else ""),
                     chip("pass", "good") if v["pass"] else chip("fail", "bad")])
    out.append('<h2 id="gates">Gates</h2>' + table(["Gate", "Rule", "Result", ""], rows))
    md += ["## Gates", "", "| Gate | Rule | Result | Pass |", "|---|---|---|---|"]
    md += [f"| {a} | {b} | {c} | {'yes' if 'good' in d else 'no / flag'} |" for a, b, c, d in rows] + [""]

    # ---------------------------------------------------------------------------------------------- contrasts
    crow, mrow = [], []
    for key, label, pred, preg, pup in PRIORS:
        c = C.get(key) or {}
        o = outcome(key, C)
        verdict_txt = "—" if o is None else ("supported" if o else "not supported")
        if key == "R1_probe_minus_NLA" and o is not None:
            verdict_txt = "yes (≥ 0)" if o else "no"
        crow.append([esc(label), str(c.get("n", "—")), fmt_ci(c), pred,
                     chip(verdict_txt, "good" if o else "bad") if o is not None else chip("—", "pend"),
                     r(preg, 2) if preg is not None else "—", r(pup, 2) if pup is not None else "—"])
        mrow.append(f"| {label} | {c.get('n', '—')} | {fmt_ci(c)} | {pred} | {verdict_txt} | "
                    f"{r(preg, 2) if preg is not None else '—'} | {r(pup, 2) if pup is not None else '—'} |")
    hrows = [[k, r(v["p"], 4), r(v["threshold"], 4), chip("rejects H0", "good") if v["rejects_null"] else chip("does not", "bad")]
             for k, v in hol.items()]
    reg = brier([(p, outcome(k, C)) for k, _, _, p, _ in PRIORS])
    upd = brier([(p, outcome(k, C)) for k, _, _, _, p in PRIORS])
    out.append(f"""<h2 id="contrasts">Pre-registered contrasts</h2>
<p>Per-concept statistics, averaged over concepts; 95% cluster-bootstrap intervals (10,000 draws). "Supported" means the
interval excludes 0 in the predicted direction. The P-family {{P11′ (NLA), P17}} is Holm-corrected on one-sided bootstrap p
at 0.025. Priors: the registered column is from commit df56646 (before any GPU run); the updated column was written after
the dev pilot and before this run (<code>005_full_run.md</code>).</p>
{table(["Contrast", "n", "Mean [95% CI]", "Predicted", "Verdict", "Registered prior", "Updated prior"], crow, num_cols=(1, 2, 5, 6))}
<h3>Holm over the P-family</h3>{table(["Contrast", "one-sided p", "Holm threshold", ""], hrows, num_cols=(1, 2))}
<p>Brier score of the registered priors: <b>{r(reg[0], 3) if reg[0] is not None else '—'}</b> over {reg[1]} resolved;
of the updated priors: <b>{r(upd[0], 3) if upd[0] is not None else '—'}</b> over {upd[1]} (always 0.5 scores 0.25).</p>""")
    md += ["## Pre-registered contrasts", "", "| Contrast | n | Mean [95% CI] | Predicted | Verdict | Registered prior | Updated prior |",
           "|---|---|---|---|---|---|---|"] + mrow + ["",
           "Holm (P-family): " + "; ".join(f"{k}: p = {r(v['p'], 4)} vs {r(v['threshold'], 4)} → "
                                          f"{'rejects H0' if v['rejects_null'] else 'does not'}" for k, v in hol.items()), "",
           f"Brier: registered priors {r(reg[0], 3) if reg[0] is not None else '—'} (n = {reg[1]}); updated priors "
           f"{r(upd[0], 3) if upd[0] is not None else '—'} (n = {upd[1]}).", ""]

    # --------------------------------------------------------------------------------------------- per contrast
    mc = T["model_curves"]
    null = T["model_null95"]

    def ser(conds, key="dR1p_median"):
        return [{"name": COND[c][0], "cls": COND[c][1], "dash": c == 7,
                 "ys": [mc.get(str(c), {}).get(str(s), {}).get(key) for s in GRID9]} for c in conds]
    s1 = ser([1, 2, 3, 4])
    f1 = figure(line_chart(s1, GRID9, [SNAME[s] for s in GRID9], "median ΔR1′ (nats)", band=[null[str(s)]["dR1p"] for s in GRID9],
                           aria="Median report log-prob change by dose and condition"), legend(s1, extra="random directions, 95th pct"),
                "Median over concepts of the change in the concept's log-prob at the open quote. Shaded: the random-direction null.")
    s3 = ser([2, 5, 6])
    f3 = figure(line_chart(s3, GRID9, [SNAME[s] for s in GRID9], "median ΔR1′ (nats)", band=[null[str(s)]["dR1p"] for s in GRID9],
                           aria="Natural-scale J and non-J parts"), legend(s3, extra="random directions, 95th pct"),
                "M3: the J and non-J parts at their natural size (they add up to the full vector).")
    out.append(f"""<h2 id="m1">M1 and M3: the J part vs the non-J part</h2>{f1}{f3}
<p>{N.get('M1', '')}</p><p>{N.get('M3', '')}</p>""")
    cur = C.get("M2_curves", {})
    s2 = [{"name": "J-atom clamp", "cls": "re", "ys": [cur.get(str(E), {}).get("extra_J", {}).get("mean") for E in ENDS]},
          {"name": "output-direction clamp", "cls": "pn", "ys": [cur.get(str(E), {}).get("extra_out", {}).get("mean") for E in ENDS]}]
    f2 = figure(line_chart(s2, ENDS, [f"L{E}" for E in ENDS], "extra reduction (nats)", xlabel="clamp end layer E",
                           aria="Clamp curves"), legend(s2), "Mean over evaluable concepts: random-token clamp minus concept clamp.")
    rows2 = [[f"L{E}", fmt_ci(cur.get(str(E), {}).get("extra_J")), fmt_ci(cur.get(str(E), {}).get("extra_out")),
              fmt_ci(cur.get(str(E), {}).get("D"))] for E in ENDS]
    out.append(f"""<h2 id="m2">M2: the clamp, with the output-direction control</h2>{f2}
{table(["End layer", "J clamp extra reduction", "Output clamp extra reduction", "D = J − output"], rows2, num_cols=(1, 2, 3))}
<p>Design clamp (to clean, L32–62), extra reduction: {fmt_ci(C.get('M2_design_clamp_10_vs_10r'))}. Evaluable concepts:
{C.get('M2', {}).get('evaluable', '—')}.</p><p>{N.get('M2', '')}</p>""")

    lt = T.get("likelihood_SM_median", {})
    s11 = [{"name": "model's report", "cls": "perp", "ys": [mc.get("4", {}).get(str(s), {}).get("SM_median") for s in GRID9]}]
    for k, cls in (("nla_L42", "j"), ("oracle_L40", "rj"), ("oracle_L44", "rp")):
        if k in lt:
            s11.append({"name": k.replace("_", " ").replace("nla", "NLA").replace("oracle", "Oracle"), "cls": cls,
                        "ys": [lt[k].get("4", {}).get(str(s)) for s in GRID9]})
    f11 = figure(line_chart(s11, GRID9, [SNAME[s] for s in GRID9], "median SM (nats)", band=[1.0] * len(GRID9),
                            aria="Specific rise of the non-J part: model vs readers"), legend(s11, extra="below 1 nat"),
                 "The non-J part: the planted concept's specific rise (own rise minus the median of the other words) at each one's forced report position.")
    on = C.get("P11prime_onsets", {})

    def dist(d):
        vals = [v for v in d.values() if v is not None]
        return " · ".join(f"{SNAME.get(GRID9[i], 'never') if i < 9 else 'never'}: {sum(1 for v in vals if v == i)}"
                          for i in range(10) if any(v == i for v in vals))
    orow = [["model's report", dist(on.get("model", {}))]] + [[k.replace("_", " "), dist(v)] for k, v in on.get("readers", {}).items()]
    out.append(f"""<h2 id="p11">P11′: does a reader detect the non-J part at a lower dose than the model reports it?</h2>{f11}
{table(["Who", "Onset (first dose with SM ≥ 1 nat): number of concepts"], orow)}
<p>NLA: {fmt_ci(C.get('P11prime_NLA'))} grid steps (negative = the reader is earlier); Oracle: {fmt_ci(C.get('P11prime_Oracle'))}.</p>
<p>{N.get('P11', '')}</p>""")
    nm = T.get("naming", {})
    nrows = []
    for k, d in nm.items():
        for cond in ("2", "3", "4", "7"):
            nrows.append([k.replace("_", " ").replace("nla", "NLA").replace("oracle", "Oracle"), COND[int(cond)][0]] +
                         [pct(d.get(cond, {}).get(str(s))) for s in TEXT_S])
    out.append(f"""<h2 id="p17">P17 and the readers' naming</h2>
{table(["Reader", "Planted", "s = 1/64", "s = 1/4", "s = 1"], nrows, num_cols=(2, 3, 4))}
<p>P17 (NLA − max Oracle, non-J part, naming): {fmt_ci(C.get('P17'))}. On the likelihood measure (secondary):
{fmt_ci(C.get('P17_likelihood_secondary'))} nats.</p><p>{N.get('P17', '')}</p>""")
    out.append(f"""<h2 id="r">R1 and R2</h2>
<p>R1, the probe (absolute criterion) minus the NLA's naming on the non-J part, direct reads at L42: {fmt_ci(C.get('R1_probe_minus_NLA'))}.
R2, the share of concepts where behaviour (B1, B2 or B4) moves at a dose where the report does not: {fmt_ci(C.get('R2_behaviour_without_report'))}.</p>
<p>{N.get('R', '')}</p>""")

    # --------------------------------------------------------------------------------------------- secondary
    sm = ser([1, 2, 3, 4, 7], "SM_median")
    fsm = figure(line_chart(sm, GRID9, [SNAME[s] for s in GRID9], "median SM (nats)", aria="Specificity margin by dose"),
                 legend(sm), "A18: the concept's own rise minus the median rise of the other words in W.")
    idr = ser([1, 2, 3, 4, 7], "ID_rate")
    fid = figure(line_chart(idr, GRID9, [SNAME[s] for s in GRID9], "share identified", ylim=(0, 1), aria="Identification rate"),
                 legend(idr), "Share of concepts whose own rise is the largest among all concepts in the choice set.")
    r2t = T.get("R2", {})
    r2rows = [[COND[int(c)][0]] + [pct(d.get(str(s))) for s in TEXT_S] for c, d in r2t.items()]
    sens = T.get("sensitivity", {})
    srows = [[k, pct((T.get("sensitivity_J_share_median") or {}).get(k), 1), fmt_ci(v.get("M1_R1p"))] for k, v in sens.items()]
    tp = T.get("token_probe_detection", {})
    trows = []
    for kind in ("direct_cq", "carried_co", "injected_cq"):
        for cond in ("2", "4"):
            for s in (1 / 64, 1 / 4, 1.0):
                d = tp.get(f"L42|{kind}|{cond}|{s}")
                if d:
                    trows.append([kind.replace("_", " "), COND[int(cond)][0], SNAME[s]] + [pct(d.get(x)) for x in ("J_NP", "J_CB", "logit", "R_CB", "probe_abs")])
    out.append(f"""<h2 id="secondary">Secondary tables</h2>{fsm}{fid}
<h3>Sampled naming by the model (R2, secondary under A1)</h3>{table(["Planted", "s = 1/64", "s = 1/4", "s = 1"], r2rows, num_cols=(1, 2, 3))}
<h3>Sensitivity: M1 on R1′ under other J-space constructions</h3>{table(["Construction", "J share (median)", "M1 on R1′"], srows, num_cols=(1, 2))}
<h3>Token readers and the probe at L42 (share of concepts detected: rank ≤ 10; probe decision > 0)</h3>
{table(["Read kind", "Planted", "Dose", "J_NP", "J_CB", "logit lens", "R_CB", "probe"], trows, num_cols=(3, 4, 5, 6, 7))}""")

    # ------------------------------------------------------------------------------------------- raw readouts
    ex = S.get("examples", {})
    erows = [[k.replace("_", " "), e["concept"], COND[e["cond"]][0], "<br>".join(f"<code>{esc(t[:200])}</code>" for t in e["samples"][:2])]
             for k, lst in ex.items() for e in lst]
    exr2 = S.get("examples_R2", [])
    rrows = [[e["concept"], COND[e["cond"]][0], "<br>".join(f"<code>{esc(t)}</code>" for t in e["texts"][:3])] for e in exr2]
    out.append(f"""<h2 id="reads">Raw readouts (s = 1/4)</h2>{table(["Reader", "Concept", "Planted", "Two samples"], erows)}
<h3>The model's own continuation after the prefill</h3>{table(["Concept", "Planted", "Samples"], rrows)}""")
    tm = T.get("timings") or {}
    trow = [[k, r(v["seconds"] / 60, 1), str(v.get("n", ""))] for k, v in tm.items()]
    out.append(f"""<h2 id="machinery">Machinery</h2><p>Wall time {r(T.get('wall_minutes'), 1)} min; peak GPU memory
{r((T.get('memory') or {}).get('phaseA_peak_GB'), 1)} GB (phase A), {r((T.get('memory') or {}).get('phaseB_peak_GB'), 1)} GB (phase B).
Likelihood scoring, float32 vs the readers' own logits, max |difference|: {", ".join(f"{k} {r(v, 3)}" for k, v in (T.get('scoring_check_maxabs') or {}).items())}.</p>
{table(["Component", "Minutes", "Items"], trow, num_cols=(1, 2))}
<p>{N.get('caveats', '')}</p>""")
    md += ["## Notes", ""] + [f"**{k}.** {v}" for k, v in N.items() if k not in ("short_html", "short_md")] + [""]
    fonts = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&'
             'family=IBM+Plex+Sans+Condensed:wght@600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">')
    toc = [("short", "The short version"), ("gates", "Gates"), ("contrasts", "Pre-registered contrasts"), ("m1", "M1 and M3"),
           ("m2", "M2 clamp"), ("p11", "P11′"), ("p17", "P17 and naming"), ("r", "R1 and R2"), ("secondary", "Secondary"),
           ("reads", "Raw readouts"), ("machinery", "Machinery")]
    nav = '<nav class="toc" aria-label="Contents">' + "".join(f'<a href="#{a}">{esc(b)}</a>' for a, b in toc) + "</nav>"
    page = (f"<title>Planted Concepts Full Run</title>\n{fonts}\n<style>{CSS}</style>\n"
            f'<div class="wrap">{nav}<main>{"".join(out)}<footer>Generated by <code>scripts/005_planted_concepts/make_full_report.py</code> '
            "from <code>results/005_planted_concepts/analysis/</code>. Branch <code>005-amended-full-run</code>.</footer></main></div>")
    return page, "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--stage", default="run", choices=("run", "rehearse"))
    a = ap.parse_args()
    S = load(RESA / ("full_summary.json" if a.stage == "run" else "full_summary_rehearse.json"))
    N = (load(RESA / "full_notes.json") or {}) if a.stage == "run" else {}
    page, md = build(S, N)
    Path(a.out).write_text(page)
    if a.stage == "run":
        (ROOT / "docs/reports/005_full_run.md").write_text(md)
    print("wrote", a.out, len(page) // 1024, "KB")


if __name__ == "__main__":
    main()
