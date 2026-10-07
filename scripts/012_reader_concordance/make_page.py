"""012 results page (artifact) built ONLY from committed JSON in results/012_reader_concordance/. Inline SVG charts,
numbers rounded half-up from full precision.  python make_page.py  ->  results/012_reader_concordance/page/index.html"""
import html
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/012_reader_concordance"
A = json.loads((R / "analysis/stageA_summary.json").read_text())["runs"]
D = json.loads((R / "analysis/stageD_analysis.json").read_text())
X = json.loads((R / "analysis/stageD_answers.json").read_text())
G = json.loads((R / "analysis/stageB_gates.json").read_text())
Q = [json.loads(l) for l in open(R / "qualitative/stageC_readouts_samples.jsonl")]
JA = json.loads((R / "analysis/stageJ_analysis.json").read_text()) if (R / "analysis/stageJ_analysis.json").exists() else None
IA = json.loads((R / "analysis/stageI_analysis.json").read_text()) if (R / "analysis/stageI_analysis.json").exists() else None
EA = json.loads((R / "analysis/stageE0_ar_profile.json").read_text()) if (R / "analysis/stageE0_ar_profile.json").exists() else None
HA = json.loads((R / "analysis/stageH_summary.json").read_text()) if (R / "analysis/stageH_summary.json").exists() else None
GA = json.loads((R / "analysis/stageG_analysis.json").read_text()) if (R / "analysis/stageG_analysis.json").exists() else None
FA = json.loads((R / "analysis/stageF_analysis.json").read_text()) if (R / "analysis/stageF_analysis.json").exists() else None
LA = json.loads((R / "analysis/stageL_analysis.json").read_text())
R13 = ROOT / "results/013_nla_hallucination"
HD = json.loads((R13 / "analysis/hallucination_all.json").read_text())
RW = json.loads((R13 / "analysis/rewrite_all.json").read_text())
TH = json.loads((R13 / "analysis/testhalf.json").read_text())
DR = json.loads((R13 / "analysis/derisk.json").read_text())
EX = [json.loads(l) for l in open(R13 / "qualitative/examples.jsonl")]


def h(x, d=2):
    return "–" if x is None else str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)).replace("-", "−")


E = html.escape
W = 640


def axis_x(x0, x1, y, ticks, sx, fmt=lambda t: h(t, 1)):
    s = f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" class="ax"/>'
    for t in ticks:
        s += (f'<line x1="{sx(t):.1f}" y1="{y}" x2="{sx(t):.1f}" y2="{y + 4}" class="ax"/>'
              f'<text x="{sx(t):.1f}" y="{y + 16}" class="tk" text-anchor="middle">{fmt(t)}</text>')
    return s


def chart_stageA():
    rows = [("J swap L24–47", "j_swap_L24-47"), ("J swap L24–39", "j_swap_L24-39"), ("J swap L36–47", "j_swap_L36-47"),
            ("J swap L40–48", "j_swap_L40-48"), ("J swap L24–35", "j_swap_L24-35"), ("logit swap L24–47", "logit_swap_L24-47"),
            ("broadcast J L24–47", "j_bcast_L24-47")]
    L, top, rh = 170, 18, 30
    sx = lambda v: L + v * (W - L - 40)
    s = "".join(f'<line x1="{sx(t):.1f}" y1="{top - 6}" x2="{sx(t):.1f}" y2="{top + rh * len(rows)}" class="grid"/>' for t in (0.2, 0.4, 0.6, 0.8))
    for i, (lab, k) in enumerate(rows):
        r = A[k]
        y = top + i * rh
        p, (lo, hi), rc = r["success"]["rate"], r["success"]["wilson95"], r["random_control"]["rate"]
        s += f'<text x="{L - 10}" y="{y + 14}" class="lb" text-anchor="end">{lab}</text>'
        s += (f'<rect x="{L}" y="{y + 4}" width="{max(sx(p) - L, 1):.1f}" height="14" rx="3" class="s1">'
              f'<title>{lab}: {h(p)} [{h(lo)}, {h(hi)}] (n {r["success"]["n"]}); random control {h(rc)}</title></rect>')
        s += f'<line x1="{sx(lo):.1f}" y1="{y + 11}" x2="{sx(hi):.1f}" y2="{y + 11}" class="ci"/>'
        s += f'<circle cx="{sx(rc):.1f}" cy="{y + 11}" r="4.5" class="rnd"><title>random control {h(rc)}</title></circle>'
        s += f'<text x="{min(sx(hi) + 8, W - 34):.1f}" y="{y + 15}" class="val">{h(p)}</text>'
    yb = top + rh * len(rows) + 4
    s += axis_x(L, W - 40, yb, [0, 0.2, 0.4, 0.6, 0.8, 1.0], sx)
    return f'<svg viewBox="0 0 {W} {yb + 26}" role="img" aria-label="Stage A swap success by band">{s}</svg>'


def chart_scatter(cell, title):
    pts = D["cells"][cell]["points"]
    L, T, w, hgt = 46, 12, 280, 210
    xs = [p["dm"] for p in pts]
    ys = [p["dJ"] for p in pts]
    x0, x1 = min(min(xs), 0) - 1, max(xs) + 1
    y0, y1 = min(min(ys), 0) - 1, max(ys) + 1
    sx = lambda v: L + (v - x0) / (x1 - x0) * w
    sy = lambda v: T + hgt - (v - y0) / (y1 - y0) * hgt
    s = f'<line x1="{sx(0):.1f}" y1="{T}" x2="{sx(0):.1f}" y2="{T + hgt}" class="grid"/><line x1="{L}" y1="{sy(0):.1f}" x2="{L + w}" y2="{sy(0):.1f}" class="grid"/>'
    for p in sorted(pts, key=lambda p: p["success"]):
        cls = "rnd" if p["direction"] == "toward_src" else ("s1" if p["success"] else "s2")
        s += (f'<circle cx="{sx(p["dm"]):.1f}" cy="{sy(p["dJ"]):.1f}" r="4" class="{cls} ring"><title>{E(p["name"])}: Δm {h(p["dm"])}, d_r {h(p["dJ"])}, '
              f'{"works" if p["success"] else "fails"}, {p["direction"]}</title></circle>')
    s += f'<rect x="{L}" y="{T}" width="{w}" height="{hgt}" class="frame"/>'
    step = 10 if x1 - x0 > 25 else 5
    s += axis_x(L, L + w, T + hgt, [t for t in range(int(x0 // step * step), int(x1) + 1, step) if x0 <= t <= x1], sx, lambda t: str(t).replace("-", "−"))
    ystep = 10 if y1 - y0 > 25 else 5
    for t in range(int(y0 // ystep * ystep), int(y1) + 1, ystep):
        if y0 <= t <= y1:
            s += f'<text x="{L - 6}" y="{sy(t) + 4:.1f}" class="tk" text-anchor="end">{str(t).replace("-", "−")}</text>'
    s += f'<text x="{L + w / 2}" y="{T + hgt + 32}" class="tk" text-anchor="middle">model Δm (nats, float32)</text>'
    s += f'<text x="12" y="{T + hgt / 2}" class="tk" text-anchor="middle" transform="rotate(-90 12 {T + hgt / 2})">reader d_r (own nats)</text>'
    return f'<figure class="sm"><figcaption>{title}</figcaption><svg viewBox="0 0 340 {T + hgt + 40}" role="img" aria-label="{title}">{s}</svg></figure>'


def chart_auc():
    order = ["oracle|L40|L24-35", "oracle|L44|L24-35", "nla|L42|L24-35", "nla_offtrain|L44|L24-35", "J_NP|L40|L24-35",
             "J_CB|L40|L24-35", "logit|L40|L24-35", "oracle|L52|L24-47", "oracle|L56|L24-47", "J_NP|L56|L24-47", "logit|L56|L24-47"]
    names = {"oracle": "Oracle", "nla": "NLA", "nla_offtrain": "NLA (off-training)", "J_NP": "J-lens NP", "J_CB": "J-lens CB", "logit": "logit lens"}
    L, top, rh = 200, 26, 24
    sx = lambda v: L + (v - 0.5) / 0.5 * (W - L - 30)
    s = "".join(f'<line x1="{sx(t):.1f}" y1="{top - 8}" x2="{sx(t):.1f}" y2="{top + rh * len(order)}" class="grid"/>' for t in (0.6, 0.7, 0.8, 0.9, 1.0))
    for b, y in (("L24-35", top), ("L24-47", top + rh * 7)):
        m = X["model_output"][b]["auc_d_model_bridge"]
        s += f'<line x1="{sx(m):.1f}" y1="{y - 6}" x2="{sx(m):.1f}" y2="{y + rh * (7 if b == "L24-35" else 4) - 6}" class="ref"><title>model own output (bridge) AUC {h(m)}</title></line>'
    for i, k in enumerate(order):
        c = X["cells"][k]
        r, lay, band = k.split("|")
        y = top + i * rh
        s += f'<text x="{L - 10}" y="{y + 4}" class="lb" text-anchor="end">{names[r]} {lay} · {band.replace("-", "–")}</text>'
        s += f'<line x1="{sx(min(c["auc_bridge"], c["auc_answer"])):.1f}" y1="{y}" x2="{sx(max(c["auc_bridge"], c["auc_answer"])):.1f}" y2="{y}" class="ci"/>'
        s += f'<circle cx="{sx(c["auc_answer"]):.1f}" cy="{y}" r="5" class="s2 ring"><title>answer shift AUC {h(c["auc_answer"])}</title></circle>'
        s += f'<circle cx="{sx(c["auc_bridge"]):.1f}" cy="{y}" r="5" class="s1 ring"><title>bridge shift AUC {h(c["auc_bridge"])}</title></circle>'
    yb = top + rh * len(order) - 8
    s += axis_x(L, W - 30, yb, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], sx)
    return f'<svg viewBox="0 0 {W} {yb + 26}" role="img" aria-label="AUC of reader shifts for predicting a working swap">{s}</svg>'


def chart_partial():
    order = ["oracle|L40|L24-35", "oracle|L44|L24-35", "nla|L42|L24-35", "J_NP|L40|L24-35", "logit|L40|L24-35",
             "oracle|L56|L24-47", "J_NP|L56|L24-47", "logit|L56|L24-47"]
    names = {"oracle": "Oracle", "nla": "NLA", "J_NP": "J-lens NP", "logit": "logit lens"}
    L, top, rh = 200, 16, 24
    sx = lambda v: L + (v + 0.4) / 1.2 * (W - L - 30)
    s = f'<line x1="{sx(0):.1f}" y1="{top - 8}" x2="{sx(0):.1f}" y2="{top + rh * len(order)}" class="ref"/>'
    for i, k in enumerate(order):
        c = X["cells"][k]
        r, lay, band = k.split("|")
        y = top + i * rh
        v, ci = c["partial_bridge_dm_given_model_output_size"], c["partial_bridge_dm_given_model_output_size_ci95"]
        s += f'<text x="{L - 10}" y="{y + 4}" class="lb" text-anchor="end">{names[r]} {lay} · {band.replace("-", "–")}</text>'
        s += f'<line x1="{sx(ci[0]):.1f}" y1="{y}" x2="{sx(ci[1]):.1f}" y2="{y}" class="ci"/>'
        s += f'<circle cx="{sx(v):.1f}" cy="{y}" r="5" class="s1 ring"><title>partial ρ {h(v)} [{h(ci[0])}, {h(ci[1])}]</title></circle>'
    yb = top + rh * len(order) - 8
    s += axis_x(L, W - 30, yb, [-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8], sx)
    return f'<svg viewBox="0 0 {W} {yb + 26}" role="img" aria-label="Partial correlation beyond the model output">{s}</svg>'


def chart_f1():
    arms = [("all positions", "J_L24-35", "rand_L24-35"), ("all but the read token", "J_allbutread_L24-35", "rand_allbutread_L24-35"),
            ("read token only", "J_readonly_L24-35", "rand_readonly_L24-35")]
    L, top, rh = 170, 14, 30
    sx = lambda v: L + v * (W - L - 40)
    s = "".join(f'<line x1="{sx(t):.1f}" y1="{top - 6}" x2="{sx(t):.1f}" y2="{top + rh * 3}" class="grid"/>' for t in (0.2, 0.4, 0.6))
    for i, (lab, j, r) in enumerate(arms):
        a, b = FA["F1"]["arms"][j], FA["F1"]["arms"][r]
        p, q = a["flips"] / a["n"], b["flips"] / b["n"]
        y = top + i * rh
        s += f'<text x="{L - 10}" y="{y + 14}" class="lb" text-anchor="end">{lab}</text>'
        s += f'<rect x="{L}" y="{y + 4}" width="{max(sx(p) - L, 1):.1f}" height="14" rx="3" class="s1"><title>{lab}: {a["flips"]}/{a["n"]} answers changed</title></rect>'
        s += f'<circle cx="{sx(q):.1f}" cy="{y + 11}" r="4.5" class="rnd"><title>random control {b["flips"]}/{b["n"]}</title></circle>'
        s += f'<text x="{sx(p) + 8:.1f}" y="{y + 15}" class="val">{a["flips"]}/{a["n"]}</text>'
    yb = top + rh * 3 + 4
    s += axis_x(L, W - 40, yb, [0, 0.2, 0.4, 0.6, 0.8, 1.0], sx)
    return f'<svg viewBox="0 0 {W} {yb + 26}" role="img" aria-label="Answer changes by edited positions">{s}</svg>'


def chart_f_auc():
    rows = [("Oracle L40", "oracle|L40"), ("Oracle L44", "oracle|L44"), ("NLA L42", "nla|L42"), ("J-lens NP L40", "J_NP|L40"),
            ("J-lens CB L40", "J_CB|L40"), ("logit lens L40", "logit|L40")]
    L, top, rh = 160, 24, 26
    sx = lambda v: L + (v - 0.3) / 0.7 * (W - L - 30)
    s = "".join(f'<line x1="{sx(t):.1f}" y1="{top - 10}" x2="{sx(t):.1f}" y2="{top + rh * len(rows) - 8}" class="grid"/>' for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    me = FA["F2"]["model_early_output_auc"]["bridge"]
    s += f'<line x1="{sx(me):.1f}" y1="{top - 10}" x2="{sx(me):.1f}" y2="{top + rh * len(rows) - 8}" class="ref"><title>model own early-position output AUC {h(me)}</title></line>'
    for i, (lab, k) in enumerate(rows):
        y = top + i * rh
        f1 = FA["F1"]["readers"][f"{k}|J_allbutread_L24-35"]["auc_bridge"]
        f2 = FA["F2"][k]
        s += f'<text x="{L - 10}" y="{y + 4}" class="lb" text-anchor="end">{lab}</text>'
        if f2["auc_ci95"]:
            s += f'<line x1="{sx(f2["auc_ci95"][0]):.1f}" y1="{y}" x2="{sx(f2["auc_ci95"][1]):.1f}" y2="{y}" class="ci"/>'
        s += f'<circle cx="{sx(f2["auc"]):.1f}" cy="{y}" r="5" class="s2 ring"><title>early written positions: AUC {h(f2["auc"])}</title></circle>'
        s += f'<circle cx="{sx(f1):.1f}" cy="{y}" r="5" class="s1 ring"><title>unwritten answer token (all-but-read edit): AUC {h(f1)}</title></circle>'
    yb = top + rh * len(rows) - 6
    s += axis_x(L, W - 30, yb, [0.3, 0.5, 0.7, 0.9], sx)
    return f'<svg viewBox="0 0 {W} {yb + 26}" role="img" aria-label="Reader AUC at the answer token vs early positions">{s}</svg>'


def chart_e0():
    Ls = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
    L0, T, w, hg = 46, 12, W - 70, 200
    sx = lambda i: L0 + i * w / (len(Ls) - 1)
    sy = lambda v: T + hg - v / 0.7 * hg
    s = "".join(f'<line x1="{L0}" y1="{sy(v):.1f}" x2="{L0 + w}" y2="{sy(v):.1f}" class="grid"/><text x="{L0 - 6}" y="{sy(v) + 4:.1f}" class="tk" text-anchor="end">{h(v, 1)}</text>' for v in (0.0, 0.2, 0.4, 0.6))
    series = [("pub_fve", lambda L: EA["layers"][str(L)]["published"]["fve"], "ref2"), ("our_fve", lambda L: EA["layers"][str(L)]["white"]["fve_scaled"], "s1"),
              ("pub_jfve", lambda L: EA["layers"][str(L)]["published"]["jfve"], "ref2"), ("our_jfve", lambda L: EA["jfve_scaled"]["layers"][str(L)]["b_mu_removed"]["fve_scaled"], "s2")]
    for name, f, cls in series:
        pts = " ".join(f"{sx(i):.1f},{sy(f(L)):.1f}" for i, L in enumerate(Ls))
        dash = ' stroke-dasharray="4 3"' if name.startswith("pub") else ""
        col = "var(--muted)" if name.startswith("pub") else ("var(--s1)" if "jfve" not in name else "var(--s2)")
        s += f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"{dash}/>'
        if not name.startswith("pub"):
            s += "".join(f'<circle cx="{sx(i):.1f}" cy="{sy(f(L)):.1f}" r="4" class="{cls} ring"><title>L{L}: ours {h(f(L))}</title></circle>' for i, L in enumerate(Ls))
    s += "".join(f'<text x="{sx(i):.1f}" y="{T + hg + 16}" class="tk" text-anchor="middle">L{L}</text>' for i, L in enumerate(Ls))
    return f'<svg viewBox="0 0 {W} {T + hg + 26}" role="img" aria-label="Our AR loader vs the published per-layer profile">{s}</svg>'


def chart_stageL():
    """AUC for hold success at H2 by layer: Stage J (L40/44) and Stage L (L48-60); model's own H2 output as reference."""
    series = [("Oracle", "oracle", "var(--s1)", ""), ("logit lens", "logit", "var(--s2)", ""),
              ("J_NP", "J_NP", "var(--fg2)", ""), ("J_CB", "J_CB", "var(--fg2)", "4 3")]
    x0, x1, y0, y1 = 56, 610, 16, 196
    Ls = [40, 44, 48, 52, 56, 60]
    sx = lambda L: x0 + (L - 40) / 20 * (x1 - x0)
    sy = lambda v: y1 - (v - 0.3) / 0.6 * (y1 - y0)
    s = ""
    for v in (0.3, 0.5, 0.7, 0.9):
        s += f'<line x1="{x0}" y1="{sy(v):.1f}" x2="{x1}" y2="{sy(v):.1f}" class="grid"/><text x="{x0 - 8}" y="{sy(v) + 4:.1f}" class="tk" text-anchor="end">{h(v, 1)}</text>'
    mo = JA["J2_model_H2_bridge_auc"]
    s += f'<line x1="{x0}" y1="{sy(mo):.1f}" x2="{x1}" y2="{sy(mo):.1f}" class="ref"><title>model\'s own next-token output at H2: AUC {h(mo)}</title></line>'
    s += f'<text x="{x1}" y="{sy(mo) - 6:.1f}" class="tk" text-anchor="end">model\'s own output {h(mo)}</text>'
    for name, rd, col, dash in series:
        pts = []
        for L in Ls:
            src = JA["readers"] if L in (40, 44) else LA["readers"]
            k = f"{rd}|L{L}|H2"
            if k in src:
                pts.append((L, src[k]["auc"]))
        s += f'<polyline points="{" ".join(f"{sx(L):.1f},{sy(v):.1f}" for L, v in pts)}" fill="none" stroke="{col}" stroke-width="2" stroke-dasharray="{dash}"/>'
        for L, v in pts:
            s += f'<circle cx="{sx(L):.1f}" cy="{sy(v):.1f}" r="3.5" fill="{col}" class="ring"><title>{name} L{L}: AUC {h(v)}</title></circle>'
    s += axis_x(x0, x1, y1 + 4, Ls, sx, fmt=lambda L: f"L{L}")
    return f'<svg viewBox="0 0 {W} {y1 + 30}" role="img" aria-label="Reader AUC at the carrier end by layer">{s}</svg>'


def chart_deletion():
    """Δ = FVE(random deletion) − FVE(cleaned), points with 95% cluster-bootstrap CIs; ±2-point band."""
    rows = [("all false text (primary)", HD["primary_delta_R1_minus_C1"], "s1"),
            ("all false text, sentence control", HD["A1_sentence_delta_RS1_minus_C1"], "s1"),
            ("invented context only", HD["A1_kind_invented_context"]["delta_clause"], "s2"),
            ("invented context, sentence control", HD["A1_kind_invented_context"]["delta_sentence"], "s2"),
            ("misquotes only", HD["A1_kind_misquote"]["delta_clause"], "s1"),
            ("misquotes, sentence control", HD["A1_kind_misquote"]["delta_sentence"], "s1")]
    x0, x1, lo, hi = 250, 610, -10, 45
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    rh = 28
    s = f'<rect x="{sx(-2):.1f}" y="6" width="{sx(2) - sx(-2):.1f}" height="{rh * len(rows)}" fill="var(--grid)" opacity="0.6"><title>±2 FVE points: no difference band</title></rect>'
    s += f'<line x1="{sx(0):.1f}" y1="6" x2="{sx(0):.1f}" y2="{6 + rh * len(rows)}" class="ref"/>'
    for i, (lab, st, cls) in enumerate(rows):
        y = 6 + rh * i + rh / 2
        s += f'<text x="{x0 - 10}" y="{y + 4:.1f}" class="lb" text-anchor="end">{lab}</text>'
        s += f'<line x1="{sx(st["ci95"][0]):.1f}" y1="{y:.1f}" x2="{sx(st["ci95"][1]):.1f}" y2="{y:.1f}" class="ci"/>'
        s += f'<circle cx="{sx(st["mean"]):.1f}" cy="{y:.1f}" r="5" class="{cls} ring"><title>{lab}: {h(st["mean"], 1)} [{h(st["ci95"][0], 1)}, {h(st["ci95"][1], 1)}], n {st["n"]}</title></circle>'
    s += axis_x(x0, x1, 10 + rh * len(rows), [-10, 0, 10, 20, 30, 40], sx, fmt=lambda v: h(v, 0))
    return f'<svg viewBox="0 0 {W} {rh * len(rows) + 38}" role="img" aria-label="Reconstruction cost of deleting false text vs random text">{s}</svg>'


def mark_spans(text, spans):
    """Explanation with judge-A contradicted spans highlighted by kind (whitespace-flexible match)."""
    import re as _re
    hits = []
    for sp in spans:
        toks = sp["text"].split()
        m = _re.search(r"\s+".join(_re.escape(x) for x in toks), text) if toks else None
        if m:
            hits.append((m.start(), m.end(), sp["kind"]))
    hits.sort()
    out, last = "", 0
    for a, b, k in hits:
        if a < last:
            continue
        out += E(text[last:a]) + f'<mark class="k-{E(k)}" title="{E(k.replace("_", " "))}">{E(text[a:b])}</mark>'
        last = b
    return out + E(text[last:])


def examples13():
    out = []
    for x in EX:
        fv = x["fve"]
        out.append(f'<details><summary>{E(x["eid"])} · {E(x["selection"])}</summary>'
                   f'<p class="rh">Prompt: {E(x["prompt"])} · model continues: {E(x["continuation"].strip())}</p>'
                   f'<div class="ros"><div class="ro"><div class="rh">NLA explanation (FVE {h(fv["O"], 1)}); false spans marked</div><p class="ex">{mark_spans(x["explanation"], x["contradicted_spans"])}</p></div>'
                   f'<div class="ro"><div class="rh">Factual rewrite (FVE {h(fv["F"], 1)})</div><p class="ex">{E(x["F"])}</p></div>'
                   f'<div class="ro"><div class="rh">Paraphrase control (FVE {h(fv["P"], 1)})</div><p class="ex">{E(x["P"])}</p></div></div></details>')
    return "".join(out)


def rewrite_table():
    fm, jm = RW["fve_mean"], RW["share_J_mean"]
    rows = [("real activation", None, jm["gold"]), ("NLA explanation as written", fm["O"], jm["O"]),
            ("factual rewrite", fm["F"], jm["F"]), ("paraphrase control", fm["P"], jm["P"])]
    body = "".join(f"<tr><th>{a}</th><td>{h(b, 1) if b is not None else '–'}</td><td>{h(c, 3)}</td></tr>" for a, b, c in rows)
    return f'<div class="wrap"><table><thead><tr><th></th><th>FVE (%)</th><th>J share of reconstruction</th></tr></thead><tbody>{body}</tbody></table></div>'


def naming_table():
    rows = []
    for rd, lay in (("oracle", "L40"), ("oracle", "L44"), ("nla", "L42")):
        def cnt(grp, cat):
            k = f"{rd}|{lay}|L24-35|{grp}"
            v = D["naming"].get(k, {})
            return v.get(cat, 0) + v.get("both", 0) if cat != "both" else v.get("both", 0), v.get("_n", 0)
        cells = []
        for grp in ("clean", "J|tt_success", "J|tt_fail"):
            (sv, n), (tv, _) = cnt(grp, "src-only"), cnt(grp, "tgt-only")
            cells.append(f"<td>{sv}/{n}</td><td>{tv}/{n}</td>")
        rows.append(f"<tr><th>{'Oracle' if rd == 'oracle' else 'NLA'} {lay}</th>{''.join(cells)}</tr>")
    return ("<table><thead><tr><th>reader</th><th>clean: src</th><th>clean: tgt</th><th>working swap: src</th><th>working swap: tgt</th>"
            "<th>failed swap: src</th><th>failed swap: tgt</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def readouts():
    by = {}
    for q in Q:
        by.setdefault((q["group"], q["item"]), []).append(q)
    out = []
    for (grp, item), qs in sorted(by.items())[:12]:
        if not grp.endswith("success"):
            continue
        cells = []
        for rd, lay in (("oracle", "40"), ("nla", "42")):
            c = next((q for q in qs if q["reader"] == rd and str(q["layer"]) == lay and q["state"] == "clean"), None)
            j = next((q for q in qs if q["reader"] == rd and str(q["layer"]) == lay and q["state"].startswith("J")), None)
            if c and j:
                cells.append(f'<div class="ro"><div class="rh">{"Oracle L40" if rd == "oracle" else "NLA L42"}</div>'
                             f'<p><b>clean</b> {E(c["samples"][0][:220])}</p><p><b>after swap</b> {E(j["samples"][0][:220])}</p></div>')
        if cells:
            out.append(f'<details><summary>{E(item)}: {E(qs[0]["src"])} → {E(qs[0]["tgt"])}</summary><div class="ros">{"".join(cells)}</div></details>')
    return "".join(out[:6])


oa = D["oracle_vs_nla_auc"]
dirs = {b: {"tt": sum(1 for p in D["cells"][f"oracle|L{'44' if b == 'L24-35' else '56'}|{b}"]["points"] if p["direction"] == "toward_tgt"),
            "ts": sum(1 for p in D["cells"][f"oracle|L{'44' if b == 'L24-35' else '56'}|{b}"]["points"] if p["direction"] == "toward_src"),
            "ts_ok": sum(1 for p in D["cells"][f"oracle|L{'44' if b == 'L24-35' else '56'}|{b}"]["points"] if p["direction"] == "toward_src" and p["success"]),
            "tt_ok": sum(1 for p in D["cells"][f"oracle|L{'44' if b == 'L24-35' else '56'}|{b}"]["points"] if p["direction"] == "toward_tgt" and p["success"])}
        for b in ("L24-35", "L24-47")}
mo = X["model_output"]
page = f"""<title>Readers on Swapped States</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400&family=Newsreader:opsz,wght@6..72,500&display=swap">
<style>
/* layout: single reading column, figures full column width, small multiples wrap */
:root {{ --bg:#f9f9f7; --surface:#fcfcfb; --fg:#0b0b0b; --fg2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --s1:#2a78d6; --s2:#eb6834; --rnd:#898781; --accent:#256abf;
  --display:"Newsreader", Georgia, serif; --body:"IBM Plex Sans", system-ui, sans-serif; --mono:"IBM Plex Mono", ui-monospace, monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0d0d0d; --surface:#1a1a19; --fg:#ffffff; --fg2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835; --s1:#3987e5; --s2:#d95926; --rnd:#898781; --accent:#86b6ef; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#0d0d0d; --surface:#1a1a19; --fg:#ffffff; --fg2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835; --s1:#3987e5; --s2:#d95926; --rnd:#898781; --accent:#86b6ef; color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.55 var(--body); }}
main {{ max-width:760px; margin:0 auto; padding-inline:16px; padding-block:28px 64px; display:grid; gap:22px; }}
h1 {{ font:500 2rem/1.15 var(--display); margin:0; text-wrap:balance; }}
h2 {{ font:500 1.35rem/1.25 var(--display); margin:10px 0 0; text-wrap:balance; }}
p {{ margin:0; max-width:68ch; color:var(--fg); }} .sub {{ color:var(--fg2); }}
.eyebrow {{ font:12px var(--mono); letter-spacing:.06em; text-transform:uppercase; color:var(--muted); }}
.lede {{ font-size:1.05rem; }}
figure {{ margin:0; background:var(--surface); border:1px solid var(--grid); border-radius:6px; padding:12px; min-width:0; }}
figcaption {{ font-size:13px; color:var(--fg2); margin-bottom:6px; }}
svg {{ width:100%; height:auto; display:block; }}
.ax {{ stroke:var(--axis); }} .grid {{ stroke:var(--grid); }} .ref {{ stroke:var(--muted); stroke-dasharray:3 3; }}
.tk {{ fill:var(--muted); font:11px var(--mono); }} .lb {{ fill:var(--fg2); font:12px var(--body); }} .val {{ fill:var(--fg); font:12px var(--mono); }}
.s1 {{ fill:var(--s1); }} .s2 {{ fill:var(--s2); }} .rnd {{ fill:var(--surface); stroke:var(--rnd); stroke-width:2; }}
.ring {{ stroke:var(--surface); stroke-width:1.5; }} .ci {{ stroke:var(--fg2); stroke-width:1.5; }} .frame {{ fill:none; stroke:var(--grid); }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; font-size:12px; color:var(--fg2); }} .legend i {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:-1px; }}
.sms {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:12px; }}
.wrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; font:13px var(--mono); font-variant-numeric:tabular-nums; width:100%; }}
th, td {{ border-bottom:1px solid var(--grid); padding:6px 8px; text-align:left; }} th {{ color:var(--fg2); font-weight:400; }}
.kv {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; }}
.kv div {{ border-top:2px solid var(--grid); padding-top:6px; }} .kv b {{ display:block; font:500 1.4rem var(--display); }} .kv span {{ font-size:12px; color:var(--fg2); }}
details {{ border:1px solid var(--grid); border-radius:6px; padding:8px 12px; background:var(--surface); }} summary {{ cursor:pointer; font:13px var(--mono); }}
.ros {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:10px; margin-top:8px; }} .ro p {{ font:12px/1.45 var(--mono); color:var(--fg2); margin-top:4px; }} .rh {{ font-size:12px; color:var(--muted); }}
.note {{ border-left:3px solid var(--accent); padding-left:12px; }}
ul {{ margin:0; padding-left:20px; max-width:68ch; }} li {{ margin:4px 0; }}
.part {{ border-top:1px solid var(--axis); padding-top:18px; margin-top:12px; }}
.ex {{ white-space:pre-wrap; }}
mark {{ color:inherit; border-radius:2px; padding:0 1px; }}
mark.k-invented_context {{ background:color-mix(in srgb, var(--s2) 28%, transparent); }}
mark.k-misquote {{ background:color-mix(in srgb, var(--s1) 24%, transparent); }}
mark.k-wrong_entity, mark.k-false_fact {{ background:color-mix(in srgb, var(--muted) 30%, transparent); }}
</style>
<main>
<div class="eyebrow">Causal-Concordance · 012 + 013 · Qwen3.6-27B · WorkspaceBench multihop (via Jan Chen's bank) · exploratory · 2026-10-07</div>
<h1>Readers on swapped states</h1>
<p class="lede">J-lens coordinate swaps of the hidden bridge (e.g. Eiffel Tower → France ⇒ Paris; swap France→Japan) change Qwen's answer on about half of the items. Every reader (Oracle Lens, NLA, J-lens, logit lens) then shifts toward the swapped-in concept, much more when the model's answer actually changed. Most of that tracking is also visible in the model's own output at the read cell, so it does not yet separate reading the bridge from reading what the model is about to say.</p>

<div class="kv">
<div><b>{h(A['j_swap_L24-47']['success']['rate'])}</b><span>J swap L24–47 success (n {A['j_swap_L24-47']['success']['n']}); random {h(A['j_swap_L24-47']['random_control']['rate'])}</span></div>
<div><b>{h(A['logit_swap_L24-47']['success']['rate'])}</b><span>logit-row swap, same band</span></div>
<div><b>{h(X['cells']['oracle|L44|L24-35']['auc_bridge'])} / {h(X['cells']['nla|L42|L24-35']['auc_bridge'])}</b><span>Oracle L44 / NLA L42: AUC of bridge shift for a working swap</span></div>
<div><b>{h(mo['L24-35']['auc_d_model_bridge'])}</b><span>the model's own read-cell output, same AUC</span></div>
</div>

<h2>1. The lever: J-row swaps move the answer</h2>
<p class="sub">Jan's phrase-lens code, unmodified, on our instance. Swap at every prompt position over the band, α 1, greedy continuation contains the counterfactual answer. Clean-correct items only (82 of 87 covered; 5 clean-wrong excluded). Bars: success with Wilson 95% interval; hollow dot: source-preserving random-target control.</p>
<figure><div class="legend"><span><i style="background:var(--s1)"></i>J-row swap success</span><span><i style="background:var(--surface);border:2px solid var(--rnd)"></i>random-target control</span></div>{chart_stageA()}</figure>
<p>J ablation at L24–47 leaves the original answer on {h(A['j_ablation_L24-47']['survives']['rate'])} of items (random: {h(A['j_ablation_L24-47']['random_control']['rate'])}). The swap works at L40–48 too when every position is edited, so 008's brew null is not about the band alone.</p>

<h2>2. The swap has a direction</h2>
<p>The paper's coordinate swap reflects the src−tgt contrast of the clean state. When the clean state already leans toward tgt (in pinv coordinates), the "swap" pushes toward src. At L24–35, {dirs['L24-35']['ts']} of 82 edits went toward src ({dirs['L24-35']['ts_ok']} worked); at L24–47, {dirs['L24-47']['ts']} ({dirs['L24-47']['ts_ok']} worked). Everything below compares working and failed swaps <em>within</em> toward-tgt edits ({dirs['L24-35']['tt']} items at L24–35: {dirs['L24-35']['tt_ok']} work).</p>

<h2>3. Every reader shifts toward the swapped-in concept, more when the answer changed</h2>
<p class="sub">d_r = change in the reader's log-likelihood preference for tgt over src (edited minus clean), in that reader's own nats. Reads are downstream of the edit: L24–35 edits read at L40/L42/L44; L24–47 edits read at L52/L56. Dots: items (blue: answer changed; orange: unchanged; hollow: edit went toward src).</p>
<div class="sms">{chart_scatter("oracle|L44|L24-35", "Oracle L44 · edit L24–35")}{chart_scatter("nla|L42|L24-35", "NLA L42 · edit L24–35")}</div>
<p>Oracle vs NLA, same items, scale-free (AUC difference for predicting a working swap): {h(oa['oracle_L44_minus_nla_L42']['auc_diff'], 3)} [{h(oa['oracle_L44_minus_nla_L42']['ci95'][0], 3)}, {h(oa['oracle_L44_minus_nla_L42']['ci95'][1], 3)}] at L44 vs L42. No detectable difference; the prediction that the Oracle would follow swaps more than the NLA is not supported.</p>

<h2>4. But the answer-facing content tracks just as well</h2>
<p class="sub">Within toward-tgt edits: AUC for predicting a working swap from the reader's bridge shift (blue) and from its answer shift, y_b vs y_a (orange). Dashed line: the same AUC for the model's own output at the read cell (bridge tokens).</p>
<figure><div class="legend"><span><i style="background:var(--s1)"></i>bridge shift</span><span><i style="background:var(--s2)"></i>answer shift</span><span>– – model's own output</span></div>{chart_auc()}</figure>
<p class="sub">Partial Spearman of each reader's bridge shift with the model's Δm, controlling for the model's own read-cell output (bridge and answer) and the edit size; relation-cluster bootstrap 95%.</p>
<figure>{chart_partial()}</figure>
<p class="note">Reading: readers follow states the model demonstrably acts on, but at the answer-producing token most of that is shared with what the model is about to say. A modest residual (partial ρ {h(min(v["partial_bridge_dm_given_model_output_size"] for v in X["cells"].values()))}–{h(max(v["partial_bridge_dm_given_model_output_size"] for v in X["cells"].values()))} across readers and layers, several intervals touching or crossing 0) is all that remains. The next informative cut reads a cell where the bridge is present and the answer is not yet output-facing.</p>

<h2>5. What the readers say</h2>
<p class="sub">Sampled readouts that name the src or tgt bridge concept (word match; 2 samples per state). Counts: names src (or both) / readouts, names tgt (or both) / readouts, at L24–35 edits.</p>
<div class="wrap">{naming_table()}</div>
{readouts()}

<h2>6. Where the change is built, and what readers see along the way (Stage F)</h2>
<p class="sub">Same J swap (L24–35), but restricted by position. Bars: answers changed (clean-correct items); hollow dot: random-target control.</p>
<figure><div class="legend"><span><i style="background:var(--s1)"></i>J-row swap</span><span><i style="background:var(--surface);border:2px solid var(--rnd)"></i>random-target control</span></div>{chart_f1()}</figure>
<p>Editing only the answer token changes nothing; editing everything except it does almost all the work. The model carries the swapped concept from earlier tokens to the answer token itself, so a reader at the answer token under that edit reads a state the model computed, not coordinates we wrote.</p>
<p class="sub">AUC for predicting a working swap. Blue: at the unwritten answer token (all-but-read edit). Orange: mean over earlier prompt tokens, which the edit writes directly (all-position edit; interval: relation-cluster bootstrap 95%). Dashed: the model's own next-token output at those earlier tokens.</p>
<figure><div class="legend"><span><i style="background:var(--s1)"></i>answer token, not written</span><span><i style="background:var(--s2)"></i>earlier tokens, written</span><span>– – model's own early output</span></div>{chart_f_auc()}</figure>
<p class="note">At the earlier, written tokens readers mostly see the write whether or not the model ends up using it (AUC {h(min(v["auc"] for k, v in FA["F2"].items() if isinstance(v, dict) and "auc" in v))}–{h(max(v["auc"] for k, v in FA["F2"].items() if isinstance(v, dict) and "auc" in v))} across readers), and the model's own output there does not move with success either. An earlier version reported the NLA at L42 as a partial exception; that came from a position-mask bug in the analysis (found by an independent audit). Corrected, the NLA is not outcome-specific either (AUC {h(FA["F2"]["nla|L42"]["auc"])}, partial ρ {h(FA["F2"]["nla|L42"]["partial_given_model_early_output_size"])}). At the answer token, the Oracle and NLA keep a moderate association with the answer change beyond the model's own output there (partial ρ about {h(FA["F1"]["readers"]["oracle|L40|J_allbutread_L24-35"]["partial_bridge_dm_given_model_output_writenorm"])} and {h(FA["F1"]["readers"]["nla|L42|J_allbutread_L24-35"]["partial_bridge_dm_given_model_output_writenorm"])}).</p>

<h2>7. Whitened swaps (the TA's suggestion)</h2>
<p>The same coordinate swap done in whitened coordinates (Ledoit–Wolf Σ for L24–35, rows as covectors) changes fewer answers: {GA["flips_white"]}/{GA["n"]} against 38/{GA["n"]} raw (random control {GA["flips_rand_white"]}/{GA["n"]}); every whitened flip is also a raw flip. Pre-registered verdict: <b>{GA["verdict_s14"]}</b> on this family. Readers on the whitened-swap states show the same tracking pattern (Oracle L44 AUC {h(GA["readers"]["oracle|L44|bridge"]["auc"])}, NLA L42 {h(GA["readers"]["nla|L42|bridge"]["auc"])}). The reconstructor-vector version of the suggestion still waits on the official reconstructor loader.</p>

<h2>8. Why the earlier brew swaps failed (Stage H)</h2>
<p>008's lens-coordinate swap on brew edited three cells at L40–48 and its answer-atom control moved the answer share by only 0.010. Rerun at every prompt position, the answer-atom control reaches 0.073 at L40–48 and 0.155 at L24–47, with no answers flipped; the hidden-colour swap stays near zero (0.019 at L24–47). By the pre-registered rule this pointed to the task family, but the rerun also changed the numerical regime and the J-row convention, and its late-band control failed the power bar, so which difference explains 008's null is <b>not established</b>. The J-row edits at the brew answer cell are only 1–2% of the activation norm: colour concepts there load weakly on their J rows, while multihop bridges load enough for every-position swaps to change about half the answers.</p>

<h2>9. The Oracle's reconstructor: our loader and its J-lens target</h2>
<p class="sub">The official reconstructor (AR) loader is not public, so we tested our own against the published per-layer validation profile (1,024 overlapping spans re-cropped from the same 12 responses used to choose the loader conventions, so reused calibration contexts rather than held-out validation; all 12 tag layers). Solid: ours (one fitted scale per layer); dashed grey: published. Blue: whitened FVE; orange: FVE of the J-lens transform of the mean-centred activation, our best guess at the AR's undocumented "jfve" term.</p>
<figure><div class="legend"><span><i style="background:var(--s1)"></i>our FVE</span><span><i style="background:var(--s2)"></i>our J-lens-target FVE</span><span>– – published</span></div>{chart_e0()}</figure>
<p>Our loader matches the published FVE profile at every layer once a per-layer scale is fitted, but its retrieval@1 falls short at L20–32, so by the pre-registered rule it is <b>not validated</b> and the TA's reconstructor-vector swaps stay blocked. The fitted scale rises from about 0.1 to 2.3 with depth, consistent with a scale-invariant training loss. The J-lens-target candidate matches the published "jfve" within 0.04 at every layer from L32 up but misses L20–28 (8/11 against a pre-set bar of 9/11; post hoc): the metadata and this profile support a J-aware reconstructor hypothesis, but the exact training loss, and whether the Oracle's reward depends on J, remain unverified. The 1,024 spans reuse the 12 responses used earlier to choose the loader conventions, so this is a calibration check, not independent validation, and the scale is fitted per layer on the same samples.</p>

<h2>10. Holding the hidden country before the question is asked (Stage I)</h2>
<p class="sub">"Fact: Consider the country of Mount Fuji. [unrelated sentence] The capital / main language / continent of that country is". The model's state at the end of the unrelated sentence is identical whichever question follows (checked bitwise). We swap the hidden country only up to that point.</p>
<p>The swap still changes the later answer on {IA["I1_lever"]["J_pre"]}/{IA["I1_lever"]["gated_n"]} vetted prompts (random control {IA["I1_lever"]["rand_pre"]}/{IA["I1_lever"]["gated_n"]}): the model holds the swapped country and uses it for whichever question comes. But at the end of the unrelated sentence no reader sees it (Oracle L44 AUC {h(IA["gated_primary"]["readers"]["oracle|L44|H2|all"]["auc"])}, NLA L42 {h(IA["gated_primary"]["readers"]["nla|L42|H2|all"]["auc"])}; readouts there almost never name either country), while the model's own next-token output there still leaks it (AUC {h(IA["gated_primary"]["I2_model_H2_bridge_auc"])}). At the end of the clue sentence, which the swap writes, every reader tracks it (AUC about {h(IA["gated_primary"]["readers"]["oracle|L44|H1|all"]["auc"])}, logit lens included). Only {IA["I1_lever"]["gated_n"]} of 168 delayed prompts passed the clean gate, so these are small numbers.</p>
<p class="note">Reading: the hidden step can be held in context and used later without being visible in the residual stream at the token that holds it. A reader that reads one token sees it where it is written or about to be said, not where it is merely held.</p>

<h2>11. Swapping only the clue, reading where the model computes it (Stage J)</h2>
<p>With a looser vetting rule (8-token answers, 8/10 samples), {JA["n_gated"]} delayed prompts pass. Swapping only the clue's own tokens (nothing after them is written) changes the later answer on {JA["J_clue"]}/{JA["n_gated"]} (random {JA["rand_clue"]}/{JA["n_gated"]}). At the clue's final token, which the model computes from the edited clue, every reader tracks whether the swap worked (Oracle L44 AUC {h(JA["readers"]["oracle|L44|H1"]["auc"])}, NLA L42 {h(JA["readers"]["nla|L42|H1"]["auc"])}, plain logit lens {h(JA["readers"]["logit|L44|H1"]["auc"])}), but the model's own next-token output there carries a similar signal (AUC {h(JA["J2_model_H1_bridge_auc"])}, above the pre-set 0.65 bar), the effect leans on Japan-target items, and Oracle and NLA do not differ (paired difference {h(JA["paired_oracle_vs_nla"]["oracle_L44_minus_nla_L42_H1"]["diff"])}). At the end of the unrelated sentence no reader at L40–44 tracks it (AUC {h(min(v["auc"] for k, v in JA["readers"].items() if k.endswith("H2")))}–{h(max(v["auc"] for k, v in JA["readers"].items() if k.endswith("H2")))}), although the model's own next-token output there does (AUC {h(JA["J2_model_H2_bridge_auc"])}), so that token is not free of output-facing content either.</p>

<h2>12. The held country does reach the carrier end, in later layers (Stage L)</h2>
<p class="sub">Same prompts and edits as Stage J (hold states reproduced bitwise, {LA["n_holds"]} holds, {LA["n_success"]} working). AUC for telling working swaps from failed ones from each reader's shift toward the swapped-in country, at the end of the unrelated sentence. Dashed line: the model's own next-token output there.</p>
<figure><div class="legend"><span><i style="background:var(--s1)"></i>Oracle</span><span><i style="background:var(--s2)"></i>logit lens</span><span><i style="background:var(--fg2)"></i>J_NP / J_CB (dashed)</span></div>{chart_stageL()}</figure>
<p>Through L48 no reader can tell (AUC {h(min(v["auc"] for k, v in LA["readers"].items() if "|L48|H2" in k))}–{h(max(v["auc"] for k, v in LA["readers"].items() if "|L48|H2" in k))}). From L52 every reader can, about as well as the model's own output ({h(min(v["auc"] for k, v in LA["readers"].items() if k.endswith("|H2") and "|L48|" not in k))}–{h(max(v["auc"] for k, v in LA["readers"].items() if k.endswith("|H2") and "|L48|" not in k))}; best {E(LA["H2_best"]["key"].split("|")[0])} at {E(LA["H2_best"]["key"].split("|")[1])}, {h(LA["H2_best"]["auc"])}). The Oracle does no better than the plain logit lens, and its sampled readouts there describe the carrier sentence and almost never name a country. So Stage J's "nothing at the carrier end" was a layer effect: the swapped country arrives between L48 and L52, and what arrives is about as readable as the model's own next-token output.</p>

<h2 class="part">Part two (013): do the NLA's false claims carry signal?</h2>
<p class="lede">NLA explanations are mostly not true of the prompt. We checked what that false text does for the NLA's own reconstructor, which turns an explanation back into an activation (fraction of variance explained, FVE).</p>
<p class="sub">Material: the {RW["n"]} NLA explanations saved in 012 (87 items × clean / J swap / random swap × 2 samples, layer 42). NLA: nla-rl av_base + iter_000400 with its final reconstructor (validated: 78.1% FVE on the release's own examples against 76.6% published). On these prompts the clean-state explanations reconstruct at {h(DR["multihop"]["fve_pct_first"], 1)}% FVE ({h(RW["fve_mean"]["O"], 1)}% over all three states); on brew at {h(DR["brew"]["fve_pct_first"], 0)}%, so brew was dropped. Judges: Gemini 3.8 Flash marks false spans and rewrites; GLM-5.3-flash labels sentences and tags each false span by kind. Hand checks: false-span precision 89/90; kind tags 75/90 (the "invented context" tag only 24/39).</p>

<h2>13. Deleting false text: invented context is padding, misquotes carry the content</h2>
<p class="sub">Δ = FVE after deleting the same amount of random text minus FVE after deleting the false text, in FVE points. Above 0: the false text carried more signal than average text. Grey band: within ±2 points. Interval: 95% cluster bootstrap over items.</p>
<figure><div class="legend"><span><i style="background:var(--s1)"></i>all false text / misquotes</span><span><i style="background:var(--s2)"></i>invented context</span></div>{chart_deletion()}</figure>
<p>False text is {h(100 * HD["flagged_token_share"]["contradicted_mean"], 0)}% of an average explanation. Two kinds behave differently. <b>Invented context</b> (made-up headers, quiz names, framings; about a quarter of the text) costs no more to delete than random text. <b>Misquotes</b> (paraphrases of the prompt in altered wording, about half the text) cost far more, because they still carry the bridge and the answer. The pre-registered headline ("false claims carry extra signal") comes from the misquotes. On 007's test half alone: invented context {h(TH["deletion_test"]["invented_context_sentence"]["mean"], 1)} [{h(TH["deletion_test"]["invented_context_sentence"]["ci95"][0], 1)}, {h(TH["deletion_test"]["invented_context_sentence"]["ci95"][1], 1)}], misquotes {h(TH["deletion_test"]["misquote_sentence"]["mean"], 1)}.</p>

<h2>14. Rewriting to keep only facts (the TA's suggestion)</h2>
<p class="sub">Each explanation rewritten minimally so every claim is true (misquotes corrected to the prompt's wording, invented context removed), against a paraphrase that changes about as many words but keeps every claim. The judge finds contradicted sentences in {h(RW["B_contradicted_sentence_rate"]["O"])} of the originals, {h(RW["B_contradicted_sentence_rate"]["F"])} of the factual rewrites and {h(RW["B_contradicted_sentence_rate"]["P"])} of the paraphrases.</p>
{rewrite_table()}
<p>The factual rewrite reconstructs {h(RW["dFVE_F_minus_P"]["mean"], 1)} points better than the paraphrase [{h(RW["dFVE_F_minus_P"]["ci95"][0], 1)}, {h(RW["dFVE_F_minus_P"]["ci95"][1], 1)}]; about half of that is the cost paraphrasing itself carries. Against the original it gains {h(RW["dFVE_F_minus_O"]["mean"], 1)} [{h(RW["dFVE_F_minus_O"]["ci95"][0], 1)}, {h(RW["dFVE_F_minus_O"]["ci95"][1], 1)}]. The reconstruction also moves further into the J subspace (+{h(RW["dJ_F_minus_P"]["mean"], 3)}), but the original was already more J than the real activation ({h(RW["share_J_mean"]["O"], 3)} against {h(RW["share_J_mean"]["gold"], 3)}), so this does not mean the cleaned text is more faithful.</p>
<p class="note">Reading: removing the NLA's hallucinations makes its reconstruction slightly better. The made-up framing is padding; the information sits in what it paraphrases from the prompt, sometimes wrongly. The reconstructor was trained together with this NLA, so this describes what that pair uses, not what the model computes.</p>

<h2>15. Sample explanations</h2>
<p class="sub">Two chosen to illustrate, four drawn at random (seed 5). Marked spans: <mark class="k-invented_context">invented context</mark> <mark class="k-misquote">misquote</mark> <mark class="k-wrong_entity">wrong entity or false fact</mark>.</p>
{examples13()}

<h2>Gates and scope</h2>
<ul>
<li>Our captures come from the exact labelled runs: greedy text identical to Jan's code on 87/87 items for clean, J and random arms; α = 0 edit bitwise; read position = WSB last-word token on 87/87.</li>
<li>Margins in float32 (Jan's scorer uses bf16 logits; identity-clamp floor median 0.001 nats).</li>
<li>Scope: Qwen3.6-27B, 82 clean-correct single-token multihop items (49 of them in 007's already-read test half), one read position, the stated bands. Japan is the target in 28 of 87 covered items. Labels mean "an answer effect under this intervention", not "the bridge is the mediator".</li>
<li>Exploratory and partly post hoc: the edit-direction stratification and the output-facing analyses were chosen after the smoke and the first summaries. Spec, ledger and code: branch 012-reader-concordance (docs/design_specs/012_reader_concordance.md, docs/ledger/012_ledger.md).</li>
<li>Stage L and 013: branch 013-nla-hallucination (docs/design_specs/012_reader_concordance.md §22; docs/design_specs/013_nla_hallucination.md with amendments A1-A2; docs/ledger/013_ledger.md). The 013 split by kind and the rewrite arm were added after the smoke, in written amendments, before their runs. Judge spend about $2.35.</li>
</ul>
</main>"""
(R / "page").mkdir(parents=True, exist_ok=True)
(R / "page/index.html").write_text(page)
print("wrote", R / "page/index.html", len(page))
