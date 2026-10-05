"""Build the 005 night report page (one HTML file, inline SVG charts) from committed/saved JSON only.

  python make_report_page.py OUT.html [--pilot pilot1] [--smoke smoke1]

Every number on the page is read from JSON and rounded half-up here; nothing is typed in by hand except the
pre-registered priors, which are quoted with the commit that registered them.
"""
import html
import json
import math
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import p005 as P                                     # noqa: E402

OUTP = ROOT / "outputs/005_planted_concepts"
RESA = ROOT / "results/005_planted_concepts/analysis"
CHK = ROOT / "results/checks_2026-10-04"
GRID9 = [1 / 256, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
GRID6 = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
SNAME = {1 / 256: "1/256", 1 / 128: "1/128", 1 / 64: "1/64", 1 / 32: "1/32", 1 / 16: "1/16", 1 / 8: "1/8",
         1 / 4: "1/4", 1 / 2: "1/2", 1.0: "1", 0.0: "0"}
COND = {1: ("pure J-lens vector", "jl"), 2: ("full concept vector", "full"), 3: ("J part (matched)", "j"),
        4: ("non-J part (matched)", "perp"), 5: ("J part (natural)", "jn"), 6: ("non-J part (natural)", "pn"),
        7: ("random direction", "rand"), 8: ("random J vector", "rj"), 9: ("random non-J vector", "rp"),
        10: ("non-J + clamp to clean", "cl"), "10r": ("non-J + random clamp to clean", "clr"),
        12: ("non-J + re-entry clamp (to L62)", "re"), "12r": ("non-J + random re-entry clamp (to L62)", "rer"),
        13: ("non-J + re-entry clamp (to L44)", "re44"), "13r": ("non-J + random re-entry clamp (to L44)", "rer44")}


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def r(x, nd=2, sign=False):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    q = Decimal(str(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    s = f"{q:.{nd}f}" if nd > 0 else f"{q:.0f}"
    if sign and float(q) > 0:
        s = "+" + s
    return s.replace("-", "−")


def pct(x, nd=0):
    return "—" if x is None else r(100 * float(x), nd) + "%"


def esc(s):
    return html.escape(str(s))


def med(x):
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else None


# ----------------------------------------------------------------------------------------------------- charts
def line_chart(series, xs, xlabels, ylabel, *, ylim=None, ylog=False, invert=False, band=None, w=560, h=300,
               aria="chart", ytick_fmt=None, xlabel="strength s (× layer mean norm, at every band layer)"):
    """series: [{name, cls, ys, dash}] with ys aligned to xs (None = missing). band: list of y upper values (null 95th
    percentile), drawn as a shaded area from the axis floor. Categorical x (equal spacing)."""
    L, Rm, T, B = 58, 14, 14, 46
    pw, ph = w - L - Rm, h - T - B
    vals = [y for s in series for y in s["ys"] if y is not None]
    if band:
        vals += [y for y in band if y is not None]
    if ylim:
        lo, hi = ylim
        if ylog:
            lo, hi = math.log10(max(lo, 1e-9)), math.log10(max(hi, 1e-9))
    elif ylog:                                            # pad in log units (linear padding can go below zero)
        lo = math.log10(max(min(vals), 1e-9)) if vals else 0.0
        hi = math.log10(max(max(vals), 1e-9)) if vals else 1.0
        if lo == hi:
            hi = lo + 1
        pad = 0.05 * (hi - lo)
        lo, hi = lo - pad, hi + pad
    else:
        lo, hi = (min(vals), max(vals)) if vals else (0, 1)
        if lo == hi:
            hi = lo + 1
        pad = 0.06 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        if lo > 0 and lo < 0.25 * hi:
            lo = 0

    def X(i):
        return L + (pw * i / max(len(xs) - 1, 1))

    def Y(v):
        v = math.log10(max(v, 1e-9)) if ylog else v
        t = (v - lo) / (hi - lo)
        t = 1 - t if not invert else t
        return T + ph * min(max(t, 0), 1)

    out = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(aria)}" preserveAspectRatio="xMidYMid meet">']
    # y ticks
    if ylog:
        ticks = [10 ** k for k in range(int(math.ceil(lo - 1e-9)), int(math.floor(hi + 1e-9)) + 1)]
    else:
        span = hi - lo
        step = 10 ** math.floor(math.log10(span / 4)) if span > 0 else 1
        for m_ in (1, 2, 2.5, 5, 10):
            if span / (step * m_) <= 6:
                step *= m_
                break
        t0 = math.ceil(lo / step) * step
        ticks = []
        t = t0
        while t <= hi + 1e-12:
            ticks.append(round(t, 10))
            t += step
    for t in ticks:
        y = Y(t)
        if T - 0.5 <= y <= T + ph + 0.5:
            lab = ytick_fmt(t) if ytick_fmt else (f"{t:g}" if not ylog else (f"{int(t):,}" if t >= 1 else f"{t:g}"))
            out.append(f'<line class="grid" x1="{L}" x2="{L + pw}" y1="{y:.1f}" y2="{y:.1f}"/>')
            out.append(f'<text class="tick" x="{L - 6}" y="{y + 4:.1f}" text-anchor="end">{esc(lab)}</text>')
    if not ylog and lo < 0 < hi:
        out.append(f'<line class="zero" x1="{L}" x2="{L + pw}" y1="{Y(0):.1f}" y2="{Y(0):.1f}"/>')
    # band
    if band and any(b is not None for b in band):
        base = Y(lo if not ylog else 10 ** lo) if not invert else Y(10 ** hi if ylog else hi)
        pts = [(X(i), Y(b)) for i, b in enumerate(band) if b is not None]
        if pts:
            poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            out.append(f'<polygon class="nullband" points="{pts[0][0]:.1f},{base:.1f} {poly} {pts[-1][0]:.1f},{base:.1f}"/>')
    # x ticks
    for i, lab in enumerate(xlabels):
        out.append(f'<text class="tick" x="{X(i):.1f}" y="{T + ph + 16}" text-anchor="middle">{esc(lab)}</text>')
    out.append(f'<text class="axlab" x="{L + pw / 2:.1f}" y="{h - 6}" text-anchor="middle">{esc(xlabel)}</text>')
    fs = min(12.5, ph / (0.6 * max(len(ylabel), 1)))         # keep the rotated label inside the plot height
    out.append(f'<text class="axlab" style="font-size:{fs:.1f}px" transform="translate(13,{T + ph / 2:.1f}) rotate(-90)" '
               f'text-anchor="middle">{esc(ylabel)}</text>')
    out.append(f'<line class="axis" x1="{L}" x2="{L}" y1="{T}" y2="{T + ph}"/><line class="axis" x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}"/>')
    for s in series:
        pts = [(X(i), Y(y)) for i, y in enumerate(s["ys"]) if y is not None]
        if len(pts) >= 2:
            d = " ".join(f"{'M' if k == 0 else 'L'}{x:.1f},{y:.1f}" for k, (x, y) in enumerate(pts))
            out.append(f'<path class="ln c-{s["cls"]}{" dash" if s.get("dash") else ""}" d="{d}"/>')
        for x, y in pts:
            out.append(f'<circle class="pt c-{s["cls"]}" cx="{x:.1f}" cy="{y:.1f}" r="2.6"/>')
    out.append("</svg>")
    return "".join(out)


def legend(series, extra=None):
    items = [f'<span class="lg"><i class="sw c-{s["cls"]}{" dash" if s.get("dash") else ""}"></i>{esc(s["name"])}</span>'
             for s in series]
    if extra:
        items.append(f'<span class="lg"><i class="sw nullsw"></i>{esc(extra)}</span>')
    return '<div class="legend">' + "".join(items) + "</div>"


def figure(svg, leg, caption):
    return f'<figure class="fig"><div class="figscroll">{svg}</div>{leg}<figcaption>{caption}</figcaption></figure>'


def table(head, rows, cls="", num_cols=()):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f'<td class="{"num" if i in num_cols else ""}">{c}</td>' for i, c in enumerate(row)) + "</tr>"
    return f'<div class="tblwrap"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def chip(text, kind):
    return f'<span class="chip {kind}">{esc(text)}</span>'


def heat(v, scale=1.0):
    """A table cell shaded by a 0..1 rate."""
    if v is None:
        return '<span class="hc na">—</span>'
    a = max(0.0, min(1.0, float(v) / scale))
    return f'<span class="hc" style="--a:{a * 85:.0f}%">{r(v, 2)}</span>'


# ------------------------------------------------------------------------------------------------------- data
class Data:
    def __init__(self, smoke="smoke1", pilot="pilot1"):
        self.c1 = load(CHK / "c1_chain_rescore.json")
        self.c2 = load(CHK / "c2_damage_decomposition.json")
        self.c3 = load(CHK / "c3_removal_energy.json")
        self.screen = load(RESA / "concept_screen.json")
        self.smoke = load(RESA / "smoke_summary.json")
        self.pilot = load(RESA / "pilot_summary.json")
        sd = OUTP / "smoke" / smoke
        self.sm_state = load(sd / "state.json") or {}
        self.sm_g1 = load(sd / "rows_g1.json") or []
        self.sm_e2e = load(sd / "rows_e2e.json") or []
        self.sm_oracle = load(sd / "oracle_reads.json") or []
        self.sm_nla = load(sd / "nla_reads.json") or []
        pd = OUTP / "pilot" / pilot
        self.pd = pd
        self.pl_state = load(pd / "state.json") or {}
        self.pl_oracle = load(pd / "oracle_reads.json") or []
        self.pl_nla = load(pd / "nla_reads.json") or []
        self.pl_rows = {}
        for f in sorted(pd.glob("rows_*_*.json")):
            _, band, pre = f.stem.split("_")
            self.pl_rows[(band, pre)] = load(f)
        self.sec = load(pd / "sec_crosscheck.json")
        self.s7 = load(pd / "step7b" / "state.json")


def curve(rows, cond, key, grid, agg="median", seed0=True):
    out = []
    for s in grid:
        v = [x[key] for x in rows if str(x["cond"]) == str(cond) and x["s"] == s and x.get(key) is not None
             and (not seed0 or x.get("seed", 0) == 0)]
        out.append((float(np.median(v)) if agg == "median" else float(np.mean(v))) if v else None)
    return out


def null_curve(rows, key, grid, q=0.95):
    out = []
    for s in grid:
        v = [x[key] for x in rows if x["cond"] == 7 and x["s"] == s]
        out.append(float(np.quantile(v, q)) if v else None)
    return out


CSS = r"""
/* Layout: a lab-notebook report. Sticky contents rail on wide screens, one reading column, data panels that may run
   wider inside their own scroll boxes. Conditions keep one colour everywhere (Okabe-Ito, colour-blind safe). */
:root{
  --paper:#f5f7fa; --panel:#ffffff; --ink:#18212d; --muted:#566273; --rule:#d8dee7; --soft:#eef2f6;
  --accent:#1d5d8a; --accent-ink:#ffffff;
  --good:#24764a; --warn:#9a6400; --bad:#ad2f27; --pend:#5f6b7b;
  --c-jl:#009e73; --c-full:#b85c9b; --c-j:#0072b2; --c-perp:#d55e00; --c-rand:#7c8592; --c-null:#d6dde6;
  --c-jn:#56b4e9; --c-pn:#e69f00; --c-rj:#2b8c8c; --c-rp:#9b6a3d;
  --c-cl:#7a3e9d; --c-clr:#b9a3c9; --c-re:#1f7a3a; --c-rer:#9ccfa9; --c-re44:#0b4f6c; --c-rer44:#8fbcd0;
  --f-display:"IBM Plex Sans Condensed","Arial Narrow",Arial,sans-serif;
  --f-body:"Source Serif 4",Georgia,"Times New Roman",serif;
  --f-mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --measure:68ch;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --paper:#0f141b; --panel:#151c26; --ink:#e2e8f0; --muted:#9aa7b7; --rule:#2a3442; --soft:#1a2230;
  --accent:#7fb7de; --accent-ink:#0f141b;
  --good:#6cc794; --warn:#e3b04f; --bad:#f08a7e; --pend:#a7b2c0;
  --c-jl:#36c99a; --c-full:#e19bc8; --c-j:#5ab4ec; --c-perp:#f2874a; --c-rand:#a2abb7; --c-null:#2c3747;
  --c-jn:#8fd0f3; --c-pn:#f2c25a; --c-rj:#5cc3c3; --c-rp:#d1a274;
  --c-cl:#c39be0; --c-clr:#6f5a82; --c-re:#5fd18a; --c-rer:#3d6b4c; --c-re44:#6cc0e6; --c-rer44:#35596a;
  color-scheme:dark}}
:root[data-theme="dark"]{
  --paper:#0f141b; --panel:#151c26; --ink:#e2e8f0; --muted:#9aa7b7; --rule:#2a3442; --soft:#1a2230;
  --accent:#7fb7de; --accent-ink:#0f141b;
  --good:#6cc794; --warn:#e3b04f; --bad:#f08a7e; --pend:#a7b2c0;
  --c-jl:#36c99a; --c-full:#e19bc8; --c-j:#5ab4ec; --c-perp:#f2874a; --c-rand:#a2abb7; --c-null:#2c3747;
  --c-jn:#8fd0f3; --c-pn:#f2c25a; --c-rj:#5cc3c3; --c-rp:#d1a274;
  --c-cl:#c39be0; --c-clr:#6f5a82; --c-re:#5fd18a; --c-rer:#3d6b4c; --c-re44:#6cc0e6; --c-rer44:#35596a;
  color-scheme:dark}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
@media (prefers-reduced-motion: reduce){html{scroll-behavior:auto}}
body{background:var(--paper);color:var(--ink);font-family:var(--f-body);font-size:17px;line-height:1.55;margin:0}
.wrap{max-width:1240px;margin:0 auto;padding-block:28px 80px;padding-inline:20px;display:grid;grid-template-columns:1fr;gap:28px}
@media (min-width:1100px){.wrap{grid-template-columns:230px minmax(0,1fr)}}
nav.toc{display:none}
@media (min-width:1100px){nav.toc{display:block;position:sticky;top:calc(env(safe-area-inset-top,0px) + 18px);align-self:start;
  font-family:var(--f-display);font-size:14px;max-height:calc(100vh - 40px);overflow:auto;padding-right:6px}}
nav.toc a{display:block;color:var(--muted);text-decoration:none;padding:3px 0 3px 10px;border-left:2px solid var(--rule)}
nav.toc a:hover,nav.toc a:focus-visible{color:var(--ink);border-left-color:var(--accent);outline:none}
nav.toc .sub{padding-left:22px;font-size:13px}
main{min-width:0}
header.top{border-bottom:1px solid var(--rule);padding-bottom:22px;margin-bottom:8px}
.eyebrow{font-family:var(--f-mono);font-size:12.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
h1{font-family:var(--f-display);font-weight:700;font-size:clamp(30px,5vw,46px);line-height:1.08;margin:.25em 0 .3em;text-wrap:balance;letter-spacing:-.01em}
h2{font-family:var(--f-display);font-weight:700;font-size:27px;line-height:1.2;margin:2.2em 0 .5em;text-wrap:balance;scroll-margin-top:16px}
h3{font-family:var(--f-display);font-weight:600;font-size:20px;margin:1.6em 0 .4em;text-wrap:balance;scroll-margin-top:16px}
h4{font-family:var(--f-display);font-weight:600;font-size:16.5px;margin:1.2em 0 .3em;color:var(--muted);letter-spacing:.02em}
p,li{max-width:var(--measure)}
p{margin:.5em 0 .9em}
ul,ol{padding-left:1.3em;margin:.4em 0 1em}
li{margin:.25em 0}
a{color:var(--accent)}
code,.mono{font-family:var(--f-mono);font-size:.86em;overflow-wrap:anywhere}
.lede{font-size:19px;color:var(--ink);max-width:var(--measure)}
.muted{color:var(--muted)}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 4px}
.chip{font-family:var(--f-mono);font-size:12px;letter-spacing:.03em;padding:3px 9px;border-radius:999px;border:1px solid currentColor;white-space:nowrap}
.chip.good{color:var(--good)} .chip.bad{color:var(--bad)} .chip.warn{color:var(--warn)} .chip.pend{color:var(--pend)}
.callout{background:var(--panel);border:1px solid var(--rule);border-left:4px solid var(--accent);padding:14px 18px;margin:16px 0;max-width:calc(var(--measure) + 40px)}
.callout h3,.callout h4{margin-top:.2em}
.qcard{background:var(--panel);border:1px solid var(--rule);padding:16px 18px;margin:18px 0}
.qcard .qhead{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px;justify-content:space-between}
.qcard .qhead h3{margin:0}
.verdict{font-family:var(--f-mono);font-size:13px;font-weight:500;padding:3px 9px;border:1px solid currentColor;border-radius:4px}
.verdict.yes{color:var(--good)} .verdict.no{color:var(--bad)} .verdict.mixed{color:var(--warn)} .verdict.pend{color:var(--pend)}
dl.kv{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:4px 14px;margin:8px 0;font-size:15.5px}
dl.kv dt{font-family:var(--f-display);color:var(--muted);overflow-wrap:anywhere} dl.kv dd{margin:0;min-width:0}
@media (max-width:640px){dl.kv{grid-template-columns:minmax(0,1fr)} dl.kv dd{margin-bottom:8px}}
.tblwrap{overflow-x:auto;margin:12px 0 18px;border:1px solid var(--rule);background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:14.5px;font-family:var(--f-body)}
th{font-family:var(--f-display);font-weight:600;text-align:left;background:var(--soft);color:var(--ink);padding:7px 10px;border-bottom:1px solid var(--rule);white-space:nowrap}
td{padding:6px 10px;border-bottom:1px solid var(--rule);vertical-align:top}
td.num{font-family:var(--f-mono);font-size:13px;font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:none}
.hc{display:inline-block;min-width:3.2em;text-align:center;font-family:var(--f-mono);font-size:12.5px;padding:1px 4px;border-radius:3px;
  background:color-mix(in srgb,var(--c-perp) var(--a,0%),transparent);font-variant-numeric:tabular-nums}
.hc.na{color:var(--muted);background:none}
.fig{margin:18px 0 26px;background:var(--panel);border:1px solid var(--rule);padding:12px 12px 10px}
.figscroll{overflow-x:auto}
svg.chart{display:block;width:100%;min-width:420px;max-width:760px;height:auto}
svg .grid{stroke:var(--rule);stroke-width:1}
svg .axis{stroke:var(--muted);stroke-width:1}
svg .zero{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3}
svg .tick{fill:var(--muted);font-family:var(--f-mono);font-size:11px}
svg .axlab{fill:var(--muted);font-family:var(--f-display);font-size:12.5px}
svg .nullband{fill:var(--c-null);opacity:.75}
svg .ln{fill:none;stroke-width:2.2;stroke-linejoin:round;stroke-linecap:round}
svg .ln.dash{stroke-dasharray:6 4}
svg .pt{stroke:none}
figcaption{font-size:14.5px;color:var(--muted);margin-top:6px;max-width:var(--measure)}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-family:var(--f-display);font-size:13.5px;margin:6px 4px 2px}
.lg{display:inline-flex;align-items:center;gap:6px}
.sw{display:inline-block;width:20px;height:0;border-top:3px solid var(--muted)}
.sw.dash{border-top-style:dashed}
.sw.nullsw{height:10px;border:none;background:var(--c-null)}
.smallmult{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,430px),1fr));gap:14px}
.smallmult .fig{margin:0}
.read{background:var(--soft);border-left:3px solid var(--rule);padding:8px 12px;margin:8px 0;font-family:var(--f-mono);font-size:12.5px;
  white-space:pre-wrap;word-break:break-word;max-width:100%}
.read .tag{display:block;font-family:var(--f-display);font-size:13px;color:var(--muted);margin-bottom:3px}
details{margin:10px 0}
details>summary{cursor:pointer;font-family:var(--f-display);font-weight:600;color:var(--accent)}
details>summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:14px}
.kpi{background:var(--panel);border:1px solid var(--rule);padding:12px 14px}
.kpi .v{font-family:var(--f-display);font-weight:700;font-size:26px;line-height:1.1}
.kpi .k{font-size:14px;color:var(--muted)}
.pending{color:var(--pend);font-style:italic}
footer{border-top:1px solid var(--rule);margin-top:40px;padding-top:14px;font-size:14px;color:var(--muted)}
"""
for _cls in ("jl", "full", "j", "perp", "rand", "jn", "pn", "rj", "rp", "cl", "clr", "re", "rer", "re44", "rer44"):
    CSS += f"svg .c-{_cls}{{stroke:var(--c-{_cls});fill:var(--c-{_cls})}} svg .ln.c-{_cls}{{fill:none}} .sw.c-{_cls}{{border-top-color:var(--c-{_cls})}}\n"


# ----------------------------------------------------------------------------------------------- scorecard
def scorecard(D):
    """Pre-registered agent priors (quoted with the registering commit) and their outcomes from JSON."""
    items = []

    def add(group, what, prior, commit, outcome, note, brier=True):
        items.append({"group": group, "what": what, "prior": prior, "commit": commit, "outcome": outcome, "note": note,
                      "brier": brier})

    c1 = D.c1["cells"]["oracle_L56"]["wsb"] if D.c1 else None
    if c1:
        gap = c1["pass_rate"] - c1["null_top1_near"]
        add("Checks", "C1: Oracle at L56 beats WSB's decoy null by ≥ 0.10 (chain)", 0.55, "e50864d", gap >= 0.10,
            f"pass {r(c1['pass_rate'], 3)} vs null {r(c1['null_top1_near'], 3)} (gap {r(gap, 3)})")
    if D.c2:
        z = D.c2["A_004_L42_full_sequence"]["share_z_plus_z2_median"]
        add("Checks", "C2: z + z′ take ≥ 50% of the off-pair increase at “:” L42", 0.25, "e50864d", z >= 0.5,
            f"median share {r(z, 2)}")
    if D.c3:
        R = D.c3["sets"]["multihop_CW4"]["R_median"]
        add("Checks", "C3: multihop content removal takes > 2× the control's energy", 0.80, "e50864d", R > 2,
            f"median ratio {r(R, 2)}")
    sm = D.sm_state
    if sm.get("G1"):
        add("Smoke", "G1 report gate passes at band L32–38", 0.65, "df56646", bool(sm["G1"][0]["decision"]["pass"]),
            "sampled naming 0% for both the pure J-lens and the full vector")
        add("Smoke", "G1 calibration shifts the grid", 0.35, "df56646",
            any(g["decision"]["calibration"] for g in sm["G1"]), "no shift: neither peak sat at a grid edge")
    if D.smoke:
        g3 = D.smoke["G3_J_share"]["median"]
        add("Smoke", "G3: median J share at k = 16 within [1%, 30%]", 0.85, "df56646", 0.01 <= g3 <= 0.30,
            f"median {pct(g3, 1)}")
        add("Smoke", "G4: probe validity passes", 0.70, "df56646", D.smoke["G4"]["pass_rule_on_medians"],
            f"held-out AUC median {r(D.smoke['G4']['loto_auc_median'], 3)}, permutation control {r(D.smoke['G4']['perm_ctrl_median'], 2)}")
        add("Smoke", "G5 flags a leaky split (> 20% of concept-layers)", 0.25, "df56646",
            D.smoke["G5"]["share_top10"] > 0.20, f"{pct(D.smoke['G5']['share_top10'], 0)} of concept-layers (earth only)")
        nla_drop = any(v["over_5pct"] for v in (D.smoke.get("nla_clean") or {}).values())
        add("Smoke", "G6 drops ≥ 1 concept for the NLA (clean naming > 5%)", 0.40, "df56646", nla_drop,
            "case, dog, role")
        on = sm.get("onset", {}).get("s_star")
        add("Smoke", "No u⊥ onset inside the 6-point grid", 0.30, "df56646", on is None,
            f"onset at s = {SNAME.get(on, on)}, the grid floor (point estimate was about 1/4)")
    pl = D.pilot
    if pl:
        if "Q1" in pl:
            add("Pilot", "Q1: the rank / log-prob report measure works", 0.90, "0b0ca0b", pl["Q1"]["answer"] == "yes",
                "rule: ≥ 3 strengths above the random null for conditions 1 and 2")
        if "Q2" in pl and any(k.endswith("_word") for k in pl["Q2"]["by_block"]):
            add("Pilot", "Q2: the ‘word’ prefill lifts naming to ≥ 25%", 0.30, "0b0ca0b", pl["Q2"]["answer"] == "yes",
                "best sampled naming under the ‘word’ prefill")
        if pl.get("Q3", {}).get("by_prefill"):
            ch = [v["rule_choice"] for v in pl["Q3"]["by_prefill"].values()]
            add("Pilot", "Q3: the wide band L24–38 is chosen", 0.50, "0b0ca0b", "B2" in ch, f"rule chose {', '.join(ch)}")
        if "Q4" in pl:
            add("Pilot", "Q4: u⊥ onset above the grid floor in at least one band", 0.60, "0b0ca0b",
                pl["Q4"]["answer"] == "yes", ", ".join(f"{k}: {SNAME.get(v['s_star'], v['s_star'])}"
                                                       for k, v in pl["Q4"]["by_block"].items()))
        q5 = (pl.get("Q5_rules_preview") or {})
        v5 = q5.get("NLA names c at carried C_o for the full vector at some strength (mean over dev > 0.10 and > rand-dir)")
        if v5 is not None:
            add("Pilot", "Q5: the NLA names c at the report position (full vector)", 0.40, "0b0ca0b", any(v5.values()),
                "mean naming over dev concepts > 0.10 at some strength")
        if "Q6" in pl:
            add("Pilot", "Q6: the clamp beats its random null", 0.25, "0b0ca0b",
                any(v["distinguishable"] for v in pl["Q6"]["by_block"].values()),
                "; ".join(f"{k}: {r(v['extra_reduction_dR1p_mean'], 2, True)} nats" for k, v in pl["Q6"]["by_block"].items()))
        if "Q7" in pl and pl["Q7"]["leaky"]:
            n = max(len(v) for v in pl["Q7"]["leaky"].values())
            add("Pilot", "Q7: at most 5 of 55 concepts have a leaky split", 0.60, "0b0ca0b", n <= 5,
                "; ".join(f"band {k}: {', '.join(v) or 'none'}" for k, v in pl["Q7"]["leaky"].items()))
        if "Q8" in pl and pl["Q8"]["median"] is not None:
            add("Pilot", "Q8: cos(leave-one-out, 194-word baseline) median < 0.9", 0.70, "0b0ca0b",
                pl["Q8"]["median"] < 0.9, f"median {r(pl['Q8']['median'], 2)}")
        if pl.get("SEC_crosscheck"):
            a = pl["SEC_crosscheck"]["agree_frac"]
            add("Pilot", "SEC cross-check agrees on ≥ 90% of cells", 0.85, "df56646", a is not None and a >= 0.9,
                f"agreement {pct(a, 0)} of {pl['SEC_crosscheck']['n']} cells")
        if pl.get("Q10", {}).get("by_band"):
            for pair, pr in (("12_vs_12r", 0.40), ("13_vs_13r", 0.25)):
                vals = [v["pairs"][pair] for v in pl["Q10"]["by_band"].values() if pair in v["pairs"]]
                if vals:
                    add("Step 7b", f"Q10: re-entry clamp {'to L62' if pair.startswith('12') else 'to L44'} beats its null",
                        pr, "c7cde05", any(v["specific"] for v in vals),
                        "; ".join(f"{b}: {r(v['pairs'][pair]['extra_reduction_mean'], 2, True)} nats" for b, v in
                                  pl["Q10"]["by_band"].items() if pair in v["pairs"]))
        if pl.get("Q11", {}).get("by_band"):
            add("Step 7b", "Q11: u⊥ becomes J-readable at the report position (≤ L62)", 0.55, "c7cde05",
                any(v["answer"] == "yes" for v in pl["Q11"]["by_band"].values()),
                "; ".join(f"{b}: {v['answer']}" + (f" (first at L{v['first_hit']['layer']})" if v.get("first_hit") else "")
                          for b, v in pl["Q11"]["by_band"].items()))
        if pl.get("Q12", {}).get("by_band"):
            q12 = pl["Q12"]["by_band"]
            flag = ("pre-registered relative rule (degenerate: fires at every dose, see A9); the absolute criterion says "
                    "the concept never reaches the report position. Excluded from the Brier score")
            add("Step 8", "Q12: the full vector reaches the report position by L44 (s = 1/4)", 0.50, "bf1da37",
                any(v["full_by_L44_at_1/4"] for v in q12.values()), flag, brier=False)
            add("Step 8", "Q12: the non-J part arrives no later than the J part (s = 1/4)", 0.60, "bf1da37",
                any(v["perp_no_later_than_J_at_1/4"] for v in q12.values()), flag, brier=False)
            add("Step 8", "Q12: something arrives by L44 inside the strict P11 window", 0.20, "bf1da37",
                any(v["window_any_by_L44"] for v in q12.values()), flag, brier=False)
        if pl.get("Q13"):
            bands = lambda hs: sorted({"L32–38" if h["band"] == "B" else "L24–38" for h in hs})
            add("Step 8", "Q13: the Oracle names the full vector at the report position (L48–60)", 0.45, "bf1da37",
                pl["Q13"]["full"]["answer"] == "yes",
                f"{len(pl['Q13']['full']['hits'])} qualifying cells, all at band {', '.join(bands(pl['Q13']['full']['hits'])) or '—'}; "
                "a few concepts carry them (see Q13)")
            add("Step 8", "Q13: the Oracle names the non-J part at the report position (L48–60)", 0.25, "bf1da37",
                pl["Q13"]["perp"]["answer"] == "yes",
                f"{len(pl['Q13']['perp']['hits'])} qualifying cells, all at band {', '.join(bands(pl['Q13']['perp']['hits'])) or '—'}; "
                "only two concepts, one with a leaky split (see Q13)")
        s9 = pl.get("step9")
        if s9 and s9.get("n_variants"):
            add("Step 9", "F1 (J vs non-J dose pattern) holds in ≥ 4 of 5 constructions", 0.55, "94be4ec",
                s9["R9a_holds_in"] >= 4, f"holds in {s9['R9a_holds_in']} of {s9['n_variants']}")
            add("Step 9", "F2 (clamp specific only to L62) holds in ≥ 4 of 5 constructions", 0.65, "94be4ec",
                s9["R9b_holds_in"] >= 4, f"holds in {s9['R9b_holds_in']} of {s9['n_variants']}")
        s10 = pl.get("step10")
        if s10 and s10.get("G10", {}).get("pass"):
            q14 = s10.get("Q14", {}).get("B")
            if q14:
                add("Step 10", "Q14: the clamp becomes specific only from L56–62 on (band L32–38)", 0.55, "3bfd2f0",
                    q14["category"] == "56-62",
                    f"E* = {('L' + str(q14['E_star'])) if q14['E_star'] else 'none'} ({q14['category']}); "
                    "other priors: 52–54 0.20, 46–50 0.15, none/other 0.10")
            q15 = s10.get("Q15", {})
            for b, nm, pr2, pr4 in (("B", "L32–38", 0.65, 0.35), ("B2", "L24–38", 0.70, 0.45)):
                if b in q15:
                    add("Step 10", f"Q15: the full vector's ΔR1′ is concept-specific (band {nm})", pr2, "3bfd2f0",
                        q15[b]["2"]["specific"], f"{q15[b]['2']['n_doses_meeting']}/9 doses meet the rule")
                    add("Step 10", f"Q15: the non-J part's ΔR1′ is concept-specific (band {nm})", pr4, "3bfd2f0",
                        q15[b]["4"]["specific"], f"{q15[b]['4']['n_doses_meeting']}/9 doses meet the rule")
            if "B" in q15 and "B2" in q15:
                add("Step 10", "Q15: the pure J-lens atom is concept-specific in both bands", 0.90, "3bfd2f0",
                    q15["B"]["1"]["specific"] and q15["B2"]["1"]["specific"],
                    f"{q15['B']['1']['n_doses_meeting']}/9 and {q15['B2']['1']['n_doses_meeting']}/9 doses")
                add("Step 10", "Q15: band L24–38 is at least as specific as L32–38 (A3 rule)", 0.60, "3bfd2f0",
                    q15["A3_B2_vs_B"]["B2_at_least_as_specific"], "full vector at s = 1/16 and 1/4")
                add("Step 10", "Q15: random directions stay at chance (ID ≤ 3/10 at every dose, both bands)", 0.90,
                    "3bfd2f0", q15["B"]["7"]["at_chance"] and q15["B2"]["7"]["at_chance"], "sanity check")
        s11 = pl.get("step11")
        if s11 and s11.get("consistency", {}).get("pass"):
            q16 = s11.get("Q16", {}).get("B")
            if q16:
                add("Step 11", "Q16: the J clamp beats the output-direction clamp at L48–52 (band L32–38)", 0.50, "6058e38",
                    q16["verdict"] == "J-specific", f"{q16['verdict']} ({q16['n_mid_D_above_0']} of 3 end layers)")
            g = s11.get("G11_by_reader", {})
            for n, nm, pr in (("oracle", "Oracle", 0.80), ("nla", "NLA", 0.55)):
                if g.get(n) is not None:
                    add("Step 11", f"G11: the {nm} identifies natural concept activations (≥ 7/10, a layer)", pr, "6058e38",
                        g[n], "positive control for the likelihood measure")
            v = s11.get("Q17", {}).get("verdicts", {})
            for n, nm in (("nla", "NLA"), ("oracle", "Oracle")):
                if n in v:
                    for cond, what, pr in (("4", "non-J part", {"nla": 0.35, "oracle": 0.45}[n]),
                                           ("2", "full vector", {"nla": 0.40, "oracle": 0.50}[n])):
                        x = v[n][cond]["1"]
                        add("Step 11", f"Q17: the {nm} is at least as sensitive as the model ({what}, 1-nat onset)", pr,
                            "6058e38", x["at_least_as_sensitive"],
                            "uninformative (failed G11)" if v[n]["uninformative"] else
                            f"reader {SNAME.get(x['reader_onset'], 'never')} vs model {SNAME.get(x['model_onset'], 'never')}")
    return items


def brier(items):
    res = [(it["prior"], 1.0 if it["outcome"] else 0.0) for it in items if it["outcome"] is not None and it.get("brier", True)]
    return (float(np.mean([(p - o) ** 2 for p, o in res])) if res else None), len(res)


# ------------------------------------------------------------------------------------------------ sections
def sec_why():
    return """
<h2 id="why">Why any of this was run</h2>
<p>The project asks whether activation readers (the NLA, the Oracle Lens, the J-lens) read what the model is actually
holding, re-solve the task from cues in the prompt, or see content the model itself cannot use or report. Today's work
was a sequence; each step was triggered by the one before it.</p>
<ol>
<li><b>Three cheap checks (C1–C3), no GPU.</b> Each could overturn a claim the spec leaned on: the one positive
"reader really reads" case (CW-7), the influence label at the ":" cell, and Cathryn's "survives removal" test.</li>
<li><b>Spec corrections (errata 01).</b> Wording that overstated results, and the claims the checks changed.</li>
<li><b>Engineering.</b> A loader for the NLA you chose (<code>nla-rl</code>), built exactly as WorkspaceBench runs it, and
the J-space pursuit that splits a concept into its J part and its non-J part.</li>
<li><b>Design 005 (planted concepts).</b> The one experiment where ground truth is known exactly: plant a concept into
Qwen, split into J and non-J parts, then ask whether the model reports each part and whether each reader detects it.
It tests the paper's central claim on Qwen (only J-space content is reportable) and the NLA's selling point (it sees
content outside J-space that the model can't report).</li>
<li><b>The smoke run.</b> Checks every part on the real model before an expensive run. Its pre-registered report gate
failed, so the design could not run as written.</li>
<li><b>The overnight pilot, dev concepts only.</b> Instead of leaving the GPU idle or spending the 45 held-out test
concepts on a design we already knew must change, the pilot answers the amendment's open questions (Q1–Q9), each with a
rule set before it ran.</li>
<li><b>Step 7b.</b> Three additions from the review you forwarded: a re-entry clamp, a layer-by-layer readout at the
report position, and the R-lens (Q10–Q11).</li>
<li><b>Step 8, the extra you asked for.</b> Where the planted concept actually reaches the report position: a reader-free
probe at every layer L32–62 and the Oracle at its later layers there (Q12–Q13), plus a power analysis for the full run.</li>
<li><b>Step 9, a second extra.</b> Whether the two headline findings (no clear J privilege at matched norm; the clamp is
specific only at the output end) survive five other ways of building J-space.</li>
<li><b>Step 10, a third extra.</b> Two checks aimed at your pending decisions: where along the depth the re-entry clamp
starts to act (A8), and whether the log-prob report measure singles out the planted concept rather than lifting every word
once Qwen's canned answer is displaced (A1, A3).</li>
<li><b>Step 11, run on your "run something extra!".</b> Two controls for step 10's caveats. Does the late clamp effect need
the J-lens coordinates, or does blocking the plain output direction do the same (A8)? And are the readers really less
sensitive than the model once both are measured by likelihood rather than naming (A6)?</li>
<li><b>The full run (5 October).</b> You adopted the amendment. I wrote the run spec, rehearsed the whole pipeline on two
dev concepts, committed the analysis code, and then ran the 45 held-out test concepts once (first section).</li>
</ol>
<p>Every run had a design spec and pre-registered priors committed before it started. No test-concept outcome has been
produced. The full 005 run waits for your decisions on the amendment.</p>
"""


def sec_checks(D):
    if not (D.c1 and D.c2 and D.c3):
        return ""
    c = D.c1["cells"]
    rows = []
    for key, lab in (("oracle_L42", "Oracle L42 (off-contract)"), ("nla_L42", "NLA L42 (training layer)"),
                     ("oracle_L56", "Oracle L56 (on-contract)"), ("nla_L56", "NLA L56 (off its layer)"),
                     ("jlens_L42", "J-lens L42"), ("jlens_L56", "J-lens L56")):
        w, a = c[key]["wsb"], c[key]["all60"]
        cw = c[key]["cw_any_number"]
        rows.append([lab, f"{a['pass']}/60 = {r(w['pass_rate'], 3)}", f"[{r(w['ci95'][0], 2)}, {r(w['ci95'][1], 2)}]",
                     r(w["null_top1_near"], 3), f"{cw['hits']}/60 ({r(cw['chance'], 1)})"])
    t1 = table(["Reader, layer", "Top computed value is an intermediate", "95% CI", "Decoy null", "Cathryn's any-number count (chance)"],
               rows, num_cols=(1, 2, 3, 4))
    A = D.c2["A_004_L42_full_sequence"]
    rows2 = [["a (recipient's answer)", r(A["delta_a_median"], 3, True), f"{A['delta_a_pos']}/33", "—"],
             ["a′ (partner's answer)", r(A["delta_a2_median"], 3, True), f"{A['delta_a2_pos']}/33", "—"],
             ["z (recipient's intermediate)", r(A["delta_z_median"], 3, True), f"{A['delta_z_pos']}/33", r(A["share_z_median"], 2)],
             ["z′ (partner's intermediate)", r(A["delta_z2_median"], 3, True), f"{A['delta_z2_pos']}/33", r(A["share_z2_median"], 2)],
             ["other 8 palette colours", r(A["delta_other_colour_median"], 3, True), f"{A['delta_other_colour_pos']}/33",
              r(A["share_other_colour_median"], 2)],
             ["non-colour tokens", r(A["delta_non_colour_median"], 3, True), f"{A['delta_non_colour_pos']}/33",
              r(A["share_non_colour_median"], 2)],
             ["z + z′", "—", "—", f"{r(A['share_z_plus_z2_median'], 2)} [{r(A['share_z_plus_z2_ci'][0], 2)}, {r(A['share_z_plus_z2_ci'][1], 2)}]"]]
    t2 = table(["Where the probability went", "Median change", "Pairs up", "Share of the off-pair increase"], rows2,
               num_cols=(1, 2, 3))
    S = D.c3["sets"]
    rows3 = []
    for k, lab in (("multihop_CW4", "Multihop CW-4, L42"), ("brew_CW9", "Brew CW-9, “:” L42")):
        x = S[k]
        rows3.append([lab, str(x["n_items"]), pct(x["r_strong_median"], 2), pct(x["r_rand_median"], 3),
                      f"{r(x['R_median'], 1)} [{r(x['R_ci'][0], 1)}, {r(x['R_ci'][1], 1)}]", f"{x['strong_gt_rand']}/{x['n_items']}"])
    t3 = table(["Set", "n", "Content removal (share of activation energy)", "Random control", "Ratio, median [CI]", "Content > control"],
               rows3, num_cols=(1, 2, 3, 4, 5))
    return f"""
<h2 id="checks">Earlier today: three checks before any new GPU work</h2>
<h3>C1. Does the Oracle really read chain intermediates? (CW-7)</h3>
<p>Cathryn's count credited any number appearing in a readout, and Oracle readouts carry 6–9 numbers each.
WorkspaceBench's stricter rule asks for the number the readout presents as <em>computed</em>, against a decoy null.
Judge: Gemini 3.8 Flash, 360 calls, $0.33.</p>
{t1}
<p><b>What it showed.</b> The Oracle result survives at about a third of its size and passes the pre-registered rule.
But J-lens, a linear readout that cannot re-solve anything, scores exactly the same 11/60 at L56, and the two pass on
mostly different items (4 of 11 shared). The NLA can't be tested on chain: at L42, its training layer, the final cell
carries no leverage yet (S ≈ 0.001 in experiment 001). The Oracle's 8/60 at that same no-leverage cell is a weak
re-solving candidate.</p>
<h3>C2. Is the influence label at ":" real, or is the patch breaking the model?</h3>
<p>Report 001 flagged whole-cell patches at ":" as damage. Exact 12-colour masses from experiment 004 (33 pairs, L42):</p>
{t2}
<p><b>What it showed.</b> The patched output becomes a mixture of the two trajectories: the answer moves to the
partner's, both intermediates rise (z is the recipient's runner-up answer in 23 of 33 pairs), and about 0.04 leaks to
non-colour tokens. "Influence" is kept at L42; a rewritten damage rule applies to future data only.</p>
<h3>C3. Were Cathryn's removal controls fair?</h3>
{t3}
<p><b>What it showed.</b> On multihop the control removes about 2.4× less of the activation than the content removal;
on Brew they are roughly matched. Both remove only 0.1–1% of the activation's energy, so "survives J-removal" can
never separate reading from re-solving. That takes an eraser shown to remove the information.</p>
"""


def dose_series(rows, grid, conds=(1, 2, 3, 4, 7), key="dR1p", agg="median"):
    return [{"name": COND[c][0], "cls": COND[c][1], "ys": curve(rows, c, key, grid, agg=agg)} for c in conds]


def sec_smoke(D):
    sm, S = D.sm_state, D.smoke
    if not sm:
        return ""
    g1 = sm.get("G1", [])
    rows_dec = []
    for g in g1:
        dc = g["decision"]["by_cond"]
        rows_dec.append([f"L{g['band'][0]}–{g['band'][1]}", f"{r(dc['1']['R2'], 2)} at s = {SNAME.get(dc['1']['best_s'])}",
                         f"{r(dc['2']['R2'], 2)} at s = {SNAME.get(dc['2']['best_s'])}",
                         chip("fail", "bad") if not g["decision"]["pass"] else chip("pass", "good")])
    t_dec = table(["Band", "Pure J-lens vector: best mean naming", "Full vector: best mean naming", "Gate"], rows_dec,
                  num_cols=(1, 2))
    G = D.sm_g1
    ser = dose_series(G, GRID6)
    band = null_curve(G, "dR1p", GRID6)
    fig1 = figure(line_chart(ser, GRID6, [SNAME[s] for s in GRID6], "median change in report log-prob (nats)",
                             band=band, aria="Smoke dose response at band L24-38"),
                  legend(ser, "random-direction 95th percentile"),
                  "Smoke, band L24–38, the 6 gate concepts (conditions 3 and 4 also run for the later checks). Median over "
                  "concepts of ΔR1′, the change in the concept's log-probability right after <code>The thought is about \"</code>. "
                  "The shaded area is what random directions reach (95th percentile over 5 seeds × 6 concepts).")
    rk = [{"name": COND[c][0], "cls": COND[c][1], "ys": curve(G, c, "R1", GRID6)} for c in (1, 2, 7)]
    fig2 = figure(line_chart(rk, GRID6, [SNAME[s] for s in GRID6], "median reciprocal rank of the concept",
                             ylim=(0, 0.55), aria="Smoke reciprocal rank"),
                  legend(rk), "The paper's own metric: median reciprocal rank of the concept at the open quote. 0.5 means the "
                              "concept is typically the second most likely next word. The clean model's top answer is "
                              "“apple” for every concept.")
    on = sm.get("onset", {})
    tr = on.get("trace", [])
    ser_on = [{"name": "non-J part (median ΔR1′)", "cls": "perp", "ys": [t["median_perp"] for t in tr]}]
    fig3 = figure(line_chart(ser_on, [t["s"] for t in tr], [SNAME[t["s"]] for t in tr], "ΔR1′ (nats)",
                             band=[t["null95"] for t in tr], aria="Smoke onset"),
                  legend(ser_on, "random-direction 95th percentile"),
                  "Onset on the 10 dev concepts (band L24–38). The non-J part is already above the random null at the "
                  "smallest dose, so the grid had no “model silent” strength. The pilot extended it down to 1/256.")
    e2e = D.sm_e2e
    er = []
    for c in ("gel", "earth"):
        row = [c]
        for cond in (1, 2, 3, 4, 5, 6, 7, 10, "10r"):
            x = [q for q in e2e if q["concept"] == c and str(q["cond"]) == str(cond) and q["s"] == 0.25 and q.get("seed", 0) == 0]
            row.append(r(x[0]["dR1p"], 1, True) if x else "—")
        er.append(row)
    t_e2e = table(["Concept"] + [COND[c][0] for c in (1, 2, 3, 4, 5, 6, 7, 10, "10r")], er,
                  num_cols=tuple(range(1, 10)))
    # readers (smoke)
    ts = on.get("text_strengths", [])
    rd = []
    for name, rows in (("Oracle", D.sm_oracle), ("NLA", D.sm_nla)):
        for q in rows:
            if q["cond"] == "clean" or q["concept"] is None:
                continue
            rd.append((name, q["layer"], q["kind"], q["concept"], q["cond"], q["s"],
                       float(np.mean([P.names(t, q["concept"]) for t in q["samples"]]))))
    rows_r = []
    for name, L in (("Oracle", 40), ("Oracle", 44), ("NLA", 42)):
        for kind in ("direct_cq", "carried_co"):
            for cond in (2, 3, 4, 7):
                vals = [x[6] for x in rd if x[0] == name and x[1] == L and x[2] == kind and x[4] == cond]
                rows_r.append([f"{name} L{L}", "direct" if kind == "direct_cq" else "report position",
                               COND[cond][0], heat(float(np.mean(vals)) if vals else None)])
    t_rd = table(["Reader", "Read", "Condition", "Mean naming (gel, earth; 3 doses × 4 samples)"], rows_r, num_cols=(3,))
    clean_rows = []
    for name in ("oracle", "nla"):
        for k, v in (S.get(f"{name}_clean") or {}).items():
            clean_rows.append([name.upper() if name == "nla" else "Oracle", k.replace("|", " · ").replace("carried_co", "report position").replace("direct_cq", "C_q (direct)"),
                               ", ".join(v["over_5pct"]) or "none", r(v["max"], 2)])
    t_clean = table(["Reader", "Cell", "Concepts named in > 5% of clean reads", "Highest clean rate"], clean_rows, num_cols=(3,))
    G0 = S.get("G0_inject", {}) if S else {}
    return f"""
<h2 id="smoke">The smoke run (19:50–20:17 UTC)</h2>
<p class="muted">Dev concepts only: 6 gate concepts, 10 for the onset, gel and earth end to end. Run <code>smoke1</code>,
code 4ce5c21. Descriptive; settles no prediction.</p>
<h3 id="smoke-g1">The report gate failed, and the paper's metric still works</h3>
<p>The gate asked whether Qwen <em>says</em> a planted concept after <code>Yes, I detect an injected thought. The thought is
about "</code>: mean naming ≥ 0.25 in 8 samples, at least 0.20 above random, without blurting the word early. It failed at
both bands:</p>
{t_dec}
<p>The planting itself works. By the paper's own metric, the pure J-lens vector lifts the concept to about the second
most likely word. But Qwen has a strong default answer at that position, and "apple" stays on top, so eight samples at
temperature 1 rarely draw the concept. Full concept vectors push the reply toward topic-like words ("ocean", "silence",
"safety") rather than the concept itself.</p>
{fig1}
{fig2}
<h3 id="smoke-onset">The non-J part acts from the smallest dose</h3>
{fig3}
<h3 id="smoke-e2e">Every condition on gel and earth (ΔR1′ in nats at s = 1/4)</h3>
{t_e2e}
<p>earth's non-J part drives the concept to rank 1 (named in 3 of 8 samples), but earth's split is leaky: its "non-J"
part is still J-readable as "earth" at several layers (split-quality check G5). This is an artifact of the split, not
evidence against privilege. At natural size, the full vector's effect sits in the non-J part (it carries about 94% of
the vector). On these two concepts the concept-specific clamp did no better than the random clamp.</p>
<h3 id="smoke-readers">Readers at the low onset doses ({", ".join(SNAME.get(s, str(s)) for s in ts)})</h3>
{t_rd}
<p>The text readers almost never name the concept at these doses. The Oracle names earth at L44 on direct reads of the
full vector and the J part at s = 1/8 (3 of 4 samples); the NLA names neither concept anywhere. Clean reads (no injection)
fix each reader's base rate on this prompt:</p>
{t_clean}
<p>The Oracle's clean read at the report position names "dog" in more than half its samples: it reads out the model's
candidate answers (cat, dog, apple…). The in-run screen G6 exists for exactly this.</p>
<h3 id="smoke-checks">Machinery</h3>
<ul>
<li>Writes are exact: {S.get('G0_inject', {}).get('n_layer_batches', '—') if S else '—'} injection batches with ρ between
{r(G0.get('rho_min'), 7)} and {r(G0.get('rho_max'), 7)} and κ ≥ {r(G0.get('kappa_min'), 7)}; clamp residual ≤
{r((S or {}).get('G0_clamp', {}).get('max_residual'), 7)}.</li>
<li>Determinism: teacher-forced outputs bitwise identical on rerun; samples identical with the same seed.</li>
<li>J share of a concept vector at k = 16: median {pct((S or {}).get('G3_J_share', {}).get('median'), 1)} (the paper reports 6–7% on Claude).</li>
<li>Probe validity: held-out-template AUC median {r((S or {}).get('G4', {}).get('loto_auc_median'), 3)}; label-permutation control median {r((S or {}).get('G4', {}).get('perm_ctrl_median'), 2)}.</li>
<li>The exact-precision SEC cross-check was skipped (the fallback band starts below its boundary); the pilot ran it at band L32–38.</li>
</ul>
"""


BLK = {"B_default": "band L32–38 · default prefill", "B_word": "band L32–38 · ‘word’ prefill",
       "B2_default": "band L24–38 · default prefill", "B2_word": "band L24–38 · ‘word’ prefill"}


def verdict(ans, yes="yes", no="no"):
    if ans is None:
        return '<span class="verdict pend">pending</span>'
    return f'<span class="verdict {"yes" if ans else "no"}">{esc(yes if ans else no)}</span>'


def qcard(qid, title, question, rule, prior, body, ans, interp):
    return f"""<div class="qcard" id="{qid.lower()}"><div class="qhead"><h3>{esc(qid)}. {esc(title)}</h3>{ans}</div>
<dl class="kv"><dt>Question</dt><dd>{question}</dd><dt>Rule (pre-registered)</dt><dd>{rule}</dd>
<dt>My prior</dt><dd>{prior}</dd></dl>{body}<p><b>What it means.</b> {interp}</p></div>"""


def sec_pilot(D, notes):
    pl = D.pilot
    if not pl or not D.pl_rows:
        return '<h2 id="pilot">The overnight pilot</h2><p class="pending">Pending: the pilot has not finished.</p>'
    out = ["""<h2 id="pilot">The overnight pilot (dev concepts only)</h2>
<p>10 dev concepts × 2 bands × 2 prefills × 9 doses (1/256 to 1) × every condition, teacher-forced, plus 8 sampled
answers for conditions 1–4 and 7. Readers on the default prefill. All on the calibration set; the 45 test concepts were
not touched. Each question below was registered with its rule and my prior before the pilot ran (design §16.1, commit
0b0ca0b).</p>"""]
    # ---- Q1
    figs = []
    for (band, pre), rows in sorted(D.pl_rows.items()):
        ser = dose_series(rows, GRID9)
        figs.append(figure(line_chart(ser, GRID9, [SNAME[s] for s in GRID9], "median ΔR1′ (nats)",
                                      band=null_curve(rows, "dR1p", GRID9), w=520, h=290,
                                      aria=f"Pilot dose response {band} {pre}"),
                           legend(ser, "random 95th pct"), f"{BLK[f'{band}_{pre}']}"))
    q1 = pl["Q1"]
    t = table(["Block", "Pure J-lens vector: doses above null", "Full vector: doses above null", "Rule met"],
              [[BLK[k], ", ".join(SNAME[s] for s in v["cond1_strengths_above_null"]) or "none",
                ", ".join(SNAME[s] for s in v["cond2_strengths_above_null"]) or "none",
                chip("yes", "good") if v["rule_met"] else chip("no", "bad")] for k, v in q1["by_block"].items()])
    out.append(qcard("Q1", "Is the paper's rank / log-prob report measure usable?",
                     "Where sampled naming failed, does the concept's log-probability at the report position respond to "
                     "planted concepts above what random directions do?",
                     "median ΔR1′ above the random-direction 95th percentile at ≥ 3 doses for the pure J-lens vector "
                     "<em>and</em> the full concept vector",
                     "0.90 yes", '<div class="smallmult">' + "".join(figs) + "</div>" + t,
                     verdict(q1["answer"] == "yes"), notes.get("Q1", "")))
    # ---- Q2
    figs2 = []
    for band in ("B", "B2"):
        ser = []
        for pre, dash in (("default", False), ("word", True)):
            rows = D.pl_rows.get((band, pre))
            if not rows:
                continue
            for c in (1, 2):
                ser.append({"name": f"{COND[c][0]} · {pre}", "cls": COND[c][1], "dash": dash,
                            "ys": curve(rows, c, "R2", GRID9, agg="mean")})
        if ser:
            figs2.append(figure(line_chart(ser, GRID9, [SNAME[s] for s in GRID9], "mean sampled naming (8 samples)",
                                           ylim=(0, max(0.3, max((y for s in ser for y in s["ys"] if y is not None), default=0) + 0.05)),
                                           w=520, h=270, aria=f"Naming {band}"),
                                legend(ser), f"band {'L32–38' if band == 'B' else 'L24–38'}: solid = default prefill, dashed = ‘word’ prefill"))
    q2 = pl["Q2"]
    t2 = table(["Block", "Pure J-lens vector: best naming (dose)", "Full vector: best naming (dose)"],
               [[BLK[k], f"{r(v['cond1_best_R2'], 2)} ({SNAME.get(v['cond1_best_s'], '—')})",
                 f"{r(v['cond2_best_R2'], 2)} ({SNAME.get(v['cond2_best_s'], '—')})"] for k, v in q2["by_block"].items()],
               num_cols=(1, 2))
    out.append(qcard("Q2", "Does the ‘word’ prefill remove the default-answer prior?",
                     "With <code>… The thought is about the word \"</code>, does Qwen actually <em>say</em> the planted concept?",
                     "under the ‘word’ prefill, mean naming at the best dose ≥ 0.25 for the pure J-lens or the full vector",
                     "0.30 yes", '<div class="smallmult">' + "".join(figs2) + "</div>" + t2,
                     verdict(q2["answer"] == "yes"), notes.get("Q2", "")))
    # ---- Q3
    q3 = pl.get("Q3", {}).get("by_prefill", {})
    if q3:
        t3 = table(["Prefill", "Band", "Full vector: best ΔR1′ (dose)", "Neutral-prompt KL there", "Pure J-lens vector: best ΔR1′", "Rule's choice"],
                   [[pre, "L32–38" if b == "B" else "L24–38", f"{r(v[b]['cond2_best_dR1p'], 2)} ({SNAME.get(v[b]['cond2_best_s'])})",
                     r(v[b]["B1_at_best"], 2), r(v[b]["cond1_best_dR1p"], 2), v["rule_choice"] if b == "B" else ""]
                    for pre, v in q3.items() for b in ("B", "B2")], num_cols=(2, 3, 4))
        out.append(qcard("Q3", "Which injection band?", "Does the wider band L24–38 give a clearly larger report effect than L32–38?",
                         "keep L32–38 unless L24–38's full-vector ΔR1′ is ≥ 1 nat larger without a larger neutral-prompt KL",
                         "0.50 that L24–38 is chosen", t3, verdict(any(v["rule_choice"] == "B2" for v in q3.values()),
                                                                  "L24–38", "L32–38"), notes.get("Q3", "")))
    # ---- Q4
    q4 = pl["Q4"]
    figs4 = []
    for k, v in q4["by_block"].items():
        tr = v["trace"]
        ser = [{"name": "non-J part (median ΔR1′)", "cls": "perp", "ys": [x["median_perp"] for x in tr]}]
        figs4.append(figure(line_chart(ser, [x["s"] for x in tr], [SNAME[x["s"]] for x in tr], "ΔR1′ (nats)",
                                       band=[x["null95"] for x in tr], w=520, h=260, aria=f"Onset {k}"),
                            legend(ser, "random 95th pct"),
                            f"{BLK[k]}: onset s* = {SNAME.get(v['s_star'], 'none')}; reader doses by the rule "
                            f"{', '.join(SNAME[s] for s in v['text_strengths_rule'])}"))
    out.append(qcard("Q4", "Where does the non-J part start to act?",
                     "On the extended grid, is there a dose below which the non-J part leaves the model's report untouched?",
                     "onset s* = the smallest dose where the median ΔR1′ of the non-J part exceeds the random 95th "
                     "percentile; useful if it sits above the grid floor (1/256)",
                     "0.60 above the floor in at least one band", '<div class="smallmult">' + "".join(figs4) + "</div>",
                     verdict(q4["answer"] == "yes"), notes.get("Q4", "")))
    return "".join(out)


READ_S = [1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0]


def reader_rates(reads):
    """{(band, layer, kind, cond, s): mean naming over concepts}, clean rates per (layer, kind)."""
    acc, clean = {}, {}
    for q in reads:
        if q["cond"] == "clean":
            clean[(q["layer"], q["kind"])] = {c: float(np.mean([P.names(t, c) for t in q["samples"]])) for c in P.CONCEPTS}
            continue
        k = (q.get("band"), q["layer"], q["kind"], str(q["cond"]), q["s"])
        acc.setdefault(k, []).append(float(np.mean([P.names(t, q["concept"]) for t in q["samples"]])))
    return {k: float(np.mean(v)) for k, v in acc.items()}, clean


def sec_pilot2(D, notes):
    pl = D.pilot
    if not pl or not D.pl_rows:
        return ""
    out = []
    # ---- Q5 readers
    rates_o, clean_o = reader_rates(D.pl_oracle)
    rates_n, clean_n = reader_rates(D.pl_nla)
    blocks_html = []
    for band in ("B", "B2"):
        rows = []
        for name, L, rates in (("NLA", 42, rates_n), ("Oracle", 40, rates_o), ("Oracle", 44, rates_o)):
            for kind, kl in (("direct_cq", "direct"), ("carried_co", "report position")):
                for cond in (2, 3, 4, 7):
                    rows.append([f"{name} L{L}", kl, COND[cond][0]] +
                                [heat(rates.get((band, L, kind, str(cond), s))) for s in READ_S])
        if any(rates_n) or any(rates_o):
            blocks_html.append(f"<h4>{'band L32–38' if band == 'B' else 'band L24–38'} · default prefill</h4>" +
                               table(["Reader", "Read", "Condition"] + [f"s = {SNAME[s]}" for s in READ_S], rows,
                                     num_cols=(3, 4, 5, 6, 7)))
    clean_rows = []
    for name, cl in (("NLA", clean_n), ("Oracle", clean_o)):
        for (L, kind), d in sorted(cl.items()):
            hi = sorted(((c, x) for c, x in d.items() if x > 0.05), key=lambda t: -t[1])
            clean_rows.append([f"{name} L{L}", "report position" if kind == "carried_co" else "C_q (direct base)",
                               ", ".join(f"{c} {r(x, 2)}" for c, x in hi) or "none"])
    t_clean = table(["Reader", "Cell", "Concepts named in > 5% of 32 clean samples (rate)"], clean_rows)
    # P11a-style preview: doses where the model's report of u_perp is unmoved
    prev = []
    for band in ("B", "B2"):
        rows = D.pl_rows.get((band, "default"))
        if not rows:
            continue
        nul = null_curve(rows, "dR1p", GRID9)
        perp = curve(rows, 4, "dR1p", GRID9)
        silent = [s for s, m, n in zip(GRID9, perp, nul) if m is not None and n is not None and m <= n and s in READ_S]
        for s in silent:
            prev.append(["L32–38" if band == "B" else "L24–38", SNAME[s],
                         heat(rates_n.get((band, 42, "carried_co", "4", s))), heat(rates_n.get((band, 42, "carried_co", "7", s))),
                         heat(rates_n.get((band, 42, "direct_cq", "4", s)))])
    t_prev = table(["Band", "Dose (model's report of the non-J part within the random null)", "NLA at report position: non-J",
                    "NLA at report position: random", "NLA direct: non-J"], prev, num_cols=(2, 3, 4)) if prev else \
        "<p class='muted'>No reader dose fell where the model's report of the non-J part was within the random null.</p>"
    p17 = []
    for band in ("B", "B2"):
        for s in READ_S:
            nla = rates_n.get((band, 42, "direct_cq", "4", s))
            ors = [rates_o.get((band, L, "direct_cq", "4", s)) for L in (40, 44)]
            ors = [x for x in ors if x is not None]
            if nla is None and not ors:
                continue
            p17.append(["L32–38" if band == "B" else "L24–38", SNAME[s], heat(nla), heat(max(ors) if ors else None),
                        heat(rates_n.get((band, 42, "direct_cq", "2", s))),
                        heat(max([x for x in (rates_o.get((band, L, "direct_cq", "2", s)) for L in (40, 44)) if x is not None], default=None))])
    t_p17 = table(["Band", "Dose", "NLA L42: non-J", "Oracle best of L40/L44: non-J", "NLA L42: full", "Oracle best: full"],
                  p17, num_cols=(2, 3, 4, 5)) if p17 else ""
    tpz = pl.get("token_probe") or {}
    tp_rows = []
    for kind, kl in (("direct_cq", "direct"), ("carried_co", "report position"), ("injected_cq", "injected position")):
        for cond in ("1", "2", "3", "4", "7"):
            cells = []
            for sv in READ_S:
                v = tpz.get(f"B|{kind}|42|{cond}|{sv}")
                cells.append(f'{heat(v.get("jnp10") if v else None)} {heat(v.get("probe_abs") if v else None)}')
            tp_rows.append([kl, COND[int(cond)][0]] + cells)
    t_tp = table(["Read (L42, band L32–38)", "Condition"] + [f"s = {SNAME[sv]}: J_NP top-10 · probe says “c”" for sv in READ_S],
                 tp_rows, num_cols=(2, 3, 4, 5, 6)) if tpz else ""
    probe_note = ("<p>Two probe criteria. The pre-registered one (above the strongest of five random directions at the same "
                  "dose) fires for almost every concept at every dose, even 1/256. That is a property of the rule: the forward "
                  "pass is deterministic and nearly linear at small doses, so a linear probe's response to the concept "
                  "direction and to a random direction shrink together and the comparison is scale-free. The table shows the "
                  "post-hoc absolute criterion instead: the probe classifies the read as the concept (decision above 0, the "
                  "threshold calibrated on natural sentences). By that standard the concept is plainly present on direct reads "
                  "and at the injected position at s ≥ 1/4, and <b>never at the report position at L42</b>.</p>")
    out.append(qcard("Q5", "Do the readers see the planted concept, and where?",
                     "Naming rates of the planted concept by the NLA and the Oracle, read directly at the written cell and at "
                     "the report position, against their random-direction and clean-read rates (a preview of P11 and P17, not a test).",
                     "descriptive; the registered expectation was that the NLA names the concept at the report position for the "
                     "full vector at some dose (mean over dev concepts > 0.10)",
                     "0.40 at the report position; 0.80 on direct reads",
                     "".join(blocks_html) + "<h4>Token reader and reader-free probe at L42</h4>" + probe_note + t_tp +
                     "<h4>Base rates on this prompt (no injection)</h4>" + t_clean +
                     "<h4>Preview of P11a: doses where the model's report of the non-J part is unmoved</h4>" + t_prev +
                     "<h4>Preview of P17: NLA vs Oracle on direct reads</h4>" + t_p17,
                     verdict(None) if not (rates_n or rates_o) else verdict(
                         any(v > 0.10 for k, v in rates_n.items() if k[2] == "carried_co" and k[3] == "2")),
                     notes.get("Q5", "")))
    # ---- Q6 standard clamp
    q6 = pl["Q6"]["by_block"]
    t6 = table(["Block", "Concepts evaluable", "Extra reduction by the c-clamp over its random null (nats)", "95% CI", "Beats null"],
               [[BLK[k], str(v["n_evaluable"]), r(v["extra_reduction_dR1p_mean"], 2, True),
                 f"[{r(v['ci'][0], 2)}, {r(v['ci'][1], 2)}]", chip("yes", "good") if v["distinguishable"] else chip("no", "bad")]
                for k, v in q6.items()], num_cols=(1, 2, 3))
    figs6 = []
    for (band, pre), rows in sorted(D.pl_rows.items()):
        if pre != "default":
            continue
        ser = [{"name": COND[c][0], "cls": COND[c][1], "dash": str(c).endswith("r"), "ys": curve(rows, c, "dR1p", GRID9)}
               for c in (4, 10, "10r")]
        figs6.append(figure(line_chart(ser, GRID9, [SNAME[s] for s in GRID9], "median ΔR1′ (nats)", w=520, h=260,
                                       aria=f"Clamp {band}"), legend(ser), BLK[f"{band}_{pre}"]))
    out.append(qcard("Q6", "Does the design's clamp beat its random null?",
                     "The paper's privilege clamp holds the concept's J coordinates at their clean values at every layer. "
                     "Does that remove more of the non-J part's report effect than clamping a same-size random token set?",
                     "at the non-J part's best dose per concept, the c-clamp lowers ΔR1′ more than the random clamp; bootstrap 95% CI "
                     "over evaluable dev concepts excludes 0", "0.25 yes",
                     '<div class="smallmult">' + "".join(figs6) + "</div>" + t6,
                     verdict(any(v["distinguishable"] for v in q6.values())), notes.get("Q6", "")))
    # ---- Q7 / Q8
    q7 = pl.get("Q7", {})
    lk = q7.get("leaky") or {}
    g5 = (D.pl_state.get("split_quality") or {}).get("g5_rank", {})
    leaky_all = sorted(set(c for v in lk.values() for c in v))
    rows7 = [[c] + [str(g5.get(c, {}).get(str(l), "—")) for l in range(24, 39)] for c in leaky_all]
    t7 = table(["Concept"] + [f"L{l}" for l in range(24, 39)], rows7, num_cols=tuple(range(1, 16))) if rows7 else ""
    out.append(qcard("Q7", "Which concepts have a leaky split?",
                     "For all 55 concepts: is the concept's own J-lens direction still readable inside its “non-J” part?",
                     "leaky = the concept is in J_NP's top 10 on its pure non-J part at ≥ 2 layers of the band in use (fixed before computing)",
                     "0.60 that at most 5 of 55 are leaky",
                     "<p>" + "; ".join(f"band {'L32–38' if b == 'B' else 'L24–38'}: <b>{', '.join(v) or 'none'}</b>" for b, v in lk.items()) +
                     "</p>" + ("<p class='muted'>J_NP rank of each leaky concept on its own non-J part, by layer:</p>" + t7 if t7 else ""),
                     verdict(max((len(v) for v in lk.values()), default=0) <= 5), notes.get("Q7", "")))
    q8 = pl.get("Q8", {})
    js = pl.get("J_share_all55", {})
    out.append(qcard("Q8", "Does the concept-vector baseline matter?",
                     "Concept vectors subtract the mean over the other 100 protocol concepts (leave-one-out, primary). How "
                     "different are they from vectors built with the 194 category words instead?",
                     "report the cosine; below 0.9 the 194-word version is mandatory as a sensitivity check",
                     "0.70 that the median is below 0.9",
                     f"<p>Median cosine {r(q8.get('median'), 3)} (lowest {r(q8.get('min'), 2)}) over 55 concepts × 18 layers. "
                     f"J share of the concept vector across the same set: median {pct(js.get('median'), 1)} "
                     f"(range {pct(js.get('min'), 1)}–{pct(js.get('max'), 1)}).</p>",
                     verdict((q8.get("median") or 1) < 0.9), notes.get("Q8", "")))
    return "".join(out)


def sec_7b(D, notes):
    pl = D.pilot or {}
    q10, q11 = pl.get("Q10"), pl.get("Q11")
    if not q10:
        return '<h2 id="step7b">Step 7b</h2><p class="pending">Pending: step 7b runs after the pilot.</p>'
    out = ["""<h2 id="step7b">Step 7b: re-entry clamp, layer sweep, R-lens</h2>
<p>Three additions from the review you forwarded, registered before they ran (design §16.3, commit c7cde05). Dev concepts,
default prefill, both bands, teacher-forced.</p>"""]
    rows, figs = [], []
    for band, v in q10["by_band"].items():
        for pair, lab in (("10_vs_10r", "clamp to clean, to L62 (the design's)"), ("12_vs_12r", "re-entry clamp, to L62"),
                          ("13_vs_13r", "re-entry clamp, to L44")):
            x = v["pairs"].get(pair)
            if not x:
                continue
            cl = x.get("clamp_logs", {})
            rows.append(["L32–38" if band == "B" else "L24–38", lab, str(x["n_evaluable"]),
                         r(x["extra_reduction_mean"], 2, True), f"[{r(x['ci'][0], 2)}, {r(x['ci'][1], 2)}]",
                         pct(cl.get("removed_share_band"), 1), pct(cl.get("removed_share_above"), 1),
                         chip("yes", "good") if x["specific"] else chip("no", "bad")])
        cv = v.get("dR1p_curves", {})
        ser = [{"name": COND[int(c) if c.isdigit() else c][0], "cls": COND[int(c) if c.isdigit() else c][1],
                "dash": c.endswith("r"), "ys": [cv[c].get(str(s)) for s in GRID9]} for c in ("4", "10", "10r", "12", "12r", "13", "13r") if c in cv]
        figs.append(figure(line_chart(ser, GRID9, [SNAME[s] for s in GRID9], "median ΔR1′ (nats)", w=520, h=280,
                                      aria=f"Re-entry clamp {band}"), legend(ser),
                           f"{'band L32–38' if band == 'B' else 'band L24–38'}; dashed = the random-token null of each clamp"))
    t10 = table(["Band", "Clamp", "n", "Extra reduction vs null (nats)", "95% CI", "Removed (band)", "Removed (above)",
                 "Beats null"], rows, num_cols=(0, 2, 3, 4, 5, 6))
    cosd = pl.get("cos_atom_vs_output") or {}
    cos_fig = ""
    if cosd:
        layers = list(range(24, 63))
        m = [med([cosd[c].get(str(l)) for c in cosd]) for l in layers]
        ser = [{"name": "median over dev concepts", "cls": "j", "ys": m}]
        cos_fig = figure(line_chart(ser, layers, [str(l) if l % 4 == 0 else "" for l in layers],
                                    "cos(J atom of c, unembedding of c)", ylim=(0, 1), xlabel="layer", w=560, h=240,
                                    aria="Atom vs output direction"), legend(ser),
                         "How close the concept's J-lens direction at each layer is to its output (unembedding) direction. "
                         "Where this approaches 1, clamping the J coordinate is close to clamping the answer itself.")
    ok10 = [x["specific"] for v in q10["by_band"].values() for k, x in v["pairs"].items() if k != "10_vs_10r"]
    out.append(qcard("Q10", "Does a re-entry clamp isolate what the model writes back into J?",
                     "Clamping c's J coordinates to <em>clean + the raw injection</em> removes only what the model's own blocks "
                     "write into them (the re-entry the paper's clamp targets) and keeps the injection itself. Does that lower "
                     "the non-J part's report effect more than the same clamp on random tokens? Run to L62 (as the design) and "
                     "only to L44 (where c's J direction is still far from its output direction).",
                     "at the non-J part's best dose per concept, the c-clamp lowers ΔR1′ more than its random null; bootstrap "
                     "95% CI over evaluable dev concepts excludes 0", "0.40 for the L62 version, 0.25 for the L44 version",
                     '<div class="smallmult">' + "".join(figs) + "</div>" + t10 + cos_fig,
                     verdict(any(ok10)) if ok10 else verdict(None), notes.get("Q10", "")))
    if q11:
        sc = pl.get("sweep_curves", {})
        figs11 = []
        for band, rd in sc.items():
            s_pick = notes.get("sweep_s", 0.25)
            for reader in ("J_NP", "logit", "R_CB"):
                cr = rd.get(reader, {})
                ser = []
                for c in (2, 3, 4, 7):
                    k = f"{c}|{s_pick}"
                    if k in cr:
                        ser.append({"name": COND[c][0], "cls": COND[c][1], "ys": [cr[k].get(str(l)) for l in range(32, 63)]})
                if "clean|0.0" in cr:
                    ser.append({"name": "clean", "cls": "rand", "dash": True, "ys": [cr["clean|0.0"].get(str(l)) for l in range(32, 63)]})
                if ser:
                    figs11.append(figure(line_chart(ser, list(range(32, 63)), [str(l) if l % 4 == 0 else "" for l in range(32, 63)],
                                                    f"{reader} rank of c (log)", ylog=True, invert=True, xlabel="layer",
                                                    ylim=(1, 2e5),
                                                    w=520, h=270, aria=f"Sweep {band} {reader}"), legend(ser),
                                         f"{'band L32–38' if band == 'B' else 'band L24–38'} · {reader} · dose {SNAME.get(s_pick)} · "
                                         "median rank over dev concepts at the report position (rank 1 at the top)"))
        rows11 = [["L32–38" if b == "B" else "L24–38", ", ".join(SNAME[s] for s in v["strengths_evaluable"]) or "none",
                   v["answer"], (f"L{v['first_hit']['layer']} at dose {SNAME.get(v['first_hit']['s'])} (median rank "
                                 f"{r(v['first_hit']['median_rank'], 0)})") if v.get("first_hit") else "—"]
                  for b, v in q11["by_band"].items()]
        out.append(qcard("Q11", "Does the non-J effect become J-readable at the report position?",
                         "Track the concept's rank at the report position (the open quote) at every layer L32–62, under the "
                         "full vector, the J part, the non-J part and random directions.",
                         "under the non-J part, at doses where it moves the report above the null, the median J_NP rank of c reaches "
                         "the top 10 at some layer and beats the random 5th-percentile rank there", "0.55 yes",
                         '<div class="smallmult">' + "".join(figs11) + "</div>" +
                         table(["Band", "Doses where the non-J part moves the report", "Readable?", "First layer"], rows11),
                         verdict(any(v["answer"] == "yes" for v in q11["by_band"].values())), notes.get("Q11", "")))
    rc = pl.get("rlens_consistency")
    if rc:
        out.append(f"""<h3 id="rlens">R-lens next to J-lens and logit lens</h3>
<p>Ranks of the planted concept on {rc['n']:,} pilot reads (direct, report position and injected position; L40/42/44):
Spearman correlation J_NP vs R_CB {r(rc['spearman_JNP_RCB'], 2)}, R_CB vs logit lens {r(rc['spearman_RCB_logit'], 2)};
top-10 agreement J_NP vs R_CB {pct(rc['top10_agree_JNP_RCB'], 0)}. Share of reads with the concept in the top 10:
J_NP {pct(rc['top10_rate']['J_NP'], 1)}, R_CB {pct(rc['top10_rate']['R_CB'], 1)}, logit lens {pct(rc['top10_rate']['logit'], 1)}.</p>
<p>{notes.get('rlens', '')}</p>""")
    return "".join(out)


def sec_8(D, notes):
    pl = D.pilot or {}
    q12, q13, pw = pl.get("Q12"), pl.get("Q13"), pl.get("power_8b")
    if not q12:
        return '<h2 id="step8">Step 8</h2><p class="pending">Pending: step 8 runs after step 7b.</p>'
    out = ["""<h2 id="step8">Step 8: where the planted concept reaches the report position</h2>
<p>Extra run you asked for (design §16.4, commit bf1da37). A reader-free probe for each dev concept, trained on natural
sentences at every layer L32–62, read at the open quote; plus the Oracle at its later on-contract layers there. The NLA can
only read L42, so this decides whether P11 at the report position is testable for it at all.</p>"""]
    figs = []
    for band, v in q12["by_band"].items():
        for sv in (1 / 16, 1 / 4, 1.0):
            ser = []
            for c in (2, 3, 4):
                cr = v["curves"].get(f"{c}|{sv}", {})
                ser.append({"name": COND[c][0], "cls": COND[c][1], "ys": [cr.get(str(l), cr.get(l)) for l in range(32, 63)]})
            ca = v.get("curves_absolute", {})
            for c in (2, 3, 4):
                cr = ca.get(f"{c}|{sv}", {})
                if cr:
                    ser.append({"name": COND[c][0] + " · probe says “c”", "cls": COND[c][1], "dash": True,
                                "ys": [cr.get(str(l), cr.get(l)) for l in range(32, 63)]})
            figs.append(figure(line_chart(ser, list(range(32, 63)), [str(l) if l % 4 == 0 else "" for l in range(32, 63)],
                                          "share of dev concepts detected", ylim=(0, 1), xlabel="layer (report position)",
                                          w=520, h=250, aria=f"Arrival {band} {sv}"), legend(ser),
                               f"{'band L32–38' if band == 'B' else 'band L24–38'} · dose {SNAME.get(sv)} · "
                               f"solid: above the strongest of 5 random directions (pre-registered); dashed: the probe classifies the read "
                               f"as the concept (post-hoc absolute criterion). The NLA reads at L42."))
    rows = []
    for band, v in q12["by_band"].items():
        for sv in READ_S:
            rows.append(["L32–38" if band == "B" else "L24–38", SNAME[sv],
                         *[(f"L{v['first_arrival'].get(f'{c}|{sv}')}" if v["first_arrival"].get(f"{c}|{sv}") else "never")
                           for c in (2, 3, 4)],
                         "yes" if sv in v["strict_P11_window_doses"] else "no"])
    t = table(["Band", "Dose", "Full: first layer ≥ 50% detected", "J part", "non-J part",
               "Inside the strict P11 window (model's report unmoved)"], rows)
    ans = [v["full_by_L44_at_1/4"] for v in q12["by_band"].values()]
    out.append(qcard("Q12", "When does the concept reach the report position?",
                     "Reader-free: at which layer does concept information first show up at the open quote, for the full "
                     "vector, its J part and its non-J part?",
                     "first layer where at least half of the dev concepts are probe-detected above the strongest of 5 random "
                     "directions", "0.50 that the full vector arrives by L44 at s = 1/4; 0.60 that the non-J part arrives no "
                     "later than the J part; 0.20 that anything arrives by L44 inside the strict P11 window",
                     '<div class="smallmult">' + "".join(figs) + "</div>" + t,
                     verdict(any(ans)) if ans else verdict(None), notes.get("Q12", "")))
    if q13:
        rate = q13["rates"]
        rows = []
        for band in ("B", "B2"):
            for L in (48, 52, 56, 60):
                for cond in ("2", "3", "4", "7"):
                    rows.append([f"L{L}", "L32–38" if band == "B" else "L24–38", COND[int(cond)][0]] +
                                [heat(rate.get(f"{band}|{L}|carried_co|{cond}|{sv}")) for sv in READ_S])
        out.append(qcard("Q13", "Does the Oracle see the concept at the report position at its later layers?",
                         "Oracle naming at the open quote at L48, L52, L56 and L60 (all on its contract).",
                         "mean naming over dev concepts above 0.10 and above random directions at some dose",
                         "0.45 for the full vector; 0.25 for the non-J part",
                         table(["Layer", "Band", "Condition"] + [f"s = {SNAME[sv]}" for sv in READ_S], rows,
                               num_cols=(3, 4, 5, 6, 7)) +
                         "<p class='muted'>Clean-read base rates above 5%: " + "; ".join(f"{k}: {', '.join(v) or 'none'}"
                                                                                          for k, v in q13["clean_over_5pct"].items()) + "</p>",
                         verdict(q13["full"]["answer"] == "yes"), notes.get("Q13", "")))
    if pw:
        rows = [[k.replace("_", " "), r(v.get("dev_mean", v.get("dev_mean_rate")), 3),
                 r(v.get("dev_sd", v.get("obs_sd")), 3), r(v.get("ci_halfwidth_n45", v.get("ci_halfwidth_n45_k4")), 3),
                 r(v.get("ci_halfwidth_n45_k8"), 3)] for k, v in pw.items()]
        out.append('<h3 id="power">Power for the full run (planning, CPU)</h3><p>Between-concept spread on the 10 dev '
                   'concepts, scaled to 45 test concepts. For text readers, the 8-sample column removes half the '
                   'within-concept binomial noise.</p>' +
                   table(["Contrast (dev preview)", "Dev mean", "Dev SD", "Predicted 95% CI half-width, n = 45 (4 samples)",
                          "… with 8 samples"], rows, num_cols=(1, 2, 3, 4)) + f"<p>{notes.get('power', '')}</p>")
    return "".join(out)


def sec_9(D, notes):
    s9 = (D.pilot or {}).get("step9")
    if not s9:
        return ""
    names = {"k10": "k = 10", "k25": "k = 25", "ls": "LS refit (span)", "raw": "raw-W_U atoms", "base194": "194-word baseline"}
    rows = []
    for k, v in s9["variants"].items():
        d = v["J_minus_perp_by_dose"]
        c12, c13 = v["clamp"]["12"], v["clamp"]["13"]
        rows.append([names.get(k, k), pct((s9.get("J_share_median") or {}).get(k), 1),
                     " / ".join(r(d[str(x)] if str(x) in d else d.get(x), 2, True) for x in (1 / 16, 1 / 8)),
                     " / ".join(r(d[str(x)] if str(x) in d else d.get(x), 2, True) for x in (1 / 2, 1.0)),
                     f"{r(v['peaks']['J'], 2)} / {r(v['peaks']['perp'], 2)}",
                     chip("holds", "good") if v["R9a_pattern_holds"] else chip("does not hold", "bad"),
                     f"{r(c12['mean'], 2, True)} [{r(c12['ci'][0], 2)}, {r(c12['ci'][1], 2)}]",
                     f"{r(c13['mean'], 2, True)} [{r(c13['ci'][0], 2)}, {r(c13['ci'][1], 2)}]",
                     chip("holds", "good") if v["R9b_pattern_holds"] else chip("does not hold", "bad")])
    t = table(["J-space construction", "J share (median)", "J − non-J, s = 1/16 / 1/8", "J − non-J, s = 1/2 / 1",
               "Peak ΔR1′, J / non-J", "F1 pattern", "Re-entry clamp to L62 vs null", "to L44 vs null", "F2 pattern"],
              rows, num_cols=(1, 2, 3, 4, 6, 7))
    return f"""<h2 id="step9">Step 9: are the two headline findings robust to how J-space is built?</h2>
<p>Pre-registered before it ran (design §16.5, commit 94be4ec). F1: at matched norm the J part beats the non-J part at middle
doses and loses at high doses. F2: the re-entry clamp is specific only when it reaches L62. Both were re-run under five
alternative constructions (band L32–38, dev concepts). The primary construction is k = 16, NNLS refit, folded atoms,
leave-one-out baseline.</p>
{t}
<p>F1 holds in <b>{s9['R9a_holds_in']} of {s9['n_variants']}</b> variants (my prior for ≥ 4 of 5: 0.55); F2 holds in
<b>{s9['R9b_holds_in']} of {s9['n_variants']}</b> (prior for ≥ 4 of 5: 0.65).</p>
<p>{notes.get('step9', '')}</p>"""


def sec_10(D, notes):
    s10 = (D.pilot or {}).get("step10")
    if not s10:
        return ""
    bn = {"B": "L32–38", "B2": "L24–38"}

    def small(x):
        return "—" if x is None else (f"{x:.1e}" if 0 < abs(x) < 1e-3 else r(x, 3))

    def ci_txt(m, ci, nd=2):
        return f"{r(m, nd, True)} [{r(ci[0], nd)}, {r(ci[1], nd)}]"

    out = ["""<h2 id="step10">Step 10: where does the clamp start to act, and is the report measure specific to the planted concept?</h2>
<p>Pre-registered before it ran (design §16.6, commit 3bfd2f0). The analysis code and one clarification (the SM statistic
is the mean over concepts) were committed while it ran, before any result was read (999fdd8). Two of your amendment
decisions rested on evidence with a gap in it.</p>
<ul><li><b>A8.</b> The re-entry clamp is specific when it runs to L62 and not when it stops at L44. Where between the two
does it start to act?</li>
<li><b>A1/A3.</b> The log-prob measure beats random directions. Does it single out the planted concept, or does displacing
Qwen's canned answer raise every word? And is the wider band's larger effect specific?</li></ul>
<p>Dev concepts only, teacher-forced, default prefill, both bands.</p>"""]
    g = s10.get("G10", {})
    gl = {"spec_B": "10b rows, band L32–38", "spec_B2": "10b rows, band L24–38",
          "clamp_B": "10a rows (unclamped, E = 44, E = 62), band L32–38",
          "clamp_B2": "10a rows (unclamped, E = 44, E = 62), band L24–38"}
    grows = [[gl.get(k, k), str(v["n"]), small(v["median_abs"]), small(v["p95_abs"]), small(v["max_abs"])]
             for k, v in g.get("checks", {}).items()]
    out.append('<h3 id="g10">Gate G10: repeated rows reproduce the stored ones</h3>'
               + table(["Rows repeated from the pilot / step 7b", "n", "Median |difference| in ΔR1′ (nats)", "95th pct", "Max"],
                       grows, num_cols=(1, 2, 3, 4))
               + f"<p>{chip('G10 passes', 'good') if g.get('pass') else chip('G10 fails', 'bad')} Rule: {esc(g.get('rule', ''))}. "
               "The rows ran in different batches from the originals, so bitwise equality was not required.</p>")
    q14 = s10.get("Q14", {})
    if q14:
        Es = list(range(44, 63, 2))
        ser = [{"name": f"band {bn[b]}", "cls": cls, "ys": [q14[b]["by_E"][str(E)]["mean"] for E in Es]}
               for b, cls in (("B", "re"), ("B2", "cl")) if b in q14]
        svg = line_chart(ser, Es, [f"L{E}" for E in Es], "extra reduction (nats)",
                         xlabel="clamp end layer E (the clamp runs from the band's first layer to E)",
                         aria="Extra reduction of the non-J part's report effect by the concept clamp over its random null, by clamp end layer")
        rows = []
        for E in Es:
            any_b = next(iter(q14.values()))["by_E"][str(E)]
            row = [f"L{E}", r(any_b.get("cos_atom_vs_output_median"), 2)]
            for b in ("B", "B2"):
                if b in q14:
                    x = q14[b]["by_E"][str(E)]
                    row += [ci_txt(x["mean"], x["ci"]), chip("specific", "good") if x["specific"] else chip("no", "bad")]
            rows.append(row)
        head = ["Clamp end layer E", "cos(J atom, output direction), median"]
        for b in ("B", "B2"):
            if b in q14:
                head += [f"Band {bn[b]}: extra reduction [95% CI]", "Specific?"]
        verdicts = "".join(
            f"<li>Band {bn[b]}: E* = <b>{('L' + str(v['E_star'])) if v['E_star'] else 'none'}</b> "
            f"(category {esc(v['category'])}; n = {len(v['evaluable'])} evaluable concepts). Pre-registered mapping: "
            f"{esc(v['amendment'])}.</li>" for b, v in q14.items())
        out.append(f"""<h3 id="q14">Q14. Where along the depth does the re-entry clamp start to block the non-J part's effect?</h3>
<p>The clamp removes what the model's own blocks write into the concept's J coordinates, from the band's first layer up to
an end layer E. The comparison is with a clamp on as many random tokens' coordinates. "Extra reduction" is how much more
the concept clamp lowers the non-J part's ΔR1′ than the random clamp does, at each concept's best dose (the Q10 measure).
E = 44 and E = 62 repeat step 7b. <b>E*</b> is the first end layer from which the clamp is specific at every later end layer.
Agent priors (band L32–38): E* in 56–62 0.55; 52–54 0.20; 46–50 0.15; none or other 0.10.</p>
{figure(svg, legend(ser), "Mean over evaluable dev concepts of ΔR1′(random-token clamp) − ΔR1′(concept clamp), non-J part injected. Above 0: the concept clamp blocks more than the random one.")}
{table(head, rows, num_cols=(1, 2, 4))}
<ul>{verdicts}</ul>
<p>{notes.get('Q14', '')}</p>""")
    q15 = s10.get("Q15", {})
    if q15.get("B") or q15.get("B2"):
        cn = {"1": "pure J-lens atom", "2": "full concept vector", "3": "J part", "4": "non-J part", "7": "random direction"}
        cls = {"1": "jl", "2": "full", "3": "j", "4": "perp", "7": "rand"}
        grid = [1 / 256, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
        figs = []
        for b in ("B", "B2"):
            if b not in q15:
                continue
            ser = [{"name": cn[c], "cls": cls[c], "dash": c == "7", "ys": [q15[b][c]["by_dose"][str(x)]["ID_rate"] for x in grid]}
                   for c in ("1", "2", "3", "4", "7")]
            svg = line_chart(ser, grid, [SNAME.get(x, str(x)) for x in grid], "share identified", ylim=(0, 1),
                             band=[0.1] * len(grid), aria=f"Identification rate by dose, band {bn[b]}")
            figs.append(figure(svg, legend(ser, extra="chance (1 in 10)"),
                               f"Band {bn[b]}: share of dev concepts whose own ΔR1′ is the largest among the 10 dev concepts "
                               "(for random directions: over 50 concept × seed rows)."))
        rows = []
        for b in ("B", "B2"):
            if b not in q15:
                continue
            for c in ("1", "2", "3", "4", "7"):
                v = q15[b][c]
                d16, d4 = v["by_dose"][str(1 / 16)], v["by_dose"][str(0.25)]
                verdict = ((chip("at chance", "good") if v.get("at_chance") else chip("above chance", "bad")) if c == "7"
                           else (chip("specific", "good") if v["specific"] else chip("not specific", "bad")))
                rows.append([bn[b], cn[c], "—" if c == "7" else f"{v['n_doses_meeting']}/9", verdict,
                             f"{pct(d16['ID_rate'])} · {ci_txt(d16['SM_mean'], d16['SM_ci'])}",
                             f"{pct(d4['ID_rate'])} · {ci_txt(d4['SM_mean'], d4['SM_ci'])}",
                             r(d4["generic_median"], 2, True), r(d4["dR1p_median"], 2, True)])
        t = table(["Band", "Injected", "Doses meeting the rule", "Verdict", "s = 1/16: ID · SM [95% CI]",
                   "s = 1/4: ID · SM [95% CI]", "Other words' rise, s = 1/4 (median)", "Own ΔR1′, s = 1/4 (median)"],
                  rows, num_cols=(2, 6, 7))
        a3 = q15.get("A3_B2_vs_B")
        a3_txt = ""
        if a3:
            a3_rows = [[SNAME.get(float(k), k), f"{r(v['SM_mean_B'], 2, True)} / {r(v['SM_mean_B2'], 2, True)}",
                        f"{pct(v['ID_B'])} / {pct(v['ID_B2'])}", chip("yes", "good") if v["B2_ge_B"] else chip("no", "bad")]
                       for k, v in a3["by_dose"].items()]
            a3_txt = ("<h4>A3: is the wider band's larger effect specific?</h4>"
                      + table(["Dose", "SM, L32–38 / L24–38", "ID, L32–38 / L24–38", "L24–38 at least as specific"], a3_rows, num_cols=(1, 2))
                      + f"<p>By the pre-registered rule, band L24–38 is "
                      f"<b>{'at least as specific as' if a3['B2_at_least_as_specific'] else 'less specific than'}</b> L32–38 "
                      "(agent prior 0.60).</p>")
        clean = s10.get("clean_top10_at_quote") or []
        clean_txt = " · ".join(f"<code>{esc(t_)}</code> {pct(p_, 1)}" for t_, p_ in clean)
        ex = s10.get("examples_cond2_s0.25", {})
        ex_rows = []
        for c in sorted(set(k for v in ex.values() for k in v)):
            ex_rows.append([c] + [" · ".join(f"<code>{esc(t_)}</code> {pct(p_)}" for t_, p_ in ex[b][c]["top10"])
                                  + f" <span class='muted'>(ΔR1′ {r(ex[b][c]['dR1p'], 2, True)})</span>"
                                  if c in ex.get(b, {}) else "—" for b in ("B", "B2")])
        out.append(f"""<h3 id="q15">Q15. Does the log-prob measure single out the planted concept?</h3>
<p>For every row the log-prob change at the open quote was recorded for {s10.get('W_size', '—')} words: the 10 dev concepts
and the protocol's baseline words that are single tokens there (no test concept). <b>ID</b>: the planted concept's own rise is
the largest among the 10 dev concepts (chance 1 in 10). <b>SM</b>: its rise minus the median rise of all other words.
A condition counts as concept-specific if, at ≥ 3 doses where its ΔR1′ beats random directions, ID ≥ 5/10 and SM's CI lies
above 0. Agent priors: full vector specific 0.65 (L32–38) / 0.70 (L24–38); non-J part 0.35 / 0.45; pure J-lens atom in both 0.90;
random directions at chance 0.90.</p>
{"".join(figs)}
{t}
{a3_txt}
<h4>Raw readouts at the open quote</h4>
<p>Clean (no injection), top 10: {clean_txt}</p>
{table(["Concept", "Full vector, s = 1/4, band L32–38: top tokens", "… band L24–38"], ex_rows)}
<p>{notes.get('Q15', '')}</p>""")
    return "".join(out)


def sec_11(D, notes):
    s11 = (D.pilot or {}).get("step11")
    if not s11:
        return ""
    bn = {"B": "L32–38", "B2": "L24–38"}

    def ci_txt(m, ci, nd=2):
        return f"{r(m, nd, True)} [{r(ci[0], nd)}, {r(ci[1], nd)}]"
    out = ["""<h2 id="step11">Step 11: is the late clamp effect about J coordinates, and are the readers really less sensitive?</h2>
<p>Pre-registered before it ran (design §16.7, commit 6058e38). The analysis code was committed while it ran, before any
result was read (e868df2, 2aa6bf2). Dev concepts only; no sampling.</p>"""]
    cons = s11.get("consistency", {})
    out.append("<p>" + (chip("consistency passes", "good") if cons.get("pass") else chip("consistency fails", "bad")) +
               " The unclamped rows repeat step 10's: " + "; ".join(
                   f"band {bn[b]} median |difference| {r(v['median_abs'], 3)}, max {r(v['max_abs'], 3)} nats (n = {v['n']})"
                   for b, v in cons.get("checks", {}).items()) + ".</p>")
    q16 = s11.get("Q16", {})
    q14 = ((D.pilot or {}).get("step10") or {}).get("Q14", {})
    if q16:
        Es = list(range(44, 63, 2))
        ser = []
        for b, dash in (("B", False), ("B2", True)):
            if b in q16:
                ser.append({"name": f"J-atom clamp, band {bn[b]}", "cls": "re", "dash": dash,
                            "ys": [q16[b]["by_E"][str(E)]["extra_J_mean"] for E in Es]})
                ser.append({"name": f"output-direction clamp, band {bn[b]}", "cls": "pn", "dash": dash,
                            "ys": [q16[b]["by_E"][str(E)]["extra_out_mean"] for E in Es]})
        svg = line_chart(ser, Es, [f"L{E}" for E in Es], "extra reduction (nats)",
                         xlabel="clamp end layer E (the clamp runs from the band's first layer to E)",
                         aria="Extra reduction by the J-atom clamp and the output-direction clamp, by end layer")
        rows = []
        for E in Es:
            row = [f"L{E}", r(q14.get("B", {}).get("by_E", {}).get(str(E), {}).get("cos_atom_vs_output_median"), 2)]
            for b in ("B", "B2"):
                if b in q16:
                    x = q16[b]["by_E"][str(E)]
                    row += [r(x["extra_J_mean"], 2, True), ci_txt(x["extra_out_mean"], x["extra_out_ci"]),
                            ci_txt(x["D_mean"], x["D_ci"]),
                            chip("J > output", "good") if x["D_above_0"] else (chip("output > J", "bad") if x["D_below_0"] else chip("no difference", "warn"))]
            rows.append(row)
        head = ["End layer E", "cos(J atom, output), median"]
        for b in ("B", "B2"):
            if b in q16:
                head += [f"{bn[b]}: J clamp", f"{bn[b]}: output clamp [95% CI]", "D = J − output [95% CI]", "D"]
        verd = "".join(f"<li>Band {bn[b]}: <b>{esc(v['verdict'])}</b> (D above 0 at {v['n_mid_D_above_0']} of L48, L50, L52). "
                       f"Pre-registered mapping: {esc(v['amendment'])}.</li>" for b, v in q16.items())
        out.append(f"""<h3 id="q16">Q16. Does blocking the plain output direction do what the J clamp does?</h3>
<p>Same tokens, same rank, same re-entry reference and best doses as step 10's J-atom clamp; only the coordinates differ.
The output-direction clamp acts on g ⊙ W_U[t] (the logit-lens coordinates, the same at every layer); the J clamp on
J<sub>ℓ</sub><sup>T</sup>(g ⊙ W_U[t]). "Extra reduction" is the Q14 measure (random-token version minus concept version).
Rule: J-specific if D's interval lies above 0 at ≥ 2 of L48, L50, L52 (band L32–38 primary). Agent prior 0.50.</p>
{figure(svg, legend(ser), "Mean over evaluable dev concepts; non-J part injected; solid band L32–38, dashed L24–38.")}
{table(head, rows, num_cols=tuple(range(1, len(head))))}
<ul>{verd}</ul>
<p>{notes.get('Q16', '')}</p>""")
    q17 = s11.get("Q17", {})
    if q17.get("readers"):
        rd = q17["readers"]
        g_rows = [[f"{v['reader'].upper() if v['reader'] == 'nla' else 'Oracle'} L{v['layer']}", f"<code>{esc(v['prefix'])}</code>",
                   str(v["W_size"]), f"{int(round(10 * v['G11_ID_natural']))}/10",
                   chip("passes", "good") if v["G11_pass"] else chip("fails", "bad")] for v in rd.values()]
        grid = [1 / 256, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
        figs = []
        mo = q17.get("model_onsets_bandB", {})
        for cond, what in (("4", "non-J part"), ("2", "full concept vector")):
            ser = [{"name": "model's report (band L32–38)", "cls": "full" if cond == "2" else "perp",
                    "ys": [mo.get(cond, {}).get("SM_median_by_dose", {}).get(str(x)) for x in grid]}]
            for k, v in rd.items():
                cls = {"nla_L42": "j", "oracle_L40": "rj", "oracle_L44": "rp"}.get(k, "rand")
                ser.append({"name": f"{'NLA' if v['reader'] == 'nla' else 'Oracle'} L{v['layer']} (direct read)", "cls": cls,
                            "dash": not v["G11_pass"], "ys": [v["by_cond"][cond]["by_dose"][str(x)]["SM_median"] for x in grid]})
            svg = line_chart(ser, grid, [SNAME.get(x, str(x)) for x in grid], "specific rise, median SM (nats)",
                             band=[1.0] * len(grid), aria=f"Median SM by dose, {what}: model vs readers")
            figs.append(figure(svg, legend(ser, extra="below the 1-nat threshold"),
                               f"{what.capitalize()}: the planted concept's specific rise (own rise minus the median of the other "
                               "words) at each one's forced report position. Shaded: below 1 nat."))
        v_rows = []
        for n, v in q17.get("verdicts", {}).items():
            for cond, what in (("2", "full vector"), ("3", "J part"), ("4", "non-J part")):
                x1, x5 = v[cond]["1"], v[cond]["0.5"]
                v_rows.append(["NLA" if n == "nla" else "Oracle", what,
                               SNAME.get(x1["reader_onset"], "never"), SNAME.get(x1["model_onset"], "never"),
                               "—" if x1["at_least_as_sensitive"] is None else (chip("yes", "good") if x1["at_least_as_sensitive"] else chip("no", "bad")),
                               SNAME.get(x5["reader_onset"], "never"), SNAME.get(x5["model_onset"], "never")])
        exs = []
        for k, lst in (s11.get("examples_top5") or {}).items():
            for e in lst:
                if e["concept"] == "storm":
                    exs.append([k.replace("nla", "NLA").replace("oracle", "Oracle").replace("_", " "),
                                f"{'non-J part' if e['cond'] == 4 else 'full vector'}, s = {SNAME.get(e['s'], e['s'])}",
                                " · ".join(f"<code>{esc(w)}</code> {r(dv, 2, True)}" for w, dv in e["top5"])])
        out.append(f"""<h3 id="q17">Q17. Under a likelihood measure, are the readers less sensitive than the model?</h3>
<p>Each reader's answer was teacher-forced to start with a fixed prefix, and the next-token log-probs of the dev
concepts and baseline words were read on direct reads of the written cell, minus the same on the clean cell. That is the
same specific-rise measure (SM) as the model's report in Q15. Onsets use an absolute size, the first dose with a median
SM ≥ 1 nat (0.5 nat secondary), because a relative-to-random criterion is scale-free in a deterministic forward (the A9
trap). Caveat registered beforehand: the model's dose is applied at 7 band layers × 20 positions, while a reader sees
one written cell at the same per-cell dose.</p>
<h4>Gate G11: can each reader identify natural concept activations under this measure?</h4>
{table(["Reader", "Forced prefix", "Words scored", "Natural ‘Tell me about c’: identified", "G11"], g_rows, num_cols=(2,))}
{"".join(figs)}
{table(["Reader", "Injected", "Reader onset (1 nat)", "Model onset (1 nat)", "Reader at least as sensitive", "Reader onset (0.5)", "Model onset (0.5)"], v_rows)}
<p>Pre-registered mapping: {esc(q17.get('amendment', ''))}.</p>
<h4>Raw readouts: the five words each reader raises most (“storm” planted)</h4>
{table(["Reader", "Planted", "Top five words by rise (nats)"], exs)}
<p>{notes.get('Q17', '')}</p>""")
    return "".join(out)


def sec_eng(D, notes):
    pl, st = D.pilot or {}, D.pl_state or {}
    tm = st.get("timings", {})
    rows = []
    for k, v in sorted(tm.items()):
        if not v.get("seconds"):
            continue
        rate = v["n"] / v["seconds"] if v.get("n") else None
        rows.append([k.replace("_", " "), r(v["seconds"] / 60, 1), str(v.get("n", "")), r(rate, 2) if rate else "—"])
    t = table(["Step (pilot)", "Minutes", "Units", "Units per second"], rows, num_cols=(1, 2, 3)) if rows else ""
    sec = pl.get("SEC_crosscheck")
    mem = st.get("memory", {})
    return f"""<h2 id="eng">Machinery: speed, memory, precision</h2>
{t}
<ul>
<li>Peak GPU memory: {", ".join(f"{k.replace('_', ' ')} {r(v, 1)} GB" for k, v in mem.items()) or "—"}.</li>
<li>Exact-precision cross-check (SEC replay, layers above 31 in true float32): {("agreement on " + pct(sec['agree_frac'], 0) + " of " + str(sec['n']) + " cells (same top answer and reciprocal rank within 0.05); median |ΔR1′| difference " + r(sec.get('median_abs_dR1p'), 3) + " nats") if sec else "pending"}.</li>
<li>{notes.get('budget', '')}</li>
</ul>"""


def sec_reads(D, notes):
    """Raw readouts: clean, full, non-J and random for one concept at one dose, both readers, both read kinds."""
    out = ['<h2 id="reads">Raw readouts</h2><p>First sample of each read, verbatim (whitespace collapsed). The pilot\'s '
           'choice of concept and dose for this gallery is fixed in code: the dev concept and reader dose with the highest '
           'NLA naming of the full vector on direct reads, so the gallery shows the readers at their best.</p>']
    src = [("NLA", D.pl_nla or D.sm_nla), ("Oracle", D.pl_oracle or D.sm_oracle)]
    nla = src[0][1]
    best = None
    for q in nla:
        if q["cond"] == 2 and q["kind"] == "direct_cq" and q.get("concept"):
            v = float(np.mean([P.names(t, q["concept"]) for t in q["samples"]]))
            key = (v, q["s"])
            if best is None or key > best[0]:
                best = (key, q["concept"], q["s"], q.get("band"))
    if best is None:
        return out[0] + "<p class='pending'>Pending.</p>"
    _, c, s, band = best
    out.append(f"<p>Gallery: concept <b>{esc(c)}</b>, dose s = {SNAME.get(s, s)}" + (f", band {'L32–38' if band == 'B' else 'L24–38'}" if band else "") + ".</p>")
    for name, reads in src:
        for kind, kl in (("direct_cq", "direct read of the written cell"), ("carried_co", "report position (open quote)")):
            for cond in ("clean", 2, 3, 4, 7):
                cand = [q for q in reads if q["kind"] == kind and (
                    (cond == "clean" and q["cond"] == "clean") or
                    (cond != "clean" and q["cond"] == cond and q.get("concept") == c and q["s"] == s and q.get("band") == band))]
                if name == "Oracle":
                    cand = [q for q in cand if q["layer"] == 44] or cand
                if not cand:
                    continue
                q = cand[0]
                txt = " ".join(q["samples"][0].split())[:700]
                lab = "no injection" if cond == "clean" else COND[cond][0]
                hit = "" if cond == "clean" else (" · names it" if P.names(q["samples"][0], c) else " · does not name it")
                out.append(f'<div class="read"><span class="tag">{esc(name)} L{q["layer"]} · {esc(kl)} · {esc(lab)}{hit}</span>{esc(txt)}</div>')
    rows = D.pl_rows.get((band or "B", "default")) or D.sm_e2e
    ans = [x for x in rows if x["concept"] == c and x["cond"] in (1, 2, 4, 7) and x["s"] == s and x.get("R2_texts")]
    if ans:
        out.append("<h3>What the model itself answered (8 samples each)</h3>")
        for x in ans:
            out.append(f'<div class="read"><span class="tag">{esc(COND[x["cond"]][0])} · top-1 next word “{esc(x.get("top1_at_quote"))}” · '
                       f'reciprocal rank {r(x["R1"], 3)}</span>' + esc(" | ".join(" ".join(t.split()) for t in x["R2_texts"])) + "</div>")
    return "".join(out)


def sec_mistakes():
    md = (ROOT / "CLAUDE.md").read_text()
    rows = []
    for line in md.splitlines():
        if line.startswith("| 10-04 |"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 3:
                rows.append([cells[1], cells[2]])
    return ('<h2 id="mistakes">Mistakes made today and how each was fixed</h2><p>Copied from the silly-mistakes log in '
            'CLAUDE.md (also in each commit message).</p>' +
            table(["Mistake", "Fix"], [[md_inline(a), md_inline(b)] for a, b in rows]))


def md_inline(s):
    s = esc(s)
    import re
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    return s


def sec_amend():
    f = ROOT / "docs/design_specs/005_amendment_01.md"
    if not f.exists():
        return ""
    rows = []
    for line in f.read_text().splitlines():
        if line.startswith("| A"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows.append([md_inline(c) for c in cells])
    return ('<h2 id="amend">The amendment (adopted 5 October)</h2><p>From <code>docs/design_specs/005_amendment_01.md</code>. '
            'You adopted every item, including the optional A18, on branch <code>005-amended-full-run</code>; the table is the '
            'draft as you saw it.</p>' +
            table(["#", "Item", "What the pilot found (dev)", "Proposal"], rows))


def correction_html():
    """Correction from the 005 follow-ups (exploratory): the model-side 'report' measure is lexical priming."""
    F = load(RESA / "followups_summary.json")
    if not F or not str(F.get("A2_naming", "")).startswith("lexical priming"):
        return ""
    b = F["A2b"]["by_cond"]["2"]
    a2 = F["A2"]["test"]["2|0.25"]
    own = F["A3"]["own_atom"]["per_concept"]
    n_own = sum(1 for o in own if o["selected_layers"] > 0)
    return (f"""<div class="callout"><h3>Correction (5 October, follow-ups A2/A2b and A3; exploratory)</h3><ul>
<li><b>What these pages call the model's "report" is lexical priming.</b> The measure is the concept's log-prob at the answer
slot (ΔR1′, R1). It rises about as much at every other prefill position as at the answer slot (full vector: answer slot minus
other positions {r(b['mean'], 2, True)} [{r(b['ci'][0], 2)}, {r(b['ci'][1], 2)}] nats), and more on an unrelated weather prompt
(median {r(a2['median_B2'], 2)} vs {r(a2['median_dR1p'], 2)} nats at s = 1/4). By the pre-registered rule it is called
"lexical priming at the answer slot". Read "report" below as "priming"; the concept-specificity results still hold, as priming.</li>
<li><b>The concept's own token is in its J part for {n_own} of {len(own)} test concepts.</b> Statements that M1 passes "partly by
construction" are withdrawn.</li></ul><p>Details: <code>docs/reports/005_followups.md</code>.</p></div>""")


def sec_full():
    """The full run under the adopted amendment: verdicts from full_summary.json (+ the interpretation in full_notes.json)."""
    S = load(RESA / "full_summary.json")
    if not S:
        return ""
    import make_full_report as MF                      # lazy: it imports this module
    N = load(RESA / "full_notes.json") or {}
    C, G = S["contrasts"], S["gates"]
    rows = []
    for key, label, pred, preg, pup in MF.PRIORS:
        c = C.get(key) or {}
        o = MF.outcome(key, C)
        v = ("not tested (G11 failed)" if c.get("uninformative") else "—") if o is None else ("supported" if o else "not supported")
        if key == "R1_probe_minus_NLA" and o is not None:
            v = "yes (≥ 0)" if o else "no"
        rows.append([esc(label), str(c.get("n", "—")), MF.fmt_ci(c, 4 if key == "M1_R1" else 2), pred,
                     chip(v, "good" if o else ("pend" if o is None else "bad")),
                     r(preg, 2) if preg is not None else "—", r(pup, 2) if pup is not None else "—"])
    reg = MF.brier([(p, MF.outcome(k, C)) for k, _, _, p, _ in MF.PRIORS])
    upd = MF.brier([(p, MF.outcome(k, C)) for k, _, _, _, p in MF.PRIORS])
    bl = N.get("short_html", "")
    bl = bl[bl.find("<ul>"):] if "<ul>" in bl else ""
    g11 = ", ".join(f"{k.replace('_', ' ').replace('nla', 'NLA').replace('oracle', 'Oracle')} {pct(v['ID_natural'])}"
                    for k, v in G.get("G11", {}).items())
    return (f"""<h2 id="full">The full run (5 October): the pre-registered tests on the 45 test concepts</h2>
<p>You adopted amendment 01 in full on 5 October. The run spec (<code>docs/design_specs/005_full_run.md</code>) and the
analysis code were committed before any test-concept output existed, after an end-to-end rehearsal on two dev concepts.
Run <code>{esc(S['run_id'])}</code> took {r(S['tables'].get('wall_minutes'), 0)} minutes. Everything lives on branch
<code>005-amended-full-run</code>, so dropping it reverses the adoption and the run. Every gate passes except the likelihood
positive control (G11) for the NLA (natural activations identified: {g11}; bar 50%), which makes P11′ for the NLA
untestable. The detailed page, with every table, chart and raw readout: <a href="https://claude.ai/artifact/5EGz2DxcAXXQs3wTtz1Lnu">Planted Concepts Full Run</a>.</p>
{bl}
{table(["Contrast", "n", "Mean [95% CI]", "Predicted", "Verdict", "Registered prior", "Updated prior"], rows, num_cols=(1, 2, 5, 6))}
<p>Brier score on the full run: registered priors <b>{r(reg[0], 3)}</b> over {reg[1]}; updated (post-pilot) priors
<b>{r(upd[0], 3)}</b> over {upd[1]}. The bullets are my interpretation; the verdicts follow the pre-registered rules.</p>""")


def sec_glossary():
    g = [("s (dose)", "Injection strength: each band layer gets s × that layer's mean residual norm × a unit direction, on every token of the question turn."),
         ("Band", "The layers written to: L32–38 (B) or L24–38 (B2). Readers read at L40–44, downstream of every write."),
         ("u, u^J, u^⊥", "The concept vector (“Tell me about c” minus the mean of the other 100 concepts); its J-space part (16 J-lens token directions by nonnegative pursuit); the rest (non-J part)."),
         ("Matched / natural norm", "Matched: each part rescaled to the same size as the full vector. Natural: each part at the size it has inside the full vector (the J part holds about 6% of the energy)."),
         ("Pure J-lens vector", "The single J-lens direction of the concept's own token: the paper's Fig. 7 positive control."),
         ("R1, R1′, ΔR1′", "At the open quote: the reciprocal rank of the concept (the paper's metric); its log-probability (log-sum-exp over spellings); the change from the clean run."),
         ("R2", "Sampled naming: share of 8 sampled answers whose first word is the concept."),
         ("B1–B4", "Behaviour: KL on a neutral prompt (B1); the concept's log-prob rise there (B2); the concept's probability before the quote (B3, should stay low); KL over the prefill in the same forward the readers see (B4)."),
         ("Null", "What random directions do at the same dose: the 95th percentile over 5 seeds × the concepts of the split."),
         ("C_q, C_o", "C_q: the comma after “If so” (inside the injected span). C_o: the open quote where the answer comes (never written to)."),
         ("Direct / report-position / injected-position reads", "Direct: the reader gets the written cell itself. Report position: C_o after the model's own processing (content arrives only by attention). Injected position: C_q after processing (near-direct, token readers only)."),
         ("Clamp to clean / re-entry clamp", "Hold the concept's J coordinates at their clean values (the design's version, which also deletes part of the injection) or at clean + raw injection (removes only what the model writes back)."),
         ("G0–G6", "Gates: G0 write fidelity; G1 report gate; G2 analysis gate; G3 J-share sanity; G4 probe validity; G5 split quality; G6 in-run base-rate screen."),
         ("P11a / P11b / P17", "The pre-registered reader tests for the full run: the NLA sees what the model doesn't report (P11a) or doesn't act on (P11b); the Oracle sees the non-J part less than the NLA (P17)."),
         ("Dev / test concepts", "10 dev concepts calibrate and gate; 45 test concepts are held out until the amended full run.")]
    return ('<h2 id="gloss">Glossary</h2><dl class="kv">' + "".join(f"<dt>{esc(a)}</dt><dd>{esc(b)}</dd>" for a, b in g) + "</dl>")


def sec_scorecard(D):
    items = scorecard(D)
    bs, n = brier(items)
    rows = [[it["group"], esc(it["what"]), r(it["prior"], 2), f"<code>{it['commit']}</code>",
             chip("happened", "good") if it["outcome"] else chip("did not happen", "bad"), esc(it["note"])] for it in items]
    FS = load(RESA / "full_summary.json")
    if FS:
        import make_full_report as MF                  # lazy: it imports this module
        FC = FS["contrasts"]

        def res(key):
            o = MF.outcome(key, FC)
            return chip("happened", "good") if o else chip("did not happen", "bad")
        pend = [["Full run", esc(x), r(p, 2), "<code>df56646</code>", res(k), f"test concepts: {MF.fmt_ci(FC.get(k), 4 if k == 'M1_R1' else 2)}"]
                for x, p, k in (("M1: J part beats non-J part on report (matched; R1, as registered)", 0.85, "M1_R1"),
                                ("M2: clamp interpretable and supported (A8: J re-entry clamp to L50)", 0.30, "M2"),
                                ("M3: J beats non-J at natural norm", 0.60, "M3"),
                                ("R1: probe detects u⊥ at least as often as the NLA", 0.70, "R1_probe_minus_NLA"),
                                ("R2: u⊥ moves behaviour without moving report", 0.50, "R2_behaviour_without_report"),
                                ("P17: Oracle sees u⊥ less than the NLA", 0.45, "P17"))]
        pend += [["Full run", esc(x), r(p, 2), "<code>df56646</code>", chip("replaced (A6)", "pend"), "replaced by P11′ under the adopted A6"]
                 for x, p in (("P11a: NLA sees what the model doesn't report", 0.20), ("P11b: NLA sees what doesn't change behaviour", 0.15))]
    else:
        pend = [["Full run", esc(x), r(p, 2), "<code>df56646</code>", chip("not tested yet", "pend"), "needs the amended full run on test concepts"]
                for x, p in (("M1: J part beats non-J part on report (matched)", 0.85), ("M2: clamp interpretable and supported", 0.30),
                             ("M3: J beats non-J at natural norm", 0.60), ("R1: probe detects u⊥ at least as often as the NLA", 0.70),
                             ("R2: u⊥ moves behaviour without moving report", 0.50), ("P11a: NLA sees what the model doesn't report", 0.20),
                             ("P11b: NLA sees what doesn't change behaviour", 0.15), ("P17: Oracle sees u⊥ less than the NLA", 0.45))]
    return (f'<h2 id="expect">What I expected, and what happened</h2><p>Every prior below was committed before the run it '
            f'concerns (commit shown). "Happened" means the event the prior was about occurred. Over the {n} resolved '
            f'pilot-era predictions my Brier score is <b>{r(bs, 3)}</b> (0 is perfect; always saying 0.5 scores 0.25). The "Full run" '
            f'rows are resolved by the full run and scored on its own (first section).</p>' +
            table(["Stage", "Prediction", "My prior", "Registered", "Outcome", "Detail"], rows + pend, num_cols=(2,)))


def sec_where():
    items = [("Design 005 (v2, priors, §16 pilot, §16.3 step 7b)", "docs/design_specs/005_planted_concepts.md"),
             ("Amendment 01 (adopted 5 Oct)", "docs/design_specs/005_amendment_01.md"),
             ("Full-run spec (pre-registered)", "docs/design_specs/005_full_run.md"),
             ("Full-run report", "docs/reports/005_full_run.md"),
             ("Full-run page", "https://claude.ai/artifact/5EGz2DxcAXXQs3wTtz1Lnu"),
             ("Smoke note", "docs/reports/005_smoke_note.md"),
             ("Checks C1–C3 report", "docs/reports/checks_2026-10-04_no_gpu.md"),
             ("Errata 01 (accepted)", "docs/context/SPEC_v2_errata_01.md"),
             ("Your review and its disposition", "docs/reviews/2026-10-04_pi_review_checks_005/"),
             ("Summaries (JSON)", "results/005_planted_concepts/analysis/{smoke,pilot,full}_summary.json"),
             ("Raw outputs", "outputs/005_planted_concepts/{smoke/smoke1, pilot/pilot1 (+ step7b … step11), run/20261005T015235Z}"),
             ("Code", "scripts/005_planted_concepts/ (p005.py, planted.py, pilot.py, pilot_7b.py, *_analysis.py, make_report_page.py)"),
             ("Plain-language notes", "/workspace/claude_reports/ (README index)")]
    return ('<h2 id="where">Where everything lives</h2><p>Repo <code>senku14x/Causal-Concordance</code> (private), branch '
            '<code>spec-v2</code> for everything up to the pilot; branch <code>005-amended-full-run</code> for the adoption and the '
            'full run.</p>' + table(["What", "Path"], [[esc(a), f"<code>{esc(b)}</code>"] for a, b in items]))


def build(D, notes, short_html):
    pl = D.pilot or {}
    have_pilot = bool(pl.get("Q1")) and len(D.pl_rows) == 4
    have_7b = bool(pl.get("Q10"))
    chips = [chip("checks C1–C3 done", "good"), chip("smoke done · report gate failed (stop rule)", "warn"),
             chip("pilot done" if have_pilot else "pilot running", "good" if have_pilot else "pend"),
             chip("step 7b done" if have_7b else "step 7b pending", "good" if have_7b else "pend"),
             chip("step 8 done" if pl.get("Q12") else "step 8 pending", "good" if pl.get("Q12") else "pend"),
             chip("step 9 done" if pl.get("step9") else "step 9 pending", "good" if pl.get("step9") else "pend"),
             chip("step 10 done" if pl.get("step10") else "step 10 pending", "good" if pl.get("step10") else "pend"),
             chip("step 11 done" if pl.get("step11") else "step 11 pending", "good" if pl.get("step11") else "pend"),
             chip("amendment 01 adopted", "good") if (ROOT / "docs/design_specs/005_amendment_01.md").exists() else chip("full run waits for your decisions", "pend")]
    if load(RESA / "full_summary.json"):
        chips.append(chip("full run done · 45 test concepts", "good"))
    toc = ([("full", "The full run (5 Oct)")] if load(RESA / "full_summary.json") else []) + [("short", "The pilot in short"), ("why", "Why any of this was run"), ("expect", "Expected vs happened"),
           ("checks", "Checks C1–C3"), ("smoke", "The smoke run"), ("pilot", "The overnight pilot"),
           ("q1", "Q1 report measure", 1), ("q2", "Q2 prefill", 1), ("q3", "Q3 band", 1), ("q4", "Q4 onset", 1),
           ("q5", "Q5 readers", 1), ("q6", "Q6 clamp", 1), ("q7", "Q7 leaky splits", 1), ("q8", "Q8 baseline", 1),
           ("step7b", "Step 7b"), ("q10", "Q10 re-entry clamp", 1), ("q11", "Q11 layer sweep", 1),
           ("step8", "Step 8"), ("q12", "Q12 arrival at the quote", 1), ("q13", "Q13 Oracle late layers", 1), ("power", "Power", 1),
           ("step9", "Step 9 robustness"),
           ("step10", "Step 10"), ("g10", "G10 consistency", 1), ("q14", "Q14 clamp end layer", 1), ("q15", "Q15 specificity", 1),
           ("step11", "Step 11"), ("q16", "Q16 output-direction clamp", 1), ("q17", "Q17 reader likelihoods", 1),
           ("reads", "Raw readouts"), ("eng", "Machinery"), ("mistakes", "Mistakes"), ("amend", "Amendment (adopted)"),
           ("gloss", "Glossary"), ("where", "Where things live")]
    nav = '<nav class="toc" aria-label="Contents">' + "".join(
        f'<a class="{"sub" if len(t) > 2 else ""}" href="#{t[0]}">{esc(t[1])}</a>' for t in toc) + "</nav>"
    head = f"""<header class="top"><div class="eyebrow">Causal-Concordance · design 005 · night of 4 October 2026</div>
<h1>Planted Concepts Pilot</h1>
<p class="lede">Everything that was run tonight, why it was run, what I expected and what came out: the three checks,
the 005 smoke run, the dev-only pilot, the extra steps 7b and 8–11, and (5 October) the full run on the 45 test concepts under the adopted amendment. Every number on this page is generated from the
saved result files.</p>
<div class="chips">{"".join(chips)}</div></header>"""
    body = [head, correction_html(), sec_full(), short_html, sec_why(), sec_scorecard(D), sec_checks(D), sec_smoke(D), sec_pilot(D, notes),
            sec_pilot2(D, notes), sec_7b(D, notes), sec_8(D, notes), sec_9(D, notes), sec_10(D, notes), sec_11(D, notes), sec_reads(D, notes), sec_eng(D, notes), sec_mistakes(), sec_amend(),
            sec_glossary(), sec_where(),
            '<footer>Generated by <code>scripts/005_planted_concepts/make_report_page.py</code> from JSON in '
            '<code>results/</code> and <code>outputs/</code>. Dev concepts only; no test-concept outcome exists yet.</footer>']
    fonts = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&'
             'family=IBM+Plex+Sans+Condensed:wght@600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">')
    return (f"<title>Planted Concepts Pilot</title>\n{fonts}\n<style>{CSS}</style>\n"
            f'<div class="wrap">{nav}<main>{"".join(body)}</main></div>')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--notes", default=str(RESA / "report_notes.json"))
    a = ap.parse_args()
    D = Data()
    notes = load(a.notes) or {}
    short = notes.get("short_html", '<h2 id="short">The short version</h2><p class="pending">Written when every result is in.</p>')
    html_ = build(D, notes, short)
    Path(a.out).write_text(html_)
    print("wrote", a.out, len(html_) // 1024, "KB")


if __name__ == "__main__":
    main()
