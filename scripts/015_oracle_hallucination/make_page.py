"""015 results page (claude.ai artifact) from committed JSON only; numbers rounded half-up from full precision.
EXPLORATORY. Interpretive text lives in TEXT below and quotes numbers only through the same helpers.

  python make_page.py   ->  results/015_oracle_hallucination/page/index.html
Sources: results/015_oracle_hallucination/{analysis/oh_all.json, analysis/oh_smoke.json, r1.json, handcheck_tallies.json},
docs/ledger/015_ledger.md (rendered in full), outputs/015_oracle_hallucination/versions/versions_all.jsonl (two samples;
mirrored to the public HF dataset).
"""
import html
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/015_oracle_hallucination"
A = json.loads((R / "analysis/oh_all.json").read_text())
SM = json.loads((R / "analysis/oh_smoke.json").read_text())
R1 = json.loads((R / "r1.json").read_text())
HC = json.loads((R / "handcheck_tallies.json").read_text())
LEDGER = (ROOT / "docs/ledger/015_ledger.md").read_text()
GH = "https://github.com/senku14x/Causal-Concordance/blob/015-oracle-hallucination/"
HF = "https://huggingface.co/datasets/senku21x/causal-concordance-outputs/tree/main/015_oracle_hallucination"
E = html.escape


def h(x, nd=1, sign=True):
    if x is None:
        return "—"
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    q = abs(q) if q == 0 else q
    s = f"{q:.{nd}f}"
    return ("+" + s if sign and q > 0 else s).replace("-", "−")


def d(s, nd=1, n=True):
    if not s:
        return "—"
    return f"{h(s['mean'], nd)} [{h(s['ci95'][0], nd)}, {h(s['ci95'][1], nd)}]" + (f" (n {s['n']})" if n else "")


def frac(a, b, nd=2):
    return f"{a}/{b} = {h(a / b, nd, False)}"


CELLS = [("oracle_texts|oracle_ar", "Oracle texts", "Oracle AR"), ("oracle_texts|nla_ar", "Oracle texts", "NLA AR"),
         ("nla_texts|nla_ar", "NLA texts", "NLA AR"), ("nla_texts|oracle_ar", "NLA texts", "Oracle AR")]
C = {k: A["cells"][k] for k, _, _ in CELLS}


# --------------------------------------------------------------------------------------------- charts
def ticks(lo, hi, n=6):
    span = hi - lo
    step = 10 ** __import__("math").floor(__import__("math").log10(span / n))
    for m in (1, 2, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    t = __import__("math").ceil(lo / step) * step
    out = []
    while t <= hi + 1e-9:
        out.append(round(t, 6))
        t += step
    return out


def forest(rows, aria, w=380, pad_x=10, rowh=46, band=2.0):
    """rows: (label, stat, cls). Drawn at its rendered width (about 380 px in the two-column grid): label and value on
    one line, then the CI. Dot = mean, line = 95% CI (item-cluster bootstrap); shaded = ±2 points (013's bands)."""
    vals = [0.0, -band, band] + [v for _, s, _ in rows if s for v in (s["ci95"][0], s["ci95"][1], s["mean"])]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad
    H = rowh * len(rows) + 26
    sx = lambda v: pad_x + (v - lo) / (hi - lo) * (w - 2 * pad_x)                                # noqa: E731
    s = [f'<svg viewBox="0 0 {w} {H}" role="img" aria-label="{E(aria)}">',
         f'<rect x="{sx(-band):.1f}" y="2" width="{sx(band) - sx(-band):.1f}" height="{H - 24}" class="band"/>']
    for t in ticks(lo, hi, 5):
        s.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="2" y2="{H - 22}" class="grid"/>'
                 f'<text x="{sx(t):.1f}" y="{H - 8}" text-anchor="middle" class="tk">{h(t, 0 if abs(t) >= 1 or t == 0 else 1)}</text>')
    s.append(f'<line x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="2" y2="{H - 22}" class="ax"/>')
    for i, (lab, st, cls) in enumerate(rows):
        y0 = 4 + i * rowh
        s.append(f'<text x="{pad_x}" y="{y0 + 13}" class="lb">{E(lab)}</text>'
                 f'<text x="{w - pad_x}" y="{y0 + 13}" text-anchor="end" class="val">{E(d(st, n=False))}</text>')
        if st:
            y = y0 + 30
            s.append(f'<line x1="{sx(st["ci95"][0]):.1f}" x2="{sx(st["ci95"][1]):.1f}" y1="{y}" y2="{y}" class="ci"/>'
                     f'<circle cx="{sx(st["mean"]):.1f}" cy="{y}" r="5.5" class="{cls} ring"><title>{E(lab)}: {E(d(st))}</title></circle>')
    s.append("</svg>")
    return "".join(s)


def fig(title, svg, cap=""):
    return f'<figure><figcaption><b>{E(title)}</b>{(" · " + cap) if cap else ""}</figcaption>{svg}</figure>'


# --------------------------------------------------------------------------------------------- ledger markdown
def inline(t):
    t = E(t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    return t


def md(text):
    out, stack, para = [], [], []

    def flush():
        if para:
            out.append("<p>" + " ".join(para) + "</p>")
            para.clear()

    def close_all():
        while stack:
            out.append("</li></ul>")
            stack.pop()

    for line in text.split("\n"):
        m = re.match(r"^(\s*)- (.*)$", line)
        if m:
            flush()
            ind = len(m.group(1))
            while stack and stack[-1] > ind:
                out.append("</li></ul>")
                stack.pop()
            if stack and stack[-1] == ind:
                out.append("</li><li>")
            else:
                out.append("<ul><li>")
                stack.append(ind)
            out.append(inline(m.group(2)))
        elif not line.strip():
            flush()
        elif stack and line.startswith(" "):
            out.append(" " + inline(line.strip()))
        else:
            close_all()
            para.append(inline(line.strip()))
    flush()
    close_all()
    return "".join(out)


def ledger_html():
    parts = re.split(r"\n## ", LEDGER)
    head, secs = parts[0], parts[1:]
    out = []
    for i, sec in enumerate(secs):
        title, _, body = sec.partition("\n")
        out.append(f'<details{" open" if i == len(secs) - 1 else ""}><summary>{inline(title)}</summary><div class="md">{md(body)}</div></details>')
    return "".join(out)


# --------------------------------------------------------------------------------------------- qualitative
def sample(eid, reader="oracle"):
    V = None
    for l in open(ROOT / "outputs/015_oracle_hallucination/versions/versions_all.jsonl"):
        x = json.loads(l)
        if x["eid"] == eid and x["reader"] == reader:
            V = x
            break
    names = [("O", "original"), ("CA", "answer spans deleted"), ("RA", "random clauses deleted (seed 0)"),
             ("PA", "same token count cut from the opening of an answer-free bullet (seed 0)"), ("C1", "contradicted spans deleted"),
             ("F", "factual rewrite (013 prompt)"), ("P", "matched paraphrase of O"), ("Fc", "correction-only rewrite (POST HOC)"),
             ("Pc", "paraphrase matched to Fc")]
    cards = "".join(f'<div class="ro"><div class="rh">{n} · {E(lab)}</div><p class="ex">{E(V["texts"][n][0])}</p></div>'
                    for n, lab in names if n in V["texts"])
    return f'<div class="ros">{cards}</div>'


# --------------------------------------------------------------------------------------------- text (written after reading oh_all.json)
TEXT = {}
exec((Path(__file__).parent / "page_text.py").read_text(), {"A": A, "C": C, "SM": SM, "R1": R1, "HC": HC, "h": h, "d": d, "frac": frac,
                                                             "TEXT": TEXT})

# --------------------------------------------------------------------------------------------- page
g = A["gates"]
cov = A["answer_arm_coverage"]
rw = A["rewrites"]
oo = C["oracle_texts|oracle_ar"]
tiles = [
    (d(oo["dA_RA_minus_CA"], n=False), f"Δ_A, Oracle texts × Oracle AR (n {oo['dA_RA_minus_CA']['n']}): deleting the answer vs random clauses"),
    (d(oo["dAP_PA_minus_CA"], n=False), f"Δ_AP, same cell (n {oo['dAP_PA_minus_CA']['n']}): answer vs the opening of an answer-free bullet"),
    (d(oo["primary_R1_minus_C1"], n=False), f"primary Δ, same cell (n {oo['primary_R1_minus_C1']['n']}): false claims vs random"),
    (d(C["nla_texts|nla_ar"]["primary_R1_minus_C1"], n=False), f"primary Δ, NLA texts × NLA AR (n {C['nla_texts|nla_ar']['primary_R1_minus_C1']['n']}; 013 found +28.6)"),
]
r1rows = "".join(f"<tr><td>{t}</td><td>{s}</td><td>{h(R1['cells'][k]['matched_minus_shuffled']['mean'])} "
                 f"[{h(R1['cells'][k]['matched_minus_shuffled']['ci95'][0])}, {h(R1['cells'][k]['matched_minus_shuffled']['ci95'][1])}]</td>"
                 f"<td>{h(R1['cells'][k]['matched_cos_mean'], 3, False)}</td><td>{h(R1['cells'][k]['retrieval_at_1'], 2, False)}</td></tr>"
                 for k, t, s in CELLS)
gp = HC["G2_contradicted_span_precision"]["pooled"]
gk = HC["G2k_kind_agreement"]["pooled"]
ga = HC["G2a_answer_span_precision"]
gates = [
    ("G1 judge A parses with verbatim spans (≥ 0.90)", h(g["G1_A_ok_rate_oracle"], 3, False), "pass"),
    ("Answer spans parse, Oracle / NLA", f"{h(g['ANS_ok_rate']['oracle'], 3, False)} / {h(g['ANS_ok_rate']['nla'], 3, False)}", "pass"),
    ("G2 hand check of contradicted spans, precision ≥ 0.80 (smoke 20 + 10 more)",
     f"{frac(gp['correct_strict'], gp['n_spans'])} strict; {frac(gp['correct_lenient'], gp['n_spans'])} lenient", "pass"),
    ("G2k kind tags, agreement ≥ 0.80", f"{frac(gk['agree'], gk['n_spans'])}; misquote tags right {gk['misquote_correct']}/{gk['misquote_tags']}",
     "pass; Oracle misquote unreliable"),
    ("G2a answer spans, precision ≥ 0.80", f"Oracle {frac(ga['oracle_smoke']['answer_statements'], ga['oracle_smoke']['n_spans'])}; "
     f"NLA {frac(ga['nla_smoke_first10']['answer_statements'], ga['nla_smoke_first10']['n_spans'])}", "pass"),
    ("G3 A–B κ per sentence (descriptive)", f"{h(g['G3_kappa_oracle'], 2, False)} over {g['G3_n_sentences_oracle']} sentences", "—"),
    ("Rewrite gate, Oracle: F contradicted-sentence rate ≤ 0.10; P ≥ 0.7 × O",
     f"F {h(g['B_contradicted_sentence_rate_oracle']['F'], 3, False)}; P {h(g['B_contradicted_sentence_rate_oracle']['P'], 3, False)} vs O "
     f"{h(g['B_contradicted_sentence_rate_oracle']['O'], 3, False)}",
     "pass" if g["rewrite_gate_F_le_0.10_oracle"] and g["rewrite_gate_P_ge_0.7_O_oracle"] else "FAIL"),
    ("Rewrite gate, NLA: F ≤ 0.10; P* ≥ 0.7 × O",
     f"F {h(g['B_contradicted_sentence_rate_nla']['F'], 3, False)}; P* {h(g['B_contradicted_sentence_rate_nla']['Pstar'], 3, False)} vs O "
     f"{h(g['B_contradicted_sentence_rate_nla']['O'], 3, False)}",
     "pass" if g["rewrite_gate_F_le_0.10_nla"] and g["rewrite_gate_Pstar_ge_0.7_O_nla"] else "FAIL"),
    ("G-repro: NLA AR on NLA texts = 013 (≤ 1e-6)", f"max |ΔMSE| {A['G_repro_nla']['max_abs_mse_013_versions']} over {A['G_repro_nla']['n_values']} "
     f"values; versions identical {A['G_repro_nla']['versions_same_as_013']}/{A['G_repro_nla']['n']}", "pass" if A["G_repro_nla"]["pass_1e-6"] else "FAIL"),
    ("Step 2 stop gate (Oracle texts × Oracle AR): matched − shuffled ≥ 10, CI > 0; ret@1 ≥ 0.5",
     f"{h(R1['cells']['oracle_texts|oracle_ar']['matched_minus_shuffled']['mean'])}; ret@1 {h(R1['cells']['oracle_texts|oracle_ar']['retrieval_at_1'], 2, False)}", "pass"),
]
gate_rows = "".join(f'<tr><td>{E(a)}</td><td>{E(b)}</td><td class="{"ok" if c.startswith("pass") else ("bad" if c == "FAIL" else "")}">{E(c)}</td></tr>'
                    for a, b, c in gates)


def cellchart(k, rows, aria):
    t = [x for x in CELLS if x[0] == k][0]
    return fig(f"{t[1]} × {t[2]}", forest(rows, aria), f"mean cos of O {h(C[k]['cos_mean']['O'], 3, False)}")


ans_charts = "".join(cellchart(k, [("Δ_A = RA − CA", C[k]["dA_RA_minus_CA"], "s1" if k.startswith("oracle") else "s2"),
                                   ("Δ_A on the PA subset", C[k]["dA_on_PA_subset"], "s1" if k.startswith("oracle") else "s2"),
                                   ("Δ_AP = PA − CA", C[k]["dAP_PA_minus_CA"], "s1" if k.startswith("oracle") else "s2")],
                                f"answer arm, {k}") for k, _, _ in CELLS)
prim_charts = "".join(cellchart(k, [("primary Δ = R1 − C1", C[k]["primary_R1_minus_C1"], "s1" if k.startswith("oracle") else "s2"),
                                    ("sentence ctrl RS1 − C1", C[k]["RS1_minus_C1"], "s1" if k.startswith("oracle") else "s2")]
                                + [(f"kind: {kk.replace('_', ' ')}", C[k][f"kind_{kk}"]["delta_clause"], "s1" if k.startswith("oracle") else "s2")
                                   for kk in ("misquote", "invented_context", "wrong_entity", "false_fact") if C[k][f"kind_{kk}"]["n"]],
                                f"primary, {k}") for k, _, _ in CELLS)


def rew_rows(k):
    c, cls = C[k], ("s1" if k.startswith("oracle") else "s2")
    rows = [("F − P", c["F_minus_P"], cls), ("F − O", c["F_minus_O"], cls), ("P − O", c["P_minus_O"], cls)]
    if k.startswith("oracle"):
        rows += [("Fc − Pc (POST HOC)", c.get("Fc_minus_Pc"), cls), ("Fc − O", c.get("Fc_minus_O"), cls), ("Pc − O", c.get("Pc_minus_O"), cls),
                 ("F − P on Fc readouts", c.get("F_minus_P_on_Fc_set"), cls)]
    else:
        rows += [("F − P* (P redo)", c.get("F_minus_Pstar"), cls)]
    return rows


rew_charts = "".join(cellchart(k, rew_rows(k), f"rewrites, {k}") for k, _, _ in CELLS)
rc_rows = ""
for cn, lab in (("primary", "Primary Δ"), ("dA", "Δ_A"), ("dAP", "Δ_AP")):
    for s, sl in (("oracle_ar", "Oracle AR"), ("nla_ar", "NLA AR")):
        x = A["reader_contrasts_nla_minus_oracle"][cn][s]
        a, e, sc = x["all"], x["english_only"], x.get("scaled_pct_of_R1_range_descriptive")
        rc_rows += (f"<tr><td>{lab}</td><td>{sl}</td><td>{h(a['raw']['mean'])} [{h(a['raw']['ci95'][0])}, {h(a['raw']['ci95'][1])}]</td>"
                    f"<td>{h(a['per_share']['value'])} [{h(a['per_share']['ci95'][0])}, {h(a['per_share']['ci95'][1])}]</td>"
                    f"<td>{(h(e['raw']['mean']) + ' [' + h(e['raw']['ci95'][0]) + ', ' + h(e['raw']['ci95'][1]) + ']') if e else '—'}</td>"
                    f"<td>{(h(sc['raw']['mean']) + ' [' + h(sc['raw']['ci95'][0]) + ', ' + h(sc['raw']['ci95'][1]) + ']') if sc else '—'}</td>"
                    f"<td>{a['n_item_states']}</td><td>{'yes' if x['claim_holds'] else 'no'}</td></tr>")
lv_rows = "".join(f"<tr><td>{t}</td><td>{s}</td>" + "".join(f"<td>{h(C[k]['fve_mean'].get(n))} ({h(C[k]['cos_mean'].get(n), 3, False)})</td>"
                                                               for n in ("O", "C1", "R1", "CA", "RA", "PA", "F", "P"))
                  + f"<td>{h(C[k]['share_J_mean']['gold'], 3, False)} / {h(C[k]['share_J_mean']['O'], 3, False)} / {h(C[k]['share_J_mean'].get('F'), 3, False)}</td></tr>"
                  for k, t, s in CELLS)
cw = A["cw20_metric_oracle_texts"]

page = f"""<title>Oracle Hallucination Ledger</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400&family=Newsreader:opsz,wght@6..72,500&display=swap">
<style>
/* layout: one reading column; small multiples (one per text × scorer cell) wrap to one column on phones */
:root {{ --bg:#f9f9f7; --surface:#fcfcfb; --fg:#0b0b0b; --fg2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --s1:#2a78d6; --s2:#eb6834; --accent:#256abf; --band:#ecebe5; --ok:#2f7d4f; --bad:#b3261e;
  --display:"Newsreader", Georgia, serif; --body:"IBM Plex Sans", system-ui, sans-serif; --mono:"IBM Plex Mono", ui-monospace, monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0d0d0d; --surface:#1a1a19; --fg:#ffffff; --fg2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835; --s1:#3987e5; --s2:#d95926; --accent:#86b6ef; --band:#222220; --ok:#6cc28f; --bad:#f2827a; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#0d0d0d; --surface:#1a1a19; --fg:#ffffff; --fg2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835; --s1:#3987e5; --s2:#d95926; --accent:#86b6ef; --band:#222220; --ok:#6cc28f; --bad:#f2827a; color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.55 var(--body); }}
main {{ max-width:800px; margin:0 auto; padding-inline:16px; padding-block:28px 64px; display:grid; gap:20px; min-width:0; }}
h1 {{ font:500 2rem/1.15 var(--display); margin:0; text-wrap:balance; }}
h2 {{ font:500 1.35rem/1.25 var(--display); margin:12px 0 0; text-wrap:balance; }}
h3 {{ font:600 1rem/1.3 var(--body); margin:4px 0 0; }}
p {{ margin:0; max-width:70ch; }} .sub {{ color:var(--fg2); }}
.eyebrow {{ font:12px var(--mono); letter-spacing:.06em; text-transform:uppercase; color:var(--muted); }}
.lede {{ font-size:1.05rem; }}
figure {{ margin:0; background:var(--surface); border:1px solid var(--grid); border-radius:6px; padding:12px; min-width:0; }}
figcaption {{ font-size:13px; color:var(--fg2); margin-bottom:6px; }} figcaption b {{ color:var(--fg); font-weight:600; }}
svg {{ width:100%; height:auto; display:block; }}
.ax {{ stroke:var(--axis); stroke-width:1.5; }} .grid {{ stroke:var(--grid); }} .band {{ fill:var(--band); }}
.tk {{ fill:var(--muted); font:10.5px var(--mono); }} .lb {{ fill:var(--fg); font:600 12px var(--body); }} .val {{ fill:var(--fg2); font:10.5px var(--mono); }}
.s1 {{ fill:var(--s1); }} .s2 {{ fill:var(--s2); }} .ring {{ stroke:var(--surface); stroke-width:1.5; }} .ci {{ stroke:var(--fg2); stroke-width:2; }}
.sms {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 340px), 1fr)); gap:12px; }}
.wrap {{ overflow-x:auto; min-width:0; }} table {{ border-collapse:collapse; font:12.5px var(--mono); font-variant-numeric:tabular-nums; width:100%; }}
th, td {{ border-bottom:1px solid var(--grid); padding:6px 8px; text-align:left; vertical-align:top; }} th {{ color:var(--fg2); font-weight:400; }}
td.ok {{ color:var(--ok); }} td.bad {{ color:var(--bad); font-weight:600; }}
.kv {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 170px), 1fr)); gap:12px; }}
.kv div {{ border-top:2px solid var(--grid); padding-top:6px; min-width:0; }} .kv b {{ display:block; font:500 1.15rem var(--display); }} .kv span {{ font-size:12px; color:var(--fg2); }}
details {{ border:1px solid var(--grid); border-radius:6px; padding:8px 12px; background:var(--surface); }} summary {{ cursor:pointer; font:13px var(--mono); }}
details .md {{ font-size:14px; margin-top:8px; display:grid; gap:8px; }} .md ul {{ margin:0; padding-left:20px; }} .md li {{ margin:3px 0; }}
code {{ font:12.5px var(--mono); background:var(--band); padding:0 3px; border-radius:3px; overflow-wrap:anywhere; }}
.ros {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 340px), 1fr)); gap:10px; }}
.ro {{ min-width:0; }} .ro p {{ font:12px/1.45 var(--mono); color:var(--fg2); margin-top:4px; }} .rh {{ font-size:12px; color:var(--muted); }}
.ex {{ white-space:pre-wrap; overflow-wrap:anywhere; }}
.note {{ border-left:3px solid var(--accent); padding-left:12px; }}
ul {{ margin:0; padding-left:20px; max-width:70ch; }} li {{ margin:4px 0; }}
a {{ color:var(--accent); }} a:focus-visible, summary:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; font-size:12px; color:var(--fg2); }} .legend i {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:-1px; }}
</style>
<main>
<div class="eyebrow">Causal-Concordance · experiment 015 · Qwen3.6-27B · multihop, final prompt token · Oracle Lens L44 and NLA L42 · exploratory · 2026-10-11</div>
<h1>Oracle hallucination test, with cross-scoring</h1>
<p class="lede">{TEXT['lede']}</p>
<div class="kv">{''.join(f'<div><b>{a}</b><span>{E(b)}</span></div>' for a, b in tiles)}</div>
<p class="sub">Δ is FVE(control) − FVE(cleaned), in each reconstructor's own FVE points (013's metric). Brackets: 95% item-cluster bootstrap (2,000 draws). Points are never compared across reconstructors. Every number on this page is read from committed JSON and rounded half-up.</p>

<h2>What was tested</h2>
{TEXT['design']}
<div class="wrap"><table><tr><th>Texts</th><th>Scored by</th><th>R1: matched − shuffled</th><th>matched cos</th><th>ret@1 / 87</th></tr>{r1rows}</table></div>
<p class="sub">Step 2 (report 45): every text source × reconstructor cell tells the right activation from a shuffled one; the NLA texts under the Oracle AR only weakly.</p>
<div class="wrap"><table><tr><th>Version</th><th>What it is</th></tr>
<tr><td>O</td><td>the reader's own text</td></tr>
<tr><td>C1 / R1</td><td>judge A's contradicted spans deleted / token-matched random clauses deleted (5 seeds). Primary Δ = R1 − C1</td></tr>
<tr><td>CA / RA</td><td>judge B's answer statements deleted / token-matched random clauses deleted. Δ_A = RA − CA</td></tr>
<tr><td>PA</td><td>the same token count cut from the opening of a bullet (or sentence) with no answer in it. Δ_AP = PA − CA</td></tr>
<tr><td>F / P</td><td>013's minimal factual rewrite / a paraphrase matched to F's edit size</td></tr>
<tr><td>Fc / Pc</td><td>POST HOC: correct the flagged spans in place, delete nothing / its matched paraphrase (Oracle only)</td></tr>
<tr><td>P*</td><td>POST HOC: the NLA's paraphrase redone with the matched-size rule</td></tr></table></div>

<h2>Gates and checks</h2>
<div class="wrap"><table><tr><th>Check</th><th>Result</th><th>Status</th></tr>{gate_rows}</table></div>
{TEXT['gates']}

<h2>The answer arm (the main contrast)</h2>
{TEXT['answer']}
<div class="legend"><span><i style="background:var(--s1)"></i>Oracle texts</span><span><i style="background:var(--s2)"></i>NLA texts</span><span><i style="background:var(--band)"></i>±2 points (013's no-difference band)</span></div>
<div class="sms">{ans_charts}</div>

<h2>False claims vs random text (primary Δ) and span kinds</h2>
{TEXT['primary']}
<p class="sub">Kinds use the clause control (Rk − C1k). Hover a dot for its n; every n is in the report's tables.</p>
<div class="sms">{prim_charts}</div>

<h2>Rewrites</h2>
{TEXT['rewrites']}
<div class="sms">{rew_charts}</div>

<h2>Is any difference reader-specific?</h2>
{TEXT['contrasts']}
<div class="wrap"><table><tr><th>Contrast</th><th>Scorer</th><th>NLA − Oracle, raw</th><th>per unit deleted share</th><th>English-only</th><th>% of each cell's R1 range (descriptive)</th><th>(item, state)</th><th>claim rule met</th></tr>{rc_rows}</table></div>

<h2>Levels and J share</h2>
<p class="sub">Mean FVE points (mean cosine) per version; J share of the reconstruction in 007's dense J basis at the scorer's layer: gold / O / F.</p>
<div class="wrap"><table><tr><th>Texts</th><th>Scorer</th><th>O</th><th>C1</th><th>R1</th><th>CA</th><th>RA</th><th>PA</th><th>F</th><th>P</th><th>J share</th></tr>{lv_rows}</table></div>
<p class="sub">CW-20's metric on Oracle texts × Oracle AR (global scale {h(cw['global_scale'], 3, False)}): scaled FVE O {h(cw['scaled_fve_mean']['O'], 3, False)} (hers 0.21), CA {h(cw['scaled_fve_mean']['CA'], 3, False)}, RA {h(cw['scaled_fve_mean']['RA'], 3, False)}, PA {h(cw['scaled_fve_mean']['PA'], 3, False)}; RA − CA {h(cw['scaled_fve_mean']['RA'] - cw['scaled_fve_mean']['CA'], 3)} (her "removing the answer −7.7" is +0.077 in this sign).</p>

<h2>What the versions look like</h2>
<p class="sub">One Oracle readout (atomic number 26, random-swap state; the model's continuation is "Fe"), every version, seed 0 where seeded. Source: <code>versions_all.jsonl</code>. The &#xFFFD; at the end of one bullet is in the readout itself: generation stopped in the middle of a character.</p>
{sample("atomic-26-symbol|rand_L24-35|1")}

<h2>Caveats</h2>
{TEXT['caveats']}

<h2>The 015 ledger, in full</h2>
<p class="sub">Append-only, rendered from <a href="{GH}docs/ledger/015_ledger.md">docs/ledger/015_ledger.md</a>. Newest entry open.</p>
{ledger_html()}

<h2>Files</h2>
<ul>
<li>Spec: <a href="{GH}docs/design_specs/015_oracle_hallucination.md">docs/design_specs/015_oracle_hallucination.md</a></li>
<li>Reports: <a href="{GH}claude_ovarida/45_015_r1_2026-10-11.md">45 (step 2)</a>, <a href="{GH}claude_ovarida/46_015_smoke_2026-10-11.md">46 (smoke)</a>, <a href="{GH}claude_ovarida/47_015_full_2026-10-11.md">47 (full run)</a></li>
<li>Code: <a href="{GH}scripts/015_oracle_hallucination/oh.py">oh.py</a>, <a href="{GH}scripts/015_oracle_hallucination/r1.py">r1.py</a>; results JSON <a href="{GH}results/015_oracle_hallucination/analysis/oh_all.json">oh_all.json</a></li>
<li>Raw outputs (judge cache, versions, scores): <a href="{HF}">public HF dataset</a></li>
</ul>
</main>
"""
(R / "page").mkdir(parents=True, exist_ok=True)
page = page.replace("\ufffd", "&#xFFFD;")        # real data: a readout cut mid-character by the generation budget
(R / "page/index.html").write_text(page)
print("wrote", R / "page/index.html", len(page))
