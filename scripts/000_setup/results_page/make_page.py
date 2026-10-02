"""Build the results page (published as a private claude.ai artifact) from committed results only.

    python scripts/000_setup/results_page/make_page.py  ->  outputs/000_setup/results_page/causal_concordance.html

Block 000 comes from `results/000_setup/analysis/*.json`; experiment 001 from `results/001_leverage_profile/
{analysis/*.json, figures/*.png}` and the predictions table of `docs/reports/001_leverage_profile.md` §10.
"""
import html
from decimal import ROUND_HALF_UP, Decimal
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import partners as P
import registry as R
import wsb

A = R.PROJECT / "results" / "000_setup" / "analysis"
X = R.PROJECT / "results" / "001_leverage_profile"
LAYERS = [40, 42, 44, 56, 60]
CELL_IDX = {"emit_asst": "cell0", "emit_think": "cell1", "emit_colon": "cell2"}
CELL_NAME = {"emit_colon": "“:” after “Answer”", "emit_asst": "newline after “assistant”",
             "emit_think": "“\\n\\n” after “</think>”"}
READER_NAME = {"logit_lens": "Logit lens", "J_NP_cosine": "J_NP (WSB cosine)", "J_NP_standard": "J_NP (standard)",
               "J_CB_standard": "J_CB", "R_CB_standard": "R_CB", "oracle_wsb": "Oracle (WSB, 1 sample)",
               "oracle": "Oracle (contract, 4 samples)", "nla": "NLA (4 samples)"}
h = html.escape


def f2(x, d=2):
    """Round half up on the stored decimal (0.585 -> 0.59), as the reports do; Python's f-format would give 0.58."""
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), ROUND_HALF_UP))


def pct(x):
    return f"{100 * x:.0f}%"


def frac(x, n):
    """'39 of 43 (90.7%)' from a fraction over n items."""
    return f"{round(x * n)} of {n} ({100 * x:.1f}%)"


def shade(s):
    """Cell background for a share S in [0, 1] (accent at increasing opacity)."""
    a = max(0.0, min(1.0, s))
    return f' style="background:color-mix(in srgb, var(--accent) {round(a * 55)}%, transparent)"' if a >= 0.02 else ""


def md_inline(s):
    s = h(s.replace("\\|", "|"))
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", s)


def predictions():
    rows = []
    for line in (R.PROJECT / "docs" / "reports" / "001_leverage_profile.md").read_text().splitlines():
        if line.startswith("| Q"):
            cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]
            rows.append(cells)
    return rows


def top3(lst):
    return ", ".join(f"{h(t.strip() or t)} {math.exp(lp):.2f}" for t, lp in lst[:3])


def region_tokens(tok, fam, name):
    """Tokens of each region of one render, in prompt order (the brew/chain templates fix every non-colour token)."""
    _, bank = wsb.load_bank(fam)
    r = wsb.render_item(tok, fam, next(x for x in bank if x.get("name") == name))
    out = {}
    for pos, g in sorted(r.regions.items()):
        out.setdefault(g, []).append(r.decoded[pos])
    return out


def heat_rows(prof, order, toks, notes):
    rows = []
    for region, group in order:
        for k, t in enumerate(toks[region]):
            es = [prof[f"{region}#{k}|L{L}"] for L in LAYERS]
            rows.append({"group": group, "tok": t, "note": notes.get(f"{region}#{k}", ""),
                         "S": [round(e["median"], 4) for e in es], "ci": [[round(x, 3) for x in e["ci95_items"]] for e in es],
                         "tow": [round(e["frac_toward"], 3) for e in es], "neg": [round(e["frac_negligible"], 3) for e in es]})
    return rows


def exp001(tok):
    """HTML for the 'what to look for' + experiment 001 sections, and the data the page's script needs."""
    s = json.loads((X / "analysis" / "summary.json").read_text())
    ss = json.loads((X / "analysis" / "snapshot_summary.json").read_text())
    sn = json.loads((X / "analysis" / "snapshot.json").read_text())
    q = json.loads((X / "analysis" / "qualitative.json").read_text())
    b, c, sub, dec = s["brew"], s["chain"], s["substitution"], s["decodability"]
    bp, cp = b["profile"], c["profile"]
    tau, eps = s["thresholds"]["tau"], s["thresholds"]["eps"]
    nr = ss["naming_rates"]
    g = s["gate"]

    def S(prof, key, L):
        return prof[f"{key}|L{L}"]["median"]

    # ---------- what to look for ----------
    nla_z = [nr[f"nla|member|cell2|L42|{a}"]["z"] for a in ("real", "shuffled")]
    orc_z = [nr[f"oracle|member|cell2|L40|{a}"]["z"] for a in ("real", "shuffled")]
    orc_off = [nr[f"oracle|member|cell2|L40|{a}"]["off"] for a in ("real", "shuffled")]
    lens_any = max(nr[f"{r}|member|cell2|L{L}|real"]["any_colour"]
                   for r in ("logit_lens", "J_NP_cosine", "J_NP_standard", "J_CB_standard", "R_CB_standard")
                   for L in (20, 24, 28, 32, 36, 40, 42, 44, 48))
    zc = [dec["z"][f"emit_colon|L{L}"]["real"] for L in (40, 42, 44)]
    ac = [dec["answer"][f"emit_colon|L{L}"]["real"] for L in (40, 42, 44)]
    mem60 = dec["membership"]["emit_colon|L60"]["pooled_auc"]
    look = [
        ("Does one cell carry the pair difference?",
         f"Yes, at the “:” cell: median share S = {f2(S(bp, 'emit_colon#0', 42))} at L42 and {f2(S(bp, 'emit_colon#0', 44))} "
         f"at L44, toward the partner in every pair. That is the cell the Oracle and NLA read. Look at the top row of the brew heatmap."),
        ("What content sits there?",
         f"Probes decode the intermediate z (balanced accuracy {f2(min(zc))}–{f2(max(zc))}) and the start colour, but barely "
         f"the answer ({f2(min(ac))}–{f2(max(ac))}) at L40–44. The answer becomes decodable only from L56. "
         "That is the order expected of a two-step computation."),
        ("Is there content that is decodable but not needed?",
         f"Only clearly at L60: whether an off-path colour is in the table is decodable there (AUC {f2(mem60)}), and swapping "
         f"that colour out barely moves the answer ({frac(sub['behaviour_frac_negligible'], sub['n_items'])} items within the bound; the pre-registered line was 90%). "
         "At L40–44 the pooled probe fails the pre-registered rule; a cross-fitted directional test still follows the edit in most items."),
        ("Do readers name the content the answer depends on?",
         f"The NLA names this table's z at {f2(nla_z[0])} vs {f2(nla_z[1])} with a shuffled activation. The Oracle names it at "
         f"{f2(orc_z[0])} vs {f2(orc_z[1])}. "
         + ("No lens names any colour in its top 10 at L20–48. " if lens_any == 0 else
            f"Lenses name a colour at L20–48 in at most {f2(lens_any)} of readouts. ") +
         "This is descriptive, 10 tables, word matching. It is the first explained disagreement, not a reader score."),
        ("Do readers name content that is not needed?",
         f"The verbalizers mention off-path candidates about as often with a shuffled activation as with the real one (Oracle "
         f"{f2(orc_off[0])} vs {f2(orc_off[1])}), and no reader tracks the substitution that the probe decodes at L60."),
        ("Can the measurements be trusted?",
         f"Every exact engine check passes on every pair (self-patch, embedding anchor, prefix, stir no-op: bitwise). The SEC "
         f"noise floor is {b['noise_floor']['upper_L42_rms_frac_G']:.1e} of the gap. Reader inputs captured under stock bf16 and SEC "
         f"agree at cosine ≥ {min(v['min_cos'] for k, v in ss['bridge'].items() if k.startswith('L')):.5f}."),
    ]
    look_html = "".join(f'<li><span class="q">{h(a)}</span><span class="a">{h(bb)}</span></li>' for a, bb in look)

    # ---------- gate ----------
    gate_rows = [("brew start pairs", "dev", g["brew|dev"]), ("brew start pairs", "test", g["brew|test"]),
                 ("chain start pairs", "dev", g["chain|dev"]), ("chain start pairs", "test", g["chain|test"]),
                 ("brew substitution", "dev", g["brew_substitution|dev"])]
    gate_tbl = ('<tr><th>Family</th><th>Split</th><th class="num">Candidates</th><th class="num">Passed</th>'
                '<th class="num">Items with a passing pair</th></tr>' +
                "".join(f'<tr><td>{a}</td><td>{sp}</td><td class="num">{v["candidates"]}</td><td class="num">{v["passed"]}</td>'
                        f'<td class="num">{v.get("items_with_a_passing_pair", v["passed"])} / {v.get("items", v["candidates"])}</td></tr>'
                        for a, sp, v in gate_rows))

    # ---------- brew profile table ----------
    brew_rows = [("emit_colon#0", "“:” after “Answer”"), ("emit_asst#0", "newline after “assistant”"),
                 ("emit_think#0", "“\\n\\n” after “</think>”"), ("start#4", "start-colour token"),
                 ("question#8", "last question token “?”")]
    shown = {k for k, _ in brew_rows}
    rest = max(abs(v["median"]) for k, v in bp.items() if k.split("|")[0] not in shown)

    def prow(prof, key, label, ci_layers=(44, 60)):
        tds = []
        for L in LAYERS:
            e = prof[f"{key}|L{L}"]
            ci = (f'<br><span class="small muted">[{f2(e["ci95_items"][0])}, {f2(e["ci95_items"][1])}]</span>'
                  if L in ci_layers and e["median"] >= tau else "")
            tds.append(f'<td class="num"{shade(e["median"])}>{f2(e["median"])}{ci}</td>')
        return f"<tr><td>{h(label)}</td>{''.join(tds)}</tr>"

    head = '<tr><th>Cell</th>' + "".join(f'<th class="num">L{L}</th>' for L in LAYERS) + "</tr>"
    brew_tbl = (head + "".join(prow(bp, k, lab) for k, lab in brew_rows) +
                f'<tr><td>every other start, question and stir token</td><td class="num" colspan="5">|S| ≤ {rest:.3f}</td></tr>')
    chain_tbl = head + prow(cp, "final#0", "final “\\n\\n” of the question", ci_layers=(56, 60))
    chain_rest = max(abs(v["median"]) for k, v in cp.items() if not k.startswith("final#0") and int(k.split("|L")[1]) <= 44)
    chain_tbl += (f'<tr><td>every other question-span cell, L40–44</td><td class="num" colspan="5">|S| ≤ {chain_rest:.3f}</td></tr>')
    spans = ([("brew", "emission (3 cells)", b["spans"]["emission"]), ("brew", "start sentence", b["spans"]["start"]),
              ("brew", "question", b["spans"]["question"]), ("brew", "start + question + emission", b["spans"]["start+question+emission"]),
              ("chain", "input digits", c["spans"]["input"]), ("chain", "question", c["spans"]["question"])])
    span_tbl = ('<tr><th>Family</th><th>Span (all five layers at once)</th><th class="num">median S</th><th class="num">toward</th></tr>' +
                "".join(f'<tr><td>{fam}</td><td>{lab}</td><td class="num">{f2(e["median"])}</td><td class="num">{pct(e["frac_toward"])}</td></tr>'
                        for fam, lab, e in spans))

    # ---------- controls ----------
    ctl = b["controls"]
    ctl_tbl = ('<tr><th>Check</th><th>Brew</th><th>Chain</th></tr>' +
               "".join(f'<tr><td>{lab}</td><td class="num">{pct(b["controls"][k])}</td><td class="num">'
                       f'{pct(c["controls"][k]) if c["controls"][k] is not None else "n/a"}</td></tr>'
                       for k, lab in [("self_patch_bitwise", "patching a cell with its own value changes nothing (bitwise)"),
                                      ("embedding_anchor_bitwise", "patching the differing token's embedding reproduces the partner (bitwise)"),
                                      ("prefix_identical", "activations before the first differing token are identical"),
                                      ("stir_noop_bitwise", "patching a stir cell (before the difference) changes nothing (bitwise)")]) +
               f'<tr><td>start-colour token at L0–4: median S (positive control, expect 1)</td><td class="num">'
               f'{min(v["median"] for v in ctl["start_token_S"].values()):.3f}–{max(v["median"] for v in ctl["start_token_S"].values()):.3f}</td>'
               f'<td class="num">{min(v["median"] for v in c["controls"]["start_token_S"].values()):.3f}–'
               f'{max(v["median"] for v in c["controls"]["start_token_S"].values()):.3f}</td></tr>'
               f'<tr><td>noise floor: RMS margin change from a 1e-4 nudge at L42, as a fraction of G</td>'
               f'<td class="num">{b["noise_floor"]["upper_L42_rms_frac_G"]:.1e}</td><td class="num">{c["noise_floor"]["upper_L42_rms_frac_G"]:.1e}</td></tr>'
               f'<tr><td>reverse direction (partner ← member) vs forward: Pearson r</td>'
               f'<td class="num">{b["reverse"]["pearson"]:.2f} (n = {b["reverse"]["n"]})</td><td class="num">{c["reverse"]["pearson"]:.2f} (n = {c["reverse"]["n"]})</td></tr>'
               f'<tr><td>pairs checked</td><td class="num">{ctl["n"]}</td><td class="num">{c["controls"]["n"]}</td></tr>')

    # ---------- decodability ----------
    def dcell(e, key="real"):
        v = e[key]
        mark = "<b>" if e["decodable"] else ""
        return f'<td class="num">{mark}{f2(v)}{"</b> ✓" if e["decodable"] else ""}</td>'

    dec_tbl = head.replace("<th>Cell</th>", "<th>Target at the “:” cell</th>")
    for t, lab in [("z", "z, the hidden step (12-way bal. acc.)"), ("start", "start colour"), ("answer", "answer")]:
        dec_tbl += f"<tr><td>{lab}</td>" + "".join(dcell(dec[t][f"emit_colon|L{L}"]) for L in LAYERS) + "</tr>"
    dec_tbl += "<tr><td>off-path colour is in the table (pooled AUC)</td>" + "".join(
        dcell(dec["membership"][f"emit_colon|L{L}"], "pooled_auc") for L in LAYERS) + "</tr>"
    sd = dec["substitution_directional"]
    dec_tbl += ("<tr><td>directional test: removed colour's score falls / inserted colour's rises</td>" +
                "".join(f'<td class="num">{pct(sd[f"emit_colon|L{L}"]["frac_removed_down"])} / {pct(sd[f"emit_colon|L{L}"]["frac_inserted_up"])}</td>'
                        for L in LAYERS) + "</tr>")
    dec_other = head.replace("<th>Cell</th>", "<th>Target · cell</th>")
    for t in ("z", "start", "answer", "membership"):
        for cell in ("emit_asst", "emit_think"):
            dec_other += (f"<tr><td>{t} · {h(CELL_NAME[cell])}</td>" +
                          "".join(dcell(dec[t][f"{cell}|L{L}"], "pooled_auc" if t == "membership" else "real") for L in LAYERS) + "</tr>")

    # ---------- substitution ----------
    tb = sub["transplant_by_cell_layer"]
    sub_tbl = ("<tr><th>Transplant at</th>" + "".join(f'<th class="num">L{L}</th>' for L in LAYERS) + "</tr>" +
               "".join(f"<tr><td>{h(CELL_NAME[cell])}</td>" + "".join(
                   f'<td class="num">{pct(tb[f"{cell}|L{L}"]["frac_negligible"])}</td>' for L in LAYERS) + "</tr>"
                   for cell in ("emit_colon", "emit_asst", "emit_think")))
    ex_rows = ""
    for e in q["substitution_exceeding"]:
        if e["direction"] != "sub->orig":
            continue
        ex_rows += (f'<tr><td><code>{h(e["item"][-4:])}</code></td><td>{h(e["c"])} → {h(e["c2"])}</td><td>{h(e["a"])}</td>'
                    f'<td class="num">{e["behaviour_dlogPa"]:+.3f}</td><td class="num">{e["behaviour_TV"]:.3f}</td>'
                    f'<td class="small">{top3(e["top5_x"])}</td><td class="small">{top3(e["top5_y"])}</td></tr>')
    ex_tbl = ('<tr><th>Table</th><th>Removed → inserted</th><th>Answer</th><th class="num">Δlog P(a)</th><th class="num">TV</th>'
              '<th>Top 3, original</th><th>Top 3, substituted</th></tr>' + ex_rows)

    # ---------- readers ----------
    rs_ = ss["substitution_sensitivity"]
    sens_rows = [("nla", 2, 42, "real"), ("nla", 2, 42, "shuffled"), ("oracle", 2, 40, "real"), ("oracle", 2, 40, "shuffled"),
                 ("oracle", 2, 44, "real"), ("J_NP_cosine", 2, 56, "real"), ("J_NP_cosine", 2, 60, "real"),
                 ("logit_lens", 2, 60, "real"), ("R_CB_standard", 2, 60, "real")]
    sens_tbl = ('<tr><th>Reader at the “:” cell</th><th>Arm</th><th class="num">removed colour c: original → substituted</th>'
                '<th class="num">inserted colour c′: original → substituted</th></tr>' +
                "".join(f'<tr><td>{READER_NAME[r]}, L{L}</td><td>{arm}</td>'
                        f'<td class="num">{f2(rs_[f"{r}|cell{ci}|L{L}|{arm}"]["c_orig"])} → {f2(rs_[f"{r}|cell{ci}|L{L}|{arm}"]["c_sub"])}</td>'
                        f'<td class="num">{f2(rs_[f"{r}|cell{ci}|L{L}|{arm}"]["c2_orig"])} → {f2(rs_[f"{r}|cell{ci}|L{L}|{arm}"]["c2_sub"])}</td></tr>'
                        for r, ci, L, arm in sens_rows))
    fv = ss["nla_fve"]
    fve_tbl = ('<tr><th>NLA explanations</th><th class="num">median FVE against this cell</th><th class="num">n</th></tr>' +
               "".join(f'<tr><td>{lab}</td><td class="num">{f2(fv[k]["median"])}</td><td class="num">{fv[k]["n"]}</td></tr>'
                       for k, lab in [("real|fve_vs_cell", "its own (real)"), ("shuffled|fve_vs_cell", "from another table's cell (shuffled)"),
                                      ("shuffled|fve_vs_source", "shuffled, scored against its own source cell"),
                                      ("real_sec|fve_vs_cell", "real, cell captured under SEC"),
                                      ("none|fve_vs_member_colon", "no activation")]))

    # naming-rate grid data (member activations)
    grid_layers = ss["protocol"]["lens_layers"]
    naming = {}
    for k, v in nr.items():
        r, role, cell, L, arm = k.split("|")
        if role != "member" or arm not in ("real", "shuffled"):
            continue
        naming.setdefault(r, {}).setdefault(cell, {}).setdefault(L[1:], {})[arm] = {
            t: v[t] for t in ("z", "start", "answer", "off", "any_colour")} | {"n": v["n"]}

    # qualitative browser data: every snapshot table, member activation at the ":" cell
    tables = []
    none_or = sn["tables"][0]["readouts"].get("none|L40|oracle_wsb|none")
    none_nla = sn["tables"][0]["readouts"].get("none|L42|nla|none")
    for t in sn["tables"]:
        Rd = t["readouts"]
        nla_r, nla_s = Rd["member|cell2|L42|nla|real"], Rd["member|cell2|L42|nla|shuffled"]
        tables.append({
            "item": t["item"], "start": t["start"], "z": t["z"], "answer": t["answer"], "candidates": t["candidates"],
            "partner": t["partner"], "substitution": t["substitution"], "shuffled_from": t["shuffled_from"],
            "lens": {f"L{L}": {r: Rd[f"member|cell2|L{L}|lenses|real"][r] for r in ("J_NP_cosine", "J_CB_standard", "logit_lens")}
                     for L in (40, 44, 52, 60)},
            "oracle": Rd["member|cell2|L40|oracle|real"], "oracle_shuffled": Rd["member|cell2|L40|oracle|shuffled"][:1],
            "oracle_none": Rd["member|cell2|L40|oracle|none"][:1],
            "nla": [[e, f] for e, f in zip(nla_r["explanations"], nla_r["fve_vs_cell"])],
            "nla_shuffled": [[e, f] for e, f in zip(nla_s["explanations"][:1], nla_s["fve_vs_cell"][:1])],
        })
    nla_none = None
    if isinstance(none_nla, dict):
        nla_none = (none_nla.get("explanations") or [None])[0]

    # ---------- fixed-seed qualitative pairs ----------
    qb, qc = "", ""
    for p in q["pairs"]:
        Sx = p["S_by_region_layer"]
        if p["family"] == "brew":
            st = sorted({int(k.split("@")[1].split("|")[0]) for k in Sx if k.startswith("start@")})[4]
            qb += (f'<tr><td><code>{h(p["pair_id"].split("-")[-1])}</code></td><td>{h(p["a"])} → {h(p["a_donor"])}</td>'
                   f'<td class="num">{p["G"]:.1f}</td><td class="num">{" / ".join(f2(Sx[f"emit_colon@164|L{L}"]) for L in (40, 44, 60))}</td>'
                   f'<td class="num">{f2(Sx[f"start@{st}|L40"])}</td><td class="small">{top3(p["read_cell_top5"]["emit_colon|L44"])}</td></tr>')
        else:
            fk = next(k for k in Sx if k.startswith("final@")).split("|")[0]
            qc += (f'<tr><td><code>{h(p["pair_id"].split("-")[-1])}</code></td><td>{h(p["a"])} → {h(p["a_donor"])}</td>'
                   f'<td class="num">{p["G"]:.1f}</td><td class="num">{" / ".join(f2(Sx[f"{fk}|L{L}"]) for L in LAYERS)}</td>'
                   f'<td class="small">{top3(p["read_cell_top5"]["final|L56"])}</td></tr>')
    qb = ('<tr><th>Pair</th><th>Member → partner answer</th><th class="num">G (nats)</th><th class="num">S at “:”, L40 / 44 / 60</th>'
          '<th class="num">start-token S, L40</th><th>Top 3 after the L44 “:” patch</th></tr>' + qb)
    qc = ('<tr><th>Pair</th><th>Member → partner answer</th><th class="num">G (nats)</th><th class="num">final-cell S, L40 / 42 / 44 / 56 / 60</th>'
          '<th>Top 3 after the L56 patch</th></tr>' + qc)
    gf = "".join(f'<tr><td>{h(x["family"])}</td><td>{h(x["answer"])}</td><td class="num">{x["n_correct"]}/10</td>'
                 f'<td class="q">{h(x["greedy"][:90])}</td><td class="q small">{" · ".join(h(y.strip().splitlines()[0][:40]) for y in x["samples"][:4])}</td></tr>'
                 for x in q["gate_failures_sample"])
    gf = ('<tr><th>Family</th><th>Gold answer</th><th class="num">correct</th><th>Greedy output</th><th>First 4 samples</th></tr>' + gf)

    # ---------- predictions + decisions ----------
    pr = "".join(f'<tr><td>{md_inline(a)}</td><td>{md_inline(bb)}</td><td class="num">{md_inline(cc)}</td><td>{md_inline(d)}</td></tr>'
                 for a, bb, cc, d in predictions())
    pr = '<tr><th>#</th><th>Prediction (before the run)</th><th class="num">Prior</th><th>Outcome</th></tr>' + pr

    exp_html = f"""
<section aria-labelledby="h-look">
  <h2 id="h-look">What to look for</h2>
  <p>Six questions. The answers so far come from experiment 001 on the dev split; nothing has been run on the test split, and no label or reader score exists yet.</p>
  <ol class="look">{look_html}</ol>
</section>

<section aria-labelledby="h-001">
  <div class="eyebrow">Experiment 001 · dev split · SEC regime (true fp32 from block 40) · report: docs/reports/001_leverage_profile.md</div>
  <h2 id="h-001">Experiment 001: where the answer depends on one cell, and what sits there</h2>
  <p><b>What was done.</b> For each gated pair (two prompts that differ only in the start colour), the partner's whole
  activation was copied into the member's run at one token position and one layer, and we measured how much of the
  answer gap G that single edit produces (share S). This was repeated at every start, stir, question and answer cell, at
  L40, 42, 44, 56 and 60. Separately: probes for what is decodable at the answer cells, an <em>off-trajectory
  substitution</em> (swap an unused colour in the table for an absent one; the path and answer stay the same), and a
  descriptive reader snapshot on 10 tables. Brew: {b['n_pairs']} pairs from {b['n_items']} tables. Chain: {c['n_pairs']} pairs
  from {c['n_items']} items (9 step groups).</p>
  <p class="small muted"><b>Why whole-cell patching and not the paper's J-space clamping?</b> A J-coordinate swap uses a reader's
  directions, so a label built from it would grade the J lens on its own axes. The labels use reader-free edits only.
  The whole-cell patch measures leverage of the full pair difference; design 002 adds 1-D swaps along mean-difference
  directions and clamp-to-mean necessity tests.</p>

  <h3>Admission gate (every candidate, dev and test)</h3>
  <p class="small muted">A pair is used only if the model answers both prompts correctly: greedy, and at least 9/10 samples at T = 0.8.</p>
  <div class="scroll"><table class="tbl">{gate_tbl}</table></div>

  <h3>Measurement checks (exact)</h3>
  <div class="scroll"><table class="tbl">{ctl_tbl}</table></div>

  <h3>Share of the answer gap carried by one cell</h3>
  <p class="small muted">Median share S over pairs, for every cell of the prompt × five layers (hover a cell for its interval and the toward / negligible fractions). “·” = |S| &lt; 0.005. No cell moves the answer the wrong way: “away” is 0% everywhere. Oracle reads the “:” cell at L40/44, the NLA at L42.</p>
  <div class="cols2">
    <div style="display:grid;gap:8px"><h3>Brew ({b['n_pairs']} pairs, {b['n_items']} tables)</h3><div class="scroll" id="ex-heat-brew"></div></div>
    <div style="display:grid;gap:8px"><h3>Chain ({c['n_pairs']} pairs, {c['n_items']} items)</h3><div class="scroll" id="ex-heat-chain"></div></div>
  </div>
  <details class="more"><summary>Brew medians with 95% intervals over table means</summary><p class="small muted">Brackets shown where S ≥ τ at L44/L60. τ = {tau}, ε′ = {eps}.</p><div class="scroll"><table class="tbl">{brew_tbl}</table></div></details>
  <ul class="findings">
    <li><b>The “:” cell carries about half the gap at L42–44</b> in every pair (100% toward), rising to {f2(S(bp, 'emit_colon#0', 60))} by L60.</li>
    <li><b>The start-colour token still carries {f2(S(bp, 'start#4', 40))} at L40</b> and nothing by L56: the colour is still read from its own position in the middle layers.</li>
    <li><b>Damage flag.</b> By the pre-registered rule the “:” patches are damage-dominated (46–67% of pairs gain > 0.10 off-pair mass). Most of that mass goes to the two pairs' intermediate colours (z and z′), so the rule overstates harm here. AMENDMENT_02 proposes splitting the readout.</li>
  </ul>
  <div class="cols2">
    <div style="display:grid;gap:8px"><h3>The start-colour token across depth</h3><div class="chart" id="ex-start"></div>
      <p class="small muted">L0–4: patching the differing token reproduces the whole flip (positive control; bars = IQR over pairs). L40–60: brew only, bars = 95% interval over table means. The colour is still read from its own position at L40 and no longer by L56.</p></div>
    <div style="display:grid;gap:8px"><h3>Spans (several cells patched at once)</h3><div class="scroll"><table class="tbl">{span_tbl}</table></div>
      <details class="more"><summary>Chain final cell with intervals</summary><div class="scroll"><table class="tbl">{chain_tbl}</table></div></details></div>
  </div>

  <h3>Decodability at the answer cells (probes)</h3>
  <p class="small muted">L2 logistic regression, 5-fold by table, nested C. Control task (Hewitt–Liang) at 0.05–0.12; ✓ = BH-significant and selectivity ≥ 0.10. Chance ≈ 0.083 for 12 colours.</p>
  <div class="scroll"><table class="tbl">{dec_tbl}</table></div>
  <details class="more"><summary>Same targets at the other two answer cells</summary><div class="scroll"><table class="tbl">{dec_other}</table></div></details>

  <h3>Off-trajectory substitution: content that is present but not needed</h3>
  <p>Swap an unused colour c in the table for an absent colour c′. The path and the answer are unchanged, and the prompts
  stay token-aligned. Bound: |Δlog P(a)| &lt; {sub['bound']['abs_dlogPa']:.2f} nats and TV &lt; {sub['bound']['TV']}.
  <b>Behaviour:</b> {frac(sub['behaviour_frac_negligible'], sub['n_items'])} items within the bound (median |Δlog P(a)| {sub['behaviour_abs_dlogPa_median']:.3f}).
  <b>Transplanting</b> the substituted state at the answer cells: {frac(sub['transplant_all_cells_frac_negligible'], sub['n_items'])} items within the bound at every cell × layer.</p>
  <div class="cols2">
    <div style="display:grid;gap:8px"><h3>Worst item per transplant cell</h3><div class="chart" id="ex-transplant"></div>
      <p class="small muted">Largest |Δlog P(a)| over the {sub['n_items']} items when the substituted state is transplanted at one answer cell. Dashed: the bound.</p></div>
    <div style="display:grid;gap:8px"><h3>Share within the bound, per transplant cell</h3><div class="scroll"><table class="tbl">{sub_tbl}</table></div></div>
  </div>
  <p class="small muted">The items over the bound all fail only TV, and narrowly: a little tail probability moves from the removed colour to the inserted one. So the answer does not depend on the swap, but its tail carries a trace of which colours are in the table.</p>
  <div class="scroll"><table class="tbl">{ex_tbl}</table></div>

  <h3>Reader snapshot (descriptive; 10 dev tables; not a score)</h3>
  <p>Fraction of readouts naming this table's colour, from the member's own activation (real) and from the same cell of a
  different table (shuffled = chance). Colours matched by word, English and Chinese. Choose a target and a cell.</p>
  <div class="seg" id="ex-target"></div>
  <div class="seg" id="ex-cell"></div>
  <div class="scroll"><table class="tbl grid" id="ex-grid"></table></div>
  <p class="small muted">Each entry shows real / shuffled. Empty = that reader wasn't run at that cell × layer (Oracle contract: “:” at L40/44; NLA: L42).</p>
  <div class="cols2">
    <div style="display:grid;gap:8px"><h3>Does any reader track the substitution?</h3><div class="scroll"><table class="tbl">{sens_tbl}</table></div>
      <p class="small muted">The changes sit within the shuffled-arm noise at n = 10. The cross-fitted probe follows the same edit in 98% of items at L60.</p></div>
    <div style="display:grid;gap:8px"><h3>NLA reconstruction</h3><div class="scroll"><table class="tbl">{fve_tbl}</table></div>
      <p class="small muted">Shuffled explanations reconstruct this cell about as well as its own. On near-identical prompts, FVE reflects the shared template, not item-specific content.</p></div>
  </div>

  <h3>Qualitative: every snapshot table, raw reader outputs at the “:” cell</h3>
  <div class="seg" id="ex-tables"></div>
  <div id="ex-browser" class="panel"></div>

  <h3>Qualitative: fixed-seed pairs (seed {q['seed']})</h3>
  <div class="scroll"><table class="tbl">{qb}</table></div>
  <div class="scroll"><table class="tbl">{qc}</table></div>
  <p class="small muted">0140__brown: the runner-up after the patch is brown, the partner's <em>start</em> colour, not an intermediate. 0013__5: the first answer token is “1” in both prompts, so the effect sits in the second digit, which is why every answer token is scored.</p>
  <details class="more"><summary>Gate failures (sample of 12): what the model says when it isn't reliable</summary><div class="scroll"><table class="tbl">{gf}</table></div></details>

  <h3>Predictions written before the run, and what happened</h3>
  <div class="scroll"><table class="tbl">{pr}</table></div>

  <h3>Decision for you (before any test-split label)</h3>
  <ul class="plain">
    <li>Keep τ = {tau} and ε′ = {eps} (the profile gives no reason to move them) and freeze the substitution bound.</li>
    <li>Split the damage readout into “intermediate / colour answers” vs “non-colour mass”, plus top-1 retention.</li>
    <li><b>Your call:</b> should reader scoring of the not-needed class use L60 only (lenses), or also L40–44 via the directional test (where the Oracle and NLA read)?</li>
  </ul>
</section>
"""
    bt, ct = region_tokens(tok, "brew", "brew-s2-r10d2-0000"), region_tokens(tok, "chain", "chain-w2-d3p-0408")
    heat = {
        "layers": LAYERS,
        "brew": heat_rows(bp, [("stir", "stir"), ("start", "start"), ("question", "question"), ("emit_asst", "answer cells"),
                               ("emit_think", "answer cells"), ("emit_colon", "answer cells")], bt,
                          {"start#4": "start colour", "emit_asst#0": "after “assistant”",
                           "emit_think#0": "after “</think>”", "emit_colon#0": "after “Answer”"}),
        "chain": heat_rows(cp, [("question", "question"), ("final", "final")], ct,
                           {"question#20": "after “assistant”", "final#0": "after “</think>”"}),
    }
    ctl_c = c["controls"]["start_token_S"]
    start = {"brew_ctrl": [[int(L), v["median"], v["q25"], v["q75"]] for L, v in sorted(ctl["start_token_S"].items(), key=lambda x: int(x[0]))],
             "chain_ctrl": [[int(L), v["median"], v["q25"], v["q75"]] for L, v in sorted(ctl_c.items(), key=lambda x: int(x[0]))],
             "brew_prof": [[L, bp[f"start#4|L{L}"]["median"], *bp[f"start#4|L{L}"]["ci95_items"]] for L in LAYERS]}
    transplant = {"bound": sub["bound"]["abs_dlogPa"],
                  "series": {cell: [[L, tb[f"{cell}|L{L}"]["max_abs_dlogPa"], tb[f"{cell}|L{L}"]["median_abs_dlogPa"]] for L in LAYERS]
                             for cell in ("emit_colon", "emit_asst", "emit_think")}}
    data = {"grid_layers": grid_layers, "naming": naming, "tables": tables, "heat": heat, "start": start, "transplant": transplant,
            "oracle_none": none_or, "nla_none": nla_none}
    return exp_html, data


STATUS_HTML = """
<section aria-labelledby="h-status">
  <h2 id="h-status">Status</h2>
  <div class="status">
    <div class="ok"><h3>Works and is tested</h3><ul class="plain">
      <li>Pinned environment: model loads, 29/29 hashes match, and a rebuild script recreates everything.</li>
      <li>WorkspaceBench renders byte-identical to WSB's producer; partners, substitutions and the gate.</li>
      <li>Margin scorer, patching engines (RHUT cross-check, SEC true-fp32 replay), stats, figures, and a model-integrity guard. 78 tests pass.</li>
      <li>Readers: logit lens, J_NP, J_CB, R_CB, Oracle Lens (L40), NLA (L42), plus their positive controls.</li>
      <li>Experiment 001 (dev): gate, smoke, run, probes, reader snapshot, analysis, figures, report.</li>
    </ul></div>
    <div class="no"><h3>Not working, parked or dropped</h3><ul class="plain">
      <li><b>vLLM acceleration:</b> the engine won't start on this GPU (sm_120 check inside FlashInfer). Parked; the HF route works.</li>
      <li><b>SAE:</b> deferred until you decide. One candidate is noted (ceselder/qwen36-27b-bsf-l42-1b).</li>
      <li><b>Skip-lens:</b> dropped. Its checkpoints are trained at L62, not L42.</li>
      <li><b>Raw outputs</b> are only on this instance (no persistent volume; the HF mirror has no name yet). Committed results and figures are safe in git.</li>
    </ul></div>
    <div class="todo"><h3>Waiting on you, then next</h3><ul class="plain">
      <li>Review report 001 and the AMENDMENT_02 proposals (above).</li>
      <li>HF dataset name for the raw outputs; paper PDFs in a public repo; merge to main.</li>
      <li>Then design 002, the labels: 1-D swaps along reader-free mean-difference directions, clamp-to-mean necessity tests, random-direction nulls. Shown to you before any run.</li>
    </ul></div>
  </div>
</section>
"""


def main():
    rs = json.loads((A / "reader_smoke.json").read_text())
    rc = json.loads((A / "regime_check.json").read_text())
    tm = json.loads((A / "timing_smoke.json").read_text())
    ec = json.loads((A / "env_check.json").read_text())
    tok = R.make_tokenizer()
    _, bank = wsb.load_bank("brew")
    it = next(x for x in bank if x["name"] == "brew-s2-r10d2-0000")
    r = wsb.render_item(tok, "brew", it)
    pair = {p.p.inp: p for p in P.brew_candidates(it)}["pink"]
    pc = rs["vector_lenses_positive_control"]
    exp_html, exp_data = exp001(tok)
    data = {
        "prompt_tokens": [[i, t, r.regions.get(i, "")] for i, t in enumerate(r.decoded)],
        "item": {"name": it["name"], "start": it["start"], "z": it["intermediates"][0], "answer": it["answer"],
                 "partner_start": pair.p.inp, "partner_z": pair.p.z, "partner_answer": pair.p.a,
                 "off": it["off_trajectory"], "candidates": it["options_adjacent"][0]},
        "env": {"greedy": ec["greedy"].strip().split()[0], "top5": ec["top5_stock_bf16"], "load_s": round(ec["load_s"], 1),
                "gpu": ec["gpu"]},
        "lenses": {L: {k: [t for t, _ in v["top10"]] for k, v in rows.items()}
                   for L, rows in rs["vector_lenses"]["readouts"].items()},
        "late": {**{f"logit_lens {k}": [t for t, _ in v["top10"]] for k, v in pc["logit_lens"].items()},
                 **{k.replace("_", " ", 1): [t for t, _ in v["top10"]] for k, v in pc["J"].items()
                    if "standard" in k and "J_NP" in k}},
        "oracle": {a: rs["oracle_lens_L40"][a]["samples"] for a in ("activation", "no_activation", "other_cell")},
        "nla": {a: rs["nla_L42_av"][a]["explanations"] for a in ("activation", "no_activation", "other_cell")},
        "fve": {"activation": rs["nla_L42_ar"]["fve_activation_explanations_vs_cell"],
                "no_activation": rs["nla_L42_ar"]["fve_no_activation_explanations_vs_cell"],
                "other_cell": rs["nla_L42_ar"]["fve_other_cell_explanations_vs_cell"],
                "other_own": rs["nla_L42_ar"]["fve_other_cell_explanations_vs_other_cell"],
                "reference": rs["nla_L42_ar"]["fve_reference"],
                "reextract_min_cos": rs["nla_reference_reextraction"]["min_cos"]},
        "timing": {k: {"s": rs[k]["seconds"], "peak": rs[k]["gpu_peak_gb"]}
                   for k in ("vector_lenses", "oracle_lens_L40", "nla_L42_av", "nla_L42_ar")},
        "regime": {"pair": rc["pair_id"], "member": rc["member"], "partner": rc["partner"],
                   "edits": [{"name": e["name"], "rhut": rc["regimes"]["rhut_fp32res_from0"]["edits"][e["name"]]["dm"],
                              "sec": rc["regimes"]["sec_true_fp32_from40"]["edits"][e["name"]]["dm"],
                              "bf16": rc["regimes"]["stock_bf16"]["edits"][e["name"]]["dm"]} for e in rc["edits"]],
                   "G": {k: v["G"] for k, v in rc["regimes"].items()},
                   "noise": {k: v["noise_floor"]["max_abs_dm"] for k, v in rc["regimes"].items()},
                   "max_frac": {k: v["max_abs_diff_over_G"] for k, v in rc["comparison_vs_sec"].items()}},
        "gates": [{"pair": a["pair_id"], "pass": a["gate"]["pass"], "m": a["gate"]["m"]["n_correct"],
                   "p": a["gate"]["p"]["n_correct"],
                   "p_samples": [s.strip().split("\n")[0][:14] for s in a["gate"]["p"]["samples"]]}
                  for a in rc["gate_attempts"]]
                 + [{"pair": a["pair_id"], "pass": a["pass"], "m": None, "p": None, "p_samples": []}
                    for a in tm["chain_gate_attempts"]],
        "exp001": exp_data,
    }
    t = open(os.path.join(HERE, "template.html")).read()
    for key in ("__EXP001__", "__STATUS__", "__DATA__"):
        assert t.count(key) == 1, key
    # Lens readouts include byte-fragment tokens that decode to U+FFFD; write it as an escape so the page has no raw one.
    js = json.dumps(data, ensure_ascii=False).replace("</", "<\\/").replace("\ufffd", "\\ufffd")
    page = (t.replace("__EXP001__", exp_html.replace("\ufffd", "&#xFFFD;")).replace("__STATUS__", STATUS_HTML)
             .replace("__DATA__", js))
    assert "\ufffd" not in page
    out = R.out_dir("outputs", "000_setup", "results_page") / "causal_concordance.html"
    out.write_text(page)
    print(out, len(page))


if __name__ == "__main__":
    main()
