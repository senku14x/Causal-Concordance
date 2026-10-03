"""Build the results page (published as a private claude.ai artifact) from committed results only.

    python scripts/000_setup/results_page/make_page.py  ->  outputs/000_setup/results_page/causal_concordance.html

Block 000 comes from `results/000_setup/analysis/*.json`; experiment 001 from `results/001_leverage_profile/
{analysis/*.json, figures/*.png}` and the predictions table of `docs/reports/001_leverage_profile.md` §10.
"""
import html
from decimal import ROUND_HALF_UP, Decimal
import json
import math

import numpy as np
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
    au = json.loads((X / "analysis" / "audit_followups.json").read_text())
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
    rd = au["readers"]
    trk = {k: rd[k]["start_pair_tracking"] for k in rd}
    look = [
        ("Does one cell carry the pair difference?",
         f"At the “:” cell the whole-cell patch moves the answer about halfway to the partner's: median share S = "
         f"{f2(S(bp, 'emit_colon#0', 42))} at L42 and {f2(S(bp, 'emit_colon#0', 44))} at L44, toward the partner in every pair. "
         "That is the cell the Oracle and NLA read. But the patch swaps start, z, answer and everything downstream together, "
         "so it cannot say which content carries the effect."),
        ("What content sits there?",
         f"The linear probe decodes the intermediate z (balanced accuracy {f2(min(zc))}–{f2(max(zc))}) and the start colour well "
         f"at L40–44, and the answer poorly ({f2(min(ac))}–{f2(max(ac))}); it decodes the answer well from L56. Consistent with a "
         "two-step computation, but a weak probe is not evidence that the answer is absent."),
        ("Is there content that is decodable but not needed?",
         f"Yes: off-path table colours. Swapping one out barely moves the answer ({frac(sub['behaviour_frac_negligible'], sub['n_items'])} "
         f"items within the bound; the line was 90%). After the probe repair (amendment A2) membership is decodable by the rule at "
         f"the “:” cell at every layer: clearly at L60 (AUC {f2(mem60)}), narrowly at L40–44 (selectivity 0.11–0.13, just above "
         "the 0.10 line)."),
        ("Do readers name the content the answer depends on?",
         "Not scored against a causal label yet, because no reliable label for the intermediate exists (003, 004). Descriptively: the NLA mentions this table's z in 17 of 40 samples at “:” (shuffled 8 of 40), "
         "on 6 of 10 tables, and gives z the right role in about 5 of those 17. A within-table contrast (swap the start, which "
         f"changes z) shows some activation-sensitive signal: NLA {f2(trk['nla|cell2|L42']['mean'])}–{f2(trk['nla|cell0|L42']['mean'])}, "
         f"Oracle {f2(trk['oracle|cell2|L40']['mean'])}–{f2(trk['oracle|cell2|L44']['mean'])}. "
         + ("No lens lists a palette colour at L20–48. " if lens_any == 0 else f"Lenses list a palette colour at L20–48 in at most {f2(lens_any)} of readouts. ") +
         "Exploratory, 10 tables, word matching. Experiment 002 (role-aware scoring, below) sharpens this: the verbalizers rarely "
         "claim z in its role (NLA 5–15%, Oracle 0–3%), while the token lenses, read inside the candidate set, track z under the "
         "start swap in 47–48 of 48 tables at L40–44 (positive difference-in-differences of scores; actual rank reversals in "
         "about half of the comparisons)."),
        ("Do readers name content that is not needed?",
         f"Off-path candidates are mentioned about as often with a shuffled activation as with the real one (Oracle "
         f"{f2(orc_off[0])} vs {f2(orc_off[1])}), and no substitution tracking was detected. Both are exploratory nulls at n = 10."),
        ("Can the measurements be trusted?",
         f"The patching engine, yes: every exact check passes on every pair (self-patch, embedding anchor, prefix, stir no-op: "
         f"bitwise), and the SEC noise floor is {b['noise_floor']['upper_L42_rms_frac_G']:.1e} of the gap. The probe controls and "
         "null were repaired and re-run (amendment A2; controls now at 0.48–0.52). 001's damage readout cannot say where "
         "most off-pair mass goes (only the top 5 tokens were saved); 004 fixed this by scoring all 12 colours on every edit. "
         "Every report was audited externally, and every audit number was recomputed."),
        ("Is there a clean causal handle on the intermediate itself?",
         "Not yet. One fitted z direction moves a small, consistent share of the answer (003, median 0.03–0.05 at “:” L40–44), "
         "but the labels are not reproducible. Swapping the whole fitted z subspace moves more (004), but no more than a PCA "
         "subspace of the same size, and most of what it moves goes to the recipient's own intermediate rather than the donor's "
         "answer. So readers are not yet scored against causal labels."),
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

    est = au["brew_estimands"]
    est_key = {"emit_colon#0": "emit_colon", "emit_asst#0": "emit_asst", "emit_think#0": "emit_think", "start#4": "start_colour",
               "question#8": "question_last"}

    def prow(prof, key, label, ci_layers=(44, 60), boot=None):
        tds = []
        for L in LAYERS:
            e = prof[f"{key}|L{L}"]
            ci_v = (boot or {}).get(f"{est_key.get(key, '')}|L{L}", {}).get("pair_median_boot95_tables") if boot is not None else None
            ci = (f'<br><span class="small muted">[{f2(ci_v[0])}, {f2(ci_v[1])}]</span>'
                  if ci_v and L in ci_layers and e["median"] >= tau else "")
            tds.append(f'<td class="num"{shade(e["median"])}>{f2(e["median"])}{ci}</td>')
        return f"<tr><td>{h(label)}</td>{''.join(tds)}</tr>"

    head = '<tr><th>Cell</th>' + "".join(f'<th class="num">L{L}</th>' for L in LAYERS) + "</tr>"
    brew_tbl = (head + "".join(prow(bp, k, lab, ci_layers=LAYERS, boot=est) for k, lab in brew_rows) +
                f'<tr><td>every other start, question and stir token</td><td class="num" colspan="5">|S| ≤ {rest:.3f}</td></tr>')
    cg = au["chain_by_step_group"]
    chain_tbl = head + prow(cp, "final#0", "final “\\n\\n” of the question", ci_layers=(56, 60),
                            boot={f"|L{L}": {"pair_median_boot95_tables": cg[f"final|L{L}"]["pair_median_boot95_groups"]} for L in LAYERS})
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

    dm = au["damage"]
    dmg_tbl = ("<tr><th>“:” cell</th>" + "".join(f'<th class="num">L{L}</th>' for L in (42, 44, 60)) + "</tr>" +
               "".join(f"<tr><td>{lab}</td>" + "".join(f'<td class="num">{fn(dm[f"emit_colon|L{L}"])}</td>' for L in (42, 44, 60)) + "</tr>"
                       for lab, fn in [("pairs flagged (off-pair increase > 0.10)", lambda d: d["flagged"]),
                                       ("off-pair mass, clean → edited", lambda d: f'{f2(d["off_pair_clean"])} → {f2(d["off_pair_edit"])}'),
                                       ("edited: z and z′ within the saved top 5", lambda d: f2(d["intermediates"])),
                                       ("edited: other colours and format tokens within the top 5", lambda d: f2(d["other_colours"] + d["format_or_other"])),
                                       ("edited: mass outside the saved top 5", lambda d: f2(d["mass_outside_top5"]))]))
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

    TR = [("nla|cell0|L42", "NLA L42, assistant newline"), ("nla|cell1|L42", "NLA L42, “</think>” newline"), ("nla|cell2|L42", "NLA L42, “:”"),
          ("oracle|cell2|L40", "Oracle L40, “:”"), ("oracle|cell2|L44", "Oracle L44, “:”")]
    trk_tbl = ('<tr><th>Reader, cell</th><th class="num">start-swap tracking</th><th class="num">tables + / − / tie</th>'
               '<th class="num">real − shuffled (z)</th><th class="num">own z − other tables\' z</th></tr>' +
               "".join(f'<tr><td>{h(lab)}</td><td class="num">{f2(rd[k]["start_pair_tracking"]["mean"])}</td>'
                       f'<td class="num">{rd[k]["start_pair_tracking"]["pos"]} / {rd[k]["start_pair_tracking"]["neg"]} / {rd[k]["start_pair_tracking"]["tie"]}</td>'
                       f'<td class="num">{f2(rd[k]["real_minus_shuffled"]["mean"])}</td><td class="num">{f2(rd[k]["own_z_minus_other_tables_z"]["mean"])}</td></tr>'
                       for k, lab in TR))
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
  <p>Seven questions, answered from experiments 001–004 (dev split only; nothing has been run on the test split). The section above tells the same story in order.</p>
  <ol class="look">{look_html}</ol>
</section>

<section aria-labelledby="h-001">
  <div class="eyebrow">Experiment 001 · dev split · SEC regime (true fp32 from block 40) · report: docs/reports/001_leverage_profile.md</div>
  <h2 id="h-001">Experiment 001: where the answer depends on one cell, and what sits there</h2>
  <p class="small" style="background:var(--warn-soft);color:var(--ink);padding:10px 14px;border-radius:8px"><b>Revised after an external audit (2026-10-02).</b> Every number was recomputed independently from the saved outputs and matched. Corrected: the damage reading, reader mention counts read as claims, the reconstruction and null wording, and interval estimands. The probe controls and null have since been repaired and re-run (amendment A2). Defensible summary: the readers share task-structure information, show some activation-sensitive colour signal, and often surround it with incorrect detail. Whether their claims track causally influential content is unresolved. Report §15 has the point-by-point table.</p>
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
  <details class="more"><summary>Brew medians with 95% intervals</summary><p class="small muted">Median over pairs; brackets: 95% bootstrap of that median, resampling tables (shown where S ≥ τ). τ = {tau}, ε′ = {eps}.</p><div class="scroll"><table class="tbl">{brew_tbl}</table></div></details>
  <ul class="findings">
    <li><b>The “:” cell carries about half the gap at L42–44</b> in every pair (100% toward), rising to {f2(S(bp, 'emit_colon#0', 60))} by L60.</li>
    <li><b>The start-colour token still carries {f2(S(bp, 'start#4', 40))} at L40</b> and nothing by L56: the colour is still read from its own position in the middle layers.</li>
    <li><b>Damage flag: it stands.</b> By the pre-registered rule the “:” patches are damage-dominated (46–67% of pairs gain > 0.10 off-pair mass). An earlier version said most of that mass goes to the intermediate colours; the saved top-5 tokens cannot show that (table below). AMENDMENT_02 proposes logging the full answer-colour distribution.</li>
  </ul>
  <div class="scroll"><table class="tbl">{dmg_tbl}</table></div>
  <div class="cols2">
    <div style="display:grid;gap:8px"><h3>The start-colour token across depth</h3><div class="chart" id="ex-start"></div>
      <p class="small muted">L0–4: patching the differing token reproduces the whole flip (positive control; bars = IQR over pairs). L40–60: brew only, bars = 95% interval over table means. The colour is still read from its own position at L40 and no longer by L56.</p></div>
    <div style="display:grid;gap:8px"><h3>Spans (several cells patched at once)</h3><div class="scroll"><table class="tbl">{span_tbl}</table></div>
      <details class="more"><summary>Chain final cell with intervals (bootstrap over the 9 step groups)</summary><div class="scroll"><table class="tbl">{chain_tbl}</table></div></details></div>
  </div>

  <h3>Decodability at the answer cells (probes)</h3>
  <p class="small muted">L2 logistic regression, 5-fold by table, nested C. Control task (Hewitt–Liang) at 0.05–0.12; ✓ = BH-significant and selectivity ≥ 0.10. Chance ≈ 0.083 for 12 colours. Repaired run (amendment A2): per-fold AUC; 20 (membership) / 10 (12-way) control draws per target, membership controls now 0.48–0.52; full-procedure permutation nulls. Membership passes at “:” L40–60, narrowly at L40–44.</p>
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
  <p>Fraction of readouts <em>mentioning</em> this table's colour, from the member's own activation (real) and from the same
  cell of a different table (shuffled = chance). Colours matched by word over the 12 brew colours, English and Chinese.
  <b>A mention is not a claim:</b> the match does not check the role the reader gives the colour. Lens top-10 lists and long
  descriptions are not comparable claim budgets. Choose a target and a cell.</p>
  <div class="seg" id="ex-target"></div>
  <div class="seg" id="ex-cell"></div>
  <div class="scroll"><table class="tbl grid" id="ex-grid"></table></div>
  <p class="small muted">Each entry shows real / shuffled. Empty = that reader wasn't run at that cell × layer (Oracle contract: “:” at L40/44; NLA: L42).</p>
  <h3>Does the mention follow the activation? (post-hoc contrasts, from the audit)</h3>
  <p class="small muted">Start-swap tracking: keep the table fixed and swap in the partner's start, which changes the correct z; per table, [(member names z − partner names z) + (partner names z′ − member names z′)] / 2 over 4 samples each. “Own z − other tables' z” controls for the readers' favourite colours (the NLA names red in 62% of readouts, the Oracle blue in 93%). Oracle and NLA read different layers; exploratory, n = 10.</p>
  <div class="scroll"><table class="tbl">{trk_tbl}</table></div>
  <p class="small muted">Role reading (author judgment, not a score) of the 17 NLA mentions of z at “:”: 5 give z its correct role (the first step's result, which the next step continues from), 7 make it the start of a two-step chain, 2 offer it as the answer, 3 are incidental. The full list is in <code>audit_followups.json</code>.</p>
  <div class="cols2">
    <div style="display:grid;gap:8px"><h3>Does any reader track the substitution?</h3><div class="scroll"><table class="tbl">{sens_tbl}</table></div>
      <p class="small muted">The changes sit within the shuffled-arm noise at n = 10. An exploratory null: it does not show the readers cannot report the change, and the probe's 98% at L60 says nothing about verbalizers reading L40–44.</p></div>
    <div style="display:grid;gap:8px"><h3>NLA reconstruction</h3><div class="scroll"><table class="tbl">{fve_tbl}</table></div>
      <p class="small muted">Shuffled explanations reconstruct this cell about as well as its own, so FVE is dominated by the shared template. But it carries a small item signal: a shuffled explanation scores higher against its true source than against the wrong cell in {au["fve_source_vs_target"]["source_wins"]} of {au["fve_source_vs_target"]["n_explanations"]} cases (mean +{au["fve_source_vs_target"]["mean_source_minus_target"]:.3f}).</p></div>
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
    <li>Log the full answer-colour distribution at flagged cells; report intermediate mass, non-colour mass and top-1 retention separately.</li>
    <li>Probes: repaired and re-run (A2). The not-needed class now exists at L40–44 too, but narrowly; decide whether to score it there.</li>
    <li>Reader scoring: mention vs correct role vs contradicted, with abstention, matched claim budgets, within-table contrasts and a prompt-only baseline; frozen on dev.</li>
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



X2 = R.PROJECT / "results" / "002_claim_scoring"


def exp002():
    """HTML + data for experiment 002 (reader claim scoring). Empty if the analysis has not run."""
    f = X2 / "analysis" / "scores.json"
    if not f.exists():
        return "", None
    A = json.loads(f.read_text())
    Q = json.loads((X2 / "analysis" / "qualitative.json").read_text())
    v, gate = A["validation"], A["validation_gate"]
    # stacked outcome bars (member renders): real vs shuffled
    arms = [("nla", 0, 42, "NLA L42, after assistant"), ("nla", 1, 42, "NLA L42, after </think>"), ("nla", 2, 42, "NLA L42, “:”"),
            ("oracle", 2, 40, "Oracle L40, “:” (k=4)"), ("oracle", 2, 44, "Oracle L44, “:” (k=4)")] + \
           [("oracle_wsb", 2, L, f"Oracle WSB grid, “:” L{L}") for L in range(20, 61, 8)]
    bars = []
    for rd, cell, L, lab in arms:
        for arm in ("real", "shuffled"):
            k = f"{rd}|member|cell{cell}|L{L}|{arm}"
            if k in A["rates"]:
                r = A["rates"][k]
                bars.append({"label": lab, "arm": arm, "n": r["n"], "correct": r["int_correct"], "other": r["int_other_candidate"],
                             "none": r["int_none"], "ct": r["int_cannot_tell"], "conf": r["int_conflicting"], "mention": r["mention_z"]})
    # contrasts table
    crow = ""
    for key, lab in [("nla|cell0|L42", "NLA L42, after assistant"), ("nla|cell1|L42", "NLA L42, after </think>"), ("nla|cell2|L42", "NLA L42, “:”"),
                     ("oracle|cell2|L40", "Oracle L40, “:”"), ("oracle|cell2|L44", "Oracle L44, “:”")]:
        c = A["contrasts"].get(key, {})
        def cell_(x):
            if not x:
                return '<td class="num">—</td>'
            return f'<td class="num">{f2(x["mean"])}<br><span class="small muted">{x["pos"]}+ / {x["neg"]}− / {x["tie"]}=, p {f2(x["signflip_p_unadj"])}</span></td>'
        br = c.get("end_to_end_bridge_intermediate_agreement") or {}
        crow += (f"<tr><td>{h(lab)}</td>{cell_(c.get('start_swap_tracking_roles'))}{cell_(c.get('real_minus_shuffled_roles'))}"
                 f"{cell_(c.get('real_minus_none_roles'))}<td class=\"num\">{f2(br['agreement']) if br.get('agreement') is not None else '—'}</td></tr>")
    ctbl = ('<tr><th>Reader, cell</th><th class="num">start-swap tracking<br>(z in the intermediate role)</th><th class="num">real − shuffled</th>'
            '<th class="num">real − no activation</th><th class="num">same claim under SEC capture<br>(end-to-end: reader + activation + judge)</th></tr>' + crow)
    # validation
    vt = ('<tr><th>Role</th><th class="num">exact agreement</th><th class="num">Cohen κ (labels)</th><th class="num">κ (correct / other / none / cannot tell / conflicting)</th><th class="num">n</th></tr>' +
          "".join(f'<tr><td>{r}</td><td class="num">{f2(v[r]["exact_agreement"])}</td><td class="num">{f2(v[r]["kappa_labels"])}</td>'
                  f'<td class="num">{f2(v[r]["kappa_categories"])}</td><td class="num">{v[r]["n"]}</td></tr>' for r in ("start", "intermediate", "answer")))
    conf = "".join(f"<tr><td>{h(k)}</td><td class=\"num\">{n}</td></tr>" for k, n in list(v["intermediate"]["confusion"].items())[:10])
    # predictions
    pr = "".join(f'<tr><td>{h(k)}</td><td class="small">{h(json.dumps(x["value"], default=str)[:120])}</td><td>{"✓ true" if x["true"] else ("✗ false" if x["true"] is not None else "—")}</td></tr>'
                 for k, x in A["predictions"].items())
    po = A["prompt_only"]
    P = A["lens_post_hoc"]
    lt = ('<tr><th>Lens, “:” cell</th><th class="num">layer</th><th class="num">z first among candidates<br>(real / shuffled)</th>'
          '<th class="num">tracks z<br>(tables + / −)</th><th class="num">tracks start</th><th class="num">tracks answer</th></tr>')
    for rd, nm in (("logit", "Logit lens"), ("J_NP_standard", "J_NP"), ("J_NP_cosine", "J_NP cosine"), ("J_CB_standard", "J_CB"), ("R_CB_standard", "R_CB")):
        for L in (40, 42, 44, 60):
            k = f"{rd}|emit_colon|L{L}"
            T_ = P["start_swap_tracking_logscore"]
            lt += (f'<tr><td>{nm}</td><td class="num">L{L}</td><td class="num">{f2(P["member_hit_z"][k])} / {f2(P["shuffled_hit_z"][k]["hit_z_shuffled"])}</td>'
                   f'<td class="num"><b>{T_[k]["pos"]} / {T_[k]["neg"]}</b></td><td class="num">{T_[k + "|start"]["pos"]} / {T_[k + "|start"]["neg"]}</td>'
                   f'<td class="num">{T_[k + "|answer"]["pos"]} / {T_[k + "|answer"]["neg"]}</td></tr>')
    ovr = "".join(f'<tr><td>{h(k.split("|")[0])}</td><td>{h(k.split("|")[1].replace("_", " "))}</td><td class="num">{v.get("both", v.get("tables"))}</td>'
                  f'<td class="num">{v.get("only_first", "")}</td><td class="num">{v.get("only_second", "")}</td><td class="num">{v.get("neither", "")}</td></tr>'
                  for k, v in A["overlap"]["pairs"].items())
    data = {"bars": bars, "lens": A["lens_closed_set"], "prompt_only": po, "qual": Q, "top10": A["lens_top10_claims"]}
    rt = A["rates"]
    nla_r = [rt[f"nla|member|cell{c}|L42|real"]["int_correct"] for c in (0, 1, 2)]
    nla_s = [rt[f"nla|member|cell{c}|L42|shuffled"]["int_correct"] for c in (0, 1, 2)]
    orc_r = [rt[f"oracle|member|cell2|L{L}|real"]["int_correct"] for L in (40, 44)]
    nla_m = [rt[f"nla|member|cell{c}|L42|real"]["mention_z"] for c in (0, 1, 2)]
    trk = A["lens_post_hoc"]["start_swap_tracking_logscore"]
    lz = [trk[f"{r}|emit_colon|L{L}"]["pos"] for r in ("J_NP_cosine", "J_CB_standard", "R_CB_standard") for L in (40, 42, 44)]
    ls = [trk[f"{r}|emit_colon|L{L}|start"]["pos"] for r in ("J_NP_cosine", "J_CB_standard", "R_CB_standard") for L in (40, 42, 44)
          if f"{r}|emit_colon|L{L}|start" in trk]
    tk2 = takeaway([
        f"<b>The verbalizers rarely put the right intermediate in its role.</b> NLA L42: {pc(min(nla_r))}–{pc(max(nla_r))} of samples with the real activation (shuffled {pc(min(nla_s))}–{pc(max(nla_s))}), although the colour word itself appears in {pc(min(nla_m))}–{pc(max(nla_m))}. Oracle at “:”: {pc(min(orc_r))}–{pc(max(orc_r))}.",
        f"<b>The token lenses follow the intermediate inside the candidate set.</b> When the start is swapped (which changes z), J_NP cosine, J_CB and R_CB move their scores toward the new z in {min(lz)}–{max(lz)} of 48 tables at “:” L40–44" + (f", but toward the new start in only {min(ls)}–{max(ls)}." if ls else "."),
        f"<b>The judge is reliable enough to use</b>: κ {f2(gate['kappa_intermediate_labels'])} with blind author codes on the intermediate role (gate 0.6).",
        f"<b>The prompt alone gives the intermediate away</b>: the stock model, asked directly, names z top-1 in {pc(po['intermediate']['top1_of_12'])} of renders. A reader could, in principle, infer z from the prompt rather than read it from the activation; the shuffled-activation arms control for that.",
    ],
        "The readers differ sharply. The token lenses carry information about the intermediate that changes with the "
        "activation, within a small set of candidate colours. The verbalizers mostly do not state it in its role; when they "
        "name the right colour, they usually give it the wrong role or wrap it in incorrect detail. Mentioning a colour, "
        "which 001 counted, is a much weaker thing than claiming it.",
        "Whether the content a reader names is content the model's answer depends on. The lens result is inside a "
        "five-candidate closed set, an easier task than naming freely. Verbalizers were scored on 10 tables. No causal label "
        "is involved.")
    html_ = f"""
<section aria-labelledby="h-002">
  <div class="eyebrow">Experiment 002 · dev · 10 snapshot tables (verbalizers), 48 tables (lenses) · judge {h(", ".join(A["judge_models_returned"]))} · design: docs/design_specs/002_claim_scoring.md</div>
  <h2 id="h-002">Experiment 002: what do the readers actually claim?</h2>
  <p>Experiment 001 counted colour <em>mentions</em>. Here a judge (Gemini 3.8 Flash, WorkspaceBench's judge) reads each
  of the {A["judge_status"].get("ok", 0)} verbalizer outputs with only the table's candidate colours and says which colour
  the text puts in each role: start, intermediate (after one rule application), answer. It may also say “none”,
  “cannot tell” or “conflicting”. The judge never sees the prompt. Token lenses are scored without a judge: does z rank
  first among the candidates, and score above the same lens on the mean activation? Judge spend:
  ${A["judge_spend_usd"]:.2f}.</p>
  {tk2}
  <h3>Is the judge trustworthy? Agreement with blind author codes (60 readouts)</h3>
  <p class="small muted">Gate: κ on the intermediate role ≥ 0.6. <b>{"Passed" if gate["passes"] else "FAILED: judge-based reader scores below are not results"}</b> (κ = {f2(gate["kappa_intermediate_labels"])}; judge errors {gate["judge_error_frac"]*100:.1f}%).</p>
  <div class="cols2">
    <div class="scroll"><table class="tbl">{vt}</table></div>
    <details class="more"><summary>Intermediate role: where judge and author differ</summary><div class="scroll"><table class="tbl"><tr><th>author | judge (categories)</th><th class="num">n</th></tr>{conf}</table></div></details>
  </div>
  <h3>What each reader puts in the intermediate role (member activation)</h3>
  <div class="chart" id="ex2-bars"></div>
  <p class="small muted">Green: z, the correct intermediate. Red: an incorrect claim: another candidate (solid) or an explicit value outside the candidates (pale red; every correct value is a candidate, so this is also wrong). Light: no claim. Amber: conflicting claims. The dot marks the 001-style mention rate for the same readouts.</p>
  <h3>Does the claim follow the activation? (within table)</h3>
  <div class="scroll"><table class="tbl">{ctbl}</table></div>
  <h3>Token lenses: closed-set hit rate on z and on the answer (all 48 dev tables, “:” cell)</h3>
  <div class="cols2"><div class="chart" id="ex2-lens-z"></div><div class="chart" id="ex2-lens-a"></div></div>
  <p class="small muted">Hit = the colour ranks first among the render's 5 candidates (start, z, answer, 2 off-path) <b>and</b> scores above the lens's output on the dev-mean activation (A01 rule). Prompt-only <b>task baseline</b> (stock model given the full prompt and asked; prompt solvability, not a ceiling on readers): z {f2(po["intermediate"]["top1_of_12"])} on all 48 tables ({f2(po["intermediate"]["snapshot_tables_top1_of_12"])} on the 10 snapshot tables), answer {f2(po["answer"]["top1_of_12"])}, start {f2(po["start"]["top1_of_12"])}.</p>
  <h3>Token lenses inside the candidate set: activation-sensitive z information, none for the start (48 tables; post hoc)</h3>
  <p class="small muted">Added after the hit rates above were seen; exploratory. “Tracks” = within the same table, the lens's score for the member's colour minus the partner's colour is higher on the member render than on the partner render (start-swap contrast; per table, averaged over partners; counts over 48 tables). Shuffled = the same lens output scored against another table's candidates.</p>
  <div class="scroll"><table class="tbl">{lt}</table></div>
  <h3>Do readers overlap? (exploratory; 10 tables; member activation)</h3>
  <p class="small muted">Mention = the colour word appears in any of the cell's samples; asserts = the judge puts z in the intermediate role in any sample; shared wrong = both readers assert the same incorrect intermediate in some sample. Oracle L40/L44 and NLA L42 read different layers; J_NP top-10 lists are a different claim budget.</p>
  <div class="scroll"><table class="tbl"><tr><th>Pair</th><th>Level</th><th class="num">both</th><th class="num">only first</th><th class="num">only second</th><th class="num">neither</th></tr>{ovr}</table></div>
  <h3>Qualitative: judged readouts</h3>
  <div class="seg" id="ex2-qsel"></div>
  <div id="ex2-q" class="panel"></div>
  <h3>Predictions written before the judge ran</h3>
  <div class="scroll"><table class="tbl"><tr><th>Prediction</th><th>Value</th><th>Outcome</th></tr>{pr}</table></div>
</section>
"""
    return html_, data



X3 = R.PROJECT / "results" / "003_labels"


def exp003():
    """HTML + data for experiment 003 (reader-free labels). Empty until the analysis has run."""
    f = X3 / "analysis" / "summary.json"
    if not f.exists():
        return "", None
    A = json.loads(f.read_text())
    S, gates = A["summary"], A["gates"]
    cells = [("emit_asst", 40), ("emit_asst", 42), ("emit_asst", 44), ("emit_think", 40), ("emit_think", 42), ("emit_think", 44),
             ("emit_colon", 40), ("emit_colon", 42), ("emit_colon", 44), ("emit_colon", 60), ("start_tok", 40)]
    nice = {"emit_asst": "after “assistant”", "emit_think": "after “</think>”", "emit_colon": "“:”", "start_tok": "start colour token"}
    chart = []
    for c, L in cells:
        row = {"label": f"{nice[c]} L{L}"}
        for k in ("z", "start", "answer"):
            x = S.get(f"{c}|L{L}|{k}") or {}
            row[k] = x.get("median")
            row[k + "_ci"] = x.get("boot95_tables")
            row[k + "_q95"] = x.get("null_q95")
        row["full"] = (S.get(f"{c}|L{L}|z") or {}).get("median_S_full")
        chart.append(row)
    lt = ('<tr><th>Cell</th><th>Content</th><th class="num">median S [95% CI]</th><th class="num">null q95</th><th class="num">used</th>'
          '<th class="num">used + necessary</th><th class="num">not needed</th><th class="num">unresolved</th><th class="num">damage</th><th class="num">1-D &gt; full</th></tr>')
    for c, L in cells:
        for k in ("z", "start", "answer"):
            x = S.get(f"{c}|L{L}|{k}")
            if not x:
                continue
            ci = x.get("boot95_tables") or [float("nan")] * 2
            un = x.get("frac_used_necessary")
            lt += (f'<tr><td>{h(nice[c])} L{L}</td><td>{k}</td><td class="num">{f2(x["median"])} [{f2(ci[0])}, {f2(ci[1])}]</td>'
                   f'<td class="num">{f2(x["null_q95"])}</td><td class="num"><b>{pct(x["frac_used"])}</b></td>'
                   f'<td class="num">{pct(un) if un is not None else "—"}</td><td class="num">{pct(x["frac_not_needed"])}</td>'
                   f'<td class="num">{pct(x["frac_unresolved"])}</td><td class="num">{pct(x["frac_damage"])}</td><td class="num">{pct(x["frac_one_d_gt_full"])}</td></tr>')
    g4 = gates["G4_label_reliability"]
    rt = ('<tr><th>Content · cell · layers</th><th class="num">n</th><th class="num">agreement</th><th class="num">κ (used)</th>'
          '<th class="num">used: fit 1 / fit 2</th><th class="num">r(S₁, S₂)</th></tr>' +
          "".join(f'<tr><td>{h(k.replace("|", " · "))}</td><td class="num">{v["n"]}</td><td class="num">{f2(v["agreement"]) if v["agreement"] is not None else "—"}</td>'
                  f'<td class="num">{f2(v["kappa_used"]) if v["kappa_used"] == v["kappa_used"] else "n/a"}</td>'
                  f'<td class="num">{pct(v["frac_used_r1"]) if v["frac_used_r1"] is not None else "—"} / {pct(v["frac_used_r2"]) if v["frac_used_r2"] is not None else "—"}</td>'
                  f'<td class="num">{f2(v["pearson_S"]) if v["pearson_S"] is not None else "—"}</td></tr>' for k, v in g4.items()))
    nv = "".join(f'<tr><td>{h(k)}</td><td class="num">{f2(v["median_S_naive"])}</td><td class="num">{f2((S.get(k.replace("|L", "|L") + "|z") or {}).get("median", float("nan")))}</td><td class="num">{f2(v["median_cos_with_decorrelated"])}</td></tr>'
                 for k, v in A["naive_vs_decorrelated"].items())
    q = "".join(f'<tr><td><code>{h(x["pair_id"].split("-")[-1])}</code></td><td class="num">{f2(x["S_z"])}</td><td class="num">{f2(x["S_start"]) if x["S_start"] is not None else "—"}</td>'
                f'<td class="num">{f2(x["S_answer"]) if x["S_answer"] is not None else "—"}</td><td class="num">{f2(x["S_full"]) if x["S_full"] is not None else "—"}</td>'
                f'<td class="small">{h(", ".join(f"{c} {v}" for c, v in x["colour_mass_clean_top4"]))}</td><td class="small">{h(", ".join(f"{c} {v}" for c, v in x["colour_mass_edit_top4"]))}</td><td>{h(x["state"])}</td></tr>'
                for x in A["qualitative"])
    pr = "".join(f'<tr><td>{h(k)}</td><td class="small">{h(json.dumps(v, default=float)[:160])}</td></tr>' for k, v in A["predictions"].items())
    G1 = gates["G1_fidelity"]
    sm = lambda key, f="median": S[key][f]
    z3 = [S[f"emit_colon|L{L}|z"] for L in (40, 42, 44)]
    g4z = gates["G4_label_reliability"]["z|emit_colon|42/44"]
    tk3 = takeaway([
        f"<b>The intermediate's own direction has a small, consistent effect</b> at “:” L40–44: median S {f2(min(x['median'] for x in z3))}–{f2(max(x['median'] for x in z3))}, positive in {pc(min(x['frac_S_positive'] for x in z3))}–{pc(max(x['frac_S_positive'] for x in z3))} of pairs. That is larger than the start ({f2(sm('emit_colon|L42|start'))}) and answer ({f2(sm('emit_colon|L42|answer'))}) directions at L42, but small against the whole cell ({f2(sm('emit_colon|L42|z', 'median_S_full'))}).",
        f"<b>It is “used” (S ≥ 0.10 and above random directions) in only {pc(min(x['frac_used'] for x in z3))}–{pc(max(x['frac_used'] for x in z3))} of pairs</b>, concentrated in a few tables.",
        f"<b>Late in the network the answer takes over:</b> at “:” L60 the answer direction alone moves {f2(sm('emit_colon|L60|answer'))} of the gap (whole cell {f2(sm('emit_colon|L60|answer', 'median_S_full'))}).",
        f"<b>After “&lt;/think&gt;”</b> z is decodable but the whole cell barely moves the answer (whole cell {f2(sm('emit_think|L42|z', 'median_S_full'))} at L42): readable, low patch leverage.",
        f"<b>The labels are not reproducible.</b> Two disjoint fitting sets give labels that agree at κ {f2(g4z['kappa_used'])} (r = {f2(g4z['pearson_S'])}) for z at “:” L42–44, below the 0.7 gate.",
    ],
        "Along a single fitted direction, the intermediate has demonstrated influence on this answer, but it accounts for "
        "a small part of what the whole cell does, and which pairs count as “used” depends on the fitting data. These "
        "labels are therefore descriptive and are not used to score readers. The gap between one direction and the whole "
        "cell is the question 004 takes up.",
        "That the intermediate is unimportant here: one direction is a narrow probe of a 5,120-dimensional state. The "
        "necessity clamps beat random clamps, but those were not size-matched (corrected after the audit), so they do not "
        "show specificity.")
    html_ = f"""
<section aria-labelledby="h-003">
  <div class="eyebrow">Experiment 003 · dev · {A["n_pairs"]} gated brew pairs ({A["n_tables"]} tables) · SEC regime · design: docs/design_specs/003_labels.md</div>
  <h2 id="h-003">Experiment 003: is the intermediate itself used? Reader-free labels</h2>
  <p>A whole-cell patch moves start, intermediate and answer together. Here each content gets its own direction:
  a least-squares fit of the activations on start, intermediate and answer together (so each direction has the other
  two regressed out), fitted on other tables only. The partner's coordinate along that one direction is swapped into the
  member's run, and the share S of the answer gap it moves is measured against norm-matched random directions. No reader
  is involved anywhere.</p>
  {tk3}
  <p class="small muted">Gates: fidelity {G1["n_fail"]} failures of {G1["n_edit_rows"]} edits (ρ {G1["rho_range"][0]:.5f}–{G1["rho_range"][1]:.5f}); engine checks {"pass" if gates["G2"] else "FAIL"};
  direction validity (held-out AUC, “:” L42) {f2(gates["G3"]["median_auc_colon_L42"])}; label reliability for z at “:” L42–44 {"passes" if gates["G4_passes_z_colon_L42_44"] else "does NOT pass: labels are descriptive only"}.</p>
  <h3>Which content carries the leverage? Median share S per content direction, against the whole cell</h3>
  <div class="chart" id="ex3-bars"></div>
  <h3>Labels</h3>
  <div class="scroll"><table class="tbl">{lt}</table></div>
  <p class="small muted">used = S ≥ τ (0.10) and above the norm-matched random-direction null (BH per pair). used + necessary = also, clamping that one direction to its fitting-set mean in both runs shrinks the gap more than 95% of random clamps. not needed = the whole cell carries |S| &lt; 0.05 while z is decodable there. Damage edits are unresolved.</p>
  <h3>Label reliability: two disjoint fitting subsets (gate G4)</h3>
  <div class="scroll"><table class="tbl">{rt}</table></div>
  <details class="more"><summary>Decorrelated vs naive (centroid-difference) z direction</summary><div class="scroll"><table class="tbl"><tr><th>Cell</th><th class="num">median S, naive</th><th class="num">median S, decorrelated</th><th class="num">cos(naive, decorrelated)</th></tr>{nv}</table></div></details>
  <h3>Qualitative: pairs at “:” L44 (largest, median and smallest z-swap S)</h3>
  <div class="scroll"><table class="tbl"><tr><th>Pair</th><th class="num">S_z</th><th class="num">S_start</th><th class="num">S_answer</th><th class="num">S_full</th><th>colour mass, clean (top 4)</th><th>after the z swap</th><th>label</th></tr>{q}</table></div>
  <details class="more"><summary>Predictions (L1–L8) and values</summary><div class="scroll"><table class="tbl">{pr}</table></div></details>
</section>
"""
    return html_, {"chart": chart}


def sg(x, d=3):
    """Signed number with a true minus sign, rounded half up."""
    return ("−" if float(x) < 0 else "+") + f2(abs(float(x)), d)


def pc(x):
    """Percent, rounded half up like f2 (0.145 -> 15%)."""
    return f2(100 * float(x), 0) + "%"


def takeaway(found, means, not_shown, found_title="What we found"):
    """Three panels after an experiment's intro: findings (list), what they mean, what they do not show (HTML strings)."""
    items = "".join(f"<li>{x}</li>" for x in found)
    return (f'<div class="takeaway"><div><span class="eyebrow">{found_title}</span><ul class="plain">{items}</ul></div>'
            f'<div><span class="eyebrow">What it means</span><p>{means}</p></div>'
            f'<div><span class="eyebrow">What it does not show</span><p>{not_shown}</p></div></div>')


def story():
    """The research arc 001 → 004 in plain language; every number from committed JSON."""
    s1 = json.loads((X / "analysis" / "summary.json").read_text())
    s2 = json.loads((X2 / "analysis" / "scores.json").read_text())
    s3 = json.loads((X3 / "analysis" / "summary.json").read_text())
    f4 = X4 / "analysis" / "summary.json"
    s4 = json.loads(f4.read_text()) if f4.exists() else None
    bp, dec, sub = s1["brew"]["profile"], s1["decodability"], s1["substitution"]
    S42, S44 = bp["emit_colon#0|L42"]["median"], bp["emit_colon#0|L44"]["median"]
    z42, a42 = dec["z"]["emit_colon|L42"]["real"], dec["answer"]["emit_colon|L42"]["real"]
    rt = s2["rates"]
    nla = [rt[f"nla|member|cell{c}|L42|real"]["int_correct"] for c in (0, 1, 2)]
    orc = [rt[f"oracle|member|cell2|L{L}|real"]["int_correct"] for L in (40, 44)]
    trk = s2["lens_post_hoc"]["start_swap_tracking_logscore"]
    lens = [trk[f"{r}|emit_colon|L{L}"]["pos"] for r in ("J_NP_cosine", "J_CB_standard", "R_CB_standard") for L in (40, 42, 44)]
    n_judged = s2["judge_status"].get("ok", 0)
    z3 = [s3["summary"][f"emit_colon|L{L}|z"] for L in (40, 42, 44)]
    g4 = s3["gates"]["G4_label_reliability"]["z|emit_colon|42/44"]
    steps = [
        ("001", "Leverage profile", "Where does the answer depend on a single cell, and what can be decoded there?", "h-001",
         "Readers are read at particular cells (a token at a layer). Before scoring what a reader says about a cell, we need "
         "to know, without any reader, whether the model's answer depends on that cell at all.",
         f"For {s1['brew']['n_pairs'] if 'n_pairs' in s1['brew'] else 82} brew pairs (two prompts that differ only in the start colour, so the "
         "intermediate z and the answer differ too), we copied the partner's whole activation into the member's run at one "
         "token and one layer, and measured the share S of the answer gap that moved. We did this for every cell of the "
         "prompt at L40, 42, 44, 56 and 60. Alongside: linear probes, a swap of an unused table colour, and a descriptive "
         "look at what each reader says on 10 tables.",
         f"The final “:” cell carries about half the gap at L42–44 (median S {f2(S42)} and {f2(S44)}), toward the partner in "
         f"every pair. z is decodable there (probe balanced accuracy {f2(z42)} at L42, chance 0.08); the answer only weakly "
         f"until L56 ({f2(a42)} at L42). Swapping an unused colour leaves the answer alone in {frac(sub['behaviour_frac_negligible'], sub['n_items'])} items.",
         "The “:” cell at L42 (the cell the NLA reads, next to the Oracle's L40 and L44) is a site where the pair difference "
         "matters and z is decodable. It is the right place to ask what readers report.",
         "Which content carries the effect. A whole-cell patch moves start, z, answer and everything downstream at once. "
         "Experiments 003 and 004 take this up."),
        ("002", "Reader claim scoring", "What do the readers actually claim, role by role?", "h-002",
         "001 only counted colour words. A reader that says “red” might mean the start, the intermediate or the answer, or "
         "might be guessing from the prompt.",
         f"A judge (Gemini 3.8 Flash, WorkspaceBench's own judge, checked against blind author codes at κ "
         f"{f2(s2['validation_gate']['kappa_intermediate_labels'])}) read {n_judged:,} verbalizer outputs and said which "
         "colour each puts in each role. The token lenses were scored without a judge, inside the table's candidate colours. "
         "Shuffled-activation and prompt-only baselines throughout.",
         f"The verbalizers rarely put the right intermediate in its role: the NLA in {pc(min(nla))}–{pc(max(nla))} of samples, "
         f"the Oracle in {pc(min(orc))}–{pc(max(orc))}. The token lenses, read inside the candidate set, follow the intermediate "
         f"when the start is swapped in {min(lens)}–{max(lens)} of 48 tables at “:” L40–44 (J_NP cosine, J_CB, R_CB).",
         "The lenses carry activation-sensitive information about z at this cell. The verbalizers mostly do not state it in "
         "a role-correct way, and often surround a correct colour with incorrect detail.",
         "Whether the content a reader names is content the answer depends on. That needs a causal label for z itself, "
         "which is what 003 set out to build."),
        ("003", "Reader-free labels", "Does the answer depend on the intermediate itself, apart from start and answer?", "h-003",
         "To score a reader's z claim against what the model uses, we need a label for z, not for the whole cell.",
         "We fitted one direction per content (start, z, answer) by least squares on other tables' activations, so each "
         "direction has the other two regressed out. Then we swapped the partner's coordinate along the z direction only, "
         "and clamped it to its mean to test necessity, against random-direction controls. 82 pairs, 37k edits, every write "
         "in float32 with fidelity checks.",
         f"The z direction has a small, consistent effect at “:” L40–44: median S {f2(min(x['median'] for x in z3))}–"
         f"{f2(max(x['median'] for x in z3))}, positive in {pc(min(x['frac_S_positive'] for x in z3))}–"
         f"{pc(max(x['frac_S_positive'] for x in z3))} of pairs, against a whole-cell {f2(S42)} at L42. It clears the 0.10 "
         f"“used” line in only {pc(min(x['frac_used'] for x in z3))}–{pc(max(x['frac_used'] for x in z3))} of pairs. Labels "
         f"from two disjoint fitting sets agree poorly (κ {f2(g4['kappa_used'])}, r {f2(g4['pearson_S'])}), so the "
         "reliability gate failed.",
         "Along the one fitted direction, z has demonstrated influence on the answer, but most of the whole-cell effect lies "
         "elsewhere. The labels are too unstable to score readers with, so they stay descriptive.",
         "That z is unimportant. One direction is a narrow probe of a 5,120-dimensional state. 004 widens it."),
    ]
    if s4:
        A4, rec = s4["arms"], lambda k: s4["arms"][k]["recovery"]["value"]
        pr = s4["paired"]
        steps.append(
            ("004", "Span assay", "Does a wider edit recover the cell, and is that more than any subspace of the same size would do?", "h-004",
             "003's single direction left most of the cell's effect unexplained. If the influence is spread over the content "
             "subspace, swapping the whole subspace should recover it. If a generic subspace of the same size does as well, "
             "the content fit adds nothing.",
             f"At “:” L42, {s4['n_pairs']} pairs (one per table), we swapped the partner's coordinates in the z-associated "
             "span (11 dimensions), the joint start/z/answer span (33 dimensions) and their complements. We compared them "
             "with PCA subspaces of the activations at the same rank (also rescaled to the same edit size), random subspaces, "
             "and, for necessity, paired random rotations. Every edit records all 12 colours' probabilities.",
             f"Share of the whole-cell effect recovered: 1-D z {f2(rec('colon|both|z1d'))}, z span {f2(rec('colon|both|Z'))}, "
             f"joint span {f2(rec('colon|both|J'))}. PCA of the same rank recovers {f2(rec('colon|both|pca_kZ'))} and "
             f"{f2(rec('colon|both|pca_kJ'))}; at the same edit size the fitted spans are still slightly behind "
             f"({sg(pr['colon|both|Z-pca_kZ_dose']['dS_all']['value'])} and {sg(pr['colon|both|J-pca_kJ_dose']['dS_all']['value'])} "
             f"of the gap). The z-span swap mostly raises the recipient's own intermediate colour, not the donor's answer "
             f"(aimed at the donor's answer in {A4['colon|both|Z']['n_qualified']} of {s4['n_pairs']} pairs).",
             "How much a subspace recovers tracks how much of the pair difference it contains, whether or not it was fitted "
             "to the content. The fitted spans sit in directions that matter (random ones do nothing) but show no "
             "content-specific advantage. 004 gives no clean causal handle on the intermediate at this cell.",
             "That the intermediate is unused at this cell, or that no method could isolate it (DAS, trained for the purpose, "
             "was not tried). Dev split, one cell, one layer."))
    lis = ""
    for num, name, q, anchor, why, did, found, means, not_ in steps:
        lis += (f'<li class="step"><div class="step-head"><span class="eyebrow">Experiment {num} · {h(name)}</span>'
                f'<h3>{q}</h3></div><dl class="qa"><dt>Why</dt><dd>{why}</dd><dt>What we did</dt><dd>{did}</dd>'
                f'<dt>What we found</dt><dd>{found}</dd><dt>What it means</dt><dd>{means}</dd>'
                f'<dt>Not shown</dt><dd>{not_}</dd></dl><a class="small" href="#{anchor}">Full results for {num} below</a></li>')
    return f"""
<section aria-labelledby="h-story">
  <h2 id="h-story">What we did, in order, and what it showed</h2>
  <p>Four experiments so far, all on Qwen3.6-27B and the dev split of WorkspaceBench. Each was designed after reading
  the previous one's results, so the order matters: each “Why” is a gap the step before left open. Numbers are dev-split
  medians; the sections below give intervals, controls and examples.</p>
  <ol class="story">{lis}</ol>
  <div class="bottomline"><span class="eyebrow">Where this leaves the project</span>
  <p>The model's answer depends heavily on the “:” cell (001), and the readers carry real, activation-sensitive task
  information there (002). But there is still no reader-free causal reference specific to the intermediate (003, 004),
  so no reader has been scored against a causal label. The recommended next step (report 004 §14, awaiting your
  confirmation) is to compare the readers' native configurations directly: what task information each carries, where
  each errs, and how they overlap. A supervised method (DAS) at matched rank and edit size is the one bounded causal
  follow-up worth considering, and only if its answer would change that plan.</p></div>
</section>
"""


X4 = R.PROJECT / "results" / "004_span_assay"
ARMS4 = [("colon", "z1d", "1-D z direction (003)"), ("colon", "Z", "z-associated span (11)"), ("colon", "pca_kZ", "PCA, same rank"),
         ("colon", "pca_kZ_dose", "PCA, same rank and edit norm"), ("colon", "J", "joint start/z/answer span (33)"),
         ("colon", "pca_kJ", "PCA, rank 33"), ("colon", "pca_kJ_dose", "PCA, rank 33, matched norm"),
         ("colon", "S", "start-associated span"), ("colon", "A", "answer-associated span"), ("colon", "Zc", "complement of the z span"),
         ("colon", "Jc", "complement of the joint span"),
         ("start_ref", "s1d", "start 1-D (reference)"), ("start_ref", "Sspan", "start span (reference)"),
         ("start_ref", "pca_kS", "PCA, same rank (reference)"), ("start_ref", "Sc", "complement of the start span")]


def exp004():
    """HTML + data for experiment 004 (span assay). Empty until the 33-pair analysis has run."""
    f = X4 / "analysis" / "summary.json"
    if not f.exists():
        return "", None
    A = json.loads(f.read_text())
    PP = json.loads((X4 / "analysis" / "per_pair.json").read_text())
    Q = json.loads((X4 / "analysis" / "examples.json").read_text())
    rs = A["recovery_subset"]
    chart = []
    for site, arm, label in ARMS4:
        row = {"label": label, "site": site}
        for fit in ("both", "r1", "r2"):
            rec = (A["arms"].get(f"{site}|{fit}|{arm}") or {}).get("recovery") or {}
            row[fit] = rec.get("value")
            if fit == "both":
                row["ci"] = rec.get("boot95_tables")
        chart.append(row)
    scatter = {k: [{"id": pid.split("-")[-1], "x": p["swap"].get(f"colon|both|{a}", {}).get("S"), "y": p["swap"].get(f"colon|both|{b}", {}).get("S"),
                    "full": p["S_full"]["colon"]} for pid, p in PP.items()]
               for k, (a, b) in {"Z_vs_pca": ("Z", "pca_kZ"), "J_vs_pca": ("J", "pca_kJ")}.items()}
    priors = {"S1": "45%", "S2": "40%", "S3": "55%", "S4": "50%", "S5": "35%", "S6": "45%"}

    def ci(x, d=2):
        if not x or x.get("value") is None:
            return "—"
        b = x.get("boot95_tables") or [None, None]
        return f2(x["value"], d) + (f" [{f2(b[0], d)}, {f2(b[1], d)}]" if b[0] is not None else "")
    def pval(k, v):
        x = v["value"]
        if k.startswith("S5"):
            mr = x.get("median_recovery") or {}
            return (f'recovery r1 {f2(mr.get("r1"))}, r2 {f2(mr.get("r2"))} (primary {f2(mr.get("both"))}); '
                    f'median |S_r1 − S_r2| {f2(x.get("median_abs_dS_r1_r2"), 3)}')
        return "—" if x is None else f2(x, 3)
    pr = "".join(f'<tr><td>{h(k.split("_", 1)[0])}</td><td>{h(k.split("_", 1)[1].replace("_", " "))}</td><td class="num">{priors[k[:2]]}</td>'
                 f'<td class="small">{h(pval(k, v))}</td>'
                 f'<td><b>{"holds" if v["holds"] else "does not hold"}</b>{" (low-effect agreement)" if v.get("low_effect_agreement") else ""}</td></tr>'
                 for k, v in A["predictions"].items())
    comp_rows = ""
    for site, arm, label in ARMS4[:7] + [("colon", "full", "whole cell")] + [ARMS4[12], ("start_ref", "full", "whole cell (reference)")]:
        key = f"{site}|{'None' if arm == 'full' else 'both'}|{arm}"
        a = A["arms"].get(key)
        if not a:
            continue
        g = [PP[p]["swap"][key]["gains"] for p in PP if key in PP[p]["swap"]]
        AP, ZP, RZ = "a'", "z'", "of_other: recipient z"
        med = lambda r: f2(float(np.median([x[r] for x in g if x.get(r) is not None])), 3) if any(x.get(r) is not None for x in g) else "—"
        comp_rows += (f'<tr><td>{h(label)}</td><td class="num">{ci(a["S_all"])}</td><td class="num">{f2(a["F_target_median"]) if a["F_target_median"] is not None else "—"} ({a["n_F_defined"]})</td>'
                      f'<td class="num">{med(AP)}</td><td class="num">{med(ZP)}</td><td class="num">{med(RZ)}</td>'
                      f'<td class="num">{a["n_qualified"]}</td><td class="num">{a["n_flagged"]}</td></tr>')
    ctrl = "".join(f'<tr><td>{h(k.replace("|", " · "))}</td><td class="num">{v["n"]}</td><td class="num">{pct(v["frac_real_above_q95"])}</td>'
                   f'<td class="num">{pct(v["frac_real_below_q05"])}</td><td class="num">{f2(v["median_rank_stat"], 3)}</td><td class="num">{f2(v["ctrl_median_of_medians"], 3)}</td></tr>'
                   for k, v in A["controls"].items())
    nec = ""
    for k, v in A["necessity"].items():
        site, fit, arm = k.split("|")
        rot = A["necessity_controls"].get(f"{site}|{arm}|rot") if fit == "both" else None
        rcl = A["necessity_controls"].get(f"{site}|{arm}|rclamp") if fit == "both" else None
        nec += (f'<tr><td>{h(site)} · {h(arm)}</td><td>{h(fit)}</td><td class="num">{ci(v["N"])}</td>'
                f'<td class="num">{pct(rot["frac_real_above_q95"]) if rot else ""}</td><td class="num">{pct(rcl["frac_real_above_q95"]) if rcl else ""}</td>'
                f'<td class="num">{v["n_flag_member"]} / {v["n_flag_partner"]}</td></tr>')
    inter = "".join(f'<tr><td>{h(k.replace("|", " · "))}</td><td class="num">{ci(v["I_signed"])}</td><td class="num">{ci(v["abs_I"])}</td>'
                    f'<td class="num">{ci(v["frac_abs_I_ge"])}</td></tr>' for k, v in A["interaction"].items())
    rep = "".join(f'<tr><td>{h(k.replace("|", " · "))}</td><td class="num">{v["n"]}</td><td class="num">{f2(v["pearson_r1_r2"]) if v["pearson_r1_r2"] is not None else "—"}</td>'
                  f'<td class="num">{f2(v["median_abs_diff"], 3)}</td><td class="num">{f2(v["signed_bias_median_r1_minus_r2"], 3)}</td>'
                  f'<td class="num">{" / ".join(f2(x) if x is not None else "—" for x in (v["median_recovery"] or {}).values()) if v["median_recovery"] else "—"}</td>'
                  f'<td>{"yes" if v.get("high_recovery_reproducible") else ("low-effect agreement" if v.get("low_effect_agreement") else ("—" if v["median_recovery"] is None else "no"))}</td></tr>'
                  for k, v in A["reproducibility"].items() if k.split("|")[2] in ("z1d", "Z", "J", "Sspan", "s1d", "pca_kZ", "pca_kJ"))
    t4 = lambda lst: ", ".join(f"{c} {v:.2f}" for c, v in lst)
    ql = ""
    merged = {}
    for e in Q:
        if e["pair_id"] in merged:
            merged[e["pair_id"]]["why"] += "; " + e["why"]
        else:
            merged[e["pair_id"]] = dict(e)
    for e in merged.values():
        arms = e["arms"]
        cells = "".join(f'<td class="num">{f2(arms[a]["S"])}</td>' if a in arms else "<td>—</td>" for a in ("z1d", "Z", "pca_kZ", "J", "full"))
        ql += (f'<tr><td><code>{h(e["pair_id"].split("-")[-1])}</code><br><span class="small muted">{h(e["why"].replace("_", " "))}</span></td>'
               f'<td class="small">{h(e["start"])}→{h(e["z"])}→<b>{h(e["a"])}</b><br>donor {h(e["start2"])}→{h(e["z2"])}→<b>{h(e["a2"])}</b></td>{cells}'
               f'<td class="small">{h(t4(e["clean_top4"]))}</td><td class="small">{h(t4(arms["Z"]["top4"])) if "Z" in arms else "—"}</td>'
               f'<td class="small">{h(t4(arms["J"]["top4"])) if "J" in arms else "—"}</td></tr>')
    # ---- extra numbers for the explanation (all from committed JSON) ----
    pre = json.loads((X4 / "preflight.json").read_text())
    man = set(json.loads((X4 / "manifest_pairs.json").read_text())["pairs"])
    share = {a: float(np.median([p["colon|both"][a]["proj_share_sq"] for p in pre["pairs"] if p["pair_id"] in man]))
             for a in ("z1d", "Z", "J", "pca_kZ", "pca_kJ")}
    rv = lambda k: A["arms"][k]["recovery"]["value"]
    sv = lambda k: A["arms"][k]["S_all"]["value"]
    dS = lambda k: A["paired"][k]["dS_all"]["value"]
    npairs = A["n_pairs"]
    cnt = lambda a, b: sum(1 for p in PP.values() if p["swap"].get(a, {}).get("S") is not None and p["swap"].get(b, {}).get("S") is not None
                           and p["swap"][a]["S"] > p["swap"][b]["S"])
    nZ = A["necessity"]["colon|both|Z"]["N"]["value"]
    rotZ = A["necessity_controls"]["colon|Z|rot"]["frac_real_above_q95"]
    IZ = A["interaction"]["colon|both|Z+Zc"]
    repJ, repZ = A["reproducibility"]["S|colon|J"], A["reproducibility"]["S|colon|Z"]
    big_rec_z = sum(1 for p in PP.values() if p["swap"]["colon|both|Z"]["gains"].get("of_other: recipient z") is not None
                    and max(((r, v) for r, v in p["swap"]["colon|both|Z"]["gains"].items() if v is not None and r != "other"),
                            key=lambda t: t[1])[0] == "of_other: recipient z")
    # composition chart: mean positive colour-mass gains by role (means are additive; medians are in the table)
    comp_chart = []
    for key, label in (("colon|both|z1d", "1-D z direction"), ("colon|both|Z", "z-associated span"),
                       ("colon|both|pca_kZ_dose", "PCA-11, same edit size"), ("colon|both|pca_kZ", "PCA-11"),
                       ("colon|both|J", "joint span"), ("colon|both|pca_kJ_dose", "PCA-33, same edit size"),
                       ("colon|both|pca_kJ", "PCA-33"), ("colon|None|full", "whole cell")):
        g = [PP[p]["swap"][key]["gains"] for p in PP if key in PP[p]["swap"]]
        m = lambda r: float(np.mean([x.get(r) or 0.0 for x in g]))
        a2, zz, rz, rs_ = m("a'"), m("z'"), m("of_other: recipient z"), m("of_other: recipient start")
        st, oth = m("start'"), m("other")
        comp_chart.append({"label": label, "a2": a2, "z2": zz, "start2": st, "rz": rz, "rest": max(0.0, oth - rz - rs_) + rs_,
                           "qualified": A["arms"][key]["n_qualified"], "n": len(g)})
    cshare = {r["label"]: {k: r[k] / max(1e-12, r["a2"] + r["z2"] + r["start2"] + r["rz"] + r["rest"]) for k in ("a2", "rz")} for r in comp_chart}
    comp_line = (f"For the z-span swap, the donor's answer gets {pc(cshare['z-associated span']['a2'])} of the gained probability on average "
                 f"and the recipient's own intermediate {pc(cshare['z-associated span']['rz'])}. For the whole cell the donor's answer gets "
                 f"{pc(cshare['whole cell']['a2'])}. PCA and the joint span sit in between, roughly in order of how much they move.")
    edits = [
        ("1-D z direction", "the partner's coordinate along 003's single fitted z direction", "the baseline 004 tries to improve on"),
        ("z-associated span (11)", "the partner's coordinates in the space spanned by all fitted z-colour contrasts", "does the intermediate's influence spread over a subspace?"),
        ("joint span (33)", "the same for start, z and answer contrasts together", "does the whole content subspace carry the cell?"),
        ("start / answer span", "start-only or answer-only contrasts at the same cell", "is any single role enough on its own?"),
        ("complement", "everything except the span", "is the effect inside the span or outside it?"),
        ("PCA, same rank", "the top principal directions of the activations themselves (no content labels)", "the generic comparator: is the content fit better than any data-aligned subspace?"),
        ("PCA, same edit size", "the PCA edit rescaled to the content edit's size", "rules out “PCA wins because its edit is bigger”"),
        ("random subspace / direction", "a random subspace of the same rank, or a random direction of the same edit size", "do arbitrary edits of this size move the answer?"),
        ("whole cell", "the partner's entire activation", "the effect everything is measured against"),
        ("clamp to the mean", "the span's coordinates set to their average, in both prompts", "necessity: does removing the span's variation shrink the gap?"),
        ("paired random rotation", "the same clamp edit, rotated into a random subspace (same sizes and angles)", "is it the orientation that matters, or just the edit size?"),
    ]
    pairs_cmp = [("colon|both|Z", "colon|both|pca_kZ"), ("colon|both|Z", "colon|both|pca_kZ_dose"), ("colon|both|J", "colon|both|pca_kJ"),
                 ("colon|both|J", "colon|both|pca_kJ_dose"), ("start_ref|both|Sspan", "start_ref|both|pca_kS")]
    pca_right = all(rv(b) > rv(a) for a, b in pairs_cmp)
    pca_line = "Here PCA is to the right of the matching content span in every pair of rows." if pca_right else "PCA and the content spans trade places across rows."
    etab = "".join(f"<tr><td><b>{h(a)}</b></td><td>{h(b)}</td><td>{h(c)}</td></tr>" for a, b, c in edits)
    found = [
        f"<b>The whole cell moves a median {f2(sv('colon|None|full'))} of the answer gap</b> at “:” L42 (all {npairs} pairs above 0.10).",
        f"<b>Wider spans recover more.</b> Share of the whole-cell effect: 1-D z {f2(rv('colon|both|z1d'))}, z span {f2(rv('colon|both|Z'))}, joint span {f2(rv('colon|both|J'))}. The z span beats the 1-D direction in {cnt('colon|both|Z', 'colon|both|z1d')} of {npairs} pairs.",
        f"<b>PCA of the same size does as well or better.</b> PCA recovers {f2(rv('colon|both|pca_kZ'))} at 11 dimensions and {f2(rv('colon|both|pca_kJ'))} at 33. Even rescaled to the same edit size it stays slightly ahead: z span − PCA {sg(dS('colon|both|Z-pca_kZ_dose'))}, joint − PCA {sg(dS('colon|both|J-pca_kJ_dose'))} of the gap (PCA larger in {cnt('colon|both|pca_kZ_dose', 'colon|both|Z')} and {cnt('colon|both|pca_kJ_dose', 'colon|both|J')} of {npairs} pairs).",
        f"<b>Recovery follows how much of the pair difference a subspace holds:</b> z span {pc(share['Z'])} of the difference's squared size, PCA-11 {pc(share['pca_kZ'])}, joint {pc(share['J'])}, PCA-33 {pc(share['pca_kJ'])}.",
        f"<b>The z-span swap mostly raises the wrong colour.</b> Its largest single gain is the recipient's own intermediate in {big_rec_z} of {npairs} pairs; it is aimed mainly at the donor's answer in only {A['arms']['colon|both|Z']['n_qualified']}.",
        f"<b>The spans do sit in directions that matter.</b> Random edits of the same rank or size do nothing (median S ≤ 0.005). Clamping the z span removes {f2(nZ)} of the gap, more than paired random rotations of the same edit in {round(rotZ * npairs)} of {npairs} pairs.",
        f"<b>The parts do not add up.</b> z span plus its complement falls {f2(-IZ['I_signed']['value'])} short of the whole cell; the shortfall is at least 0.15 in {pc(IZ['frac_abs_I_ge']['value'])} of pairs.",
        f"<b>Stability:</b> two half-sized fits agree on the z span only at a low level (recovery {f2(repZ['median_recovery']['r1'])} and {f2(repZ['median_recovery']['r2'])}). The joint span reaches {f2(repJ['median_recovery']['r1'])} and {f2(repJ['median_recovery']['r2'])}, but their per-pair disagreement ({f2(repJ['median_abs_diff'], 3)}) misses the pre-set 0.05 tolerance.",
    ]
    tk = takeaway(found,
                  "Swapping a larger subspace moves the answer more, in proportion to how much of the pair difference that "
                  "subspace contains, and a subspace chosen without any content labels (PCA) does at least as well. So the "
                  "fitted spans are not picking out the intermediate in a special way, and most of what they move does not go "
                  "to the donor's answer. By the design's own reading this is a descriptive geometry result: 004 does not give "
                  "a content-specific causal reference for the intermediate at this cell, and no reader is scored against it.",
                  "That the intermediate is unused at “:” L42, or that no method could isolate it. A supervised method trained "
                  "for this (DAS) was not tried. Scope: dev split, one pair per table, one cell and layer, one answer endpoint.")
    html_ = f"""
<section aria-labelledby="h-004">
  <div class="eyebrow">Experiment 004 · dev · {npairs} pairs, one per table · SEC regime · “:” L42 and the start token L40 · design: docs/design_specs/004_span_assay.md (v2 + §13) · report: docs/reports/004_span_assay.md</div>
  <h2 id="h-004">Experiment 004: does a fitted span recover what the whole cell does?</h2>
  <p><b>Why.</b> In 001 the whole “:” cell at L42 carried about half the answer gap. In 003 the partner's coordinate
  along one fitted z direction carried almost none of it. Either the intermediate's influence is spread over more than
  one direction, or the fitted directions miss it. 004 tests the first possibility, and asks whether any gain is
  specific to the content or would come from any subspace of the same size.</p>
  <p><b>What we did.</b> One pair per table ({npairs} pairs). At “:” L42 the partner's coordinates in a subspace are swapped
  into the member's run, and we measure the share S of the answer gap that moves. <b>Recovery</b> is that share divided
  by the whole-cell share for the same pair (1 = the edit does everything the whole cell does). Every edit is fitted on
  other tables only, written in float32 with a fidelity check, and scored on all 12 colours. The start-colour token at
  L40 is a reference site, and two half-sized fits (r1, r2) test stability. No reader is involved.</p>
  <h3>The edits, and why each is in the design</h3>
  <div class="scroll"><table class="tbl"><tr><th>Edit</th><th>What it changes in the member's activation</th><th>Question it answers</th></tr>{etab}</table></div>
  <p class="small muted">Gates: the preflight passes (basis identities hold to float32 precision; zero writes are no-ops). The 2-pair smoke passes (self-swap bitwise; whole-cell share equals 001 exactly; clean margins equal 003). Fidelity is checked on every write of the run (rho/kappa at ≥ 1e-6 ‖h‖, read-back bound below). {len(A["quarantined_rows"])} quarantined rows. Recovery subset: whole-cell share ≥ 0.10 (“:” {rs["colon"]["n"]} of {npairs}, start token {rs["start_ref"]["n"]}).</p>
  {tk}
  <h3>Recovery of the whole-cell effect</h3>
  <div class="chart" id="ex4-rec"></div>
  <p class="small muted"><b>How to read it.</b> Each row is one kind of edit; the filled dot is the median recovery over pairs with the primary fit, with its 95% table-bootstrap interval. ○ and □ are the two half-sized fits. The dashed line at 1 is the whole cell. Read the rows in pairs: each content span sits next to the PCA subspace of the same size, so the gap between them is the content fit's advantage. {pca_line} Complements (everything except the span) show what lies outside it.</p>
  <h3>Per pair: fitted span against PCA of the same rank</h3>
  <div class="arms"><div class="arm"><div class="chart" id="ex4-sc1"></div></div><div class="arm"><div class="chart" id="ex4-sc2"></div></div></div>
  <p class="small muted"><b>How to read it.</b> One dot per pair; x is the share S moved by the fitted span, y by PCA of the same rank, at “:” L42. Dots above the diagonal are pairs where PCA moves more. These are the unscaled PCA edits, which are larger; the same-edit-size comparison is in the findings above.</p>
  <h3>Where the moved probability goes</h3>
  <div class="chart" id="ex4-comp"></div>
  <p class="small muted"><b>How to read it.</b> Each bar is the mean total probability that colours <em>gain</em> after the edit, split by whose colour gains it: the donor's answer (the target), the donor's intermediate, the donor's start, the recipient's own intermediate, and other colours. A target-directed edit is mostly the first segment. {comp_line} Means are shown so the segments add up; medians are in the table below.</p>
  <h3>Predictions (priors written before any 004 run)</h3>
  <div class="scroll"><table class="tbl"><tr><th>#</th><th>Prediction</th><th class="num">prior</th><th>value</th><th>outcome</th></tr>{pr}</table></div>
  <h3>Where the added probability goes: medians per edit (primary fit, all pairs)</h3>
  <div class="scroll"><table class="tbl"><tr><th>Edit</th><th class="num">median S [95% CI]</th><th class="num">median F_target (n defined)</th><th class="num">gain to donor answer</th><th class="num">to donor z</th><th class="num">to recipient z</th><th class="num">target-directed</th><th class="num">flagged</th></tr>{comp_rows}</table></div>
  <p class="small muted">F_target = the donor answer's share of all positive colour-mass gains (defined when they total ≥ 0.01). Target-directed = F_target &gt; 0.5, log P(a′) up beyond the numerical floor, and no warning flag. Flags: mass outside the accepted spellings up &gt; 0.05, or both endpoint log-probabilities down &gt; 0.05 nats. Flagged rows stay in every median.</p>
  <h3>Qualitative: fixed-seed random pairs beside the median, extreme and flagged cases</h3>
  <p class="small muted">Each row is one pair: the member's path (start → intermediate → <b>answer</b>) and the donor's, the share each edit moves, and the top four colours before and after. Look for the recipient's own intermediate rising after the z-span swap (for example 0012, white; 0041, black).</p>
  <div class="scroll"><table class="tbl"><tr><th>Pair</th><th>recipient / donor path</th><th class="num">S 1-D</th><th class="num">S z span</th><th class="num">S PCA</th><th class="num">S joint</th><th class="num">S whole</th><th>colour mass, clean (top 4)</th><th>after the z-span swap</th><th>after the joint-span swap</th></tr>{ql}</table></div>
  <details class="more"><summary>Random controls (primary fit; descriptive, no per-item significance)</summary><div class="scroll"><table class="tbl"><tr><th>Arm · control</th><th class="num">n</th><th class="num">real above control q95</th><th class="num">real below control q05</th><th class="num">median rank stat (min 1/17)</th><th class="num">control median S</th></tr>{ctrl}</table></div></details>
  <details class="more"><summary>Necessity: clamping the span to its fitting mean (N = share of the gap removed)</summary><div class="scroll"><table class="tbl"><tr><th>Span</th><th>fit</th><th class="num">median N [95% CI]</th><th class="num">above paired-rotation q95</th><th class="num">above random-clamp q95</th><th class="num">flags member / partner</th></tr>{nec}</table></div></details>
  <details class="more"><summary>Additivity: I = S(span) + S(complement) − S(whole cell)</summary><div class="scroll"><table class="tbl"><tr><th>Span + complement</th><th class="num">signed I</th><th class="num">|I|</th><th class="num">fraction |I| ≥ 0.15</th></tr>{inter}</table></div></details>
  <details class="more"><summary>Reproducibility between the two half-sized fits (r1 vs r2)</summary><div class="scroll"><table class="tbl"><tr><th>Quantity · site · arm</th><th class="num">n</th><th class="num">Pearson r</th><th class="num">median |Δ|</th><th class="num">median r1 − r2</th><th class="num">median recovery primary / r1 / r2</th><th>high recovery reproduced</th></tr>{rep}</table></div></details>
  <div class="bottomline"><span class="eyebrow">Next decision (report 004 §14)</span>
  <p>The design said that if 004 does not nominate a construction, the project returns to comparing the readers
  directly rather than adding estimators automatically. Recommended: go back to the reader panel. Optional, bounded: a
  supervised method (DAS) at the same rank and edit size against PCA, only if its answer would change that plan.</p></div>
</section>
"""
    return html_, {"chart": chart, "scatter": scatter, "comp": comp_chart}


STATUS_HTML = """
<section aria-labelledby="h-status">
  <h2 id="h-status">Status</h2>
  <div class="status">
    <div class="ok"><h3>Works and is tested</h3><ul class="plain">
      <li>Pinned environment: model loads, 29/29 hashes match, and a rebuild script recreates everything.</li>
      <li>WorkspaceBench renders byte-identical to WSB's producer; partners, substitutions and the gate.</li>
      <li>Margin scorer, patching engines (RHUT cross-check, SEC true-fp32 replay), stats, figures, a model-integrity guard, and a reader-free import boundary. 65 CPU tests pass.</li>
      <li>Readers: logit lens, J_NP, J_CB, R_CB, Oracle Lens (L40), NLA (L42), plus their positive controls.</li>
      <li>Experiment 001 (dev): gate, smoke, run, probes (repaired, A2), reader snapshot, analysis, report (audited).</li>
      <li>Experiment 002 (dev): role-aware claim scoring of every snapshot readout (Gemini 3.8 Flash judge, validated), lens closed-set scores, prompt-only baseline.</li>
      <li>Experiment 003 (dev): reader-free 1-D labels. Gates G1–G3 pass; label reliability (G4) fails, so the labels are descriptive.</li>
      <li>Experiment 004 (dev, 33 pairs): span assay at “:” L42. Larger fitted spans recover more of the cell (joint span 0.72), but no more than PCA of the same rank; descriptive geometry result, no intervention nominated.</li>
      <li>Raw outputs mirrored to the public HF dataset senku21x/causal-concordance-outputs (MANIFEST.json with sha256).</li>
    </ul></div>
    <div class="no"><h3>Not working, parked or dropped</h3><ul class="plain">
      <li><b>vLLM acceleration:</b> the engine won't start on this GPU (sm_120 check inside FlashInfer). Parked; the HF route works.</li>
      <li><b>SAE:</b> deferred until you decide. One candidate is noted (ceselder/qwen36-27b-bsf-l42-1b).</li>
      <li><b>Skip-lens:</b> dropped. Its checkpoints are trained at L62, not L42.</li>
      <li><b>Modulation lens, phrase-J:</b> dropped. <b>Template lens:</b> deferred (exists for this model; its vocabulary lacks teal).</li>
    </ul></div>
    <div class="todo"><h3>Waiting on you, then next</h3><ul class="plain">
      <li>Confirm the framing: reader-panel comparison as the primary line, causal work as a validation branch (004 nominates no construction).</li>
      <li>Review reports 001–004; AMENDMENT_02 (τ, ε′, substitution bound) before any test-split label.</li>
      <li>Next (report 004 §14): return to the reader panel; a supervised follow-up (DAS vs PCA at matched rank and norm) only if it would change a decision.</li>
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
    exp2_html, exp2_data = exp002()
    exp3_html, exp3_data = exp003()
    exp4_html, exp4_data = exp004()
    story_html = story()
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
        "exp002": exp2_data,
        "exp003": exp3_data,
        "exp004": exp4_data,
    }
    t = open(os.path.join(HERE, "template.html")).read()
    for key in ("__EXP001__", "__STATUS__", "__DATA__"):
        assert t.count(key) == 1, key
    # Lens readouts include byte-fragment tokens that decode to U+FFFD; write it as an escape so the page has no raw one.
    js = json.dumps(data, ensure_ascii=False).replace("</", "<\\/").replace("\ufffd", "\\ufffd")
    page = (t.replace("__EXP001__", (story_html + exp_html + exp2_html + exp3_html + exp4_html).replace("\ufffd", "&#xFFFD;")).replace("__STATUS__", STATUS_HTML)
             .replace("__DATA__", js))
    assert "\ufffd" not in page
    out = R.out_dir("outputs", "000_setup", "results_page") / "causal_concordance.html"
    out.write_text(page)
    print(out, len(page))


if __name__ == "__main__":
    main()
