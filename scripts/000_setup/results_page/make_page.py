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
         f"Candidate: off-path table colours. Swapping one out barely moves the answer ({frac(sub['behaviour_frac_negligible'], sub['n_items'])} "
         f"items within the bound; the line was 90%), and a cross-fitted probe follows the swap (98% of items at L60). The pooled "
         f"membership probe passes only at L60 (AUC {f2(mem60)}), and that call is provisional: its control task sits far from "
         "chance (median AUC 0.58, max 0.84), so the probes need repair first."),
        ("Do readers name the content the answer depends on?",
         "Not scored yet (no labels). Descriptively: the NLA mentions this table's z in 17 of 40 samples at “:” (shuffled 8 of 40), "
         "on 6 of 10 tables, and gives z the right role in about 5 of those 17. A within-table contrast (swap the start, which "
         f"changes z) shows some activation-sensitive signal: NLA {f2(trk['nla|cell2|L42']['mean'])}–{f2(trk['nla|cell0|L42']['mean'])}, "
         f"Oracle {f2(trk['oracle|cell2|L40']['mean'])}–{f2(trk['oracle|cell2|L44']['mean'])}. "
         + ("No lens lists a palette colour at L20–48. " if lens_any == 0 else f"Lenses list a palette colour at L20–48 in at most {f2(lens_any)} of readouts. ") +
         "Exploratory, 10 tables, word matching."),
        ("Do readers name content that is not needed?",
         f"Off-path candidates are mentioned about as often with a shuffled activation as with the real one (Oracle "
         f"{f2(orc_off[0])} vs {f2(orc_off[1])}), and no substitution tracking was detected. Both are exploratory nulls at n = 10."),
        ("Can the measurements be trusted?",
         f"The patching engine, yes: every exact check passes on every pair (self-patch, embedding anchor, prefix, stir no-op: "
         f"bitwise), and the SEC noise floor is {b['noise_floor']['upper_L42_rms_frac_G']:.1e} of the gap. Two diagnostics need "
         "repair: the probe controls and permutation null, and the damage readout (the saved top 5 tokens cannot say where "
         "most off-pair mass goes). Found by an external audit; see report §15."),
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
  <p>Six questions. The answers so far come from experiment 001 on the dev split; nothing has been run on the test split, and no label or reader score exists yet.</p>
  <ol class="look">{look_html}</ol>
</section>

<section aria-labelledby="h-001">
  <div class="eyebrow">Experiment 001 · dev split · SEC regime (true fp32 from block 40) · report: docs/reports/001_leverage_profile.md</div>
  <h2 id="h-001">Experiment 001: where the answer depends on one cell, and what sits there</h2>
  <p class="small" style="background:var(--warn-soft);color:var(--ink);padding:10px 14px;border-radius:8px"><b>Revised after an external audit (2026-10-02).</b> Every number was recomputed independently from the saved outputs and matched. Corrected: the damage reading, reader mention counts read as claims, the reconstruction and null wording, and interval estimands. Still to repair: the probe controls and the permutation null. Defensible summary: the readers share task-structure information, show some activation-sensitive colour signal, and often surround it with incorrect detail. Whether their claims track causally influential content is unresolved. Report §15 has the point-by-point table.</p>
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
  <p class="small muted">L2 logistic regression, 5-fold by table, nested C. Control task (Hewitt–Liang) at 0.05–0.12; ✓ = BH-significant and selectivity ≥ 0.10. Chance ≈ 0.083 for 12 colours. <b>Membership row provisional:</b> its control task sits far from chance (median AUC 0.58, max 0.84), and the permutation null reuses the regularisation chosen on the real labels. Both need repair before any membership call is used.</p>
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
    <li>Repair the probes (a permutation null that repeats C tuning; diagnose the off-chance control) before any membership call is used.</li>
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
    data = {"bars": bars, "lens": A["lens_closed_set"], "prompt_only": po, "qual": Q, "top10": A["lens_top10_claims"]}
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
  <h3>Qualitative: judged readouts</h3>
  <div class="seg" id="ex2-qsel"></div>
  <div id="ex2-q" class="panel"></div>
  <h3>Predictions written before the judge ran</h3>
  <div class="scroll"><table class="tbl"><tr><th>Prediction</th><th>Value</th><th>Outcome</th></tr>{pr}</table></div>
</section>
"""
    return html_, data


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
      <li><b>Modulation lens, phrase-J:</b> dropped. <b>Template lens:</b> deferred (exists for this model; its vocabulary lacks teal).</li>
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
    exp2_html, exp2_data = exp002()
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
    }
    t = open(os.path.join(HERE, "template.html")).read()
    for key in ("__EXP001__", "__STATUS__", "__DATA__"):
        assert t.count(key) == 1, key
    # Lens readouts include byte-fragment tokens that decode to U+FFFD; write it as an escape so the page has no raw one.
    js = json.dumps(data, ensure_ascii=False).replace("</", "<\\/").replace("\ufffd", "\\ufffd")
    page = (t.replace("__EXP001__", (exp_html + exp2_html).replace("\ufffd", "&#xFFFD;")).replace("__STATUS__", STATUS_HTML)
             .replace("__DATA__", js))
    assert "\ufffd" not in page
    out = R.out_dir("outputs", "000_setup", "results_page") / "causal_concordance.html"
    out.write_text(page)
    print(out, len(page))


if __name__ == "__main__":
    main()
