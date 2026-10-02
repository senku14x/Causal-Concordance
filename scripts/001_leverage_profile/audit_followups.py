"""Follow-ups to the external audit of report 001 (Codex, 2026-10-02; /workspace/codex_reports/2026-10-02_reader_audit.md).

Recomputes every quantity the audit raised, independently, from saved 001 outputs only
(results/001_leverage_profile/analysis/*.json, outputs/001_leverage_profile/run/*.jsonl). No model, no reader, no GPU.
Everything here is exploratory and descriptive: the contrasts were chosen after the run, n = 10 snapshot tables.

    python scripts/001_leverage_profile/audit_followups.py  ->  results/001_leverage_profile/analysis/audit_followups.json
"""
import itertools
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import leverage_profile as LP  # noqa: E402  the snapshot's own colour scorer (colours_in_text / colours_in_tokens)
import partners as P  # noqa: E402
import registry as R  # noqa: E402
import splits as SP  # noqa: E402
import stats as ST  # noqa: E402
import wsb  # noqa: E402

A = R.PROJECT / "results" / "001_leverage_profile" / "analysis"
RUN = R.PROJECT / "outputs" / "001_leverage_profile" / "run"
LAYERS = [40, 42, 44, 56, 60]
EXTRA_COLOURS = ["yellow", "cyan", "violet", "magenta", "silver", "beige", "maroon", "navy", "indigo", "turquoise",
                 "crimson", "amber", "lime", "olive", "tan", "bronze", "copper", "scarlet", "黄", "紫红"]


def jl(p):
    return [json.loads(l) for l in open(p)]


def signflip_p(v):
    """Exact two-sided sign-flip p over per-table values (unadjusted; exploratory)."""
    v = np.asarray(v, float)
    obs = abs(v.mean())
    flips = np.array(list(itertools.product([1, -1], repeat=len(v))))
    return float(np.mean(np.abs((flips * v).mean(1)) >= obs - 1e-12))


def summarise(v):
    v = np.asarray(v, float)
    bs = ST.cluster_bootstrap(v, np.arange(len(v)))
    return {"mean": float(v.mean()), "boot95_tables": bs["ci"], "pos": int((v > 0).sum()), "neg": int((v < 0).sum()),
            "tie": int((v == 0).sum()), "signflip_p_unadj": signflip_p(v), "per_table": [round(float(x), 4) for x in v]}


# ------------------------------------------------------------------------------------------------ readers
def reader_contrasts(sn):
    T = sn["tables"]
    specs = [("nla", "cell0", 42), ("nla", "cell1", 42), ("nla", "cell2", 42), ("oracle", "cell2", 40), ("oracle", "cell2", 44)]

    def texts(t, role, rd, cell, L, arm):
        v = t["readouts"][f"{role}|{cell}|L{L}|{rd}|{arm}"]
        return v["explanations"] if rd == "nla" else v

    def rate(t, role, rd, cell, L, arm, colour):
        xs = texts(t, role, rd, cell, L, arm)
        return float(np.mean([colour in LP.colours_in_text(x or "") for x in xs]))

    out = {}
    for rd, cell, L in specs:
        rows = defaultdict(list)
        for i, t in enumerate(T):
            z, zp = t["z"], t["partner"]["z"]
            c, c2 = t["substitution"]["c"], t["substitution"]["c2"]
            r = lambda role, arm, col: rate(t, role, rd, cell, L, arm, col)  # noqa: E731
            rows["real_z"].append(r("member", "real", z))
            rows["shuffled_z"].append(r("member", "shuffled", z))
            rows["real_minus_shuffled"].append(r("member", "real", z) - r("member", "shuffled", z))
            rows["start_pair_tracking"].append(((r("member", "real", z) - r("partner", "real", z)) +
                                                (r("partner", "real", zp) - r("member", "real", zp))) / 2)
            rows["substitution_tracking"].append(((r("member", "real", c) - r("substituted", "real", c)) +
                                                  (r("substituted", "real", c2) - r("member", "real", c2))) / 2)
            path = {t["start"], z, t["answer"]}
            others = [u["z"] for j, u in enumerate(T) if j != i and u["z"] not in path]
            rows["own_z_minus_other_tables_z"].append(r("member", "real", z) - float(np.mean([r("member", "real", o) for o in others])))
        out[f"{rd}|{cell}|L{L}"] = {k: summarise(v) if k not in ("real_z", "shuffled_z") else
                                    {"mean": float(np.mean(v)), "tables_nonzero": int(np.sum(np.array(v) > 0)),
                                     "per_table": v} for k, v in rows.items()}
    # colour base rates over every real readout of the reader at that cell
    for rd, cell, L in specs:
        n, cnt = 0, defaultdict(int)
        for t in T:
            for role in ("member", "partner", "substituted"):
                for x in texts(t, role, rd, cell, L, "real"):
                    n += 1
                    for col in LP.colours_in_text(x or ""):
                        cnt[col] += 1
        out[f"{rd}|{cell}|L{L}"]["colour_base_rates"] = {k: round(v / n, 3) for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])}
    return out


def nla_z_mentions(sn):
    """Every member/real NLA explanation at ':' L42 that mentions z, with the sentence(s) naming it (for role reading)."""
    rows = []
    for t in sn["tables"]:
        for k, x in enumerate(t["readouts"]["member|cell2|L42|nla|real"]["explanations"]):
            if x and t["z"] in LP.colours_in_text(x):
                sents = [s.strip() for s in re.split(r"(?<=[.;\n])\s+", x) if re.search(rf"\b{t['z']}\b", s, re.I)]
                rows.append({"table": t["item"], "trajectory": f'{t["start"]} -> {t["z"]} -> {t["answer"]}', "sample": k + 1,
                             "z_sentences": sents, "colours_named": sorted(LP.colours_in_text(x))})
    return rows


def lens_offpalette(sn):
    """Lens tokens at ':' (member, real, L20-48) that name a colour outside the scorer's 12-colour palette."""
    hits = []
    for t in sn["tables"]:
        for key, v in t["readouts"].items():
            p = key.split("|")
            if len(p) == 5 and p[0] == "member" and p[1] == "cell2" and p[3] == "lenses" and p[4] == "real" and int(p[2][1:]) <= 48:
                for lens, toks in v.items():
                    for tk in toks:
                        w = re.sub(r"[^a-z]", "", tk.lower())
                        if w in EXTRA_COLOURS or any(ch in tk for ch in EXTRA_COLOURS if not ch.isascii()):
                            hits.append({"table": t["item"], "layer": p[2], "lens": lens, "token": tk})
    return hits


def fve_source_vs_target(sn):
    """Shuffled-arm NLA explanations scored by the AR against their true source cell vs the (wrong) target cell."""
    d, per_table = [], defaultdict(list)
    for t in sn["tables"]:
        for key, v in t["readouts"].items():
            p = key.split("|")
            if len(p) == 5 and p[3] == "nla" and p[4] == "shuffled":
                for a, b in zip(v["fve_vs_source"], v["fve_vs_cell"]):
                    if a is not None and b is not None:
                        d.append(a - b)
                        per_table[t["item"]].append(a - b)
    d = np.array(d)
    return {"n_explanations": int(len(d)), "source_wins": int((d > 0).sum()), "mean_source_minus_target": float(d.mean()),
            "per_target_table": summarise([np.mean(v) for v in per_table.values()])}


# ------------------------------------------------------------------------------------------------ assay
def damage(bank):
    """Off-pair mass (next-token mass outside both answers' first-token sets) at the brew emission cells: clean vs edited,
    and what the saved top-5 of the edited distribution can and cannot account for."""
    items = {it["name"]: it for it in bank}
    zz = {}
    for it in bank:
        for p in P.brew_candidates(it):
            zz[p.pair_id] = (p.m.z, p.p.z, p.m.a, p.p.a)
    rows = [r for r in jl(RUN / "profile_brew.jsonl") if r["direction"] == "p->m" and r["region"].startswith("emit")]
    out = {}
    for region in ("emit_colon", "emit_asst", "emit_think"):
        for L in LAYERS:
            rr = [r for r in rows if r["region"] == region and r["layer"] == L]
            dec = defaultdict(list)
            for r in rr:
                z, zp, a, ap = zz[r["pair_id"]]
                parts = defaultdict(float)
                for tok, lp in r.get("top5") or []:
                    w = re.sub(r"[^a-z]", "", tok.lower())
                    k = ("answers" if w in (a, ap) else "intermediates" if w in (z, zp) else
                         "other_colours" if w in LP._ZH else "format_or_other")
                    parts[k] += float(np.exp(lp))
                for k in ("answers", "intermediates", "other_colours", "format_or_other"):
                    dec[k].append(parts[k])
                dec["top5_total"].append(sum(parts.values()))
                dec["off_pair_clean"].append(r["off_pair_mass_clean"])
                dec["off_pair_edit"].append(r["off_pair_mass_edit"])
                dec["off_pair_increase"].append(r["off_pair_mass_edit"] - r["off_pair_mass_clean"])
            m = {k: float(np.mean(v)) for k, v in dec.items()}
            m["flagged_frac"] = float(np.mean(np.array(dec["off_pair_increase"]) > 0.1))
            m["flagged"] = f'{int(np.sum(np.array(dec["off_pair_increase"]) > 0.1))}/{len(rr)}'
            m["off_pair_in_top5"] = m["intermediates"] + m["other_colours"] + m["format_or_other"]
            m["mass_outside_top5"] = 1 - m["top5_total"]
            out[f"{region}|L{L}"] = m
    return out


def estimands(bank):
    """Brew: pair-level median S with a table-cluster bootstrap CI, next to the table-mean with its cluster-t CI."""
    rows = [r for r in jl(RUN / "profile_brew.jsonl") if r["direction"] == "p->m"]
    q_last = max(r["pos"] for r in rows if r["region"] == "question")
    cells = {"emit_colon": 164, "emit_asst": 158, "emit_think": 162, "start_colour": 142, "question_last": q_last}
    out = {}
    for name, pos in cells.items():
        for L in LAYERS:
            rr = [r for r in rows if r["pos"] == pos and r["layer"] == L]
            S = np.array([r["S"] for r in rr])
            tab = np.array([r["pair_id"].split("__")[0] for r in rr])
            med = ST.cluster_bootstrap(S, tab, stat=np.median)
            means = [S[tab == t].mean() for t in sorted(set(tab))]
            ct = ST.cluster_t(means)
            out[f"{name}|L{L}"] = {"pair_median": med["stat"], "pair_median_boot95_tables": med["ci"],
                                   "table_mean": ct["mean"], "table_mean_t95": ct["ci"], "n_pairs": len(S), "n_tables": len(means)}
    return out


def chain_by_group():
    """Chain: the same cells clustered by step-sequence group (the independent split unit), not by item."""
    _, bank = wsb.load_bank("chain")
    grp = {n: k for k, names in SP.chain_groups(bank).items() for n in names}
    rows = [r for r in jl(RUN / "profile_chain.jsonl") if r["direction"] == "p->m"]
    out = {}
    for region, pick in (("final", lambda r: r["region"] == "final"),):
        for L in LAYERS:
            rr = [r for r in rows if pick(r) and r["layer"] == L]
            S = np.array([r["S"] for r in rr])
            g = np.array([grp[r["pair_id"].split("__")[0]] for r in rr])
            means = [S[g == k].mean() for k in sorted(set(g))]
            ct = ST.cluster_t(means)
            med = ST.cluster_bootstrap(S, g, stat=np.median)
            out[f"{region}|L{L}"] = {"pair_median": med["stat"], "pair_median_boot95_groups": med["ci"], "group_mean": ct["mean"],
                                     "group_mean_t95": ct["ci"], "n_pairs": len(S), "n_groups": len(means),
                                     "pairs_per_group": sorted([int((g == k).sum()) for k in set(g)], reverse=True)}
    return out


def probe_controls():
    pr = json.load(open(A / "probes.json"))
    cells, layers = pr["cells"], pr["layers"]
    mem = [r for r in pr["rows"] if r.get("kind") != "12-way" and r.get("status") == "ok"]
    multi = [r for r in pr["rows"] if r.get("kind") == "12-way" and r.get("status") == "ok"]
    ctl = [r["control"] for r in mem if r["control"] == r["control"]]
    top = sorted(mem, key=lambda r: -(r["control"] if r["control"] == r["control"] else -1))[:8]
    return {
        "twelve_way_control_range": [min(r["control"] for r in multi), max(r["control"] for r in multi)],
        "membership_control": {"n": len(ctl), "median": float(np.median(ctl)), "q90": float(np.quantile(ctl, 0.9)),
                               "max": float(max(ctl)), "n_above_0.6": int(sum(c > 0.6 for c in ctl)), "n_above_0.7": int(sum(c > 0.7 for c in ctl)),
                               "highest": [{"colour": r["target"], "cell": cells[r["cell"]], "layer": layers[r["layer"]],
                                            "control": round(r["control"], 3), "real": round(r["real"], 3), "n_tables": r["n_tables"],
                                            "classes": r["classes"]} for r in top]},
        "permutation_null_note": "null fits reuse the C chosen per outer fold on the real labels (C_fixed=Cs); the real and "
                                 "control fits tune C by inner CV. The null therefore does not repeat the full procedure.",
    }


def main():
    sn = json.load(open(A / "snapshot.json"))
    _, bank = wsb.load_bank("brew")
    res = {"source": "scripts/001_leverage_profile/audit_followups.py", "status": "exploratory, post hoc, n = 10 snapshot tables",
           "readers": reader_contrasts(sn), "nla_colon_z_mentions": nla_z_mentions(sn), "lens_off_palette_L20_48": lens_offpalette(sn),
           "fve_source_vs_target": fve_source_vs_target(sn), "damage": damage(bank), "brew_estimands": estimands(bank),
           "chain_by_step_group": chain_by_group(), "probe_controls": probe_controls()}
    (A / "audit_followups.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    print(A / "audit_followups.json")


if __name__ == "__main__":
    main()
