"""012 Stage I analysis (CPU, EXPLORATORY; rules pre-registered in spec §18).

Unit = hold state (clue x carrier); its outcome = fraction of its discriminating questions whose answer flipped under J_pre;
hold success = fraction >= 0.5. PRIMARY cohort (as pre-registered): gated discriminating trials only. SENSITIVITY
(labelled, not pre-registered): all discriminating trials (ungated); readers without the toward_tgt-at-H2 restriction
(post hoc: the Stage I report showed the H2 direction does not predict flips); Japan targets excluded.
I-1 lever; I-2 model's own H2 bridge shift AUC < 0.65; I-3 reader d_r at H2 AUC >= 0.75 (with I-2 met).
"""
import ast
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stageD_analysis import auc, cluster_boot

ROOT = Path(__file__).resolve().parents[2]
I = ROOT / "outputs/012_reader_concordance/stageI"
RES = ROOT / "results/012_reader_concordance"
tf = lambda x: x is True or x == "True"


def num(x):
    if isinstance(x, dict):
        for k in ("bridge", "d_model_bridge", "d_bridge"):
            if k in x:
                return float(x[k])
        return None
    if isinstance(x, str):
        try:
            return num(ast.literal_eval(x))
        except Exception:
            return float(x)
    return None if x is None else float(x)


def holds(trials, gated_only):
    hs = defaultdict(lambda: {"flips": [], "rand": [], "clue": None, "tgt": None, "dir": None, "mb": None})
    for t in trials:
        if not tf(t["discriminating"]) or (gated_only and not tf(t["gated"])):
            continue
        h = hs[f"{t['clue_id']}|{t['carrier']}"]
        h["flips"].append(tf(t["arms"]["J_pre"]["success"]))
        h["rand"].append(tf(t["arms"]["rand_pre"]["success"]))
        h["clue"], h["tgt"], h["dir"] = t["clue_id"], t["tgt"], t["edit_direction_H2"]
        h["mb"] = num(t["arms"]["J_pre"].get("d_model_H2"))
    for h in hs.values():
        h["frac"] = float(np.mean(h["flips"]))
        h["success"] = h["frac"] >= 0.5
    return dict(hs)


def main():
    trials = [json.loads(l) for l in open(I / "i_items.jsonl")]
    gated = [t for t in trials if tf(t["gated"]) and tf(t["discriminating"])]
    disc = [t for t in trials if tf(t["discriminating"])]
    out = {"label": "EXPLORATORY (012 Stage I; rules in spec §18)",
           "I1_lever": {"gated_n": len(gated), "J_pre": sum(tf(t["arms"]["J_pre"]["success"]) for t in gated),
                        "rand_pre": sum(tf(t["arms"]["rand_pre"]["success"]) for t in gated),
                        "J_all": sum(tf(t["arms"]["J_all"]["success"]) for t in gated),
                        "ungated_disc_n": len(disc), "ungated_J_pre": sum(tf(t["arms"]["J_pre"]["success"]) for t in disc),
                        "ungated_rand_pre": sum(tf(t["arms"]["rand_pre"]["success"]) for t in disc),
                        "gated_by_relation": {r: sum(t["relation"] == r for t in gated) for r in ("capital", "language", "continent")}}}
    L = out["I1_lever"]
    L["passes"] = L["J_pre"] / L["gated_n"] >= 0.20 and (L["J_pre"] - L["rand_pre"]) / L["gated_n"] >= 0.15
    dr = defaultdict(dict)
    for f in ("i_hold_ll_oracle_lenses.jsonl", "i_hold_ll_nla.jsonl"):
        for l in open(I / f):
            r = json.loads(l)
            if r["state"] == "J_pre" and r.get("d_r") not in (None, "None"):
                dr[(r["reader"], str(r["layer"]), r["pos"])][r["hold"]] = float(r["d_r"])
    for cohort, gonly in (("gated_primary", True), ("ungated_sensitivity", False)):
        hs = holds(trials, gonly)
        res = {"n_holds": len(hs), "n_success": sum(h["success"] for h in hs.values()),
               "n_clues": len({h["clue"] for h in hs.values()}),
               "bridge_level_all_flipped_share": float(np.mean([all(h["flips"]) for h in hs.values()])) if hs else None}
        mrows = [{"rel": h["clue"], "s": h["success"], "x": h["mb"]} for h in hs.values() if h["mb"] is not None]
        a2 = auc([x["x"] for x in mrows], [x["s"] for x in mrows])
        res["I2_model_H2_bridge_auc"] = a2
        res["I2_met"] = a2 is not None and a2 < 0.65
        rd = {}
        for (reader, layer, pos), per in sorted(dr.items()):
            for strat in ("toward_tgt_H2", "all"):
                rows = [{"rel": hs[k]["clue"], "s": hs[k]["success"], "d": v} for k, v in per.items()
                        if k in hs and (strat == "all" or hs[k]["dir"] == "toward_tgt")]
                rows_nj = [{"rel": hs[k]["clue"], "s": hs[k]["success"], "d": v} for k, v in per.items() if k in hs and hs[k]["tgt"] != "Japan"]
                f = lambda xs: auc([x["d"] for x in xs], [x["s"] for x in xs])
                a = f(rows)
                rd[f"{reader}|L{layer}|{pos}|{strat}"] = {
                    "n": len(rows), "n_success": sum(x["s"] for x in rows), "auc": a, "auc_ci95": cluster_boot(f, rows),
                    "auc_no_japan": f(rows_nj) if strat == "all" else None,
                    "d_median_success": float(np.median([x["d"] for x in rows if x["s"]])) if any(x["s"] for x in rows) else None,
                    "d_median_fail": float(np.median([x["d"] for x in rows if not x["s"]])) if any(not x["s"] for x in rows) else None,
                    "I3_passes": bool(a is not None and a >= 0.75 and res["I2_met"])}
        res["readers"] = rd
        out[cohort] = res
    # naming at H2 (Oracle L44, NLA L42), clean vs J_pre, by primary hold success
    hs = holds(trials, True)
    nam = defaultdict(lambda: defaultdict(int))
    for f in ("i_readouts_oracle.jsonl", "i_readouts_nla.jsonl"):
        for l in open(I / f):
            r = json.loads(l)
            if r["hold"] not in hs or r["state"] not in ("clean", "J_pre"):
                continue
            g = "clean" if r["state"] == "clean" else ("J_pre|success" if hs[r["hold"]]["success"] else "J_pre|fail")
            cats = r["category"] if isinstance(r["category"], list) else ast.literal_eval(r["category"])
            for c in cats:
                nam[f"{r['reader']} L{r['layer']}|{g}"][c] += 1
    out["naming_H2"] = {k: dict(v) for k, v in sorted(nam.items())}
    (RES / "analysis/stageI_analysis.json").write_text(json.dumps(out, indent=1))
    r2 = lambda x: None if x is None else round(x, 2)
    print("I1", L)
    for c in ("gated_primary", "ungated_sensitivity"):
        o = out[c]
        print(c, {k: r2(v) if isinstance(v, float) else v for k, v in o.items() if k != "readers"})
        for k, v in o["readers"].items():
            if "|H2|" in k:
                print("  ", f"{k:30s} n={v['n']} s={v['n_success']} AUC {r2(v['auc'])} {[r2(x) for x in v['auc_ci95'] or []]} noJ {r2(v['auc_no_japan'])} d s/f {r2(v['d_median_success'])}/{r2(v['d_median_fail'])} I3 {v['I3_passes']}")
    print(json.dumps(out["naming_H2"]))


if __name__ == "__main__":
    main()
