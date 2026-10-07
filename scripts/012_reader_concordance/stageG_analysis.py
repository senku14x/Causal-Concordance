"""012 Stage G analysis (CPU, EXPLORATORY; rules in spec §14): whitened vs raw J swap at L24-35; readers at the read
cell (bridge/answer AUC for J_white success within toward_tgt in whitened coordinates)."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stageD_analysis import auc, cluster_boot, rel

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / "outputs/012_reader_concordance/stageG"
RES = ROOT / "results/012_reader_concordance"


def main():
    keep = {json.loads(l)["name"] for l in open(ROOT / "outputs/012_reader_concordance/stageB/items.jsonl")
            if (lambda r: r.get("status") == "evaluated" and r["clean_correct"] and not r["clean_contains_y_b"])(json.loads(l))}
    it = {}
    for l in open(G / "g_items.jsonl"):
        r = json.loads(l)
        if r["name"] not in keep:
            continue
        a = r["arms"]
        tf = lambda x: x is True or x == "True"
        it[r["name"]] = {"rel": rel(r["name"]), "s": tf(a["J_white_L24-35"]["success"]), "sr": tf(a["rand_white_L24-35"]["success"]),
                         "dm": float(a["J_white_L24-35"]["dm"]), "dir": r.get("edit_direction_white") or a["J_white_L24-35"].get("edit_direction_white")}
    n = len(it)
    out = {"label": "EXPLORATORY (012 Stage G; §14)", "n": n, "flips_white": sum(v["s"] for v in it.values()),
           "flips_rand_white": sum(v["sr"] for v in it.values()), "flips_raw_reference": 38,
           "verdict_s14": None, "readers": {}}
    d = (out["flips_white"] - 38) / n
    out["verdict_s14"] = ("whitening helps" if d >= 0.10 and out["flips_rand_white"] / n <= 0.05 else
                          "whitening hurts" if d <= -0.10 else "no material difference")
    ll = defaultdict(dict)
    for f in ("g_ll_oracle_lenses.jsonl", "g_ll_nla.jsonl"):
        for l in open(G / f):
            r = json.loads(l)
            if r["item"] in it and r["state"] == "J_white_L24-35" and r.get("d") not in (None, "None"):
                ll[(r["reader"], str(r["layer"]), r["kind"])][r["item"]] = float(r["d"])
    for (rd, lay, kind), per in sorted(ll.items()):
        rows = [{"rel": it[k]["rel"], "d": v, "s": it[k]["s"]} for k, v in per.items() if it[k]["dir"] == "toward_tgt"]
        f = lambda xs: auc([x["d"] for x in xs], [x["s"] for x in xs])
        out["readers"][f"{rd}|L{lay}|{kind}"] = {"n_toward_tgt": len(rows), "n_success": sum(x["s"] for x in rows),
                                                 "auc": f(rows), "auc_ci95": cluster_boot(f, rows), "d_median": float(np.median([x["d"] for x in rows]))}
    (RES / "analysis/stageG_analysis.json").write_text(json.dumps(out, indent=1))
    print({k: out[k] for k in ("n", "flips_white", "flips_rand_white", "verdict_s14")})
    for k, v in out["readers"].items():
        if k.endswith("bridge"):
            print(f"{k:22s} n={v['n_toward_tgt']} s={v['n_success']} AUC {round(v['auc'], 2)} {[round(c, 2) for c in v['auc_ci95'] or []]}")


if __name__ == "__main__":
    main()
