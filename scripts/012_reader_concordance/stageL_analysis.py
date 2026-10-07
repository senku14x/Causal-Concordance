"""012 Stage L analysis (CPU, EXPLORATORY; rule in spec §22). Hold success and cohort exactly as stageJ_analysis
(J1-gated discriminating trials, J_clue, hold success = >= half flipped). Readers' d_r at H2 / H1 on the Stage L
captures (token lenses L48/52/56/60, Oracle L52/56/60); AUC for hold success over all holds; no-Japan AUC; model's own
H2/H1 output AUC from Stage J for reference.
"""
import ast
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stageD_analysis import auc, cluster_boot

ROOT = Path(__file__).resolve().parents[2]
J = ROOT / "outputs/012_reader_concordance/stageJ"
L = ROOT / "outputs/012_reader_concordance/stageL"
RES = ROOT / "results/012_reader_concordance"
tf = lambda x: x is True or x == "True"
lit = lambda x: ast.literal_eval(x) if isinstance(x, str) else x


def main():
    tr = [json.loads(l) for l in open(J / "j2_items.jsonl")]
    g = [t for t in tr if tf(t["j1_gated"]) and tf(t["discriminating"])]
    hs = defaultdict(list)
    for t in g:
        hs[f"{t['clue_id']}|{t['carrier']}"].append(tf(lit(t["arms"])["J_clue"]["success"]))
    hold = {k: {"s": np.mean(v) >= 0.5, "clue": k.split("|")[0]} for k, v in hs.items()}
    tgt = {json.loads(l)["hold"]: json.loads(l)["tgt"] for l in open(J / "j2_hold.jsonl")}
    dr = defaultdict(dict)
    for l in open(L / "l_hold_ll_oracle_lenses.jsonl"):
        r = json.loads(l)
        if r["state"] == "J_clue" and r["hold"] in hold and r.get("d_r") not in (None, "None"):
            dr[(r["reader"], int(r["layer"]), r["pos"])][r["hold"]] = float(r["d_r"])
    out = {"label": "EXPLORATORY (012 Stage L; spec §22)", "n_holds": len(hold), "n_success": int(sum(h["s"] for h in hold.values())),
           "model_own_output_auc_from_stageJ": {k: json.load(open(RES / "analysis/stageJ_analysis.json"))[k]
                                                for k in ("J2_model_H1_bridge_auc", "J2_model_H2_bridge_auc")}, "readers": {}}
    for (rd, layer, pos), per in sorted(dr.items()):
        rows = [{"rel": hold[k]["clue"], "s": hold[k]["s"], "d": v, "k": k} for k, v in per.items()]
        f = lambda xs: auc([x["d"] for x in xs], [x["s"] for x in xs])
        out["readers"][f"{rd}|L{layer}|{pos}"] = {"n": len(rows), "auc": f(rows), "auc_ci95": cluster_boot(f, rows),
                                                  "auc_no_japan": f([x for x in rows if tgt.get(x["k"]) != "Japan"]),
                                                  "d_median_success": float(np.median([x["d"] for x in rows if x["s"]])),
                                                  "d_median_fail": float(np.median([x["d"] for x in rows if not x["s"]]))}
    h2 = {k: v["auc"] for k, v in out["readers"].items() if k.endswith("|H2")}
    best = max(h2.values()) if h2 else None
    out["H2_best"] = {"key": max(h2, key=h2.get), "auc": best} if h2 else None
    out["verdict_s22"] = ("arrives late" if best is not None and best >= 0.70 else
                          "not readable through L60" if best is not None and best <= 0.60 else "partial")
    (RES / "analysis/stageL_analysis.json").write_text(json.dumps(out, indent=1))
    r2 = lambda x: None if x is None else round(x, 2)
    for k, v in out["readers"].items():
        print(f"{k:22s} n={v['n']} AUC {r2(v['auc'])} {[r2(c) for c in v['auc_ci95'] or []]} noJ {r2(v['auc_no_japan'])}")
    print("verdict:", out["verdict_s22"], out["H2_best"])


if __name__ == "__main__":
    main()
