"""012 Stage J analysis (CPU, EXPLORATORY; rules in spec §19, specificity per §20 paired contrast).
J2: clue-only edit (positions < H1), J1-gated discriminating trials. Unit = hold (clue x carrier); hold success = >= half of its
gated questions flipped under J_clue. J-1 lever; J-2 model's own H1 next-token bridge shift AUC < 0.65; J-3 reader d_r at H1
(not written) AUC >= 0.75 with J-1, J-2 met. Oracle-vs-NLA: paired AUC difference, cluster bootstrap by clue."""
import ast
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stageD_analysis import auc, cluster_boot

ROOT = Path(__file__).resolve().parents[2]
J = ROOT / "outputs/012_reader_concordance/stageJ"
RES = ROOT / "results/012_reader_concordance"
tf = lambda x: x is True or x == "True"
lit = lambda x: ast.literal_eval(x) if isinstance(x, str) else x


def main():
    tr = [json.loads(l) for l in open(J / "j2_items.jsonl")]
    g = [t for t in tr if tf(t["j1_gated"]) and tf(t["discriminating"])]
    arms = lambda t: lit(t["arms"])
    out = {"label": "EXPLORATORY (012 Stage J; §19, §20)", "n_gated": len(g),
           "J_clue": sum(tf(arms(t)["J_clue"]["success"]) for t in g), "rand_clue": sum(tf(arms(t)["rand_clue"]["success"]) for t in g)}
    out["J1_lever"] = out["J_clue"] / len(g) >= 0.20 and (out["J_clue"] - out["rand_clue"]) / len(g) >= 0.15
    hs = defaultdict(list)
    for t in g:
        hs[f"{t['clue_id']}|{t['carrier']}"].append(tf(arms(t)["J_clue"]["success"]))
    hold = {k: {"s": np.mean(v) >= 0.5, "clue": k.split("|")[0]} for k, v in hs.items()}
    tgt = {}
    for l in open(J / "j2_hold.jsonl"):
        r = json.loads(l)
        if r["hold"] in hold:
            d = lit(r["d_model"])
            hold[r["hold"]]["mb_H1"] = d.get("J_clue|H1", {}).get("bridge")
            hold[r["hold"]]["mb_H2"] = d.get("J_clue|H2", {}).get("bridge")
            tgt[r["hold"]] = r["tgt"]
    out["n_holds"], out["n_hold_success"] = len(hold), int(sum(h["s"] for h in hold.values()))
    for pos in ("H1", "H2"):
        rows = [h for h in hold.values() if h.get(f"mb_{pos}") is not None]
        out[f"J2_model_{pos}_bridge_auc"] = auc([h[f"mb_{pos}"] for h in rows], [h["s"] for h in rows])
    out["J2_met_H1"] = out["J2_model_H1_bridge_auc"] is not None and out["J2_model_H1_bridge_auc"] < 0.65
    dr = defaultdict(dict)
    for f in ("j2_hold_ll_oracle_lenses.jsonl", "j2_hold_ll_nla.jsonl"):
        for l in open(J / f):
            r = json.loads(l)
            if r["state"] == "J_clue" and r["hold"] in hold and r.get("d_r") not in (None, "None"):
                dr[(r["reader"], str(r["layer"]), r["pos"])][r["hold"]] = float(r["d_r"])
    rd = {}
    for (reader, layer, pos), per in sorted(dr.items()):
        rows = [{"rel": hold[k]["clue"], "s": hold[k]["s"], "d": v, "k": k} for k, v in per.items()]
        f = lambda xs: auc([x["d"] for x in xs], [x["s"] for x in xs])
        a = f(rows)
        rd[f"{reader}|L{layer}|{pos}"] = {"n": len(rows), "auc": a, "auc_ci95": cluster_boot(f, rows),
                                          "auc_no_japan": f([x for x in rows if tgt.get(x["k"]) != "Japan"]),
                                          "d_median_success": float(np.median([x["d"] for x in rows if x["s"]])),
                                          "d_median_fail": float(np.median([x["d"] for x in rows if not x["s"]])),
                                          "J3_passes": bool(pos == "H1" and out["J1_lever"] and out["J2_met_H1"] and a is not None and a >= 0.75)}
    out["readers"] = rd
    pc = {}
    for ol in ("40", "44"):
        ko, kn = ("oracle", ol, "H1"), ("nla", "42", "H1")
        common = [k for k in dr[ko] if k in dr[kn]]
        rows = [{"rel": hold[k]["clue"], "s": hold[k]["s"], "o": dr[ko][k], "n_": dr[kn][k]} for k in common]
        fd = lambda xs: (auc([x["o"] for x in xs], [x["s"] for x in xs]) or np.nan) - (auc([x["n_"] for x in xs], [x["s"] for x in xs]) or np.nan)
        pc[f"oracle_L{ol}_minus_nla_L42_H1"] = {"n": len(rows), "diff": fd(rows), "ci95": cluster_boot(fd, rows)}
    out["paired_oracle_vs_nla"] = pc
    (RES / "analysis/stageJ_analysis.json").write_text(json.dumps(out, indent=1))
    r2 = lambda x: None if x is None else round(x, 2)
    print({k: (r2(v) if isinstance(v, float) else v) for k, v in out.items() if k not in ("readers", "paired_oracle_vs_nla")})
    for k, v in rd.items():
        print(f"{k:16s} n={v['n']} AUC {r2(v['auc'])} {[r2(c) for c in v['auc_ci95'] or []]} noJ {r2(v['auc_no_japan'])} d s/f {r2(v['d_median_success'])}/{r2(v['d_median_fail'])} J3 {v['J3_passes']}")
    print({k: {"diff": r2(v["diff"]), "ci": [r2(c) for c in v["ci95"] or []]} for k, v in pc.items()})


if __name__ == "__main__":
    main()
