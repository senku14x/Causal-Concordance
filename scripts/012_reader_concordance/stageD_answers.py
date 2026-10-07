"""012 Stage D add-on (CPU, EXPLORATORY, POST HOC): is the readers' bridge shift more than answer-facing content?

Within toward_tgt items (primary cohort), per reader x layer x band:
  - AUC for swap success of the bridge shift d_r and of the answer shift d_ans (same reader, same states);
  - Spearman(d_r, d_ans);
  - partial Spearman(d_r, Δm | d_ans, edit size): bridge tracking beyond the reader's own answer shift;
  - partial Spearman(d_r, Δm | model's own read-position output for bridge and answer, edit size): beyond what the
    model's output distribution at that cell already carries (the 006b / 009 output-facing confound).
Also: AUC of the model's own output shifts (d_model_bridge, d_model_answer).
d values are in each instrument's own nats; compare AUCs and rank statistics, not raw magnitudes.

  python stageD_answers.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stageD_analysis import B, C, RES, auc, cluster_boot, rank, rel

ROOT = Path(__file__).resolve().parents[2]


def partial(a, b, Z):
    if len(a) < 6:
        return None
    X = np.c_[np.ones(len(a)), np.column_stack([rank(z) for z in Z])]
    res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.corrcoef(res(rank(a)), res(rank(b)))[0, 1])


def main():
    items = {}
    for l in open(B):
        r = json.loads(l)
        if r.get("status") != "evaluated" or not r["clean_correct"] or r["clean_contains_y_b"]:
            continue
        items[r["name"]] = {"rel": rel(r["name"]), "japan": r["tgt"] == "Japan"}
        for b in ("L24-35", "L24-47"):
            J = r["arms"][f"J_{b}"]
            dh = [v["dh"] / v["h"] for v in J["diag"].values() if v.get("h")]
            items[r["name"]][b] = {"success": bool(J["success"]), "dm": J["dm"], "direction": r["edit_direction"][b],
                                   "size": float(np.mean(dh))}
    mo = {}
    for l in open(ROOT / "outputs/012_reader_concordance/stageB/model_output.jsonl"):
        r = json.loads(l)
        arms = r["arms"] if isinstance(r["arms"], dict) else json.loads(r["arms"])
        mo[r["name"]] = {b: (arms.get(f"J_{b}", {}).get("d_model_bridge"), arms.get(f"J_{b}", {}).get("d_model_answer"))
                         for b in ("L24-35", "L24-47")}
    vals = defaultdict(lambda: defaultdict(dict))
    for f, k in (("ll_oracle_lenses.jsonl", "dJ"), ("ll_nla.jsonl", "dJ"),
                 ("ll_answers_oracle_lenses.jsonl", "dA"), ("ll_answers_nla.jsonl", "dA")):
        for l in open(C / f):
            r = json.loads(l)
            if r["item"] not in items or not r["state"].startswith("J_"):
                continue
            v = r.get("d_r") if k == "dJ" else r.get("d_ans")
            if v in (None, "None", ""):
                continue
            key = (r["reader"] + ("_offtrain" if r.get("off_training") in (True, "True") else ""), str(r["layer"]), r["band"])
            vals[key][r["item"]][k] = float(v)
    out = {"label": "EXPLORATORY, POST HOC (012 Stage D add-on)", "cells": {}, "model_output": {}}
    for b in ("L24-35", "L24-47"):
        tt = [n for n in items if items[n][b]["direction"] == "toward_tgt" and mo.get(n, {}).get(b, (None,))[0] is not None]
        y = [items[n][b]["success"] for n in tt]
        out["model_output"][b] = {"n": len(tt),
                                  "auc_d_model_bridge": auc([mo[n][b][0] for n in tt], y),
                                  "auc_d_model_answer": auc([mo[n][b][1] for n in tt if mo[n][b][1] is not None],
                                                            [items[n][b]["success"] for n in tt if mo[n][b][1] is not None])}
    for (reader, layer, band), per in sorted(vals.items()):
        rows = [dict(name=n, rel=items[n]["rel"], s=items[n][band]["success"], dm=items[n][band]["dm"],
                     size=items[n][band]["size"], dJ=v["dJ"], dA=v["dA"], mb=mo[n][band][0], ma=mo[n][band][1])
                for n, v in per.items() if "dJ" in v and "dA" in v and items[n][band]["direction"] == "toward_tgt"
                and mo.get(n, {}).get(band, (None, None))[1] is not None]
        if len(rows) < 10:
            continue
        g = lambda k: [x[k] for x in rows]
        pb = lambda xs: partial([x["dJ"] for x in xs], [x["dm"] for x in xs],
                                [[x["mb"] for x in xs], [x["ma"] for x in xs], [x["size"] for x in xs]])
        out["cells"][f"{reader}|L{layer}|{band}"] = {
            "n_toward_tgt": len(rows), "n_success": sum(g("s")),
            "auc_bridge": auc(g("dJ"), g("s")), "auc_answer": auc(g("dA"), g("s")),
            "spearman_bridge_answer": float(np.corrcoef(rank(g("dJ")), rank(g("dA")))[0, 1]),
            "partial_bridge_dm_given_answer_size": partial(g("dJ"), g("dm"), [g("dA"), g("size")]),
            "partial_bridge_dm_given_model_output_size": pb(rows),
            "partial_bridge_dm_given_model_output_size_ci95": cluster_boot(pb, rows),
        }
    (RES / "analysis/stageD_answers.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["model_output"], indent=0))
    for k, v in out["cells"].items():
        r2 = lambda x: None if x is None else round(x, 2)
        print(f"{k:26s} n={v['n_toward_tgt']} AUC bridge {r2(v['auc_bridge'])} answer {r2(v['auc_answer'])}  "
              f"rho(b,a) {r2(v['spearman_bridge_answer'])}  pS|ans {r2(v['partial_bridge_dm_given_answer_size'])}  "
              f"pS|model-out {r2(v['partial_bridge_dm_given_model_output_size'])} {[r2(c) for c in (v['partial_bridge_dm_given_model_output_size_ci95'] or [])]}")


if __name__ == "__main__":
    main()
