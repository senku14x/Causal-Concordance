"""012 Stage F analysis (CPU, EXPLORATORY; rules pre-registered in spec §13).

F1 (position split, J swap L24-35): flips per arm; readers at the read cell under `allbutread` (cell never written):
    AUC of bridge d_r and answer d_ans for allbutread success (pre-registered: unstratified; toward_tgt secondary),
    partial Spearman(d_r, Δm | model's own read-cell output, total write norm).
F2 (all-position J swap L24-35, reads at early positions): per item the mean over early positions (read token excluded;
    the token before the read token also excluded when the read token is " is") of reader d_r and of the model's own
    next-token output shift; AUC within toward_tgt; partial Spearman(d_r_early, Δm | model early output, edit size).
Rules (§13): F1 (i) allbutread flip >= 0.2 and >> control, readers AUC >= 0.8 -> readers follow model-computed states.
F2: AUC >= 0.75 and partial CI > 0 -> bridge content beyond output-facing; AUC < 0.65 -> not outcome-specific;
d ~ 0 -> absent.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stageD_analysis import auc, cluster_boot, rank, rel
from stageD_answers import partial

ROOT = Path(__file__).resolve().parents[2]
F = ROOT / "outputs/012_reader_concordance/stageF"
RES = ROOT / "results/012_reader_concordance"


def main():
    base = {}
    for l in open(ROOT / "outputs/012_reader_concordance/stageB/items.jsonl"):
        r = json.loads(l)
        if r.get("status") == "evaluated" and r["clean_correct"] and not r["clean_contains_y_b"]:
            J = r["arms"]["J_L24-35"]
            dh = [v["dh"] / v["h"] for v in J["diag"].values() if v.get("h")]
            base[r["name"]] = {"rel": rel(r["name"]), "read_token": r["read_token"], "dir": r["edit_direction"]["L24-35"],
                               "succ_all": bool(J["success"]), "dm_all": J["dm"], "size": float(np.mean(dh))}
    f1 = {}
    for l in open(F / "f1_items.jsonl"):
        r = json.loads(l)
        if r["name"] not in base:
            continue
        arms = r["arms"]
        f1[r["name"]] = {a: {"success": bool(v["success"]), "dm": v.get("dm"),
                             "mb": (v.get("model_out") or {}).get("d_model_bridge"), "ma": (v.get("model_out") or {}).get("d_model_answer"),
                             "wn": v.get("write_norm_other_pos_max")}
                         for a, v in arms.items() if "success" in v}
    out = {"label": "EXPLORATORY (012 Stage F; rules pre-registered in spec §13)", "n": len(f1), "F1": {}, "F2": {}}
    n = len(f1)
    for a in ("J_L24-35", "rand_L24-35", "J_allbutread_L24-35", "rand_allbutread_L24-35", "J_readonly_L24-35", "rand_readonly_L24-35"):
        xs = [f1[k][a] for k in f1 if a in f1[k]]
        out["F1"].setdefault("arms", {})[a] = {"flips": sum(x["success"] for x in xs), "n": len(xs),
                                               "dm_median": float(np.median([x["dm"] for x in xs if x["dm"] is not None]))}
    ll = defaultdict(dict)
    for f in ("f1_ll_oracle_lenses.jsonl", "f1_ll_nla.jsonl"):
        for l in open(F / f):
            r = json.loads(l)
            if r["item"] in f1 and r["state"] in ("J_allbutread_L24-35", "J_readonly_L24-35") and r.get("d") not in (None, "None"):
                ll[(r["reader"], str(r["layer"]), r["state"], r["kind"])][r["item"]] = float(r["d"])
    cells = {}
    for (reader, layer, state, kind), per in sorted(ll.items()):
        if kind != "bridge":
            continue
        ans = ll.get((reader, layer, state, "answer"), {})
        rows = [{"rel": base[k]["rel"], "d": v, "da": ans.get(k), "s": f1[k][state]["success"], "dm": f1[k][state]["dm"],
                 "mb": f1[k][state]["mb"], "ma": f1[k][state]["ma"], "wn": f1[k][state]["wn"], "dir": base[k]["dir"]}
                for k, v in per.items()]
        rows_ok = [x for x in rows if None not in (x["mb"], x["ma"], x["wn"])]
        pf = lambda xs: partial([x["d"] for x in xs], [x["dm"] for x in xs], [[x["mb"] for x in xs], [x["ma"] for x in xs], [x["wn"] for x in xs]])
        tt = [x for x in rows if x["dir"] == "toward_tgt"]
        cells[f"{reader}|L{layer}|{state}"] = {
            "n": len(rows), "n_success": sum(x["s"] for x in rows), "d_median": float(np.median([x["d"] for x in rows])),
            "auc_bridge": auc([x["d"] for x in rows], [x["s"] for x in rows]),
            "auc_bridge_ci95": cluster_boot(lambda xs: auc([x["d"] for x in xs], [x["s"] for x in xs]), rows),
            "auc_answer": auc([x["da"] for x in rows if x["da"] is not None], [x["s"] for x in rows if x["da"] is not None]),
            "auc_bridge_toward_tgt": auc([x["d"] for x in tt], [x["s"] for x in tt]),
            "partial_bridge_dm_given_model_output_writenorm": pf(rows_ok) if state.startswith("J_allbut") else None,
            "partial_ci95": cluster_boot(pf, rows_ok) if state.startswith("J_allbut") else None}
    out["F1"]["readers"] = cells
    mo = [f1[k]["J_allbutread_L24-35"] for k in f1]
    out["F1"]["model_output_auc_allbutread"] = {"bridge": auc([x["mb"] for x in mo if x["mb"] is not None], [x["success"] for x in mo if x["mb"] is not None]),
                                               "answer": auc([x["ma"] for x in mo if x["ma"] is not None], [x["success"] for x in mo if x["ma"] is not None])}
    # F2
    # ONE shared (item, pos) mask for readers and the model covariate (fix 2026-10-07, Codex audit): the reader rows
    # carry no is_read_minus_1 flag, so the exclusion set is built from the model rows and applied to both by (item, pos)
    excl = set()
    for l in open(F / "f2_model.jsonl"):
        r = json.loads(l)
        it = r.get("item", r.get("name"))
        if it in base and r.get("is_read_minus_1") in (True, "True") and base[it]["read_token"].strip() == "is":
            excl.add((it, int(r["pos"])))
    keep = lambda r: r["item"] in base and (r["item"], int(r["pos"])) not in excl
    mod = defaultdict(lambda: {"mb": [], "ma": []})
    for l in open(F / "f2_model.jsonl"):
        r = json.loads(l)
        if not keep({**r, "item": r.get("item", r.get("name"))}):
            continue
        it = r.get("item", r.get("name"))
        J = r["J_L24-35"] if "J_L24-35" in r else r.get("arms", {}).get("J_L24-35", {})
        if J.get("d_model_bridge_p") is not None:
            mod[it]["mb"].append(J["d_model_bridge_p"])
        if J.get("d_model_answer_p") is not None:
            mod[it]["ma"].append(J["d_model_answer_p"])
    rd = defaultdict(lambda: defaultdict(list))
    prof = defaultdict(lambda: defaultdict(list))
    for f in ("f2_ll_oracle_lenses.jsonl", "f2_ll_nla.jsonl"):
        for l in open(F / f):
            r = json.loads(l)
            if r["state"] != "J_L24-35" or not keep(r) or r.get("d_r") in (None, "None"):
                continue
            key = f"{r['reader']}|L{r['layer']}"
            rd[key][r["item"]].append(float(r["d_r"]))
            if base[r["item"]]["dir"] == "toward_tgt":
                prof[key][(int(r["pos"]) - int(r.get("n_prompt", 0)), base[r["item"]]["succ_all"])].append(float(r["d_r"]))
    for key, per in sorted(rd.items()):
        rows = [{"rel": base[k]["rel"], "d": float(np.mean(v)), "s": base[k]["succ_all"], "dm": base[k]["dm_all"], "size": base[k]["size"],
                 "mb": float(np.mean(mod[k]["mb"])) if mod[k]["mb"] else None, "ma": float(np.mean(mod[k]["ma"])) if mod[k]["ma"] else None}
                for k, v in per.items() if base[k]["dir"] == "toward_tgt"]
        ok = [x for x in rows if x["mb"] is not None and x["ma"] is not None]
        pf = lambda xs: partial([x["d"] for x in xs], [x["dm"] for x in xs], [[x["mb"] for x in xs], [x["ma"] for x in xs], [x["size"] for x in xs]])
        a = auc([x["d"] for x in rows], [x["s"] for x in rows])
        pc = cluster_boot(pf, ok)
        verdict = ("bridge content beyond output-facing" if a is not None and a >= 0.75 and pc and pc[0] > 0 else
                   "not outcome-specific" if a is not None and a < 0.65 else "intermediate")
        out["F2"][key] = {"n_toward_tgt": len(rows), "d_early_median_success": float(np.median([x["d"] for x in rows if x["s"]])),
                          "d_early_median_fail": float(np.median([x["d"] for x in rows if not x["s"]])),
                          "auc": a, "auc_ci95": cluster_boot(lambda xs: auc([x["d"] for x in xs], [x["s"] for x in xs]), rows),
                          "partial_given_model_early_output_size": pf(ok), "partial_ci95": pc, "verdict_s13": verdict,
                          "profile_by_offset": {f"{o}|{'works' if s else 'fails'}": float(np.median(v)) for (o, s), v in sorted(prof[key].items()) if len(v) >= 5}}
    tt_items = [k for k in base if base[k]["dir"] == "toward_tgt" and mod[k]["mb"]]
    out["F2"]["model_early_output_auc"] = {"bridge": auc([np.mean(mod[k]["mb"]) for k in tt_items], [base[k]["succ_all"] for k in tt_items]),
                                          "answer": auc([np.mean(mod[k]["ma"]) for k in tt_items if mod[k]["ma"]], [base[k]["succ_all"] for k in tt_items if mod[k]["ma"]])}
    (RES / "analysis/stageF_analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["F1"]["arms"]), json.dumps(out["F1"]["model_output_auc_allbutread"]), json.dumps(out["F2"]["model_early_output_auc"]), sep="\n")
    r2 = lambda x: None if x is None else round(x, 2)
    for k, v in out["F1"]["readers"].items():
        print("F1", f"{k:36s}", "n", v["n"], "med d", r2(v["d_median"]), "AUCb", r2(v["auc_bridge"]), v["auc_bridge_ci95"] and [r2(c) for c in v["auc_bridge_ci95"]],
              "AUCa", r2(v["auc_answer"]), "pS", r2(v["partial_bridge_dm_given_model_output_writenorm"]), v["partial_ci95"] and [r2(c) for c in v["partial_ci95"]])
    for k, v in out["F2"].items():
        if k != "model_early_output_auc":
            print("F2", f"{k:12s}", "n", v["n_toward_tgt"], "d succ/fail", r2(v["d_early_median_success"]), r2(v["d_early_median_fail"]),
                  "AUC", r2(v["auc"]), [r2(c) for c in (v["auc_ci95"] or [])], "pS", r2(v["partial_given_model_early_output_size"]),
                  [r2(c) for c in (v["partial_ci95"] or [])], "->", v["verdict_s13"])


if __name__ == "__main__":
    main()
