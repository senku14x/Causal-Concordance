"""012 Stage R analysis (CPU, EXPLORATORY; rule pre-registered in spec §16): does the NLA's early-position signal
(Stage F2) replicate on fresh broadcast items? Same F2 statistics; clusters = source concept (repeated across functions)."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stageD_analysis import auc, cluster_boot
from stageD_answers import partial

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs/012_reader_concordance/stageR"
RES = ROOT / "results/012_reader_concordance"


def main():
    base = {}
    for l in open(D / "r_items.jsonl"):
        r = json.loads(l)
        if r.get("status") != "evaluated" or not r["clean_correct"] or r.get("clean_contains_y_b"):
            continue
        J = r["arms"]["J_L24-35"]
        diag = J.get("diag") or r.get("diag") or {}
        dh = [v["dh"] / v["h"] for v in diag.values() if isinstance(v, dict) and v.get("h")]
        ed = r["edit_direction"]["L24-35"] if isinstance(r["edit_direction"], dict) else r["edit_direction"]
        base[r["name"]] = {"rel": r["src"], "read_token": r.get("read_token", ""), "dir": ed, "s": J["success"] in (True, "True"),
                           "dm": float(J["dm"]), "size": float(np.mean(dh)) if dh else 0.0}
    keep = lambda r: r["item"] in base and not (r.get("is_read_minus_1") in (True, "True") and base[r["item"]]["read_token"].strip() == "is")
    mod = defaultdict(lambda: {"mb": [], "ma": []})
    for l in open(D / "f2_model.jsonl"):
        r = json.loads(l)
        r["item"] = r.get("item", r.get("name"))
        if not keep(r):
            continue
        J = r.get("J_L24-35", {})
        if J.get("d_model_bridge_p") is not None:
            mod[r["item"]]["mb"].append(J["d_model_bridge_p"])
        if J.get("d_model_answer_p") is not None:
            mod[r["item"]]["ma"].append(J["d_model_answer_p"])
    rd = defaultdict(lambda: defaultdict(list))
    for f in ("f2_ll_oracle_lenses.jsonl", "f2_ll_nla.jsonl"):
        for l in open(D / f):
            r = json.loads(l)
            if r["state"] == "J_L24-35" and keep(r) and r.get("d_r") not in (None, "None"):
                rd[f"{r['reader']}|L{r['layer']}"][r["item"]].append(float(r["d_r"]))
    out = {"label": "EXPLORATORY (012 Stage R; rule in spec §16)", "n_primary": len(base),
           "n_toward_tgt": sum(v["dir"] == "toward_tgt" for v in base.values()),
           "n_tt_success": sum(v["dir"] == "toward_tgt" and v["s"] for v in base.values()), "readers": {}}
    tt_items = [k for k in base if base[k]["dir"] == "toward_tgt" and mod[k]["mb"] and mod[k]["ma"]]
    out["model_early_output_auc"] = {"bridge": auc([np.mean(mod[k]["mb"]) for k in tt_items], [base[k]["s"] for k in tt_items]),
                                     "answer": auc([np.mean(mod[k]["ma"]) for k in tt_items], [base[k]["s"] for k in tt_items])}
    for key, per in sorted(rd.items()):
        rows = [{"rel": base[k]["rel"], "d": float(np.mean(v)), "s": base[k]["s"], "dm": base[k]["dm"], "size": base[k]["size"],
                 "mb": float(np.mean(mod[k]["mb"])), "ma": float(np.mean(mod[k]["ma"]))}
                for k, v in per.items() if k in tt_items]
        pf = lambda xs: partial([x["d"] for x in xs], [x["dm"] for x in xs], [[x["mb"] for x in xs], [x["ma"] for x in xs], [x["size"] for x in xs]])
        a = auc([x["d"] for x in rows], [x["s"] for x in rows])
        pc = cluster_boot(pf, rows)
        rep = a is not None and a >= 0.65 and pc is not None and pc[0] > 0
        out["readers"][key] = {"n": len(rows), "auc": a, "auc_ci95": cluster_boot(lambda xs: auc([x["d"] for x in xs], [x["s"] for x in xs]), rows),
                               "partial": pf(rows), "partial_ci95": pc, "passes_s16_test": rep}
    nla = out["readers"].get("nla|L42", {})
    others = [k for k in out["readers"] if not k.startswith("nla")]
    out["verdict_s16"] = ("replicates, NLA-specific" if nla.get("passes_s16_test") and not any(out["readers"][k]["passes_s16_test"] for k in others)
                          else "replicates, not NLA-specific" if nla.get("passes_s16_test") else "does not replicate")
    (RES / "analysis/stageR_analysis.json").write_text(json.dumps(out, indent=1))
    r2 = lambda x: None if x is None else round(x, 2)
    print({k: out[k] for k in ("n_primary", "n_toward_tgt", "n_tt_success", "verdict_s16")}, "model early AUC", {k: r2(v) for k, v in out["model_early_output_auc"].items()})
    for k, v in out["readers"].items():
        print(f"{k:12s} n={v['n']} AUC {r2(v['auc'])} {[r2(c) for c in v['auc_ci95'] or []]} partial {r2(v['partial'])} {[r2(c) for c in v['partial_ci95'] or []]} pass={v['passes_s16_test']}")


if __name__ == "__main__":
    main()
