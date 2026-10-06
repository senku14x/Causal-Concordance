"""012 Stage D (CPU, EXPLORATORY): do readers' src->tgt shifts track the model's demonstrated answer change, beyond the
edit's direction and size?

Primary cohort: clean-correct items whose clean continuation lacks y_b (spec §12). For each band and reader x layer:
  1. d_r after the J swap, success vs fail, WITHIN edit_direction == toward_tgt (the swap's reflection moves the
     state toward tgt). Success and direction are confounded otherwise: toward_src edits push readers toward src.
  2. Spearman(d_r, Δm) and the partial Spearman controlling for edit size (mean ||Δh||/||h|| at the read position over
     band layers), all items and toward_tgt only.
  3. Scale-free reader comparison: AUC of d_r for predicting success within toward_tgt; Oracle vs NLA paired
     bootstrap (clusters = relation keys).
  4. Sensitivity: Japan-target items excluded.
Naming: src/tgt naming in sampled readouts, clean vs J, success vs fail (toward_tgt).
d_r is in each reader's own nats: never compare raw magnitudes across readers.

  python stageD_analysis.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "outputs/012_reader_concordance/stageB/items.jsonl"
C = ROOT / "outputs/012_reader_concordance/stageC"
RES = ROOT / "results/012_reader_concordance"
SEED, NB = 20261007, 2000


def rel(name):
    return name.split("-")[-1]


def rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    for v in np.unique(x):                       # average ties
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    if len(a) < 4:
        return None
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def partial_spearman(a, b, z):
    if len(a) < 5:
        return None
    ra, rb, rz = rank(a), rank(b), rank(z)
    X = np.c_[np.ones(len(rz)), rz]
    res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.corrcoef(res(ra), res(rb))[0, 1])


def auc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    r = rank(s)
    return float((r[y].sum() - y.sum() * (y.sum() - 1) / 2 - y.sum() * 0) / (y.sum() * (~y).sum()) - 0.5 / (~y).sum() * 0
                 if False else (r[y].sum() - y.sum() * (y.sum() - 1) / 2) / (y.sum() * (~y).sum()))


def cluster_boot(f, items, nb=NB, seed=SEED):
    """Percentile CI of statistic f(list_of_items) resampling relation clusters."""
    cl = defaultdict(list)
    for it in items:
        cl[it["rel"]].append(it)
    keys = sorted(cl)
    rng = random.Random(seed)
    vals = []
    for _ in range(nb):
        smp = [x for k in (rng.choice(keys) for _ in keys) for x in cl[k]]
        v = f(smp)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) > 50 else None


def main():
    items = {}
    for l in open(B):
        r = json.loads(l)
        if r.get("status") != "evaluated" or not r["clean_correct"] or r["clean_contains_y_b"]:
            continue
        rec = {"name": r["name"], "rel": rel(r["name"]), "tgt": r["tgt"], "src": r["src"], "japan": r["tgt"] == "Japan"}
        for b in ("L24-35", "L24-47"):
            J = r["arms"][f"J_{b}"]
            dh = [v["dh"] / v["h"] for v in J["diag"].values() if v.get("h")]
            rec[b] = {"success": bool(J["success"]), "dm": J["dm"], "dm_rand": r["arms"][f"rand_{b}"]["dm"],
                      "direction": r["edit_direction"][b], "edit_size": float(np.mean(dh)) if dh else None,
                      "contrast_read": r["swap_direction"][b].get("mean_cs_minus_ct_read")}
        items[r["name"]] = rec
    d = defaultdict(dict)                                   # (reader, layer, band) -> item -> {state: d_r}
    for f in ("ll_oracle_lenses.jsonl", "ll_nla.jsonl"):
        for l in open(C / f):
            r = json.loads(l)
            if r["item"] in items and r.get("d_r") is not None and r["state"] != "clean":
                key = (r["reader"] + ("_offtrain" if r.get("off_training") in (True, "True") else ""), str(r["layer"]), r["band"])
                d[key].setdefault(r["item"], {})[r["state"].split("_")[0]] = float(r["d_r"])
    out = {"label": "EXPLORATORY (012 Stage D); POST HOC analyses after seeing the Stage B/C summary: the direction "
                    "stratification and edit-size control were chosen because the smoke showed the swap direction issue",
           "n_primary": len(items), "cells": {}}
    for (reader, layer, band), per in sorted(d.items()):
        rows = [dict(items[n][band], name=n, rel=items[n]["rel"], japan=items[n]["japan"], dJ=v.get("J"), dR=v.get("rand"))
                for n, v in per.items() if v.get("J") is not None]
        tt = [x for x in rows if x["direction"] == "toward_tgt"]
        med = lambda xs, k: float(np.median([x[k] for x in xs])) if xs else None
        succ = [x for x in tt if x["success"]]
        fail = [x for x in tt if not x["success"]]
        diff = lambda xs: (np.median([x["dJ"] for x in xs if x["success"]]) - np.median([x["dJ"] for x in xs if not x["success"]])
                           if any(x["success"] for x in xs) and any(not x["success"] for x in xs) else None)
        a = lambda xs: auc([x["dJ"] for x in xs], [x["success"] for x in xs])
        out["cells"][f"{reader}|L{layer}|{band}"] = {
            "n": len(rows), "n_toward_tgt": len(tt), "n_tt_success": len(succ), "n_tt_fail": len(fail),
            "dJ_median_all": med(rows, "dJ"), "dRand_median_all": med([x for x in rows if x["dR"] is not None], "dR"),
            "toward_src_dJ_median": med([x for x in rows if x["direction"] == "toward_src"], "dJ"),
            "tt_success_dJ_median": med(succ, "dJ"), "tt_fail_dJ_median": med(fail, "dJ"),
            "tt_success_minus_fail": diff(tt), "tt_success_minus_fail_ci95": cluster_boot(diff, tt),
            "auc_tt": a(tt), "auc_tt_ci95": cluster_boot(a, tt),
            "auc_tt_no_japan": a([x for x in tt if not x["japan"]]),
            "spearman_dJ_dm_all": spearman([x["dJ"] for x in rows], [x["dm"] for x in rows]),
            "partial_spearman_dJ_dm_given_size_all": partial_spearman([x["dJ"] for x in rows], [x["dm"] for x in rows],
                                                                      [x["edit_size"] for x in rows]),
            "spearman_dJ_dm_tt": spearman([x["dJ"] for x in tt], [x["dm"] for x in tt]),
            "partial_spearman_dJ_dm_given_size_tt": partial_spearman([x["dJ"] for x in tt], [x["dm"] for x in tt],
                                                                     [x["edit_size"] for x in tt]),
            "spearman_size_dm_tt": spearman([x["edit_size"] for x in tt], [x["dm"] for x in tt]),
            "points": [{k: x[k] for k in ("name", "dJ", "dR", "dm", "success", "direction", "edit_size", "japan")} for x in rows]}
    # Oracle vs NLA, same items, same band (L24-35), scale-free AUC difference with paired cluster bootstrap
    cmp = {}
    for o_layer in ("40", "44"):
        ko, kn = ("oracle", o_layer, "L24-35"), ("nla", "42", "L24-35")
        common = [n for n in d[ko] if n in d[kn] and "J" in d[ko][n] and "J" in d[kn][n]
                  and items[n]["L24-35"]["direction"] == "toward_tgt"]
        rows = [{"rel": items[n]["rel"], "o": d[ko][n]["J"], "n_": d[kn][n]["J"], "s": items[n]["L24-35"]["success"]} for n in common]
        f = lambda xs: (auc([x["o"] for x in xs], [x["s"] for x in xs]) or np.nan) - (auc([x["n_"] for x in xs], [x["s"] for x in xs]) or np.nan)
        cmp[f"oracle_L{o_layer}_minus_nla_L42"] = {"n": len(rows), "auc_diff": f(rows), "ci95": cluster_boot(f, rows)}
    out["oracle_vs_nla_auc"] = cmp
    # naming
    nam = defaultdict(lambda: defaultdict(int))
    for f in ("readouts_oracle.jsonl", "readouts_nla.jsonl"):
        for l in open(C / f):
            r = json.loads(l)
            if r["item"] not in items:
                continue
            band = r["band"]
            it = items[r["item"]][band]
            grp = "clean" if r["state"] == "clean" else (r["state"].split("_")[0] + ("|tt_success" if it["direction"] == "toward_tgt" and it["success"]
                                                                                    else "|tt_fail" if it["direction"] == "toward_tgt" else "|toward_src"))
            key = f"{r['reader']}{'_offtrain' if r.get('off_training') in (True, 'True') else ''}|L{r['layer']}|{band}|{grp}"
            for c in r["category"]:
                nam[key][c] += 1
                nam[key]["_n"] += 1
    out["naming"] = {k: dict(v) for k, v in sorted(nam.items())}
    (RES / "analysis").mkdir(parents=True, exist_ok=True)
    (RES / "analysis/stageD_analysis.json").write_text(json.dumps(out, indent=1))
    # qualitative: seeded readouts, clean vs J, for toward_tgt success and fail items
    rng = random.Random(SEED)
    pick = {k: rng.sample(sorted(n for n, v in items.items() if v["L24-35"]["direction"] == "toward_tgt" and v["L24-35"]["success"] == s),
                          k=6) for k, s in (("success", True), ("fail", False))}
    keep = []
    for f in ("readouts_oracle.jsonl", "readouts_nla.jsonl"):
        for l in open(C / f):
            r = json.loads(l)
            for k, ns in pick.items():
                if r["item"] in ns and r["band"] == "L24-35" and r["state"] in ("clean", "J_L24-35"):
                    keep.append({"group": f"toward_tgt_{k}", **{x: r[x] for x in ("item", "reader", "layer", "state", "src", "tgt", "samples", "category", "answer_category")}})
    (RES / "qualitative").mkdir(parents=True, exist_ok=True)
    (RES / "qualitative/stageC_readouts_samples.jsonl").write_text("".join(json.dumps(x) + "\n" for x in keep))
    for k, v in out["cells"].items():
        print(f"{k:28s} n={v['n']:2d} tt={v['n_toward_tgt']:2d} (s{v['n_tt_success']}/f{v['n_tt_fail']})  "
              f"tt succ-fail {v['tt_success_minus_fail'] and round(v['tt_success_minus_fail'], 2)} {v['tt_success_minus_fail_ci95'] and [round(c, 2) for c in v['tt_success_minus_fail_ci95']]}  "
              f"AUC {v['auc_tt'] and round(v['auc_tt'], 2)}  pS|size {v['partial_spearman_dJ_dm_given_size_tt'] and round(v['partial_spearman_dJ_dm_given_size_tt'], 2)}")
    print(json.dumps(cmp, indent=0))


if __name__ == "__main__":
    main()
