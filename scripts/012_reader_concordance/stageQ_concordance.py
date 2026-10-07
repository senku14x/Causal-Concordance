"""012 Stage Q (CPU, EXPLORATORY, POST HOC): reader agreement and ensembles on the Stage C states (J swap L24-35).

For the PI's question (spec §1, Q1/Q2/Q4). Cohort: clean-correct, toward_tgt items (the swap pushed toward tgt).
  - Q1 agreement: pairwise Spearman of readers' bridge d_r across items; Cohen's kappa of "names tgt in >= 1 of 2
    samples" between the Oracle and the NLA (sampled readers only).
  - Q2 coverage: per reader, working swaps it misses (d_r <= 0) and failed swaps where it shifts strongly
    (d_r above the median d_r of working swaps); naming-based misses for Oracle/NLA.
  - Q4 ensembles vs the label "the model's answer changed": AUC of each reader, of a family-balanced rank-average
    ensemble (Oracle, NLA, J-lens, logit families weighted equally), of leave-one-family-out ensembles, and of the
    model's own read-cell output with and without the readers added.
d_r is in each reader's own nats, so ensembles average within-reader ranks, never raw nats.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stageD_analysis import B, C, RES, auc, cluster_boot, rank, rel

ROOT = Path(__file__).resolve().parents[2]
READERS = {"Oracle L40": ("oracle", "40"), "Oracle L44": ("oracle", "44"), "NLA L42": ("nla", "42"),
           "J_NP L40": ("J_NP", "40"), "J_CB L40": ("J_CB", "40"), "logit L40": ("logit", "40")}
FAMILY = {"Oracle L40": "oracle", "Oracle L44": "oracle", "NLA L42": "nla", "J_NP L40": "jlens", "J_CB L40": "jlens", "logit L40": "logit"}


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else None


def main():
    items = {}
    for l in open(B):
        r = json.loads(l)
        if r.get("status") == "evaluated" and r["clean_correct"] and not r["clean_contains_y_b"] and r["edit_direction"]["L24-35"] == "toward_tgt":
            items[r["name"]] = {"rel": rel(r["name"]), "s": bool(r["arms"]["J_L24-35"]["success"])}
    d = defaultdict(dict)
    for f in ("ll_oracle_lenses.jsonl", "ll_nla.jsonl"):
        for l in open(C / f):
            r = json.loads(l)
            if r["item"] in items and r["state"] == "J_L24-35" and r.get("off_training") not in (True, "True"):
                for name, (rd, lay) in READERS.items():
                    if r["reader"] == rd and str(r["layer"]) == lay:
                        d[name][r["item"]] = float(r["d_r"])
    mo = {}
    for l in open(ROOT / "outputs/012_reader_concordance/stageB/model_output.jsonl"):
        r = json.loads(l)
        a = r["arms"] if isinstance(r["arms"], dict) else json.loads(r["arms"])
        mo[r["name"]] = a.get("J_L24-35", {}).get("d_model_bridge")
    names = [n for n in items if all(n in d[k] for k in READERS) and mo.get(n) is not None]
    y = [items[n]["s"] for n in names]
    out = {"label": "EXPLORATORY, POST HOC (012 Stage Q)", "n": len(names), "n_success": sum(y)}
    # Q1
    out["spearman_matrix"] = {a: {b: float(np.corrcoef(rank([d[a][n] for n in names]), rank([d[b][n] for n in names]))[0, 1])
                                  for b in READERS} for a in READERS}
    tgt_named = defaultdict(dict)
    for f, rd, lay in (("readouts_oracle.jsonl", "oracle", "40"), ("readouts_oracle.jsonl", "oracle", "44"), ("readouts_nla.jsonl", "nla", "42")):
        for l in open(C / f):
            r = json.loads(l)
            if r["item"] in items and r["state"] == "J_L24-35" and str(r["layer"]) == lay and r.get("off_training") not in (True, "True"):
                tgt_named[f"{rd} L{lay}"][r["item"]] = any(c in ("tgt-only", "both") for c in r["category"])
    common = [n for n in names if all(n in tgt_named[k] for k in tgt_named)]
    out["kappa_tgt_named"] = {f"{a} vs {b}": kappa([tgt_named[a][n] for n in common], [tgt_named[b][n] for n in common])
                              for a, b in (("oracle L40", "nla L42"), ("oracle L44", "nla L42"), ("oracle L40", "oracle L44"))}
    # Q2
    cov = {}
    for k in READERS:
        succ = [d[k][n] for n in names if items[n]["s"]]
        thr = float(np.median(succ))
        cov[k] = {"misses_working_dr_le_0": sum(1 for n in names if items[n]["s"] and d[k][n] <= 0),
                  "strong_shift_on_failed": sum(1 for n in names if not items[n]["s"] and d[k][n] > thr)}
    for k, v in tgt_named.items():
        cov[k + " (naming)"] = {"working_swaps_not_naming_tgt": sum(1 for n in common if items[n]["s"] and not v[n]),
                                "failed_swaps_naming_tgt": sum(1 for n in common if not items[n]["s"] and v[n])}
    only = defaultdict(int)
    for n in common:
        if items[n]["s"]:
            who = [k for k in tgt_named if tgt_named[k][n]]
            if len({w.split()[0] for w in who}) == 1:
                only[who[0].split()[0] + "_only"] += 1
            elif not who:
                only["none"] += 1
    out["coverage"] = cov
    out["working_swaps_tgt_named_by_only_one_family"] = dict(only)
    out["working_swaps_tgt_named_by_only_one_family_note"] = (
        "pooled: Oracle has 2 layers x 2 samples vs NLA 1 layer x 2 samples (unequal opportunity); use "
        "naming_equal_opportunity (added after Codex audit 12, 2026-10-07)")
    eq = {}
    for o in ("oracle L40", "oracle L44"):
        w = [n for n in common if items[n]["s"]]
        eq[f"{o} vs nla L42"] = {"n_working": len(w),
                                 "oracle_only": sum(1 for n in w if tgt_named[o][n] and not tgt_named["nla L42"][n]),
                                 "nla_only": sum(1 for n in w if tgt_named["nla L42"][n] and not tgt_named[o][n]),
                                 "both": sum(1 for n in w if tgt_named[o][n] and tgt_named["nla L42"][n]),
                                 "neither": sum(1 for n in w if not tgt_named[o][n] and not tgt_named["nla L42"][n])}
    out["naming_equal_opportunity"] = eq
    # Q4
    R = {k: rank([d[k][n] for n in names]) / len(names) for k in READERS}
    fam = defaultdict(list)
    for k in READERS:
        fam[FAMILY[k]].append(R[k])
    famr = {f: np.mean(v, 0) for f, v in fam.items()}
    ens = lambda fs: np.mean([famr[f] for f in fs], 0)
    rows = [{"rel": items[n]["rel"], "i": i} for i, n in enumerate(names)]
    boot = lambda score: cluster_boot(lambda xs: auc([score[x["i"]] for x in xs], [y[x["i"]] for x in xs]), rows)
    mor = rank([mo[n] for n in names]) / len(names)
    out["auc"] = {k: auc(R[k], y) for k in READERS}
    out["auc"]["family_ensemble"] = auc(ens(list(famr)), y)
    out["auc"]["family_ensemble_ci95"] = boot(ens(list(famr)))
    for f in famr:
        out["auc"][f"ensemble_without_{f}"] = auc(ens([g for g in famr if g != f]), y)
    out["auc"]["model_output"] = auc(mor, y)
    out["auc"]["model_output_plus_family_ensemble"] = auc((mor + ens(list(famr))) / 2, y)
    best = max(READERS, key=lambda k: out["auc"][k])
    out["auc"]["best_single"] = best
    diff = lambda xs: auc([ens(list(famr))[x["i"]] for x in xs], [y[x["i"]] for x in xs]) - auc([R[best][x["i"]] for x in xs], [y[x["i"]] for x in xs])
    out["auc"]["ensemble_minus_best_ci95"] = cluster_boot(diff, rows)
    # paired increments over the model's own output (added after Codex audit 12, 2026-10-07)
    pair = lambda a, b: (lambda xs: auc([a[x["i"]] for x in xs], [y[x["i"]] for x in xs]) - auc([b[x["i"]] for x in xs], [y[x["i"]] for x in xs]))
    out["auc"]["model_output_plus_family_ensemble_minus_model_output"] = out["auc"]["model_output_plus_family_ensemble"] - out["auc"]["model_output"]
    out["auc"]["model_output_plus_family_ensemble_minus_model_output_ci95"] = cluster_boot(pair((mor + ens(list(famr))) / 2, mor), rows)
    for k in ("Oracle L44", "NLA L42"):
        out["auc"][f"{k}_minus_model_output_ci95"] = cluster_boot(pair(R[k], mor), rows)
    (RES / "analysis/stageQ_concordance.json").write_text(json.dumps(out, indent=1))
    r2 = lambda v: v if not isinstance(v, float) else round(v, 3)
    print("n", out["n"], "success", out["n_success"])
    print("kappa", {k: r2(v) for k, v in out["kappa_tgt_named"].items()})
    print("AUC", {k: (r2(v) if not isinstance(v, list) else [r2(c) for c in v]) for k, v in out["auc"].items()})
    print("coverage", cov, dict(only))
    print("spearman Oracle L44 vs NLA", r2(out["spearman_matrix"]["Oracle L44"]["NLA L42"]), "J_NP vs NLA", r2(out["spearman_matrix"]["J_NP L40"]["NLA L42"]),
          "Oracle vs J_NP", r2(out["spearman_matrix"]["Oracle L44"]["J_NP L40"]), "logit vs Oracle", r2(out["spearman_matrix"]["logit L40"]["Oracle L44"]))
    off = [v for a, row in out["spearman_matrix"].items() for b, v in row.items() if FAMILY[a] != FAMILY[b]]
    print("spearman across families: min", r2(min(off)), "max", r2(max(off)), "(full matrix in the JSON; quote ranges from it, not from this line)")


if __name__ == "__main__":
    main()
