"""011 analysis (POST HOC, exploratory): per-condition medians and the paired contrasts of amendment 3, from the per-pair
rows of a swap run.

  python olens_swaps_analysis.py --run-id s1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "008_trackb"))
sys.path.insert(0, str(HERE.parent / "common"))
import ar_vectors as AV                            # noqa: E402
import c008 as C8                                  # noqa: E402
import stats as ST                                 # noqa: E402

POWER = 0.15
EQUIV = 0.05


def boot(x, tables):
    r = ST.cluster_bootstrap(np.asarray(x, float), np.asarray(tables), stat=np.median, B=10000, seed=C8.BOOT_SEED)
    return {"median": float(np.median(x)), "ci": [float(r["ci"][0]), float(r["ci"][1])], "n": len(x),
            "pos": int((np.asarray(x) > 0).sum()), "neg": int((np.asarray(x) < 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="s1")
    a = ap.parse_args()
    rows = [json.loads(l) for f in sorted((AV.OUT / a.run_id / "pairs").glob("*.jsonl")) for l in open(f)]
    by = {}
    for r in rows:
        if r.get("arm") and r["arm"] != "G0" and not r.get("quarantined"):
            by.setdefault(r["pair_id"], {})[r["name"]] = r
    pids = sorted(by)
    tab = [by[p]["Bref_full_3L"]["table"] for p in pids]

    def col(name):
        return np.array([by[p][name]["S"] for p in pids])

    names = sorted({n for p in pids for n in by[p]})
    out = {"run_id": a.run_id, "label": "POST HOC, exploratory (011 amendments 2-3)", "n_pairs": len(pids),
           "conditions": {n: {**boot(col(n), tab), "flip_rate": float(np.mean([by[p][n]["flip"] for p in pids])),
                              "edit_norm_sum_median": float(np.median([by[p][n]["edit_norm_sum_base"] for p in pids]))}
                          for n in names}}
    rawrand = (col("C_raw_rand_0") + col("C_raw_rand_1")) / 2
    drand = (col("D_rand_0") + col("D_rand_1")) / 2
    out["contrasts"] = {
        "C_rawcoord - C_raw_rand": boot(col("C_rawcoord") - rawrand, tab),
        "D_ar - D_classmean": boot(col("D_ar") - col("D_classmean"), tab),
        "D_ar - D_rand": boot(col("D_ar") - drand, tab),
        "D_classmean - D_rand": boot(col("D_classmean") - drand, tab),
        "C_white - Crand": boot(col("C_white") - np.mean([col(f"Crand_{i}") for i in range(4)], 0), tab),
    }
    c, k = out["conditions"], out["contrasts"]
    v = {}
    v["C_white"] = ("not read: answer-colour control below power" if c["Cctrl_answer"]["median"] < POWER
                    else f"read: median S {c['C_white']['median']:.3f}")
    if c["C_raw_answer"]["median"] < POWER:
        v["C_rawcoord"] = "not read: raw answer-colour control below power"
    else:
        v["C_rawcoord"] = ("colour-specific (paired CI vs matched random above 0)" if k["C_rawcoord - C_raw_rand"]["ci"][0] > 0
                           else "not colour-specific (paired CI vs matched random includes 0 or is below)")
    if c["D_classmean"]["median"] < POWER:
        v["additive"] = "not read: class-mean additive lever below power at this norm"
    else:
        lo, hi = k["D_ar - D_classmean"]["ci"]
        if -EQUIV <= lo and hi <= EQUIV:
            v["additive"] = "AR colour difference as good a handle as the class-mean difference"
        elif k["D_ar - D_rand"]["ci"][0] <= 0 <= k["D_ar - D_rand"]["ci"][1]:
            v["additive"] = "AR colour difference not a handle (indistinguishable from random at matched norm)"
        else:
            v["additive"] = "partial: AR difference beats random but not equivalent to the class-mean difference"
    out["verdicts"] = v
    res = AV.RES / f"olens_swaps_analysis_{a.run_id}.json"
    res.write_text(json.dumps(out, indent=1))
    for n in names:
        x = c[n]
        print(f"{n:14s} S {x['median']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]  flips {x['flip_rate']:.2f}  "
              f"+{x['pos']}/-{x['neg']}  edit {x['edit_norm_sum_median']:.1f}")
    for n, x in k.items():
        print(f"{n:24s} {x['median']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]  +{x['pos']}/-{x['neg']}")
    print(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
