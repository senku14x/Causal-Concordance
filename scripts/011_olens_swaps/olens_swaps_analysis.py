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
    ap.add_argument("--expect-pairs", type=int, default=82, help="gated dev pairs (008's set)")
    ap.add_argument("--allow-partial", action="store_true", help="analyse an incomplete run; output is marked PARTIAL")
    a = ap.parse_args()
    rows = [json.loads(l) for f in sorted((AV.OUT / a.run_id / "pairs").glob("*.jsonl")) for l in open(f)]
    # completeness guard (report 31): every pair file must hold every condition, and the run must hold every pair
    names_by = {}
    for r in rows:
        if r.get("arm") and r["arm"] != "G0":
            names_by.setdefault(r["pair_id"], set()).add(r["name"])
    allnames = set().union(*names_by.values()) if names_by else set()
    incomplete = sorted(p for p, s in names_by.items() if s != allnames)
    n_done = len(names_by)
    partial = n_done < a.expect_pairs or bool(incomplete)
    if partial and not a.allow_partial:
        raise SystemExit(f"run {a.run_id}: {n_done}/{a.expect_pairs} pairs, {len(incomplete)} pair files missing "
                         f"conditions; finish the run or pass --allow-partial")
    # quarantine handling: a quarantined row drops that (pair, condition) only; each statistic uses the pairs that
    # have every condition it needs, and reports its n
    quar = sorted((r["pair_id"], r["name"]) for r in rows if r.get("quarantined"))
    by = {}
    for r in rows:
        if r.get("arm") and r["arm"] != "G0" and not r.get("quarantined"):
            by.setdefault(r["pair_id"], {})[r["name"]] = r
    pids = sorted(by)

    def cols(*names):
        ps = [p for p in pids if all(n in by[p] for n in names)]
        return ps, [by[p][names[0]]["table"] for p in ps], [np.array([by[p][n]["S"] for p in ps]) for n in names]

    def cond(n):
        ps, tab, (x,) = cols(n)
        return {**boot(x, tab), "flip_rate": float(np.mean([by[p][n]["flip"] for p in ps])),
                "edit_norm_sum_median": float(np.median([by[p][n]["edit_norm_sum_base"] for p in ps]))}

    def contrast(f, *names):
        ps, tab, xs = cols(*names)
        return boot(f(*xs), tab)

    names = sorted(allnames)
    out = {"run_id": a.run_id, "label": "POST HOC, exploratory (011 amendments 2-3)", "n_pairs": len(pids),
           "expect_pairs": a.expect_pairs, "partial": partial, "incomplete_pair_files": incomplete,
           "quarantined_rows": quar, "conditions": {n: cond(n) for n in names}}
    out["contrasts"] = {
        "C_rawcoord - C_raw_rand": contrast(lambda x, r0, r1: x - (r0 + r1) / 2, "C_rawcoord", "C_raw_rand_0", "C_raw_rand_1"),
        "D_ar - D_classmean": contrast(lambda x, y: x - y, "D_ar", "D_classmean"),
        "D_ar - D_rand": contrast(lambda x, r0, r1: x - (r0 + r1) / 2, "D_ar", "D_rand_0", "D_rand_1"),
        "D_classmean - D_rand": contrast(lambda x, r0, r1: x - (r0 + r1) / 2, "D_classmean", "D_rand_0", "D_rand_1"),
        "C_white - Crand": contrast(lambda x, *r: x - np.mean(r, 0), "C_white", *[f"Crand_{i}" for i in range(4)]),
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
    res = AV.RES / f"olens_swaps_analysis_{a.run_id}{'_PARTIAL' if partial else ''}.json"
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
