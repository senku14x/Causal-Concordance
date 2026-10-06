"""008 Track B analysis (CPU): per condition S (median and mean over pairs, 95% table-cluster bootstrap, 10,000 draws,
seed 20261004), top-1 flip rate, gates; the design's power gates and reading rule (J vs non-J, 2b's rule) per site x
lens x metric x magnitude; the 33-pair subset (004's frozen pairs) alongside.

  python trackb_analysis.py --run-id b1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
import c008 as C                                   # noqa: E402
import stats as ST                                 # noqa: E402

B = 10000


def cb(vals, tabs, stat=np.median):
    r = ST.cluster_bootstrap(np.asarray(vals, float), np.asarray(tabs), stat=stat, B=B, seed=C.BOOT_SEED)
    return r


def summarize(rows, subset=None):
    by = {}
    for r in rows:
        if subset is not None and r["pair_id"] not in subset:
            continue
        if r.get("name") in ("clean", "G_repro") or r.get("arm") == "G0":
            continue
        if r.get("quarantined"):
            continue
        by.setdefault((r["site"], r["name"]), []).append(r)
    out = {}
    for (site, name), rs in sorted(by.items()):
        S = np.array([r["S"] for r in rs])
        T = np.array([r["table"] for r in rs])
        med = cb(S, T, np.median)
        mean = cb(S, T, np.mean)
        out[f"{site}|{name}"] = {"n": len(rs), "n_tables": int(len(set(T))), "median_S": float(np.median(S)),
                                 "median_ci": med.get("ci"), "mean_S": float(S.mean()), "mean_ci": mean.get("ci"),
                                 "flip_rate": float(np.mean([r["flip"] for r in rs])),
                                 "pos": int((S > 0).sum()), "neg": int((S < 0).sum()),
                                 "top1_retention_median": float(np.nanmedian([r["top1_retention"] for r in rs])),
                                 **{k: rs[0].get(k) for k in ("arm", "lens", "metric", "mag", "part", "alpha")}}
    return out


def paired(rows, site, a, b, subset=None):
    """Per pair S(a) - S(b), median with table-cluster CI."""
    Sa = {r["pair_id"]: (r["S"], r["table"]) for r in rows if r.get("site") == site and r.get("name") == a
          and not r.get("quarantined") and (subset is None or r["pair_id"] in subset)}
    Sb = {r["pair_id"]: r["S"] for r in rows if r.get("site") == site and r.get("name") == b and not r.get("quarantined")}
    k = sorted(set(Sa) & set(Sb))
    if len(k) < 2:
        return None
    d = np.array([Sa[p][0] - Sb[p] for p in k])
    t = np.array([Sa[p][1] for p in k])
    r = cb(d, t, np.median)
    return {"n": len(k), "median_diff": float(np.median(d)), "ci": r.get("ci"), "pos": int((d > 0).sum()),
            "neg": int((d < 0).sum())}


def readings(summ, rows, subset=None):
    """Power gates and 2b's reading rule per site x lens x metric x magnitude (arm B), and arm A's power gate."""
    out = {"power": {}, "B": {}, "A": {}}
    for site in ("emit3", "colon"):
        for metric in ("raw", "white"):
            full = summ.get(f"{site}|B_{metric}_full")
            if full:
                out["power"][f"B|{site}|{metric}"] = {"median_S_full": full["median_S"],
                                                      "pass": full["median_S"] >= C.POWER_MIN_S}
        for lens in ("J_NP", "J_CB"):
            ans = summ.get(f"{site}|Actrl_answer_raw_{lens}")
            if ans:
                out["power"][f"A|{site}|{lens}"] = {"median_S_answer_swap": ans["median_S"],
                                                    "pass": ans["median_S"] >= C.POWER_MIN_S}
            for metric, mag, suff in (("raw", "natural", ""), ("white", "natural", ""), ("raw", "rawnorm", "_rawnorm"),
                                      ("raw", "whitenorm", "_whitenorm")):
                nmJ = f"B_{metric}_J_{lens}{suff}"
                nmN = f"B_{metric}_nonJ_{lens}{suff}"
                if f"{site}|{nmJ}" not in summ:
                    continue
                full_key = f"{site}|B_{'white' if metric == 'white' else 'raw'}_full"
                full = summ[full_key]["median_S"]
                Sj, Sn = summ[f"{site}|{nmJ}"]["median_S"], summ[f"{site}|{nmN}"]["median_S"]
                pdiff = paired(rows, site, nmJ, nmN, subset)
                pw = full >= C.POWER_MIN_S
                if not pw:
                    reading = "not read (power gate failed)"
                elif pdiff and pdiff["median_diff"] > 0 and pdiff["ci"][0] > 0 and Sj >= 0.5 * full:
                    reading = "privilege replicated"
                elif Sn >= Sj:
                    reading = "not replicated"
                else:
                    reading = "mixed"
                out["B"][f"{site}|{lens}|{metric}|{mag}"] = {"S_full": full, "S_J": Sj, "S_nonJ": Sn, "J_minus_nonJ": pdiff,
                                                            "reading": reading}
            for metric in ("raw", "white"):
                nm = f"A_{metric}_a1_{lens}"
                if f"{site}|{nm}" in summ:
                    out["A"][f"{site}|{lens}|{metric}|a1"] = {k: summ[f"{site}|{nm}"][k] for k in
                                                              ("median_S", "median_ci", "mean_S", "flip_rate")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="b1")
    a = ap.parse_args()
    d = C.OUT / a.run_id
    rows = [json.loads(l) for f in sorted((d / "pairs").glob("*.jsonl")) for l in open(f)]
    state = json.loads((d / "state.json").read_text())
    man = json.loads((C.PROJECT / "results" / "004_span_assay" / "manifest_pairs.json").read_text())
    sub33 = set(man["pairs"])
    rep = [r for r in rows if r.get("name") == "G_repro"]
    g0 = [r for r in rows if r.get("arm") == "G0"]
    q = [r for r in rows if r.get("quarantined")]
    cond_rows = [r for r in rows if "site" in r and r.get("arm") != "G0"]
    gates = {"G_repro": {"n": len(rep), "pass": int(sum(r["pass"] for r in rep)),
                         "max_abs_diff": float(max(abs(r["S"] - r["S_001"]) for r in rep)) if rep else None},
             "G0": {"n": len(g0), "pass": int(sum(r["pass"] for r in g0)), "max_abs_dm": float(max(abs(r["dm"]) for r in g0))
                    if g0 else None},
             "G1_and_G_clamp": {"rows": len(cond_rows), "quarantined": len(q),
                                "clamp_err_max": float(max(r["clamp_err_max"] for r in cond_rows if r.get("clamp_err_max")
                                                           is not None)) if cond_rows else None},
             "G_RA": {"ra_err_max": state.get("ra_err_max"), "pass": state.get("ra_err_max", 1) <= 1e-5}}
    summ = summarize(rows)
    summ33 = summarize(rows, sub33)
    out = {"run_id": a.run_id, "n_pairs": len({r["pair_id"] for r in rows}), "gates": gates, "conditions": summ,
           "readings": readings(summ, rows), "subset33": {"conditions": summ33, "readings": readings(summ33, rows, sub33)},
           "complementary": {}}
    for site in ("emit3",):
        for lens in ("J_NP", "J_CB"):
            for nm, base in ((f"B_raw_nonJ_clampJ_{lens}", f"B_raw_nonJ_{lens}"), (f"B_raw_J_clampNonJ_{lens}", f"B_raw_J_{lens}")):
                out["complementary"][f"{site}|{nm}"] = {"S_clamped": summ.get(f"{site}|{nm}", {}).get("median_S"),
                                                        "S_unclamped": summ.get(f"{site}|{base}", {}).get("median_S"),
                                                        "paired_clamped_minus_unclamped": paired(rows, site, nm, base)}
    C.dump(C.RES / "analysis" / f"trackb_{a.run_id}.json", out)
    print(json.dumps(gates, indent=1))
    for k, v in summ.items():
        print(f"{k:40s} n={v['n']:3d} medS={v['median_S']:+.3f} {np.round(v['median_ci'], 3) if v['median_ci'] else ''} "
              f"meanS={v['mean_S']:+.3f} flip={v['flip_rate']:.2f}")
    print(json.dumps(out["readings"], indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
