"""007 model-side analysis (CPU): Stage 2 (design §5.3) and Stage 2b (design §6.5 + A1).

  python analysis_model.py stage2 --run-id s1
  python analysis_model.py stage2b --run-id s1

Writes results/007_jsubspace/analysis/stage2.json and stage2b.json. Model-side, reader-free: no labels.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p007 as C                                                           # noqa: E402

COLS = C.COLOURS
B_BOOT = 10000
MARGIN = 0.10
S001 = {40: 0.35, 42: 0.48, 44: 0.50}


def load(stage, run_id):
    rows = []
    for f in sorted((C.OUT / stage / run_id / "pairs").glob("*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    return rows


def f_target(row, clean_logq, a2):
    d = {c: float(np.exp(q1) - np.exp(q0)) for c, q1, q0 in zip(COLS, row["logq"], clean_logq)}
    tp = sum(max(v, 0.0) for v in d.values())
    return (max(d[a2], 0.0) / tp) if tp >= 0.01 else None


def boot_median(x, B=B_BOOT, seed=C.BOOT_SEED):
    x = np.asarray(x, float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(B, len(x)))
    m = np.median(x[idx], axis=1)
    return {"median": float(np.median(x)), "ci": [float(np.quantile(m, .025)), float(np.quantile(m, .975))],
            "ci90": [float(np.quantile(m, .05)), float(np.quantile(m, .95))], "n": int(len(x)),
            "pos": int((x > 0).sum()), "neg": int((x < 0).sum())}


def stage2(run_id):
    from sklearn.isotonic import IsotonicRegression
    rows = load("stage2", run_id)
    clean = {r["pair_id"]: r for r in rows if r.get("cond") == "clean"}
    cond = [r for r in rows if r.get("cond") not in ("clean",) and "layer" in r]
    res = {"run_id": run_id, "n_pairs": len(clean), "quarantined": int(sum(bool(r.get("quarantined")) for r in cond)),
           "layers": {}}
    for L in C.LAYERS:
        R = [r for r in cond if r["layer"] == L and not r.get("quarantined")]
        by = {}
        for r in R:
            by.setdefault(r["pair_id"], {})[r["cond"]] = r
        Sfull = {p: d["whole"]["S"] for p, d in by.items() if "whole" in d}
        rec_pairs = [p for p, s in Sfull.items() if s >= 0.10]
        rec = {p: {c: r["S"] / Sfull[p] for c, r in by[p].items()} for p in rec_pairs}
        out = {"S_full": boot_median(list(Sfull.values())), "S_full_001_median": S001[L], "n_recovery_subset": len(rec_pairs)}
        fam = {}
        for r in R:
            fam.setdefault(r["family"], []).append(r)
        out["families"] = {}
        for f_, rs in fam.items():
            pairs_f = sorted({r["pair_id"] for r in rs})
            S_pp = [np.mean([r["S"] for r in rs if r["pair_id"] == p]) for p in pairs_f]
            ds_pp = [np.mean([r["delta_share"] for r in rs if r["pair_id"] == p]) for p in pairs_f]
            rc_pp = [np.mean([r["S"] / Sfull[p] for r in rs if r["pair_id"] == p]) for p in pairs_f if p in rec]
            ft = [f_target(r, clean[r["pair_id"]]["logq_clean"], clean[r["pair_id"]]["a2"]) for r in rs]
            ft = [x for x in ft if x is not None]
            out["families"][f_] = {"S": boot_median(S_pp), "delta_share_median": float(np.median(ds_pp)),
                                   "recovery": boot_median(rc_pp) if rc_pp else None,
                                   "F_target_median": float(np.median(ft)) if ft else None, "F_target_n": len(ft),
                                   "flag_outside_share": float(np.mean([r["flag_outside"] for r in rs])),
                                   "flag_both_down_share": float(np.mean([r["flag_both_down"] for r in rs])),
                                   "n_rows": len(rs)}
        for c in ("PCA1", "PCA2", "PCA4", "PCA8", "PCA16"):
            rs = [r for r in R if r["cond"] == c]
            out["families"][c] = {"S": boot_median([r["S"] for r in rs]), "delta_share_median": float(np.median([r["delta_share"] for r in rs])),
                                  "recovery": boot_median([r["S"] / Sfull[r["pair_id"]] for r in rs if r["pair_id"] in rec])}
        # primary contrasts (per pair, recovery subset)
        dJP = [rec[p]["J"] - rec[p]["PCA4"] for p in rec_pairs if "J" in rec[p] and "PCA4" in rec[p]]
        dJR = [rec[p]["J"] - np.mean([v for k, v in rec[p].items() if k.startswith("Rdelta_")]) for p in rec_pairs if "J" in rec[p]]
        cJP, cJR = boot_median(dJP), boot_median(dJR)
        beyond = all(c["median"] >= MARGIN and c["ci"][0] > 0 for c in (cJP, cJR))
        tracks = all(-MARGIN < c["ci90"][0] and c["ci90"][1] < MARGIN for c in (cJP, cJR))
        below = all(c["median"] <= -MARGIN and c["ci"][1] < 0 for c in (cJP, cJR))
        out["primary"] = {"R_J_minus_R_PCA4": cJP, "R_J_minus_R_Rdelta": cJR,
                          "reading": "J beyond content share" if beyond else "tracks delta-share" if tracks else
                          "J below" if below else "mixed"}
        # secondary: isotonic recovery vs delta-share over all non-J conditions; J's residual with a refit bootstrap
        pts = [(r["delta_share"], r["S"] / Sfull[r["pair_id"]], r["pair_id"]) for r in R
               if r["pair_id"] in rec and r["cond"] not in ("J", "whole")]
        Jpts = {p: (by[p]["J"]["delta_share"], rec[p]["J"]) for p in rec_pairs if "J" in rec[p]}
        x, y, pid = np.array([a for a, _, _ in pts]), np.array([b for _, b, _ in pts]), np.array([c for _, _, c in pts])
        iso = IsotonicRegression(out_of_bounds="clip").fit(x, y)
        resid = [Jpts[p][1] - float(iso.predict([Jpts[p][0]])[0]) for p in Jpts]
        rng = np.random.default_rng(C.BOOT_SEED)
        ps = sorted(Jpts)
        bmed = []
        for _ in range(2000):
            samp = rng.choice(ps, len(ps))
            m = np.concatenate([np.where(pid == p)[0] for p in samp])
            f = IsotonicRegression(out_of_bounds="clip").fit(x[m], y[m])
            bmed.append(np.median([Jpts[p][1] - float(f.predict([Jpts[p][0]])[0]) for p in samp]))
        out["regression"] = {"J_residual_median": float(np.median(resid)),
                             "ci": [float(np.quantile(bmed, .025)), float(np.quantile(bmed, .975))],
                             "n_points": int(len(x)), "n_J": len(Jpts)}
        inter = [by[p]["J"]["S"] + by[p]["rest_J"]["S"] - by[p]["whole"]["S"] for p in by if {"J", "rest_J", "whole"} <= set(by[p])]
        out["interaction_J"] = boot_median(inter)
        res["layers"][L] = out
    C.dump(C.RES / "analysis" / "stage2.json", res)
    for L, o in res["layers"].items():
        print(L, "S_full", round(o["S_full"]["median"], 3), "|", o["primary"]["reading"], "| J-PCA4",
              round(o["primary"]["R_J_minus_R_PCA4"]["median"], 3), o["primary"]["R_J_minus_R_PCA4"]["ci"],
              "| J-Rdelta", round(o["primary"]["R_J_minus_R_Rdelta"]["median"], 3), o["primary"]["R_J_minus_R_Rdelta"]["ci"],
              "| J resid", round(o["regression"]["J_residual_median"], 3), o["regression"]["ci"])
    return res


def stage2b(run_id):
    rows = load("stage2b", run_id)
    clean = {r["pair_id"]: r for r in rows if r.get("cond") == "clean"}
    st = json.loads((C.OUT / "stage2b" / run_id / "state.json").read_text())
    cond = [r for r in rows if r.get("cond") != "clean" and "layer" in r and not r.get("quarantined")]
    res = {"run_id": run_id, "smoke_check": st.get("smoke_check"), "quarantined": st.get("quarantined"), "layers": {},
           "name": "adapted selectivity test (A1)"}
    for L in C.LAYERS:
        R = [r for r in cond if r["layer"] == L]
        by = {}
        for r in R:
            by.setdefault(r["pair_id"], {})[r["cond"]] = r

        def top1_flip(r):
            return COLS[int(np.argmax(r["logq"]))] == clean[r["pair_id"]]["a2"]

        def dmg(r):
            return bool(r["flag_outside"] or r["flag_both_down"])
        out = {"full": {"S": boot_median([d["full"]["S"] for d in by.values()]),
                        "flip_share": float(np.mean([top1_flip(d["full"]) for d in by.values()]))},
               "rand": {"S": boot_median([np.mean([d[f"rand_{i}"]["S"] for i in range(4)]) for d in by.values()])}, "lenses": {}}
        for ln in ("J_NP", "J_CB"):
            SJ = np.array([d[f"J_{ln}"]["S"] for d in by.values()])
            SN = np.array([d[f"nonJ_{ln}"]["S"] for d in by.values()])
            SR = np.array([np.mean([d[f"rand_{i}"]["S"] for i in range(4)]) for d in by.values()])
            SF = np.array([d["full"]["S"] for d in by.values()])
            ftJ = [f_target(d[f"J_{ln}"], clean[p]["logq_clean"], clean[p]["a2"]) for p, d in by.items()]
            ftN = [f_target(d[f"nonJ_{ln}"], clean[p]["logq_clean"], clean[p]["a2"]) for p, d in by.items()]
            ftJ, ftN = [x for x in ftJ if x is not None], [x for x in ftN if x is not None]
            dJ = float(np.mean([dmg(d[f"J_{ln}"]) for d in by.values()]))
            dN = float(np.mean([dmg(d[f"nonJ_{ln}"]) for d in by.values()]))
            cJN, cJR = boot_median(SJ - SN), boot_median(SJ - SR)
            crit = {"J_minus_nonJ_CI_above_0": cJN["ci"][0] > 0,
                    "S_J_ge_half_S_full": float(np.median(SJ)) >= 0.5 * float(np.median(SF)),
                    "S_J_above_random_CI_above_0": cJR["ci"][0] > 0,
                    "F_target_J_gt_nonJ": (bool(ftJ) and bool(ftN) and float(np.median(ftJ)) > float(np.median(ftN))),
                    "damage_J_le_nonJ": dJ <= dN}
            replicated = all(crit.values())
            not_rep = float(np.median(SN)) >= float(np.median(SJ))
            out["lenses"][ln] = {"S_J": boot_median(SJ), "S_nonJ": boot_median(SN), "J_minus_nonJ": cJN, "J_minus_rand": cJR,
                                 "F_target_J": float(np.median(ftJ)) if ftJ else None, "F_target_nonJ": float(np.median(ftN)) if ftN else None,
                                 "damage_rate_J": dJ, "damage_rate_nonJ": dN,
                                 "flip_share_J": float(np.mean([top1_flip(d[f"J_{ln}"]) for d in by.values()])),
                                 "flip_share_nonJ": float(np.mean([top1_flip(d[f"nonJ_{ln}"]) for d in by.values()])),
                                 "criteria": crit,
                                 "reading": "privilege replicated (adapted selectivity test)" if replicated else
                                 "not demonstrated under this adaptation" if not_rep else "mixed"}
        res["layers"][L] = out
    C.dump(C.RES / "analysis" / "stage2b.json", res)
    for L, o in res["layers"].items():
        print(L, "full S", round(o["full"]["S"]["median"], 4), "rand", round(o["rand"]["S"]["median"], 4),
              {ln: (round(v["S_J"]["median"], 4), round(v["S_nonJ"]["median"], 4), v["reading"]) for ln, v in o["lenses"].items()})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stage2", "stage2b"])
    ap.add_argument("--run-id", default="s1")
    a = ap.parse_args()
    {"stage2": stage2, "stage2b": stage2b}[a.what](a.run_id)


if __name__ == "__main__":
    main()
