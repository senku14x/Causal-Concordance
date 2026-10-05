"""Analysis for design 006b (docs/design_specs/006b_model_output_baseline.md). CPU only.

  python model_output_analysis.py [--model-run m1]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
import reader_likelihood_analysis as A            # noqa: E402
import registry as R                               # noqa: E402

COL, ROLES = A.COLOURS, ("start", "z", "answer")
CI = {c: i for i, c in enumerate(COL)}
CELLS = ("emit_asst", "emit_think", "emit_colon")
PRIMARY = {"NLA L42": "ll_nla_L42_p0.npz", "Oracle L40": "ll_oracle_L40_p0.npz", "Oracle L44": "ll_oracle_L44_p0.npz"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-run", default="m1")
    a = ap.parse_args()
    D = R.PROJECT / "outputs/006_reader_likelihood"
    meta = json.loads((R.PROJECT / "outputs/001_leverage_profile/run/captures_brew_meta.json").read_text())
    RS = [r for r in meta["renders"] if r["role"] in ("member", "partner")]
    z = np.load(D / f"model_{a.model_run}" / "ll_model.npz")
    assert [str(k) for k in z["keys"]] == [r["key"] for r in RS]
    own = [str(o) for o in z["owner"]]
    M = {c: A.per_colour(z["LL"][:, i], own) for i, c in enumerate(CELLS)}           # [231, 12] per cell
    items = sorted({r["item"] for r in RS})
    tab = {it: {"m": [n for n, r in enumerate(RS) if r["item"] == it and r["role"] == "member"][0],
                "p": [n for n, r in enumerate(RS) if r["item"] == it and r["role"] == "partner"]} for it in items}
    rng = np.random.default_rng(20261004)

    def collides(m, p, t):
        others = {RS[b][o] for b in (m, p) for o in ROLES if o != t}
        return RS[m][t] in others or RS[p][t] in others

    def pairs(LL, t, cf):
        return {(it, p): float((LL[tab[it]["m"], CI[RS[tab[it]["m"]][t]]] - LL[tab[it]["m"], CI[RS[p][t]]])
                               - (LL[p, CI[RS[tab[it]["m"]][t]]] - LL[p, CI[RS[p][t]]]))
                for it in items for p in tab[it]["p"]
                if RS[p][t] != RS[tab[it]["m"]][t] and not (cf and collides(tab[it]["m"], p, t))}

    def per_table(d):
        by = {}
        for (it, _), v in d.items():
            by.setdefault(it, []).append(v)
        return np.array([np.mean(v) for v in by.values()])

    def boot(x):
        m = rng.choice(x, (10000, len(x))).mean(1)
        return {"n": int(len(x)), "mean": float(x.mean()), "ci": [float(np.quantile(m, .025)), float(np.quantile(m, .975))],
                "pos": int((x > 0).sum()), "neg": int((x < 0).sum())}
    S = {"design": "docs/design_specs/006b_model_output_baseline.md", "model_run": a.model_run,
         "gate": json.loads((D / f"model_{a.model_run}" / "state.json").read_text()).get("G_score_m"), "model_D": {}}
    for c in CELLS:
        S["model_D"][c] = {t: boot(per_table(pairs(M[c], t, t != "z"))) for t in ROLES}
    # rank of z at ':' on member renders; runner-up = z second, answer first
    ranks, runner = [], []
    for it in items:
        m = tab[it]["m"]
        order = list(np.argsort(-M["emit_colon"][m]))
        rz, ra = order.index(CI[RS[m]["z"]]) + 1, order.index(CI[RS[m]["answer"]]) + 1
        ranks.append(rz)
        runner.append(rz == 2 and ra == 1)
    S["rank_z_colon_member"] = {"median": float(np.median(ranks)), "counts": {str(k): ranks.count(k) for k in sorted(set(ranks))},
                                "runner_up_share": float(np.mean(runner)),
                                "answer_rank1_share": float(np.mean([list(np.argsort(-M["emit_colon"][tab[it]["m"]])).index(
                                    CI[RS[tab[it]["m"]]["answer"]]) == 0 for it in items]))}
    # B2: reader D_z on model D_z across pairs (':' cell), table-cluster bootstrap of the OLS intercept
    mz = pairs(M["emit_colon"], "z", False)
    keys = sorted(mz)
    B2 = {}
    for nm, f in PRIMARY.items():
        zz = np.load(D / "run1" / f)
        LLr = A.per_colour(zz["emit_colon"], [str(o) for o in zz["owner"]])
        rz = pairs(LLr, "z", False)
        x, y = np.array([mz[k] for k in keys]), np.array([rz[k] for k in keys])
        tid = np.array([items.index(k[0]) for k in keys])
        beta = np.polyfit(x, y, 1)
        ints = []
        for _ in range(10000):
            pick = rng.choice(len(items), len(items))
            idx = np.concatenate([np.where(tid == t)[0] for t in pick])
            ints.append(np.polyfit(x[idx], y[idx], 1)[1])
        ints = np.array(ints)
        sub = np.where(x <= 0)[0]                          # primary: pairs where the model's own output does not favour z
        rec = {"secondary_intercept": float(beta[1]), "slope": float(beta[0]),
               "intercept_ci": [float(np.quantile(ints, .025)), float(np.quantile(ints, .975))],
               "r": float(np.corrcoef(x, y)[0, 1]), "n_pairs": len(keys), "n_model_nonpositive": int(len(sub))}
        if len(sub) >= 15:
            st, means = np.unique(tid[sub]), []
            for _ in range(10000):
                pick = rng.choice(st, len(st))
                means.append(np.mean(np.concatenate([y[sub][tid[sub] == t] for t in pick])))
            means = np.array(means)
            rec.update({"testable": True, "mean_reader_Dz": float(y[sub].mean()),
                        "ci": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
                        "p_one_sided": float(np.mean(means <= 0)), "n_tables": int(len(st))})
        else:
            rec.update({"testable": False, "p_one_sided": None})
        B2[nm] = rec
    for k in B2:
        B2[k].setdefault("beyond_model", None)
    order = sorted([k for k in B2 if B2[k]["testable"]], key=lambda k: B2[k]["p_one_sided"])
    stop = False
    for i, k in enumerate(order):
        thr = 0.025 / (len(order) - i)
        ok = (not stop) and B2[k]["p_one_sided"] <= thr
        stop = stop or not ok
        B2[k].update({"holm_threshold": thr, "beyond_model": ok})
    S["B2_reader_beyond_model"] = B2
    ph = json.loads((R.PROJECT / "results/006_reader_likelihood/analysis/posthoc.json").read_text())
    mt = S["model_D"]["emit_think"]["z"]
    S["rules"] = {"B1_model_tracks_z_colon": S["model_D"]["emit_colon"]["z"]["ci"][0] > 0,
                  "B2_beyond_model": {k: v["beyond_model"] for k, v in B2.items()},
                  "B3_think_model_not_readers_yes": bool(mt["ci"][0] <= 0 and all(ph["cells"][nm]["emit_think"]["ci"][0] > 0 for nm in PRIMARY)),
                  "B4_runner_up_ge_half": S["rank_z_colon_member"]["runner_up_share"] >= 0.5}
    out = R.PROJECT / "results/006_reader_likelihood/analysis/model_output.json"
    out.write_text(json.dumps(S, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
