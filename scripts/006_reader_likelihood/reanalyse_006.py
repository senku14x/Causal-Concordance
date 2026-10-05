"""006 post-hoc reanalysis (PI-forwarded review, 2026-10-05; every item here is post hoc and labelled so):
secondary cells, the positive control with the colour prior removed, a 200-derangement shuffle null, the role-collision
census with collision-free D, and per-pair reader agreement. Writes results/006_reader_likelihood/analysis/posthoc.json."""
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
D = R.PROJECT / "outputs/006_reader_likelihood/run1"
BLOCKS = {"NLA L42": "ll_nla_L42_p0.npz", "Oracle L40": "ll_oracle_L40_p0.npz", "Oracle L44": "ll_oracle_L44_p0.npz"}
meta = json.loads((R.PROJECT / "outputs/001_leverage_profile/run/captures_brew_meta.json").read_text())
RS = [r for r in meta["renders"] if r["role"] in ("member", "partner")]
ITEMS = sorted({r["item"] for r in RS})
TAB = {it: {"m": [n for n, r in enumerate(RS) if r["item"] == it and r["role"] == "member"][0],
            "p": [n for n, r in enumerate(RS) if r["item"] == it and r["role"] == "partner"]} for it in ITEMS}
rng = np.random.default_rng(20261004)


def load(f, cell):
    z = np.load(D / f)
    own = [str(o) for o in z["owner"]]
    return A.per_colour(z[cell], own), A.per_colour(z["positive"], own)


def collides(m, p, t):
    others = {RS[b][o] for b in (m, p) for o in ROLES if o != t}
    return RS[m][t] in others or RS[p][t] in others


def per_pair(LL, t, collision_free=False):
    out = {}
    for it in ITEMS:
        m = TAB[it]["m"]
        for p in TAB[it]["p"]:
            if RS[p][t] == RS[m][t] or (collision_free and collides(m, p, t)):
                continue
            cm, cp = CI[RS[m][t]], CI[RS[p][t]]
            out[(it, p)] = float((LL[m, cm] - LL[m, cp]) - (LL[p, cm] - LL[p, cp]))
    return out


def per_table(d):
    by = {}
    for (it, _), v in d.items():
        by.setdefault(it, []).append(v)
    return [float(np.mean(v)) for v in by.values()]


def boot(x):
    x = np.asarray(x, float)
    m = rng.choice(x, (10000, len(x))).mean(1)
    return {"n": int(len(x)), "mean": float(x.mean()), "ci": [float(np.quantile(m, .025)), float(np.quantile(m, .975))],
            "pos": int((x > 0).sum()), "neg": int((x < 0).sum())}


S = {"note": "post hoc (PI-forwarded review of 006, 2026-10-05); dev, 48 brew tables; prefix p0",
     "patch_share_001_L42": {"emit_colon": 0.48, "emit_asst": 0.29, "emit_think": 0.01,
                             "source": "docs/reports/001_leverage_profile.md, whole-cell patch share table (median S)"},
     "cells": {}, "positive_control": {}, "derangement_null": {}, "collisions": {}, "agreement": {}}
for nm, f in BLOCKS.items():
    dz = {c: per_table(per_pair(load(f, c)[0], "z")) for c in ("emit_colon", "emit_think", "emit_asst")}
    S["cells"][nm] = {c: boot(v) for c, v in dz.items()}
    S["cells"][nm]["think_minus_colon"] = boot(np.array(dz["emit_think"]) - np.array(dz["emit_colon"]))
    _, P = load(f, "emit_colon")
    loo = np.array([P[i] - np.delete(P, i, 0).mean(0) for i in range(12)])
    S["positive_control"][nm] = {"raw_ID": float(np.mean(P.argmax(1) == np.arange(12))),
                                 "argmax": [COL[i] for i in P.argmax(1)],
                                 "loo_prior_removed_ID": float(np.mean(loo.argmax(1) == np.arange(12))),
                                 "pairwise_double_difference_positive": int(sum(((P[i, i] - P[i, j]) - (P[j, i] - P[j, j])) > 0
                                                                                for i in range(12) for j in range(i + 1, 12))),
                                 "pairs": 66}
    LL, _ = load(f, "emit_colon")
    nul = {}
    r2 = np.random.default_rng(1)
    perms = []
    while len(perms) < 200:
        p_ = r2.permutation(len(ITEMS))
        if not (p_ == np.arange(len(ITEMS))).any():
            perms.append(p_)
    for t in ROLES:
        vals = []
        for perm in perms:
            per = []
            for k, it in enumerate(ITEMS):
                m, o = TAB[it]["m"], TAB[ITEMS[perm[k]]]
                v = [(LL[o["m"], CI[RS[m][t]]] - LL[o["m"], CI[RS[p][t]]]) - (LL[o["p"][j % len(o["p"])], CI[RS[m][t]]]
                     - LL[o["p"][j % len(o["p"])], CI[RS[p][t]]]) for j, p in enumerate([p for p in TAB[it]["p"] if RS[p][t] != RS[m][t]])]
                if v:
                    per.append(np.mean(v))
            vals.append(float(np.mean(per)))
        nul[t] = {"mean": float(np.mean(vals)), "p2.5": float(np.quantile(vals, .025)), "p97.5": float(np.quantile(vals, .975))}
    S["derangement_null"][nm] = nul
    S["collisions"][nm] = {t: boot(per_table(per_pair(LL, t, collision_free=True))) for t in ("start", "answer")}
S["collisions"]["census"] = {t: int(sum(collides(TAB[it]["m"], p, t) for it in ITEMS for p in TAB[it]["p"])) for t in ROLES}
S["collisions"]["n_pairs"] = int(sum(len(TAB[it]["p"]) for it in ITEMS))
pp = {nm: per_pair(load(f, "emit_colon")[0], "z") for nm, f in BLOCKS.items()}
keys = sorted(pp["NLA L42"])
for o in ("Oracle L40", "Oracle L44"):
    a, b = np.array([pp["NLA L42"][k] for k in keys]), np.array([pp[o][k] for k in keys])
    S["agreement"][f"NLA L42 vs {o}"] = {"same_sign": float(np.mean(np.sign(a) == np.sign(b))), "r": float(np.corrcoef(a, b)[0, 1]),
                                         "n_pairs": len(keys)}
for nm, f in BLOCKS.items():
    b = per_pair(load(f.replace("_p0", "_p1"), "emit_colon")[0], "z")
    S["agreement"][f"{nm} p0 vs p1"] = {"r": float(np.corrcoef([pp[nm][k] for k in keys], [b[k] for k in keys])[0, 1])}
out = R.PROJECT / "results/006_reader_likelihood/analysis/posthoc.json"
out.write_text(json.dumps(S, indent=1))
print("wrote", out)
