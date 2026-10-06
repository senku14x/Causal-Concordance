"""009 reader-phase analysis (CPU; decision rules: docs/design_specs/009b_reader_phase.md). Reader-side.

Per instrument x layer x task x cell x piece: 006's D for z, f and start (start / f on collision-free pairs) as a mean
over tables, 95% table-cluster bootstrap (10,000 draws, seed 20261004), sign counts, a 200-derangement null. Primary
contrast at cells where the z probe decodes under both tasks: Delta = D_z(final) - D_z(start), paired by table.
Instruments: Oracle L40/L44, NLA L42/L44 (pieces), token lenses (whole), the prompt-only floor, the model's own output.
"""
import json
from pathlib import Path

import numpy as np

import c009 as C

CI = {c: i for i, c in enumerate(C.COLOURS)}
NB = 10000
N_DERANGE = 200
DELTA = 0.5


def colour_ll(arr, owner):
    arr = np.asarray(arr, np.float64)
    owner = np.asarray(owner)
    out = np.empty((arr.shape[0], len(C.COLOURS)))
    for j, c in enumerate(C.COLOURS):
        cols = arr[:, owner == c]
        m = cols.max(1, keepdims=True)
        out[:, j] = (m + np.log(np.exp(cols - m).sum(1, keepdims=True)))[:, 0]
    return out


def owner24():
    return np.array([c for c in C.COLOURS for _ in range(2)])


def pairs_of(meta):
    idx = {(r["item"], r["role"], r["partner"]): i for i, r in enumerate(meta["renders"])}
    out = []
    for i, r in enumerate(meta["renders"]):
        if r["role"] == "partner":
            m = idx[(r["item"], "member", None)]
            mr = meta["renders"][m]
            roles_m = {k: mr[k] for k in ("start", "z", "f")}
            roles_p = {k: r[k] for k in ("start", "z", "f")}
            coll = {}
            for k in ("start", "z", "f"):
                others = {v for kk, v in roles_m.items() if kk != k} | {v for kk, v in roles_p.items() if kk != k}
                coll[k] = bool({roles_m[k], roles_p[k]} & others)
            out.append({"m": m, "p": i, "item": r["item"], "m_roles": roles_m, "p_roles": roles_p, "collide": coll})
    return out


def pair_D(LLc, pairs, role, rows=None):
    v = np.full(len(pairs), np.nan)
    for i, p in enumerate(pairs):
        if role != "z" and p["collide"][role]:
            continue
        tm, tp = p["m_roles"][role], p["p_roles"][role]
        if tm == tp:
            continue
        a, b = (p["m"], p["p"]) if rows is None else rows[i]
        v[i] = (LLc[a, CI[tm]] - LLc[a, CI[tp]]) - (LLc[b, CI[tm]] - LLc[b, CI[tp]])
    return v


def table_means(v, pairs):
    items = np.array([p["item"] for p in pairs])
    ok = ~np.isnan(v)
    ts = sorted(set(items[ok]))
    return {t: float(v[ok & (items == t)].mean()) for t in ts}


def boot_mean(x, seed=C.BOOT_SEED, conf=0.95):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float("nan"), float("nan")], float("nan")
    rng = np.random.default_rng(seed)
    bm = rng.choice(x, (NB, len(x))).mean(1)
    lo, hi = np.quantile(bm, [(1 - conf) / 2, (1 + conf) / 2])
    p = float(min(1.0, 2 * min((bm <= 0).mean(), (bm >= 0).mean())))
    return [float(lo), float(hi)], max(p, 1e-4)


def derange_null(LLc, pairs, role, seed=C.SEED if hasattr(C, "SEED") else 20261005):
    """Replace each pair's activation rows with those of a pair from another table (roles kept)."""
    items = sorted({p["item"] for p in pairs})
    by = {t: [i for i, p in enumerate(pairs) if p["item"] == t] for t in items}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_DERANGE):
        perm = list(items)
        while True:
            rng.shuffle(perm)
            if all(a != b for a, b in zip(items, perm)):
                break
        sig = dict(zip(items, perm))
        rows = []
        for i, p in enumerate(pairs):
            src = by[sig[p["item"]]]
            q = pairs[src[i % len(src)]]
            rows.append((q["m"], q["p"]))
        tm = table_means(pair_D(LLc, pairs, role, rows), pairs)
        vals.append(np.mean(list(tm.values())) if tm else np.nan)
    v = np.array(vals)
    return {"mean": float(np.nanmean(v)), "p2.5": float(np.nanquantile(v, .025)), "p97.5": float(np.nanquantile(v, .975))}


def D_summary(LLc, pairs, role, null=False):
    tm = table_means(pair_D(LLc, pairs, role), pairs)
    x = np.array(list(tm.values()))
    ci, p = boot_mean(x)
    out = {"mean": float(x.mean()) if len(x) else float("nan"), "ci": ci, "p": p, "n_tables": len(x),
           "pos": int((x > 0).sum()), "neg": int((x < 0).sum())}
    if null:
        out["derangement_null"] = derange_null(LLc, pairs, role)
    return out, tm


def contrast(tm_final, tm_start):
    ts = sorted(set(tm_final) & set(tm_start))
    d = np.array([tm_final[t] - tm_start[t] for t in ts])
    ci95, p = boot_mean(d)
    ci90, _ = boot_mean(d, conf=0.90)
    out = {"mean": float(d.mean()) if len(d) else float("nan"), "ci95": ci95, "ci90": ci90, "p": p, "n_tables": len(d),
           "pos": int((d > 0).sum()), "neg": int((d < 0).sum())}
    return out


def categorize(c, Df, Ds):
    """009b rules. Df / Ds: D_z summaries under final / start."""
    if c["ci95"][0] > 0 and c["mean"] >= DELTA:
        return "follows the task (higher where z has demonstrated leverage)"
    if c["ci90"][0] > -DELTA and c["ci90"][1] < DELTA and Ds["ci"][0] > 0:
        return "task-blind (z decodable through the reader under both tasks, equal within delta)"
    if Df["ci"][0] <= 0 <= Df["ci"][1] and Ds["ci"][0] <= 0 <= Ds["ci"][1]:
        return "absent under both tasks"
    if c["ci95"][1] < 0 and c["mean"] <= -DELTA:
        return "higher under the start task"
    return "unresolved"


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def main(a):
    rd = C.OUT / "readers" / a.run_id
    cap = C.OUT / "captures" / a.cap_run
    metas = {t: json.loads((cap / f"{t}_meta.json").read_text()) for t in C.TASKS}
    pairs = {t: pairs_of(metas[t]) for t in C.TASKS}
    probes = json.loads((C.RES / "analysis" / "probes.json").read_text())
    both = probes["z_decodes_both_tasks"]
    res = {"run_id": a.run_id, "rules": "docs/design_specs/009b_reader_phase.md", "instruments": {}, "primary": {},
           "n_pairs": {t: len(pairs[t]) for t in C.TASKS}}
    tms = {}

    def add(name, L, t, cell, piece, LLc, null=False):
        key = f"{name}|L{L}|{t}|{cell}|{piece}"
        out = {}
        for role in ("z", "f", "start"):
            s, tm = D_summary(LLc, pairs[t], role, null=(null and role == "z"))
            out[role] = s
            tms[(key, role)] = tm
        res["instruments"][key] = out
    # readers (pieces)
    for rdr, layers in (("oracle", (40, 44)), ("nla", (42, 44))):
        st = rd / f"state_{rdr}.json"
        if not st.exists():
            continue
        for L in layers:
            for t in C.TASKS:
                for cell in C.EMIT_CELLS:
                    for pn in ("whole", "J", "rest_J", "R0", "R1", "R2", "rest_R0"):
                        f = rd / f"ll_{rdr}_L{L}_{t}_{cell}_{pn}.npz"
                        if f.exists():
                            add(rdr, L, t, cell, pn, colour_ll(np.load(f)["LL"], owner24()), null=(pn == "whole"))
    # token lenses
    f = rd / "ll_tokenlens.npz"
    if f.exists():
        z = np.load(f)
        for k in z.files:
            nm, Ls, t, cell = k.split("|")
            add(nm, int(Ls[1:]), t, cell, "whole", np.asarray(z[k], np.float64))
    # prompt-only floor
    f = rd / "ll_promptonly.npz"
    if f.exists():
        z = np.load(f)
        for t in C.TASKS:
            for cell in C.EMIT_CELLS:
                add("promptonly", 0, t, cell, "-", colour_ll(z[f"{t}|{cell}"], z["owner"]))
    # model's own output (outfacing run)
    of = sorted((C.OUT / "outfacing").glob("*/ll_slots.npz"))
    if of:
        z = np.load(of[-1])
        keys = list(z["keys"])
        slots = list(z["slots"])
        for t in C.TASKS:
            rows = [keys.index(r["key"]) for r in metas[t]["renders"]]
            for cell in C.EMIT_CELLS:
                add("model_output", 0, t, cell, "-", z["LL"][rows, slots.index(cell), :])
    # primary contrasts
    prim = []
    for k in res["instruments"]:
        nm, Ls, t, cell, pn = k.split("|")
        if t != "final" or pn not in ("whole", "-"):
            continue
        ks = f"{nm}|{Ls}|start|{cell}|{pn}"
        if ks not in res["instruments"]:
            continue
        c = contrast(tms[(k, "z")], tms[(ks, "z")])
        L = int(Ls[1:])
        eligible = bool(both.get(f"{cell}|L{L}", False)) if L else None
        res["primary"][f"{nm}|{Ls}|{cell}|{pn}"] = {"contrast": c, "D_z_final": res["instruments"][k]["z"],
                                                    "D_z_start": res["instruments"][ks]["z"], "z_decodes_both": eligible,
                                                    "category": categorize(c, res["instruments"][k]["z"],
                                                                           res["instruments"][ks]["z"])}
        if nm in ("oracle", "nla") and eligible:
            prim.append(f"{nm}|{Ls}|{cell}|{pn}")
    adj = holm([res["primary"][k]["contrast"]["p"] for k in prim]) if prim else []
    for k, p in zip(prim, adj):
        res["primary"][k]["holm_p"] = float(p)
    res["primary_family"] = prim
    # positive controls
    for rdr in ("oracle", "nla"):
        st = rd / f"state_{rdr}.json"
        if st.exists():
            s = json.loads(st.read_text())
            res.setdefault("positive_control", {}).update(s.get("positive_control", {}))
            res.setdefault("gates", {})[rdr] = {k: s.get(k) for k in ("G_score1", "G_score2", "gates_pass")}
    C.dump(C.RES / "analysis" / f"readers_{a.run_id}.json", res)
    for k, v in res["primary"].items():
        c = v["contrast"]
        print(f"{k:38s} final {v['D_z_final']['mean']:+6.2f} start {v['D_z_start']['mean']:+6.2f} "
              f"delta {c['mean']:+6.2f} {np.round(c['ci95'], 2)} both={v['z_decodes_both']} -> {v['category']}")
