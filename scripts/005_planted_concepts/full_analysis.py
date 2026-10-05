"""Analysis of the 005 full run under amendment 01 (docs/design_specs/005_full_run.md §3-§4). CPU only.

  python full_analysis.py RUN_ID [--stage run|rehearse]

Writes results/005_planted_concepts/analysis/full_summary.json (stage run) or full_summary_rehearse.json. Gates first
(G0, G3, G4, G5/leaky, G6, G2, G11), then the pre-registered contrasts (M1, M2 with its output-direction control, M3,
R1, R2, P11' for the NLA and the Oracle, P17), Holm over the P-family {P11' (NLA), P17}, then the secondary tables.
Statistics: per-concept values, cluster bootstrap over concepts (10,000 draws, seed 20261004).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p005 as P                                   # noqa: E402

GRID9 = [1 / 256, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
TEXT_S = [1 / 64, 1 / 4, 1.0]
ENDS = list(range(44, 63, 2))
NB = 10000
ALPHA = 0.025                                      # one-sided, matching the two-sided 95% CI rule


def med(x):
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else None


def mean(x):
    x = [v for v in x if v is not None]
    return float(np.mean(x)) if x else None


class Boot:
    """Cluster bootstrap over concepts (one value per concept); one generator, seeded once."""

    def __init__(self):
        self.rng = np.random.default_rng(P.SEED)

    def __call__(self, x):
        x = np.asarray([v for v in x if v is not None], float)
        if len(x) < 2:
            return {"n": int(len(x)), "mean": float(x.mean()) if len(x) else None, "ci": [None, None], "means": None}
        m = self.rng.choice(x, size=(NB, len(x)), replace=True).mean(1)
        return {"n": int(len(x)), "mean": float(x.mean()), "ci": [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))],
                "means": m}


def verdict(b, direction):
    """'supported' if the 95% CI excludes 0 in the predicted direction; one-sided bootstrap p for Holm."""
    if b["means"] is None:
        return {"n": b["n"], "mean": b["mean"], "ci": b["ci"], "supported": None, "p_one_sided": None}
    lo, hi = b["ci"]
    sup = lo > 0 if direction > 0 else hi < 0
    p = float(np.mean(b["means"] <= 0)) if direction > 0 else float(np.mean(b["means"] >= 0))
    return {"n": b["n"], "mean": b["mean"], "ci": b["ci"], "supported": bool(sup), "p_one_sided": p}


def gt(a, b):
    return a is not None and b is not None and a > b


def le(a, b):
    return a is not None and b is not None and a <= b


def holm(ps, alpha=ALPHA):
    items = sorted([(p, k) for k, p in ps.items() if p is not None])
    out, m, stop = {}, len(items), False
    for i, (p, k) in enumerate(items):
        thr = alpha / (m - i)
        ok = (not stop) and p <= thr
        stop = stop or not ok
        out[k] = {"p": p, "threshold": thr, "rejects_null": ok}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--stage", default="run", choices=("run", "rehearse"))
    a = ap.parse_args()
    D = P.ROOT / "outputs/005_planted_concepts/run" / a.run_id
    OUT = P.ROOT / "results/005_planted_concepts/analysis" / ("full_summary.json" if a.stage == "run" else "full_summary_rehearse.json")
    st = json.loads((D / "state.json").read_text())
    concepts, id_set = st["concepts"], st["id_set"]
    boot = Boot()
    S = {"run_id": a.run_id, "stage": a.stage, "n_concepts": len(concepts),
         "note": "pre-registered analysis of docs/design_specs/005_full_run.md; per-concept statistics, cluster bootstrap"}

    def load(name):
        f = D / name
        return json.loads(f.read_text()) if f.exists() else None
    model = load("model_rows.json")
    r2 = load("r2_rows.json") or []
    clamps = load("clamp_rows.json") or []
    tpr = load("token_probe_reads.json") or []
    readers = {"oracle": load("oracle_reads.json") or [], "nla": load("nla_reads.json") or []}
    lik = {f.stem.replace("lik_", ""): json.loads(f.read_text()) for f in sorted(D.glob("lik_*_L*.json"))}

    idx = {}
    for r in model:
        idx[(r["concept"], str(r["cond"]), r["s"], r.get("seed", 0))] = r

    def M(c, cond, s, key="dR1p", seed=0):
        r = idx.get((c, str(cond), s, seed))
        return None if r is None else r.get(key)

    def null95(key, s):
        v = [r[key] for r in model if r["cond"] == 7 and r["s"] == s and r.get(key) is not None]
        return float(np.quantile(v, 0.95)) if v else None

    # ------------------------------------------------------------------------------------------------ gates
    gates = {}
    g0 = st.get("G0", {})
    gates["G0"] = {"blocks": g0, "pass": all(v.get("failures", 1) == 0 for v in g0.values()) and bool(g0)}
    mat = st.get("material", {})
    js = mat.get("J_share_median")
    gates["G3"] = {"J_share_median": js, "flag": js is None or not (0.01 <= js <= 0.30)}
    g4 = st.get("G4_probe_validity", {})
    lo, pe = med([v["loto_auc"] for v in g4.values()]), med([v["perm_ctrl_auc"] for v in g4.values()])
    gates["G4"] = {"loto_auc_median": lo, "perm_ctrl_auc_median": pe, "pass": lo is not None and lo >= 0.9 and pe <= 0.6}
    leaky = mat.get("leaky_bandB", [])
    gates["G5_leaky"] = {"leaky": leaky, "rule": "c in J_NP top-10 on pure u_perp at >= 2 band layers (§16.1 Q7)"}
    nonleaky = [c for c in concepts if c not in leaky]

    # G6: clean-read naming per reader cell
    g6 = {}
    for name, reads in readers.items():
        for q in reads:
            if q["cond"] == "clean":
                key = f"{name}_L{q['layer']}"
                rates = {c: float(np.mean([P.names(t, c) for t in q["samples"]])) for c in concepts}
                g6[key] = {"dropped": sorted(c for c, v in rates.items() if v > 0.05), "rates_over_0": {c: v for c, v in rates.items() if v > 0}}
    gates["G6"] = g6

    def naming(name, layer, c, cond, s, seed=0):
        for q in readers[name]:
            if (q["layer"] == layer and q["concept"] == c and str(q["cond"]) == str(cond) and q["s"] == s
                    and q.get("seed", 0) == seed):
                return float(np.mean([P.names(t, c) for t in q["samples"]]))
        return None
    reader_layers = {"nla": [42], "oracle": [40, 44]}

    # G2: per reader-layer, naming of the full vector above rand-dir at a reader dose where the model moves
    moves = [s for s in TEXT_S if gt(med([M(c, 2, s) for c in concepts]), null95("dR1p", s))
             or gt(med([M(c, 2, s, "B1") for c in concepts]), null95("B1", s))]
    g2 = {}
    for name, layers in reader_layers.items():
        for l in layers:
            keep = [c for c in concepts if c not in g6.get(f"{name}_L{l}", {}).get("dropped", [])]
            res = {}
            for s in moves:
                d = [naming(name, l, c, 2, s) - naming(name, l, c, 7, s) for c in keep
                     if naming(name, l, c, 2, s) is not None and naming(name, l, c, 7, s) is not None]
                b = boot(d)
                res[str(s)] = {"mean": b["mean"], "ci": b["ci"]}
            g2[f"{name}_L{l}"] = {"by_dose": res, "pass": any(v["ci"][0] is not None and v["ci"][0] > 0 for v in res.values())}
    gates["G2"] = {"model_moves_at": moves, "by_reader": g2}

    # likelihood reads: per reader-layer, per row SM; G11 on natural activations
    def lik_tables(d):
        W, clean = d["W"], d["clean"]
        ids = [w for w in id_set if w in W]
        nat = {x["concept"]: max(ids, key=lambda w: x["lpW"][w]) == x["concept"] for x in d["natural"]}
        rows = {}
        for r in d["rows"]:
            dw = {w: r["lpW"][w] - clean[w] for w in W}
            sp = P.specificity(dw, r["concept"], ids)
            rows[(r["concept"], str(r["cond"]), r["s"], r.get("seed", 0))] = {"SM": sp["SM"], "ID": sp["ID"], "dc": dw[r["concept"]],
                                                                             "generic": sp["generic"]}
        return rows, nat
    LT, g11 = {}, {}
    for key, d in lik.items():
        LT[key], nat = lik_tables(d)
        rate = float(np.mean(list(nat.values()))) if nat else None
        g11[key] = {"ID_natural": rate, "n": len(nat), "pass": rate is not None and rate >= 0.5,
                    "missed": sorted(c for c, ok in nat.items() if not ok), "W_size": len(d["W"])}
    gates["G11"] = g11
    S["gates"] = gates

    # ---------------------------------------------------------------------------- model-side SM (A18) per row
    def sm_model(c, cond, s, seed=0):
        r = idx.get((c, str(cond), s, seed))
        if r is None or "dW" not in r:
            return None
        ids = [w for w in id_set if w in r["dW"]]
        return P.specificity(r["dW"], c, ids)

    def onset_idx(fn):
        for i, s in enumerate(GRID9):
            v = fn(s)
            if v is not None and v >= 1.0:
                return i
        return len(GRID9)

    # ----------------------------------------------------------------------------------------------- contrasts
    C = {}
    # M1: peak over doses, J part (3) minus non-J part (4); R1' primary (A1), R1 as registered
    for key, nm in (("dR1p", "M1_R1p"), ("R1", "M1_R1")):
        d = [max(M(c, 3, s, key) for s in GRID9) - max(M(c, 4, s, key) for s in GRID9) for c in nonleaky]
        C[nm] = verdict(boot(d), +1)
    C["M1_per_dose"] = {str(s): {"median_J_minus_perp": med([M(c, 3, s) - M(c, 4, s) for c in nonleaky])} for s in GRID9}

    # M2 (A8): re-entry clamp to L50, J atoms and output directions, at each evaluable concept's best non-J dose
    cidx = {(r["concept"], r["cond"], r["s"]): r for r in clamps}
    ev = {}
    for c in nonleaky:
        s_b = max(GRID9, key=lambda s: M(c, 4, s, "R1"))
        if gt(M(c, 4, s_b), null95("dR1p", s_b)):
            ev[c] = s_b

    def extra(c, fam, E):
        s_b = ev[c]
        a_, b_ = cidx.get((c, f"{fam}{E}r", s_b)), cidx.get((c, f"{fam}{E}", s_b))
        return None if a_ is None or b_ is None else a_["dR1p"] - b_["dR1p"]
    curves = {}
    for E in ENDS:
        xj = [extra(c, "E", E) for c in ev]
        xo = [extra(c, "O", E) for c in ev]
        dd = [a_ - b_ for a_, b_ in zip(xj, xo) if a_ is not None and b_ is not None]
        bj, bo, bd = boot(xj), boot(xo), boot(dd)
        curves[str(E)] = {"extra_J": {"mean": bj["mean"], "ci": bj["ci"]}, "extra_out": {"mean": bo["mean"], "ci": bo["ci"]},
                          "D": {"mean": bd["mean"], "ci": bd["ci"]}}
    C["M2"] = dict(verdict(boot([extra(c, "E", 50) for c in ev]), +1), evaluable=len(ev), evaluable_concepts=ev)
    C["M2_control_D_at_L50"] = verdict(boot([extra(c, "E", 50) - extra(c, "O", 50) for c in ev
                                              if extra(c, "E", 50) is not None and extra(c, "O", 50) is not None]), +1)
    C["M2_output_clamp_at_L50"] = verdict(boot([extra(c, "O", 50) for c in ev]), +1)
    C["M2_design_clamp_10_vs_10r"] = verdict(boot([M(c, "10r", ev[c]) - M(c, 10, ev[c]) for c in ev]), +1)
    C["M2_curves"] = curves

    # M3: natural norm, J-natural (5) minus non-J-natural (6), at the doses where the full vector moves R1'
    s_full = [s for s in GRID9 if gt(med([M(c, 2, s) for c in concepts]), null95("dR1p", s))]
    C["M3"] = dict(verdict(boot([mean([M(c, 5, s) - M(c, 6, s) for s in s_full]) for c in nonleaky]) if s_full else
                           {"n": 0, "mean": None, "ci": [None, None], "means": None}, +1), doses=s_full)

    # R1: probe (absolute criterion, direct, L42) minus NLA naming (direct, L42) on u_perp at the reader doses
    pidx = {(q["concept"], str(q["cond"]), q["s"], q["layer"], q["kind"], q.get("seed", 0)): q for q in tpr}
    nla_drop = g6.get("nla_L42", {}).get("dropped", [])
    r1 = []
    for c in nonleaky:
        if c in nla_drop:
            continue
        v = [(1.0 if pidx[(c, "4", s, 42, "direct_cq", 0)]["probe"] > 0 else 0.0) - naming("nla", 42, c, 4, s)
             for s in TEXT_S if (c, "4", s, 42, "direct_cq", 0) in pidx and naming("nla", 42, c, 4, s) is not None]
        r1.append(mean(v))
    C["R1_probe_minus_NLA"] = verdict(boot(r1), +1)

    # R2: behaviour without report on u_perp at a reader dose
    r2i = {(r["concept"], str(r["cond"]), r["s"]): r.get("R2") for r in r2}
    has = []
    for c in nonleaky:
        ok = False
        for s in TEXT_S:
            beh = any(gt(M(c, 4, s, k), null95(k, s)) for k in ("B1", "B2", "B4"))
            rep_in = le(M(c, 4, s), null95("dR1p", s)) and le(r2i.get((c, "4", s)), r2i.get((c, "7", s)))
            ok = ok or (beh and rep_in)
        has.append(1.0 if ok else 0.0)
    C["R2_behaviour_without_report"] = verdict(boot(has), +1)

    # P11': onset index (first dose with SM >= 1 nat; never = 9), reader minus model, on u_perp (direct reads)
    def reader_onset(key, c):
        T = LT.get(key)
        if T is None:
            return None
        return onset_idx(lambda s: (T.get((c, "4", s, 0)) or {}).get("SM"))
    model_on = {c: onset_idx(lambda s, c=c: (sm_model(c, 4, s) or {}).get("SM")) for c in nonleaky}
    p11 = {}
    for name, keys in (("nla", ["nla_L42"]), ("oracle", ["oracle_L40", "oracle_L44"])):
        ok_keys = [k for k in keys if g11.get(k, {}).get("pass")]
        d = []
        for c in nonleaky:
            ons = [reader_onset(k, c) for k in ok_keys]
            ons = [o for o in ons if o is not None]
            if ons:
                d.append(min(ons) - model_on[c])
        p11[name] = dict(verdict(boot(d), -1), layers_used=ok_keys, uninformative=not ok_keys)
    C["P11prime_NLA"], C["P11prime_Oracle"] = p11["nla"], p11["oracle"]
    C["P11prime_onsets"] = {"model": model_on,
                            "readers": {k: {c: reader_onset(k, c) for c in nonleaky} for k in LT}}

    # P17: NLA minus max(Oracle L40, L44) naming of u_perp, direct reads, mean over the reader doses
    o_drop = {l: g6.get(f"oracle_L{l}", {}).get("dropped", []) for l in (40, 44)}
    p17 = []
    for c in nonleaky:
        if c in nla_drop or all(c in o_drop[l] for l in (40, 44)):
            continue
        v = []
        for s in TEXT_S:
            n_ = naming("nla", 42, c, 4, s)
            o_ = [naming("oracle", l, c, 4, s) for l in (40, 44) if c not in o_drop[l]]
            o_ = [x for x in o_ if x is not None]
            if n_ is not None and o_:
                v.append(n_ - max(o_))
        p17.append(mean(v))
    C["P17"] = verdict(boot(p17), +1)
    # P17 on the likelihood measure (secondary): NLA SM minus the larger Oracle SM, u_perp, mean over all doses
    p17l = []
    for c in nonleaky:
        v = []
        for s in GRID9:
            n_ = (LT.get("nla_L42", {}).get((c, "4", s, 0)) or {}).get("SM")
            o_ = [(LT.get(k, {}).get((c, "4", s, 0)) or {}).get("SM") for k in ("oracle_L40", "oracle_L44")]
            o_ = [x for x in o_ if x is not None]
            if n_ is not None and o_:
                v.append(n_ - max(o_))
        p17l.append(mean(v))
    C["P17_likelihood_secondary"] = verdict(boot(p17l), +1)
    C["holm_P_family"] = holm({"P11prime_NLA": C["P11prime_NLA"]["p_one_sided"], "P17": C["P17"]["p_one_sided"]})
    S["contrasts"] = C

    # ------------------------------------------------------------------------------------------ secondary tables
    T = {}
    T["model_curves"] = {str(cond): {str(s): {"dR1p_median": med([M(c, cond, s) for c in concepts]),
                                               "R1_median": med([M(c, cond, s, "R1") for c in concepts]),
                                               "B1_median": med([M(c, cond, s, "B1") for c in concepts]),
                                               "B4_median": med([M(c, cond, s, "B4") for c in concepts]),
                                               "SM_median": med([(sm_model(c, cond, s) or {}).get("SM") for c in concepts]),
                                               "ID_rate": mean([1.0 if (sm_model(c, cond, s) or {}).get("ID") else 0.0
                                                                for c in concepts if sm_model(c, cond, s)])}
                                     for s in GRID9} for cond in (1, 2, 3, 4, 5, 6, 8, 9, 10, "10r")}
    T["model_null95"] = {str(s): {k: null95(k, s) for k in ("dR1p", "R1", "B1", "B2", "B4")} for s in GRID9}
    T["R2"] = {str(cond): {str(s): mean([r2i.get((c, str(cond), s)) for c in concepts]) for s in TEXT_S} for cond in (1, 2, 3, 4, 7)}
    T["naming"] = {f"{name}_L{l}": {str(cond): {str(s): mean([naming(name, l, c, cond, s) for c in concepts
                                                              if c not in g6.get(f"{name}_L{l}", {}).get("dropped", [])])
                                                 for s in TEXT_S} for cond in (2, 3, 4, 7)}
                   for name, layers in reader_layers.items() for l in layers}
    T["likelihood_SM_median"] = {k: {str(cond): {str(s): med([(LT[k].get((c, str(cond), s, 0)) or {}).get("SM") for c in concepts])
                                                 for s in GRID9} for cond in (2, 3, 4, 7)} for k in LT}
    tok = {}
    for q in tpr:
        k = (q["layer"], q["kind"], str(q["cond"]), q["s"])
        d = tok.setdefault(k, {"n": 0, "probe_abs": 0, **{r: 0 for r in q["token"]}})
        d["n"] += 1
        d["probe_abs"] += 1 if q["probe"] > 0 else 0
        for rd, v in q["token"].items():
            d[rd] += 1 if v["rank"] <= 10 else 0
    T["token_probe_detection"] = {f"L{l}|{kind}|{cond}|{s}": {k: (v / d["n"] if k != "n" else v) for k, v in d.items()}
                                  for (l, kind, cond, s), d in tok.items()}
    sens = {}
    for f in sorted(D.glob("sens_*.json")):
        rows = json.loads(f.read_text())
        si = {(r["concept"], r["cond"], r["s"]): r for r in rows}
        pk = [max(si[(c, 3, s)]["dR1p"] for s in GRID9) - max(si[(c, 4, s)]["dR1p"] for s in GRID9) for c in nonleaky]
        sens[f.stem.replace("sens_", "")] = {"M1_R1p": verdict(boot(pk), +1),
                                             "J_minus_perp_median_by_dose": {str(s): med([si[(c, 3, s)]["dR1p"] - si[(c, 4, s)]["dR1p"]
                                                                                          for c in nonleaky]) for s in GRID9}}
    T["sensitivity"] = sens
    T["sensitivity_J_share_median"] = st.get("sensitivity_J_share_median")
    T["scoring_check_maxabs"] = st.get("scoring_check_maxabs")
    T["timings"], T["memory"], T["wall_minutes"] = st.get("timings"), st.get("memory"), st.get("wall_minutes")
    S["tables"] = T

    # raw readouts kept next to the numbers: per reader-layer, two fixed concepts, u_perp and the full vector at s = 1/4
    ex = {}
    for name, reads in readers.items():
        for q in reads:
            if q["concept"] in concepts[:2] and q["cond"] in (2, 4) and q["s"] == 0.25:
                ex.setdefault(f"{name}_L{q['layer']}", []).append({"concept": q["concept"], "cond": q["cond"],
                                                                    "samples": [t[:240] for t in q["samples"]]})
    S["examples"] = ex
    S["examples_R2"] = [{"concept": r["concept"], "cond": r["cond"], "s": r["s"], "texts": r.get("R2_texts", [])[:4]}
                        for r in r2 if r["concept"] in concepts[:2] and r["cond"] in (1, 2, 4) and r["s"] == 0.25]
    OUT.write_text(json.dumps(S, indent=1, default=str))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
