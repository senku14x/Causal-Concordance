"""005 follow-ups, analysis (docs/design_specs/005_followups.md; EXPLORATORY). CPU only.

  python followups_analysis.py [--run 20261005T015235Z] [--fu fu1]

A1 reader overlap and ensembles (full-run outputs), A2 ΔR1′ vs B2 (full-run rows; dev pilot rows secondary), A2b the
position profile, A3 cosine regression and own-atom shares, Q-c J atom vs logit-lens direction. Writes
results/005_planted_concepts/analysis/followups_summary.json.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p005 as P                                   # noqa: E402

GRID9 = [1 / 256, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
TEXT_S = [1 / 64, 1 / 4, 1.0]
MID = [1 / 16, 1 / 8, 1 / 4]
rng = np.random.default_rng(P.SEED)


def med(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.median(x)) if x else None


def mean(x):
    x = [v for v in x if v is not None]
    return float(np.mean(x)) if x else None


def boot(x):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) < 2:
        return {"n": int(len(x)), "mean": float(x.mean()) if len(x) else None, "ci": [None, None]}
    m = rng.choice(x, size=(10000, len(x)), replace=True).mean(1)
    return {"n": int(len(x)), "mean": float(x.mean()), "ci": [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]}


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = float(np.mean(a == b))
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return None if pe >= 1 else (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="20261005T015235Z")
    ap.add_argument("--fu", default="fu1")
    a = ap.parse_args()
    D = P.ROOT / "outputs/005_planted_concepts/run" / a.run
    F = P.ROOT / "outputs/005_planted_concepts/followups" / a.fu
    OUT = P.ROOT / "results/005_planted_concepts/analysis/followups_summary.json"
    S = {"note": "EXPLORATORY: the test concepts' 005 outcomes were seen before these analyses (docs/design_specs/005_followups.md)",
         "run": a.run, "followups": a.fu}
    model = json.loads((D / "model_rows.json").read_text())
    concepts = sorted({r["concept"] for r in model})
    M = {(r["concept"], str(r["cond"]), r["s"], r.get("seed", 0)): r for r in model}

    # ------------------------------------------------------------------------------------------------ A1
    tpr = json.loads((D / "token_probe_reads.json").read_text())
    T = {(q["concept"], str(q["cond"]), q["s"], q.get("seed", 0), q["layer"], q["kind"]): q for q in tpr}
    text = {}
    for name in ("nla", "oracle"):
        for q in json.loads((D / f"{name}_reads.json").read_text()):
            if q["cond"] == "clean":
                continue
            k = sum(P.names(t, q["concept"]) for t in q["samples"])
            text[(f"{name.upper() if name == 'nla' else 'Oracle'}_L{q['layer']}", q["concept"], str(q["cond"]), q["s"])] = k

    def det(reader, c, cond, s, kind="direct_cq", k_min=1):
        if reader in ("NLA_L42", "Oracle_L40", "Oracle_L44"):
            v = text.get((reader, c, cond, s))
            return None if (v is None or kind != "direct_cq") else v >= k_min
        fam, layer = reader.rsplit("_L", 1)
        q = T.get((c, cond, s, 0, int(layer), kind))
        if q is None:
            return None
        return q["probe"] > 0 if fam == "probe" else q["token"][fam]["rank"] <= 10
    layers = (40, 42, 44)
    allr = ["NLA_L42", "Oracle_L40", "Oracle_L44"] + [f"{f}_L{l}" for f in ("probe", "J_NP", "J_CB", "R_CB", "logit") for l in layers]
    panel = ["NLA_L42", "Oracle_L40", "Oracle_L44", "probe_L42", "J_NP_L42", "J_CB_L42", "R_CB_L42", "logit_L42"]
    A1 = {"readers": allr, "ensemble_panel": panel, "by_dose": {}}
    for k_min in (1, 2):
        for s in TEXT_S:
            key = f"k{k_min}|{s}"
            res = {"hit_rate": {}, "kappa_direct_nonJ": {}, "unique_catches_nonJ": {}, "ensembles": {}}
            for r_ in allr:
                res["hit_rate"][r_] = {cond: mean([float(det(r_, c, cond, s, k_min=k_min)) for c in concepts
                                                  if det(r_, c, cond, s, k_min=k_min) is not None]) for cond in ("2", "3", "4", "7")}
            for x, y in itertools.combinations(panel, 2):
                vx = [det(x, c, "4", s, k_min=k_min) for c in concepts]
                vy = [det(y, c, "4", s, k_min=k_min) for c in concepts]
                pairs = [(p, q) for p, q in zip(vx, vy) if p is not None and q is not None]
                if pairs:
                    px, py = zip(*pairs)
                    res["kappa_direct_nonJ"][f"{x}|{y}"] = {"kappa": kappa(px, py), "both": int(sum(p and q for p, q in pairs)),
                                                            "x_only": int(sum(p and not q for p, q in pairs)),
                                                            "y_only": int(sum(q and not p for p, q in pairs)), "n": len(pairs)}
            for r_ in panel:
                res["unique_catches_nonJ"][r_] = int(sum(1 for c in concepts if det(r_, c, "4", s, k_min=k_min)
                                                         and not any(det(o, c, "4", s, k_min=k_min) for o in panel if o != r_)))
            for cond in ("2", "3", "4", "7"):
                hits = {r_: [det(r_, c, cond, s, k_min=k_min) for c in concepts] for r_ in panel}
                union = [float(any(hits[r_][i] for r_ in panel)) for i in range(len(concepts))]
                major = [float(sum(bool(hits[r_][i]) for r_ in panel) >= len(panel) / 2) for i in range(len(concepts))]
                rates = {r_: float(np.mean([bool(v) for v in hits[r_]])) for r_ in panel}
                best = max(rates, key=rates.get)
                bvec = [float(bool(v)) for v in hits[best]]
                res["ensembles"][cond] = {"union": float(np.mean(union)), "majority": float(np.mean(major)), "best_single": best,
                                          "best_rate": rates[best],
                                          "union_minus_best": boot([u - b for u, b in zip(union, bvec)]),
                                          "majority_minus_best": boot([m - b for m, b in zip(major, bvec)])}
            A1["by_dose"][key] = res
    carried = {}
    for kind in ("carried_co", "injected_cq"):
        for s in TEXT_S:
            carried[f"{kind}|{s}"] = {f"{f}_L{l}": {cond: mean([float(det(f'{f}_L{l}', c, cond, s, kind=kind)) for c in concepts])
                                                   for cond in ("2", "4", "7")}
                                     for f in ("probe", "J_NP", "J_CB", "R_CB", "logit") for l in layers}
    A1["token_probe_carried_and_injected"] = carried
    S["A1"] = A1

    # ------------------------------------------------------------------------------------------------ A2
    A2 = {"test": {}, "dev_pilot": {}}
    rhos = []
    for cond in ("1", "2", "3", "4"):
        for s in GRID9:
            x = [M[(c, cond, s, 0)]["dR1p"] for c in concepts]
            y = [M[(c, cond, s, 0)]["B2"] for c in concepts]
            rho = float(stats.spearmanr(x, y).correlation)
            A2["test"][f"{cond}|{s}"] = {"rho": rho, "median_dR1p": med(x), "median_B2": med(y),
                                         "B2_minus_dR1p": boot([b - d for b, d in zip(y, x)])}
            if s in (1 / 16, 1 / 8, 1 / 4, 1 / 2):
                rhos.append(rho)
    A2["median_rho_rule_cells"] = float(np.median(rhos))
    A2["tracks_closely"] = A2["median_rho_rule_cells"] >= 0.7
    pil = P.ROOT / "outputs/005_planted_concepts/pilot/pilot1/rows_B_default.json"
    if pil.exists():
        pr = json.loads(pil.read_text())
        PM = {(r["concept"], str(r["cond"]), r["s"]): r for r in pr if r.get("seed", 0) == 0}
        dv = sorted({r["concept"] for r in pr})
        for cond in ("1", "2", "3", "4"):
            for s in GRID9:
                x = [PM[(c, cond, s)]["dR1p"] for c in dv if (c, cond, s) in PM]
                y = [PM[(c, cond, s)]["B2"] for c in dv if (c, cond, s) in PM]
                if len(x) > 2:
                    A2["dev_pilot"][f"{cond}|{s}"] = {"rho": float(stats.spearmanr(x, y).correlation), "n": len(x)}
    S["A2"] = A2

    # ----------------------------------------------------------------------------------------------- A2b
    a2b = json.loads((F / "a2b_rows.json").read_text())
    B = {(r["concept"], str(r["cond"]), r["s"]): r for r in a2b}
    A2b = {"by_cond": {}, "profile_median": {}}
    nonans = [str(p) for p in range(126, 138)]
    for cond in ("1", "2", "3", "4", "7"):
        per = []
        for c in concepts:
            v = []
            for s in MID:
                d = B[(c, cond, s)]["delta_by_pos"]
                v.append(d["138"] - float(np.median([d[p] for p in nonans])))
            per.append(float(np.mean(v)))
        b = boot(per)
        A2b["by_cond"][cond] = dict(b, slot_specific=b["ci"][0] is not None and b["ci"][0] > 0,
                                    median_answer_slot={str(s): med([B[(c, cond, s)]["delta_by_pos"]["138"] for c in concepts]) for s in GRID9},
                                    median_nonanswer={str(s): med([float(np.median([B[(c, cond, s)]["delta_by_pos"][p] for p in nonans]))
                                                                   for c in concepts]) for s in GRID9})
        A2b["profile_median"][cond] = {str(s): {p: med([B[(c, cond, s)]["delta_by_pos"][p] for c in concepts])
                                                for p in [str(q) for q in range(125, 139)]} for s in (1 / 8, 1 / 4)}
    A2b["consistency_vs_full_run_dR1p_maxabs"] = float(max(abs(B[(c, cond, s)]["dR1p"] - M[(c, cond, s, 0)]["dR1p"])
                                                            for c in concepts for cond in ("1", "2", "3", "4") for s in GRID9))
    S["A2b"] = A2b
    S["A2_naming"] = ("lexical priming at the answer slot" if (A2["tracks_closely"] or not A2b["by_cond"]["2"]["slot_specific"])
                      else "report measure (tracks priming loosely; slot-specific component present)")

    # ------------------------------------------------------------------------------------------------ A3
    geo = json.loads((F / "a3_geometry.json").read_text())
    A3 = {}
    conds = ["1", "2", "3", "4", "7", "8", "9"]
    for which in ("cos_quote", "cos_space"):
        for s in (1 / 8, 1 / 4):
            X, Y, G = [], [], []
            for c in concepts:
                for cond in conds:
                    X.append(geo[c][which][cond])
                    Y.append(M[(c, cond, s, 0)]["dR1p"])
                    G.append(cond)
            X, Y = np.array(X), np.array(Y)
            A1m = np.column_stack([np.ones_like(X), X])
            beta, *_ = np.linalg.lstsq(A1m, Y, rcond=None)
            r2 = 1 - np.sum((Y - A1m @ beta) ** 2) / np.sum((Y - Y.mean()) ** 2)
            dums = np.column_stack([np.array([g == k for g in G], float) for k in conds[1:]])
            A2m = np.column_stack([A1m, dums])
            beta2, *_ = np.linalg.lstsq(A2m, Y, rcond=None)
            rss1, rss2 = np.sum((Y - A1m @ beta) ** 2), np.sum((Y - A2m @ beta2) ** 2)
            r2b = 1 - rss2 / np.sum((Y - Y.mean()) ** 2)
            df1, df2 = A2m.shape[1] - A1m.shape[1], len(Y) - A2m.shape[1]
            Fst = ((rss1 - rss2) / df1) / (rss2 / df2)
            within = {}
            for k in conds:
                idx = [i for i, g in enumerate(G) if g == k]
                if np.std(X[idx]) > 0:
                    sl = stats.linregress(X[idx], Y[idx])
                    within[k] = {"slope": float(sl.slope), "r": float(sl.rvalue), "median_cos": float(np.median(X[idx]))}
            A3[f"{which}|{s}"] = {"slope": float(beta[1]), "R2": float(r2), "R2_with_conditions": float(r2b),
                                  "delta_R2": float(r2b - r2), "F": float(Fst), "F_p": float(1 - stats.f.cdf(Fst, df1, df2)),
                                  "n": int(len(Y)), "within_condition": within,
                                  "explains": bool(r2 >= 0.5 and (r2b - r2 < 0.05 or 1 - stats.f.cdf(Fst, df1, df2) >= 0.05))}
    A3["rule_met_primary"] = A3["cos_quote|0.125"]["explains"] and A3["cos_quote|0.25"]["explains"]
    own = []
    for c in concepts:
        la = geo[c]["own_atom"]
        own.append({"concept": c, "selected_layers": int(sum(x["selected"] for x in la)),
                    "coef_mean": float(np.mean([x["coef"] for x in la])),
                    "energy_share_mean": float(np.mean([x["energy_share"] or 0 for x in la])),
                    "cos2_uJ_atom_mean": float(np.mean([x["cos2_uJ_atom"] for x in la]))})
    A3["own_atom"] = {"per_concept": own, "share_selected_all_band_layers": float(np.mean([o["selected_layers"] == 7 for o in own])),
                      "median_energy_share": med([o["energy_share_mean"] for o in own]),
                      "median_cos2": med([o["cos2_uJ_atom_mean"] for o in own])}
    S["A3"] = A3

    # ------------------------------------------------------------------------------------------------ Q-c
    qc = json.loads((F / "qc_rows.json").read_text())
    Q = {(r["concept"], str(r["cond"]), r["s"], r.get("seed", 0)): r for r in qc}
    qcs = sorted({(r["concept"], r["split"]) for r in qc})
    QC = {}
    for split in ("all", "dev", "test"):
        cs = [c for c, sp_ in qcs if split == "all" or sp_ == split]
        d = [float(np.mean([Q[(c, "1", s, 0)]["dR1p"] - Q[(c, "L", s, 0)]["dR1p"] for s in MID])) for c in cs]
        b = boot(d)
        QC[split] = dict(b, J_lens_adds_something=b["ci"][0] is not None and (b["ci"][0] > 0 or b["ci"][1] < 0))
    QC["curves"] = {cond: {str(s): {"dR1p": med([Q[(c, cond, s, 0)]["dR1p"] for c, _ in qcs]),
                                    "R1": med([Q[(c, cond, s, 0)]["R1"] for c, _ in qcs]),
                                    "B1": med([Q[(c, cond, s, 0)]["B1"] for c, _ in qcs]),
                                    "B2": med([Q[(c, cond, s, 0)]["B2"] for c, _ in qcs]),
                                    "B3": med([Q[(c, cond, s, 0)]["B3"] for c, _ in qcs])} for s in GRID9}
                    for cond in ("1", "L", "2")}
    QC["null95_dR1p"] = {str(s): float(np.quantile([r["dR1p"] for r in qc if r["cond"] == 7 and r["s"] == s], 0.95)) for s in GRID9}
    QC["broken_doses_B1_over_5"] = [f"{cond}|{s}" for cond in ("1", "L", "2") for s in GRID9 if (QC["curves"][cond][str(s)]["B1"] or 0) > 5]
    QC["consistency_cond1_vs_full_run_maxabs"] = float(max(abs(Q[(c, "1", s, 0)]["dR1p"] - M[(c, "1", s, 0)]["dR1p"])
                                                            for c, sp_ in qcs if sp_ == "test" for s in GRID9))
    S["Qc"] = QC
    fst = json.loads((F / "state.json").read_text())
    S["G0"], S["timings"], S["qc_concepts"] = fst.get("G0"), fst.get("timings"), fst.get("qc_concepts")
    OUT.write_text(json.dumps(S, indent=1, default=str))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
