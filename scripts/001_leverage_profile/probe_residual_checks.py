"""Design 001 amendment A2, residual checks requested by the external review of design 002 (point 7).

1. Span vs full solver across the whole C grid, on real outer folds: L2-logistic objective at each solution, held-out
   prediction agreement and metric, and whether either solver stopped at max_iter.
2. 12-way permutation null: the A2 null permutes labels across renders. Labels and folds are grouped by table, so a
   grouped alternative permutes whole tables (blocks of equal size carry their label vectors) and keeps the label
   clustering. Both nulls are run through the full procedure on the borderline targets; their means and 95th
   percentiles are compared with the real score.

    python scripts/001_leverage_profile/probe_residual_checks.py  ->  results/001_leverage_profile/analysis/probe_residual_checks.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import leverage_profile as LP  # noqa: E402

N_NULL = 100


def _data():
    out = LP.OUT / "run"
    X = np.load(out / "captures_brew.npz")["X"]
    meta = json.loads((out / "captures_brew_meta.json").read_text())
    return X, meta


def _objective(model, Xtr, ytr, C):
    """sklearn's L2-logistic objective at the fitted solution: 0.5||w||^2 + C * sum(log-loss), in the scaled space."""
    from sklearn.metrics import log_loss
    if isinstance(model, LP._SpanLR):
        w = model.lr.coef_ @ model.V.T
        proba = model.lr.predict_proba(model._z(Xtr))
        classes = model.lr.classes_
    else:
        lr = model[-1]
        w = lr.coef_
        proba = model.predict_proba(Xtr)
        classes = lr.classes_
    return float(0.5 * (w ** 2).sum() + C * log_loss(ytr, proba, labels=classes, normalize=False))


def solver_check(X, meta):
    from sklearn.model_selection import StratifiedGroupKFold
    R_ = meta["renders"]
    g_all = np.array([r["item"] for r in R_])
    nat = np.array([r["role"] in ("member", "partner") for r in R_])
    cells, layers = meta["cells"], meta["layers"]
    targets = [("answer", "emit_colon", 40), ("z", "emit_colon", 44), ("answer", "emit_asst", 44)]
    out = []
    for tgt, cell, L in targets:
        Xs = X[nat, cells.index(cell), layers.index(L)]
        y = np.array([str(r[tgt]) for r in R_])[nat]
        g = g_all[nat]
        for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=LP.QUAL_SEED).split(Xs, y, g)):
            if k > 1:
                break
            for C in LP.C_GRID:
                row = {"target": tgt, "cell": cell, "layer": L, "fold": k, "C": C}
                for name, fit in (("full", LP._fit_full), ("span", LP._fit)):
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        m = fit(Xs[tr], y[tr], C, True)
                    row[f"{name}_objective"] = _objective(m, Xs[tr], y[tr], C)
                    row[f"{name}_heldout_acc"] = float(np.mean(m.predict(Xs[te]) == y[te]))
                    row[f"{name}_convergence_warning"] = any("converge" in str(x.message).lower() for x in w)
                    row[f"_{name}_pred"] = m.predict(Xs[te])
                row["heldout_pred_agreement"] = float(np.mean(row.pop("_full_pred") == row.pop("_span_pred")))
                row["objective_rel_diff_span_minus_full"] = (row["span_objective"] - row["full_objective"]) / abs(row["full_objective"])
                out.append(row)
                print(row, flush=True)
    return out


def _grouped_perm(y, g, rng):
    """Permute whole tables among tables with the same number of renders; each table carries its label vector."""
    tabs = sorted(set(g))
    by_size = {}
    for t in tabs:
        by_size.setdefault(int((g == t).sum()), []).append(t)
    mapping = {}
    for size, ts in by_size.items():
        perm = rng.permutation(ts)
        mapping.update(dict(zip(ts, perm)))
    yp = y.copy()
    for t in tabs:
        src = mapping[t]
        yp[g == t] = y[g == src]           # same size, so the blocks align render-for-render in stored order
    return yp


def null_check(X, meta):
    from joblib import Parallel, delayed
    R_ = meta["renders"]
    g_all = np.array([r["item"] for r in R_])
    nat = np.array([r["role"] in ("member", "partner") for r in R_])
    cells, layers = meta["cells"], meta["layers"]
    targets = [("answer", "emit_colon", 40), ("answer", "emit_colon", 44), ("answer", "emit_asst", 40), ("z", "emit_colon", 42)]

    def one(Xs, y, g, kind, i):
        rng = np.random.default_rng(LP._seed("resid", kind, i))
        yp = rng.permutation(y) if kind == "row" else _grouped_perm(y, g, rng)
        try:
            pred, _, _, _ = LP._cv_scores(Xs, yp, g, True, seed=LP.QUAL_SEED)
        except AssertionError:
            return None
        return LP._metric(yp, pred, True)

    res = []
    for tgt, cell, L in targets:
        Xs = X[nat, cells.index(cell), layers.index(L)]
        y = np.array([str(r[tgt]) for r in R_])[nat]
        g = g_all[nat]
        pred, _, _, _ = LP._cv_scores(Xs, y, g, True, seed=LP.QUAL_SEED)
        real = LP._metric(y, pred, True)
        sizes = {}
        for t in set(g):
            sizes.setdefault(int((g == t).sum()), []).append(t)
        row = {"target": tgt, "cell": cell, "layer": L, "real": real,
               "grouped_null_tables_fixed_in_singleton_size_classes": int(sum(len(v) for v in sizes.values() if len(v) == 1)),
               "n_tables": len(set(g)), "size_classes": {k: len(v) for k, v in sorted(sizes.items())}}
        for kind in ("row", "grouped"):
            vals = Parallel(n_jobs=100)(delayed(one)(Xs, y, g, kind, i) for i in range(N_NULL))
            v = np.array([x for x in vals if x is not None])
            row[kind] = {"n_valid": int(len(v)), "n_failed": int(N_NULL - len(v)), "mean": float(v.mean()), "sd": float(v.std()),
                         "q95": float(np.quantile(v, 0.95)), "p": float((1 + (v >= real).sum()) / (1 + len(v)))}
        res.append(row)
        print(row, flush=True)
    return res


def main():
    t0 = time.time()
    X, meta = _data()
    out = {"solver_check": solver_check(X, meta), "null_check": null_check(X, meta), "seconds": None,
           "note": "exploratory diagnostics for amendment A2; see the review of design 002, point 7"}
    out["seconds"] = time.time() - t0
    (LP.ANA / "probe_residual_checks.json").write_text(json.dumps(out, indent=1, default=float))
    print("done", out["seconds"])


if __name__ == "__main__":
    main()
