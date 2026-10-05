"""007 geometry (CPU only, reader-side: uses the J_NP lens; never imported by label code).

Stages (each writes JSON under results/007_jsubspace/analysis/; bases go to outputs/007_jsubspace/bases/):
  bases       J subspaces at L40/42/44 under two conventions (raw W_U, as Cathryn's make_jbasis.py; and W_U * (1 + w),
              the final-norm gain), plus the logit-lens subspaces (J = I) under both conventions.
              Construction (her file's): d_t = W_c[t] @ J_L over the whole vocabulary, rows unit-normalised,
              G = sum_t unit(d_t) unit(d_t)^T (float32 chunk products accumulated in float64), eigenvectors, largest first.
  crosscheck  The advisor's geometry numbers on her 50 saved multihop residuals (runs/abl50/resid_L42.pt), recomputed:
              a positive control for this file's metric code (no forward passes).
  overlaps    Subspace overlaps: ours vs her file; raw vs gain; across layers; J vs logit-lens; J vs her PCA-1,024
              (runs/cw12/pcabasis.pt @ her 04484b2); J vs a within-table PCA of the 001 Brew captures.
  brew        On the 001 Brew captures (48 dev tables, 48 member + 183 partner renders), cells ':' and '</think>',
              L40/42/44: per piece, energy share, item fraction, item variance held, share of the member-partner
              difference, the reader-visible (unit-normalised) difference ratio, and a simple cross-fitted z probe.

Overlap = mean squared cosine of the principal angles between two orthonormal bases (= ||A^T B||_F^2 / min(a, b));
two random subspaces of dims a <= b in R^d give b / d in expectation (0.2 for 1024 in 5120).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts" / "common"))
OUT = PROJECT / "outputs" / "007_jsubspace"
RES = PROJECT / "results" / "007_jsubspace" / "analysis"
BASES = OUT / "bases"
CW = Path("/workspace/sources/repos/nla-vs-lenses")
CW_BASIS = CW / "jbasis" / "jbasis_L42_qwen36_27b.pt"                      # @ 9a46bf9
CW_RESID = CW / "runs" / "abl50" / "resid_L42.pt"                          # @ 9a46bf9
CW_PCA = OUT / "inputs" / "cw12_pcabasis.pt"                               # git show 04484b2:runs/cw12/pcabasis.pt
CAPS = PROJECT / "outputs" / "001_leverage_profile" / "run" / "captures_brew.npz"
CAPS_META = PROJECT / "outputs" / "001_leverage_profile" / "run" / "captures_brew_meta.json"
LAYERS = (40, 42, 44)
K = 1024
KEEP = 2560
D = 5120
CHUNK = 16384
SEED = 20261005
FINE_K = False
OUT_NAME = "geometry_brew.json"
torch.set_num_threads(32)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}]", *a, flush=True)


def dump(name, obj):
    RES.mkdir(parents=True, exist_ok=True)
    p = RES / name
    p.write_text(json.dumps(obj, indent=1))
    log("wrote", p)


# ------------------------------------------------------------------------------------------------ helpers
def orth(M):
    """Orthonormal basis of the column span (QR in float64), same column order."""
    q, r = torch.linalg.qr(M.double())
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def overlap(A, B):
    """A [d, a], B [d, b] orthonormal. Mean cos^2 over the min(a, b) principal angles, and summaries."""
    s = torch.linalg.svdvals(A.double().T @ B.double()).clamp(max=1.0)
    c2 = s ** 2
    return {"mean_cos2": float(c2.mean()), "min_cos2": float(c2.min()), "frac_cos2_gt_0.9": float((c2 > 0.9).double().mean()),
            "frac_cos2_lt_0.1": float((c2 < 0.1).double().mean()), "dims": [A.shape[1], B.shape[1]],
            "random_expectation": max(A.shape[1], B.shape[1]) / A.shape[0]}


def load_W_and_gain():
    from safetensors import safe_open
    snap = next((Path("/workspace/.hf_home/hub/models--Qwen--Qwen3.6-27B/snapshots")).iterdir())
    idx = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(snap / idx["lm_head.weight"]), framework="pt") as f:
        W = f.get_tensor("lm_head.weight").float()
    with safe_open(str(snap / idx["model.language_model.norm.weight"]), framework="pt") as f:
        w = f.get_tensor("model.language_model.norm.weight").float()
    cfg = json.loads((snap / "config.json").read_text())
    assert cfg.get("tie_word_embeddings") is False and cfg["text_config"].get("tie_word_embeddings") is False
    assert W.shape == (248320, D) and w.shape == (D,), (W.shape, w.shape)
    return W, 1.0 + w                                                       # Qwen3_5RMSNorm: gain = 1 + weight


def gram(Wc, J):
    G = torch.zeros(D, D, dtype=torch.float64)
    for i in range(0, Wc.shape[0], CHUNK):
        Dm = Wc[i:i + CHUNK] if J is None else Wc[i:i + CHUNK] @ J
        Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp_min(1e-9)
        G += (Dm.T @ Dm).double()
    return G


def eig_basis(G):
    ev, Q = torch.linalg.eigh(G)
    ev, Q = ev.flip(0), Q.flip(1)
    return ev, Q


def load_basis(name, k=K):
    b = torch.load(BASES / f"{name}.pt", weights_only=False)
    return orth(b["Q"][:, :k].double())


def cw_basis(k=K):
    b = torch.load(CW_BASIS, weights_only=False)
    return orth(b["Q"].float()[:, :k]), b["ev"]


def cw_pca(k=K):
    b = torch.load(CW_PCA, weights_only=False)
    return orth(b["P"].float()[:, :k]), b


def rand_basis(seed, k=K):
    g = torch.Generator().manual_seed(seed)
    return orth(torch.randn(D, k, generator=g, dtype=torch.float64))


# ------------------------------------------------------------------------------------------------ stage: bases
def stage_bases():
    import readers
    t0 = time.time()
    W, g = load_W_and_gain()
    Wg = W * g.unsqueeze(0)
    lens = readers.load_lenses(("J_NP",), layers=list(LAYERS), device="cpu", verify_sha=True)["J_NP"]
    BASES.mkdir(parents=True, exist_ok=True)
    meta = {"J_NP": "readers.load_lenses (sha256-verified, registry pin)", "W": "lm_head.weight (untied)",
            "gain": "1 + model.language_model.norm.weight", "construction": "eig of sum_t unit(W_c[t] @ J) outer itself",
            "bases": {}}
    jobs = [(f"J_{c}_L{L}", (W if c == "raw" else Wg), lens.jacobians[L]) for L in LAYERS for c in ("raw", "gain")]
    jobs += [("LL_raw", W, None), ("LL_gain", Wg, None)]
    for name, Wc, J in jobs:
        t = time.time()
        ev, Q = eig_basis(gram(Wc, J))
        share = (ev.cumsum(0) / ev.sum())
        torch.save({"Q": Q[:, :KEEP].float(), "ev": ev}, BASES / f"{name}.pt")
        meta["bases"][name] = {"share_top": {str(k): float(share[k - 1]) for k in (16, 64, 256, 1024, 2560)},
                               "trace": float(ev.sum()), "seconds": round(time.time() - t, 1)}
        log(name, meta["bases"][name])
    meta["seconds_total"] = round(time.time() - t0, 1)
    dump("geometry_bases.json", meta)


# ------------------------------------------------------------------------------------------------ piece metrics
def piece_stats(Pfun, H, mu_loo):
    """Pfun(X) projects rows of X. H [n, d]; mu_loo [n, d] leave-one-out (or leave-one-table-out) means."""
    PH = Pfun(H)
    E = (PH ** 2).sum(1) / (H ** 2).sum(1)
    C = H - mu_loo
    PC = Pfun(C)
    item_frac = (PC ** 2).sum(1) / (PH ** 2).sum(1).clamp_min(1e-30)
    item_held = (PC ** 2).sum(1) / (C ** 2).sum(1)
    return E, item_frac, item_held


def proj(Q):
    return lambda X: (X @ Q) @ Q.T


def comp(Q):
    return lambda X: X - (X @ Q) @ Q.T


def med(x):
    return float(np.median(np.asarray(x)))


def summ(x):
    x = np.asarray(x, dtype=np.float64)
    return {"median": float(np.median(x)), "q25": float(np.quantile(x, 0.25)), "q75": float(np.quantile(x, 0.75)), "n": int(x.size)}


# ------------------------------------------------------------------------------------------------ stage: crosscheck
def stage_crosscheck():
    cells = [c for c in torch.load(CW_RESID, weights_only=False) if c["layer"] == 42]
    H = torch.stack([c["h"].double() for c in cells])                       # [50, d]
    n = H.shape[0]
    mu_loo = (H.sum(0, keepdim=True) - H) / (n - 1)
    out = {"n": n, "positions": sorted({int(c["pos"]) for c in cells}), "pieces": {}}
    Qcw, _ = cw_basis()
    pieces = {"whole": lambda X: X, "J_cw": proj(Qcw), "rest_J_cw": comp(Qcw),
              "J_raw_L42": proj(load_basis("J_raw_L42")), "J_gain_L42": proj(load_basis("J_gain_L42")),
              "LL_raw": proj(load_basis("LL_raw")), "rest_LL_raw": comp(load_basis("LL_raw")),
              "LL_gain": proj(load_basis("LL_gain")), "rest_LL_gain": comp(load_basis("LL_gain"))}
    for nm, P in pieces.items():
        E, F, V = piece_stats(P, H, mu_loo)
        out["pieces"][nm] = {"energy": med(E), "item_held": med(V), "item_frac": med(F)}
    rs = {"energy": [], "item_held": [], "item_frac": [], "rest_energy": [], "rest_item_held": [], "rest_item_frac": []}
    for s in range(10):
        Qr = rand_basis(SEED + s)
        for pre, P in (("", proj(Qr)), ("rest_", comp(Qr))):
            E, F, V = piece_stats(P, H, mu_loo)
            rs[pre + "energy"].append(med(E)); rs[pre + "item_held"].append(med(V)); rs[pre + "item_frac"].append(med(F))
    out["pieces"]["random_1024 (median of 10 draws)"] = {k: med(rs[k]) for k in ("energy", "item_held", "item_frac")}
    out["pieces"]["all_but_random (median of 10 draws)"] = {k.replace("rest_", ""): med(rs[k]) for k in ("rest_energy", "rest_item_held", "rest_item_frac")}
    mu = H.mean(0)
    out["mean_norm2_over_h_norm2_median"] = med(((mu.norm() ** 2) / (H ** 2).sum(1)).numpy())          # ||mu||^2 / ||h||^2
    out["cos2_h_mu_median"] = med((((H @ mu) ** 2) / ((H ** 2).sum(1) * (mu ** 2).sum())).numpy())   # h's energy along mu
    out["mean_in_J_cw_share"] = float(((Qcw.T @ mu) ** 2).sum() / (mu ** 2).sum())
    out["advisor_reported"] = {"whole": [1.00, 1.00, 0.32], "J": [0.15, 0.32, 0.62], "rest_J": [0.84, 0.68, 0.25],
                               "random": [0.20, 0.20, 0.31], "all_but_random": [0.80, 0.80, 0.31],
                               "LL": [0.10, 0.18, 0.50], "rest_LL": [0.90, 0.82, 0.29],
                               "order": ["energy", "item_held", "item_frac"], "mean_energy": 0.69, "mean_in_J": 0.085}
    dump("geometry_crosscheck.json", out)


# ------------------------------------------------------------------------------------------------ brew material
def brew_material():
    X = np.load(CAPS, mmap_mode="r")["X"]
    meta = json.loads(CAPS_META.read_text())
    assert meta["layers"] == [40, 42, 44, 56, 60] and meta["cells"] == ["emit_asst", "emit_think", "emit_colon"], meta["cells"]
    R = meta["renders"]
    keep = [i for i, r in enumerate(R) if r["role"] in ("member", "partner")]
    assert len(keep) == 231, len(keep)
    tables = sorted({R[i]["item"] for i in keep})
    assert len(tables) == 48
    t_of = np.array([tables.index(R[i]["item"]) for i in keep])
    role = [R[i]["role"] for i in keep]
    z = [R[i]["z"] for i in keep]
    mem = {t_of[j]: j for j in range(len(keep)) if role[j] == "member"}
    pairs = [(mem[t_of[j]], j) for j in range(len(keep)) if role[j] == "partner"]
    assert len(mem) == 48 and len(pairs) == 183
    return X, keep, t_of, z, pairs


def table_folds(n_tables, n_folds=4, seed=SEED):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_tables)
    return [set(perm[f::n_folds].tolist()) for f in range(n_folds)]


def within_table_pca(H, t_of, rows):
    """004-style: within-table centred, rows weighted 1/sqrt(n_t). Returns orthonormal right singular vectors [d, r]."""
    rows = np.asarray(rows)
    Hs = H[rows]
    ts = t_of[rows]
    C = torch.empty_like(Hs)
    for t in np.unique(ts):
        m = torch.from_numpy(ts == t)
        C[m] = (Hs[m] - Hs[m].mean(0, keepdim=True)) / float(np.sqrt(int(m.sum())))
    _, S, Vh = torch.linalg.svd(C, full_matrices=False)
    r = int((S > S[0] * 1e-8).sum())
    return Vh[:r].T.contiguous(), S[:r]


# ------------------------------------------------------------------------------------------------ stage: overlaps
def stage_overlaps():
    out = {}
    Qcw, ev_cw = cw_basis()
    b42 = torch.load(BASES / "J_raw_L42.pt", weights_only=False)
    out["ours_raw_L42_vs_cw"] = overlap(load_basis("J_raw_L42"), Qcw)
    out["ours_gain_L42_vs_cw"] = overlap(load_basis("J_gain_L42"), Qcw)
    ev_ours = b42["ev"]
    out["spectrum_raw_L42_vs_cw"] = {"max_abs_rel_diff_top2560": float(((ev_ours[:KEEP] - ev_cw[:KEEP]).abs() / ev_cw[:KEEP].abs().clamp_min(1e-12)).max()),
                                     "trace_ours": float(ev_ours.sum()), "trace_cw": float(ev_cw.sum()),
                                     "share_top1024_ours": float(ev_ours[:K].sum() / ev_ours.sum()),
                                     "share_top1024_cw": float(ev_cw[:K].sum() / ev_cw.sum()),
                                     "eig_gap_at_1024_ours": [float(ev_ours[K - 1]), float(ev_ours[K])]}
    for L in LAYERS:
        out[f"raw_vs_gain_L{L}"] = overlap(load_basis(f"J_raw_L{L}"), load_basis(f"J_gain_L{L}"))
        for c in ("raw", "gain"):
            out[f"J_{c}_L{L}_vs_LL_raw"] = overlap(load_basis(f"J_{c}_L{L}"), load_basis("LL_raw"))
            out[f"J_{c}_L{L}_vs_LL_gain"] = overlap(load_basis(f"J_{c}_L{L}"), load_basis("LL_gain"))
    for c in ("raw", "gain"):
        out[f"J_{c}_L40_vs_L42"] = overlap(load_basis(f"J_{c}_L40"), load_basis(f"J_{c}_L42"))
        out[f"J_{c}_L44_vs_L42"] = overlap(load_basis(f"J_{c}_L44"), load_basis(f"J_{c}_L42"))
        out[f"J_{c}_L40_vs_L44"] = overlap(load_basis(f"J_{c}_L40"), load_basis(f"J_{c}_L44"))
    out["LL_raw_vs_LL_gain"] = overlap(load_basis("LL_raw"), load_basis("LL_gain"))
    out["random_vs_random"] = overlap(rand_basis(SEED), rand_basis(SEED + 1))
    Pcw, pmeta = cw_pca()
    out["cw_pca_meta"] = {"n": int(pmeta["n"]), "her_reported_overlap_with_her_J": float(pmeta["overlap"]),
                          "her_var_in_J": float(pmeta["var_in_J"]), "var_top1024": float(pmeta["var"][:K].sum())}
    out["cw_vs_cw_pca (recomputed)"] = overlap(Qcw, Pcw)
    for c in ("raw", "gain"):
        out[f"J_{c}_L42_vs_cw_pca"] = overlap(load_basis(f"J_{c}_L42"), Pcw)
    out["LL_gain_vs_cw_pca"] = overlap(load_basis("LL_gain"), Pcw)
    # J vs within-table PCA of the 001 Brew captures (fitted on all 231 renders per cell and layer)
    X, keep, t_of, z, pairs = brew_material()
    out["brew_within_table_pca"] = {}
    for ci, cell in ((2, "colon"), (1, "think")):
        for li, L in enumerate(LAYERS):
            H = torch.from_numpy(np.asarray(X[keep, ci, li], dtype=np.float64))
            V, S = within_table_pca(H, t_of, range(len(keep)))
            var = (S ** 2) / (S ** 2).sum()
            rec = {"rank": V.shape[1]}
            for nm in (f"J_raw_L{L}", f"J_gain_L{L}", "LL_raw", "LL_gain"):
                QJ = load_basis(nm)
                r_ = {}
                for r in (11, 33, 64, V.shape[1]):
                    o = overlap(V[:, :r], QJ)
                    r_[f"top{r}"] = o["mean_cos2"]                              # share of the PCA-r subspace inside J
                cJ = ((V.T @ QJ) ** 2).sum(1)                                       # per PC, share inside J
                r_["variance_weighted"] = float((var * cJ).sum())                   # share of within-table variance in J
                rec[nm] = r_
            rec["cw_pca"] = {f"top{r}": overlap(V[:, :r], Pcw)["mean_cos2"] for r in (11, 33, 64, V.shape[1])}
            rec["cw_pca"]["variance_weighted"] = float((var * ((V.T @ Pcw) ** 2).sum(1)).sum())
            rec["random_expectation"] = K / D
            out["brew_within_table_pca"][f"{cell}_L{L}"] = rec
            log("brew pca overlap", cell, L, {k: v for k, v in rec.items() if k in ("rank",)})
    dump("geometry_overlaps.json", out)


# ------------------------------------------------------------------------------------------------ stage: brew
def z_probe(Fe, z, t_of, folds, seed=SEED, n_perm=2):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    y = np.array(z)

    def run(labels):
        pred = np.empty(len(labels), dtype=object)
        for f in folds:
            te = np.array([t in f for t in t_of])
            sc = StandardScaler().fit(Fe[~te])
            clf = LogisticRegression(C=0.05, max_iter=4000).fit(sc.transform(Fe[~te]), labels[~te])
            pred[te] = clf.predict(sc.transform(Fe[te]))
        return balanced_accuracy_score(labels, pred)
    real = run(y)
    rng = np.random.default_rng(seed)
    tz = {t: y[t_of == t] for t in np.unique(t_of)}
    perm = []
    for _ in range(n_perm):                                                      # permute whole tables' label sets
        ts = list(tz)
        sh = rng.permutation(ts)
        yp = y.copy()
        for a, b in zip(ts, sh):
            ia, ib = np.where(t_of == a)[0], np.where(t_of == b)[0]
            m = min(len(ia), len(ib))
            yp[ia[:m]] = y[ib[:m]]
        perm.append(run(yp))
    return {"balanced_acc": float(real), "shuffled_tables_mean": float(np.mean(perm))}


def stage_brew(probe=True):
    X, keep, t_of, z, pairs = brew_material()
    folds = table_folds(48)
    out = {"n_renders": len(keep), "n_pairs": len(pairs), "n_tables": 48, "K": K,
           "definitions": {
               "energy": "||P h||^2 / ||h||^2 over renders",
               "item_frac": "||P (h - mu_LOTO)||^2 / ||P h||^2, mu_LOTO = mean over renders of the other 47 tables",
               "item_held": "||P (h - mu_LOTO)||^2 / ||h - mu_LOTO||^2",
               "delta_share": "||P (h_m - h_p)||^2 / ||h_m - h_p||^2 over the 183 member-partner pairs",
               "reader_visible_ratio": "||unit(P h_m) - unit(P h_p)|| / ||unit(h_m) - unit(h_p)|| (readers rescale their input)",
               "PCA_k": "004-style within-table PCA (centred within table, rows / sqrt(n_t)), cross-fitted over 4 table folds; "
                        "k chosen as the smallest k whose pooled held-out median delta_share >= the J piece's (PCA_k_delta) "
                        "or whose item_held >= the J piece's (PCA_k_item)",
               "mean_removed": "I - u u^T, u = unit(mu_LOTO) (rank-1 removal of the other tables' mean direction)",
               "z_probe": "exploratory: 12-way logistic (C 0.05, standardised) on unit(P h), folds = 4 table folds; "
                          "control = whole tables' label sets permuted (2 draws)"},
           "cells": {}}
    for ci, cell in ((2, "colon"), (1, "think")):
        for li, L in enumerate(LAYERS):
            t = time.time()
            H = torch.from_numpy(np.asarray(X[keep, ci, li], dtype=np.float64))
            n = H.shape[0]
            tot = H.sum(0)
            mu_lo = torch.empty_like(H)
            for tb in range(48):
                m = torch.from_numpy(t_of == tb)
                mu_lo[m] = (tot - H[m].sum(0)) / (n - int(m.sum()))
            iM = np.array([a for a, b in pairs]); iP = np.array([b for a, b in pairs])
            Dl = H[iM] - H[iP]
            pieces = {"whole": (lambda X_: X_)}
            for nm in (f"J_raw_L{L}", f"J_gain_L{L}", "LL_raw", "LL_gain"):
                Q = load_basis(nm)
                short = nm.replace(f"_L{L}", "")
                pieces[short] = proj(Q); pieces["rest_" + short] = comp(Q)
            # mean-removed: per-render projector depends on the render's table
            U = mu_lo / mu_lo.norm(dim=1, keepdim=True)
            rec = {}

            def stats_for(P, rowproj=None):
                if rowproj is None:
                    E, F, V = piece_stats(P, H, mu_lo)
                    PD = P(Dl); PM = P(H[iM]); PP = P(H[iP])
                else:
                    E, F, V = rowproj(H, mu_lo)
                    PD, PM, PP = rowproj(None, None, pairs_mode=True)
                ds = (PD ** 2).sum(1) / (Dl ** 2).sum(1)
                un = lambda A: A / A.norm(dim=1, keepdim=True)
                rv = (un(PM) - un(PP)).norm(dim=1) / (un(H[iM]) - un(H[iP])).norm(dim=1)
                return {"energy": summ(E), "item_frac": summ(F), "item_held": summ(V), "delta_share": summ(ds),
                        "reader_visible_ratio": summ(rv)}

            for nm, P in pieces.items():
                rec[nm] = stats_for(P)

            def mr(Hx, mux, pairs_mode=False):
                if pairs_mode:
                    Ut = U[iM]                                                      # same table for member and partner
                    f = lambda A: A - (A * Ut).sum(1, keepdim=True) * Ut
                    return f(Dl), f(H[iM]), f(H[iP])
                f = lambda A: A - (A * U).sum(1, keepdim=True) * U
                PH = f(Hx); C = Hx - mux; PC = f(C)
                return (PH ** 2).sum(1) / (Hx ** 2).sum(1), (PC ** 2).sum(1) / (PH ** 2).sum(1), (PC ** 2).sum(1) / (C ** 2).sum(1)
            rec["mean_removed"] = stats_for(None, rowproj=mr)
            # random 1,024 (10 draws): per-draw medians, then median and range over draws
            rd = {}
            for s in range(10):
                Qr = rand_basis(SEED + s)
                for pre, P in (("random", proj(Qr)), ("rest_random", comp(Qr))):
                    st = stats_for(P)
                    for mk, v in st.items():
                        rd.setdefault(pre, {}).setdefault(mk, []).append(v["median"])
            for pre, d in rd.items():
                rec[pre + " (10 draws)"] = {mk: {"median_of_draw_medians": med(v), "min": float(min(v)), "max": float(max(v))} for mk, v in d.items()}
            # PCA-k, cross-fitted over table folds; k matched to the J_raw piece on delta_share and on item_held
            J_ds = rec["J_raw"]["delta_share"]["median"]; J_ih = rec["J_raw"]["item_held"]["median"]
            ds_by_k, ih_by_k, Vf = {}, {}, []
            for f in folds:
                tr = [j for j in range(n) if t_of[j] not in f]
                V, _ = within_table_pca(H, t_of, tr)
                Vf.append((f, V))
            kmax = min(V.shape[1] for _, V in Vf)
            if FINE_K:                                                   # added after the first run (k grid too coarse)
                ks = list(range(1, kmax + 1))
            else:
                ks = sorted(set([1, 2, 4, 8, 11, 16, 24, 33, 48, 64, 96, 128] + [kmax]))
                ks = [k for k in ks if k <= kmax]
            for k in ks:
                dsl, ihl = [], []
                for f, V in Vf:
                    Qk = V[:, :k]
                    pm = [i for i, (a, b) in enumerate(pairs) if t_of[a] in f]
                    rows = [j for j in range(n) if t_of[j] in f]
                    PD = (Dl[pm] @ Qk) @ Qk.T
                    dsl += ((PD ** 2).sum(1) / (Dl[pm] ** 2).sum(1)).tolist()
                    C = H[rows] - mu_lo[rows]
                    PC = (C @ Qk) @ Qk.T
                    ihl += ((PC ** 2).sum(1) / (C ** 2).sum(1)).tolist()
                ds_by_k[k] = med(dsl); ih_by_k[k] = med(ihl)
            if FINE_K:                                                   # closest held-out median, not first >=
                k_ds = min(ks, key=lambda k: abs(ds_by_k[k] - J_ds))
                k_ih = min(ks, key=lambda k: abs(ih_by_k[k] - J_ih))
            else:
                k_ds = next((k for k in ks if ds_by_k[k] >= J_ds), None)
                k_ih = next((k for k in ks if ih_by_k[k] >= J_ih), None)
            rec["PCA_curve"] = {"k": ks, "heldout_delta_share": [ds_by_k[k] for k in ks], "heldout_item_held": [ih_by_k[k] for k in ks],
                                "k_matched_delta": k_ds, "k_matched_item": k_ih, "kmax": kmax}
            for tag, kk in (("PCA_k_delta", k_ds), ("PCA_k_item", k_ih)):
                if kk is None:
                    rec[tag] = {"not_reachable_with_kmax": kmax}
                    continue
                # cross-fitted piece: each render / pair uses the basis fitted without its table
                Hp = torch.empty_like(H); Hr = torch.empty_like(H); Mup = torch.empty_like(H)
                for f, V in Vf:
                    Qk = V[:, :kk]
                    m = torch.from_numpy(np.array([t in f for t in t_of]))
                    Hp[m] = (H[m] @ Qk) @ Qk.T
                    Mup[m] = ((H[m] - mu_lo[m]) @ Qk) @ Qk.T
                for pre, A, B in (("", Hp, Mup), ("rest_", H - Hp, (H - mu_lo) - Mup)):
                    E = (A ** 2).sum(1) / (H ** 2).sum(1)
                    F = (B ** 2).sum(1) / (A ** 2).sum(1)
                    V_ = (B ** 2).sum(1) / ((H - mu_lo) ** 2).sum(1)
                    PDm = A[iM] - A[iP]
                    ds = (PDm ** 2).sum(1) / (Dl ** 2).sum(1)
                    un = lambda Z: Z / Z.norm(dim=1, keepdim=True)
                    rv = (un(A[iM]) - un(A[iP])).norm(dim=1) / (un(H[iM]) - un(H[iP])).norm(dim=1)
                    rec[pre + tag] = {"k": kk, "energy": summ(E), "item_frac": summ(F), "item_held": summ(V_),
                                      "delta_share": summ(ds), "reader_visible_ratio": summ(rv)}
            if probe:
                pr = {}
                for nm in ("whole", "J_raw", "rest_J_raw"):
                    A = pieces[nm](H)
                    Fe = (A / A.norm(dim=1, keepdim=True)).numpy()
                    pr[nm] = z_probe(Fe, z, t_of, folds)
                Qr = rand_basis(SEED)
                for nm, P in (("random(seed0)", proj(Qr)), ("rest_random(seed0)", comp(Qr))):
                    A = P(H); Fe = (A / A.norm(dim=1, keepdim=True)).numpy()
                    pr[nm] = z_probe(Fe, z, t_of, folds)
                rec["z_probe"] = pr
            out["cells"][f"{cell}_L{L}"] = rec
            log(cell, L, f"{time.time() - t:.0f}s", {k: round(v["delta_share"]["median"], 3) for k, v in rec.items()
                                                     if isinstance(v, dict) and "delta_share" in v and "median" in v["delta_share"]})
    out["fine_k_grid"] = FINE_K
    dump(OUT_NAME, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["bases", "crosscheck", "overlaps", "brew", "all"])
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--fine-k", action="store_true", help="integer k grid, k closest to J's (writes geometry_brew_finek.json)")
    a = ap.parse_args()
    global FINE_K, OUT_NAME
    if a.fine_k:
        FINE_K, OUT_NAME = True, "geometry_brew_finek.json"
    for s in (["bases", "crosscheck", "overlaps", "brew"] if a.stage == "all" else [a.stage]):
        log("stage", s)
        {"bases": stage_bases, "crosscheck": stage_crosscheck, "overlaps": stage_overlaps,
         "brew": lambda: stage_brew(probe=not a.no_probe)}[s]()


if __name__ == "__main__":
    main()
