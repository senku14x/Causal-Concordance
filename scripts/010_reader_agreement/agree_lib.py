"""010 helpers (exploratory, CPU only): cluster-bootstrap correlations (Pearson, Spearman), sign agreement, partial
correlations on a control variable, table-derangement nulls, and round-half-up. Pure numpy/scipy; no model, no GPU.

Bootstrap convention: clusters (brew tables) are resampled with replacement, B = 10,000 draws from
np.random.default_rng(20261004).integers(0, n, (B, n)) (007's `analysis.boot_idx`). A resample is applied to pair-level
data as integer weights (each pair inherits its table's draw count), so every statistic in one block uses the same
draws and differences between statistics are paired. Spearman under a resample uses weighted mid-ranks, which equal
the ranks of the expanded resample.
"""
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from scipy import stats as sst

B_BOOT = 10000
BOOT_SEED = 20261004
N_NULL = 1000
NULL_SEED = 1                     # 006's derangement generator (reanalyse_006: default_rng(1)); first 200 = 006's draws
CHUNK = 500


def rhu(x, nd=2):
    """Round half-up from the full-precision value (project convention)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return x
    q = Decimal(1).scaleb(-nd)
    return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def boot_idx(n, B=B_BOOT, seed=BOOT_SEED):
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def counts(idx, n):
    W = np.zeros((idx.shape[0], n))
    for b in range(idx.shape[0]):
        W[b] = np.bincount(idx[b], minlength=n)
    return W


def derangements(n, k=N_NULL, seed=NULL_SEED):
    r2 = np.random.default_rng(seed)
    perms = []
    while len(perms) < k:
        p_ = r2.permutation(n)
        if not (p_ == np.arange(n)).any():
            perms.append(p_)
    return np.array(perms)


def summarize(x, idx):
    """Table-mean summary with the bootstrap percentile CI and sign counts (007 `analysis.summarize`)."""
    x = np.asarray(x, float)
    m = x[idx].mean(1)
    return {"n": int(len(x)), "mean": float(x.mean()), "ci": [float(np.quantile(m, .025)), float(np.quantile(m, .975))],
            "pos": int((x > 0).sum()), "neg": int((x < 0).sum()), "zero": int((x == 0).sum())}


def weighted_ranks(x, W):
    """Mid-ranks of x in each resample. x [n]; W [b, n] integer weights. Returns [b, n] (rows with weight 0 get a rank
    that never enters any weighted moment)."""
    u, inv = np.unique(x, return_inverse=True)
    M = np.zeros((len(x), len(u)))
    M[np.arange(len(x)), inv] = 1.0
    Wg = W @ M
    cum = np.cumsum(Wg, 1)
    mid = cum - (Wg - 1.0) / 2.0
    return mid[:, inv]


def _moments(X, W):
    """X [k, n] or [k, b, n]; W [b, n]. Weighted covariance matrices [b, k, k]."""
    sw = W.sum(1)
    if X.ndim == 2:
        m = (W @ X.T) / sw[:, None]
        M2 = np.einsum("bn,in,jn->bij", W, X, X, optimize=True) / sw[:, None, None]
    else:
        m = np.einsum("bn,ibn->bi", W, X, optimize=True) / sw[:, None]
        M2 = np.einsum("bn,ibn,jbn->bij", W, X, X, optimize=True) / sw[:, None, None]
    return M2 - m[:, :, None] * m[:, None, :]


def _corr(cov):
    d = np.sqrt(np.clip(np.einsum("...ii->...i", cov), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / (d[..., :, None] * d[..., None, :])


def _sign_same(X, W):
    P, N = (X > 0).astype(float), (X < 0).astype(float)
    sw = W.sum(1)
    return (np.einsum("bn,in,jn->bij", W, P, P, optimize=True) + np.einsum("bn,in,jn->bij", W, N, N, optimize=True)) \
        / sw[:, None, None]


def boot_stats(X, W):
    """All instruments at once. X [k, n] (finite); W [b, n]. Returns dict of [b, k, k]: cov, pearson, spearman (and the
    rank covariance), sign-agreement share."""
    Xc = X - X.mean(1, keepdims=True)
    out = {"cov": [], "pearson": [], "spearman": [], "rcov": [], "sign": []}
    for s in range(0, W.shape[0], CHUNK):
        Wc = W[s:s + CHUNK]
        cov = _moments(Xc, Wc)
        R = np.stack([weighted_ranks(X[i], Wc) for i in range(X.shape[0])])          # [k, b, n]
        rcov = _moments(R, Wc)
        out["cov"].append(cov)
        out["pearson"].append(_corr(cov))
        out["rcov"].append(rcov)
        out["spearman"].append(_corr(rcov))
        out["sign"].append(_sign_same(X, Wc))
    res = {k: np.concatenate(v) for k, v in out.items()}
    # sign agreement expected from the two marginals (independence) and the excess over it (Cohen's kappa)
    sw = W.sum(1)
    pp, pn = (W @ (X > 0).T.astype(float)) / sw[:, None], (W @ (X < 0).T.astype(float)) / sw[:, None]
    res["sign_expected"] = pp[:, :, None] * pp[:, None, :] + pn[:, :, None] * pn[:, None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        res["sign_kappa"] = (res["sign"] - res["sign_expected"]) / (1 - res["sign_expected"])
    return res


def point_stats(X):
    return {k: v[0] for k, v in boot_stats(X, np.ones((1, X.shape[1]))).items()}


def partial(r, c):
    """Partial correlation matrix controlling for instrument c. r [..., k, k]."""
    rc = r[..., :, c]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (r - rc[..., :, None] * rc[..., None, :]) / np.sqrt((1 - rc[..., :, None] ** 2) * (1 - rc[..., None, :] ** 2))


def model_cov_share(cov, c):
    """Share of cov(x, y) carried by the control term: cov(x,m) cov(y,m) / (var(m) cov(x,y)). Equals 1 - residual
    covariance share; can leave [0, 1] when the residual covariance has the opposite sign or cov(x, y) is near 0."""
    cc = cov[..., :, c]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (cc[..., :, None] * cc[..., None, :]) / (cov[..., c, c][..., None, None] * cov)


def ci(a):
    a = np.asarray(a, float)
    ok = np.isfinite(a)
    if ok.sum() < 10:
        return [None, None]
    return [float(np.quantile(a[ok], .025)), float(np.quantile(a[ok], .975))]


def null_stats(Xt, perms):
    """Table-derangement null: instrument j's per-table values are permuted across tables (instrument i fixed).
    Xt [k, nT]; perms [P, nT]. Returns dict of [P, k, k] (pearson, spearman, sign)."""
    def z(A):
        return (A - A.mean(1, keepdims=True)) / A.std(1, keepdims=True)
    Z = z(Xt)
    Rk = z(np.stack([sst.rankdata(x) for x in Xt]))
    n = Xt.shape[1]
    Zp, Rp = Z[:, perms].transpose(1, 0, 2), Rk[:, perms].transpose(1, 0, 2)     # [P, k, nT]
    P, N = (Xt > 0).astype(float), (Xt < 0).astype(float)
    Pp, Np_ = P[:, perms].transpose(1, 0, 2), N[:, perms].transpose(1, 0, 2)
    return {"pearson": np.einsum("in,pjn->pij", Z, Zp) / n, "spearman": np.einsum("in,pjn->pij", Rk, Rp) / n,
            "sign": (np.einsum("in,pjn->pij", P, Pp) + np.einsum("in,pjn->pij", N, Np_)) / n}


def table_means(X, tabs):
    """X [k, n_pairs]; tabs [n_pairs]. Returns ([k, nT], table ids)."""
    ut = np.unique(tabs)
    return np.stack([X[:, tabs == t].mean(1) for t in ut], 1), ut


def sign_counts(x, y):
    return {"both_pos": int(((x > 0) & (y > 0)).sum()), "both_neg": int(((x < 0) & (y < 0)).sum()),
            "x_pos_y_neg": int(((x > 0) & (y < 0)).sum()), "x_neg_y_pos": int(((x < 0) & (y > 0)).sum())}


def agreement_block(names, X, tabs, control=None, with_null=True, B=B_BOOT, keep=False):
    """Agreement between every pair of instruments over pairs and over tables.

    names [k]; X [k, n_pairs] per-pair D (finite); tabs [n_pairs] cluster ids. control: name of the control instrument
    for partial correlations (pair and table level). Returns (records keyed 'A || B', bootstrap arrays if keep)."""
    names = list(names)
    k = len(names)
    Xt, ut = table_means(X, tabs)
    nT = len(ut)
    idx = boot_idx(nT, B)
    WT = counts(idx, nT)
    tpos = np.searchsorted(ut, tabs)
    WP = WT[:, tpos]
    lv = {"pairs": (X, WP), "tables": (Xt, WT)}
    pt, bs = {}, {}
    for L, (A, W) in lv.items():
        pt[L] = point_stats(A)
        bs[L] = boot_stats(A, W)
    nul = null_stats(Xt, derangements(nT)) if with_null else None
    c = names.index(control) if control is not None else None
    if c is not None:
        for L in lv:
            for src, dst in (("pearson", "partial_pearson"), ("spearman", "partial_spearman")):
                pt[L][dst] = partial(pt[L][src], c)
                bs[L][dst] = partial(bs[L][src], c)
            pt[L]["model_cov_share"] = model_cov_share(pt[L]["cov"], c)
            bs[L]["model_cov_share"] = model_cov_share(bs[L]["cov"], c)
            pt[L]["spearman_model_cov_share"] = model_cov_share(pt[L]["rcov"], c)
            bs[L]["spearman_model_cov_share"] = model_cov_share(bs[L]["rcov"], c)
    out = {}
    for i in range(k):
        for j in range(i + 1, k):
            rec = {"a": names[i], "b": names[j]}
            for L, (A, _) in lv.items():
                r = {"n": int(A.shape[1])}
                for s in ("pearson", "spearman", "sign", "sign_expected", "sign_kappa"):
                    r[s] = float(pt[L][s][i, j])
                    r[f"{s}_ci"] = ci(bs[L][s][:, i, j])
                r["sign_counts"] = sign_counts(A[i], A[j])
                if c is not None and c not in (i, j):
                    r["partial_on"] = control
                    r["r_a_control"], r["r_b_control"] = float(pt[L]["pearson"][i, c]), float(pt[L]["pearson"][j, c])
                    for s in ("partial_pearson", "partial_spearman", "model_cov_share", "spearman_model_cov_share"):
                        r[s] = float(pt[L][s][i, j])
                        r[f"{s}_ci"] = ci(bs[L][s][:, i, j])
                    r["cov_ab"] = float(pt[L]["cov"][i, j])
                    r["cov_ab_model_term"] = float(pt[L]["cov"][i, c] * pt[L]["cov"][j, c] / pt[L]["cov"][c, c])
                    r["cov_ab_residual"] = r["cov_ab"] - r["cov_ab_model_term"]
                rec[L] = r
            if nul is not None:
                rec["tables"]["derangement_null"] = {
                    s: {"mean": float(nul[s][:, i, j].mean()), "p2.5": float(np.quantile(nul[s][:, i, j], .025)),
                        "p97.5": float(np.quantile(nul[s][:, i, j], .975)),
                        "p_one_sided_ge": float((1 + (nul[s][:, i, j] >= pt["tables"][s][i, j] - 1e-12).sum()) / (1 + len(nul[s])))}
                    for s in ("pearson", "spearman", "sign")}
                rec["tables"]["derangement_null"]["n_draws"] = int(len(nul["pearson"]))
                rec["tables"]["derangement_null"]["permuted"] = names[j]
            out[f"{names[i]} || {names[j]}"] = rec
    return out, ({"bs": bs, "pt": pt, "names": names, "ut": ut, "WT": WT, "WP": WP, "idx": idx} if keep else None)


def diff_record(bsL, ptL, names, s, pair_a, pair_b):
    """Paired bootstrap difference of statistic s between two instrument pairs (same draws)."""
    ia, ja = names.index(pair_a[0]), names.index(pair_a[1])
    ib, jb = names.index(pair_b[0]), names.index(pair_b[1])
    d = bsL[s][:, ia, ja] - bsL[s][:, ib, jb]
    return {"estimate": float(ptL[s][ia, ja] - ptL[s][ib, jb]), "ci": ci(d),
            "a": float(ptL[s][ia, ja]), "b": float(ptL[s][ib, jb])}
