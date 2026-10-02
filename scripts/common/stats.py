"""Cluster-level inference. Ported from RHUT common/scripts/stats.py (cluster_t, equivalence, ratio_of_means,
holm, paired_p) with BH-FDR and a cluster bootstrap added."""
import numpy as np
from scipy import stats as st


def cluster_t(values, conf=0.95):
    """Mean and Student-t interval over clusters; sign counts; per-cluster values."""
    v = np.asarray(values, float)
    n = len(v)
    m = float(v.mean())
    if n < 2:
        return {"mean": m, "ci": [float("nan"), float("nan")], "n": n, "signs_pos": int((v > 0).sum()),
                "signs_neg": int((v < 0).sum()), "per_cluster": v.tolist()}
    h = st.t.ppf(0.5 + conf / 2, n - 1) * float(v.std(ddof=1)) / np.sqrt(n)
    return {"mean": m, "ci": [m - h, m + h], "n": n, "signs_pos": int((v > 0).sum()), "signs_neg": int((v < 0).sum()),
            "sd": float(v.std(ddof=1)), "per_cluster": v.tolist()}


def equivalence(values, eps, conf=0.90):
    """Two one-sided tests: the (conf) interval lies entirely within [-eps, +eps]."""
    r = cluster_t(values, conf)
    r["eps"] = eps
    r["equivalent"] = bool(r["ci"][0] > -eps and r["ci"][1] < eps)
    return r


def ratio_of_means(num, den, B=2000, seed=20260907):
    """Ratio of cluster means with a paired cluster-bootstrap interval."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = len(num)
    if n == 0 or not (np.isfinite(num).all() and np.isfinite(den).all()):
        return {"ratio": float("nan"), "ci": [float("nan"), float("nan")], "numerator_mean": float("nan"),
                "denominator_mean": float("nan"), "n": n, "B": B}
    rng = np.random.default_rng(seed)
    point = float(num.mean() / den.mean())
    bs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        d = den[idx].mean()
        bs.append(num[idx].mean() / d if d != 0 else np.nan)
    bs = np.array(bs)
    bs = bs[np.isfinite(bs)]
    return {"ratio": point, "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "numerator_mean": float(num.mean()), "denominator_mean": float(den.mean()), "n": n, "B": B}


def holm(pvals):
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adj[i] = running
    return adj.tolist()


def paired_p(values):
    v = np.asarray(values, float)
    return float(st.ttest_1samp(v, 0.0).pvalue) if len(v) > 1 else float("nan")


def bh_fdr(pvals, q=0.05):
    """Benjamini-Hochberg step-up. Returns (rejected list[bool], adjusted p-values list[float]).
    Adjusted p_(i) = min_{j>=i} m p_(j) / j, capped at 1 (monotone), in the original order."""
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return [], []
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    adj = np.empty(m)
    adj[order] = adj_sorted
    return (adj <= q).tolist(), adj.tolist()


def empirical_p(observed, null_samples, greater=True):
    """One-sided empirical p-value with the +1 correction: (1 + #null >= obs) / (1 + n_null)."""
    null = np.asarray(null_samples, float)
    k = int((null >= observed).sum()) if greater else int((null <= observed).sum())
    return (1 + k) / (1 + len(null))


def cluster_bootstrap(values, clusters, stat=np.mean, B=2000, seed=20261002, conf=0.95):
    """Percentile bootstrap resampling CLUSTERS with replacement (all observations of a drawn cluster
    enter together). `stat` maps the pooled 1-D array of resampled observations to a scalar."""
    v = np.asarray(values, float)
    c = np.asarray(clusters)
    ids = np.unique(c)
    groups = [v[c == k] for k in ids]
    rng = np.random.default_rng(seed)
    point = float(stat(v))
    bs = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(ids), len(ids))
        bs[b] = stat(np.concatenate([groups[i] for i in pick]))
    lo, hi = np.percentile(bs, [100 * (1 - conf) / 2, 100 * (1 + conf) / 2])
    return {"stat": point, "ci": [float(lo), float(hi)], "n_clusters": len(ids), "n_obs": len(v), "B": B, "seed": seed}
