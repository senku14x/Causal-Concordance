"""007 D statistics (design §3.3; 006's D reused exactly). Pure numpy; unit-tested in tests/test_007_dstat.py."""
import numpy as np

from p007 import COLOURS


def colour_ll(arr, owner):
    """arr [n, S] per-spelling full-sequence log-probs; owner [S] colour names. Returns [n, 12] log-sum-exp per colour
    (column order = COLOURS)."""
    arr = np.asarray(arr, dtype=np.float64)
    owner = np.asarray(owner)
    out = np.empty((arr.shape[0], len(COLOURS)))
    for j, c in enumerate(COLOURS):
        cols = arr[:, owner == c]
        assert cols.shape[1] >= 1, c
        m = cols.max(1, keepdims=True)
        out[:, j] = (m + np.log(np.exp(cols - m).sum(1, keepdims=True)))[:, 0]
    return out


CI = {c: i for i, c in enumerate(COLOURS)}


def pair_D(LLc, pairs, role, act_of=None):
    """Per pair D for a role: [LL_m(t_m) - LL_m(t_p)] - [LL_p(t_m) - LL_p(t_p)]. role in z/start/answer/off.
    pairs: stage-0 role rows. act_of: optional mapping (m_index, p_index) -> rows of LLc to use (derangements).
    Returns (values [n_pairs] with NaN where the role is not scorable, tables [n_pairs])."""
    v = np.full(len(pairs), np.nan)
    tabs = np.array([p["table"] for p in pairs])
    for i, p in enumerate(pairs):
        if role == "off":
            if p["off"] is None:
                continue
            tm, tp = p["off"]
        else:
            if role != "z" and p["collide"][role]:
                continue
            tm, tp = p["m_roles"][role], p["p_roles"][role]
            if tm == tp:
                continue
        a, b = (p["m"], p["p"]) if act_of is None else act_of(p)
        v[i] = (LLc[a, CI[tm]] - LLc[a, CI[tp]]) - (LLc[b, CI[tm]] - LLc[b, CI[tp]])
    return v, tabs


def table_means(v, tabs, n_tables=None):
    """Average over each table's scorable pairs; returns [n_tables_with_data] and the table ids."""
    ok = ~np.isnan(v)
    ts = np.unique(tabs[ok])
    return np.array([v[ok & (tabs == t)].mean() for t in ts]), ts


def D_mean(LLc, pairs, role):
    v, tabs = pair_D(LLc, pairs, role)
    tm, _ = table_means(v, tabs)
    return float(tm.mean()), int(len(tm))
