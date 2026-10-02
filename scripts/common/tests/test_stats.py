import numpy as np
import pytest
from scipy import stats as st

import stats as S


def test_cluster_t_matches_scipy():
    rng = np.random.default_rng(0)
    v = rng.normal(0.3, 1.0, 17)
    r = S.cluster_t(v)
    lo, hi = st.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=st.sem(v))
    assert r["ci"][0] == pytest.approx(lo) and r["ci"][1] == pytest.approx(hi)
    assert r["signs_pos"] + r["signs_neg"] == 17


def test_bh_fdr_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(1)
    p = np.concatenate([rng.uniform(0, 0.01, 10), rng.uniform(0, 1, 90)])
    rej, adj = S.bh_fdr(p, q=0.05)
    rej2, adj2, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    assert rej == list(rej2)
    np.testing.assert_allclose(adj, adj2, rtol=0, atol=1e-12)


def test_holm_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    p = np.array([0.001, 0.02, 0.03, 0.2, 0.5])
    np.testing.assert_allclose(S.holm(p), multipletests(p, method="holm")[1], atol=1e-12)


def test_cluster_bootstrap_covers_and_respects_clusters():
    rng = np.random.default_rng(2)
    clusters = np.repeat(np.arange(30), 5)
    cluster_effect = rng.normal(0, 1, 30)[clusters]
    v = 0.5 + cluster_effect + rng.normal(0, 0.1, len(clusters))
    r = S.cluster_bootstrap(v, clusters, B=1000, seed=3)
    assert r["n_clusters"] == 30 and r["ci"][0] < 0.5 < r["ci"][1]
    # cluster resampling is wider than treating the 150 observations as independent
    naive = st.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=st.sem(v))
    assert (r["ci"][1] - r["ci"][0]) > (naive[1] - naive[0]) * 1.5


def test_ratio_of_means_and_empirical_p():
    r = S.ratio_of_means([2, 4, 6], [1, 2, 3], B=200)
    assert r["ratio"] == pytest.approx(2.0)
    assert S.empirical_p(5.0, [0, 1, 2, 3]) == pytest.approx(1 / 5)
    assert S.empirical_p(1.5, [0, 1, 2, 3]) == pytest.approx(3 / 5)


def test_equivalence():
    assert S.equivalence([0.01, -0.02, 0.0, 0.015, -0.01], eps=0.1)["equivalent"]
    assert not S.equivalence([0.5, 0.6, 0.55], eps=0.1)["equivalent"]
