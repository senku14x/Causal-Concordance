"""J-space pursuit (jspace.py): exact recovery of synthetic sparse nonnegative signals, nonnegativity, the fixed-k stopping
rule, refits, atoms and the norm gain. CPU only.

Exact recovery here is on random (incoherent) dictionaries, the easy case: it shows the code is correct, not that the
pursuit recovers anything unique from real J-lens atoms, which are far more coherent (near-duplicate vocabulary
directions)."""
import pytest
import torch

import jspace as JS


def _dict(V, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(V, d, generator=g)
    return D / D.norm(dim=1, keepdim=True), g


@pytest.mark.parametrize("V,d,s", [(500, 64, 4), (2000, 256, 8), (5000, 512, 16)])
def test_exact_recovery_of_sparse_nonnegative_signal(V, d, s):
    D, g = _dict(V, d)
    X, truth = [], []
    for _ in range(20):
        S = torch.randperm(V, generator=g)[:s]
        c = 1 + torch.rand(s, generator=g)
        X.append(c @ D[S])
        truth.append(dict(zip(S.tolist(), c.tolist())))
    out = JS.decompose(torch.stack(X), D, k=s, refit="nnls")
    for i, t in enumerate(truth):
        assert set(out["support"][i]) == set(t)
        for a, c in zip(out["support"][i], out["coef"][i]):
            assert abs(c - t[a]) < 1e-5
        assert float(out["x_nonj"][i].norm() / X[i].norm()) < 1e-5
        assert abs(float(out["energy_j"][i]) - 1.0) < 1e-5


def test_coefficients_nonnegative_and_negative_parts_left_out():
    D, g = _dict(400, 64, seed=1)
    x = 2.0 * D[3] + 1.5 * D[10] - 2.0 * D[20]              # one negative component
    out = JS.decompose(x[None], D, k=6, refit="nnls")
    assert all(c > 0 for c in out["coef"][0])
    assert 20 not in out["support"][0]
    assert {3, 10} <= set(out["support"][0])
    idx, coef = JS.gradient_pursuit(x[None], D, 6)
    assert bool((coef >= 0).all())


def test_fixed_k_support_bound_and_determinism():
    D, g = _dict(800, 64, seed=2)
    x = torch.randn(5, 64, generator=g)                        # dense signals: the pursuit runs all k iterations
    for k in (1, 10, 25):
        a = JS.decompose(x, D, k=k, refit=None)
        b = JS.decompose(x, D, k=k, refit=None)
        assert all(len(s) <= k for s in a["support"])
        assert a["support"] == b["support"] and torch.equal(a["x_j"], b["x_j"])


def test_nnls_refit_kkt_and_energy_split():
    D, g = _dict(800, 64, seed=3)
    x = torch.randn(4, 64, generator=g)
    out = JS.decompose(x, D, k=10, refit="nnls")
    for i in range(4):
        r = out["x_nonj"][i].double()
        Ds = D[out["support"][i]].double()
        assert float((Ds @ r).abs().max()) < 1e-4 * float(x[i].norm())   # KKT: residual ⟂ active atoms
        e = float(out["x_j"][i].norm() ** 2 + r.norm() ** 2) / float(x[i].norm() ** 2)
        assert abs(e - 1) < 1e-4                                       # x_j ⟂ x_nonj


def test_ls_refit_fits_at_least_as_well_on_the_same_support():
    D, g = _dict(800, 64, seed=4)
    x = torch.randn(4, 64, generator=g)
    idx, coef = JS.gradient_pursuit(x, D, 10)
    nn = JS.decompose(x, D, k=10, refit="nnls")
    ls = JS.decompose(x, D, k=10, refit="ls")
    for i in range(4):
        assert float(ls["x_nonj"][i].norm()) <= float(nn["x_nonj"][i].norm()) + 1e-5


def test_chunked_scores_equal_unchunked():
    D, g = _dict(1000, 32, seed=5)
    x = torch.randn(3, 32, generator=g)
    a = JS.gradient_pursuit(x, D, 8, chunk=None)
    b = JS.gradient_pursuit(x, D, 8, chunk=128)
    assert torch.equal(a[0], b[0]) and torch.allclose(a[1], b[1])


def test_atoms_unit_and_match_jlens_cosine_direction():
    g = torch.Generator().manual_seed(6)
    W, J = torch.randn(300, 16, generator=g), torch.randn(16, 16, generator=g)
    gain = 0.5 + torch.rand(16, generator=g)
    A = JS.atoms(J, W, chunk=64)
    torch.testing.assert_close(A.norm(dim=1), torch.ones(300))
    h = torch.randn(16, generator=g)
    cos_score = (h @ J.T @ W.T) / (W @ J).norm(dim=1)          # readers.VectorLenses.j_cosine
    torch.testing.assert_close(A @ h, cos_score, rtol=1e-4, atol=1e-5)
    Af = JS.atoms(J, W, gain=gain)
    v = (gain * W[7]) @ J
    torch.testing.assert_close(Af[7], v / v.norm())


@pytest.mark.parametrize("one_plus", [False, True])
def test_effective_gain_recovers_rmsnorm_gain(one_plus):
    class RMS(torch.nn.Module):
        def __init__(self, w, eps):
            super().__init__()
            self.weight, self.variance_epsilon = torch.nn.Parameter(w), eps

        def forward(self, x):
            y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
            return y * ((1 + self.weight) if one_plus else self.weight)

    w = torch.linspace(-0.5, 0.5, 12)
    g_true = (1 + w) if one_plus else w
    torch.testing.assert_close(JS.effective_gain(RMS(w, 1e-6), 12), g_true, rtol=1e-5, atol=1e-6)
