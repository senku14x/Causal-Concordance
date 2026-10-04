"""J-space decomposition (SPEC_v2 review (c) c1; workspace paper §2.3 and App. A.8). The J-space component of x is a
sparse nonnegative combination of unit-norm J-lens vectors, found by gradient pursuit over the whole vocabulary; the
non-J component is the remainder.

  atoms              v_t = J_l^T (g ⊙ W_U[t]), normalised to unit norm; g = the final norm's effective gain (the folded
                     J-lens vector, primary) or None (raw W_U, the WSB JLens cosine direction; sensitivity)
  pursuit            GDM's nonnegative gradient pursuit (Blumensath & Davies 2008; GDM progress update #1): exactly k
                     iterations; each adds the atom with the largest signed inner product with the residual, takes the
                     exact line-search gradient step on the support's coefficients and clips negatives to 0 (a clipped
                     atom leaves the support). The final support has at most k atoms.
  refit              "nnls" (primary): nonnegative least squares on the pursuit's support, so x_J is the projection onto
                     that cone and x_J ⟂ x - x_J; "ls": unconstrained least squares on the same support (the A.8 span
                     reading, sensitivity); None: the pursuit's own coefficients.

Reader-side code (it is built from J-lens directions): label code must never import it (tests/test_import_boundary.py).
"""
import numpy as np
import torch


def effective_gain(norm, d, device="cpu"):
    """The elementwise gain of an RMSNorm-style module, whatever its weight convention (w or 1 + w): norm(1) * sqrt(1 + eps),
    since rms(1) = sqrt(1 + eps)."""
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 0.0)))
    one = torch.ones(1, d, device=device, dtype=torch.float32)
    with torch.no_grad():
        p = next(norm.parameters(), None)
        x = one.to(p.dtype) if p is not None else one
        return norm(x).float()[0] * float(np.sqrt(1.0 + eps))


@torch.no_grad()
def atoms(J, W_U, gain=None, chunk=32768):
    """Unit-norm J-lens atoms [V, d]: row t = normalise((g ⊙ W_U[t]) @ J) (= J^T (g ⊙ W_U[t])). J maps the layer-l residual
    to the final residual ([d_out, d_in])."""
    J = J.float()
    out = []
    for i in range(0, W_U.shape[0], chunk):
        w = W_U[i:i + chunk].to(J.device).float()
        if gain is not None:
            w = w * gain.to(J.device).float()
        v = w @ J
        out.append(v / v.norm(dim=1, keepdim=True).clamp_min(1e-12))
    return torch.cat(out)


def _scores(r, D, chunk):
    """r @ D^T in row chunks of D: [n, V]."""
    if chunk is None or D.shape[0] <= chunk:
        return r @ D.T
    return torch.cat([r @ D[i:i + chunk].T for i in range(0, D.shape[0], chunk)], dim=1)


@torch.no_grad()
def gradient_pursuit(x, D, k, chunk=None):
    """Nonnegative gradient pursuit, exactly k iterations. x [n, d] (float32), D [V, d] unit rows on the same device.
    Returns (idx [n, k] long, coef [n, k] float32): slot j holds the atom added at iteration j (-1 if that iteration's
    best atom was already in the support); a coefficient of 0 means the atom is not (or no longer) in the support."""
    x = x.float()
    n = x.shape[0]
    idx = torch.full((n, k), -1, dtype=torch.long, device=x.device)
    coef = torch.zeros(n, k, dtype=torch.float32, device=x.device)
    rows = torch.arange(n, device=x.device)
    for it in range(k):
        safe = idx.clamp_min(0)
        live = (idx >= 0) & (coef > 0)                                  # GDM: selected = weights != 0
        recon = torch.einsum("nk,nkd->nd", coef * live, D[safe])
        r = x - recon
        best = _scores(r, D, chunk).argmax(dim=1)
        dup = ((idx == best[:, None]) & live).any(dim=1)
        idx[rows[~dup], it] = best[~dup]
        sel = (idx >= 0) & ((coef > 0) | (torch.arange(k, device=x.device)[None, :] == it))
        Ds = D[idx.clamp_min(0)]                                          # [n, k, d]
        g = torch.einsum("nkd,nd->nk", Ds, r) * sel                       # -gradient of 0.5||x - c D||^2 on the support
        c = torch.einsum("nk,nkd->nd", g, Ds)
        den = (c * c).sum(dim=1)
        step = torch.where(den > 0, (c * r).sum(dim=1) / den.clamp_min(1e-30), torch.zeros_like(den))
        coef = torch.relu(coef + step[:, None] * g) * sel
    return idx, coef


def _nnls(A, b):
    from scipy.optimize import nnls
    c, _ = nnls(A, b, maxiter=50 * A.shape[1])
    return c


@torch.no_grad()
def decompose(x, D, k, refit="nnls", chunk=None):
    """J-space split of each row of x. Returns a dict: support (list of atom-index lists), coef (matching lists), x_j and
    x_nonj ([n, d] float32, x = x_j + x_nonj), energy_j = ||x_j||^2 / ||x||^2 per row, and the settings."""
    assert refit in ("nnls", "ls", None), refit
    x = x.float()
    idx, coef = gradient_pursuit(x, D, k, chunk=chunk)
    supports, coefs, xj = [], [], torch.zeros_like(x)
    for i in range(x.shape[0]):
        keep = (idx[i] >= 0) & (coef[i] > 0)
        S, c = idx[i][keep], coef[i][keep]
        if refit is not None and len(S):
            A = D[S].double().cpu().numpy().T                             # [d, |S|]
            b = x[i].double().cpu().numpy()
            if refit == "nnls":
                c = torch.from_numpy(_nnls(A, b))
            else:
                c = torch.from_numpy(np.linalg.lstsq(A, b, rcond=None)[0])
            c = c.to(x.device, torch.float64)
            if refit == "nnls":
                S, c = S[c > 0], c[c > 0]
            xj[i] = (c[None, :] @ D[S].double()).float()[0] if len(S) else 0.0
        elif len(S):
            xj[i] = c @ D[S]
        order = torch.argsort(-c.abs()) if len(S) else torch.arange(0)
        supports.append([int(t) for t in S[order]])
        coefs.append([float(v) for v in c[order]])
    energy = (xj.double().norm(dim=1) ** 2 / x.double().norm(dim=1).clamp_min(1e-30) ** 2).float()
    return {"support": supports, "coef": coefs, "x_j": xj, "x_nonj": x - xj, "energy_j": energy,
            "k": k, "refit": refit}
