"""011 AR J-bias check (EXPLORATORY, CPU, post hoc; report 31 step 1, PI go 2026-10-06).

Question: does the Oracle Lens's reconstructor (AR, the ruler its RL reward is computed on) over-weight 007's dense J
subspace relative to the activations it reconstructs? Its meta.json names the Neuronpedia J-lens and a loss mix
(`loss_mix_lambda` 0.25, `val_jfve`), whose definition is not public.

Data: the saved 011 diagnostics (`outputs/011_olens_swaps/d2/diag_tensors.pt`): 768 held-out assistant spans, AR head
outputs under the accepted convention (read `ln_pre`, raw output; target = block-L output at the token preceding the
span), L40 and L44 (007 has J bases there).

Subspaces (all 1,024-dim, orthonormal, raw coordinates): 007's J (`bases/J_raw_L{L}.pt`, top-1,024 of
Σ_t unit(W_U[t] J_L)); PCA-1,024 of the centred targets (variance-matched control, fit on the even spans and
evaluated on the odd spans); 3 random.

Measures, on vectors centred over the evaluation spans:
  share(X, S) = ||P_S X||² / ||X||²  for X = targets and X = AR outputs; ratio = share(AR) / share(targets);
  fve(S) = 1 − ||P_S h − s P_S ŷ||² / ||P_S h||², with one least-squares scale s per subspace (scale-free).

Decision rule (written before computing): "J-biased" if the J ratio ≥ 1.5 at both layers AND exceeds the PCA and
random ratios; "not J-biased" if the J ratio ≤ 1.2 at both layers; otherwise ambiguous.

  python ar_jbias.py
"""
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DIAG = ROOT / "outputs/011_olens_swaps/d2/diag_tensors.pt"
BASES = ROOT / "outputs/007_jsubspace/bases"
RES = ROOT / "results/011_olens_swaps"
K = 1024


def orth(M):
    return torch.linalg.qr(M.double())[0]


def share(X, Q):
    return float((X @ Q).pow(2).sum() / X.pow(2).sum())


def fve(h, y, Q):
    a, b = h @ Q, y @ Q
    s = float((a * b).sum() / b.pow(2).sum())
    return float(1 - (a - s * b).pow(2).sum() / a.pow(2).sum()), s


def main():
    t = torch.load(DIAG, weights_only=False)
    out = {"label": "EXPLORATORY, post hoc (011; report 31 step 1)", "convention": "block_L|ln_pre|raw", "k": K,
           "n_spans_eval": None, "layers": {}}
    g = torch.Generator().manual_seed(20261006)
    for L in (40, 44):
        h = t["t_prev"][L].double()
        y = t["reads"][L]["ln_pre"].double()
        tr, ev = slice(0, None, 2), slice(1, None, 2)
        he, ye = h[ev] - h[ev].mean(0), y[ev] - y[ev].mean(0)
        out["n_spans_eval"] = int(he.shape[0])
        Q = {"J": orth(torch.load(BASES / f"J_raw_L{L}.pt", weights_only=False)["Q"][:, :K])}
        htr = h[tr] - h[tr].mean(0)
        Q["PCA"] = torch.linalg.svd(htr, full_matrices=False)[2][:K].T.contiguous()
        for i in range(3):
            Q[f"rand{i}"] = orth(torch.randn(h.shape[1], K, generator=g, dtype=torch.float64))
        rows = {}
        for nm, q in Q.items():
            sh_h, sh_y = share(he, q), share(ye, q)
            f, s = fve(he, ye, q)
            rows[nm] = {"share_targets": sh_h, "share_ar": sh_y, "ratio": sh_y / sh_h, "fve_scaled": f, "scale": s}
        f_all, s_all = fve(he, ye, torch.eye(h.shape[1], dtype=torch.float64))
        rows["whole"] = {"fve_scaled": f_all, "scale": s_all}
        rand_ratio = max(rows[f"rand{i}"]["ratio"] for i in range(3))
        rows["J_ratio_minus_PCA"] = rows["J"]["ratio"] - rows["PCA"]["ratio"]
        rows["J_ratio_minus_max_random"] = rows["J"]["ratio"] - rand_ratio
        out["layers"][L] = rows
    jr = [out["layers"][L]["J"]["ratio"] for L in (40, 44)]
    beats = all(out["layers"][L]["J"]["ratio"] > max(out["layers"][L]["PCA"]["ratio"],
                                                      *[out["layers"][L][f"rand{i}"]["ratio"] for i in range(3)])
                for L in (40, 44))
    out["verdict"] = ("J-biased" if min(jr) >= 1.5 and beats else "not J-biased" if max(jr) <= 1.2 else "ambiguous")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "ar_jbias.json").write_text(json.dumps(out, indent=1))
    for L, rows in out["layers"].items():
        for nm, r in rows.items():
            if isinstance(r, dict):
                print(L, f"{nm:6s}", {k: round(v, 4) for k, v in r.items()})
    print("verdict:", out["verdict"])


if __name__ == "__main__":
    main()
