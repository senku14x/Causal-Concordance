"""011, exploratory: score the saved AR head outputs (ar_vectors.py --save-tensors) under every output convention on CPU.

  python ar_offline_diag.py --run d2

The head's output convention is undocumented. Candidates, each scored against the stock activation preceding the span
(the WSB post's target) in the AR's training whitener W = (Sigma + c * tr(Sigma)/d * I)^(-1/2), c in {0, 0.1}
(the AR config's ridge_c is 0.1):
  zca       y^ is already W (h - mu)                 -> compare y^ with W (y - mu)
  raw       y^ is the raw activation h               -> compare W (y^ - mu) with W (y - mu)
  rawc      y^ is the centred raw activation h - mu  -> compare W y^ with W (y - mu)
  pca       y^ is the PCA-whitened activation        -> compare y^ with P (y - mu)
Metrics: FVE (1 - SSE / SST about the validation mean), FVE0 (SST about mu, i.e. about 0 in whitened space), mean cosine,
retrieval@1 among all spans and within random batches of 24 (the AR's micro-batch), median norm ratio.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ar_vectors as AV                            # noqa: E402


def metrics(yh, y, groups=24, seed=0):
    sst = ((y - y.mean(0)) ** 2).sum()
    sse = ((y - yh) ** 2).sum()
    yn, hn = torch.nn.functional.normalize(y, dim=1), torch.nn.functional.normalize(yh, dim=1)
    S = hn @ yn.T
    ret_all = float((S.argmax(1) == torch.arange(len(y))).double().mean())
    perm = torch.from_numpy(np.random.default_rng(seed).permutation(len(y)))
    hits = []
    for j in range(0, len(y) - groups + 1, groups):
        ix = perm[j:j + groups]
        hits.append((S[ix][:, ix].argmax(1) == torch.arange(groups)).double().mean())
    return {"fve": float(1 - sse / sst), "fve0": float(1 - sse / (y ** 2).sum()), "cos": float((yn * hn).sum(1).mean()),
            "ret_all": ret_all, "ret_b24": float(torch.stack(hits).mean()),
            "norm_ratio": float((yh.norm(dim=1) / y.norm(dim=1)).median())}


def whiteners(L, c):
    w = AV.whitener(L)
    ev = w["ev"] + c * w["ev"].mean()
    W = (w["U"] * ev.rsqrt()) @ w["U"].T
    P = (w["U"] * ev.rsqrt()).T
    return w["mu"], W, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="d2")
    a = ap.parse_args()
    T = torch.load(AV.OUT / a.run / "diag_tensors.pt", weights_only=False)
    out = {}
    for L in AV.LAYERS:
        for c in (0.0, 0.1):
            mu, W, P = whiteners(L, c)
            for lname, l in (("block_L", L), ("block_L-1", L - 1)):
                y = T["t_prev"][l].double()
                yw, yp = (y - mu) @ W.T, (y - mu) @ P.T
                for read, yh in T["reads"][L].items():
                    yh = yh.double()
                    conv = {"zca": (yh, yw), "raw": ((yh - mu) @ W.T, yw), "rawc": (yh @ W.T, yw), "pca": (yh, yp)}
                    for cv, (pred, targ) in conv.items():
                        out[f"L{L}|ridge{c}|{lname}|{read}|{cv}"] = metrics(pred, targ)
    res = AV.RES / f"ar_offline_{a.run}.json"
    res.write_text(json.dumps(out, indent=1))
    top = sorted(out.items(), key=lambda kv: -kv[1]["fve"])[:10]
    for k, v in top:
        print(k, {kk: round(vv, 3) for kk, vv in v.items()})
    print("published val FVE L40/44/48: 0.224 / 0.220 / 0.228; ret1 0.92 / 0.93 / 0.96")


if __name__ == "__main__":
    main()
