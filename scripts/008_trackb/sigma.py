"""008 Σ per layer (design §1.4): ~50k chat-style tokens (UltraChat-200k test_sft, Qwen chat template, thinking off,
<= 512 tokens per conversation, first 4 positions dropped), captured under SEC (bf16 blocks 0-39, fp32 replay 40-63),
layers L40-48. Ledoit-Wolf shrinkage toward (tr S / d) I; lambda reported per layer. Saves mu, eigenvalues and
eigenvectors of Sigma_lambda (float64) per layer.

  python sigma.py [--tokens 50000]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
import c008 as C                                   # noqa: E402
import regime as RG                                # noqa: E402
import registry as R                               # noqa: E402

DROP_FIRST = 4
MAX_LEN = 512
MIN_LEN = 64


def conversations(tok, n_tokens, seed=C.SEED):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("HuggingFaceH4/ultrachat_200k", "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
                        repo_type="dataset")
    df = pd.read_parquet(p)
    order = np.random.default_rng(seed).permutation(len(df))
    out, tot = [], 0
    for i in order:
        msgs = [{"role": m["role"], "content": m["content"]} for m in df.iloc[i]["messages"]]
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, enable_thinking=False)
        ids = list(enc["input_ids"] if hasattr(enc, "keys") else enc)[:MAX_LEN]
        if len(ids) < MIN_LEN:
            continue
        out.append((int(i), ids))
        tot += len(ids) - DROP_FIRST
        if tot >= n_tokens:
            break
    return out, p


def ledoit_wolf(X):
    """X [n, d] float64 (GPU). Returns mu, S_lambda, lambda, diagnostics."""
    n, d = X.shape
    mu = X.mean(0)
    Xc = X - mu
    S = (Xc.T @ Xc) / n
    trS = float(torch.trace(S))
    m = trS / d
    fro2 = float((S * S).sum())
    delta2 = fro2 - 2 * m * trS + m * m * d
    x4 = float(((Xc * Xc).sum(1) ** 2).sum())
    beta_bar2 = (x4 - n * fro2) / (n * n)
    beta2 = min(beta_bar2, delta2)
    lam = beta2 / delta2
    Sl = (1 - lam) * S
    Sl.diagonal().add_(lam * m)
    return mu, Sl, lam, {"n": n, "trace_over_d": m, "delta2": delta2, "beta_bar2": beta_bar2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=50000)
    a = ap.parse_args()
    assert not torch.backends.cuda.matmul.allow_tf32
    out = C.OUT / "sigma"
    out.mkdir(parents=True, exist_ok=True)
    log = C.Logger(out / "log.txt")
    tok = R.make_tokenizer()
    convs, src = conversations(tok, a.tokens)
    log(f"{len(convs)} conversations, {sum(len(c) - DROP_FIRST for _, c in convs)} tokens after dropping {DROP_FIRST}")
    model = R.load_model()
    store = {L: [] for L in C.SIGMA_LAYERS}
    t0 = time.time()
    norms = []
    with RG.SecEngine(model, l0=39) as sec:
        for j, (i, ids) in enumerate(convs):
            _, acts = sec.forward(ids, record=list(C.SIGMA_LAYERS))
            for L in C.SIGMA_LAYERS:
                store[L].append(acts[L][DROP_FIRST:].double().cpu())
            norms.append(acts[44][DROP_FIRST:].norm(dim=1).cpu().numpy())
            sec.clear()
            if j % 20 == 0:
                log(f"conv {j + 1}/{len(convs)} ({time.time() - t0:.0f}s)")
    del model
    torch.cuda.empty_cache()
    nall = np.concatenate(norms)
    rep = {"source": "HuggingFaceH4/ultrachat_200k test_sft", "file": src, "n_conversations": len(convs),
           "conversation_rows": [i for i, _ in convs], "drop_first": DROP_FIRST, "max_len": MAX_LEN, "layers": {},
           "regime": "SEC (bf16 0-39, fp32 40-63)", "L44_norm_quantiles": {q: float(np.quantile(nall, q)) for q in
                                                                          (0.01, 0.5, 0.99, 0.999, 1.0)}}
    for L in C.SIGMA_LAYERS:
        X = torch.cat(store[L]).to("cuda")
        mu, Sl, lam, diag = ledoit_wolf(X)
        ev, U = torch.linalg.eigh(Sl)
        torch.save({"mu": mu.cpu(), "evals": ev.cpu(), "evecs": U.cpu(), "lambda": lam, **diag},
                   out / f"sigma_L{L}.pt")
        rep["layers"][L] = {"lambda": lam, **diag, "eig_min": float(ev[0]), "eig_max": float(ev[-1]),
                            "cond": float(ev[-1] / ev[0]), "top10_share": float(ev[-10:].sum() / ev.sum()),
                            "top100_share": float(ev[-100:].sum() / ev.sum())}
        log(f"L{L}: n {diag['n']} lambda {lam:.4f} eig [{float(ev[0]):.3e}, {float(ev[-1]):.3e}] "
            f"top10 share {rep['layers'][L]['top10_share']:.3f}")
        del X, Sl, U
        torch.cuda.empty_cache()
    rep["minutes"] = (time.time() - t0) / 60
    rep["manifest"] = R.manifest_block()
    C.dump(C.RES / "analysis" / "sigma.json", rep)
    log("done")


if __name__ == "__main__":
    main()
