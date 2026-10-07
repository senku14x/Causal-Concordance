"""012 Stage G Σ for L24-35 (design §14; EXPLORATORY). Thin wrapper on 008's recipe (scripts/008_trackb/sigma.py):
the same UltraChat-200k test_sft selection (`sigma.conversations`, seed c008.SEED, <= 512 tokens, >= 64, first 4
positions dropped, ~50k tokens) and Ledoit-Wolf (`sigma.ledoit_wolf`), lambda per layer. Blocks 0-39 are stock bf16
under SEC, so L24-35 are captured from a stock forward of registry.load_model() (008's model) with block-output hooks.
Saves per layer mu, W = Σ^{-1/2}, W^{-1} (float64). Gates: G-Σ (min eigenvalue > 0) and round trip max|W W^{-1} - I|
<= 1e-6.

  python stageG_sigma.py [--tokens 50000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/008_trackb"))
sys.path.insert(0, str(ROOT / "scripts/common"))
import registry as R                               # noqa: E402
import sigma as S8                                 # noqa: E402  (008's recipe, imported unchanged)

LAYERS = list(range(24, 36))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=50000)
    a = ap.parse_args()
    assert not torch.backends.cuda.matmul.allow_tf32
    out = ROOT / "outputs/012_reader_concordance/stageG"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok = R.make_tokenizer()
    convs, src = S8.conversations(tok, a.tokens)
    model = R.load_model()
    blocks = R.blocks(model)
    store, grab = {L: [] for L in LAYERS}, {}
    hooks = [blocks[L].register_forward_hook(lambda m, i, o, L=L: grab.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))
             for L in LAYERS]
    try:
        with torch.no_grad():
            for i, ids in convs:
                model(torch.tensor([ids], device=model.device), logits_to_keep=1)
                for L in LAYERS:
                    store[L].append(grab[L][S8.DROP_FIRST:])
    finally:
        for h in hooks:
            h.remove()
    del model
    torch.cuda.empty_cache()
    t_cap = time.time() - t0
    rep = {"source": "HuggingFaceH4/ultrachat_200k test_sft (008 sigma.conversations)", "file": src,
           "n_conversations": len(convs), "conversation_rows": [i for i, _ in convs], "drop_first": S8.DROP_FIRST,
           "regime": "stock bf16 forward (== SEC below block 40), registry.load_model (eager)", "layers": {}}
    sig = {}
    for L in LAYERS:
        X = torch.cat(store[L]).double().to("cuda")
        mu, Sl, lam, diag = S8.ledoit_wolf(X)
        ev, U = torch.linalg.eigh(Sl)
        W = (U * ev.rsqrt()) @ U.T
        Wi = (U * ev.sqrt()) @ U.T
        rt = float((W @ Wi - torch.eye(W.shape[0], dtype=W.dtype, device=W.device)).abs().max())
        sig[L] = {"mu": mu.cpu(), "W": W.cpu(), "Wi": Wi.cpu(), "lambda": lam}
        rep["layers"][L] = {"lambda": lam, **diag, "eig_min": float(ev[0]), "eig_max": float(ev[-1]),
                            "cond": float(ev[-1] / ev[0]), "roundtrip_max_abs": rt}
        print(f"L{L}: n {diag['n']} lambda {lam:.4f} eig [{float(ev[0]):.3e}, {float(ev[-1]):.3e}] roundtrip {rt:.1e}", flush=True)
        del X, Sl, U, W, Wi
        store[L] = None
        torch.cuda.empty_cache()
    torch.save(sig, out / "sigma_L24-35.pt")
    g = {"G_sigma": all(v["eig_min"] > 0 for v in rep["layers"].values()),
         "G_roundtrip": all(v["roundtrip_max_abs"] <= 1e-6 for v in rep["layers"].values())}
    rep.update(gates=g, pass_=all(g.values()), capture_s=round(t_cap, 1), wall_s=round(time.time() - t0, 1),
               manifest=R.manifest_block())
    rep["pass"] = rep.pop("pass_")
    (ROOT / "results/012_reader_concordance/analysis/stageG_sigma.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({"gates": g, "wall_s": rep["wall_s"]}), flush=True)


if __name__ == "__main__":
    main()
