"""014 Part A/B data (spec docs/design_specs/014_typo.md; EXPLORATORY; reader side, reading only).

  python typo.py subject [--n 10] [--tag smoke]   # captures (L40/42/44, final prompt token, + echo control), pieces,
                                                  # token-lens ranks, Oracle L40/L44 readouts (2 samples per piece)
  python typo.py nla     [--tag smoke]            # NLA L42 readouts on the saved pieces (subject freed)
  python typo.py analyze [--tag smoke]            # Part A naming table, Part B agreement -> results/014_typo/analysis/

Pieces as 007: whole, J (J_raw_L{L}, first 1,024 cols, orthonormalised), rest = whole − J, R0 (rand_basis_f32(
rand_seed("r1024", 0), 1024)), MR (whole minus its projection on the unit mean of the typo whole activations).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in ("common", "007_jsubspace", "005_planted_concepts", "012_reader_concordance"):
    sys.path.insert(0, str(ROOT / "scripts" / p))

BANK = Path("/workspace/sources/repos/workspace-bench/evals/typo/items.json")
LAYERS = [40, 42, 44]
PIECES = ["whole", "J", "rest_J", "R0", "MR"]


def out_dir(tag):
    d = ROOT / "outputs/014_typo" / ("" if tag == "full" else tag)
    d.mkdir(parents=True, exist_ok=True)
    return d


def bank(n=None):
    items = json.loads(BANK.read_text())
    items = items if isinstance(items, list) else items["items"]
    return items[:n] if n else items


def echo_text(prompt, target):
    head, _, _last = prompt.rstrip().rpartition(" ")
    return f"{head} {target}"


def pieces_for(H, L, U):
    """H [n, d] float32 -> {piece: [n, d] float32}; bases as 007 (float64 projections)."""
    import p007 as C
    from pieces import rand_basis_f32
    QJ = torch.linalg.qr(torch.load(ROOT / f"outputs/007_jsubspace/bases/J_raw_L{L}.pt", weights_only=False)["Q"][:, :C.K].double())[0]
    R0 = rand_basis_f32(C.rand_seed("r1024", 0), C.K)[0].double()
    Hd = H.double()
    pj = lambda Q: (Hd @ Q) @ Q.T
    J = pj(QJ)
    out = {"whole": Hd, "J": J, "rest_J": Hd - J, "R0": pj(R0), "MR": Hd - (Hd @ U.double())[:, None] * U.double()[None]}
    rec = float(((out["J"] + out["rest_J"] - Hd).norm(dim=1) / Hd.norm(dim=1)).max())
    return {k: v.float() for k, v in out.items()}, rec


def phase_subject(n, tag):
    import readers as RD
    import registry as R
    import p007 as C
    D = out_dir(tag)
    items = bank(n)
    tok, model = R.make_tokenizer(), R.load_model()
    caps = {L: [] for L in LAYERS}
    echo = {L: [] for L in LAYERS}
    meta, gates = [], {"render": [], "piece_recon_max": 0.0}
    t0 = time.time()
    for it in items:
        tgt = it["intermediates"][0]
        ids = tok(it["prompt"], add_special_tokens=False)["input_ids"]
        ids_e = tok(echo_text(it["prompt"], tgt), add_special_tokens=False)["input_ids"]
        last_word = it["prompt"].rstrip().rpartition(" ")[2]
        dec = tok.decode(ids[-1:])
        if not (dec.strip() and last_word.endswith(dec.strip())):
            gates["render"].append({"item": it["name"], "last_token": dec, "last_word": last_word})
        c = RD.capture_stock(model, ids, LAYERS, [len(ids) - 1])
        ce = RD.capture_stock(model, ids_e, LAYERS, [len(ids_e) - 1])
        for L in LAYERS:
            caps[L].append(c[L][0].float())
            echo[L].append(ce[L][0].float())
        with torch.no_grad():
            g = model.generate(torch.tensor([ids], device=model.device), max_new_tokens=8, do_sample=False)
        meta.append({"item": it["name"], "prompt": it["prompt"], "target": tgt, "echo_prompt": echo_text(it["prompt"], tgt),
                     "continuation": tok.decode(g[0, len(ids):], skip_special_tokens=True), "read_token": dec, "n_tokens": len(ids)})
    H = {L: torch.stack(caps[L]) for L in LAYERS}
    E = {L: torch.stack(echo[L]) for L in LAYERS}
    U = {L: torch.nn.functional.normalize(H[L].double().mean(0), dim=0) for L in LAYERS}
    P = {}
    for L in LAYERS:
        P[L], rec = pieces_for(H[L], L, U[L])
        gates["piece_recon_max"] = max(gates["piece_recon_max"], rec)
    torch.save({"items": [m["item"] for m in meta], "layers": LAYERS, "H": H, "echo": E, "U": U, "pieces": P}, D / "captures.pt")
    (D / "meta.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in meta))
    # token lenses: rank of the target token on the whole activation (and the echo control)
    fid = [tok.encode(" " + m["target"], add_special_tokens=False)[0] for m in meta]
    VL = RD.VectorLenses(model, tok, LAYERS, lens_names=("J_NP", "J_CB"))
    ranks = []
    with torch.no_grad():
        for L in LAYERS:
            for src, X in (("whole", H[L]), ("echo", E[L])):
                Xd = X.to(VL.W.device).float()
                lg = {"logit": VL._n(Xd) @ VL.W.T}
                for ln in ("J_NP", "J_CB"):
                    lg[ln] = VL._n(VL.lenses[ln].transport(Xd, L)) @ VL.W.T
                for rd, M in lg.items():
                    for i, f in enumerate(fid):
                        ranks.append({"item": meta[i]["item"], "lens": rd, "layer": L, "src": src, "rank": int((M[i] > M[i, f]).sum()) + 1})
    del VL
    torch.cuda.empty_cache()
    (D / "lens_ranks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ranks))
    # Oracle readouts: every piece at L40 / L44, plus the echo control (whole)
    ol = RD.OracleLens(model, tok)
    rows = []
    try:
        for L in (40, 44):
            for pc in PIECES + ["echo"]:
                X = E[L] if pc == "echo" else P[L][pc]
                outs = []
                for j in range(0, len(X), 32):
                    outs += ol.read_batch(list(X[j:j + 32]), L, k=2, seed=C.SEED)
                for i, smp in enumerate(outs):
                    rows.append({"item": meta[i]["item"], "reader": "oracle", "layer": L, "piece": pc, "samples": smp})
            print(f"[subject] oracle L{L} done ({time.time() - t0:.0f}s)", flush=True)
    finally:
        ol.unload()
    (D / "readouts_oracle.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    gates.update(n_items=len(meta), render_ok=not gates["render"], piece_recon_ok=gates["piece_recon_max"] <= 1e-6, wall_s=round(time.time() - t0, 1))
    (D / "gates_subject.json").write_text(json.dumps(gates, indent=1))
    print(json.dumps(gates), flush=True)


def phase_nla(tag):
    import readers as RD
    import p007 as C
    D = out_dir(tag)
    cap = torch.load(D / "captures.pt")
    meta = [json.loads(l) for l in open(D / "meta.jsonl")]
    nla = RD.NLArl()
    rows, t0 = [], time.time()
    try:
        for pc in PIECES + ["echo"]:
            X = cap["echo"][42] if pc == "echo" else cap["pieces"][42][pc]
            outs = []
            for j in range(0, len(X), 32):
                outs += nla.read_batch(list(X[j:j + 32]), k=2, seed=C.SEED)
            for i, smp in enumerate(outs):
                rows.append({"item": meta[i]["item"], "reader": "nla", "layer": 42, "piece": pc, "samples": smp})
            print(f"[nla] {pc} done ({time.time() - t0:.0f}s) write {nla.last_write}", flush=True)
    finally:
        nla.close()
    (D / "readouts_nla.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def boot(vals, clus, nb=2000, seed=0):
    g = {}
    for v, c in zip(vals, clus):
        g.setdefault(c, []).append(v)
    keys, rng = sorted(g), random.Random(seed)
    b = [np.mean([v for k in (rng.choice(keys) for _ in keys) for v in g[k]]) for _ in range(nb)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else None


def phase_analyze(tag):
    from c012 import _hit
    D = out_dir(tag)
    meta = {json.loads(l)["item"]: json.loads(l) for l in open(D / "meta.jsonl")}
    rows = [json.loads(l) for f in ("readouts_oracle.jsonl", "readouts_nla.jsonl") if (D / f).exists() for l in open(D / f)]
    cells, named = {}, {}
    T = {i: m["target"] for i, m in meta.items()}
    for r in rows:
        k = f"{r['reader']} L{r['layer']}|{r['piece']}"
        hits = [bool(_hit(s, [T[r["item"]]])) for s in r["samples"]]
        # POST HOC (added after the smoke, before the full run): other-target null = the share of the other items'
        # targets a readout names (typo targets are common words, so incidental naming must be ruled out)
        null = [float(np.mean([_hit(s, [t]) for i, t in T.items() if i != r["item"]])) for s in r["samples"]]
        c = cells.setdefault(k, {"v": [], "i": [], "null": []})
        c["v"] += hits
        c["null"] += null
        c["i"] += [r["item"]] * len(hits)
        named[(k, r["item"])] = any(hits)
    out = {"label": f"EXPLORATORY (014 Part A/B, tag {tag})", "n_items": len(meta), "naming": {}, "lens_rank": {}, "agreement": {}}
    for k, c in sorted(cells.items()):
        out["naming"][k] = {"share": float(np.mean(c["v"])), "ci95": boot(c["v"], c["i"]), "n_readouts": len(c["v"]),
                            "other_target_null": float(np.mean(c["null"]))}
    lr = [json.loads(l) for l in open(D / "lens_ranks.jsonl")]
    for r in lr:
        out["lens_rank"].setdefault(f"{r['lens']} L{r['layer']}|{r['src']}", []).append(r["rank"])
    out["lens_rank"] = {k: {"median_rank": float(np.median(v)), "top10_share": float(np.mean([x <= 10 for x in v])), "n": len(v)}
                        for k, v in out["lens_rank"].items()}
    its = sorted(meta)
    for a, b in (("oracle L40", "nla L42"), ("oracle L44", "nla L42"), ("oracle L40", "oracle L44")):
        xa = [named.get((f"{a}|whole", i), False) for i in its]
        xb = [named.get((f"{b}|whole", i), False) for i in its]
        out["agreement"][f"{a} vs {b}"] = {"kappa": kappa(xa, xb), "only_first": int(sum(x and not y for x, y in zip(xa, xb))),
                                           "only_second": int(sum(y and not x for x, y in zip(xa, xb))), "both": int(sum(x and y for x, y in zip(xa, xb)))}
    p = ROOT / "results/014_typo/analysis"
    p.mkdir(parents=True, exist_ok=True)
    (p / f"parts_{tag}.json").write_text(json.dumps(out, indent=1))
    for k, v in out["naming"].items():
        print(f"{k:22s} {v['share']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] null {v['other_target_null']:.2f} n={v['n_readouts']}")
    print(json.dumps(out["agreement"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["subject", "nla", "analyze"])
    ap.add_argument("--n", type=int)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    if a.phase == "subject":
        phase_subject(a.n, a.tag)
    elif a.phase == "nla":
        phase_nla(a.tag)
    else:
        phase_analyze(a.tag)
