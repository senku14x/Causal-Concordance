"""014 content-matched control (EXPLORATORY, POST HOC; claude_ovarida). Rules fixed here before any read.

Why (report 37): removing 007's J subspace from the typo activations removes about 2x the item-varying (centred) variance
that a random 1,024-dim subspace does. So 014's "removing J costs the Oracle 0.14 / 0.225 (L40 / L44) but the NLA 0.025"
may be about losing item-specific content, not about J.

New pieces, built from 014's saved captures (outputs/014_typo/captures.pt; same activations 014 read):
  rest_PCA : whole minus its projection on the top-k principal components of the 100 centred typo activations at that
             layer, k = the smallest k whose centred share >= J's (report 37: L40 23, L42 20, L44 17). Content-matched.
  rest_R0, rest_R1 : whole minus its projection on a random 1,024-dim subspace (007's r1024 seeds 0 and 1). Size-matched.
Readers exactly as 014: Oracle L40 / L44 (read_batch, k=2, seed p007.SEED, chunks of 32), NLA L42 (same); naming = whole
word, case-insensitive (c012._hit), share of the 2 samples per item. 'whole' and 'rest_J' come from 014's own readouts
(outputs/014_typo/readouts_*.jsonl; same seed and chunking).

Contrasts (per item, item bootstrap 2,000, seed 0):
  cost(piece) = share(whole) - share(piece)
  D_PCA = cost(rest_J) - cost(rest_PCA);  D_R = cost(rest_J) - mean(cost(rest_R0), cost(rest_R1))
  Oracle - NLA difference in D_PCA (L40 vs NLA L42, L44 vs NLA L42).
Readings (fixed now):
  For a reader, "J-specific beyond content" if D_PCA's CI excludes 0 (> 0) and D_PCA >= 0.05;
  "content, not J" if D_PCA's CI includes 0 and cost(rest_PCA) >= 0.5 * cost(rest_J); otherwise "unresolved".
  The 014 Oracle-vs-NLA J contrast "survives content matching" if the Oracle - NLA difference in D_PCA has a CI > 0 at
  L40 or L44; "does not survive" if both CIs include 0 and both point estimates are < 0.05; otherwise "unresolved".

  python typo_content_control.py read     # GPU: pieces + Oracle reads (subject loaded), then NLA reads (subject freed)
  python typo_content_control.py analyze  # CPU
"""
import json, random, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in ("common", "007_jsubspace", "005_planted_concepts", "012_reader_concordance"):
    sys.path.insert(0, str(ROOT / "scripts" / p))
SRC = ROOT / "outputs/014_typo"
OUT = ROOT / "outputs/claude_ovarida/014_content_control"
RES = ROOT / "claude_ovarida/results/typo_content_control.json"
K = {40: 23, 42: 20, 44: 17}
NEW = ["rest_PCA", "rest_R0", "rest_R1"]


def pieces():
    import p007 as C
    from pieces import rand_basis_f32
    cap = torch.load(SRC / "captures.pt", weights_only=False)
    P, gate = {}, {}
    for L in cap["layers"]:
        H = cap["H"][L].double()
        Xc = H - H.mean(0, keepdim=True)
        Vt = torch.linalg.svd(Xc, full_matrices=False)[2]
        QP = Vt[:K[L]].T
        QJ = torch.linalg.qr(torch.load(ROOT / f"outputs/007_jsubspace/bases/J_raw_L{L}.pt", weights_only=False)["Q"][:, :C.K].double())[0]
        share = lambda X, Q: float((((X @ Q) ** 2).sum(1) / (X ** 2).sum(1)).mean())
        out = {"rest_PCA": H - (H @ QP) @ QP.T}
        for r in (0, 1):
            Q = rand_basis_f32(C.rand_seed("r1024", r), C.K)[0].double()
            out[f"rest_R{r}"] = H - (H @ Q) @ Q.T
        # gate: rest_J rebuilt here equals 014's saved rest_J piece; PCA-k centred share >= J's
        rJ = H - (H @ QJ) @ QJ.T
        gate[L] = {"rest_J_vs_saved_max_rel": float(((rJ.float() - cap["pieces"][L]["rest_J"]).norm(dim=1) / H.float().norm(dim=1)).max()),
                   "centred_share_J": share(Xc, QJ), "centred_share_PCA_k": share(Xc, QP), "k": K[L],
                   "energy_removed_share": {"J": share(H, QJ), "PCA": share(H, QP),
                                            **{f"R{r}": 1 - float(((out[f"rest_R{r}"] ** 2).sum(1) / (H ** 2).sum(1)).mean()) for r in (0, 1)}}}
        P[L] = {k: v.float() for k, v in out.items()}
    return cap, P, gate


def phase_read():
    import readers as RD, registry as R, p007 as C
    OUT.mkdir(parents=True, exist_ok=True)
    cap, P, gate = pieces()
    (OUT / "gates.json").write_text(json.dumps(gate, indent=1))
    print(json.dumps(gate, indent=1), flush=True)
    assert all(g["rest_J_vs_saved_max_rel"] < 1e-5 for g in gate.values()), "piece rebuild does not match 014"
    items, t0 = cap["items"], time.time()
    tok, model = R.make_tokenizer(), R.load_model()
    ol, rows = RD.OracleLens(model, tok), []
    try:
        for L in (40, 44):
            for pc in NEW:
                X, outs = P[L][pc], []
                for j in range(0, len(X), 32):
                    outs += ol.read_batch(list(X[j:j + 32]), L, k=2, seed=C.SEED)
                rows += [{"item": items[i], "reader": "oracle", "layer": L, "piece": pc, "samples": s} for i, s in enumerate(outs)]
                print(f"oracle L{L} {pc} ({time.time() - t0:.0f}s)", flush=True)
    finally:
        ol.unload()
    (OUT / "readouts_oracle.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    del model
    torch.cuda.empty_cache()
    nla, rows = RD.NLArl(), []
    try:
        for pc in NEW:
            X, outs = P[42][pc], []
            for j in range(0, len(X), 32):
                outs += nla.read_batch(list(X[j:j + 32]), k=2, seed=C.SEED)
            rows += [{"item": items[i], "reader": "nla", "layer": 42, "piece": pc, "samples": s} for i, s in enumerate(outs)]
            print(f"nla {pc} ({time.time() - t0:.0f}s) write {nla.last_write}", flush=True)
    finally:
        nla.close()
    (OUT / "readouts_nla.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"done {time.time() - t0:.0f}s", flush=True)


def boot(v, nb=2000, seed=0):
    rng, n = random.Random(seed), len(v)
    b = [np.mean([v[rng.randrange(n)] for _ in range(n)]) for _ in range(nb)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def phase_analyze():
    from c012 import _hit
    T = {json.loads(l)["item"]: json.loads(l)["target"] for l in open(SRC / "meta.jsonl")}
    S = {}
    files = [SRC / "readouts_oracle.jsonl", SRC / "readouts_nla.jsonl", OUT / "readouts_oracle.jsonl", OUT / "readouts_nla.jsonl"]
    for f in files:
        for l in open(f):
            r = json.loads(l)
            if r["piece"] in ["whole", "rest_J"] + NEW:
                S[(f"{r['reader']} L{r['layer']}", r["piece"], r["item"])] = float(np.mean([_hit(s, [T[r["item"]]]) for s in r["samples"]]))
    its = sorted(T)
    res = {"label": "EXPLORATORY, POST HOC (claude_ovarida): 014 content-matched control", "k_pca": K, "n_items": len(its),
           "gates": json.loads((OUT / "gates.json").read_text()), "readers": {}, "oracle_minus_nla": {}}
    D = {}
    for rd in ("oracle L40", "oracle L44", "nla L42"):
        sh = {pc: [S[(rd, pc, i)] for i in its] for pc in ["whole", "rest_J"] + NEW}
        cost = {pc: [w - x for w, x in zip(sh["whole"], sh[pc])] for pc in ["rest_J"] + NEW}
        dP = [a - b for a, b in zip(cost["rest_J"], cost["rest_PCA"])]
        dR = [a - (b + c) / 2 for a, b, c in zip(cost["rest_J"], cost["rest_R0"], cost["rest_R1"])]
        D[rd] = dP
        m, ci = float(np.mean(dP)), boot(dP)
        cJ, cP = float(np.mean(cost["rest_J"])), float(np.mean(cost["rest_PCA"]))
        verdict = ("J-specific beyond content" if ci[0] > 0 and m >= 0.05 else
                   "content, not J" if ci[0] <= 0 <= ci[1] and cP >= 0.5 * cJ else "unresolved")
        res["readers"][rd] = {"share": {pc: float(np.mean(v)) for pc, v in sh.items()},
                              "cost": {pc: {"mean": float(np.mean(v)), "ci95": boot(v)} for pc, v in cost.items()},
                              "D_PCA": {"mean": m, "ci95": ci}, "D_R": {"mean": float(np.mean(dR)), "ci95": boot(dR)}, "verdict": verdict}
    for o in ("oracle L40", "oracle L44"):
        d = [a - b for a, b in zip(D[o], D["nla L42"])]
        res["oracle_minus_nla"][f"{o} - nla L42 | D_PCA"] = {"mean": float(np.mean(d)), "ci95": boot(d)}
    om = list(res["oracle_minus_nla"].values())
    res["verdict_014_contrast"] = ("survives content matching" if any(x["ci95"][0] > 0 for x in om) else
                                   "does not survive" if all(x["ci95"][0] <= 0 <= x["ci95"][1] and x["mean"] < 0.05 for x in om) else "unresolved")
    RES.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("readers", "oracle_minus_nla", "verdict_014_contrast")}, indent=1))


if __name__ == "__main__":
    {"read": phase_read, "analyze": phase_analyze}[sys.argv[1]]()
