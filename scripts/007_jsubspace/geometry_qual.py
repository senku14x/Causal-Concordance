"""007 geometry: qualitative samples (CPU, reader-side). For 4 fixed-seed member-partner pairs at ':' L40/L42, the J_NP
cosine readout (WSB's J arm: (W_U J h) / ||J^T W_U[t]||, raw W_U) of each piece: top-10 tokens, and the rank of the
member's start / z / answer colours among the 12 palette colours (lowercase ' c' tokens). Also each pair's delta_share
per piece. Writes results/007_jsubspace/analysis/geometry_qual.json and .md.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "common"))
import geometry as G  # noqa: E402

COLOURS = ("black", "blue", "brown", "gold", "gray", "green", "orange", "pink", "purple", "red", "teal", "white")


def main():
    import readers
    from transformers import AutoTokenizer
    snap = next(Path("/workspace/.hf_home/hub/models--Qwen--Qwen3.6-27B/snapshots").iterdir())
    tok = AutoTokenizer.from_pretrained(str(snap))
    W, _ = G.load_W_and_gain()
    lens = readers.load_lenses(("J_NP",), layers=[40, 42], device="cpu", verify_sha=True)["J_NP"]
    cid = []
    for c in COLOURS:
        ids = tok.encode(" " + c, add_special_tokens=False)
        cid.append(ids[0])                                   # " teal" may be 2 tokens: first token's score (noted)
    multi = [c for c in COLOURS if len(tok.encode(" " + c, add_special_tokens=False)) > 1]
    X, keep, t_of, z, pairs = G.brew_material()
    meta = json.loads(G.CAPS_META.read_text())["renders"]
    R_ = [meta[i] for i in keep]
    rng = np.random.default_rng(G.SEED)
    sel = sorted(rng.choice(len(pairs), 4, replace=False).tolist())
    out = {"pairs": [], "note": f"multi-token colours scored by their first token: {multi}"}
    md = ["# 007 geometry: qualitative samples (J_NP cosine readout of each piece)", "",
          f"4 member-partner pairs drawn with numpy default_rng({G.SEED}); ':' cell. Colour ranks are among the 12 palette "
          f"colours by the cosine readout (1 = top). Multi-token colours use their first token: {multi}.", ""]
    for L in (40, 42):
        J = lens.jacobians[L].float()
        den = torch.cat([(W[i:i + G.CHUNK] @ J).norm(dim=1) for i in range(0, W.shape[0], G.CHUNK)]).clamp_min(1e-9)
        QJ = G.load_basis(f"J_raw_L{L}").float()
        Qr = G.rand_basis(G.SEED).float()
        li = [40, 42, 44].index(L)
        H = torch.from_numpy(np.asarray(X[keep, 2, li], dtype=np.float32))
        n = H.shape[0]
        tot = H.sum(0)
        pieces = {"whole": lambda h: h, "J": lambda h: (h @ QJ) @ QJ.T, "rest_J": lambda h: h - (h @ QJ) @ QJ.T,
                  "random(seed0)": lambda h: (h @ Qr) @ Qr.T}
        for pi in sel:
            a, b = pairs[pi]
            tb = t_of[a]
            m = torch.from_numpy(t_of == tb)
            mu = (tot - H[m].sum(0)) / (n - int(m.sum()))
            u = mu / mu.norm()
            pieces["mean_removed"] = lambda h, u=u: h - (h @ u) * u
            rec = {"layer": L, "pair": pi, "table": R_[a]["item"]}
            for role, j in (("member", a), ("partner", b)):
                r = R_[j]
                rec[role] = {"start": r["start"], "z": r["z"], "answer": r["answer"], "pieces": {}}
                for nm, P in pieces.items():
                    v = P(H[j])
                    sc = (W @ (J @ v)) / den
                    top = [tok.decode([i]) for i in torch.topk(sc, 10).indices.tolist()]
                    cs = sc[cid]
                    order = torch.argsort(cs, descending=True).tolist()
                    rank = {COLOURS[k]: order.index(k) + 1 for k in range(12)}
                    rec[role]["pieces"][nm] = {"top10": top, "rank_start": rank[r["start"]], "rank_z": rank[r["z"]],
                                               "rank_answer": rank[r["answer"]]}
            Dl = H[a] - H[b]
            rec["delta_share"] = {nm: float((P(Dl) ** 2).sum() / (Dl ** 2).sum()) for nm, P in pieces.items() if nm != "whole"}
            out["pairs"].append(rec)
            md.append(f"## L{L}, table {rec['table']} (pair {pi})")
            md.append("")
            md.append("delta_share: " + ", ".join(f"{k} {v:.2f}" for k, v in rec["delta_share"].items()))
            md.append("")
            md.append("| render | piece | rank start / z / answer | J_NP cosine top-10 |")
            md.append("|---|---|---|---|")
            for role in ("member", "partner"):
                r = rec[role]
                for nm, pv in r["pieces"].items():
                    toks = " · ".join(repr(t)[1:-1] for t in pv["top10"])
                    md.append(f"| {role} ({r['start']}→{r['z']}→{r['answer']}) | {nm} | {pv['rank_start']} / {pv['rank_z']} / {pv['rank_answer']} | {toks} |")
            md.append("")
    G.dump("geometry_qual.json", out)
    (G.RES / "geometry_qual.md").write_text("\n".join(md))
    G.log("wrote", G.RES / "geometry_qual.md")


if __name__ == "__main__":
    main()
