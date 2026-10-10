"""012 Stage E (design `docs/design_specs/012E_ar_concept_swaps.md` §5): 007's dense J subspace at every edit layer.

CPU, READER-SIDE (uses the J_NP lens); never imported by label code. Uses 007's own construction unchanged
(`geometry.gram` + `geometry.eig_basis`: eig of sum_t unit(W[t] @ J_L) outer itself, W = lm_head, no gain), so the
bases are the same object as 007's `J_raw_L{L}`.

Sanity gate (design §5): the rebuilt L40, L42 and L44 bases must reproduce 007's committed ones (top 1,024, mean cos^2
>= 0.9999, `geometry.overlap`). If they don't, nothing here is used.

  python stageE_bases.py          -> outputs/012_reader_concordance/stageE/bases/J_raw_L{L}.pt
                                     results/012_reader_concordance/analysis/stageE_bases.json
"""
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "scripts" / "007_jsubspace"))
sys.path.insert(0, str(PROJECT / "scripts" / "common"))
import geometry as G                                                       # noqa: E402

OUT = PROJECT / "outputs" / "012_reader_concordance" / "stageE" / "bases"
RES = PROJECT / "results" / "012_reader_concordance" / "analysis" / "stageE_bases.json"
LAYERS = list(range(36, 49))           # multihop band L36-47 (nearest-tag edits at every layer) + brew L40/44/48
CHECK = (40, 42, 44)                   # 007's committed bases


def main():
    import readers
    t0 = time.time()
    W, _gain = G.load_W_and_gain()
    lens = readers.load_lenses(("J_NP",), layers=LAYERS, device="cpu", verify_sha=True)["J_NP"]
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"label": "012 Stage E: 007 dense J (J_raw) bases rebuilt at the edit layers; 007 construction unchanged",
            "construction": "eig of sum_t unit(W[t] @ J_L) outer itself; W = lm_head (raw, no gain); top 1,024 used",
            "layers": {}, "sanity_vs_007": {}}
    for L in LAYERS:
        t = time.time()
        ev, Q = G.eig_basis(G.gram(W, lens.jacobians[L]))
        torch.save({"Q": Q[:, :G.KEEP].float(), "ev": ev}, OUT / f"J_raw_L{L}.pt")
        share = ev.cumsum(0) / ev.sum()
        meta["layers"][str(L)] = {"share_top_1024": float(share[1023]), "seconds": round(time.time() - t, 1)}
        G.log(f"J_raw_L{L}", meta["layers"][str(L)])
        if L in CHECK:
            ours = G.orth(Q[:, :G.K].double())
            ref = G.load_basis(f"J_raw_L{L}")
            meta["sanity_vs_007"][str(L)] = G.overlap(ours, ref)
            G.log(f"sanity L{L}", meta["sanity_vs_007"][str(L)])
    meta["sanity_pass"] = all(v["mean_cos2"] >= 0.9999 for v in meta["sanity_vs_007"].values())
    meta["seconds_total"] = round(time.time() - t0, 1)
    RES.parent.mkdir(parents=True, exist_ok=True)
    RES.write_text(json.dumps(meta, indent=1))
    G.log("sanity_pass", meta["sanity_pass"], "wrote", RES)


if __name__ == "__main__":
    main()
