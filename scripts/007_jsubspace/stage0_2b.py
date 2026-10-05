"""007 Stage 0, part for Stage 2b (design §6.1-6.2; A1): class-mean z directions at ':' L40/42/44 from the dev SEC
captures (reader-free: labels from the bank), each split by nonnegative gradient pursuit (k = 25, NNLS refit) over the
folded J-lens atoms (gain convention) of J_NP and of J_CB, into a J part and the remainder.

  d_c^(-T) = mean(h | z = c, tables != T) - mean(h | tables != T)      (for the table T of each of 004's 33 pairs)
  d_c      = the same over all 48 tables                               (for the per-colour report)

Writes outputs/007_jsubspace/stage0/stage2b_directions.pt and results/007_jsubspace/analysis/stage0_2b.json.
Reader-side (jspace); never imported by label code.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
import geometry as G                                                      # noqa: E402
import jspace                                                             # noqa: E402
import p007 as C                                                          # noqa: E402

torch.set_num_threads(16)
K_PURSUIT = 25
log = C.Logger(C.STAGE0 / "log_2b.txt")


def pairs_004():
    """004's 33 frozen pairs: item, member z, partner z (from 004's saved clean rows)."""
    man = json.loads((C.PROJECT / "results" / "004_span_assay" / "manifest_pairs.json").read_text())
    out = []
    for pid in man["pairs"]:
        r = json.loads(open(C.PROJECT / "outputs" / "004_span_assay" / "run" / "pairs" / f"{pid}.jsonl").readline())
        assert r["kind"] == "clean" and r["pair_id"] == pid
        out.append({"pair_id": pid, "item": r["item"], "z": r["z"], "z2": r["z2"], "a": r["a"], "a2": r["a2"]})
    return out


def main():
    import readers
    from transformers import AutoTokenizer
    X, keep, t_of, R_, pairs = G.brew_material()
    meta = json.loads(G.CAPS_META.read_text())["renders"]
    rend = [meta[i] for i in keep]
    tables = sorted({r["item"] for r in rend})
    z = np.array([r["z"] for r in rend])
    P4 = pairs_004()
    for p in P4:
        assert p["item"] in tables, p["item"]
        p["table"] = tables.index(p["item"])
    snap = next(Path("/workspace/.hf_home/hub/models--Qwen--Qwen3.6-27B/snapshots").iterdir())
    tok = AutoTokenizer.from_pretrained(str(snap))
    own = {c: sorted({tok.encode(" " + c, add_special_tokens=False)[0], tok.encode(" " + c.capitalize(), add_special_tokens=False)[0]})
           for c in C.COLOURS}
    W, g = G.load_W_and_gain()
    out_t, rep = {}, {"k": K_PURSUIT, "refit": "nnls", "atoms": "unit((g * W_U[t]) @ J_l), g = 1 + final-norm weight",
                      "pairs": [{k: p[k] for k in ("pair_id", "item", "table", "z", "z2")} for p in P4], "layers": {}}
    for li, L in enumerate(C.LAYERS):
        H = torch.from_numpy(np.asarray(X[keep, 2, li], dtype=np.float64))     # ':' cell
        rows, keys = [], []
        for p in P4:                                                         # leave the pair's table out
            m = t_of != p["table"]
            gm = H[torch.from_numpy(m)].mean(0)
            for c in (p["z"], p["z2"]):
                sel = torch.from_numpy(m & (z == c))
                assert int(sel.sum()) >= 1, (p["pair_id"], c)
                rows.append(H[sel].mean(0) - gm)
                keys.append(("loto", p["table"], c))
        gm = H.mean(0)
        for c in C.COLOURS:                                                  # all tables (per-colour report)
            sel = torch.from_numpy(z == c)
            if int(sel.sum()) == 0:
                continue
            rows.append(H[sel].mean(0) - gm)
            keys.append(("all", -1, c))
        Xd = torch.stack(rows).float()
        rep["layers"][L] = {}
        for lens_name in ("J_NP", "J_CB"):
            lens = readers.load_lenses((lens_name,), layers=[L], device="cpu", verify_sha=True)[lens_name]
            Dm = jspace.atoms(lens.jacobians[L].float(), W, gain=g)
            res = jspace.decompose(Xd, Dm, K_PURSUIT, refit="nnls", chunk=65536)
            del Dm, lens
            for (kind, tb, c), xj, xn, e, sup in zip(keys, res["x_j"], res["x_nonj"], res["energy_j"], res["support"]):
                out_t[(L, lens_name, kind, tb, c)] = {"full": None, "J": xj.clone(), "nonJ": xn.clone()}
            allrows = [(c, float(e), any(t in own[c] for t in sup)) for (kind, tb, c), e, sup in
                       zip(keys, res["energy_j"], res["support"]) if kind == "all"]
            loto = [float(e) for (kind, _, _), e in zip(keys, res["energy_j"]) if kind == "loto"]
            loto_own = [any(t in own[c] for t in sup) for (kind, _, c), sup in zip(keys, res["support"]) if kind == "loto"]
            rep["layers"][L][lens_name] = {
                "per_colour_all_tables": {c: {"energy_J": e, "own_atom_in_support": o} for c, e, o in allrows},
                "median_energy_J_all_tables": float(np.median([e for _, e, _ in allrows])),
                "loto_median_energy_J": float(np.median(loto)), "loto_own_atom_share": float(np.mean(loto_own)),
                "support_tokens_example": {c: [tok.decode([t]) for t in sup[:10]] for (kind, _, c), sup in
                                           zip(keys, res["support"]) if kind == "all"}}
            log(f"L{L} {lens_name}: median J energy (all tables) {rep['layers'][L][lens_name]['median_energy_J_all_tables']:.3f}; "
                f"own atom in support {sum(o for _, _, o in allrows)}/{len(allrows)}")
        for (kind, tb, c), x in zip(keys, Xd):
            for lens_name in ("J_NP", "J_CB"):
                out_t[(L, lens_name, kind, tb, c)]["full"] = x.clone()
    torch.save({"|".join(map(str, k)): v for k, v in out_t.items()}, C.STAGE0 / "stage2b_directions.pt")
    C.dump(C.RES / "analysis" / "stage0_2b.json", rep)
    log("stage 0 (2b directions) done")


if __name__ == "__main__":
    main()
