"""007 brew test replication, material (design §7 + O1/O2; AMENDMENT_02 use 1).

  python brew_test.py capture     # GPU, SEC as in 001: the 3 emission cells at L40/42/44 for every test render
  python brew_test.py pieces      # CPU: roles (off pair, collisions, 001 gate flag), the 15 test pieces with dev-frozen
                                  # operators, positive-control pieces, G-recon

Material (O1): all test start candidates (members + every token-aligned partner) of the 49 test tables; 001's existing
reader-free gate result is attached per pair (secondary gated-only analysis). Reader-side; never imported by label code.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
import p007 as C                                                          # noqa: E402

D_T = C.OUT / "brew_test"
CELLS3 = ("emit_asst", "emit_think", "emit_colon")
WANT = {"emit_asst": ("assistant", "\n"), "emit_think": ("</think>", "\n\n"), "emit_colon": ("Answer", ":")}


def entries(tok):
    import brew_material as BM
    return [e for e in BM.brew_pairs(tok) if e["split"] == "test"]


def capture():
    import brew_material as BM
    import regime as RG
    import registry as R
    D_T.mkdir(parents=True, exist_ok=True)
    log = C.Logger(D_T / "log.txt")
    tok = R.make_tokenizer()
    ents = entries(tok)
    renders, meta = {}, []
    for e in ents:
        it = e["item"]
        for role, r, start, z, a in (("member", e["r_m"], e["pair"].m.inp, e["pair"].m.z, e["pair"].m.a),
                                     ("partner", e["r_p"], e["pair"].p.inp, e["pair"].p.z, e["pair"].p.a)):
            k = BM.ids_key(r.ids)
            if k not in renders:
                renders[k] = r
                meta.append({"key": k, "item": it["name"], "role": role, "start": start, "z": z, "answer": a})
    log(f"test renders: {len(meta)} ({sum(m['role'] == 'member' for m in meta)} members) over {len({m['item'] for m in meta})} tables")
    model = R.load_model()
    X = np.zeros((len(meta), 3, len(C.LAYERS), C.D), dtype=np.float32)
    with RG.SecEngine(model, l0=39) as sec:
        for i, m in enumerate(meta):
            r = renders[m["key"]]
            em = {g: p for p, g in r.regions.items() if g.startswith("emit_")}
            for c in CELLS3:                                                 # the cell-region assertion
                assert r.decoded[em[c] - 1] == WANT[c][0] and r.decoded[em[c]] == WANT[c][1], (m["item"], c)
            _, acts = sec.forward(r.ids, record=list(C.LAYERS))
            for j, L in enumerate(C.LAYERS):
                X[i, :, j] = acts[L][[em[c] for c in CELLS3]].float().cpu().numpy()
            sec.clear()
    np.savez(D_T / "captures_test.npz", X=X)
    C.dump(D_T / "captures_test_meta.json", {"layers": list(C.LAYERS), "cells": list(CELLS3), "renders": meta,
                                              "regime": "SEC, l0 = 39 (001)"})
    log("captured", X.shape)


def pieces():
    import wsb
    from pieces import f32_orth, rand_basis_f32
    log = C.Logger(D_T / "log.txt")
    meta = json.loads((D_T / "captures_test_meta.json").read_text())
    R_ = meta["renders"]
    X = np.load(D_T / "captures_test.npz")["X"]
    tables = sorted({r["item"] for r in R_})
    t_of = np.array([tables.index(r["item"]) for r in R_])
    mem = {r["item"]: i for i, r in enumerate(R_) if r["role"] == "member"}
    pairs_idx = [(mem[r["item"]], i) for i, r in enumerate(R_) if r["role"] == "partner"]
    gate = {}
    for l in open(C.PROJECT / "results" / "001_leverage_profile" / "analysis" / "gate_pairs.jsonl"):
        g = json.loads(l)
        gate[g["pair_id"]] = g
    _, brew = wsb.load_bank("brew")
    items = {it["name"]: it for it in brew}
    rng = np.random.default_rng(C.SEED)
    rows = []
    for a, b in pairs_idx:
        m, p = R_[a], R_[b]
        path = {m["start"], m["z"], m["answer"], p["start"], p["z"], p["answer"]}
        offs = [c for c in items[m["item"]]["options_adjacent"][0] if c not in path]
        rest = [c for c in C.COLOURS if c not in path and c not in offs]
        rest = [rest[i] for i in rng.permutation(len(rest))]
        cand = (offs + rest)[:2]
        if len(cand) == 2 and rng.random() < 0.5:
            cand = cand[::-1]
        pid = f"{m['item']}__{p['start']}"
        rows.append({"m": int(a), "p": int(b), "table": int(t_of[a]), "item": m["item"], "pair_id": pid,
                     "gated": bool(gate.get(pid, {}).get("pass", False)), "gate_found": pid in gate,
                     "m_roles": {k: m[k] for k in ("start", "z", "answer")}, "p_roles": {k: p[k] for k in ("start", "z", "answer")},
                     "collide": {k: m[k] == p[k] or bool({m[k], p[k]} & {x[o] for x in (m, p) for o in ("start", "z", "answer") if o != k})
                                 for k in ("start", "z", "answer")},
                     "off": cand if len(cand) == 2 else None})
    C.dump(D_T / "roles.json", {"renders": R_, "t_of": t_of.tolist(), "pairs": rows,
                                "n_tables": len(tables), "n_pairs": len(rows),
                                "n_gated": sum(r["gated"] for r in rows), "n_gate_found": sum(r["gate_found"] for r in rows)})
    log(f"test roles: {len(rows)} pairs, {len(tables)} tables, gated {sum(r['gated'] for r in rows)} "
        f"(gate rows found {sum(r['gate_found'] for r in rows)})")
    B = torch.load(C.STAGE0 / "bases_f32.pt")
    pca = torch.load(C.STAGE0 / "pca_bases.pt")
    pc = torch.load(C.POS_ACTS)
    (D_T / "pieces").mkdir(parents=True, exist_ok=True)
    out = {"G_recon": {}}
    n = len(R_)
    for li, L in enumerate(C.LAYERS):
        H = torch.from_numpy(X[:, 2, li].astype(np.float64))                  # ':'
        tot = H.sum(0)
        mu = torch.empty_like(H)
        for tb in range(len(tables)):
            m = torch.from_numpy(t_of == tb)
            mu[m] = (tot - H[m].sum(0)) / (n - int(m.sum()))
        U = mu / mu.norm(dim=1, keepdim=True)
        Pj = lambda A, Q: (A @ Q) @ Q.T
        QJ = B[f"J_L{L}"].double()
        Qk = f32_orth(pca[f"colon_L{L}"]["V_all"][:, :C.PCA_K[("colon", L)]])[0].double()   # dev fit, k frozen
        MR = H - (H * U).sum(1, keepdim=True) * U
        pcs = {"whole": H, "J": Pj(H, QJ), "rest_J": H - Pj(H, QJ), "PCA": Pj(H, Qk), "MR": MR, "J_c": Pj(MR, QJ)}
        Rb = {r: rand_basis_f32(C.rand_seed("r1024", r), C.K)[0].double() for r in range(3)}
        for r in range(3):
            pcs[f"R{r}"] = Pj(H, Rb[r])
            pcs[f"rest_R{r}"] = H - pcs[f"R{r}"]
            pcs[f"R{r}_c"] = Pj(MR, Rb[r])
        assert set(pcs) == set(C.TEST_PIECES)
        p32 = {k: v.float() for k, v in pcs.items()}
        np.savez(D_T / "pieces" / f"pieces_colon_L{L}.npz", **{k: v.numpy() for k, v in p32.items()})
        hn = H.norm(dim=1)
        rec = max(float(((p32[a].double() + p32[b].double() - H).norm(dim=1) / hn).max())
                  for a, b in [("J", "rest_J")] + [(f"R{r}", f"rest_R{r}") for r in range(3)])
        orth = float(((p32["MR"].double() * U).sum(1).abs() / hn).max())
        out["G_recon"][L] = {"max_rel_recon_err": rec, "max_rel_MR_dot_u": orth, "pass": rec <= 1e-6 and orth <= 1e-6}
        # positive-control pieces with the test-stage operators (colour-mean centring and the frozen test-mean u_b)
        h = pc[:, C.CAP_LAYERS.index(L)].double()
        mu_c = (h.sum(0, keepdim=True) - h) / 11.0
        Uc = mu_c / mu_c.norm(dim=1, keepdim=True)
        MRc = h - (h * Uc).sum(1, keepdim=True) * Uc
        ub = H.mean(0)
        ub = ub / ub.norm()
        MRb = h - (h @ ub).unsqueeze(1) * ub.unsqueeze(0)
        q = {"whole": h, "J": Pj(h, QJ), "rest_J": h - Pj(h, QJ), "PCA": Pj(h, Qk), "MR": MRc, "J_c": Pj(MRc, QJ),
             "MR@brew": MRb, "J_c@brew": Pj(MRb, QJ)}
        for r in range(3):
            q[f"R{r}"] = Pj(h, Rb[r])
            q[f"rest_R{r}"] = h - q[f"R{r}"]
            q[f"R{r}_c"] = Pj(MRc, Rb[r])
            q[f"R{r}_c@brew"] = Pj(MRb, Rb[r])
        np.savez(D_T / "pieces" / f"pcpieces_colon_L{L}.npz", **{k: v.float().numpy() for k, v in q.items()})
    C.dump(C.RES / "analysis" / "brew_test_material.json", out)
    log("test pieces", out)
    assert all(v["pass"] for v in out["G_recon"].values()), "G-recon (test) failed"


if __name__ == "__main__":
    {"capture": capture, "pieces": pieces}[sys.argv[1]]()
