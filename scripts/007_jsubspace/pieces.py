"""007 Stage 0 (CPU; design §3.1, §4.6, §6.2): float32 bases re-orthonormalised by QR (errors logged), every Stage 1a/1b
piece of the 001 brew dev captures, the positive-control pieces, the PCA bases Stage 2 and the brew test need, the
role table (incl. the off-trajectory colour pair per member-partner pair), and gates G-bases, G-geom, G-recon.
Reader-side; never imported by label code.

  python pieces.py            # everything; writes outputs/007_jsubspace/stage0/ and results/007_jsubspace/analysis/stage0.json
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
from p007 import (CAPS, CAP_LAYERS, CELLS, CELL_INDEX, COLOURS, COMPLEMENTS, D, K, LAYERS, N_RAND, N_RAND_C,  # noqa: E402
                  N_REST_RAND, PCA_K, POS_ACTS, RES, SEED, STAGE0, Logger, dump, rand_seed)

torch.set_num_threads(32)
log = Logger(STAGE0 / "log.txt")
RDELTA_DIM = {40: 1418, 42: 1245, 44: 1429}                               # round(5120 x median J delta-share at ':')


def f32_orth(Q):
    """Cast to float32 and re-orthonormalise by QR (sign-fixed). Returns (Q32, err_before, err_after)."""
    Q = Q.float()
    I = torch.eye(Q.shape[1])
    before = float((Q.T @ Q - I).abs().max())
    q, r = torch.linalg.qr(Q)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    after = float((q.T @ q - I).abs().max())
    return q.contiguous(), before, after


def rand_basis_f32(seed, k):
    return f32_orth(G.rand_basis(seed, k))


def material():
    """The 231 member/partner renders of the 48 dev tables, in capture order (= 006's order)."""
    X, keep, t_of, z, pairs = G.brew_material()
    meta = json.loads(G.CAPS_META.read_text())["renders"]
    R_ = [meta[i] for i in keep]
    return X, keep, t_of, R_, pairs


def role_table(R_, pairs, t_of):
    """Per pair: the member/partner indices, roles, collision flags (006 census) and the off-trajectory pair (o1, o2):
    colours in none of {start, z, answer} of either member, from WSB's option list (options_adjacent[0], as 002's
    table candidates) in order, then a seeded draw (SEED) from the rest of the palette; o1/o2 order seeded."""
    sys.path.insert(0, str(HERE.parent / "common"))
    import wsb
    _, brew = wsb.load_bank("brew")
    items = {it["name"]: it for it in brew}
    rng = np.random.default_rng(SEED)
    rows = []
    for a, b in pairs:
        m, p = R_[a], R_[b]
        it = items[m["item"]]
        path = {m["start"], m["z"], m["answer"], p["start"], p["z"], p["answer"]}
        offs = [c for c in it["options_adjacent"][0] if c not in path]
        rest = [c for c in COLOURS if c not in path and c not in offs]
        rest = [rest[i] for i in rng.permutation(len(rest))]
        cand = (offs + rest)[:2]
        if len(cand) == 2 and rng.random() < 0.5:
            cand = cand[::-1]
        rows.append({"m": int(a), "p": int(b), "table": int(t_of[a]), "item": m["item"],
                     "m_roles": {k: m[k] for k in ("start", "z", "answer")},
                     "p_roles": {k: p[k] for k in ("start", "z", "answer")},
                     # 006's role-collision census (reanalyse_006.collides): the role's colour of either member
                     # appears in another role of either member; start/answer D use collision-free pairs only
                     "collide": {k: m[k] == p[k] or bool({m[k], p[k]} & {x[o] for x in (m, p) for o in
                                                                       ("start", "z", "answer") if o != k})
                                 for k in ("start", "z", "answer")},
                     "off": cand if len(cand) == 2 else None,
                     "off_source": ["wsb" if c in offs else "seeded" for c in cand]})
    return rows


def main():
    STAGE0.mkdir(parents=True, exist_ok=True)
    out = {"design": "docs/design_specs/007_jsubspace.md §3.1, §4.6", "gates": {}, "bases": {}, "geometry": {}}
    X, keep, t_of, R_, pairs = material()
    roles = role_table(R_, pairs, t_of)
    dump(STAGE0 / "roles.json", {"renders": [{k: r[k] for k in ("key", "item", "role", "start", "z", "answer")} for r in R_],
                                 "t_of": t_of.tolist(), "pairs": roles})
    n_off = sum(r["off"] is not None for r in roles)
    out["roles"] = {"pairs": len(roles), "pairs_with_off_pair": n_off,
                    "off_from_wsb_both": sum(r["off"] is not None and r["off_source"] == ["wsb", "wsb"] for r in roles),
                    "collision_free": {k: sum(not r["collide"][k] for r in roles) for k in ("start", "z", "answer")}}
    log("roles", out["roles"])

    # ---------------------------------------------------------------- bases: float32, QR re-orthonormalised (fix 11)
    errs = {}
    B = {}
    for L in LAYERS:
        B[f"J_L{L}"], b, a = f32_orth(torch.load(G.BASES / f"J_raw_L{L}.pt", weights_only=False)["Q"][:, :K])
        errs[f"J_raw_L{L}"] = (b, a)
    B["LL"], b, a = f32_orth(torch.load(G.BASES / "LL_gain.pt", weights_only=False)["Q"][:, :K])
    errs["LL_gain"] = (b, a)
    for r in range(N_RAND):
        B[f"R{r}"], b, a = rand_basis_f32(rand_seed("r1024", r), K)
        errs[f"r1024_{r}"] = (b, a)
    for L in LAYERS:                                                     # Stage 2 families: checked here, regenerated there
        for r in range(10):
            _, b, a = rand_basis_f32(rand_seed("rdelta", r), RDELTA_DIM[L])
            errs[f"rdelta_L{L}_{r}"] = (b, a)
    for kind, k in (("r512", 512), ("r2560", 2560)):
        for r in range(3):
            _, b, a = rand_basis_f32(rand_seed(kind, r), k)
            errs[f"{kind}_{r}"] = (b, a)
    # chain (Stage 5): J_raw at L56, built as report 22's bases (raw W_U, J_NP pin)
    f56 = G.BASES / "J_raw_L56.pt"
    if not f56.exists():
        import readers
        W, _ = G.load_W_and_gain()
        lens = readers.load_lenses(("J_NP",), layers=[56], device="cpu", verify_sha=True)["J_NP"]
        ev, Q = G.eig_basis(G.gram(W, lens.jacobians[56].float()))
        torch.save({"Q": Q[:, :G.KEEP].float(), "ev": ev}, f56)
        del W, lens
        log("built J_raw_L56")
    B["J_L56"], b, a = f32_orth(torch.load(f56, weights_only=False)["Q"][:, :K])
    errs["J_raw_L56"] = (b, a)
    out["bases"] = {k: {"err_before": v[0], "err_after": v[1]} for k, v in errs.items()}
    worst = max(v[1] for v in errs.values())
    out["gates"]["G_bases"] = {"max_err_after_qr": worst, "threshold": 1e-5, "pass": worst <= 1e-5}
    log("G-bases worst |Q^T Q - I| after QR", worst)
    assert worst <= 1e-5, "G-bases failed"
    torch.save({k: v for k, v in B.items() if not k.startswith("R")}, STAGE0 / "bases_f32.pt")

    # ---------------------------------------------------------------- PCA bases (cross-fitted 4 table folds; full-dev fit)
    folds = G.table_folds(48)
    pca = {}
    geo_ref = json.loads((RES / "analysis" / "geometry_brew_finek.json").read_text())["cells"]
    n = len(keep)
    for cname, cell in CELLS.items():
        ci = CELL_INDEX[cell]
        for li, L in enumerate(LAYERS):
            H = torch.from_numpy(np.asarray(X[keep, ci, li], dtype=np.float64))
            Vf = []
            for f in folds:
                tr = [j for j in range(n) if t_of[j] not in f]
                V, _ = G.within_table_pca(H, t_of, tr)
                Vf.append(V[:, :16].float())
            Vall, _ = G.within_table_pca(H, t_of, list(range(n)))
            pca[(cname, L)] = {"folds": [sorted(f) for f in folds], "V_folds": Vf, "V_all": Vall[:, :16].float()}
            assert geo_ref[f"{cname}_L{L}"]["PCA_k_delta"]["k"] == PCA_K[(cname, L)]
    torch.save({f"{c}_L{L}": v for (c, L), v in pca.items()}, STAGE0 / "pca_bases.pt")

    # ---------------------------------------------------------------- pieces + G-geom + G-recon
    pc = torch.load(POS_ACTS)                                              # [12, 5, d]
    recon_worst, orth_worst, geom_worst = 0.0, 0.0, 0.0
    iM = np.array([a for a, b in pairs]); iP = np.array([b for a, b in pairs])
    un = lambda A: A / A.norm(dim=1, keepdim=True)
    for cname, cell in CELLS.items():
        ci = CELL_INDEX[cell]
        for li, L in enumerate(LAYERS):
            H = torch.from_numpy(np.asarray(X[keep, ci, li], dtype=np.float64))
            tot = H.sum(0)
            mu = torch.empty_like(H)
            for tb in range(48):
                m = torch.from_numpy(t_of == tb)
                mu[m] = (tot - H[m].sum(0)) / (n - int(m.sum()))
            U = mu / mu.norm(dim=1, keepdim=True)
            P = lambda A, Q: (A @ Q) @ Q.T
            QJ, QL = B[f"J_L{L}"].double(), B["LL"].double()
            k = PCA_K[(cname, L)]
            Hp = torch.empty_like(H)
            for f, V in zip(folds, pca[(cname, L)]["V_folds"]):
                m = torch.from_numpy(np.array([t in f for t in t_of]))
                Qk = f32_orth(V[:, :k])[0].double()
                Hp[m] = P(H[m], Qk)
            MRp = H - (H * U).sum(1, keepdim=True) * U
            pcs = {"whole": H, "J": P(H, QJ), "rest_J": H - P(H, QJ), "PCA": Hp, "rest_PCA": H - Hp,
                   "LL": P(H, QL), "rest_LL": H - P(H, QL), "MR": MRp, "J_c": P(MRp, QJ)}
            for r in range(N_RAND):
                QR = B[f"R{r}"].double()
                pcs[f"R{r}"] = P(H, QR)
                if r < N_REST_RAND:
                    pcs[f"rest_R{r}"] = H - pcs[f"R{r}"]
                if r < N_RAND_C:
                    pcs[f"R{r}_c"] = P(MRp, QR)
            p32 = {k_: v.float() for k_, v in pcs.items()}
            np.savez(STAGE0 / f"pieces_{cname}_L{L}.npz", **{k_: v.numpy() for k_, v in p32.items()})
            # G-recon (float64 from the stored float32 pieces)
            hn = H.norm(dim=1)
            for a_, b_ in COMPLEMENTS:
                e = ((p32[a_].double() + p32[b_].double() - H).norm(dim=1) / hn).max()
                recon_worst = max(recon_worst, float(e))
            orth_worst = max(orth_worst, float(((p32["MR"].double() * U).sum(1).abs() / hn).max()))
            # geometry of every piece (context) and G-geom against report 22 (v1 pieces)
            Dl = H[iM] - H[iP]
            geo = {}
            for nm, A in pcs.items():
                ds = ((A[iM] - A[iP]) ** 2).sum(1) / (Dl ** 2).sum(1)
                rv = (un(A[iM]) - un(A[iP])).norm(dim=1) / (un(H[iM]) - un(H[iP])).norm(dim=1)
                geo[nm] = {"delta_share": float(ds.median()), "visible_ratio": float(rv.median()),
                           "energy": float(((A ** 2).sum(1) / (H ** 2).sum(1)).median())}
            ref = geo_ref[f"{cname}_L{L}"]
            pairs_ref = {"J": "J_raw", "rest_J": "rest_J_raw", "LL": "LL_gain", "rest_LL": "rest_LL_gain",
                         "MR": "mean_removed", "PCA": "PCA_k_delta", "rest_PCA": "rest_PCA_k_delta"}
            gg = {}
            for ours, theirs in pairs_ref.items():
                for mk, rk in (("delta_share", "delta_share"), ("visible_ratio", "reader_visible_ratio")):
                    gg[f"{ours}.{mk}"] = abs(geo[ours][mk] - ref[theirs][rk]["median"])
            for mk, rk in (("delta_share", "delta_share"), ("visible_ratio", "reader_visible_ratio")):
                gg[f"R(10).{mk}"] = abs(float(np.median([geo[f"R{r}"][mk] for r in range(N_RAND)])) -
                                        ref["random (10 draws)"][rk]["median_of_draw_medians"])
            # A1 (first Stage 0 execution failed here): the rest-of-random comparison must be like for like. Per draw
            # r = 0..2 (and 0..9 for random), report 22's own float64 construction vs our float32 piece.
            for r in range(N_RAND):
                Q64 = G.rand_basis(SEED + r)
                for nm, A in ((f"R{r}", P(H, Q64)), (f"rest_R{r}", H - P(H, Q64))):
                    if nm not in pcs:
                        continue
                    ds = float((((A[iM] - A[iP]) ** 2).sum(1) / (Dl ** 2).sum(1)).median())
                    rv = float(((un(A[iM]) - un(A[iP])).norm(dim=1) / (un(H[iM]) - un(H[iP])).norm(dim=1)).median())
                    gg[f"{nm}.delta_share(per-draw)"] = abs(ds - geo[nm]["delta_share"])
                    gg[f"{nm}.visible_ratio(per-draw)"] = abs(rv - geo[nm]["visible_ratio"])
            geom_worst = max(geom_worst, max(gg.values()))
            # A1: the Visible model's P5 prediction from the measured post-projection geometry (random draws 0-2)
            sep = lambda A: float((un(A[iM]) - un(A[iP])).norm(dim=1).median())
            rc = [f"R{r}_c" for r in range(N_RAND_C)]
            p5 = {"J_c": {"delta_share": geo["J_c"]["delta_share"], "visible_ratio": geo["J_c"]["visible_ratio"],
                          "normalised_separation": sep(pcs["J_c"])},
                  "R_c(mean of 0-2)": {"delta_share": float(np.mean([geo[x]["delta_share"] for x in rc])),
                                       "visible_ratio": float(np.mean([geo[x]["visible_ratio"] for x in rc])),
                                       "normalised_separation": float(np.mean([sep(pcs[x]) for x in rc]))}}
            vj, vr = p5["J_c"]["visible_ratio"], p5["R_c(mean of 0-2)"]["visible_ratio"]
            p5["visible_ratio_rel_diff"] = abs(vj - vr) / min(vj, vr)
            p5["visible_model_P5_prediction"] = "≈" if p5["visible_ratio_rel_diff"] <= 0.15 else "no prediction"
            out.setdefault("P5_geometry", {})[f"{cname}_L{L}"] = p5
            out["geometry"][f"{cname}_L{L}"] = {"pieces": geo, "G_geom_absdiff": gg, "PCA_k": k}
            log(f"pieces {cname} L{L}: G-geom worst {max(gg.values()):.5f}; J_c ds {geo['J_c']['delta_share']:.3f} "
                f"vis {geo['J_c']['visible_ratio']:.2f}; R0_c ds {geo['R0_c']['delta_share']:.3f} vis {geo['R0_c']['visible_ratio']:.2f}")

        # positive-control pieces (cell-specific only through PCA, which uses the full-dev fit of that cell)
        for L in LAYERS:
            h = pc[:, CAP_LAYERS.index(L)].double()                       # [12, d]
            tot = h.sum(0)
            mu = (tot.unsqueeze(0) - h) / 11.0
            U = mu / mu.norm(dim=1, keepdim=True)
            P = lambda A, Q: (A @ Q) @ Q.T
            QJ, QL = B[f"J_L{L}"].double(), B["LL"].double()
            Qk = f32_orth(pca[(cname, L)]["V_all"][:, :PCA_K[(cname, L)]])[0].double()
            MRp = h - (h * U).sum(1, keepdim=True) * U
            pcs = {"whole": h, "J": P(h, QJ), "rest_J": h - P(h, QJ), "PCA": P(h, Qk), "rest_PCA": h - P(h, Qk),
                   "LL": P(h, QL), "rest_LL": h - P(h, QL), "MR": MRp, "J_c": P(MRp, QJ)}
            for r in range(N_RAND):
                QR = B[f"R{r}"].double()
                pcs[f"R{r}"] = P(h, QR)
                if r < N_REST_RAND:
                    pcs[f"rest_R{r}"] = h - pcs[f"R{r}"]
                if r < N_RAND_C:
                    pcs[f"R{r}_c"] = P(MRp, QR)
            # A1: the frozen brew operators on the colour activations, alongside the colour-mean version above.
            # Template mean: u_b = unit(mean of the 231 dev renders at this cell and layer). The generic-mean versions
            # (GMR, J o gc, R o gc) need u_gen and are built at run time in Phase A.
            Hb = torch.from_numpy(np.asarray(X[keep, CELL_INDEX[cell], LAYERS.index(L)], dtype=np.float64))
            ub = Hb.mean(0); ub = ub / ub.norm()
            MRb = h - (h @ ub).unsqueeze(1) * ub.unsqueeze(0)
            pcs["MR@brew"] = MRb
            pcs["J_c@brew"] = P(MRb, QJ)
            for r in range(N_RAND_C):
                pcs[f"R{r}_c@brew"] = P(MRb, B[f"R{r}"].double())
            np.savez(STAGE0 / f"pcpieces_{cname}_L{L}.npz", **{k_: v.float().numpy() for k_, v in pcs.items()})
            torch.save(ub.float(), STAGE0 / f"u_brew_{cname}_L{L}.pt")
    out["gates"]["G_recon"] = {"max_rel_recon_err": recon_worst, "max_rel_MR_dot_u": orth_worst, "threshold": 1e-6,
                               "pass": recon_worst <= 1e-6 and orth_worst <= 1e-6}
    out["gates"]["G_geom"] = {"max_absdiff_vs_report22": geom_worst, "threshold": 0.005, "pass": geom_worst <= 0.005}
    log("G-recon", out["gates"]["G_recon"], "G-geom", out["gates"]["G_geom"])
    dump(RES / "analysis" / "stage0.json", out)
    assert out["gates"]["G_recon"]["pass"], "G-recon failed"
    assert out["gates"]["G_geom"]["pass"], "G-geom failed"
    log("stage 0 (pieces) passed")


if __name__ == "__main__":
    main()
