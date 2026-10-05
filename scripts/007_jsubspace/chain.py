"""007 Stage 5 — chain (design §10; exploratory; dropped first if over budget). Reader-side.

  python chain.py --run-id c1

001's gated dev chain start pairs (same steps, different start; single-digit intermediates by construction), restricted
to pairs whose intermediates and answers are all single digits for both members; dropped if fewer than 10 pairs.
Cell: 001's final cell (the render's last token) at L56; reader: the Oracle L56 only (on-contract). 21 v1 pieces with
the J_raw_L56 basis; PCA-k within step-sequence group (cross-fitted, delta-matched); mean-removed with the
leave-one-group-out mean. D on the first intermediate (digits 0-9, spellings " d" and "d").
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parent / "005_planted_concepts"))
import p005 as P5                                 # noqa: E402
import partners as PT                             # noqa: E402
import readers as RD                              # noqa: E402
import registry as R                              # noqa: E402
import splits as SP                               # noqa: E402
import wsb                                        # noqa: E402
import p007 as C                                  # noqa: E402
import reader_split as RS                         # noqa: E402
from pieces import f32_orth, rand_basis_f32       # noqa: E402

L56 = 56
DIGITS = [str(d) for d in range(10)]


def material(tok):
    _, chain = wsb.load_bank("chain")
    csplit, _ = SP.chain_split(chain)
    gate = {}
    for l in open(C.PROJECT / "results" / "001_leverage_profile" / "analysis" / "gate_pairs.jsonl"):
        g = json.loads(l)
        gate[g["pair_id"]] = g
    seen, out = set(), []
    for it in chain:
        if not PT.chain_single_token_item(it) or csplit[it["name"]] != "dev":
            continue
        for pair in PT.chain_candidates(it):
            key = frozenset((pair.m.text, pair.p.text))
            if key in seen:
                continue
            al = PT.token_align(tok, pair, it)
            if al is None:
                continue
            seen.add(key)
            g = gate.get(pair.pair_id, {})
            if not g.get("pass") or g.get("family") != "chain":
                continue
            zs = list(pair.m.z) + list(pair.p.z) + [pair.m.a, pair.p.a]
            if not all(isinstance(v, int) and 0 <= v <= 9 for v in zs):
                continue
            out.append({"pair": pair, "item": it, "r_m": al[0], "r_p": al[1], "group": " | ".join(it["steps"])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="c1")
    a = ap.parse_args()
    d = C.OUT / "stage5" / a.run_id
    (d / "ll").mkdir(parents=True, exist_ok=True)
    log = C.Logger(d / "log.txt")
    budget = C.OUT / "budget_drop.json"
    if budget.exists() and "5" in json.loads(budget.read_text()).get("dropped", []):
        log("Stage 5 dropped by the budget rule (design §13.3)")
        return
    disabled = json.loads(RS.DISABLED.read_text()) if RS.DISABLED.exists() else {}
    if "oracle" in disabled:
        log("Stage 5 not run: the Oracle path is disabled (A1)", disabled["oracle"])
        return
    tok = R.make_tokenizer()
    mats = material(tok)
    (d / "manifest.json").write_text(json.dumps({"run_id": a.run_id, "manifest": R.manifest_block(), "n_pairs": len(mats),
                                                 "design": "docs/design_specs/007_jsubspace.md (v2 + A1)"}, indent=1))
    log(f"chain single-digit gated dev pairs: {len(mats)} over {len({e['item']['name'] for e in mats})} items")
    if len(mats) < 10:
        log("Stage 5 dropped: fewer than 10 pairs (design §10)")
        C.dump(C.RES / "analysis" / "stage5.json", {"n_pairs": len(mats), "dropped": "fewer than 10 pairs"})
        return
    renders, idx = [], {}
    for e in mats:
        for role, r in (("member", e["r_m"]), ("partner", e["r_p"])):
            k = tuple(r.ids)
            if k not in idx:
                idx[k] = len(renders)
                renders.append({"r": r, "role": role, "group": e["group"], "item": e["item"]["name"]})
    pairs = []
    items = sorted({e["item"]["name"] for e in mats})
    for e in mats:
        pairs.append({"m": idx[tuple(e["r_m"].ids)], "p": idx[tuple(e["r_p"].ids)], "table": items.index(e["item"]["name"]),
                      "pair_id": e["pair"].pair_id, "z_m": int(e["pair"].m.z[0]), "z_p": int(e["pair"].p.z[0]),
                      "a_m": int(e["pair"].m.a), "a_p": int(e["pair"].p.a)})
    model = R.load_model()
    blocks = R.blocks(model)
    dev = next(model.parameters()).device
    X = np.zeros((len(renders), C.D), np.float32)
    with torch.no_grad():
        for n, rr in enumerate(renders):
            r = rr["r"]
            p = r.read_positions[-1]
            assert p == len(r.ids) - 1
            rec = P5.RowRecorder(blocks, [L56], [p])
            with rec:
                P5.text_model(model)(input_ids=torch.tensor([r.ids], device=dev), use_cache=False)
            X[n] = rec.acts[L56][0, 0].cpu().numpy()
    # pieces
    B = torch.load(C.STAGE0 / "bases_f32.pt")
    QJ, QL = B["J_L56"].double(), B["LL"].double()
    Rb = {r: rand_basis_f32(C.rand_seed("r1024", r), C.K)[0].double() for r in range(10)}
    H = torch.from_numpy(X.astype(np.float64))
    grp = np.array([rr["group"] for rr in renders])
    ug = sorted(set(grp))
    gfold = {g: i % 4 for i, g in enumerate(np.random.default_rng(C.SEED).permutation(ug))}
    Pj = lambda A, Q: (A @ Q) @ Q.T

    def wpca(rows):
        Hs, gs = H[rows], grp[rows]
        Cm = torch.zeros_like(Hs)
        for g in np.unique(gs):
            m = torch.from_numpy(gs == g)
            if int(m.sum()) >= 2:
                Cm[m] = (Hs[m] - Hs[m].mean(0, keepdim=True)) / float(np.sqrt(int(m.sum())))
        _, S_, Vh = torch.linalg.svd(Cm, full_matrices=False)
        return Vh[:int((S_ > S_[0] * 1e-8).sum())].T
    Vf = {f: wpca([i for i in range(len(renders)) if gfold[grp[i]] != f]) for f in range(4)}
    Dl = torch.stack([H[p["m"]] - H[p["p"]] for p in pairs])
    jds = float(np.median(((Pj(Dl, QJ) ** 2).sum(1) / (Dl ** 2).sum(1)).numpy()))
    kmax = min(V.shape[1] for V in Vf.values())
    best = None
    for k in range(1, kmax + 1):
        ds = [float(((Vf[gfold[grp[p["m"]]]][:, :k].T @ (H[p["m"]] - H[p["p"]])) ** 2).sum() / ((H[p["m"]] - H[p["p"]]) ** 2).sum())
              for p in pairs]
        gap = abs(float(np.median(ds)) - jds)
        if best is None or gap < best[1]:
            best = (k, gap)
    k = best[0]
    Hp = torch.stack([Pj(H[i:i + 1], f32_orth(Vf[gfold[grp[i]]][:, :k].float())[0].double())[0] for i in range(len(renders))])
    MR = torch.empty_like(H)
    for i in range(len(renders)):
        u = H[torch.from_numpy(grp != grp[i])].mean(0)
        u = u / u.norm()
        MR[i] = H[i] - (H[i] @ u) * u
    pcs = {"whole": H, "J": Pj(H, QJ), "rest_J": H - Pj(H, QJ), "PCA": Hp, "rest_PCA": H - Hp, "LL": Pj(H, QL),
           "rest_LL": H - Pj(H, QL), "MR": MR}
    for r in range(10):
        pcs[f"R{r}"] = Pj(H, Rb[r])
        if r < 3:
            pcs[f"rest_R{r}"] = H - pcs[f"R{r}"]
    np.savez(d / "pieces_L56.npz", **{nm: v.float().numpy() for nm, v in pcs.items()})
    C.dump(d / "material.json", {"pairs": pairs, "items": items, "PCA_k": k, "J_median_delta_share": jds,
                                 "renders": [{kk: rr[kk] for kk in ("role", "group", "item")} for rr in renders]})
    pc_ids = [P5.render_tell(tok, f"the number {dd}") for dd in DIGITS]
    pc = P5.capture_last(model, pc_ids, [L56])[:, 0]
    W32 = model.get_output_embeddings().weight.detach().float()
    RD.seq_logprob_from_hidden = RS.seq_logprob_fast
    ol = RD.OracleLens(model, tok)
    st = {}
    try:
        prefix = RS.PREFIX0["oracle"]
        base = tok.encode(prefix, add_special_tokens=False)
        seqs, owner = [], []
        for dd in DIGITS:
            for form in (" " + dd, dd):
                full = tok.encode(prefix + form, add_special_tokens=False)
                assert full[:len(base)] == base
                if full[len(base):] not in seqs:
                    seqs.append(full[len(base):])
                    owner.append(dd)
        score = lambda hs, check=False, apb=16: RS.oracle_fast(ol, hs, L56, base, seqs, W32, acts_per_batch=apb, check=check)
        four = [torch.from_numpy(pcs["whole"][i].float().numpy()) for i in range(4)]
        single = [i for i, s in enumerate(seqs) if len(s) == 1]
        a_ = score(four)[:, single].numpy()
        b_ = ol.score_next(four, L56, base, [seqs[i][0] for i in single], W32)[0].cpu().numpy()
        fo = ol.score_seq(four, L56, base, seqs, W32, acts_per_batch=2, check=False).numpy()
        g1 = RS.floor_ok(a_, b_)
        gf = RS.floor_ok(score(four).numpy(), fo)
        score(four[:2], True, 2)
        st["gates"] = {"G_score1": g1, "G_fastpath": gf, "G_score2": getattr(ol, "last_check", None),
                       "pass": bool(g1[2] and gf[2] and getattr(ol, "last_check", 9) <= 1.0)}
        log("gates", st["gates"])
        if st["gates"]["pass"]:
            arrs = {nm: score([v[i].float() for i in range(v.shape[0])]).numpy() for nm, v in pcs.items()}
            arrs["none"] = score([None]).numpy()
            np.savez(d / "ll" / "oracle_L56.npz", **arrs, owner=np.array(owner))
            np.savez(d / "ll" / "oracle_L56_positive.npz", whole=score([pc[i] for i in range(10)]).numpy(), owner=np.array(owner))
            log("Oracle L56 scored")
        else:
            log("STOP GATE: Stage 5 gates failed")
    finally:
        ol.unload()
        del ol
        gc.collect()
    C.dump(d / "state.json", st)


if __name__ == "__main__":
    main()
