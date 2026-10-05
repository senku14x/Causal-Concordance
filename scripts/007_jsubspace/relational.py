"""007 Stage 4 — relational multihop (design §9 + A1; secondary; exploratory; whole bank).

  python relational.py profile --run-id q1     # 4a: gate (stock) + alignment + SEC whole-cell patch profile at 5 shared
                                               #     positions x L40..60 step 2; noise floor; position selection
  python relational.py parta --run-id q1       # 4b: Part A at the selected higher- and lower-leverage positions

4a is reader-free. 4b is reader-side (Oracle L40/L44, NLA L42; 21 v1 pieces; pooled relation-word candidates).
Wording (A1): "higher / lower observed patch leverage", never "used / unused".
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parent / "005_planted_concepts"))
import margin as MG                               # noqa: E402
import partners as PT                             # noqa: E402
import registry as R                              # noqa: E402
import wsb                                        # noqa: E402
import p007 as C                                  # noqa: E402

BANK = Path("/workspace/sources/repos/workspace-bench/evals/relational_multihop/items.json")
PROFILE_LAYERS = list(range(40, 61, 2))
POSNAMES = ("final_s", "possessor", "is", "cloze_subject", "period_s2")


def positions(r):
    """The 5 shared positions (design §9.2) from the rendered tokens: the final 's, the possessor name's last token, the
    last ' is', the cloze subject's last token, and the '.' that ends sentence 2."""
    dec = r.decoded
    T = len(dec)
    fin = T - 1
    assert dec[fin].strip() in ("'s", "’s"), (r.item, dec[fin])
    is_pos = max(i for i in range(T - 1) if dec[i] == " is")
    in_pos = max(i for i in range(T) if dec[i].strip() == "In")
    per = in_pos - 1
    assert dec[per].strip().endswith("."), (r.item, dec[per])
    return {"final_s": fin, "possessor": fin - 1, "is": is_pos, "cloze_subject": is_pos - 1, "period_s2": per}


def material(tok):
    items = json.loads(BANK.read_text())
    by_id = {it["id"]: it for it in items}
    out = []
    for pr in PT.relational_pairs(items):
        al = PT.token_align(tok, pr, by_id[pr.item])
        out.append({"pair": pr, "aligned": al is not None, "r_m": al[0] if al else None, "r_p": al[1] if al else None,
                    "diff": al[2] if al else None, "a": by_id[pr.m.inp] if False else by_id[pr.item],
                    "b": by_id[pr.item[:-1] + "b"]})
    return out


def boot_median_ci(x, seed=C.BOOT_SEED, B=10000):
    x = np.asarray(x, float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(B, len(x)))
    m = np.median(x[idx], axis=1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def profile(run_id):
    import regime as RG
    d = C.OUT / "stage4" / run_id
    d.mkdir(parents=True, exist_ok=True)
    log = C.Logger(d / "log.txt")
    tok = R.make_tokenizer()
    mats = material(tok)
    model = R.load_model()
    # -------- gate (both members; greedy + >= 9/10 at T 0.8, top-p 0.95; partners.GATE) on aligned scenes
    gate = {}
    fg = d / "gate.jsonl"
    if fg.exists():
        for l in open(fg):
            g = json.loads(l)
            gate[g["pair_id"]] = g
    with open(fg, "a") as f:
        for n, e in enumerate(mats):
            pid = e["pair"].pair_id
            if pid in gate or not e["aligned"]:
                continue
            g = PT.gate_pair(model, tok, e["pair"], e["r_m"], e["r_p"], seed=20261002 + n)
            row = {"pair_id": pid, "pass": g["pass"], "m": {k: g["m"][k] for k in ("greedy", "greedy_ok", "n_correct")},
                   "p": {k: g["p"][k] for k in ("greedy", "greedy_ok", "n_correct")}}
            f.write(json.dumps(row) + "\n")
            gate[pid] = row
    keep = [e for e in mats if e["aligned"] and gate.get(e["pair"].pair_id, {}).get("pass")]
    summ = {"scenes": len(mats), "aligned": sum(e["aligned"] for e in mats), "gated_aligned": len(keep)}
    log("relational material", summ)
    C.dump(d / "material.json", {**summ, "kept": [e["pair"].pair_id for e in keep]})
    if len(keep) < 10:
        log("STOP: fewer than 10 scenes after gate + alignment; Stage 4 stops (design §9.1)")
        C.dump(C.RES / "analysis" / "stage4a.json", {**summ, "stopped": "fewer than 10 scenes"})
        return
    # -------- SEC profile: whole-cell patch partner -> member at the 5 positions x L40..60 step 2; reverse on 30%
    rng = np.random.default_rng(C.SEED)
    rev = set(rng.choice(len(keep), int(round(0.3 * len(keep))), replace=False).tolist())
    rows, noise = [], []
    fr = d / "profile.jsonl"
    done = set()
    if fr.exists():
        for l in open(fr):
            r = json.loads(l)
            done.add((r["pair_id"], r["direction"]))
            rows.append(r)
    with RG.SecEngine(model, l0=39) as sec, open(fr, "a") as fo:
        for n, e in enumerate(keep):
            for reverse in ((False, True) if n in rev else (False,)):
                direction = "m->p" if reverse else "p->m"
                if (e["pair"].pair_id, direction) in done:
                    continue
                t0 = time.time()
                rx, ry = (e["r_p"], e["r_m"]) if reverse else (e["r_m"], e["r_p"])
                ax, ay = (e["pair"].p.a, e["pair"].m.a) if reverse else (e["pair"].m.a, e["pair"].p.a)
                Sa, Sb = MG.spellings(tok, "relational", ax), MG.spellings(tok, "relational", ay)
                pos = positions(rx)
                assert pos == positions(ry), "member / partner shared positions differ"
                assert min(pos.values()) > max(e["diff"]), "a shared position precedes a changed relation word"
                for nm, p in pos.items():
                    assert rx.ids[p] == ry.ids[p], (nm, p)
                conts = [()] + [s for s in dict.fromkeys(Sa + Sb) if len(s) > 1]
                src = {c: sec.forward(ry.ids + list(c), record=PROFILE_LAYERS)[1] for c in conts}
                clean = sec.score(rx.ids, Sa, Sb)
                donor = sec.score(ry.ids, Sa, Sb)
                G = donor["m"] - clean["m"]
                for nm, p in pos.items():
                    for L in PROFILE_LAYERS:
                        ed = sec.score(rx.ids, Sa, Sb, edit_fn=lambda c, L=L, p=p: ({L: RG.patch_fn([p], src[c][L][[p]])}, []))
                        dm = RG.sec_damage(sec, clean, ed, p, len(rx))
                        r = {"pair_id": e["pair"].pair_id, "direction": direction, "pos_name": nm, "pos": p, "layer": L,
                             "G": G, "dm": ed["m"] - clean["m"], "S": (ed["m"] - clean["m"]) / G,
                             "off_pair_mass_clean": dm["off_pair_mass_clean"], "off_pair_mass_edit": dm["off_pair_mass_edit"]}
                        rows.append(r)
                        fo.write(json.dumps(r) + "\n")
                fo.flush()
                sec.clear()
                log(f"[{n + 1}/{len(keep)}] {e['pair'].pair_id} {direction} G {G:.2f} ({time.time() - t0:.0f}s)")
            if n < 4:                                                      # noise floor (001 style) on the first 4 scenes
                rx = e["r_m"]
                Sa, Sb = MG.spellings(tok, "relational", e["pair"].m.a), MG.spellings(tok, "relational", e["pair"].p.a)
                conts = [()] + [s for s in dict.fromkeys(Sa + Sb) if len(s) > 1]
                own = {c: sec.forward(rx.ids + list(c), record=[42])[1] for c in conts}
                clean = sec.score(rx.ids, Sa, Sb)
                G = sec.score(e["r_p"].ids, Sa, Sb)["m"] - clean["m"]
                p = len(rx) - 1
                for k in range(8):
                    v = torch.randn(5120, generator=torch.Generator().manual_seed(k))
                    v = v / v.norm()

                    def ef(c, v=v):
                        h = own[c][42][p]
                        return {42: RG.patch_fn([p], (h + 1e-4 * h.norm() * v.to(h.device))[None])}, []
                    noise.append(abs(sec.score(rx.ids, Sa, Sb, edit_fn=ef)["m"] - clean["m"]) / abs(G))
                sec.clear()
    # -------- summary and the pre-registered position selection (design §9.2)
    fwd = [r for r in rows if r["direction"] == "p->m"]
    summ_pl = {}
    for nm in POSNAMES:
        for L in PROFILE_LAYERS:
            S = [r["S"] for r in fwd if r["pos_name"] == nm and r["layer"] == L]
            dmg = [(r["off_pair_mass_edit"] - r["off_pair_mass_clean"]) > 0.10 for r in fwd if r["pos_name"] == nm and r["layer"] == L]
            summ_pl[f"{nm}|L{L}"] = {"median_S": float(np.median(S)), "ci": boot_median_ci(S), "n": len(S),
                                     "damage_share": float(np.mean(dmg)), "damage_dominated": bool(np.mean(dmg) > 0.20)}
    hi = [nm for nm in POSNAMES if summ_pl[f"{nm}|L42"]["median_S"] >= 0.25 and summ_pl[f"{nm}|L40"]["median_S"] >= 0.15
          and summ_pl[f"{nm}|L44"]["median_S"] >= 0.15 and not any(summ_pl[f"{nm}|L{L}"]["damage_dominated"] for L in (40, 42, 44))]
    lo = [nm for nm in POSNAMES if all(summ_pl[f"{nm}|L{L}"]["median_S"] <= 0.05 and summ_pl[f"{nm}|L{L}"]["ci"][1] <= 0.10
                                       for L in (40, 42, 44))]
    sel = {"higher": None, "lower": None}
    if hi and lo:
        sel["higher"] = max(hi, key=lambda nm: summ_pl[f"{nm}|L42"]["median_S"])
        p_hi = positions(keep[0]["r_m"])
        sel["lower"] = min(lo, key=lambda nm: abs(p_hi[nm] - p_hi[sel["higher"]]))
    rev_rows = {(r["pair_id"], r["pos_name"], r["layer"]): r["S"] for r in rows if r["direction"] == "m->p"}
    pairs_fr = [(r["S"], rev_rows[(r["pair_id"], r["pos_name"], r["layer"])]) for r in fwd
                if (r["pair_id"], r["pos_name"], r["layer"]) in rev_rows]
    out = {**summ, "profile": summ_pl, "higher_candidates": hi, "lower_candidates": lo, "selected": sel,
           "noise_floor_rel_median": float(np.median(noise)) if noise else None,
           "noise_floor_rel_max": float(np.max(noise)) if noise else None,
           "reverse_agreement_r": float(np.corrcoef(*zip(*pairs_fr))[0, 1]) if len(pairs_fr) > 2 else None,
           "stop": not (sel["higher"] and sel["lower"])}
    C.dump(C.RES / "analysis" / "stage4a.json", out)
    log("4a selection", sel, "stop" if out["stop"] else "")


def parta(run_id):
    """4b: Part A at the two selected positions (design §9.3)."""
    import p005 as P5
    import readers as RD
    import reader_split as RS
    from pieces import f32_orth, rand_basis_f32
    sel = json.loads((C.RES / "analysis" / "stage4a.json").read_text())
    d = C.OUT / "stage4" / run_id
    log = C.Logger(d / "log_4b.txt")
    if sel.get("stop", True):
        log("4b not run: 4a did not yield both a higher- and a lower-leverage position")
        return
    budget = C.OUT / "budget_drop.json"
    if budget.exists() and "4b" in json.loads(budget.read_text()).get("dropped", []):
        log("4b dropped by the budget rule (design §13.3)")
        return
    tok = R.make_tokenizer()
    mats = material(tok)
    kept = set(json.loads((d / "material.json").read_text())["kept"])
    keep = [e for e in mats if e["pair"].pair_id in kept]
    words = sorted({w for e in keep for w in (e["pair"].m.z[0], e["pair"].m.z[1])})
    model = R.load_model()
    blocks = R.blocks(model)
    dev = next(model.parameters()).device
    POS = {k: sel["selected"][k] for k in ("higher", "lower")}
    renders = [(e, "a", e["r_m"]) for e in keep] + [(e, "b", e["r_p"]) for e in keep]
    X = {k: np.zeros((len(renders), len(C.LAYERS), C.D), np.float32) for k in POS}
    with torch.no_grad():
        for n, (e, ab, r) in enumerate(renders):
            pos = positions(r)
            for k, nm in POS.items():
                rec = P5.RowRecorder(blocks, list(C.LAYERS), [pos[nm]])
                with rec:
                    P5.text_model(model)(input_ids=torch.tensor([r.ids], device=dev), use_cache=False)
                for li, L in enumerate(C.LAYERS):
                    X[k][n, li] = rec.acts[L][0, 0].cpu().numpy()
    scene_of = np.array([keep.index(e) for e, _, _ in renders])
    # pieces: fixed projectors, within-scene PCA (cross-fitted over 4 scene folds, delta-matched k), leave-one-scene-out mean
    B = torch.load(C.STAGE0 / "bases_f32.pt")
    Rb = {r: rand_basis_f32(C.rand_seed("r1024", r), C.K)[0].double() for r in range(10)}
    folds = np.random.default_rng(C.SEED).permutation(len(keep)) % 4
    Pj = lambda A, Q: (A @ Q) @ Q.T
    pieces = {}
    meta = {}
    for k in POS:
        pieces[k] = {}
        for li, L in enumerate(C.LAYERS):
            H = torch.from_numpy(X[k][:, li].astype(np.float64))
            QJ, QL = B[f"J_L{L}"].double(), B["LL"].double()
            n = H.shape[0]
            na = len(keep)
            Dl = H[:na] - H[na:]
            jds = float(np.median(((Pj(Dl, QJ) ** 2).sum(1) / (Dl ** 2).sum(1)).numpy()))
            Vf = {}
            for f in range(4):
                tr = [s for s in range(na) if folds[s] != f]
                Cm = torch.cat([(H[tr] - H[[s + na for s in tr]]) / 2.0, (H[[s + na for s in tr]] - H[tr]) / 2.0]) / np.sqrt(2)
                _, S_, Vh = torch.linalg.svd(Cm, full_matrices=False)
                Vf[f] = Vh[:int((S_ > S_[0] * 1e-8).sum())].T
            kmax = min(V.shape[1] for V in Vf.values())
            best = None
            for kk in range(1, min(kmax, 64) + 1):
                ds = [float(((Vf[folds[s]][:, :kk].T @ Dl[s]) ** 2).sum() / (Dl[s] ** 2).sum()) for s in range(na)]
                gap = abs(float(np.median(ds)) - jds)
                if best is None or gap < best[1]:
                    best = (kk, gap)
            kk = best[0]
            Hp = torch.stack([Pj(H[i:i + 1], f32_orth(Vf[folds[scene_of[i]]][:, :kk].float())[0].double())[0] for i in range(n)])
            MR = torch.empty_like(H)
            for i in range(n):
                m = torch.from_numpy(scene_of != scene_of[i])
                u = H[m].mean(0)
                u = u / u.norm()
                MR[i] = H[i] - (H[i] @ u) * u
            pc_ = {"whole": H, "J": Pj(H, QJ), "rest_J": H - Pj(H, QJ), "PCA": Hp, "rest_PCA": H - Hp, "LL": Pj(H, QL),
                   "rest_LL": H - Pj(H, QL), "MR": MR}
            for r in range(10):
                pc_[f"R{r}"] = Pj(H, Rb[r])
                if r < 3:
                    pc_[f"rest_R{r}"] = H - pc_[f"R{r}"]
            pieces[k][L] = {nm: v.float() for nm, v in pc_.items()}
            meta[f"{k}|L{L}"] = {"PCA_k": kk, "J_median_delta_share": jds}
            np.savez(d / f"pieces_{k}_L{L}.npz", **{nm: v.numpy() for nm, v in pieces[k][L].items()})
    C.dump(d / "parta_material.json", {"positions": POS, "words": words, "renders": [(e["pair"].pair_id, ab) for e, ab, _ in renders],
                                       "meta": meta})
    # positive control activations: "Tell me about {relation word}"
    pc_acts = P5.capture_last(model, [P5.render_tell(tok, w) for w in words], list(C.LAYERS))
    W32 = model.get_output_embeddings().weight.detach().float()
    RD.seq_logprob_from_hidden = RS.seq_logprob_fast
    disabled = json.loads(RS.DISABLED.read_text()) if RS.DISABLED.exists() else {}

    def run_reader(name, L, score, prefix, rtok):
        base = rtok.encode(prefix, add_special_tokens=False)
        seqs, owner = [], []
        for w in words:
            full = rtok.encode(prefix + " " + w, add_special_tokens=False)
            assert full[:len(base)] == base
            seqs.append(full[len(base):])
            owner.append(w)
        for k in POS:
            f = d / f"ll_{name}_L{L}_{k}.npz"
            if f.exists():
                continue
            arrs = {nm: score([v[i] for i in range(v.shape[0])], base, seqs).numpy() for nm, v in pieces[k][L].items()}
            arrs["none"] = score([None], base, seqs).numpy()
            np.savez(f, **arrs, owner=np.array(owner))
        fpc = d / f"ll_{name}_L{L}_positive.npz"
        if not fpc.exists():
            h = pc_acts[:, C.LAYERS.index(L)]
            np.savez(fpc, whole=score([h[i] for i in range(h.shape[0])], base, seqs).numpy(), owner=np.array(owner))
        log(f"4b {name} L{L} scored")
    if "oracle" not in disabled:
        ol = RD.OracleLens(model, tok)
        try:
            for L in (40, 44):
                run_reader("oracle", L, lambda hs, b, s, L=L: RS.oracle_fast(ol, hs, L, b, s, W32, acts_per_batch=16),
                           RS.PREFIX0["oracle"], tok)
        finally:
            ol.unload()
            del ol
            gc.collect()
    del model, W32
    gc.collect()
    torch.cuda.empty_cache()
    if "nla" not in disabled:
        nla = RD.NLArl()
        try:
            run_reader("nla", 42, lambda hs, b, s: RS.nla_fast(nla, hs, b, s, acts_per_batch=16), RS.PREFIX0["nla"], nla.tok)
        finally:
            nla.close()
    log("4b done (the shared-path gates for these readers were established in Stage 1a; see disabled_readers.json)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["profile", "parta"])
    ap.add_argument("--run-id", default="q1")
    a = ap.parse_args()
    d = C.OUT / "stage4" / a.run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / f"manifest_{a.what}.json").write_text(json.dumps({"run_id": a.run_id, "manifest": R.manifest_block(),
                                                          "design": "docs/design_specs/007_jsubspace.md (v2 + A1)"}, indent=1))
    {"profile": profile, "parta": parta}[a.what](a.run_id)


if __name__ == "__main__":
    main()
