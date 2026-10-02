"""Design 003 — reader-free causal labels at the brew answer cells (docs/design_specs/003_labels.md).

    python scripts/003_labels/labels.py capture      # SEC activations at the start-colour token (GPU, ~2 min)
    python scripts/003_labels/labels.py directions   # least-squares content directions per fold / fit set + G3 (CPU)
    python scripts/003_labels/labels.py smoke        # G1/G2 on 3 pairs + timing (GPU)
    python scripts/003_labels/labels.py run          # all gated dev brew pairs (GPU, resumable)
    python scripts/003_labels/labels.py analysis     # labels, reliability, necessity, attribution (CPU)

Label code: imports no reader code (readers.py, jlens, peft); directions come from activations and task labels only.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "001_leverage_profile"))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import leverage_profile as LP  # noqa: E402  materials, gating, render keys, helpers (label code only)
import registry as R  # noqa: E402

EXP = "003_labels"
RES = R.out_dir("results", EXP)
ANA = R.out_dir("results", EXP, "analysis")
OUT = R.out_dir("outputs", EXP)
CAP001 = R.PROJECT / "outputs" / "001_leverage_profile" / "run"
COLOURS = list(LP.COLOURS)
ROLES = ("start", "z", "answer")
LAYERS = [40, 42, 44, 56, 60]                 # layers available in the 001 captures
CELLS = ["emit_asst", "emit_think", "emit_colon", "start_tok"]
GRID = [(c, L) for c in ("emit_asst", "emit_think", "emit_colon") for L in (40, 42, 44)] + [("emit_colon", 60), ("start_tok", 40)]
NEC_GRID = [(c, L) for c in ("emit_asst", "emit_think", "emit_colon") for L in (40, 42, 44)]
N_RAND = 8
TAU, EPS = 0.10, 0.05
SEED = 20261006
MIN_OCC = 2                                   # a colour must take the role at least this often in the fit set


def jl(p):
    return [json.loads(l) for l in open(p)]


def write_manifest(stage, extra):
    m = {"stage": stage, "time_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), **R.manifest_block(), **extra}
    (RES / f"manifest_{stage}.json").write_text(json.dumps(m, indent=1, default=str))


# ------------------------------------------------------------------------------------------- material
def natural_meta():
    meta = json.loads((CAP001 / "captures_brew_meta.json").read_text())
    nat = [i for i, r in enumerate(meta["renders"]) if r["role"] in ("member", "partner")]
    return meta, nat


def folds():
    """3 folds of the 48 dev tables, seeded, balanced on render count (greedy on a seeded order)."""
    meta, nat = natural_meta()
    cnt = {}
    for i in nat:
        cnt[meta["renders"][i]["item"]] = cnt.get(meta["renders"][i]["item"], 0) + 1
    order = list(np.random.default_rng(SEED).permutation(sorted(cnt)))
    order.sort(key=lambda t: -cnt[t])
    load, out = [0, 0, 0], {}
    for t in order:
        k = int(np.argmin(load))
        out[str(t)] = k
        load[k] += cnt[t]
    return out


def start_pos(r):
    return sorted(p for p, g in r.regions.items() if g == "start")[4]       # the start-colour token (" blue" etc.)


def cell_pos(r, cell):
    if cell == "start_tok":
        return start_pos(r)
    return [p for p, g in r.regions.items() if g == cell][0]


# --------------------------------------------------------------------------------------------- capture
def stage_capture():
    """SEC activations at the start-colour token, LAYERS, for the 231 natural dev renders (001 capture order)."""
    import regime as RG
    import torch
    meta, nat = natural_meta()
    tok = R.make_tokenizer()
    mats = LP.materials(tok)
    by_key = {}
    for e in mats["brew"]:
        if e["split"] == "dev":
            by_key[LP.ids_key(e["r_m"].ids)] = e["r_m"]
            by_key[LP.ids_key(e["r_p"].ids)] = e["r_p"]
    model = R.load_model()
    X = np.zeros((len(nat), len(LAYERS), 5120), dtype=np.float32)
    keys = []
    with RG.SecEngine(model, l0=39) as sec:
        for n, i in enumerate(nat):
            m = meta["renders"][i]
            r = by_key[m["key"]]
            _, acts = sec.forward(r.ids, record=LAYERS)
            p = start_pos(r)
            for li, L in enumerate(LAYERS):
                X[n, li] = acts[L][p].float().cpu().numpy()
            keys.append(m["key"])
            sec.clear()
    d = OUT / "captures"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "start_token.npz", X=X, keys=np.array(keys))
    write_manifest("capture", {"n": len(keys)})
    print("captured", X.shape)


def load_acts():
    """{cell: [n_nat, n_layers, d]} aligned with natural_meta() order, plus the render metadata."""
    meta, nat = natural_meta()
    X = np.load(CAP001 / "captures_brew.npz")["X"]
    acts = {c: X[nat, meta["cells"].index(c)] for c in ("emit_asst", "emit_think", "emit_colon")}
    st = np.load(OUT / "captures" / "start_token.npz")
    assert list(st["keys"]) == [meta["renders"][i]["key"] for i in nat]
    acts["start_tok"] = st["X"]
    return acts, [meta["renders"][i] for i in nat]


# ------------------------------------------------------------------------------------------ directions
def fit_coefs(H, rows):
    """Least squares H = b0 + sum_role onehot(colour) B_role (reference coding: the first colour seen per role is the
    reference, coefficient 0). Returns ({role: {colour: vector}}, {role: {colour: count}}, rank, n_params)."""
    cols = []
    ref = {}
    for role in ROLES:
        seen = sorted({r[role] for r in rows})
        ref[role] = seen[0]
        cols += [(role, c) for c in seen[1:]]
    D = np.zeros((len(rows), 1 + len(cols)))
    D[:, 0] = 1
    for j, (role, c) in enumerate(cols, 1):
        D[:, j] = [r[role] == c for r in rows]
    B, *_ = np.linalg.lstsq(D, H.astype(np.float64), rcond=None)
    rank = int(np.linalg.matrix_rank(D))
    coef = {role: {ref[role]: np.zeros(H.shape[1])} for role in ROLES}
    for j, (role, c) in enumerate(cols, 1):
        coef[role][c] = B[j]
    counts = {role: {c: sum(r[role] == c for r in rows) for c in COLOURS} for role in ROLES}
    return coef, counts, rank, D.shape[1]


def fit_sets():
    """Per evaluation fold f: 'both' = the other two folds; 'r1', 'r2' = each of them alone (reliability)."""
    fo = folds()
    out = {}
    for f in range(3):
        others = [g for g in range(3) if g != f]
        out[f] = {"both": set(others), "r1": {others[0]}, "r2": {others[1]}}
    return fo, out


def stage_directions():
    """Fit and save content coefficients (and naive z centroids) per evaluation fold x fit set x cell x layer; G3."""
    acts, rows = load_acts()
    fo, fs = fit_sets()
    fold_of = np.array([fo[r["item"]] for r in rows])
    store, info = {}, {}
    val = []
    for f in range(3):
        for name, use in fs[f].items():
            idx = np.where(np.isin(fold_of, list(use)))[0]
            rr = [rows[i] for i in idx]
            for cell, L in GRID + [("emit_colon", 56)]:
                li = LAYERS.index(L)
                H = acts[cell][idx, li]
                coef, counts, rank, npar = fit_coefs(H, rr)
                key = f"f{f}|{name}|{cell}|L{L}"
                for role in ROLES:
                    for c, v in coef[role].items():
                        store[f"{key}|{role}|{c}"] = v.astype(np.float32)
                for c in COLOURS:
                    sel = [k for k, r in enumerate(rr) if r["z"] == c]
                    if sel:
                        store[f"{key}|naive_z|{c}"] = H[sel].mean(0).astype(np.float32)
                store[f"{key}|fitmean"] = H.mean(0).astype(np.float32)
                info[key] = {"n": len(rr), "rank": rank, "n_params": npar, "counts": counts}
                # G3: held-out separation of z contrasts on the evaluation fold (primary fit set only)
                if name == "both":
                    ev = np.where(fold_of == f)[0]
                    He, re = acts[cell][ev, li], [rows[i] for i in ev]
                    for a in COLOURS:
                        for b in COLOURS:
                            if a >= b or min(counts["z"][a], counts["z"][b]) < MIN_OCC:
                                continue
                            xa = [k for k, r in enumerate(re) if r["z"] == a]
                            xb = [k for k, r in enumerate(re) if r["z"] == b]
                            if not xa or not xb:
                                continue
                            u = coef["z"][b] - coef["z"][a]
                            u = u / np.linalg.norm(u)
                            pa, pb = He[xa] @ u, He[xb] @ u
                            auc = float(np.mean([(y > x) + 0.5 * (y == x) for x in pa for y in pb]))
                            val.append({"fold": f, "cell": cell, "layer": L, "a": a, "b": b, "n_a": len(xa), "n_b": len(xb), "auc": auc})
    np.savez_compressed(OUT / "directions.npz", **store)
    (OUT / "directions_info.json").write_text(json.dumps({"folds": fo, "info": info}, indent=1))
    summ = {}
    for cell, L in GRID + [("emit_colon", 56)]:
        v = [x["auc"] for x in val if x["cell"] == cell and x["layer"] == L]
        summ[f"{cell}|L{L}"] = {"n_contrasts": len(v), "median_auc": float(np.median(v)) if v else None,
                                "frac_ge_0.75": float(np.mean(np.array(v) >= 0.75)) if v else None}
    g3 = summ["emit_colon|L42"]["median_auc"]
    out = {"gate_G3": {"median_auc_colon_L42": g3, "passes": bool(g3 is not None and g3 >= 0.75)}, "by_cell": summ,
           "contrasts": val, "fold_sizes": {k: sum(1 for v in fo.values() if v == k) for k in range(3)}}
    (ANA / "directions_validity.json").write_text(json.dumps(out, indent=1))
    write_manifest("directions", {"gate_G3": out["gate_G3"]})
    print(json.dumps(out["gate_G3"]), {k: v["median_auc"] for k, v in summ.items()})


class Directions:
    def __init__(self):
        with np.load(OUT / "directions.npz") as npz:
            self.z = {k: npz[k] for k in npz.files}          # ~100 MB, loaded once
        self.info = json.loads((OUT / "directions_info.json").read_text())
        self.fold = self.info["folds"]

    def get(self, f, name, cell, L, role, c, c2):
        """unit(B_role[c2] - B_role[c]) or None if either colour is too rare in that role in the fit set."""
        key = f"f{f}|{name}|{cell}|L{L}"
        cnt = self.info["info"][key]["counts"][role if role != "naive_z" else "z"]
        if min(cnt[c], cnt[c2]) < MIN_OCC:
            return None
        a, b = f"{key}|{role}|{c}", f"{key}|{role}|{c2}"
        if a not in self.z or b not in self.z:
            return None
        u = self.z[b].astype(np.float64) - self.z[a].astype(np.float64)
        n = np.linalg.norm(u)
        return (u / n).astype(np.float32) if n > 0 else None

    def fitmean(self, f, cell, L):
        return self.z[f"f{f}|both|{cell}|L{L}|fitmean"]


# ------------------------------------------------------------------------------------------- measurement
class Fidelity:
    def __init__(self):
        self.rho, self.kappa = [], []

    def add(self, requested, realized):
        import torch
        rn = float(requested.norm())
        if rn == 0:
            return
        self.rho.append(float(realized.norm()) / rn)
        self.kappa.append(float(torch.nn.functional.cosine_similarity(requested[None], realized[None])[0]))


def _write(pos, delta_fn, fid):
    """Upper-stack edit at one position: x[pos] += delta_fn(x[pos]); fidelity measured on the realized write."""
    def f(x):
        x = x.clone()
        before = x[pos].clone()
        d = delta_fn(before)
        x[pos] = before + d
        fid.add(d, x[pos] - before)
        return x
    return f


def colour_mass(lp_last, first_ids):
    """First-token colour mass at the answer position: 12 colours + the rest."""
    p = lp_last.exp()
    m = {c: float(p[list(ids)].sum()) for c, ids in first_ids.items()}
    m["other"] = max(0.0, 1.0 - sum(m.values()))
    return m


def measure(sec, tok, e, dirs, fold, first_ids, fold_acts=None, timing=False):
    """All 003 edits for one pair (recipient = member, donor = partner). Returns a list of rows."""
    import margin as MG
    import regime as RG
    import torch
    pair = e["pair"]
    rx, ry = e["r_m"], e["r_p"]
    Sa, Sb = MG.spellings(tok, "brew", pair.m.a), MG.spellings(tok, "brew", pair.p.a)
    T = len(rx)
    conts = [()] + [s for s in dict.fromkeys(Sa + Sb) if len(s) > 1]
    layers = sorted({L for _, L in GRID})
    own = {c: sec.forward(rx.ids + list(c), record=layers)[1] for c in conts}
    src = {c: sec.forward(ry.ids + list(c), record=layers)[1] for c in conts}
    clean = sec.score(rx.ids, Sa, Sb)
    donor = sec.score(ry.ids, Sa, Sb)
    G = donor["m"] - clean["m"]
    vals = {"start": (pair.m.inp, pair.p.inp), "z": (pair.m.z, pair.p.z), "answer": (pair.m.a, pair.p.a)}
    rng = np.random.default_rng(SEED + sum(map(ord, pair.pair_id)))
    rows = []
    base = {"pair_id": pair.pair_id, "item": pair.item, "fold": fold, "G": G, "m_clean": clean["m"], "m_donor": donor["m"]}
    cm_clean = colour_mass(clean["lp_last"], first_ids)

    def run_edit(kind, cell, L, content, u, mode, extra=None):
        """mode 'swap': coordinate along u -> donor's; mode ('dir', r, delta_of): h += delta * r with delta = u.(h_d - h);
        mode ('clamp', u, mu): coordinate along u -> mu (on the run being scored)."""
        pos = cell_pos(rx, cell)
        fid = Fidelity()
        ut = torch.tensor(u, device=sec.model.device) if u is not None else None

        def edit_fn(c):
            h_d = src[c][L][pos]
            if mode == "swap":
                fn = _write(pos, lambda h: (h_d.float() @ ut - h @ ut) * ut, fid)
            elif mode[0] == "dir":
                r = torch.tensor(mode[1], device=sec.model.device)
                delta = float(h_d.float() @ ut - own[c][L][pos].float() @ ut)
                fn = _write(pos, lambda h: delta * r, fid)
            else:
                raise ValueError(mode)
            return {L: fn}, []
        t0 = time.time()
        ed = sec.score(rx.ids, Sa, Sb, edit_fn=edit_fn)
        row = {**base, "kind": kind, "cell": cell, "layer": L, "content": content, "dm": ed["m"] - clean["m"],
               "S": (ed["m"] - clean["m"]) / G, **RG.sec_damage(sec, clean, ed, pos, T),
               "fid_rho": [min(fid.rho), max(fid.rho)] if fid.rho else None, "fid_kappa_min": min(fid.kappa) if fid.kappa else None}
        if kind == "swap":
            row["colour_mass_edit"] = colour_mass(ed["lp_last"], first_ids)
            row["colour_mass_clean"] = cm_clean
        if extra:
            row.update(extra)
        if timing:
            row["seconds"] = time.time() - t0
        rows.append(row)
        return row

    def run_clamp(kind, cell, L, u, mu, which):
        """Necessity: clamp the coordinate along u to mu in the member's ('m') or partner's ('p') run."""
        r_ = rx if which == "m" else ry
        pos = cell_pos(r_, cell)
        fid = Fidelity()
        ut = torch.tensor(u, device=sec.model.device)
        fn = _write(pos, lambda h: (float(mu) - h @ ut) * ut, fid)
        sc = sec.score(r_.ids, Sa, Sb, edit_fn=lambda c: ({L: fn}, []))
        return sc["m"], fid

    for cell, L in GRID:
        for content in ROLES:
            c, c2 = vals[content]
            u = dirs.get(fold, "both", cell, L, content, c, c2)
            if u is None:
                rows.append({**base, "kind": "swap", "cell": cell, "layer": L, "content": content, "status": "no direction"})
                continue
            cosines = {}
            for other in ROLES:
                if other != content:
                    v = dirs.get(fold, "both", cell, L, other, *vals[other])
                    cosines[other] = None if v is None else float(np.dot(u, v))
            run_edit("swap", cell, L, content, u, "swap", {"cos_other_contents": cosines})
            for rel in ("r1", "r2"):
                ur = dirs.get(fold, rel, cell, L, content, c, c2)
                if ur is None:
                    rows.append({**base, "kind": f"swap_{rel}", "cell": cell, "layer": L, "content": content, "status": "no direction"})
                else:
                    run_edit(f"swap_{rel}", cell, L, content, ur, "swap", {"cos_with_primary": float(np.dot(u, ur))})
            for k in range(N_RAND):
                r = rng.standard_normal(5120).astype(np.float32)
                r /= np.linalg.norm(r)
                run_edit("null", cell, L, content, u, ("dir", r), {"rand_index": k})
            if content == "z":
                un = dirs.get(fold, "both", cell, L, "naive_z", c, c2)
                if un is not None:
                    run_edit("swap_naive", cell, L, content, un, "swap", {"cos_with_primary": float(np.dot(u, un))})
    # necessity (z content), member and partner runs both clamped
    for cell, L in NEC_GRID:
        c, c2 = vals["z"]
        u = dirs.get(fold, "both", cell, L, "z", c, c2)
        if u is None:
            rows.append({**base, "kind": "necessity", "cell": cell, "layer": L, "content": "z", "status": "no direction"})
            continue
        fm = dirs.fitmean(fold, cell, L)
        for k, uu in enumerate([u] + [(lambda r: r / np.linalg.norm(r))(rng.standard_normal(5120).astype(np.float32)) for _ in range(N_RAND)]):
            mu = float(np.dot(fm, uu)) if k == 0 else float(fold_acts[(cell, L)] @ uu) if fold_acts is not None else float(np.dot(fm, uu))
            mm, f1 = run_clamp("necessity", cell, L, uu, mu, "m")
            mp, f2_ = run_clamp("necessity", cell, L, uu, mu, "p")
            Gabl = mp - mm
            rows.append({**base, "kind": "necessity" if k == 0 else "necessity_null", "cell": cell, "layer": L, "content": "z",
                         "G_abl": Gabl, "N": 1 - Gabl / G, "mu": mu,
                         "fid_rho": [min(f1.rho + f2_.rho), max(f1.rho + f2_.rho)] if f1.rho else None,
                         "fid_kappa_min": min(f1.kappa + f2_.kappa) if f1.kappa else None})
    sec.clear()
    return rows


def _first_ids(tok):
    import margin as MG
    return {c: sorted({s[0] for s in MG.spellings(tok, "brew", c)}) for c in COLOURS}


def _entries(tok):
    mats = LP.materials(tok)
    return LP.gated_dev(mats, "brew")


def stage_smoke():
    """G2: self-swap bitwise, clean margin equals 001's; G1 fidelity; timing (3 pairs)."""
    import margin as MG
    import regime as RG
    import torch
    tok = R.make_tokenizer()
    entries = _entries(tok)[:3]
    dirs = Directions()
    p001 = {h["pair_id"]: h for h in jl(CAP001 / "pairs_brew.jsonl") if h["direction"] == "p->m"}
    model = R.load_model()
    out = {"pairs": []}
    first = _first_ids(tok)
    with RG.SecEngine(model, l0=39) as sec:
        for e in entries:
            pair = e["pair"]
            rx = e["r_m"]
            Sa, Sb = MG.spellings(tok, "brew", pair.m.a), MG.spellings(tok, "brew", pair.p.a)
            clean = sec.score(rx.ids, Sa, Sb)
            res = {"pair_id": pair.pair_id, "m_clean": clean["m"], "m_clean_001": p001[pair.pair_id]["m_clean"],
                   "m_matches_001": abs(clean["m"] - p001[pair.pair_id]["m_clean"]) < 1e-4}
            # self-swap: coordinate along a real z direction set to its own value -> must be bitwise identical
            fold = dirs.fold[pair.item]
            u = dirs.get(fold, "both", "emit_colon", 42, "z", pair.m.z, pair.p.z)
            pos = cell_pos(rx, "emit_colon")
            ut = torch.tensor(u, device=model.device)

            def selfswap(x):
                return x                     # target = own coordinate: no write at all (the identity edit)
            own = sec.forward(rx.ids, record=[42])[1][42][pos]
            fid = Fidelity()
            fn = _write(pos, lambda h: (own.float() @ ut - h @ ut) * ut, fid)
            ss = sec.score(rx.ids, Sa, Sb, edit_fn=lambda c: ({42: fn}, []))
            res["self_swap_max_abs_dm"] = abs(ss["m"] - clean["m"])
            res["self_swap_bitwise"] = bool(torch.equal(ss["fh"], clean["fh"]))
            t0 = time.time()
            rows = measure(sec, tok, e, dirs, fold, first, timing=True)
            res["seconds_pair"] = time.time() - t0
            res["n_edits"] = sum(1 for r in rows if "S" in r) + 2 * sum(1 for r in rows if r["kind"].startswith("necessity") and "N" in r)
            rho = [x for r in rows if r.get("fid_rho") for x in r["fid_rho"]]
            kap = [r["fid_kappa_min"] for r in rows if r.get("fid_kappa_min") is not None]
            res["fidelity_rho_range"] = [min(rho), max(rho)]
            res["fidelity_kappa_min"] = min(kap)
            res["no_direction"] = sum(1 for r in rows if r.get("status") == "no direction")
            res["example_S"] = {f'{r["kind"]}|{r["cell"]}|L{r["layer"]}|{r["content"]}': round(r["S"], 4) for r in rows
                                if r["kind"] in ("swap", "swap_naive") and "S" in r}
            out["pairs"].append(res)
            print(json.dumps({k: v for k, v in res.items() if k != "example_S"}), flush=True)
    ok = all(p["m_matches_001"] and p["self_swap_max_abs_dm"] < 1e-6 and 0.9 <= p["fidelity_rho_range"][0]
             and p["fidelity_rho_range"][1] <= 1.1 and p["fidelity_kappa_min"] >= 0.99 for p in out["pairs"])
    out["all_pass"] = ok
    (ANA / "smoke_checks.json").write_text(json.dumps(out, indent=1))
    write_manifest("smoke", {"all_pass": ok})
    print("smoke", "PASS" if ok else "FAIL")


def stage_run():
    import regime as RG
    assert json.loads((ANA / "smoke_checks.json").read_text())["all_pass"], "smoke did not pass"
    tok = R.make_tokenizer()
    entries = _entries(tok)
    dirs = Directions()
    first = _first_ids(tok)
    acts, rows_meta = load_acts()
    fo = dirs.fold
    fold_of = np.array([fo[r["item"]] for r in rows_meta])
    # fit-set means for the random-ablation null are taken per fold on the fly: {(fold, cell, L): mean vector}
    fmeans = {}
    for f in range(3):
        idx = np.where(fold_of != f)[0]
        for cell, L in NEC_GRID:
            fmeans[(f, cell, L)] = acts[cell][idx, LAYERS.index(L)].mean(0)
    out = OUT / "run"
    out.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["pair_id"] for l in open(out / "pairs_done.jsonl")} if (out / "pairs_done.jsonl").exists() else set()
    model = R.load_model()
    t0 = time.time()
    with RG.SecEngine(model, l0=39) as sec:
        for i, e in enumerate(entries):
            pid = e["pair"].pair_id
            if pid in done:
                continue
            fold = fo[e["pair"].item]
            fa = {(c, L): fmeans[(fold, c, L)] for c, L in NEC_GRID}
            rows = measure(sec, tok, e, dirs, fold, first, fold_acts=fa)
            with open(out / "edits.jsonl", "a") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows))
            with open(out / "pairs_done.jsonl", "a") as f:
                f.write(json.dumps({"pair_id": pid, "n_rows": len(rows)}) + "\n")
            print(f"[003 run] {i + 1}/{len(entries)} {pid} {time.time() - t0:.0f}s", flush=True)
    write_manifest("run", {"seconds": time.time() - t0, "n_pairs": len(entries)})
    print("run done", time.time() - t0)


# ---------------------------------------------------------------------------------------------- analysis
def _bh(pvals, q=0.05):
    import stats as ST
    return ST.bh_fdr(pvals, q=q)


def _kappa2(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if len(a) == 0:
        return float("nan")
    po = float(np.mean(a == b))
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def _med_ci(vals, tables):
    import stats as ST
    v = np.asarray(vals, float)
    if len(v) == 0:
        return None
    b = ST.cluster_bootstrap(v, np.asarray(tables), stat=np.median)
    return {"median": float(np.median(v)), "boot95_tables": b["ci"], "n": int(len(v)), "n_tables": b["n_clusters"]}


def stage_analysis():
    rows = jl(OUT / "run" / "edits.jsonl")
    pairs = sorted({r["pair_id"] for r in rows})
    # whole-cell reference from 001 (same pairs, same regime)
    prof = [r for r in jl(CAP001 / "profile_brew.jsonl") if r["direction"] == "p->m"]
    cellpos = {"emit_asst": 158, "emit_think": 162, "emit_colon": 164, "start_tok": 142}
    sfull = {(r["pair_id"], r["pos"], r["layer"]): r["S"] for r in prof}
    item_of = {r["pair_id"]: r["item"] for r in rows}
    sw = [r for r in rows if r["kind"] == "swap" and "S" in r]
    nul = [r for r in rows if r["kind"] == "null"]
    pool = {}
    for r in nul:
        pool.setdefault((r["cell"], r["layer"], r["content"]), []).append(r["S"])
    pool = {k: np.array(v) for k, v in pool.items()}

    def pval(r):
        v = pool[(r["cell"], r["layer"], r["content"])]
        return float((1 + (v >= r["S"]).sum()) / (1 + len(v)))

    # primary labels
    lab = {}
    for pid in pairs:
        rs = [r for r in sw if r["pair_id"] == pid]
        ps = [pval(r) for r in rs]
        rej, adj = _bh(ps)
        for r, rj, a in zip(rs, rej, adj):
            k = (pid, r["cell"], r["layer"], r["content"])
            sf = sfull.get((pid, cellpos[r["cell"]], r["layer"]))
            dmg = (r["off_pair_mass_edit"] - r["off_pair_mass_clean"] > 0.10 and
                   r["colour_mass_edit"]["other"] - r["colour_mass_clean"]["other"] > 0.05) or \
                  (r.get("top1_retention") is not None and r["top1_retention"] == r["top1_retention"] and r["top1_retention"] < 0.95)
            fid_ok = r["fid_rho"] is not None and 0.9 <= r["fid_rho"][0] and r["fid_rho"][1] <= 1.1 and r["fid_kappa_min"] >= 0.99
            used = bool(r["S"] >= TAU and rj)
            lab[k] = {"S": r["S"], "p": ps[rs.index(r)], "p_bh": a, "used": used and not dmg and fid_ok, "used_raw": used,
                      "damage": bool(dmg), "fidelity_ok": bool(fid_ok), "S_full": sf,
                      "one_d_gt_full": bool(sf is not None and r["S"] > sf + EPS), "item": item_of[pid],
                      "cos_other": r.get("cos_other_contents")}
    # necessity (z)
    nec = {(r["pair_id"], r["cell"], r["layer"]): r["N"] for r in rows if r["kind"] == "necessity" and "N" in r}
    necnull = {}
    for r in rows:
        if r["kind"] == "necessity_null":
            necnull.setdefault((r["cell"], r["layer"]), []).append(r["N"])
    nq95 = {k: float(np.quantile(v, 0.95)) for k, v in necnull.items()}
    for (pid, cell, L), N in nec.items():
        k = (pid, cell, L, "z")
        if k in lab:
            lab[k]["N"] = N
            lab[k]["necessary"] = bool(N > nq95[(cell, L)])
    # final state
    for k, v in lab.items():
        pid, cell, L, content = k
        if v["used"]:
            v["state"] = "used + necessary" if v.get("necessary") else "used"
        elif content == "z" and cell != "start_tok" and v["S_full"] is not None and abs(v["S_full"]) < EPS:
            v["state"] = "readable, low patch leverage"     # family-level decodability only (audit of 003, §5)
        else:
            v["state"] = "unresolved"
    # reliability (G4): same rule with unadjusted p, two disjoint fits
    rel = {}
    for r in rows:
        if r["kind"] in ("swap_r1", "swap_r2") and "S" in r:
            rel.setdefault((r["pair_id"], r["cell"], r["layer"], r["content"]), {})[r["kind"]] = r
    G4 = {}
    for content in ROLES:
        for cl in [("emit_colon", (42, 44)), ("emit_colon", (40,)), ("emit_asst", (40, 42, 44)), ("emit_think", (40, 42, 44))]:
            cell, Ls = cl
            a, b, s1, s2 = [], [], [], []
            for k, v in rel.items():
                if k[1] == cell and k[2] in Ls and k[3] == content and len(v) == 2:
                    r1, r2 = v["swap_r1"], v["swap_r2"]
                    a.append(r1["S"] >= TAU and pval(r1) < 0.05)
                    b.append(r2["S"] >= TAU and pval(r2) < 0.05)
                    s1.append(r1["S"]); s2.append(r2["S"])
            G4[f"{content}|{cell}|{'/'.join(map(str, Ls))}"] = {"n": len(a), "kappa_used": _kappa2(a, b),
                                                                "agreement": float(np.mean(np.array(a) == np.array(b))) if a else None,
                                                                "frac_used_r1": float(np.mean(a)) if a else None, "frac_used_r2": float(np.mean(b)) if b else None,
                                                                "pearson_S": float(np.corrcoef(s1, s2)[0, 1]) if len(s1) > 2 else None}
    # summaries
    summ = {}
    for cell, L in GRID:
        for content in ROLES:
            ks = [k for k in lab if k[1] == cell and k[2] == L and k[3] == content]
            if not ks:
                continue
            S = [lab[k]["S"] for k in ks]
            summ[f"{cell}|L{L}|{content}"] = {
                **(_med_ci(S, [lab[k]["item"] for k in ks]) or {}),
                "frac_used": float(np.mean([lab[k]["used"] for k in ks])),
                "frac_used_necessary": float(np.mean([lab[k]["state"] == "used + necessary" for k in ks])) if content == "z" and (cell, L) in NEC_GRID else None,
                "frac_not_needed": float(np.mean([lab[k]["state"] == "readable, low patch leverage" for k in ks])),
                "frac_necessity_above_null": float(np.mean([lab[k].get("N", -1e9) > (nq95.get((cell, L)) or 1e9) for k in ks])) if content == "z" and (cell, L) in NEC_GRID else None,
                "frac_S_positive": float(np.mean([lab[k]["S"] > 0 for k in ks])),
                "frac_unresolved": float(np.mean([lab[k]["state"] == "unresolved" for k in ks])),
                "frac_damage": float(np.mean([lab[k]["damage"] for k in ks])),
                "frac_one_d_gt_full": float(np.mean([lab[k]["one_d_gt_full"] for k in ks])),
                "median_S_full": float(np.median([lab[k]["S_full"] for k in ks if lab[k]["S_full"] is not None])),
                "null_q95": float(np.quantile(pool[(cell, L, content)], 0.95)),
                "null_median": float(np.median(pool[(cell, L, content)])),
                "median_N": float(np.median([lab[k]["N"] for k in ks if "N" in lab[k]])) if content == "z" and (cell, L) in NEC_GRID else None,
                "necessity_null_q95": nq95.get((cell, L)) if content == "z" else None}
    naive = {}
    for cell, L in GRID:
        a = [r["S"] for r in rows if r["kind"] == "swap_naive" and r["cell"] == cell and r["layer"] == L and "S" in r]
        cs = [r["cos_with_primary"] for r in rows if r["kind"] == "swap_naive" and r["cell"] == cell and r["layer"] == L and "S" in r]
        if a:
            naive[f"{cell}|L{L}"] = {"median_S_naive": float(np.median(a)), "median_cos_with_decorrelated": float(np.median(cs)), "n": len(a)}
    attrib = {}
    for cell, L in GRID:
        per = []
        for pid in pairs:
            ss = {c: lab.get((pid, cell, L, c), {}).get("S") for c in ROLES}
            sf = sfull.get((pid, cellpos[cell], L))
            if None in ss.values() or sf is None:
                continue
            per.append({"z": ss["z"], "start": ss["start"], "answer": ss["answer"], "full": sf,
                        "full_minus_max_single": sf - max(ss.values()),   # NOT an attribution to other content (audit §6)
                        "largest": max(ss, key=ss.get)})
        if per:
            attrib[f"{cell}|L{L}"] = {"median_S": {k: float(np.median([x[k] for x in per])) for k in ("z", "start", "answer", "full", "full_minus_max_single")},
                                      "largest_content_counts": {c: sum(x["largest"] == c for x in per) for c in ROLES}, "n": len(per),
                                      "median_cos": {f"{a}-{b}": float(np.nanmedian([lab[(pid, cell, L, a)]["cos_other"].get(b) if lab.get((pid, cell, L, a)) and lab[(pid, cell, L, a)]["cos_other"] and lab[(pid, cell, L, a)]["cos_other"].get(b) is not None else np.nan for pid in pairs]))
                                                     for a, b in (("z", "start"), ("z", "answer"), ("start", "answer"))}}
    fid = [x for r in rows if r.get("fid_rho") for x in r["fid_rho"]]
    kap = [r["fid_kappa_min"] for r in rows if r.get("fid_kappa_min") is not None]
    n_edits = sum(1 for r in rows if r.get("fid_rho"))
    n_fail = sum(1 for r in rows if r.get("fid_rho") and not (0.9 <= r["fid_rho"][0] and r["fid_rho"][1] <= 1.1 and r["fid_kappa_min"] >= 0.99))
    gates = {"G1_fidelity": {"n_edit_rows": n_edits, "n_fail": n_fail, "rho_range": [min(fid), max(fid)], "kappa_min": min(kap), "passes": n_fail <= 0.01 * n_edits},
             "G3": json.loads((ANA / "directions_validity.json").read_text())["gate_G3"],
             "G2": json.loads((ANA / "smoke_checks.json").read_text())["all_pass"],
             "G4_label_reliability": G4,
             "G4_passes_z_colon_L42_44": bool((G4.get("z|emit_colon|42/44") or {}).get("kappa_used", 0) >= 0.7)}
    g = lambda k: summ.get(k, {})  # noqa: E731
    pred = {
        "L1 z swap at ':' L42/L44 used in >= 50% of pairs": [g("emit_colon|L42|z").get("frac_used"), g("emit_colon|L44|z").get("frac_used")],
        "L2 ':' L40-44 median S_z > median S_start": [(g(f"emit_colon|L{L}|z").get("median"), g(f"emit_colon|L{L}|start").get("median")) for L in (40, 42, 44)],
        "L3 ':' L42-44 median S_z > median S_answer": [(g(f"emit_colon|L{L}|z").get("median"), g(f"emit_colon|L{L}|answer").get("median")) for L in (42, 44)],
        "L4 z necessity at ':' L40-44 above null q95 in >= 30%": [g(f"emit_colon|L{L}|z").get("frac_necessity_above_null") for L in (40, 42, 44)],
        "L5 G4 kappa >= 0.7 for z at ':' L42-44": (G4.get("z|emit_colon|42/44") or {}).get("kappa_used"),
        "L6 start swap at the start token L40 used in >= 70%": g("start_tok|L40|start").get("frac_used"),
        "L7 ':' median S_z at L60 > at L44": [g("emit_colon|L60|z").get("median"), g("emit_colon|L44|z").get("median")],
        "L8 naive z gives a larger median S than decorrelated at ':' L42-44": [((naive.get(f"emit_colon|L{L}") or {}).get("median_S_naive"), g(f"emit_colon|L{L}|z").get("median")) for L in (42, 44)],
    }
    # qualitative: per (':' L44) the 4 pairs with the largest z-swap S, with colour-mass shifts
    qual = []
    zr = sorted([r for r in sw if r["cell"] == "emit_colon" and r["layer"] == 44 and r["content"] == "z"], key=lambda r: -r["S"])
    for r in zr[:3] + zr[len(zr) // 2: len(zr) // 2 + 1] + zr[-2:]:
        cm0, cm1 = r["colour_mass_clean"], r["colour_mass_edit"]
        top = lambda cm: sorted([(c, round(v, 3)) for c, v in cm.items()], key=lambda x: -x[1])[:4]  # noqa: E731
        qual.append({"pair_id": r["pair_id"], "S_z": r["S"], "S_full": lab[(r["pair_id"], "emit_colon", 44, "z")]["S_full"],
                     "S_start": lab.get((r["pair_id"], "emit_colon", 44, "start"), {}).get("S"),
                     "S_answer": lab.get((r["pair_id"], "emit_colon", 44, "answer"), {}).get("S"),
                     "colour_mass_clean_top4": top(cm0), "colour_mass_edit_top4": top(cm1),
                     "state": lab[(r["pair_id"], "emit_colon", 44, "z")]["state"]})
    out = {"gates": gates, "summary": summ, "naive_vs_decorrelated": naive, "attribution": attrib, "predictions": pred,
           "n_pairs": len(pairs), "n_tables": len({item_of[p] for p in pairs}), "qualitative": qual}
    (ANA / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    (ANA / "labels.json").write_text(json.dumps([{"pair_id": k[0], "cell": k[1], "layer": k[2], "content": k[3], **v} for k, v in lab.items()], default=float))
    write_manifest("analysis", {"n_pairs": len(pairs)})
    print(json.dumps(gates, indent=1, default=float)[:1500])


if __name__ == "__main__":
    {"capture": stage_capture, "directions": stage_directions, "smoke": stage_smoke, "run": stage_run,
     "analysis": stage_analysis}[sys.argv[1]]()
