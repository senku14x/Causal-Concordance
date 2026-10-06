"""008 Track B (design docs/design_specs/008_009_trackb_brewcode.md, Part 1; execution notes in the design's §E):
the paper's §3.3 protocol on brew dev — per-layer coordinate clamps over the emit cells x L40-48, the true J-lens
coordinate swap and class-mean z-direction swaps (J part / non-J part), raw and whitened. Experiments about J geometry;
no labels; never used to score readers.

  python trackb.py --run-id b1 [--limit-pairs N]

One model session:
  Phase 1  SEC captures of the 231 dev renders at the 3 emit cells, L40-48          -> OUT/captures.pt
  Phase 2  class-mean z directions per (cell, layer), leaving each pair's table out; J split by GPU pursuit (k = 25,
           NNLS refit) over folded J-lens atoms (raw) and over unit(Sigma^1/2 v) (whitened), J_NP and J_CB
                                                                                       -> OUT/directions.pt
  Phase 3  every condition on every gated dev pair (82); gates G0 / G1 / G-clamp / G-RA / G-repro
                                                                                       -> OUT/<run>/pairs/<pid>.jsonl
The clamp primitive: at each band layer and site position, c(h) = R (h - mu); h <- h + A (t - c(h)), R A = I; the target
t comes from the CLEAN run's coordinates at that layer and position (swap: t = c + alpha (sigma(c) - c)).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parent / "004_span_assay"))
import brew_material as BM                        # noqa: E402
import c008 as C                                  # noqa: E402
import jspace                                     # noqa: E402  (reader-side geometry; this is not label code)
import margin as MG                               # noqa: E402
import regime as RG                               # noqa: E402
import registry as R                              # noqa: E402
import span_assay as SA                           # noqa: E402

LENSES = ("J_NP", "J_CB")
REPRO_TOL = 0.01
RA_TOL = 1e-5
CLAMP_TOL = 1e-3


# ------------------------------------------------------------------------------------------------ material
def material(tok):
    entries = BM.brew_pairs(tok)
    dev = [e for e in entries if e["split"] == "dev"]
    rend = {}
    for e in dev:
        for role, r, mem in (("member", e["r_m"], e["pair"].m), ("partner", e["r_p"], e["pair"].p)):
            rend.setdefault(BM.ids_key(r.ids), {"r": r, "item": e["item"]["name"], "z": mem.z, "role": role})
    keys = sorted(rend)
    assert len(keys) == 231, len(keys)
    gated = BM.gated_dev_brew(entries)
    assert len(gated) == 82, len(gated)
    tables = sorted({rend[k]["item"] for k in keys})
    assert len(tables) == 48
    return rend, keys, tables, gated


def p001():
    """001's whole-cell S at ':' L42 (p->m) per pair: the G-repro reference."""
    out = {}
    for l in open(R.PROJECT / "outputs" / "001_leverage_profile" / "run" / "profile_brew.jsonl"):
        r = json.loads(l)
        if r["direction"] == "p->m" and r["region"] == "emit_colon" and r["layer"] == 42:
            out[r["pair_id"]] = r["S"]
    return out


# ------------------------------------------------------------------------------------------------ phase 1
def phase1(sec, rend, keys, log):
    f = C.OUT / "captures.pt"
    if f.exists():
        return torch.load(f)
    H = {cell: {L: torch.zeros(len(keys), 5120) for L in C.BAND} for cell in C.EMIT}
    for i, k in enumerate(keys):
        r = rend[k]["r"]
        _, acts = sec.forward(r.ids, record=list(C.BAND))
        for cell in C.EMIT:
            p = SA.cell_pos(r, cell)
            for L in C.BAND:
                H[cell][L][i] = acts[L][p].float().cpu()
        sec.clear()
    cap = {"H": H, "keys": keys}
    torch.save(cap, f)
    log(f"phase 1: captured {len(keys)} renders x {len(C.EMIT)} cells x {len(C.BAND)} layers")
    return cap


# ------------------------------------------------------------------------------------------------ Sigma
_SIG = {}
_WCUDA = {}


def wcuda(L):
    if L not in _WCUDA:
        _WCUDA[L] = load_sigma(L)["W"].float().cuda()
    return _WCUDA[L]


def load_sigma(L):
    if L in _SIG:
        return _SIG[L]
    s = torch.load(C.OUT / "sigma" / f"sigma_L{L}.pt")
    ev, U = s["evals"].double(), s["evecs"].double()
    W = (U * ev.rsqrt()) @ U.T
    Wi = (U * ev.sqrt()) @ U.T
    _SIG[L] = {"mu": s["mu"].double(), "W": W, "Wi": Wi, "lambda": s["lambda"]}
    return _SIG[L]


# ------------------------------------------------------------------------------------------------ phase 2
def colour_ids(tok):
    low = {c: tok.encode(" " + c, add_special_tokens=False) for c in C.COLOURS}
    cap = {c: tok.encode(" " + c.capitalize(), add_special_tokens=False) for c in C.COLOURS}
    assert all(len(v) == 1 for v in low.values()), low
    return {c: low[c][0] for c in C.COLOURS}, {c: cap[c][0] for c in C.COLOURS}, {c: len(cap[c]) for c in C.COLOURS}


def phase2(model, cap, rend, keys, tables, gated, tok, log):
    f = C.OUT / "directions.pt"
    if f.exists():
        return torch.load(f)
    import readers
    t_of = np.array([tables.index(rend[k]["item"]) for k in keys])
    zs = np.array([rend[k]["z"] for k in keys])
    need = sorted({(tables.index(e["pair"].item), c) for e in gated for c in (e["pair"].m.z, e["pair"].p.z)})
    W_U = model.get_output_embeddings().weight.detach().float().to("cuda")
    g = jspace.effective_gain(R.final_norm(model), 5120, device="cuda")
    low, capid, caplen = colour_ids(tok)
    out = {"need": need, "dirs": {}, "mu_dev": {}, "split": {}, "atoms_sel": {}, "report": {}}
    for L in C.BAND:
        sg = load_sigma(L)
        Wf, Wif = sg["W"].float().cuda(), sg["Wi"].float().cuda()
        rows, rkeys = [], []
        for cell in C.EMIT:
            Hc = cap["H"][cell][L].double()
            for T in sorted({t for t, _ in need}):
                m = t_of != T
                mu = Hc[torch.from_numpy(m)].mean(0)
                out["mu_dev"][(cell, L, T)] = mu.float()
                for (T2, c) in need:
                    if T2 != T:
                        continue
                    sel = torch.from_numpy(m & (zs == c))
                    assert int(sel.sum()) >= 1, (cell, L, T, c)
                    d = Hc[sel].mean(0) - mu
                    out["dirs"][(cell, L, T, c)] = d.float()
                    rows.append(d.float())
                    rkeys.append((cell, T, c))
        X = torch.stack(rows).cuda()
        Xw = (X.double() @ sg["W"].cuda()).float()          # whitened displacements W d (W symmetric)
        for lens_name in LENSES:
            lens = readers.load_lenses((lens_name,), layers=[L], device="cuda", verify_sha=True)[lens_name]
            Dm = jspace.atoms(lens.jacobians[L], W_U, gain=g)                     # [V, d] unit, raw
            out["atoms_sel"][(L, lens_name)] = {"low": Dm[[low[c] for c in C.COLOURS]].cpu(),
                                                "cap": Dm[[capid[c] for c in C.COLOURS]].cpu()}
            res = jspace.decompose(X, Dm, C.K_PURSUIT, refit="nnls", chunk=65536)
            own = []
            for (cell, T, c), xj, xn, e, sup in zip(rkeys, res["x_j"], res["x_nonj"], res["energy_j"], res["support"]):
                out["split"][("raw", cell, L, lens_name, T, c)] = {"J": xj.cpu(), "nonJ": xn.cpu(), "support": sup,
                                                                   "energy": float(e)}
                own.append(low[c] in sup or capid[c] in sup)
            erw = [float(e) for e in res["energy_j"]]
            # whitened atoms: unit(Sigma^1/2 v_t), built in chunks over the vocabulary
            for i in range(0, Dm.shape[0], 32768):
                blk = Dm[i:i + 32768] @ Wif
                Dm[i:i + 32768] = blk / blk.norm(dim=1, keepdim=True).clamp_min(1e-12)
            resw = jspace.decompose(Xw, Dm, C.K_PURSUIT, refit="nnls", chunk=65536)
            ownw = []
            for (cell, T, c), xj, xn, e, sup in zip(rkeys, resw["x_j"], resw["x_nonj"], resw["energy_j"], resw["support"]):
                out["split"][("white", cell, L, lens_name, T, c)] = {"J": xj.cpu(), "nonJ": xn.cpu(), "support": sup,
                                                                     "energy": float(e)}
                ownw.append(low[c] in sup or capid[c] in sup)
            out["report"][(L, lens_name)] = {
                "n_rows": len(rkeys), "raw_median_energy_J": float(np.median(erw)), "raw_own_atom_share": float(np.mean(own)),
                "white_median_energy_J": float(np.median([float(e) for e in resw["energy_j"]])),
                "white_own_atom_share": float(np.mean(ownw))}
            log(f"phase 2 L{L} {lens_name}: {out['report'][(L, lens_name)]}")
            del Dm, lens, res, resw
            torch.cuda.empty_cache()
        del Wf, Wif, X, Xw
        torch.cuda.empty_cache()
    out["colour_ids"] = {"low": low, "cap": capid, "cap_len": caplen}
    torch.save(out, f)
    C.dump(C.RES / "analysis" / "directions_report.json",
           {f"L{L}|{ln}": v for (L, ln), v in out["report"].items()})
    return out


# ------------------------------------------------------------------------------------------------ phase 3
def pinv(V):
    return torch.linalg.pinv(V)


def swap_target(c, pairs, alpha=1.0):
    t = c.clone()
    for i, j in pairs:
        t[i] = c[i] + alpha * (c[j] - c[i])
        t[j] = c[j] + alpha * (c[i] - c[j])
    return t


class Pair:
    def __init__(self, sec, tok, e, groups, cap_keys, tables):
        self.sec, self.tok, self.e, self.groups = sec, tok, e, groups
        pr = e["pair"]
        self.pair, self.rx, self.ry = pr, e["r_m"], e["r_p"]
        self.T = tables.index(pr.item)
        self.Sa, self.Sb = MG.spellings(tok, "brew", pr.m.a), MG.spellings(tok, "brew", pr.p.a)
        self.pos = {cell: SA.cell_pos(self.rx, cell) for cell in C.EMIT}
        for cell in C.EMIT:
            assert self.pos[cell] == SA.cell_pos(self.ry, cell)
        self.conts = [()] + [s for g in groups for s in g if len(s) > 1]
        self.clean, self.donor42 = {}, {}
        for c in self.conts:
            _, a = sec.forward(self.rx.ids + list(c), record=list(C.BAND))
            self.clean[c] = {L: {p: a[L][p].clone() for p in self.pos.values()} for L in C.BAND}
            _, b = sec.forward(self.ry.ids + list(c), record=[42])
            self.donor42[c] = b[42][self.pos["emit_colon"]].clone()
        self.cs = self.score(self.rx.ids)
        self.ds = self.score(self.ry.ids)
        self.css, self.dss = SA.summarize(self.cs, groups), SA.summarize(self.ds, groups)
        self.G = self.ds["m"] - self.cs["m"]
        self.base = {"pair_id": pr.pair_id, "item": pr.item, "table": self.T, "G": self.G, "m_clean": self.cs["m"],
                     "s": pr.m.inp, "s2": pr.p.inp, "z": pr.m.z, "z2": pr.p.z, "a": pr.m.a, "a2": pr.p.a}

    def score(self, ids, edit_fn=None):
        return self.sec.score(ids, self.Sa, self.Sb, edit_fn=edit_fn, extra_spellings=self.groups)

    def top_colour(self, sc):
        lq = SA.summarize(sc, self.groups)["logq"]
        return C.COLOURS[int(np.argmax(lq))]

    def evaluate(self, cond):
        """cond: {"name", "specs": {L: [(p, A, R, mu, tfun)]}} with A [d,k], R [k,d], mu [d] float32 cuda;
        tfun(c_clean [k], cont) -> target [k]."""
        t0 = time.time()
        fid = SA.Fidelity()
        cerr, enorm = [], []

        def edit_fn(cont):
            cl = self.clean[cont]
            fns = {}
            for L, specs in cond["specs"].items():
                prep = []
                for (p, A, Rm, mu, tfun) in specs:
                    with torch.autocast("cuda", enabled=False):
                        c0 = Rm @ (cl[L][p] - mu)
                        prep.append((p, A, Rm, mu, tfun(c0, L, p)))

                def f(x, prep=prep):
                    with torch.autocast("cuda", enabled=False):
                        for p, A, Rm, mu, t in prep:
                            h = x[p]
                            d = A @ (t - Rm @ (h - mu))
                            x, req, real, hh = SA.apply_write(x, p, d=d)
                            fid.add(req, real, hh)
                            enorm.append(float(req.norm()))
                            cerr.append(float((Rm @ (x[p] - mu) - t).abs().max()) / max(1.0, float(t.abs().max())))
                    return x
                fns[L] = f
            return fns, []
        ed = self.score(self.rx.ids, edit_fn=edit_fn)
        es = SA.summarize(ed, self.groups)
        first = min(p for specs in cond["specs"].values() for (p, *_r) in specs)
        dmg = RG.sec_damage(self.sec, self.cs, ed, first, len(self.rx))
        sm = fid.summary()
        top = self.top_colour(ed)
        row = {**self.base, **{k: v for k, v in cond.items() if k not in ("specs",)},
               "dm": ed["m"] - self.cs["m"], "S": (ed["m"] - self.cs["m"]) / self.G,
               "d_logP_a": es["lse_a"] - self.css["lse_a"], "d_logP_b": es["lse_b"] - self.css["lse_b"],
               "top1": top, "flip": top == self.pair.p.a, "top1_retention": dmg["top1_retention"],
               "off_pair_mass_clean": dmg["off_pair_mass_clean"], "off_pair_mass_edit": dmg["off_pair_mass_edit"],
               "fid_ok": fid.ok(), "fid_rho": sm["fid_rho"], "fid_kappa_min": sm["fid_kappa_min"], "fid_modes": sm["fid_modes"],
               "clamp_err_max": max(cerr) if cerr else None, "clamp_ok": bool(cerr) and max(cerr) <= CLAMP_TOL,
               "edit_norm_sum_base": float(sum(enorm[:len(enorm) // max(1, len(self.conts))])), "seconds": time.time() - t0}
        return row

    def repro(self):
        p = self.pos["emit_colon"]

        def edit_fn(cont):
            src = self.donor42[cont]

            def f(x):
                with torch.autocast("cuda", enabled=False):
                    x, *_ = SA.apply_write(x, p, target=src)
                return x
            return {42: f}, []
        ed = self.score(self.rx.ids, edit_fn=edit_fn)
        return (ed["m"] - self.cs["m"]) / self.G


class Builder:
    """Condition matrices for one pair (float64 on CPU, cast to float32 cuda at the end)."""

    def __init__(self, P, dirs, sig):
        self.P, self.d, self.sig = P, dirs, sig

    @staticmethod
    def fin(A, Rm, mu):
        RA = Rm @ A
        err = float((RA - torch.eye(RA.shape[0], dtype=RA.dtype)).abs().max())
        return (A.float().cuda(), Rm.float().cuda(), mu.float().cuda()), err

    def lens_V(self, L, lens, cols):
        a = self.d["atoms_sel"][(L, lens)]
        idx = {c: i for i, c in enumerate(C.COLOURS)}
        V = [a["low" if sp == "low" else "cap"][idx[c]].double() for c, sp in cols]
        return torch.stack(V, 1)

    def mats(self, kind, V, L, mu):
        """kind: raw (A=V, R=V+), white (V are raw-space objects mapped by `wmap`)."""
        if kind == "raw":
            return self.fin(V, pinv(V), mu)
        sg = self.sig[L]
        W, Wi = sg["W"], sg["Wi"]
        if kind == "white_cov":                     # covectors (J-lens atoms): V~ = unit(W^-1 v)
            Vt = Wi @ V
            Vt = Vt / Vt.norm(dim=0, keepdim=True)
        elif kind == "white_disp":                  # displacements already in whitened space (passed as V~)
            Vt = V
        else:
            raise ValueError(kind)
        return self.fin(Wi @ Vt, pinv(Vt) @ W, mu)


def build_conditions(P, D, sig, rng_seed):
    """Every condition for pair P. Returns a list of cond dicts and the max RA error."""
    pr = P.pair
    z, z2, a, a2, s, s2 = pr.m.z, pr.p.z, pr.m.a, pr.p.a, pr.m.inp, pr.p.inp
    B = Builder(P, D, sig)
    ra_errs = []
    zero_mu = torch.zeros(5120, dtype=torch.float64)
    traj = {s, z, a, s2, z2, a2}
    off = [c for c in C.COLOURS if c not in traj]
    rng = np.random.default_rng(SA.seed_of(pr.pair_id, "008_off"))
    o1, o2 = (list(rng.choice(off, 2, replace=False)) if len(off) >= 2 else (None, None))
    caplen = D["colour_ids"]["cap_len"]

    def spec_for(site, make):
        """make(L, cell, p) -> ((A, R, mu), tfun) ; returns {L: [(p, A, R, mu, tfun)]}."""
        out = {}
        for L in C.BAND:
            out[L] = []
            for cell in C.SITES[site]:
                p = P.pos[cell]
                (A, Rm, mu), tf = make(L, cell, p)
                out[L].append((p, A, Rm, mu, tf))
        return out

    def sw(alpha=1.0, pairs=((0, 1),)):
        return lambda c, L, p: swap_target(c, pairs, alpha)

    conds = []

    def add(name, site, make, **meta):
        conds.append({"name": name, "site": site, "specs": spec_for(site, make), **meta})

    # ---------------- (A) the paper's lens-coordinate swap on colour atoms
    def lens_make(lens, cols, kind, mu_kind, alpha=1.0, pairs=((0, 1),)):
        def make(L, cell, p):
            V = B.lens_V(L, lens, cols)
            mu = zero_mu if mu_kind == "zero" else sig[L]["mu"]
            mats, err = B.mats(kind, V, L, mu)
            ra_errs.append(err)
            return mats, sw(alpha, pairs)
        return make

    for site in ("emit3", "colon"):
        for lens in LENSES:
            add(f"A_raw_a1_{lens}", site, lens_make(lens, [(z, "low"), (z2, "low")], "raw", "zero"),
                arm="A", lens=lens, metric="raw", alpha=1.0, mag="natural")
            add(f"A_white_a1_{lens}", site, lens_make(lens, [(z, "low"), (z2, "low")], "white_cov", "corpus"),
                arm="A", lens=lens, metric="white", alpha=1.0, mag="natural")
            if site == "emit3":
                add(f"A_raw_a2_{lens}", site, lens_make(lens, [(z, "low"), (z2, "low")], "raw", "zero", 2.0),
                    arm="A", lens=lens, metric="raw", alpha=2.0, mag="natural")
                add(f"A_white_a2_{lens}", site, lens_make(lens, [(z, "low"), (z2, "low")], "white_cov", "corpus", 2.0),
                    arm="A", lens=lens, metric="white", alpha=2.0, mag="natural")
                add(f"A_rawc_a1_{lens}", site, lens_make(lens, [(z, "low"), (z2, "low")], "raw", "corpus"),
                    arm="A", lens=lens, metric="raw_centred", alpha=1.0, mag="natural")
                add(f"Actrl_answer_raw_{lens}", site, lens_make(lens, [(a, "low"), (a2, "low")], "raw", "zero"),
                    arm="A_ctrl_answer", lens=lens, metric="raw", alpha=1.0, mag="natural")
                if o1 is not None:
                    add(f"Actrl_off_raw_{lens}", site, lens_make(lens, [(o1, "low"), (o2, "low")], "raw", "zero"),
                        arm="A_ctrl_off", lens=lens, metric="raw", alpha=1.0, mag="natural", off=[o1, o2])
                    add(f"Actrl_off_white_{lens}", site, lens_make(lens, [(o1, "low"), (o2, "low")], "white_cov", "corpus"),
                        arm="A_ctrl_off", lens=lens, metric="white", alpha=1.0, mag="natural", off=[o1, o2])
        if site == "emit3" and caplen[z] == 1 and caplen[z2] == 1:
            add("A_raw_both_J_NP", site, lens_make("J_NP", [(z, "low"), (z, "cap"), (z2, "low"), (z2, "cap")], "raw",
                                                   "zero", 1.0, ((0, 2), (1, 3))),
                arm="A_both", lens="J_NP", metric="raw", alpha=1.0, mag="natural")

    # ---------------- (B) class-mean z directions (Fig. 16 style)
    def V_part(metric, lens, part, L, cell):
        if part == "full":
            dz, dz2 = D["dirs"][(cell, L, P.T, z)].double(), D["dirs"][(cell, L, P.T, z2)].double()
            if metric == "white":
                W = sig[L]["W"]
                return torch.stack([W @ dz, W @ dz2], 1)
            return torch.stack([dz, dz2], 1)
        key = lambda c: (metric, cell, L, lens, P.T, c)
        return torch.stack([D["split"][key(z)][part].double(), D["split"][key(z2)][part].double()], 1)

    def B_make(metric, lens, part, mag="natural", clamp=None):
        def make(L, cell, p):
            mu = D["mu_dev"][(cell, L, P.T)].double()
            V = V_part(metric, lens, part, L, cell)
            if clamp is not None:                               # complementary clamp: extra columns held at clean
                if clamp == "Jatoms":
                    sups = set(D["split"][("raw", cell, L, lens, P.T, z)]["support"]) | \
                        set(D["split"][("raw", cell, L, lens, P.T, z2)]["support"])
                    lowid, capid = D["colour_ids"]["low"], D["colour_ids"]["cap"]
                    sups |= {lowid[z], lowid[z2], capid[z], capid[z2]}
                    Y = jatoms(L, lens, sorted(sups))
                else:                                           # "nonJ": the non-J parts' coordinates
                    Y = V_part("raw", lens, "nonJ", L, cell)
                V = torch.cat([V, Y], 1)
            kind = "raw" if metric == "raw" else "white_disp"
            mats, err = B.mats(kind, V, L, mu)
            ra_errs.append(err)
            A_, R_, mu_ = mats
            if mag == "natural":
                tf = sw()
            else:                                               # rescale the edit to the full condition's norm
                Vf = V_part(metric, lens, "full", L, cell)
                (Af, Rf, muf), e2 = B.mats(kind, Vf, L, mu)
                ra_errs.append(e2)
                Wl = wcuda(L)

                def tf(c, L_, p_, A_=A_, R_=R_, mu_=mu_, Af=Af, Rf=Rf, muf=muf, Wl=Wl):
                    h = P.clean[()][L_][p_]
                    cf = Rf @ (h - muf)
                    ef = Af @ (swap_target(cf, ((0, 1),)) - cf)
                    ex = A_ @ (swap_target(c, ((0, 1),)) - c)
                    if mag == "rawnorm":
                        sc = float(ef.norm() / ex.norm().clamp_min(1e-12))
                    else:
                        sc = float((Wl @ ef).norm() / (Wl @ ex).norm().clamp_min(1e-12))
                    return c + sc * (swap_target(c, ((0, 1),)) - c)
            return mats, tf
        return make

    atom_cache = {}

    def jatoms(L, lens, ids):
        # raw folded atoms for arbitrary token ids (complementary clamp), rebuilt from the lens + W_U lazily
        key = (L, lens, tuple(ids))
        if key not in atom_cache:
            atom_cache[key] = D["_atom_fn"](L, lens, ids)
        return atom_cache[key]

    for site in ("emit3", "colon"):
        add("B_raw_full", site, B_make("raw", None, "full"), arm="B", lens=None, part="full", metric="raw", mag="natural")
        add("B_white_full", site, B_make("white", None, "full"), arm="B", lens=None, part="full", metric="white", mag="natural")
        for lens in LENSES:
            for part in ("J", "nonJ"):
                add(f"B_raw_{part}_{lens}", site, B_make("raw", lens, part), arm="B", lens=lens, part=part, metric="raw",
                    mag="natural")
                add(f"B_white_{part}_{lens}", site, B_make("white", lens, part), arm="B", lens=lens, part=part,
                    metric="white", mag="natural")
                add(f"B_raw_{part}_{lens}_rawnorm", site, B_make("raw", lens, part, "rawnorm"), arm="B", lens=lens,
                    part=part, metric="raw", mag="rawnorm")
                if site == "emit3":
                    add(f"B_raw_{part}_{lens}_whitenorm", site, B_make("raw", lens, part, "whitenorm"), arm="B",
                        lens=lens, part=part, metric="raw", mag="whitenorm")
            if site == "emit3":
                add(f"B_raw_nonJ_clampJ_{lens}", site, B_make("raw", lens, "nonJ", clamp="Jatoms"), arm="B_comp",
                    lens=lens, part="nonJ|clampJ", metric="raw", mag="natural")
                add(f"B_raw_J_clampNonJ_{lens}", site, B_make("raw", lens, "J", clamp="nonJ"), arm="B_comp",
                    lens=lens, part="J|clampNonJ", metric="raw", mag="natural")
        # random direction pairs, edits matched to the full natural raw edit norm
        for i in range(4):
            def make(L, cell, p, i=i):
                g1 = torch.from_numpy(SA.unit(SA.seed_of(pr.pair_id, "008_rand", i, 0))).double()
                g2 = torch.from_numpy(SA.unit(SA.seed_of(pr.pair_id, "008_rand", i, 1))).double()
                V = torch.stack([g1, g2], 1)
                mu = D["mu_dev"][(cell, L, P.T)].double()
                mats, err = B.mats("raw", V, L, mu)
                ra_errs.append(err)
                Vf = V_part("raw", None, "full", L, cell)
                (Af, Rf, muf), _ = B.mats("raw", Vf, L, mu)
                A_, R_, mu_ = mats

                def tf(c, L_, p_, A_=A_, Af=Af, Rf=Rf, muf=muf):
                    h = P.clean[()][L_][p_]
                    cf = Rf @ (h - muf)
                    ef = Af @ (swap_target(cf, ((0, 1),)) - cf)
                    ex = A_ @ (swap_target(c, ((0, 1),)) - c)
                    sc = float(ef.norm() / ex.norm().clamp_min(1e-12))
                    return c + sc * (swap_target(c, ((0, 1),)) - c)
                return mats, tf
            add(f"B_rand_{i}", site, make, arm="B_rand", lens=None, part="rand", metric="raw", mag="rawnorm", draw=i)
    return conds, (max(ra_errs) if ra_errs else 0.0)


def zero_condition(P, D, sig):
    """G0: the class-mean full raw clamp with target = clean coordinates (an exact no-op)."""
    z, z2 = P.pair.m.z, P.pair.p.z

    def make(L, cell, p):
        mu = D["mu_dev"][(cell, L, P.T)].double()
        V = torch.stack([D["dirs"][(cell, L, P.T, z)].double(), D["dirs"][(cell, L, P.T, z2)].double()], 1)
        (A, Rm, mu_), _ = Builder.fin(V, pinv(V), mu)
        return (A, Rm, mu_), (lambda c, L_, p_: c.clone())
    specs = {L: [] for L in C.BAND}
    for L in C.BAND:
        for cell in C.EMIT:
            p = P.pos[cell]
            (A, Rm, mu), tf = make(L, cell, p)
            specs[L].append((p, A, Rm, mu, tf))
    return {"name": "G0_zero", "site": "emit3", "arm": "G0", "specs": specs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="b1")
    ap.add_argument("--limit-pairs", type=int, default=0)
    a = ap.parse_args()
    assert not torch.backends.cuda.matmul.allow_tf32 and torch.get_float32_matmul_precision() == "highest", "TF32 must be off"
    d = C.OUT / a.run_id
    (d / "pairs").mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"run_id": a.run_id, "manifest": R.manifest_block(),
                                                 "design": "docs/design_specs/008_009_trackb_brewcode.md Part 1"}, indent=1))
    log = C.Logger(d / "log.txt")
    t_start = time.time()
    tok = R.make_tokenizer()
    rend, keys, tables, gated = material(tok)
    groups = SA.spelling_groups(tok)
    ref001 = p001()
    model = R.load_model()
    with RG.SecEngine(model, l0=39) as sec:
        cap = phase1(sec, rend, keys, log)
    log("phase 1 done; SEC restored for phase 2", f"free GPU {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")
    D = phase2(model, cap, rend, keys, tables, gated, tok, log)
    sig = {L: load_sigma(L) for L in C.BAND}
    # lazy raw atoms for the complementary clamp (lens + W_U kept on CPU; built per call)
    import readers
    W_U = model.get_output_embeddings().weight.detach().float().cpu()
    g = jspace.effective_gain(R.final_norm(model), 5120, device="cpu")
    lens_cpu = {}

    def atom_fn(L, lens, ids):
        if lens not in lens_cpu:
            lens_cpu[lens] = readers.load_lenses((lens,), layers=list(C.BAND), device="cpu", verify_sha=True)[lens].jacobians
        Jm = lens_cpu[lens][L]
        v = (W_U[ids] * g) @ Jm
        return (v / v.norm(dim=1, keepdim=True)).T.double()
    D["_atom_fn"] = atom_fn
    state = {"smoke": {}, "quarantined": 0, "ra_err_max": 0.0, "repro_fail": []}
    ents = gated[: a.limit_pairs] if a.limit_pairs else gated
    with RG.SecEngine(model, l0=39) as sec:
        for n, e in enumerate(ents):
            pid = e["pair"].pair_id
            f = d / "pairs" / f"{pid}.jsonl"
            if f.exists():
                continue
            t0 = time.time()
            P = Pair(sec, tok, e, groups, keys, tables)
            rows = [{**P.base, "name": "clean", "member": P.css, "partner": P.dss}]
            S_rep = P.repro()
            rep_ok = abs(S_rep - ref001[pid]) <= REPRO_TOL
            rows.append({**P.base, "name": "G_repro", "S": S_rep, "S_001": ref001[pid], "pass": rep_ok})
            if not rep_ok:
                state["repro_fail"].append(pid)
                log(f"G-repro FAIL {pid}: {S_rep:.4f} vs 001 {ref001[pid]:.4f}")
            if n < 2:
                r0 = P.evaluate(zero_condition(P, D, sig))
                g0 = abs(r0["dm"]) <= C.FLOOR and r0["fid_ok"]
                state["smoke"][pid] = {"G0_dm": r0["dm"], "G0_pass": g0}
                rows.append({**r0, "pass": g0})
                log(f"G0 {pid}: dm {r0['dm']:.2e} pass {g0}")
            conds, raerr = build_conditions(P, D, sig, n)
            state["ra_err_max"] = max(state["ra_err_max"], raerr)
            for cond in conds:
                r = P.evaluate(cond)
                if not (r["fid_ok"] and r["clamp_ok"]):
                    r["quarantined"] = True
                    state["quarantined"] += 1
                rows.append(r)
            tmp = f.with_suffix(".tmp")
            with open(tmp, "w") as fo:
                for r in rows:
                    fo.write(json.dumps(r, default=float) + "\n")
            tmp.rename(f)
            sec.clear()
            log(f"[{n + 1}/{len(ents)}] {pid} {len(conds)} conds {time.time() - t0:.0f}s repro {S_rep:.3f}/{ref001[pid]:.3f} "
                f"RA {raerr:.1e} q {state['quarantined']}")
            C.dump(d / "state.json", state)
    state["minutes"] = (time.time() - t_start) / 60
    state["G_RA_pass"] = state["ra_err_max"] <= RA_TOL
    C.dump(d / "state.json", state)
    log(f"done in {state['minutes']:.1f} min; RA max {state['ra_err_max']:.2e}; quarantined {state['quarantined']}")


if __name__ == "__main__":
    main()
