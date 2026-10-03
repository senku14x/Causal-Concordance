"""Design 004 v2 — span assay diagnostic at ':' L42 (docs/design_specs/004_span_assay.md).

    python scripts/004_span_assay/span_assay.py preflight   # CPU: pair manifest, bases, identities, zero writes, arm manifest
    python scripts/004_span_assay/span_assay.py smoke       # GPU: 2 pairs end to end + G1/G2 checks, timing
    python scripts/004_span_assay/span_assay.py run --pi-go # GPU: all 33 pairs (only after the smoke passes and the PI's go)

Reader-free: imports scripts/common label modules only (brew_material, registry, margin, regime); never the mixed 001
module or the 003 module (tests/test_import_boundary.py walks this file's import closure).
Reuses 003's saved least-squares coefficients and fitting means (outputs/003_labels/directions.npz), the 001 emission
captures and the 003 start-token captures, unchanged.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import brew_material as BM  # noqa: E402
import registry as R  # noqa: E402

EXP = "004_span_assay"
RES = R.out_dir("results", EXP)
OUT = R.out_dir("outputs", EXP)
D003 = R.PROJECT / "outputs" / "003_labels"
CAP001 = R.PROJECT / "outputs" / "001_leverage_profile" / "run"
SEED = 20261007
SVD_RTOL = 1e-6
MIN_OCC = 2
NEG_REL = 1e-4          # a write with ||d|| < NEG_REL ||h|| is applied but called negligible: fidelity not assessable
FLOOR = 2e-4            # SEC numerical floor (nats), 001 v3
FLAG_OUTSIDE = 0.05     # warning: mass outside the accepted spellings up by more than this
FLAG_BOTH_DOWN = 0.05   # warning (swaps): log P(a) and log P(a') both down by more than this (nats)
N_DRAWS = 16
ROLES = ("start", "z", "answer")
COLOURS = ("black", "blue", "brown", "gold", "gray", "green", "orange", "pink", "purple", "red", "teal", "white")
FITS = ("both", "r1", "r2")
SITES = {"colon": ("emit_colon", 42), "start_ref": ("start_tok", 40)}
LAYERS_CAP = [40, 42, 44, 56, 60]       # layer axis of the 001 / 003 captures
CTRL_ARMS = [("colon", "z1d"), ("colon", "Z"), ("colon", "J"), ("start_ref", "Sspan")]   # swap controls and necessity


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def write_manifest(stage, extra):
    m = {"stage": stage, "time_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), **R.manifest_block(), **extra}
    write_json(RES / f"manifest_{stage}.json", m)


def seed_of(*parts):
    return int(hashlib.sha256("|".join(map(str, (SEED,) + parts)).encode()).hexdigest()[:15], 16)


# ------------------------------------------------------------------------------------------- material
def natural_meta():
    meta = json.loads((CAP001 / "captures_brew_meta.json").read_text())
    nat = [i for i, r in enumerate(meta["renders"]) if r["role"] in ("member", "partner")]
    return meta, nat


def load_acts():
    """{cell: [n_nat, 5 layers, d]} for the two 004 cells (001 / 003 captures, 003's row order) and the render rows."""
    meta, nat = natural_meta()
    X = np.load(CAP001 / "captures_brew.npz")["X"]
    acts = {"emit_colon": X[nat, meta["cells"].index("emit_colon")]}
    st = np.load(D003 / "captures" / "start_token.npz")
    assert list(st["keys"]) == [meta["renders"][i]["key"] for i in nat]
    acts["start_tok"] = st["X"]
    return acts, [meta["renders"][i] for i in nat]


def start_pos(r):
    return sorted(p for p, g in r.regions.items() if g == "start")[4]       # the start-colour token (" blue" etc.)


def cell_pos(r, cell):
    if cell == "start_tok":
        return start_pos(r)
    return [p for p, g in r.regions.items() if g == cell][0]


def pair_manifest(entries):
    """One gated dev pair per table, seeded; frozen to results/004_span_assay/manifest_pairs.json."""
    by_table = {}
    for e in entries:
        by_table.setdefault(e["pair"].item, []).append(e["pair"].pair_id)
    rng = np.random.default_rng(SEED)
    chosen = {t: sorted(v)[int(rng.integers(len(v)))] for t, v in sorted(by_table.items())}
    return {"seed": SEED, "n_tables": len(chosen), "pairs": [chosen[t] for t in sorted(chosen)], "exclusions": [],
            "eligible_pairs_per_table": {t: len(v) for t, v in sorted(by_table.items())},
            "source": "003 gated dev brew pairs (82), via scripts/common/brew_material.py"}


def material(tok):
    entries = BM.gated_dev_brew(BM.brew_pairs(tok))
    man = json.loads((RES / "manifest_pairs.json").read_text())
    by_id = {e["pair"].pair_id: e for e in entries}
    return [by_id[p] for p in man["pairs"]], man


# ------------------------------------------------------------------------------------------- bases
def orth(M):
    """Orthonormal basis (float64) of the column span of M, SVD with relative cutoff SVD_RTOL."""
    if M.shape[1] == 0:
        return M
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, s > s[0] * SVD_RTOL]


def haar(seed, k):
    """Haar-random 5120 x k orthonormal frame: Gaussian QR with the sign correction."""
    G = np.random.default_rng(seed).standard_normal((5120, k))
    Q, Rr = np.linalg.qr(G)
    return Q * np.sign(np.diag(Rr))


def unit(seed):
    r = np.random.default_rng(seed).standard_normal(5120)
    return r / np.linalg.norm(r)


class Fits:
    """003 least-squares coefficients per (fold, fit, cell, layer): role spans, the joint span, 1-D directions,
    each fit's own fitting mean, and the PCA comparator built from that fit's own fitting rows."""

    def __init__(self, acts=None, rows=None):
        with np.load(D003 / "directions.npz") as z:
            self.z = {k: z[k] for k in z.files}
        self.info = json.loads((D003 / "directions_info.json").read_text())
        self.fold_of_table = self.info["folds"]
        self.acts, self.rows = acts, rows
        self._pca = {}

    def key(self, fold, fit, cell, L):
        return f"f{fold}|{fit}|{cell}|L{L}"

    def fit_folds(self, fold, fit):
        others = [g for g in range(3) if g != fold]                 # 003 fit_sets(): both / r1 / r2
        return {"both": set(others), "r1": {others[0]}, "r2": {others[1]}}[fit]

    def fit_rows(self, fold, fit):
        use = self.fit_folds(fold, fit)
        return [i for i, r in enumerate(self.rows) if self.fold_of_table[r["item"]] in use]

    def eligible(self, fold, fit, cell, L, role):
        k = self.key(fold, fit, cell, L)
        cnt = self.info["info"][k]["counts"][role]
        return [c for c in COLOURS if cnt[c] >= MIN_OCC and f"{k}|{role}|{c}" in self.z]

    def coef(self, fold, fit, cell, L, role, c):
        return self.z[f"{self.key(fold, fit, cell, L)}|{role}|{c}"].astype(np.float64)

    def fitmean(self, fold, fit, cell, L):
        return self.z[f"{self.key(fold, fit, cell, L)}|fitmean"].astype(np.float64)        # that fit's own mean

    def role_span(self, fold, fit, cell, L, role):
        el = self.eligible(fold, fit, cell, L, role)
        if len(el) < 2:
            return np.zeros((5120, 0)), el
        B = np.stack([self.coef(fold, fit, cell, L, role, c) for c in el])
        return orth((B[1:] - B[0]).T), el

    def joint_span(self, fold, fit, cell, L):
        return orth(np.concatenate([self.role_span(fold, fit, cell, L, r)[0] for r in ROLES], axis=1))

    def direction(self, fold, fit, cell, L, role, c, c2):
        el = self.eligible(fold, fit, cell, L, role)
        if c not in el or c2 not in el:
            return None
        u = self.coef(fold, fit, cell, L, role, c2) - self.coef(fold, fit, cell, L, role, c)
        return u / np.linalg.norm(u)

    def pca(self, fold, fit, cell, L, k):
        """Top-k PCA directions of the fit's own rows at the cell: centred within table, rows weighted 1/sqrt(n_t)."""
        key = (fold, fit, cell, L)
        if key not in self._pca:
            idx = self.fit_rows(fold, fit)
            X = self.acts[cell][idx, LAYERS_CAP.index(L)].astype(np.float64)
            items = np.array([self.rows[i]["item"] for i in idx])
            parts = []
            for t in sorted(set(items)):
                Xt = X[items == t]
                parts.append((Xt - Xt.mean(0)) / np.sqrt(len(Xt)))
            _, s, Vt = np.linalg.svd(np.concatenate(parts), full_matrices=False)
            self._pca[key] = (Vt.T, s)
        V, s = self._pca[key]
        return V[:, :k]

    def pair_bases(self, pair):
        """{fit: {site: {arm: basis [5120, k] float64}, 'mu': {site: mean}}} for one pair (its table's fold)."""
        fold = self.fold_of_table[pair.item]
        out = {}
        for fit in FITS:
            b = {}
            cell, L = SITES["colon"]
            QZ, _ = self.role_span(fold, fit, cell, L, "z")
            QS, _ = self.role_span(fold, fit, cell, L, "start")
            QA, _ = self.role_span(fold, fit, cell, L, "answer")
            J = self.joint_span(fold, fit, cell, L)
            u = self.direction(fold, fit, cell, L, "z", pair.m.z, pair.p.z)
            P = self.pca(fold, fit, cell, L, max(QZ.shape[1], J.shape[1]))
            b["colon"] = {"z1d": None if u is None else u[:, None], "Z": QZ, "J": J, "S": QS, "A": QA,
                          "pca_kZ": P[:, :QZ.shape[1]], "pca_kJ": P[:, :J.shape[1]], "mu": self.fitmean(fold, fit, cell, L)}
            cell, L = SITES["start_ref"]
            QSt, _ = self.role_span(fold, fit, cell, L, "start")
            us = self.direction(fold, fit, cell, L, "start", pair.m.inp, pair.p.inp)
            b["start_ref"] = {"s1d": None if us is None else us[:, None], "Sspan": QSt,
                              "pca_kS": self.pca(fold, fit, cell, L, QSt.shape[1]), "mu": self.fitmean(fold, fit, cell, L)}
            out[fit] = b
        return fold, out


# ------------------------------------------------------------------------------------------- conditions
def conditions(pair_id, bases):
    """Every scored condition for one pair (design §3). op: span | comp | full | normdir | pca_dose | nec | rot | rclamp.
    prompts = 1 (member run) or 2 (necessity: member and partner runs)."""
    C = []

    def add(kind, site, fit, arm, op, basis=None, ref=None, draw=None, seed=None, k=None):
        C.append({"kind": kind, "site": site, "fit": fit, "arm": arm, "op": op, "basis": basis, "ref": ref, "draw": draw,
                  "seed": seed, "k": k, "prompts": 2 if op in ("nec", "rot", "rclamp") else 1})

    for site in SITES:
        add("swap", site, None, "full", "full", k=5120)
    for fit in FITS:
        b = bases[fit]
        for arm in ("z1d", "Z", "J", "S", "A"):
            add("swap", "colon", fit, arm, "span", basis=arm, k=None if b["colon"][arm] is None else b["colon"][arm].shape[1])
        for arm in ("Z", "J"):
            add("swap", "colon", fit, arm + "c", "comp", basis=arm, k=b["colon"][arm].shape[1])
        add("swap", "start_ref", fit, "s1d", "span", basis="s1d", k=None if b["start_ref"]["s1d"] is None else 1)
        add("swap", "start_ref", fit, "Sspan", "span", basis="Sspan", k=b["start_ref"]["Sspan"].shape[1])
        add("swap", "start_ref", fit, "Sc", "comp", basis="Sspan", k=b["start_ref"]["Sspan"].shape[1])
        for site, pk, content in (("colon", "pca_kZ", "Z"), ("colon", "pca_kJ", "J"), ("start_ref", "pca_kS", "Sspan")):
            k = b[site][pk].shape[1]
            add("pca", site, fit, pk, "span", basis=pk, k=k)
            add("pca", site, fit, pk + "_dose", "pca_dose", basis=pk, ref=content, k=k)
        for site, arm in CTRL_ARMS:
            B = b[site][arm]
            add("nec", site, fit, arm, "nec", basis=arm, k=None if B is None else B.shape[1])
    b = bases["both"]
    for site, arm in CTRL_ARMS:
        B = b[site][arm]
        if B is None:
            continue
        k = B.shape[1]
        for i in range(N_DRAWS):
            add("swap_ctrl", site, "both", f"rank:{arm}", "span", draw=i, seed=seed_of(pair_id, "rank", arm, i), k=k)
            add("swap_ctrl", site, "both", f"norm:{arm}", "normdir", ref=arm, draw=i, seed=seed_of(pair_id, "norm", arm, i), k=k)
            if arm == "J":          # random complement shares the rank-matched J draw (same R)
                add("swap_ctrl", site, "both", "rcomp:J", "comp", draw=i, seed=seed_of(pair_id, "rank", arm, i), k=k)
            add("nec_ctrl", site, "both", f"rot:{arm}", "rot", basis=arm, draw=i, seed=seed_of(pair_id, "rot", arm, i), k=k)
            add("nec_ctrl", site, "both", f"rclamp:{arm}", "rclamp", draw=i, seed=seed_of(pair_id, "rclamp", arm, i), k=k)
    return C


def draw_matrix(c):
    """The random matrix of a control condition (float64), regenerated from its saved seed."""
    if c["op"] == "normdir":
        return unit(c["seed"])[:, None]
    return haar(c["seed"], c["k"])


# ------------------------------------------------------------------------------------------- writes
def apply_write(x, pos, d=None, target=None):
    """x [T, d] float32 -> (new x, requested, realized, h). Either an additive d or a direct assignment `target`."""
    x = x.clone()
    h = x[pos].clone()
    if target is not None:
        x[pos] = target
        req = target - h
    else:
        x[pos] = h + d
        req = d
    return x, req, x[pos] - h, h


class Fidelity:
    def __init__(self):
        self.rho, self.kappa, self.negligible, self.req_norm, self.h_norm = [], [], 0, [], []

    def add(self, req, real, h):
        import torch
        rn, hn = float(req.norm()), float(h.norm())
        self.req_norm.append(rn)
        self.h_norm.append(hn)
        if rn < NEG_REL * hn:
            self.negligible += 1
            return
        self.rho.append(float(real.norm()) / rn)
        self.kappa.append(float(torch.nn.functional.cosine_similarity(req[None], real[None])[0]))

    def summary(self):
        return {"fid_rho": [min(self.rho), max(self.rho)] if self.rho else None,
                "fid_kappa_min": min(self.kappa) if self.kappa else None, "fid_negligible": self.negligible,
                "write_norm": self.req_norm[0] if self.req_norm else None, "h_norm": self.h_norm[0] if self.h_norm else None}

    def ok(self):
        return all(0.9 <= r <= 1.1 for r in self.rho) and all(k >= 0.99 for k in self.kappa)


def make_delta(op, M, h, hd, mu):
    """The additive write for one op at one position (float32 torch). M: dict of float32 matrices for the condition."""
    if op == "span":
        D = hd - h
        return M["B"] @ (M["B"].T @ D)
    if op == "comp":
        D = hd - h
        return D - M["B"] @ (M["B"].T @ D)
    if op == "normdir":
        D = hd - h
        return (M["Bref"] @ (M["Bref"].T @ D)).norm() * M["r"][:, 0]
    if op == "pca_dose":
        D = hd - h
        p = M["B"] @ (M["B"].T @ D)
        return p * ((M["Bref"] @ (M["Bref"].T @ D)).norm() / p.norm())
    if op == "nec":
        return -(M["B"] @ (M["B"].T @ (h - mu)))
    if op == "rot":
        return -(M["U"] @ (M["B"].T @ (h - mu)))
    if op == "rclamp":
        return -(M["U"] @ (M["U"].T @ (h - mu)))
    raise ValueError(op)


# ------------------------------------------------------------------------------------------- scoring
def spelling_groups(tok):
    import margin as MG
    return [MG.spellings(tok, "brew", c) for c in COLOURS]


def summarize(sc, groups):
    logq = [float(x) for x in sc["extra_lse"]]
    M = float(np.sum(np.exp(logq)))
    lp = sc["lp_last"]
    nt = {c: float(lp[sorted({s[0] for s in g})].exp().sum()) for c, g in zip(COLOURS, groups)}
    return {"m": sc["m"], "lse_a": sc["lse_a"], "lse_b": sc["lse_b"], "logq": logq, "M": M, "outside": 1.0 - M,
            "next_token_mass": nt}


class PairRunner:
    """All 004 conditions for one pair under SEC. Recipient = member, donor = partner (003's direction p -> m)."""

    def __init__(self, sec, tok, e, F, groups):
        import margin as MG
        import torch
        self.sec, self.tok, self.e, self.F, self.groups = sec, tok, e, F, groups
        self.torch, self.dev = torch, sec.model.device
        pair = e["pair"]
        self.pair, self.rx, self.ry = pair, e["r_m"], e["r_p"]
        self.Sa, self.Sb = MG.spellings(tok, "brew", pair.m.a), MG.spellings(tok, "brew", pair.p.a)
        self.pos = {s: cell_pos(self.rx, SITES[s][0]) for s in SITES}
        assert self.pos == {s: cell_pos(self.ry, SITES[s][0]) for s in SITES}, "member / partner positions differ"
        self.conts = [()] + [s for g in groups for s in g if len(s) > 1]
        layers = sorted({L for _, L in SITES.values()})
        self.own, self.src = {}, {}
        for c in self.conts:                   # sources at the matching length for every continuation forward
            a = sec.forward(self.rx.ids + list(c), record=layers)[1]
            self.own[c] = {s: a[L][self.pos[s]].clone() for s, (_, L) in SITES.items()}
            a = sec.forward(self.ry.ids + list(c), record=layers)[1]
            self.src[c] = {s: a[L][self.pos[s]].clone() for s, (_, L) in SITES.items()}
        self.fold, self.bases = F.pair_bases(pair)
        self.clean = self.score(self.rx.ids)
        self.donor = self.score(self.ry.ids)
        self.cs, self.ds = summarize(self.clean, groups), summarize(self.donor, groups)
        self.G = self.donor["m"] - self.clean["m"]
        self.base = {"pair_id": pair.pair_id, "item": pair.item, "fold": self.fold, "G": self.G, "m_clean": self.clean["m"],
                     "m_donor": self.donor["m"], "a": pair.m.a, "a2": pair.p.a, "z": pair.m.z, "z2": pair.p.z,
                     "start": pair.m.inp, "start2": pair.p.inp}

    def t(self, a):
        return self.torch.tensor(np.ascontiguousarray(a), dtype=self.torch.float32, device=self.dev)

    def score(self, ids, edit_fn=None):
        return self.sec.score(ids, self.Sa, self.Sb, edit_fn=edit_fn, extra_spellings=self.groups)

    def mats(self, c):
        b = self.bases[c["fit"]][c["site"]] if c["fit"] else None
        M = {}
        if c["op"] in ("span", "comp") and c["draw"] is not None:
            M["B"] = self.t(draw_matrix(c))
        elif c["basis"] is not None:
            M["B"] = self.t(b[c["basis"]])
        if c["op"] in ("normdir", "pca_dose"):
            M["Bref"] = self.t(b[c["ref"]])
        if c["op"] == "normdir":
            M["r"] = self.t(draw_matrix(c))
        if c["op"] in ("rot", "rclamp"):
            M["U"] = self.t(draw_matrix(c))
        mu = self.t(b["mu"]) if c["op"] in ("nec", "rot", "rclamp") else None
        return M, mu

    def edit_fn(self, c, M, mu, run, fid, donor=None, identity=False):
        """edit_fn(cont) for sec.score: one write at (site, layer) on the `run` prompt ('m' or 'p'), rebuilt per continuation
        with that continuation's donor source; donor overrides the source (self-swap checks)."""
        torch = self.torch
        site = c["site"]
        L, pos = SITES[site][1], self.pos[site]

        def fn_for(cont):
            hd = (donor or self.src)[cont][site]

            def f(x):
                with torch.autocast("cuda", enabled=False):
                    if c["op"] == "full" and not identity:
                        x, req, real, h = apply_write(x, pos, target=hd)
                    elif identity:
                        x, req, real, h = apply_write(x, pos, d=M["I"] @ (M["I"].T @ (hd - x[pos])))
                    else:
                        x, req, real, h = apply_write(x, pos, d=make_delta(c["op"], M, x[pos].clone(), hd, mu))
                    fid.add(req, real, h)
                return x
            return {L: f}, []
        return fn_for

    def pca_dose_defined(self, c, M):
        """Dose matching is undefined when the PCA-projected Delta is numerically zero at any continuation."""
        for cont in self.conts:
            D = self.src[cont][c["site"]] - self.own[cont][c["site"]]
            p = M["B"] @ (M["B"].T @ D)
            if float(p.norm()) < NEG_REL * float(self.own[cont][c["site"]].norm()):
                return False
        return True

    def evaluate(self, c, donor=None, identity=False, extra=None):
        t0 = time.time()
        row = {**self.base, **{k: c[k] for k in ("kind", "site", "fit", "arm", "op", "draw", "seed", "k")}}
        if extra:
            row.update(extra)
        if c["k"] is None:
            row["status"] = "no direction"
            return row
        M, mu = self.mats(c) if c["op"] != "full" else ({}, None)
        if identity:
            M["I"] = self.torch.eye(5120, dtype=self.torch.float32, device=self.dev)
        if c["op"] == "pca_dose" and not self.pca_dose_defined(c, M):
            row["status"] = "dose undefined (PCA-projected Delta ~ 0)"
            return row
        L, pos = SITES[c["site"]][1], self.pos[c["site"]]
        if c["prompts"] == 1:
            fid = Fidelity()
            ed = self.score(self.rx.ids, edit_fn=self.edit_fn(c, M, mu, "m", fid, donor=donor, identity=identity))
            es = summarize(ed, self.groups)
            import regime as RG
            row.update({"dm": ed["m"] - self.clean["m"], "S": (ed["m"] - self.clean["m"]) / self.G, "edit": es,
                        "d_logP_a": es["lse_a"] - self.cs["lse_a"], "d_logP_b": es["lse_b"] - self.cs["lse_b"],
                        "outside_up": es["outside"] - self.cs["outside"],
                        **{k: v for k, v in RG.sec_damage(self.sec, self.clean, ed, pos, len(self.rx)).items()
                           if k in ("top1_retention", "n_retention_positions")},
                        **fid.summary(), "fid_ok": fid.ok()})
            row["flag_outside"] = row["outside_up"] > FLAG_OUTSIDE
            row["flag_both_down"] = row["d_logP_a"] < -FLAG_BOTH_DOWN and row["d_logP_b"] < -FLAG_BOTH_DOWN
        else:
            out = {}
            ok = True
            for run, r_, ref in (("m", self.rx, self.cs), ("p", self.ry, self.ds)):
                fid = Fidelity()
                ed = self.score(r_.ids, edit_fn=self.edit_fn(c, M, mu, run, fid))
                es = summarize(ed, self.groups)
                out[run] = {**es, "outside_up": es["outside"] - ref["outside"], "d_lse_a": es["lse_a"] - ref["lse_a"],
                            "d_lse_b": es["lse_b"] - ref["lse_b"], **fid.summary()}
                out[run]["flag_outside"] = out[run]["outside_up"] > FLAG_OUTSIDE
                ok = ok and fid.ok()
            G_abl = out["p"]["m"] - out["m"]["m"]
            row.update({"G_abl": G_abl, "N": 1 - G_abl / self.G, "member_run": out["m"], "partner_run": out["p"], "fid_ok": ok})
        row["seconds"] = time.time() - t0
        return row

    def run_all(self, conds):
        rows = [{**self.base, "kind": "clean", "member": self.cs, "partner": self.ds}]
        for c in conds:
            rows.append(self.evaluate(c))
        return rows


# ------------------------------------------------------------------------------------------- preflight
def stage_preflight():
    import torch
    tok = R.make_tokenizer()
    entries = BM.gated_dev_brew(BM.brew_pairs(tok))
    man = pair_manifest(entries)
    write_json(RES / "manifest_pairs.json", man)
    acts, rows = load_acts()
    F = Fits(acts, rows)
    by_id = {e["pair"].pair_id: e for e in entries}
    key_idx = {r["key"]: i for i, r in enumerate(rows)}
    pre = {"pairs": [], "fits": {}}

    # 1. eligibility and support per fit; the fit rows reproduce 003's n and fitting mean
    for fold in range(3):
        for fit in FITS:
            idx = F.fit_rows(fold, fit)
            for site, (cell, L) in SITES.items():
                k = F.key(fold, fit, cell, L)
                inf = F.info["info"][k]
                mean_err = float(np.abs(acts[cell][idx, LAYERS_CAP.index(L)].astype(np.float64).mean(0) - F.fitmean(fold, fit, cell, L)).max())
                tables = {role: {c: len({rows[i]["item"] for i in idx if rows[i][role] == c}) for c in COLOURS} for role in ROLES}
                F.pca(fold, fit, cell, L, 1)
                sv = F._pca[(fold, fit, cell, L)][1]
                pre["fits"][f"f{fold}|{fit}|{site}"] = {
                    "n_renders": inf["n"], "n_rows_recomputed": len(idx), "n_tables": len({rows[i]["item"] for i in idx}),
                    "fitmean_max_abs_err_vs_003": mean_err, "rank": inf["rank"], "render_counts": inf["counts"],
                    "table_support": tables, "eligible": {role: F.eligible(fold, fit, cell, L, role) for role in ROLES},
                    "pca_var_share_top11": float((sv[:11] ** 2).sum() / (sv ** 2).sum()),
                    "pca_var_share_top33": float((sv[:33] ** 2).sum() / (sv ** 2).sum())}
                assert len(idx) == inf["n"], (k, len(idx), inf["n"])

    # 2. basis identities (float64 and float32), 1-D containment, span + complement = Delta; geometry to PCA
    err = {k: 0.0 for k in ("QtQ64", "QtQ32", "idem64", "idem32", "recon64", "recon32", "contain64", "contain32")}
    for pid in man["pairs"]:
        e = by_id[pid]
        fold, bases = F.pair_bases(e["pair"])
        im, ip = key_idx[BM.ids_key(e["r_m"].ids)], key_idx[BM.ids_key(e["r_p"].ids)]
        rec = {"pair_id": pid, "fold": fold, "item": e["pair"].item}
        for site, (cell, L) in SITES.items():
            d64 = (acts[cell][ip, LAYERS_CAP.index(L)] - acts[cell][im, LAYERS_CAP.index(L)]).astype(np.float64)
            d32 = d64.astype(np.float32)
            for fit in FITS:
                b = bases[fit][site]
                info = {}
                for arm, B in b.items():
                    if arm == "mu" or B is None:
                        continue
                    I = np.eye(B.shape[1])
                    B32 = B.astype(np.float32)
                    err["QtQ64"] = max(err["QtQ64"], float(np.abs(B.T @ B - I).max()))
                    err["QtQ32"] = max(err["QtQ32"], float(np.abs(B32.T @ B32 - I).max()))
                    x = np.random.default_rng(0).standard_normal(5120)
                    x32 = x.astype(np.float32)
                    err["idem64"] = max(err["idem64"], float(np.abs(B @ (B.T @ (B @ (B.T @ x))) - B @ (B.T @ x)).max() / np.abs(x).max()))
                    err["idem32"] = max(err["idem32"], float(np.abs(B32 @ (B32.T @ (B32 @ (B32.T @ x32))) - B32 @ (B32.T @ x32)).max() / np.abs(x32).max()))
                    sp = B @ (B.T @ d64)
                    err["recon64"] = max(err["recon64"], float(np.abs(sp + (d64 - sp) - d64).max() / np.abs(d64).max()))
                    sp32 = B32 @ (B32.T @ d32)
                    err["recon32"] = max(err["recon32"], float(np.abs(sp32 + (d32 - sp32) - d32).max() / np.abs(d32).max()))
                    info[arm] = {"k": int(B.shape[1]), "proj_norm": float(np.linalg.norm(B.T @ d64)), "proj_share_sq": float(np.linalg.norm(B.T @ d64) ** 2 / np.linalg.norm(d64) ** 2)}
                one = "z1d" if site == "colon" else "s1d"
                span = "Z" if site == "colon" else "Sspan"
                if b[one] is not None:
                    u, Q = b[one][:, 0], b[span]
                    err["contain64"] = max(err["contain64"], float(np.linalg.norm(u - Q @ (Q.T @ u))))
                    u32, Q32 = u.astype(np.float32), Q.astype(np.float32)
                    err["contain32"] = max(err["contain32"], float(np.linalg.norm(u32 - Q32 @ (Q32.T @ u32))))
                pk = "pca_kZ" if site == "colon" else "pca_kS"
                cosang = np.linalg.svd(b[span].T @ b[pk], compute_uv=False)
                info["principal_cos_span_vs_pca"] = [float(v) for v in cosang]
                info["delta_norm"] = float(np.linalg.norm(d64))
                rec[f"{site}|{fit}"] = info
        pre["pairs"].append(rec)
    tol = {k: float(max(10 * v, np.finfo(np.float64 if k.endswith("64") else np.float32).eps)) for k, v in err.items()}
    pre["identity_max_errors"], pre["identity_tolerances"] = err, tol

    # 3. zero and negligible writes (the hook's own write function, float32 on CPU, real h at ':' L42)
    H = torch.tensor(acts["emit_colon"][:32, LAYERS_CAP.index(42)], dtype=torch.float32)
    zw = {"zero_write_bitwise_noop": True, "by_scale": {}}
    for i in range(H.shape[0]):
        x = H[i:i + 1].clone()
        y, *_ = apply_write(x, 0, d=torch.zeros(5120))
        zw["zero_write_bitwise_noop"] &= bool(torch.equal(x, y))
    for scale in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        rhos, kaps = [], []
        for i in range(H.shape[0]):
            x = H[i:i + 1].clone()
            g = torch.tensor(np.random.default_rng(i).standard_normal(5120), dtype=torch.float32)
            d = g / g.norm() * scale * x[0].norm()
            _, req, real, h = apply_write(x, 0, d=d)
            rhos.append(float(real.norm() / req.norm()))
            kaps.append(float(torch.nn.functional.cosine_similarity(req[None], real[None])[0]))
        zw["by_scale"][str(scale)] = {"rho_range": [min(rhos), max(rhos)], "kappa_min": min(kaps)}
    ok_scales = [float(s) for s, v in zw["by_scale"].items() if abs(v["rho_range"][0] - 1) < 0.01 and abs(v["rho_range"][1] - 1) < 0.01 and v["kappa_min"] > 0.999]
    zw["NEG_REL"] = NEG_REL
    zw["smallest_clean_scale"] = min(ok_scales)
    zw["neg_rel_is_safe"] = bool(min(ok_scales) <= NEG_REL)
    pre["zero_writes"] = zw

    # 4. dependency hygiene (the boundary test checks the static closure; this checks what actually loaded)
    pre["reader_modules_loaded"] = sorted(m for m in sys.modules if m.split(".")[0] in ("readers", "jlens", "peft", "vllm", "vllm_lens", "leverage_profile", "labels"))

    # 5. arm manifest
    groups = spelling_groups(tok)
    n_multi = sum(len(s) > 1 for g in groups for s in g)
    flat = [s for g in groups for s in g]
    prefix_conflicts = [(a, b) for a in flat for b in flat if a != b and len(a) <= len(b) and b[:len(a)] == a]
    per_pair, tot = [], {}
    for pid in man["pairs"]:
        _, bases = F.pair_bases(by_id[pid]["pair"])
        C = conditions(pid, bases)
        cnt = {}
        for c in C:
            k = f"{c['kind']}|{c['site']}|{c['fit']}|{c['arm'].split(':')[0] if ':' in c['arm'] else c['arm']}"
            cnt[k] = cnt.get(k, 0) + 1
        prompts = 2 + sum(c["prompts"] for c in C if c["k"] is not None)
        per_pair.append({"pair_id": pid, "conditions": len(C), "scored_prompts": prompts,
                         "forwards": prompts * (1 + n_multi) + 2 * (1 + n_multi)})
        for k, v in cnt.items():
            tot[k] = tot.get(k, 0) + v
    arm_manifest = {"design": "docs/design_specs/004_span_assay.md v2", "n_pairs": len(per_pair), "n_draws": N_DRAWS,
                    "spelling_events": {c: [list(s) for s in g] for c, g in zip(COLOURS, groups)},
                    "multi_token_spellings": n_multi, "prefix_conflicts": prefix_conflicts,
                    "forwards_note": "per scored prompt: 1 + one forward per multi-token spelling; per pair: + capture forwards (member and partner, each continuation)",
                    "per_pair": per_pair, "conditions_by_kind_site_fit_arm": tot,
                    "totals": {k: int(sum(p[k] for p in per_pair)) for k in ("conditions", "scored_prompts", "forwards")}}
    write_json(RES / "arm_manifest.json", arm_manifest)

    pre["passes"] = bool(all(err[k] <= tol[k] for k in err) and err["QtQ64"] < 1e-10 and err["QtQ32"] < 1e-5
                         and err["contain64"] < 1e-8 and zw["zero_write_bitwise_noop"] and zw["neg_rel_is_safe"]
                         and not pre["reader_modules_loaded"] and not prefix_conflicts
                         and all(v["n_renders"] == v["n_rows_recomputed"] and v["fitmean_max_abs_err_vs_003"] < 1e-3
                                 for v in pre["fits"].values()))
    write_json(RES / "preflight.json", pre)
    write_manifest("preflight", {"passes": pre["passes"]})
    ks = {f"{s}|{a}|{f}": sorted({p[f"{s}|{f}"][a]["k"] for p in pre["pairs"]}) for s, a in (("colon", "Z"), ("colon", "J"), ("start_ref", "Sspan")) for f in FITS}
    print(json.dumps({"n_pairs": len(man["pairs"]), "ranks": ks, "identity_max_errors": err,
                      "zero_writes": {k: v for k, v in zw.items() if k != "by_scale"},
                      "fitmean_max_err": max(v["fitmean_max_abs_err_vs_003"] for v in pre["fits"].values()),
                      "arm_totals": arm_manifest["totals"], "reader_modules": pre["reader_modules_loaded"], "passes": pre["passes"]}, indent=1))


# ------------------------------------------------------------------------------------------- GPU stages
def _jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def _setup():
    import regime as RG
    import torch
    assert not torch.backends.cuda.matmul.allow_tf32 and torch.get_float32_matmul_precision() == "highest", "TF32 must be off"
    tok = R.make_tokenizer()
    ents, man = material(tok)
    acts, rows = load_acts()
    F = Fits(acts, rows)
    groups = spelling_groups(tok)
    model = R.load_model()
    return RG, torch, tok, ents, F, groups, model


def stage_smoke():
    """G1 fidelity, G2 (self-swap exact, identity projector vs direct assignment, S_full and clean margin vs 001 / 003),
    timing, and a hand-inspection table; 2 pairs end to end (the first two of the frozen manifest)."""
    RG, torch, tok, ents, F, groups, model = _setup()
    d = OUT / "smoke"
    d.mkdir(parents=True, exist_ok=True)
    if (d / "rows.jsonl").exists():          # a rerun archives the earlier pass; nothing scored is deleted
        (d / "rows.jsonl").rename(d / f"rows_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.jsonl")
    prof = {(r["pair_id"], r["pos"], r["layer"]): r["S"] for r in _jl(CAP001 / "profile_brew.jsonl") if r["direction"] == "p->m"}
    e003 = {}
    for r in _jl(D003 / "run" / "edits.jsonl"):
        e003.setdefault(r["pair_id"], (r["m_clean"], r["G"]))
    checks, all_rows = {"pairs": []}, []
    with RG.SecEngine(model, l0=39) as sec:
        for e in ents[:2]:
            t0 = time.time()
            pr = PairRunner(sec, tok, e, F, groups)
            t_setup = time.time() - t0
            C = conditions(e["pair"].pair_id, pr.bases)
            rows = pr.run_all(C)
            t_all = time.time() - t0
            g = {"pair_id": e["pair"].pair_id, "seconds_total": t_all, "seconds_setup": t_setup,
                 "scored_prompts": 2 + sum(c["prompts"] for c in C if c["k"] is not None)}
            # G2a self-swap through the span, complement and whole-cell paths: bitwise equal to clean
            self_rows = [pr.evaluate(c, donor=pr.own, extra={"check": "self_swap"}) for c in C
                         if c["kind"] == "swap" and c["fit"] in (None, "both") and c["arm"] in ("full", "Z", "Jc", "Sspan")]
            g["self_swap_dm"] = {f"{r['site']}|{r['arm']}": r["dm"] for r in self_rows}
            g["self_swap_exact"] = all(r["dm"] == 0.0 and r["edit"]["logq"] == pr.cs["logq"] for r in self_rows)
            # G2b whole cell through the identity projector vs direct assignment
            full = {r["site"]: r for r in rows if r.get("kind") == "swap" and r.get("arm") == "full"}
            ident = [pr.evaluate(c, identity=True, extra={"check": "identity_projector"}) for c in C if c["arm"] == "full"]
            g["identity_vs_direct_abs_dm"] = {r["site"]: abs(r["dm"] - full[r["site"]]["dm"]) for r in ident}
            g["identity_ok"] = all(v <= FLOOR for v in g["identity_vs_direct_abs_dm"].values())
            # G2c S_full vs 001's whole-cell profile, clean margin and G vs 003
            g["S_full"] = {s: full[s]["S"] for s in SITES}
            g["S_full_001"] = {s: prof.get((e["pair"].pair_id, pr.pos[s], SITES[s][1])) for s in SITES}
            g["S_full_abs_diff"] = {s: None if g["S_full_001"][s] is None else abs(g["S_full"][s] - g["S_full_001"][s]) for s in SITES}
            m3, G3 = e003.get(e["pair"].pair_id, (None, None))
            g["m_clean_vs_003"] = None if m3 is None else pr.clean["m"] - m3
            g["G_vs_003"] = None if G3 is None else pr.G - G3
            g["S_full_ok"] = all(v is not None and v <= 1e-3 for v in g["S_full_abs_diff"].values())
            g["clean_ok"] = g["m_clean_vs_003"] is not None and abs(g["m_clean_vs_003"]) <= 1e-6 and abs(g["G_vs_003"]) <= 1e-6
            # G1 fidelity on every write; mass bookkeeping
            scored = [r for r in rows if r.get("kind") not in ("clean",) and "status" not in r]
            g["n_conditions"], g["n_scored"] = len(rows) - 1, len(scored)
            g["status_counts"] = {}
            for r in rows:
                if "status" in r:
                    g["status_counts"][r["status"]] = g["status_counts"].get(r["status"], 0) + 1
            g["fidelity_ok"] = all(r["fid_ok"] for r in scored)
            g["fid_negligible_writes"] = int(sum(r.get("fid_negligible", 0) or 0 for r in scored)
                                             + sum((r[k].get("fid_negligible", 0) or 0) for r in scored if "member_run" in r for k in ("member_run", "partner_run")))
            Ms = [pr.cs["M"], pr.ds["M"]] + [r["edit"]["M"] for r in scored if "edit" in r] + \
                 [r[k]["M"] for r in scored if "member_run" in r for k in ("member_run", "partner_run")]
            g["M_max"], g["M_min"] = max(Ms), min(Ms)
            g["mass_ok"] = max(Ms) <= 1 + 1e-6
            g["n_flag_outside"] = int(sum(bool(r.get("flag_outside")) for r in scored))
            g["n_flag_both_down"] = int(sum(bool(r.get("flag_both_down")) for r in scored))
            checks["pairs"].append(g)
            with open(d / "rows.jsonl", "a") as f:
                for r in rows + self_rows + ident:
                    f.write(json.dumps(r, default=float) + "\n")
            all_rows += rows
            sec.clear()
            print(json.dumps({k: v for k, v in g.items() if k not in ("self_swap_dm",)}, default=float))
    keys = ("self_swap_exact", "identity_ok", "S_full_ok", "clean_ok", "fidelity_ok", "mass_ok")
    checks["gates"] = {k: all(p[k] for p in checks["pairs"]) for k in keys}
    checks["passes"] = all(checks["gates"].values())
    checks["seconds_per_pair"] = [p["seconds_total"] for p in checks["pairs"]]
    checks["projected_run_minutes"] = float(np.mean(checks["seconds_per_pair"]) * len(json.loads((RES / "manifest_pairs.json").read_text())["pairs"]) / 60)
    write_json(RES / "smoke_checks.json", checks)
    write_manifest("smoke", {"passes": checks["passes"], "gates": checks["gates"]})
    print(json.dumps({"gates": checks["gates"], "passes": checks["passes"], "seconds_per_pair": checks["seconds_per_pair"],
                      "projected_run_minutes": checks["projected_run_minutes"]}))


def composition(row, clean_logq, a2):
    """d_c = q_edit(c) - q_clean(c); T+ = sum max(d_c, 0); F_target = max(d_a', 0) / T+ (defined for T+ >= 0.01)."""
    d = {c: float(np.exp(q1) - np.exp(q0)) for c, q1, q0 in zip(COLOURS, row["edit"]["logq"], clean_logq)}
    tp = sum(max(v, 0.0) for v in d.values())
    return d, tp, (max(d[a2], 0.0) / tp if tp >= 0.01 else None)


def stage_inspect(src="smoke"):
    """Hand-inspection table for the smoke (or any pass): colour distributions, composition, flags, fidelity, controls."""
    text = inspect_text(_jl(OUT / src / "rows.jsonl"))
    (RES / f"{src}_inspection.md").write_text(text)
    print(text)


def inspect_text(rows, only=None):
    lines = []
    fmt = lambda x: "  n/a" if x is None else f"{x:+.3f}"
    for pid in (only or dict.fromkeys(r["pair_id"] for r in rows)):
        R_ = [r for r in rows if r["pair_id"] == pid and "check" not in r]
        cl = R_[0]
        top = lambda lq: ", ".join(f"{c} {np.exp(q):.3f}" for c, q in sorted(zip(COLOURS, lq), key=lambda t: -t[1])[:4])
        lines += [f"## {pid}  (a {cl['a']} -> a' {cl['a2']}; z {cl['z']} -> {cl['z2']}; start {cl['start']} -> {cl['start2']})",
                  f"G = {cl['G']:.3f}   m_clean {cl['m_clean']:.3f}   m_donor {cl['m_donor']:.3f}",
                  f"clean q: {top(cl['member']['logq'])}   outside {cl['member']['outside']:.3f}",
                  f"donor q: {top(cl['partner']['logq'])}   outside {cl['partner']['outside']:.3f}", "",
                  "| site | fit | arm | k | S | dlogP(a) | dlogP(a') | outside up | T+ | F_target | top gain | |write| | rho | flags |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in R_:
            if r["kind"] not in ("swap", "pca") or "status" in r:
                if r["kind"] in ("swap", "pca") and "status" in r:
                    lines.append(f"| {r['site']} | {r['fit']} | {r['arm']} | {r['k']} | {r['status']} |")
                continue
            d, tp, ft = composition(r, cl["member"]["logq"], cl["a2"])
            g = max(d, key=d.get)
            fl = ",".join(k for k in ("flag_outside", "flag_both_down") if r[k]) or "-"
            rho = "neg" if r["fid_rho"] is None else f"{r['fid_rho'][0]:.4f}"
            lines.append(f"| {r['site']} | {r['fit']} | {r['arm']} | {r['k']} | {r['S']:+.3f} | {r['d_logP_a']:+.3f} | {r['d_logP_b']:+.3f} | "
                         f"{r['outside_up']:+.3f} | {tp:.3f} | {fmt(ft)} | {g} {d[g]:+.3f} | {r['write_norm']:.2f} | {rho} | {fl} |")
        lines += ["", "| site | arm (primary) | real S | rank-matched (Jc: random complement) median / q95 | norm-matched median / q95 | rank stat vs rank-matched (Jc: lower tail) |",
                  "|---|---|---|---|---|---|"]
        for site, arm in CTRL_ARMS + [("colon", "Jc")]:
            real = [r for r in R_ if r["kind"] == "swap" and r["fit"] == "both" and r["site"] == site and r["arm"] == arm and "S" in r]
            if not real:
                continue
            ck = "rcomp:J" if arm == "Jc" else f"rank:{arm}"
            rk = [r["S"] for r in R_ if r["kind"] == "swap_ctrl" and r["arm"] == ck and "S" in r]
            nm = [r["S"] for r in R_ if r["kind"] == "swap_ctrl" and r["arm"] == f"norm:{arm}" and "S" in r]
            s = real[0]["S"]
            stat = ((1 + sum(x <= s for x in rk)) if arm == "Jc" else (1 + sum(x >= s for x in rk))) / (1 + len(rk)) if rk else None
            q = lambda v: f"{np.median(v):+.3f} / {np.quantile(v, 0.95):+.3f}" if v else "-"
            lines.append(f"| {site} | {arm} | {s:+.3f} | {q(rk)} | {q(nm)} | {stat if stat is None else f'{stat:.3f}'} |")
        lines += ["", "| site | arm | fit | N | G_abl | member outside up | partner outside up | rotation N median / q95 | rand-clamp N median / q95 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in R_:
            if r["kind"] != "nec" or "N" not in r:
                continue
            ctl = lambda kind: [x["N"] for x in R_ if x["kind"] == "nec_ctrl" and x["arm"] == f"{kind}:{r['arm']}" and "N" in x]
            q = lambda v: f"{np.median(v):+.3f} / {np.quantile(v, 0.95):+.3f}" if v else "-"
            lines.append(f"| {r['site']} | {r['arm']} | {r['fit']} | {r['N']:+.3f} | {r['G_abl']:.3f} | {r['member_run']['outside_up']:+.3f} | "
                         f"{r['partner_run']['outside_up']:+.3f} | {q(ctl('rot')) if r['fit'] == 'both' else ''} | {q(ctl('rclamp')) if r['fit'] == 'both' else ''} |")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------- analysis (design §4-§7)
REC_MIN = 0.10          # recovery subset: remeasured S_full >= 0.10 at that site
TOL_MATERIAL = 0.05     # r1 / r2 material-disagreement tolerance (share units)
I_BIG = 0.15            # |I| at or above this counts as a large interaction
PAIRED = {"colon": [("Z", "z1d"), ("J", "Z"), ("Z", "pca_kZ"), ("Z", "pca_kZ_dose"), ("J", "pca_kJ"), ("J", "pca_kJ_dose")],
          "start_ref": [("Sspan", "s1d"), ("Sspan", "pca_kS"), ("Sspan", "pca_kS_dose")]}
INTERACT = [("colon", "Z", "Zc"), ("colon", "J", "Jc"), ("start_ref", "Sspan", "Sc")]


def _med_ci(vals, tables, stat=np.median):
    import stats as ST
    v = np.asarray(vals, float)
    if len(v) == 0:
        return None
    if len(v) == 1:
        return {"value": float(stat(v)), "boot95_tables": [None, None], "n": 1}
    b = ST.cluster_bootstrap(v, np.asarray(tables), stat=stat)
    return {"value": float(stat(v)), "boot95_tables": b["ci"], "n": int(len(v)), "n_pos": int((v > 0).sum())}


def roles_of(cl):
    """Role labels per colour for the composition breakdown; a colour holding several roles is labelled with all of them
    and counted once, under the first of a' > z' > start'."""
    lab = {}
    for name, key in (("a'", "a2"), ("z'", "z2"), ("start'", "start2"), ("a", "a"), ("z", "z"), ("start", "start")):
        lab.setdefault(cl[key], []).append(name)
    return lab


def swap_summary(r, cl):
    d, tp, ft = composition(r, cl["member"]["logq"], cl["a2"])
    lab = roles_of(cl)
    gains, used = {}, set()
    for role, key in (("a'", "a2"), ("z'", "z2"), ("start'", "start2")):
        c = cl[key]
        gains[role] = None if c in used else max(d[c], 0.0)
        used.add(c)
    gains["other"] = tp - sum(v for v in gains.values() if v)
    for role, key in (("of_other: recipient z", "z"), ("of_other: recipient start", "start")):   # descriptive sub-split
        c = cl[key]
        gains[role] = None if c in used else max(d[c], 0.0)
        used.add(c)
    flagged = bool(r["flag_outside"] or r["flag_both_down"])
    return {"S": r["S"], "T_plus": tp, "F_target": ft, "gains": gains, "role_labels": {c: lab[c] for c in lab},
            "flagged": flagged, "flag_outside": r["flag_outside"], "flag_both_down": r["flag_both_down"],
            "qualified": bool(ft is not None and ft > 0.5 and r["d_logP_b"] > FLOOR and not flagged),
            "d_logP_a": r["d_logP_a"], "d_logP_b": r["d_logP_b"], "outside_up": r["outside_up"], "write_norm": r["write_norm"]}


def per_pair_tables(rows):
    out = {}
    for pid in dict.fromkeys(r["pair_id"] for r in rows):
        R_ = [r for r in rows if r["pair_id"] == pid and "check" not in r]
        cl = next(r for r in R_ if r["kind"] == "clean")
        p = {"item": cl["item"], "G": cl["G"], "a": cl["a"], "a2": cl["a2"], "swap": {}, "status": {}, "nec": {},
             "ctrl": {}, "nec_ctrl": {}}
        for r in R_:
            key = f"{r.get('site')}|{r.get('fit')}|{r.get('arm')}"
            if "status" in r:
                p["status"][f"{r['kind']}|{key}"] = r["status"]
            elif r["kind"] in ("swap", "pca"):
                p["swap"][key] = swap_summary(r, cl)
            elif r["kind"] == "nec":
                p["nec"][key] = {"N": r["N"], "G_abl": r["G_abl"], "flag_member": r["member_run"]["flag_outside"],
                                 "flag_partner": r["partner_run"]["flag_outside"]}
            elif r["kind"] == "swap_ctrl":
                p["ctrl"].setdefault(f"{r['site']}|{r['arm']}", []).append(r["S"])
            elif r["kind"] == "nec_ctrl":
                p["nec_ctrl"].setdefault(f"{r['site']}|{r['arm']}", []).append(r["N"])
        p["S_full"] = {s: p["swap"][f"{s}|None|full"]["S"] for s in SITES}
        out[pid] = p
    return out


def _S(p, site, fit, arm):
    v = p["swap"].get(f"{site}|{fit}|{arm}")
    return None if v is None else v["S"]


def stage_analysis(src="run"):
    rows = [r for r in _jl(OUT / src / "rows.jsonl") if "check" not in r]
    P = per_pair_tables(rows)
    pids = list(P)
    tab = {pid: P[pid]["item"] for pid in pids}
    rec_sub = {s: [pid for pid in pids if P[pid]["S_full"][s] >= REC_MIN] for s in SITES}
    A = {"n_pairs": len(pids), "recovery_subset": {s: {"n": len(v), "pairs": v, "rule": f"S_full >= {REC_MIN}"} for s, v in rec_sub.items()},
         "S_full": {s: _med_ci([P[p]["S_full"][s] for p in pids], [tab[p] for p in pids]) for s in SITES},
         "arms": {}, "paired": {}, "controls": {}, "necessity": {}, "necessity_controls": {}, "reproducibility": {},
         "interaction": {}, "status_counts": {}}
    for p in pids:
        for k, v in P[p]["status"].items():
            A["status_counts"][f"{k}|{v}"] = A["status_counts"].get(f"{k}|{v}", 0) + 1
    arm_keys = sorted({k for p in pids for k in P[p]["swap"]})
    for k in arm_keys:
        site, fit, arm = k.split("|")
        have = [p for p in pids if k in P[p]["swap"]]
        sw = [P[p]["swap"][k] for p in have]
        rs = [p for p in rec_sub[site] if k in P[p]["swap"]]
        A["arms"][k] = {
            "S_all": _med_ci([s["S"] for s in sw], [tab[p] for p in have]),
            "n_missing": len(pids) - len(have),
            "recovery": _med_ci([P[p]["swap"][k]["S"] / P[p]["S_full"][site] for p in rs], [tab[p] for p in rs]),
            "n_flagged": int(sum(s["flagged"] for s in sw)), "n_flag_outside": int(sum(s["flag_outside"] for s in sw)),
            "n_flag_both_down": int(sum(s["flag_both_down"] for s in sw)), "n_qualified": int(sum(s["qualified"] for s in sw)),
            "F_target_median": float(np.median([s["F_target"] for s in sw if s["F_target"] is not None])) if any(s["F_target"] is not None for s in sw) else None,
            "n_F_defined": int(sum(s["F_target"] is not None for s in sw)),
            "S_qualified_only": _med_ci([s["S"] for p, s in zip(have, sw) if s["qualified"]], [tab[p] for p, s in zip(have, sw) if s["qualified"]]),
            "S_unflagged_only": _med_ci([s["S"] for p, s in zip(have, sw) if not s["flagged"]], [tab[p] for p, s in zip(have, sw) if not s["flagged"]])}
    for site, comps in PAIRED.items():
        for fit in FITS:
            for a1, a2 in comps:
                have = [p for p in pids if _S(P[p], site, fit, a1) is not None and _S(P[p], site, fit, a2) is not None]
                rs = [p for p in have if p in rec_sub[site]]
                A["paired"][f"{site}|{fit}|{a1}-{a2}"] = {
                    "dS_all": _med_ci([_S(P[p], site, fit, a1) - _S(P[p], site, fit, a2) for p in have], [tab[p] for p in have]),
                    "dS_recovery_subset": _med_ci([_S(P[p], site, fit, a1) - _S(P[p], site, fit, a2) for p in rs], [tab[p] for p in rs]),
                    "d_recovery": _med_ci([(_S(P[p], site, fit, a1) - _S(P[p], site, fit, a2)) / P[p]["S_full"][site] for p in rs], [tab[p] for p in rs])}
    # swap controls (primary fit): descriptive exceedance counts and rank statistics, no per-item significance
    for site, arm in CTRL_ARMS + [("colon", "Jc")]:
        for ck in (["rcomp:J"] if arm == "Jc" else [f"rank:{arm}", f"norm:{arm}"]):
            per = []
            for p in pids:
                s = _S(P[p], site, "both", arm)
                c = P[p]["ctrl"].get(f"{site}|{ck}")
                if s is None or not c:
                    continue
                c = np.asarray(c)
                per.append({"pair_id": p, "S": s, "ctrl_median": float(np.median(c)), "ctrl_q05": float(np.quantile(c, 0.05)),
                            "ctrl_q95": float(np.quantile(c, 0.95)), "exceedances": int((c >= s).sum()),
                            "rank_stat": float((1 + (c >= s).sum()) / (1 + len(c))),
                            "rank_stat_lower": float((1 + (c <= s).sum()) / (1 + len(c)))})
            A["controls"][f"{site}|{arm}|{ck}"] = {
                "n": len(per), "n_draws": N_DRAWS, "min_rank_stat": 1 / (1 + N_DRAWS),
                "tail_note": "complements: the informative tail is the lower one (real complement below random complements)",
                "frac_real_above_q95": float(np.mean([x["S"] > x["ctrl_q95"] for x in per])) if per else None,
                "frac_real_below_q05": float(np.mean([x["S"] < x["ctrl_q05"] for x in per])) if per else None,
                "frac_rank_stat_at_min": float(np.mean([x["exceedances"] == 0 for x in per])) if per else None,
                "median_rank_stat_lower": float(np.median([x["rank_stat_lower"] for x in per])) if per else None,
                "median_rank_stat": float(np.median([x["rank_stat"] for x in per])) if per else None,
                "ctrl_median_of_medians": float(np.median([x["ctrl_median"] for x in per])) if per else None, "per_pair": per}
    # necessity
    nec_keys = sorted({k for p in pids for k in P[p]["nec"]})
    for k in nec_keys:
        have = [p for p in pids if k in P[p]["nec"]]
        A["necessity"][k] = {"N": _med_ci([P[p]["nec"][k]["N"] for p in have], [tab[p] for p in have]),
                             "n_flag_member": int(sum(P[p]["nec"][k]["flag_member"] for p in have)),
                             "n_flag_partner": int(sum(P[p]["nec"][k]["flag_partner"] for p in have))}
    for site, arm in CTRL_ARMS:
        k = f"{site}|both|{arm}"
        for kind in ("rot", "rclamp"):
            per = []
            for p in pids:
                if k not in P[p]["nec"] or not P[p]["nec_ctrl"].get(f"{site}|{kind}:{arm}"):
                    continue
                c = np.asarray(P[p]["nec_ctrl"][f"{site}|{kind}:{arm}"])
                n = P[p]["nec"][k]["N"]
                per.append({"pair_id": p, "N": n, "ctrl_median": float(np.median(c)), "ctrl_q95": float(np.quantile(c, 0.95)),
                            "exceedances": int((c >= n).sum()), "rank_stat": float((1 + (c >= n).sum()) / (1 + len(c)))})
            A["necessity_controls"][f"{site}|{arm}|{kind}"] = {
                "n": len(per), "frac_real_above_q95": float(np.mean([x["N"] > x["ctrl_q95"] for x in per])) if per else None,
                "median_rank_stat": float(np.median([x["rank_stat"] for x in per])) if per else None, "per_pair": per}
    # reproducibility (S and N): r1 vs r2
    for kind, keyset, get in (("S", sorted({k.split("|", 2)[0] + "|" + k.split("|", 2)[2] for k in arm_keys if "|r1|" in k}),
                               lambda p, s, f, a: _S(P[p], s, f, a)),
                              ("N", sorted({k.split("|", 2)[0] + "|" + k.split("|", 2)[2] for k in nec_keys if "|r1|" in k}),
                               lambda p, s, f, a: P[p]["nec"].get(f"{s}|{f}|{a}", {}).get("N"))):
        for sa in keyset:
            site, arm = sa.split("|")
            have = [p for p in pids if get(p, site, "r1", arm) is not None and get(p, site, "r2", arm) is not None]
            if len(have) < 2:
                continue
            x1 = np.array([get(p, site, "r1", arm) for p in have])
            x2 = np.array([get(p, site, "r2", arm) for p in have])
            x0 = np.array([get(p, site, "both", arm) for p in have], dtype=float)
            rs = [i for i, p in enumerate(have) if p in rec_sub[site]]
            rec = lambda x: float(np.median([x[i] / P[have[i]]["S_full"][site] for i in rs])) if rs else None
            A["reproducibility"][f"{kind}|{site}|{arm}"] = {
                "n": len(have), "pearson_r1_r2": float(np.corrcoef(x1, x2)[0, 1]) if x1.std() > 0 and x2.std() > 0 else None,
                "signed_bias_median_r1_minus_r2": float(np.median(x1 - x2)), "median_abs_diff": float(np.median(np.abs(x1 - x2))),
                "range": [float(min(x1.min(), x2.min())), float(max(x1.max(), x2.max()))],
                "median": {"both": float(np.median(x0)), "r1": float(np.median(x1)), "r2": float(np.median(x2))},
                "median_recovery": {"both": rec(x0), "r1": rec(x1), "r2": rec(x2)} if kind == "S" else None}
    # interaction I = S_span + S_complement - S_full
    for site, a1, a2 in INTERACT:
        for fit in FITS:
            have = [p for p in pids if _S(P[p], site, fit, a1) is not None and _S(P[p], site, fit, a2) is not None]
            I = [_S(P[p], site, fit, a1) + _S(P[p], site, fit, a2) - P[p]["S_full"][site] for p in have]
            t = [tab[p] for p in have]
            A["interaction"][f"{site}|{fit}|{a1}+{a2}"] = {
                "I_signed": _med_ci(I, t), "abs_I": _med_ci(np.abs(I), t),
                "frac_abs_I_ge": _med_ci(I, t, stat=lambda v: float(np.mean(np.abs(v) >= I_BIG))), "threshold": I_BIG}
    # predictions (priors in design §7; definitions fixed here before the run)
    rec = lambda k: (A["arms"].get(k) or {}).get("recovery") or {}
    z_verdict = lambda f: (rec(f"colon|{f}|Z").get("value") or -np.inf) >= 0.25
    rz = A["reproducibility"].get("S|colon|Z", {}).get("median_recovery") or {}
    nz = A["necessity_controls"].get("colon|Z|rot", {})
    pz = A["paired"].get("colon|both|Z-pca_kZ", {}).get("dS_recovery_subset") or {}
    A["predictions"] = {
        "S1_median_Z_recovery_ge_0.25": {"value": rec("colon|both|Z").get("value"), "holds": z_verdict("both")},
        "S2_median_J_recovery_ge_0.5": {"value": rec("colon|both|J").get("value"), "holds": (rec("colon|both|J").get("value") or -np.inf) >= 0.5},
        "S3_median_start_span_recovery_ge_0.5": {"value": rec("start_ref|both|Sspan").get("value"), "holds": (rec("start_ref|both|Sspan").get("value") or -np.inf) >= 0.5},
        "S4_Z_beats_unscaled_PCA_kZ": {"value": pz.get("value"), "holds": (pz.get("value") or -np.inf) > 0,
                                       "definition": "paired median dS (Z - PCA_kZ unscaled), recovery subset, primary fit"},
        "S5_Z_reproducible_under_stress": {"value": rz, "holds": bool(rz and None not in rz.values() and z_verdict("r1") == z_verdict("both") == z_verdict("r2")
                                                                      and abs(rz["r1"] - rz["r2"]) <= TOL_MATERIAL),
                                           "definition": f"r1 and r2 give S1's verdict and |median recovery r1 - r2| <= {TOL_MATERIAL}"},
        "S6_Z_clamp_above_rotation_q95_in_half": {"value": nz.get("frac_real_above_q95"), "holds": (nz.get("frac_real_above_q95") or 0) >= 0.5}}
    d = RES / ("analysis" if src == "run" else f"analysis_{src}")
    zr = sorted((P[p]["swap"]["colon|both|Z"]["S"] / P[p]["S_full"]["colon"], p) for p in rec_sub["colon"] if "colon|both|Z" in P[p]["swap"])
    ex = {"random_seeded": [str(x) for x in np.random.default_rng(SEED).choice(pids, size=min(3, len(pids)), replace=False)],
          "median_Z_recovery": zr[len(zr) // 2][1] if zr else None, "max_Z_recovery": zr[-1][1] if zr else None,
          "min_Z_recovery": zr[0][1] if zr else None,
          "flagged_or_failed": [p for p in pids if P[p]["swap"].get("colon|both|Z", {}).get("flagged")
                                or (P[p]["swap"].get("colon|both|Z", {}).get("S", 0) < 0)][:3]}
    A["examples"] = ex
    sel = list(dict.fromkeys([x for v in ex.values() for x in (v if isinstance(v, list) else [v]) if x]))
    d.mkdir(parents=True, exist_ok=True)
    (d / "examples.md").write_text("# 004 qualitative examples\n\n" + json.dumps(ex, indent=1) + "\n\n" + inspect_text(rows, sel))
    write_json(d / "summary.json", A)
    write_json(d / "per_pair.json", P)
    print(json.dumps({"n_pairs": len(pids), "recovery_subset": {s: len(v) for s, v in rec_sub.items()},
                      "predictions": {k: v["holds"] for k, v in A["predictions"].items()}}, indent=1, default=float))


def stage_run():
    assert "--pi-go" in sys.argv, "the 33-pair run needs the PI's go (pass --pi-go)"
    sm = json.loads((RES / "smoke_checks.json").read_text())
    assert sm["passes"], "smoke gates failed"
    RG, torch, tok, ents, F, groups, model = _setup()
    d = OUT / "run"
    d.mkdir(parents=True, exist_ok=True)
    done = {r["pair_id"] for r in _jl(d / "pairs_done.jsonl")}
    write_manifest("run", {"n_pairs": len(ents), "already_done": len(done)})
    with RG.SecEngine(model, l0=39) as sec:
        for n, e in enumerate(ents):
            pid = e["pair"].pair_id
            if pid in done:
                continue
            t0 = time.time()
            pr = PairRunner(sec, tok, e, F, groups)
            rows = pr.run_all(conditions(pid, pr.bases))
            with open(d / "rows.jsonl", "a") as f:
                for r in rows:
                    f.write(json.dumps(r, default=float) + "\n")
            with open(d / "pairs_done.jsonl", "a") as f:
                f.write(json.dumps({"pair_id": pid, "seconds": time.time() - t0, "n_rows": len(rows),
                                    "fid_ok": all(r.get("fid_ok", True) for r in rows)}) + "\n")
            sec.clear()
            print(f"[{n + 1}/{len(ents)}] {pid} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    {"preflight": stage_preflight, "smoke": stage_smoke, "run": stage_run, "inspect": lambda: stage_inspect(*sys.argv[2:3]),
     "analysis": lambda: stage_analysis(*sys.argv[2:3])}[sys.argv[1]]()
