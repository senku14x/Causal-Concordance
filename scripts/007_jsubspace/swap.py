"""007 Stage 2 (Brew Part B; design §5) and Stage 2b (the paper's §3.3 test as an adapted selectivity test; design §6 +
A1): model-side swaps at the brew ':' cell, L40/42/44, on 004's 33 frozen pairs, SEC regime, 004's write path
(apply_write + Fidelity G1), scoring (full-sequence answer margin over the 24 spellings) and damage readouts.
Experiments about J geometry: no labels, never used to score readers.

  python swap.py --run-id s1

Order in one model session: smoke (2 pairs: G0 zero write, G1, G-repro vs 004) -> Stage 2 on 33 pairs -> Stage 2b on 33
pairs (its 2-pair check is a reported smoke check, not a gate, A1). Every pair writes one JSONL per stage; reruns skip
finished pairs. G1 failures quarantine the row (recorded, excluded) and do not stop the run (design §5.4).
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
import margin as MG                               # noqa: E402
import regime as RG                               # noqa: E402
import registry as R                              # noqa: E402
import span_assay as SA                           # noqa: E402  (apply_write, Fidelity, summarize, spelling_groups, cell_pos, seed_of)
import p007 as C                                  # noqa: E402
from pieces import RDELTA_DIM, f32_orth, rand_basis_f32   # noqa: E402

FLOOR = 2e-4
REPRO_TOL = 0.01


def tables_sorted():
    meta = json.loads(C.CAPS_META.read_text())["renders"]
    return sorted({r["item"] for r in meta if r["role"] in ("member", "partner")})


def s004_full():
    """004's whole-cell S at ':' L42 per pair (G-repro reference)."""
    out = {}
    for f in (C.PROJECT / "outputs" / "004_span_assay" / "run" / "pairs").glob("*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r.get("kind") == "swap" and r.get("site") == "colon" and r.get("arm") == "full":
                out[r["pair_id"]] = r["S"]
    return out


class Bank:
    """Every subspace basis of Stage 2, built once (float32, QR re-orthonormalised; CPU, moved per use)."""

    def __init__(self):
        self.J = {L: b for L, b in ((L, torch.load(C.STAGE0 / "bases_f32.pt")[f"J_L{L}"]) for L in C.LAYERS)}
        self.pca = torch.load(C.STAGE0 / "pca_bases.pt")
        self.rand, self.errs = {}, {}
        for r in range(10):
            self.rand[("r1024", r)], _, e = rand_basis_f32(C.rand_seed("r1024", r), 1024)
            self.errs[f"r1024_{r}"] = e
        for kind, k in (("r512", 512), ("r2560", 2560)):
            for r in range(3):
                self.rand[(kind, r)], _, e = rand_basis_f32(C.rand_seed(kind, r), k)
                self.errs[f"{kind}_{r}"] = e
        for L in C.LAYERS:
            for r in range(10):
                self.rand[("rdelta", L, r)], _, e = rand_basis_f32(C.rand_seed("rdelta", r), RDELTA_DIM[L])
                self.errs[f"rdelta_L{L}_{r}"] = e
        assert max(self.errs.values()) <= 1e-5

    def pca_basis(self, L, table, k):
        P = self.pca[f"colon_L{L}"]
        (f,) = [i for i, fold in enumerate(P["folds"]) if table in fold]
        return f32_orth(P["V_folds"][f][:, :k])[0]

    def conditions(self, L, table):
        """Stage 2's 34 conditions at one layer: (name, family, op, basis or None, dim, draw)."""
        c = [("whole", "whole", "full", None, 5120, None), ("J", "J", "span", self.J[L], 1024, None),
             ("rest_J", "rest_J", "comp", self.J[L], 4096, None)]
        for k in (1, 2, 4, 8, 16):
            c.append((f"PCA{k}", "PCA", "span", self.pca_basis(L, table, k), k, None))
        for r in range(10):
            c.append((f"R1024_{r}", "R1024", "span", self.rand[("r1024", r)], 1024, r))
        for r in range(10):
            c.append((f"Rdelta_{r}", "Rdelta", "span", self.rand[("rdelta", L, r)], RDELTA_DIM[L], r))
        for r in range(3):
            c.append((f"R512_{r}", "R512", "span", self.rand[("r512", r)], 512, r))
            c.append((f"R2560_{r}", "R2560", "span", self.rand[("r2560", r)], 2560, r))
        assert len(c) == 34
        return c


class Runner:
    """One 004 pair under SEC: member (recipient) and partner (donor) captures at ':' L40/42/44 for every continuation."""

    def __init__(self, sec, tok, e, groups, table):
        self.sec, self.tok, self.e, self.groups, self.table = sec, tok, e, groups, table
        pair = e["pair"]
        self.pair, self.rx, self.ry = pair, e["r_m"], e["r_p"]
        self.Sa, self.Sb = MG.spellings(tok, "brew", pair.m.a), MG.spellings(tok, "brew", pair.p.a)
        self.pos = SA.cell_pos(self.rx, "emit_colon")
        assert self.pos == SA.cell_pos(self.ry, "emit_colon")
        self.conts = [()] + [s for g in groups for s in g if len(s) > 1]
        self.own, self.src = {}, {}
        for c in self.conts:
            a = sec.forward(self.rx.ids + list(c), record=list(C.LAYERS))[1]
            self.own[c] = {L: a[L][self.pos].clone() for L in C.LAYERS}
            a = sec.forward(self.ry.ids + list(c), record=list(C.LAYERS))[1]
            self.src[c] = {L: a[L][self.pos].clone() for L in C.LAYERS}
        self.clean = self.score(self.rx.ids)
        self.donor = self.score(self.ry.ids)
        self.cs, self.ds = SA.summarize(self.clean, groups), SA.summarize(self.donor, groups)
        self.G = self.donor["m"] - self.clean["m"]
        self.base = {"pair_id": pair.pair_id, "item": pair.item, "table": table, "G": self.G, "m_clean": self.clean["m"],
                     "m_donor": self.donor["m"], "a": pair.m.a, "a2": pair.p.a, "z": pair.m.z, "z2": pair.p.z,
                     "logq_clean": self.cs["logq"], "outside_clean": self.cs["outside"]}

    def score(self, ids, edit_fn=None):
        return self.sec.score(ids, self.Sa, self.Sb, edit_fn=edit_fn, extra_spellings=self.groups)

    def evaluate(self, L, delta, info):
        """delta(h, hd) -> additive write (float32, device) or the string 'full' (direct assignment of the donor cell)."""
        t0 = time.time()
        fid = SA.Fidelity()
        pos = self.pos

        def edit_fn(cont):
            hd = self.src[cont][L]

            def f(x):
                with torch.autocast("cuda", enabled=False):
                    h = x[pos].clone()
                    d = delta(h, hd)
                    if isinstance(d, str):
                        x, req, real, hh = SA.apply_write(x, pos, target=hd)
                    else:
                        x, req, real, hh = SA.apply_write(x, pos, d=d)
                    fid.add(req, real, hh)
                return x
            return {L: f}, []
        ed = self.score(self.rx.ids, edit_fn=edit_fn)
        es = SA.summarize(ed, self.groups)
        dmg = RG.sec_damage(self.sec, self.clean, ed, pos, len(self.rx))
        row = {**self.base, **info, "layer": L, "dm": ed["m"] - self.clean["m"], "S": (ed["m"] - self.clean["m"]) / self.G,
               "d_logP_a": es["lse_a"] - self.cs["lse_a"], "d_logP_b": es["lse_b"] - self.cs["lse_b"],
               "outside_up": es["outside"] - self.cs["outside"], "logq": es["logq"],
               "top1_retention": dmg["top1_retention"], "off_pair_mass_clean": dmg["off_pair_mass_clean"],
               "off_pair_mass_edit": dmg["off_pair_mass_edit"], "fid_ok": fid.ok()}
        sm = fid.summary()
        row.update({k: sm[k] for k in ("fid_rho", "fid_kappa_min", "fid_modes", "write_norm", "h_norm")})
        row["flag_outside"] = row["outside_up"] > SA.FLAG_OUTSIDE
        row["flag_both_down"] = row["d_logP_a"] < -SA.FLAG_BOTH_DOWN and row["d_logP_b"] < -SA.FLAG_BOTH_DOWN
        row["seconds"] = time.time() - t0
        return row

    def delta_share(self, L, op, B):
        D = (self.src[()][L] - self.own[()][L]).double().cpu()
        if op == "full":
            return 1.0
        Bd = B.double()
        p = Bd @ (Bd.T @ D)
        if op == "comp":
            p = D - p
        return float((p ** 2).sum() / (D ** 2).sum())


def span_delta(op, B):
    if op == "full":
        return lambda h, hd: "full"
    Bt = B.to("cuda")
    if op == "span":
        return lambda h, hd: Bt @ (Bt.T @ (hd - h))
    if op == "comp":
        return lambda h, hd: (hd - h) - Bt @ (Bt.T @ (hd - h))
    raise ValueError(op)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="s1")
    a = ap.parse_args()
    assert not torch.backends.cuda.matmul.allow_tf32 and torch.get_float32_matmul_precision() == "highest", "TF32 must be off"
    d2, d2b = C.OUT / "stage2" / a.run_id, C.OUT / "stage2b" / a.run_id
    for d in (d2, d2b):
        (d / "pairs").mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps({"run_id": a.run_id, "manifest": R.manifest_block(),
                                                     "design": "docs/design_specs/007_jsubspace.md (v2 + A1)"}, indent=1))
    log = C.Logger(d2 / "log.txt")
    t_start = time.time()
    tok = R.make_tokenizer()
    ents, man = SA.material(tok)
    tabs = tables_sorted()
    groups = SA.spelling_groups(tok)
    bank = Bank()
    ref004 = s004_full()
    dirs = torch.load(C.STAGE0 / "stage2b_directions.pt")
    state = {"bank_qr_err_max": max(bank.errs.values()), "smoke": {}, "quarantined": 0}
    model = R.load_model()
    with RG.SecEngine(model, l0=39) as sec:
        # ------------------------------------------------ smoke: G0 (zero write), G1, G-repro on the first 2 pairs
        smoke_ok = True
        for e in ents[:2]:
            pr = Runner(sec, tok, e, groups, tabs.index(e["pair"].item))
            g = {}
            for L in C.LAYERS:
                r0 = pr.evaluate(L, lambda h, hd: torch.zeros_like(h), {"cond": "zero"})
                g[f"G0_L{L}"] = {"dm": r0["dm"], "fid_ok": r0["fid_ok"], "pass": abs(r0["dm"]) <= FLOOR and r0["fid_ok"]}
            rw = pr.evaluate(42, span_delta("full", None), {"cond": "whole"})
            g["G_repro_L42"] = {"S": rw["S"], "S_004": ref004[e["pair"].pair_id],
                                "pass": abs(rw["S"] - ref004[e["pair"].pair_id]) <= REPRO_TOL and rw["fid_ok"]}
            state["smoke"][e["pair"].pair_id] = g
            smoke_ok = smoke_ok and all(v["pass"] for v in g.values())
            log("smoke", e["pair"].pair_id, json.dumps(g, default=float))
            sec.clear()
        state["smoke_pass"] = smoke_ok
        C.dump(d2 / "state.json", state)
        if not smoke_ok:
            log("STOP GATE: Stage 2 smoke failed; Stage 2 and 2b (same write path) are skipped")
            return
        # ------------------------------------------------ Stage 2
        for n, e in enumerate(ents):
            pid = e["pair"].pair_id
            f = d2 / "pairs" / f"{pid}.jsonl"
            if f.exists():
                continue
            t0 = time.time()
            pr = Runner(sec, tok, e, groups, tabs.index(e["pair"].item))
            rows = [{**pr.base, "cond": "clean", "member": pr.cs, "partner": pr.ds}]
            for L in C.LAYERS:
                for name, fam, op, B, k, draw in bank.conditions(L, pr.table):
                    r = pr.evaluate(L, span_delta(op, B), {"cond": name, "family": fam, "op": op, "k": k, "draw": draw})
                    r["delta_share"] = pr.delta_share(L, op, B)
                    if not r["fid_ok"]:
                        r["quarantined"] = True
                        state["quarantined"] += 1
                    rows.append(r)
            whole42 = [r for r in rows if r.get("cond") == "whole" and r.get("layer") == 42][0]
            rep = abs(whole42["S"] - ref004[pid])
            if rep > REPRO_TOL:
                log(f"STOP GATE: G-repro failed for {pid}: S {whole42['S']:.4f} vs 004 {ref004[pid]:.4f}")
                state["G_repro_fail"] = {"pair": pid, "S": whole42["S"], "S_004": ref004[pid]}
                C.dump(d2 / "state.json", state)
                return
            tmp = f.with_suffix(".tmp")
            with open(tmp, "w") as fo:
                for r in rows:
                    fo.write(json.dumps(r, default=float) + "\n")
            tmp.rename(f)
            sec.clear()
            log(f"[2 {n + 1}/{len(ents)}] {pid} {time.time() - t0:.0f}s (G-repro |dS| {rep:.4f})")
        state["stage2_minutes"] = (time.time() - t_start) / 60
        C.dump(d2 / "state.json", state)
        # ------------------------------------------------ Stage 2b (all 33 pairs; 2-pair smoke check reported)
        log2 = C.Logger(d2b / "log.txt")
        st2b = {"quarantined": 0}
        t2 = time.time()
        for n, e in enumerate(ents):
            pid = e["pair"].pair_id
            f = d2b / "pairs" / f"{pid}.jsonl"
            if f.exists():
                continue
            t0 = time.time()
            tb = tabs.index(e["pair"].item)
            pr = Runner(sec, tok, e, groups, tb)
            rows = [{**pr.base, "cond": "clean", "member": pr.cs, "partner": pr.ds}]
            for L in C.LAYERS:
                vm = {ln: dirs[f"{L}|{ln}|loto|{tb}|{pr.pair.m.z}"] for ln in ("J_NP", "J_CB")}
                vp = {ln: dirs[f"{L}|{ln}|loto|{tb}|{pr.pair.p.z}"] for ln in ("J_NP", "J_CB")}
                full = (vp["J_NP"]["full"] - vm["J_NP"]["full"]).double()
                nf = float(full.norm())
                edits = [("full", "full", None, full)]
                for ln in ("J_NP", "J_CB"):
                    for part in ("J", "nonJ"):
                        dx = (vp[ln][part] - vm[ln][part]).double()
                        edits.append((f"{part}_{ln}", part, ln, dx * (nf / float(dx.norm()))))
                for i in range(4):
                    rv = SA.unit(SA.seed_of(pid, "2b_rand", L, i))
                    edits.append((f"rand_{i}", "rand", None, torch.from_numpy(rv) * nf))
                for name, part, ln, dx in edits:
                    dxt = dx.float().to("cuda")
                    r = pr.evaluate(L, lambda h, hd, dxt=dxt: dxt, {"cond": name, "part": part, "lens": ln})
                    r["edit_norm"] = float(dx.norm())
                    r["full_norm"] = nf
                    if not r["fid_ok"]:
                        r["quarantined"] = True
                        st2b["quarantined"] += 1
                    rows.append(r)
            tmp = f.with_suffix(".tmp")
            with open(tmp, "w") as fo:
                for r in rows:
                    fo.write(json.dumps(r, default=float) + "\n")
            tmp.rename(f)
            sec.clear()
            if n < 2:                                                   # the A1 smoke check (reported, not a gate)
                fs = [r["S"] for r in rows if r.get("cond") == "full"]
                st2b.setdefault("smoke_check_full_S", {})[pid] = fs
            log2(f"[2b {n + 1}/{len(ents)}] {pid} {time.time() - t0:.0f}s")
        smk = [s for v in st2b.get("smoke_check_full_S", {}).values() for s in v]
        st2b["smoke_check"] = {"median_full_S": float(np.median(smk)) if smk else None, "criterion": ">= 0.02",
                               "pass": bool(smk) and float(np.median(smk)) >= 0.02, "note": "reported only (A1)"}
        st2b["minutes"] = (time.time() - t2) / 60
        C.dump(d2b / "state.json", st2b)
    state["wall_minutes"] = (time.time() - t_start) / 60
    C.dump(d2 / "state.json", state)
    log(f"stage 2 + 2b done ({state['wall_minutes']:.1f} min)")


if __name__ == "__main__":
    main()
