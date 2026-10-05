"""Design 005 full run under amendment 01 (docs/design_specs/005_full_run.md; branch 005-amended-full-run).

  python full_run.py --stage rehearse --run-id rehearse1     (2 dev concepts; W without any test concept)
  python full_run.py --stage run --run-id <UTC timestamp>     (the 45 test concepts)

Phase A (subject resident): A1 model side, A2 sampled naming, A3 clamps (J-atom and output-direction re-entry clamps,
E = 44..62), A4 sensitivity variants, A5 token readers + probe, A6 Oracle naming, A7 Oracle likelihoods. Phase B
(subject freed): NLA naming and likelihoods. Every block writes one file and is skipped on a rerun. The log prints
progress, gates and timings only, never an outcome.
"""
import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p005 as P                                  # noqa: E402
import pilot as PI                                # noqa: E402  (sets PL.SAMPLE_CHUNK = 192)
import planted as PL                              # noqa: E402
import registry as R                              # noqa: E402

GRID9 = PI.GRID9
TEXT_S = [1 / 64, 1 / 4, 1.0]
ENDS = list(range(44, 63, 2))
BAND = list(P.BAND)
MODEL_CONDS = [(c, 0) for c in (1, 2, 3, 4, 5, 6, 8, 9, 10, "10r")] + [(7, k) for k in range(5)]
R2_CONDS = [(1, 0), (2, 0), (3, 0), (4, 0), (7, 0)]
TEXT_CONDS = [(2, 0), (3, 0), (4, 0), (7, 0)]
LIK_CONDS = [(2, 0), (3, 0), (4, 0)] + [(7, k) for k in range(5)]
ORACLE_L, NLA_L = [40, 44], 42
PREFIX = {"oracle": "-", "nla": "<explanation>\nThe text is about"}
VARIANTS = {"k10": dict(k=10, refit="nnls", raw=False, base=False),
            "k25": dict(k=25, refit="nnls", raw=False, base=False),
            "ls": dict(k=16, refit="ls", raw=False, base=False),
            "raw": dict(k=16, refit="nnls", raw=True, base=False),
            "base194": dict(k=16, refit="nnls", raw=False, base=True)}


class FullRun(PI.Pilot):
    def __init__(self, stage, run_id):
        self.stage, self.run_id = stage, run_id
        self.dir = PL.OUT / "run" / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = PL.Log(self.dir / "log.txt")
        self.timings, self.mem, self.notes = {}, {}, []
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.concepts = list(P.TEST) if stage == "run" else ["gel", "earth"]
        self.id_set = list(P.CONCEPTS) if stage == "run" else list(P.DEV)
        self.state.update({"stage": stage, "concepts": self.concepts, "id_set": self.id_set})

    # ------------------------------------------------------------------------------------------------- setup
    def prepare(self):
        self.setup()
        self.mat = torch.load(PL.OUT / "pilot" / "pilot1" / "material.pt")   # same renders and model as the pilot
        self.ids = P.render_intro(self.tok, "default")
        self.clean()
        self.load_lenses()
        import readers as RD
        self.JR = RD.load_lenses(("R_CB",), layers=P.READ_LAYERS, device=self.dev)["R_CB"].jacobians
        self.vectors(sorted(set(BAND) | set(P.READ_LAYERS)), self.concepts)
        W = {}
        for w in self.id_set + list(self.base):
            try:
                W[w] = P.spellings(self.tok, w)["quote"]
            except AssertionError:
                continue
        assert set(self.id_set) <= set(W)
        if self.stage == "rehearse":
            assert not set(W) & set(P.TEST)
        self.W_spec = W
        self.clean_w = {w: float(torch.logsumexp(self.lp_clean[-1, t], 0)) for w, t in W.items()}
        leaky = sorted(c for c in self.concepts if sum(self.g5[(c, l)] <= 10 for l in BAND) >= 2)
        self.state["material"] = {"W_size": len(W), "leaky_bandB": leaky,
                                  "J_share_median": float(np.median([self.V[(c, l)]["J_share"] for c in self.concepts for l in BAND])),
                                  "J_share": {f"{c}|L{l}": self.V[(c, l)]["J_share"] for c in self.concepts for l in BAND},
                                  "g5_rank": {f"{c}|L{l}": self.g5[(c, l)] for c in self.concepts for l in BAND},
                                  "cos_loo_base": {f"{c}|L{l}": self.V[(c, l)]["cos_loo_base"] for c in self.concepts for l in BAND}}
        self.save_state()
        self.log(f"prepared: {len(self.concepts)} concepts, |W| = {len(W)}, leaky at band B: {len(leaky)}")

    def g0(self, fids, clogs, tag):
        """G0: every write rho in [0.9, 1.1] and kappa >= 0.99; every clamp residual <= 1e-4. A failure stops the run."""
        bad = []
        for f in fids:
            for l, v in f.items():
                if not (0.9 <= v["rho_min"] and v["rho_max"] <= 1.1 and v["kappa_min"] >= 0.99):
                    bad.append([tag, "write", l, v])
        for c in clogs:
            for b, lg in c.items():
                for l, v in lg.items():
                    if v["residual"] > 1e-4:
                        bad.append([tag, "clamp", b, l, v["residual"]])
        g = self.state.setdefault("G0", {}).setdefault(tag, {"writes_checked": 0, "clamp_rows_checked": 0,
                                                             "failures": 0, "examples": []})
        g["writes_checked"] += sum(len(f) for f in fids)              # accumulated over a block's chunks
        g["clamp_rows_checked"] += sum(len(c) for c in clogs)
        g["failures"] += len(bad)
        g["examples"] = (g["examples"] + bad)[:5]
        self.save_state()
        if bad:
            raise RuntimeError(f"G0 failed in {tag}: {bad[:3]}")

    def token_sets(self):
        """Clamp token sets, exactly planted.Smoke.clamp_sets's (same code and seeds): {c: (T_c, T_r)}."""
        vocab = self.tok.convert_ids_to_tokens(list(range(len(self.tok))))
        excl = set()
        for c in self.all_concepts:
            for s in (" " + c, " " + c.capitalize(), c, c.capitalize()):
                t = self.tok.encode(s, add_special_tokens=False)
                if len(t) == 1:
                    excl.add(t[0])
        pool = [i for i, t in enumerate(vocab) if t is not None and re.fullmatch(r"Ġ[A-Za-z]{3,}", t) and i not in excl]
        out = {}
        for c in self.concepts:
            Tc = sorted(set(t for l in BAND for t in self.V[(c, l)]["support"]) | set(self.sp[c]["all"]))
            g = torch.Generator().manual_seed(P.seed_of("rand-clamp", c))
            out[c] = (Tc, [pool[j] for j in torch.randperm(len(pool), generator=g)[:len(Tc)].tolist()])
        return out

    # ------------------------------------------------------------------------------------------- A1 model side
    def block_model(self):
        f = self.dir / "model_rows.json"
        if f.exists():
            self.rows_model = json.loads(f.read_text())
            self.caps = torch.load(self.dir / "caps.pt")
            self.log("A1 model side: loaded")
            return
        self.clamp_sets(self.concepts, BAND)               # design clamp (conditions 10/10r, 11/11r): to clean, L32-62
        rows = [{"concept": c, "cond": cond, "s": s, "seed": sd} for c in self.concepts for s in GRID9
                for cond, sd in MODEL_CONDS]
        rows += [{"concept": c, "cond": cond, "s": 0.0, "seed": 0} for c in self.concepts for cond in (11, "11r")]
        self.log(f"A1 model side: {len(rows)} rows")
        caps, fid, clog = self.teacher_forced(rows, BAND, "model")
        self.g0(fid, clog, "A1")
        del self.Q
        torch.save(caps, self.dir / "caps.pt")
        PL.jdump({"inject": fid, "clamp": clog}, self.dir / "fidelity_model.json")
        PL.jdump(rows, f)
        self.rows_model, self.caps = rows, caps
        self.save_timings()

    # ---------------------------------------------------------------------------------------- A2 sampled naming
    def block_r2(self):
        f = self.dir / "r2_rows.json"
        if f.exists():
            return
        rows = [{"concept": c, "cond": cond, "s": s, "seed": sd} for c in self.concepts for s in TEXT_S for cond, sd in R2_CONDS]
        self.log(f"A2 sampled naming: {len(rows)} rows x {P.NAME_SAMPLES} samples")
        self.sample(rows, BAND, "r2")
        PL.jdump(rows, f)
        self.save_timings()

    # ------------------------------------------------------------------------------------------------ A3 clamps
    def block_clamps(self):
        f = self.dir / "clamp_rows.json"
        if f.exists():
            return
        sets = self.token_sets()
        self.clamp_sets(self.concepts, BAND)
        for c, (Tc, Tr) in sets.items():
            cs = self.state["clamp_sets"][c]
            assert cs["c_tokens"] == [self.tok.decode([t]) for t in Tc] and cs["r_tokens"] == [self.tok.decode([t]) for t in Tr]
        Qo = {}
        for c, (Tc, Tr) in sets.items():
            for kind, T in (("c", Tc), ("r", Tr)):
                Qm, _ = torch.linalg.qr((self.gain * self.W_U[T]).double().T)
                Qo[(c, kind)] = Qm.float()
        full = list(range(BAND[0], 63))
        rows = [{"concept": c, "cond": f"{fam}{E}{'r' if k == 'r' else ''}", "fam": fam, "E": E, "kind": k, "s": s, "seed": 0}
                for c in self.concepts for s in GRID9 for E in ENDS for fam in ("E", "O") for k in ("c", "r")]
        self.log(f"A3 clamps: {len(rows)} rows")
        lpos = list(range(125, len(self.ids)))
        clogs = []
        for j in range(0, len(rows), PL.TF_CHUNK):
            ch = rows[j:j + PL.TF_CHUNK]
            dl = {l: torch.stack([r["s"] * self.Nbar[l] * self.vec_of(dict(r, cond=4), l) for r in ch]) for l in BAND}
            Qr, off = {}, {}
            for b, r in enumerate(ch):
                layers = list(range(BAND[0], r["E"] + 1))
                Qr[b] = {l: (self.Q[(r["concept"], r["kind"])][l] if r["fam"] == "E" else Qo[(r["concept"], r["kind"])])
                         for l in layers}
                acc, A = torch.zeros(R.D_MODEL, device=self.dev), {}
                for l in layers:
                    if l in dl:
                        acc = acc + dl[l][b]
                    A[l] = acc.clone()
                off[b] = A
            with PL.Timer(self.timings, "clamps", len(ch)):
                inj = P.BandInject(self.blocks, dl, P.POS["span"])
                cl = P.BandClamp(self.blocks, Qr, self.hc, full, offsets=off, offset_pos=P.POS["span"])
                lp, _ = P.forward_rows(self.model, self.ids, len(ch), self.W_U, lpos, writes=[inj, cl])
            self.g0([inj.fid], [], "A3_inject")
            clogs.append({str(j + b): v for b, v in cl.log.items()})
            for b, r in enumerate(ch):
                r.update(P.intro_metrics(lp[b:b + 1], self.lp_clean, self.sp[r["concept"]])[0])
                lg = cl.log[b]
                r["removed_share_band"] = float(np.median([lg[l]["removed_share"] for l in BAND if l in lg]))
                r["max_residual"] = float(max(v["residual"] for v in lg.values()))
            if (j // PL.TF_CHUNK) % 100 == 0:
                self.log(f"A3 clamps {j + len(ch)}/{len(rows)}")
        self.g0([], clogs, "A3_clamp")
        del self.Q, Qo
        torch.cuda.empty_cache()
        PL.jdump(rows, f)
        self.save_timings()

    # ------------------------------------------------------------------------------------------- A4 sensitivity
    def variant_vectors(self, k, refit, raw, base):
        H, words = self.mat["H_tell"], self.mat["words"]
        idx = {w: i for i, w in enumerate(words)}
        nconc = len(self.all_concepts)
        base_rows = list(range(nconc, len(words)))
        V = {}
        for l in BAND:
            Hl = H[:, PL.L_ALL.index(l)].to(self.dev)
            if base:
                bm = Hl[base_rows].mean(0)
                X = torch.stack([Hl[idx[c]] - bm for c in self.concepts])
            else:
                sumc = Hl[:nconc].sum(0)
                X = torch.stack([Hl[idx[c]] - (sumc - Hl[idx[c]]) / (nconc - 1) for c in self.concepts])
            D = self.JS.atoms(self.JNP[l], self.W_U, None if raw else self.gain)
            dec = self.JS.decompose(X, D, k=k, refit=refit, chunk=65536)
            for i, c in enumerate(self.concepts):
                V[(c, l)] = {3: P.unit(dec["x_j"][i]), 4: P.unit(dec["x_nonj"][i]), "J_share": float(dec["energy_j"][i])}
            del D
            torch.cuda.empty_cache()
        return V

    def block_sensitivity(self):
        lpos = list(range(125, len(self.ids)))
        for name, cfg in VARIANTS.items():
            f = self.dir / f"sens_{name}.json"
            if f.exists():
                continue
            V = self.variant_vectors(**cfg)
            rows = [{"concept": c, "cond": cond, "s": s, "variant": name} for c in self.concepts for s in GRID9 for cond in (3, 4)]
            for j in range(0, len(rows), PL.TF_CHUNK):
                ch = rows[j:j + PL.TF_CHUNK]
                dl = {l: torch.stack([r["s"] * self.Nbar[l] * V[(r["concept"], l)][r["cond"]] for r in ch]) for l in BAND}
                with PL.Timer(self.timings, "sensitivity", len(ch)):
                    inj = P.BandInject(self.blocks, dl, P.POS["span"])
                    lp, _ = P.forward_rows(self.model, self.ids, len(ch), self.W_U, lpos, writes=[inj])
                self.g0([inj.fid], [], f"A4_{name}")
                for b, r in enumerate(ch):
                    r.update(P.intro_metrics(lp[b:b + 1], self.lp_clean, self.sp[r["concept"]])[0])
            self.state.setdefault("sensitivity_J_share_median", {})[name] = float(np.median([v["J_share"] for v in V.values()]))
            PL.jdump(rows, f)
            del V
            self.log(f"A4 sensitivity {name} done")
        self.save_timings()

    # -------------------------------------------------------------------------------- A5 token readers + probe
    def direct_x(self, r, l):
        return self.hc[l][P.POS["cq"]].cpu() + r["s"] * self.Nbar[l] * self.vec_of(r, l).cpu()

    def block_token_probe(self):
        f = self.dir / "token_probe_reads.json"
        if f.exists():
            return
        with PL.Timer(self.timings, "probe_training"):
            self.state["G4_probe_validity"] = self.train_probes(self.concepts)
        self.save_state()
        clean_x = {(l, k): (self.hc[l][self.co] if k == "carried_co" else self.hc[l][P.POS["cq"]]).cpu()
                   for l in P.READ_LAYERS for k in ("carried_co", "injected_cq", "direct_cq")}
        out = []
        for l in P.READ_LAYERS:
            reads = []
            for b, r in enumerate(self.rows_model):
                if r["cond"] in (10, "10r", 11, "11r"):
                    continue
                for kind, x in (("carried_co", self.caps[l][b, 1]), ("injected_cq", self.caps[l][b, 0]),
                                ("direct_cq", self.direct_x(r, l))):
                    reads.append({"concept": r["concept"], "cond": r["cond"], "seed": r["seed"], "s": r["s"],
                                  "layer": l, "kind": kind, "x": x})
            with PL.Timer(self.timings, "token_reads", len(reads)):
                for j in range(0, len(reads), 256):
                    ch = reads[j:j + 256]
                    X = torch.stack([q["x"] for q in ch])
                    tr = self.token_reads(X, l, [q["concept"] for q in ch])
                    for q, t in zip(ch, tr):
                        q["token"] = {k: {"rank": v["rank"]} for k, v in t.items()}
                    for c in set(q["concept"] for q in ch):
                        idx = [i for i, q in enumerate(ch) if q["concept"] == c]
                        ps = self.probe_scores(X[idx], l, c)
                        pc = {k: float(self.probe_scores(clean_x[(l, k)][None], l, c)[0])
                              for k in ("carried_co", "injected_cq", "direct_cq")}
                        for i, p in zip(idx, ps):
                            ch[i]["probe"] = float(p)
                            ch[i]["probe_clean"] = pc[ch[i]["kind"]]
            out += [{k: v for k, v in q.items() if k != "x"} for q in reads]
            del reads
            gc.collect()
            self.log(f"A5 token readers + probe L{l} done")
        PL.jdump(out, f)
        self.save_timings()

    # --------------------------------------------------------------------------------------- A6/A7 text readers
    def text_items(self, layer):
        return [{"layer": layer, "x": self.direct_x(dict(concept=c, cond=cond, s=s, seed=sd), layer),
                 "meta": {"concept": c, "cond": cond, "s": s, "seed": sd}}
                for c in self.concepts for s in TEXT_S for cond, sd in TEXT_CONDS]

    def lik_inputs(self, layer):
        rows = [{"concept": c, "cond": cond, "s": s, "seed": sd} for c in self.concepts for s in GRID9 for cond, sd in LIK_CONDS]
        X = torch.stack([self.direct_x(r, layer) for r in rows])
        nat = torch.stack([self.mat["H_tell"][self.mat["words"].index(c), PL.L_ALL.index(layer)] for c in self.concepts])
        return rows, X, self.hc[layer][P.POS["cq"]].cpu(), nat

    def score_reader(self, name, layer, scorer, tok, rows, X, clean, nat):
        f = self.dir / f"lik_{name}_L{layer}.json"
        if f.exists():
            return
        pre, cands = P.forced_candidates(tok, PREFIX[name], self.id_set + list(self.base))
        if self.stage == "rehearse":
            assert not set(cands) & set(P.TEST)
        assert set(self.id_set) <= set(cands), f"{name}: a concept has no single-token candidate after the prefix"
        cand_ids = sorted({t for ts in cands.values() for t in ts})
        col = {t: i for i, t in enumerate(cand_ids)}
        hs = list(X) + [clean] + list(nat)
        with PL.Timer(self.timings, f"lik_{name}_L{layer}", len(hs)):
            lp, own = scorer(hs, pre, cand_ids)
        diff = float((lp - own).abs().max())
        self.state.setdefault("scoring_check_maxabs", {})[f"{name}_L{layer}"] = diff
        assert diff < 1.0, f"{name} L{layer}: float32 vs own-logit log-probs differ by {diff}"

        def lse(row):
            return {w: float(torch.logsumexp(row[[col[t] for t in ts]], 0)) for w, ts in cands.items()}
        n = len(rows)
        PL.jdump({"reader": name, "layer": layer, "prefix": PREFIX[name], "prefix_ids": pre, "W": sorted(cands),
                  "clean": lse(lp[n]), "natural": [{"concept": c, "lpW": lse(lp[n + 1 + i])} for i, c in enumerate(self.concepts)],
                  "rows": [dict(r, lpW=lse(lp[i])) for i, r in enumerate(rows)]}, f)
        self.save_state()
        self.log(f"{name} L{layer} likelihoods: {len(hs)} reads (float32 vs own logits max |diff| {diff:.3f})")

    def block_oracle(self):
        f = self.dir / "oracle_reads.json"
        ol = self.RD.OracleLens(self.model, self.tok)
        try:
            if not f.exists():
                out = []
                for l in ORACLE_L:
                    items = self.text_items(l)
                    for j in range(0, len(items), PL.READ_CHUNK):
                        ch = items[j:j + PL.READ_CHUNK]
                        with PL.Timer(self.timings, "oracle_samples", len(ch) * P.TEXT_K):
                            outs = ol.read_batch([q["x"] for q in ch], l, k=P.TEXT_K, seed=P.seed_of("oracle-run", l, j))
                        out += [{"reader": "oracle", "layer": l, "kind": "direct_cq", **q["meta"], "samples": smp}
                                for q, smp in zip(ch, outs)]
                        if (j // PL.READ_CHUNK) % 10 == 0:
                            self.log(f"A6 Oracle L{l} {j + len(ch)}/{len(items)}")
                    smp = ol.read_batch([self.hc[l][P.POS["cq"]].cpu()], l, k=P.CLEAN_K, seed=P.seed_of("oracle-run-clean", l))[0]
                    out.append({"reader": "oracle", "layer": l, "kind": "direct_cq", "concept": None, "cond": "clean",
                                "s": 0.0, "samples": smp})
                PL.jdump(out, f)
                self.save_timings()
            for l in ORACLE_L:
                rows, X, clean, nat = self.lik_inputs(l)
                self.score_reader("oracle", l, lambda hs, pre, ci, l=l: ol.score_next(hs, l, pre, ci, self.W_U),
                                  self.tok, rows, X, clean, nat)
        finally:
            base = ol.unload()
            assert base is self.model
            del ol, base
            gc.collect()
        self.save_timings()

    # ----------------------------------------------------------------------------------------------- phase B
    def phase_b(self):
        rows42, X42, clean42, nat42 = self.lik_inputs(NLA_L)
        items = self.text_items(NLA_L)
        self.log("phase B: free the subject; load nla-rl")
        del self.model, self.blocks
        for name in ("JNP", "JCB", "JR", "den", "V", "hcn", "caps"):
            self.__dict__.pop(name, None)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with PL.Timer(self.timings, "nla_load"):
            nla = self.RD.NLArl()
        self.state["nla_scaling"] = nla.scaling
        try:
            f = self.dir / "nla_reads.json"
            if not f.exists():
                out = []
                for j in range(0, len(items), PL.READ_CHUNK):
                    ch = items[j:j + PL.READ_CHUNK]
                    with PL.Timer(self.timings, "nla_samples", len(ch) * P.TEXT_K):
                        outs = nla.read_batch([q["x"] for q in ch], k=P.TEXT_K, seed=P.seed_of("nla-run", j))
                    out += [{"reader": "nla", "layer": NLA_L, "kind": "direct_cq", **q["meta"], "samples": smp,
                             "write": nla.last_write} for q, smp in zip(ch, outs)]
                    if (j // PL.READ_CHUNK) % 10 == 0:
                        self.log(f"B1 NLA {j + len(ch)}/{len(items)}")
                smp = nla.read_batch([clean42], k=P.CLEAN_K, seed=P.seed_of("nla-run-clean"))[0]
                out.append({"reader": "nla", "layer": NLA_L, "kind": "direct_cq", "concept": None, "cond": "clean",
                            "s": 0.0, "samples": smp})
                PL.jdump(out, f)
                self.save_timings()
            self.score_reader("nla", NLA_L, lambda hs, pre, ci: nla.score_next(hs, pre, ci), nla.tok,
                              rows42, X42, clean42, nat42)
        finally:
            nla.close()
        self.mem["phaseB_peak_GB"] = torch.cuda.max_memory_allocated() / 1e9
        self.save_timings()

    def save_timings(self):
        self.state["timings"], self.state["memory"] = self.timings, self.mem
        self.save_state()

    def run(self):
        t0 = time.time()
        self.prepare()
        torch.cuda.reset_peak_memory_stats()
        self.block_model()
        self.block_r2()
        self.block_clamps()
        self.block_sensitivity()
        self.block_token_probe()
        self.mem["phaseA_peak_GB"] = torch.cuda.max_memory_allocated() / 1e9
        self.log("A6/A7 Oracle naming + likelihoods")
        self.block_oracle()
        self.phase_b()
        self.state["wall_minutes"] = (time.time() - t0) / 60
        self.save_timings()
        self.log(f"{self.stage} done ({self.state['wall_minutes']:.1f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("rehearse", "run"), required=True)
    ap.add_argument("--run-id", default=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    a = ap.parse_args()
    fr = FullRun(a.stage, a.run_id)
    PL.jdump({"stage": a.stage, "run_id": a.run_id, "manifest": R.manifest_block(),
              "design": "docs/design_specs/005_full_run.md (amendment 01 adopted 2026-10-05)",
              "branch": "005-amended-full-run", "concepts": fr.concepts},
             PL.RES / f"manifest_{'run' if a.stage == 'run' else 'rehearse'}.json")
    fr.run()


if __name__ == "__main__":
    main()
