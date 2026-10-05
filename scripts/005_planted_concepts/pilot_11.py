"""Design 005 §16.7 step 11 (dev-only; pre-registered before it runs): 11a the re-entry clamp on the concept tokens'
output directions (logit-lens coordinates) next to step 10's J-atom clamp; 11b reader likelihoods at a forced answer
position on direct reads of the written cell (Oracle L40/L44, NLA L42), with a natural-activation positive control.

  python pilot_11.py --run-id pilot1      (writes into outputs/005_planted_concepts/pilot/<run_id>/step11/)
"""
import argparse
import gc
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p005 as P                                  # noqa: E402
import pilot as PI                                # noqa: E402
import planted as PL                              # noqa: E402

ENDS = list(range(44, 63, 2))
RCONDS = [(2, 0), (3, 0), (4, 0)] + [(7, k) for k in range(5)]
ORACLE_L = [40, 44]
NLA_L = 42
PREFIX = {"oracle": "-", "nla": "<explanation>\nThe text is about"}


class Step11(PI.Pilot):
    def __init__(self, run_id):
        super().__init__(run_id)
        self.pdir = self.dir
        self.dir = self.pdir / "step11"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = PL.Log(self.dir / "log.txt")
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def capture_material(self):
        self.mat = torch.load(self.pdir / "material.pt")

    def token_sets(self, band):
        """Step 10's clamp token sets (planted.Smoke.clamp_sets, same code and seeds): {c: (T_c, T_r)}."""
        vocab = self.tok.convert_ids_to_tokens(list(range(len(self.tok))))
        excl = set()
        for c in self.all_concepts:
            for s in (" " + c, " " + c.capitalize(), c, c.capitalize()):
                t = self.tok.encode(s, add_special_tokens=False)
                if len(t) == 1:
                    excl.add(t[0])
        pool = [i for i, t in enumerate(vocab) if t is not None and re.fullmatch(r"Ġ[A-Za-z]{3,}", t) and i not in excl]
        out = {}
        for c in P.DEV:
            Tc = sorted(set(t for l in band for t in self.V[(c, l)]["support"]) | set(self.sp[c]["all"]))
            g = torch.Generator().manual_seed(P.seed_of("rand-clamp", c))
            Tr = [pool[j] for j in torch.randperm(len(pool), generator=g)[:len(Tc)].tolist()]
            out[c] = (Tc, Tr)
        return out

    # ------------------------------------------------------------------------------- 11a output-direction clamp
    def out_clamp_sweep(self, bname, band):
        f = self.dir / f"outclamp_{bname}.json"
        if f.exists():
            self.log("skip output clamp", bname)
            return
        sets = self.token_sets(band)
        self.clamp_sets(P.DEV, band)                     # step 10's J sets, only to check the token sets agree
        for c, (Tc, Tr) in sets.items():
            cs = self.state["clamp_sets"][c]
            assert cs["c_tokens"] == [self.tok.decode([t]) for t in Tc] and cs["r_tokens"] == [self.tok.decode([t]) for t in Tr]
        del self.Q
        Qo = {}
        for c, (Tc, Tr) in sets.items():
            for kind, T in (("c", Tc), ("r", Tr)):
                Qm, _ = torch.linalg.qr((self.gain * self.W_U[T]).double().T)
                Qo[(c, kind)] = Qm.float()
        full = list(range(band[0], 63))
        rows = [{"concept": c, "cond": 4, "E": None, "kind": None, "s": s, "seed": 0, "band": bname}
                for c in P.DEV for s in PI.GRID9]
        rows += [{"concept": c, "cond": f"O{E}{'r' if k == 'r' else ''}", "E": E, "kind": k, "s": s, "seed": 0,
                  "band": bname} for c in P.DEV for s in PI.GRID9 for E in ENDS for k in ("c", "r")]
        lpos = list(range(125, len(self.ids)))
        for j in range(0, len(rows), PL.TF_CHUNK):
            ch = rows[j:j + PL.TF_CHUNK]
            dl = {l: torch.stack([r["s"] * self.Nbar[l] * self.vec_of(dict(r, cond=4), l) for r in ch]) for l in band}
            Qr, off = {}, {}
            for b, r in enumerate(ch):
                if r["E"] is None:
                    continue
                layers = list(range(band[0], r["E"] + 1))
                Qr[b] = {l: Qo[(r["concept"], r["kind"])] for l in layers}
                acc, A = torch.zeros(P.R.D_MODEL, device=self.dev), {}
                for l in layers:
                    if l in dl:
                        acc = acc + dl[l][b]
                    A[l] = acc.clone()
                off[b] = A
            with PL.Timer(self.timings, f"outclamp_{bname}", len(ch)):
                inj = P.BandInject(self.blocks, dl, P.POS["span"])
                cl = P.BandClamp(self.blocks, Qr, self.hc, full, offsets=off, offset_pos=P.POS["span"])
                lp, _ = P.forward_rows(self.model, self.ids, len(ch), self.W_U, lpos, writes=[inj, cl])
            for b, r in enumerate(ch):
                r.update(P.intro_metrics(lp[b:b + 1], self.lp_clean, self.sp[r["concept"]])[0])
                if b in cl.log:
                    lg = cl.log[b]
                    r["removed_share_band"] = float(np.median([lg[l]["removed_share"] for l in band if l in lg]))
                    r["max_residual"] = float(max(v["residual"] for v in lg.values()))
        PL.jdump(rows, f)
        torch.cuda.empty_cache()
        self.log("output clamp done", bname)

    # ------------------------------------------------------------------------------------- 11b reader inputs
    def reader_inputs(self):
        rows = [{"concept": c, "cond": cond, "s": s, "seed": sd} for c in P.DEV for s in PI.GRID9 for cond, sd in RCONDS]
        X = {}
        for l in ORACLE_L + [NLA_L]:
            hc = self.hc[l][P.POS["cq"]].float()
            X[l] = torch.stack([hc + r["s"] * self.Nbar[l] * self.vec_of(r, l) for r in rows]).cpu()
            X[(l, "clean")] = hc.cpu()
            X[(l, "natural")] = torch.stack([self.mat["H_tell"][self.mat["words"].index(c), PL.L_ALL.index(l)] for c in P.DEV])
        return rows, X

    def lse_by_word(self, lp, cands, col):
        return {w: float(torch.logsumexp(lp[[col[t] for t in ts]], 0)) for w, ts in cands.items()}

    def score_reader(self, name, layer, scorer, tok, rows, X):
        f = self.dir / f"reads_{name}_L{layer}.json"
        if f.exists():
            self.log("skip", f.name)
            return
        pre, cands = P.forced_candidates(tok, PREFIX[name], list(P.DEV) + list(self.base))
        assert set(P.DEV) <= set(cands), f"{name}: a dev concept has no single-token candidate after the prefix"
        cand_ids = sorted({t for ts in cands.values() for t in ts})
        col = {t: i for i, t in enumerate(cand_ids)}
        hs = list(X[layer]) + [X[(layer, "clean")]] + list(X[(layer, "natural")])
        with PL.Timer(self.timings, f"score_{name}_L{layer}", len(hs)):
            lp, own = scorer(hs, pre, cand_ids)
        diff = float((lp - own).abs().max())
        self.state.setdefault("scoring_check_maxabs", {})[f"{name}_L{layer}"] = diff
        assert diff < 1.0, f"{name} L{layer}: float32 vs own-logit log-probs differ by {diff}"
        n = len(rows)
        out = {"reader": name, "layer": layer, "prefix": PREFIX[name], "prefix_ids": pre, "W": sorted(cands),
               "clean": self.lse_by_word(lp[n], cands, col),
               "natural": [{"concept": c, "lpW": self.lse_by_word(lp[n + 1 + i], cands, col)} for i, c in enumerate(P.DEV)],
               "rows": [dict(r, lpW=self.lse_by_word(lp[i], cands, col)) for i, r in enumerate(rows)]}
        PL.jdump(out, f)
        self.save_state()
        self.log(f"{name} L{layer} scored ({len(hs)} reads; float32 vs own logits max |diff| {diff:.3f})")

    def run(self):
        self.setup()
        self.capture_material()
        self.ids = P.render_intro(self.tok, "default")
        self.clean()
        self.load_lenses()
        self.log("vectors (dev; identical seeds to the pilot)")
        self.vectors(sorted(set(P.BAND2) | set(ORACLE_L) | {NLA_L}), P.DEV)
        for bname, band in PI.BANDS.items():
            self.log("output clamp", bname)
            self.out_clamp_sweep(bname, band)
        rows, X = self.reader_inputs()
        self.log("Oracle scoring")
        ol = self.RD.OracleLens(self.model, self.tok)
        for l in ORACLE_L:
            self.score_reader("oracle", l, lambda hs, pre, ci, l=l: ol.score_next(hs, l, pre, ci, self.W_U), self.tok, rows, X)
        base = ol.unload()
        assert base is self.model
        del ol, base
        torch.save({"rows": rows, "X": {k: v for k, v in X.items() if k == NLA_L or (isinstance(k, tuple) and k[0] == NLA_L)}},
                   self.dir / "nla_inputs.pt")
        self.state["timings"] = self.timings
        self.save_state()
        self.log("free the subject; load nla-rl")
        del self.model, self.blocks
        self.__dict__.pop("JNP", None)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with PL.Timer(self.timings, "nla_load"):
            nla = self.RD.NLArl()
        self.score_reader("nla", NLA_L, lambda hs, pre, ci: nla.score_next(hs, pre, ci), nla.tok, rows, X)
        nla.close()
        self.state["timings"] = self.timings
        self.state["peak_GB_nla_phase"] = torch.cuda.max_memory_allocated() / 1e9
        self.save_state()
        self.log("step 11 done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="pilot1")
    a = ap.parse_args()
    Step11(a.run_id).run()


if __name__ == "__main__":
    main()
