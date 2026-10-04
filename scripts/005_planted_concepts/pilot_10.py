"""Design 005 §16.6 step 10 (dev-only; pre-registered before it runs): 10a the re-entry clamp's end-layer sweep
(E = 44, 46, ..., 62, each with its random-token null) and 10b the concept specificity of the report measure (ΔR1′ at
C_o of every word in W). Teacher-forced, default prefill, bands B and B2.

  python pilot_10.py --run-id pilot1      (writes into outputs/005_planted_concepts/pilot/<run_id>/step10/)
"""
import argparse
import json
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
SPEC_CONDS = [(1, 0), (2, 0), (3, 0), (4, 0)] + [(7, k) for k in range(5)]


class Step10(PI.Pilot):
    def __init__(self, run_id):
        super().__init__(run_id)
        self.pdir = self.dir
        self.dir = self.pdir / "step10"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = PL.Log(self.dir / "log.txt")
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def capture_material(self):
        self.mat = torch.load(self.pdir / "material.pt")

    def word_set(self):
        """W = the 10 dev concepts + the baseline words whose quote spellings are single tokens (§16.6 10b)."""
        W = {}
        for w in list(P.DEV) + list(self.base):
            try:
                W[w] = P.spellings(self.tok, w)["quote"]
            except AssertionError:
                continue
        assert not set(W) & (set(P.TEST) | set(P.EXCLUDED)) and set(P.DEV) <= set(W)
        return W

    def top10(self, lp_row):
        t = lp_row.topk(10)
        return [[self.tok.decode([int(i)]), float(v)] for i, v in zip(t.indices.tolist(), t.values.exp().tolist())]

    # ------------------------------------------------------------------------------------- 10b specificity
    def specificity(self, bname, band):
        f = self.dir / f"spec_{bname}.json"
        if f.exists():
            self.log("skip specificity", bname)
            return
        W, lpos = self.W, list(range(125, len(self.ids)))
        co = len(lpos) - 1
        clean_w = {w: float(torch.logsumexp(self.lp_clean[co, t], 0)) for w, t in W.items()}
        rows = [{"concept": c, "cond": cond, "s": s, "seed": sd, "band": bname}
                for c in P.DEV for s in PI.GRID9 for cond, sd in SPEC_CONDS]
        for j in range(0, len(rows), PL.TF_CHUNK):
            ch = rows[j:j + PL.TF_CHUNK]
            dl = {l: torch.stack([r["s"] * self.Nbar[l] * self.vec_of(r, l) for r in ch]) for l in band}
            with PL.Timer(self.timings, f"spec_{bname}", len(ch)):
                inj = P.BandInject(self.blocks, dl, P.POS["span"])
                lp, _ = P.forward_rows(self.model, self.ids, len(ch), self.W_U, lpos, writes=[inj])
            for b, r in enumerate(ch):
                r.update(P.intro_metrics(lp[b:b + 1], self.lp_clean, self.sp[r["concept"]])[0])
                r["dW"] = {w: float(torch.logsumexp(lp[b, co, t], 0)) - clean_w[w] for w, t in W.items()}
                r["top10"] = self.top10(lp[b, co])
        PL.jdump(rows, f)
        self.log("specificity done", bname)

    # ------------------------------------------------------------------------------------- 10a end-layer sweep
    def clamp_sweep(self, bname, band):
        f = self.dir / f"clampsweep_{bname}.json"
        if f.exists():
            self.log("skip clamp sweep", bname)
            return
        self.clamp_sets(P.DEV, band)
        full = list(range(band[0], 63))
        rows = [{"concept": c, "cond": 4, "E": None, "kind": None, "s": s, "seed": 0, "band": bname}
                for c in P.DEV for s in PI.GRID9]
        rows += [{"concept": c, "cond": f"E{E}{'r' if k == 'r' else ''}", "E": E, "kind": k, "s": s, "seed": 0,
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
                Qr[b] = {l: self.Q[(r["concept"], r["kind"])][l] for l in layers}
                acc, A = torch.zeros(P.R.D_MODEL, device=self.dev), {}
                for l in layers:
                    if l in dl:
                        acc = acc + dl[l][b]
                    A[l] = acc.clone()
                off[b] = A
            with PL.Timer(self.timings, f"clampsweep_{bname}", len(ch)):
                inj = P.BandInject(self.blocks, dl, P.POS["span"])
                cl = P.BandClamp(self.blocks, Qr, self.hc, full, offsets=off, offset_pos=P.POS["span"])
                lp, _ = P.forward_rows(self.model, self.ids, len(ch), self.W_U, lpos, writes=[inj, cl])
            for b, r in enumerate(ch):
                r.update(P.intro_metrics(lp[b:b + 1], self.lp_clean, self.sp[r["concept"]])[0])
                if b in cl.log:
                    lg = cl.log[b]
                    r["removed_share_band"] = float(np.median([lg[l]["removed_share"] for l in band if l in lg]))
                    above = [v["removed_share"] for l, v in lg.items() if l > band[-1]]
                    r["removed_share_above"] = float(np.median(above)) if above else None
                    r["max_residual"] = float(max(v["residual"] for v in lg.values()))
        PL.jdump(rows, f)
        del self.Q
        torch.cuda.empty_cache()
        self.log("clamp sweep done", bname)

    def run(self):
        self.setup()
        self.capture_material()
        self.ids = P.render_intro(self.tok, "default")
        self.clean()
        self.load_lenses()
        self.log("vectors (dev; identical seeds to the pilot)")
        self.vectors(sorted(set(P.BAND2)), P.DEV)
        self.W = self.word_set()
        self.state["W"] = {w: [self.tok.decode([t]) for t in ts] for w, ts in self.W.items()}
        self.state["clean_top10_at_quote"] = self.top10(self.lp_clean[-1])
        self.save_state()
        for bname, band in PI.BANDS.items():
            self.log("specificity", bname)
            self.specificity(bname, band)
        for bname, band in PI.BANDS.items():
            self.log("clamp sweep", bname)
            self.clamp_sweep(bname, band)
        self.state["timings"] = self.timings
        self.state["peak_GB"] = torch.cuda.max_memory_allocated() / 1e9
        self.save_state()
        self.log("step 10 done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="pilot1")
    a = ap.parse_args()
    Step10(a.run_id).run()


if __name__ == "__main__":
    main()
