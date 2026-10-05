"""005 follow-ups, GPU part (docs/design_specs/005_followups.md; EXPLORATORY, test outcomes already seen):
A2b the concept's log-prob rise at every prefill position (answer slot vs non-answer positions), A3 the geometry of the
injected directions (cosine with the concept's output direction; own-atom share in the J part), Q-c the logit-lens
direction g * W_U[" c"] vs the pure J-lens atom as a write direction. Band L32-38, default prefill, float32-residual
regime, G0 on every write.

  python followups.py --run-id fu1
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
import p005 as P                                  # noqa: E402
import pilot as PI                                # noqa: E402
import planted as PL                              # noqa: E402
import registry as R                              # noqa: E402

GRID9 = PI.GRID9
BAND = list(P.BAND)
GEO_CONDS = [1, 2, 3, 4, (7, 0), 8, 9]
A2B_CONDS = [(1, 0), (2, 0), (3, 0), (4, 0), (7, 0)]
QC_CONDS = [(1, 0), ("L", 0), (2, 0)] + [(7, k) for k in range(5)]


def qc_concepts():
    drawn = np.random.default_rng(20261004).choice(np.array(P.TEST), 20, replace=False).tolist()
    return list(P.DEV), drawn


class Followups(PI.Pilot):
    def __init__(self, run_id):
        self.run_id = run_id
        self.dir = PL.OUT / "followups" / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = PL.Log(self.dir / "log.txt")
        self.timings, self.mem, self.notes = {}, {}, []
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def g0(self, fids, tag):
        bad = [[tag, l, v] for f in fids for l, v in f.items()
               if not (0.9 <= v["rho_min"] and v["rho_max"] <= 1.1 and v["kappa_min"] >= 0.99)]
        g = self.state.setdefault("G0", {}).setdefault(tag, {"writes_checked": 0, "failures": 0, "examples": []})
        g["writes_checked"] += sum(len(f) for f in fids)
        g["failures"] += len(bad)
        g["examples"] = (g["examples"] + bad)[:5]
        if bad:
            self.save_state()
            raise RuntimeError(f"G0 failed in {tag}: {bad[:3]}")

    def quote_lower(self, c):
        (t,) = [t for t in self.sp[c]["quote"] if self.tok.decode([t]) == c]
        return t

    # ------------------------------------------------------------------------------------------- A3 geometry
    def geometry(self):
        out = {}
        for c in P.CONCEPTS:
            oq = P.unit(self.gain * self.W_U[self.quote_lower(c)])
            osp = P.unit(self.gain * self.W_U[self.sp[c]["space_lower"]])
            t = self.sp[c]["space_lower"]
            rec = {"cos_quote": {}, "cos_space": {}, "own_atom": {}}
            for cond in GEO_CONDS:
                key = str(cond[0]) if isinstance(cond, tuple) else str(cond)
                vs = [self.V[(c, l)]["vec"][cond] for l in BAND]
                rec["cos_quote"][key] = float(np.mean([float(P.unit(v) @ oq) for v in vs]))
                rec["cos_space"][key] = float(np.mean([float(P.unit(v) @ osp) for v in vs]))
            per_l = []
            for l in BAND:
                V = self.V[(c, l)]
                sup, coef = [int(x) for x in V["support"]], [float(x) for x in V["coef"]]
                cf = float(coef[sup.index(t)]) if t in sup else 0.0
                uj_norm = float(np.sqrt(max(V["J_share"], 0.0)) * V["u_norm"])
                per_l.append({"layer": l, "selected": t in sup, "coef": cf,
                              "energy_share": (cf ** 2) / (uj_norm ** 2) if uj_norm > 0 else None,
                              "cos2_uJ_atom": float((V["vec"][3] @ V["vec"][1]) ** 2), "J_share": V["J_share"]})
            rec["own_atom"] = per_l
            out[c] = rec
        PL.jdump(out, self.dir / "a3_geometry.json")
        self.log("A3 geometry done")

    # -------------------------------------------------------------------------------- A2b position profile
    def position_profile(self):
        f = self.dir / "a2b_rows.json"
        if f.exists():
            return
        lpos = list(range(125, len(self.ids)))
        clean = {c: torch.logsumexp(self.lp_clean[:, self.sp[c]["all"]], -1) for c in P.TEST}       # [14]
        rows = [{"concept": c, "cond": cond, "s": s, "seed": sd} for c in P.TEST for s in GRID9 for cond, sd in A2B_CONDS]
        self.log(f"A2b: {len(rows)} rows")
        for j in range(0, len(rows), PL.TF_CHUNK):
            ch = rows[j:j + PL.TF_CHUNK]
            dl = {l: torch.stack([r["s"] * self.Nbar[l] * self.vec_of(r, l) for r in ch]) for l in BAND}
            with PL.Timer(self.timings, "a2b", len(ch)):
                inj = P.BandInject(self.blocks, dl, P.POS["span"])
                lp, _ = P.forward_rows(self.model, self.ids, len(ch), self.W_U, lpos, writes=[inj])
            self.g0([inj.fid], "A2b")
            for b, r in enumerate(ch):
                c = r["concept"]
                prof = torch.logsumexp(lp[b][:, self.sp[c]["all"]], -1) - clean[c].to(lp.device)
                r["delta_by_pos"] = {str(p): float(v) for p, v in zip(lpos, prof)}
                r.update(P.intro_metrics(lp[b:b + 1], self.lp_clean, self.sp[c])[0])
        PL.jdump(rows, f)
        self.log("A2b done")

    # ------------------------------------------------------------------------------------------------ Q-c
    def qc(self):
        f = self.dir / "qc_rows.json"
        if f.exists():
            return
        dev, drawn = qc_concepts()
        self.state["qc_concepts"] = {"dev": dev, "test_drawn": drawn,
                                     "draw": "numpy.random.default_rng(20261004).choice(TEST, 20, replace=False)"}
        rows = [{"concept": c, "cond": cond, "s": s, "seed": sd, "split": "dev" if c in dev else "test"}
                for c in dev + drawn for s in GRID9 for cond, sd in QC_CONDS]
        self.log(f"Q-c: {len(rows)} rows")
        _, fid, _ = self.teacher_forced(rows, BAND, "qc")
        self.g0(fid, "Q-c")
        PL.jdump(rows, f)
        self.log("Q-c done")

    def run(self):
        t0 = time.time()
        self.setup()
        self.mat = torch.load(PL.OUT / "pilot" / "pilot1" / "material.pt")
        self.ids = P.render_intro(self.tok, "default")
        self.clean()
        self.load_lenses()
        self.vectors(BAND, P.CONCEPTS)
        for c in P.CONCEPTS:                                   # the logit-lens direction of the J atom's own token
            u = P.unit(self.gain * self.W_U[self.sp[c]["space_lower"]])
            for l in BAND:
                self.V[(c, l)]["vec"]["L"] = u
        self.geometry()
        self.position_profile()
        self.qc()
        self.state["timings"] = self.timings
        self.state["wall_minutes"] = (time.time() - t0) / 60
        self.save_state()
        self.log(f"follow-ups done ({self.state['wall_minutes']:.1f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="fu1")
    a = ap.parse_args()
    PL.jdump({"stage": "followups", "run_id": a.run_id, "manifest": R.manifest_block(),
              "design": "docs/design_specs/005_followups.md (exploratory)"}, PL.RES / "manifest_followups.json")
    Followups(a.run_id).run()


if __name__ == "__main__":
    main()
