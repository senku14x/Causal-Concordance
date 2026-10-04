"""Design 005 §16.3 step 7b (dev-only; pre-registered c7cde05): the re-entry clamp (full range L62 and band-limited
L44, each with a random-token null), a layer sweep of c's rank at the report position (L32-62: J_NP cosine, logit
lens, R_CB standard), and the R-lens on the pilot's stored captures. Teacher-forced, default prefill, bands B and B2.

  python pilot_7b.py --run-id pilot1        (writes into outputs/005_planted_concepts/pilot/<run_id>/step7b/)
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
import p005 as P                                  # noqa: E402
import pilot as PI                                # noqa: E402
import planted as PL                              # noqa: E402

SWEEP = list(range(32, 63))


class Step7b(PI.Pilot):
    def __init__(self, run_id):
        super().__init__(run_id)
        self.pdir = self.dir                       # the pilot's directory (stored captures)
        self.dir = self.pdir / "step7b"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = PL.Log(self.dir / "log.txt")
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def capture_material(self):
        self.mat = torch.load(self.pdir / "material.pt")

    def load_rlens(self):
        import readers as RD
        with PL.Timer(self.timings, "load_rlens"):
            self.JR = RD.load_lenses(("R_CB",), layers=sorted(set(SWEEP) | set(P.READ_LAYERS)), device=self.dev)["R_CB"].jacobians

    def ranks_at(self, H, l, concepts):
        """Per row: rank of its concept (best single-token form) under J_NP cosine, logit lens and R_CB standard."""
        norm = R_norm(self.model)
        S = {"J_NP": (H @ self.JNP[l].T) @ self.W_U.T / self.raw_den("J_NP", l).clamp_min(1e-9),
             "logit": norm(H).float() @ self.W_U.T,
             "R_CB": norm(H @ self.JR[l].T).float() @ self.W_U.T}
        return [{k: P.rank_of(v[b], self.sp[c]["all"]) for k, v in S.items()} for b, c in enumerate(concepts)]

    # -------------------------------------------------------------------------------------------- 7b.1 re-entry
    def reentry(self, bname, band):
        f = self.dir / f"reentry_{bname}.json"
        if f.exists():
            self.log("skip re-entry", bname)
            return
        self.clamp_sets(P.DEV, band)
        full = list(range(band[0], 63))
        lim = list(range(band[0], 45))
        conds = [4, 10, "10r", 12, "12r", 13, "13r"]
        rows = [{"concept": c, "cond": cond, "s": s, "seed": 0, "band": bname}
                for c in P.DEV for s in PI.GRID9 for cond in conds]
        lpos = list(range(125, len(self.ids)))
        logs = {}
        for j in range(0, len(rows), PL.TF_CHUNK):
            ch = rows[j:j + PL.TF_CHUNK]
            dl = {l: torch.stack([r["s"] * self.Nbar[l] * self.vec_of(dict(r, cond=4), l) for r in ch]) for l in band}
            Qr, off = {}, {}
            for b, r in enumerate(ch):
                if r["cond"] == 4:
                    continue
                kind = "r" if str(r["cond"]).endswith("r") else "c"
                layers = lim if str(r["cond"]).startswith("13") else full
                Qr[b] = {l: self.Q[(r["concept"], kind)][l] for l in layers}
                if str(r["cond"]).startswith(("12", "13")):
                    acc, A = torch.zeros(P.R.D_MODEL, device=self.dev), {}
                    for l in range(band[0], 63):
                        if l in dl:
                            acc = acc + dl[l][b]
                        if l in layers:
                            A[l] = acc.clone()
                    off[b] = A
            with PL.Timer(self.timings, f"reentry_{bname}", len(ch)):
                inj = P.BandInject(self.blocks, dl, P.POS["span"])
                cl = P.BandClamp(self.blocks, Qr, self.hc, full, offsets=off, offset_pos=P.POS["span"])
                lp, _ = P.forward_rows(self.model, self.ids, len(ch), self.W_U, lpos, writes=[inj, cl])
            for b, r in enumerate(ch):
                r.update(P.intro_metrics(lp[b:b + 1], self.lp_clean, self.sp[r["concept"]])[0])
                r["top1_at_quote"] = self.tok.decode([int(lp[b, -1].argmax())])
                if b in cl.log:
                    lg = cl.log[b]
                    r["removed_share_band"] = float(np.median([lg[l]["removed_share"] for l in band if l in lg]))
                    r["removed_share_above"] = float(np.median([v["removed_share"] for l, v in lg.items() if l > band[-1]]))
                    r["max_residual"] = float(max(v["residual"] for v in lg.values()))
        PL.jdump(rows, f)
        del self.Q
        torch.cuda.empty_cache()
        self.log("re-entry done", bname)

    # ------------------------------------------------------------------------------------------- 7b.2 sweep
    def sweep(self, bname, band):
        f = self.dir / f"sweep_{bname}.json"
        if f.exists():
            self.log("skip sweep", bname)
            return
        conds = [(2, 0), (3, 0), (4, 0)] + [(7, k) for k in range(5)]
        rows = [{"concept": c, "cond": cond, "s": s, "seed": sd, "band": bname}
                for c in P.DEV for s in PI.GRID9 for cond, sd in conds]
        rows += [{"concept": c, "cond": "clean", "s": 0.0, "seed": 0, "band": bname} for c in P.DEV]
        for j in range(0, len(rows), PL.TF_CHUNK):
            ch = rows[j:j + PL.TF_CHUNK]
            dl = {l: torch.stack([r["s"] * self.Nbar[l] * (self.vec_of(r, l) if r["cond"] != "clean"
                                                             else torch.zeros(P.R.D_MODEL, device=self.dev))
                                  for r in ch]) for l in band}
            with PL.Timer(self.timings, f"sweep_{bname}", len(ch)):
                inj = P.BandInject(self.blocks, dl, P.POS["span"])
                _, acts = P.forward_rows(self.model, self.ids, len(ch), self.W_U, [self.co], writes=[inj],
                                         record=SWEEP, record_pos=[self.co])
                for l in SWEEP:
                    rk = self.ranks_at(acts[l][:, 0], l, [r["concept"] for r in ch])
                    for r, x in zip(ch, rk):
                        r.setdefault("ranks", {})[l] = x
        PL.jdump(rows, f)
        self.log("sweep done", bname)

    # ------------------------------------------------------------------------------------- 7b.3 R-lens on captures
    def rlens_on_captures(self):
        f = self.dir / "rlens_reads.json"
        if f.exists():
            return
        out = []
        for bname in ("B", "B2"):
            d = torch.load(self.pdir / f"caps_{bname}.pt")
            rows, caps = d["rows"], d["caps"]
            for l in P.READ_LAYERS:
                items = []
                for b, r in enumerate(rows):
                    if r["cond"] in (10, "10r", 11, "11r"):
                        continue
                    items.append(("carried_co", r, caps[l][b, 1]))
                    items.append(("injected_cq", r, caps[l][b, 0]))
                    items.append(("direct_cq", r, self.hc[l][P.POS["cq"]].cpu()
                                  + r["s"] * self.Nbar[l] * self.vec_of(r, l).cpu()))
                for j in range(0, len(items), 256):
                    ch = items[j:j + 256]
                    H = torch.stack([x for _, _, x in ch]).to(self.dev).float()
                    norm = R_norm(self.model)
                    S = norm(H @ self.JR[l].T).float() @ self.W_U.T
                    for i, (kind, r, _) in enumerate(ch):
                        out.append({"band": bname, "layer": l, "kind": kind, "concept": r["concept"], "cond": r["cond"],
                                    "seed": r["seed"], "s": r["s"],
                                    "R_CB_rank": P.rank_of(S[i], self.sp[r["concept"]]["space"])})
        PL.jdump(out, f)
        self.log("R-lens on captures done", len(out))

    def run(self):
        self.setup()
        self.capture_material()
        self.ids = P.render_intro(self.tok, "default")
        self.clean()
        self.load_lenses()
        self.load_rlens()
        layers = sorted(set(P.BAND2) | set(P.READ_LAYERS))
        self.log("vectors (dev; identical seeds to the pilot)")
        self.vectors(layers, P.DEV)
        # where does c's own J atom become its output direction? (cos with g * W_U[c], per layer)
        cosd = {}
        for c in P.DEV:
            t = self.sp[c]["space_lower"]
            u = self.gain * self.W_U[t]
            cosd[c] = {l: float(torch.nn.functional.cosine_similarity(self.atom(t, l), u / u.norm(), dim=0))
                       for l in range(24, 63)}
        self.state["cos_atom_vs_output"] = cosd
        self.save_state()
        for bname, band in PI.BANDS.items():
            self.log("re-entry clamp", bname)
            self.reentry(bname, band)
            self.log("sweep", bname)
            self.sweep(bname, band)
        self.rlens_on_captures()
        self.state["timings"] = self.timings
        self.state["peak_GB"] = torch.cuda.max_memory_allocated() / 1e9
        self.save_state()
        self.log("step 7b done")


def R_norm(model):
    import registry as R
    return R.final_norm(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="pilot1")
    a = ap.parse_args()
    Step7b(a.run_id).run()


if __name__ == "__main__":
    main()
