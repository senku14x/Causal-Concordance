"""Design 005 §16.5 step 9 (dev-only; pre-registered 94be4ec): robustness of F1 (J vs non-J at matched norm) and F2 (the
re-entry clamp is specific only when it reaches L62) to the J-space construction: k = 10, k = 25, LS refit, raw-W_U atoms,
194-word baseline. Band L32-38, default prefill, 9-point grid, teacher-forced.

  python pilot_9.py --run-id pilot1      (writes into outputs/005_planted_concepts/pilot/<run_id>/step9/)
"""
import argparse
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

VARIANTS = {"k10": dict(k=10, refit="nnls", raw=False, base=False),
            "k25": dict(k=25, refit="nnls", raw=False, base=False),
            "ls": dict(k=16, refit="ls", raw=False, base=False),
            "raw": dict(k=16, refit="nnls", raw=True, base=False),
            "base194": dict(k=16, refit="nnls", raw=False, base=True)}
CONDS = [3, 4, 12, "12r", 13, "13r"]


class Step9(PI.Pilot):
    def __init__(self, run_id):
        super().__init__(run_id)
        self.pdir = self.dir
        self.dir = self.pdir / "step9"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = PL.Log(self.dir / "log.txt")
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def capture_material(self):
        self.mat = torch.load(self.pdir / "material.pt")

    def variant_vectors(self, k, refit, raw, base):
        JS = self.JS
        H = self.mat["H_tell"]
        words = self.mat["words"]
        idx = {w: i for i, w in enumerate(words)}
        nconc = len(self.all_concepts)
        base_rows = list(range(nconc, len(words)))
        V = {}
        for l in P.BAND:
            Hl = H[:, PL.L_ALL.index(l)].to(self.dev)
            if base:
                bm = Hl[base_rows].mean(0)
                X = torch.stack([Hl[idx[c]] - bm for c in P.DEV])
            else:
                sumc = Hl[:nconc].sum(0)
                X = torch.stack([Hl[idx[c]] - (sumc - Hl[idx[c]]) / (nconc - 1) for c in P.DEV])
            D = JS.atoms(self.JNP[l], self.W_U, None if raw else self.gain)
            dec = JS.decompose(X, D, k=k, refit=refit, chunk=65536)
            for i, c in enumerate(P.DEV):
                V[(c, l)] = {"vec": {3: P.unit(dec["x_j"][i]), 4: P.unit(dec["x_nonj"][i])}, "support": dec["support"][i],
                             "J_share": float(dec["energy_j"][i])}
            del D
            torch.cuda.empty_cache()
        return V

    def clamp_sets_variant(self, V, raw, tag):
        vocab = self.tok.convert_ids_to_tokens(list(range(len(self.tok))))
        excl = set()
        for c in self.all_concepts:
            for sp in (" " + c, " " + c.capitalize(), c, c.capitalize()):
                t = self.tok.encode(sp, add_special_tokens=False)
                if len(t) == 1:
                    excl.add(t[0])
        pool = [i for i, t in enumerate(vocab) if t is not None and re.fullmatch(r"Ġ[A-Za-z]{3,}", t) and i not in excl]
        Q = {}
        for c in P.DEV:
            Tc = sorted(set(t for l in P.BAND for t in V[(c, l)]["support"]) | set(self.sp[c]["all"]))
            g = torch.Generator().manual_seed(P.seed_of("rand-clamp", c, tag))
            Tr = [pool[j] for j in torch.randperm(len(pool), generator=g)[:len(Tc)].tolist()]
            for kind, T in (("c", Tc), ("r", Tr)):
                q = {}
                for l in range(P.BAND[0], 63):
                    W = self.W_U[T] if raw else self.gain * self.W_U[T]
                    A = (W @ self.JNP[l]).double()
                    Qm, _ = torch.linalg.qr(A.T)
                    q[l] = Qm.float()
                Q[(c, kind)] = q
        return Q

    def run(self):
        self.setup()
        self.capture_material()
        self.ids = P.render_intro(self.tok, "default")
        self.clean()
        self.load_lenses()
        band = list(P.BAND)
        full, lim = list(range(band[0], 63)), list(range(band[0], 45))
        lpos = list(range(125, len(self.ids)))
        for name, cfg in VARIANTS.items():
            f = self.dir / f"rows_{name}.json"
            if f.exists():
                self.log("skip", name)
                continue
            self.log("variant", name, json.dumps(cfg))
            with PL.Timer(self.timings, f"vectors_{name}"):
                V = self.variant_vectors(**cfg)
            Q = self.clamp_sets_variant(V, cfg["raw"], name)
            rows = [{"concept": c, "cond": cond, "s": s, "variant": name} for c in P.DEV for s in PI.GRID9 for cond in CONDS]
            for j in range(0, len(rows), PL.TF_CHUNK):
                ch = rows[j:j + PL.TF_CHUNK]
                dl = {l: torch.stack([r["s"] * self.Nbar[l] * V[(r["concept"], l)]["vec"][3 if r["cond"] == 3 else 4]
                                      for r in ch]) for l in band}
                Qr, off = {}, {}
                for b, r in enumerate(ch):
                    if r["cond"] in (3, 4):
                        continue
                    kind = "r" if str(r["cond"]).endswith("r") else "c"
                    layers = lim if str(r["cond"]).startswith("13") else full
                    Qr[b] = {l: Q[(r["concept"], kind)][l] for l in layers}
                    acc, A = torch.zeros(P.R.D_MODEL, device=self.dev), {}
                    for l in range(band[0], 63):
                        if l in dl:
                            acc = acc + dl[l][b]
                        if l in layers:
                            A[l] = acc.clone()
                    off[b] = A
                with PL.Timer(self.timings, f"rows_{name}", len(ch)):
                    inj = P.BandInject(self.blocks, dl, P.POS["span"])
                    cl = P.BandClamp(self.blocks, Qr, self.hc, full, offsets=off, offset_pos=P.POS["span"])
                    lp, _ = P.forward_rows(self.model, self.ids, len(ch), self.W_U, lpos, writes=[inj, cl])
                for b, r in enumerate(ch):
                    r.update(P.intro_metrics(lp[b:b + 1], self.lp_clean, self.sp[r["concept"]])[0])
            self.state.setdefault("J_share_median", {})[name] = float(np.median([v["J_share"] for v in V.values()]))
            PL.jdump(rows, f)
            del Q, V
            torch.cuda.empty_cache()
            self.log("variant done", name)
        self.state["timings"] = self.timings
        self.save_state()
        self.log("step 9 done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="pilot1")
    a = ap.parse_args()
    Step9(a.run_id).run()


if __name__ == "__main__":
    main()
