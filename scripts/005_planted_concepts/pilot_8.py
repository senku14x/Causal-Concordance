"""Design 005 §16.4 step 8 (dev-only; pre-registered bf1da37): where the planted concept reaches the report position.

  python pilot_8.py --run-id pilot1      (writes into outputs/005_planted_concepts/pilot/<run_id>/step8/)

8a: teacher-forced rows (conditions 2, 3, 4, 7 x seeds 0-4; s in {1/256, 1/64, 1/16, 1/4, 1}; bands B, B2; default
prefill) with h(C_o) recorded at L32-62; per-concept probes trained at every layer L32-62 on natural template activations
(stock bf16 captures); Delta-probe per row and layer; Oracle reads at C_o at L48, L52, L56, L60 (+ clean reads).
"""
import argparse
import gc
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

SWEEP = list(range(32, 63))
READ_S = [1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0]
ORACLE_LATE = [48, 52, 56, 60]


class Step8(PI.Pilot):
    def __init__(self, run_id):
        super().__init__(run_id)
        self.pdir = self.dir
        self.dir = self.pdir / "step8"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = PL.Log(self.dir / "log.txt")
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def capture_material(self):
        self.mat = torch.load(self.pdir / "material.pt")

    def probes_all_layers(self):
        """Per dev concept x layer L32-62: final probe fit on the 6 natural templates x 295 words (stock captures)."""
        from joblib import Parallel, delayed
        f = self.dir / "tpl_L32_62.pt"
        words = self.mat["words"]
        if f.exists():
            Ht = torch.load(f)
        else:
            tpl_ids = [P.render_template(self.tok, t, w) for w in words for t in P.TEMPLATES]
            with PL.Timer(self.timings, "capture_templates_L32_62", len(tpl_ids)):
                Ht = P.capture_last(self.model, tpl_ids, SWEEP)          # [words*templates, layers, d]
            torch.save(Ht, f)
        nT = len(P.TEMPLATES)
        tmpl = np.tile(np.arange(nT), len(words))
        jobs = []
        for li, l in enumerate(SWEEP):
            X = Ht[:, li].numpy()
            for c in P.DEV:
                y = np.repeat(np.array([w == c for w in words]), nT).astype(int)
                jobs.append((c, l, X, y))
        with PL.Timer(self.timings, "probe_fits_L32_62", len(jobs)):
            res = Parallel(n_jobs=48, backend="loky")(delayed(PL._probe_job)(X, tmpl, y, None, "final")
                                                      for c, l, X, y in jobs)
        self.probes = {(c, l): m for (c, l, _, _), m in zip(jobs, res)}
        del Ht
        gc.collect()

    def run(self):
        self.setup()
        self.capture_material()
        self.ids = P.render_intro(self.tok, "default")
        self.clean()
        self.load_lenses()
        self.log("vectors (dev; identical seeds to the pilot)")
        self.vectors(sorted(set(P.BAND2)), P.DEV)
        self.log("probes at L32-62")
        self.probes_all_layers()
        conds = [(2, 0), (3, 0), (4, 0)] + [(7, k) for k in range(5)]
        oracle_items = []
        for bname, band in PI.BANDS.items():
            f = self.dir / f"probe_sweep_{bname}.json"
            rows = [{"concept": c, "cond": cond, "s": s, "seed": sd, "band": bname}
                    for c in P.DEV for s in READ_S for cond, sd in conds]
            caps = {l: [] for l in SWEEP}
            self.log(f"forwards {bname}: {len(rows)} rows")
            for j in range(0, len(rows), PL.TF_CHUNK):
                ch = rows[j:j + PL.TF_CHUNK]
                dl = {l: torch.stack([r["s"] * self.Nbar[l] * self.vec_of(r, l) for r in ch]) for l in band}
                with PL.Timer(self.timings, f"forward_{bname}", len(ch)):
                    _, acts = P.forward_rows(self.model, self.ids, len(ch), self.W_U, [self.co],
                                             writes=[P.BandInject(self.blocks, dl, P.POS["span"])],
                                             record=SWEEP, record_pos=[self.co])
                for l in SWEEP:
                    caps[l].append(acts[l][:, 0].cpu())
            caps = {l: torch.cat(v) for l, v in caps.items()}
            with PL.Timer(self.timings, f"probe_scores_{bname}", len(rows)):
                for l in SWEEP:
                    xc = self.hc[l][self.co].cpu()[None]
                    for c in P.DEV:
                        idx = [i for i, r in enumerate(rows) if r["concept"] == c]
                        sc, m = self.probes[(c, l)]
                        ps = m.decision_function(sc.transform(caps[l][idx].numpy()))
                        p0 = float(m.decision_function(sc.transform(xc.numpy()))[0])
                        for i, p in zip(idx, ps):
                            rows[i].setdefault("dprobe", {})[l] = float(p) - p0
            PL.jdump(rows, f)
            for i, r in enumerate(rows):
                if r["cond"] in (2, 3, 4) or (r["cond"] == 7 and r["seed"] == 0):
                    for l in ORACLE_LATE:
                        oracle_items.append({"layer": l, "x": caps[l][i], "meta": {k: r[k] for k in ("concept", "cond", "s", "seed", "band")}})
            del caps
            gc.collect()
            self.log(f"probe sweep {bname} done")
        f = self.dir / "oracle_late.json"
        if not f.exists():
            self.log(f"Oracle late layers: {len(oracle_items)} reads")
            ol = self.RD.OracleLens(self.model, self.tok)
            out = []
            for l in ORACLE_LATE:
                sel = [q for q in oracle_items if q["layer"] == l]
                for j in range(0, len(sel), PL.READ_CHUNK):
                    ch = sel[j:j + PL.READ_CHUNK]
                    with PL.Timer(self.timings, "oracle_late_samples", len(ch) * P.TEXT_K):
                        outs = ol.read_batch([q["x"] for q in ch], l, k=P.TEXT_K, seed=P.seed_of("oracle-late", l, j))
                    for q, smp in zip(ch, outs):
                        out.append({"reader": "oracle", "layer": l, "kind": "carried_co", **q["meta"], "samples": smp})
                smp = ol.read_batch([self.hc[l][self.co].cpu()], l, k=P.CLEAN_K, seed=P.seed_of("oracle-late-clean", l))[0]
                out.append({"reader": "oracle", "layer": l, "kind": "carried_co", "concept": None, "cond": "clean",
                            "s": 0.0, "samples": smp})
                PL.jdump(out, f)
                self.log(f"Oracle L{l} done")
            base = ol.unload()
            assert base is self.model
        self.state["timings"] = self.timings
        self.state["peak_GB"] = torch.cuda.max_memory_allocated() / 1e9
        self.save_state()
        self.log("step 8 done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="pilot1")
    a = ap.parse_args()
    Step8(a.run_id).run()


if __name__ == "__main__":
    main()
