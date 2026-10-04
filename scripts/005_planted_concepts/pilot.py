"""Design 005 §16: overnight dev-only pilot (PI go 2026-10-04). Exploratory and descriptive; no test-concept outcome.

  python pilot.py --run-id ID [--material-from smoke1]

Phase A (subject): split quality for all 55 concepts; for the 10 dev concepts x bands {B, B2} x prefills {default,
word} x a 9-point grid, every model-side condition (teacher-forced) + sampled naming for conditions 1-4, 7; token
readers + probes and Oracle reads (default prefill). Phase B: NLA reads. Every block is checkpointed to disk.
"""
import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p005 as P                                  # noqa: E402
import planted as PL                              # noqa: E402
import registry as R                              # noqa: E402

GRID9 = [1 / 256, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
READ_S = [1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0]
BANDS = {"B": list(P.BAND), "B2": list(P.BAND2)}
VARIANTS = ["default", "word"]
PL.SAMPLE_CHUNK = 192


class Pilot(PL.Smoke):
    def __init__(self, run_id):
        self.run_id = run_id
        self.dir = PL.OUT / "pilot" / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = PL.Log(self.dir / "log.txt")
        self.timings, self.mem, self.notes = {}, {}, []
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def clean_variants(self):
        self.cv = {}
        for v in VARIANTS:
            self.ids = P.render_intro(self.tok, v)
            self.clean()                                # sets ids-dependent lp_clean, hc, co, Nbar; neutral too
            self.cv[v] = {"ids": self.ids, "lp_clean": self.lp_clean, "hc": self.hc, "co": self.co,
                          "Nbar": dict(self.Nbar)}
        self.state["Nbar_by_variant"] = {v: d["Nbar"] for v, d in self.cv.items()}
        self.state["clean_top5_by_variant"] = {}
        for v, d in self.cv.items():
            self.state["clean_top5_by_variant"][v] = [self.tok.decode([int(t)]) for t in d["lp_clean"][-1].topk(5).indices]
        self.save_state()

    def set_variant(self, v):
        d = self.cv[v]
        self.ids, self.lp_clean, self.hc, self.co = d["ids"], d["lp_clean"], d["hc"], d["co"]
        self.Nbar = self.cv["default"]["Nbar"]         # injection scale: the default prompt's (the span is identical)

    def run(self):
        st = self.state
        self.setup()
        self.capture_material()
        self.clean_variants()
        self.load_lenses()
        torch.cuda.reset_peak_memory_stats()

        # ---- split quality for all 55 concepts (material only) + dev vectors at every needed layer
        layers = sorted(set(P.BAND2) | set(P.READ_LAYERS))
        self.log("vectors: all 55 concepts at", layers[0], "-", layers[-1], "+ read layers")
        with PL.Timer(self.timings, "vectors_all55"):
            self.vectors(layers, P.CONCEPTS)
        g5 = {}
        for (c, l), rk in self.g5.items():
            g5.setdefault(c, {})[l] = rk
        leaky = {}
        for bname, band in BANDS.items():
            leaky[bname] = sorted(c for c in P.CONCEPTS if sum(g5[c][l] <= 10 for l in band) >= 2)
        st["split_quality"] = {"g5_rank": {c: {str(l): r for l, r in d.items()} for c, d in g5.items()},
                               "leaky_rule": "c in J_NP top-10 on pure u_perp at >= 2 layers of the band (§16.1 Q7)",
                               "leaky": leaky,
                               "J_share": {f"{c}|L{l}": v["J_share"] for (c, l), v in self.V.items()},
                               "cos_loo_base": {f"{c}|L{l}": v["cos_loo_base"] for (c, l), v in self.V.items()},
                               "sensitivity": {f"{c}|L{l}": v for (c, l), v in self.sens.items()}}
        self.save_state()
        self.log("leaky concepts", json.dumps(leaky))
        # free the vectors of test concepts (their condition vectors are never used)
        for k in [k for k in self.V if k[0] not in P.DEV]:
            del self.V[k]
        torch.cuda.empty_cache()

        # ---- model side: bands x prefills
        conds = [(c, 0) for c in (1, 2, 3, 4, 5, 6, 8, 9, 10, "10r")] + [(7, k) for k in range(5)]
        caps_by_band = {}
        for bname, band in BANDS.items():
            self.clamp_sets(P.DEV, band)
            for v in VARIANTS:
                f = self.dir / f"rows_{bname}_{v}.json"
                if f.exists() and (v != "default" or (self.dir / f"caps_{bname}.pt").exists()):
                    self.log("skip (done)", bname, v)
                    if v == "default":
                        caps_by_band[bname] = torch.load(self.dir / f"caps_{bname}.pt")
                    continue
                self.set_variant(v)
                self.log(f"model side band {bname} prefill {v}")
                rows = [{"concept": c, "cond": cond, "s": s, "seed": sd, "band": bname, "prefill": v}
                        for c in P.DEV for s in GRID9 for cond, sd in conds]
                rows += [{"concept": c, "cond": cond, "s": 0.0, "seed": 0, "band": bname, "prefill": v}
                         for c in P.DEV for cond in (11, "11r")]
                caps, fid, clog = self.teacher_forced(rows, band, f"{bname}_{v}")
                self.sample([r for r in rows if r["cond"] in (1, 2, 3, 4) or (r["cond"] == 7 and r["seed"] == 0)],
                            band, f"{bname}_{v}")
                onset_rows = [r for r in rows if r["cond"] == 4 or r["cond"] == 7]
                s_star, trace = P.onset(onset_rows, GRID9)
                st.setdefault("onset", {})[f"{bname}_{v}"] = {"s_star": s_star, "trace": trace}
                PL.jdump(rows, f)
                PL.jdump({"inject": fid, "clamp": clog}, self.dir / f"fidelity_{bname}_{v}.json")
                if v == "default":
                    torch.save({"rows": rows, "caps": caps}, self.dir / f"caps_{bname}.pt")
                    caps_by_band[bname] = {"rows": rows, "caps": caps}
                self.save_state()
                self.log(f"onset {bname} {v}: {s_star}")
            del self.Q
            gc.collect()
            torch.cuda.empty_cache()
        self.set_variant("default")
        self.mem["phaseA_model_side_peak_GB"] = torch.cuda.max_memory_allocated() / 1e9

        # ---- token readers + probes (default prefill, both bands, all kinds)
        self.log("token readers + probes")
        with PL.Timer(self.timings, "probe_training"):
            st["G4_probe_validity"] = self.train_probes(P.DEV)
        clean_x = {(l, k): (self.hc[l][self.co] if k == "carried_co" else self.hc[l][P.POS["cq"]]).cpu()
                   for l in P.READ_LAYERS for k in ("carried_co", "injected_cq", "direct_cq")}
        reads = []
        for bname, d in caps_by_band.items():
            rows, caps = d["rows"], d["caps"]
            for b, r in enumerate(rows):
                if r["cond"] in (10, "10r", 11, "11r"):
                    continue
                for l in P.READ_LAYERS:
                    xs = {"carried_co": caps[l][b, 1], "injected_cq": caps[l][b, 0],
                          "direct_cq": self.hc[l][P.POS["cq"]].cpu() + r["s"] * self.Nbar[l] * self.vec_of(r, l).cpu()}
                    for kind, x in xs.items():
                        reads.append({"band": bname, "concept": r["concept"], "cond": r["cond"], "seed": r["seed"],
                                      "s": r["s"], "layer": l, "kind": kind, "x": x})
        with PL.Timer(self.timings, "token_reads", len(reads)):
            for l in P.READ_LAYERS:
                sel = [q for q in reads if q["layer"] == l]
                for j in range(0, len(sel), 256):
                    ch = sel[j:j + 256]
                    tr = self.token_reads(torch.stack([q["x"] for q in ch]), l, [q["concept"] for q in ch])
                    X = torch.stack([q["x"] for q in ch])
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
        PL.jdump([{k: v for k, v in q.items() if k != "x"} for q in reads], self.dir / "token_probe_reads.json")
        del reads
        gc.collect()

        # ---- Oracle (default prefill): conditions 2,3,4,7(seed 0) x READ_S x {direct, carried C_o} x {L40, L44}
        self.log("Oracle reads")
        items = []
        for bname, d in caps_by_band.items():
            rows, caps = d["rows"], d["caps"]
            for b, r in enumerate(rows):
                if not (r["cond"] in (2, 3, 4) or (r["cond"] == 7 and r["seed"] == 0)) or r["s"] not in READ_S:
                    continue
                for l in (40, 42, 44):
                    xd = self.hc[l][P.POS["cq"]].cpu() + r["s"] * self.Nbar[l] * self.vec_of(r, l).cpu()
                    meta = {k: r[k] for k in ("concept", "cond", "s", "seed", "band")}
                    items.append({"layer": l, "kind": "carried_co", "x": caps[l][b, 1], "meta": meta, "band": bname})
                    items.append({"layer": l, "kind": "direct_cq", "x": xd, "meta": meta, "band": bname})
        nla_items = [q for q in items if q["layer"] == 42]
        torch.save({"items": nla_items, "clean": {"carried_co": self.hc[42][self.co].cpu(),
                                                  "direct_cq": self.hc[42][P.POS["cq"]].cpu()}},
                   self.dir / "nla_inputs.pt")
        if not (self.dir / "oracle_reads.json").exists():
            ol = self.RD.OracleLens(self.model, self.tok)
            oracle = []
            for l in (40, 44):
                sel = [q for q in items if q["layer"] == l]
                for j in range(0, len(sel), PL.READ_CHUNK):
                    ch = sel[j:j + PL.READ_CHUNK]
                    with PL.Timer(self.timings, "oracle_samples", len(ch) * P.TEXT_K):
                        outs = ol.read_batch([q["x"] for q in ch], l, k=P.TEXT_K, seed=P.seed_of("oracle", l, j))
                    for q, smp in zip(ch, outs):
                        oracle.append({"reader": "oracle", "layer": l, "kind": q["kind"], **q["meta"], "samples": smp})
                    if j % (PL.READ_CHUNK * 10) == 0:
                        self.log(f"oracle L{l} {j + len(ch)}/{len(sel)}")
                for kind, x in (("carried_co", self.hc[l][self.co]), ("direct_cq", self.hc[l][P.POS["cq"]])):
                    smp = ol.read_batch([x.cpu()], l, k=P.CLEAN_K, seed=P.seed_of("oracle-clean", l, kind))[0]
                    oracle.append({"reader": "oracle", "layer": l, "kind": kind, "concept": None, "cond": "clean",
                                   "s": 0.0, "samples": smp})
                PL.jdump(oracle, self.dir / "oracle_reads.json")
            base = ol.unload()
            assert base is self.model
            del ol, base
            gc.collect()
        self.mem["phaseA_oracle_peak_GB"] = torch.cuda.max_memory_allocated() / 1e9
        st["timings"], st["memory"], st["notes"] = self.timings, self.mem, self.notes
        self.save_state()

        # ---- SEC l0 = 31 cross-check (band B, default prefill, gate concepts, conditions 1-4 and 7 seed 0, all
        #      strengths): completes the smoke item that was skipped at band B2. Frees every other GPU tensor first.
        if not (self.dir / "sec_crosscheck.json").exists():
            import regime as RG
            rowsB = caps_by_band["B"]["rows"]
            sel = [r for r in rowsB if r["concept"] in P.GATE and r["s"] > 0
                   and (r["cond"] in (1, 2, 3, 4) or (r["cond"] == 7 and r["seed"] == 0))]
            vecs = [{l: (r["s"] * self.Nbar[l] * self.vec_of(r, l)).detach().clone() for l in P.BAND} for r in sel]
            ids, co = list(self.ids), self.co
            for name in ("JNP", "JCB", "den", "W_U", "V", "hc", "hcn", "cv", "lp_clean", "lpn_clean"):
                if hasattr(self, name):
                    delattr(self, name)
            gc.collect()
            torch.cuda.empty_cache()
            self.log(f"SEC cross-check l0 = 31: {len(sel)} rows")
            out = []
            with PL.Timer(self.timings, "sec_build"):
                se = RG.SecEngine(self.model, l0=31)
            try:
                for r, dv in zip(sel, vecs):
                    ed = {}
                    for l, d in dv.items():
                        def fn(x, d=d):
                            x = x.clone()
                            x[P.POS["span"]] = x[P.POS["span"]] + d.to(x.device)
                            return x
                        ed[l] = fn
                    with PL.Timer(self.timings, "sec_forward", 1):
                        fh, _ = se.forward(ids, upper=ed)
                        lp = se.logprobs(fh, [co])[0]
                    q = self.sp[r["concept"]]["quote"]
                    out.append({"concept": r["concept"], "cond": r["cond"], "s": r["s"],
                                "R1_sec": 1.0 / P.rank_of(lp, q), "R1_fp32res": r["R1"],
                                "R1p_sec": float(torch.logsumexp(lp[q], 0)), "R1p_fp32res": r["R1p"],
                                "top1_sec": self.tok.decode([int(lp.argmax())]), "top1_fp32res": r["top1_at_quote"]})
            finally:
                se.restore()
            del se
            gc.collect()
            torch.cuda.empty_cache()
            agree = [x["top1_sec"] == x["top1_fp32res"] and abs(x["R1_sec"] - x["R1_fp32res"]) <= 0.05 for x in out]
            st["SEC_crosscheck"] = {"n": len(out), "agree_frac": float(np.mean(agree)) if agree else None,
                                    "median_abs_dR1p": float(np.median([abs(x["R1p_sec"] - x["R1p_fp32res"])
                                                                        for x in out])) if out else None}
            PL.jdump(out, self.dir / "sec_crosscheck.json")
            self.save_state()
            self.log("SEC", json.dumps(st["SEC_crosscheck"]))

        # ---- phase B: NLA
        self.log("phase B: free subject, load nla-rl")
        del self.model, self.blocks
        for name in ("JNP", "JCB", "den", "W_U", "V", "hc", "hcn", "cv"):
            if hasattr(self, name):
                delattr(self, name)
        gc.collect()
        torch.cuda.empty_cache()
        PL.nla_phase(self.dir, None, self.log)
        self.log("pilot done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    ap.add_argument("--material-from", default="smoke1")
    a = ap.parse_args()
    p = Pilot(a.run_id)
    src = PL.OUT / "smoke" / a.material_from / "material.pt"
    if src.exists() and not (p.dir / "material.pt").exists():
        shutil.copy(src, p.dir / "material.pt")          # identical renders and model: reuse the smoke's captures
    PL.jdump({"stage": "pilot", "run_id": a.run_id, "manifest": R.manifest_block(),
              "design": "docs/design_specs/005_planted_concepts.md §16", "authorised": "PI 2026-10-04 (dev-only pilot)",
              "material_from": a.material_from}, PL.RES / "manifest_pilot.json")
    p.run()


if __name__ == "__main__":
    main()
