"""Design 006 (docs/design_specs/006_reader_likelihood.md): the readers' full-sequence colour likelihoods at the brew
cells of the saved 001 captures, the scorer checks (G-score1/2) and the positive control (G-pc). Reader-side code.

  python reader_likelihood.py --run-id <id>

Phase A (subject resident): positive-control activations ("Tell me about {colour}", stock forward), then the Oracle
(L40, L44, L56, L60). Phase B (subject freed): the NLA (L42). Every block writes one file and is skipped on a rerun.
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
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parent / "005_planted_concepts"))
import p005 as P5                                 # noqa: E402  (render_tell, capture_last)
import readers as RD                              # noqa: E402
import registry as R                              # noqa: E402

COLOURS = ("black", "blue", "brown", "gold", "gray", "green", "orange", "pink", "purple", "red", "teal", "white")
PREFIX = {"oracle": ["-", "- The colour is"], "nla": ["<explanation>\nThe text is about", "<explanation>\nThe colour is"]}
ORACLE_L, NLA_L = [40, 44, 56, 60], 42
CAP = R.PROJECT / "outputs" / "001_leverage_profile" / "run"
OUT = R.PROJECT / "outputs" / "006_reader_likelihood"


def spellings(tok, prefix):
    """Candidate token sequences (" c", " C") after `prefix`, checked to tokenize as prefix + candidate."""
    base = tok.encode(prefix, add_special_tokens=False)
    seqs, owner = [], []
    for c in COLOURS:
        for form in (" " + c, " " + c.capitalize()):
            full = tok.encode(prefix + form, add_special_tokens=False)
            assert full[:len(base)] == base, (prefix, form)
            seqs.append(full[len(base):])
            owner.append(c)
    return base, seqs, owner


class Run:
    def __init__(self, run_id):
        self.dir = OUT / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.t0 = time.time()

    def log(self, *a):
        msg = f"[{time.strftime('%H:%M:%S', time.gmtime())} +{time.time() - self.t0:6.0f}s] " + " ".join(map(str, a))
        print(msg, flush=True)
        with open(self.dir / "log.txt", "a") as f:
            f.write(msg + "\n")

    def save(self):
        self.state_path.write_text(json.dumps(self.state, indent=1))

    def material(self):
        meta = json.loads((CAP / "captures_brew_meta.json").read_text())
        X = np.load(CAP / "captures_brew.npz")["X"]
        self.cells, self.layers = meta["cells"], meta["layers"]
        self.nat = [n for n, r in enumerate(meta["renders"]) if r["role"] in ("member", "partner")]
        self.renders = [meta["renders"][n] for n in self.nat]
        self.X = torch.tensor(X[self.nat])                             # [231, 3 cells, 5 layers, d]
        self.ref = self.X.mean(0)                                      # dev-mean (no-activation reference)
        assert len(self.nat) == 231 and self.cells == ["emit_asst", "emit_think", "emit_colon"]
        self.state["material"] = {"n_renders": len(self.nat), "cells": self.cells, "layers": self.layers,
                                  "render_keys": [r["key"] for r in self.renders]}
        self.save()

    def acts(self, cell, layer):
        ci, li = self.cells.index(cell), self.layers.index(layer)
        return list(self.X[:, ci, li]) + [self.ref[ci, li]]           # last row = reference

    def score_block(self, name, layer, scorer, tok, pc_acts):
        for k, prefix in enumerate(PREFIX[name]):
            f = self.dir / f"ll_{name}_L{layer}_p{k}.npz"
            if f.exists():
                continue
            base, seqs, owner = spellings(tok, prefix)
            arrs, checks = {}, {}
            for cell in self.cells:
                t = time.time()
                arrs[cell] = scorer(self.acts(cell, layer), base, seqs, True).numpy()
                checks[cell] = self.last_check()
                self.log(f"{name} L{layer} prefix {k} {cell}: {arrs[cell].shape[0]} reads in {time.time() - t:.0f}s "
                         f"(own-logit check {checks[cell]:.3f})")
            arrs["positive"] = scorer(pc_acts, base, seqs, False).numpy()
            np.savez(f, **arrs, owner=np.array(owner))
            self.state.setdefault("G_score2_own_logits_maxabs", {})[f"{name}_L{layer}_p{k}"] = checks
            self.save()

    def g_score1(self, name, layer, seq_scorer, next_scorer, tok, acts):
        """Single-token spellings: score_seq (padded batch) must equal score_next (unpadded) within 1e-3 nats."""
        base, seqs, owner = spellings(tok, PREFIX[name][0])
        single = [i for i, s in enumerate(seqs) if len(s) == 1]
        a = seq_scorer(acts, base, seqs, False)[:, single]
        b = next_scorer(acts, base, [seqs[i][0] for i in single])
        d = float((a - b).abs().max())
        self.state.setdefault("G_score1_seq_vs_next_maxabs", {})[f"{name}_L{layer}"] = d
        self.save()
        self.log(f"G-score1 {name} L{layer}: max |seq - next| = {d:.2e}")
        assert d <= 1e-3, f"G-score1 failed for {name} L{layer}: {d}"

    def run(self):
        self.material()
        tok = R.make_tokenizer()
        model = R.load_model()
        f = self.dir / "positive_acts.pt"
        if f.exists():
            pc = torch.load(f)
        else:
            ids = [P5.render_tell(tok, c) for c in COLOURS]
            pc = P5.capture_last(model, ids, self.layers)                 # [12, 5 layers, d], stock forward
            torch.save(pc, f)
        W32 = model.get_output_embeddings().weight.detach().float()
        ol = RD.OracleLens(model, tok)
        try:
            for layer in ORACLE_L:
                li = self.layers.index(layer)
                self.last_check = lambda: getattr(ol, "last_check", float("nan"))
                if f"oracle_L{layer}" not in self.state.get("G_score1_seq_vs_next_maxabs", {}):
                    self.g_score1("oracle", layer,
                                  lambda hs, b, s, c, l=layer: ol.score_seq(hs, l, b, s, W32, check=c),
                                  lambda hs, b, ci, l=layer: ol.score_next(hs, l, b, ci, W32)[0],
                                  tok, self.acts("emit_colon", layer)[:4])
                self.score_block("oracle", layer, lambda hs, b, s, c, l=layer: ol.score_seq(hs, l, b, s, W32, check=c),
                                 tok, list(pc[:, li]))
        finally:
            base_model = ol.unload()
            assert base_model is model
            del ol, base_model
            gc.collect()
        self.log("phase B: free the subject; load nla-rl")
        del model, W32
        gc.collect()
        torch.cuda.empty_cache()
        nla = RD.NLArl()
        try:
            self.last_check = lambda: getattr(nla, "last_check", float("nan"))
            li = self.layers.index(NLA_L)
            if f"nla_L{NLA_L}" not in self.state.get("G_score1_seq_vs_next_maxabs", {}):
                self.g_score1("nla", NLA_L, lambda hs, b, s, c: nla.score_seq(hs, b, s, check=c),
                              lambda hs, b, ci: nla.score_next(hs, b, ci)[0], nla.tok, self.acts("emit_colon", NLA_L)[:4])
            self.score_block("nla", NLA_L, lambda hs, b, s, c: nla.score_seq(hs, b, s, check=c), nla.tok, list(pc[:, li]))
        finally:
            nla.close()
        self.state["wall_minutes"] = (time.time() - self.t0) / 60
        self.save()
        self.log(f"006 done ({self.state['wall_minutes']:.1f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    a = ap.parse_args()
    res = R.PROJECT / "results" / "006_reader_likelihood"
    res.mkdir(parents=True, exist_ok=True)
    (res / "manifest_run.json").write_text(json.dumps({"stage": "run", "run_id": a.run_id, "manifest": R.manifest_block(),
                                                       "design": "docs/design_specs/006_reader_likelihood.md"}, indent=1))
    Run(a.run_id).run()


if __name__ == "__main__":
    main()
