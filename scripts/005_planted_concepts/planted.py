"""Design 005 pipeline (docs/design_specs/005_planted_concepts.md v2).

  python planted.py preflight            CPU: renders, spellings, material hashes, pools -> results/.../preflight.json
  python planted.py smoke [--run-id ID]  GPU smoke (§11; PI go 2026-10-04): G1 + calibration + onset, end-to-end on
                                         gel / earth, fidelity, determinism, SEC l0 = 31 cross-check, throughput,
                                         memory, raw readouts; phase A (subject) then phase B (NLA). Resumable.

The full run is deliberately not a stage here: it needs the PI's go after the smoke review (design status line).
Reader-side code: never imported by label code.
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p005 as P                     # noqa: E402  (also puts scripts/common on sys.path)
import registry as R                 # noqa: E402

OUT = P.ROOT / "outputs/005_planted_concepts"
RES = P.ROOT / "results/005_planted_concepts"
L_ALL = list(range(24, 45))          # captured "Tell me about" layers (band, fallback band, read layers)
L_J = list(range(24, 63))            # J_NP layers kept (atoms, clamp, G5, token readers)
TF_CHUNK, SAMPLE_CHUNK, READ_CHUNK = 32, 96, 24


# --------------------------------------------------------------------------------------------------- utilities
class Log:
    def __init__(self, path):
        self.path = Path(path)
        self.t0 = time.time()

    def __call__(self, *a):
        msg = f"[{time.strftime('%H:%M:%S')} +{time.time() - self.t0:7.0f}s] " + " ".join(str(x) for x in a)
        print(msg, flush=True)
        with open(self.path, "a") as f:
            f.write(msg + "\n")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __init__(self, store, key, n=None):
        self.store, self.key, self.n = store, key, n

    def __enter__(self):
        sync()
        self.t = time.time()
        return self

    def __exit__(self, *a):
        sync()
        dt = time.time() - self.t
        e = self.store.setdefault(self.key, {"seconds": 0.0, "n": 0})
        e["seconds"] += dt
        e["n"] += self.n or 0


def jdump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def _probe_job(X, tmpl, y, t, kind):
    """One probe fit (joblib worker). kind 'final': fit on all rows, return (scaler, model); else leave template t out
    and return the held-out AUC (None if the held-out labels are constant)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    with threadpool_limits(2):
        if kind == "final":
            sc = StandardScaler().fit(X)
            return sc, LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y)
        tr, te = tmpl != t, tmpl == t
        if y[te].min() == y[te].max():
            return None
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        return float(roc_auc_score(y[te], m.decision_function(sc.transform(X[te]))))


# --------------------------------------------------------------------------------------------------- preflight
def preflight():
    import re
    tok = R.make_tokenizer()
    _, _, concepts, base = P.protocol()
    ids = P.render_intro(tok)
    nid = P.render_neutral(tok)
    sp = {c: P.spellings(tok, c) for c in P.CONCEPTS}
    base_single = [w for w in base if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
    vocab = tok.convert_ids_to_tokens(list(range(len(tok))))
    excl = {t for c in concepts for t in (tok.encode(" " + c, add_special_tokens=False)
                                         + tok.encode(" " + c.capitalize(), add_special_tokens=False))}
    pool = [i for i, t in enumerate(vocab) if t is not None and re.fullmatch(r"Ġ[A-Za-z]{3,}", t) and i not in excl]
    res = {"intro_len": len(ids), "neutral_len": len(nid), "positions": P.POS, "neutral": P.NEUTRAL,
           "n_concepts": len(P.CONCEPTS), "dev": P.DEV, "test": P.TEST, "excluded": P.EXCLUDED,
           "spellings": {c: {k: ([tok.decode([t]) for t in v] if isinstance(v, list) else tok.decode([v]))
                             for k, v in s.items()} for c, s in sp.items()},
           "base_words": len(base), "base_single_token": len(base_single), "clamp_pool": len(pool),
           "tell_lengths": sorted({len(P.render_tell(tok, w)) for w in concepts + base}),
           "protocol_sha256": P.PROTOCOL_SHA}
    jdump(res, RES / "analysis/preflight.json")
    print(json.dumps({k: v for k, v in res.items() if k not in ("spellings", "dev", "test")}, indent=1))
    return res


# ------------------------------------------------------------------------------------------------------- smoke
class Smoke:
    def __init__(self, run_id):
        self.run_id = run_id
        self.dir = OUT / "smoke" / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = Log(self.dir / "log.txt")
        self.timings, self.mem, self.notes = {}, {}, []
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def save_state(self):
        jdump(self.state, self.state_path)

    # ------------------------------------------------------------------------------------------- setup + material
    def setup(self):
        import readers as RD
        self.log("setup: tokenizer, model, renders")
        self.tok = R.make_tokenizer()
        self.model = R.load_model()
        self.blocks = R.blocks(self.model)
        self.dev = next(self.model.parameters()).device
        self.W_U = self.model.get_output_embeddings().weight.detach().float()
        import jspace as JS
        self.JS = JS
        norm = R.final_norm(self.model)
        self.gain = JS.effective_gain(norm, R.D_MODEL, device=self.dev)
        gerr = float((self.gain - (1.0 + norm.weight.float())).abs().max())
        assert gerr < 1e-5, f"effective gain differs from 1 + weight by {gerr}"
        self.state["gain_check_maxabs"] = gerr
        self.ids = P.render_intro(self.tok)
        self.nid = P.render_neutral(self.tok)
        _, _, self.all_concepts, self.base = P.protocol()
        self.sp = {c: P.spellings(self.tok, c) for c in P.CONCEPTS}
        self.RD = RD

    def capture_material(self):
        f = self.dir / "material.pt"
        if f.exists():
            self.mat = torch.load(f)
            self.log("material: loaded")
            return
        self.log("material: 'Tell me about' renders (101 + 194) and probe templates (295 x 6)")
        words = self.all_concepts + self.base
        with Timer(self.timings, "capture_tell", len(words)):
            H = P.capture_last(self.model, [P.render_tell(self.tok, w) for w in words], L_ALL)
        tpl_ids = [P.render_template(self.tok, t, w) for w in words for t in P.TEMPLATES]
        with Timer(self.timings, "capture_templates", len(tpl_ids)):
            Ht = P.capture_last(self.model, tpl_ids, P.READ_LAYERS)
        self.mat = {"words": words, "H_tell": H, "H_tpl": Ht.reshape(len(words), len(P.TEMPLATES), len(P.READ_LAYERS), -1)}
        torch.save(self.mat, f)

    def clean(self):
        """Clean float32-residual forwards: introspection (all positions at L24-62, log-probs at 125-138) and the
        neutral prompt with its clean greedy continuation (all positions at L24-62, log-probs at 18-25)."""
        self.log("clean forwards")
        rec_layers = L_J
        self.co = len(self.ids) - 1
        lpos = list(range(125, len(self.ids)))
        lp, acts = P.forward_rows(self.model, self.ids, 1, self.W_U, lpos, record=rec_layers,
                                  record_pos=list(range(len(self.ids))))
        self.lp_clean = lp[0]
        self.hc = {l: acts[l][0] for l in rec_layers}                        # [139, d]
        self.Nbar = {l: float(self.hc[l][P.POS["span"]].norm(dim=-1).mean()) for l in rec_layers}
        # neutral: clean greedy continuation (cache-free, same regime)
        x = list(self.nid)
        for _ in range(P.NEUTRAL["n_cont"]):
            lp1, _ = P.forward_rows(self.model, x, 1, self.W_U, [len(x) - 1])
            x.append(int(lp1[0, 0].argmax()))
        self.nseq = x
        self.npos = list(range(P.NEUTRAL["T"] - 1, P.NEUTRAL["T"] - 1 + P.NEUTRAL["n_cont"]))
        lpn, actn = P.forward_rows(self.model, self.nseq, 1, self.W_U, self.npos, record=rec_layers,
                                   record_pos=list(range(len(self.nseq))))
        self.lpn_clean = lpn[0]
        self.hcn = {l: actn[l][0] for l in rec_layers}
        self.state["neutral_continuation"] = self.tok.decode(self.nseq[P.NEUTRAL["T"]:])
        self.state["Nbar"] = self.Nbar
        self.state["clean_top5_at_quote"] = [self.tok.decode([int(t)]) for t in self.lp_clean[-1].topk(5).indices]
        self.save_state()

    # ------------------------------------------------------------------------------------------------ vectors
    def load_lenses(self):
        import readers as RD
        self.log("lenses: J_NP L24-62, J_CB at read layers")
        with Timer(self.timings, "load_lenses"):
            self.JNP = RD.load_lenses(("J_NP",), layers=L_J, device=self.dev)["J_NP"].jacobians
            self.JCB = RD.load_lenses(("J_CB",), layers=P.READ_LAYERS, device=self.dev)["J_CB"].jacobians
        self.den = {}

    def raw_den(self, name, l):
        """|J_l^T W_U[t]| per token (the WSB JLens cosine denominator)."""
        key = (name, l)
        if key not in self.den:
            J = (self.JNP if name == "J_NP" else self.JCB)[l]
            self.den[key] = torch.cat([(self.W_U[i:i + 32768] @ J).norm(dim=1) for i in range(0, self.W_U.shape[0], 32768)])
        return self.den[key]

    def atom(self, t, l):
        v = (self.gain * self.W_U[t]) @ self.JNP[l]
        return v / v.norm()

    def vectors(self, layers, concepts):
        """u (leave-one-out primary, 194-word baseline as sensitivity), decompositions at `layers` for `concepts`,
        and every condition vector; sensitivity J shares; G5 ranks. Cached per layer in self.V."""
        JS = self.JS
        if not hasattr(self, "V"):
            self.V, self.sens, self.g5 = {}, {}, {}
        H = self.mat["H_tell"]
        idx = {w: i for i, w in enumerate(self.mat["words"])}
        nconc = len(self.all_concepts)
        base_rows = list(range(nconc, len(self.mat["words"])))
        todo = [l for l in layers if any((c, l) not in self.V for c in concepts)]
        for l in todo:
            li = L_ALL.index(l)
            Hl = H[:, li].to(self.dev)
            sumc = Hl[:nconc].sum(0)
            u_loo = {c: Hl[idx[c]] - (sumc - Hl[idx[c]]) / (nconc - 1) for c in concepts}
            u_base = {c: Hl[idx[c]] - Hl[base_rows].mean(0) for c in concepts}
            with Timer(self.timings, "atoms", 1):
                D = JS.atoms(self.JNP[l], self.W_U, self.gain)
            X = torch.stack([u_loo[c] for c in concepts])
            G = torch.stack([P.gaussian(R.D_MODEL, "rand-perp", c, l, device=self.dev) for c in concepts])
            with Timer(self.timings, "pursuit_primary", len(concepts) * 2):
                dec = JS.decompose(torch.cat([X, G]), D, k=P.K, refit="nnls", chunk=65536)
            n = len(concepts)
            # sensitivity J shares (k 10/25, LS refit, base baseline); raw-W_U atoms below
            with Timer(self.timings, "pursuit_sensitivity", len(concepts) * 4):
                s10 = JS.decompose(X, D, k=10, refit="nnls", chunk=65536)["energy_j"]
                s25 = JS.decompose(X, D, k=25, refit="nnls", chunk=65536)["energy_j"]
                sls = JS.decompose(X, D, k=P.K, refit="ls", chunk=65536)["energy_j"]
                sbase = JS.decompose(torch.stack([u_base[c] for c in concepts]), D, k=P.K, refit="nnls",
                                     chunk=65536)["energy_j"]
            base_single = [self.tok.encode(" " + w, add_special_tokens=False) for w in self.base]
            base_single = [t[0] for t in base_single if len(t) == 1]
            for i, c in enumerate(concepts):
                u = X[i]
                uJ, uP = dec["x_j"][i], dec["x_nonj"][i]
                coefs = sorted(dec["coef"][i], reverse=True)
                g = torch.Generator().manual_seed(P.seed_of("rand-J", c, l))
                toks = [base_single[j] for j in torch.randperm(len(base_single), generator=g)[:max(len(coefs), 1)].tolist()]
                rj = sum(cf * D[t] for cf, t in zip(coefs or [1.0], toks))
                vec = {1: P.unit(D[self.sp[c]["space_lower"]]), 2: P.unit(u), 3: P.unit(uJ), 4: P.unit(uP),
                       5: uJ / u.norm(), 6: uP / u.norm(), 8: P.unit(rj), 9: P.unit(dec["x_nonj"][n + i])}
                for sd in range(5):
                    vec[(7, sd)] = P.unit(P.gaussian(R.D_MODEL, "rand-dir", c, l, sd, device=self.dev))
                self.V[(c, l)] = {"vec": vec, "support": dec["support"][i], "coef": dec["coef"][i],
                                  "support_tokens": [self.tok.decode([t]) for t in dec["support"][i]],
                                  "J_share": float(dec["energy_j"][i]), "u_norm": float(u.norm()),
                                  "cos_loo_base": float(torch.nn.functional.cosine_similarity(u, u_base[c], dim=0))}
                self.sens[(c, l)] = {"k10": float(s10[i]), "k25": float(s25[i]), "ls": float(sls[i]),
                                     "base": float(sbase[i])}
            del D
            torch.cuda.empty_cache()
            with Timer(self.timings, "atoms_raw", 1):
                Draw = JS.atoms(self.JNP[l], self.W_U, None)
            sraw = JS.decompose(X, Draw, k=P.K, refit="nnls", chunk=65536)["energy_j"]
            del Draw
            torch.cuda.empty_cache()
            den = self.raw_den("J_NP", l)
            for i, c in enumerate(concepts):
                self.sens[(c, l)]["raw_atoms"] = float(sraw[i])
                uP = self.V[(c, l)]["vec"][4]
                score = (self.W_U @ (self.JNP[l] @ uP)) / den.clamp_min(1e-9)
                self.g5[(c, l)] = P.rank_of(score, self.sp[c]["space"])
            self.log(f"vectors L{l}: J share median {np.median([self.V[(c, l)]['J_share'] for c in concepts]):.4f}")

    def clamp_sets(self, concepts, band):
        """Q bases for the c-clamp and the random clamp at layers band[0]..62 (design §4)."""
        import re
        vocab = self.tok.convert_ids_to_tokens(list(range(len(self.tok))))
        excl = set()
        for c in self.all_concepts:
            for s in (" " + c, " " + c.capitalize(), c, c.capitalize()):
                t = self.tok.encode(s, add_special_tokens=False)
                if len(t) == 1:
                    excl.add(t[0])
        pool = [i for i, t in enumerate(vocab) if t is not None and re.fullmatch(r"Ġ[A-Za-z]{3,}", t) and i not in excl]
        self.Q = {}
        layers = list(range(band[0], 63))
        for c in concepts:
            Tc = sorted(set(t for l in band for t in self.V[(c, l)]["support"]) | set(self.sp[c]["all"]))
            g = torch.Generator().manual_seed(P.seed_of("rand-clamp", c))
            Tr = [pool[j] for j in torch.randperm(len(pool), generator=g)[:len(Tc)].tolist()]
            for kind, T in (("c", Tc), ("r", Tr)):
                q = {}
                for l in layers:
                    A = torch.stack([(self.gain * self.W_U[t]) @ self.JNP[l] for t in T]).double()
                    Qm, _ = torch.linalg.qr(A.T)
                    q[l] = Qm.float()
                self.Q[(c, kind)] = q
            self.state.setdefault("clamp_sets", {})[c] = {"n_tokens": len(Tc),
                                                          "c_tokens": [self.tok.decode([t]) for t in Tc],
                                                          "r_tokens": [self.tok.decode([t]) for t in Tr]}
        self.save_state()

    # ---------------------------------------------------------------------------------------------- model side
    def vec_of(self, r, l):
        cond = r["cond"]
        if cond in (10, "10r"):
            cond = 4
        if cond in (11, "11r"):
            return torch.zeros(R.D_MODEL, device=self.dev)
        key = (7, r.get("seed", 0)) if cond == 7 else cond
        return self.V[(r["concept"], l)]["vec"][key]

    def deltas(self, rows, band):
        return {l: torch.stack([r["s"] * self.Nbar[l] * self.vec_of(r, l) for r in rows]) for l in band}

    def clamp_rows(self, rows):
        out = {}
        for b, r in enumerate(rows):
            if r["cond"] in (10, 11):
                out[b] = self.Q[(r["concept"], "c")]
            elif r["cond"] in ("10r", "11r"):
                out[b] = self.Q[(r["concept"], "r")]
        return out

    def teacher_forced(self, rows, band, tag):
        """Introspection + neutral teacher-forced forwards for `rows`; metrics added in place; captures returned."""
        caps = {l: [] for l in P.READ_LAYERS}
        fid, clog = [], []
        lpos = list(range(125, len(self.ids)))
        for j in range(0, len(rows), TF_CHUNK):
            chunk = rows[j:j + TF_CHUNK]
            dl = self.deltas(chunk, band)
            Qr = self.clamp_rows(chunk)
            clamp_layers = list(range(band[0], 63))
            with Timer(self.timings, f"tf_{tag}", len(chunk)):
                inj = P.BandInject(self.blocks, dl, P.POS["span"])
                cl = P.BandClamp(self.blocks, Qr, self.hc, clamp_layers) if Qr else None
                lp, acts = P.forward_rows(self.model, self.ids, len(chunk), self.W_U, lpos,
                                          writes=[w for w in (inj, cl) if w], record=P.READ_LAYERS,
                                          record_pos=[P.POS["cq"], self.co])
                injn = P.BandInject(self.blocks, dl, P.NEUTRAL["span"])
                cln = P.BandClamp(self.blocks, Qr, self.hcn, clamp_layers) if Qr else None
                lpn, _ = P.forward_rows(self.model, self.nseq, len(chunk), self.W_U, self.npos,
                                        writes=[w for w in (injn, cln) if w])
            fid.append(inj.fid)
            fid.append(injn.fid)
            if cl:
                clog.append({str(j + b): v for b, v in cl.log.items()})
            for b, r in enumerate(chunk):
                sp = self.sp[r["concept"]]
                r.update(P.intro_metrics(lp[b:b + 1], self.lp_clean, sp)[0])
                r.update(P.neutral_metrics(lpn[b:b + 1], self.lpn_clean, sp)[0])
                r["top1_at_quote"] = self.tok.decode([int(lp[b, -1].argmax())])
                if getattr(self, "W_spec", None):            # full run (A18): Δ log-prob at C_o of every word of W
                    r["dW"] = {w: float(torch.logsumexp(lp[b, -1, t], 0)) - self.clean_w[w] for w, t in self.W_spec.items()}
            for l in P.READ_LAYERS:
                caps[l].append(acts[l].cpu())
        return {l: torch.cat(v) for l, v in caps.items()}, fid, clog

    def sample(self, rows, band, tag, seed_base=0):
        """R2: NAME_SAMPLES samples per row (cache-free loop); naming rate and the raw texts added in place."""
        rep = [r for r in rows for _ in range(P.NAME_SAMPLES)]
        texts = []
        for j in range(0, len(rep), SAMPLE_CHUNK):
            chunk = rep[j:j + SAMPLE_CHUNK]
            dl = {l: torch.stack([r["s"] * self.Nbar[l] * self.vec_of(r, l) for r in chunk]) for l in band}
            with Timer(self.timings, f"sample_{tag}", len(chunk)):
                out = P.sample_rows(self.model, self.ids, self.W_U, dl, P.POS["span"], P.NAME_NEW,
                                    seed=P.seed_of("R2", tag, seed_base, j))
            texts += self.tok.batch_decode(out, skip_special_tokens=True)
        for i, r in enumerate(rows):
            tt = texts[i * P.NAME_SAMPLES:(i + 1) * P.NAME_SAMPLES]
            r["R2"] = float(np.mean([P.named_in_quote(t, r["concept"]) for t in tt]))
            r["R2_texts"] = tt
        return texts

    # --------------------------------------------------------------------------------------------- G1 / onset
    def g1(self, band, grid):
        rows_tf = [{"concept": c, "cond": cond, "s": s, "seed": sd} for c in P.GATE for s in grid
                   for cond, sd in [(1, 0), (2, 0), (3, 0), (4, 0)] + [(7, k) for k in range(5)]]
        self.teacher_forced(rows_tf, band, "g1")
        rows_s = [r for r in rows_tf if r["cond"] in (1, 2) or (r["cond"] == 7 and r["seed"] == 0)]
        self.sample(rows_s, band, "g1")
        for r in rows_tf:
            if "R2" not in r:
                r["R2"] = None
        dec = P.g1_decide([r for r in rows_tf if r["cond"] in (1, 2, 7) and r["R2"] is not None], grid)
        return rows_tf, dec

    # ------------------------------------------------------------------------------------------------- readers
    def token_reads(self, H, l, concepts_of_rows):
        """J_NP / J_CB cosine and logit-lens ranks of each row's concept (and of all 55 concepts) at layer l."""
        H = H.to(self.dev).float()
        out = []
        sp_space = {c: self.sp[c]["space"] for c in P.CONCEPTS}
        norm = R.final_norm(self.model)
        scores = {"J_NP": (H @ self.JNP[l].T) @ self.W_U.T / self.raw_den("J_NP", l).clamp_min(1e-9),
                  "J_CB": (H @ self.JCB[l].T) @ self.W_U.T / self.raw_den("J_CB", l).clamp_min(1e-9),
                  "logit": norm(H).float() @ self.W_U.T}
        if getattr(self, "JR", None) is not None:              # full run (A14): R_CB standard read
            scores["R_CB"] = norm(H @ self.JR[l].T).float() @ self.W_U.T
        sets = [sp_space[cc] for cc in P.CONCEPTS]
        for b, c in enumerate(concepts_of_rows):
            rec = {}
            for name, S in scores.items():
                rk = dict(zip(P.CONCEPTS, P.ranks_many(S[b], sets)))
                rec[name] = {"rank": rk[c], "ranks_all": rk}
            out.append(rec)
        return out

    def train_probes(self, concepts):
        """Per concept and read layer: logistic one-vs-rest on natural template activations; G4 validity
        (leave-one-template-out AUC; sample-level label-permutation control: a Hewitt-Liang type-level control is
        uninformative for single-word identity probes, since any word's identity is decodable). Fits run in parallel."""
        from joblib import Parallel, delayed
        Ht = self.mat["H_tpl"]                        # [words, templates, layers, d]
        words = self.mat["words"]
        nW, nT = Ht.shape[0], Ht.shape[1]
        tmpl = np.tile(np.arange(nT), nW)
        Xs = {l: Ht[:, :, li].reshape(nW * nT, -1).numpy() for li, l in enumerate(P.READ_LAYERS)}
        jobs = []
        for l in P.READ_LAYERS:
            for c in concepts:
                y = np.repeat(np.array([w == c for w in words]), nT).astype(int)
                rng = np.random.default_rng(P.seed_of("perm", c, l) % (2 ** 32))
                yp = rng.permutation(y)
                for t in range(nT):
                    jobs.append((c, l, "loto", t, y))
                    jobs.append((c, l, "perm", t, yp))
                jobs.append((c, l, "final", None, y))
        res = Parallel(n_jobs=48, backend="loky")(delayed(_probe_job)(Xs[l], tmpl, y, t, kind)
                                                  for c, l, kind, t, y in jobs)
        self.probes, val = {}, {}
        for (c, l, kind, t, y), r in zip(jobs, res):
            v = val.setdefault(f"{c}|L{l}", {"loto": [], "perm": []})
            if kind == "final":
                self.probes[(c, l)] = r
            elif r is not None:
                v[kind].append(r)
        return {k: {"loto_auc": float(np.mean(v["loto"])) if v["loto"] else None,
                    "perm_ctrl_auc": float(np.mean(v["perm"])) if v["perm"] else None} for k, v in val.items()}

    def probe_scores(self, H, l, c):
        sc, m = self.probes[(c, l)]
        return m.decision_function(sc.transform(H.float().cpu().numpy()))

    # ------------------------------------------------------------------------------------------------ run it
    def run(self):
        st = self.state
        self.setup()
        self.capture_material()
        self.clean()
        self.load_lenses()
        band, grid = list(P.BAND), list(P.GRID)
        torch.cuda.reset_peak_memory_stats()

        # ---- vectors for the dev concepts at the band and read layers
        self.log("vectors: band + read layers, dev concepts")
        self.vectors(band + P.READ_LAYERS, P.DEV)

        # ---- determinism (one teacher-forced batch twice; one sampling batch twice)
        self.log("determinism")
        r0 = [{"concept": "gel", "cond": c, "s": 0.25, "seed": 0} for c in (1, 2, 3, 4)]
        lpos = list(range(125, 139))
        outs = []
        for _ in range(2):
            dl = self.deltas(r0, band)
            lp, _ = P.forward_rows(self.model, self.ids, len(r0), self.W_U, lpos,
                                   writes=[P.BandInject(self.blocks, dl, P.POS["span"])])
            outs.append(lp.cpu())
        smp = [P.sample_rows(self.model, self.ids, self.W_U, self.deltas(r0, band), P.POS["span"], 4, seed=7)
               for _ in range(2)]
        st["determinism"] = {"teacher_forced_bitwise": bool(torch.equal(outs[0], outs[1])),
                             "sampling_identical": bool(torch.equal(smp[0], smp[1]))}
        self.log("determinism", st["determinism"])
        del outs
        self.save_state()

        # ---- G1 (+ calibration once, + fallback band once)
        self.log("G1 at band", band[0], band[-1], "grid", grid)
        rows_g1, dec = self.g1(band, grid)
        st["G1"] = [{"band": [band[0], band[-1]], "grid": grid, "decision": dec}]
        self.log("G1 decision", json.dumps(dec))
        if dec["calibration"]:
            grid = [s * (4 if dec["calibration"] == "x4" else 0.25) for s in grid]
            self.log("G1 calibration", dec["calibration"], "-> grid", grid)
            rows_g1, dec = self.g1(band, grid)
            st["G1"].append({"band": [band[0], band[-1]], "grid": grid, "decision": dec})
        if not dec["pass"]:
            band = list(P.BAND2)
            self.log("G1 failed at B; fallback band B2", band[0], band[-1])
            self.vectors(band, P.GATE)
            rows_g1, dec = self.g1(band, grid)
            st["G1"].append({"band": [band[0], band[-1]], "grid": grid, "decision": dec})
            if not dec["pass"]:
                self.notes.append("G1 failed at B and B2: stop rule (report contrast floor-limited); the smoke "
                                  "continues for engineering checks only")
        st["final_band"], st["final_grid"] = [band[0], band[-1]], grid
        jdump(rows_g1, self.dir / "rows_g1.json")
        self.save_state()

        # ---- onset on the dev concepts -> text-reader strengths
        self.log("onset: dev concepts, u_perp vs rand-dir")
        self.vectors(band, P.DEV)
        rows_on = [{"concept": c, "cond": cond, "s": s, "seed": sd} for c in P.DEV for s in grid
                   for cond, sd in [(4, 0)] + [(7, k) for k in range(5)]]
        self.teacher_forced(rows_on, band, "onset")
        s_star, trace = P.onset(rows_on, grid)
        ts = P.text_strengths(s_star, grid)
        st["onset"] = {"s_star": s_star, "trace": trace, "text_strengths": ts}
        self.log("onset", s_star, "text strengths", ts)
        jdump(rows_on, self.dir / "rows_onset.json")
        self.save_state()

        # ---- end to end on the smoke concepts: all conditions, model side
        self.log("end-to-end model side: gel, earth")
        self.clamp_sets(P.SMOKE, band)
        conds = [(c, 0) for c in (1, 2, 3, 4, 5, 6, 8, 9, 10, "10r")] + [(7, k) for k in range(5)]
        rows_e = [{"concept": c, "cond": cond, "s": s, "seed": sd} for c in P.SMOKE for s in grid for cond, sd in conds]
        rows_e += [{"concept": c, "cond": cond, "s": 0.0, "seed": 0} for c in P.SMOKE for cond in (11, "11r")]
        caps_e, fid_e, clog_e = self.teacher_forced(rows_e, band, "e2e")
        self.sample([r for r in rows_e if r["cond"] in (1, 2, 3, 4, 5, 6, 8, 9) or (r["cond"] == 7 and r["seed"] == 0)],
                    band, "e2e")
        jdump(rows_e, self.dir / "rows_e2e.json")
        jdump({"inject": fid_e, "clamp": clog_e}, self.dir / "fidelity_e2e.json")
        torch.save(caps_e, self.dir / "caps_e2e.pt")
        self.mem["phaseA_model_side_peak_GB"] = torch.cuda.max_memory_allocated() / 1e9

        # ---- token readers + probes on all three kinds (smoke concepts)
        self.log("token readers + probes")
        with Timer(self.timings, "probe_training", 1):
            st["G4_probe_validity"] = self.train_probes(P.DEV)
        reads = []
        for b, r in enumerate(rows_e):
            if r["cond"] in (10, "10r", 11, "11r"):
                continue
            for li, l in enumerate(P.READ_LAYERS):
                xs = {"carried_co": caps_e[l][b, 1], "injected_cq": caps_e[l][b, 0],
                      "direct_cq": self.hc[l][P.POS["cq"]].cpu() + r["s"] * self.Nbar[l] * self.vec_of(r, l).cpu()}
                for kind, x in xs.items():
                    reads.append({"row": b, "concept": r["concept"], "cond": r["cond"], "seed": r.get("seed", 0),
                                  "s": r["s"], "layer": l, "kind": kind, "x": x})
        clean_x = {(l, k): (self.hc[l][P.POS["co"]] if k == "carried_co" else self.hc[l][P.POS["cq"]]).cpu()
                   for l in P.READ_LAYERS for k in ("carried_co", "injected_cq", "direct_cq")}
        with Timer(self.timings, "token_reads", len(reads)):
            for l in P.READ_LAYERS:
                sel = [q for q in reads if q["layer"] == l]
                for j in range(0, len(sel), 256):
                    ch = sel[j:j + 256]
                    tr = self.token_reads(torch.stack([q["x"] for q in ch]), l, [q["concept"] for q in ch])
                    for q, t in zip(ch, tr):
                        q["token"] = t
                    for q in ch:
                        q["probe"] = float(self.probe_scores(q["x"][None], l, q["concept"])[0])
                        q["probe_clean"] = float(self.probe_scores(clean_x[(l, q["kind"])][None], l, q["concept"])[0])
        jdump([{k: v for k, v in q.items() if k != "x"} for q in reads], self.dir / "token_probe_reads.json")
        # G5 on the dev concepts (pure u_perp at band layers)
        st["G5"] = {f"{c}|L{l}": rk for (c, l), rk in self.g5.items() if l in band}
        st["G3"] = {f"{c}|L{l}": v["J_share"] for (c, l), v in self.V.items()}
        st["sensitivity_J_share"] = {f"{c}|L{l}": v for (c, l), v in self.sens.items()}
        st["cos_loo_vs_base"] = {f"{c}|L{l}": v["cos_loo_base"] for (c, l), v in self.V.items()}
        st["supports"] = {f"{c}|L{l}": list(zip(v["support_tokens"], [round(x, 4) for x in v["coef"]]))
                          for (c, l), v in self.V.items() if c in P.SMOKE}
        self.save_state()

        # ---- Oracle (phase A, subject resident): conditions 2,3,4,7 x text strengths x {direct, carried C_o}
        self.log("Oracle reads")
        text_rows = [b for b, r in enumerate(rows_e) if r["cond"] in (2, 3, 4) or (r["cond"] == 7 and r["seed"] == 0)]
        text_rows = [b for b in text_rows if rows_e[b]["s"] in ts]
        ol = self.RD.OracleLens(self.model, self.tok)
        oracle = []
        for l in (40, 44):
            items = []
            for b in text_rows:
                r = rows_e[b]
                items.append(("carried_co", b, caps_e[l][b, 1]))
                items.append(("direct_cq", b, self.hc[l][P.POS["cq"]].cpu() + r["s"] * self.Nbar[l] * self.vec_of(r, l).cpu()))
            for j in range(0, len(items), READ_CHUNK):
                ch = items[j:j + READ_CHUNK]
                with Timer(self.timings, "oracle_samples", len(ch) * P.TEXT_K):
                    outs = ol.read_batch([x for _, _, x in ch], l, k=P.TEXT_K, seed=P.seed_of("oracle", l, j))
                for (kind, b, _), smp in zip(ch, outs):
                    r = rows_e[b]
                    oracle.append({"reader": "oracle", "layer": l, "kind": kind, "concept": r["concept"],
                                   "cond": r["cond"], "s": r["s"], "samples": smp})
            for kind, x in (("carried_co", self.hc[l][P.POS["co"]]), ("direct_cq", self.hc[l][P.POS["cq"]])):
                with Timer(self.timings, "oracle_samples", P.CLEAN_K):
                    smp = ol.read_batch([x.cpu()], l, k=P.CLEAN_K, seed=P.seed_of("oracle-clean", l, kind))[0]
                oracle.append({"reader": "oracle", "layer": l, "kind": kind, "concept": None, "cond": "clean",
                               "s": 0.0, "samples": smp})
        base = ol.unload()
        assert base is self.model
        del ol, base
        gc.collect()
        jdump(oracle, self.dir / "oracle_reads.json")
        self.mem["phaseA_oracle_peak_GB"] = torch.cuda.max_memory_allocated() / 1e9

        # ---- NLA inputs (stored now; read in phase B)
        nla_items = []
        for b in text_rows:
            r = rows_e[b]
            nla_items.append({"kind": "carried_co", "row": b, "x": caps_e[42][b, 1]})
            nla_items.append({"kind": "direct_cq", "row": b,
                              "x": self.hc[42][P.POS["cq"]].cpu() + r["s"] * self.Nbar[42] * self.vec_of(r, 42).cpu()})
        nla_clean = {"carried_co": self.hc[42][P.POS["co"]].cpu(), "direct_cq": self.hc[42][P.POS["cq"]].cpu()}
        torch.save({"items": nla_items, "clean": nla_clean}, self.dir / "nla_inputs.pt")

        # ---- SEC l0 = 31 cross-check (gate concepts, conditions 1-4 and 7, all strengths); only for band B, whose
        #      writes (L32-38) all lie above l0 = 31
        sec_ok = band[0] > 31
        self.log("SEC cross-check l0 = 31" if sec_ok else "SEC cross-check skipped (fallback band starts below l0)")
        sec_vec = {(r["concept"], r["cond"], r["seed"], r["s"]): r for r in rows_g1}
        V_gate = {k: v for k, v in self.V.items()}
        Nbar = dict(self.Nbar)
        for name in ("JNP", "JCB", "den", "hc", "hcn", "Q"):
            if hasattr(self, name):
                delattr(self, name)
        del self.W_U
        gc.collect()
        torch.cuda.empty_cache()
        import regime as RG
        sec_rows = []
        with Timer(self.timings, "sec_build"):
            se = RG.SecEngine(self.model, l0=31) if sec_ok else None
        try:
            for (c, cond, sd, s), r in (sec_vec.items() if sec_ok else []):
                if cond not in (1, 2, 3, 4) and not (cond == 7 and sd == 0):
                    continue
                key = (7, 0) if cond == 7 else cond
                ed = {}
                for l in band:
                    d = (s * Nbar[l] * V_gate[(c, l)]["vec"][key]).to(self.dev)

                    def fn(x, d=d):
                        x = x.clone()
                        x[P.POS["span"]] = x[P.POS["span"]] + d
                        return x
                    ed[l] = fn
                with Timer(self.timings, "sec_forward", 1):
                    fh, _ = se.forward(self.ids, upper=ed)
                    lp = se.logprobs(fh, [P.POS["co"]])[0]
                q = self.sp[c]["quote"]
                rr = 1.0 / P.rank_of(lp, q)
                sec_rows.append({"concept": c, "cond": cond, "s": s, "R1_sec": rr, "R1_fp32res": r["R1"],
                                 "top1_sec": self.tok.decode([int(lp.argmax())]), "top1_fp32res": r["top1_at_quote"]})
        finally:
            if se is not None:
                se.restore()
        del se
        gc.collect()
        torch.cuda.empty_cache()
        agree = [x["top1_sec"] == x["top1_fp32res"] and abs(x["R1_sec"] - x["R1_fp32res"]) <= 0.05 for x in sec_rows]
        st["SEC_crosscheck"] = {"n": len(sec_rows), "agree_frac": float(np.mean(agree)) if agree else None}
        jdump(sec_rows, self.dir / "sec_crosscheck.json")
        self.save_state()
        self.mem["phaseA_sec_peak_GB"] = torch.cuda.max_memory_allocated() / 1e9

        # ---- phase B: free the subject; NLA reads
        self.log("phase B: free subject, load nla-rl")
        del self.model, self.blocks
        gc.collect()
        torch.cuda.empty_cache()
        st["timings"], st["memory"], st["notes"] = self.timings, self.mem, self.notes
        self.save_state()
        nla_phase(self.dir, rows_e, self.log)
        self.log("smoke done")


def nla_phase(run_dir, rows, log):
    """Phase B (subject not resident): NLA reads of every stored input + clean reads; writes nla_reads.json and adds
    timings / memory to state.json. Usable on its own to resume a run whose phase A finished."""
    import readers as RD
    run_dir = Path(run_dir)
    st = json.loads((run_dir / "state.json").read_text())
    timings, mem = st.get("timings", {}), st.get("memory", {})
    inp = torch.load(run_dir / "nla_inputs.pt")
    items, clean = inp["items"], inp["clean"]
    torch.cuda.reset_peak_memory_stats()
    with Timer(timings, "nla_load"):
        nla = RD.NLArl()
    st["nla_scaling"] = nla.scaling
    out = []
    for j in range(0, len(items), READ_CHUNK):
        ch = items[j:j + READ_CHUNK]
        with Timer(timings, "nla_samples", len(ch) * P.TEXT_K):
            outs = nla.read_batch([q["x"] for q in ch], k=P.TEXT_K, seed=P.seed_of("nla", j))
        for q, smp in zip(ch, outs):
            r = q.get("meta") or rows[q["row"]]
            out.append({"reader": "nla", "layer": 42, "kind": q["kind"], "concept": r["concept"], "cond": r["cond"],
                        "s": r["s"], "seed": r.get("seed", 0), "band": q.get("band", r.get("band")), "samples": smp,
                        "write": nla.last_write})
        log(f"nla {j + len(ch)}/{len(items)}")
    for kind, x in clean.items():
        with Timer(timings, "nla_samples", P.CLEAN_K):
            smp = nla.read_batch([x], k=P.CLEAN_K, seed=P.seed_of("nla-clean", kind))[0]
        out.append({"reader": "nla", "layer": 42, "kind": kind, "concept": None, "cond": "clean", "s": 0.0,
                    "samples": smp})
    mem["phaseB_nla_peak_GB"] = torch.cuda.max_memory_allocated() / 1e9
    nla.close()
    jdump(out, run_dir / "nla_reads.json")
    st["timings"], st["memory"] = timings, mem
    jdump(st, run_dir / "state.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["preflight", "smoke", "nla-resume"])
    ap.add_argument("--run-id", default=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    a = ap.parse_args()
    if a.stage == "preflight":
        preflight()
    elif a.stage == "nla-resume":
        d = OUT / "smoke" / a.run_id
        rows = json.loads((d / "rows_e2e.json").read_text())
        nla_phase(d, rows, Log(d / "log.txt"))
    else:
        s = Smoke(a.run_id)
        meta = {"stage": "smoke", "run_id": a.run_id, "manifest": R.manifest_block(),
                "design": "docs/design_specs/005_planted_concepts.md (v2)", "authorised": "PI 2026-10-04 (smoke only)"}
        jdump(meta, RES / "manifest_smoke.json")
        s.run()


if __name__ == "__main__":
    main()
