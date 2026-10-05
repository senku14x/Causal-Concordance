"""007 template lens (design §3.7 + §1 O4; secondary reader, outside the Holm family). CPU only. Reader-side.

Artifact: camilablank/workspace-lenses @ d740106d, qwen3.6-27b/template-lens/ (sha256-pinned in registry.json). Score of
an activation h for word w at layer l: cos(h, T[l, w]) (the authors' README: "scored by cosine of the per-layer
residual against each template direction"; metadata space: raw). Only the needed rows of each needed layer are kept.

  python template_lens.py brew --stage stage1a        # every Stage 1a piece at L40/42/44, + the sanity check
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
import analysis as AN                                                      # noqa: E402
import dstat                                                               # noqa: E402
import p007 as C                                                           # noqa: E402

SNAP = Path("/workspace/.hf_home/hub/models--camilablank--workspace-lenses/snapshots/"
            "d740106d1e0f95456dc8718fba2895e9c8ffd6ef/qwen3.6-27b/template-lens")
VARIANTS = {"primary": ("templates.safetensors", "template_words.txt"),
            "phrases_v3": ("templates+phrases_v3.safetensors", "template_words+phrases_v3.txt")}
PIN = {"templates.safetensors": "2368c2ed10537b1e2b07af72c7973b93807484abfcf193c150c83206ce93ea98",
       "template_words.txt": "5b247e71def1a12689cf6f92f5838b24ae39f2275c66992ffe028c51be63508d",
       "templates+phrases_v3.safetensors": "c8529864c78b081867450bbad274bc4f2863f22fa8d196ae23bcb8962158cf91",
       "template_words+phrases_v3.txt": "77e81a0374ee20916d65f467b4db186d0026cd681184c91a6b6679f04cd3dc12"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 24), b""):
            h.update(ch)
    return h.hexdigest()


def verify():
    out = {}
    for f, want in PIN.items():
        got = sha(SNAP / f)
        out[f] = {"sha256": got, "ok": got == want}
        assert got == want, (f, got)
    return out


class TL:
    def __init__(self, variant, words, layers):
        from safetensors import safe_open
        fs, fw = VARIANTS[variant]
        rows = [l.split("\t", 1) for l in (SNAP / fw).read_text().splitlines() if "\t" in l]
        text = [t for _, t in rows]
        low = {}
        for i, t in enumerate(text):
            low.setdefault(t.lower(), i)
        self.row = {w: low.get(w.lower()) for w in words}
        self.words = [w for w in words if self.row[w] is not None]
        self.T = {}
        with safe_open(str(SNAP / fs), framework="pt") as f:
            lay = f.get_tensor("layers").tolist()
            sl = f.get_slice("templates")
            for L in layers:
                li = lay.index(L)
                M = sl[li].float()                                          # [n_rows, d] one layer
                idx = torch.tensor([self.row[w] for w in self.words])
                T = M[idx]
                self.T[L] = T / T.norm(dim=1, keepdim=True)
                del M

    def score(self, H, L, words):
        """H [n, d] -> [n, len(words)] cosines (NaN for out-of-vocabulary words)."""
        H = torch.as_tensor(np.asarray(H), dtype=torch.float32)
        Hn = H / H.norm(dim=1, keepdim=True).clamp_min(1e-12)
        S = (Hn @ self.T[L].T).numpy()
        out = np.full((H.shape[0], len(words)), np.nan)
        for j, w in enumerate(words):
            if w in self.words:
                out[:, j] = S[:, self.words.index(w)]
        return out


def brew(stage):
    cname = {"stage1a": "colon", "stage1b": "think", "brew_test": "colon"}[stage]
    res = {"stage": stage, "scoring": "cos(h, T[l, w]) (README; space raw); teal absent from the vocabulary",
           "pins": verify(), "variants": {}}
    roles = json.loads((C.OUT / "brew_test" / "roles.json" if stage == "brew_test" else C.STAGE0 / "roles.json").read_text())
    pairs = roles["pairs"]
    pcs = torch.load(C.POS_ACTS).numpy()
    u_gen = torch.load(C.OUT / "stage1a" / "u_gen.pt")
    perms = AN.derangements(len({p["table"] for p in pairs}))
    for variant in VARIANTS:
        tl = TL(variant, list(C.COLOURS), C.LAYERS)
        V = {"vocab_missing": [c for c in C.COLOURS if c not in tl.words], "layers": {}}
        for L in C.LAYERS:
            pc = tl.score(pcs[:, C.CAP_LAYERS.index(L)], L, list(C.COLOURS))           # sanity (design §3.7)
            ok = ~np.isnan(pc[0])
            own = [int(np.nanargmax(pc[i]) == i) for i in range(12) if ok[i]]
            src = (C.OUT / "brew_test" / "pieces" / f"pieces_colon_L{L}.npz") if stage == "brew_test" else \
                (C.STAGE0 / f"pieces_{cname}_L{L}.npz")
            P = dict(np.load(src))
            if "GMR" not in P and stage != "brew_test":
                u = u_gen[L].double()
                H = torch.from_numpy(P["whole"]).double()
                P["GMR"] = (H - (H @ u).unsqueeze(1) * u.unsqueeze(0)).float().numpy()
            pieces = {}
            raw = {}
            idx_cache = {}
            for nm, H in P.items():
                S = tl.score(H, L, list(C.COLOURS))
                raw[nm] = S.astype(np.float32)
                rec = {}
                for role in ("z", "start", "answer", "off"):
                    v, tb = dstat.pair_D(S, pairs, role)                    # NaN where a role colour is teal
                    m, ts = dstat.table_means(v, tb)
                    key = (role, tuple(ts))
                    idx_cache.setdefault(key, AN.boot_idx(len(ts)))
                    rec[role] = AN.summarize(m, idx_cache[key])
                    rec[role]["n_pairs_scorable"] = int((~np.isnan(v)).sum())
                    nul = AN.deranged_D(np.nan_to_num(S, nan=np.nan), pairs, role, perms)
                    rec[role]["derangement_null"] = {"p2.5": float(np.nanquantile(nul, .025)), "p97.5": float(np.nanquantile(nul, .975))}
                    rec[role]["_tm"] = m.tolist()
                # z-first share among the 11 colours (member renders) and 006's 5-way forced choice
                # ({start, z, answer} + the table's off-trajectory pair; teal candidates dropped)
                zfirst, fc = [], []
                first_pair = {}
                for p in pairs:
                    first_pair.setdefault(p["m"], p)
                for i, p in first_pair.items():
                    zc = roles["renders"][i]["z"]
                    if zc == "teal":
                        continue
                    row = S[i]
                    order = np.argsort(-np.nan_to_num(row, nan=-9))
                    zfirst.append(int(C.COLOURS[order[0]] == zc))
                    cand = [roles["renders"][i][k] for k in ("start", "z", "answer")] + list(p["off"] or [])
                    cand = [c for c in cand if c != "teal"]
                    sc = {c: row[dstat.CI[c]] for c in cand}
                    fc.append(int(max(sc, key=sc.get) == zc))
                rec["z_first_share_members"] = float(np.mean(zfirst)) if zfirst else None
                rec["forced_choice_z_members"] = float(np.mean(fc)) if fc else None
                pieces[nm] = rec
            # contrasts P1-P5 (descriptive; cosine units, no delta, no Holm)
            con = {}
            rand = [n for n in P if n.startswith("R") and n[1:].isdigit()]
            defs = {"P1": (["J"], ["PCA"]), "P2": (["J"], rand), "P3": ([n for n in P if n.startswith("rest_R")], ["rest_J"]),
                    "P4": (["MR"], ["whole"]), "P5": (["J_c"], [n for n in P if n.startswith("R") and n.endswith("_c")])}
            for k, (a, b) in defs.items():
                if not a or not b or not all(n in pieces for n in a + b):
                    continue
                A = np.mean([pieces[n]["z"]["_tm"] for n in a], 0)
                B_ = np.mean([pieces[n]["z"]["_tm"] for n in b], 0)
                c = AN.contrast(A, B_, AN.boot_idx(len(A)), delta=np.inf)
                con[k] = {kk: c[kk] for kk in ("estimate", "ci", "n")}
            for rec in pieces.values():
                for role in ("z", "start", "answer", "off"):
                    rec[role].pop("_tm", None)
            wz, rz = pieces.get("whole", {}).get("z", {}).get("mean"), pieces.get("rest_J", {}).get("z", {}).get("mean")
            V["layers"][L] = {"sanity": {"own_colour_first": int(sum(own)), "of": int(len(own)), "pass": bool(sum(own) >= 9)},
                              "pieces": pieces, "contrasts_descriptive": con,
                              "rest_J_over_whole_Dz": (rz / wz) if (wz not in (None, 0) and rz is not None) else None}
            np.savez(C.OUT / "template_lens" / f"scores_{stage}_{variant}_L{L}.npz", **raw)
        res["variants"][variant] = V
    C.dump(C.RES / "analysis" / f"template_lens_{stage}.json", res)
    for variant, V in res["variants"].items():
        for L, X in V["layers"].items():
            print(variant, L, X["sanity"], {k: round(v["estimate"], 4) for k, v in X["contrasts_descriptive"].items()},
                  "restJ/whole", None if X["rest_J_over_whole_Dz"] is None else round(X["rest_J_over_whole_Dz"], 3))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["brew", "verify"])
    ap.add_argument("--stage", default="stage1a")
    a = ap.parse_args()
    (C.OUT / "template_lens").mkdir(parents=True, exist_ok=True)
    if a.what == "verify":
        print(verify())
    else:
        brew(a.stage)


if __name__ == "__main__":
    main()
