"""007 CPU analysis §11.2 (exploratory): the mean-removed token-lens readout on all 48 brew dev tables, against the
whole activation. 002's five readouts (logit; J_NP standard and cosine; J_CB standard; R_CB standard; single-token
spellings, max over a colour's spellings, as 002) plus the template lens (cosine; teal absent), at L40/42/44 and the
three emission cells. Statistics: 002-style closed-set reciprocal rank of z (member path + partner path + WSB options),
z-first share among the colours (member renders), within-table tracking D for z/start/answer (collision-free start and
answer). Gate G-002: whole-activation token-lens scores reproduce 002's colour_scores.json.
CPU only. Reader-side; never imported by label code.
"""
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
import geometry as G                                                       # noqa: E402
import p007 as C                                                           # noqa: E402

torch.set_num_threads(16)
READOUTS = ("logit", "J_NP_standard", "J_CB_standard", "R_CB_standard", "J_NP_cosine")
F002 = C.PROJECT / "results" / "002_claim_scoring" / "lens" / "colour_scores.json"


def main():
    import readers
    import wsb
    from transformers import AutoTokenizer
    log = C.Logger(C.OUT / "token_lens_cpu" / "log.txt")
    snap = next(Path("/workspace/.hf_home/hub/models--Qwen--Qwen3.6-27B/snapshots").iterdir())
    tok = AutoTokenizer.from_pretrained(str(snap))
    cfg = json.loads((snap / "config.json").read_text())
    eps = float(cfg["text_config"].get("rms_norm_eps", 1e-6))
    W, g = G.load_W_and_gain()
    ids, flat = {}, []
    for c in C.COLOURS:
        ids[c] = []
        for form in (" " + c, " " + c.capitalize()):
            t = tok(form, add_special_tokens=False)["input_ids"]
            if len(t) == 1:
                ids[c].append(t[0])
    flat = sorted({i for v in ids.values() for i in v})
    pos = {i: j for j, i in enumerate(flat)}
    Wf = W[flat]

    def per_colour(vals):
        return torch.stack([vals[:, [pos[i] for i in ids[c]]].max(dim=1).values for c in C.COLOURS], dim=1)

    def norm(x):
        x = x.float()
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * g

    def softmax_cols(Hn):
        out = []
        for i in range(0, Hn.shape[0], 64):
            lg = Hn[i:i + 64] @ W.T
            out.append(torch.softmax(lg, -1)[:, flat])
        return torch.cat(out)
    lenses = readers.load_lenses(("J_NP", "J_CB", "R_CB"), layers=list(C.LAYERS), device="cpu", verify_sha=True)
    X, keep, t_of, z, pairs_ = G.brew_material()
    roles = json.loads((C.STAGE0 / "roles.json").read_text())
    pairs = roles["pairs"]
    ref002 = json.loads(F002.read_text())
    meta = json.loads(G.CAPS_META.read_text())["renders"]
    nat_idx = [i for i, r in enumerate(meta) if r["role"] in ("member", "partner")]
    _, brew = wsb.load_bank("brew")
    items = {it["name"]: it for it in brew}
    sys.path.insert(0, str(HERE))
    import template_lens as TLm
    tl = TLm.TL("primary", list(C.COLOURS), C.LAYERS)
    res = {"readouts": list(READOUTS) + ["template_lens"], "G_002": {}, "cells": {}}
    worst = 0.0
    n = len(keep)
    for ci, cell in enumerate(("emit_asst", "emit_think", "emit_colon")):
        for li, L in enumerate(C.LAYERS):
            H = torch.from_numpy(np.asarray(X[keep, ci, li], dtype=np.float32))
            Hd = H.double()
            tot = Hd.sum(0)
            mu = torch.empty_like(Hd)
            for tb in range(48):
                m = torch.from_numpy(t_of == tb)
                mu[m] = (tot - Hd[m].sum(0)) / (n - int(m.sum()))
            U = mu / mu.norm(dim=1, keepdim=True)
            MR = (Hd - (Hd * U).sum(1, keepdim=True) * U).float()
            for piece, A in (("whole", H), ("MR", MR)):
                sc = {"logit": per_colour(softmax_cols(norm(A)))}
                for ln in ("J_NP", "J_CB", "R_CB"):
                    sc[f"{ln}_standard"] = per_colour(softmax_cols(norm(lenses[ln].transport(A, L))))
                J = lenses["J_NP"].jacobians[L].float()
                den = (Wf @ J).norm(dim=1).clamp_min(1e-9)
                sc["J_NP_cosine"] = per_colour(((A @ J.T) @ Wf.T) / den)
                sc["template_lens"] = torch.from_numpy(tl.score(A.numpy(), L, list(C.COLOURS)))
                for rd, S in sc.items():
                    S = S.double().numpy()
                    if piece == "whole" and rd != "template_lens":
                        r002 = np.array(ref002["scores"][f"{rd}|{ci}|{C.CAP_LAYERS.index(L)}"])[nat_idx]
                        err = float(np.max(np.abs(S - r002) / np.maximum(np.abs(r002), 1e-3)))
                        res["G_002"][f"{rd}|{cell}|L{L}"] = err
                        worst = max(worst, err)
                    rec = {}
                    for role in ("z", "start", "answer"):
                        v, tb = dstat.pair_D(S, pairs, role)
                        mm, ts = dstat.table_means(v, tb)
                        rec[f"D_{role}"] = AN.summarize(mm, AN.boot_idx(len(ts)))
                    # closed-set reciprocal rank of the member's z (member path + partner path + WSB options)
                    rr, zf = [], []
                    for p in pairs:
                        mrole, prole = p["m_roles"], p["p_roles"]
                        cand = list(dict.fromkeys([mrole["start"], mrole["z"], mrole["answer"], prole["start"], prole["z"],
                                                   prole["answer"]] + items[p["item"]]["options_adjacent"][0]))
                        cand = [c for c in cand if not np.isnan(S[p["m"], dstat.CI[c]])]
                        if mrole["z"] not in cand:
                            continue
                        row = {c: S[p["m"], dstat.CI[c]] for c in cand}
                        order = sorted(cand, key=lambda c: -row[c])
                        rr.append((p["table"], 1.0 / (order.index(mrole["z"]) + 1)))
                    tabs = sorted({t for t, _ in rr})
                    rec["closed_set_RR_z"] = float(np.mean([np.mean([x for t, x in rr if t == tb]) for tb in tabs]))
                    mem = sorted({p["m"] for p in pairs})
                    for i in mem:
                        zc = roles["renders"][i]["z"]
                        if np.isnan(S[i, dstat.CI[zc]]):
                            continue
                        zf.append(int(C.COLOURS[int(np.nanargmax(S[i]))] == zc))
                    rec["z_first_share_members"] = float(np.mean(zf))
                    res["cells"].setdefault(f"{cell}|L{L}", {}).setdefault(rd, {})[piece] = rec
            log(f"{cell} L{L} done")
    res["G_002_worst_rel"] = worst
    res["G_002_pass"] = worst <= 1e-4                                         # design §11.2 (pre-registered)
    C.dump(C.RES / "analysis" / "token_lens_cpu.json", res)
    log("G-002 worst relative error", worst)


if __name__ == "__main__":
    main()
