"""009 reader phase (design docs/design_specs/008_009_trackb_brewcode.md §2.7; decision rules in
docs/design_specs/009b_reader_phase.md, committed before the first reader score). READER-SIDE code: it imports reader
modules; no 009 label/admission module imports it (tests/test_import_boundary.py).

  python read009.py oracle   --gate-run g2 --cap-run c1 --run-id r1   # subject + Oracle (L40, L44), token lenses, prompt-only
  python read009.py nla      --gate-run g2 --cap-run c1 --run-id r1   # NLA av_base (L42, L44); the subject is not loaded
  python read009.py analysis --gate-run g2 --cap-run c1 --run-id r1   # CPU

Readers score the 12 colours (" c" / " C", full-sequence LSE) after their primary prefixes (Oracle "-", NLA
"<explanation>\\nThe text is about") on SEC captures at the 3 emit cells, for 007's pieces (whole, J, rest_J, R0-R2,
rest_R0; 007's raw-W_U J subspace and random 1,024-dim bases), both tasks.
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
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parent / "007_jsubspace"))
import bank as B                                    # noqa: E402
import c009 as C                                    # noqa: E402
import derisk as DR                                 # noqa: E402

PIECES = ("whole", "J", "rest_J", "R0", "R1", "R2", "rest_R0")
READ = {"oracle": (40, 44), "nla": (42, 44)}
PREFIX = {"oracle": "-", "nla": "<explanation>\nThe text is about"}
PO_PREFILL = "The model is thinking about"
QUAL_N_TABLES = 8
G1_MED, G1_P99 = 0.05, 0.25             # per-reader noise-aware G-score1 (report 25 / A3 convention)
G2_MAX = 1.0


def caps(cap_run):
    out = {}
    for t in C.TASKS:
        d = C.OUT / "captures" / cap_run
        meta = json.loads((d / f"{t}_meta.json").read_text())
        X = np.load(d / f"{t}.npy", mmap_mode="r")
        out[t] = (X, meta)
    return out


def bases():
    import p007 as P7
    from pieces import rand_basis_f32
    b = torch.load(P7.STAGE0 / "bases_f32.pt")
    J = {L: b[f"J_L{L}"].float() for L in (40, 42, 44)}
    Rb = {r: rand_basis_f32(P7.rand_seed("r1024", r), 1024)[0] for r in range(3)}
    return J, Rb


def make_piece(h, name, L, J, Rb):
    """h [n, d] float32 (cpu). 007's operators: projections onto fixed orthonormal bases and their complements."""
    if name == "whole":
        return h
    if name in ("J", "rest_J"):
        Q = J[L]
        p = (h @ Q) @ Q.T
        return p if name == "J" else h - p
    r = int(name[-1])
    Q = Rb[r]
    p = (h @ Q) @ Q.T
    return p if name.startswith("R") else h - p


def cell_index(meta, cell):
    return meta["positions"].index(cell)


def layer_index(meta, L):
    return meta["layers"].index(L)


def qual_rows(meta):
    """Fixed-seed render subset for qualitative samples: QUAL_N_TABLES tables, member + first partner."""
    items = sorted({r["item"] for r in meta["renders"]})
    rng = np.random.default_rng(C.BOOT_SEED)
    pick = set(rng.choice(items, min(QUAL_N_TABLES, len(items)), replace=False).tolist())
    out, seen = [], set()
    for i, r in enumerate(meta["renders"]):
        if r["item"] in pick and (r["role"] == "member" or (r["item"], "p") not in seen):
            out.append(i)
            if r["role"] == "partner":
                seen.add((r["item"], "p"))
    return out


# ------------------------------------------------------------------------------------------------ scoring loops
def score_all(score_fn, CP, J, Rb, layers, out, log, tag):
    """score_fn(hs list of [d] tensors, L) -> [n, 24] log-probs. Saves one npz per (layer, task, cell, piece)."""
    for L in layers:
        for t in C.TASKS:
            X, meta = CP[t]
            li = layer_index(meta, L)
            for cell in C.EMIT_CELLS:
                ci = cell_index(meta, cell)
                H = torch.from_numpy(np.asarray(X[:, ci, li], np.float32))
                for pn in PIECES:
                    f = out / f"ll_{tag}_L{L}_{t}_{cell}_{pn}.npz"
                    if f.exists():
                        continue
                    P = make_piece(H, pn, L, J, Rb)
                    t0 = time.time()
                    LL = score_fn([P[i] for i in range(P.shape[0])], L).numpy()
                    np.savez(f, LL=LL, keys=np.array([r["key"] for r in meta["renders"]]))
                    log(f"{tag} L{L} {t} {cell} {pn}: {LL.shape} {time.time() - t0:.0f}s")


def positive_control(score_fn, layers, J, Rb, owner, tag, state):
    import p007 as P7
    import dstat
    from analysis import pc_stats, pc_pass
    pos = torch.load(P7.POS_ACTS)                                       # [12, (40,42,44,56,60), d]
    lay = [40, 42, 44, 56, 60]
    for L in layers:
        Hn = pos[:, lay.index(L)].float()
        for pn in PIECES:
            P = make_piece(Hn, pn, L, J, Rb)
            LL = dstat.colour_ll(score_fn([P[i] for i in range(12)], L).numpy(), owner)
            st = pc_stats(LL)
            state.setdefault("positive_control", {})[f"{tag}|L{L}|{pn}"] = {**st, "pass": pc_pass(st)}


def gate_score1(fast_fn, slow_fn, CP, L, n=32):
    X, meta = CP["final"]
    H = torch.from_numpy(np.asarray(X[:n, cell_index(meta, "emit_colon"), layer_index(meta, L)], np.float32))
    a = fast_fn([H[i] for i in range(H.shape[0])], L).numpy()
    b = slow_fn([H[i] for i in range(H.shape[0])], L).numpy()
    d = np.abs(a - b).ravel()
    return {"median": float(np.median(d)), "p99": float(np.quantile(d, .99)), "max": float(d.max()), "n_acts": n,
            "pass": bool(np.median(d) <= G1_MED and np.quantile(d, .99) <= G1_P99)}


def prompt_only(model, tok, renders_by_key, CP, out, log):
    """Stock model, WSB prompt_only_summary template, the render up to and including the cell token; teacher-forced
    "The model is thinking about" + colour (007's reader floor)."""
    import registry as R
    from reader_split import P5, WSB_PO, _assemble, _rows_for, spellings
    f = out / "ll_promptonly.npz"
    if f.exists():
        return
    tmpl = json.loads(WSB_PO.read_text())["_source"]["prompt_template"]
    inner = P5.text_model(model)
    W32 = model.get_output_embeddings().weight.detach().float()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    res = {}
    for t in C.TASKS:
        _, meta = CP[t]
        for cell in C.EMIT_CELLS:
            items = []
            for r in meta["renders"]:
                rr = renders_by_key[r["key"]]
                p = rr["cells"][cell]
                content = tmpl.format(token=tok.decode([rr["ids"][p]]), prefix=tok.decode(rr["ids"][:p + 1]))
                text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False) + PO_PREFILL
                base, seqs, owner = spellings(tok, text)
                items.append((base, seqs))
            LL = np.zeros((len(items), 24), np.float32)
            with torch.no_grad():
                for j in range(0, len(items), 16):
                    ch = items[j:j + 16]
                    per = [_rows_for(b, s_, pad) for b, s_ in ch]
                    T = max(len(pr[0][0]) for pr in per)
                    rows, mask, row0, Lps, gms = [], [], [], [], []
                    for (b, s_), (rw, mk, gm, single, multi, stok) in zip(ch, per):
                        row0.append(len(rows))
                        Lps.append(len(b))
                        gms.append(gm)
                        for r_, m_ in zip(rw, mk):
                            rows.append(r_ + [pad] * (T - len(r_)))
                            mask.append(m_ + [0] * (T - len(m_)))
                    x = torch.tensor(rows, device=W32.device)
                    m = torch.tensor(mask, device=W32.device)
                    Hn = inner(input_ids=x, attention_mask=m, use_cache=False).last_hidden_state
                    LL[j:j + len(ch)] = _assemble(Hn, W32, row0, Lps, gms, single, multi, stok, 24).cpu().numpy()
                    del Hn
            res[f"{t}|{cell}"] = LL
            log(f"prompt-only {t} {cell} done")
    np.savez(f, **res, owner=np.array(owner))


def token_lenses(model, tok, CP, out, log):
    """Logit lens and J_NP / J_CB / R_CB cosine scores of the 12 lowercase colour tokens on the whole activation."""
    import readers
    f = out / "ll_tokenlens.npz"
    if f.exists():
        return
    VL = readers.VectorLenses(model, tok, [40, 42, 44])
    cid = torch.tensor([tok.encode(" " + c, add_special_tokens=False)[0] for c in C.COLOURS], device=VL.W.device)
    res = {}
    with torch.no_grad():
        for t in C.TASKS:
            X, meta = CP[t]
            for L in (40, 42, 44):
                li = layer_index(meta, L)
                for cell in C.EMIT_CELLS:
                    H = torch.from_numpy(np.asarray(X[:, cell_index(meta, cell), li], np.float32)).to(VL.W.device)
                    res[f"logit|L{L}|{t}|{cell}"] = torch.log_softmax(VL._n(H) @ VL.W.T, -1)[:, cid].cpu().numpy()
                    for nm in ("J_NP", "J_CB", "R_CB"):
                        Jm = VL.lenses[nm].jacobians[L]
                        key = (nm, L)
                        if key not in VL._den:
                            VL._den[key] = torch.cat([(VL.W[i:i + 32768] @ Jm).norm(dim=1) for i in range(0, VL.W.shape[0], 32768)])
                        s = ((H @ Jm.T) @ VL.W[cid].T) / VL._den[key][cid].clamp_min(1e-9)
                        res[f"{nm}cos|L{L}|{t}|{cell}"] = s.cpu().numpy()
    np.savez(f, **res)
    log("token lenses done")


def stage_oracle(a):
    import registry as R
    import readers
    from reader_split import oracle_fast, spellings
    out = C.OUT / "readers" / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    log = C.Logger(out / "log_oracle.txt")
    state_f = out / "state_oracle.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    tok = R.make_tokenizer()
    CP = caps(a.cap_run)
    J, Rb = bases()
    _, renders = DR.gated_material(a.gate_run)
    rbk = {}
    for r in renders:
        rbk[r["key"]] = {"ids": B.render(tok, r["text"]).ids, "cells": r["cells"]}
    model = R.load_model()
    W32 = model.get_output_embeddings().weight.detach().float()
    prompt_only(model, tok, rbk, CP, out, log)
    token_lenses(model, tok, CP, out, log)
    ol = readers.OracleLens(model, tok)
    base, seqs, owner = spellings(tok, PREFIX["oracle"])

    def fast(hs, L):
        return oracle_fast(ol, hs, L, base, seqs, W32, acts_per_batch=16)

    def slow(hs, L):
        return ol.score_seq(hs, L, base, seqs, W32, acts_per_batch=2)
    for L in READ["oracle"]:
        state.setdefault("G_score1", {})[f"L{L}"] = gate_score1(fast, slow, CP, L)
        oracle_fast(ol, [torch.from_numpy(np.asarray(CP["final"][0][0, -1, layer_index(CP["final"][1], L)], np.float32))],
                    L, base, seqs, W32, check=True)
        state.setdefault("G_score2", {})[f"L{L}"] = {"max": float(ol.last_check), "pass": float(ol.last_check) <= G2_MAX}
        state.setdefault("noact", {})[f"L{L}"] = oracle_fast(ol, [None], L, base, seqs, W32)[0].tolist()
        log(f"oracle L{L} gates", state["G_score1"][f"L{L}"], state["G_score2"][f"L{L}"])
    state["gates_pass"] = all(v["pass"] for v in state["G_score1"].values()) and all(v["pass"] for v in state["G_score2"].values())
    C.dump(state_f, state)
    if not state["gates_pass"]:
        log("STOP GATE: Oracle G-score1/2 failed")
        return
    positive_control(fast, READ["oracle"], J, Rb, owner, "oracle", state)
    C.dump(state_f, state)
    score_all(fast, CP, J, Rb, READ["oracle"], out, log, "oracle")
    qual = []
    for L in READ["oracle"]:
        for t in C.TASKS:
            X, meta = CP[t]
            rows = qual_rows(meta)
            for cell in C.EMIT_CELLS:
                H = torch.from_numpy(np.asarray(X[rows, cell_index(meta, cell), layer_index(meta, L)], np.float32))
                for pn in ("whole", "J", "rest_J"):
                    P = make_piece(H, pn, L, J, Rb)
                    samp = ol.read_batch([P[i] for i in range(P.shape[0])], L, k=2, seed=C.BOOT_SEED)
                    for i, s in zip(rows, samp):
                        r = meta["renders"][i]
                        qual.append({"reader": "oracle", "layer": L, "task": t, "cell": cell, "piece": pn, "key": r["key"],
                                     "role": r["role"], "start": r["start"], "z": r["z"], "f": r["f"], "samples": s})
    (out / "qualitative_oracle.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qual))
    state["done"] = True
    C.dump(state_f, state)
    log("oracle stage done")


def stage_nla(a):
    import readers
    from reader_split import nla_fast, spellings
    out = C.OUT / "readers" / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    log = C.Logger(out / "log_nla.txt")
    state_f = out / "state_nla.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    CP = caps(a.cap_run)
    J, Rb = bases()
    nla = readers.NLArl()
    base, seqs, owner = spellings(nla.tok, PREFIX["nla"])

    def fast(hs, L):
        return nla_fast(nla, hs, base, seqs, acts_per_batch=16)

    def slow(hs, L):
        return nla.score_seq(hs, base, seqs, acts_per_batch=2)
    for L in READ["nla"]:
        state.setdefault("G_score1", {})[f"L{L}"] = gate_score1(fast, slow, CP, L)
        nla_fast(nla, [torch.from_numpy(np.asarray(CP["final"][0][0, -1, layer_index(CP["final"][1], L)], np.float32))],
                 base, seqs, check=True)
        state.setdefault("G_score2", {})[f"L{L}"] = {"max": float(nla.last_check), "pass": float(nla.last_check) <= G2_MAX}
        state.setdefault("noact", {})[f"L{L}"] = nla_fast(nla, [None], base, seqs)[0].tolist()
        log(f"nla L{L} gates", state["G_score1"][f"L{L}"], state["G_score2"][f"L{L}"])
    state["gates_pass"] = all(v["pass"] for v in state["G_score1"].values()) and all(v["pass"] for v in state["G_score2"].values())
    C.dump(state_f, state)
    if not state["gates_pass"]:
        log("STOP GATE: NLA G-score1/2 failed")
        return
    positive_control(fast, READ["nla"], J, Rb, owner, "nla", state)
    C.dump(state_f, state)
    score_all(fast, CP, J, Rb, READ["nla"], out, log, "nla")
    qual = []
    for L in READ["nla"]:
        for t in C.TASKS:
            X, meta = CP[t]
            rows = qual_rows(meta)
            for cell in C.EMIT_CELLS:
                H = torch.from_numpy(np.asarray(X[rows, cell_index(meta, cell), layer_index(meta, L)], np.float32))
                for pn in ("whole", "J", "rest_J"):
                    P = make_piece(H, pn, L, J, Rb)
                    samp = nla.read_batch([P[i] for i in range(P.shape[0])], k=2, seed=C.BOOT_SEED)
                    for i, s in zip(rows, samp):
                        r = meta["renders"][i]
                        qual.append({"reader": "nla", "layer": L, "task": t, "cell": cell, "piece": pn, "key": r["key"],
                                     "role": r["role"], "start": r["start"], "z": r["z"], "f": r["f"], "samples": s})
    (out / "qualitative_nla.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qual))
    state["done"] = True
    C.dump(state_f, state)
    nla.close()
    log("nla stage done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("oracle", "nla", "analysis"))
    ap.add_argument("--gate-run", required=True)
    ap.add_argument("--cap-run", required=True)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    if a.stage == "analysis":
        import read009_analysis
        read009_analysis.main(a)
        return
    {"oracle": stage_oracle, "nla": stage_nla}[a.stage](a)


if __name__ == "__main__":
    main()
