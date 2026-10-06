"""009 reader-free de-risk on gated dev pairs (design docs/design_specs/008_009_trackb_brewcode.md §2.5). No reader code.

  python derisk.py outfacing --gate-run g2 --run-id o1      # GPU: stock greedy continuations, then SEC scoring
  python derisk.py captures  --gate-run g2 --run-id c1      # GPU: SEC captures, question span + 3 emit cells, 7 layers
  python derisk.py probes    --gate-run g2 --run-id c1      # CPU: z / f / s probes on the captures of run c1
  python derisk.py profile   --gate-run g2 --run-id p1 [--limit-pairs 48]   # GPU: whole-cell patch share S (p -> m)

Label quantities use the SEC regime (stock bf16 blocks <= 39, float32 replay of blocks 40-63, float32 logits).
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
import bank as B                                    # noqa: E402
import c009 as C                                    # noqa: E402

SUPPRESS = ["<|im_end|>", "<|endoftext|>"]
N_CONT = 24
FLOOR = 2e-4
RANK_MIN = 100
SPELLED_MAX = 0.20
NB = 10000


# --------------------------------------------------------------------------------------------- material
def gated_material(gate_run):
    """(depth, [render dicts]) for every gated dev pair: the member and each gated partner of every table with >= 1 pair,
    under both tasks. A render dict: key, item, role, partner, task, start, z, f, answer, codes, fold, text."""
    S = json.loads((C.RES / "analysis" / f"gate_{gate_run}.json").read_text())
    assert S.get("complete"), f"gate run {gate_run} is not complete"
    depth = S["depth"]
    items = {it["name"]: it for it in B.load("dev", depth)}
    keep = {}
    for p in S["pair_list"]:
        if p["pass"]:
            keep.setdefault(p["item"], []).append(p["partner"])
    out = []
    for name in sorted(keep):
        it = items[name]
        for t in C.TASKS:
            for role, j in [("member", None)] + [("partner", j) for j in sorted(keep[name])]:
                m = it["member"] if role == "member" else it["partners"][j]
                text = it["prompts"][t]["member"] if role == "member" else it["prompts"][t]["partners"][j]
                out.append({"key": f"{name}|{role}|{j}|{t}", "item": name, "role": role, "partner": j, "task": t,
                            "start": m["start"], "z": m["z"], "f": m["f"], "answer": m["answer"][t],
                            "codes": it["codes"], "fold": it["fold"], "text": text, "cells": it["cells"]})
    return depth, out


def pairs_of(renders, task):
    """[(member index, partner index)] into `renders` for one task."""
    idx = {(r["item"], r["role"], r["partner"], r["task"]): i for i, r in enumerate(renders)}
    out = []
    for i, r in enumerate(renders):
        if r["task"] == task and r["role"] == "partner":
            out.append((idx[(r["item"], "member", None, task)], i))
    return out


def colour_spellings(tok):
    """{colour: [token tuples]} for " c" and " C" (" Teal" is two tokens)."""
    out = {}
    for c in C.COLOURS:
        sp = []
        for form in (" " + c, " " + c.capitalize()):
            s = tuple(tok.encode(form, add_special_tokens=False))
            if s not in sp:
                sp.append(s)
        out[c] = sp
    return out


def code_spellings(tok, code):
    sp = []
    for form in (" " + str(code), str(code)):
        s = tuple(tok.encode(form, add_special_tokens=False))
        if s not in sp:
            sp.append(s)
    return sp


def lse(xs):
    xs = np.asarray(xs, np.float64)
    m = xs.max()
    return float(m + np.log(np.exp(xs - m).sum()))


# --------------------------------------------------------------------------------------------- D statistic (006b)
def d_stat(LL, renders, pairs, role, rng_seed=C.BOOT_SEED):
    """006b's D for one role at one slot: per pair [LL_m(t_m) - LL_m(t_p)] - [LL_p(t_m) - LL_p(t_p)], averaged over a
    table's partners, then a 10,000-draw table-cluster bootstrap of the mean. LL: [n_renders, 12] (colour order C.COLOURS)."""
    ci = {c: k for k, c in enumerate(C.COLOURS)}
    per = {}
    for m, p in pairs:
        tm, tp = renders[m][role], renders[p][role]
        if tm == tp:
            continue
        v = (LL[m, ci[tm]] - LL[m, ci[tp]]) - (LL[p, ci[tm]] - LL[p, ci[tp]])
        per.setdefault(renders[m]["item"], []).append(float(v))
    x = np.array([np.mean(v) for v in per.values()])
    if len(x) == 0:
        return {"n": 0}
    rng = np.random.default_rng(rng_seed)
    bm = rng.choice(x, (NB, len(x))).mean(1)
    return {"n_tables": int(len(x)), "n_pairs": int(sum(len(v) for v in per.values())), "mean": float(x.mean()),
            "ci": [float(np.quantile(bm, .025)), float(np.quantile(bm, .975))], "pos": int((x > 0).sum()),
            "neg": int((x < 0).sum())}


def inside(ci, delta=C.DELTA):
    return bool(ci[0] >= -delta and ci[1] <= delta)


# --------------------------------------------------------------------------------------------- outfacing
def stage_outfacing(a):
    import torch
    import registry as R
    import regime as RG
    depth, renders = gated_material(a.gate_run)
    out = C.OUT / "outfacing" / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    log = C.Logger(out / "log.txt")
    tok = R.make_tokenizer()
    for r in renders:
        r["ids"] = B.render(tok, r["text"]).ids
        r["ans_ids"] = list(code_spellings(tok, r["answer"])[0])           # " NN"
        assert len(r["ans_ids"]) == 3 and tok.decode(r["ans_ids"]) == " " + str(r["answer"])
    csp = colour_spellings(tok)
    single = {c: [s[0] for s in sp if len(s) == 1] for c, sp in csp.items()}
    multi = {c: [s for s in sp if len(s) > 1] for c, sp in csp.items()}
    model = R.load_model()
    # ---- Phase A (stock bf16, model resident): greedy continuations after the teacher-forced answer, member renders
    sup = [i for i in (tok.convert_tokens_to_ids(t) for t in SUPPRESS) if isinstance(i, int) and i >= 0]
    mem = [i for i, r in enumerate(renders) if r["role"] == "member"]
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(mem), 32):
            idx = mem[b:b + 32]
            seqs = [renders[i]["ids"] + renders[i]["ans_ids"] for i in idx]
            assert len({len(s) for s in seqs}) == 1
            x = torch.tensor(seqs, device=model.device)
            g = model.generate(x, attention_mask=torch.ones_like(x), do_sample=False, max_new_tokens=N_CONT,
                               suppress_tokens=sup, pad_token_id=tok.eos_token_id)
            for i, row in zip(idx, g):
                renders[i]["cont_ids"] = [int(t) for t in row[x.shape[1]:]]
                renders[i]["cont_text"] = tok.decode(renders[i]["cont_ids"], skip_special_tokens=True)
    log(f"phase A: {len(mem)} continuations in {time.time() - t0:.0f}s")
    # ---- Phase B (SEC): colour log-probs at the slots
    slots = ("emit_asst", "emit_think", "emit_colon", "after_answer")
    LL = np.full((len(renders), len(slots), len(C.COLOURS)), np.nan)
    rank = np.full((len(renders), 2), np.nan)                               # z, f rank at the answer slot (members)
    cont = {}
    t0 = time.time()
    with RG.SecEngine(model, l0=39) as sec, torch.no_grad():
        for n, r in enumerate(renders):
            ids = r["ids"]
            T = len(ids)
            seq = ids + r["ans_ids"] + r.get("cont_ids", [])
            fh, _ = sec.forward(seq)
            pos = {"emit_asst": r["cells"]["emit_asst"], "emit_think": r["cells"]["emit_think"],
                   "emit_colon": T - 1, "after_answer": T + len(r["ans_ids"]) - 1}
            assert pos["emit_colon"] == r["cells"]["emit_colon"]
            lp = sec.logprobs(fh, [pos[s] for s in slots]).double().cpu()
            for si, s in enumerate(slots):
                prefix = seq[:pos[s] + 1]
                for ci_, c in enumerate(C.COLOURS):
                    vals = [float(lp[si, t]) for t in single[c]]
                    for sp in multi[c]:
                        fh2, _ = sec.forward(prefix + list(sp[:-1]))
                        lp2 = sec.logprobs(fh2, range(len(prefix) - 1, len(prefix) - 1 + len(sp))).double().cpu()
                        vals.append(float(sum(lp2[i, t] for i, t in enumerate(sp))))
                    LL[n, si, ci_] = lse(vals)
            if r["role"] == "member":
                lpc = lp[slots.index("emit_colon")]
                for k, role in enumerate(("z", "f")):
                    tid = csp[r[role]][0][0]                                   # " c" (single token for every colour)
                    rank[n, k] = 1 + int((lpc > lpc[tid]).sum())
                if r.get("cont_ids"):
                    cpos = list(range(pos["after_answer"], pos["after_answer"] + len(r["cont_ids"])))
                    lpk = sec.logprobs(fh, cpos).double().cpu()
                    per = {role: [lse([float(lpk[i, t]) for t in single[r[role]]]) for i in range(len(cpos))]
                           for role in ("z", "f")}
                    found = {m_.group(1).lower().replace("grey", "gray") for m_ in B.COLOUR_RE.finditer(r["cont_text"])}
                    spelled = {role: r[role] in found for role in ("z", "f")}
                    cont[r["key"]] = {"text": r["cont_text"], "max": {k: max(v) for k, v in per.items()},
                                      "mean": {k: float(np.mean(v)) for k, v in per.items()}, "spelled": spelled}
            sec.clear()
            if n % 50 == 0:
                log(f"[{n + 1}/{len(renders)}] {time.time() - t0:.0f}s")
    np.savez(out / "ll_slots.npz", LL=LL, rank=rank, keys=np.array([r["key"] for r in renders]), slots=np.array(slots))
    (out / "continuations.json").write_text(json.dumps(cont, indent=1))
    S = summarize_outfacing(renders, LL, rank, cont, slots, depth)
    S.update({"gate_run": a.gate_run, "run_id": a.run_id, "minutes": (time.time() - t0) / 60})
    C.dump(C.RES / "analysis" / "outfacing.json", S)
    log("outfacing", json.dumps(S["pass"], default=float))


def summarize_outfacing(renders, LL, rank, cont, slots, depth):
    S = {"design": "§2.5.1", "depth": depth, "delta": C.DELTA, "rank_min": RANK_MIN, "spelled_max": SPELLED_MAX,
         "D": {}, "rank": {}, "continuation": {}}
    for t in C.TASKS:
        pr = pairs_of(renders, t)
        S["D"][t] = {s: {role: d_stat(LL[:, si, :], renders, pr, role) for role in ("z", "f", "start")}
                     for si, s in enumerate(slots)}
        mem = [i for i, r in enumerate(renders) if r["role"] == "member" and r["task"] == t]
        S["rank"][t] = {role: {"median": float(np.median(rank[mem, k])), "q10": float(np.quantile(rank[mem, k], .1)),
                               "min": float(np.min(rank[mem, k]))} for k, role in enumerate(("z", "f"))}
        ck = [renders[i]["key"] for i in mem if renders[i]["key"] in cont]
        S["continuation"][t] = {
            "n": len(ck),
            "spelled_z_or_f_share": float(np.mean([cont[k]["spelled"]["z"] or cont[k]["spelled"]["f"] for k in ck])) if ck else None,
            "spelled_z_share": float(np.mean([cont[k]["spelled"]["z"] for k in ck])) if ck else None,
            "spelled_f_share": float(np.mean([cont[k]["spelled"]["f"] for k in ck])) if ck else None,
            "max_logprob_median": {role: float(np.median([cont[k]["max"][role] for k in ck])) for role in ("z", "f")} if ck else None,
            "examples": [cont[k]["text"] for k in ck[:8]]}
    f = S["D"]["final"]["emit_colon"]
    S["pass"] = {
        "D_z_answer_slot_inside": inside(f["z"]["ci"]), "D_f_answer_slot_inside": inside(f["f"]["ci"]),
        "rank_z_gt": S["rank"]["final"]["z"]["median"] > RANK_MIN, "rank_f_gt": S["rank"]["final"]["f"]["median"] > RANK_MIN,
        "spelled_le": (S["continuation"]["final"]["spelled_z_or_f_share"] or 0.0) <= SPELLED_MAX}
    S["pass"]["all"] = all(S["pass"].values())
    return S


# --------------------------------------------------------------------------------------------- captures
def capture_positions(cells):
    names = [f"q{i}" for i in range(C.QUESTION_TOKENS)] + list(C.EMIT_CELLS)
    return names, list(cells["question"]) + [cells[c] for c in C.EMIT_CELLS]


def stage_captures(a):
    import torch
    import registry as R
    import regime as RG
    depth, renders = gated_material(a.gate_run)
    out = C.OUT / "captures" / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    log = C.Logger(out / "log.txt")
    tok = R.make_tokenizer()
    model = R.load_model()
    t0 = time.time()
    with RG.SecEngine(model, l0=39) as sec, torch.no_grad():
        for t in C.TASKS:
            rs = [r for r in renders if r["task"] == t]
            X = np.zeros((len(rs), C.QUESTION_TOKENS + len(C.EMIT_CELLS), len(C.CAP_LAYERS), 5120), np.float32)
            names = None
            for n, r in enumerate(rs):
                ids = B.render(tok, r["text"]).ids
                names, pos = capture_positions(r["cells"])
                _, acts = sec.forward(ids, record=list(C.CAP_LAYERS))
                for li, L in enumerate(C.CAP_LAYERS):
                    X[n, :, li] = acts[L][pos].float().cpu().numpy()
                sec.clear()
            np.save(out / f"{t}.npy", X)
            meta = {"task": t, "depth": depth, "positions": names, "layers": list(C.CAP_LAYERS), "regime": "SEC l0=39",
                    "renders": [{k: r[k] for k in ("key", "item", "role", "partner", "start", "z", "f", "answer", "codes",
                                                   "fold")} for r in rs]}
            (out / f"{t}_meta.json").write_text(json.dumps(meta, indent=1))
            log(f"{t}: {len(rs)} renders captured ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------------------------- probes (CPU)
def balanced_accuracy(y, yhat, classes):
    accs = [np.mean(yhat[y == c] == c) for c in classes if np.any(y == c)]
    return float(np.mean(accs))


def probe_one(X, y, folds, items, seed, n_pca=256, C_reg=1.0):
    """Cross-validated 12-way logistic probe (train-fold standardisation and PCA), plus a permuted-label control (train
    labels permuted within the training folds). Returns held-out predictions for both."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    pred, ctrl = np.empty(len(y), dtype=object), np.empty(len(y), dtype=object)
    for f in sorted(set(folds)):
        tr, te = folds != f, folds == f
        sc = StandardScaler().fit(X[tr])
        k = min(n_pca, int(tr.sum()) - 1, X.shape[1])
        pc = PCA(n_components=k, random_state=0).fit(sc.transform(X[tr]))
        Ztr, Zte = pc.transform(sc.transform(X[tr])), pc.transform(sc.transform(X[te]))
        clf = LogisticRegression(C=C_reg, max_iter=3000).fit(Ztr, y[tr])
        pred[te] = clf.predict(Zte)
        yp = y[tr].copy()
        rng.shuffle(yp)
        ctrl[te] = LogisticRegression(C=C_reg, max_iter=3000).fit(Ztr, yp).predict(Zte)
    return pred, ctrl


def ba_boot(y, yhat, items, seed, nb=2000):
    classes = sorted(set(y))
    tabs = sorted(set(items))
    by = {t: np.where(items == t)[0] for t in tabs}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(nb):
        idx = np.concatenate([by[t] for t in rng.choice(tabs, len(tabs))])
        bs.append(balanced_accuracy(y[idx], yhat[idx], classes))
    return [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def probe_stats(X, y, folds, items, seed):
    pred, ctrl = probe_one(X, y, folds, items, seed)
    classes = sorted(set(y))
    ba, bc = balanced_accuracy(y, pred, classes), balanced_accuracy(y, ctrl, classes)
    ci = ba_boot(y, pred, items, seed + 1)
    chance = 1.0 / 12
    return {"ba": ba, "ci": ci, "control_ba": bc, "n": int(len(y)),
            "decodes": bool(ci[0] > chance + 0.15 and bc <= chance + 0.05)}


def _probe_job(path, pi, li, targets, folds, items, seed):
    """One (task, position, layer) slice, loaded from the memory-mapped capture; every target probed on it."""
    X = np.asarray(np.load(path, mmap_mode="r")[:, pi, li], np.float64)
    return {name: probe_stats(X, y, folds, items, seed) for name, y in targets.items()}


def stage_probes(a):
    from joblib import Parallel, delayed
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cap = C.OUT / "captures" / a.run_id
    log = C.Logger(cap / "probes_log.txt")
    res = {"design": "§2.5.3", "captures_run": a.run_id, "rule": "decodes = BA CI lower > 1/12 + 0.15 and control BA <= 1/12 + 0.05",
           "probe": "12-way logistic (C=1) on train-fold standardised PCA-256; 5 table folds; permuted-label control",
           "tasks": {}}
    jobs, keys = [], []
    for t in C.TASKS:
        meta = json.loads((cap / f"{t}_meta.json").read_text())
        rs = meta["renders"]
        folds = np.array([r["fold"] for r in rs])
        items = np.array([r["item"] for r in rs])
        targets = {name: np.array([r[name] for r in rs]) for name in ("z", "f", "start")}
        for pi, pname in enumerate(meta["positions"]):
            for li, L in enumerate(meta["layers"]):
                jobs.append((str(cap / f"{t}.npy"), pi, li, targets, folds, items, C.BOOT_SEED + 1000 * pi + li))
                keys.append((t, pname, L))
    log(f"{len(jobs)} probe jobs (3 targets each)")
    out = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(_probe_job)(*j) for j in jobs)
    for (t, pname, L), rr in zip(keys, out):
        for target, r in rr.items():
            res["tasks"].setdefault(t, {}).setdefault(target, {}).setdefault(pname, {})[str(L)] = r
    both = {}
    for pname in res["tasks"]["final"]["z"]:
        for L in res["tasks"]["final"]["z"][pname]:
            both[f"{pname}|L{L}"] = bool(res["tasks"]["final"]["z"][pname][L]["decodes"] and
                                         res["tasks"]["start"]["z"][pname][L]["decodes"])
    res["z_decodes_both_tasks"] = both
    res["reader_layers_ok"] = {k: v for k, v in both.items() if k.split("|")[0] in C.EMIT_CELLS and
                               k.split("|L")[1] in ("40", "42", "44") and v}
    C.dump(C.RES / "analysis" / "probes.json", res)
    log("probes done;", sum(both.values()), "cells x layers decode z under both tasks")


# --------------------------------------------------------------------------------------------- profile (whole-cell S)
def margin_sec(sec, ids, S_a, S_b, upper_fn=None):
    """m = LSE_{S_b} log P - LSE_{S_a} log P, full-sequence teacher forcing (every token of every spelling); one forward
    per spelling on ids + s[:-1] (cont = s[:-1]), with upper_fn(cont) -> {layer: fn} edits."""
    import torch
    T = len(ids)
    lp = {}
    for s in dict.fromkeys(list(S_a) + list(S_b)):
        cont = tuple(s[:-1])
        up = upper_fn(cont) if upper_fn is not None else {}
        fh, _ = sec.forward(list(ids) + list(cont), upper=up)
        l = sec.logprobs(fh, range(T - 1, T - 1 + len(s))).double()
        lp[s] = float(sum(l[i, t] for i, t in enumerate(s)))
    la = float(torch.logsumexp(torch.tensor([lp[s] for s in S_a], dtype=torch.float64), 0))
    lb = float(torch.logsumexp(torch.tensor([lp[s] for s in S_b], dtype=torch.float64), 0))
    return {"m": lb - la, "lse_a": la, "lse_b": lb, "spell": {str(k): v for k, v in lp.items()}}


def select_pairs(renders, task, limit, seed=C.BANK_SEED):
    """Up to `limit` pairs: one per table first (seeded), then the rest, in a seeded order."""
    pr = pairs_of(renders, task)
    rng = random.Random(f"{seed}|profile|{task}")
    by = {}
    for m, p in pr:
        by.setdefault(renders[m]["item"], []).append((m, p))
    first, rest = [], []
    for name in sorted(by):
        v = list(by[name])
        rng.shuffle(v)
        first.append(v[0])
        rest += v[1:]
    rng.shuffle(first)
    rng.shuffle(rest)
    sel = first + rest
    return sel if limit is None else sel[:limit]


def stage_profile(a):
    import torch
    import registry as R
    import regime as RG
    depth, renders = gated_material(a.gate_run)
    out = C.OUT / "profile" / a.run_id
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    log = C.Logger(out / "log.txt")
    tok = R.make_tokenizer()
    for r in renders:
        r["ids"] = B.render(tok, r["text"]).ids
    model = R.load_model()
    state = {"gates": {}, "depth": depth, "limit_pairs": a.limit_pairs}
    t_all = time.time()
    with RG.SecEngine(model, l0=39) as sec, torch.no_grad():
        for t in C.TASKS:
            sel = select_pairs(renders, t, a.limit_pairs)
            for n, (mi, pi) in enumerate(sel):
                m, p = renders[mi], renders[pi]
                pid = f"{m['item']}__{p['partner']}__{t}"
                f = out / "pairs" / f"{pid}.jsonl"
                if f.exists():
                    continue
                t0 = time.time()
                S_a, S_b = code_spellings(tok, m["answer"]), code_spellings(tok, p["answer"])
                conts = sorted({tuple(s[:-1]) for s in S_a + S_b})
                src, own = {}, {}
                for c in conts:
                    _, ap = sec.forward(p["ids"] + list(c), record=list(C.CAP_LAYERS))
                    _, am = sec.forward(m["ids"] + list(c), record=list(C.CAP_LAYERS))
                    src[c] = {L: ap[L] for L in C.CAP_LAYERS}
                    own[c] = {L: am[L] for L in C.CAP_LAYERS}
                clean = margin_sec(sec, m["ids"], S_a, S_b)
                donor = margin_sec(sec, p["ids"], S_a, S_b)
                G = donor["m"] - clean["m"]
                rows = [{"pair": pid, "item": m["item"], "task": t, "kind": "clean", "m_clean": clean["m"],
                         "m_donor": donor["m"], "G": G}]
                if n < 2:                                                     # G0 and the same-source check
                    pos = m["cells"]["emit_colon"]
                    g = {}
                    for L in (C.CAP_LAYERS[0], C.CAP_LAYERS[-1]):
                        def zero(c, L=L):
                            def fn(x):
                                x = x.clone()
                                x[pos] = x[pos] + 0.0
                                return x
                            return {L: fn}
                        e0 = margin_sec(sec, m["ids"], S_a, S_b, zero)
                        ss = margin_sec(sec, m["ids"], S_a, S_b,
                                        lambda c, L=L: {L: RG.patch_fn([pos], own[c][L][pos][None])})
                        g[f"L{L}"] = {"G0_dm": e0["m"] - clean["m"], "same_source_dm": ss["m"] - clean["m"],
                                      "pass": abs(e0["m"] - clean["m"]) <= FLOOR and ss["m"] == clean["m"]}
                    state["gates"][pid] = g
                    if not all(v["pass"] for v in g.values()):
                        log("STOP GATE: G0 / same-source failed", pid, json.dumps(g))
                        C.dump(out / "state.json", state)
                        return
                for cell in C.EMIT_CELLS:
                    pos = m["cells"][cell]
                    for L in C.CAP_LAYERS:
                        e = margin_sec(sec, m["ids"], S_a, S_b,
                                       lambda c, L=L, pos=pos: {L: RG.patch_fn([pos], src[c][L][pos][None])})
                        dm = e["m"] - clean["m"]
                        rows.append({"pair": pid, "item": m["item"], "task": t, "kind": "patch", "cell": cell, "layer": L,
                                     "dm": dm, "S": dm / G if G != 0 else math.nan,
                                     "d_lse_a": e["lse_a"] - clean["lse_a"], "d_lse_b": e["lse_b"] - clean["lse_b"]})
                tmp = f.with_suffix(".tmp")
                tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
                tmp.rename(f)
                sec.clear()
                log(f"[{t} {n + 1}/{len(sel)}] {pid} G {G:.2f} {time.time() - t0:.1f}s")
    state["minutes"] = (time.time() - t_all) / 60
    C.dump(out / "state.json", state)
    summarize_profile(out, a.run_id)


def summarize_profile(out, run_id):
    rows = [json.loads(l) for f in sorted((out / "pairs").glob("*.jsonl")) for l in open(f)]
    S = {"design": "§2.5.4", "run_id": run_id, "label": "single-cell leverage (whole-cell patch share S, p -> m)", "cells": {}}
    rng = np.random.default_rng(C.BOOT_SEED)
    for t in C.TASKS:
        cl = [r for r in rows if r["kind"] == "clean" and r["task"] == t]
        S.setdefault("G", {})[t] = {"n": len(cl), "median": float(np.median([r["G"] for r in cl])) if cl else None}
        for cell in C.EMIT_CELLS:
            for L in C.CAP_LAYERS:
                rr = [r for r in rows if r["kind"] == "patch" and r["task"] == t and r["cell"] == cell and r["layer"] == L
                      and np.isfinite(r["S"])]
                if not rr:
                    continue
                x = np.array([r["S"] for r in rr])
                items = np.array([r["item"] for r in rr])
                tabs = sorted(set(items))
                by = {tt: x[items == tt] for tt in tabs}
                bm = [np.median(np.concatenate([by[tt] for tt in rng.choice(tabs, len(tabs))])) for _ in range(2000)]
                S["cells"].setdefault(t, {}).setdefault(cell, {})[str(L)] = {
                    "n": int(len(x)), "median_S": float(np.median(x)), "mean_S": float(x.mean()),
                    "ci_median": [float(np.quantile(bm, .025)), float(np.quantile(bm, .975))]}
    C.dump(C.RES / "analysis" / f"profile_{run_id}.json", S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("outfacing", "captures", "probes", "profile"))
    ap.add_argument("--gate-run", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--limit-pairs", type=int, default=48)
    ap.add_argument("--n-jobs", type=int, default=24)
    a = ap.parse_args()
    {"outfacing": stage_outfacing, "captures": stage_captures, "probes": stage_probes, "profile": stage_profile}[a.stage](a)


if __name__ == "__main__":
    main()
