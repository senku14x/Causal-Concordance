"""Experiment 001 — leverage profile (brew, chain) and off-trajectory content (brew).
Design: docs/design_specs/001_leverage_profile.md (v3, approved 2026-10-02).

Stages (run in order; each writes results/001_leverage_profile/manifest_<stage>.json):
    gate      every distinct prompt of every candidate pair (dev + test start pairs; dev substitution prompts):
              greedy + 10 samples at T = 0.8; decodes saved in full
    smoke     engine checks + a few profile cells on 3 brew, 1 chain and 2 substitution dev pairs (must pass first)
    run       whole-cell profile, spans, controls, substitution arm, noise floor, decodability captures (dev only)
    probes    decodability probes on the captures (CPU)
    snapshot  descriptive reader snapshot (§3.7), only after every reader-free output is hashed
    analysis  summary.json + qualitative.json from the committed raw outputs (decision rules evaluated)
    figures   ICLR-style figures from summary.json only
Usage: python scripts/001_leverage_profile/leverage_profile.py <stage>
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import partners as P
import registry as R
import splits as SP
import wsb

EXP = "001_leverage_profile"
RUN_ID = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
RES = R.out_dir("results", EXP)
ANA = R.out_dir("results", EXP, "analysis")
OUT = R.out_dir("outputs", EXP)


def ids_key(ids):
    return hashlib.sha1(json.dumps(list(ids)).encode()).hexdigest()[:16]


def sha256_file(p):
    return R.sha256(p)


def write_manifest(stage, extra):
    m = {"experiment": EXP, "stage": stage, "run_id": RUN_ID, "design": "docs/design_specs/001_leverage_profile.md (v3)",
         **R.manifest_block(), **extra}
    (RES / f"manifest_{stage}.json").write_text(json.dumps(m, indent=1, default=str))


# ------------------------------------------------------------------------------------------------ materials
def materials(tok):
    """All candidate pairs with their renders and split. Start pairs for every item (dev and test), chain pairs
    de-duplicated as unordered prompt pairs, one seeded substitution per brew item."""
    _, brew = wsb.load_bank("brew")
    _, chain = wsb.load_bank("chain")
    bsplit = SP.brew_split(brew)
    csplit, cgroups = SP.chain_split(chain)
    mats = {"brew": [], "chain": [], "sub": [], "brew_items": {it["name"]: it for it in brew},
            "chain_items": {it["name"]: it for it in chain}, "bsplit": bsplit, "csplit": csplit, "cgroups": cgroups}
    for it in brew:
        for pair in P.brew_candidates(it):
            al = P.token_align(tok, pair, it)
            if al is None:
                continue
            mats["brew"].append({"pair": pair, "item": it, "r_m": al[0], "r_p": al[1], "diff": al[2],
                                 "split": bsplit[it["name"]]})
        sub = P.brew_substitution(tok, it)
        al = P.token_align(tok, sub, it)
        assert al is not None, sub.pair_id
        mats["sub"].append({"pair": sub, "item": it, "r_m": al[0], "r_p": al[1], "diff": al[2], "split": bsplit[it["name"]]})
    seen = set()
    for it in chain:
        if not P.chain_single_token_item(it):
            continue
        for pair in P.chain_candidates(it):
            key = frozenset((pair.m.text, pair.p.text))
            if key in seen:
                continue
            al = P.token_align(tok, pair, it)
            if al is None:
                continue
            seen.add(key)
            mats["chain"].append({"pair": pair, "item": it, "r_m": al[0], "r_p": al[1], "diff": al[2],
                                  "split": csplit[it["name"]], "group": " | ".join(it["steps"])})
    return mats


# ----------------------------------------------------------------------------------------------------- gate
def stage_gate():
    tok = R.make_tokenizer()
    mats = materials(tok)
    model = R.load_model()
    prompts = {}            # ids_key -> {family, ids, answer, text, users: [pair ids]}

    def add(fam, r, answer, pid):
        k = ids_key(r.ids)
        e = prompts.setdefault(k, {"family": fam, "ids": r.ids, "answer": answer, "text": r.text, "users": []})
        assert e["answer"] == answer, (k, e["answer"], answer)
        e["users"].append(pid)
    for e in mats["brew"]:
        add("brew", e["r_m"], e["pair"].m.a, e["pair"].pair_id)
        add("brew", e["r_p"], e["pair"].p.a, e["pair"].pair_id)
    for e in mats["chain"]:
        add("chain", e["r_m"], e["pair"].m.a, e["pair"].pair_id)
        add("chain", e["r_p"], e["pair"].p.a, e["pair"].pair_id)
    for e in mats["sub"]:
        if e["split"] == "dev":
            add("brew", e["r_m"], e["pair"].m.a, e["pair"].pair_id)
            add("brew", e["r_p"], e["pair"].p.a, e["pair"].pair_id)
    path = OUT / "gate" / "gate_prompts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if path.exists():                                  # resumable: never re-score a prompt already gated
        for line in open(path):
            d = json.loads(line)
            done[d["key"]] = d
    t0 = time.time()
    with open(path, "a") as f:
        for i, (k, e) in enumerate(sorted(prompts.items())):
            if k in done:
                continue
            seed = 20261002 + int(k[:6], 16) % 1000003
            g = P.gate_member(model, tok, e["ids"], e["family"], e["answer"], seed)
            row = {"key": k, "family": e["family"], "answer": e["answer"], "text": e["text"], "seed": seed,
                   "users": e["users"], **{x: g[x] for x in ("greedy", "greedy_ok", "samples", "n_correct", "k", "pass")}}
            f.write(json.dumps(row) + "\n")
            f.flush()
            done[k] = row
            if i % 25 == 0:
                print(f"[gate] {len(done)}/{len(prompts)} prompts, {time.time() - t0:.0f}s", flush=True)
    # pair-level outcomes and summary
    def pair_rows(entries, fam):
        rows = []
        for e in entries:
            km, kp = ids_key(e["r_m"].ids), ids_key(e["r_p"].ids)
            if km not in done or kp not in done:
                continue
            rows.append({"pair_id": e["pair"].pair_id, "family": fam, "item": e["pair"].item, "split": e["split"],
                         "group": e.get("group"), "z": str(e["pair"].m.z), "z_p": str(e["pair"].p.z),
                         "m_correct": done[km]["n_correct"], "p_correct": done[kp]["n_correct"],
                         "m_greedy_ok": done[km]["greedy_ok"], "p_greedy_ok": done[kp]["greedy_ok"],
                         "pass": bool(done[km]["pass"] and done[kp]["pass"])})
        return rows
    rows = pair_rows(mats["brew"], "brew") + pair_rows(mats["chain"], "chain") + \
        pair_rows([e for e in mats["sub"] if e["split"] == "dev"], "brew_substitution")
    (ANA / "gate_pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summ = {}
    for fam in ("brew", "chain", "brew_substitution"):
        for split in ("dev", "test"):
            rr = [r for r in rows if r["family"] == fam and r["split"] == split]
            if not rr:
                continue
            s = {"candidates": len(rr), "passed": sum(r["pass"] for r in rr),
                 "items_with_a_passing_pair": len({r["item"] for r in rr if r["pass"]}),
                 "items": len({r["item"] for r in rr})}
            if fam == "chain":
                s["groups_with_a_passing_pair"] = len({r["group"] for r in rr if r["pass"]})
                s["groups"] = len({r["group"] for r in rr})
            by_z = {}
            for r in rr:
                b = by_z.setdefault(r["z"], [0, 0])
                b[0] += r["pass"]
                b[1] += 1
            s["pass_by_member_z"] = {z: f"{a}/{n}" for z, (a, n) in sorted(by_z.items())}
            s["per_item_passing_pairs"] = sorted(sum(r["pass"] for r in rr if r["item"] == it) for it in {r["item"] for r in rr})
            summ[f"{fam}|{split}"] = s
    (ANA / "gate_summary.json").write_text(json.dumps(summ, indent=1))
    write_manifest("gate", {"n_prompts": len(prompts), "seconds": time.time() - t0,
                            "raw": {"gate_prompts.jsonl": sha256_file(path)}})
    print(json.dumps(summ, indent=1))


# ------------------------------------------------------------------------------------------- measurement core
LAYERS = [40, 42, 44, 56, 60]
CTRL_LAYERS = [0, 1, 2, 4]
COLOURS = list(P.BREW_COLOURS)
QUAL_SEED = 20261003


def gated_dev(mats, fam):
    rows = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ANA / "gate_pairs.jsonl")}
    key = {"brew": "brew", "chain": "chain", "sub": "brew_substitution"}[fam]
    return [e for e in mats[fam] if e["split"] == "dev" and rows.get(e["pair"].pair_id, {}).get("pass")
            and rows[e["pair"].pair_id]["family"] == key]


def top5(tok, lp):
    v, i = lp.topk(5)
    return [(tok.decode([int(t)]), round(float(x), 3)) for x, t in zip(v, i)]


def cells_of(fam, r):
    """(position, region) read cells: brew = the 24 pinned cells; chain = the question span (" What" -> final)."""
    if fam == "brew":
        return sorted(r.regions.items())
    return [(p, "final" if p == r.read_positions[-1] else "question") for p in r.read_positions]


def spans_of(fam, r, diff):
    if fam == "brew":
        reg = lambda *names: sorted(p for p, g in r.regions.items() if g in names)
        em = reg("emit_asst", "emit_think", "emit_colon")
        return {"start": reg("start"), "question": reg("question"), "emission": em,
                "start+question+emission": reg("start", "question") + em}
    return {"input": list(diff), "question": list(r.read_positions)}


def measure_pair(sec, tok, e, fam, reverse=False):
    """Whole-cell profile of one start pair under SEC. reverse=False: recipient = member (x), donor = partner (y)."""
    import margin as MG
    import regime as RG
    import torch
    import engine as E
    pair, diff = e["pair"], e["diff"]
    rx, ry = (e["r_p"], e["r_m"]) if reverse else (e["r_m"], e["r_p"])
    ax, ay = (pair.p.a, pair.m.a) if reverse else (pair.m.a, pair.p.a)
    Sa, Sb = MG.spellings(tok, fam, ax), MG.spellings(tok, fam, ay)
    T = len(rx)
    conts = [()] + [s for s in dict.fromkeys(Sa + Sb) if len(s) > 1]
    rec = LAYERS + CTRL_LAYERS + [20, 39, 63]
    src = {c: sec.forward(ry.ids + list(c), record=rec)[1] for c in conts}
    own = {c: sec.forward(rx.ids + list(c), record=rec)[1] for c in conts}
    clean = sec.score(rx.ids, Sa, Sb)
    donor = sec.score(ry.ids, Sa, Sb)
    G = donor["m"] - clean["m"]
    head = {"pair_id": pair.pair_id, "family": fam, "item": pair.item, "direction": "p->m" if not reverse else "m->p",
            "T": T, "m_clean": clean["m"], "m_donor": donor["m"], "G": G, "a": str(ax), "a_donor": str(ay),
            "top5_clean": top5(tok, clean["lp_last"]), "top5_donor": top5(tok, donor["lp_last"])}
    rows = []
    for pos, region in cells_of(fam, rx):
        for L in LAYERS:
            ed = sec.score(rx.ids, Sa, Sb, edit_fn=lambda c, L=L, pos=pos: ({L: RG.patch_fn([pos], src[c][L][[pos]])}, []))
            row = {"pair_id": pair.pair_id, "direction": head["direction"], "pos": pos, "region": region, "layer": L,
                   "dm": ed["m"] - clean["m"], "S": (ed["m"] - clean["m"]) / G,
                   "bitwise_noop": bool(torch.equal(ed["fh"], clean["fh"])), **RG.sec_damage(sec, clean, ed, pos, T)}
            if region in ("emit_asst", "emit_think", "emit_colon", "final"):
                row["top5"] = top5(tok, ed["lp_last"])
            rows.append(row)
    # controls
    ctrl = {"pair_id": pair.pair_id, "direction": head["direction"]}
    read_last = cells_of(fam, rx)[-1][0]
    ctrl["self_patch_bitwise"] = all(torch.equal(
        sec.score(rx.ids, Sa, Sb, edit_fn=lambda c, L=L: ({L: RG.patch_fn([read_last], own[c][L][[read_last]])}, []))["fh"],
        clean["fh"]) for L in LAYERS)
    emb_src = {c: sec.embed(torch.tensor([ry.ids + list(c)], device=sec.model.device))[0, diff].detach() for c in conts}
    anc = sec.score(rx.ids, Sa, Sb, edit_fn=lambda c: ({}, [E.EmbedPatch(sec.embed, diff, emb_src[c])]))
    ctrl["embedding_anchor_bitwise"] = bool(torch.equal(anc["fh"], donor["fh"])) and abs(anc["m"] - donor["m"]) == 0.0
    d0 = diff[0]
    ctrl["prefix_identical"] = all(torch.equal(src[()][l][:d0], own[()][l][:d0]) for l in rec)
    ctrl["start_token_S"] = {}
    for L in CTRL_LAYERS:
        ed = sec.score(rx.ids, Sa, Sb, edit_fn=lambda c, L=L: ({}, [E.CellPatch(sec.blocks, L, diff, src[c][L][diff])]))
        ctrl["start_token_S"][L] = (ed["m"] - clean["m"]) / G
    spans = []
    for name, posl in spans_of(fam, rx, diff).items():
        ed = sec.score(rx.ids, Sa, Sb, edit_fn=lambda c, posl=posl: ({L: RG.patch_fn(posl, src[c][L][posl]) for L in LAYERS}, []))
        spans.append({"pair_id": pair.pair_id, "direction": head["direction"], "span": name, "n_pos": len(posl),
                      "S": (ed["m"] - clean["m"]) / G})
    sec.clear()
    return head, rows, ctrl, spans


def colour_sets(tok):
    import margin as MG
    return [MG.spellings(tok, "brew", c) for c in COLOURS]


def colour_dist(sc):
    import torch
    l = torch.tensor(sc["extra_lse"], dtype=torch.float64)
    return torch.softmax(l, 0)


def measure_sub(sec, tok, e, reverse=False):
    """Off-trajectory substitution arm (design 001 §3.3): behaviour, emission-cell transplants, key-token reference."""
    import margin as MG
    import regime as RG
    import torch
    import engine as E
    pair, diff = e["pair"], e["diff"]
    rx, ry = (e["r_p"], e["r_m"]) if reverse else (e["r_m"], e["r_p"])
    a = pair.m.a
    Sa = MG.spellings(tok, "brew", a)
    ALLS = colour_sets(tok)
    T = len(rx)
    conts = [()] + [s for s in dict.fromkeys([s for g in ALLS for s in g]) if len(s) > 1]
    src = {c: sec.forward(ry.ids + list(c), record=LAYERS + [0])[1] for c in conts}
    cx = sec.score(rx.ids, Sa, Sa, extra_spellings=ALLS)
    cy = sec.score(ry.ids, Sa, Sa, extra_spellings=ALLS)
    px, py = colour_dist(cx), colour_dist(cy)
    head = {"pair_id": pair.pair_id, "item": pair.item, "direction": "sub->orig" if not reverse else "orig->sub",
            "c": pair.meta["c"], "c2": pair.meta["c2"], "a": a,
            "succ_answer": pair.meta.get("succ_answer"), "c_is_succ_answer": bool(pair.meta.get("c_is_succ_answer", False)),
            "behaviour_dlogPa": cy["lse_a"] - cx["lse_a"],
            "behaviour_TV": float(0.5 * (px - py).abs().sum()), "top5_x": top5(tok, cx["lp_last"]),
            "top5_y": top5(tok, cy["lp_last"]), "dist_x": [round(float(v), 4) for v in px],
            "dist_y": [round(float(v), 4) for v in py]}
    rows = []
    r_reg = {p: g for p, g in rx.regions.items() if g.startswith("emit")}
    for pos, region in sorted(r_reg.items()):
        for L in LAYERS:
            ed = sec.score(rx.ids, Sa, Sa, extra_spellings=ALLS,
                           edit_fn=lambda c, L=L, pos=pos: ({L: RG.patch_fn([pos], src[c][L][[pos]])}, []))
            pe = colour_dist(ed)
            rows.append({"pair_id": pair.pair_id, "direction": head["direction"], "pos": pos, "region": region, "layer": L,
                         "dlogPa": ed["lse_a"] - cx["lse_a"], "TV": float(0.5 * (pe - px).abs().sum()),
                         "top5": top5(tok, ed["lp_last"]), **RG.sec_damage(sec, cx, ed, pos, T)})
    ed = sec.score(rx.ids, Sa, Sa, extra_spellings=ALLS,
                   edit_fn=lambda c: ({}, [E.CellPatch(sec.blocks, 0, [diff[0]], src[c][0][[diff[0]]])]))
    head["key_token_L0_dlogPa"] = ed["lse_a"] - cx["lse_a"]
    head["key_token_L0_TV"] = float(0.5 * (colour_dist(ed) - px).abs().sum())
    sec.clear()
    return head, rows


def noise_floor(sec, tok, e, fam, n=8, rel=1e-4, seed=0):
    """|dm| from 1e-4-relative random perturbations of the recipient's own state: at the last read cell, L42 (SEC upper
    stack) and at the first differing token, L2 (bf16 lower stack). One direction per draw, shared by every forward of
    a multi-token spelling."""
    import margin as MG
    import regime as RG
    import torch
    import engine as E
    pair, diff = e["pair"], e["diff"]
    rx = e["r_m"]
    Sa, Sb = MG.spellings(tok, fam, pair.m.a), MG.spellings(tok, fam, pair.p.a)
    conts = [()] + [s for s in dict.fromkeys(Sa + Sb) if len(s) > 1]
    clean = sec.score(rx.ids, Sa, Sb)
    G = sec.score(e["r_p"].ids, Sa, Sb)["m"] - clean["m"]
    read_last = cells_of(fam, rx)[-1][0]
    own = {c: sec.forward(rx.ids + list(c), record=[42, 2])[1] for c in conts}
    out = {"pair_id": pair.pair_id, "family": fam, "G": G, "upper_L42": [], "lower_L2": []}
    for k in range(n):
        v = torch.randn(5120, generator=torch.Generator().manual_seed(seed + k))
        v = v / v.norm()
        for where, L, pos in (("upper_L42", 42, read_last), ("lower_L2", 2, diff[0])):
            def ef(c, L=L, pos=pos, where=where):
                h = own[c][L][pos]
                src = (h + rel * h.norm() * v.to(h.device))[None]
                return ({L: RG.patch_fn([pos], src)}, []) if where == "upper_L42" else ({}, [E.CellPatch(sec.blocks, L, [pos], src)])
            out[where].append(abs(sec.score(rx.ids, Sa, Sb, edit_fn=ef)["m"] - clean["m"]))
    sec.clear()
    return out


def capture_brew(sec, mats):
    """SEC activations at the three emission cells x LAYERS for every dev brew render (members, all partner candidates,
    substitution renders), with labels for the probes."""
    import numpy as np
    renders, meta = {}, []
    for e in mats["brew"]:
        if e["split"] != "dev":
            continue
        it = e["item"]
        for role, r, start, z, a in (("member", e["r_m"], e["pair"].m.inp, e["pair"].m.z, e["pair"].m.a),
                                     ("partner", e["r_p"], e["pair"].p.inp, e["pair"].p.z, e["pair"].p.a)):
            k = ids_key(r.ids)
            if k not in renders:
                renders[k] = r
                meta.append({"key": k, "item": it["name"], "role": role, "start": start, "z": z, "answer": a,
                             "present": sorted(set(it["table"]) | {v["stir"] for v in it["table"].values()}),
                             "trajectory": [start, z, a], "succ_answer": it["table"].get(a, {}).get("stir")})
    for e in mats["sub"]:
        if e["split"] != "dev":
            continue
        it, pr = e["item"], e["pair"]
        k = ids_key(e["r_p"].ids)
        c, c2 = pr.meta["c"], pr.meta["c2"]
        present = sorted((set(it["table"]) | {v["stir"] for v in it["table"].values()}) - {c} | {c2})
        renders[k] = e["r_p"]
        meta.append({"key": k, "item": it["name"], "role": "substituted", "start": pr.m.inp, "z": pr.m.z, "answer": pr.m.a,
                     "present": present, "trajectory": [pr.m.inp, pr.m.z, pr.m.a], "sub": [c, c2],
                     "succ_answer": pr.meta.get("succ_answer"), "c_is_succ_answer": bool(pr.meta.get("c_is_succ_answer", False)),
                     "orig_key": ids_key(e["r_m"].ids)})
    X = np.zeros((len(meta), 3, len(LAYERS), 5120), dtype=np.float32)
    for i, m in enumerate(meta):
        r = renders[m["key"]]
        em = sorted(p for p, g in r.regions.items() if g.startswith("emit"))
        _, acts = sec.forward(r.ids, record=LAYERS)
        for j, L in enumerate(LAYERS):
            X[i, :, j] = acts[L][em].float().cpu().numpy()
        sec.clear()
    return X, meta


# ------------------------------------------------------------------------------------------- stages: smoke, run
def _jsonl_append(path, rows):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")


def _done(path, key="pair_id", extra="direction"):
    if not os.path.exists(path):
        return set()
    return {(d[key], d.get(extra)) for d in map(json.loads, open(path))}


def stage_smoke():
    """Engine checks + a few profile cells on 3 brew, 1 chain and 2 substitution dev pairs. Exit 1 if any exact check
    fails (design 001 §3.5: then the engine is broken and nothing else runs)."""
    import regime as RG
    tok = R.make_tokenizer()
    mats = materials(tok)
    model = R.load_model()
    sec = RG.SecEngine(model, l0=39)
    t0 = time.time()
    res = {"pairs": [], "subs": []}
    ok = True
    for fam, n in (("brew", 3), ("chain", 1)):
        for e in gated_dev(mats, fam)[:n]:
            t1 = time.time()
            head, rows, ctrl, spans = measure_pair(sec, tok, e, fam)
            stir = [r["bitwise_noop"] for r in rows if r["region"] == "stir"]
            checks = {"self_patch_bitwise": ctrl["self_patch_bitwise"], "embedding_anchor_bitwise": ctrl["embedding_anchor_bitwise"],
                      "prefix_identical": ctrl["prefix_identical"], "stir_noop_bitwise": all(stir) if stir else None}
            ok &= all(v for v in checks.values() if v is not None)
            em = {(r["region"], r["layer"]): round(r["S"], 3) for r in rows if r["region"] in ("emit_colon", "final")}
            res["pairs"].append({"pair_id": head["pair_id"], "family": fam, "G": head["G"], "checks": checks,
                                 "start_token_S": ctrl["start_token_S"], "read_cell_S": {f"{k[0]}|L{k[1]}": v for k, v in em.items()},
                                 "spans": {s_["span"]: round(s_["S"], 3) for s_ in spans}, "seconds": time.time() - t1,
                                 "top5_clean": head["top5_clean"], "top5_donor": head["top5_donor"]})
            print(json.dumps(res["pairs"][-1], default=str)[:1200], flush=True)
    for e in gated_dev(mats, "sub")[:2]:
        t1 = time.time()
        head, rows = measure_sub(sec, tok, e)
        res["subs"].append({**{k: head[k] for k in ("pair_id", "c", "c2", "behaviour_dlogPa", "behaviour_TV",
                                                     "key_token_L0_dlogPa", "top5_x", "top5_y")},
                            "max_abs_transplant_dlogPa": max(abs(r["dlogPa"]) for r in rows),
                            "max_transplant_TV": max(r["TV"] for r in rows), "seconds": time.time() - t1})
        print(json.dumps(res["subs"][-1], default=str)[:800], flush=True)
    res["all_engine_checks_pass"] = bool(ok)
    res["seconds"] = time.time() - t0
    (ANA / "smoke_checks.json").write_text(json.dumps(res, indent=1, default=str))
    write_manifest("smoke", {"all_engine_checks_pass": bool(ok), "seconds": res["seconds"]})
    sec.restore()
    print("ENGINE CHECKS:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def stage_run():
    import random
    import numpy as np
    import regime as RG
    assert json.loads((ANA / "smoke_checks.json").read_text())["all_engine_checks_pass"], "smoke checks did not pass"
    tok = R.make_tokenizer()
    mats = materials(tok)
    model = R.load_model()
    sec = RG.SecEngine(model, l0=39)
    out = OUT / "run"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = random.Random(QUAL_SEED)
    for fam in ("brew", "chain"):
        entries = gated_dev(mats, fam)
        rev = set(rng.sample([e["pair"].pair_id for e in entries], k=round(0.3 * len(entries))))
        done = _done(out / f"pairs_{fam}.jsonl")
        for i, e in enumerate(entries):
            for reverse in ([False, True] if e["pair"].pair_id in rev else [False]):
                d = "m->p" if reverse else "p->m"
                if (e["pair"].pair_id, d) in done:
                    continue
                head, rows, ctrl, spans = measure_pair(sec, tok, e, fam, reverse=reverse)
                _jsonl_append(out / f"profile_{fam}.jsonl", rows)
                _jsonl_append(out / f"controls_{fam}.jsonl", [ctrl])
                _jsonl_append(out / f"spans_{fam}.jsonl", spans)
                _jsonl_append(out / f"pairs_{fam}.jsonl", [head])
            print(f"[run] {fam} {i + 1}/{len(entries)} {time.time() - t0:.0f}s", flush=True)
    subs = gated_dev(mats, "sub")
    rev = set(rng.sample([e["pair"].pair_id for e in subs], k=round(0.3 * len(subs))))
    done = _done(out / "subs.jsonl")
    for i, e in enumerate(subs):
        for reverse in ([False, True] if e["pair"].pair_id in rev else [False]):
            d = "orig->sub" if reverse else "sub->orig"
            if (e["pair"].pair_id, d) in done:
                continue
            head, rows = measure_sub(sec, tok, e, reverse=reverse)
            _jsonl_append(out / "sub_transplants.jsonl", rows)
            _jsonl_append(out / "subs.jsonl", [head])
        print(f"[run] sub {i + 1}/{len(subs)} {time.time() - t0:.0f}s", flush=True)
    if not (out / "noise.jsonl").exists():
        nf = []
        for fam in ("brew", "chain"):
            entries = gated_dev(mats, fam)
            for e in random.Random(QUAL_SEED + 1).sample(entries, k=min(20, len(entries))):
                nf.append(noise_floor(sec, tok, e, fam, seed=QUAL_SEED))
        _jsonl_append(out / "noise.jsonl", nf)
    if not (out / "captures_brew.npz").exists():
        X, meta = capture_brew(sec, mats)
        np.savez_compressed(out / "captures_brew.npz", X=X)
        (out / "captures_brew_meta.json").write_text(json.dumps({"layers": LAYERS, "cells": ["emit_asst", "emit_think", "emit_colon"],
                                                                 "renders": meta}, default=str))
    sec.restore()
    hashes = {f.name: sha256_file(f) for f in sorted(out.iterdir()) if f.is_file()}
    write_manifest("run", {"seconds": time.time() - t0, "raw": hashes})
    print("run done", time.time() - t0)


# ----------------------------------------------------------------------------------------------- stage: probes
C_GRID = (1e-3, 1e-2, 1e-1, 1.0)
N_PERM = 200


def _fit(Xtr, ytr, C, multi):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    return clf.fit(Xtr, ytr)


def _choose_C(Xtr, ytr, gtr, multi):
    """Inner grouped CV (3 folds by table) over C_GRID; falls back to C = 1e-2 when a fold lacks a class."""
    import numpy as np
    from sklearn.model_selection import GroupKFold
    best, best_s = 1e-2, -1
    for C in C_GRID:
        sc = []
        for tr, te in GroupKFold(3).split(Xtr, ytr, gtr):
            if len(set(ytr[tr])) < len(set(ytr)) or len(set(ytr[tr])) < 2:
                continue
            m = _fit(Xtr[tr], ytr[tr], C, multi)
            sc.append(float(np.mean(m.predict(Xtr[te]) == ytr[te])))
        if sc and np.mean(sc) > best_s:
            best, best_s = C, float(np.mean(sc))
    return best


def _cv_scores(X, y, g, multi, seed=0, C_fixed=None):
    """Outer StratifiedGroupKFold(5) by table. Returns (held-out scores per row, fold index per row, Cs, models).
    Binary: decision_function; multiclass: predicted labels."""
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    pred = np.empty(len(y), dtype=object if multi else float)
    fold_of = np.full(len(y), -1)
    Cs, models = [], []
    for k, (tr, te) in enumerate(folds.split(X, y, g)):
        assert len(set(y[tr])) >= 2 and (multi or len(set(y[te])) == 2), "a fold lacks a class"
        C = C_fixed[k] if C_fixed is not None else _choose_C(X[tr], y[tr], g[tr], multi)
        m = _fit(X[tr], y[tr], C, multi)
        pred[te] = m.predict(X[te]) if multi else m.decision_function(X[te])
        fold_of[te] = k
        Cs.append(C)
        models.append(m)
    return pred, fold_of, Cs, models


def _metric(y, pred, multi):
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    return float(balanced_accuracy_score(y, pred.astype(y.dtype))) if multi else float(roc_auc_score(y, pred.astype(float)))


def _one_target(X, y, g, multi, label_perm_unit, seed):
    """Real score, control-task score, permutation null (labels permuted at the table level for table-level targets,
    at the render level otherwise), and fitted fold models for directional scoring."""
    import numpy as np
    rng = np.random.default_rng(seed)
    pred, fold_of, Cs, models = _cv_scores(X, y, g, multi, seed=seed)
    real = _metric(y, pred, multi)
    # control task (Hewitt-Liang analogue): a random label per table (binary) / per (table, colour) (multiclass)
    if multi:
        lab = sorted(set(y))
        perm_of = {t: dict(zip(lab, rng.permutation(lab))) for t in sorted(set(g))}
        yc = np.array([perm_of[t][v] for t, v in zip(g, y)])
    else:
        tabs = sorted(set(g))
        frac = float(np.mean([y[g == t][0] for t in tabs]))
        tl = {t: int(rng.random() < frac) for t in tabs}
        yc = np.array([tl[t] for t in g])
    try:
        pc, _, _, _ = _cv_scores(X, yc, g, multi, seed=seed)
        control = _metric(yc, pc, multi)
    except AssertionError:
        control = float("nan")
    null = []
    tabs = sorted(set(g))
    for i in range(N_PERM):
        if label_perm_unit == "table":
            tl = dict(zip(tabs, rng.permutation([y[g == t][0] for t in tabs])))
            yp = np.array([tl[t] for t in g])
        else:
            yp = rng.permutation(y)
        try:
            pp, _, _, _ = _cv_scores(X, yp, g, multi, seed=seed, C_fixed=Cs)
            null.append(_metric(yp, pp, multi))
        except AssertionError:
            continue
    null = np.array(null)
    p = float((1 + (null >= real).sum()) / (1 + len(null)))
    return {"real": real, "control": control, "selectivity": real - control if control == control else None,
            "null_mean": float(null.mean()) if len(null) else None, "null_q95": float(np.quantile(null, 0.95)) if len(null) else None,
            "n_null": int(len(null)), "null": [round(float(x), 5) for x in null], "p": p, "Cs": Cs, "n": int(len(y)),
            "n_tables": int(len(set(g))),
            "classes": {str(k): int((y == k).sum()) for k in sorted(set(y))}}, models, fold_of


def stage_probes():
    import numpy as np
    from joblib import Parallel, delayed
    out = OUT / "run"
    X = np.load(out / "captures_brew.npz")["X"]
    meta = json.loads((out / "captures_brew_meta.json").read_text())
    R_ = meta["renders"]
    nat = np.array([r["role"] in ("member", "partner") for r in R_])
    g_all = np.array([r["item"] for r in R_])
    jobs = []
    for ci, cell in enumerate(meta["cells"]):
        for li, L in enumerate(meta["layers"]):
            for tgt in ("z", "start", "answer"):
                y = np.array([str(r[tgt]) for r in R_])[nat]
                jobs.append(("multi", tgt, ci, li, X[nat, ci, li], y, g_all[nat], "render"))
            for c in COLOURS:
                keep = np.array([r["role"] in ("member", "partner") and c not in r["trajectory"] for r in R_])
                y = np.array([int(c in r["present"]) for r in R_])[keep]
                if min((y == 0).sum(), (y == 1).sum()) == 0 or len(set(g_all[keep][y == 0])) < 5 or len(set(g_all[keep][y == 1])) < 5:
                    jobs.append(("skip", c, ci, li, None, None, None, None))
                    continue
                jobs.append(("member", c, ci, li, X[keep, ci, li], y, g_all[keep], "table"))

    def run(job):
        kind, tgt, ci, li, Xs, y, g, unit = job
        if kind == "skip":
            return {"kind": "membership", "target": tgt, "cell": ci, "layer": li, "status": "insufficient support"}, None
        res, models, fold_of = _one_target(Xs, y, g, kind == "multi", unit, seed=QUAL_SEED)
        res.update({"kind": "12-way" if kind == "multi" else "membership", "target": tgt, "cell": ci, "layer": li, "status": "ok"})
        keep_models = None
        if kind == "member":                      # keep fold models + table->fold map for the directional test
            keep_models = (models, dict(zip(g, fold_of)))
        return res, keep_models
    t0 = time.time()
    results = Parallel(n_jobs=96, verbose=5)(delayed(run)(j) for j in jobs)
    rows = [r for r, _ in results]
    models = {(r["target"], r["cell"], r["layer"]): m for (r, m) in results if m is not None}
    # directional substitution test (cross-fitted: the fold model that never saw the item)
    subs = [(i, r) for i, r in enumerate(R_) if r["role"] == "substituted"]
    key2idx = {r["key"]: i for i, r in enumerate(R_)}
    direc = []
    for ci, cell in enumerate(meta["cells"]):
        for li, L in enumerate(meta["layers"]):
            for i, r in subs:
                c, c2 = r["sub"]
                j = key2idx.get(r["orig_key"])
                if j is None:
                    continue
                row = {"item": r["item"], "c": c, "c2": c2, "cell": ci, "layer": li}
                for col, name in ((c, "removed"), (c2, "inserted")):
                    m = models.get((col, ci, li))
                    if m is None or r["item"] not in m[1]:
                        row[name] = None
                        continue
                    fm = m[0][m[1][r["item"]]]
                    row[name] = float(fm.decision_function(X[[i], ci, li])[0] - fm.decision_function(X[[j], ci, li])[0])
                direc.append(row)
    (ANA / "probes.json").write_text(json.dumps({"rows": rows, "cells": meta["cells"], "layers": meta["layers"],
                                                 "n_perm": N_PERM, "seconds": time.time() - t0}, indent=1, default=float))
    (ANA / "probes_substitution_directional.json").write_text(json.dumps(direc, indent=1, default=float))
    write_manifest("probes", {"seconds": time.time() - t0, "n_jobs": len(jobs)})
    print("probes done", time.time() - t0)


# --------------------------------------------------------------------------------------------- stage: snapshot
WSB_GRID = list(range(20, 61, 4))                       # WSB's reader grid: layers 20..60 every 4 (Oracle's training grid)
SNAP_LENS_LAYERS = sorted(set(WSB_GRID) | set(LAYERS))  # + L42 (the NLA layer and a profile layer)


def stage_snapshot():
    """Descriptive reader snapshot (design 001 v3 §3.7 + amendment A1). Runs only after the reader-free outputs are
    hashed; never feeds any 001 decision. Every readout is saved in full (qualitative record).

    WSB protocol (A1): lenses top-10 at layers 20..60 every 4 (+ L42) on the three emission cells; Oracle Lens 1 sample
    per cell at the same grid; NLA at L42. Contract runs (v3): Oracle k = 4 at emit_colon L40/L44 and NLA k = 4, each with
    no-activation, shuffled and SEC-activation arms."""
    import gc
    import numpy as np
    import torch
    import readers as RD
    from transformers import AutoTokenizer
    assert (RES / "manifest_run.json").exists() and (RES / "manifest_probes.json").exists(), "reader-free stages first"
    tok = R.make_tokenizer()
    mats = materials(tok)
    gp = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ANA / "gate_pairs.jsonl")}
    starts = {}
    for e in mats["brew"]:
        if e["split"] == "dev" and gp.get(e["pair"].pair_id, {}).get("pass"):
            starts.setdefault(e["pair"].item, e)
    subs = {e["pair"].item: e for e in mats["sub"] if e["split"] == "dev" and gp.get(e["pair"].pair_id, {}).get("pass")}
    tables = [n for n in mats["brew_items"] if n in starts and n in subs][:10]
    assert len(tables) == 10, len(tables)
    roles = ("member", "partner", "substituted")
    renders = {(t, "member"): starts[t]["r_m"], (t, "partner"): starts[t]["r_p"], (t, "substituted"): subs[t]["r_p"]}
    cap = np.load(OUT / "run" / "captures_brew.npz")["X"]
    cmeta = json.loads((OUT / "run" / "captures_brew_meta.json").read_text())
    kidx = {r["key"]: i for i, r in enumerate(cmeta["renders"])}
    model = R.load_model()
    stock = {}
    for (t, role), r in renders.items():
        em = sorted(p for p, g in r.regions.items() if g.startswith("emit"))
        acts = RD.capture_stock(model, r.ids, SNAP_LENS_LAYERS, em)
        for L in SNAP_LENS_LAYERS:
            for ci in range(3):
                stock[(t, role, ci, L)] = acts[L][ci]
    secv = {(t, role, ci, L): torch.from_numpy(cap[kidx[ids_key(r.ids)], ci, LAYERS.index(L)])
            for (t, role), r in renders.items() for ci in range(3) for L in LAYERS}
    shuf = {t: tables[(i + 1) % len(tables)] for i, t in enumerate(tables)}       # fixed derangement
    snap = {"tables": [], "bridge": [], "protocol": {"lens_layers": SNAP_LENS_LAYERS, "oracle_wsb_layers": WSB_GRID,
                                                     "oracle_contract_layers": [40, 44], "nla_layer": 42, "cells": ["emit_asst", "emit_think", "emit_colon"]}}
    for t in tables:
        it = mats["brew_items"][t]
        sp = subs[t]["pair"]
        snap["tables"].append({"item": t, "start": it["start"], "z": it["intermediates"][0], "answer": it["answer"],
                               "candidates": it["options_adjacent"][0],
                               "partner": {"pair_id": starts[t]["pair"].pair_id, "start": starts[t]["pair"].p.inp,
                                           "z": starts[t]["pair"].p.z, "answer": starts[t]["pair"].p.a},
                               "substitution": {"pair_id": sp.pair_id, "c": sp.meta["c"], "c2": sp.meta["c2"]},
                               "shuffled_from": shuf[t], "readouts": {}})
    for k, v in secv.items():
        snap["bridge"].append({"table": k[0], "role": k[1], "cell": k[2], "layer": k[3],
                               "cos_stock_sec": float(torch.nn.functional.cosine_similarity(stock[k].float(), v.float(), 0))})
    T = {d["item"]: d for d in snap["tables"]}
    t0 = time.time()

    def hs_for(role, ci, L, arm, rows=tables):
        return [None if arm == "none" else stock[(t, role, ci, L)] if arm == "real" else
                stock[(shuf[t], role, ci, L)] if arm == "shuffled" else secv[(t, role, ci, L)] for t in rows]
    # ---- lenses: real + shuffled at every snapshot layer; SEC activations at the profile layers
    vl = RD.VectorLenses(model, tok, SNAP_LENS_LAYERS)
    for (t, role, ci, L), h in stock.items():
        arms = [("real", h), ("shuffled", stock[(shuf[t], role, ci, L)])] + ([("real_sec", secv[(t, role, ci, L)])] if L in LAYERS else [])
        for arm, v in arms:
            ro = {"logit_lens": vl.logit_lens(v), "J_NP_cosine": vl.j_cosine("J_NP", v, L), "J_NP_standard": vl.j_standard("J_NP", v, L),
                  "J_CB_standard": vl.j_standard("J_CB", v, L), "R_CB_standard": vl.j_standard("R_CB", v, L)}
            T[t]["readouts"][f"{role}|cell{ci}|L{L}|lenses|{arm}"] = {k: [x for x, _ in val] for k, val in ro.items()}
    del vl
    gc.collect()
    torch.cuda.empty_cache()
    print("lenses", round(time.time() - t0), flush=True)
    ol = RD.OracleLens(model, tok)
    # ---- Oracle, WSB protocol: 1 sample per cell, layers 20..60 every 4, real / shuffled (all roles in one batch), none
    for L in WSB_GRID:
        for ci in range(3):
            for arm in ("real", "shuffled"):
                batch = [(t, role) for role in roles for t in tables]
                hs = [stock[(t, role, ci, L)] if arm == "real" else stock[(shuf[t], role, ci, L)] for t, role in batch]
                outs = ol.read_batch(hs, L, k=1, seed=QUAL_SEED + L)
                for (t, role), o in zip(batch, outs):
                    T[t]["readouts"][f"{role}|cell{ci}|L{L}|oracle_wsb|{arm}"] = o
        outs = ol.read_batch([None] * len(tables), L, k=1, seed=QUAL_SEED + L)     # no activation: depends on L only
        for t, o in zip(tables, outs):
            T[t]["readouts"][f"none|L{L}|oracle_wsb|none"] = o
        print("oracle wsb", L, round(time.time() - t0), flush=True)
    # ---- Oracle contract runs: k = 4 at emit_colon, L40 / L44
    for L in (40, 44):
        for role in roles:
            for arm in ("real", "none", "shuffled", "real_sec"):
                outs = ol.read_batch(hs_for(role, 2, L, arm), L, k=4, seed=QUAL_SEED)
                for t, o in zip(tables, outs):
                    T[t]["readouts"][f"{role}|cell2|L{L}|oracle|{arm}"] = o
    print("oracle contract", round(time.time() - t0), flush=True)
    model = ol.unload()
    del ol
    # ---- NLA at L42 (merges the SFT adapter into the model IN PLACE: every capture is done above)
    _, pth, _ = RD.nla_paths()
    av = RD.NLAVerbalizer(model, AutoTokenizer.from_pretrained(pth["av_sft_lora"]))
    for ci in range(3):
        for role in roles:
            for arm in ("real", "shuffled") + (("real_sec",) if ci == 2 else ()):
                outs = av.read_batch(hs_for(role, ci, 42, arm), k=4, seed=QUAL_SEED)
                for t, o in zip(tables, outs):
                    T[t]["readouts"][f"{role}|cell{ci}|L42|nla|{arm}"] = {"raw": o, "explanations": [RD.extract_explanation(x) for x in o]}
    outs = av.read_batch([None] * len(tables), k=4, seed=QUAL_SEED)               # no activation: one batch
    for t, o in zip(tables, outs):
        T[t]["readouts"]["none|L42|nla|none"] = {"raw": o, "explanations": [RD.extract_explanation(x) for x in o]}
    print("nla", round(time.time() - t0), flush=True)
    av.close()
    del av, model
    gc.collect()
    torch.cuda.empty_cache()
    # ---- AR: FVE of every NLA explanation against the cell it describes (shuffled: also against its source)
    ar = RD.NLAReconstructor()
    ref, _ = RD.nla_reference_activations()
    for t in tables:
        for key, d in T[t]["readouts"].items():
            if "|nla|" not in key:
                continue
            if key.startswith("none|"):
                golds = {"fve_vs_member_colon": stock[(t, "member", 2, 42)]}
            else:
                role, cell = key.split("|")[0], int(key.split("|")[1][4:])
                golds = {"fve_vs_cell": stock[(t, role, cell, 42)]}
                if key.endswith("|shuffled"):
                    golds["fve_vs_source"] = stock[(shuf[t], role, cell, 42)]
            for name, gold in golds.items():
                d[name] = [None if e is None else round(ar.fve(ar.predict(tok, e), gold, ref)[0], 4) for e in d["explanations"]]
    del ar
    snap["seconds"] = time.time() - t0
    (ANA / "snapshot.json").write_text(json.dumps(snap, indent=1, default=float, ensure_ascii=False))
    write_manifest("snapshot", {"seconds": snap["seconds"], "tables": tables})
    print("snapshot done", snap["seconds"])


# --------------------------------------------------------------------------------------------- stage: analysis
TAU, EPS = 0.10, 0.05


def _load_jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def _item_means(rows, key):
    by = {}
    for r in rows:
        by.setdefault(r["item"], []).append(r[key])
    return [sum(v) / len(v) for v in by.values()]


def _summ(rows, key="S"):
    import numpy as np
    import stats as ST
    v = np.array([r[key] for r in rows], dtype=float)
    im = _item_means(rows, key)
    ct = ST.cluster_t(im) if len(im) > 1 else {"mean": float(np.mean(im)), "ci": [float("nan")] * 2}
    eq = ST.equivalence(im, EPS) if len(im) > 1 else {"equivalent": None}
    return {"n_pairs": int(len(v)), "n_items": len(im), "median": float(np.median(v)), "mean_items": ct["mean"],
            "ci95_items": ct["ci"], "frac_toward": float(np.mean(v >= TAU)), "frac_negligible": float(np.mean(np.abs(v) < EPS)),
            "frac_away": float(np.mean(v <= -TAU)), "frac_intermediate": float(np.mean((np.abs(v) >= EPS) & (v < TAU) & (v > -TAU))),
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"), "tost_negligible_items": eq["equivalent"]}


def stage_analysis():
    import random
    import numpy as np
    import stats as ST
    out = OUT / "run"
    gate = json.loads((ANA / "gate_summary.json").read_text())
    smoke = json.loads((ANA / "smoke_checks.json").read_text())
    S = {"gate": gate, "smoke_all_engine_checks_pass": smoke["all_engine_checks_pass"], "thresholds": {"tau": TAU, "eps": EPS}}
    pair_item = {}
    for fam in ("brew", "chain"):
        heads = _load_jsonl(out / f"pairs_{fam}.jsonl")
        for h in heads:
            pair_item[h["pair_id"]] = h["item"]
        rows = _load_jsonl(out / f"profile_{fam}.jsonl")
        for r in rows:
            r["item"] = pair_item[r["pair_id"]]
        fwd = [r for r in rows if r["direction"] == "p->m"]
        # cell identity across items: region + rank within region
        rank = {}
        for r in fwd:
            rank.setdefault((r["pair_id"], r["region"]), set()).add(r["pos"])
        for r in rows:
            ps = sorted(rank.get((r["pair_id"], r["region"]), {r["pos"]}))
            r["cell"] = f'{r["region"]}#{ps.index(r["pos"]) if r["pos"] in ps else 0}'
        table = {}
        for r in fwd:
            table.setdefault((r["cell"], r["layer"]), []).append(r)
        prof = {f"{c}|L{L}": _summ(v) for (c, L), v in sorted(table.items(), key=lambda kv: (kv[0][1], kv[0][0]))}
        for k, v in prof.items():
            rr = table[(k.split("|L")[0], int(k.split("|L")[1]))]
            v["frac_damage_offpair_gt_0.1"] = float(np.mean([(r["off_pair_mass_edit"] - r["off_pair_mass_clean"]) > 0.1 for r in rr]))
            v["median_top1_retention"] = float(np.nanmedian([r["top1_retention"] for r in rr])) if rr else None
        G = np.array([h["G"] for h in heads if h["direction"] == "p->m"])
        S[fam] = {"n_pairs": int(len(G)), "n_items": len({h["item"] for h in heads}), "G_median": float(np.median(G)) if len(G) else None,
                  "G_min": float(G.min()) if len(G) else None, "profile": prof}
        ctrls = _load_jsonl(out / f"controls_{fam}.jsonl")
        S[fam]["controls"] = {
            "n": len(ctrls),
            "self_patch_bitwise": float(np.mean([c["self_patch_bitwise"] for c in ctrls])) if ctrls else None,
            "embedding_anchor_bitwise": float(np.mean([c["embedding_anchor_bitwise"] for c in ctrls])) if ctrls else None,
            "prefix_identical": float(np.mean([c["prefix_identical"] for c in ctrls])) if ctrls else None,
            "stir_noop_bitwise": float(np.mean([r["bitwise_noop"] for r in fwd if r["region"] == "stir"])) if fam == "brew" and fwd else None,
            "start_token_S": {L: {"median": float(np.median([c["start_token_S"][str(L)] if str(L) in c["start_token_S"] else c["start_token_S"][L] for c in ctrls])),
                                  "q25": float(np.quantile([c["start_token_S"].get(str(L), c["start_token_S"].get(L)) for c in ctrls], 0.25)),
                                  "q75": float(np.quantile([c["start_token_S"].get(str(L), c["start_token_S"].get(L)) for c in ctrls], 0.75)),
                                  "frac_ge_0.9": float(np.mean([c["start_token_S"].get(str(L), c["start_token_S"].get(L)) >= 0.9 for c in ctrls]))}
                              for L in CTRL_LAYERS} if ctrls else {}}
        sp = [dict(x, item=pair_item[x["pair_id"]]) for x in _load_jsonl(out / f"spans_{fam}.jsonl") if x["direction"] == "p->m"]
        S[fam]["spans"] = {name: _summ([x for x in sp if x["span"] == name]) for name in sorted({x["span"] for x in sp})}
        # reverse-direction agreement at the read cells
        both = {}
        for r in rows:
            if r["region"] in ("emit_asst", "emit_think", "emit_colon", "final"):
                both.setdefault((r["pair_id"], r["cell"], r["layer"]), {})[r["direction"]] = r["S"]
        pr = [(v["p->m"], v["m->p"]) for v in both.values() if len(v) == 2]
        S[fam]["reverse"] = {"n": len(pr), "pearson": float(np.corrcoef(np.array(pr).T)[0, 1]) if len(pr) > 2 else None,
                             "median_abs_diff": float(np.median([abs(a - b) for a, b in pr])) if pr else None}
        # within-position variation diagnostic
        noise = [n for n in _load_jsonl(out / "noise.jsonl") if n["family"] == fam]
        nsd = float(np.sqrt(np.mean([(x / n["G"]) ** 2 for n in noise for x in n["upper_L42"]]))) if noise else float("nan")
        S[fam]["noise_floor"] = {"n_pairs": len(noise), "upper_L42_rms_frac_G": nsd,
                                 "upper_L42_max_abs_dm": float(max(x for n in noise for x in n["upper_L42"])) if noise else None,
                                 "lower_L2_max_abs_dm": float(max(x for n in noise for x in n["lower_L2"])) if noise else None,
                                 "lower_L2_rms_frac_G": float(np.sqrt(np.mean([(x / n["G"]) ** 2 for n in noise for x in n["lower_L2"]]))) if noise else None}
        S[fam]["within_position_variation"] = [k for k, v in prof.items() if v["n_pairs"] >= 20 and v["frac_toward"] >= 0.1
                                               and v["frac_negligible"] >= 0.1 and v["sd"] > 3 * nsd]
    # substitution arm
    subs = _load_jsonl(out / "subs.jsonl")
    tr = _load_jsonl(out / "sub_transplants.jsonl")
    Gmed = S["brew"]["G_median"] or float("nan")
    bound = {"abs_dlogPa": EPS * Gmed, "TV": 0.05}
    fs = [h for h in subs if h["direction"] == "sub->orig"]
    beh_ok = {h["pair_id"]: abs(h["behaviour_dlogPa"]) < bound["abs_dlogPa"] and h["behaviour_TV"] < bound["TV"] for h in fs}
    trf = [r for r in tr if r["direction"] == "sub->orig"]
    tr_ok = {}
    for r in trf:
        ok = abs(r["dlogPa"]) < bound["abs_dlogPa"] and r["TV"] < bound["TV"]
        tr_ok[r["pair_id"]] = tr_ok.get(r["pair_id"], True) and ok
    by_cl = {}
    for r in trf:
        by_cl.setdefault(f'{r["region"]}|L{r["layer"]}', []).append(r)
    # c = T(a) (the answer's successor, look-ahead content) is tagged at construction and reported separately from
    # truly unrelated off-path colours; both groups stay in every pooled figure above (a tag, not a filter).
    # backfill the tag for heads written before it existed (run stage at 647dcb5): c is T(a), the answer's successor
    _, _bank = wsb.load_bank("brew")
    _succ = {it["name"]: it["table"].get(it["answer"], {}).get("stir") for it in _bank}
    for h in fs:
        if "c_is_succ_answer" not in h:
            h["succ_answer"] = _succ[h["item"]]
            h["c_is_succ_answer"] = h["c"] == _succ[h["item"]]
    succ_ids = {h["pair_id"] for h in fs if h.get("c_is_succ_answer")}
    by_tag = {}
    for name, ids in (("c_is_succ_answer", succ_ids), ("c_unrelated", {h["pair_id"] for h in fs} - succ_ids)):
        hh = [h for h in fs if h["pair_id"] in ids]
        by_tag[name] = {"n_items": len(hh),
                        "behaviour_frac_negligible": float(np.mean([beh_ok[h["pair_id"]] for h in hh])) if hh else None,
                        "behaviour_abs_dlogPa_median": float(np.median([abs(h["behaviour_dlogPa"]) for h in hh])) if hh else None,
                        "transplant_all_cells_frac_negligible": float(np.mean([tr_ok.get(h["pair_id"], True) for h in hh])) if hh else None,
                        "transplant_max_abs_dlogPa": float(max((abs(r["dlogPa"]) for r in trf if r["pair_id"] in ids), default=float("nan"))) if hh else None,
                        "items": sorted(h["pair_id"] for h in hh) if name == "c_is_succ_answer" else None}
    S["substitution"] = {
        "n_items": len(fs), "bound": bound,
        "behaviour_frac_negligible": float(np.mean(list(beh_ok.values()))) if beh_ok else None,
        "behaviour_abs_dlogPa_median": float(np.median([abs(h["behaviour_dlogPa"]) for h in fs])) if fs else None,
        "behaviour_TV_median": float(np.median([h["behaviour_TV"] for h in fs])) if fs else None,
        "transplant_all_cells_frac_negligible": float(np.mean(list(tr_ok.values()))) if tr_ok else None,
        "transplant_by_cell_layer": {k: {"frac_negligible": float(np.mean([abs(r["dlogPa"]) < bound["abs_dlogPa"] and r["TV"] < bound["TV"] for r in v])),
                                         "median_abs_dlogPa": float(np.median([abs(r["dlogPa"]) for r in v])),
                                         "max_abs_dlogPa": float(max(abs(r["dlogPa"]) for r in v)),
                                         "median_TV": float(np.median([r["TV"] for r in v]))} for k, v in sorted(by_cl.items())},
        "by_removed_colour_role": by_tag,
        "key_token_L0_abs_dlogPa_median": float(np.median([abs(h["key_token_L0_dlogPa"]) for h in fs])) if fs else None,
        "exceeding": [h["pair_id"] for h in fs if not (beh_ok.get(h["pair_id"]) and tr_ok.get(h["pair_id"], True))]}
    # probes
    P_ = json.loads((ANA / "probes.json").read_text()) if (ANA / "probes.json").exists() else None
    if P_:
        rows = [r for r in P_["rows"] if r["status"] == "ok"]
        cells, layers = P_["cells"], P_["layers"]
        dec = {}
        for tgt in ("z", "start", "answer"):
            tab = {}
            for r in rows:
                if r["kind"] == "12-way" and r["target"] == tgt:
                    tab[f'{cells[r["cell"]]}|L{layers[r["layer"]]}'] = {k: r[k] for k in ("real", "control", "selectivity", "p", "null_mean", "n", "n_tables")}
            keys = sorted(tab)
            rej, adj = ST.bh_fdr([tab[k]["p"] for k in keys], q=0.05)
            for k, rj, a in zip(keys, rej, adj):
                tab[k]["p_bh"] = a
                tab[k]["decodable"] = bool(rj and (tab[k]["selectivity"] or 0) >= 0.10)
            dec[tgt] = tab
        mem = {}
        for ci, cell in enumerate(cells):
            for li, L in enumerate(layers):
                rr = [r for r in rows if r["kind"] == "membership" and r["cell"] == ci and r["layer"] == li]
                if not rr:
                    continue
                real = float(np.mean([r["real"] for r in rr]))
                nn = min(len(r["null"]) for r in rr)
                null = np.mean([r["null"][:nn] for r in rr], axis=0)
                sel = float(np.mean([r["selectivity"] for r in rr if r["selectivity"] is not None]))
                mem[f"{cell}|L{L}"] = {"pooled_auc": real, "n_colours": len(rr), "null_mean": float(null.mean()),
                                       "p": float((1 + (null >= real).sum()) / (1 + nn)), "selectivity": sel,
                                       "per_colour": {r["target"]: round(r["real"], 3) for r in rr}}
        keys = sorted(mem)
        rej, adj = ST.bh_fdr([mem[k]["p"] for k in keys], q=0.05)
        for k, rj, a in zip(keys, rej, adj):
            mem[k]["p_bh"] = a
            mem[k]["decodable"] = bool(rj and mem[k]["selectivity"] >= 0.10)
        dec["membership"] = mem
        D = json.loads((ANA / "probes_substitution_directional.json").read_text())
        dirtab = {}
        for ci, cell in enumerate(cells):
            for li, L in enumerate(layers):
                rr = [d for d in D if d["cell"] == ci and d["layer"] == li]
                rem = [d["removed"] for d in rr if d["removed"] is not None]
                ins = [d["inserted"] for d in rr if d["inserted"] is not None]
                dirtab[f"{cell}|L{L}"] = {"n_removed": len(rem), "frac_removed_down": float(np.mean(np.array(rem) < 0)) if rem else None,
                                          "mean_removed": float(np.mean(rem)) if rem else None,
                                          "n_inserted": len(ins), "frac_inserted_up": float(np.mean(np.array(ins) > 0)) if ins else None,
                                          "mean_inserted": float(np.mean(ins)) if ins else None}
        dec["substitution_directional"] = dirtab
        S["decodability"] = dec
    # decision rules
    em_cells = ("emit_asst", "emit_think", "emit_colon")
    mem_dec = any(v["decodable"] for k, v in S.get("decodability", {}).get("membership", {}).items())
    sub_neg = (S["substitution"]["behaviour_frac_negligible"] or 0) >= 0.9 and (S["substitution"]["transplant_all_cells_frac_negligible"] or 0) >= 0.9
    eng_ok = smoke["all_engine_checks_pass"] and all((S[f]["controls"].get(k) in (None, 1.0)) for f in ("brew", "chain")
                                                     for k in ("self_patch_bitwise", "embedding_anchor_bitwise", "prefix_identical", "stir_noop_bitwise"))
    S["decision_rules"] = {
        "1_within_position_variation (diagnostic)": {"brew": S["brew"]["within_position_variation"], "chain": S["chain"]["within_position_variation"]},
        "2_brew_not_needed_class": {"membership_decodable_at_emission": mem_dec, "substitution_negligible": sub_neg,
                                    "brew_supplies_not_needed_class": bool(mem_dec and sub_neg)},
        "3_engine_checks_pass": bool(eng_ok),
        "4_power": {"brew_dev_pairs": S["brew"]["n_pairs"], "chain_dev_pairs": S["chain"]["n_pairs"],
                    "ok": S["brew"]["n_pairs"] >= 20 and S["chain"]["n_pairs"] >= 15}}
    (ANA / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    # qualitative sample (fixed seed): 8 brew + 4 chain pairs, every cell-layer share at the read cells, top-5 tokens
    qual = {"seed": QUAL_SEED, "pairs": []}
    for fam, n in (("brew", 8), ("chain", 4)):
        heads = [h for h in _load_jsonl(out / f"pairs_{fam}.jsonl") if h["direction"] == "p->m"]
        pick = random.Random(QUAL_SEED).sample(heads, k=min(n, len(heads)))
        rows = _load_jsonl(out / f"profile_{fam}.jsonl")
        ctr = {c["pair_id"]: c for c in _load_jsonl(out / f"controls_{fam}.jsonl") if c["direction"] == "p->m"}
        for h in pick:
            rr = [r for r in rows if r["pair_id"] == h["pair_id"] and r["direction"] == "p->m"]
            qual["pairs"].append({**h, "family": fam, "start_token_S": ctr[h["pair_id"]]["start_token_S"],
                                  "S_by_region_layer": {f'{r["region"]}@{r["pos"]}|L{r["layer"]}': round(r["S"], 3) for r in rr},
                                  "read_cell_top5": {f'{r["region"]}|L{r["layer"]}': r.get("top5") for r in rr if r.get("top5")}})
    qual["substitution_exceeding"] = [h for h in subs if h["pair_id"] in S["substitution"]["exceeding"]]
    # gate failures (qualitative): a fixed-seed sample of failing prompts with every decode
    gp = _load_jsonl(OUT / "gate" / "gate_prompts.jsonl")
    fails = [g for g in gp if not g["pass"]]
    qual["gate_failures_sample"] = [{k: g[k] for k in ("family", "answer", "greedy", "samples", "n_correct", "text")}
                                    for g in random.Random(QUAL_SEED).sample(fails, k=min(12, len(fails)))]
    (ANA / "qualitative.json").write_text(json.dumps(qual, indent=1, default=float, ensure_ascii=False))
    write_manifest("analysis", {})
    print(json.dumps(S["decision_rules"], indent=1, default=float))


# ---------------------------------------------------------------------------------------------- stage: figures
def stage_figures():
    """ICLR-style figures from results/001_leverage_profile/analysis/summary.json only."""
    import numpy as np
    import figures as F
    ip = F.setup()
    S = json.loads((ANA / "summary.json").read_text())
    FIG = R.out_dir("results", EXP, "figures")

    def cell_order(keys):
        order = {"stir": 0, "start": 1, "question": 2, "emit_asst": 3, "emit_think": 4, "emit_colon": 5, "final": 6}
        return sorted(keys, key=lambda c: (order.get(c.split("#")[0], 9), int(c.split("#")[1]) if "#" in c else 0))
    for fam in ("brew", "chain"):
        prof = S[fam]["profile"]
        cells = cell_order({k.split("|L")[0] for k in prof})
        M = np.array([[prof.get(f"{c}|L{L}", {}).get("median", np.nan) for L in LAYERS] for c in cells])
        f, ax = ip.figure(width="half", aspect=1.6 if fam == "brew" else 1.3)
        ip.heatmap(ax, M, xlabels=[f"L{L}" for L in LAYERS], ylabels=cells, signed=True)
        ax.set_title(f"{fam}: median whole-cell share S (n = {S[fam]['n_pairs']} dev pairs)", fontsize=8)
        F.save(f, str(FIG / f"profile_{fam}"))
    f, ax = ip.figure(width="half")
    for i, fam in enumerate(("brew", "chain")):
        st = S[fam]["controls"].get("start_token_S", {})
        if not st:
            continue
        xs = [int(L) for L in st]
        med = [st[L]["median"] for L in st]
        lo = [st[L]["q25"] for L in st]
        hi = [st[L]["q75"] for L in st]
        ip.line_ci(ax, xs, med, lo, hi, label=fam, color=ip.CATEGORICAL[i])
    ax.set_xlabel("block output layer (start token patched)")
    ax.set_ylabel("share S (median, IQR)")
    ax.legend(frameon=False)
    F.save(f, str(FIG / "start_token_by_layer"))
    sub = S["substitution"]["transplant_by_cell_layer"]
    if sub:
        f, ax = ip.figure(width="half")
        for i, reg in enumerate(("emit_asst", "emit_think", "emit_colon")):
            ys = [sub.get(f"{reg}|L{L}", {}).get("max_abs_dlogPa", np.nan) for L in LAYERS]
            ax.plot(LAYERS, ys, marker="o", color=ip.CATEGORICAL[i], label=reg)
        ip.reference_line(ax, y=S["substitution"]["bound"]["abs_dlogPa"], label="negligible bound")
        ax.set_xlabel("layer")
        ax.set_ylabel("max |Δ log P(answer)| over items")
        ax.legend(frameon=False, fontsize=7)
        F.save(f, str(FIG / "substitution_transplants"))
    print("figures written to", FIG)


if __name__ == "__main__":
    stage = sys.argv[1]
    {"gate": stage_gate, "smoke": stage_smoke, "run": stage_run, "probes": stage_probes, "snapshot": stage_snapshot, "analysis": stage_analysis, "figures": stage_figures}[stage]()
