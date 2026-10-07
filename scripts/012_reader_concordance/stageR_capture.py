"""012 Stage R model side (design §16; EXPLORATORY; label code, no reader imports): Stage F2's protocol, unchanged, on
Jan's causal_st_broadcast bank. Covered items = src and tgt in Stage A's bank_j. Render (rstripped prompt), budget,
matcher and random-target seed root (benchmark, item, "j", "rand_swap") as Jan / Stage A.
Arms: clean, J_L24-35, rand_L24-35 (Jan's SwapEdit, every prompt position, prefill only, alpha 1).
Per item (r_items.jsonl, Stage B style): clean_correct, clean_contains_y_b, success, float32 m / dm (Jan's bf16 m kept),
edit_direction at the read position (Stage B's rule: band-layer mean of clean c_s - c_t > 0 -> toward_tgt), per band
layer ||dh|| and coordinate error at the read position, the model's own float32 read-position output.
F2 (f2_model.jsonl, f2_captures.pt; Stage F schemas): block outputs at every position p <= T-2 at L40/42/44 for clean /
J / rand, and the model's next-token log-probs at p for the src/tgt resolved tokens and y_a/y_b first tokens.
Gates: G0 (alpha 0 bitwise: scores, logits, greedy ids); clean greedy text and budget identical to Stage A's broadcast rows.

  PYTHONPATH=/workspace/sources/ext/phrase-lens-src/src python stageR_capture.py [--limit 4] [--tag smoke]
"""
import argparse
import json
import time

import torch
from phrase_lens.benchmarks import load_benchmark
from phrase_lens.interventions import BandEditor, SwapEdit, answer_generation_budget, random_directions
from phrase_lens.lenses import DirectionBank
from phrase_lens.metrics import causal_outcome
from phrase_lens.modeling import ModelRunner

from stageB_capture import ARM_ID, MODEL, REV, ROOT, SA, Logged, lse
from stageF_capture import AllPos
from tokforms012 import first_token_sets, lse_ids

BENCH = "causal_st_broadcast"
BAND = list(range(24, 36))
LAYERS = [40, 42, 44]
R_ARMS = ["clean", "J_L24-35", "rand_L24-35"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    out = ROOT / "outputs/012_reader_concordance/stageR" / ("" if a.tag == "full" else a.tag)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    runner = ModelRunner.from_pretrained(MODEL, device="cuda", dtype="bfloat16", revision=REV, local_files_only=True)
    tok, blocks, norm = runner.tokenizer, runner.blocks, runner.final_norm
    W32 = runner.model.get_output_embeddings().weight.detach().float()
    bank = DirectionBank.load(SA / "bank_j")
    words = set(bank.words)
    forms = json.loads((SA / "words_st_report.json").read_text())["resolved_form"]
    fid = {w: tok.encode(f, add_special_tokens=False)[0] for w, f in forms.items()}
    stA = {r["name"]: r for r in map(json.loads, open(SA / "runs/j_bcast_L24-47.jsonl"))}
    items = load_benchmark(BENCH)["items"]
    items = items[:a.limit] if a.limit else items

    def gen(ids, edit, budget):
        n, grab = len(ids), {}

        def pre(_m, args):
            if "x" not in grab and args[0].shape[1] == n:
                grab["x"] = args[0][0, :n].detach()
        hk = norm.register_forward_pre_hook(pre)
        ctx = BandEditor(blocks, edit.band, edit, n) if edit is not None else torch.no_grad()
        try:
            with ctx, AllPos(blocks, LAYERS, n) as cap:
                g = runner.generate(ids, edit=None, max_new_tokens=budget, return_token_ids=True)
        finally:
            hk.remove()
        with torch.no_grad():
            lp = torch.log_softmax(norm(grab["x"].float()) @ W32.T, -1).cpu()
        return g, torch.stack([cap.acts[L] for L in LAYERS]), lp

    def one(prompt, s, edit):
        grab = {}
        hk = norm.register_forward_pre_hook(lambda _m, args: grab.__setitem__("x", args[0]))
        try:
            r = runner.continuation_scores(prompt, s, edit=edit)
        finally:
            hk.remove()
        n, tg = r["prompt_tokens"], r["token_ids"]
        with torch.no_grad():
            y = norm(grab["x"][0, n - 1:n - 1 + len(tg)].float())
            lp32 = torch.log_softmax(y @ W32.T, -1)[torch.arange(len(tg)), torch.tensor(tg)].double().sum()
        return float(lp32), r["logprob_sum"]

    rows, f2_rows, capsR, capsF2, names = [], [], [], {}, []
    gates = {"G0_fail": [], "parity_mismatch": [], "n_uncovered": 0}
    for k, it in enumerate(items):
        t1 = time.time()
        ya, yb = [it["y_a"], *it.get("y_a_alts", [])], [it["y_b"], *it.get("y_b_alts", [])]
        base = {"name": it["name"], "src": it["src"], "tgt": it["tgt"], "y_a": ya, "y_b": yb}
        if it["src"] not in words or it["tgt"] not in words:
            rows.append({**base, "status": "uncovered"})
            gates["n_uncovered"] += 1
            continue
        prompt = it["prompt"].rstrip()
        ids = runner.encode(prompt)
        n, pos = len(ids), len(ids) - 1
        ida, idb, _ = first_token_sets(tok, ya, yb)
        ts, tt = fid[it["src"]], fid[it["tgt"]]
        sa = [answer_generation_budget(tok, it["prompt"], ya), answer_generation_budget(tok, it["prompt"], yb)]
        budget = max(sa)
        c = runner.capture_ids(ids, positions=list(range(n)), layers=BAND)
        clean_acts = {L: c[i] for i, L in enumerate(BAND)}
        s_dirs, t_dirs = bank.concept(it["src"], BAND), bank.concept(it["tgt"], BAND)
        r_dirs = random_directions(s_dirs.shape[-1], [BENCH, it["name"], ARM_ID, "rand_swap"], BAND)
        J, R = SwapEdit(s_dirs, t_dirs, clean_acts, BAND, 1.0), SwapEdit(s_dirs, r_dirs, clean_acts, BAND, 1.0)
        gap = []
        for j, L in enumerate(BAND):                                        # Stage B's edit-direction rule
            cc = torch.linalg.pinv(torch.stack([s_dirs[j], t_dirs[j]], 1)) @ clean_acts[L][pos].float()
            gap.append(float(cc[0] - cc[1]))
        direction = "toward_tgt" if sum(gap) / len(gap) > 0 else "toward_src"

        def scores(edit):
            both = {s: one(prompt, s, edit) for s in dict.fromkeys(ya + yb)}
            mk = lambda j: lse([both[s][j] for s in yb]) - lse([both[s][j] for s in ya])
            return {"lp": {s: v[0] for s, v in both.items()}, "m": mk(0), "m_bf16_jan": mk(1)}

        arms, X, LP = {}, {}, {}
        for arm, e in (("clean", None), ("J_L24-35", J), ("rand_L24-35", R)):
            lg = Logged(e, pos) if e is not None else None
            g, X[arm], LP[arm] = gen(ids, lg, budget)
            lp = LP[arm][pos]
            rec = {"ids": g, "text": tok.decode(g, skip_special_tokens=True), **scores(e),
                   "model_out": {"lp_src": float(lp[ts]), "lp_tgt": float(lp[tt]), "lp_ya": lse_ids(lp, ida), "lp_yb": lse_ids(lp, idb)}}
            if e is None:
                rec["clean_correct"] = causal_outcome(it, tok.decode(g[:sa[0]], skip_special_tokens=True), condition="clean")
                ra = stA.get(it["name"], {})
                if ra.get("status") != "evaluated" or ra["result"]["clean"] != rec["text"] or ra["result"]["generation_budgets"]["generated"] != budget \
                        or ra["clean_correct"] != rec["clean_correct"]:
                    gates["parity_mismatch"].append(it["name"])
            else:
                tp = tok.decode(g[:sa[1]], skip_special_tokens=True)
                cm, mo = arms["clean"]["model_out"], rec["model_out"]
                mo["d_model_bridge"] = (mo["lp_tgt"] - mo["lp_src"]) - (cm["lp_tgt"] - cm["lp_src"])
                mo["d_model_answer"] = None if None in (mo["lp_ya"], mo["lp_yb"], cm["lp_ya"], cm["lp_yb"]) else \
                    (mo["lp_yb"] - mo["lp_ya"]) - (cm["lp_yb"] - cm["lp_ya"])
                rec.update(target_prefix=tp, success=causal_outcome(it, tp, condition="swap", collision=e.flags["proxy_collision_all_layers"]),
                           dm=rec["m"] - arms["clean"]["m"], dm_bf16_jan=rec["m_bf16_jan"] - arms["clean"]["m_bf16_jan"],
                           flags=e.flags, diag={str(L): v for L, v in lg.log.items()})
            arms[arm] = rec
        z = SwapEdit(s_dirs, t_dirs, clean_acts, BAND, 0.0)                # G0
        gz, _, lpz = gen(ids, z, budget)
        sz = scores(z)
        g0 = {"greedy_same": gz == arms["clean"]["ids"], "max_abs_dlp": max(abs(sz["lp"][s] - arms["clean"]["lp"][s]) for s in sz["lp"]),
              "max_abs_dlogit": float((lpz - LP["clean"]).abs().max())}
        if not (g0["greedy_same"] and g0["max_abs_dlp"] == 0.0 and g0["max_abs_dlogit"] == 0.0):
            gates["G0_fail"].append([it["name"], g0])
        rows.append({**base, "status": "evaluated", "prompt": prompt, "read_pos": pos, "read_token": tok.decode([ids[pos]]),
                     "budgets": sa, "clean_correct": arms["clean"]["clean_correct"],
                     "clean_contains_y_b": causal_outcome(it, arms["clean"]["text"], condition="swap"),
                     "edit_direction": direction, "mean_cs_minus_ct_read": sum(gap) / len(gap), "answer_first_token_ids": [ida, idb],
                     "stageA_L24-47_success": stA.get(it["name"], {}).get("success"),
                     "arms": {k_: {kk: vv for kk, vv in v.items() if kk != "lp"} for k_, v in arms.items()}, "G0": g0,
                     "sec": round(time.time() - t1, 1)})
        capsR.append(torch.stack([X[arm][:, pos] for arm in R_ARMS]))
        capsF2[it["name"]] = torch.stack([X[arm][:, :pos] for arm in R_ARMS])
        names.append(it["name"])
        cl = LP["clean"]
        for p in range(pos):
            row = {"item": it["name"], "pos": p, "token": tok.decode([ids[p]]), "n_prompt": n,
                   "is_read_minus_1": p == pos - 1, "is_first": p == 0}
            c_ = {"lp_src": float(cl[p, ts]), "lp_tgt": float(cl[p, tt]), "lp_ya": lse_ids(cl[p], ida), "lp_yb": lse_ids(cl[p], idb)}
            for st in ("J_L24-35", "rand_L24-35"):
                e_ = LP[st]
                v = {"lp_src": float(e_[p, ts]), "lp_tgt": float(e_[p, tt]), "lp_ya": lse_ids(e_[p], ida), "lp_yb": lse_ids(e_[p], idb)}
                row[st] = {**v, "clean": c_, "d_model_bridge_p": (v["lp_tgt"] - v["lp_src"]) - (c_["lp_tgt"] - c_["lp_src"]),
                           "d_model_answer_p": None if None in (v["lp_ya"], v["lp_yb"], c_["lp_ya"], c_["lp_yb"]) else
                           (v["lp_yb"] - v["lp_ya"]) - (c_["lp_yb"] - c_["lp_ya"])}
            f2_rows.append(row)
        J_ = arms["J_L24-35"]
        print(f"[{k + 1}/{len(items)}] {it['name']} {time.time() - t1:.1f}s cc={int(arms['clean']['clean_correct'])} "
              f"J:{J_['dm']:+.2f}{'*' if J_['success'] else ''} rand:{arms['rand_L24-35']['dm']:+.2f} dir={direction}", flush=True)

    torch.save({"items": names, "arms": R_ARMS, "layers": LAYERS, "X": torch.stack(capsR),
                "note": "block outputs (float32) at the read position, each arm's greedy prefill"}, out / "r_captures.pt")
    torch.save({"layers": LAYERS, "states": R_ARMS, "X": capsF2,
                "note": "per item [3 states, 3 layers, positions 0..T-2, D] float32, greedy prefill"}, out / "f2_captures.pt")
    for fn, rr in (("r_items.jsonl", rows), ("f2_model.jsonl", f2_rows)):
        with open(out / fn, "w") as f:
            for r in rr:
                f.write(json.dumps(r) + "\n")
    ev = [r for r in rows if r["status"] == "evaluated"]
    gates.update(n_items=len(rows), n_covered=len(ev), wall_s=round(time.time() - t0, 1),
                 n_clean_correct=sum(r["clean_correct"] for r in ev),
                 n_primary=sum(r["clean_correct"] and not r["clean_contains_y_b"] for r in ev))
    gates["pass"] = not (gates["G0_fail"] or gates["parity_mismatch"])
    gp = ROOT / "results/012_reader_concordance/analysis" / ("stageR_gates.json" if a.tag == "full" else f"stageR_gates_{a.tag}.json")
    gp.write_text(json.dumps(gates, indent=1))
    print(json.dumps(gates), flush=True)


if __name__ == "__main__":
    main()
