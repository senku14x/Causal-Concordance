"""012 Stage F model side (design §13; EXPLORATORY). Label code: no reader imports.

Band L24-35 only; Jan's SwapEdit with Stage B's clean coordinates and seed root; float32 margins as stageB_capture.
Arms: clean; J_L24-35 / rand_L24-35 (all positions, existing; re-run for parity and for F2);
      J_readonly_L24-35 / rand_readonly_L24-35 (only the read position written, every band layer);
      J_allbutread_L24-35 / rand_allbutread_L24-35 (every prompt position but the read position written).
A mask wraps Jan's edit: positions outside it get the residual back unchanged; the per-position write norm is logged
in the greedy prefill (gate: zero outside the mask at every band layer).
F1: greedy (Jan's budget, matcher), m (float32; Jan's bf16 kept), dm vs clean, read-position captures L40/42/44, the
model's own float32 read-position output (src/tgt resolved tokens, y_a/y_b first tokens -> d_model_bridge/answer).
F2: for clean / J_L24-35 / rand_L24-35, block outputs at every position p <= T-2 (L40/42/44) and the model's own
next-token log-probs at p for the same tokens (d_model_bridge_p / d_model_answer_p).
Gates: G0 (alpha = 0 under the allbutread mask reproduces clean bitwise: scores and greedy ids); write masks;
parity of clean / J_L24-35 / rand_L24-35 with Stage B (read-position captures bitwise, greedy ids, float32 m).

  PYTHONPATH=/workspace/sources/ext/phrase-lens-src/src python stageF_capture.py [--limit 4] [--tag smoke]
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

from stageB_capture import ARM_ID, ARMS as B_ARMS, BENCH, CAP_LAYERS as B_LAYERS, MODEL, REV, ROOT, SA, lse
from tokforms012 import first_token_sets, lse_ids

BAND = list(range(24, 36))
LAYERS = [40, 42, 44]
F_ARMS = ["clean", "J_L24-35", "rand_L24-35", "J_readonly_L24-35", "rand_readonly_L24-35",
          "J_allbutread_L24-35", "rand_allbutread_L24-35"]


class Masked:
    """Jan's edit at the masked positions only; the rest get the residual back unchanged (bitwise)."""
    def __init__(self, edit, positions, n):
        self.edit, self.band = edit, edit.band
        self.mask = torch.zeros(n, dtype=torch.bool)
        self.mask[list(positions)] = True
        self.write = {}

    def __call__(self, layer, residual):
        out = self.edit(layer, residual)
        out = torch.where(self.mask.to(out.device)[:, None], out, residual)
        if layer not in self.write:
            self.write[layer] = (out - residual).norm(dim=-1).tolist()
        return out


class AllPos:
    """Block outputs at every prompt position in the prefill (registered after any BandEditor)."""
    def __init__(self, blocks, layers, n):
        self.blocks, self.layers, self.n, self.acts, self.h = blocks, layers, n, {}, []

    def __enter__(self):
        for L in self.layers:
            self.h.append(self.blocks[L].register_forward_hook(self._hook(L)))
        return self

    def __exit__(self, *exc):
        for h in self.h:
            h.remove()

    def _hook(self, L):
        def f(_m, _a, out):
            x = out[0] if isinstance(out, tuple) else out
            if L not in self.acts and x.shape[1] == self.n:
                self.acts[L] = x[0, :self.n].detach().float().cpu()
        return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    sub = "" if a.tag == "full" else a.tag
    out = ROOT / "outputs/012_reader_concordance/stageF" / sub
    out.mkdir(parents=True, exist_ok=True)
    bdir = ROOT / "outputs/012_reader_concordance/stageB" / sub
    t0 = time.time()
    runner = ModelRunner.from_pretrained(MODEL, device="cuda", dtype="bfloat16", revision=REV, local_files_only=True)
    tok, blocks, norm = runner.tokenizer, runner.blocks, runner.final_norm
    W32 = runner.model.get_output_embeddings().weight.detach().float()
    bank = DirectionBank.load(SA / "bank_j")
    forms = json.loads((SA / "words_st_report.json").read_text())["resolved_form"]
    fid = {w: tok.encode(f, add_special_tokens=False)[0] for w, f in forms.items()}
    capB = torch.load(bdir / "captures.pt")
    rowsB = {r["name"]: r for r in map(json.loads, open(bdir / "items.jsonl")) if r["status"] == "evaluated"}
    items = [it for it in load_benchmark(BENCH)["items"] if it["name"] in rowsB][:a.limit]

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
            y = norm(grab["x"].float())
            assert y.dtype == torch.float32
            lp = torch.log_softmax(y @ W32.T, -1).cpu()                   # [n, V] float32
        return g, torch.stack([cap.acts[L] for L in LAYERS]), lp       # X [3 layers, n, D]

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

    f1, f2_rows, capsF1, capsF2, names = [], [], [], {}, []
    gates = {"write_mask_violations": [], "parity_mismatch": [], "G0_fail": [], "max_write_outside_mask": 0.0}
    for k, it in enumerate(items):
        t1 = time.time()
        rb = rowsB[it["name"]]
        prompt = it["prompt"].rstrip()
        ids = runner.encode(prompt)
        n, pos = len(ids), len(ids) - 1
        ya, yb = [it["y_a"], *it.get("y_a_alts", [])], [it["y_b"], *it.get("y_b_alts", [])]
        ida, idb, _ = first_token_sets(tok, ya, yb)
        ts, tt = fid[it["src"]], fid[it["tgt"]]
        sa = [answer_generation_budget(tok, it["prompt"], ya), answer_generation_budget(tok, it["prompt"], yb)]
        budget = max(sa)
        c = runner.capture_ids(ids, positions=list(range(n)), layers=BAND)
        clean_acts = {L: c[i] for i, L in enumerate(BAND)}
        s_dirs, t_dirs = bank.concept(it["src"], BAND), bank.concept(it["tgt"], BAND)
        r_dirs = random_directions(s_dirs.shape[-1], [BENCH, it["name"], ARM_ID, "rand_swap"], BAND)
        J, R = SwapEdit(s_dirs, t_dirs, clean_acts, BAND, 1.0), SwapEdit(s_dirs, r_dirs, clean_acts, BAND, 1.0)
        make = {"clean": lambda: None, "J_L24-35": lambda: J, "rand_L24-35": lambda: R,
                "J_readonly_L24-35": lambda: Masked(J, [pos], n), "rand_readonly_L24-35": lambda: Masked(R, [pos], n),
                "J_allbutread_L24-35": lambda: Masked(J, range(pos), n),
                "rand_allbutread_L24-35": lambda: Masked(R, range(pos), n)}

        def scores(edit):
            both = {s: one(prompt, s, edit) for s in dict.fromkeys(ya + yb)}
            mk = lambda j: lse([both[s][j] for s in yb]) - lse([both[s][j] for s in ya])
            return {"lp": {s: v[0] for s, v in both.items()}, "m": mk(0), "m_bf16_jan": mk(1)}

        arms, Xr, LP = {}, {}, {}
        for arm in F_ARMS:
            e = make[arm]()
            g, X, lp = gen(ids, e, budget)
            Xr[arm], LP[arm] = X, lp
            tp = dec = tok.decode(g[:sa[1]], skip_special_tokens=True)
            rec = {"ids": g, "text": tok.decode(g, skip_special_tokens=True), **scores(make[arm]())}
            if arm == "clean":
                rec["clean_correct"] = causal_outcome(it, tok.decode(g[:sa[0]], skip_special_tokens=True), condition="clean")
            else:
                rec.update(target_prefix=tp, success=causal_outcome(it, dec, condition="swap",
                                                                   collision=(J if arm.startswith("J") else R).flags["proxy_collision_all_layers"]),
                           dm=rec["m"] - arms["clean"]["m"], dm_bf16_jan=rec["m_bf16_jan"] - arms["clean"]["m_bf16_jan"])
            if isinstance(e, Masked):                                      # write-mask gate
                outside = max(v for L in BAND for p, v in enumerate(e.write[L]) if not e.mask[p])
                inside_read = [e.write[L][pos] for L in BAND]
                gates["max_write_outside_mask"] = max(gates["max_write_outside_mask"], outside)
                if outside != 0.0 or set(e.write) != set(BAND):
                    gates["write_mask_violations"].append([it["name"], arm, outside])
                rec["write_norm_read_pos"] = inside_read
                rec["write_norm_other_pos_max"] = max(v for L in BAND for p, v in enumerate(e.write[L]) if p != pos)
            rec["model_out"] = {"lp_src": float(lp[pos, ts]), "lp_tgt": float(lp[pos, tt]),
                                "lp_ya": lse_ids(lp[pos], ida), "lp_yb": lse_ids(lp[pos], idb)}
            if arm in B_ARMS:                                               # parity with Stage B
                xb = capB["X"][capB["items"].index(it["name"]), B_ARMS.index(arm)]
                dx = float((X[:, pos] - xb[[B_LAYERS.index(L) for L in LAYERS]]).abs().max())
                if dx != 0.0 or g != rb["arms"][arm]["ids"] or rec["m"] != rb["arms"][arm]["m"]:
                    gates["parity_mismatch"].append([it["name"], arm, dx, g == rb["arms"][arm]["ids"],
                                                     rec["m"] - rb["arms"][arm]["m"]])
            arms[arm] = rec
        mo = lambda r_, k1, k2: None if None in (r_[k1], r_[k2], arms["clean"]["model_out"][k1], arms["clean"]["model_out"][k2]) \
            else (r_[k2] - r_[k1]) - (arms["clean"]["model_out"][k2] - arms["clean"]["model_out"][k1])
        for arm in F_ARMS[1:]:
            m_ = arms[arm]["model_out"]
            m_["d_model_bridge"], m_["d_model_answer"] = mo(m_, "lp_src", "lp_tgt"), mo(m_, "lp_ya", "lp_yb")
        z = Masked(SwapEdit(s_dirs, t_dirs, clean_acts, BAND, 0.0), range(pos), n)   # G0
        gz, _, lpz = gen(ids, z, budget)
        sz = scores(Masked(SwapEdit(s_dirs, t_dirs, clean_acts, BAND, 0.0), range(pos), n))
        g0 = {"greedy_same": gz == arms["clean"]["ids"], "max_abs_dlp": max(abs(sz["lp"][s] - arms["clean"]["lp"][s]) for s in sz["lp"]),
              "max_abs_dlogit": float((lpz - LP["clean"]).abs().max())}
        if not (g0["greedy_same"] and g0["max_abs_dlp"] == 0.0 and g0["max_abs_dlogit"] == 0.0):
            gates["G0_fail"].append([it["name"], g0])
        f1.append({"name": it["name"], "src": it["src"], "tgt": it["tgt"], "y_a": ya, "y_b": yb, "read_pos": pos,
                   "clean_correct": arms["clean"]["clean_correct"], "budgets": sa, "answer_first_token_ids": [ida, idb],
                   "arms": {k_: {kk: vv for kk, vv in v.items() if kk != "lp"} for k_, v in arms.items()}, "G0": g0,
                   "edit_direction_L24-35": rb["edit_direction"]["L24-35"], "sec": round(time.time() - t1, 1)})
        capsF1.append(torch.stack([Xr[arm][:, pos] for arm in F_ARMS]))     # [7 arms, 3 layers, D]
        capsF2[it["name"]] = torch.stack([Xr[arm][:, :pos] for arm in ("clean", "J_L24-35", "rand_L24-35")])
        names.append(it["name"])
        cl = LP["clean"]
        for p in range(pos):
            row = {"item": it["name"], "pos": p, "token": tok.decode([ids[p]]), "n_prompt": n,
                   "is_read_minus_1": p == pos - 1, "is_first": p == 0}
            for st in ("J_L24-35", "rand_L24-35"):
                e_ = LP[st]
                v = {"lp_src": float(e_[p, ts]), "lp_tgt": float(e_[p, tt]), "lp_ya": lse_ids(e_[p], ida), "lp_yb": lse_ids(e_[p], idb)}
                c_ = {"lp_src": float(cl[p, ts]), "lp_tgt": float(cl[p, tt]), "lp_ya": lse_ids(cl[p], ida), "lp_yb": lse_ids(cl[p], idb)}
                row[st] = {**v, "clean": c_, "d_model_bridge_p": (v["lp_tgt"] - v["lp_src"]) - (c_["lp_tgt"] - c_["lp_src"]),
                           "d_model_answer_p": None if None in (v["lp_ya"], v["lp_yb"]) else
                           (v["lp_yb"] - v["lp_ya"]) - (c_["lp_yb"] - c_["lp_ya"])}
            f2_rows.append(row)
        print(f"[{k + 1}/{len(items)}] {it['name']} {time.time() - t1:.1f}s "
              + " ".join(f"{a_.replace('_L24-35', '')}:{arms[a_]['dm']:+.2f}{'*' if arms[a_]['success'] else ''}" for a_ in F_ARMS[1:]),
              flush=True)

    torch.save({"items": names, "arms": F_ARMS, "layers": LAYERS, "X": torch.stack(capsF1),
                "note": "block outputs (float32) at the read position, each arm's greedy prefill"}, out / "f1_captures.pt")
    torch.save({"layers": LAYERS, "states": ["clean", "J_L24-35", "rand_L24-35"], "X": capsF2,
                "note": "per item [3 states, 3 layers, positions 0..T-2, D] float32, greedy prefill"}, out / "f2_captures.pt")
    for fn, rows in (("f1_items.jsonl", f1), ("f2_model.jsonl", f2_rows)):
        with open(out / fn, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    gates.update(n_items=len(f1), wall_s=round(time.time() - t0, 1),
                 pass_=not (gates["write_mask_violations"] or gates["parity_mismatch"] or gates["G0_fail"]))
    gates["pass"] = gates.pop("pass_")
    gp = ROOT / "results/012_reader_concordance/analysis" / ("stageF_gates.json" if a.tag == "full" else f"stageF_gates_{a.tag}.json")
    gp.write_text(json.dumps(gates, indent=1))
    print(json.dumps(gates), flush=True)


if __name__ == "__main__":
    main()
