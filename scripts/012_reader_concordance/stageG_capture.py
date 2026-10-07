"""012 Stage G model side (design §14; EXPLORATORY). Label code: no reader imports.

Whitened J-row swap at every prompt position, band L24-35, alpha 1 (008 §1.4 covector construction, Σ from
stageG_sigma.py): V~ = unit(W^-1 [s, t]); c~ = V~^+ W (h - mu); clamp to swap(c~_clean) (clean run, same layer and
position); h' = h + W^-1 V~ (swap(c~_clean) - c~). Control rand_white: src row + Jan's seeded random unit target (same
seed root), whitened the same way. Same engine as Stage B/F (Jan's ModelRunner, BandEditor, generate path, budget,
matcher; float32 margins; Jan's bf16 m kept).
Per item x arm {clean, J_white_L24-35, rand_white_L24-35}: greedy + success, m, dm, realised ||dh|| and coordinate
error per band layer at the read position, edit direction from the whitened clean coordinates at the read position
(toward_tgt iff the band-layer mean of c~_s - c~_t > 0), captures at the read position L40/42/44, the model's own
float32 read-position output (d_model_bridge / d_model_answer).
Gates: G0 (alpha = 0 whitened edit reproduces clean bitwise: scores, read-position logits, greedy ids); R A = I per layer
(float64, <= 1e-9); clean parity with Stage B (captures, greedy ids, float32 m).

  PYTHONPATH=/workspace/sources/ext/phrase-lens-src/src python stageG_capture.py [--limit 4] [--tag smoke]
"""
import argparse
import json
import time

import torch
from phrase_lens.benchmarks import load_benchmark
from phrase_lens.interventions import BandEditor, answer_generation_budget, random_directions
from phrase_lens.lenses import DirectionBank
from phrase_lens.metrics import causal_outcome
from phrase_lens.modeling import ModelRunner

from stageB_capture import ARM_ID, ARMS as B_ARMS, BENCH, CAP_LAYERS as B_LAYERS, MODEL, REV, ROOT, SA, PrefillCapture, lse
from tokforms012 import first_token_sets, lse_ids

BAND = list(range(24, 36))
LAYERS = [40, 42, 44]
G_ARMS = ["clean", "J_white_L24-35", "rand_white_L24-35"]


class WhiteSwap:
    """Jan's clamp in 008's whitened coordinates (J rows as covectors)."""
    def __init__(self, s_dirs, t_dirs, clean_acts, band, sig, alpha=1.0, pos=None, collision_cos=.999):
        self.band, self.alpha, self.pos = list(band), alpha, pos
        self.A, self.Rm, self.mu, self.targets, self.c_clean, self.log = {}, {}, {}, {}, {}, {}
        self.cos_white, self.ra_err, self.collided = [], [], []
        for j, L in enumerate(self.band):
            sg = sig[L]
            Vt = sg["Wi"] @ torch.stack([s_dirs[j], t_dirs[j]], 1).double()
            Vt = Vt / Vt.norm(dim=0, keepdim=True)
            self.cos_white.append(float(Vt[:, 0] @ Vt[:, 1]))
            if self.cos_white[-1] > collision_cos:
                self.collided.append(L)
                continue
            A, Rm = sg["Wi"] @ Vt, torch.linalg.pinv(Vt) @ sg["W"]
            self.ra_err.append(float((Rm @ A - torch.eye(2, dtype=torch.float64)).abs().max()))
            c = Rm @ (clean_acts[L].double() - sg["mu"]).T                 # [2, n] whitened clean coordinates
            self.c_clean[L] = c
            self.targets[L] = c[[1, 0]].float()
            self.A[L], self.Rm[L], self.mu[L] = A.float(), Rm.float(), sg["mu"].float()
        self._dev = {}

    def _on(self, L, dev):
        if L not in self._dev:
            self._dev[L] = tuple(x.to(dev) for x in (self.A[L], self.Rm[L], self.mu[L], self.targets[L]))
        return self._dev[L]

    def __call__(self, layer, residual):
        if layer in self.collided:
            return residual
        A, Rm, mu, tg = self._on(layer, residual.device)
        out = residual + self.alpha * (A @ (tg - Rm @ (residual - mu).T)).T
        if self.pos is not None and layer not in self.log:
            p = self.pos
            a32, abf = Rm @ (out[p] - mu), Rm @ (out[p].to(torch.bfloat16).float() - mu)
            self.log[layer] = {"dh": float((out[p] - residual[p]).norm()), "h": float(residual[p].norm()),
                               "coord_err_fp32": float((a32 - tg[:, p]).abs().max()),
                               "coord_err_bf16": float((abf - tg[:, p]).abs().max())}
        return out

    @property
    def flags(self):
        return {"swap_mode": "whitened_clamp_covector", "alpha": self.alpha, "cos_white": self.cos_white,
                "proxy_collision_layers": self.collided, "proxy_collision_all_layers": len(self.collided) == len(self.band),
                "ra_err_max": max(self.ra_err) if self.ra_err else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    sub = "" if a.tag == "full" else a.tag
    out = ROOT / "outputs/012_reader_concordance/stageG" / sub
    out.mkdir(parents=True, exist_ok=True)
    bdir = ROOT / "outputs/012_reader_concordance/stageB" / sub
    t0 = time.time()
    sig = torch.load(ROOT / "outputs/012_reader_concordance/stageG/sigma_L24-35.pt")
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
                grab["x"] = args[0][0, n - 1].detach()
        hk = norm.register_forward_pre_hook(pre)
        ctx = BandEditor(blocks, edit.band, edit, n) if edit is not None else torch.no_grad()
        try:
            with ctx, PrefillCapture(blocks, LAYERS, n, n - 1) as pc:
                g = runner.generate(ids, edit=None, max_new_tokens=budget, return_token_ids=True)
        finally:
            hk.remove()
        with torch.no_grad():
            lp = torch.log_softmax(norm(grab["x"].float()) @ W32.T, -1).cpu()
        return g, torch.stack([pc.acts[L] for L in LAYERS]), lp

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

    rows, caps, names = [], [], []
    gates = {"parity_mismatch": [], "G0_fail": [], "ra_err_max": 0.0, "collisions": []}
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
        c = runner.capture_ids(ids, positions=list(range(n)), layers=BAND)
        clean_acts = {L: c[i] for i, L in enumerate(BAND)}
        s_dirs, t_dirs = bank.concept(it["src"], BAND), bank.concept(it["tgt"], BAND)
        r_dirs = random_directions(s_dirs.shape[-1], [BENCH, it["name"], ARM_ID, "rand_swap"], BAND)
        make = {"clean": lambda **kw: None,
                "J_white_L24-35": lambda **kw: WhiteSwap(s_dirs, t_dirs, clean_acts, BAND, sig, **kw),
                "rand_white_L24-35": lambda **kw: WhiteSwap(s_dirs, r_dirs, clean_acts, BAND, sig, **kw)}

        def scores(edit):
            both = {s: one(prompt, s, edit) for s in dict.fromkeys(ya + yb)}
            mk = lambda j: lse([both[s][j] for s in yb]) - lse([both[s][j] for s in ya])
            return {"lp": {s: v[0] for s, v in both.items()}, "m": mk(0), "m_bf16_jan": mk(1)}

        arms, LP, X = {}, {}, {}
        for arm in G_ARMS:
            e = make[arm](pos=pos)
            g, X[arm], LP[arm] = gen(ids, e, max(sa))
            rec = {"ids": g, "text": tok.decode(g, skip_special_tokens=True), **scores(make[arm]())}
            lp = LP[arm]
            rec["model_out"] = {"lp_src": float(lp[ts]), "lp_tgt": float(lp[tt]), "lp_ya": lse_ids(lp, ida), "lp_yb": lse_ids(lp, idb)}
            if arm == "clean":
                rec["clean_correct"] = causal_outcome(it, tok.decode(g[:sa[0]], skip_special_tokens=True), condition="clean")
                xb = capB["X"][capB["items"].index(it["name"]), B_ARMS.index("clean")]
                dx = float((X[arm] - xb[[B_LAYERS.index(L) for L in LAYERS]]).abs().max())
                if dx != 0.0 or g != rb["arms"]["clean"]["ids"] or rec["m"] != rb["arms"]["clean"]["m"]:
                    gates["parity_mismatch"].append([it["name"], dx, g == rb["arms"]["clean"]["ids"], rec["m"] - rb["arms"]["clean"]["m"]])
            else:
                tp = tok.decode(g[:sa[1]], skip_special_tokens=True)
                gap = sum(float(e.c_clean[L][0, pos] - e.c_clean[L][1, pos]) for L in e.c_clean) / max(len(e.c_clean), 1)
                rec.update(target_prefix=tp, success=causal_outcome(it, tp, condition="swap", collision=e.flags["proxy_collision_all_layers"]),
                           dm=rec["m"] - arms["clean"]["m"], dm_bf16_jan=rec["m_bf16_jan"] - arms["clean"]["m_bf16_jan"],
                           flags=e.flags, diag={str(L): v for L, v in e.log.items()},
                           white_coords_read={str(L): [float(e.c_clean[L][0, pos]), float(e.c_clean[L][1, pos])] for L in e.c_clean},
                           mean_cs_minus_ct_read_white=gap, edit_direction_white="toward_tgt" if gap > 0 else "toward_src")
                gates["ra_err_max"] = max(gates["ra_err_max"], e.flags["ra_err_max"] or 0.0)
                if e.collided:
                    gates["collisions"].append([it["name"], arm, e.collided])
                mo, cm = rec["model_out"], arms["clean"]["model_out"]
                mo["d_model_bridge"] = (mo["lp_tgt"] - mo["lp_src"]) - (cm["lp_tgt"] - cm["lp_src"])
                mo["d_model_answer"] = None if None in (mo["lp_ya"], mo["lp_yb"]) else (mo["lp_yb"] - mo["lp_ya"]) - (cm["lp_yb"] - cm["lp_ya"])
            arms[arm] = rec
        z = WhiteSwap(s_dirs, t_dirs, clean_acts, BAND, sig, alpha=0.0)                  # G0
        gz, _, lpz = gen(ids, z, max(sa))
        sz = scores(WhiteSwap(s_dirs, t_dirs, clean_acts, BAND, sig, alpha=0.0))
        g0 = {"greedy_same": gz == arms["clean"]["ids"], "max_abs_dlp": max(abs(sz["lp"][s] - arms["clean"]["lp"][s]) for s in sz["lp"]),
              "max_abs_dlogit": float((lpz - LP["clean"]).abs().max())}
        if not (g0["greedy_same"] and g0["max_abs_dlp"] == 0.0 and g0["max_abs_dlogit"] == 0.0):
            gates["G0_fail"].append([it["name"], g0])
        rows.append({"name": it["name"], "src": it["src"], "tgt": it["tgt"], "y_a": ya, "y_b": yb, "read_pos": pos,
                     "clean_correct": arms["clean"]["clean_correct"], "budgets": sa,
                     "edit_direction_raw_L24-35": rb["edit_direction"]["L24-35"],
                     "raw_J_L24-35_success": rb["arms"]["J_L24-35"]["success"],
                     "arms": {k_: {kk: vv for kk, vv in v.items() if kk != "lp"} for k_, v in arms.items()}, "G0": g0,
                     "sec": round(time.time() - t1, 1)})
        caps.append(torch.stack([X[arm] for arm in G_ARMS]))
        names.append(it["name"])
        J_, R_ = arms["J_white_L24-35"], arms["rand_white_L24-35"]
        print(f"[{k + 1}/{len(items)}] {it['name']} {time.time() - t1:.1f}s J_white:{J_['dm']:+.2f}{'*' if J_['success'] else ''} "
              f"rand_white:{R_['dm']:+.2f}{'*' if R_['success'] else ''} dir={J_['edit_direction_white']} "
              f"dh_read={J_['diag'][str(BAND[-1])]['dh']:.2f}", flush=True)

    torch.save({"items": names, "arms": G_ARMS, "layers": LAYERS, "X": torch.stack(caps),
                "note": "block outputs (float32) at the read position, each arm's greedy prefill"}, out / "g_captures.pt")
    with open(out / "g_items.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    gates.update(n_items=len(rows), wall_s=round(time.time() - t0, 1))
    gates["pass"] = not (gates["parity_mismatch"] or gates["G0_fail"]) and gates["ra_err_max"] <= 1e-9
    gp = ROOT / "results/012_reader_concordance/analysis" / ("stageG_gates.json" if a.tag == "full" else f"stageG_gates_{a.tag}.json")
    gp.write_text(json.dumps(gates, indent=1))
    print(json.dumps(gates), flush=True)


if __name__ == "__main__":
    main()
