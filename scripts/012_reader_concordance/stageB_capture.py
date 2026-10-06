"""012 Stage B-lite (model side; design §12): Jan's J swap re-run with margins, edit diagnostics and captures. EXPLORATORY.

Label code: never imports reader code (no scripts/common/readers, no lens). Built on phrase-lens's own objects
(ModelRunner, SwapEdit, BandEditor, random_directions, answer_generation_budget, causal_outcome), with the exact Stage A
recipe (render = bank prompt rstripped; random target seed root = (benchmark, item name, arm "j"), "rand_swap").

Per covered item of causal_st_multihop and arm (clean; J_L24-35 + rand_L24-35; J_L24-47 + rand_L24-47):
  greedy continuation (Jan's budget, matcher), teacher-forced log P of every y_a / y_b spelling (+alts) via
  continuation_scores -> m = LSE(y_b) - LSE(y_a), dm vs clean. Primary m is float32 (margin.py convention: final-norm
  output recomputed in float32 from the same edited forward x float32 W_U, every answer token of every spelling); Jan's
  bf16-logit m is kept as m_bf16_jan. Swap-direction diagnostic per band layer (clean pinv coordinates c_s, c_t at the
  read position and their mean difference over prompt positions, u.h_clean with u = (s-t)/|s-t|; edit_direction =
  toward_tgt iff the band-layer mean of (c_s - c_t) at the read position is > 0). Edit diagnostics per band layer
  (||dh||, coordinates before / after the clamp, coordinate error in fp32 and after the bf16 cast); captures of the
  block outputs at the read position (final token of the render), L40/42/44/52/56, taken in the greedy run's prefill
  (the exact labelled forward), float32.
Gates: G-parity (clean / J / random greedy text identical to Stage A's rows, paired success table); G0 (the alpha = 0
edit reproduces clean bitwise: every bf16 and float32 score, and the greedy ids). Not a gate: floor_identity_clamp =
|dm float32| of Jan's clamp written back to the unswapped clean coordinates (the bf16-write noise floor). Read position checked against WSB's last_word_token offsets.

  PYTHONPATH=/workspace/sources/ext/phrase-lens-src/src python stageB_capture.py [--limit 4] [--tag smoke]
"""
import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from phrase_lens.benchmarks import load_benchmark
from phrase_lens.interventions import BandEditor, SwapEdit, answer_generation_budget, random_directions
from phrase_lens.lenses import DirectionBank
from phrase_lens.metrics import causal_outcome
from phrase_lens.modeling import ModelRunner

ROOT = Path(__file__).resolve().parents[2]
SA = ROOT / "outputs/012_reader_concordance/stageA"
WSB = Path("/workspace/sources/repos/workspace-bench/evals/multihop/items.json")
MODEL, REV = "Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
BENCH, ARM_ID = "causal_st_multihop", "j"
BANDS = {"L24-35": list(range(24, 36)), "L24-47": list(range(24, 48))}
ARMS = ["clean", "J_L24-35", "rand_L24-35", "J_L24-47", "rand_L24-47"]
CAP_LAYERS = [40, 42, 44, 52, 56]


class Logged:
    """Delegates to a SwapEdit unchanged; records per band layer what the clamp did at the read position."""
    def __init__(self, edit, pos):
        self.edit, self.band, self.pos, self.log = edit, edit.band, pos, {}

    def __call__(self, layer, residual):
        out = self.edit(layer, residual)
        if layer not in self.log:
            p, rec = self.pos, {}
            rec["dh"] = float((out[p] - residual[p]).norm())
            rec["h"] = float(residual[p].norm())
            if layer in self.edit.targets:
                i = self.edit.band.index(layer)
                pinv = torch.linalg.pinv(torch.stack([self.edit.s[i], self.edit.t[i]], 1).to(out))
                tgt = self.edit.targets[layer][:, p].to(out)
                after32, afterbf = pinv @ out[p], pinv @ out[p].to(torch.bfloat16).float()
                rec.update(coord_before=(pinv @ residual[p]).tolist(), coord_target=tgt.tolist(),
                           coord_err_fp32=float((after32 - tgt).abs().max()),
                           coord_err_bf16=float((afterbf - tgt).abs().max()))
            self.log[layer] = rec
        return out


class PrefillCapture:
    """Block outputs at one position in the prefill (first forward with n tokens); registered after any BandEditor,
    so band layers are captured post-edit."""
    def __init__(self, blocks, layers, n, pos):
        self.blocks, self.layers, self.n, self.pos, self.acts, self.h = blocks, layers, n, pos, {}, []

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
                self.acts[L] = x[0, self.pos].detach().float().cpu()
        return f


def lse(xs):
    return float(torch.logsumexp(torch.tensor(xs, dtype=torch.float64), 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    out = ROOT / "outputs/012_reader_concordance/stageB" / ("" if a.tag == "full" else a.tag)
    out.mkdir(parents=True, exist_ok=True)
    gates_path = ROOT / "results/012_reader_concordance/analysis" / (
        "stageB_gates.json" if a.tag == "full" else f"stageB_gates_{a.tag}.json")
    gates_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    runner = ModelRunner.from_pretrained(MODEL, device="cuda", dtype="bfloat16", revision=REV, local_files_only=True)
    tok, blocks = runner.tokenizer, runner.blocks
    bank = DirectionBank.load(SA / "bank_j")
    items = load_benchmark(BENCH)["items"][:a.limit] if a.limit else load_benchmark(BENCH)["items"]
    wsb = {it["name"]: it for it in json.loads(WSB.read_text())["items"]}
    stageA = {b: {r["name"]: r for r in map(json.loads, open(SA / f"runs/j_swap_{b}.jsonl"))} for b in BANDS}
    words = set(bank.words)
    W32 = runner.model.get_output_embeddings().weight.detach().float()
    norm = runner.final_norm
    rows, caps, cap_names = [], [], []
    print(f"loaded in {time.time() - t0:.0f}s; {len(items)} items", flush=True)

    def gen(ids, edit, budget, pos):
        ctx = BandEditor(blocks, edit.band, edit, len(ids)) if edit is not None else nullcontext()
        with ctx, PrefillCapture(blocks, CAP_LAYERS, len(ids), pos) as cap:   # == runner.generate(ids, edit=edit)
            gids = runner.generate(ids, edit=None, max_new_tokens=budget, return_token_ids=True)
        assert set(cap.acts) == set(CAP_LAYERS)
        return gids, torch.stack([cap.acts[L] for L in CAP_LAYERS])

    for k, it in enumerate(items):
        t1 = time.time()
        row = {"name": it["name"], "src": it["src"], "tgt": it["tgt"], "y_a": it["y_a"], "y_b": it["y_b"],
               "y_a_alts": it.get("y_a_alts", []), "y_b_alts": it.get("y_b_alts", []),
               "headline_eligible": it["headline_eligible"]}
        if it["src"] not in words or it["tgt"] not in words:
            rows.append({**row, "status": "uncovered"})
            continue
        prompt = it["prompt"].rstrip()
        ids = runner.encode(prompt)
        pos = len(ids) - 1
        # read position vs WSB last_word_token (offsets counted from the end of the unstripped tokenization)
        w = wsb.get(it["name"])
        if w is None:
            rp = {"in_wsb": False}
        else:
            full = list(tok(w["prompt"], add_special_tokens=True).input_ids)
            wpos = len(full) + w["readout"]["offsets"][0]
            rp = {"in_wsb": True, "wsb_pos": wpos, "wsb_prompt_same": w["prompt"] == it["prompt"],
                  "match": wpos == pos and full[:pos + 1] == ids, "token": tok.decode([ids[pos]])}
        sa = [answer_generation_budget(tok, it["prompt"], [it["y_a"], *row["y_a_alts"]]),
              answer_generation_budget(tok, it["prompt"], [it["y_b"], *row["y_b_alts"]])]
        budget = max(sa)
        spell_a, spell_b = [it["y_a"], *row["y_a_alts"]], [it["y_b"], *row["y_b_alts"]]

        def one(s, edit):
            grab = {}
            hk = norm.register_forward_pre_hook(lambda _m, args: grab.__setitem__("x", args[0]))
            try:
                r = runner.continuation_scores(prompt, s, edit=edit)
            finally:
                hk.remove()
            n, tg = r["prompt_tokens"], r["token_ids"]
            with torch.no_grad():
                y = norm(grab["x"][0, n - 1:n - 1 + len(tg)].float())
                assert y.dtype == torch.float32
                lp32 = torch.log_softmax(y @ W32.T, -1)[torch.arange(len(tg)), torch.tensor(tg)].double().sum()
            return float(lp32), r["logprob_sum"]

        def scores(edit):
            both = {s: one(s, edit) for s in dict.fromkeys(spell_a + spell_b)}
            lp, lpj = {s: v[0] for s, v in both.items()}, {s: v[1] for s, v in both.items()}
            mk = lambda d: lse([d[s] for s in spell_b]) - lse([d[s] for s in spell_a])
            return {"lp": lp, "m": mk(lp), "lp_bf16_jan": lpj, "m_bf16_jan": mk(lpj)}

        dec = lambda x: tok.decode(x, skip_special_tokens=True)
        arms, X = {}, {}
        gids, X["clean"] = gen(ids, None, budget, pos)
        arms["clean"] = {"ids": gids, "text": dec(gids), "source_prefix": dec(gids[:sa[0]]), **scores(None)}
        arms["clean"]["clean_correct"] = causal_outcome(it, arms["clean"]["source_prefix"], condition="clean")
        g0, swap_dir = {}, {}
        for b, layers in BANDS.items():
            cap = runner.capture_ids(ids, positions=list(range(len(ids))), layers=layers)
            clean_acts = {L: cap[i] for i, L in enumerate(layers)}
            s_dirs = bank.concept(it["src"], layers)
            t_dirs = bank.concept(it["tgt"], layers)
            sd = {}
            for j, L in enumerate(layers):                                  # swap-direction diagnostic (clean coords)
                V = torch.stack([s_dirs[j], t_dirs[j]], 1)
                c = torch.linalg.pinv(V) @ clean_acts[L].float().T            # [2, n_prompt]
                u = (s_dirs[j] - t_dirs[j]) / (s_dirs[j] - t_dirs[j]).norm()
                sd[str(L)] = {"c_s": float(c[0, pos]), "c_t": float(c[1, pos]),
                              "mean_cs_minus_ct_all_pos": float((c[0] - c[1]).mean()),
                              "u_dot_h": float(u @ clean_acts[L][pos].float()),
                              "mean_u_dot_h_all_pos": float((clean_acts[L].float() @ u).mean())}
            gap = sum(v["c_s"] - v["c_t"] for v in sd.values()) / len(sd)
            swap_dir[b] = {"edit_direction": "toward_tgt" if gap > 0 else "toward_src",
                           "mean_cs_minus_ct_read": gap, "layers": sd}
            edits = {f"J_{b}": SwapEdit(s_dirs, bank.concept(it["tgt"], layers), clean_acts, layers, 1.0),
                     f"rand_{b}": SwapEdit(s_dirs, random_directions(s_dirs.shape[-1], [BENCH, it["name"], ARM_ID, "rand_swap"],
                                                                      layers), clean_acts, layers, 1.0)}
            for arm, e in edits.items():
                lg = Logged(e, pos)
                gids, X[arm] = gen(ids, lg, budget, pos)
                tp = dec(gids[:sa[1]])
                arms[arm] = {"ids": gids, "text": dec(gids), "target_prefix": tp,
                             "success": causal_outcome(it, tp, condition="swap", collision=e.flags["proxy_collision_all_layers"]),
                             "flags": e.flags, "applied_layers": [L for L in layers if L in e.targets and L in lg.log],
                             "diag": {str(L): v for L, v in lg.log.items()}, **scores(e)}
                arms[arm]["dm"] = arms[arm]["m"] - arms["clean"]["m"]
                arms[arm]["dm_bf16_jan"] = arms[arm]["m_bf16_jan"] - arms["clean"]["m_bf16_jan"]
            if b == "L24-47":                                               # G0: zero edits on Jan's clamp object
                z = SwapEdit(s_dirs, bank.concept(it["tgt"], layers), clean_acts, layers, 0.0)
                ident = SwapEdit(s_dirs, bank.concept(it["tgt"], layers), clean_acts, layers, 1.0)
                ident.targets = {L: t[[1, 0]] for L, t in ident.targets.items()}   # un-swap: clamp to own clean coords
                for nm, e in (("alpha0", z), ("identity_clamp", ident)):
                    sc = scores(e)
                    gi, _ = gen(ids, e, budget, pos)
                    g0[nm] = {"max_abs_dlp": max(abs(sc["lp"][s] - arms["clean"]["lp"][s]) for s in sc["lp"]),
                              "max_abs_dlp_bf16_jan": max(abs(sc["lp_bf16_jan"][s] - arms["clean"]["lp_bf16_jan"][s])
                                                          for s in sc["lp"]),
                              "dm": sc["m"] - arms["clean"]["m"], "dm_bf16_jan": sc["m_bf16_jan"] - arms["clean"]["m_bf16_jan"],
                              "greedy_same": gi == arms["clean"]["ids"]}
        # G-parity against Stage A rows (decoded continuations as Stage A saved them)
        par = {}
        for b in BANDS:
            r = stageA[b][it["name"]]
            res = r["result"]
            par[b] = {"clean_same": res["clean"] == arms["clean"]["text"],
                      "J_same": res["intervention"] == arms[f"J_{b}"]["text"],
                      "rand_same": res["random"] == arms[f"rand_{b}"]["text"],
                      "stageA_success": r["success"], "ours_success": arms[f"J_{b}"]["success"],
                      "stageA_random_success": r["random_success"], "ours_random_success": arms[f"rand_{b}"]["success"],
                      "stageA_clean_correct": r["clean_correct"], "budget_same": res["generation_budgets"]["generated"] == budget,
                      "prompt_tokens_same": res["prompt_tokens"] == len(ids)}
        row.update(status="evaluated", prompt=prompt, read_pos=pos, read_token=tok.decode([ids[pos]]), read_check=rp,
                   budgets={"source": sa[0], "target": sa[1], "generated": budget},
                   clean_correct=arms["clean"]["clean_correct"],
                   clean_contains_y_b=causal_outcome(it, arms["clean"]["text"], condition="swap"),
                   arms=arms, G0=g0, floor_identity_clamp=abs(g0["identity_clamp"]["dm"]), swap_direction=swap_dir,
                   edit_direction={b_: v["edit_direction"] for b_, v in swap_dir.items()}, parity=par, sec=round(time.time() - t1, 1))
        rows.append(row)
        caps.append(torch.stack([X[a_] for a_ in ARMS]))
        cap_names.append(it["name"])
        print(f"[{k + 1}/{len(items)}] {it['name']} {time.time() - t1:.1f}s m_clean={arms['clean']['m']:.2f} "
              + " ".join(f"{a_}:dm={arms[a_]['dm']:+.3f}" for a_ in ARMS[1:]) + f" dir={row['edit_direction']}", flush=True)

    torch.save({"items": cap_names, "arms": ARMS, "layers": CAP_LAYERS, "X": torch.stack(caps),
                "note": "block outputs (float32) at the read position, captured in each arm's greedy prefill"},
               out / "captures.pt")
    with open(out / "items.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ev = [r for r in rows if r["status"] == "evaluated"]
    gates = {"n_items": len(rows), "n_evaluated": len(ev), "wall_s": round(time.time() - t0, 1)}
    for b in BANDS:
        mism = [{"name": r["name"], **{k_: v for k_, v in r["parity"][b].items() if k_.endswith("_same") and not v}}
                for r in ev if not all(v for k_, v in r["parity"][b].items() if k_.endswith("_same"))]
        tab = {}
        for r in ev:
            key = f"stageA_{int(r['parity'][b]['stageA_success'])}_ours_{int(r['parity'][b]['ours_success'])}"
            tab[key] = tab.get(key, 0) + 1
        gates[f"G_parity_{b}"] = {"pass": not mism, "mismatches": mism, "paired_success": tab,
                                  "random_success_agree": sum(r["parity"][b]["stageA_random_success"] == r["parity"][b]["ours_random_success"] for r in ev)}
    g0a = max(max(r["G0"]["alpha0"]["max_abs_dlp"], r["G0"]["alpha0"]["max_abs_dlp_bf16_jan"]) for r in ev)
    g0g = all(r["G0"]["alpha0"]["greedy_same"] for r in ev)
    fl = sorted(r["floor_identity_clamp"] for r in ev)
    gates["G0"] = {"alpha0_max_abs_dlp_f32_and_bf16": g0a, "alpha0_greedy_same": g0g,
                   "rule": "alpha = 0 edit reproduces clean bitwise (every float32 and bf16 score) and the greedy ids",
                   "pass": g0a == 0.0 and g0g}
    gates["floor_identity_clamp"] = {"note": "not a gate: |dm float32| of the identity clamp (bf16-write floor)",
                                     "median": fl[len(fl) // 2], "max": fl[-1],
                                     "greedy_same_all": all(r["G0"]["identity_clamp"]["greedy_same"] for r in ev)}
    gates["read_position"] = {"n_checked": sum(r["read_check"].get("in_wsb", False) for r in ev),
                              "mismatches": [{"name": r["name"], **r["read_check"]} for r in ev
                                             if r["read_check"].get("in_wsb") and not r["read_check"]["match"]],
                              "not_in_wsb": [r["name"] for r in ev if not r["read_check"].get("in_wsb")]}
    gates_path.write_text(json.dumps(gates, indent=1))
    print(json.dumps({k_: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ("mismatches", "not_in_wsb")})
                      for k_, v in gates.items()}, indent=1), flush=True)


if __name__ == "__main__":
    main()
