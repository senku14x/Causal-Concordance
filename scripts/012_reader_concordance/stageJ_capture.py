"""012 Stage J model side (design §19; EXPLORATORY; label code, no reader imports). Stage I's prompts and engine.

J1 (relaxed gate, fixed 8-token budget everywhere): clean greedy + 10 samples at T 0.8 / top_p 0.95 (seed SEED + prompt
index, as Stage I); gate = greedy contains y_a (Jan's matcher), >= 8/10 samples contain y_a, clean greedy does not contain
y_b; greedy for J_pre / rand_pre / J_all (Stage I's edits, masks and seeds) -> success at budget 8.
J2 (clue-only edit): J_clue / rand_clue = Jan's SwapEdit (band L24-35, J_NP rows of Stage A's bank_j; random target seed
root ["012I_delayed", clue_id, "j", "rand_swap"] as Stage I) masked to positions < H1. Per prompt: greedy (budget 8) +
success, float32 m / dm, G0 (alpha 0 under the clue mask: scores, greedy, H1/H2 logits bitwise), write-mask check (no
write at positions >= H1). Per hold (clue x carrier): H1/H2 captures L40/42/44 for clean / J_clue / rand_clue (bitwise
identity across the 3 questions checked; clean parity with Stage I's hold_captures.pt checked), the model's own float32
next-token log-prob shift at H1 and H2 (src/tgt tokens; first tokens of the three relations' answers).

  PYTHONPATH=/workspace/sources/ext/phrase-lens-src/src python stageJ_capture.py [--clues 2] [--tag smoke]
"""
import argparse
import json
import time
from collections import defaultdict

import torch
from phrase_lens.interventions import BandEditor, SwapEdit, random_directions
from phrase_lens.lenses import DirectionBank
from phrase_lens.metrics import strict_match
from phrase_lens.modeling import ModelRunner

from stageB_capture import MODEL, REV, ROOT, SA, lse
from stageF_capture import Masked
from stageI_capture import BAND, LAYERS, SEED, Cap, D as DI
from tokforms012 import first_token_sets, lse_ids

BUDGET = 8
D = ROOT / "outputs/012_reader_concordance/stageJ"
J1_ARMS = ["J_pre", "rand_pre", "J_all"]
J2_ARMS = ["J_clue", "rand_clue"]
HOLD_ARMS = ["clean", "J_clue", "rand_clue"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clues", type=int)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    out = D / ("" if a.tag == "full" else a.tag)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    spec = json.loads((DI / "prompts.json").read_text())
    clues = list(dict.fromkeys(p["clue_id"] for p in spec["items"]))[:a.clues]
    items = [p for p in spec["items"] if p["clue_id"] in clues]
    capI = torch.load(DI / ("" if a.tag == "full" else a.tag) / "hold_captures.pt")
    runner = ModelRunner.from_pretrained(MODEL, device="cuda", dtype="bfloat16", revision=REV, local_files_only=True)
    tok, blocks, norm = runner.tokenizer, runner.blocks, runner.final_norm
    W32 = runner.model.get_output_embeddings().weight.detach().float()
    bank = DirectionBank.load(SA / "bank_j")
    forms = json.loads((SA / "words_st_report.json").read_text())["resolved_form"]
    fid = {w: tok.encode(f, add_special_tokens=False)[0] for w, f in forms.items()}
    enc = lambda s: list(tok(s, add_special_tokens=True).input_ids)
    dec = lambda g: tok.decode(g, skip_special_tokens=True)

    def gen(ids, edit, positions):
        n, grab = len(ids), {}

        def pre(_m, args):
            if "x" not in grab and args[0].shape[1] == n:
                grab["x"] = args[0][0, positions].detach()
        hk = norm.register_forward_pre_hook(pre)
        ctx = BandEditor(blocks, edit.band, edit, n) if edit is not None else torch.no_grad()
        try:
            with ctx, Cap(blocks, LAYERS, n, positions) as cap:
                g = runner.generate(ids, edit=None, max_new_tokens=BUDGET, return_token_ids=True)
        finally:
            hk.remove()
        with torch.no_grad():
            lp = torch.log_softmax(norm(grab["x"].float()) @ W32.T, -1).cpu()
        return g, torch.stack([cap.acts[L] for L in LAYERS], 1), lp

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

    rel_answers = defaultdict(dict)
    for p in spec["items"]:
        rel_answers[p["clue_id"]][p["relation"]] = ([p["y_a"], *p["y_a_alts"]], [p["y_b"], *p["y_b_alts"]])
    gates = {"G0_fail": [], "write_mask_violations": [], "hold_identity": {}, "clean_parity_vs_stageI": {}}
    j1, j2, hold = [], [], {}
    for k, it in enumerate(spec["items"]):
        if it["clue_id"] not in clues:
            continue
        t1 = time.time()
        prompt = it["prompt"].rstrip()
        ids = enc(prompt)
        n = len(ids)
        H1, H2 = len(enc(it["prefix_H1"])) - 1, len(enc(it["prefix_H2"])) - 1
        assert ids[:H2 + 1] == enc(it["prefix_H2"])
        ya, yb = [it["y_a"], *it["y_a_alts"]], [it["y_b"], *it["y_b_alts"]]
        c = runner.capture_ids(ids, positions=list(range(n)), layers=BAND)
        clean_acts = {L: c[i] for i, L in enumerate(BAND)}
        s_dirs, t_dirs = bank.concept(it["src"], BAND), bank.concept(it["tgt"], BAND)
        r_dirs = random_directions(s_dirs.shape[-1], ["012I_delayed", it["clue_id"], "j", "rand_swap"], BAND)
        J, Rr = SwapEdit(s_dirs, t_dirs, clean_acts, BAND, 1.0), SwapEdit(s_dirs, r_dirs, clean_acts, BAND, 1.0)
        pre_pos, clue_pos = range(H2 + 1), range(H1)
        make = {"clean": lambda: None, "J_pre": lambda: Masked(J, pre_pos, n), "rand_pre": lambda: Masked(Rr, pre_pos, n),
                "J_all": lambda: J, "J_clue": lambda: Masked(J, clue_pos, n), "rand_clue": lambda: Masked(Rr, clue_pos, n)}
        ft = {r: first_token_sets(tok, *rel_answers[it["clue_id"]][r]) for r in rel_answers[it["clue_id"]]}
        ts, tt = fid[it["src"]], fid[it["tgt"]]

        def outs(lp):
            return [{"lp_src": float(l[ts]), "lp_tgt": float(l[tt]),
                     **{f"lp_{r}_{s}": lse_ids(l, ft[r][i]) for r in ft for i, s in ((0, "ya"), (1, "yb"))}} for l in lp]

        def scores(edit):
            both = {s: one(prompt, s, edit) for s in dict.fromkeys(ya + yb)}
            mk = lambda j: lse([both[s][j] for s in yb]) - lse([both[s][j] for s in ya])
            return {"lp": {s: v[0] for s, v in both.items()}, "m": mk(0), "m_bf16_jan": mk(1)}

        res, X, LP = {}, {}, {}
        for arm in ["clean", *J1_ARMS, *J2_ARMS]:
            e = make[arm]()
            g, X[arm], LP[arm] = gen(ids, e, [H1, H2])
            rec = {"ids": g, "text": dec(g)}
            if arm == "clean":
                torch.manual_seed(SEED + k)
                x = torch.tensor([ids] * 10, device=runner.device)
                with torch.no_grad():
                    smp = runner.model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=True, temperature=0.8,
                                                top_p=0.95, max_new_tokens=BUDGET, pad_token_id=tok.eos_token_id)
                st = [dec(s[n:]) for s in smp]
                rec.update(samples=st, k_of_10=sum(strict_match(it["y_a"], s, it["y_a_alts"]) for s in st),
                           greedy_ya=strict_match(it["y_a"], rec["text"], it["y_a_alts"]),
                           greedy_yb=strict_match(it["y_b"], rec["text"], it["y_b_alts"]))
            else:
                rec["success"] = strict_match(it["y_b"], rec["text"], it["y_b_alts"])
            if arm in ("clean", *J2_ARMS):
                rec.update(scores(make[arm]()))
            if isinstance(e, Masked):
                outside = max(v for L in BAND for p, v in enumerate(e.write[L]) if not e.mask[p])
                if outside != 0.0 or set(e.write) != set(BAND):
                    gates["write_mask_violations"].append([it["id"], arm, outside])
                if arm in J2_ARMS:
                    rec["write_ge_H1_max"] = max(v for L in BAND for p, v in enumerate(e.write[L]) if p >= H1)
            res[arm] = rec
        for arm in J2_ARMS:
            res[arm]["dm"] = res[arm]["m"] - res["clean"]["m"]
            res[arm]["dm_bf16_jan"] = res[arm]["m_bf16_jan"] - res["clean"]["m_bf16_jan"]
        z = Masked(SwapEdit(s_dirs, t_dirs, clean_acts, BAND, 0.0), clue_pos, n)
        gz, _, lpz = gen(ids, z, [H1, H2])
        sz = scores(Masked(SwapEdit(s_dirs, t_dirs, clean_acts, BAND, 0.0), clue_pos, n))
        g0 = {"greedy_same": gz == res["clean"]["ids"], "max_abs_dlp": max(abs(sz["lp"][s] - res["clean"]["lp"][s]) for s in sz["lp"]),
              "max_abs_dlogit_H": float((lpz - LP["clean"]).abs().max())}
        if not (g0["greedy_same"] and g0["max_abs_dlp"] == 0.0 and g0["max_abs_dlogit_H"] == 0.0):
            gates["G0_fail"].append([it["id"], g0])
        cr = res["clean"]
        gate = cr["greedy_ya"] and cr["k_of_10"] >= 8 and not cr["greedy_yb"]
        base = {"id": it["id"], "clue_id": it["clue_id"], "carrier": it["carrier"], "relation": it["relation"], "src": it["src"],
                "tgt": it["tgt"], "y_a": ya, "y_b": yb, "discriminating": it["discriminating"], "H1": H1, "H2": H2, "budget": BUDGET}
        j1.append({**base, "j1_gated": gate, "k_of_10": cr["k_of_10"], "greedy_ya": cr["greedy_ya"], "greedy_yb": cr["greedy_yb"],
                   "clean_text": cr["text"], "samples": cr["samples"],
                   "arms": {a_: {"text": res[a_]["text"], "success": res[a_]["success"]} for a_ in J1_ARMS}})
        j2.append({**base, "j1_gated": gate, "m_clean": cr["m"], "m_clean_bf16_jan": cr["m_bf16_jan"], "G0": g0,
                   "arms": {a_: {kk: vv for kk, vv in res[a_].items() if kk not in ("lp", "ids")} for a_ in J2_ARMS}})
        hk = f"{it['clue_id']}|{it['carrier']}"
        hd = hold.setdefault(hk, {"X": {a_: [] for a_ in HOLD_ARMS}, "LP": {a_: [] for a_ in HOLD_ARMS}, "ft": ft, "src": it["src"], "tgt": it["tgt"]})
        for a_ in HOLD_ARMS:
            hd["X"][a_].append(X[a_])
            hd["LP"][a_].append(LP[a_])
        print(f"[{len(j1)}/{len(items)}] {it['id']} {time.time() - t1:.1f}s j1gate={int(gate)} k={cr['k_of_10']} "
              + " ".join(f"{a_}:{int(res[a_]['success'])}" for a_ in J1_ARMS + J2_ARMS)
              + f" dm_clue {res['J_clue']['dm']:+.2f} rand {res['rand_clue']['dm']:+.2f}", flush=True)

    # holds: identity across questions, clean parity with Stage I, own-output shifts at H1 / H2
    keys = list(hold)
    hrows, HX = [], []
    for hk in keys:
        hd = hold[hk]
        dx = max(float((x - hd["X"][a_][0]).abs().max()) for a_ in HOLD_ARMS for x in hd["X"][a_])
        dl = max(float((x - hd["LP"][a_][0]).abs().max()) for a_ in HOLD_ARMS for x in hd["LP"][a_])
        gates["hold_identity"][hk] = {"n_prompts": len(hd["X"]["clean"]), "max_abs_diff_X": dx, "max_abs_diff_logprob": dl}
        xi = capI["X"][capI["holds"].index(hk), capI["arms"].index("clean")]
        gates["clean_parity_vs_stageI"][hk] = float((hd["X"]["clean"][0] - xi).abs().max())
        HX.append(torch.stack([hd["X"][a_][0] for a_ in HOLD_ARMS]))
        o = {a_: outs(hd["LP"][a_][0]) for a_ in HOLD_ARMS}
        sh = {}
        for a_ in J2_ARMS:
            for pi, P in enumerate(("H1", "H2")):
                e_, c_ = o[a_][pi], o["clean"][pi]
                sh[f"{a_}|{P}"] = {"bridge": (e_["lp_tgt"] - e_["lp_src"]) - (c_["lp_tgt"] - c_["lp_src"]),
                                   **{r: None if None in (e_[f"lp_{r}_ya"], e_[f"lp_{r}_yb"], c_[f"lp_{r}_ya"], c_[f"lp_{r}_yb"]) else
                                      (e_[f"lp_{r}_yb"] - e_[f"lp_{r}_ya"]) - (c_[f"lp_{r}_yb"] - c_[f"lp_{r}_ya"]) for r in hd["ft"]}}
        hrows.append({"hold": hk, "clue_id": hk.split("|")[0], "carrier": hk.split("|")[1], "src": hd["src"], "tgt": hd["tgt"],
                      "own_logprobs": o, "d_model": sh, "answer_first_tokens": hd["ft"]})
    torch.save({"holds": keys, "arms": HOLD_ARMS, "positions": ["H1", "H2"], "layers": LAYERS, "X": torch.stack(HX),
                "meta": {hk: {"clue_id": hk.split("|")[0], "carrier": hk.split("|")[1], "src": hold[hk]["src"], "tgt": hold[hk]["tgt"]}
                         for hk in keys}}, out / "j2_hold_captures.pt")
    for fn, rr in (("j1_items.jsonl", j1), ("j2_items.jsonl", j2), ("j2_hold.jsonl", hrows)):
        with open(out / fn, "w") as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    gates.update(n_prompts=len(j1), n_j1_gated=sum(r["j1_gated"] for r in j1),
                 n_j1_gated_discriminating=sum(r["j1_gated"] and r["discriminating"] for r in j1),
                 hold_identity_all_bitwise=all(v["max_abs_diff_X"] == 0.0 and v["max_abs_diff_logprob"] == 0.0 for v in gates["hold_identity"].values()),
                 clean_parity_all_bitwise=all(v == 0.0 for v in gates["clean_parity_vs_stageI"].values()), wall_s=round(time.time() - t0, 1))
    gates["pass"] = not (gates["G0_fail"] or gates["write_mask_violations"])
    gp = ROOT / "results/012_reader_concordance/analysis" / ("stageJ_gates.json" if a.tag == "full" else f"stageJ_gates_{a.tag}.json")
    gp.write_text(json.dumps(gates, indent=1))
    print(json.dumps({k_: v for k_, v in gates.items() if k_ not in ("hold_identity", "clean_parity_vs_stageI")}), flush=True)


if __name__ == "__main__":
    main()
