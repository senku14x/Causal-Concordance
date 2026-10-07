"""012 Stage I model side (design §18; EXPLORATORY; label code, no reader imports): delayed question.

Prompts: outputs/012_reader_concordance/stageI/prompts.json (read only). Plain render, trailing whitespace stripped (Jan).
H1 / H2 = last token of encode(prefix_H1) / encode(prefix_H2) (asserted to be prefixes of the prompt's tokens).
Gates: clean greedy contains y_a (Jan's budget over y_a / y_b spellings, his matcher on the source-budget prefix),
>= 9/10 at T 0.8 / top_p 0.95 (same budget, seeded), clean output does not contain y_b; tgt fact checks greedy-correct.
Arms (band L24-35, J_NP rows of Stage A's bank_j, Jan's SwapEdit, prefill only): J_pre (positions 0..H2 via stageF's
Masked), rand_pre (same mask, source-preserving random target, seed root ["012I_delayed", clue_id, "j", "rand_swap"]),
J_all (every position). Per trial: greedy + success, float32 m / dm, G0 (alpha 0 under the J_pre mask, bitwise), write
mask check, edit direction at H2 (band-layer mean of clean pinv c_s - c_t), the model's own float32 next-token
log-prob shift at H2 for the src/tgt tokens and the first tokens of all three relations' answers (tokforms012).
Hold states: residuals at H1 / H2, L40/42/44, for clean / J_pre / rand_pre; bitwise identity across the three
questions of a (clue, carrier) is checked; one state per (clue, carrier) is saved.

  PYTHONPATH=/workspace/sources/ext/phrase-lens-src/src python stageI_capture.py [--clues 2] [--tag smoke]
"""
import argparse
import json
import time
from collections import defaultdict

import torch
from phrase_lens.interventions import BandEditor, SwapEdit, answer_generation_budget, random_directions
from phrase_lens.lenses import DirectionBank
from phrase_lens.metrics import causal_outcome, strict_match
from phrase_lens.modeling import ModelRunner

from stageB_capture import MODEL, REV, ROOT, SA, lse
from stageF_capture import Masked
from tokforms012 import first_token_sets, lse_ids

BAND = list(range(24, 36))
LAYERS = [40, 42, 44]
ARMS = ["clean", "J_pre", "rand_pre", "J_all"]
HOLD_ARMS = ["clean", "J_pre", "rand_pre"]
SEED = 20261007
D = ROOT / "outputs/012_reader_concordance/stageI"


class Cap:
    """Block outputs at given positions in the prefill (after any BandEditor hook)."""
    def __init__(self, blocks, layers, n, positions):
        self.blocks, self.layers, self.n, self.pos, self.acts, self.h = blocks, layers, n, positions, {}, []

    def __enter__(self):
        for L in self.layers:
            self.h.append(self.blocks[L].register_forward_hook(self._hook(L)))
        return self

    def __exit__(self, *e):
        for h in self.h:
            h.remove()

    def _hook(self, L):
        def f(_m, _a, out):
            x = out[0] if isinstance(out, tuple) else out
            if L not in self.acts and x.shape[1] == self.n:
                self.acts[L] = x[0, self.pos].detach().float().cpu()
        return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clues", type=int)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    out = D / ("" if a.tag == "full" else a.tag)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    spec = json.loads((D / "prompts.json").read_text())
    clues = list(dict.fromkeys(p["clue_id"] for p in spec["items"]))[:a.clues]
    items = [p for p in spec["items"] if p["clue_id"] in clues]
    runner = ModelRunner.from_pretrained(MODEL, device="cuda", dtype="bfloat16", revision=REV, local_files_only=True)
    tok, blocks, norm = runner.tokenizer, runner.blocks, runner.final_norm
    W32 = runner.model.get_output_embeddings().weight.detach().float()
    bank = DirectionBank.load(SA / "bank_j")
    forms = json.loads((SA / "words_st_report.json").read_text())["resolved_form"]
    fid = {w: tok.encode(f, add_special_tokens=False)[0] for w, f in forms.items()}
    enc = lambda s: list(tok(s, add_special_tokens=True).input_ids)

    def gen(ids, edit, budget, positions):
        n, grab = len(ids), {}

        def pre(_m, args):
            if "x" not in grab and args[0].shape[1] == n:
                grab["x"] = args[0][0, positions].detach()
        hk = norm.register_forward_pre_hook(pre)
        ctx = BandEditor(blocks, edit.band, edit, n) if edit is not None else torch.no_grad()
        try:
            with ctx, Cap(blocks, LAYERS, n, positions) as cap:
                g = runner.generate(ids, edit=None, max_new_tokens=budget, return_token_ids=True)
        finally:
            hk.remove()
        with torch.no_grad():
            lp = torch.log_softmax(norm(grab["x"].float()) @ W32.T, -1).cpu()       # [len(positions), V]
        return g, torch.stack([cap.acts[L] for L in LAYERS], 1), lp               # X [len(positions), 3 layers, D]

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

    gates = {"prefix_mismatch": [], "G0_fail": [], "write_mask_violations": [], "hold_identity": {}, "tgt_checks": []}
    # tgt fact checks
    for c in spec["tgt_fact_checks"]:
        if c["country"] not in {p["tgt"] for p in items}:
            continue
        pr = c["prompt"].rstrip()
        b = answer_generation_budget(tok, c["prompt"], c["answers"])
        g = runner.generate(enc(pr), max_new_tokens=b, return_token_ids=True)
        txt = tok.decode(g, skip_special_tokens=True)
        gates["tgt_checks"].append({**c, "text": txt, "pass": strict_match(c["answers"][0], txt, c["answers"][1:])})
    rel_answers = defaultdict(dict)                      # clue -> relation -> (ya list, yb list)
    for p in spec["items"]:
        rel_answers[p["clue_id"]][p["relation"]] = ([p["y_a"], *p["y_a_alts"]], [p["y_b"], *p["y_b_alts"]])
    rows, holds, holdX = [], {}, {}
    for k, it in enumerate(items):
        t1 = time.time()
        prompt = it["prompt"].rstrip()
        ids = enc(prompt)
        n = len(ids)
        h = []
        for key in ("prefix_H1", "prefix_H2"):
            q = enc(it[key])
            if ids[:len(q)] != q:
                gates["prefix_mismatch"].append([it["id"], key])
            h.append(len(q) - 1)
        H1, H2 = h
        ya, yb = [it["y_a"], *it["y_a_alts"]], [it["y_b"], *it["y_b_alts"]]
        sa = [answer_generation_budget(tok, it["prompt"], ya), answer_generation_budget(tok, it["prompt"], yb)]
        budget = max(sa)
        item = {"y_a": it["y_a"], "y_a_alts": it["y_a_alts"], "y_b": it["y_b"], "y_b_alts": it["y_b_alts"]}
        c = runner.capture_ids(ids, positions=list(range(n)), layers=BAND)
        clean_acts = {L: c[i] for i, L in enumerate(BAND)}
        s_dirs, t_dirs = bank.concept(it["src"], BAND), bank.concept(it["tgt"], BAND)
        r_dirs = random_directions(s_dirs.shape[-1], ["012I_delayed", it["clue_id"], "j", "rand_swap"], BAND)
        J, Rr = SwapEdit(s_dirs, t_dirs, clean_acts, BAND, 1.0), SwapEdit(s_dirs, r_dirs, clean_acts, BAND, 1.0)
        pre_pos = range(H2 + 1)
        make = {"clean": lambda: None, "J_pre": lambda: Masked(J, pre_pos, n), "rand_pre": lambda: Masked(Rr, pre_pos, n),
                "J_all": lambda: J}
        gap = []
        for j, L in enumerate(BAND):
            cc = torch.linalg.pinv(torch.stack([s_dirs[j], t_dirs[j]], 1)) @ clean_acts[L][H2].float()
            gap.append(float(cc[0] - cc[1]))
        ft = {r: first_token_sets(tok, *rel_answers[it["clue_id"]][r]) for r in rel_answers[it["clue_id"]]}
        ts, tt = fid[it["src"]], fid[it["tgt"]]

        def scores(edit):
            both = {s: one(prompt, s, edit) for s in dict.fromkeys(ya + yb)}
            mk = lambda j: lse([both[s][j] for s in yb]) - lse([both[s][j] for s in ya])
            return {"lp": {s: v[0] for s, v in both.items()}, "m": mk(0), "m_bf16_jan": mk(1)}

        arms, LP = {}, {}
        for arm in ARMS:
            e = make[arm]()
            g, X, lp = gen(ids, e, budget, [H1, H2])
            LP[arm] = lp
            if arm in HOLD_ARMS:
                holdX.setdefault((it["clue_id"], it["carrier"]), {}).setdefault(arm, []).append(X)
            l2 = lp[1]
            rec = {"ids": g, "text": tok.decode(g, skip_special_tokens=True), **scores(make[arm]()),
                   "out_H2": {"lp_src": float(l2[ts]), "lp_tgt": float(l2[tt]),
                              **{f"lp_{r}_{s}": lse_ids(l2, ft[r][i]) for r in ft for i, s in ((0, "ya"), (1, "yb"))}}}
            if arm == "clean":
                rec["clean_correct"] = causal_outcome(item, tok.decode(g[:sa[0]], skip_special_tokens=True), condition="clean")
                rec["clean_contains_y_b"] = strict_match(it["y_b"], rec["text"], it["y_b_alts"])
                torch.manual_seed(SEED + k)
                x = torch.tensor([ids] * 10, device=runner.device)
                with torch.no_grad():
                    smp = runner.model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=True, temperature=0.8,
                                                top_p=0.95, max_new_tokens=budget, pad_token_id=tok.eos_token_id)
                st = [tok.decode(s[n:n + sa[0]], skip_special_tokens=True) for s in smp]
                rec["k_of_10"] = sum(strict_match(it["y_a"], s, it["y_a_alts"]) for s in st)
                rec["samples"] = st
            else:
                tp = tok.decode(g[:sa[1]], skip_special_tokens=True)
                rec.update(target_prefix=tp, success=causal_outcome(item, tp, condition="swap", collision=(Rr if arm == "rand_pre" else J).flags["proxy_collision_all_layers"]),
                           dm=rec["m"] - arms["clean"]["m"], dm_bf16_jan=rec["m_bf16_jan"] - arms["clean"]["m_bf16_jan"])
                o, cl = rec["out_H2"], arms["clean"]["out_H2"]
                rec["d_model_H2"] = {"bridge": (o["lp_tgt"] - o["lp_src"]) - (cl["lp_tgt"] - cl["lp_src"]),
                                     **{r: None if None in (o[f"lp_{r}_ya"], o[f"lp_{r}_yb"], cl[f"lp_{r}_ya"], cl[f"lp_{r}_yb"]) else
                                        (o[f"lp_{r}_yb"] - o[f"lp_{r}_ya"]) - (cl[f"lp_{r}_yb"] - cl[f"lp_{r}_ya"]) for r in ft}}
                if isinstance(e, Masked):
                    outside = max(v for L in BAND for p, v in enumerate(e.write[L]) if not e.mask[p])
                    if outside != 0.0 or set(e.write) != set(BAND):
                        gates["write_mask_violations"].append([it["id"], arm, outside])
                    rec["write_norm_H2"] = [e.write[L][H2] for L in BAND]
            arms[arm] = rec
        z = Masked(SwapEdit(s_dirs, t_dirs, clean_acts, BAND, 0.0), pre_pos, n)
        gz, _, lpz = gen(ids, z, budget, [H1, H2])
        sz = scores(Masked(SwapEdit(s_dirs, t_dirs, clean_acts, BAND, 0.0), pre_pos, n))
        g0 = {"greedy_same": gz == arms["clean"]["ids"], "max_abs_dlp": max(abs(sz["lp"][s] - arms["clean"]["lp"][s]) for s in sz["lp"]),
              "max_abs_dlogit_H": float((lpz - LP["clean"]).abs().max())}
        if not (g0["greedy_same"] and g0["max_abs_dlp"] == 0.0 and g0["max_abs_dlogit_H"] == 0.0):
            gates["G0_fail"].append([it["id"], g0])
        cr = arms["clean"]
        gated = cr["clean_correct"] and cr["k_of_10"] >= 9 and not cr["clean_contains_y_b"]
        rows.append({"id": it["id"], "clue_id": it["clue_id"], "carrier": it["carrier"], "relation": it["relation"],
                     "src": it["src"], "tgt": it["tgt"], "y_a": ya, "y_b": yb, "discriminating": it["discriminating"],
                     "prompt": prompt, "H1": H1, "H2": H2, "n_prompt": n, "budgets": sa, "gated": gated,
                     "edit_direction_H2": "toward_tgt" if sum(gap) / len(gap) > 0 else "toward_src",
                     "mean_cs_minus_ct_H2": sum(gap) / len(gap), "answer_first_tokens": ft,
                     "arms": {k_: {kk: vv for kk, vv in v.items() if kk != "lp"} for k_, v in arms.items()}, "G0": g0,
                     "sec": round(time.time() - t1, 1)})
        print(f"[{k + 1}/{len(items)}] {it['id']} {time.time() - t1:.1f}s gated={int(gated)} k={cr['k_of_10']} "
              + " ".join(f"{a_}:{arms[a_]['dm']:+.2f}{'*' if arms[a_]['success'] else ''}" for a_ in ARMS[1:]), flush=True)
    # hold-state identity across the questions of a (clue, carrier); one state saved per (clue, carrier)
    keys = sorted(holdX)
    for key in keys:
        per = holdX[key]
        dev = max(float((x - per[arm][0]).abs().max()) for arm in HOLD_ARMS for x in per[arm])
        gates["hold_identity"]["|".join(key)] = {"n_prompts": len(per["clean"]), "max_abs_diff": dev}
        holds["|".join(key)] = torch.stack([per[arm][0] for arm in HOLD_ARMS])                      # [3 arms, 2 pos, 3 layers, D]
    torch.save({"holds": list(holds), "arms": HOLD_ARMS, "positions": ["H1", "H2"], "layers": LAYERS,
                "X": torch.stack([holds[k_] for k_ in holds]),
                "meta": {k_: {"clue_id": k_.split("|")[0], "carrier": k_.split("|")[1],
                              "src": next(r["src"] for r in rows if r["clue_id"] == k_.split("|")[0]),
                              "tgt": next(r["tgt"] for r in rows if r["clue_id"] == k_.split("|")[0])} for k_ in holds}},
               out / "hold_captures.pt")
    with open(out / "i_items.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    gates.update(n_prompts=len(rows), n_gated=sum(r["gated"] for r in rows),
                 n_gated_discriminating=sum(r["gated"] and r["discriminating"] for r in rows),
                 n_tgt_checks=len(gates["tgt_checks"]), n_tgt_checks_pass=sum(c["pass"] for c in gates["tgt_checks"]),
                 hold_identity_all_bitwise=all(v["max_abs_diff"] == 0.0 for v in gates["hold_identity"].values()),
                 wall_s=round(time.time() - t0, 1))
    gates["pass"] = not (gates["prefix_mismatch"] or gates["G0_fail"] or gates["write_mask_violations"])   # hold identity: recorded
    gp = ROOT / "results/012_reader_concordance/analysis" / ("stageI_gates.json" if a.tag == "full" else f"stageI_gates_{a.tag}.json")
    gp.write_text(json.dumps(gates, indent=1, ensure_ascii=False))
    print(json.dumps({k_: v for k_, v in gates.items() if k_ not in ("tgt_checks", "hold_identity")}), flush=True)


if __name__ == "__main__":
    main()
