"""012 Stage B add-on (model side; no reader code): the model's own float32 output at the read position per arm.
EXPLORATORY (lead's request after Stage D: the output-facing covariate for the readers' bridge d_r).

Per covered item x arm {clean, J_L24-35, rand_L24-35, J_L24-47, rand_L24-47}: the same edits as stageB_capture.py
(Jan's SwapEdit / random_directions, same seed root), prefill only through the identical generate path
(max_new_tokens = 1). Float32 logits at the read position = float32 final norm of the block-63 output x float32 W_U.
log-softmax of the src and tgt resolved single-token forms (Stage A's words_st_report.json; NA if unresolved) and of
y_a / y_b first tokens (tokforms012 rule; LSE over ids). d_model_bridge / d_model_answer = edited minus clean.
Parity (gate): every arm's block outputs at L40/42/44/52/56 at the read position are bitwise equal to Stage B's saved
captures, and the greedy first token equals Stage B's. Also: the alpha = 0 J edit gives bitwise-equal clean logits.

  PYTHONPATH=/workspace/sources/ext/phrase-lens-src/src python stageB_model_output.py [--limit 4] [--tag smoke]
"""
import argparse
import json
import time

import torch
from phrase_lens.benchmarks import load_benchmark
from phrase_lens.interventions import BandEditor, SwapEdit, random_directions
from phrase_lens.lenses import DirectionBank
from phrase_lens.modeling import ModelRunner

from stageB_capture import ARM_ID, ARMS, BANDS, BENCH, CAP_LAYERS, MODEL, REV, ROOT, SA, PrefillCapture
from tokforms012 import first_token_sets, lse_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    out = ROOT / "outputs/012_reader_concordance/stageB" / ("" if a.tag == "full" else a.tag)
    t0 = time.time()
    runner = ModelRunner.from_pretrained(MODEL, device="cuda", dtype="bfloat16", revision=REV, local_files_only=True)
    tok, blocks, norm = runner.tokenizer, runner.blocks, runner.final_norm
    W32 = runner.model.get_output_embeddings().weight.detach().float()
    bank = DirectionBank.load(SA / "bank_j")
    forms = json.loads((SA / "words_st_report.json").read_text())["resolved_form"]
    fid = {w: tok.encode(f, add_special_tokens=False) for w, f in forms.items()}
    assert all(len(v) == 1 for v in fid.values())
    cap = torch.load(out / "captures.pt")
    rowsB = {r["name"]: r for r in map(json.loads, open(out / "items.jsonl")) if r["status"] == "evaluated"}
    items = [it for it in load_benchmark(BENCH)["items"] if it["name"] in rowsB][:a.limit]

    def prefill(ids, edit):
        n, grab = len(ids), {}

        def pre(_m, args):
            if "x" not in grab and args[0].shape[1] == n:
                grab["x"] = args[0][0, n - 1].detach()
        hk = norm.register_forward_pre_hook(pre)
        ctx = BandEditor(blocks, edit.band, edit, n) if edit is not None else torch.no_grad()
        try:
            with ctx, PrefillCapture(blocks, CAP_LAYERS, n, n - 1) as pc:
                g = runner.generate(ids, edit=None, max_new_tokens=1, return_token_ids=True)
        finally:
            hk.remove()
        with torch.no_grad():
            y = norm(grab["x"].float())
            assert y.dtype == torch.float32
            lp = torch.log_softmax(y @ W32.T, -1).cpu()
        return lp, torch.stack([pc.acts[L] for L in CAP_LAYERS]), g[0]

    rows, worst = [], {"capture_max_abs": 0.0, "first_token_mismatch": [], "alpha0_logit_max_abs": 0.0}
    for it in items:
        rb = rowsB[it["name"]]
        ids = runner.encode(it["prompt"].rstrip())
        ci = cap["items"].index(it["name"])
        ya, yb = [it["y_a"], *it.get("y_a_alts", [])], [it["y_b"], *it.get("y_b_alts", [])]
        ida, idb, shared = first_token_sets(tok, ya, yb)
        edits = {"clean": None}
        alpha0 = None
        for b, layers in BANDS.items():
            c = runner.capture_ids(ids, positions=list(range(len(ids))), layers=layers)
            clean_acts = {L: c[i] for i, L in enumerate(layers)}
            s_dirs, t_dirs = bank.concept(it["src"], layers), bank.concept(it["tgt"], layers)
            edits[f"J_{b}"] = SwapEdit(s_dirs, t_dirs, clean_acts, layers, 1.0)
            edits[f"rand_{b}"] = SwapEdit(s_dirs, random_directions(s_dirs.shape[-1], [BENCH, it["name"], ARM_ID, "rand_swap"],
                                                                    layers), clean_acts, layers, 1.0)
            if b == "L24-47":
                alpha0 = SwapEdit(s_dirs, t_dirs, clean_acts, layers, 0.0)
        sc = {}
        for arm, e in edits.items():
            lp, X, g0 = prefill(ids, e)
            worst["capture_max_abs"] = max(worst["capture_max_abs"], float((X - cap["X"][ci, ARMS.index(arm)]).abs().max()))
            if g0 != rb["arms"][arm]["ids"][0]:
                worst["first_token_mismatch"].append([it["name"], arm])
            sc[arm] = {"lp_src": float(lp[fid[it["src"]][0]]) if it["src"] in fid else None,
                       "lp_tgt": float(lp[fid[it["tgt"]][0]]) if it["tgt"] in fid else None,
                       "lp_ya": lse_ids(lp, ida), "lp_yb": lse_ids(lp, idb)}
            if arm == "clean":
                lp_clean = lp
        lp0, _, _ = prefill(ids, alpha0)
        worst["alpha0_logit_max_abs"] = max(worst["alpha0_logit_max_abs"], float((lp0 - lp_clean).abs().max()))
        d = lambda x, k1, k2: None if None in (x[k1], x[k2], sc["clean"][k1], sc["clean"][k2]) else \
            (x[k2] - x[k1]) - (sc["clean"][k2] - sc["clean"][k1])
        for arm in ARMS:
            sc[arm]["d_model_bridge"] = d(sc[arm], "lp_src", "lp_tgt")
            sc[arm]["d_model_answer"] = d(sc[arm], "lp_ya", "lp_yb")
        rows.append({"name": it["name"], "src": it["src"], "tgt": it["tgt"], "y_a": ya, "y_b": yb,
                     "answer_first_token_ids": {"y_a": ida, "y_b": idb, "shared_dropped": shared},
                     "answer_first_tokens": {"y_a": [tok.decode([i]) for i in ida], "y_b": [tok.decode([i]) for i in idb]},
                     "read_pos": len(ids) - 1, "arms": sc})
    with open(out / "model_output.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    gate = {"n_items": len(rows), **worst,
            "rule": "captures bitwise equal to Stage B's, greedy first token equal, alpha0 logits bitwise equal to clean",
            "pass": worst["capture_max_abs"] == 0.0 and not worst["first_token_mismatch"] and worst["alpha0_logit_max_abs"] == 0.0,
            "n_answer_NA": sum(r["arms"]["clean"]["lp_ya"] is None or r["arms"]["clean"]["lp_yb"] is None for r in rows),
            "wall_s": round(time.time() - t0, 1)}
    gp = ROOT / "results/012_reader_concordance/analysis" / (
        "model_output_parity.json" if a.tag == "full" else f"model_output_parity_{a.tag}.json")
    gp.write_text(json.dumps(gate, indent=1))
    print(json.dumps(gate), flush=True)


if __name__ == "__main__":
    main()
