"""012 Stage H (design §15; EXPLORATORY; model side only, no readers): 008's arm A on brew dev, every prompt position.

Material: 008's 82 gated dev pairs via the same loaders 008's `material()` uses (common/brew_material: brew_pairs +
gated_dev_brew, 001 gate record), member renders (chat template + "Answer:" prefill), 008's spellings (margin.spellings
"brew") and 008's recorded clean gap G (outputs/008_trackb/b1/pairs/<pid>.jsonl, row "clean"; our engine's G is also
computed and compared). Edit: Jan's SwapEdit (raw coordinates, unit J_NP rows from a token bank built with Jan's
build-token for the 12 colours, his surface-form rule), every prompt position, prefill only, alpha 1, member prompt.
Arms Az (z -> z'), Aa (a -> a'; the answer-atom positive control) x bands L24-47 / L40-48, each with a
source-preserving random-target control (Jan's random_directions, seed root ["012H_brew_dev", pair_id, Az|Aa,
"rand_swap"]). Scores: float32 m = LSE log P(a') - LSE log P(a) over every spelling and token (008's margin
convention: final-norm output in float32 x float32 W_U, teacher forced), S = dm / G_008, greedy flip to a' (Jan's
budget rule on the spellings, his strict matcher), ||dh|| per band layer at the ':' cell, G0 (alpha 0 bitwise).

  PYTHONPATH=/workspace/sources/ext/phrase-lens-src/src python stageH.py [--limit 4] [--tag smoke]
"""
import argparse
import json
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
from phrase_lens.interventions import BandEditor, SwapEdit, random_directions
from phrase_lens.lenses import DirectionBank
from phrase_lens.metrics import strict_match
from phrase_lens.modeling import ModelRunner

from stageB_capture import MODEL, REV, ROOT, Logged, lse

sys.path.insert(0, str(ROOT / "scripts/common"))
import brew_material as BM                          # noqa: E402  (label module; what 008's material() calls)
import margin as MG                                 # noqa: E402
import registry as R                                # noqa: E402
import stats as ST                                  # noqa: E402

OUT = ROOT / "outputs/012_reader_concordance/stageH"
BANDS = {"L24-47": list(range(24, 48)), "L40-48": list(range(40, 49))}
ARMS = [f"{k}_{b}" for k in ("Az", "Aa") for b in BANDS] + [f"rand_{k}_{b}" for k in ("Az", "Aa") for b in BANDS]
POWER = 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    out = OUT / ("" if a.tag == "full" else a.tag)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok8 = R.make_tokenizer()
    entries = BM.brew_pairs(tok8)
    gated = BM.gated_dev_brew(entries)
    assert len(gated) == 82, len(gated)
    gated = gated[:a.limit] if a.limit else gated
    runner = ModelRunner.from_pretrained(MODEL, device="cuda", dtype="bfloat16", revision=REV, local_files_only=True)
    tok, blocks, norm = runner.tokenizer, runner.blocks, runner.final_norm
    W32 = runner.model.get_output_embeddings().weight.detach().float()
    bank = DirectionBank.load(OUT / "bank_brew_j")
    rows, gates = [], {"G0_fail": [], "clean_not_a": [], "G_diff_max": 0.0}

    def tf(ids, spell, edit):
        """Float32 full-sequence log P(spell | ids), edit on the prompt positions only (as continuation_scores)."""
        full, n, grab = list(ids) + list(spell), len(ids), {}
        hk = norm.register_forward_pre_hook(lambda _m, args: grab.__setitem__("x", args[0]))
        ctx = BandEditor(blocks, edit.band, edit, n) if edit is not None else nullcontext()
        try:
            with ctx, torch.no_grad():
                runner.model(input_ids=torch.tensor([full], device=runner.device), use_cache=False, logits_to_keep=1)
        finally:
            hk.remove()
        with torch.no_grad():
            y = norm(grab["x"][0, n - 1:len(full) - 1].float())
            return float(torch.log_softmax(y @ W32.T, -1)[torch.arange(len(spell)), torch.tensor(spell)].double().sum())

    for k, e in enumerate(gated):
        t1 = time.time()
        pr = e["pair"]
        pid, ids, ids_p = pr.pair_id, list(e["r_m"].ids), list(e["r_p"].ids)
        rec8 = json.loads(open(ROOT / f"outputs/008_trackb/b1/pairs/{pid}.jsonl").readline())
        assert rec8["name"] == "clean" and rec8["a"] == pr.m.a and rec8["a2"] == pr.p.a and rec8["z"] == pr.m.z
        G8, table = rec8["G"], rec8["table"]
        Sa, Sb = MG.spellings(tok8, "brew", pr.m.a), MG.spellings(tok8, "brew", pr.p.a)
        budget = min(max(len(s) for s in Sa + Sb) + 2, 8)
        colon = [p for p, g in e["r_m"].regions.items() if g == "emit_colon"][0]
        n = len(ids)

        def m_of(x_ids, edit):
            return lse([tf(x_ids, s, edit) for s in Sb]) - lse([tf(x_ids, s, edit) for s in Sa])

        def greedy(edit):
            ctx = BandEditor(blocks, edit.band, edit, n) if edit is not None else nullcontext()
            with ctx:
                g = runner.generate(ids, edit=None, max_new_tokens=budget, return_token_ids=True)
            return g, tok.decode(g, skip_special_tokens=True)

        m_clean = m_of(ids, None)
        G_ours = m_of(ids_p, None) - m_clean
        gates["G_diff_max"] = max(gates["G_diff_max"], abs(G_ours - G8))
        g_clean, txt_clean = greedy(None)
        if not strict_match(pr.m.a, txt_clean):
            gates["clean_not_a"].append([pid, txt_clean])
        res = {}
        for b, layers in BANDS.items():
            c = runner.capture_ids(ids, positions=list(range(n)), layers=layers)
            clean_acts = {L: c[i] for i, L in enumerate(layers)}
            for fam, (src, tgt) in (("Az", (pr.m.z, pr.p.z)), ("Aa", (pr.m.a, pr.p.a))):
                s_dirs = bank.concept(src, layers)
                rnd = random_directions(s_dirs.shape[-1], ["012H_brew_dev", pid, fam, "rand_swap"], layers)
                for arm, t_dirs in ((f"{fam}_{b}", bank.concept(tgt, layers)), (f"rand_{fam}_{b}", rnd)):
                    ed = SwapEdit(s_dirs, t_dirs, clean_acts, layers, 1.0)
                    lg = Logged(ed, colon)
                    g, txt = greedy(lg)
                    m = m_of(ids, ed)
                    res[arm] = {"m": m, "dm": m - m_clean, "S": (m - m_clean) / G8, "text": txt,
                                "flip": strict_match(pr.p.a, txt) and not ed.flags["proxy_collision_all_layers"],
                                "dh_colon": {str(L): v["dh"] for L, v in lg.log.items()},
                                "dh_rel_colon_max": max(v["dh"] / v["h"] for v in lg.log.values()),
                                "cos_st": ed.flags["cos_st"], "n_collision_layers": ed.flags["n_collision_layers"]}
            if b == "L40-48":                                                # G0 on Jan's clamp, alpha 0
                z0 = SwapEdit(bank.concept(pr.m.z, layers), bank.concept(pr.p.z, layers), clean_acts, layers, 0.0)
                g0, _ = greedy(z0)
                d0 = abs(m_of(ids, z0) - m_clean)
                if g0 != g_clean or d0 != 0.0:
                    gates["G0_fail"].append([pid, g0 == g_clean, d0])
        rows.append({"pair_id": pid, "item": pr.item, "table": table, "G_008": G8, "G_ours": G_ours, "m_clean": m_clean,
                     "z": pr.m.z, "z2": pr.p.z, "a": pr.m.a, "a2": pr.p.a, "colon_pos": colon, "n_prompt": n,
                     "clean_text": txt_clean, "budget": budget, "arms": res, "sec": round(time.time() - t1, 1)})
        print(f"[{k + 1}/{len(gated)}] {pid} {time.time() - t1:.1f}s G8 {G8:.2f} G_ours {G_ours:.2f} "
              + " ".join(f"{x}:{res[x]['S']:+.2f}{'*' if res[x]['flip'] else ''}" for x in ARMS), flush=True)

    with open(out / "h_pairs.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summ = {"n_pairs": len(rows), "n_tables": len({r["table"] for r in rows}), "gates": gates,
            "gates_pass": not gates["G0_fail"], "arms": {}}
    for arm in ARMS:
        S = np.array([r["arms"][arm]["S"] for r in rows])
        T = np.array([r["table"] for r in rows])
        cb = ST.cluster_bootstrap(S, T, stat=np.median, B=10000, seed=20261004) if len(set(T)) > 1 else {}
        summ["arms"][arm] = {"median_S": float(np.median(S)), "median_ci": cb.get("ci"), "mean_S": float(S.mean()),
                             "flips": int(sum(r["arms"][arm]["flip"] for r in rows)), "n": len(rows),
                             "median_dh_rel_colon_max": float(np.median([r["arms"][arm]["dh_rel_colon_max"] for r in rows]))}
    ctl = summ["arms"]["Aa_L40-48"]["median_S"]
    summ["verdict"] = {"rule": "§15: answer-atom control Aa_L40-48 median S >= 0.15 -> positions explain 008's null "
                               "(then read Az); else family (Az not read)",
                       "Aa_L40-48_median_S": ctl,
                       "verdict": "positions explain 008's null" if ctl >= POWER else "family explains it",
                       "read_Az": ctl >= POWER}
    summ["wall_s"] = round(time.time() - t0, 1)
    sp = ROOT / "results/012_reader_concordance/analysis" / ("stageH_summary.json" if a.tag == "full" else f"stageH_summary_{a.tag}.json")
    sp.write_text(json.dumps(summ, indent=1))
    print(json.dumps({k_: v for k_, v in summ.items() if k_ != "arms"}), flush=True)


if __name__ == "__main__":
    main()
