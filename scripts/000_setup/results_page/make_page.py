"""Build the block-000 results page (published as a private claude.ai artifact) from the committed analysis JSON only.

    python scripts/000_setup/results_page/make_page.py  ->  outputs/000_setup/results_page/causal_concordance.html
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import partners as P
import registry as R
import wsb

A = R.PROJECT / "results" / "000_setup" / "analysis"


def main():
    rs = json.loads((A / "reader_smoke.json").read_text())
    rc = json.loads((A / "regime_check.json").read_text())
    tm = json.loads((A / "timing_smoke.json").read_text())
    ec = json.loads((A / "env_check.json").read_text())
    tok = R.make_tokenizer()
    _, bank = wsb.load_bank("brew")
    it = next(x for x in bank if x["name"] == "brew-s2-r10d2-0000")
    r = wsb.render_item(tok, "brew", it)
    pair = {p.p.inp: p for p in P.brew_candidates(it)}["pink"]
    pc = rs["vector_lenses_positive_control"]
    data = {
        "prompt_tokens": [[i, t, r.regions.get(i, "")] for i, t in enumerate(r.decoded)],
        "item": {"name": it["name"], "start": it["start"], "z": it["intermediates"][0], "answer": it["answer"],
                 "partner_start": pair.p.inp, "partner_z": pair.p.z, "partner_answer": pair.p.a,
                 "off": it["off_trajectory"], "candidates": it["options_adjacent"][0]},
        "env": {"greedy": ec["greedy"].strip().split()[0], "top5": ec["top5_stock_bf16"], "load_s": round(ec["load_s"], 1),
                "gpu": ec["gpu"]},
        "lenses": {L: {k: [t for t, _ in v["top10"]] for k, v in rows.items()}
                   for L, rows in rs["vector_lenses"]["readouts"].items()},
        "late": {**{f"logit_lens {k}": [t for t, _ in v["top10"]] for k, v in pc["logit_lens"].items()},
                 **{k.replace("_", " ", 1): [t for t, _ in v["top10"]] for k, v in pc["J"].items()
                    if "standard" in k and "J_NP" in k}},
        "oracle": {a: rs["oracle_lens_L40"][a]["samples"] for a in ("activation", "no_activation", "other_cell")},
        "nla": {a: rs["nla_L42_av"][a]["explanations"] for a in ("activation", "no_activation", "other_cell")},
        "fve": {"activation": rs["nla_L42_ar"]["fve_activation_explanations_vs_cell"],
                "no_activation": rs["nla_L42_ar"]["fve_no_activation_explanations_vs_cell"],
                "other_cell": rs["nla_L42_ar"]["fve_other_cell_explanations_vs_cell"],
                "other_own": rs["nla_L42_ar"]["fve_other_cell_explanations_vs_other_cell"],
                "reference": rs["nla_L42_ar"]["fve_reference"],
                "reextract_min_cos": rs["nla_reference_reextraction"]["min_cos"]},
        "timing": {k: {"s": rs[k]["seconds"], "peak": rs[k]["gpu_peak_gb"]}
                   for k in ("vector_lenses", "oracle_lens_L40", "nla_L42_av", "nla_L42_ar")},
        "regime": {"pair": rc["pair_id"], "member": rc["member"], "partner": rc["partner"],
                   "edits": [{"name": e["name"], "rhut": rc["regimes"]["rhut_fp32res_from0"]["edits"][e["name"]]["dm"],
                              "sec": rc["regimes"]["sec_true_fp32_from40"]["edits"][e["name"]]["dm"],
                              "bf16": rc["regimes"]["stock_bf16"]["edits"][e["name"]]["dm"]} for e in rc["edits"]],
                   "G": {k: v["G"] for k, v in rc["regimes"].items()},
                   "noise": {k: v["noise_floor"]["max_abs_dm"] for k, v in rc["regimes"].items()},
                   "max_frac": {k: v["max_abs_diff_over_G"] for k, v in rc["comparison_vs_sec"].items()}},
        "gates": [{"pair": a["pair_id"], "pass": a["gate"]["pass"], "m": a["gate"]["m"]["n_correct"],
                   "p": a["gate"]["p"]["n_correct"],
                   "p_samples": [s.strip().split("\n")[0][:14] for s in a["gate"]["p"]["samples"]]}
                  for a in rc["gate_attempts"]]
                 + [{"pair": a["pair_id"], "pass": a["pass"], "m": None, "p": None, "p_samples": []}
                    for a in tm["chain_gate_attempts"]],
    }
    t = open(os.path.join(HERE, "template.html")).read()
    out = R.out_dir("outputs", "000_setup", "results_page") / "causal_concordance.html"
    out.write_text(t.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    print(out)


if __name__ == "__main__":
    main()
