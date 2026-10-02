"""Block c: one readout per reader on one brew emission cell. Engineering smoke test, not an experiment: no reader
output here enters any label, and nothing is scored against causal labels.

Cell: brew-s2-r10d2-0000 (start blue -> z green -> answer black), emit_colon (position 164, the last prompt token).
Readers: logit lens, J_NP / J_CB / R_CB (standard and cosine readouts) at L40/42/44; Oracle Lens on-contract at L40;
NLA-L42 AV (+ one AR reconstruction per explanation). Sensitivity arms for the verbalizers: no activation, and an
activation from another cell (first stir token, position 131). NLA positive controls on its own reference set: the
stored layer-42 activation is re-extracted from its text with our capture, and AV -> AR FVE is measured on 8 rows.

Writes results/000_setup/analysis/reader_smoke.json.
"""
import gc
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import torch

import readers as RD
import registry as R
import wsb

ITEM, POS, OTHER = "brew-s2-r10d2-0000", 164, 131
LENS_LAYERS, ORACLE_LAYER, NLA_LAYER = [40, 42, 44], 40, 42
K, SEED, N_REF = 4, 20261002, 8


def gb():
    return round(torch.cuda.memory_allocated() / 2**30, 2)


def stage(res, name, fn):
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    try:
        out = fn()
        res[name] = {"status": "works", **(out or {})}
    except Exception as e:
        res[name] = {"status": "fails", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
    res[name]["seconds"] = round(time.time() - t0, 1)
    res[name]["gpu_alloc_gb_after"] = gb()
    res[name]["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print(f"== {name}: {res[name]['status']} ({res[name]['seconds']} s, peak {res[name]['gpu_peak_gb']} GB)"
          + (f" {res[name]['error']}" if res[name]["status"] == "fails" else ""), flush=True)


def main():
    res = {"cell": {"item": ITEM, "position": POS, "region": "emit_colon", "other_cell_position": OTHER},
           "manifest": R.manifest_block()}
    tok = R.make_tokenizer()
    t0 = time.time()
    model = R.load_model()
    res["model_load"] = {"seconds": round(time.time() - t0, 1), "gpu_alloc_gb": gb()}
    _, bank = wsb.load_bank("brew")
    it = next(x for x in bank if x["name"] == ITEM)
    r = wsb.render_item(tok, "brew", it)
    wsb.assert_brew_positions(r, it)
    assert r.regions[POS] == "emit_colon" and r.regions[OTHER] == "stir"
    res["cell"].update({"start": it["start"], "z": it["intermediates"], "answer": it["answer"],
                        "off_trajectory": it["off_trajectory"], "token": r.decoded[POS], "other_token": r.decoded[OTHER]})
    layers = sorted(set(LENS_LAYERS) | {ORACLE_LAYER, NLA_LAYER})
    acts = RD.capture_stock(model, r.ids, layers, [POS, OTHER])
    h = {l: acts[l][0] for l in layers}
    h_other = {l: acts[l][1] for l in layers}
    res["cell"]["norms"] = {l: float(h[l].norm()) for l in layers}
    ref, df = RD.nla_reference_activations()

    # -------------------------------------------------------------- NLA reference re-extraction (base model)
    def reextract():
        rows = sorted(range(len(df)), key=lambda i: int(df.n_raw_tokens[i]))[:N_REF]
        out = []
        for i in rows:
            ids = tok(df.detokenized_text_truncated[i], add_special_tokens=False)["input_ids"]
            a = RD.capture_stock(model, ids, [NLA_LAYER], [len(ids) - 1])[NLA_LAYER][0]
            g = ref[i]
            out.append({"row": i, "n_tokens": len(ids), "cos": float(torch.nn.functional.cosine_similarity(a, g, 0)),
                        "rel_norm_diff": float((a - g).norm() / g.norm())})
        return {"rows": out, "min_cos": min(x["cos"] for x in out), "ref_rows": rows}
    stage(res, "nla_reference_reextraction", reextract)
    ref_rows = res["nla_reference_reextraction"].get("ref_rows", list(range(N_REF)))

    # --------------------------------------------------------------------------------------- vector lenses
    def lenses():
        vl = RD.VectorLenses(model, tok, LENS_LAYERS)
        out = {}
        for L in LENS_LAYERS:
            row = {"logit_lens": vl.logit_lens(h[L])}
            for nm in ("J_NP", "J_CB", "R_CB"):
                row[f"{nm}_standard"] = vl.j_standard(nm, h[L], L)
                row[f"{nm}_cosine"] = vl.j_cosine(nm, h[L], L)
            out[f"L{L}"] = {k: {"top10": v, "colours_named": RD.colours_named(" ".join(t for t, _ in v))}
                            for k, v in row.items()}
        del vl
        gc.collect()
        torch.cuda.empty_cache()
        return {"readouts": out}
    stage(res, "vector_lenses", lenses)

    # ---------------------------------------------------------------------------------------------- oracle
    holder = {"model": model}

    def oracle():
        ol = RD.OracleLens(holder["model"], tok)
        ids, slot, ch = ol.carrier(ORACLE_LAYER)
        out = {"marker_char": ch, "slot": slot, "prompt_len": len(ids)}
        for arm, vec, inject in (("activation", h[ORACLE_LAYER], True), ("no_activation", None, False),
                                 ("other_cell", h_other[ORACLE_LAYER], True)):
            t = time.time()
            s = ol.read(vec, ORACLE_LAYER, k=K, seed=SEED, inject=inject)
            out[arm] = {"samples": s, "colours_named": [RD.colours_named(x) for x in s], "seconds": round(time.time() - t, 1)}
        holder["model"] = ol.unload()
        return out
    stage(res, "oracle_lens_L40", oracle)
    model = holder["model"]

    # -------------------------------------------------------------------------------------------- NLA AV
    expl = {}

    def nla_av():
        from transformers import AutoTokenizer
        _, p, _ = RD.nla_paths()
        tok_av = AutoTokenizer.from_pretrained(p["av_sft_lora"])
        av = RD.NLAVerbalizer(holder["model"], tok_av)
        meta = RD.nla_meta()
        base_ids = tok.apply_chat_template([{"role": "user", "content": meta["prompt_templates"]["actor"].format(
            injection_char=meta["tokens"]["injection_char"])}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        base_ids = list(base_ids["input_ids"] if hasattr(base_ids, "keys") else base_ids)
        out = {"prompt_len": len(av.ids), "marker_pos": av.marker, "av_prompt_ids_equal_base_tokenizer": base_ids == av.ids}
        for arm, vec in (("activation", h[NLA_LAYER]), ("no_activation", None), ("other_cell", h_other[NLA_LAYER])):
            t = time.time()
            n0 = av.injections
            s = av.read(vec, k=K, seed=SEED)
            ex = [RD.extract_explanation(x) for x in s]
            expl[arm] = ex
            out[arm] = {"raw": s, "explanations": ex, "colours_named": [RD.colours_named(x or "") for x in ex],
                        "n_parsed": sum(x is not None for x in ex), "injections": av.injections - n0,
                        "seconds": round(time.time() - t, 1)}
        t = time.time()
        expl["reference"] = []
        for i in ref_rows:
            s = av.read(ref[i], k=1, seed=SEED + i)
            expl["reference"].append(RD.extract_explanation(s[0]))
        out["reference"] = {"rows": ref_rows, "explanations": expl["reference"], "seconds": round(time.time() - t, 1)}
        av.close()
        holder["model"] = None
        del av
        return out
    stage(res, "nla_L42_av", nla_av)
    model = holder["model"] = None
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------------------------- NLA AR
    def nla_ar():
        ar = RD.NLAReconstructor()
        out = {"n_layers": ar.n_layers}

        def score(ex_list, gold):
            rows = []
            for ex in ex_list:
                if ex is None:
                    rows.append(None)
                    continue
                f, mse, den = ar.fve(ar.predict(tok, ex), gold, ref)
                rows.append(round(f, 4))
            return rows
        for arm in ("activation", "no_activation", "other_cell"):
            out[f"fve_{arm}_explanations_vs_cell"] = score(expl.get(arm, []), h[NLA_LAYER])
        out["fve_other_cell_explanations_vs_other_cell"] = score(expl.get("other_cell", []), h_other[NLA_LAYER])
        out["fve_reference"] = [score([e], ref[i])[0] for e, i in zip(expl.get("reference", []), ref_rows)]
        _, _, den = ar.fve(ref[0], ref[0], ref)
        out["fve_denominator_reference_var"] = den
        return out
    stage(res, "nla_L42_ar", nla_ar)

    od = R.out_dir("results", "000_setup", "analysis")
    (od / "reader_smoke.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "manifest"}, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()
