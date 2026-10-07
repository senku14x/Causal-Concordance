"""013 amendment A2 (spec §12; EXPLORATORY, post hoc): factual rewrite F against paraphrase control P.

  python rewrite.py judge   --subset smoke|all   # A writes F, then P (edit size matched to F); B labels F and P
  python rewrite.py handset --subset smoke|all   # hand-check file: O / F / P side by side
  python rewrite.py score   --subset smoke|all   # reconstructor on O, F, P (GPU): FVE and J share
  python rewrite.py analyze --subset smoke|all   # ΔFVE = F − P, ΔJ = F − P, gates
"""
import argparse
import difflib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hallucination as H  # noqa: E402

OUT, RES = H.OUT / "rewrite", H.RES

PROMPT_F = ("You edit an automatically generated explanation of a language model's internal state so that it contains only "
            "factual information.\n\n{inputs}\n\n" + H.DEFS + "\n\n"
            "Rewrite the EXPLANATION MINIMALLY so that every claim in it is true given the inputs:\n"
            "- correct misquotes of the prompt to the prompt's exact wording;\n"
            "- remove invented context (text, headers, instructions, quiz names, options or earlier questions that are not in the input) "
            "and unsupported claims;\n"
            "- correct wrong entities or answers so they agree with the model's continuation;\n"
            "- keep every true claim with its wording unchanged, keep the overall structure, and add nothing new beyond what a correction needs.\n\n"
            "Return JSON only: {{\"rewrite\": \"...\"}}.")

PROMPT_P = ("You paraphrase an automatically generated explanation of a language model's internal state.\n\n"
            "EXPLANATION:\n<<<{expl}>>>\n\n"
            "Rewrite it MINIMALLY by rewording phrases, changing about {n} words in total. Keep the meaning of EVERY claim exactly, "
            "including claims that may be false or odd: do not correct, remove or add any claim, quote or detail. Keep the structure.\n\n"
            "Return JSON only: {{\"rewrite\": \"...\"}}.")


def words_changed(a, b):
    wa, wb = a.split(), b.split()
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=wa, b=wb, autojunk=False).get_opcodes()
               if tag != "equal")


def label(r, text, cache):
    sents = H.sentences(text)
    rec, hit = H.call(H.MODEL_B, H.PROMPT_B.format(inputs=H.judge_inputs({**r, "explanation": text}),
                                                   sentences="\n".join(f"{i + 1}. {s}" for i, s in enumerate(sents))), cache, "BL")
    b = H.parse_json(rec["response"]["choices"][0]["message"]["content"] or "")
    labs = (b or {}).get("labels")
    ok = isinstance(labs, list) and len(labs) == len(sents)
    cost = 0.0 if hit else float((rec["response"].get("usage") or {}).get("cost") or 0.0)
    return (labs if ok else None), cost


def phase_judge(which):
    rows = H.subset(H.explanations(), which)
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    def get(model, prompt, tag):
        rec, hit = H.call(model, prompt, cache, tag)
        j = H.parse_json(rec["response"]["choices"][0]["message"]["content"] or "")
        if not (j and isinstance(j.get("rewrite"), str) and j["rewrite"].strip()):
            rec, hit = H.call(model, prompt + "\n\nReturn valid JSON only, with a non-empty \"rewrite\".", cache, tag)
            j = H.parse_json(rec["response"]["choices"][0]["message"]["content"] or "")
        cost = 0.0 if hit else float((rec["response"].get("usage") or {}).get("cost") or 0.0)
        txt = j.get("rewrite").strip() if j and isinstance(j.get("rewrite"), str) else None
        return txt, cost

    def one(r):
        spent = 0.0
        F, c = get(H.MODEL_A, PROMPT_F.format(inputs=H.judge_inputs(r)), "F")
        spent += c
        nF = words_changed(r["explanation"], F) if F else None
        P, c = get(H.MODEL_A, PROMPT_P.format(expl=r["explanation"], n=max(nF or 1, 1)), "P") if F else (None, 0.0)
        spent += c
        labs = {}
        for name, txt in (("O", r["explanation"]), ("F", F), ("P", P)):
            if txt:
                labs[name], c = label(r, txt, cache)
                spent += c
        return {**{k: r[k] for k in ("eid", "item", "state", "sample")}, "F": F, "P": P, "words_changed_F": nF,
                "words_changed_P": words_changed(r["explanation"], P) if P else None, "labels": labs}, spent

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(one, rows))
    res, spent = [o[0] for o in outs], sum(o[1] for o in outs)
    with open(OUT / f"rewrites_{which}.jsonl", "w") as f:
        for x in res:
            f.write(json.dumps(x) + "\n")
    meta = {"which": which, "n": len(res), "new_spend_usd": spent, "models": {"A": H.MODEL_A, "B": H.MODEL_B},
            "prompt_F": PROMPT_F, "prompt_P": PROMPT_P, "ok_F": sum(bool(x["F"]) for x in res), "ok_P": sum(bool(x["P"]) for x in res)}
    (OUT / f"rewrite_meta_{which}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if not k.startswith("prompt")}), flush=True)


def phase_handset(which):
    rows = {r["eid"]: r for r in H.explanations()}
    R = [json.loads(l) for l in open(OUT / f"rewrites_{which}.jsonl")]
    lines = []
    for x in R:
        r = rows[x["eid"]]
        lines.append(f"### {x['eid']} (src {r['src']} -> tgt {r['tgt']})\nPROMPT: {r['prompt']}\nCONTINUATION: {r['continuation']!r}\n"
                     f"\n--- O ---\n{r['explanation']}\n\n--- F (words changed {x['words_changed_F']}) ---\n{x['F']}\n\n"
                     f"--- P (words changed {x['words_changed_P']}) ---\n{x['P']}\n")
    (RES / "qualitative").mkdir(parents=True, exist_ok=True)
    (RES / f"qualitative/rewrite_handcheck_{which}.md").write_text("\n".join(lines))
    print("wrote", RES / f"qualitative/rewrite_handcheck_{which}.md")


def phase_score(which):
    import yaml
    from transformers import AutoTokenizer
    from nla.models import NLACriticModel
    from nla.schema import normalize_activation, compute_predict_mean_baselines
    from nla.utils.critic import critic_predict
    import readers as RD
    root = Path(RD.nla_rl_paths()[0])
    template = yaml.safe_load(open(root / "nla_meta.yaml"))["prompt_templates"]["critic"]
    scale = math.sqrt(5120)
    tok = AutoTokenizer.from_pretrained(root / "av_base")
    gold = H.gold_vectors()
    G = torch.stack(list(gold.values()))
    _, B = compute_predict_mean_baselines(G, scale)
    mu = normalize_activation(G, scale).mean(0)
    QJ = torch.linalg.qr(torch.load(H.ROOT / "outputs/007_jsubspace/bases/J_raw_L42.pt", weights_only=False)["Q"][:, :1024].double())[0].float()
    share = lambda v: float(((v - mu) @ QJ).pow(2).sum() / (v - mu).pow(2).sum())
    critic = NLACriticModel.from_pretrained(str(root / "ar_reconstructor"), dtype=torch.bfloat16, device_map="cuda").eval()

    @torch.no_grad()
    def recon(text):
        ids = tok.encode(template.format(explanation=text), add_special_tokens=False)[:1024]
        return normalize_activation(critic_predict(critic, torch.tensor([ids], device="cuda"), None, scale)[0].float().cpu()[None], scale)[0]

    rows = {r["eid"]: r for r in H.explanations()}
    R = [json.loads(l) for l in open(OUT / f"rewrites_{which}.jsonl")]
    out = []
    for x in R:
        r = rows[x["eid"]]
        g = normalize_activation(gold[(r["item"], r["state"])][None], scale)[0]
        rec = {**{k: x[k] for k in ("eid", "item", "state", "sample")}, "mse": {}, "share_J": {"gold": share(g)}}
        for name, txt in (("O", r["explanation"]), ("F", x["F"]), ("P", x["P"])):
            if txt:
                p = recon(txt)
                rec["mse"][name] = float(((p - g) ** 2).mean())
                rec["share_J"][name] = share(p)
        out.append(rec)
    with open(OUT / f"scored_{which}.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    (OUT / f"score_meta_{which}.json").write_text(json.dumps({"baseline_B": B, "n": len(out)}, indent=1))
    print("[score] done", len(out), flush=True)


def phase_analyze(which):
    B = json.loads((OUT / f"score_meta_{which}.json").read_text())["baseline_B"]
    S = [json.loads(l) for l in open(OUT / f"scored_{which}.jsonl")]
    R = {json.loads(l)["eid"]: json.loads(l) for l in open(OUT / f"rewrites_{which}.jsonl")}
    ok = [s for s in S if all(k in s["mse"] for k in ("O", "F", "P"))]
    fve = lambda s, k: 100 * (1 - s["mse"][k] / B)

    def pair(rows, f):
        d = [f(s) for s in rows]
        return {"n": len(d), "mean": float(np.mean(d)), "ci95": H.boot(d, [s["item"] for s in rows]), "median": float(np.median(d))} if d else None

    res = {"label": "EXPLORATORY, POST HOC (013 amendment A2, spec §12)", "which": which, "n": len(S), "n_ok": len(ok), "baseline_B": B,
           "fve_mean": {k: float(np.mean([fve(s, k) for s in ok])) for k in ("O", "F", "P")},
           "share_J_mean": {k: float(np.mean([s["share_J"][k] for s in ok])) for k in ("gold", "O", "F", "P")},
           "dFVE_F_minus_P": pair(ok, lambda s: fve(s, "F") - fve(s, "P")),
           "dJ_F_minus_P": pair(ok, lambda s: s["share_J"]["F"] - s["share_J"]["P"]),
           "dFVE_F_minus_O": pair(ok, lambda s: fve(s, "F") - fve(s, "O")), "dFVE_P_minus_O": pair(ok, lambda s: fve(s, "P") - fve(s, "O")),
           "dJ_F_minus_O": pair(ok, lambda s: s["share_J"]["F"] - s["share_J"]["O"]), "dJ_P_minus_O": pair(ok, lambda s: s["share_J"]["P"] - s["share_J"]["O"]),
           "per_state": {st: {"dFVE_F_minus_P": pair([s for s in ok if s["state"] == st], lambda s: fve(s, "F") - fve(s, "P")),
                              "dJ_F_minus_P": pair([s for s in ok if s["state"] == st], lambda s: s["share_J"]["F"] - s["share_J"]["P"])}
                         for st in H.STATES}}
    d = res["dFVE_F_minus_P"]
    if d:
        res["reading_dFVE"] = ("removing hallucinations improves reconstruction" if d["mean"] >= 2 and d["ci95"][0] > 0 else
                               "the reconstructor relies on hallucinated content" if d["mean"] <= -2 and d["ci95"][1] < 0 else
                               "no difference beyond ±2 points" if d["ci95"][0] >= -2 and d["ci95"][1] <= 2 else "unresolved")
    j = res["dJ_F_minus_P"]
    if j:
        res["reading_dJ"] = "more J" if j["ci95"][0] > 0 else "less J" if j["ci95"][1] < 0 else "no detected change"
    contra = lambda k: [np.mean([l == "contradicted" for l in R[s["eid"]]["labels"][k]]) for s in ok if R[s["eid"]]["labels"].get(k)]
    rates = {k: float(np.mean(contra(k))) if contra(k) else None for k in ("O", "F", "P")}
    res["B_contradicted_sentence_rate"] = rates
    res["gate_F_rate_le_0.10"] = rates["F"] is not None and rates["F"] <= 0.10
    res["gate_P_rate_ge_0.7_O"] = rates["P"] is not None and rates["O"] is not None and rates["P"] >= 0.7 * rates["O"]
    res["words_changed_mean"] = {"F": float(np.mean([R[s["eid"]]["words_changed_F"] for s in ok])),
                                 "P": float(np.mean([R[s["eid"]]["words_changed_P"] for s in ok]))}
    (RES / "analysis").mkdir(parents=True, exist_ok=True)
    (RES / f"analysis/rewrite_{which}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["judge", "handset", "score", "analyze"])
    ap.add_argument("--subset", choices=["smoke", "all"], default="smoke")
    a = ap.parse_args()
    {"judge": phase_judge, "handset": phase_handset, "score": phase_score, "analyze": phase_analyze}[a.phase](a.subset)
