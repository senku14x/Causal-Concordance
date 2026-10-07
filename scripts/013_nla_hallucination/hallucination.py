"""013 main pipeline (spec docs/design_specs/013_nla_hallucination.md; EXPLORATORY).

  python hallucination.py judge   --subset smoke|all   # §3: judges A (cleaner) and B (labeller), cached, cost-logged
  python hallucination.py handset --subset smoke|all   # writes the hand-check file (§7 G2) - read BEFORE any Δ
  python hallucination.py score   --subset smoke|all   # §4-5: versions + reconstructor (GPU), G-repro
  python hallucination.py analyze --subset smoke|all   # §5-7: Δ, rules, G1/G3, per state, J share

Sections referenced (§n) are the spec's.
"""
import argparse
import ast
import hashlib
import json
import math
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "common"))
EASYNLA = Path("/workspace/sources/repos/EasyNLA")
sys.path.insert(0, str(EASYNLA))
OUT = ROOT / "outputs/013_nla_hallucination"
RES = ROOT / "results/013_nla_hallucination"
STATES = ["clean", "J_L24-35", "rand_L24-35"]
MODEL_A, MODEL_B = "google/gemini-3.8-flash", "z-ai/glm-5.3-flash"
N_SEEDS, MIN_TOK = 5, 10
SPEND_CAP = 10.0

# --------------------------------------------------------------------------------------------- §2 material
def explanations():
    """The 522 explanations in a fixed order (item, state, sample)."""
    from nla.schema import extract_explanation
    items = {json.loads(l)["name"]: json.loads(l) for l in open(ROOT / "outputs/012_reader_concordance/stageB/items.jsonl")}
    rows = []
    for l in open(ROOT / "outputs/012_reader_concordance/stageC/readouts_nla.jsonl"):
        r = json.loads(l)
        if str(r["layer"]) != "42" or r.get("off_training") in (True, "True") or r["state"] not in STATES:
            continue
        s = ast.literal_eval(r["samples"]) if isinstance(r["samples"], str) else r["samples"]
        it = items[r["item"]]
        arms = it["arms"] if isinstance(it["arms"], dict) else json.loads(it["arms"])
        for k, raw in enumerate(s):
            e = extract_explanation(raw)
            assert e is not None, (r["item"], r["state"], k)
            rows.append({"eid": f"{r['item']}|{r['state']}|{k}", "item": r["item"], "state": r["state"], "sample": k,
                         "prompt": it["prompt"], "continuation": arms[r["state"]]["text"], "src": r["src"], "tgt": r["tgt"],
                         "explanation": e})
    rows.sort(key=lambda x: (x["item"], STATES.index(x["state"]), x["sample"]))
    assert len(rows) == 522, len(rows)
    return rows


def subset(rows, which):
    if which == "all":
        return rows
    rng = random.Random(13)
    pick = []
    for st, n in (("clean", 7), ("J_L24-35", 7), ("rand_L24-35", 6)):
        pool = [r for r in rows if r["state"] == st]
        pick += rng.sample(pool, n)
    return sorted(pick, key=lambda x: x["eid"])

# --------------------------------------------------------------------------------------------- §3 judges
DEFS = """Definitions (apply them strictly):
- CONTRADICTED: the claim is false given the inputs. (a) It asserts that the input contains text or structure that is not there (invented quotes, headers, answer options, earlier questions, instructions, misquotes of the prompt). (b) It names an entity or answer that conflicts with the model's own continuation. (c) It is a factually false statement about the world.
- UNSUPPORTED: the claim is not false, but it is neither stated in the inputs nor reasonably inferable from them (speculative context, guessed intent, guessed situation). A reasonable inference about genre or task (e.g. that a "Fact:" prompt asks for a factual completion) is NOT unsupported.
- Everything else is fine: true statements about the prompt, reasonable inferences, correct world knowledge, descriptions of what the next token should be that agree with the continuation."""

STATE_NOTE = {"clean": "The model ran normally.",
              "edit": ("The model's internal state was edited before it produced its continuation. The continuation is what the "
                       "model actually computed. For claims about which entity or answer the model is working with, treat the "
                       "continuation as the truth, not the prompt.")}


def judge_inputs(r):
    note = STATE_NOTE["clean" if r["state"] == "clean" else "edit"]
    return (f"PROMPT given to the language model (exact text):\n<<<{r['prompt']}>>>\n\n"
            f"The model's own continuation of that prompt:\n<<<{r['continuation']}>>>\n\n{note}\n\n"
            f"EXPLANATION (a description of the model's internal activation at the last prompt token, written by another system):\n"
            f"<<<{r['explanation']}>>>")


PROMPT_A = ("You check an automatically generated explanation of a language model's internal state for false claims.\n\n"
            "{inputs}\n\n" + DEFS + "\n\n"
            "Return JSON only: {{\"contradicted\": [...], \"unsupported\": [...]}}. Each list holds spans copied EXACTLY, "
            "character for character, from the EXPLANATION. Each span covers one whole claim (a sentence or a clause). Spans "
            "must not overlap. Do not flag fine claims. Empty lists are allowed.")

PROMPT_B = ("You label the sentences of an automatically generated explanation of a language model's internal state.\n\n"
            "{inputs}\n\n" + DEFS + "\n- FORMAT: the sentence makes no claim (empty, a heading, a fragment).\n\n"
            "The explanation is split into these numbered sentences:\n{sentences}\n\n"
            "Return JSON only: {{\"labels\": [\"consistent\" | \"contradicted\" | \"unsupported\" | \"format\", ...]}}, "
            "one label per numbered sentence, in order. Label a sentence \"contradicted\" if it contains any contradicted claim, "
            "else \"unsupported\" if it contains any unsupported claim.")


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def api_key():
    return re.search(r'OPENROUTER_API_KEY="?([^"\n]+)', (Path("/workspace/.env")).read_text()).group(1)


def call(model, prompt, cache_dir, tag):
    h = hashlib.sha256((model + "\n" + prompt).encode()).hexdigest()[:24]
    f = cache_dir / f"{tag}_{h}.json"
    if f.exists():
        return json.loads(f.read_text()), True
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "reasoning": {"effort": "low"}, "usage": {"include": True}}
    for attempt in range(4):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"})
            resp = json.load(urllib.request.urlopen(req, timeout=180))
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    rec = {"model": model, "request_hash": h, "response": resp}
    f.write_text(json.dumps(rec))
    return rec, False


def parse_json(txt):
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", txt, flags=re.S)
    return json.loads(m.group(0)) if m else None


def ws_regex(span):
    toks = span.split()
    return re.compile(r"\s+".join(re.escape(t) for t in toks)) if toks else None


def locate(spans, text):
    """Char ranges of verbatim spans (whitespace-flexible). Returns (ranges, n_bad)."""
    out, bad = [], 0
    for s in spans:
        rx = ws_regex(s) if isinstance(s, str) else None
        m = rx.search(text) if rx else None
        if not m:
            bad += 1
            continue
        out.append((m.start(), m.end()))
    out.sort()
    merged = []
    for a, b in out:
        if merged and a < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged, bad


def phase_judge(which):
    rows = subset(explanations(), which)
    cache = OUT / "judge/cache"
    cache.mkdir(parents=True, exist_ok=True)
    spent, results = 0.0, []
    for n, r in enumerate(rows):
        inp = judge_inputs(r)
        sents = sentences(r["explanation"])
        recA, hitA = call(MODEL_A, PROMPT_A.format(inputs=inp), cache, "A")
        a = parse_json(recA["response"]["choices"][0]["message"]["content"] or "")
        if a is None or not isinstance(a.get("contradicted"), list):     # one retry (G1)
            recA, hitA = call(MODEL_A, PROMPT_A.format(inputs=inp) + "\n\nReturn valid JSON only.", cache, "A")
            a = parse_json(recA["response"]["choices"][0]["message"]["content"] or "")
        recB, hitB = call(MODEL_B, PROMPT_B.format(inputs=inp, sentences="\n".join(f"{i + 1}. {s}" for i, s in enumerate(sents))), cache, "B")
        b = parse_json(recB["response"]["choices"][0]["message"]["content"] or "")
        for rec, hit in ((recA, hitA), (recB, hitB)):
            if not hit:
                spent += float((rec["response"].get("usage") or {}).get("cost") or 0.0)
        cr, bad_c = locate((a or {}).get("contradicted", []), r["explanation"]) if a else ([], 0)
        ur, bad_u = locate((a or {}).get("unsupported", []), r["explanation"]) if a else ([], 0)
        okA = a is not None and isinstance(a.get("contradicted"), list) and isinstance(a.get("unsupported", []), list) and bad_c + bad_u == 0
        labels = (b or {}).get("labels") if b else None
        okB = isinstance(labels, list) and len(labels) == len(sents)
        results.append({**{k: r[k] for k in ("eid", "item", "state", "sample")}, "A_raw": a, "A_ok": okA, "A_bad_spans": bad_c + bad_u,
                        "contradicted_ranges": cr, "unsupported_ranges": ur, "B_labels": labels, "B_ok": okB, "sentences": sents,
                        "usage": {"A": recA["response"].get("usage"), "B": recB["response"].get("usage")}})
        if (n + 1) % 10 == 0:
            print(f"[judge] {n + 1}/{len(rows)} new spend ${spent:.3f}", flush=True)
        if spent > SPEND_CAP:
            raise SystemExit(f"spend cap ${SPEND_CAP} passed")
    (OUT / "judge").mkdir(parents=True, exist_ok=True)
    with open(OUT / f"judge/judged_{which}.jsonl", "w") as f:
        for x in results:
            f.write(json.dumps(x) + "\n")
    meta = {"which": which, "n": len(results), "models": {"A": MODEL_A, "B": MODEL_B}, "new_spend_usd": spent,
            "prompt_A_template": PROMPT_A, "prompt_B_template": PROMPT_B, "defs": DEFS, "state_note": STATE_NOTE,
            "G1_A_ok_rate": sum(x["A_ok"] for x in results) / len(results), "B_ok_rate": sum(x["B_ok"] for x in results) / len(results)}
    (OUT / f"judge/judge_meta_{which}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if not k.startswith(("prompt", "defs", "state"))}), flush=True)


def phase_handset(which):
    rows = {r["eid"]: r for r in explanations()}
    J = [json.loads(l) for l in open(OUT / f"judge/judged_{which}.jsonl")]
    lines = []
    for x in J:
        r = rows[x["eid"]]
        e = r["explanation"]
        lines.append(f"### {x['eid']}  (src {r['src']} -> tgt {r['tgt']})\nPROMPT: {r['prompt']}\nCONTINUATION: {r['continuation']!r}\n"
                     f"EXPLANATION:\n{e}\n\nA contradicted:\n" + "\n".join(f"  C{i}: {e[a:b]!r}" for i, (a, b) in enumerate(x["contradicted_ranges"]))
                     + "\nA unsupported:\n" + "\n".join(f"  U{i}: {e[a:b]!r}" for i, (a, b) in enumerate(x["unsupported_ranges"]))
                     + f"\nB labels: {list(zip(range(1, 99), x['B_labels'] or []))}\n")
    (RES / "qualitative").mkdir(parents=True, exist_ok=True)
    (RES / f"qualitative/handcheck_{which}.md").write_text("\n".join(lines))
    print("wrote", RES / f"qualitative/handcheck_{which}.md")

# --------------------------------------------------------------------------------------------- §4 versions
def tidy(t):
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"\s+([,.;:])", r"\1", t)
    return t.strip()


def delete(text, ranges):
    out, last = [], 0
    for a, b in sorted(ranges):
        out.append(text[last:a])
        last = b
    out.append(text[last:])
    return tidy("".join(out))


def units(text):
    return [(m.start(), m.end()) for m in re.finditer(r"[^\n.!?;:,]+[.!?;:,]?", text) if m.group(0).strip()]


def random_match(text, pool, target, ntok, seed):
    rng = random.Random(seed)
    order = pool[:]
    rng.shuffle(order)
    chosen, tot = [], 0
    for u in order:
        t = ntok(text[u[0]:u[1]])
        if tot + t <= target:
            chosen.append(u)
            tot += t
    rest = [u for u in order if u not in chosen]
    if rest:
        best = min(rest, key=lambda u: abs(tot + ntok(text[u[0]:u[1]]) - target))
        if abs(tot + ntok(text[best[0]:best[1]]) - target) < abs(tot - target):
            chosen.append(best)
            tot += ntok(text[best[0]:best[1]])
    return chosen, tot


def overlaps(u, ranges):
    return any(u[0] < b and a < u[1] for a, b in ranges)


def build_versions(e, cr, ur, ntok, eid):
    v = {"O": e, "O_tidy": tidy(e), "C1": delete(e, cr), "C2": delete(e, cr + ur)}
    info = {"tok_O": ntok(e), "tok_C1": ntok(v["C1"]), "tok_C2": ntok(v["C2"])}
    us = units(e)
    seed0 = int(hashlib.sha256(eid.encode()).hexdigest()[:8], 16)
    for name, target_from in (("R1", "C1"), ("R2", "C2")):
        tgt = info["tok_O"] - info[f"tok_{target_from}"]
        v[name] = []
        for s in range(N_SEEDS):
            ch, tot = random_match(e, us, tgt, ntok, seed0 + 100 * s + (0 if name == "R1" else 7))
            v[name].append(delete(e, ch))
        info[f"target_{name}"] = tgt
    flagged = cr + ur
    pool = [u for u in us if not overlaps(u, flagged)]
    tgt = info["tok_O"] - info["tok_C1"]
    pool_tok = sum(ntok(e[a:b]) for a, b in pool)
    info["K1_computable"] = pool_tok >= tgt and tgt > 0
    v["K1"] = [delete(e, random_match(e, pool, tgt, ntok, seed0 + 100 * s + 3)[0]) for s in range(N_SEEDS)] if info["K1_computable"] else []
    info["nothing_to_clean"] = len(cr) == 0
    info["cleaned_away"] = info["tok_C1"] < MIN_TOK
    return v, info

# --------------------------------------------------------------------------------------------- §5 scoring
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
    ntok = lambda t: len(tok.encode(t, add_special_tokens=False))
    caps = torch.load(ROOT / "outputs/012_reader_concordance/stageB/captures.pt", map_location="cpu", weights_only=False)
    li = caps["layers"].index(42)
    gold = {(it, st): caps["X"][n, caps["arms"].index(st), li].float() for n, it in enumerate(caps["items"]) for st in STATES}
    G = torch.stack(list(gold.values()))
    _, B = compute_predict_mean_baselines(G, scale)
    Gn = normalize_activation(G, scale)
    mu = Gn.mean(0)
    Qb = torch.load(ROOT / "outputs/007_jsubspace/bases/J_raw_L42.pt", weights_only=False)["Q"][:, :1024].double()
    QJ = torch.linalg.qr(Qb)[0].float()
    share = lambda v: float(((v - mu) @ QJ).pow(2).sum() / (v - mu).pow(2).sum())
    critic = NLACriticModel.from_pretrained(str(root / "ar_reconstructor"), dtype=torch.bfloat16, device_map="cuda").eval()

    @torch.no_grad()
    def recon(text):
        ids = tok.encode(template.format(explanation=text), add_special_tokens=False)[:1024]
        p = critic_predict(critic, torch.tensor([ids], device="cuda"), None, scale)[0].float().cpu()
        return normalize_activation(p[None], scale)[0]

    rows = {r["eid"]: r for r in explanations()}
    J = [json.loads(l) for l in open(OUT / f"judge/judged_{which}.jsonl")]
    out, t0 = [], time.time()
    for n, x in enumerate(J):
        r = rows[x["eid"]]
        g = normalize_activation(gold[(r["item"], r["state"])][None], scale)[0]
        v, info = build_versions(r["explanation"], [tuple(a) for a in x["contradicted_ranges"]], [tuple(a) for a in x["unsupported_ranges"]], ntok, x["eid"])
        rec = {**{k: x[k] for k in ("eid", "item", "state", "sample")}, **info, "A_ok": x["A_ok"], "mse": {}, "share_J": {}, "texts": {}}
        for name, val in v.items():
            vals = val if isinstance(val, list) else [val]
            if not vals:
                continue
            ps = [recon(t) for t in vals]
            rec["mse"][name] = float(np.mean([float(((p - g) ** 2).mean()) for p in ps]))
            rec["share_J"][name] = float(np.mean([share(p) for p in ps]))
            rec["texts"][name] = vals
        rec["share_J"]["gold"] = share(g)
        out.append(rec)
        if (n + 1) % 10 == 0:
            print(f"[score] {n + 1}/{len(J)} ({time.time() - t0:.0f}s)", flush=True)
    # G-repro against the de-risk scores (clean, first sample)
    dr = [json.loads(l) for l in open(OUT / "derisk/multihop_ar_scores.jsonl")]
    order = sorted({r["item"] for r in rows.values()})
    ref = {(order[d["i"]], d["sample"]): d["mse"] for d in dr if "mse" in d}
    diffs = [abs(o["mse"]["O"] - ref[(o["item"], 0)]) for o in out if o["state"] == "clean" and o["sample"] == 0 and (o["item"], 0) in ref]
    meta = {"which": which, "baseline_B": B, "n": len(out), "G_repro_n": len(diffs), "G_repro_max_abs": max(diffs) if diffs else None,
            "G_repro_pass": bool(diffs) and max(diffs) <= 1e-6, "wall_s": time.time() - t0}
    with open(OUT / f"scored_{which}.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    (OUT / f"score_meta_{which}.json").write_text(json.dumps(meta, indent=1))
    print("[score] done", json.dumps({k: v for k, v in meta.items() if k.startswith(("G_", "n", "wall"))}), flush=True)

# --------------------------------------------------------------------------------------------- §6-7 analysis
def boot(vals, clus, f=np.mean, nb=2000, seed=0):
    cl = {}
    for v, c in zip(vals, clus):
        cl.setdefault(c, []).append(v)
    keys, rng, bs = sorted(cl), random.Random(seed), []
    for _ in range(nb):
        s = [v for k in (rng.choice(keys) for _ in keys) for v in cl[k]]
        bs.append(f(s))
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def verdict(m, ci):
    if m >= 2 and ci[0] > 0:
        return "false claims carry extra signal (H-signal)"
    if m <= -2 and ci[1] < 0:
        return "false claims carry less signal than average text (H-padding)"
    if ci[0] >= -2 and ci[1] <= 2:
        return "no difference beyond ±2 points"
    return "unresolved"


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else None


def phase_analyze(which):
    meta = json.loads((OUT / f"score_meta_{which}.json").read_text())
    B = meta["baseline_B"]
    S = [json.loads(l) for l in open(OUT / f"scored_{which}.jsonl")]
    J = {json.loads(l)["eid"]: json.loads(l) for l in open(OUT / f"judge/judged_{which}.jsonl")}
    fve = lambda s, k: 100 * (1 - s["mse"][k] / B)
    elig = [s for s in S if s["A_ok"] and not s["nothing_to_clean"] and not s["cleaned_away"]]
    res = {"label": "EXPLORATORY (013; spec §5-7)", "which": which, "n": len(S), "n_eligible": len(elig),
           "n_nothing_to_clean": sum(s["nothing_to_clean"] for s in S), "n_cleaned_away": sum(s["cleaned_away"] for s in S),
           "G_repro": {k: meta[k] for k in meta if k.startswith("G_repro")}, "baseline_B": B}
    def stat(rows, a, b):
        d = [fve(s, a) - fve(s, b) for s in rows if a in s["mse"] and b in s["mse"]]
        cl = [s["item"] for s in rows if a in s["mse"] and b in s["mse"]]
        if not d:
            return None
        ci = boot(d, cl)
        return {"n": len(d), "mean": float(np.mean(d)), "ci95": ci, "median": float(np.median(d))}
    res["primary_delta_R1_minus_C1"] = stat(elig, "R1", "C1")
    if res["primary_delta_R1_minus_C1"]:
        p = res["primary_delta_R1_minus_C1"]
        res["primary_verdict"] = verdict(p["mean"], p["ci95"])
    res["secondary_delta_R2_minus_C2"] = stat([s for s in elig if s["tok_C2"] >= MIN_TOK], "R2", "C2")
    res["K1_minus_C1"] = stat([s for s in elig if s["K1_computable"]], "K1", "C1")
    res["O_minus_C1"] = stat(elig, "O", "C1")
    res["O_minus_Otidy"] = stat(S, "O", "O_tidy")
    res["fve_mean"] = {k: float(np.mean([fve(s, k) for s in elig if k in s["mse"]])) for k in ("O", "O_tidy", "C1", "C2", "R1", "R2", "K1")
                       if any(k in s["mse"] for s in elig)}
    res["per_state"] = {st: {"delta_R1_minus_C1": stat([s for s in elig if s["state"] == st], "R1", "C1"),
                             "O_minus_C1": stat([s for s in elig if s["state"] == st], "O", "C1")} for st in STATES}
    res["flagged_token_share"] = {"contradicted_mean": float(np.mean([(s["tok_O"] - s["tok_C1"]) / s["tok_O"] for s in S])),
                                  "contradicted_or_unsupported_mean": float(np.mean([(s["tok_O"] - s["tok_C2"]) / s["tok_O"] for s in S]))}
    dj = lambda s, k: s["share_J"][k] - s["share_J"]["O"]
    jr = [dj(s, "C1") - dj(s, "R1") for s in elig]
    res["J_share"] = {"gold_mean": float(np.mean([s["share_J"]["gold"] for s in S])), "O_mean": float(np.mean([s["share_J"]["O"] for s in S])),
                      "dJ_C1_minus_dJ_R1_mean": float(np.mean(jr)) if jr else None, "ci95": boot(jr, [s["item"] for s in elig]) if jr else None}
    # G1 / G3
    res["G1_A_ok_rate"] = sum(J[s["eid"]]["A_ok"] for s in S) / len(S)
    # G3: per sentence, "contains an A-contradicted span" vs B's "contradicted"
    a_flag, b_flag = [], []
    allrows = {r["eid"]: r for r in explanations()}
    for s in S:
        x = J[s["eid"]]
        if not (x["A_ok"] and x["B_ok"]):
            continue
        e = allrows[s["eid"]]["explanation"]
        pos = 0
        for sent, lab in zip(x["sentences"], x["B_labels"]):
            rx = ws_regex(sent)
            m = rx.search(e, pos) if rx else None
            if not m:
                continue
            pos = m.end()
            a_flag.append(overlaps((m.start(), m.end()), [tuple(r) for r in x["contradicted_ranges"]]))
            b_flag.append(lab == "contradicted")
    res["G3_kappa_sentence_contradicted"] = kappa(a_flag, b_flag) if a_flag else None
    res["G3_n_sentences"] = len(a_flag)
    res["G3_rates"] = {"A": float(np.mean(a_flag)) if a_flag else None, "B": float(np.mean(b_flag)) if b_flag else None}
    (RES / "analysis").mkdir(parents=True, exist_ok=True)
    (RES / f"analysis/hallucination_{which}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["judge", "handset", "score", "analyze"])
    ap.add_argument("--subset", choices=["smoke", "all"], default="smoke")
    a = ap.parse_args()
    {"judge": phase_judge, "handset": phase_handset, "score": phase_score, "analyze": phase_analyze}[a.phase](a.subset)
