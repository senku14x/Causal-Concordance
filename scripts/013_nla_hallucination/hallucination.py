"""013 main pipeline (spec docs/design_specs/013_nla_hallucination.md; EXPLORATORY).

  python hallucination.py judge   --subset smoke|all   # §3: judges A (cleaner) and B (labeller), cached, cost-logged
  python hallucination.py handset --subset smoke|all   # writes the hand-check file (§7 G2) - read BEFORE any Δ
  python hallucination.py score   --subset smoke|all   # §4-5: versions + reconstructor (GPU), G-repro
  python hallucination.py analyze --subset smoke|all   # §5-7: Δ, rules, G1/G3, per state, J share
  NLAH_FAMILY=typo python hallucination.py gate        # 014 Part C: the 013 de-risk gate R1 on typo (before judging)

NLAH_FAMILY=typo (014 Part C, docs/design_specs/014_typo.md §5) swaps the material for typo's whole-activation NLA L42
readouts (outputs/014_typo/) and writes to outputs/014_typo/nlah, results/014_typo; everything else is unchanged.

Sections referenced (§n) are the spec's.
"""
import argparse
import ast
import hashlib
import json
import math
import os
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
FAMILY = os.environ.get("NLAH_FAMILY", "multihop")
assert FAMILY in ("multihop", "typo"), FAMILY
TYPO = ROOT / "outputs/014_typo"
if FAMILY == "typo":
    OUT, RES, STATES = TYPO / "nlah", ROOT / "results/014_typo", ["clean"]
else:
    OUT, RES, STATES = ROOT / "outputs/013_nla_hallucination", ROOT / "results/013_nla_hallucination", ["clean", "J_L24-35", "rand_L24-35"]
MODEL_A, MODEL_B = "google/gemini-3.8-flash", "z-ai/glm-5.3-flash"
N_SEEDS, MIN_TOK = 5, 10
SPEND_CAP = 10.0

# --------------------------------------------------------------------------------------------- §2 material
def explanations():
    """The 522 explanations in a fixed order (item, state, sample); typo: the 200 whole-activation ones."""
    from nla.schema import extract_explanation
    if FAMILY == "typo":
        meta = {json.loads(l)["item"]: json.loads(l) for l in open(TYPO / "meta.jsonl")}
        rows = []
        for l in open(TYPO / "readouts_nla.jsonl"):
            r = json.loads(l)
            if r["piece"] != "whole":
                continue
            m = meta[r["item"]]
            for k, raw in enumerate(r["samples"]):
                e = extract_explanation(raw)
                assert e is not None, (r["item"], k)
                rows.append({"eid": f"{r['item']}|clean|{k}", "item": r["item"], "state": "clean", "sample": k, "prompt": m["prompt"],
                             "continuation": m["continuation"], "src": None, "tgt": m["target"], "explanation": e})
        rows.sort(key=lambda x: (x["item"], x["sample"]))
        assert len(rows) == 2 * len(meta), len(rows)
        return rows
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
    if FAMILY == "typo":
        return sorted(rng.sample(rows, 5), key=lambda x: x["eid"])
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


KINDS = ["misquote", "invented_context", "wrong_entity", "false_fact"]
PROMPT_K = ("You classify false claims found in an automatically generated explanation of a language model's internal state.\n\n"
            "{inputs}\n\nThese spans of the EXPLANATION were judged false (contradicted):\n{spans}\n\n"
            "Give each span exactly one kind. Use the FIRST kind in this list that applies:\n"
            "- misquote: it quotes or paraphrases the prompt's own content in altered wording (including versions that swap in an "
            "entity or answer).\n"
            "- invented_context: it asserts text, structure or framing that is not in the input (headers, instructions, quiz names, "
            "answer options, earlier questions, source documents).\n"
            "- wrong_entity: it names an entity or answer that conflicts with the model's continuation, without misquoting the prompt.\n"
            "- false_fact: it is a false statement about the world.\n\n"
            "Return JSON only: {{\"kinds\": [\"misquote\" | \"invented_context\" | \"wrong_entity\" | \"false_fact\", ...]}}, one per "
            "numbered span, in order.")


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
    def one(r):
        spent = 0.0
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
        # Amendment A1: kind of each contradicted span (judge B)
        cspans = [s for s in ((a or {}).get("contradicted") or []) if isinstance(s, str)]
        kinds, okK, recK = None, None, None
        if okA and cspans:
            pk = PROMPT_K.format(inputs=inp, spans="\n".join(f"{i + 1}. {s}" for i, s in enumerate(cspans)))
            recK, hitK = call(MODEL_B, pk, cache, "K")
            kk = parse_json(recK["response"]["choices"][0]["message"]["content"] or "")
            if not (kk and isinstance(kk.get("kinds"), list) and len(kk["kinds"]) == len(cspans)):
                recK, hitK = call(MODEL_B, pk + "\n\nReturn valid JSON only, one kind per span.", cache, "K")
                kk = parse_json(recK["response"]["choices"][0]["message"]["content"] or "")
            if not hitK:
                spent += float((recK["response"].get("usage") or {}).get("cost") or 0.0)
            kinds = kk.get("kinds") if kk else None
            okK = isinstance(kinds, list) and len(kinds) == len(cspans) and all(k in KINDS for k in kinds)
        kind_ranges = {k: [] for k in KINDS}
        if okK:
            for s, k in zip(cspans, kinds):
                rr, _ = locate([s], r["explanation"])
                kind_ranges[k] += rr
        item = ({**{k: r[k] for k in ("eid", "item", "state", "sample")}, "A_raw": a, "A_ok": okA, "A_bad_spans": bad_c + bad_u,
                        "contradicted_ranges": cr, "unsupported_ranges": ur, "B_labels": labels, "B_ok": okB, "sentences": sents,
                        "span_kinds": kinds, "K_ok": okK, "kind_ranges": kind_ranges,
                        "usage": {"A": recA["response"].get("usage"), "B": recB["response"].get("usage"),
                                  "K": recK["response"].get("usage") if recK else None}})
        return item, spent

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(one, rows))
    results = [o[0] for o in outs]
    spent = sum(o[1] for o in outs)
    print(f"[judge] {len(results)} done, new spend ${spent:.3f}", flush=True)
    if spent > SPEND_CAP:
        print(f"WARNING: spend ${spent:.2f} above cap ${SPEND_CAP}", flush=True)
    (OUT / "judge").mkdir(parents=True, exist_ok=True)
    with open(OUT / f"judge/judged_{which}.jsonl", "w") as f:
        for x in results:
            f.write(json.dumps(x) + "\n")
    meta = {"which": which, "n": len(results), "models": {"A": MODEL_A, "B": MODEL_B}, "new_spend_usd": spent,
            "prompt_A_template": PROMPT_A, "prompt_B_template": PROMPT_B, "defs": DEFS, "state_note": STATE_NOTE,
            "G1_A_ok_rate": sum(x["A_ok"] for x in results) / len(results), "B_ok_rate": sum(x["B_ok"] for x in results) / len(results),
            "prompt_K_template": PROMPT_K, "K_ok_rate_among_tagged": (lambda t: sum(x["K_ok"] for x in t) / len(t) if t else None)(
                [x for x in results if x["K_ok"] is not None]),
            "kind_counts": {k: sum(len(x["kind_ranges"][k]) for x in results) for k in KINDS}}
    (OUT / f"judge/judge_meta_{which}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if not k.startswith(("prompt", "defs", "state"))}), flush=True)


def phase_handset(which):
    rows = {r["eid"]: r for r in explanations()}
    if which in ("extra10", "kinds"):
        allj = [json.loads(l) for l in open(OUT / "judge/judged_all.jsonl")]
        smoke = {r["eid"] for r in subset(explanations(), "smoke")}
        if which == "kinds":
            J = [x for x in allj if x["eid"] in smoke]
        else:
            rest = [x for x in allj if x["eid"] not in smoke]
            J = sorted(random.Random(29).sample(rest, 10), key=lambda x: x["eid"])
    else:
        J = [json.loads(l) for l in open(OUT / f"judge/judged_{which}.jsonl")]
    lines = []
    for x in J:
        r = rows[x["eid"]]
        e = r["explanation"]
        lines.append(f"### {x['eid']}  (src {r['src']} -> tgt {r['tgt']})\nPROMPT: {r['prompt']}\nCONTINUATION: {r['continuation']!r}\n"
                     f"EXPLANATION:\n{e}\n\nA contradicted (kind from B):\n" + "\n".join(
                         f"  C{i} [{k}]: {s!r}" for i, (s, k) in enumerate(zip((x["A_raw"] or {}).get("contradicted") or [], x.get("span_kinds") or ["?"] * 99)))
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


def sentence_units(text):
    out, pos = [], 0
    for s in sentences(text):
        rx = ws_regex(s)
        m = rx.search(text, pos) if rx else None
        if m:
            out.append((m.start(), m.end()))
            pos = m.end()
    return out


def matched(e, pool, target, ntok, seed0, salt):
    return [delete(e, random_match(e, pool, target, ntok, seed0 + 100 * s + salt)[0]) for s in range(N_SEEDS)]


def build_versions(e, cr, ur, ntok, eid, kr=None):
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
    # Amendment A1: per-kind cleaning, clause- and sentence-level matched controls
    ss = sentence_units(e)
    v["RS1"] = matched(e, ss, info["tok_O"] - info["tok_C1"], ntok, seed0, 11) if not info["nothing_to_clean"] else []
    for kname, short in (("invented_context", "i"), ("misquote", "ii")):
        rr = (kr or {}).get(kname) or []
        info[f"has_{short}"] = len(rr) > 0
        if not rr:
            continue
        v[f"C1{short}"] = delete(e, [tuple(x) for x in rr])
        info[f"tok_C1{short}"] = ntok(v[f"C1{short}"])
        tg = info["tok_O"] - info[f"tok_C1{short}"]
        v[f"R{short}"] = matched(e, us, tg, ntok, seed0, 21 if short == "i" else 31)
        v[f"RS{short}"] = matched(e, ss, tg, ntok, seed0, 41 if short == "i" else 51)
    return v, info

# --------------------------------------------------------------------------------------------- §5 scoring
def gold_vectors():
    """{(item, state): L42 activation, float32}, the exact vectors the NLA read."""
    if FAMILY == "typo":
        caps = torch.load(TYPO / "captures.pt", map_location="cpu", weights_only=False)
        return {(it, "clean"): caps["H"][42][n].float() for n, it in enumerate(caps["items"])}
    caps = torch.load(ROOT / "outputs/012_reader_concordance/stageB/captures.pt", map_location="cpu", weights_only=False)
    li = caps["layers"].index(42)
    return {(it, st): caps["X"][n, caps["arms"].index(st), li].float() for n, it in enumerate(caps["items"]) for st in STATES}


def critic_setup():
    import yaml
    from transformers import AutoTokenizer
    from nla.models import NLACriticModel
    from nla.schema import normalize_activation
    from nla.utils.critic import critic_predict
    import readers as RD
    root = Path(RD.nla_rl_paths()[0])
    template = yaml.safe_load(open(root / "nla_meta.yaml"))["prompt_templates"]["critic"]
    scale = math.sqrt(5120)
    tok = AutoTokenizer.from_pretrained(root / "av_base")
    critic = NLACriticModel.from_pretrained(str(root / "ar_reconstructor"), dtype=torch.bfloat16, device_map="cuda").eval()

    @torch.no_grad()
    def recon(text):
        ids = tok.encode(template.format(explanation=text), add_special_tokens=False)[:1024]
        p = critic_predict(critic, torch.tensor([ids], device="cuda"), None, scale)[0].float().cpu()
        return normalize_activation(p[None], scale)[0]
    return recon, scale


def phase_gate(which):
    """014 Part C gate = 013's de-risk rule R1 (FVE of the first sample >= 40% and matched - shuffled >= 20 points; B =
    predict-the-mean on the family's own activations; seeded derangement over items, as derisk.py)."""
    assert FAMILY == "typo", "the multihop gate is derisk.py's"
    from nla.schema import normalize_activation, compute_predict_mean_baselines
    recon, scale = critic_setup()
    gold = gold_vectors()
    _, B = compute_predict_mean_baselines(torch.stack(list(gold.values())), scale)
    first = [r for r in explanations() if r["sample"] == 0]
    gn = [normalize_activation(gold[(r["item"], "clean")][None], scale)[0] for r in first]
    P = [recon(r["explanation"]) for r in first]
    idx = list(range(len(first)))
    perm, rng = idx[:], random.Random(0)
    while any(a == b for a, b in zip(perm, idx)):
        rng.shuffle(perm)
    mse = [float(((P[i] - gn[i]) ** 2).mean()) for i in idx]
    msh = [float(((P[j] - gn[i]) ** 2).mean()) for i, j in zip(idx, perm)]
    fve = lambda xs: 100 * (1 - float(np.mean(xs)) / B)
    rb = random.Random(0)
    boots = [fve([mse[k] for k in (rb.randrange(len(idx)) for _ in idx)]) for _ in range(2000)]
    m, s_ = fve(mse), fve(msh)
    res = {"label": "EXPLORATORY (014 Part C gate = 013 de-risk R1)", "n": len(idx), "baseline_B": B, "fve_pct_first": m,
           "fve_pct_first_boot95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
           "fve_pct_shuffled": s_, "matched_minus_shuffled": m - s_, "rule": {"R1_min_fve_pct": 40.0, "R1_min_matched_minus_shuffled": 20.0},
           "R1_pass": m >= 40.0 and m - s_ >= 20.0}
    (RES / "analysis").mkdir(parents=True, exist_ok=True)
    (RES / "analysis/nlah_gate.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


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
    gold = gold_vectors()
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
        v, info = build_versions(r["explanation"], [tuple(a) for a in x["contradicted_ranges"]], [tuple(a) for a in x["unsupported_ranges"]], ntok, x["eid"],
                                 x.get("kind_ranges") if x.get("K_ok") else None)
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
    dr = [json.loads(l) for l in open(OUT / "derisk/multihop_ar_scores.jsonl")] if FAMILY == "multihop" else []
    order = sorted({r["item"] for r in rows.values()})
    ref = {(order[d["i"]], d["sample"]): d["mse"] for d in dr if "mse" in d}  # typo: no reference (the gate phase stands in)
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
    # Amendment A1 (POST HOC; §6 bands applied descriptively)
    res["A1_sentence_delta_RS1_minus_C1"] = stat(elig, "RS1", "C1")
    for short, kname in (("i", "invented_context"), ("ii", "misquote")):
        ek = [s for s in S if s["A_ok"] and s.get(f"has_{short}") and s.get(f"tok_C1{short}", 0) >= MIN_TOK]
        out_k = {"n_eligible": len(ek),
                 "delta_clause": stat(ek, f"R{short}", f"C1{short}"), "delta_sentence": stat(ek, f"RS{short}", f"C1{short}"),
                 "O_minus_C1k": stat(ek, "O", f"C1{short}"),
                 "token_share_deleted_mean": float(np.mean([(s["tok_O"] - s[f"tok_C1{short}"]) / s["tok_O"] for s in ek])) if ek else None,
                 "per_state_delta_clause": {st: stat([s for s in ek if s["state"] == st], f"R{short}", f"C1{short}") for st in STATES}}
        for kk in ("delta_clause", "delta_sentence"):
            if out_k[kk]:
                out_k[kk]["descriptive_band"] = verdict(out_k[kk]["mean"], out_k[kk]["ci95"])
        res[f"A1_kind_{kname}"] = out_k
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
    ap.add_argument("phase", choices=["judge", "handset", "score", "analyze", "gate"])
    ap.add_argument("--subset", choices=["smoke", "all", "extra10", "kinds"], default="smoke")
    a = ap.parse_args()
    {"judge": phase_judge, "handset": phase_handset, "score": phase_score, "analyze": phase_analyze, "gate": phase_gate}[a.phase](a.subset)
