"""012 Stage E step 2b (design `docs/design_specs/012E_ar_concept_swaps.md` §3-§4): phrases and cue-varied prompts from
an LLM through OpenRouter. CPU + API only; no model. EXPLORATORY.

  python stageE_generate.py phrases [--only c1,c2] [--tag smoke]   -> outputs/.../stageE/gen/phrases[_tag].json
  python stageE_generate.py prompts [--only c1,c2] [--tag smoke]   -> outputs/.../stageE/gen/prompts_checked[_tag].json

Every request and response is saved raw (gen/cache/, keyed by a hash of model + request body + attempt), so reruns
reuse them and never re-spend. The key is read from /workspace/.env and never logged. Code checks (stageE_common)
run before anything reaches the AR or the model.
"""
import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stageE_common as E                                                  # noqa: E402

MODEL = "google/gemini-3.8-flash"
PARAMS = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 16000, "reasoning": {"effort": "minimal"},
          "usage": {"include": True}}
SEED0 = 20261010
PROMPT_REQUESTS = 3
GEN = E.OUT / "gen"
CACHE = GEN / "cache"

# design §3, exact text
PHRASE_PROMPT = ('Write 40 short English phrases (3–12 words each) of the kind that appear in an assistant\'s reply. '
                 'Every phrase must {rule}. Vary the topic and the sentence form. Do not use any of these words: '
                 '{forbidden}. No numbering, quotes or lists. Return JSON: {{"phrases": [...]}}.')
RULES = {"starts": 'begin with the word "{c}"', "mentions": 'contain the word "{c}", but not as its first word'}

# design §4; text fixed here before any call (the design fixed its constraints, not its wording)
PROMPT_PROMPT = (
    'Write 24 short factual two-hop prompts in exactly this format: "Fact: The <relation> of <clue> is". The <clue> '
    'must describe "{c}" unambiguously without naming it, and the next word after "is" must be the answer. Example, '
    'for a different concept (Kenya): "Fact: The capital of the country where the Maasai Mara reserve lies is" '
    '(answer: Nairobi). Rules: never write the word "{c}"; use a different clue in every prompt; use at least 8 '
    'different relations and at most {max_rel} prompts per relation; every answer must be one to three words, and no '
    'answer may be used by more than 8 prompts; do not use any of these words: {forbidden}. Return JSON: '
    '{{"prompts": [{{"prompt": "...", "clue": "...", "relation": "...", "answer": "..."}}]}}.')


def api_key():
    return re.search(r"(?m)^OPENROUTER_API_KEY=(\S+)", Path("/workspace/.env").read_text()).group(1)


def call(prompt, seed, attempt=0):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "seed": seed, **PARAMS}
    h = hashlib.sha256((json.dumps(body, sort_keys=True) + f"#{attempt}").encode()).hexdigest()[:24]
    f = CACHE / f"{h}.json"
    if f.exists():
        return json.loads(f.read_text()), True
    for k in range(4):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {api_key()}",
                                                  "Content-Type": "application/json"})
            resp = json.load(urllib.request.urlopen(req, timeout=180))
            break
        except Exception:  # noqa: BLE001  (network / rate limit; retried, then raised)
            if k == 3:
                raise
            time.sleep(5 * (k + 1))
    rec = {"request": body, "attempt": attempt, "response": resp, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec))
    return rec, False


def parse_json(txt):
    txt = re.sub(r"^```(?:json)?|```$", "", (txt or "").strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", txt, flags=re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        pass
    # Fallback (2026-10-10): the prompt says "no ... quotes", and some responses list the phrases unquoted
    # ({"phrases": [Green tea ..., Green ...]}). Split at commas followed by a capital letter; the code checks reject
    # any broken fragment. Used for phrases only; flagged in the record.
    m = re.search(r'\{\s*"phrases"\s*:\s*\[(.*)\]\s*\}', txt, flags=re.S)
    if m:
        parts = [p.strip() for p in re.split(r",\s*(?=[A-Z0-9])", m.group(1)) if p.strip()]
        return {"phrases": parts, "_fallback_parse": True}
    return None


def ask(prompt, seed, key):
    """One logical request with up to 3 retries on a parse failure (each attempt saved and counted)."""
    log = []
    for attempt in range(4):
        rec, cached = call(prompt, seed, attempt)
        msg = rec["response"].get("choices", [{}])[0].get("message", {}).get("content")
        js = parse_json(msg)
        cost = (rec["response"].get("usage") or {}).get("cost") or 0.0
        log.append({"attempt": attempt, "cached": cached, "parsed": js is not None and key in js, "cost": cost,
                    "fallback_parse": bool(js and js.get("_fallback_parse"))})
        if js is not None and isinstance(js.get(key), list):
            return js[key], log
    return [], log


def run_phrases(fam, grams, only, workers):
    jobs = []
    for family, inv in fam.items():
        for i, (c, v) in enumerate(sorted(inv.items())):
            if only and c not in only:
                continue
            forb = ", ".join(v["forbidden"]) or "none"
            for mode in ("starts", "mentions"):
                prompt = PHRASE_PROMPT.format(rule=RULES[mode].format(c=c), forbidden=forb)
                jobs.append((family, c, mode, prompt, SEED0 + i))
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda j: ask(j[3], j[4], "phrases"), jobs))
    out, summ = {}, {"requests": len(jobs), "cost_usd": 0.0, "fail_counts": {}, "dropped": []}
    for (family, c, mode, prompt, seed), (raw, log) in zip(jobs, res):
        v = fam[family][c]
        kept, seen, fails = [], set(), {}
        for p in raw:
            if not isinstance(p, str):
                fails["not_string"] = fails.get("not_string", 0) + 1
                continue
            p = p.strip().strip('"“”').strip()
            why = E.check_phrase(p, c, mode, v["forbidden"], grams)
            if why is None and E.norm_text(p) in seen:
                why = "duplicate"
            if why:
                fails[why] = fails.get(why, 0) + 1
                continue
            seen.add(E.norm_text(p))
            kept.append(p)
        kept = kept[:E.N_KEEP]
        rec = {"n_returned": len(raw), "n_pass": len(seen), "kept": kept, "fails": fails, "calls": log, "seed": seed,
               "dropped": len(kept) < E.N_MIN}
        out.setdefault(family, {}).setdefault(c, {"kind": v["kind"], "weak_input": v["weak_input"],
                                                  "forbidden": v["forbidden"]})[mode] = rec
        summ["cost_usd"] += sum(x["cost"] for x in log if not x["cached"])
        for k, n in fails.items():
            summ["fail_counts"][k] = summ["fail_counts"].get(k, 0) + n
        if rec["dropped"]:
            summ["dropped"].append([family, c, mode, len(kept)])
    for family, inv in out.items():                                         # template and single-word sets (no LLM)
        for c, d in inv.items():
            d["template"] = {"kept": [t.format(c=c) for t in E.TEMPLATES]}
            d["word"] = {"kept": [c]}
    return out, summ


def run_prompts(fam, grams, only, workers):
    inv = {c: v for c, v in fam["multihop"].items() if v["kind"] == "mh" and (not only or c in only)}
    order = {c: i for i, c in enumerate(sorted(fam["multihop"]))}
    # 2026-10-10, after the France gate smoke (7/21 passed): PROMPT_REQUESTS requests of 24 per concept, merged in
    # request order. Request 0 keeps its original seed (cached); the text, checks, gate and caps are unchanged.
    jobs = [(c, PROMPT_PROMPT.format(c=c, max_rel=E.MAX_PER_REL, forbidden=", ".join(v["forbidden"]) or "none"),
             SEED0 + 100000 * (k + 1) + order[c]) for c, v in sorted(inv.items()) for k in range(PROMPT_REQUESTS)]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda j: ask(j[1], j[2], "prompts"), jobs))
    merged = {}
    for (c, prompt, seed), (raw, log) in zip(jobs, res):
        m = merged.setdefault(c, {"raw": [], "log": [], "seeds": []})
        m["raw"] += raw
        m["log"] += log
        m["seeds"].append(seed)
    out, summ = {}, {"requests": len(jobs), "requests_per_concept": PROMPT_REQUESTS, "cost_usd": 0.0, "fail_counts": {}}
    for c, m in merged.items():
        raw, log, seed = m["raw"], m["log"], m["seeds"]
        ok, seen, fails = [], set(), {}
        for r in raw:
            why = E.check_prompt(r, c, inv[c]["forbidden"], grams) if isinstance(r, dict) else "not_dict"
            if why is None and E.norm_text(r["prompt"]) in seen:
                why = "duplicate"
            if why:
                fails[why] = fails.get(why, 0) + 1
                continue
            seen.add(E.norm_text(r["prompt"]))
            ok.append({k: r[k].strip() for k in ("prompt", "clue", "relation", "answer")})
        out[c] = {"n_returned": len(raw), "checked": ok, "fails": fails, "calls": log, "seed": seed,
                  "n_relations": len({r["relation"].lower() for r in ok})}
        summ["cost_usd"] += sum(x["cost"] for x in log if not x["cached"])
        for k, n in fails.items():
            summ["fail_counts"][k] = summ["fail_counts"].get(k, 0) + n
    return out, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("phrases", "prompts"))
    ap.add_argument("--only", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    only = {x for x in a.only.split(",") if x}
    fam, _items = E.inventory()
    grams = set().union(*[E.ngrams(p) for p in E.eval_prompts()])
    t0 = time.time()
    out, summ = (run_phrases if a.what == "phrases" else run_prompts)(fam, grams, only, a.workers)
    sfx = f"_{a.tag}" if a.tag else ""
    GEN.mkdir(parents=True, exist_ok=True)
    name = "phrases" if a.what == "phrases" else "prompts_checked"
    (GEN / f"{name}{sfx}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    meta = {"label": "EXPLORATORY; 012 Stage E generation (design 012E §3-§4)", "model": MODEL, "params": PARAMS,
            "seed0": SEED0, "prompt_text": PHRASE_PROMPT if a.what == "phrases" else PROMPT_PROMPT,
            "only": sorted(only), **summ, "seconds": round(time.time() - t0, 1)}
    E.RES.mkdir(parents=True, exist_ok=True)
    (E.RES / f"stageE_generation_{a.what}{sfx}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps({k: meta[k] for k in ("requests", "cost_usd", "fail_counts", "seconds")}, indent=None))


if __name__ == "__main__":
    main()
