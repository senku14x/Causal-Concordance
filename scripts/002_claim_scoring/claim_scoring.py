"""Design 002 — reader claim scoring (docs/design_specs/002_claim_scoring.md).

    python scripts/002_claim_scoring/claim_scoring.py pilot        # judge the 20 pilot readouts, print them
    python scripts/002_claim_scoring/claim_scoring.py judge        # judge every verbalizer readout (resumable)
    python scripts/002_claim_scoring/claim_scoring.py lens         # token-reader closed-set scores (judge-free)
    python scripts/002_claim_scoring/claim_scoring.py prompt_only  # stock-model prompt-only baseline (GPU)
    python scripts/002_claim_scoring/claim_scoring.py analysis

Reads only saved 001 outputs. The judge sees the readout text and the table's 5 candidates, never the prompt or truth.
The OpenRouter key is read from /workspace/.env and never logged.
"""
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import make_sets as MS  # noqa: E402
import registry as R  # noqa: E402

EXP = "002_claim_scoring"
RES = R.out_dir("results", EXP)
JDIR = R.out_dir("results", EXP, "judge")
ANA = R.out_dir("results", EXP, "analysis")
SNAP = MS.SNAP
CT, NONE, CONF = "cannot tell", "none", "conflicting"
SPECIAL = [NONE, CT, CONF]
ROLES = ("start", "intermediate", "answer")

JUDGE = {"model": "google/gemini-3.8-flash", "reasoning": {"effort": "minimal"}, "temperature": 0, "max_tokens": 2000,
         "concurrency": 16, "attempts": 4, "spend_cap_usd": 10.0}
SYSTEM = "You extract claims from text about a colour puzzle. Reply with JSON only."
PROMPT = """You will read a short text written by an AI system. The text may describe a colour puzzle in which a colour rule is
applied twice: START colour -> INTERMEDIATE colour (after one application) -> ANSWER colour (after two).

Candidate colours: {candidates}

According to the text, which candidate is the START colour, which is the INTERMEDIATE colour, and which is the ANSWER
colour? Rules:
- If the text says to apply the rule twice to X (or that X is where the chain begins), X is the START.
- If the text says a first step, first iteration or first answer produced X, X is the INTERMEDIATE.
- A chain written as A -> B -> C means START = A, INTERMEDIATE = B, ANSWER = C.
- A colour the text presents as its answer, result or final colour (including a bare answer such as "- white") is the
  ANSWER.
- If the text gives a role a colour that is not a candidate, answer "none".
- If the text does not state a role, or only lists or mentions colours without roles, answer "cannot tell".
- If the text gives the same role different colours in different places, or offers alternatives ("X or Y"), answer
  "conflicting", even when none of those colours is a candidate.
Use only the text. Do not solve the puzzle yourself.

TEXT:
{readout}"""


def prompt_sha():
    return hashlib.sha256((SYSTEM + "\n" + PROMPT).encode()).hexdigest()


def cand_order(rid, candidates):
    """Per-readout candidate order (the same rule make_sets.py used for the pilot/validation files)."""
    return [str(c) for c in np.random.default_rng(MS.SEED + sum(map(ord, rid))).permutation(candidates)]


def parse_id(rid):
    item, *key, k = rid.split("|")
    if key[0] == "none":                      # none|L42|nla|none
        return {"table": item, "role": "none", "cell": None, "layer": int(key[1][1:]), "reader": key[2], "arm": "none",
                "sample": int(k)}
    return {"table": item, "role": key[0], "cell": int(key[1][4:]), "layer": int(key[2][1:]), "reader": key[3],
            "arm": key[4], "sample": int(k)}


def all_readouts():
    sn = json.load(open(SNAP))
    tab = {t["item"]: t for t in sn["tables"]}
    rows = MS.readouts(sn)
    for r in rows:
        r["candidates"] = cand_order(r["id"], MS.table_candidates(tab[r["table"]]))
    return rows, tab


def _key():
    m = re.search(r"(?m)^OPENROUTER_API_KEY=(\S+)", Path("/workspace/.env").read_text())
    if not m or not m.group(1).startswith("sk-or-"):
        raise SystemExit("OPENROUTER_API_KEY missing from /workspace/.env")
    return m.group(1)


def _schema(cands):
    enum = list(cands) + SPECIAL
    return {"name": "roles", "strict": True,
            "schema": {"type": "object", "additionalProperties": False,
                       "properties": {r: {"type": "string", "enum": enum} for r in ROLES}, "required": list(ROLES)}}


class Spend:
    def __init__(self, start=0.0):
        self.usd = start
        self.calls = 0


async def _judge_one(client, key, r, spend, sem):
    body = {"model": JUDGE["model"], "max_tokens": JUDGE["max_tokens"], "temperature": JUDGE["temperature"],
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": PROMPT.format(candidates=", ".join(r["candidates"]), readout=r["text"])}],
            "response_format": {"type": "json_schema", "json_schema": _schema(r["candidates"])},
            "reasoning": JUDGE["reasoning"], "usage": {"include": True}, "provider": {"require_parameters": True}}
    last = None
    async with sem:
        for attempt in range(1, JUDGE["attempts"] + 1):
            if spend.usd >= JUDGE["spend_cap_usd"]:
                return {"id": r["id"], "status": "spend_cap", "attempts": attempt - 1}
            try:
                resp = await client.post("https://openrouter.ai/api/v1/chat/completions",
                                         headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
                d = resp.json()
                spend.calls += 1
                usage = d.get("usage") or {}
                spend.usd += float(usage.get("cost") or 0.0)
                if resp.status_code != 200 or not d.get("choices"):
                    raise ValueError(f"HTTP {resp.status_code}: {json.dumps(d)[:200]}")
                out = json.loads(d["choices"][0]["message"]["content"] or "")
                enum = set(r["candidates"]) | set(SPECIAL)
                if not isinstance(out, dict) or any(out.get(k) not in enum for k in ROLES):
                    raise ValueError(f"schema violation: {str(out)[:200]}")
                return {"id": r["id"], "status": "ok", "judge": {k: out[k] for k in ROLES}, "candidates": r["candidates"],
                        "model": d.get("model"), "provider": d.get("provider"), "attempts": attempt,
                        "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "cost")},
                        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")}
            except Exception as e:                                  # noqa: BLE001 — retried, then recorded
                last = f"{type(e).__name__}: {str(e)[:200]}"
                await asyncio.sleep(2 ** attempt)
    return {"id": r["id"], "status": "judge_error", "error": last, "candidates": r["candidates"], "attempts": JUDGE["attempts"]}


async def _run(rows, out_path):
    import httpx
    done = {}
    if out_path.exists():
        for line in open(out_path):
            x = json.loads(line)
            if x["status"] == "ok":
                done[x["id"]] = x
    todo = [r for r in rows if r["id"] not in done]
    prior = 0.0                                   # the cap counts every judge call ever made (pilot, discarded runs)
    for f in JDIR.glob("*.jsonl"):
        if f != out_path:
            prior += sum(float((json.loads(l).get("usage") or {}).get("cost") or 0) for l in open(f))
    spend = Spend(prior + sum(float((x.get("usage") or {}).get("cost") or 0) for x in done.values()))
    print(f"{len(done)} already judged, {len(todo)} to go; spend so far ${spend.usd:.3f}", flush=True)
    sem = asyncio.Semaphore(JUDGE["concurrency"])
    key = _key()
    t0 = time.time()
    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(_judge_one(client, key, r, spend, sem)) for r in todo]
        with open(out_path, "a") as f:
            for i, fut in enumerate(asyncio.as_completed(tasks), 1):
                x = await fut
                x.update(parse_id(x["id"]))
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
                f.flush()
                if i % 200 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} judged, ${spend.usd:.3f}, {time.time() - t0:.0f}s", flush=True)
    return spend


def write_manifest(stage, extra):
    m = {"stage": stage, "time_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), "judge": JUDGE,
         "prompt_sha256": prompt_sha(), **R.manifest_block(), **extra}
    (RES / f"manifest_{stage}.json").write_text(json.dumps(m, indent=1, default=str))


def stage_pilot():
    rows = [json.loads(l) for l in open(RES / "validation" / "pilot.jsonl")]
    out = JDIR / "pilot_claims.jsonl"
    if out.exists():
        out.unlink()                               # the pilot is for prompt debugging; each pass starts fresh
    spend = asyncio.run(_run(rows, out))
    texts = {r["id"]: r for r in rows}
    for line in open(out):
        x = json.loads(line)
        r = texts[x["id"]]
        print(f"\n== {x['id']} [{r['reader']}/{r['arm']}] {x['status']} {x.get('judge')} cands={r['candidates']}")
        print("   " + r["text"].replace("\n", " ⏎ ")[:700])
    print(f"\npilot spend ${spend.usd:.4f}, prompt sha {prompt_sha()[:12]}")


def stage_judge():
    rows, _ = all_readouts()
    spend = asyncio.run(_run(rows, JDIR / "claims.jsonl"))
    xs = [json.loads(l) for l in open(JDIR / "claims.jsonl")]
    status = {s: sum(x["status"] == s for x in xs) for s in {x["status"] for x in xs}}
    write_manifest("judge", {"n_readouts": len(rows), "status": status, "spend_usd": spend.usd, "calls": spend.calls})
    print("judge done", status, f"${spend.usd:.3f}")


if __name__ == "__main__":
    stage = sys.argv[1]
    {"pilot": stage_pilot, "judge": stage_judge}[stage]()
