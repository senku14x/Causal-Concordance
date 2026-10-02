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


SCHEMA_VERSION = "v2.1"          # one closed candidate set per table (both pair paths + WSB off-path options)
FROZEN_SHA = "e1cc74d77771b23ed2d9f0c3594503123b8d3c8f6ee2b38a360d8d9fae240ab0"   # prompt frozen after pilot pass 2


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
                        "prompt_sha": prompt_sha(), "schema": SCHEMA_VERSION, "text_sha": hashlib.sha256(r["text"].encode()).hexdigest()[:16],
                        "model": d.get("model"), "provider": d.get("provider"), "attempts": attempt,
                        "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "cost")},
                        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")}
            except Exception as e:                                  # noqa: BLE001 — retried, then recorded
                last = f"{type(e).__name__}: {str(e)[:200]}"
                await asyncio.sleep(2 ** attempt)
    return {"id": r["id"], "status": "judge_error", "error": last, "candidates": r["candidates"], "attempts": JUDGE["attempts"]}


async def _run(rows, out_path):
    import httpx
    done, stale = {}, 0
    want = {r["id"]: r for r in rows}
    if out_path.exists():
        for line in open(out_path):
            x = json.loads(line)
            if x["status"] != "ok":
                continue
            # a cached row is reused only if produced under this prompt, schema and candidate order (review point 4);
            # rows of the first v2.1 run predate the per-row fields and carry the frozen prompt by construction
            same = (x.get("prompt_sha", FROZEN_SHA) == prompt_sha() and x.get("schema", SCHEMA_VERSION) == SCHEMA_VERSION
                    and x["id"] in want and x["candidates"] == want[x["id"]]["candidates"])
            if same:
                done[x["id"]] = x
            else:
                stale += 1
    if stale:
        raise SystemExit(f"{stale} cached rows were produced under another prompt/schema/candidate order: "
                         f"move {out_path.name} aside (keep it as superseded) and rerun")
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
    n = 1
    while (JDIR / f"pilot_claims_pass{n}.jsonl").exists() or (n == 2 and (JDIR / "pilot_claims.jsonl").exists()):
        n += 1
    out = JDIR / f"pilot_claims_pass{n}.jsonl"    # every pilot pass is archived (pass 1 of v2 was overwritten; pass 2 = pilot_claims.jsonl)
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


# ------------------------------------------------------------------------------------------- token readers
COLOURS = ("black", "blue", "brown", "gold", "gray", "green", "orange", "pink", "purple", "red", "teal", "white")
LENS_READOUTS = ("logit", "J_NP_standard", "J_NP_cosine", "J_CB_standard", "R_CB_standard")


def _colour_token_ids(tok):
    """Single-token spellings " c" / " C" per colour; multi-token spellings are skipped (and listed)."""
    ids, skipped = {}, []
    for c in COLOURS:
        ids[c] = []
        for form in (" " + c, " " + c.capitalize()):
            t = tok(form, add_special_tokens=False)["input_ids"]
            (ids[c].append(t[0]) if len(t) == 1 else skipped.append(form))
    return ids, skipped


def stage_lens():
    """Design 002 §5.2: every lens's score for all 12 colours, per render x cell x layer, on the saved SEC captures, plus
    the same lens on the dev-mean activation (the no-activation reference). Candidate sets are applied in analysis."""
    import torch
    import readers as RD
    out = R.PROJECT / "outputs" / "001_leverage_profile" / "run"
    X = np.load(out / "captures_brew.npz")["X"]
    meta = json.loads((out / "captures_brew_meta.json").read_text())
    renders, cells, layers = meta["renders"], meta["cells"], meta["layers"]
    nat = np.array([r["role"] in ("member", "partner") for r in renders])
    tok = R.make_tokenizer()
    model = R.load_model()
    VL = RD.VectorLenses(model, tok, layers)
    ids, skipped = _colour_token_ids(tok)
    flat = sorted({i for v in ids.values() for i in v})
    col_of = {i: c for c, v in ids.items() for i in v}
    W = VL.W
    res = {"cells": cells, "layers": layers, "colours": list(COLOURS), "readouts": list(LENS_READOUTS),
           "skipped_spellings": skipped, "renders": [{k: r[k] for k in ("key", "item", "role", "start", "z", "answer", "present", "trajectory")}
                                                     for r in renders], "scores": {}, "reference": {}}

    def per_colour(vals):                     # [n, len(flat)] -> [n, 12]: max over a colour's spellings
        idx = {i: j for j, i in enumerate(flat)}
        return torch.stack([vals[:, [idx[i] for i in ids[c]]].max(dim=1).values for c in COLOURS], dim=1)

    with torch.no_grad():
        for ci in range(len(cells)):
            for li, L in enumerate(layers):
                H = torch.tensor(X[:, ci, li], dtype=torch.float32, device=W.device)
                Href = H[torch.tensor(nat, device=W.device)].mean(0, keepdim=True)
                for name, Hs in (("scores", H), ("reference", Href)):
                    out_rows = {}
                    out_rows["logit"] = per_colour(torch.softmax(VL._n(Hs) @ W.T, -1)[:, flat])
                    for lens in ("J_NP", "J_CB", "R_CB"):
                        t = VL.lenses[lens].transport(Hs, L)
                        out_rows[f"{lens}_standard"] = per_colour(torch.softmax(VL._n(t) @ W.T, -1)[:, flat])
                    J = VL.lenses["J_NP"].jacobians[L]
                    den = (W[flat] @ J).norm(dim=1).clamp_min(1e-9)
                    out_rows["J_NP_cosine"] = per_colour(((Hs @ J.T) @ W[flat].T) / den)
                    for k, v in out_rows.items():
                        res[name][f"{k}|{ci}|{li}"] = v.float().cpu().numpy().round(7).tolist()
                print("lens", cells[ci], L, flush=True)
    d = R.out_dir("results", EXP, "lens")
    (d / "colour_scores.json").write_text(json.dumps(res))
    write_manifest("lens", {"n_renders": len(renders), "skipped_spellings": skipped})
    print("lens done; skipped spellings:", skipped)


def _prompt_only_text(text, start_sentence, question):
    """Replace the item's question with `question` (the rules, stir and start sentences are unchanged)."""
    q = "What colour is the potion at the end?"
    assert text.count(q) == 1
    return text.replace(q, question)


def stage_prompt_only():
    """Design 002 §6 baseline (A01 C18): stock Qwen3.6-27B reads the prompt and is asked for the start, the intermediate
    and the final colour; each of the 12 colours is scored as the full answer word (LSE over spellings, every token
    scored, logits in float32 from the final norm). Brew render: user turn + "Answer:" prefill, thinking off."""
    import torch
    import margin as MG
    import partners as P
    import wsb
    out = R.PROJECT / "outputs" / "001_leverage_profile" / "run"
    meta = json.loads((out / "captures_brew_meta.json").read_text())
    renders = [r for r in meta["renders"] if r["role"] in ("member", "partner")]
    _, bank = wsb.load_bank("brew")
    items = {it["name"]: it for it in bank}
    tok = R.make_tokenizer()
    model = R.load_model()
    norm, Wu = R.final_norm(model), model.get_output_embeddings().weight
    QUESTIONS = {"start": "What colour does the potion start out?",
                 "intermediate": "What colour is the potion after the first stir?",
                 "answer": "What colour is the potion at the end?"}
    sp = {c: MG.spellings(tok, "brew", c) for c in COLOURS}
    rows = []
    t0 = time.time()
    with torch.no_grad():
        for n, r in enumerate(renders):
            it = items[r["item"]]
            base = P.BREW_START_FMT.format(it["start"])
            text = it["prompt"].replace(base, P.BREW_START_FMT.format(r["start"])) if r["role"] == "partner" else it["prompt"]
            row = {k: r[k] for k in ("key", "item", "role", "start", "z", "answer")}
            for role, q in QUESTIONS.items():
                ids = wsb.render_text(tok, "brew", _prompt_only_text(text, base, q))
                seqs = [(c, s) for c in COLOURS for s in sp[c]]
                L = max(len(s) for _, s in seqs)
                batch = torch.full((len(seqs), len(ids) + L), tok.pad_token_id or 0, dtype=torch.long)
                att = torch.zeros_like(batch)
                for j, (_, s) in enumerate(seqs):
                    batch[j, :len(ids) + len(s)] = torch.tensor(ids + list(s))
                    att[j, :len(ids) + len(s)] = 1
                hs = model.model(input_ids=batch.to(model.device), attention_mask=att.to(model.device)).last_hidden_state
                lp = {}
                for j, (c, s) in enumerate(seqs):
                    pos = torch.arange(len(ids) - 1, len(ids) - 1 + len(s), device=hs.device)
                    logits = norm(hs[j, pos]).float() @ Wu.float().T
                    tot = float(torch.log_softmax(logits, -1)[torch.arange(len(s)), torch.tensor(s, device=hs.device)].sum())
                    lp.setdefault(c, []).append(tot)
                row[role] = {c: float(torch.logsumexp(torch.tensor(v, dtype=torch.float64), 0)) for c, v in lp.items()}
            rows.append(row)
            if (n + 1) % 25 == 0:
                print(f"prompt-only {n + 1}/{len(renders)} {time.time() - t0:.0f}s", flush=True)
    d = R.out_dir("results", EXP, "prompt_only")
    (d / "colour_logprobs.json").write_text(json.dumps({"questions": QUESTIONS, "rows": rows}))
    write_manifest("prompt_only", {"n_renders": len(rows), "seconds": time.time() - t0})
    print("prompt-only done", time.time() - t0)


# ---------------------------------------------------------------------------------------------- analysis
def _signflip_p(v):
    import itertools
    v = np.asarray(v, float)
    if len(v) == 0 or not np.any(v):
        return 1.0
    flips = np.array(list(itertools.product([1, -1], repeat=len(v))))
    return float(np.mean(np.abs((flips * v).mean(1)) >= abs(v.mean()) - 1e-12))


def _summ(v):
    import stats as ST
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "boot95_tables": ST.cluster_bootstrap(v, np.arange(len(v)))["ci"],
            "pos": int((v > 0).sum()), "neg": int((v < 0).sum()), "tie": int((v == 0).sum()),
            "signflip_p_unadj": _signflip_p(v), "per_table": [round(float(x), 4) for x in v]}


def _kappa(a, b):
    labs = sorted(set(a) | set(b))
    if len(labs) < 2:
        return float("nan")
    po = float(np.mean([x == y for x, y in zip(a, b)]))
    pe = sum((a.count(l) / len(a)) * (b.count(l) / len(b)) for l in labs)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def _truth(t, role):
    if role == "partner":
        return {"start": t["partner"]["start"], "intermediate": t["partner"]["z"], "answer": t["partner"]["answer"]}
    return {"start": t["start"], "intermediate": t["z"], "answer": t["answer"]}   # member, substituted, none


def _category(label, true_colour):
    """correct / incorrect (another candidate or an explicit non-candidate value) / no claim / conflicting."""
    if label == true_colour:
        return "correct"
    if label == CT:
        return "no claim"
    if label == CONF:
        return "conflicting"
    return "incorrect"


def stage_analysis():
    import importlib
    sys.path.insert(0, os.path.join(HERE, "..", "001_leverage_profile"))
    LP = importlib.import_module("leverage_profile")          # the 001 colour scorer, for mention(z)
    sn = json.load(open(SNAP))
    tab = {t["item"]: t for t in sn["tables"]}
    texts = {r["id"]: r["text"] for r in MS.readouts(sn)}
    claims = [json.loads(l) for l in open(JDIR / "claims.jsonl")]
    ok = [c for c in claims if c["status"] == "ok"]
    A = {}
    A["judge_status"] = {s: sum(c["status"] == s for c in claims) for s in sorted({c["status"] for c in claims})}
    A["judge_spend_usd"] = sum(float((c.get("usage") or {}).get("cost") or 0) for c in claims)
    A["judge_models_returned"] = sorted({c.get("model") for c in ok})

    # ---- validation: judge vs author (60)
    codes = {json.loads(l)["id"]: json.loads(l) for l in open(RES / "validation" / "author_codes.jsonl")}
    byid = {c["id"]: c for c in ok}
    val = {}
    for role in ROLES:
        a = [codes[i][role] for i in codes if i in byid]
        j = [byid[i]["judge"][role] for i in codes if i in byid]
        tr = [_truth(tab[parse_id(i)["table"]], parse_id(i)["role"])[role] for i in codes if i in byid]
        ca = [_category(x, t) for x, t in zip(a, tr)]
        cj = [_category(x, t) for x, t in zip(j, tr)]
        conf = {}
        for x, y in zip(ca, cj):
            conf[f"author={x} | judge={y}"] = conf.get(f"author={x} | judge={y}", 0) + 1
        val[role] = {"n": len(a), "exact_agreement": float(np.mean([x == y for x, y in zip(a, j)])), "kappa_labels": _kappa(a, j),
                     "category_agreement": float(np.mean([x == y for x, y in zip(ca, cj)])), "kappa_categories": _kappa(ca, cj),
                     "confusion": dict(sorted(conf.items(), key=lambda kv: -kv[1]))}
        per_reader = {}
        for rd in ("nla", "oracle", "oracle_wsb"):
            ids = [i for i in codes if i in byid and parse_id(i)["reader"] == rd]
            per_reader[rd] = {"n": len(ids), "exact_agreement": float(np.mean([codes[i][role] == byid[i]["judge"][role] for i in ids]))}
        val[role]["per_reader"] = per_reader
    for role in ROLES:
        ids = [i for i in codes if i in byid]
        val[role]["explicit_candidate_rate"] = {"author": float(np.mean([codes[i][role] not in SPECIAL for i in ids])),
                                                "judge": float(np.mean([byid[i]["judge"][role] not in SPECIAL for i in ids]))}
    A["validation"] = val
    pos_f = RES / "validation" / "positive_codes.jsonl"
    if pos_f.exists():                         # positive-claim supplement (review point 5): judge asserted a candidate
        pcodes = [json.loads(l) for l in open(pos_f)]
        pv = {}
        for role in ROLES:
            a = [p[role] for p in pcodes]; j = [byid[p["id"]]["judge"][role] for p in pcodes]
            pv[role] = {"n": len(a), "exact_agreement": float(np.mean([x == y for x, y in zip(a, j)])), "kappa_labels": _kappa(a, j)}
        pv["per_reader_intermediate"] = {rd: float(np.mean([p["intermediate"] == byid[p["id"]]["judge"]["intermediate"] for p in pcodes
                                                            if parse_id(p["id"])["reader"] == rd])) for rd in ("nla", "oracle", "oracle_wsb")}
        A["validation_positive"] = pv
    A["validation_gate"] = {"kappa_intermediate_labels": val["intermediate"]["kappa_labels"],
                            "passes": bool(val["intermediate"]["kappa_labels"] >= 0.6),
                            "judge_error_frac": A["judge_status"].get("judge_error", 0) / max(1, len(claims))}

    # ---- per-readout scores
    rows = []
    for c in ok:
        t = tab[c["table"]]
        tr = _truth(t, c["role"])
        j = c["judge"]
        z = tr["intermediate"]
        row = {k: c[k] for k in ("id", "table", "role", "cell", "layer", "reader", "arm", "sample")}
        assert c["candidates"] == cand_order(c["id"], MS.table_candidates(t)), c["id"]      # every row is v2.1
        row.update({"int_correct": j["intermediate"] == z, "start_correct": j["start"] == tr["start"],
                    "answer_correct": j["answer"] == tr["answer"],
                    # v2.1: every correct role value is a candidate, so "none" (an explicit non-candidate value) is an
                    # incorrect claim, not an abstention (review point 3)
                    "int_incorrect": j["intermediate"] not in (CT, CONF) and j["intermediate"] != z,
                    "int_other_candidate": j["intermediate"] not in SPECIAL and j["intermediate"] != z,
                    "z_misplaced": z in (j["start"], j["answer"]),
                    "int_none": j["intermediate"] == NONE, "int_cannot_tell": j["intermediate"] == CT,
                    "int_conflicting": j["intermediate"] == CONF,
                    "mention_z": z in LP.colours_in_text(texts[c["id"]]), "judge": j})
        rows.append(row)
    keys = ("int_correct", "int_incorrect", "start_correct", "answer_correct", "int_other_candidate", "z_misplaced", "int_none",
            "int_cannot_tell", "int_conflicting", "mention_z")
    agg = {}
    for r in rows:
        k = f'{r["reader"]}|{r["role"]}|cell{r["cell"]}|L{r["layer"]}|{r["arm"]}'
        agg.setdefault(k, []).append(r)
    A["rates"] = {k: {"n": len(v), **{m: float(np.mean([x[m] for x in v])) for m in keys}} for k, v in sorted(agg.items())}

    # ---- contrasts (within table) for the k=4 arms and the WSB grid
    def rate(rd, role, cell, L, arm, table, colour, field="intermediate"):
        xs = [r for r in rows if r["reader"] == rd and r["role"] == role and r["cell"] == cell and r["layer"] == L
              and r["arm"] == arm and r["table"] == table]
        return float(np.mean([x["judge"][field] == colour for x in xs])) if xs else None

    specs = [("nla", 0, 42), ("nla", 1, 42), ("nla", 2, 42), ("oracle", 2, 40), ("oracle", 2, 44)] + \
            [("oracle_wsb", c, L) for c in (0, 1, 2) for L in range(20, 61, 4)]
    con = {}
    for rd, cell, L in specs:
        tr_, rs_, rn_ = [], [], []
        for item, t in tab.items():
            z, zp = t["z"], t["partner"]["z"]
            m_z, p_z = rate(rd, "member", cell, L, "real", item, z), rate(rd, "partner", cell, L, "real", item, z)
            p_zp, m_zp = rate(rd, "partner", cell, L, "real", item, zp), rate(rd, "member", cell, L, "real", item, zp)
            if None not in (m_z, p_z, p_zp, m_zp):
                tr_.append(((m_z - p_z) + (p_zp - m_zp)) / 2)
            sh = rate(rd, "member", cell, L, "shuffled", item, z)
            if None not in (m_z, sh):
                rs_.append(m_z - sh)
        key = f"{rd}|cell{cell}|L{L}"
        con[key] = {"start_swap_tracking_roles": _summ(tr_) if tr_ else None, "real_minus_shuffled_roles": _summ(rs_) if rs_ else None}
    # real - none and the SEC bridge on the contract arms
    for rd, cell, L in (("nla", 2, 42), ("oracle", 2, 40), ("oracle", 2, 44)):
        key = f"{rd}|cell{cell}|L{L}"
        rn = []
        for item, t in tab.items():
            m_z = rate(rd, "member", cell, L, "real", item, t["z"])
            if rd == "nla":
                xs = [r for r in rows if r["reader"] == "nla" and r["arm"] == "none" and r["table"] == item]
                nz = float(np.mean([x["judge"]["intermediate"] == t["z"] for x in xs])) if xs else None
            else:
                nz = rate(rd, "member", cell, L, "none", item, t["z"])
            if None not in (m_z, nz):
                rn.append(m_z - nz)
        con[key]["real_minus_none_roles"] = _summ(rn) if rn else None
        pairs = {}
        for r in rows:
            if r["reader"] == rd and r["cell"] == cell and r["layer"] == L and r["arm"] in ("real", "real_sec"):
                pairs.setdefault((r["table"], r["role"], r["sample"]), {})[r["arm"]] = r["judge"]["intermediate"]
        both = [v for v in pairs.values() if len(v) == 2]
        con[key]["end_to_end_bridge_intermediate_agreement"] = {"n": len(both), "agreement": float(np.mean([v["real"] == v["real_sec"] for v in both])) if both else None}
    A["contrasts"] = con

    # ---- token readers: closed-set scores (lens stage) and top-10 claims (snapshot lists)
    ls = json.load(open(RES / "lens" / "colour_scores.json"))
    cols = ls["colours"]
    lens_rows = {}
    for ci, cell in enumerate(ls["cells"]):
        for li, L in enumerate(ls["layers"]):
            for rd in ls["readouts"]:
                S = np.array(ls["scores"][f"{rd}|{ci}|{li}"])
                Ref = np.array(ls["reference"][f"{rd}|{ci}|{li}"])[0]
                out = {"z": [], "start": [], "answer": [], "z12": [], "rr_z": []}
                for n, r in enumerate(ls["renders"]):
                    if r["role"] not in ("member", "partner"):
                        continue
                    off = sorted(set(r["present"]) - set(r["trajectory"]))
                    off = [str(x) for x in np.random.default_rng(MS.SEED + sum(map(ord, r["key"]))).permutation(off)[:2]]
                    cand = [r["start"], r["z"], r["answer"]] + off
                    sc = {c: S[n, cols.index(c)] for c in cand}
                    top = max(cand, key=lambda c: sc[c])
                    for role, colour in (("z", r["z"]), ("start", r["start"]), ("answer", r["answer"])):
                        out[role].append(bool(top == colour and sc[colour] > Ref[cols.index(colour)]))
                    order = sorted(cand, key=lambda c: -sc[c])
                    out["rr_z"].append(1.0 / (order.index(r["z"]) + 1))
                    out["z12"].append(bool(int(np.argmax(S[n])) == cols.index(r["z"])))
                lens_rows[f"{rd}|{cell}|L{L}"] = {"n": len(out["z"]), "hit_z": float(np.mean(out["z"])), "hit_start": float(np.mean(out["start"])),
                                                   "hit_answer": float(np.mean(out["answer"])), "mrr_z": float(np.mean(out["rr_z"])),
                                                   "top1_of_12_is_z": float(np.mean(out["z12"]))}
    A["lens_closed_set"] = lens_rows
    top10 = {}
    for item, t in tab.items():
        cand = MS.table_candidates(t)
        for key, v in t["readouts"].items():
            p = key.split("|")
            if len(p) == 5 and p[3] == "lenses" and p[0] == "member" and p[4] in ("real", "shuffled"):
                for lens, toks in v.items():
                    claim = next((c for tk in toks for c in [next(iter(LP.colours_in_tokens([tk])), None)] if c in cand), None)
                    k = f"{lens}|{p[1]}|{p[2]}|{p[4]}"
                    top10.setdefault(k, []).append({"claim": claim, "z": t["z"], "answer": t["answer"]})
    A["lens_top10_claims"] = {k: {"n": len(v), "claims_z": float(np.mean([x["claim"] == x["z"] for x in v])),
                                  "claims_answer": float(np.mean([x["claim"] == x["answer"] for x in v])),
                                  "abstains": float(np.mean([x["claim"] is None for x in v]))} for k, v in sorted(top10.items())}

    # ---- prompt-only baseline
    po = json.load(open(RES / "prompt_only" / "colour_logprobs.json"))["rows"]
    pacc = {}
    for role, truth in (("start", "start"), ("intermediate", "z"), ("answer", "answer")):
        pacc[role] = {"n": len(po), "top1_of_12": float(np.mean([max(r[role], key=r[role].get) == r[truth] for r in po]))}
    snap_keys = {(t["item"], t["start"]) for t in tab.values()} | {(t["item"], t["partner"]["start"]) for t in tab.values()}
    po10 = [r for r in po if (r["item"], r["start"]) in snap_keys]
    for role, truth in (("start", "start"), ("intermediate", "z"), ("answer", "answer")):
        pacc[role]["snapshot_tables_n"] = len(po10)
        pacc[role]["snapshot_tables_top1_of_12"] = float(np.mean([max(r[role], key=r[role].get) == r[truth] for r in po10]))
    pacc["note"] = "task baseline (prompt solvability), not a ceiling on readers; full prompt, not a prefix"
    A["prompt_only"] = pacc

    # ---- exploratory overlap on common items (member, real; 10 tables). Different layers are labelled as such.
    def det_text(rd, cell, L, item, z):
        xs = [r for r in rows if r["reader"] == rd and r["role"] == "member" and r["cell"] == cell and r["layer"] == L
              and r["arm"] == "real" and r["table"] == item]
        return {"asserts_z": any(x["judge"]["intermediate"] == z for x in xs), "mentions_z": any(x["mention_z"] for x in xs),
                "wrong_values": sorted({x["judge"]["intermediate"] for x in xs if x["judge"]["intermediate"] not in (CT, CONF, z)})}
    def det_lens(lens, cell, L, t):
        toks = t["readouts"].get(f"member|cell{cell}|L{L}|lenses|real", {}).get(lens, [])
        return {"asserts_z": False, "mentions_z": t["z"] in LP.colours_in_tokens(toks), "wrong_values": []}
    units = {"NLA L42 “:”": lambda t: det_text("nla", 2, 42, t["item"], t["z"]),
             "Oracle L40 “:”": lambda t: det_text("oracle", 2, 40, t["item"], t["z"]),
             "Oracle L44 “:”": lambda t: det_text("oracle", 2, 44, t["item"], t["z"]),
             "J_NP cosine L42 “:” (top-10)": lambda t: det_lens("J_NP_cosine", 2, 42, t),
             "J_NP cosine L60 “:” (top-10)": lambda t: det_lens("J_NP_cosine", 2, 60, t)}
    D = {u: {item: f(t) for item, t in tab.items()} for u, f in units.items()}
    ov = {}
    names = list(units)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for lvl in ("mentions_z", "asserts_z"):
                if lvl == "asserts_z" and ("top-10" in a or "top-10" in b):
                    continue
                xa = [D[a][k][lvl] for k in tab]; xb = [D[b][k][lvl] for k in tab]
                ov[f"{a} vs {b}|{lvl}"] = {"both": sum(x and y for x, y in zip(xa, xb)), "only_first": sum(x and not y for x, y in zip(xa, xb)),
                                            "only_second": sum(y and not x for x, y in zip(xa, xb)), "neither": sum(not x and not y for x, y in zip(xa, xb))}
            if "top-10" not in a and "top-10" not in b:
                ov[f"{a} vs {b}|shared_wrong_intermediate"] = {"tables": sum(bool(set(D[a][k]["wrong_values"]) & set(D[b][k]["wrong_values"])) for k in tab)}
    A["overlap"] = {"note": "exploratory; 10 tables; member activation, real arm; detection = any of the cell's samples; "
                            "Oracle (L40/L44) and NLA (L42) read different layers", "pairs": ov}

    # ---- predictions R1-R7
    g = lambda k, m: A["rates"].get(k, {}).get(m)  # noqa: E731
    nla_int = g("nla|member|cell2|L42|real", "int_correct")
    orc_c = con.get("oracle|cell2|L40", {}).get("real_minus_shuffled_roles") or {}
    trk = con.get("nla|cell0|L42", {}).get("start_swap_tracking_roles") or {}
    l4044 = [lens_rows[k]["hit_z"] for k in lens_rows if any(f"|L{L}" in k for L in (40, 42, 44))]
    l60a = [lens_rows[k]["hit_answer"] for k in lens_rows if k.endswith("|L60")]
    orc_rows = [r for r in rows if r["reader"] == "oracle" and r["arm"] == "real"]
    labs = {}
    for r in orc_rows:
        lab = _category(r["judge"]["intermediate"], _truth(tab[r["table"]], r["role"])["intermediate"])
        labs[lab] = labs.get(lab, 0) + 1
    A["predictions"] = {
        "R1 NLA L42 ':' role-correct(z) real < 0.25": {"value": nla_int, "true": None if nla_int is None else nla_int < 0.25},
        "R2 Oracle L40 ':' real - shuffled >= 0.10": {"value": orc_c.get("mean"), "true": None if not orc_c else orc_c["mean"] >= 0.10},
        "R3 NLA start-swap tracking (roles) positive in >= 7/10 tables, assistant newline": {"value": trk.get("pos"), "true": None if not trk else trk["pos"] >= 7},
        "R4 judge-author kappa (intermediate) >= 0.6": {"value": val["intermediate"]["kappa_labels"], "true": val["intermediate"]["kappa_labels"] >= 0.6},
        "R5 lens hit z <= 0.2 at L40-44 for every lens and answer hit >= 0.8 at L60": {"value": {"max_hit_z_L40_44": max(l4044), "min_hit_answer_L60": min(l60a)},
                                                                                     "true": max(l4044) <= 0.2 and min(l60a) >= 0.8},
        "R6 prompt-only picks z >= 0.95": {"value": pacc["intermediate"]["top1_of_12"], "true": pacc["intermediate"]["top1_of_12"] >= 0.95},
        "R7 'conflicting' is the most common Oracle contract (real) intermediate outcome": {"value": labs, "true": max(labs, key=labs.get) == CONF if labs else None},
    }

    # ---- qualitative: per reader, two examples of each outcome (member, real; seeded)
    rng = np.random.default_rng(MS.SEED)
    qual = {}
    for rd in ("nla", "oracle", "oracle_wsb"):
        for name, f in (("role-correct z", lambda r: r["int_correct"]), ("other candidate as intermediate", lambda r: r["int_other_candidate"]),
                        ("z misplaced", lambda r: r["z_misplaced"]), ("conflicting", lambda r: r["int_conflicting"]),
                        ("cannot tell", lambda r: r["int_cannot_tell"]), ("none", lambda r: r["int_none"])):
            pool = [r for r in rows if r["reader"] == rd and r["arm"] == "real" and f(r)]
            pick = [pool[i] for i in rng.permutation(len(pool))[:2]] if pool else []
            qual[f"{rd}|{name}"] = [{"id": r["id"], "truth": _truth(tab[r["table"]], r["role"]), "judge": r["judge"],
                                     "text": texts[r["id"]]} for r in pick]
    (ANA / "scores.json").write_text(json.dumps(A, indent=1, default=float))
    (ANA / "readout_scores.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    (ANA / "qualitative.json").write_text(json.dumps(qual, indent=1, ensure_ascii=False))
    write_manifest("analysis", {"n_scored": len(rows)})
    print(json.dumps({"validation_gate": A["validation_gate"], "judge_status": A["judge_status"], "spend": A["judge_spend_usd"]}, indent=1))


if __name__ == "__main__":
    stage = sys.argv[1]
    {"pilot": stage_pilot, "judge": stage_judge, "lens": stage_lens, "prompt_only": stage_prompt_only,
     "analysis": stage_analysis}[stage]()
