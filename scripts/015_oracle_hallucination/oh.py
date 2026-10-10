"""015: the hallucination test on the Oracle Lens, with cross-scoring (design `docs/design_specs/015_oracle_hallucination.md`).
EXPLORATORY. Reuses 013's judge, version and scoring code (`hallucination.py`, `rewrite.py`) with a reader switch.

  python oh.py judge   --subset smoke|all     # Oracle: A, B, K, answer spans, F, P (+ re-ask), B labels of O/F/P;
                                              # NLA (same eids): answer spans, P* (matched to 013's F), B labels of P*
  python oh.py handset --subset smoke         # hand-check files (G2, G2k, G2a, rewrites); written before any Δ is seen
  python oh.py versions --subset smoke|all    # every version of both readers' texts (§5)
  CUDA_VISIBLE_DEVICES=0 python oh.py score_oracle_ar --subset ...   # Oracle AR (L44) on every version, both readers
  CUDA_VISIBLE_DEVICES=1 python oh.py score_nla_ar    --subset ...   # NLA AR (L42) on every version, both readers
  python oh.py analyze --subset smoke|all

Judge texts (§4): 013's DEFS / STATE_NOTE / PROMPT_A / B / K verbatim; for Oracle texts "EXPLANATION" -> "READOUT" and
the §4 READOUT framing; the new answer-span prompt; P with F's realised edit size and one re-ask if off by > 20%.
"""
import argparse
import ast
import difflib
import hashlib
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in ("scripts/common", "scripts/011_olens_swaps", "scripts/013_nla_hallucination"):
    sys.path.insert(0, str(ROOT / p))
sys.path.insert(0, "/workspace/sources/repos/EasyNLA")
import hallucination as H                                                   # noqa: E402
import rewrite as RW                                                        # noqa: E402

OUT = ROOT / "outputs/015_oracle_hallucination"
RES = ROOT / "results/015_oracle_hallucination"
STATES = H.STATES
SPEND_CAP = 10.0
KINDS = H.KINDS

# --------------------------------------------------------------------------------------------- material
def oracle_rows():
    items = {json.loads(l)["name"]: json.loads(l) for l in open(ROOT / "outputs/012_reader_concordance/stageB/items.jsonl")}
    rows = []
    for l in open(ROOT / "outputs/012_reader_concordance/stageC/readouts_oracle.jsonl"):
        r = json.loads(l)
        if r["layer"] != 44 or r["state"] not in STATES:
            continue
        s = ast.literal_eval(r["samples"]) if isinstance(r["samples"], str) else r["samples"]
        it = items[r["item"]]
        arms = it["arms"] if isinstance(it["arms"], dict) else json.loads(it["arms"])
        for k, t in enumerate(s):
            rows.append({"eid": f"{r['item']}|{r['state']}|{k}", "item": r["item"], "state": r["state"], "sample": k,
                         "prompt": it["prompt"], "continuation": arms[r["state"]]["text"], "src": r["src"], "tgt": r["tgt"],
                         "explanation": t, "reader": "oracle"})
    rows.sort(key=lambda x: (x["item"], STATES.index(x["state"]), x["sample"]))
    assert len(rows) == 522, len(rows)
    return rows


def nla_rows():
    return [{**r, "reader": "nla"} for r in H.explanations()]


def non_english(t):
    return bool(re.search(r"[^\x00-\x7FÀ-ɏ‘-‟–—…°-¿]", t))


# --------------------------------------------------------------------------------------------- §4 judge texts
ORACLE_FRAME = ("READOUT (text written by another system from the model's internal activation at the last prompt token. It is a "
                "list of bullets; each bullet continues the prompt, usually with an answer and then a follow-on. Bullets may be "
                "cut off mid-sentence. Some text may be in Chinese or another language: judge it by its meaning):")
NLA_FRAME = ("EXPLANATION (a description of the model's internal activation at the last prompt token, written by another system. "
             "Some text may be in another language: judge it by its meaning):")


def inputs(r):
    note = H.STATE_NOTE["clean" if r["state"] == "clean" else "edit"]
    frame = ORACLE_FRAME if r["reader"] == "oracle" else NLA_FRAME
    return (f"PROMPT given to the language model (exact text):\n<<<{r['prompt']}>>>\n\n"
            f"The model's own continuation of that prompt:\n<<<{r['continuation']}>>>\n\n{note}\n\n{frame}\n<<<{r['explanation']}>>>")


def oracle_version(t):
    return t.replace("EXPLANATION", "READOUT")


PROMPT_A_O, PROMPT_B_O, PROMPT_K_O = map(oracle_version, (H.PROMPT_A, H.PROMPT_B, H.PROMPT_K))
PROMPT_F_O, PROMPT_P_O = oracle_version(RW.PROMPT_F), oracle_version(RW.PROMPT_P)
PROMPT_ANS = (
    "You find statements of the model's answer in an automatically generated text about a language model's internal state.\n\n"
    "{inputs}\n\n"
    "An ANSWER STATEMENT is a span that states the model's continuation (shown above) as the answer or as the next words, in this "
    "or other wording, in any language. Copy the shortest span that does this: the answer word or phrase, together with any words "
    "in the same clause whose only job is to present it as the answer (for example \"the answer is\", \"the next word is\"). Do not "
    "include follow-on questions or other content. If a bullet or sentence opens with the answer, the span is that opening. A span "
    "that states a DIFFERENT answer is not an answer statement.\n\n"
    "Return JSON only: {{\"answer_spans\": [...]}}. Each span is copied EXACTLY, character for character, from the text. Spans must "
    "not overlap. An empty list is allowed.")
REASK = "\n\nYour previous paraphrase changed {n} words; change about {target}."
# POST HOC (ledger, full-run amendment; report 46 §4b): correction-only rewrite of the Oracle's flagged spans, nothing deleted
PROMPT_FC = (
    "You correct false claims in an automatically generated text about a language model's internal state.\n\n"
    "{inputs}\n\n" + H.DEFS + "\n\n"
    "These spans of the READOUT were judged false (contradicted):\n{spans}\n\n"
    "Rewrite the READOUT so that each listed span is corrected IN PLACE:\n"
    "- correct wrong entities or answers so they agree with the model's continuation;\n"
    "- correct false statements about the world to true ones;\n"
    "- correct misquotes of the prompt to the prompt's exact wording;\n"
    "- if a listed span cannot be corrected in place (for example invented context), leave it unchanged.\n"
    "Change nothing else: do not delete or add any sentence, question, placeholder, bullet or follow-on, even if it is "
    "invented or odd, and keep the wording of everything outside the listed spans. Keep the structure.\n\n"
    "Return JSON only: {{\"rewrite\": \"...\"}}.")


def parse(txt):
    """013's parse_json, but malformed JSON returns None (so the one retry applies) instead of raising."""
    try:
        return H.parse_json(txt)
    except (json.JSONDecodeError, ValueError):
        return None


def cost(rec, hit):
    return 0.0 if hit else float((rec["response"].get("usage") or {}).get("cost") or 0.0)


def content(rec):
    return rec["response"]["choices"][0]["message"]["content"] or ""


def ask_json(model, prompt, cache, tag, key, check=None):
    rec, hit = H.call(model, prompt, cache, tag)
    j = parse(content(rec))
    c = cost(rec, hit)
    ok = j is not None and key in j and (check is None or check(j[key]))
    if not ok:
        rec, hit = H.call(model, prompt + "\n\nReturn valid JSON only.", cache, tag)
        j = parse(content(rec))
        c += cost(rec, hit)
        ok = j is not None and key in j and (check is None or check(j[key]))
    return (j[key] if ok else None), c, rec


def b_labels(r, text, cache, tag):
    sents = H.sentences(text)
    pb = (PROMPT_B_O if r["reader"] == "oracle" else H.PROMPT_B).format(
        inputs=inputs({**r, "explanation": text}), sentences="\n".join(f"{i + 1}. {s}" for i, s in enumerate(sents)))
    labs, c, _ = ask_json(H.MODEL_B, pb, cache, tag, "labels", lambda x: isinstance(x, list) and len(x) == len(sents))
    return labs, sents, c


def answer_spans(r, cache):
    spans, c, _ = ask_json(H.MODEL_B, PROMPT_ANS.format(inputs=inputs(r)), cache, "ANS", "answer_spans",
                           lambda x: isinstance(x, list) and all(isinstance(s, str) for s in x))
    ranges, bad = H.locate(spans or [], r["explanation"]) if spans is not None else ([], 0)
    return {"spans": spans, "ranges": ranges, "bad": bad, "ok": spans is not None and bad == 0}, c


def paraphrase(r, F, cache, tag, prompt_p):
    target = max(RW.words_changed(r["explanation"], F), 1)
    base = prompt_p.format(expl=r["explanation"], n=target)
    P, c, _ = ask_json(H.MODEL_A, base, cache, tag, "rewrite", lambda x: isinstance(x, str) and x.strip())
    n1 = RW.words_changed(r["explanation"], P) if P else None
    reasked = False
    if P and abs(n1 - target) > 0.2 * target:
        P2, c2, _ = ask_json(H.MODEL_A, base + REASK.format(n=n1, target=target), cache, tag + "R", "rewrite",
                             lambda x: isinstance(x, str) and x.strip())
        c += c2
        if P2:
            P, reasked = P2, True
    n = RW.words_changed(r["explanation"], P) if P else None
    return {"P": P, "target": target, "words_changed_P": n, "first_try_words": n1, "reasked": reasked,
            "match_ratio": (n / target) if n is not None else None}, c


def judge_oracle(r, cache):
    spent = 0.0
    inp = inputs(r)
    sents = H.sentences(r["explanation"])
    a, c, _ = ask_json(H.MODEL_A, PROMPT_A_O.format(inputs=inp), cache, "A", "contradicted", lambda x: isinstance(x, list))
    spent += c
    A_raw = None
    if a is not None:
        recA, _ = H.call(H.MODEL_A, PROMPT_A_O.format(inputs=inp), cache, "A")
        A_raw = parse(content(recA))
        if not (A_raw and isinstance(A_raw.get("contradicted"), list)):
            recA, _ = H.call(H.MODEL_A, PROMPT_A_O.format(inputs=inp) + "\n\nReturn valid JSON only.", cache, "A")
            A_raw = parse(content(recA))
    cr, bad_c = H.locate((A_raw or {}).get("contradicted", []), r["explanation"]) if A_raw else ([], 0)
    ur, bad_u = H.locate((A_raw or {}).get("unsupported", []) or [], r["explanation"]) if A_raw else ([], 0)
    okA = A_raw is not None and isinstance(A_raw.get("contradicted"), list) and bad_c + bad_u == 0
    labs, _, c = b_labels(r, r["explanation"], cache, "B")
    spent += c
    cspans = [s for s in ((A_raw or {}).get("contradicted") or []) if isinstance(s, str)]
    kinds, okK = None, None
    kind_ranges = {k: [] for k in KINDS}
    if okA and cspans:
        pk = PROMPT_K_O.format(inputs=inp, spans="\n".join(f"{i + 1}. {s}" for i, s in enumerate(cspans)))
        kinds, c, _ = ask_json(H.MODEL_B, pk, cache, "K", "kinds", lambda x: isinstance(x, list) and len(x) == len(cspans))
        spent += c
        okK = isinstance(kinds, list) and all(k in KINDS for k in kinds)
        if okK:
            for s, k in zip(cspans, kinds):
                kind_ranges[k] += H.locate([s], r["explanation"])[0]
    ans, c = answer_spans(r, cache)
    spent += c
    F, c, _ = ask_json(H.MODEL_A, PROMPT_F_O.format(inputs=inp), cache, "F", "rewrite", lambda x: isinstance(x, str) and x.strip())
    spent += c
    par, c = paraphrase(r, F, cache, "P", PROMPT_P_O) if F else ({"P": None}, 0.0)
    spent += c
    lab = {}
    for name, txt in (("F", F), ("P", par.get("P"))):
        if txt:
            lab[name], _, c = b_labels(r, txt, cache, "BL")
            spent += c
    fc = {"Fc": None, "Pc": None}                                      # POST HOC Fc arm (only with >= 1 flagged span)
    if okA and cspans:
        Fc, c, _ = ask_json(H.MODEL_A, PROMPT_FC.format(inputs=inp, spans="\n".join(f"{i + 1}. {s}" for i, s in enumerate(cspans))),
                            cache, "Fc", "rewrite", lambda x: isinstance(x, str) and x.strip())
        spent += c
        if Fc:
            pc, c = paraphrase(r, Fc, cache, "Pc", PROMPT_P_O)
            spent += c
            fc = {"Fc": Fc, "words_changed_Fc": RW.words_changed(r["explanation"], Fc),
                  "Fc_lines_same": len(Fc.strip().split("\n")) == len(r["explanation"].strip().split("\n")),
                  "Pc": pc["P"], **{f"Pc_{k}": v for k, v in pc.items() if k != "P"}}
    return {**{k: r[k] for k in ("eid", "item", "state", "sample", "reader")}, **fc, "A_raw": A_raw, "A_ok": okA,
            "A_bad_spans": bad_c + bad_u, "contradicted_ranges": cr, "unsupported_ranges": ur, "B_labels": labs,
            "B_ok": isinstance(labs, list), "sentences": sents, "span_kinds": kinds, "K_ok": okK, "kind_ranges": kind_ranges,
            "answer": ans, "F": F, "words_changed_F": RW.words_changed(r["explanation"], F) if F else None, **par,
            "labels_F": lab.get("F"), "labels_P": lab.get("P"), "non_english": non_english(r["explanation"])}, spent


def judge_nla(r, cache, rw013):
    spent = 0.0
    ans, c = answer_spans(r, cache)
    spent += c
    F = (rw013.get(r["eid"]) or {}).get("F")
    par, c = paraphrase(r, F, cache, "Pstar", RW.PROMPT_P) if F else ({"P": None}, 0.0)
    spent += c
    lab = None
    if par.get("P"):
        lab, _, c = b_labels(r, par["P"], cache, "BL")
        spent += c
    return {**{k: r[k] for k in ("eid", "item", "state", "sample", "reader")}, "answer": ans, "F_013": F,
            "Pstar": par.get("P"), **{f"Pstar_{k}": v for k, v in par.items() if k != "P"}, "labels_Pstar": lab,
            "non_english": non_english(r["explanation"])}, spent


def phase_judge(which):
    cache = OUT / "judge/cache"
    cache.mkdir(parents=True, exist_ok=True)
    orows = H.subset(oracle_rows(), which)
    nrows = H.subset(nla_rows(), which)
    assert [r["eid"] for r in orows] == [r["eid"] for r in nrows], "smoke eids must be paired across readers"
    rw013 = {json.loads(l)["eid"]: json.loads(l) for l in open(H.OUT / "rewrite/rewrites_all.jsonl")}
    jobs = [("o", r) for r in orows] + [("n", r) for r in nrows]
    res, spent, t0 = [], 0.0, time.time()
    with ThreadPoolExecutor(16) as ex:                                 # §9: abort if the projected spend passes the cap
        for j in range(0, len(jobs), 64):
            ch = jobs[j:j + 64]
            res += list(ex.map(lambda kr: judge_oracle(kr[1], cache) if kr[0] == "o" else judge_nla(kr[1], cache, rw013), ch))
            spent = sum(c for _, c in res)
            proj = spent / len(res) * len(jobs)                    # Oracle rows first and dearer, so this over-projects
            print(f"[judge] {len(res)}/{len(jobs)} spent ${spent:.3f} projected ${proj:.2f} ({time.time() - t0:.0f}s)", flush=True)
            if proj > SPEND_CAP:
                raise SystemExit(f"ABORT: projected spend ${proj:.2f} > cap ${SPEND_CAP}")
    oo, nn = res[:len(orows)], res[len(orows):]
    (OUT / "judge").mkdir(parents=True, exist_ok=True)
    for name, xs in (("oracle", oo), ("nla", nn)):
        with open(OUT / f"judge/judged_{name}_{which}.jsonl", "w") as f:
            for x, _ in xs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    O = [x for x, _ in oo]
    meta = {"which": which, "n_oracle": len(oo), "n_nla": len(nn), "new_spend_usd": spent, "models": {"A": H.MODEL_A, "B": H.MODEL_B},
            "texts": {"ORACLE_FRAME": ORACLE_FRAME, "NLA_FRAME": NLA_FRAME, "PROMPT_ANS": PROMPT_ANS, "REASK": REASK,
                      "PROMPT_A_O": PROMPT_A_O, "PROMPT_B_O": PROMPT_B_O, "PROMPT_K_O": PROMPT_K_O, "PROMPT_F_O": PROMPT_F_O,
                      "PROMPT_P_O": PROMPT_P_O, "DEFS": H.DEFS, "STATE_NOTE": H.STATE_NOTE},
            "G1_A_ok_rate_oracle": float(np.mean([x["A_ok"] for x in O])),
            "B_ok_rate_oracle": float(np.mean([x["B_ok"] for x in O])),
            "K_ok_rate_oracle": (lambda t: float(np.mean(t)) if t else None)([x["K_ok"] for x in O if x["K_ok"] is not None]),
            "ANS_ok_rate": {"oracle": float(np.mean([x["answer"]["ok"] for x in O])),
                            "nla": float(np.mean([x["answer"]["ok"] for x, _ in nn]))},
            "kind_counts_oracle": {k: sum(len(x["kind_ranges"][k]) for x in O) for k in KINDS},
            "F_ok": sum(bool(x["F"]) for x in O), "P_ok": sum(bool(x["P"]) for x in O),
            "P_match_ratio_median_oracle": float(np.median([x["match_ratio"] for x in O if x.get("match_ratio") is not None])),
            "Pstar_match_ratio_median_nla": float(np.median([x["Pstar_match_ratio"] for x, _ in nn if x.get("Pstar_match_ratio") is not None]))}
    (OUT / f"judge/judge_meta_{which}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in meta.items() if k != "texts"}, indent=1), flush=True)
    if spent > SPEND_CAP:
        print(f"WARNING: spend ${spent:.2f} above the cap ${SPEND_CAP}", flush=True)


def phase_handset(which):
    O = {r["eid"]: r for r in oracle_rows()}
    N = {r["eid"]: r for r in nla_rows()}
    if which == "extra10":                     # §8 G2's 10 more: seeded, outside the smoke, from the full judge run
        smoke = {r["eid"] for r in H.subset(oracle_rows(), "smoke")}
        allo = [json.loads(l) for l in open(OUT / "judge/judged_oracle_all.jsonl")]
        jo = sorted(random.Random(15).sample([x for x in allo if x["eid"] not in smoke], 10), key=lambda x: x["eid"])
        jn = []
    else:
        jo = [json.loads(l) for l in open(OUT / f"judge/judged_oracle_{which}.jsonl")]
        jn = [json.loads(l) for l in open(OUT / f"judge/judged_nla_{which}.jsonl")]
    lines = ["# 015 smoke hand checks (G2, G2k, G2a on Oracle readouts; G2a on 10 NLA explanations; F/P)\n"]
    for x in jo:
        r = O[x["eid"]]
        e = r["explanation"]
        cs = (x["A_raw"] or {}).get("contradicted") or []
        lines.append(f"## {x['eid']} (src {r['src']} -> tgt {r['tgt']})  non-English: {x['non_english']}\nPROMPT: {r['prompt']}\n"
                     f"CONTINUATION: {r['continuation']!r}\n\nREADOUT:\n{e}\n\nA contradicted [kind]:\n"
                     + "\n".join(f"  C{i} [{k}]: {s!r}" for i, (s, k) in enumerate(zip(cs, x.get("span_kinds") or ["?"] * 99)))
                     + "\nA unsupported:\n" + "\n".join(f"  U{i}: {e[a:b]!r}" for i, (a, b) in enumerate(x["unsupported_ranges"]))
                     + "\nANSWER spans:\n" + "\n".join(f"  S{i}: {s!r}" for i, s in enumerate(x["answer"]["spans"] or []))
                     + f"\n\nF (words changed {x['words_changed_F']}):\n{x['F']}\n\nP (target {x.get('target')}, changed "
                     f"{x.get('words_changed_P')}, re-asked {x.get('reasked')}):\n{x['P']}\n"
                     + (f"\nFc (POST HOC; words changed {x.get('words_changed_Fc')}, same line count {x.get('Fc_lines_same')}):\n{x['Fc']}\n"
                        f"\nPc (target {x.get('Pc_target')}, changed {x.get('Pc_words_changed_P')}, re-asked {x.get('Pc_reasked')}):\n{x['Pc']}\n"
                        if x.get("Fc") else ""))
    if jn:
        lines.append("\n# NLA answer spans (first 10 smoke explanations)\n")
    for x in jn[:10]:
        r = N[x["eid"]]
        lines.append(f"## {x['eid']}\nCONTINUATION: {r['continuation']!r}\nEXPLANATION:\n{r['explanation']}\nANSWER spans:\n"
                     + "\n".join(f"  S{i}: {s!r}" for i, s in enumerate(x["answer"]["spans"] or [])) + "\n")
    (RES / "qualitative").mkdir(parents=True, exist_ok=True)
    p = RES / f"qualitative/handcheck_{which}.md"
    p.write_text("\n".join(lines))
    print("wrote", p)


# --------------------------------------------------------------------------------------------- §5 versions
# 013's build_versions makes O, O_tidy, C1, C2, R1, R2, K1, RS1 and the invented_context (i) / misquote (ii) kinds with
# its own salts; the two new kinds use the same construction with new salts (ledger, smoke entry).
NEW_KINDS = {"wrong_entity": ("we", 61, 81), "false_fact": ("ff", 71, 91)}
KSHORT = {"invented_context": "i", "misquote": "ii", "wrong_entity": "we", "false_fact": "ff"}


def strip_delims(t):
    """Ledger (smoke, before any score): 8 of the 20 Oracle P texts echo PROMPT_P's <<< >>> delimiters; strip them."""
    if not t:
        return t
    s = t.strip()
    s = s[3:] if s.startswith("<<<") else s
    s = s[:-3] if s.endswith(">>>") else s
    return s.strip()


def f_composition(o, f):
    """Word-level alignment of O and F (PI change 5): words in deleted runs vs substituted runs vs inserted runs."""
    d = {"deleted": 0, "replaced_O": 0, "replaced_F": 0, "inserted": 0, "words_O": len(o.split())}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=o.split(), b=f.split(), autojunk=False).get_opcodes():
        if tag == "delete":
            d["deleted"] += i2 - i1
        elif tag == "replace":
            d["replaced_O"] += i2 - i1
            d["replaced_F"] += j2 - j1
        elif tag == "insert":
            d["inserted"] += j2 - j1
    return d


def oracle_lines(t):
    """Bullet contents for the Oracle AR: '- ' stripped (r1 / CW-20); a bullet emptied by a deletion is dropped."""
    out = []
    for line in t.split("\n"):
        s = line.strip()
        s = s[2:] if s.startswith("- ") else s[1:].strip() if s.startswith("-") else s
        if re.search(r"\w", s):
            out.append(s)
    return out


def oracle_hosts(e):
    """Char ranges of each bullet's content (after '- '): the PA hosts for Oracle texts."""
    out, pos = [], 0
    for line in e.split("\n"):
        a, b = pos, pos + len(line)
        pos = b + 1
        lead = len(line) - len(line.lstrip())
        a += lead + (2 if line.lstrip().startswith("- ") else 0)
        b -= len(line) - len(line.rstrip())
        if a < b:
            out.append((a, b))
    return out


def seed_of(eid):
    return int(hashlib.sha256(eid.encode()).hexdigest()[:8], 16)


def add_new_kinds(v, info, e, kr, ntok, eid):
    us, ss, s0 = H.units(e), H.sentence_units(e), seed_of(eid)
    for kname, (short, sc, sse) in NEW_KINDS.items():
        rr = (kr or {}).get(kname) or []
        info[f"has_{short}"] = len(rr) > 0
        if not rr:
            continue
        v[f"C1{short}"] = H.delete(e, [tuple(x) for x in rr])
        info[f"tok_C1{short}"] = ntok(v[f"C1{short}"])
        tg = info["tok_O"] - info[f"tok_C1{short}"]
        v[f"R{short}"] = H.matched(e, us, tg, ntok, s0, sc)
        v[f"RS{short}"] = H.matched(e, ss, tg, ntok, s0, sse)


def pa_versions(e, ar, hosts, tok, eid):
    """PA: per answer span (l tokens, longest first) delete the first l tokens of a different answer-free host with >= l
    tokens; hosts drawn seeded, without replacement; 5 seeds. Not computable if any seed runs out of hosts."""
    ntok = lambda t: len(tok(t, add_special_tokens=False).input_ids)                         # noqa: E731
    free = [h for h in hosts if not H.overlaps(h, ar)]
    ls = sorted((ntok(e[a:b]) for a, b in ar), reverse=True)
    out, s0 = [], seed_of(eid)
    for s in range(H.N_SEEDS):
        rng = random.Random(s0 + 100 * s + 111)
        pool = free[:]
        rng.shuffle(pool)
        cuts = []
        for l in ls:
            h = next((h for h in pool if ntok(e[h[0]:h[1]]) >= l), None)
            if h is None:
                return [], len(free)
            pool.remove(h)
            off = tok(e[h[0]:h[1]], add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            cuts.append((h[0], h[0] + off[l - 1][1]))
        out.append(H.delete(e, cuts))
    return out, len(free)


def add_answer_arm(v, info, e, ans, ntok, tok, eid, hosts):
    ar = [tuple(a) for a in (ans or {}).get("ranges") or []] if (ans or {}).get("ok") else []
    info.update({"ans_ok": bool((ans or {}).get("ok")), "n_ans": len(ar), "ans_eligible": False, "PA_computable": False})
    if not ar:
        return
    v["CA"] = H.delete(e, ar)
    info["tok_CA"] = ntok(v["CA"])
    info["target_RA"] = info["tok_O"] - info["tok_CA"]
    v["RA"] = H.matched(e, H.units(e), info["target_RA"], ntok, seed_of(eid), 101)
    pa, nfree = pa_versions(e, ar, hosts, tok, eid)
    info["n_answer_free_hosts"] = nfree
    if pa:
        v["PA"] = pa
        info["PA_computable"] = True
        info["tok_PA_deleted_mean"] = float(np.mean([info["tok_O"] - ntok(p) for p in pa]))
    info["ans_eligible"] = info["tok_CA"] >= H.MIN_TOK


def phase_versions(which):
    from transformers import AutoTokenizer
    import readers as RD
    import registry as R
    qtok = R.make_tokenizer()
    avtok = AutoTokenizer.from_pretrained(Path(RD.nla_rl_paths()[0]) / "av_base")
    assert qtok.is_fast and avtok.is_fast
    q_n = lambda t: len(qtok(t, add_special_tokens=False).input_ids)                          # noqa: E731
    av_n = lambda t: len(avtok.encode(t, add_special_tokens=False))                           # noqa: E731
    O = {r["eid"]: r for r in oracle_rows()}
    N = {r["eid"]: r for r in nla_rows()}
    jo = [json.loads(l) for l in open(OUT / f"judge/judged_oracle_{which}.jsonl")]
    jn = [json.loads(l) for l in open(OUT / f"judge/judged_nla_{which}.jsonl")]
    want = {x["eid"] for x in jn}
    j13 = {x["eid"]: x for x in map(json.loads, open(H.OUT / "judge/judged_all.jsonl")) if x["eid"] in want}
    s13 = {x["eid"]: x for x in map(json.loads, open(H.OUT / "scored_all.jsonl")) if x["eid"] in want}
    rw13 = {x["eid"]: x for x in map(json.loads, open(H.OUT / "rewrite/rewrites_all.jsonl")) if x["eid"] in want}
    out = []
    for x in jo:
        r, e = O[x["eid"]], O[x["eid"]]["explanation"]
        v, info = H.build_versions(e, [tuple(a) for a in x["contradicted_ranges"]], [tuple(a) for a in x["unsupported_ranges"]],
                                   q_n, x["eid"], x["kind_ranges"] if x.get("K_ok") else None)
        add_new_kinds(v, info, e, x["kind_ranges"] if x.get("K_ok") else None, q_n, x["eid"])
        add_answer_arm(v, info, e, x["answer"], q_n, qtok, x["eid"], oracle_hosts(e))
        P = strip_delims(x.get("P"))
        if x.get("F"):
            v["F"] = x["F"]
            info["F_composition"] = f_composition(e, x["F"])
            info["words_changed_F"] = RW.words_changed(e, x["F"])
        if P:
            v["P"] = P
            info["P_had_delimiters"] = P != x["P"]
            info["words_changed_P"] = RW.words_changed(e, P)
            info["P_match_ratio"] = info["words_changed_P"] / max(info["words_changed_F"], 1)
        Pc = strip_delims(x.get("Pc"))                                 # POST HOC Fc arm
        if x.get("Fc") and Pc:
            v["Fc"], v["Pc"] = x["Fc"], Pc
            info["words_changed_Fc"] = RW.words_changed(e, x["Fc"])
            info["words_changed_Pc"] = RW.words_changed(e, Pc)
            info["Pc_match_ratio"] = info["words_changed_Pc"] / max(info["words_changed_Fc"], 1)
            info["Pc_had_delimiters"] = Pc != x["Pc"]
            info["Fc_lines_same"] = x.get("Fc_lines_same")
        info["A_ok"], info["non_english_O"] = x["A_ok"], non_english(e)
        info["tok_F"] = q_n(v["F"]) if "F" in v else None
        out.append({"eid": x["eid"], "reader": "oracle", "item": r["item"], "state": r["state"], "sample": r["sample"],
                    "info": info, "texts": {k: (t if isinstance(t, list) else [t]) for k, t in v.items()}})
    for y in jn:
        r, e, jj = N[y["eid"]], N[y["eid"]]["explanation"], j13[y["eid"]]
        v, info = H.build_versions(e, [tuple(a) for a in jj["contradicted_ranges"]], [tuple(a) for a in jj["unsupported_ranges"]],
                                   av_n, y["eid"], jj.get("kind_ranges") if jj.get("K_ok") else None)
        same = {k: k in v and (v[k] if isinstance(v[k], list) else [v[k]]) == t for k, t in s13[y["eid"]]["texts"].items()}
        info["versions_same_as_013"] = all(same.values())
        info["versions_differing_from_013"] = sorted(k for k, ok in same.items() if not ok)
        add_new_kinds(v, info, e, jj.get("kind_ranges") if jj.get("K_ok") else None, av_n, y["eid"])
        add_answer_arm(v, info, e, y["answer"], av_n, avtok, y["eid"], H.sentence_units(e))
        F, P13, Ps = rw13[y["eid"]].get("F"), rw13[y["eid"]].get("P"), strip_delims(y.get("Pstar"))
        if F:
            v["F"] = F
            info["F_composition"] = f_composition(e, F)
            info["words_changed_F"] = RW.words_changed(e, F)
        if P13:
            v["P"] = P13
            info["words_changed_P"] = RW.words_changed(e, P13)
        if Ps:
            v["Pstar"] = Ps
            info["Pstar_had_delimiters"] = Ps != y["Pstar"]
            info["words_changed_Pstar"] = RW.words_changed(e, Ps)
            info["Pstar_match_ratio"] = info["words_changed_Pstar"] / max(info.get("words_changed_F") or 0, 1)
        info["A_ok"], info["non_english_O"] = jj["A_ok"], non_english(e)
        out.append({"eid": y["eid"], "reader": "nla", "item": r["item"], "state": r["state"], "sample": r["sample"],
                    "info": info, "texts": {k: (t if isinstance(t, list) else [t]) for k, t in v.items()}})
    (OUT / "versions").mkdir(parents=True, exist_ok=True)
    with open(OUT / f"versions/versions_{which}.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    n_txt = sum(len(t) for o in out for t in o["texts"].values())
    summ = {"n_records": len(out), "n_texts": n_txt,
            "nla_versions_same_as_013": sum(o["info"].get("versions_same_as_013", False) for o in out if o["reader"] == "nla"),
            "PA_computable": {rd: sum(o["info"]["PA_computable"] for o in out if o["reader"] == rd) for rd in ("oracle", "nla")},
            "ans_eligible": {rd: sum(o["info"]["ans_eligible"] for o in out if o["reader"] == rd) for rd in ("oracle", "nla")}}
    print(json.dumps(summ, indent=1), flush=True)


# --------------------------------------------------------------------------------------------- §3 / §6 scoring
def entries(which):
    """Unique (reader, text) jobs and the (eid, reader, version, seed) entries that point at them."""
    V = [json.loads(l) for l in open(OUT / f"versions/versions_{which}.jsonl")]
    uniq, idx, ent = [], {}, []
    for rec in V:
        for name, texts in rec["texts"].items():
            for si, t in enumerate(texts):
                k = (rec["reader"], t)
                if k not in idx:
                    idx[k] = len(uniq)
                    uniq.append(k)
                ent.append({"eid": rec["eid"], "reader": rec["reader"], "item": rec["item"], "state": rec["state"],
                            "version": name, "seed": si, "u": idx[k]})
    return uniq, ent


def jbasis(L):
    Q = torch.load(ROOT / f"outputs/007_jsubspace/bases/J_raw_L{L}.pt", weights_only=False)["Q"][:, :1024].double()
    return torch.linalg.qr(Q)[0]


def phase_score_oracle_ar(which):
    import ar_vectors as AV
    import registry as R
    sys.path.insert(0, str(HERE))
    import r1 as R1
    (OUT / "score").mkdir(parents=True, exist_ok=True)
    log = AV.Logger(OUT / f"score/oracle_ar_{which}_log.txt")
    tok = R.make_tokenizer()
    model = R.load_model()
    pt, head, heads = AV.attach_ar(model, log)
    uniq, ent = entries(which)
    g = R1.gold(44)
    w = AV.whitener(44)
    tw = lambda x: (x.double() - w["mu"]) @ w["W"].T                                            # noqa: E731
    unit = lambda x: x / x.norm()                                                              # noqa: E731
    QJ = jbasis(44)
    mu_raw = torch.stack([unit(v) for v in g.values()]).mean(0)                                # 013's form: unit-normalised raw

    def share(p):
        c = unit(p) - mu_raw
        return float((c @ QJ).pow(2).sum() / c.pow(2).sum())

    pieces = []
    for reader, t in uniq:
        pl, ns = R1.to_pieces(oracle_lines(t) if reader == "oracle" else H.sentences(t), tok)
        pieces.append((pl, ns))
    log(f"{len(uniq)} unique texts, {sum(len(p) for p, _ in pieces)} pieces, {len(ent)} entries")
    preds, t0, CH = [None] * len(uniq), time.time(), 2048
    flat = [(u, p) for u, (pl, _) in enumerate(pieces) for p in pl]
    acc = {}
    for j in range(0, len(flat), CH):
        ch = flat[j:j + CH]
        y = AV.ar_forward(pt, head, heads, [p for _, p in ch], 44)["ln_pre"].double()
        for (u, _), yy in zip(ch, y):
            s_, n_ = acc.get(u, (0.0, 0))
            acc[u] = (s_ + yy, n_ + 1)
        if (j // CH) % 10 == 0 or j + CH >= len(flat):
            log(f"pieces {min(j + CH, len(flat))}/{len(flat)} ({time.time() - t0:.0f}s)")
    for u, (s_, n_) in acc.items():
        preds[u] = s_ / n_
    gold_share = {f"{k[0]}|{k[1]}": share(v) for k, v in g.items()}
    with open(OUT / f"score/oracle_ar_{which}.jsonl", "w") as f:
        for e_ in ent:
            p = preds[e_["u"]]
            rec = {k: e_[k] for k in ("eid", "reader", "version", "seed")}
            rec.update({"n_pieces": len(pieces[e_["u"]][0]), "n_split": pieces[e_["u"]][1], "degenerate": p is None})
            if p is not None:
                P, T = tw(p), tw(g[(e_["item"], e_["state"])])
                rec.update({"cos": float(P @ T / (P.norm() * T.norm())), "pt": float(P @ T), "pp": float(P @ P),
                            "tt": float(T @ T), "share_J": share(p)})
            f.write(json.dumps(rec) + "\n")
    (OUT / f"score/oracle_ar_meta_{which}.json").write_text(json.dumps(
        {"n_unique_texts": len(uniq), "n_entries": len(ent), "n_pieces": len(flat), "n_texts_split": sum(ns > 0 for _, ns in pieces),
         "gold_share_J": gold_share, "wall_s": time.time() - t0}, indent=1))
    log("saved")


def phase_score_nla_ar(which, shard=None):
    """shard 'i/n': unique texts with index % n == i (one text per forward, so sharding cannot change a score)."""
    from nla.schema import normalize_activation
    (OUT / "score").mkdir(parents=True, exist_ok=True)
    recon, scale = H.critic_setup()
    uniq, ent = entries(which)
    tag = which
    if shard:
        i, n = map(int, shard.split("/"))
        keep = {u for u in range(len(uniq)) if u % n == i}
        uniq = [t if u in keep else None for u, t in enumerate(uniq)]
        ent = [e_ for e_ in ent if e_["u"] in keep]
        tag = f"{which}.shard{i}of{n}"
    gold = H.gold_vectors()
    Gn = normalize_activation(torch.stack(list(gold.values())), scale)
    mu = Gn.mean(0)
    QJ = jbasis(42).float()
    share = lambda v: float(((v - mu) @ QJ).pow(2).sum() / (v - mu).pow(2).sum())             # noqa: E731 (013's)
    preds, t0, done = [], time.time(), 0
    for jt in uniq:
        preds.append(recon(jt[1]) if jt is not None else None)
        done += jt is not None
        if jt is not None and done % 500 == 0:
            print(f"[nla_ar] {done}/{sum(x is not None for x in uniq)} ({time.time() - t0:.0f}s)", flush=True)
    with open(OUT / f"score/nla_ar_{tag}.jsonl", "w") as f:
        for e_ in ent:
            p = preds[e_["u"]]
            gg = normalize_activation(gold[(e_["item"], e_["state"])][None], scale)[0]
            rec = {k: e_[k] for k in ("eid", "reader", "version", "seed")}
            rec.update({"mse": float(((p - gg) ** 2).mean()), "cos": float(p @ gg / (p.norm() * gg.norm())), "share_J": share(p),
                        "degenerate": False})
            f.write(json.dumps(rec) + "\n")
    (OUT / f"score/nla_ar_meta_{tag}.json").write_text(json.dumps(
        {"n_unique_texts": sum(x is not None for x in uniq), "n_entries": len(ent), "scale": scale, "shard": shard,
         "gold_share_J": {f"{k[0]}|{k[1]}": share(normalize_activation(v[None], scale)[0]) for k, v in gold.items()},
         "wall_s": time.time() - t0}, indent=1))
    print("[nla_ar] saved", flush=True)


# --------------------------------------------------------------------------------------------- §7 analysis
def baselines():
    sys.path.insert(0, str(HERE))
    import ar_vectors as AV
    import r1 as R1
    g44, g42 = R1.gold(44), R1.gold(42)
    w = AV.whitener(44)
    keys = sorted(g44)
    U = torch.stack([((g44[k] - w["mu"]) @ w["W"].T) for k in keys])
    U = U / U.norm(dim=1, keepdim=True)
    V = torch.stack([g42[k] for k in keys])
    V = V / V.norm(dim=1, keepdim=True)
    return {"oracle_ar": float(((U - U.mean(0)) ** 2).sum(1).mean()), "nla_ar": float(((V - V.mean(0)) ** 2).sum(1).mean())}


def phase_analyze(which):
    B = baselines()
    r1j = json.loads((RES / "r1.json").read_text())["B"]
    assert all(abs(B[k] - r1j[k]) < 1e-9 for k in B), (B, r1j)
    V = {(x["eid"], x["reader"]): x for x in map(json.loads, open(OUT / f"versions/versions_{which}.jsonl"))}
    SC, meta = {}, {}
    for scorer in ("oracle_ar", "nla_ar"):
        files = [OUT / f"score/{scorer}_{which}.jsonl"]
        if not files[0].exists():                                      # NLA AR shards (one text per forward)
            files = sorted((OUT / "score").glob(f"{scorer}_{which}.shard*.jsonl"))
        for fp in files:
            for x in map(json.loads, open(fp)):
                SC.setdefault((scorer, x["eid"], x["reader"]), {}).setdefault(x["version"], []).append(x)
        ms = [json.loads((OUT / f"score/{scorer}_meta_{fp.name[len(scorer) + 1:-6]}.json").read_text()) for fp in files]
        meta[scorer] = {**ms[0], "n_unique_texts": sum(m["n_unique_texts"] for m in ms), "n_entries": sum(m["n_entries"] for m in ms),
                        "wall_s": max(m["wall_s"] for m in ms), "n_files": len(ms)}
    V_ent = sum(len(t) for x in V.values() for t in x["texts"].values())
    for scorer in ("oracle_ar", "nla_ar"):
        n_sc = sum(len(xs) for k, d_ in SC.items() if k[0] == scorer for xs in d_.values())
        assert n_sc == V_ent, (scorer, n_sc, V_ent)                    # every version entry scored exactly once
    eids = sorted({k[0] for k in V})
    item_of = {e: V[(e, "oracle")]["item"] for e in eids}
    eng = {e for e in eids if not V[(e, "oracle")]["info"]["non_english_O"]}

    def fve(s, e, rd, name):
        xs = SC.get((s, e, rd), {}).get(name)
        if not xs or any(x["degenerate"] for x in xs):
            return None
        return float(np.mean([100 * (1 - (2 - 2 * x["cos"]) / B[s]) for x in xs]))

    def cosm(s, e, rd, name):
        xs = SC.get((s, e, rd), {}).get(name)
        return float(np.mean([x["cos"] for x in xs])) if xs and not any(x["degenerate"] for x in xs) else None

    def band(m, ci):
        """013 §6 bands, in the scorer's own points, worded neutrally (they are applied to every contrast)."""
        return ("H-signal (>= +2, CI above 0)" if m >= 2 and ci[0] > 0 else "H-padding (<= -2, CI below 0)" if m <= -2 and ci[1] < 0
                else "no difference beyond ±2" if ci[0] >= -2 and ci[1] <= 2 else "unresolved")

    def stat(s, rd, a, b, rows):
        d = [(fve(s, e, rd, a), fve(s, e, rd, b), e) for e in rows]
        d = [(x - y, e) for x, y, e in d if x is not None and y is not None]
        if not d:
            return None
        vals, cl = [x for x, _ in d], [item_of[e] for _, e in d]
        m, ci = float(np.mean(vals)), H.boot(vals, cl)
        return {"n": len(vals), "mean": m, "ci95": ci, "median": float(np.median(vals)), "band": band(m, ci),
                "cos_mean": {a: float(np.mean([cosm(s, e, rd, a) for _, e in d])), b: float(np.mean([cosm(s, e, rd, b) for _, e in d]))}}

    def per_share(s, rd, a, b, rows, tokkey):
        """Ratio of means: sum of deltas / sum of deleted token shares (ledger, smoke entry); item-cluster bootstrap."""
        d, cl = [], []
        for e in rows:
            x, y, inf = fve(s, e, rd, a), fve(s, e, rd, b), V[(e, rd)]["info"]
            if x is None or y is None or inf.get(tokkey) is None:
                continue
            d.append((x - y, (inf["tok_O"] - inf[tokkey]) / inf["tok_O"]))
            cl.append(item_of[e])
        if not d:
            return None
        f_ = lambda xs: sum(u for u, _ in xs) / sum(v for _, v in xs)                           # noqa: E731
        return {"n": len(d), "points_per_unit_share": f_(d), "ci95": H.boot(d, cl, f=f_),
                "deleted_share_mean": float(np.mean([v for _, v in d]))}

    def elig_primary(rd):
        return [e for e in eids if (inf := V[(e, rd)]["info"])["A_ok"] and not inf["nothing_to_clean"] and not inf["cleaned_away"]]

    def elig_ans(rd, pa=False):
        return [e for e in eids if (inf := V[(e, rd)]["info"])["ans_eligible"] and (inf["PA_computable"] or not pa)]

    res = {"label": "EXPLORATORY (015 smoke; spec §5-§8)" if which == "smoke" else "EXPLORATORY (015; spec §5-§8)", "which": which,
           "B": B, "n_eids": len(eids), "n_english_only_eids": len(eng), "cells": {}}
    for rd in ("oracle", "nla"):
        for s in ("oracle_ar", "nla_ar"):
            ep, ea, eap = elig_primary(rd), elig_ans(rd), elig_ans(rd, True)
            c = {"n_eligible_primary": len(ep), "n_eligible_answer": len(ea), "n_eligible_PA": len(eap),
                 "primary_R1_minus_C1": stat(s, rd, "R1", "C1", ep),
                 "primary_per_share": per_share(s, rd, "R1", "C1", ep, "tok_C1"),
                 "primary_english_only": stat(s, rd, "R1", "C1", [e for e in ep if e in eng]),
                 "dA_RA_minus_CA": stat(s, rd, "RA", "CA", ea),
                 "dA_per_share": per_share(s, rd, "RA", "CA", ea, "tok_CA"),
                 "dA_english_only": stat(s, rd, "RA", "CA", [e for e in ea if e in eng]),
                 "dAP_PA_minus_CA": stat(s, rd, "PA", "CA", eap),
                 "dAP_per_share": per_share(s, rd, "PA", "CA", eap, "tok_CA"),
                 "dAP_english_only": stat(s, rd, "PA", "CA", [e for e in eap if e in eng]),
                 "R2_minus_C2": stat(s, rd, "R2", "C2", [e for e in ep if V[(e, rd)]["info"]["tok_C2"] >= H.MIN_TOK]),
                 "K1_minus_C1": stat(s, rd, "K1", "C1", [e for e in ep if V[(e, rd)]["info"]["K1_computable"]]),
                 "RS1_minus_C1": stat(s, rd, "RS1", "C1", ep),
                 "O_minus_C1": stat(s, rd, "O", "C1", ep), "O_minus_CA": stat(s, rd, "O", "CA", ea),
                 "O_minus_Otidy": stat(s, rd, "O", "O_tidy", eids),
                 "F_minus_P": stat(s, rd, "F", "P", eids), "F_minus_O": stat(s, rd, "F", "O", eids),
                 "P_minus_O": stat(s, rd, "P", "O", eids),
                 "per_state_primary": {st: stat(s, rd, "R1", "C1", [e for e in ep if V[(e, rd)]["state"] == st]) for st in STATES},
                 "fve_mean": {}, "cos_mean": {}, "share_J_mean": {}}
            c["dA_on_PA_subset"] = stat(s, rd, "RA", "CA", eap)            # ledger full-run amendment (4c)
            if rd == "oracle":                                                 # POST HOC Fc arm (4b)
                efc = [e for e in eids if "Fc" in V[(e, rd)]["texts"] and "Pc" in V[(e, rd)]["texts"]]
                c["Fc_minus_Pc"] = stat(s, rd, "Fc", "Pc", efc)
                c["Fc_minus_Pc_within20pct"] = stat(s, rd, "Fc", "Pc", [e for e in efc if abs(V[(e, rd)]["info"]["Pc_match_ratio"] - 1) <= 0.2])
                c["Fc_minus_O"] = stat(s, rd, "Fc", "O", efc)
                c["Pc_minus_O"] = stat(s, rd, "Pc", "O", efc)
                c["F_minus_P_on_Fc_set"] = stat(s, rd, "F", "P", efc)
            mk = "P_match_ratio" if rd == "oracle" else "Pstar_match_ratio"                    # ledger: matched-size subset
            matched20 = [e for e in eids if abs(V[(e, rd)]["info"].get(mk, 9.0) - 1) <= 0.2]
            c["F_minus_" + ("P" if rd == "oracle" else "Pstar") + "_within20pct"] = stat(s, rd, "F", "P" if rd == "oracle" else "Pstar", matched20)
            if rd == "nla":
                c["F_minus_Pstar"] = stat(s, rd, "F", "Pstar", eids)
                c["Pstar_minus_O"] = stat(s, rd, "Pstar", "O", eids)
            for kname, short in KSHORT.items():
                ek = [e for e in eids if V[(e, rd)]["info"]["A_ok"] and V[(e, rd)]["info"].get(f"has_{short}")
                      and V[(e, rd)]["info"].get(f"tok_C1{short}", 0) >= H.MIN_TOK]
                c[f"kind_{kname}"] = {"n": len(ek), "delta_clause": stat(s, rd, f"R{short}", f"C1{short}", ek),
                                      "delta_sentence": stat(s, rd, f"RS{short}", f"C1{short}", ek),
                                      "O_minus_C1k": stat(s, rd, "O", f"C1{short}", ek),
                                      "token_share_deleted_mean": float(np.mean([(V[(e, rd)]["info"]["tok_O"] - V[(e, rd)]["info"][f"tok_C1{short}"])
                                                                                 / V[(e, rd)]["info"]["tok_O"] for e in ek])) if ek else None}
            names = sorted({n for e in eids for n in SC.get((s, e, rd), {})})
            for n in names:
                fv = [fve(s, e, rd, n) for e in eids]
                fv = [x for x in fv if x is not None]
                c["fve_mean"][n] = float(np.mean(fv)) if fv else None
                cs = [cosm(s, e, rd, n) for e in eids]
                c["cos_mean"][n] = float(np.mean([x for x in cs if x is not None])) if fv else None
                sh = [x["share_J"] for e in eids for x in SC.get((s, e, rd), {}).get(n, []) if not x["degenerate"]]
                c["share_J_mean"][n] = float(np.mean(sh)) if sh else None
            c["share_J_mean"]["gold"] = float(np.mean([meta[s]["gold_share_J"][f"{item_of[e]}|{V[(e, rd)]['state']}"] for e in eids]))
            c["n_degenerate_entries"] = sum(x["degenerate"] for e in eids for xs in SC.get((s, e, rd), {}).values() for x in xs)
            res["cells"][f"{rd}_texts|{s}"] = c

    # pre-named reader contrasts (spec §7): NLA - Oracle, paired on (item, state), averaged over samples
    r1c = json.loads((RES / "r1.json").read_text())["cells"]
    R1RANGE = {(rd, s): r1c[f"{rd}_texts|{s}"]["matched_minus_shuffled"]["mean"] for rd in ("oracle", "nla") for s in ("oracle_ar", "nla_ar")}

    def reader_diff(s, a, b, elig, tokkey, english=False, scaled=False):
        """scaled (4a, POST HOC, descriptive): each reader's Δ as % of its cell's step-2 matched - shuffled range."""
        by = {}
        for rd in ("nla", "oracle"):
            for e in elig(rd):
                if english and e not in eng:
                    continue
                x, y = fve(s, e, rd, a), fve(s, e, rd, b)
                if x is None or y is None:
                    continue
                inf = V[(e, rd)]["info"]
                key = (item_of[e], V[(e, rd)]["state"])
                dd = (x - y) * (100 / R1RANGE[(rd, s)] if scaled else 1)
                by.setdefault(key, {}).setdefault(rd, []).append((dd, (inf["tok_O"] - inf[tokkey]) / inf["tok_O"]))
        rows = [(k, v) for k, v in sorted(by.items()) if "nla" in v and "oracle" in v]
        if not rows:
            return None
        raw = [np.mean([d for d, _ in v["nla"]]) - np.mean([d for d, _ in v["oracle"]]) for _, v in rows]
        cl = [k[0] for k, _ in rows]
        tup = [(np.mean([d for d, _ in v["nla"]]), np.mean([sh for _, sh in v["nla"]]),
                np.mean([d for d, _ in v["oracle"]]), np.mean([sh for _, sh in v["oracle"]])) for _, v in rows]
        f_ = lambda xs: sum(t[0] for t in xs) / sum(t[1] for t in xs) - sum(t[2] for t in xs) / sum(t[3] for t in xs)  # noqa: E731
        return {"n_item_states": len(rows), "raw": {"mean": float(np.mean(raw)), "ci95": H.boot(raw, cl)},
                "per_share": {"value": float(f_(tup)), "ci95": H.boot(tup, cl, f=f_)}}

    contrasts = {"primary": ("R1", "C1", elig_primary, "tok_C1"), "dA": ("RA", "CA", elig_ans, "tok_CA"),
                 "dAP": ("PA", "CA", lambda rd: elig_ans(rd, True), "tok_CA")}
    res["reader_contrasts_nla_minus_oracle"] = {}
    for cname, (a, b, el, tk) in contrasts.items():
        rc = {s: {"all": reader_diff(s, a, b, el, tk), "english_only": reader_diff(s, a, b, el, tk, True),
                  "scaled_pct_of_R1_range_descriptive": reader_diff(s, a, b, el, tk, scaled=True)} for s in ("oracle_ar", "nla_ar")}
        for s, o in (("oracle_ar", "nla_ar"), ("nla_ar", "oracle_ar")):
            x = rc[s]["all"]
            if not x or not rc[o]["all"]:
                rc[s]["claim_holds"] = None
                continue
            sg = np.sign(x["raw"]["mean"])
            rc[s]["claim_holds"] = bool((x["raw"]["ci95"][0] > 0 or x["raw"]["ci95"][1] < 0) and abs(x["raw"]["mean"]) >= 2
                                        and np.sign(rc[o]["all"]["raw"]["mean"]) == sg
                                        and rc[s]["english_only"] is not None and np.sign(rc[s]["english_only"]["raw"]["mean"]) == sg
                                        and np.sign(x["per_share"]["value"]) == sg
                                        and (x["per_share"]["ci95"][0] > 0 or x["per_share"]["ci95"][1] < 0))
        res["reader_contrasts_nla_minus_oracle"][cname] = rc

    # CW-20's secondary metric on Oracle texts x Oracle AR: one global scale fitted on the O versions
    oo = [x for e in eids for x in SC.get(("oracle_ar", e, "oracle"), {}).get("O", [])]
    a_ = sum(x["pt"] for x in oo) / sum(x["pp"] for x in oo)
    sf = lambda x: 1 - (x["tt"] - 2 * a_ * x["pt"] + a_ ** 2 * x["pp"]) / x["tt"]               # noqa: E731
    res["cw20_metric_oracle_texts"] = {"global_scale": a_, "scaled_fve_mean": {
        n: float(np.mean([np.mean([sf(x) for x in SC[("oracle_ar", e, "oracle")][n]]) for e in eids if n in SC.get(("oracle_ar", e, "oracle"), {})]))
        for n in ("O", "C1", "R1", "CA", "RA", "PA", "F", "P", "Fc", "Pc")}}

    # gates and checks
    jo = {x["eid"]: x for x in map(json.loads, open(OUT / f"judge/judged_oracle_{which}.jsonl"))}
    jn = {x["eid"]: x for x in map(json.loads, open(OUT / f"judge/judged_nla_{which}.jsonl"))}
    rw13 = {x["eid"]: x for x in map(json.loads, open(H.OUT / "rewrite/rewrites_all.jsonl")) if x["eid"] in jn}
    j13 = {x["eid"]: x for x in map(json.loads, open(H.OUT / "judge/judged_all.jsonl")) if x["eid"] in jn}
    rate = lambda labs: float(np.mean([l == "contradicted" for l in labs])) if labs else None   # noqa: E731
    mr = lambda xs: float(np.mean([x for x in xs if x is not None])) if any(x is not None for x in xs) else None  # noqa: E731
    g = {"G1_A_ok_rate_oracle": float(np.mean([x["A_ok"] for x in jo.values()])),
         "K_ok_rate_oracle": float(np.mean([bool(x["K_ok"]) for x in jo.values() if x["K_ok"] is not None])),
         "ANS_ok_rate": {"oracle": float(np.mean([x["answer"]["ok"] for x in jo.values()])),
                         "nla": float(np.mean([x["answer"]["ok"] for x in jn.values()]))},
         "B_contradicted_sentence_rate_oracle": {"O": mr([rate(x["B_labels"]) for x in jo.values()]),
                                                 "F": mr([rate(x["labels_F"]) for x in jo.values()]),
                                                 "P": mr([rate(x["labels_P"]) for x in jo.values()])},
         "B_contradicted_sentence_rate_nla": {"O": mr([rate(rw13[e]["labels"].get("O")) for e in jn]),
                                              "F": mr([rate(rw13[e]["labels"].get("F")) for e in jn]),
                                              "P_013": mr([rate(rw13[e]["labels"].get("P")) for e in jn]),
                                              "Pstar": mr([rate(x["labels_Pstar"]) for x in jn.values()])}}
    for rd, key in (("oracle", "P"), ("nla", "Pstar")):
        r_ = g[f"B_contradicted_sentence_rate_{rd}"]
        g[f"rewrite_gate_F_le_0.10_{rd}"] = r_["F"] is not None and r_["F"] <= 0.10
        g[f"rewrite_gate_{key}_ge_0.7_O_{rd}"] = r_[key] is not None and r_["O"] is not None and r_[key] >= 0.7 * r_["O"]
    # G3: A-contradicted vs B-contradicted per sentence (Oracle texts)
    a_f, b_f = [], []
    O = {r["eid"]: r for r in oracle_rows()}
    for e, x in jo.items():
        if not (x["A_ok"] and x["B_ok"]):
            continue
        pos = 0
        for sent, lab in zip(x["sentences"], x["B_labels"]):
            rx = H.ws_regex(sent)
            m = rx.search(O[e]["explanation"], pos) if rx else None
            if not m:
                continue
            pos = m.end()
            a_f.append(H.overlaps((m.start(), m.end()), [tuple(r) for r in x["contradicted_ranges"]]))
            b_f.append(lab == "contradicted")
    g["G3_kappa_oracle"] = H.kappa(a_f, b_f) if a_f else None
    g["G3_n_sentences_oracle"] = len(a_f)
    res["gates"] = g
    inf = lambda rd: [V[(e, rd)]["info"] for e in eids]                                         # noqa: E731
    res["rewrites"] = {
        "oracle": {"P_had_delimiters": sum(i.get("P_had_delimiters", False) for i in inf("oracle")),
                   "words_changed_F_mean": mr([i.get("words_changed_F") for i in inf("oracle")]),
                   "words_changed_P_mean": mr([i.get("words_changed_P") for i in inf("oracle")]),
                   "P_match_ratio_median": float(np.median([i["P_match_ratio"] for i in inf("oracle") if "P_match_ratio" in i])),
                   "P_within_20pct": sum(abs(i["P_match_ratio"] - 1) <= 0.2 for i in inf("oracle") if "P_match_ratio" in i),
                   "F_composition_sum": {k: sum(i["F_composition"][k] for i in inf("oracle") if "F_composition" in i)
                                         for k in ("deleted", "replaced_O", "replaced_F", "inserted", "words_O")},
                   "Fc": {"n": sum("words_changed_Fc" in i for i in inf("oracle")),
                          "words_changed_Fc_mean": mr([i.get("words_changed_Fc") for i in inf("oracle")]),
                          "words_changed_Pc_mean": mr([i.get("words_changed_Pc") for i in inf("oracle")]),
                          "Pc_match_ratio_median": float(np.median([i["Pc_match_ratio"] for i in inf("oracle") if "Pc_match_ratio" in i]))
                          if any("Pc_match_ratio" in i for i in inf("oracle")) else None,
                          "Pc_within_20pct": sum(abs(i["Pc_match_ratio"] - 1) <= 0.2 for i in inf("oracle") if "Pc_match_ratio" in i),
                          "Pc_had_delimiters": sum(i.get("Pc_had_delimiters", False) for i in inf("oracle")),
                          "Fc_lines_same": sum(bool(i.get("Fc_lines_same")) for i in inf("oracle") if "words_changed_Fc" in i)}},
        "nla": {"words_changed_F_mean": mr([i.get("words_changed_F") for i in inf("nla")]),
                "words_changed_P013_mean": mr([i.get("words_changed_P") for i in inf("nla")]),
                "words_changed_Pstar_mean": mr([i.get("words_changed_Pstar") for i in inf("nla")]),
                "Pstar_match_ratio_median": float(np.median([i["Pstar_match_ratio"] for i in inf("nla") if "Pstar_match_ratio" in i])),
                "Pstar_within_20pct": sum(abs(i["Pstar_match_ratio"] - 1) <= 0.2 for i in inf("nla") if "Pstar_match_ratio" in i),
                "F_composition_sum": {k: sum(i["F_composition"][k] for i in inf("nla") if "F_composition" in i)
                                      for k in ("deleted", "replaced_O", "replaced_F", "inserted", "words_O")}}}
    res["answer_arm_coverage"] = {rd: {"n_with_answer_spans": sum(i["n_ans"] > 0 for i in inf(rd)), "ans_eligible": sum(i["ans_eligible"] for i in inf(rd)),
                                       "PA_computable": sum(i["PA_computable"] for i in inf(rd)),
                                       "answer_token_share_mean": mr([(i["tok_O"] - i["tok_CA"]) / i["tok_O"] if i.get("tok_CA") is not None else None for i in inf(rd)])}
                                  for rd in ("oracle", "nla")}
    res["flagged_token_share_C1"] = {rd: float(np.mean([(i["tok_O"] - i["tok_C1"]) / i["tok_O"] for i in inf(rd)])) for rd in ("oracle", "nla")}
    # G-repro (NLA texts x NLA AR) against 013's scores, every 013 version, and the 013 rewrite scores
    s13 = {x["eid"]: x for x in map(json.loads, open(H.OUT / "scored_all.jsonl")) if x["eid"] in jn}
    r13 = {x["eid"]: x for x in map(json.loads, open(H.OUT / "rewrite/scored_all.jsonl")) if x["eid"] in jn}
    dv = [abs(float(np.mean([x["mse"] for x in SC[("nla_ar", e, "nla")][n]])) - m) for e in jn for n, m in s13[e]["mse"].items()]
    dr = [abs(SC[("nla_ar", e, "nla")][n][0]["mse"] - m) for e in jn for n, m in r13[e]["mse"].items() if n in ("F", "P")]
    res["G_repro_nla"] = {"versions_same_as_013": sum(V[(e, "nla")]["info"]["versions_same_as_013"] for e in jn), "n": len(jn),
                          "max_abs_mse_013_versions": max(dv), "n_values": len(dv), "max_abs_mse_013_rewrites": max(dr) if dr else None,
                          "pass_1e-6": max(dv) <= 1e-6 and (not dr or max(dr) <= 1e-6)}
    r1o = torch.load(ROOT / "outputs/015_oracle_hallucination/r1/oracle_ar_preds.pt", weights_only=False)["preds"]["oracle_bullets"]
    import ar_vectors as AV
    w = AV.whitener(44)
    sys.path.insert(0, str(HERE))
    import r1 as R1
    g44 = R1.gold(44)
    tw = lambda x: (x.double() - w["mu"]) @ w["W"].T                                            # noqa: E731
    lay = []
    for e in eids:
        if e in r1o:
            T = tw(g44[(item_of[e], V[(e, "oracle")]["state"])])
            P = tw(r1o[e])
            c0 = float(P @ T / (P.norm() * T.norm()))
            lay.append(abs(100 * 2 * (c0 - SC[("oracle_ar", e, "oracle")]["O"][0]["cos"]) / B["oracle_ar"]))
    res["oracle_O_vs_r1_layout_points"] = {"n": len(lay), "max": max(lay) if lay else None}
    res["pieces"] = {s: {k: meta[s][k] for k in meta[s] if k not in ("gold_share_J",)} for s in meta}
    (RES / "analysis").mkdir(parents=True, exist_ok=True)
    (RES / f"analysis/oh_{which}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("gates", "G_repro_nla", "answer_arm_coverage", "rewrites", "oracle_O_vs_r1_layout_points")},
                     indent=1, default=float), flush=True)
    for k, c in res["cells"].items():
        f = lambda d: "—" if not d else f"{d['mean']:+.1f} [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}] n{d['n']}"  # noqa: E731
        print(f"{k:22s} primary {f(c['primary_R1_minus_C1'])} | dA {f(c['dA_RA_minus_CA'])} | dAP {f(c['dAP_PA_minus_CA'])} | "
              f"F-P {f(c['F_minus_P'])}", flush=True)


def phase_examples(which):
    """Qualitative versions file: O, C1, R1, CA, RA, PA, F, P (seed 0) for the first Oracle readouts with PA computable."""
    V = [json.loads(l) for l in open(OUT / f"versions/versions_{which}.jsonl")]
    lines = ["# 015 versions, qualitative samples (seed 0 where seeded)\n"]
    for x in [v for v in V if v["info"]["PA_computable"]][:6]:
        lines.append(f"## {x['reader']} | {x['eid']}\n")
        for n in ("O", "C1", "R1", "CA", "RA", "PA", "F", "P", "Pstar"):
            if n in x["texts"]:
                lines.append(f"### {n}\n```\n{x['texts'][n][0]}\n```\n")
    (RES / "qualitative").mkdir(parents=True, exist_ok=True)
    (RES / f"qualitative/versions_{which}.md").write_text("\n".join(lines))
    print("wrote", RES / f"qualitative/versions_{which}.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--subset", default="smoke")
    ap.add_argument("--shard", default=None)
    a = ap.parse_args()
    if a.phase == "score_nla_ar":
        phase_score_nla_ar(a.subset, a.shard)
        raise SystemExit
    {"judge": phase_judge, "handset": phase_handset, "versions": phase_versions, "score_oracle_ar": phase_score_oracle_ar,
     "score_nla_ar": phase_score_nla_ar, "analyze": phase_analyze, "examples": phase_examples}[a.phase](a.subset)
