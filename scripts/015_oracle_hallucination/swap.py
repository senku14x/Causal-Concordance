"""015 follow-up (POST HOC, descriptive; ledger amendment 2026-10-11 "swap test and runner-up test"): the answer swap test.

Oracle readouts with >= 1 true-answer opening and >= 1 wrong-answer opening. Only the readout's own strings are moved,
so no new wording enters (unlike Fc):
  ALLW  every true opening replaced by a wrong opening of the same readout (5 seeds)
  ALLT  every wrong opening replaced by a true opening of the same readout (5 seeds)
  SWAP  min(#true, #wrong) true and wrong openings exchange bullets (5 seeds); the multiset of openings is unchanged
Measures (FVE points, both reconstructors): O − ALLW, ALLT − O, O − SWAP; item-cluster bootstrap, ±2 bands.

True opening: a judge-B answer span (located in order, oh.locate_seq) that starts a bullet (after leading markup).
Wrong opening: openings() below, shared with runnerup.py.

  python swap.py versions        -> outputs/015_oracle_hallucination/versions/versions_swap.jsonl, wrong_openings.jsonl
  CUDA_VISIBLE_DEVICES=0 python oh.py score_oracle_ar --subset swap
  CUDA_VISIBLE_DEVICES=1 python oh.py score_nla_ar --subset swap
  python swap.py analyze         -> results/015_oracle_hallucination/analysis/swap.json
"""
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_argv, sys.argv = sys.argv, sys.argv[:1]
import oh  # noqa: E402
sys.argv = _argv

LEAD = " *\"“”‘’'`"
STOP = ".,;:!?()[]\"“”*|、。，（）\n"
NUMW = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                                         "fifteen sixteen seventeen eighteen nineteen twenty".split())}


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().casefold()
    s = " ".join(NUMW.get(w, w) for w in re.findall(r"[a-z0-9]+", s))
    return s


def true_answer(continuation):
    c = continuation.strip().lstrip(LEAD)
    m = re.match(r"[^" + re.escape(STOP) + r"]*", c)
    return m.group(0).strip(" '\"")


def is_latin(s):
    return not re.search(r"[^\x00-\x7FÀ-ɏ‘-‟–—…]", s)


def openings(text, continuation):
    """Per bullet: (content_start, opening_start, opening_end, opening) for the rule-based opening, plus a status:
    'wrong' (kept), or why it was skipped. Rule: the bullet's text after '- ' and leading markup/quotes, up to the first
    punctuation; Latin script; <= 4 words; not a placeholder ('[' or '_' start); not the true answer (normalised, with
    number words; containment either way counts as the true answer)."""
    ta = norm(true_answer(continuation))
    out, pos = [], 0
    for line in text.split("\n"):
        a, pos = pos, pos + len(line) + 1
        if not line.strip():
            continue
        lead_ws = len(line) - len(line.lstrip())
        c0 = a + lead_ws + (2 if line.lstrip().startswith("- ") else 0)
        body = text[c0:a + len(line)]
        k = len(body) - len(body.lstrip(LEAD))
        rest = body[k:]
        if not rest or rest[0] in "[_" or "_" in re.match(r"[^" + re.escape(STOP) + r"]*", rest).group(0):
            out.append({"c0": c0, "status": "placeholder"})
            continue
        m = re.match(r"[^" + re.escape(STOP) + r"]*", rest)
        op = m.group(0).rstrip()
        s0, s1 = c0 + k, c0 + k + len(op)
        rec = {"c0": c0, "s0": s0, "s1": s1, "opening": op}
        no = norm(op)
        if not op.strip():
            rec["status"] = "empty"
        elif not is_latin(op):
            rec["status"] = "non_latin"
        elif len(op.split()) > 4:
            rec["status"] = "too_long"
        elif not no or not ta or no == ta or ta in no or no in ta:
            rec["status"] = "true_or_ambiguous"
        else:
            rec["status"] = "wrong"
        out.append(rec)
    return out, true_answer(continuation)


def phase_versions():
    O = {r["eid"]: r for r in oh.oracle_rows()}
    J = {x["eid"]: x for x in map(json.loads, open(oh.OUT / "judge/judged_oracle_all.jsonl"))}
    recs, wrong_rows = [], []
    for e, r in O.items():
        text, ans = r["explanation"], J[e]["answer"]
        ops, ta = openings(text, r["continuation"])
        ar = oh.locate_seq(ans.get("spans") or [], text) if ans.get("ok") else []
        true_slots, wrong_slots = [], []
        for o in ops:
            c0 = o["c0"]
            body_lead = c0 + (len(text[c0:]) - len(text[c0:].lstrip(LEAD)))
            hit = [x for x in ar if c0 <= x[0] <= body_lead]               # an answer span that starts the bullet (markup allowed)
            if hit:
                true_slots.append((hit[0][0], hit[0][1]))
            elif o["status"] == "wrong" and not any(x[0] < o["s1"] and o["s0"] < x[1] for x in ar):
                wrong_slots.append((o["s0"], o["s1"]))
        for o in ops:
            if o["status"] == "wrong" and (o["s0"], o["s1"]) in wrong_slots:
                wrong_rows.append({"eid": e, "item": r["item"], "state": r["state"], "sample": r["sample"], "opening": o["opening"],
                                   "true_answer": ta, "prompt": r["prompt"], "continuation": r["continuation"]})
        info = {"n_true": len(true_slots), "n_wrong": len(wrong_slots), "statuses": [o["status"] for o in ops]}
        if not true_slots or not wrong_slots:
            recs.append({"eid": e, "reader": "oracle", "item": r["item"], "state": r["state"], "sample": r["sample"], "info": {**info, "eligible": False},
                         "texts": {}})
            continue
        T = [text[a:b] for a, b in true_slots]
        W = [text[a:b] for a, b in wrong_slots]

        def rebuild(repl):                                                 # repl: {(a, b): new string}
            t = text
            for (a, b), s_ in sorted(repl.items(), reverse=True):
                t = t[:a] + s_ + t[b:]
            return t
        v = {"O": [text], "ALLW": [], "ALLT": [], "SWAP": []}
        s0 = oh.seed_of(e)
        for s in range(5):
            rng = random.Random(s0 + 100 * s + 211)
            v["ALLW"].append(rebuild({sl: rng.choice(W) for sl in true_slots}))
            v["ALLT"].append(rebuild({sl: rng.choice(T) for sl in wrong_slots}))
            ts, ws = true_slots[:], wrong_slots[:]
            rng.shuffle(ts)
            rng.shuffle(ws)
            k = min(len(ts), len(ws))
            v["SWAP"].append(rebuild({**{ts[i]: text[ws[i][0]:ws[i][1]] for i in range(k)}, **{ws[i]: text[ts[i][0]:ts[i][1]] for i in range(k)}}))
        recs.append({"eid": e, "reader": "oracle", "item": r["item"], "state": r["state"], "sample": r["sample"],
                     "info": {**info, "eligible": True, "true_strings": T, "wrong_strings": W}, "texts": v})
    (oh.OUT / "versions").mkdir(parents=True, exist_ok=True)
    with open(oh.OUT / "versions/versions_swap.jsonl", "w") as f:
        for x in recs:
            if x["info"]["eligible"]:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(oh.OUT / "versions/wrong_openings.jsonl", "w") as f:
        for x in wrong_rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    st = {}
    for x in recs:
        for s_ in x["info"]["statuses"]:
            st[s_] = st.get(s_, 0) + 1
    summ = {"n_readouts": len(recs), "n_eligible": sum(x["info"]["eligible"] for x in recs), "n_wrong_openings": len(wrong_rows),
            "opening_status_counts": st}
    (oh.OUT / "versions/swap_versions_meta.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def phase_analyze():
    A = json.loads((oh.RES / "analysis/oh_all.json").read_text())
    B = A["B"]
    V = {x["eid"]: x for x in map(json.loads, open(oh.OUT / "versions/versions_swap.jsonl"))}
    meta = json.loads((oh.OUT / "versions/swap_versions_meta.json").read_text())
    res = {"label": "POST HOC, descriptive (015 swap test; ledger amendment 2026-10-11)", "B": B, "versions_meta": meta, "cells": {}}
    for s in ("oracle_ar", "nla_ar"):
        files = [oh.OUT / f"score/{s}_swap.jsonl"] if (oh.OUT / f"score/{s}_swap.jsonl").exists() else sorted((oh.OUT / "score").glob(f"{s}_swap.shard*.jsonl"))
        F = {}
        for fp in files:
            for x in map(json.loads, open(fp)):
                if not x["degenerate"]:
                    F.setdefault((x["eid"], x["version"]), []).append(100 * (1 - (2 - 2 * x["cos"]) / B[s]))
        f = lambda e, n: float(np.mean(F[(e, n)]))                                               # noqa: E731
        es = [e for e in V if all((e, n) in F for n in ("O", "ALLW", "ALLT", "SWAP"))]

        def stat(fn, rows):
            d = [fn(e) for e in rows]
            if not d:
                return None
            m, ci = float(np.mean(d)), oh.H.boot(d, [V[e]["item"] for e in rows])
            band = ("H-signal" if m >= 2 and ci[0] > 0 else "H-padding" if m <= -2 and ci[1] < 0 else
                    "no difference beyond ±2" if ci[0] >= -2 and ci[1] <= 2 else "unresolved")
            return {"n": len(d), "mean": m, "ci95": ci, "band": band}
        c = {"n": len(es),
             "O_minus_ALLW": stat(lambda e: f(e, "O") - f(e, "ALLW"), es),
             "ALLT_minus_O": stat(lambda e: f(e, "ALLT") - f(e, "O"), es),
             "O_minus_SWAP": stat(lambda e: f(e, "O") - f(e, "SWAP"), es),
             "ALLT_minus_ALLW": stat(lambda e: f(e, "ALLT") - f(e, "ALLW"), es),
             "per_state": {st: {k: stat(fn, [e for e in es if V[e]["state"] == st]) for k, fn in
                                (("O_minus_ALLW", lambda e: f(e, "O") - f(e, "ALLW")), ("ALLT_minus_O", lambda e: f(e, "ALLT") - f(e, "O")),
                                 ("O_minus_SWAP", lambda e: f(e, "O") - f(e, "SWAP")))} for st in oh.STATES},
             "fve_mean": {n: float(np.mean([f(e, n) for e in es])) for n in ("O", "ALLW", "ALLT", "SWAP")}}
        res["cells"][f"oracle_texts|{s}"] = c
    (oh.RES / "analysis/swap.json").write_text(json.dumps(res, indent=1))
    for k, c in res["cells"].items():
        print(k, c["n"], {m: (round(c[m]["mean"], 2), [round(x, 2) for x in c[m]["ci95"]], c[m]["band"]) for m in
                          ("O_minus_ALLW", "ALLT_minus_O", "O_minus_SWAP", "ALLT_minus_ALLW")})


if __name__ == "__main__":
    {"versions": phase_versions, "analyze": phase_analyze}[sys.argv[1]]()
