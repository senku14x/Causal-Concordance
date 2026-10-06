"""009 gate (design docs/design_specs/008_009_trackb_brewcode.md §2.3-2.4): the project gate (partners.GATE: greedy correct
and >= 9/10 correct at T = 0.8, top_p 0.95, top_k 0, 16 new tokens, EOS suppressed) for every dev render under both tasks.
Stock bf16 forward (model resident; behavioural). Renders of one table x task have equal length and are batched (greedy:
one batched call; sampling: one call with num_return_sequences = 10, seeded per table x task). A pair is in if both
members pass both tasks. Reader-free.

  python gate.py --depth 2 --run-id g2 [--limit-tables N] [--max-partners K]
  python gate.py --depth 2 --run-id g2 --summarize-only

Resumable: one JSONL per table under outputs/009_brewcode/gate/<run_id>/ (finished tables are skipped).
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
import bank as B                                    # noqa: E402
import c009 as C                                    # noqa: E402

GATE = {"greedy": True, "k": 10, "min_correct": 9, "temperature": 0.8, "top_p": 0.95, "max_new_tokens": 16,
        "suppress": ["<|im_end|>", "<|endoftext|>"]}          # = partners.GATE (asserted in main)


def seed_of(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def select_partners(it, max_partners):
    """Optional cap on partners per table (seeded); None keeps every partner."""
    idx = list(range(len(it["partners"])))
    if max_partners is not None and len(idx) > max_partners:
        idx = sorted(random.Random(seed_of(it["name"], "cap", C.BANK_SEED)).sample(idx, max_partners))
    return idx


def table_renders(it, pidx):
    """[(role, partner_index or None, start, {task: answer}, {task: text})] for the member and the selected partners."""
    out = [("member", None, it["member"]["start"], it["member"]["answer"],
            {t: it["prompts"][t]["member"] for t in C.TASKS})]
    for j in pidx:
        p = it["partners"][j]
        out.append(("partner", j, p["start"], p["answer"], {t: it["prompts"][t]["partners"][j] for t in C.TASKS}))
    return out


def gate_batch(model, tok, ids_list, answers, seed, gate=GATE):
    """Greedy + k samples for a batch of equal-length renders. Returns one dict per render."""
    import torch
    import registry
    registry.assert_subject_model_intact(model)
    registry.assert_subject_model_resident(model)
    assert len({len(x) for x in ids_list}) == 1, "renders in a batch must have equal length"
    sup = [tok.convert_tokens_to_ids(t) for t in gate["suppress"]]
    sup = [i for i in sup if isinstance(i, int) and i >= 0]
    T = len(ids_list[0])
    x = torch.tensor(ids_list, device=model.device)
    with torch.no_grad():
        g = model.generate(x, attention_mask=torch.ones_like(x), do_sample=False, max_new_tokens=gate["max_new_tokens"],
                           suppress_tokens=sup, pad_token_id=tok.eos_token_id)
        greedy = [tok.decode(row[T:], skip_special_tokens=True) for row in g]
        torch.manual_seed(seed)
        s = model.generate(x, attention_mask=torch.ones_like(x), do_sample=True, temperature=gate["temperature"],
                           top_p=gate["top_p"], top_k=0, max_new_tokens=gate["max_new_tokens"],
                           num_return_sequences=gate["k"], suppress_tokens=sup, pad_token_id=tok.eos_token_id)
        dec = [tok.decode(row[T:], skip_special_tokens=True) for row in s]
    out = []
    for i, a in enumerate(answers):
        samples = dec[i * gate["k"]:(i + 1) * gate["k"]]          # generate expands each input row consecutively
        ok_g = B.code_correct(greedy[i], a)
        n_ok = sum(B.code_correct(t, a) for t in samples)
        out.append({"greedy": greedy[i], "greedy_ok": ok_g, "samples": samples, "n_correct": n_ok, "k": gate["k"],
                    "pass": bool(ok_g and n_ok >= gate["min_correct"]),
                    "greedy_names_colour": B.names_colour(greedy[i]),
                    "n_samples_naming_colour": sum(B.names_colour(t) for t in samples)})
    return out


def summarize(items, rows_by_item, depth, run_id, complete):
    """Per-task render pass rates, pair pass rates (pair in = both members pass both tasks), and the §2.4 decision."""
    by = {}
    for it in items:
        rows = rows_by_item.get(it["name"])
        if rows is None:
            continue
        by[it["name"]] = {(r["task"], r["role"], r["partner"]): r for r in rows}
    n_tab = len(by)
    rend = {t: [r for d in by.values() for (tt, _, _), r in d.items() if tt == t] for t in C.TASKS}
    pairs, pair_in, pair_task = [], [], {t: 0 for t in C.TASKS}
    tables_with_pair = 0
    for name, d in by.items():
        m_ok = {t: d[(t, "member", None)]["pass"] for t in C.TASKS}
        any_pair = False
        for (t, role, j), r in d.items():
            if role != "partner" or t != C.TASKS[0]:
                continue
            p_ok = {tt: d[(tt, "partner", j)]["pass"] for tt in C.TASKS}
            ok = all(m_ok.values()) and all(p_ok.values())
            for tt in C.TASKS:
                pair_task[tt] += int(m_ok[tt] and p_ok[tt])
            pairs.append({"item": name, "partner": j, "pass": ok,
                          "pass_by_task": {tt: bool(m_ok[tt] and p_ok[tt]) for tt in C.TASKS}})
            pair_in.append(ok)
            any_pair = any_pair or ok
        tables_with_pair += int(any_pair)
    n_c, n_in = len(pairs), int(sum(pair_in))
    share = n_in / n_c if n_c else 0.0
    S = {"run_id": run_id, "depth": depth, "complete": complete, "n_tables_gated": n_tab, "gate": GATE,
         "renders": {t: {"n": len(rend[t]), "pass_rate": (sum(r["pass"] for r in rend[t]) / len(rend[t])) if rend[t] else None,
                         "greedy_rate": (sum(r["greedy_ok"] for r in rend[t]) / len(rend[t])) if rend[t] else None,
                         "greedy_names_colour_rate": (sum(r["greedy_names_colour"] for r in rend[t]) / len(rend[t]))
                         if rend[t] else None,
                         "mean_n_correct": (sum(r["n_correct"] for r in rend[t]) / len(rend[t])) if rend[t] else None}
                     for t in C.TASKS},
         "pairs": {"candidate": n_c, "in": n_in, "share_in": share,
                   "share_in_by_task": {t: (pair_task[t] / n_c if n_c else None) for t in C.TASKS},
                   "tables_with_pair": tables_with_pair},
         "decision_rule": f"depth 2 ok iff share_in >= {C.GATE_MIN_SHARE} and pairs in >= {C.GATE_MIN_PAIRS} "
                          "(over the full dev bank); else fall back to depth 1 (design §2.4)"}
    ok = share >= C.GATE_MIN_SHARE and n_in >= C.GATE_MIN_PAIRS
    if complete:
        S["decision"] = ("depth 2 ok" if ok else "fall back to depth 1") if depth == 2 else ("ok" if ok else "below the bar")
    else:
        proj = n_in * (len(items) / n_tab) if n_tab else 0
        S["decision"] = f"partial ({n_tab}/{len(items)} tables): share {share:.3f}, projected pairs in ~{proj:.0f}"
    S["pair_list"] = pairs
    return S


def load_rows(d):
    out = {}
    for f in sorted(Path(d).glob("*.jsonl")):
        out[f.stem] = [json.loads(l) for l in open(f)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--limit-tables", type=int, default=None)
    ap.add_argument("--max-partners", type=int, default=None)
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    import partners as P
    assert P.GATE == GATE, "gate settings drifted from partners.GATE"
    items = B.load("dev", a.depth)
    if a.limit_tables:
        items = items[:a.limit_tables]
    d = C.OUT / "gate" / a.run_id
    d.mkdir(parents=True, exist_ok=True)
    log = C.Logger(d / "log.txt")
    if not a.summarize_only:
        import registry as R
        tok = R.make_tokenizer()
        (d / "manifest.json").write_text(json.dumps({"run_id": a.run_id, "depth": a.depth, "max_partners": a.max_partners,
                                                     "limit_tables": a.limit_tables, "manifest": R.manifest_block(),
                                                     "design": "docs/design_specs/008_009_trackb_brewcode.md §2.3"}, indent=1))
        model = R.load_model()
        t_all = time.time()
        for n, it in enumerate(items):
            f = d / f"{it['name']}.jsonl"
            if f.exists():
                continue
            t0 = time.time()
            rend = table_renders(it, select_partners(it, a.max_partners))
            rows = []
            for t in C.TASKS:
                ids = [B.render(tok, txt[t]).ids for _, _, _, _, txt in rend]
                res = gate_batch(model, tok, ids, [ans[t] for _, _, _, ans, _ in rend], seed_of(it["name"], t, C.BANK_SEED))
                for (role, j, s, ans, _), r in zip(rend, res):
                    rows.append({"item": it["name"], "task": t, "role": role, "partner": j, "start": s, "answer": ans[t],
                                 **r})
            tmp = f.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
            tmp.rename(f)
            npass = {t: sum(r["pass"] for r in rows if r["task"] == t) for t in C.TASKS}
            log(f"[{n + 1}/{len(items)}] {it['name']} renders {len(rend)} pass final {npass['final']} start {npass['start']} "
                f"{time.time() - t0:.1f}s (total {(time.time() - t_all) / 60:.1f} min)")
    rows = load_rows(d)
    complete = all(it["name"] in rows for it in items) and not a.limit_tables
    S = summarize(items, rows, a.depth, a.run_id, complete)
    C.dump(C.RES / "analysis" / f"gate_{a.run_id}.json", S)
    log("summary", json.dumps({k: v for k, v in S.items() if k != "pair_list"}, default=float))


if __name__ == "__main__":
    main()
