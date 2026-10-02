"""Margin scorer (AMENDMENT_01 §3-4): m = LSE_{s in S(a')} log P(s | prompt) - LSE_{s in S(a)} log P(s | prompt).

Full-sequence endpoint: log P(s | prompt) = sum_i log P(s_i | prompt, s_<i) by teacher forcing, every token of every
spelling scored. A single-token spelling is read from the prompt-only forward; a multi-token spelling gets its own
forward on prompt + s, with the same writes applied (a write at a prompt position is clamped for every answer token).
Writes are built per forward by `writes_fn(cont)`, `cont` = the continuation appended to the prompt (() for the
prompt-only forward), so a patch can use a source captured at the matching sequence length.
"""
import math

import torch

SPELLING_FORMS = {
    "brew": lambda a: [" " + a, " " + a.capitalize()],        # after "Answer:"; " Teal" is two tokens
    "chain": lambda a: [str(a)],                                # after "</think>\n\n"; digits are single tokens
    "relational": lambda a: [" " + a],                          # after "... is Avery's"
}


def spellings(tok, family, answer):
    """Token-id tuples of the accepted spellings of `answer` (deduplicated, order kept)."""
    out = []
    for form in SPELLING_FORMS[family](answer):
        ids = tuple(tok(form, add_special_tokens=False)["input_ids"])
        if ids and ids not in out:
            out.append(ids)
    return out


def _lse(xs):
    t = torch.tensor(xs, dtype=torch.float64)
    return float(torch.logsumexp(t, 0))


@torch.no_grad()
def score(engine, prompt_ids, S_a, S_b, writes_fn=None):
    """Score both answer sets on one prompt under the same writes.

    Returns {"m", "lse_a", "lse_b", "spell": {spelling: {"lp", "per_token"}}, "run": prompt-only Run,
    "writes": [hook objects of every forward]}."""
    prompt_ids = list(prompt_ids)
    T = len(prompt_ids)
    writes_all = []

    def fwd(cont):
        w = list(writes_fn(tuple(cont))) if writes_fn is not None else []
        writes_all.extend(w)
        return engine.forward(prompt_ids + list(cont), writes=w)

    base = fwd(())
    lp_last = engine.logprobs(base, [T - 1])[0]
    spell = {}
    for s in dict.fromkeys(list(S_a) + list(S_b)):
        if len(s) == 1:
            per = [float(lp_last[s[0]])]
        else:
            run = fwd(s)
            lp = engine.logprobs(run, range(T - 1, T - 1 + len(s)))
            per = [float(lp[i, t]) for i, t in enumerate(s)]
            del run
        spell[s] = {"lp": float(sum(per)), "per_token": per}
    lse_a = _lse([spell[s]["lp"] for s in S_a])
    lse_b = _lse([spell[s]["lp"] for s in S_b])
    return {"m": lse_b - lse_a, "lse_a": lse_a, "lse_b": lse_b, "spell": spell, "run": base, "writes": writes_all,
            "first_tokens": sorted({s[0] for s in list(S_a) + list(S_b)}), "lp_last": lp_last}


def share(dm, G):
    """Full-cell patch share S = dm / G (G = m(partner) - m(member), clean)."""
    return dm / G if G != 0 else math.nan
