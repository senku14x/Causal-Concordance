"""012 first-token answer forms (stdlib only; safe for model- and reader-side code).

Single-position scores (token lenses, the model's own read-position output) cannot score a whole answer. The rule:
for every accepted spelling s of an answer, the surface forms " s", "s", " S", "S" (S = first character swapcased,
Stage A's resolved-form rule); the first non-whitespace token of each (Qwen splits " 26" into " ", "2", "6"); ids
shared by y_a and y_b (e.g. "2" for 20 vs 26) are dropped from both; a side left empty is NA.
"""


def first_token_sets(tok, ya, yb):
    def ids(lst):
        out = set()
        for s in lst:
            for f in dict.fromkeys([" " + s, s, " " + s[:1].swapcase() + s[1:], s[:1].swapcase() + s[1:]]):
                t = [x for x in tok.encode(f, add_special_tokens=False) if tok.decode([x]).strip()]
                if t:
                    out.add(t[0])
        return out
    a, b = ids(ya), ids(yb)
    shared = a & b
    return sorted(a - shared), sorted(b - shared), sorted(shared)


def lse_ids(logprobs_row, ids):
    """LSE of a 1-D log-prob tensor over token ids; None when ids is empty."""
    if not ids:
        return None
    return float(logprobs_row[list(ids)].double().logsumexp(0))
