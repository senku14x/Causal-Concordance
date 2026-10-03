"""Reader-free brew material helpers for label/assay experiments (004+), factored out of the mixed 001 module
(scripts/001_leverage_profile/leverage_profile.py, which also contains reader code in its snapshot stage).

Behaviour is identical to the 001 functions they replace (materials restricted to brew start pairs; gated_dev;
ids_key); tests/test_brew_material.py checks equality against the saved 001 gate and capture records.
"""
import hashlib
import json

import partners as P
import registry as R
import splits as SP
import wsb

GATE_FILE = R.PROJECT / "results" / "001_leverage_profile" / "analysis" / "gate_pairs.jsonl"


def ids_key(ids):
    return hashlib.sha1(json.dumps(list(ids)).encode()).hexdigest()[:16]


def brew_pairs(tok):
    """Every token-aligned brew start pair (dev and test) with its renders and split, as in 001 materials()."""
    _, brew = wsb.load_bank("brew")
    bsplit = SP.brew_split(brew)
    out = []
    for it in brew:
        for pair in P.brew_candidates(it):
            al = P.token_align(tok, pair, it)
            if al is None:
                continue
            out.append({"pair": pair, "item": it, "r_m": al[0], "r_p": al[1], "diff": al[2], "split": bsplit[it["name"]]})
    return out


def gated_dev_brew(entries):
    """The gated dev brew start pairs (001 gate record)."""
    rows = {json.loads(l)["pair_id"]: json.loads(l) for l in open(GATE_FILE)}
    return [e for e in entries if e["split"] == "dev" and rows.get(e["pair"].pair_id, {}).get("pass")
            and rows[e["pair"].pair_id]["family"] == "brew"]
