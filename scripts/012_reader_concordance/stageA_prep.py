"""012 Stage A prep (CPU): single-token vocabulary for Jan's `build-token` (J-lens / logit direction banks).

Collects src/tgt concepts of phrase-lens's ST causal banks (multihop_st, broadcast_st) and keeps those that resolve to
one token under phrase-lens's surface-form rule (" w", "w", " W", "W"; first single-token form). Unresolved concepts
are recorded, not invented (their items are reported as uncovered by the reference run).

  python stageA_prep.py
"""
import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
BANKS = Path("/workspace/sources/ext/phrase-lens-src/src/phrase_lens/data/benchmarks/causal")
OUT = ROOT / "outputs/012_reader_concordance/stageA"
MODEL, REV = "Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"


def single_token_form(tok, w):
    for s in (" " + w, w, " " + w[:1].upper() + w[1:], w[:1].upper() + w[1:]):
        if len(tok.encode(s, add_special_tokens=False)) == 1:
            return s
    return None


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    concepts = {}
    for bank in ("multihop_st", "broadcast_st"):
        for it in json.loads((BANKS / f"{bank}.json").read_text())["items"]:
            for side in ("src", "tgt"):
                concepts.setdefault(it[side], set()).add(bank)
    ok, bad = {}, {}
    for w in sorted(concepts):
        f = single_token_form(tok, w)
        (ok if f else bad)[w] = f or sorted(concepts[w])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "words_st.json").write_text(json.dumps(sorted(ok), indent=0))
    (OUT / "words_st_report.json").write_text(json.dumps({"n_concepts": len(concepts), "n_single_token": len(ok),
                                                          "resolved_form": ok, "unresolved": bad}, indent=1))
    print(f"{len(ok)}/{len(concepts)} concepts resolve to one token; unresolved: {sorted(bad)}")


if __name__ == "__main__":
    main()
