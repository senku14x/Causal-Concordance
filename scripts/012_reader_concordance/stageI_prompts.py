"""012 Stage I prompt builder (CPU): delayed-question variants of Jan's ST multihop country items (spec §18).

Template:  "Fact: Consider {clue}. {carrier} {question}"
  clue      from Jan's prompt "Fact: The {relation} of {clue} is[ the]" (country/nation clues only)
  carrier   one of two neutral sentences (no country content)
  question  capital / main language / continent of "that country"
Hold tokens: H1 = the "." ending the clue sentence; H2 = the final token of the carrier (its "."). The model's state at
H1/H2 does not depend on which question follows (causal LM), so one edited hold state must serve all three questions.

A (clue, relation) trial is DISCRIMINATING when the src and tgt answer sets do not overlap; non-discriminating trials
are kept but excluded from flip statistics. The fact table below is the only hand-written content (common facts; the
Stage I gate checks that Qwen knows each tgt fact directly).

  python stageI_prompts.py  ->  outputs/012_reader_concordance/stageI/prompts.json
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BANK = Path("/workspace/sources/ext/phrase-lens-src/src/phrase_lens/data/benchmarks/causal/multihop_st.json")
OUT = ROOT / "outputs/012_reader_concordance/stageI"

FACTS = {  # capital, main language(s), continent(s): accepted answer strings
    "Argentina": (["Buenos Aires"], ["Spanish"], ["South America"]),
    "Australia": (["Canberra"], ["English"], ["Australia", "Oceania"]),
    "Brazil": (["Brasília", "Brasilia"], ["Portuguese"], ["South America"]),
    "Canada": (["Ottawa"], ["English", "French"], ["North America"]),
    "China": (["Beijing"], ["Chinese", "Mandarin"], ["Asia"]),
    "Denmark": (["Copenhagen"], ["Danish"], ["Europe"]),
    "Egypt": (["Cairo"], ["Arabic"], ["Africa"]),
    "Finland": (["Helsinki"], ["Finnish"], ["Europe"]),
    "France": (["Paris"], ["French"], ["Europe"]),
    "Greece": (["Athens"], ["Greek"], ["Europe"]),
    "India": (["New Delhi", "Delhi"], ["Hindi", "English"], ["Asia"]),
    "Ireland": (["Dublin"], ["English", "Irish"], ["Europe"]),
    "Italy": (["Rome"], ["Italian"], ["Europe"]),
    "Japan": (["Tokyo"], ["Japanese"], ["Asia"]),
    "Kenya": (["Nairobi"], ["Swahili", "English"], ["Africa"]),
    "Mexico": (["Mexico City"], ["Spanish"], ["North America"]),
    "Morocco": (["Rabat"], ["Arabic"], ["Africa"]),
    "Netherlands": (["Amsterdam"], ["Dutch"], ["Europe"]),
    "Norway": (["Oslo"], ["Norwegian"], ["Europe"]),
    "Russia": (["Moscow"], ["Russian"], ["Europe", "Asia"]),
    "Scotland": (["Edinburgh"], ["English", "Scottish Gaelic"], ["Europe"]),
    "Sweden": (["Stockholm"], ["Swedish"], ["Europe"]),
    "Switzerland": (["Bern", "Berne"], ["German", "French", "Italian"], ["Europe"]),
}
QUESTIONS = {"capital": "The capital of that country is", "language": "The main language of that country is",
             "continent": "The continent of that country is"}
REL_INDEX = {"capital": 0, "language": 1, "continent": 2}
CARRIERS = {"c1": "Meanwhile, the morning train left the station on time and the passengers settled into their seats.",
            "c2": "Elsewhere, a light rain fell over the hills while the shops along the main street opened."}
RX = re.compile(r"^Fact: The (.+?) of (the (?:country|nation) .+?) is( the)?\s*$")


def main():
    items = json.loads(BANK.read_text())["items"]
    out, skipped = [], []
    for it in items:
        m = RX.match(it["prompt"].strip())
        if not m:
            continue
        src, tgt, clue = it["src"], it["tgt"], m.group(2)
        if src not in FACTS or tgt not in FACTS:
            skipped.append((it["name"], src, tgt))
            continue
        for cid, carrier in CARRIERS.items():
            head = f"Fact: Consider {clue}."
            for rel, q in QUESTIONS.items():
                ya, yb = FACTS[src][REL_INDEX[rel]], FACTS[tgt][REL_INDEX[rel]]
                out.append({"id": f"{it['name']}|{cid}|{rel}", "clue_id": it["name"], "src": src, "tgt": tgt, "clue": clue,
                            "carrier": cid, "relation": rel, "prompt": f"{head} {carrier} {q}",
                            "prefix_H1": head, "prefix_H2": f"{head} {carrier}",
                            "y_a": ya[0], "y_a_alts": ya[1:], "y_b": yb[0], "y_b_alts": yb[1:],
                            "discriminating": not ({a.lower() for a in ya} & {b.lower() for b in yb}),
                            "jan_relation": m.group(1), "jan_prompt": it["prompt"].strip()})
    OUT.mkdir(parents=True, exist_ok=True)
    tgt_checks = sorted({(p["tgt"], p["relation"]) for p in out})
    (OUT / "prompts.json").write_text(json.dumps({"items": out, "skipped": skipped,
                                                  "tgt_fact_checks": [{"country": c, "relation": r, "prompt": f"Fact: The {r if r != 'language' else 'main language'} of {c} is",
                                                                       "answers": FACTS[c][REL_INDEX[r]]} for c, r in tgt_checks]}, indent=1))
    n_clue = len({p["clue_id"] for p in out})
    print(f"{len(out)} prompts from {n_clue} clues ({len(skipped)} items skipped: {skipped}); "
          f"discriminating {sum(p['discriminating'] for p in out)}; tgt fact checks {len(tgt_checks)}")
    print(out[0]["prompt"])


if __name__ == "__main__":
    main()
