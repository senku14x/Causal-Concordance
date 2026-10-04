"""005 concept screen (PI review 2026-10-04, design 005 v2 §2): exclude concepts that readers name anyway.
Base rate = per-sample naming rate of each eligible concept in existing readouts that did not target it, using the
design's variant match (c, C, c+s, c+es; whole word, case-insensitive). Rule fixed before computing: exclude if the
pooled NLA rate or the pooled Oracle rate exceeds 0.05 (PI threshold). Brew readouts are not counted for palette
colours (Brew targets colours). CPU only; read-only on the CW repo.
Usage: python screen_concepts.py /workspace/nla-vs-lenses"""
import glob, json, re, sys
from pathlib import Path

CW = Path(sys.argv[1])
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/005_planted_concepts/analysis/concept_screen.json"
THRESH = 0.05
PALETTE = {"black", "blue", "brown", "gold", "gray", "green", "orange", "pink", "purple", "red", "teal", "white"}
ELIGIBLE = ("loss enemy union skin sin storm profit frame gel earth corner thumb ear review custom dog symbol free ship "
            "role friend cook stand gray curve method weight summary time line photo bags desk pool ash steel tap path "
            "case box trade bomb soap process picture diamond ease hat sheet information race hair plane judge board "
            "formula sock scene").split()
assert len(ELIGIBLE) == 58
PAT = {c: re.compile(r"\b(?:%s)\b" % "|".join(re.escape(v) for v in (c, c + "s", c + "es")), re.I) for c in ELIGIBLE}


def samples_cw():
    """(reader, source, family, text) from every CW readout file that carries samples."""
    for f in sorted(glob.glob(str(CW / "runs/*/*.jsonl"))):
        name = Path(f).name
        reader = "nla" if name.startswith("nla") else "oracle" if name.startswith("olens") else None
        if reader is None or "judge" in name:
            continue
        fam = "brew" if "brew" in f else "chain" if "chain" in f else "multihop"
        for line in open(f):
            r = json.loads(line)
            for s in r.get("samples") or []:
                yield reader, "cw:" + str(Path(f).relative_to(CW)), fam, s


def samples_001():
    d = json.load(open(ROOT / "results/001_leverage_profile/analysis/snapshot.json"))
    for t in d["tables"]:
        for k, v in t["readouts"].items():
            rd = k.split("|")[-2]
            if rd == "nla":
                for s in v.get("raw") or []:
                    yield "nla", "001:nla-L42", "brew", s
            elif rd in ("oracle", "oracle_wsb"):
                for s in v:
                    yield "oracle", "001:" + rd, "brew", s


counts = {}                                                      # (reader, source) -> [n, {c: hits}]
for reader, src, fam, text in list(samples_cw()) + list(samples_001()):
    n_h = counts.setdefault((reader, src), [0, {c: 0 for c in ELIGIBLE}, {c: 0 for c in ELIGIBLE}])
    n_h[0] += 1
    for c in ELIGIBLE:
        if fam == "brew" and c in PALETTE:
            continue
        n_h[2][c] += 1
        if PAT[c].search(text):
            n_h[1][c] += 1

res = {"threshold": THRESH, "rule": "exclude if pooled NLA or pooled Oracle per-sample rate > threshold",
       "sources": {f"{r}|{s}": v[0] for (r, s), v in counts.items()}, "concepts": {}}
for c in ELIGIBLE:
    row = {}
    for reader in ("nla", "oracle"):
        hits = sum(v[1][c] for (r, s), v in counts.items() if r == reader)
        den = sum(v[2][c] for (r, s), v in counts.items() if r == reader)
        row[reader] = {"hits": hits, "n": den, "rate": hits / den if den else None}
    row["excluded"] = any((row[r]["rate"] or 0) > THRESH for r in ("nla", "oracle"))
    res["concepts"][c] = row
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1))
print("samples per source:", res["sources"])
for c, r in sorted(res["concepts"].items(), key=lambda kv: -max(kv[1]["nla"]["rate"] or 0, kv[1]["oracle"]["rate"] or 0)):
    if r["excluded"] or max(r["nla"]["rate"] or 0, r["oracle"]["rate"] or 0) > 0.01:
        print(f"{c:12s} nla {r['nla']['rate']:.3f} ({r['nla']['hits']}/{r['nla']['n']})  oracle {r['oracle']['rate']:.3f} "
              f"({r['oracle']['hits']}/{r['oracle']['n']})  {'EXCLUDED' if r['excluded'] else ''}")
print("excluded:", [c for c, r in res["concepts"].items() if r["excluded"]])
