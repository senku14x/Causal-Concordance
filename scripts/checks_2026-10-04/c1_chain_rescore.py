"""C1 (pre-registered in docs/design_specs/checks_2026-10-04_no_gpu.md): re-score the CW pilot's chain readouts
(CW-6 / CW-7) with WSB's chain_intermediates judge at the pinned commit, unmodified, next to CW's any-number count.
Read-only on the CW repo. Without --real (or without OPENROUTER_API_KEY) it runs WSB's dry run only: row format,
item match and read positions are checked and no number is produced.
Usage: python c1_chain_rescore.py /workspace/nla-vs-lenses [--real]"""
import json, os, subprocess, sys
from pathlib import Path

CW = Path(sys.argv[1])
REAL = "--real" in sys.argv
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/common"))
import splits as SP  # noqa: E402
import wsb  # noqa: E402

WSBENCH = "/venv/wsb/bin/wsbench"
WSB_DIR = ROOT.parent / "sources/repos/workspace-bench"
RAW = ROOT / "outputs/checks_2026-10-04/c1"                   # WSB results.json per file (gitignored, test rows inside)
OUT = ROOT / "results/checks_2026-10-04/c1_chain_rescore.json"
FILES = [("oracle", 42, "runs/chain60/olens_L42.chain.jsonl"), ("nla", 42, "runs/chain60/nla_L42.chain.jsonl"),
         ("oracle", 56, "runs/chain60_L56/olens_L56.chain.jsonl"), ("nla", 56, "runs/chain60_L56/nla_L56.chain.jsonl"),
         ("jlens", 42, "runs/chain60/jlens.chain.jsonl"), ("jlens", 56, "runs/chain60_L56/jlens.chain.jsonl")]

# CW's own counter, imported unmodified (its module level reads sys.argv[1] as the run dir)
_argv, sys.argv = sys.argv, ["score_chain", str(CW / "runs/chain60")]
sys.path.insert(0, str(CW / "scripts"))
import score_chain as cw  # noqa: E402
sys.argv = _argv

_, bank = wsb.load_bank("chain")
items = {it["name"]: it for it in bank}
csplit, _ = SP.chain_split(bank)
real = REAL and bool(os.environ.get("OPENROUTER_API_KEY"))
if REAL and not real:
    sys.exit("--real needs OPENROUTER_API_KEY")


def cw_count(reader, layer, rows):
    """CW's any-number rule: does the readout state any intermediate; chance = scored against the other items."""
    rows = [r for r in rows if r["layer"] == layer]
    if reader == "jlens":
        S = {r["id"]: cw.nums_in_tokens(r["tokens"]) for r in rows}
    elif reader == "nla":
        S = {r["id"]: cw.nums_in_text(r["samples"][0]) for r in rows}
    else:
        S = {r["id"]: cw.nums_in_text(" ".join(r["samples"])) for r in rows}
    ids = sorted(S)
    I = {i: set(items[i]["intermediates"]) for i in ids}
    hit = {i: bool(S[i] & I[i]) for i in ids}
    chance = sum(sum(bool(S[i] & I[j]) for j in ids if j != i) / (len(ids) - 1) for i in ids)
    return hit, chance, {i: len(S[i]) for i in ids}


def subset_summary(rows, ids):
    rs = [r for r in rows if r["id"] in ids]
    dec = [r for r in rs if r["pass"] is not None]
    nulls = [r["null_top1_near"] for r in rs if r.get("null_top1_near") is not None]
    return {"n": len(rs), "n_decided": len(dec), "pass": sum(bool(r["pass"]) for r in dec),
            "pass_rate": sum(bool(r["pass"]) for r in dec) / len(dec) if dec else None,
            "null_top1_near_mean": sum(nulls) / len(nulls) if nulls else None,
            "any_of_3": sum(bool(r.get("any_of_3")) for r in rs)}


res = {"design": "docs/design_specs/checks_2026-10-04_no_gpu.md#c1", "cw_commit": "58d34a8", "real": real,
       "wsb_commit": subprocess.run(["git", "-C", str(WSB_DIR), "rev-parse", "HEAD"], capture_output=True,
                                    text=True).stdout.strip(), "cells": {}}
test_out = {}
for reader, layer, rel in FILES:
    rows = [json.loads(l) for l in open(CW / rel)]
    ids = sorted({r["id"] for r in rows})
    assert all(i in items and items[i]["prompt"] for i in ids), "readout id not in the pinned bank"
    out = RAW / f"{reader}_L{layer}"
    cmd = [WSBENCH, "judge", "family=chain_intermediates", f"readouts={CW / rel}", f"layers={layer}",
           f"items={','.join(ids)}", f"out={out}", f"dry_run={not real}"]
    env = os.environ.copy() if real else {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    p = subprocess.run(cmd, cwd=WSB_DIR, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    d = json.loads((out / "results.json").read_text())
    c, e = d["counts"], d["numbers"]["extras"]
    assert c["n_expected_cells"] == len(ids) == 60 and c["n_missing_cells"] == 0, c
    assert e["n_rows_not_last_token"] == 0, e
    hit, chance, n_nums = cw_count(reader, layer, rows)
    usable = [i for i in ids if i in csplit]
    dev = {i for i in usable if csplit[i] == "dev"}
    cell = {"file": rel, "n_items": d["n_items"], "kind": d["config"]["kind"], "counts": c,
            "config": d["config"],
            "cw_any_number": {"hits": sum(hit.values()), "n": len(hit), "chance": chance,
                              "median_distinct_numbers": sorted(n_nums.values())[len(n_nums) // 2]},
            "n_usable": len(usable), "n_usable_dev": len(dev)}
    if real:
        cell["wsb"] = {"pass_rate": d["numbers"]["value"], "ci95": d["numbers"]["ci95"],
                       **{k: e.get(k) for k in ("null_top1_near", "any_of_3_rate", "committed_rate",
                                                "n_items_decided", "n_items_undecided", "kinds", "basis")}}
        cell["all60"] = subset_summary(d["rows"], set(ids))
        cell["usable_dev"] = subset_summary(d["rows"], dev)
        cell["cw_any_number"]["usable_dev_hits"] = sum(hit[i] for i in dev)
        test_out[f"{reader}_L{layer}"] = subset_summary(d["rows"], set(usable) - dev)  # quarantined (§4.R)
    res["cells"][f"{reader}_L{layer}"] = cell
    print(f"{reader:6s} L{layer}: items {d['n_items']} missing {c['n_missing_cells']} calls {e['n_calls']}"
          f" | CW any-number {sum(hit.values())}/{len(hit)} (chance {chance:.1f})"
          + (f" | WSB pass {d['numbers']['value']} null {e.get('null_top1_near')}" if real else " | dry run"))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1, default=str))
if real:
    (RAW / "usable_test_QUARANTINED.json").write_text(json.dumps(test_out, indent=1))
print("wrote", OUT)
