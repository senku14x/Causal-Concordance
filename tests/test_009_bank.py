"""CPU-only tests for 009 (brew-code bank): generator determinism and constraints, token alignment, cell positions, answer
parsing, gate summary, D statistic, probe sanity and the reader-free import boundary. No model is loaded."""
import ast
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "009_brewcode"))
sys.path.insert(0, str(ROOT / "scripts" / "common"))
import bank as B          # noqa: E402
import c009 as C          # noqa: E402
import derisk as DR       # noqa: E402
import gate as GT         # noqa: E402


@pytest.fixture(scope="module")
def tok():
    import registry as R
    return R.make_tokenizer()


def test_generator_deterministic_and_streams_separate():
    a, _ = B.build("dev", 2, 20, C.BANK_SEED)
    b, _ = B.build("dev", 2, 20, C.BANK_SEED)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    t1, _ = B.build("test", 2, 5, C.BANK_SEED + 1)
    a2, _ = B.build("dev", 2, 20, C.BANK_SEED)            # building test in between does not move dev
    assert json.dumps(a, sort_keys=True) == json.dumps(a2, sort_keys=True)
    assert {it["name"] for it in t1}.isdisjoint({it["name"] for it in a})


@pytest.mark.parametrize("split,depth", [("dev", 2), ("dev", 1), ("test", 2), ("test", 1)])
def test_saved_bank_constraints(split, depth):
    items = B.load(split, depth)
    assert len(items) == (C.N_DEV if split == "dev" else C.N_TEST)
    for it in items:
        tb = it["table"]
        assert len(tb) == C.N_RULES and all(d in tb and d != c for c, d in tb.items())
        codes = it["codes"]
        assert sorted(codes) == sorted(C.COLOURS) and len(set(codes.values())) == 12
        assert all(10 <= v <= 99 for v in codes.values())
        m = it["member"]
        assert B.roles(tb, m["start"], depth) == (m["z"], m["f"])
        assert m["answer"] == {"final": codes[m["f"]], "start": codes[m["start"]]}
        assert it["partners"], it["name"]
        md = {"s": m["start"], "z": m["z"], "f": m["f"]}
        for p in it["partners"]:
            assert B.roles(tb, p["start"], depth) == (p["z"], p["f"])
            assert B.partner_ok(md, {"s": p["start"], "z": p["z"], "f": p["f"]}, depth)
            assert p["answer"]["final"] != m["answer"]["final"] and p["answer"]["start"] != m["answer"]["start"]
        for t in C.TASKS:
            txt = it["prompts"][t]["member"]
            assert txt.splitlines()[-1] == C.QUESTION[t] and txt.splitlines()[-2] == C.START_FMT.format(m["start"])
            assert txt.splitlines()[-3] == C.STIR[depth]
        if split == "dev":
            assert it["fold"] in range(C.N_FOLDS)
    if split == "dev":
        f = [it["fold"] for it in items]
        assert all(f.count(k) == len(items) // C.N_FOLDS for k in range(C.N_FOLDS))


@pytest.mark.parametrize("depth", [2, 1])
def test_alignment_and_cells(tok, depth):
    items = B.load("dev", depth)
    for it in random.Random(7).sample(items, 50):
        pos, T = B.check_item(tok, it)
        assert pos == it["cells"] and T == it["n_tokens"]
        r = B.render(tok, it["prompts"]["final"]["member"])
        d = r.decoded
        assert d[pos["start"]] == " " + it["member"]["start"]
        assert "".join(d[p] for p in pos["question"]) == C.QUESTION["final"]
        assert d[pos["emit_asst"]] == "\n" and d[pos["emit_asst"] - 1] == "assistant"
        assert d[pos["emit_think"]] == "\n\n" and d[pos["emit_think"] - 1] == "</think>"
        assert d[pos["emit_colon"]] == ":" and pos["emit_colon"] == T - 1


@pytest.mark.parametrize("text,want", [("47", 47), (" 47", 47), ("Answer: 47", 47), ("147", None), ("4 7", None),
                                       ("The code is 47.", 47), (" 47\n\nThe potion is purple", 47), ("", None),
                                       ("purple, 47", 47), (" 7", None)])
def test_parse_code(text, want):
    assert B.parse_code(text) == want
    assert B.code_correct(text, 47) == (want == 47)


def test_names_colour():
    assert B.names_colour(" 47 (purple)") and B.names_colour("Grey") and not B.names_colour(" 47\n\nDone.")


def test_gate_summary_and_decision():
    items = [{"name": f"x{i}"} for i in range(4)]
    rows = {}
    for i, it in enumerate(items):
        rr = []
        for t in C.TASKS:
            rr.append({"task": t, "role": "member", "partner": None, "pass": True, "greedy_ok": True, "n_correct": 10,
                       "greedy_names_colour": False})
            for j in range(2):
                ok = not (i == 0 and j == 1 and t == "start")
                rr.append({"task": t, "role": "partner", "partner": j, "pass": ok, "greedy_ok": ok, "n_correct": 10 if ok else 3,
                           "greedy_names_colour": False})
        rows[it["name"]] = rr
    S = GT.summarize(items, rows, 2, "t", complete=True)
    assert S["pairs"]["candidate"] == 8 and S["pairs"]["in"] == 7
    assert S["pairs"]["share_in_by_task"]["final"] == 1.0 and S["pairs"]["share_in_by_task"]["start"] == 7 / 8
    assert S["decision"] == "fall back to depth 1"                        # 7 pairs < 200
    assert S["pairs"]["tables_with_pair"] == 4


def test_select_partners_cap():
    it = {"name": "y", "partners": list(range(8))}
    assert GT.select_partners(it, None) == list(range(8))
    a = GT.select_partners(it, 4)
    assert len(a) == 4 and a == GT.select_partners(it, 4)


def test_d_stat_known_value():
    rs = [{"item": "t", "role": "member", "z": "red", "f": "blue"}, {"item": "t", "role": "partner", "z": "green", "f": "pink"}]
    LL = np.zeros((2, 12))
    ci = {c: k for k, c in enumerate(C.COLOURS)}
    LL[0, ci["red"]] = 2.0          # member favours its own z
    LL[1, ci["green"]] = 1.0        # partner favours its own z
    d = DR.d_stat(LL, rs, [(0, 1)], "z")
    assert d["mean"] == pytest.approx(3.0) and d["n_pairs"] == 1
    assert DR.d_stat(LL, rs, [(0, 1)], "f")["mean"] == pytest.approx(0.0)
    assert DR.inside([-0.4, 0.4]) and not DR.inside([-0.6, 0.1])


def test_probe_sanity():
    rng = np.random.default_rng(0)
    n, d = 360, 64
    y = np.array([C.COLOURS[i % 12] for i in range(n)])
    items = np.array([f"t{i // 6}" for i in range(n)])
    folds = np.array([int(i // 6) % 5 for i in range(n)])
    means = rng.normal(size=(12, d)) * 3
    X = np.stack([means[C.COLOURS.index(c)] for c in y]) + rng.normal(size=(n, d))
    r = DR.probe_stats(X, y, folds, items, 1)
    assert r["ba"] > 0.9 and r["decodes"]
    r0 = DR.probe_stats(rng.normal(size=(n, d)), y, folds, items, 1)
    assert r0["ba"] < 0.3 and not r0["decodes"]


def test_select_pairs_deterministic():
    rs = []
    for i in range(6):
        rs.append({"item": f"t{i}", "role": "member", "partner": None, "task": "final"})
        for j in range(3):
            rs.append({"item": f"t{i}", "role": "partner", "partner": j, "task": "final"})
    a = DR.select_pairs(rs, "final", 4)
    assert a == DR.select_pairs(rs, "final", 4) and len(a) == 4
    assert len({rs[m]["item"] for m, _ in a}) == 4                         # one per table first


def test_gated_material(tmp_path, monkeypatch):
    items = B.load("dev", 2)[:3]
    S = {"complete": True, "depth": 2,
         "pair_list": [{"item": items[0]["name"], "partner": 0, "pass": True},
                       {"item": items[1]["name"], "partner": 0, "pass": False}]}
    (tmp_path / "analysis").mkdir()
    (tmp_path / "analysis" / "gate_fake.json").write_text(json.dumps(S))
    monkeypatch.setattr(C, "RES", tmp_path)
    depth, rs = DR.gated_material("fake")
    assert depth == 2 and len(rs) == 4                                     # member + 1 partner, x 2 tasks
    assert {r["task"] for r in rs} == set(C.TASKS)
    pr = DR.pairs_of(rs, "final")
    assert len(pr) == 1 and rs[pr[0][0]]["role"] == "member"


def test_reader_free_imports():
    bad = {"readers", "jspace", "jlens", "peft", "p005", "planted", "vllm", "vllm_lens"}
    for f in (ROOT / "scripts" / "009_brewcode").glob("*.py"):
        names = set()
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        assert not (names & bad), (f.name, names & bad)
