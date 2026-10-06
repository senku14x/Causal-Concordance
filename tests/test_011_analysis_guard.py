"""011 analysis: completeness guard and quarantine handling (report 31). CPU only, synthetic rows."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent / "scripts"
sys.path[:0] = [str(HERE / "011_olens_swaps"), str(HERE / "008_trackb"), str(HERE / "common")]
import ar_vectors as AV                     # noqa: E402
import olens_swaps_analysis as OA           # noqa: E402

CONDS = ["Bref_full_3L", "C_white", "Cctrl_answer", "C_rawcoord", "C_raw_answer", "C_raw_rand_0", "C_raw_rand_1",
         "D_ar", "D_ar_answer", "D_classmean", "D_rand_0", "D_rand_1"] + [f"Crand_{i}" for i in range(4)]


def make_run(root, n_pairs, quarantine=None):
    d = root / "out" / "t1" / "pairs"
    d.mkdir(parents=True)
    for i in range(n_pairs):
        pid = f"t{i // 2:02d}__c{i}"
        rows = [{"pair_id": pid, "name": "G_repro", "pass": True}]
        for j, c in enumerate(CONDS):
            r = {"pair_id": pid, "arm": "X", "name": c, "S": 0.01 * (i + j), "table": i // 2, "flip": False,
                 "edit_norm_sum_base": 1.0}
            if quarantine == (pid, c):
                r["quarantined"] = True
            rows.append(r)
        (d / f"{pid}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def run(monkeypatch, tmp_path, *args):
    monkeypatch.setattr(AV, "OUT", tmp_path / "out")
    monkeypatch.setattr(AV, "RES", tmp_path / "res")
    (tmp_path / "res").mkdir(exist_ok=True)
    monkeypatch.setattr(sys, "argv", ["x", "--run-id", "t1", *args])
    OA.main()


def test_refuses_partial_run(monkeypatch, tmp_path):
    make_run(tmp_path, 6)
    with pytest.raises(SystemExit):
        run(monkeypatch, tmp_path, "--expect-pairs", "8")
    assert not list((tmp_path / "res").glob("*.json"))


def test_partial_is_labelled(monkeypatch, tmp_path):
    make_run(tmp_path, 6)
    run(monkeypatch, tmp_path, "--expect-pairs", "8", "--allow-partial")
    out = json.loads((tmp_path / "res" / "olens_swaps_analysis_t1_PARTIAL.json").read_text())
    assert out["partial"] and out["n_pairs"] == 6


def test_quarantine_drops_only_that_condition(monkeypatch, tmp_path):
    make_run(tmp_path, 6, quarantine=("t00__c0", "D_ar"))
    run(monkeypatch, tmp_path, "--expect-pairs", "6")
    out = json.loads((tmp_path / "res" / "olens_swaps_analysis_t1.json").read_text())
    assert out["quarantined_rows"] == [["t00__c0", "D_ar"]]
    assert out["conditions"]["D_ar"]["n"] == 5 and out["conditions"]["D_classmean"]["n"] == 6
    assert out["contrasts"]["D_ar - D_classmean"]["n"] == 5 and out["contrasts"]["D_classmean - D_rand"]["n"] == 6
