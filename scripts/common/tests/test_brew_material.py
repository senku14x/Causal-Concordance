"""brew_material reproduces the 001 material it was factored out of: the gated dev brew pairs (ids, order) and the
render keys of the 001 capture record. CPU only (tokenizer + WSB bank); the capture check skips without outputs/."""
import json

import pytest

import brew_material as BM
import registry as R


@pytest.fixture(scope="module")
def entries():
    return BM.brew_pairs(R.make_tokenizer())


def test_gated_dev_brew_matches_gate_record(entries):
    rows = [json.loads(l) for l in open(BM.GATE_FILE)]
    want = {r["pair_id"] for r in rows if r["pass"] and r["family"] == "brew"}
    got = BM.gated_dev_brew(entries)
    assert len(got) == 82
    assert {e["pair"].pair_id for e in got} <= want
    assert all(e["split"] == "dev" for e in got)


def test_render_keys_match_001_captures(entries):
    p = R.PROJECT / "outputs" / "001_leverage_profile" / "run" / "captures_brew_meta.json"
    if not p.exists():
        pytest.skip("001 capture record not present (outputs/ is mirrored, not in git)")
    keys = {r["key"] for r in json.loads(p.read_text())["renders"] if r["role"] in ("member", "partner")}
    got = BM.gated_dev_brew(entries)
    mine = {BM.ids_key(e["r_m"].ids) for e in got} | {BM.ids_key(e["r_p"].ids) for e in got}
    assert mine <= keys
