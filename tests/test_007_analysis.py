"""CPU unit tests for 007's statistics (design §3.3, §4.4 + A1): D incl. the off-trajectory role, table means, 006's
derangement null, Holm, TOST, categories and the three-model mapping (every category incl. uninformative)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "007_jsubspace"))
import analysis as AN  # noqa: E402
import dstat  # noqa: E402
from p007 import COLOURS  # noqa: E402

CI = {c: i for i, c in enumerate(COLOURS)}


def _pairs():
    # 3 tables; member index 0/3/6, partners 1,2 / 4,5 / 7,8
    P = []
    roles = [("red", "blue", "green"), ("black", "pink", "gold"), ("white", "gray", "orange")]
    proles = [[("brown", "purple", "teal"), ("teal", "black", "brown")],
              [("red", "green", "blue"), ("gray", "orange", "purple")],
              [("pink", "gold", "black"), ("blue", "brown", "green")]]
    for t in range(3):
        m = 3 * t
        for j in range(2):
            mr = dict(zip(("start", "z", "answer"), roles[t]))
            pr = dict(zip(("start", "z", "answer"), proles[t][j]))
            col = {k: mr[k] == pr[k] or bool({mr[k], pr[k]} & {x[o] for x in (mr, pr) for o in ("start", "z", "answer") if o != k})
                   for k in ("start", "z", "answer")}
            used = set(mr.values()) | set(pr.values())
            off = [c for c in COLOURS if c not in used][:2]
            P.append({"m": m, "p": m + 1 + j, "table": t, "m_roles": mr, "p_roles": pr, "collide": col, "off": off})
    return P


def test_colour_ll_logsumexp():
    owner = np.array([c for c in COLOURS for _ in range(2)])
    arr = np.log(np.full((1, 24), 0.5))
    out = dstat.colour_ll(arr, owner)
    assert np.allclose(out, 0.0)


def test_pair_D_exact_and_tracking():
    P = _pairs()
    LL = np.zeros((9, 12))
    for p in P:                                         # each render's own z gets +1: D_z = 2 for every pair
        LL[p["m"], CI[p["m_roles"]["z"]]] = 1.0
        LL[p["p"], CI[p["p_roles"]["z"]]] = 1.0
    v, tb = dstat.pair_D(LL, P, "z")
    assert np.allclose(v, 2.0)
    vo, _ = dstat.pair_D(LL, P, "off")
    assert np.allclose(vo, 0.0)                         # off-trajectory colours untouched -> 0
    m, ts = dstat.table_means(v, tb)
    assert list(ts) == [0, 1, 2] and np.allclose(m, 2.0)


def test_collision_free_skips():
    P = _pairs()
    LL = np.random.default_rng(0).normal(size=(9, 12))
    v, _ = dstat.pair_D(LL, P, "start")
    for p, x in zip(P, v):
        assert np.isnan(x) == p["collide"]["start"]


def test_derangement_null_is_centred():
    P = _pairs()
    LL = np.zeros((9, 12))
    for p in P:
        LL[p["m"], CI[p["m_roles"]["z"]]] = 1.0
        LL[p["p"], CI[p["p_roles"]["z"]]] = 1.0
    perms = AN.derangements(3, k=20)
    assert all((pm != np.arange(3)).all() for pm in perms)
    nul = AN.deranged_D(LL, P, "z", perms)
    assert np.all(np.abs(nul) < 2.0)                   # mismatched activations cannot reproduce the real D of 2


def test_holm_and_tost():
    assert AN.holm([0.001, 0.04, 0.03]) == [True, False, False]
    assert AN.holm([0.01, 0.02, 0.03]) == [True, True, True]
    idx = AN.boot_idx(40, B=2000, seed=1)
    a = np.random.default_rng(2).normal(0, 0.1, 40)
    c = AN.contrast(a, np.zeros(40), idx, delta=0.5)
    assert c["p_tost"] < 0.05 and c["p_two"] > 0.001      # equivalent within 0.5, not different
    c2 = AN.contrast(a + 2.0, np.zeros(40), idx, delta=0.5)
    assert c2["p_two"] < 0.01 and c2["p_tost"] > 0.5


def test_categories_and_mapping():
    assert AN.category(0.8, True, False, True) == "+"
    assert AN.category(-0.8, True, False, True) == "−"
    assert AN.category(0.1, False, True, True) == "≈"
    assert AN.category(0.3, True, False, True) == "unresolved"     # significant but below delta, not equivalent
    assert AN.category(0.8, True, False, False) == "uninformative"
    pred = {"Raw": {"P1": "≈", "P2": "≈", "P3": "≈", "P4": "≈", "P5": "≈"},
            "Visible": {"P1": "−", "P2": "+", "P3": "≈", "P4": "+", "P5": "≈"},
            "H_J": {"P1": "+", "P2": "+", "P3": "+", "P4": None, "P5": "+"}}
    vis = {"P1": "−", "P2": "+", "P3": "≈", "P4": "+", "P5": "≈"}
    m = AN.model_mapping(vis, pred)
    assert m == {"Raw": "contradicted", "Visible": "supported", "H_J": "contradicted"}
    hj = {"P1": "+", "P2": "+", "P3": "+", "P4": "unresolved", "P5": "+"}
    assert AN.model_mapping(hj, pred)["H_J"] == "supported"        # P4 is "no prediction" for H_J
    part = {"P1": "uninformative", "P2": "+", "P3": "≈", "P4": "+", "P5": "≈"}
    assert AN.model_mapping(part, pred)["Visible"] == "unresolved"  # an uninformative entry blocks support
    pred2 = {k: dict(v) for k, v in pred.items()}
    pred2["Visible"]["P5"] = None                                   # A1: no Visible P5 prediction when ratios differ
    assert AN.model_mapping({**vis, "P5": "+"}, pred2)["Visible"] == "supported"


def test_pc_stats():
    P = np.eye(12) * 3.0
    s = AN.pc_stats(P)
    assert s["loo_prior_ID"] == 1.0 and s["pairwise_positive"] == 66 and AN.pc_pass(s)
    Q = P.copy()
    Q[:, CI["teal"]] = np.nan                                       # template lens: teal column absent
    s2 = AN.pc_stats(Q)
    assert s2["n_colours"] == 11 and s2["pairs"] == 55
