"""Design 001 amendment A2 (probe repair): the span-solved logistic regression matches the full fit, the binary metric is
per-fold, and the repaired null / control repeat the full procedure (C tuned by inner CV). CPU only, synthetic data."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "001_leverage_profile"))
import leverage_profile as LP  # noqa: E402


def _data(n=120, d=600, k=3, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(k), n // k)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X[:, :5] += y[:, None] * 0.8                       # a planted, decodable signal
    g = np.repeat(np.arange(n // 4), 4)                # groups of 4 rows ("tables")
    return X, y, g


@pytest.mark.parametrize("C", [1e-3, 1e-2])
def test_span_fit_matches_full_fit(C):
    X, y, _ = _data()
    a = LP._fit_full(X[:90], y[:90], C, True)
    b = LP._SpanLR(C).fit(X[:90], y[:90])
    da, db = a.decision_function(X[90:]), b.decision_function(X[90:])
    assert np.abs(da - db).max() < 0.02 * np.abs(da).max()
    assert np.mean(a.predict(X[90:]) == b.predict(X[90:])) >= 0.95


def test_binary_metric_is_per_fold():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    fold = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    pred = np.array([0.1, 0.9, 0.2, 0.8, 10.1, 10.9, 10.2, 10.8])   # perfect within folds, offset between folds
    assert LP._metric(y, pred, False, fold) == 1.0
    assert LP._metric(y, pred, False) < 1.0                          # the pooled AUC is hurt by the fold offset


def test_null_and_control_jobs_tune_C(monkeypatch):
    seen = []

    def fake_cv(X, y, g, multi, seed=0, C_fixed=None):
        seen.append(C_fixed)
        return np.zeros(len(y)), np.zeros(len(y), int), [1e-2] * 5, []
    monkeypatch.setattr(LP, "_cv_scores", fake_cv)
    monkeypatch.setattr(LP, "_probe_X", lambda: np.zeros((4, 1, 1, 3), np.float32))
    y = np.array(["a", "b", "a", "b"])
    for kind in ("real", "ctrl", "null"):
        LP._probe_job((("multi", "z", 0, 0), kind, 0, True, np.arange(4), 0, 0, y, np.array([0, 0, 1, 1])))
    assert seen == [None, None, None]


def test_seeds_differ_across_targets():
    assert LP._seed("ctrl", "member", "red", 2, 3, 0) != LP._seed("ctrl", "member", "blue", 2, 3, 0)
