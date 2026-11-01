"""Grouped cross-validation keeps every session entirely in one fold."""

import numpy as np

from mochi_analysis.evaluate import cross_validate, grouped_folds
from mochi_analysis.io import PairData


def fake_data(n_sessions=15, picks=40, seed=0):
    rng = np.random.default_rng(seed)
    n = n_sessions * picks
    session = np.repeat([f"s{i:02d}" for i in range(n_sessions)], picks).astype(object)
    X = rng.uniform(size=(n, 20)) - rng.uniform(size=(n, 20))
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-X[:, 0] * 3))).astype(int)
    return PairData("home", X, y, session, session.copy(), np.zeros(n, dtype=int))


def test_no_session_in_two_folds():
    data = fake_data()
    for seed in range(10):
        folds = grouped_folds(data.session, n_folds=5, seed=seed)
        for s in data.sessions:
            assert len(set(folds[data.session == s])) == 1


def test_train_and_test_sessions_disjoint_in_every_fold():
    data = fake_data(n_sessions=13)  # uneven split on purpose
    folds = grouped_folds(data.session, n_folds=5, seed=1)
    sizes = []
    for f in range(folds.max() + 1):
        train = set(data.session[folds != f])
        test = set(data.session[folds == f])
        assert train.isdisjoint(test)
        assert train | test == set(data.sessions)
        sizes.append(len(test))
    assert max(sizes) - min(sizes) <= 1


def test_every_pick_predicted_once_and_out_of_fold():
    data = fake_data()
    cv = cross_validate(data, n_folds=5, seed=0)
    assert np.all(np.isfinite(cv.p_heldout))
    # Each fold model must differ from the others (they saw different sessions).
    biases = {round(m.bias, 12) for m in cv.models}
    assert len(biases) == len(cv.models)


def test_fewer_sessions_than_folds_is_leave_one_session_out():
    data = fake_data(n_sessions=3)
    folds = grouped_folds(data.session, n_folds=5, seed=0)
    assert folds.max() + 1 == 3
