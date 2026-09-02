import numpy as np
import pytest

from alphabeta_memory import JohnsonMobiusEncoder, MaxMemory


def test_single_feature_code():
    enc = JohnsonMobiusEncoder(length=5, offset=0).fit([[0]])
    np.testing.assert_array_equal(enc.transform([[0], [2], [5]]),
                                  [[0, 0, 0, 0, 0], [0, 0, 0, 1, 1], [1, 1, 1, 1, 1]])


def test_round_trip_multi_feature():
    X = np.array([[3, 10], [0, 12], [7, 11]])
    enc = JohnsonMobiusEncoder()
    B = enc.fit_transform(X)
    assert enc.n_bits == 7 + 2
    np.testing.assert_array_equal(enc.inverse_transform(B), X)


def test_out_of_range():
    enc = JohnsonMobiusEncoder(length=3, offset=0).fit([[0]])
    with pytest.raises(ValueError):
        enc.transform([[4]])


def test_end_to_end_with_memory():
    X = np.array([[1, 4], [3, 0], [2, 2]])
    enc = JohnsonMobiusEncoder(length=4, offset=0).fit(X)
    B = enc.fit_transform(X)
    mem = MaxMemory().fit(B)
    np.testing.assert_array_equal(enc.inverse_transform(mem.recall(B)), X)
