import numpy as np
import pytest

from alphabeta_memory import alpha, beta


def test_alpha_table():
    assert alpha(0, 0) == 1
    assert alpha(0, 1) == 0
    assert alpha(1, 0) == 2
    assert alpha(1, 1) == 1


def test_beta_table():
    for x, y, expected in [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1), (2, 0, 1), (2, 1, 1)]:
        assert beta(x, y) == expected


def test_broadcast_and_dtype():
    out = alpha(np.array([[0], [1]]), np.array([[0, 1]]))
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, [[1, 0], [2, 1]])


def test_validation():
    with pytest.raises(ValueError):
        alpha(2, 0)
    with pytest.raises(ValueError):
        beta(3, 0)
