import itertools

import numpy as np
import pytest

from alphabeta_memory import alpha, beta
from alphabeta_memory.operators import validate_binary, validate_ternary

A = (0, 1)
B = (0, 1, 2)


# ------------------------------------------------------------------ the tables
def test_alpha_table():
    assert alpha(0, 0) == 1
    assert alpha(0, 1) == 0
    assert alpha(1, 0) == 2
    assert alpha(1, 1) == 1


def test_beta_table():
    for x, y, expected in [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1), (2, 0, 1), (2, 1, 1)]:
        assert beta(x, y) == expected


def test_alpha_maps_into_B_and_beta_into_A():
    assert {int(alpha(x, y)) for x, y in itertools.product(A, A)} <= set(B)
    assert {int(beta(x, y)) for x, y in itertools.product(B, A)} <= set(A)


def test_beta_inverts_alpha_on_the_second_argument():
    """beta(alpha(y, x), x) == y for all x, y in A -- the identity the whole
    recall guarantee is built on."""
    for x, y in itertools.product(A, A):
        assert beta(alpha(y, x), x) == y


# ----------------------------------------------------------------- vectorised
def test_broadcast_and_dtype():
    out = alpha(np.array([[0], [1]]), np.array([[0, 1]]))
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, [[1, 0], [2, 1]])


def test_beta_broadcasts():
    out = beta(np.array([[0], [1], [2]]), np.array([[0, 1]]))
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, [[0, 0], [0, 1], [1, 1]])


def test_vectorised_matches_scalar():
    x, y = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
    np.testing.assert_array_equal(alpha(x, y), [alpha(a, b) for a, b in zip(x, y)])
    xb, yb = np.array([0, 1, 2, 2]), np.array([1, 1, 0, 1])
    np.testing.assert_array_equal(beta(xb, yb), [beta(a, b) for a, b in zip(xb, yb)])


def test_accepts_lists_and_bools():
    np.testing.assert_array_equal(alpha([0, 1], [1, 1]), [0, 1])
    np.testing.assert_array_equal(alpha(np.array([True, False]), np.array([False, False])), [2, 1])
    np.testing.assert_array_equal(beta([2, 0], [False, True]), [1, 0])


def test_validate_false_skips_checks_but_agrees_on_valid_input():
    x, y = np.array([0, 1, 1], np.uint8), np.array([1, 0, 1], np.uint8)
    np.testing.assert_array_equal(alpha(x, y, validate=False), alpha(x, y))
    xb = np.array([0, 1, 2], np.uint8)
    np.testing.assert_array_equal(beta(xb, y, validate=False), beta(xb, y))
    alpha(np.array([7]), np.array([0]), validate=False)  # no raise: caller's promise


# ----------------------------------------------------------------- validation
@pytest.mark.parametrize("bad", [2, -1, 1.5, np.nan])
def test_alpha_rejects_values_outside_A(bad):
    with pytest.raises(ValueError):
        alpha(bad, 0)
    with pytest.raises(ValueError):
        alpha(0, bad)


@pytest.mark.parametrize("bad", [3, -1, np.nan])
def test_beta_rejects_first_argument_outside_B(bad):
    with pytest.raises(ValueError):
        beta(bad, 0)


@pytest.mark.parametrize("bad", [2, -1, np.nan])
def test_beta_rejects_second_argument_outside_A(bad):
    with pytest.raises(ValueError):
        beta(0, bad)


def test_validators_coerce_to_uint8():
    assert validate_binary(np.array([True, False])).dtype == np.uint8
    assert validate_binary(np.array([0.0, 1.0])).dtype == np.uint8
    assert validate_ternary(np.array([0, 1, 2])).dtype == np.uint8
    with pytest.raises(ValueError, match="X must contain only 0/1"):
        validate_binary(np.array([5]), "X")
    with pytest.raises(ValueError, match="M must contain only 0/1/2"):
        validate_ternary(np.array([5]), "M")
