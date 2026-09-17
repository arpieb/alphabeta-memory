import numpy as np
import pytest

from alphabeta_memory import JohnsonMobiusEncoder, MaxMemory, MinMemory


# ------------------------------------------------------------------ encoding
def test_single_feature_code():
    enc = JohnsonMobiusEncoder(length=5, offset=0).fit([[0]])
    np.testing.assert_array_equal(
        enc.transform([[0], [2], [5]]),
        [[0, 0, 0, 0, 0], [0, 0, 0, 1, 1], [1, 1, 1, 1, 1]],
    )


def test_code_is_monotone():
    """The whole design rests on this: +1 in value == exactly one 0->1 flip."""
    enc = JohnsonMobiusEncoder(length=6, offset=0).fit([[0]])
    codes = enc.transform(np.arange(7)[:, None])
    for k in range(6):
        diff = codes[k + 1].astype(int) - codes[k].astype(int)
        assert (diff >= 0).all() and diff.sum() == 1


def test_round_trip_multi_feature():
    X = np.array([[3, 10], [0, 12], [7, 11]])
    enc = JohnsonMobiusEncoder()
    B = enc.fit_transform(X)
    assert enc.n_bits == 7 + 2
    np.testing.assert_array_equal(enc.lengths_, [7, 2])
    np.testing.assert_array_equal(enc.offsets_, [0, 10])
    np.testing.assert_array_equal(enc.inverse_transform(B), X)


def test_fit_infers_offset_and_length_per_feature():
    enc = JohnsonMobiusEncoder().fit([[5, 100], [9, 103]])
    np.testing.assert_array_equal(enc.offsets_, [5, 100])
    np.testing.assert_array_equal(enc.lengths_, [4, 3])


def test_explicit_length_and_offset_broadcast_across_features():
    enc = JohnsonMobiusEncoder(length=3, offset=1).fit([[1, 1], [4, 4]])
    np.testing.assert_array_equal(enc.lengths_, [3, 3])
    np.testing.assert_array_equal(enc.offsets_, [1, 1])
    enc = JohnsonMobiusEncoder(length=[2, 3], offset=[0, 1]).fit([[0, 1]])
    np.testing.assert_array_equal(enc.lengths_, [2, 3])
    np.testing.assert_array_equal(enc.offsets_, [0, 1])
    assert enc.n_bits == 5


def test_one_d_input_is_treated_as_a_single_column():
    enc = JohnsonMobiusEncoder()
    B = enc.fit_transform([1, 2, 3])
    assert B.shape == (3, 2)
    np.testing.assert_array_equal(enc.inverse_transform(B), [[1], [2], [3]])


def test_constant_feature_gets_zero_length_code():
    enc = JohnsonMobiusEncoder()
    X = np.array([[5, 1], [5, 3]])
    B = enc.fit_transform(X)
    np.testing.assert_array_equal(enc.lengths_, [0, 2])
    assert B.shape == (2, 2)  # the constant column contributes no bits
    np.testing.assert_array_equal(enc.inverse_transform(B), X)


def test_integral_floats_are_accepted():
    np.testing.assert_array_equal(
        JohnsonMobiusEncoder(length=2, offset=0).fit([[0]]).transform(np.array([[1.0], [2.0]])),
        [[0, 1], [1, 1]],
    )


# ------------------------------------------------------------------ decoding
def test_inverse_transform_single_row_is_1d():
    enc = JohnsonMobiusEncoder(length=3, offset=0).fit([[0]])
    out = enc.inverse_transform(np.array([0, 1, 1], dtype=np.uint8))
    assert out.shape == (1,)
    np.testing.assert_array_equal(out, [2])


def test_inverse_transform_counts_ones_in_malformed_codes():
    """Noisy recall produces codes that are not valid runs; decoding must still
    degrade gracefully rather than mis-parse the run structure."""
    enc = JohnsonMobiusEncoder(length=4, offset=0).fit([[0]])
    np.testing.assert_array_equal(enc.inverse_transform(np.array([[1, 0, 1, 0]], np.uint8)), [[2]])
    np.testing.assert_array_equal(enc.inverse_transform(np.array([[1, 1, 1, 1]], np.uint8)), [[4]])


def test_inverse_transform_applies_the_offset():
    enc = JohnsonMobiusEncoder().fit([[10], [14]])
    np.testing.assert_array_equal(enc.inverse_transform(np.array([[0, 0, 0, 0]], np.uint8)), [[10]])
    np.testing.assert_array_equal(enc.inverse_transform(np.array([[1, 1, 1, 1]], np.uint8)), [[14]])


# -------------------------------------------------------------------- errors
def test_value_above_range():
    enc = JohnsonMobiusEncoder(length=3, offset=0).fit([[0]])
    with pytest.raises(ValueError):
        enc.transform([[4]])


def test_value_below_offset():
    enc = JohnsonMobiusEncoder().fit([[5], [9]])
    with pytest.raises(ValueError):
        enc.transform([[3]])


def test_negative_length_rejected():
    with pytest.raises(ValueError):
        JohnsonMobiusEncoder(length=-1).fit([[0]])


def test_non_integer_values_rejected():
    with pytest.raises(ValueError):
        JohnsonMobiusEncoder().fit_transform(np.array([[1.5]]))


def test_three_dimensional_input_rejected():
    with pytest.raises(ValueError):
        JohnsonMobiusEncoder().fit(np.zeros((2, 2, 2), dtype=int))


def test_inverse_transform_wrong_bit_count():
    enc = JohnsonMobiusEncoder(length=3, offset=0).fit([[0]])
    with pytest.raises(ValueError):
        enc.inverse_transform(np.zeros((1, 5), dtype=np.uint8))


def test_unfitted_encoder():
    enc = JohnsonMobiusEncoder()
    with pytest.raises(RuntimeError):
        enc.transform([[1]])
    with pytest.raises(RuntimeError):
        enc.inverse_transform(np.zeros((1, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError):
        _ = enc.n_bits


# ---------------------------------------------------------------- end-to-end
def test_end_to_end_with_memory():
    X = np.array([[1, 4], [3, 0], [2, 2]])
    enc = JohnsonMobiusEncoder(length=4, offset=0)
    B = enc.fit_transform(X)
    mem = MaxMemory().fit(B)
    np.testing.assert_array_equal(enc.inverse_transform(mem.recall(B)), X)


def test_value_increase_is_additive_noise_a_max_memory_absorbs():
    """The encoder's monotonicity is what couples it to the max/min asymmetry:
    a value that drifts up is exactly additive noise on the code."""
    X = np.array([[2, 5], [6, 1]])
    enc = JohnsonMobiusEncoder(length=8, offset=0)
    B = enc.fit_transform(X)
    mem = MaxMemory().fit(B)
    drifted = enc.transform(np.array([[3, 6]]))  # both features +1
    np.testing.assert_array_equal(enc.inverse_transform(mem.recall(drifted)), [[2, 5]])


def test_value_decrease_is_subtractive_noise_a_min_memory_absorbs():
    X = np.array([[2, 5], [6, 1]])
    enc = JohnsonMobiusEncoder(length=8, offset=0)
    B = enc.fit_transform(X)
    mem = MinMemory().fit(B)
    drifted = enc.transform(np.array([[1, 4]]))  # both features -1
    np.testing.assert_array_equal(enc.inverse_transform(mem.recall(drifted)), [[2, 5]])
