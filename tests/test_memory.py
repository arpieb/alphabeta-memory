import numpy as np
import pytest

from alphabeta_memory import AlphaBetaMemory, MaxMemory, MinMemory

rng = np.random.default_rng(0)


@pytest.mark.parametrize("cls", [MaxMemory, MinMemory])
def test_autoassociative_perfect_recall(cls):
    X = rng.integers(0, 2, size=(20, 40), dtype=np.uint8)
    mem = cls().fit(X)
    assert mem.autoassociative
    np.testing.assert_array_equal(mem.recall(X), X)
    np.testing.assert_array_equal(mem(X[3]), X[3])


def test_max_memory_tolerates_additive_noise():
    X = rng.integers(0, 2, size=(5, 60), dtype=np.uint8)
    mem = MaxMemory().fit(X)
    noisy = X.copy()
    zeros = np.argwhere(noisy[0] == 0)[:5].ravel()
    noisy[0, zeros] = 1  # 0 -> 1 flips
    np.testing.assert_array_equal(mem.recall(noisy[0]), X[0])


def test_min_memory_tolerates_subtractive_noise():
    X = rng.integers(0, 2, size=(5, 60), dtype=np.uint8)
    mem = MinMemory().fit(X)
    noisy = X.copy()
    ones = np.argwhere(noisy[0] == 1)[:5].ravel()
    noisy[0, ones] = 0  # 1 -> 0 flips
    np.testing.assert_array_equal(mem.recall(noisy[0]), X[0])


def test_heteroassociative_shapes_and_simple_recall():
    X = np.eye(4, dtype=np.uint8)            # one-hot inputs
    Y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.uint8)
    mem = AlphaBetaMemory("max").fit(X, Y)
    assert not mem.autoassociative
    assert mem.weights.shape == (2, 4)
    assert mem.recall(X).shape == (4, 2)


def test_partial_fit_matches_fit():
    X = rng.integers(0, 2, size=(30, 25), dtype=np.uint8)
    full = MaxMemory(chunk_size=7).fit(X)
    inc = MaxMemory().partial_fit(X[:10]).partial_fit(X[10:])
    np.testing.assert_array_equal(full.weights, inc.weights)
    assert inc.n_patterns == 30


def test_errors():
    mem = MaxMemory()
    with pytest.raises(RuntimeError):
        _ = mem.weights
    with pytest.raises(ValueError):
        _ = AlphaBetaMemory("median")
    mem.fit(np.zeros((2, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        mem.recall(np.zeros(4, dtype=np.uint8))
    with pytest.raises(ValueError):
        mem.partial_fit(np.zeros((2, 3)), np.zeros((2, 2)))  # mixing auto/hetero


def test_save_load(tmp_path):
    X = rng.integers(0, 2, size=(5, 8), dtype=np.uint8)
    mem = MinMemory().fit(X)
    path = tmp_path / "mem.npz"
    mem.save(str(path))
    loaded = AlphaBetaMemory.load(str(path))
    assert loaded.kind == "min"
    np.testing.assert_array_equal(loaded.weights, mem.weights)
    np.testing.assert_array_equal(loaded.recall(X), X)
