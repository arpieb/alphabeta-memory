import numpy as np
import pytest

from alphabeta_memory import AlphaBetaMemory, MaxMemory, MinMemory

# Each test seeds its own generator. A module-level shared rng would make every
# test's fixture depend on which tests ran before it.
KINDS = [MaxMemory, MinMemory]


def patterns(seed, p, n):
    return np.random.default_rng(seed).integers(0, 2, size=(p, n), dtype=np.uint8)


# --------------------------------------------------------------------- recall
@pytest.mark.parametrize("cls", KINDS)
def test_autoassociative_perfect_recall(cls):
    X = patterns(0, 20, 40)
    mem = cls().fit(X)
    assert mem.autoassociative
    np.testing.assert_array_equal(mem.recall(X), X)
    np.testing.assert_array_equal(mem(X[3]), X[3])


@pytest.mark.parametrize("cls", KINDS)
def test_single_pattern_recall_is_1d(cls):
    X = patterns(1, 6, 12)
    mem = cls().fit(X)
    out = mem.recall(X[2])
    assert out.shape == (12,)
    assert out.dtype == np.uint8
    assert mem.recall(X).shape == (6, 12)


@pytest.mark.parametrize("cls", KINDS)
def test_recall_chunking_matches_unchunked(cls):
    # The recall chunk loop only runs >1 iteration when chunk_size < len(X);
    # a partial final chunk must land in the right output rows.
    X = patterns(2, 33, 20)
    mem = cls(chunk_size=8).fit(X)
    np.testing.assert_array_equal(mem.recall(X), cls(chunk_size=1000).fit(X).recall(X))
    np.testing.assert_array_equal(mem.recall(X), X)


# ----------------------------------------------------------------- noise/asym
@pytest.mark.parametrize("seed", range(5))
def test_max_memory_tolerates_additive_noise(seed):
    X = patterns(seed, 3, 60)
    mem = MaxMemory().fit(X)
    noisy = X[0].copy()
    noisy[np.argwhere(noisy == 0).ravel()[:5]] = 1  # 0 -> 1 flips
    np.testing.assert_array_equal(mem.recall(noisy), X[0])


@pytest.mark.parametrize("seed", range(5))
def test_min_memory_tolerates_subtractive_noise(seed):
    X = patterns(seed, 3, 60)
    mem = MinMemory().fit(X)
    noisy = X[0].copy()
    noisy[np.argwhere(noisy == 1).ravel()[:5]] = 0  # 1 -> 0 flips
    np.testing.assert_array_equal(mem.recall(noisy), X[0])


@pytest.mark.parametrize("seed", range(20))
def test_noise_degrades_in_one_direction_only(seed):
    """Exact recall has a capacity limit, but the *direction* of error does not.

    A max memory never drops a 1 under additive noise, and a min memory never
    adds one under subtractive noise -- at any load. This is the invariant that
    makes the max/min choice meaningful, so assert it rather than exact recall.
    """
    rng = np.random.default_rng(seed)
    p = int(rng.integers(2, 30))
    X = rng.integers(0, 2, size=(p, 40), dtype=np.uint8)
    i = int(rng.integers(0, p))

    noisy = X[i].copy()
    zeros = np.argwhere(noisy == 0).ravel()
    if len(zeros):
        noisy[rng.choice(zeros, min(len(zeros), 5), replace=False)] = 1
    assert (MaxMemory().fit(X).recall(noisy) >= X[i]).all()

    noisy = X[i].copy()
    ones = np.argwhere(noisy == 1).ravel()
    if len(ones):
        noisy[rng.choice(ones, min(len(ones), 5), replace=False)] = 0
    assert (MinMemory().fit(X).recall(noisy) <= X[i]).all()


@pytest.mark.parametrize("seed", range(5))
def test_wrong_memory_for_the_noise_direction_fails(seed):
    """The other half of the asymmetry: the wrong memory does NOT recover."""
    X = patterns(seed, 12, 60)
    subtractive = X[0].copy()
    subtractive[np.argwhere(subtractive == 1).ravel()[:5]] = 0
    assert not np.array_equal(MaxMemory().fit(X).recall(subtractive), X[0])

    additive = X[0].copy()
    additive[np.argwhere(additive == 0).ravel()[:5]] = 1
    assert not np.array_equal(MinMemory().fit(X).recall(additive), X[0])


# ------------------------------------------------------------------- hetero
def test_heteroassociative_recall_values():
    X = np.eye(4, dtype=np.uint8)
    Y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.uint8)
    mem = AlphaBetaMemory("min").fit(X, Y)
    assert not mem.autoassociative
    assert mem.weights.shape == (2, 4)
    assert (mem.input_dim, mem.output_dim) == (4, 2)
    np.testing.assert_array_equal(mem.recall(X), Y)
    np.testing.assert_array_equal(mem.recall(X[2]), Y[2])


def test_heteroassociative_max_memory_saturates_on_one_hot_inputs():
    """One-hot inputs are the degenerate case for a V memory: it saturates to
    all-2s and recalls all-ones. Pinned so the asymmetry is not mistaken for a
    regression, and so the min-memory test above is understood as the contrast."""
    X = np.eye(4, dtype=np.uint8)
    Y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.uint8)
    mem = AlphaBetaMemory("max").fit(X, Y)
    np.testing.assert_array_equal(mem.weights, np.full((2, 4), 2, dtype=np.uint8))
    np.testing.assert_array_equal(mem.recall(X), np.ones((4, 2), dtype=np.uint8))


# ----------------------------------------------------------------- learning
@pytest.mark.parametrize("cls", KINDS)
def test_learning_is_order_independent(cls):
    X = patterns(3, 20, 16)
    shuffled = X[np.random.default_rng(4).permutation(len(X))]
    np.testing.assert_array_equal(cls().fit(X).weights, cls().fit(shuffled).weights)


@pytest.mark.parametrize("cls", KINDS)
def test_learning_is_idempotent(cls):
    X = patterns(5, 12, 16)
    once = cls().fit(X)
    twice = cls().fit(X).partial_fit(X)
    np.testing.assert_array_equal(once.weights, twice.weights)
    assert twice.n_patterns == 24  # counts calls, weights unchanged


@pytest.mark.parametrize("cls", KINDS)
def test_partial_fit_matches_fit(cls):
    X = patterns(6, 30, 25)
    full = cls(chunk_size=7).fit(X)
    inc = cls().partial_fit(X[:10]).partial_fit(X[10:])
    np.testing.assert_array_equal(full.weights, inc.weights)
    np.testing.assert_array_equal(full.weights, cls(chunk_size=1).fit(X).weights)
    assert inc.n_patterns == 30


@pytest.mark.parametrize("cls", KINDS)
def test_fit_accepts_a_single_1d_pattern(cls):
    x = patterns(14, 1, 10)[0]
    mem = cls().fit(x)
    assert mem.weights.shape == (10, 10)
    assert mem.n_patterns == 1
    np.testing.assert_array_equal(mem.recall(x), x)
    np.testing.assert_array_equal(mem.weights, cls().fit(x[None, :]).weights)


def test_heteroassociative_fit_accepts_1d_x_and_y():
    x = np.array([1, 0, 1, 1], dtype=np.uint8)
    y = np.array([0, 1], dtype=np.uint8)
    mem = AlphaBetaMemory("min").fit(x, y)
    assert not mem.autoassociative
    assert mem.weights.shape == (2, 4)
    np.testing.assert_array_equal(mem.recall(x), y)


def test_fit_resets_previous_training():
    A, B = patterns(7, 5, 10), patterns(8, 5, 10)
    mem = MaxMemory().fit(A).fit(B)
    assert mem.n_patterns == 5
    np.testing.assert_array_equal(mem.weights, MaxMemory().fit(B).weights)


def test_bool_input_is_accepted():
    X = patterns(9, 6, 10).astype(bool)
    mem = MaxMemory().fit(X)
    out = mem.recall(X)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, X.astype(np.uint8))


# -------------------------------------------------------------------- state
def test_unfitted_state():
    mem = MaxMemory()
    assert not mem.is_fitted
    assert mem.n_patterns == 0
    assert "n_patterns=0" in repr(mem)
    with pytest.raises(RuntimeError):
        _ = mem.weights
    with pytest.raises(RuntimeError):
        _ = mem.autoassociative
    with pytest.raises(RuntimeError):
        _ = mem.input_dim
    with pytest.raises(RuntimeError):
        mem.recall(np.zeros(3, dtype=np.uint8))


def test_fitted_state_and_repr():
    mem = MinMemory().fit(patterns(10, 4, 7))
    assert mem.is_fitted and mem.n_patterns == 4
    assert (mem.input_dim, mem.output_dim) == (7, 7)
    r = repr(mem)
    assert "MinMemory" in r and "kind='min'" in r and "shape=(7, 7)" in r


@pytest.mark.parametrize("cls", KINDS)
def test_empty_batch_is_a_no_op(cls):
    """Regression: an empty batch used to set _autoassociative while leaving the
    weights unbuilt, so .autoassociative answered True but .weights raised."""
    mem = cls()
    assert mem.fit(np.zeros((0, 5), dtype=np.uint8)) is mem
    assert not mem.is_fitted
    assert mem.n_patterns == 0
    with pytest.raises(RuntimeError):
        _ = mem.autoassociative  # not half-initialised

    X = patterns(11, 4, 5)
    mem.fit(X)
    before = mem.weights.copy()
    mem.partial_fit(np.zeros((0, 5), dtype=np.uint8))
    np.testing.assert_array_equal(mem.weights, before)
    assert mem.n_patterns == 4


# ------------------------------------------------------------------- errors
def test_constructor_validation():
    with pytest.raises(ValueError):
        AlphaBetaMemory("median")
    with pytest.raises(ValueError):
        AlphaBetaMemory("max", chunk_size=0)
    with pytest.raises(ValueError):
        MaxMemory(chunk_size=-1)


def test_fit_input_validation():
    with pytest.raises(ValueError):
        MaxMemory().fit(np.zeros((2, 3, 4), dtype=np.uint8))       # 3-D
    with pytest.raises(ValueError):
        MaxMemory().fit(np.array([[0, 2]]))                        # non-binary
    with pytest.raises(ValueError):
        MaxMemory().fit(np.zeros((3, 4), np.uint8), np.zeros((2, 4), np.uint8))  # p mismatch


def test_shape_and_mode_locking():
    mem = MaxMemory().fit(np.zeros((2, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        mem.partial_fit(np.zeros((2, 3)), np.zeros((2, 2)))        # auto -> hetero
    with pytest.raises(ValueError):
        mem.partial_fit(np.zeros((2, 5), dtype=np.uint8))          # width change
    with pytest.raises(ValueError):
        mem.recall(np.zeros(4, dtype=np.uint8))                    # wrong input dim

    hetero = AlphaBetaMemory("max").fit(np.zeros((2, 3), np.uint8), np.zeros((2, 4), np.uint8))
    with pytest.raises(ValueError):
        hetero.partial_fit(np.zeros((2, 3), dtype=np.uint8))       # hetero -> auto


# ------------------------------------------------------------------ persist
@pytest.mark.parametrize("cls", KINDS)
def test_save_load_round_trip(cls, tmp_path):
    X = patterns(12, 5, 8)
    mem = cls().fit(X)
    path = str(tmp_path / "mem.npz")
    mem.save(path)

    loaded = AlphaBetaMemory.load(path)
    assert loaded.kind == mem.kind
    assert loaded.n_patterns == 5
    assert loaded.autoassociative is True
    np.testing.assert_array_equal(loaded.weights, mem.weights)
    np.testing.assert_array_equal(loaded.recall(X), X)


@pytest.mark.parametrize("caller", [AlphaBetaMemory, MaxMemory, MinMemory])
@pytest.mark.parametrize("cls", KINDS)
def test_load_works_called_on_any_subclass(caller, cls, tmp_path):
    """Regression: MaxMemory.load() raised TypeError, because load() passed
    kind= to a subclass __init__ that does not accept it."""
    X = patterns(13, 5, 8)
    mem = cls().fit(X)
    path = str(tmp_path / "mem.npz")
    mem.save(path)

    loaded = caller.load(path)          # called on the base class or either subclass
    assert loaded.kind == mem.kind      # kind comes from the file, not the caller
    assert type(loaded) is cls          # ...and so does the concrete class
    np.testing.assert_array_equal(loaded.weights, mem.weights)
    np.testing.assert_array_equal(loaded.recall(X), X)


def test_save_load_preserves_heteroassociative_flag(tmp_path):
    X, Y = np.eye(4, dtype=np.uint8), np.eye(4, dtype=np.uint8)[:, :2]
    mem = AlphaBetaMemory("max").fit(X, Y)
    path = str(tmp_path / "h.npz")
    mem.save(path)
    loaded = AlphaBetaMemory.load(path)
    assert loaded.autoassociative is False
    assert (loaded.input_dim, loaded.output_dim) == (4, 2)


def test_load_rejects_a_file_with_an_unknown_kind(tmp_path):
    path = str(tmp_path / "bad.npz")
    np.savez_compressed(path, M=np.zeros((2, 2), np.uint8), kind="median",
                        n_patterns=1, auto=True)
    with pytest.raises(ValueError, match="unknown kind"):
        AlphaBetaMemory.load(path)


def test_save_requires_a_fitted_memory(tmp_path):
    with pytest.raises(RuntimeError):
        MaxMemory().save(str(tmp_path / "x.npz"))
