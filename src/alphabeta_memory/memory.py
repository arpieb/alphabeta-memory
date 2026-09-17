"""Alpha-Beta associative memories.

Learning (p pattern pairs, x^mu in A^n, y^mu in A^m):

    V_ij = max_mu alpha(y_i^mu, x_j^mu)      # "max" memory
    L_ij = min_mu alpha(y_i^mu, x_j^mu)      # "min" memory

Recall for an input x in A^n:

    y_i = min_j beta(V_ij, x_j)              # max memory  (V  Delta_beta  x)
    y_i = max_j beta(L_ij, x_j)              # min memory  (L  Nabla_beta  x)

The autoassociative case (Y = X) has guaranteed perfect recall of the
training set.  Max memories are robust to *additive* noise (0 -> 1 flips),
min memories to *subtractive* noise (1 -> 0 flips).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .operators import alpha, beta, validate_binary

__all__ = ["AlphaBetaMemory", "Kind", "MaxMemory", "MinMemory"]

Kind = Literal["max", "min"]


class AlphaBetaMemory:
    """A max- or min-type Alpha-Beta associative memory.

    Parameters
    ----------
    kind
        ``"max"`` builds the V memory (robust to additive noise);
        ``"min"`` builds the Lambda memory (robust to subtractive noise).
    chunk_size
        Number of training patterns processed per vectorised block. Each block
        materialises a ``(chunk, m, n)`` uint8 tensor; lower this for large
        pattern dimensions.
    """

    def __init__(self, kind: Kind = "max", *, chunk_size: int = 256) -> None:
        if kind not in ("max", "min"):
            raise ValueError("kind must be 'max' or 'min'")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self.kind: Kind = kind
        self.chunk_size = chunk_size
        self._M: NDArray[np.uint8] | None = None
        self._n_patterns = 0
        self._autoassociative: bool | None = None

    # ------------------------------------------------------------------ props
    @property
    def weights(self) -> NDArray[np.uint8]:
        """The learned ``(m, n)`` matrix with entries in {0, 1, 2}."""
        if self._M is None:
            raise RuntimeError("memory has not been trained; call fit() first")
        return self._M

    @property
    def is_fitted(self) -> bool:
        return self._M is not None

    @property
    def n_patterns(self) -> int:
        return self._n_patterns

    @property
    def input_dim(self) -> int:
        return self.weights.shape[1]

    @property
    def output_dim(self) -> int:
        return self.weights.shape[0]

    @property
    def autoassociative(self) -> bool:
        if self._autoassociative is None:
            raise RuntimeError("memory has not been trained; call fit() first")
        return self._autoassociative

    def __repr__(self) -> str:
        shape = self._M.shape if self._M is not None else None
        return f"{type(self).__name__}(kind={self.kind!r}, shape={shape}, n_patterns={self._n_patterns})"

    # --------------------------------------------------------------- learning
    def fit(self, X: ArrayLike, Y: ArrayLike | None = None) -> AlphaBetaMemory:
        """Learn from scratch. ``X`` is ``(p, n)``; ``Y`` is ``(p, m)`` or ``None`` (autoassociative)."""
        self._M = None
        self._n_patterns = 0
        self._autoassociative = None
        return self.partial_fit(X, Y)

    def partial_fit(self, X: ArrayLike, Y: ArrayLike | None = None) -> AlphaBetaMemory:
        """Incrementally add patterns. Alpha-Beta learning is order-independent, so this is exact."""
        X = validate_binary(X, "X")
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2:
            raise ValueError("X must be 1-D (single pattern) or 2-D (patterns x features)")

        auto = Y is None
        if auto:
            Y = X
        else:
            Y = validate_binary(Y, "Y")
            if Y.ndim == 1:
                Y = Y[None, :]
            if Y.shape[0] != X.shape[0]:
                raise ValueError(f"X has {X.shape[0]} patterns but Y has {Y.shape[0]}")

        if X.shape[0] == 0:
            # No patterns: a true no-op. Returning early (rather than falling through
            # an empty chunk loop) keeps the object from being left half-initialised
            # with _autoassociative set but no weights.
            return self

        if self._autoassociative is None:
            self._autoassociative = auto
        elif self._autoassociative != auto:
            raise ValueError("cannot mix auto- and heteroassociative patterns in one memory")

        p, n = X.shape
        m = Y.shape[1]
        if self._M is not None and self._M.shape != (m, n):
            raise ValueError(f"pattern shape mismatch: memory is {self._M.shape}, got ({m}, {n})")

        reduce = np.max if self.kind == "max" else np.min
        for start in range(0, p, self.chunk_size):
            xb = X[start : start + self.chunk_size]
            yb = Y[start : start + self.chunk_size]
            # (chunk, m, n) outer product under alpha
            block = alpha(yb[:, :, None], xb[:, None, :], validate=False)
            block = reduce(block, axis=0)
            if self._M is None:
                self._M = block
            else:
                self._M = np.maximum(self._M, block) if self.kind == "max" else np.minimum(self._M, block)
        self._n_patterns += p
        return self

    # ----------------------------------------------------------------- recall
    def recall(self, X: ArrayLike) -> NDArray[np.uint8]:
        """Recall output patterns for one ``(n,)`` or many ``(k, n)`` inputs."""
        M = self.weights
        X = validate_binary(X, "X")
        single = X.ndim == 1
        if single:
            X = X[None, :]
        if X.ndim != 2 or X.shape[1] != M.shape[1]:
            raise ValueError(f"expected inputs of dimension {M.shape[1]}, got shape {X.shape}")

        reduce = np.min if self.kind == "max" else np.max
        out = np.empty((X.shape[0], M.shape[0]), dtype=np.uint8)
        for start in range(0, X.shape[0], self.chunk_size):
            xb = X[start : start + self.chunk_size]
            # (chunk, m, n) then reduce over n
            out[start : start + self.chunk_size] = reduce(beta(M[None, :, :], xb[:, None, :], validate=False), axis=2)
        return out[0] if single else out

    __call__ = recall

    # ---------------------------------------------------------------- persist
    def save(self, path: str) -> None:
        np.savez_compressed(path, M=self.weights, kind=self.kind,
                            n_patterns=self._n_patterns, auto=bool(self.autoassociative))

    @classmethod
    def load(cls, path: str) -> AlphaBetaMemory:
        """Restore a memory from ``save``.

        ``kind`` comes from the file, so this works called on any subclass: the
        returned object is the class matching the stored kind, not ``cls``.
        """
        data = np.load(path)
        kind = str(data["kind"])
        if kind not in ("max", "min"):
            raise ValueError(f"file declares unknown kind {kind!r}")
        mem = _KIND_CLASSES[kind]()
        mem._M = data["M"].astype(np.uint8)
        mem._n_patterns = int(data["n_patterns"])
        mem._autoassociative = bool(data["auto"])
        return mem


class MaxMemory(AlphaBetaMemory):
    """Alpha-Beta V memory: robust to additive noise."""

    def __init__(self, *, chunk_size: int = 256) -> None:
        super().__init__("max", chunk_size=chunk_size)


class MinMemory(AlphaBetaMemory):
    """Alpha-Beta Lambda memory: robust to subtractive noise."""

    def __init__(self, *, chunk_size: int = 256) -> None:
        super().__init__("min", chunk_size=chunk_size)


#: Maps a stored ``kind`` back to its convenience subclass; used by ``load``.
_KIND_CLASSES: dict[str, type[AlphaBetaMemory]] = {"max": MaxMemory, "min": MinMemory}
