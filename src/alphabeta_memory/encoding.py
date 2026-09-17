"""Modified Johnson-Möbius code, the standard way to feed non-binary data to Alpha-Beta memories.

An integer ``k`` in ``[0, L]`` is encoded as ``L - k`` zeros followed by ``k`` ones,
e.g. with ``L = 5``:  0 -> 00000, 2 -> 00011, 5 -> 11111.

The code is monotone, so an increase in value is *additive* noise and a decrease is
*subtractive* noise -- exactly the two noise types the max/min memories tolerate.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["JohnsonMobiusEncoder"]


class JohnsonMobiusEncoder:
    """Encode integer feature matrices as concatenated Johnson-Möbius codes.

    Parameters
    ----------
    length
        Code length ``L`` per feature. If ``None``, inferred in :meth:`fit` as the
        maximum value observed per feature (one length per column).
    offset
        Subtracted from the data before encoding (per feature if array-like); if
        ``None``, inferred in :meth:`fit` as the per-feature minimum.
    """

    def __init__(
        self, length: int | ArrayLike | None = None, offset: int | ArrayLike | None = None
    ) -> None:
        self._length_arg = length
        self._offset_arg = offset
        self.lengths_: NDArray[np.int64] | None = None
        self.offsets_: NDArray[np.int64] | None = None

    def fit(self, X: ArrayLike) -> JohnsonMobiusEncoder:
        X = self._as_int_matrix(X)
        n_features = X.shape[1]
        self.offsets_ = (
            X.min(axis=0)
            if self._offset_arg is None
            else np.broadcast_to(np.asarray(self._offset_arg, dtype=np.int64), (n_features,)).copy()
        )
        shifted = X - self.offsets_
        if self._length_arg is None:
            self.lengths_ = shifted.max(axis=0).astype(np.int64)
        else:
            self.lengths_ = np.broadcast_to(
                np.asarray(self._length_arg, dtype=np.int64), (n_features,)
            ).copy()
        if (self.lengths_ < 0).any():
            raise ValueError("code lengths must be non-negative")
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.uint8]:
        self._check_fitted()
        X = self._as_int_matrix(X)
        shifted = X - self.offsets_
        if (shifted < 0).any() or (shifted > self.lengths_).any():
            raise ValueError("values out of range for the fitted encoder")
        cols = []
        for j, L in enumerate(self.lengths_):
            thresholds = np.arange(L, 0, -1)  # L, L-1, ..., 1
            cols.append((shifted[:, j : j + 1] >= thresholds[None, :]).astype(np.uint8))
        return np.concatenate(cols, axis=1) if cols else np.zeros((X.shape[0], 0), dtype=np.uint8)

    def fit_transform(self, X: ArrayLike) -> NDArray[np.uint8]:
        return self.fit(X).transform(X)

    def inverse_transform(self, B: ArrayLike) -> NDArray[np.int64]:
        """Decode by counting ones per block (tolerant of malformed codes)."""
        self._check_fitted()
        B = np.asarray(B)
        single = B.ndim == 1
        if single:
            B = B[None, :]
        expected = int(self.lengths_.sum())
        if B.shape[1] != expected:
            raise ValueError(f"expected {expected} bits, got {B.shape[1]}")
        out = np.empty((B.shape[0], len(self.lengths_)), dtype=np.int64)
        start = 0
        for j, L in enumerate(self.lengths_):
            out[:, j] = B[:, start : start + L].sum(axis=1) + self.offsets_[j]
            start += L
        return out[0] if single else out

    @property
    def n_bits(self) -> int:
        self._check_fitted()
        return int(self.lengths_.sum())

    # -------------------------------------------------------------- helpers
    def _check_fitted(self) -> None:
        if self.lengths_ is None or self.offsets_ is None:
            raise RuntimeError("encoder has not been fitted; call fit() first")

    @staticmethod
    def _as_int_matrix(X: ArrayLike) -> NDArray[np.int64]:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError("X must be 1-D or 2-D")
        if not np.issubdtype(X.dtype, np.integer) and not np.all(np.mod(X, 1) == 0):
            raise ValueError("Johnson-Möbius encoding requires integer values; quantise first")
        return X.astype(np.int64)
