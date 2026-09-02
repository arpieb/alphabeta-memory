"""The two binary operators that define Alpha-Beta associative memories.

Sets:  A = {0, 1}    B = {0, 1, 2}

alpha : A x A -> B          beta : B x A -> A
    alpha(0, 0) = 1             beta(0, 0) = 0   beta(0, 1) = 0
    alpha(0, 1) = 0             beta(1, 0) = 0   beta(1, 1) = 1
    alpha(1, 0) = 2             beta(2, 0) = 1   beta(2, 1) = 1
    alpha(1, 1) = 1

Both admit closed arithmetic forms, which is what makes a vectorised
NumPy implementation trivial:

    alpha(x, y) = 1 + x - y
    beta(x, y)  = 1  if x + y >= 2  else 0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["alpha", "beta", "validate_binary", "validate_ternary"]


def validate_binary(a: ArrayLike, name: str = "array") -> NDArray[np.uint8]:
    """Coerce to uint8 and check every element is in A = {0, 1}."""
    arr = np.asarray(a)
    if arr.dtype == bool:
        return arr.astype(np.uint8)
    if not np.isin(arr, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0/1 values")
    return arr.astype(np.uint8)


def validate_ternary(a: ArrayLike, name: str = "array") -> NDArray[np.uint8]:
    """Coerce to uint8 and check every element is in B = {0, 1, 2}."""
    arr = np.asarray(a)
    if not np.isin(arr, (0, 1, 2)).all():
        raise ValueError(f"{name} must contain only 0/1/2 values")
    return arr.astype(np.uint8)


def alpha(x: ArrayLike, y: ArrayLike, *, validate: bool = True) -> NDArray[np.uint8]:
    """Element-wise alpha operator, broadcasting over its inputs. Returns values in B."""
    if validate:
        x, y = validate_binary(x, "x"), validate_binary(y, "y")
    return (1 + np.asarray(x, dtype=np.int8) - np.asarray(y, dtype=np.int8)).astype(np.uint8)


def beta(x: ArrayLike, y: ArrayLike, *, validate: bool = True) -> NDArray[np.uint8]:
    """Element-wise beta operator, broadcasting over its inputs. Returns values in A."""
    if validate:
        x, y = validate_ternary(x, "x"), validate_binary(y, "y")
    return ((np.asarray(x, dtype=np.int8) + np.asarray(y, dtype=np.int8)) >= 2).astype(np.uint8)
