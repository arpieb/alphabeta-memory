# alphabeta-memory

NumPy implementation of **Alpha-Beta associative memories** (Yáñez-Márquez, CIC-IPN, 2002):
max (V) and min (Λ) memories, auto- and heteroassociative, plus the modified
Johnson-Möbius encoder for non-binary data.

## Install

```bash
uv add alphabeta-memory        # or: pip install alphabeta-memory
```

## Usage

```python
import numpy as np
from alphabeta_memory import MaxMemory, MinMemory, JohnsonMobiusEncoder

X = np.random.default_rng(0).integers(0, 2, size=(4, 64), dtype=np.uint8)

mem = MaxMemory().fit(X)  # autoassociative; robust to 0->1 noise
assert (mem.recall(X) == X).all()  # perfect recall of the training set

noisy = X[0] | (np.random.default_rng(1).random(64) < 0.1)  # additive noise
assert (mem.recall(noisy) == X[0]).all()

# A min memory is the mirror image: robust to 1->0 noise.
faded = X[0] & (np.random.default_rng(2).random(64) > 0.1)  # subtractive noise
assert (MinMemory().fit(X).recall(faded) == X[0]).all()

# Heteroassociative: map each pattern to a one-hot label.
Y = np.eye(4, dtype=np.uint8)
hmem = MinMemory().fit(X, Y)
assert (hmem.recall(X[3]) == Y[3]).all()

# Integer data, via the Johnson-Mobius code.
V = [[3, 10], [0, 12], [7, 11]]
enc = JohnsonMobiusEncoder()
B = enc.fit_transform(V)
assert (enc.inverse_transform(MaxMemory().fit(B).recall(B)) == V).all()
```

Every line above is asserted, so the example either holds or raises.

Recall of the *training set* is exact for any number of stored patterns. Recall from a
*noisy* input is capacity-limited: it is reliable for a handful of 64-bit patterns and
degrades as you store more, so pick the memory that matches your noise direction and
keep an eye on the load. What does not degrade is the direction of the error — a max
memory never drops a 1, and a min memory never adds one.

Low-level operators are exported as `alpha(x, y)` and `beta(x, y)`.

## Development

```bash
uv sync                # creates .venv with dev deps
uv run pytest
uv run ruff check .      # CI also gates on `uv run ruff format --check .`
uv build               # -> dist/*.whl, dist/*.tar.gz
uv publish             # needs UV_PUBLISH_TOKEN or --token
```

## References

- C. Yáñez-Márquez, *Memorias asociativas basadas en relaciones de orden y operadores binarios*, PhD thesis, CIC-IPN, 2002.
- Yáñez-Márquez et al., "Alpha-Beta associative memories" (various), Computación y Sistemas.
