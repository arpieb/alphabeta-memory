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

X = np.random.default_rng(0).integers(0, 2, size=(10, 64), dtype=np.uint8)

mem = MaxMemory().fit(X)          # autoassociative; robust to 0->1 noise
assert (mem.recall(X) == X).all() # perfect recall of the training set

noisy = X[0] | (np.random.default_rng(1).random(64) < 0.1)
mem.recall(noisy)                 # -> X[0]

# Heteroassociative
Y = np.eye(10, dtype=np.uint8)
hmem = MinMemory().fit(X, Y)
hmem.recall(X[3])

# Integer data
enc = JohnsonMobiusEncoder().fit([[3, 10], [0, 12], [7, 11]])
B = enc.transform([[3, 10]])
enc.inverse_transform(MaxMemory().fit(B).recall(B))
```

Low-level operators are exported as `alpha(x, y)` and `beta(x, y)`.

## Development

```bash
uv sync                # creates .venv with dev deps
uv run pytest
uv run ruff check .
uv build               # -> dist/*.whl, dist/*.tar.gz
uv publish             # needs UV_PUBLISH_TOKEN or --token
```

## References

- C. Yáñez-Márquez, *Memorias asociativas basadas en relaciones de orden y operadores binarios*, PhD thesis, CIC-IPN, 2002.
- Yáñez-Márquez et al., "Alpha-Beta associative memories" (various), Computación y Sistemas.
