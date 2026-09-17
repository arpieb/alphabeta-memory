# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                      # create .venv with dev deps (editable install of the package)
uv run pytest                # full suite (testpaths = tests/)
uv run pytest tests/test_memory.py::test_partial_fit_matches_fit   # single test
uv run ruff check .          # lint (line-length 100, target py310)
uv run ruff check --fix .    # autofix (import sorting is the usual offender)
uv build                     # -> dist/*.whl, dist/*.tar.gz
uv publish                   # needs UV_PUBLISH_TOKEN or --token

# Coverage is not a dev dependency; pull it in per-invocation.
uv run --with pytest-cov pytest --cov=alphabeta_memory --cov-branch --cov-report=term-missing
```

## Architecture

A small NumPy library implementing Alpha-Beta associative memories (Yáñez-Márquez,
CIC-IPN, 2002). Three layers, bottom-up:

**`operators.py`** — the two binary operators the whole model rests on:
`alpha: {0,1}² → {0,1,2}` and `beta: {0,1,2} × {0,1} → {0,1}`. Both have closed
arithmetic forms (`alpha(x,y) = 1 + x - y`; `beta(x,y) = [x + y ≥ 2]`), which is why
everything downstream is a broadcast NumPy expression rather than a table lookup.
`validate_binary` / `validate_ternary` are the domain checks; both operators take
`validate=False` so hot inner loops skip revalidation of already-checked arrays.
The identity the recall guarantee is built on is `beta(alpha(y, x), x) == y`.

**`memory.py`** — `AlphaBetaMemory(kind="max"|"min")` with `MaxMemory`/`MinMemory`
subclasses. Learning is `M_ij = reduce_mu alpha(y_i^mu, x_j^mu)` (max for V memories,
min for Λ); recall is the dual reduction over `beta(M_ij, x_j)`. Both learning and
recall materialise a `(chunk, m, n)` uint8 tensor per block and reduce it — `chunk_size`
is the memory/speed knob, and is the only reason the code has loops at all.

Key invariants to preserve when editing:
- Learning is an idempotent, order-independent max/min reduction, so `partial_fit`
  is exact, not approximate; `fit` is just reset + `partial_fit`. Tests assert all
  three: shuffled `fit` matches, re-fitting the same patterns is a no-op on the
  weights, and chunked `fit` matches two-call `partial_fit`.
- A memory is locked to auto- or heteroassociative on first fit and refuses to mix.
- An empty batch is a **true no-op**: `partial_fit` validates its input, then returns
  before touching any state. Do not let it fall through to the chunk loop — that is
  how it used to leave `_autoassociative` set with no weights built, so
  `.autoassociative` answered `True` while `.weights` raised.
- `save`/`load` round-trip through `np.savez_compressed`. `load` is a classmethod
  that reads `kind` from the file and returns **the subclass matching that kind**,
  not `cls` — so it works called on `AlphaBetaMemory`, `MaxMemory` or `MinMemory`.
  It must not construct via `cls(kind=...)`: the subclass `__init__`s do not accept
  `kind`, which is what previously made `MaxMemory.load()` raise `TypeError`.
  `chunk_size` is deliberately *not* persisted; a reloaded memory gets the default.

Noise behaviour, which is what makes the max/min choice meaningful:
- Max memories tolerate additive (0→1) noise; min memories tolerate subtractive (1→0)
  noise. This asymmetry drives which memory a caller picks.
- *Exact* recall under noise is capacity-limited, and the limit is tight. For 60-bit
  patterns with 5 flipped bits it goes from 100% at 3 stored patterns to ~39% at 8
  and 0% by 20. Do not write tests that assert exact recall under noise without
  pinning the pattern count — that is how the original suite ended up on an operating
  point that failed for ~10% of rng seeds.
- What *does* hold at every load is the direction of the error: a max memory never
  drops a 1 under additive noise (`recall >= true`), and a min memory never adds one
  under subtractive noise (`recall <= true`). Prefer these as assertions.
- Heteroassociative recall has **no** perfect-recall guarantee, unlike the
  autoassociative case. One-hot inputs are the degenerate case for a V memory: they
  saturate it to all-2s, which then recalls all-ones regardless of `Y`. A Λ memory
  recalls the same fixture exactly. Both behaviours are pinned by tests.

**`encoding.py`** — `JohnsonMobiusEncoder` is how non-binary data enters the system.
`k` in `[0, L]` becomes `L-k` zeros then `k` ones. The code is deliberately monotone:
a value increase is exactly additive noise and a decrease exactly subtractive, matching
the max/min robustness above — so a feature drifting up is absorbed by a `MaxMemory`
and one drifting down by a `MinMemory`. Per-feature `lengths_`/`offsets_` are inferred
from the data in `fit` unless passed to the constructor; a constant feature infers
`L = 0` and contributes no bits. `inverse_transform` decodes by *counting* ones per
block rather than parsing the run structure, so it degrades gracefully on the
malformed codes a noisy recall can produce.

The public surface is re-exported from `__init__.py`; keep `__all__` and `__version__`
(mirrored in `pyproject.toml`) in sync when adding exports. `tests/test_package.py`
enforces both.

## Testing

The suite is at 100% line and branch coverage; keep it there when adding code.

- **Never share an rng across tests.** Each test seeds its own
  `np.random.default_rng(seed)` via the `patterns()` helper in `test_memory.py`. A
  module-level generator makes every test's fixture depend on which tests ran before
  it, so adding, reordering or `-k`-filtering a test silently changes the data.
- Tests for `max` vs `min` behaviour are parametrized over `KINDS` rather than
  duplicated; anything asserting the asymmetry is the exception and names its kind.
- When fixing a bug, verify the new test actually fails with the fix reverted.
  Several branches here are only reachable through argument shapes that are easy to
  miss — `fit()` on a single 1-D pattern went uncovered by the entire original suite.
