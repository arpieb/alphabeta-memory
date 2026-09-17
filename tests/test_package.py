import pathlib
import sys

import pytest

import alphabeta_memory

# tomllib is stdlib only from 3.11, but the package supports 3.10.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only on Python 3.10
    tomllib = None


def test_all_names_are_importable():
    for name in alphabeta_memory.__all__:
        assert hasattr(alphabeta_memory, name), name


def test_all_is_sorted_and_complete():
    public = {n for n in vars(alphabeta_memory) if not n.startswith("_")}
    public -= {"encoding", "memory", "operators"}  # submodules
    assert public <= set(alphabeta_memory.__all__)
    assert alphabeta_memory.__all__ == sorted(alphabeta_memory.__all__)


@pytest.mark.skipif(tomllib is None, reason="tomllib requires Python 3.11+")
def test_version_matches_pyproject():
    """CLAUDE.md requires __version__ and pyproject.toml stay in sync."""
    root = pathlib.Path(__file__).resolve().parent.parent
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    assert alphabeta_memory.__version__ == pyproject["project"]["version"]


def test_requires_python_floor_is_satisfiable():
    assert sys.version_info >= (3, 10)
