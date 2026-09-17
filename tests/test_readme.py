"""Execute the Python examples in README.md.

This is a syntax-and-API guard, not a truth guard. It runs what the README shows
and checks the assertions the README makes, so an example that rots against a code
change fails here. A false statement written as prose or as a trailing comment
still passes, which is exactly why every claim in the README's example is written
as an `assert` rather than a `# -> like this` comment. Keep it that way.
"""

import pathlib
import re

import pytest

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"
BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _blocks() -> list[tuple[int, str]]:
    """Every ```python block in the README, as (0-based start line, source)."""
    text = README.read_text()
    return [(text[: m.start(1)].count("\n"), m.group(1)) for m in BLOCK_RE.finditer(text)]


_BLOCKS = _blocks()


def test_readme_contains_python_examples():
    """Guard against the parametrised test below silently becoming a no-op.

    An empty parametrize list generates zero tests rather than failing, so a
    README edit that deleted every example would otherwise pass in silence.
    """
    assert _BLOCKS, f"no ```python blocks found in {README}"


@pytest.mark.parametrize(
    ("start_line", "source"),
    _BLOCKS,
    ids=[f"README.md:{line + 1}" for line, _ in _BLOCKS],
)
def test_readme_example_runs(start_line, source, tmp_path, monkeypatch):
    """Run one block. Every block is checked, not just the first."""
    # Pad with blank lines so a traceback points at the real line in README.md.
    # Without this, line numbers are relative to the extracted block and resolve
    # to whatever happens to sit there in the file -- usually the wrong section.
    padded = "\n" * start_line + source
    code = compile(padded, str(README), "exec")

    # Each block runs standalone, in a scratch directory: a reader copy-pastes a
    # single block, so one must not depend on a previous one or write into the repo.
    monkeypatch.chdir(tmp_path)
    exec(code, {"__name__": "__readme__"})  # noqa: S102 - running the docs is the point
