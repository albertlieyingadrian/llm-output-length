"""No undefined names anywhere in the package, or in the tests.

This exists because a real one shipped. `extract.completions_dir` reached the
cluster with `completions` used but never bound, so every re-encode run died
with NameError after loading the model. Syntax checks do not catch it: the file
parses, imports, and only fails on the branch that uses the name.

pyflakes does catch it, statically, in under a second.

`tests/` is scanned too, and for the same reason: a test that raises NameError
does not report a passing guard, it reports a failure whose message is about the
missing import rather than about the property under test. That is a slow way to
learn you forgot `import torch`.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [ROOT / "src" / "miol", ROOT / "tests"]


def _pyflakes_findings(target):
    out = subprocess.run(
        [sys.executable, "-m", "pyflakes", str(target)],
        capture_output=True, text=True,
    ).stdout
    return [ln for ln in out.splitlines() if "undefined name" in ln]


def _ast_findings(target):
    """A cheap stand-in for pyflakes, for environments without it.

    Coarser -- it pools every binding in a file rather than tracking scopes, so
    it under-reports -- but it catches the case this guard exists for: a name
    used and never bound anywhere in the file. Better than the `importorskip`
    that used to stand here, which turned the whole guard into a silent no-op on
    any machine without pyflakes installed. A guard nobody notices is off is
    worse than no guard: this one skipped on the dev container for months.
    """
    import ast
    import builtins

    out = []
    for path in sorted(Path(target).rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(), str(path))
        except SyntaxError as exc:
            out.append(f"{path}:{exc.lineno}: syntax error: {exc.msg}")
            continue
        bound = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)):
                bound.add(n.name)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                bound.add(n.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    bound.add((a.asname or a.name).split(".")[0])
            elif isinstance(n, ast.arg):
                bound.add(n.arg)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                bound.add(n.name)
            elif isinstance(n, ast.Global):
                bound.update(n.names)
        for n in ast.walk(tree):
            if (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                    and n.id not in bound):
                out.append(f"{path}:{n.lineno}: undefined name '{n.id}'")
    return out


@pytest.mark.parametrize("target", TARGETS, ids=lambda p: p.name)
def test_no_undefined_names(target):
    try:
        import pyflakes  # noqa: F401
        bad = _pyflakes_findings(target)
    except ImportError:
        bad = _ast_findings(target)
    assert not bad, "undefined names:\n" + "\n".join(bad)


def test_the_guard_can_actually_fail(tmp_path):
    """Proves whichever checker ran is looking at something.

    The `importorskip` this replaced meant the guard reported `s` -- skipped --
    on any machine without pyflakes, which reads as a pass in a summary line.
    """
    (tmp_path / "boom.py").write_text("def f():\n    return undefined_thing\n")
    try:
        import pyflakes  # noqa: F401
        bad = _pyflakes_findings(tmp_path)
    except ImportError:
        bad = _ast_findings(tmp_path)
    assert bad and "undefined_thing" in bad[0], bad
