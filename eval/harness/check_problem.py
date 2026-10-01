"""Self-check a single problem: run its tests against its reference solution.

Usage: python check_problem.py path/to/problem_dir
Exit 0 if tests pass, 1 otherwise.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import uuid


def _scratch_dir() -> pathlib.Path:
    d = pathlib.Path(__file__).resolve().parent.parent / ".scratch"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python check_problem.py <problem_dir>", file=sys.stderr)
        return 2
    d = pathlib.Path(sys.argv[1])
    for f in ("problem.json", "tests.py", "reference.py"):
        if not (d / f).exists():
            print(f"[{d.name}] missing {f}", file=sys.stderr)
            return 1
    td = _scratch_dir() / uuid.uuid4().hex
    td.mkdir(parents=True)
    try:
        shutil.copy(d / "reference.py", td / "solution.py")
        shutil.copy(d / "tests.py", td / "tests.py")
        (td / "__init__.py").write_text("", encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(td)}
        r = subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "tests.py", "-q", "-p", "no:cacheprovider"],
            cwd=td, env=env,
        )
        print(f"[{d.name}] {'PASS' if r.returncode == 0 else 'FAIL'}")
        return r.returncode
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
