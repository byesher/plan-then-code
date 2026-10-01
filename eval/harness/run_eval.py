"""Eval harness: multi-function module generation -> compile rate + test pass rate.

Usage (run from this harness directory):
  # self-check: reference solutions must score 100%
  python run_eval.py --problems ../benchmark/problems --generator reference
  # pre-generated solutions (generate on GPU, evaluate here)
  python run_eval.py --problems ../benchmark/problems --generator file --solutions-dir ../results/solutions
  # local HF model (on AutoDL)
  python run_eval.py --problems ../benchmark/problems --generator local --model Qwen/Qwen2.5-7B-Instruct
  # OpenAI-compatible API (DashScope / vLLM)
  python run_eval.py --problems ../benchmark/problems --generator openai --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --model qwen2.5-7b-instruct
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import uuid

from extract import extract_code
from prompt import render_prompt
from generators import (
    ReferenceGenerator,
    FileGenerator,
    LocalModelGenerator,
    OpenAICompatibleGenerator,
)


def load_problems(root: pathlib.Path) -> list[dict]:
    problems = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        jf = d / "problem.json"
        if not jf.exists():
            continue
        data = json.loads(jf.read_text(encoding="utf-8"))
        data["dir"] = str(d)
        data["dir_name"] = d.name
        problems.append(data)
    return problems


def count_tests(tests_file: pathlib.Path) -> int:
    return len(re.findall(r"^\s*def test_", tests_file.read_text(encoding="utf-8"), re.MULTILINE))


def check_compile(td: pathlib.Path, timeout: int) -> tuple[bool, str]:
    env = {**os.environ, "PYTHONPATH": str(td)}
    try:
        r = subprocess.run(
            [sys.executable, "-B", "-c", "import solution"],
            cwd=td, env=env, capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode == 0, r.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "compile check timed out"


def run_tests(td: pathlib.Path, timeout: int) -> tuple[bool, str]:
    env = {**os.environ, "PYTHONPATH": str(td)}
    try:
        r = subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "tests.py", "-q", "--tb=short", "-p", "no:cacheprovider"],
            cwd=td, env=env, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "tests timed out"
    return r.returncode == 0, (r.stdout + "\n" + r.stderr).strip()


def _scratch_dir() -> pathlib.Path:
    d = pathlib.Path(__file__).resolve().parent.parent / ".scratch"
    d.mkdir(parents=True, exist_ok=True)
    return d


def evaluate_one(problem: dict, generator, timeout: int, solutions_dir: pathlib.Path | None = None) -> dict:
    prompt = render_prompt(problem)
    raw = generator.generate(prompt, problem["dir_name"])
    code = extract_code(raw)

    if solutions_dir is not None:
        sdir = solutions_dir / problem["dir_name"]
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "raw.txt").write_text(raw, encoding="utf-8")
        (sdir / "solution.py").write_text(code, encoding="utf-8")

    result = {
        "id": problem["id"],
        "title": problem["title"],
        "domain": problem["domain"],
        "difficulty": problem["difficulty"],
        "n_functions": problem.get("n_functions"),
        "n_tests": 0,
        "compile_ok": False,
        "test_pass": False,
        "compile_err": "",
        "test_output": "",
    }
    if not code.strip():
        result["compile_err"] = "empty solution (no code extracted)"
        return result

    td = _scratch_dir() / uuid.uuid4().hex
    td.mkdir(parents=True)
    try:
        (td / "solution.py").write_text(code, encoding="utf-8")
        shutil.copy(pathlib.Path(problem["dir"]) / "tests.py", td / "tests.py")
        result["n_tests"] = count_tests(td / "tests.py")

        result["compile_ok"], result["compile_err"] = check_compile(td, timeout)
        if result["compile_ok"]:
            result["test_pass"], result["test_output"] = run_tests(td, timeout)
    finally:
        shutil.rmtree(td, ignore_errors=True)
    return result


def aggregate(results: list[dict]) -> dict:
    total = len(results)
    compiled = sum(1 for r in results if r["compile_ok"])
    passed = sum(1 for r in results if r["test_pass"])
    return {
        "total": total,
        "compiled": compiled,
        "passed": passed,
        "compile_rate": (compiled / total) if total else 0.0,
        "test_pass_rate": (passed / total) if total else 0.0,
        "test_pass_rate_among_compiled": (passed / compiled) if compiled else 0.0,
    }


def build_generator(args):
    if args.generator == "reference":
        return ReferenceGenerator(pathlib.Path(args.problems))
    if args.generator == "file":
        return FileGenerator(pathlib.Path(args.solutions_dir))
    if args.generator == "local":
        return LocalModelGenerator(args.model, max_new_tokens=args.max_new_tokens)
    if args.generator == "openai":
        return OpenAICompatibleGenerator(args.base_url, args.model, args.api_key,
                                         max_tokens=args.max_tokens)
    raise ValueError(f"unknown generator: {args.generator}")


def check_pytest_available() -> bool:
    try:
        r = subprocess.run(
            [sys.executable, "-B", "-c", "import pytest"],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True)
    ap.add_argument("--generator", required=True, choices=["reference", "file", "local", "openai"])
    ap.add_argument("--solutions-dir", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out-dir", default="../results")
    args = ap.parse_args()

    problems = load_problems(pathlib.Path(args.problems))
    if not problems:
        print("no problems found", file=sys.stderr)
        return 1

    if not check_pytest_available():
        print("ERROR: pytest is not installed in this Python environment.", file=sys.stderr)
        print("Install it with: pip install pytest", file=sys.stderr)
        return 1

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir = out_dir / "solutions"

    gen = build_generator(args)
    results = []
    for p in problems:
        r = evaluate_one(p, gen, args.timeout, solutions_dir)
        results.append(r)
        status = "PASS" if r["test_pass"] else ("COMPILE" if r["compile_ok"] else "FAIL")
        print(f"[{r['id']}] {status}  {r['title']}  ({r['n_tests']} tests)")

    agg = aggregate(results)
    print("\n=== aggregate ===")
    print(json.dumps(agg, indent=2, ensure_ascii=False))

    report = {"aggregate": agg, "results": results}
    out_path = out_dir / f"report_{args.generator}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
