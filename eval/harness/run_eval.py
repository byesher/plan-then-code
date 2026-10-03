"""Eval harness: multi-function module generation -> compile rate + test pass rate.

Usage (run from this harness directory):
  # self-check: reference solutions must score 100%
  python run_eval.py --problems ../benchmark/problems --generator reference
  # pre-generated solutions (generate on GPU, evaluate here)
  python run_eval.py --problems ../benchmark/problems --generator file --solutions-dir ../results/runs/<run_name>
  # local HF model (on AutoDL)
  python run_eval.py --problems ../benchmark/problems --generator local --model Qwen/Qwen2.5-7B-Instruct
  # OpenAI-compatible API (DashScope / vLLM)
  python run_eval.py --problems ../benchmark/problems --generator openai --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --model qwen2.5-7b-instruct
"""
from __future__ import annotations

import argparse
import ast
import datetime
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import uuid

from extract import extract_code
from prompt import render_prompt, render_io_prompt
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


STDLIB_MODULES = set(sys.stdlib_module_names)


def analyze_solution(code: str, problem: dict) -> dict:
    """Parse the model's code with AST; report missing interface names and non-stdlib imports."""
    info = {"missing_names": [], "non_stdlib_imports": [], "nested_in_class": []}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return info  # compile check will report this case

    defined = set()
    class_methods: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
            class_methods[node.name] = {
                sub.name for sub in node.body
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
            }

    missing: list[str] = []
    for unit in problem.get("interface", []):
        name = unit["name"]
        if unit.get("type") == "class":
            if name not in defined:
                missing.append(name)
            else:
                for m in unit.get("methods", []):
                    if m["name"] not in class_methods.get(name, set()):
                        missing.append(f"{name}.{m['name']}")
        else:
            if name not in defined:
                missing.append(name)
    info["missing_names"] = missing
    all_nested = {m for methods in class_methods.values() for m in methods}
    info["nested_in_class"] = [n for n in missing if n in all_nested]

    non_stdlib = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in STDLIB_MODULES:
                    non_stdlib.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top not in STDLIB_MODULES:
                    non_stdlib.add(top)
    info["non_stdlib_imports"] = sorted(non_stdlib)
    return info


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


def evaluate_one(problem: dict, generator, timeout: int, solutions_dir: pathlib.Path | None = None, sample_index: int | None = None) -> dict:
    prompt = render_prompt(problem)
    raw = generator.generate(prompt, problem["dir_name"])
    code = extract_code(raw)

    if solutions_dir is not None:
        sdir = solutions_dir / problem["dir_name"]
        sdir.mkdir(parents=True, exist_ok=True)
        suffix = "" if (sample_index is None or sample_index == 0) else f"_{sample_index}"
        (sdir / f"raw{suffix}.txt").write_text(raw, encoding="utf-8")
        (sdir / f"solution{suffix}.py").write_text(code, encoding="utf-8")

    analysis = analyze_solution(code, problem)
    result = {
        "id": problem["id"],
        "title": problem["title"],
        "domain": problem["domain"],
        "difficulty": problem["difficulty"],
        "n_functions": problem.get("n_functions"),
        "n_tests": 0,
        "compile_ok": False,
        "test_pass": False,
        "interface_ok": len(analysis["missing_names"]) == 0,
        "missing_names": analysis["missing_names"],
        "nested_in_class": analysis["nested_in_class"],
        "non_stdlib_imports": analysis["non_stdlib_imports"],
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


def _normalize_output(text: str) -> str:
    """Normalize I/O output: strip per-line whitespace, lowercase standalone true/false.

    Models often print Python bools (``True``/``False``) while the spec means the
    lowercase strings ``true``/``false``; treat them as equal.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln.lower() if ln.lower() in ("true", "false") else ln for ln in lines]
    return "\n".join(lines)


def evaluate_io(problem: dict, generator, timeout: int, solutions_dir: pathlib.Path | None = None, sample_index: int | None = None) -> dict:
    prompt = render_io_prompt(problem)
    raw = generator.generate(prompt, problem["dir_name"])
    code = extract_code(raw)

    if solutions_dir is not None:
        sdir = solutions_dir / problem["dir_name"]
        sdir.mkdir(parents=True, exist_ok=True)
        suffix = "" if (sample_index is None or sample_index == 0) else f"_{sample_index}"
        (sdir / f"raw{suffix}.txt").write_text(raw, encoding="utf-8")
        (sdir / f"solution{suffix}.py").write_text(code, encoding="utf-8")

    cases = problem.get("tests", [])
    result = {
        "id": problem["id"],
        "title": problem["title"],
        "domain": problem["domain"],
        "difficulty": problem["difficulty"],
        "n_tests": len(cases),
        "n_passed": 0,
        "compile_ok": False,
        "test_pass": False,
        "interface_ok": True,
        "missing_names": [],
        "non_stdlib_imports": [],
        "test_output": "",
    }
    if not code.strip():
        result["test_output"] = "empty solution (no code extracted)"
        return result

    td = _scratch_dir() / uuid.uuid4().hex
    td.mkdir(parents=True)
    try:
        (td / "solution.py").write_text(code, encoding="utf-8")
        env = {**os.environ}
        passed = 0
        notes = []
        for case in cases:
            try:
                r = subprocess.run(
                    [sys.executable, "-B", "solution.py"],
                    cwd=td, env=env, input=case["input"], capture_output=True, text=True, timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                notes.append("[case] timed out")
                continue
            result["compile_ok"] = True
            if r.returncode != 0:
                notes.append(f"[case] runtime error: {r.stderr.strip()[:300]}")
                continue
            if _normalize_output(r.stdout) == _normalize_output(case["output"]):
                passed += 1
            else:
                notes.append(f"[case] mismatch:\n  expected={case['output']!r}\n  actual  ={r.stdout!r}")
        result["test_pass"] = passed == len(cases) and len(cases) > 0
        result["n_passed"] = passed
        result["test_output"] = f"{passed}/{len(cases)} passed" if result["test_pass"] else "\n".join(notes)
    finally:
        shutil.rmtree(td, ignore_errors=True)
    return result


def aggregate(results: list[dict]) -> dict:
    total = len(results)
    compiled = sum(1 for r in results if r["compile_ok"])
    passed = sum(1 for r in results if r["test_pass"])
    interface_ok = sum(1 for r in results if r.get("interface_ok"))
    non_stdlib = sum(1 for r in results if r.get("non_stdlib_imports"))
    n_samples = results[0].get("n_samples", 1) if results else 1
    total_passed_samples = sum(r.get("n_passed_samples", 1 if r["test_pass"] else 0) for r in results)
    total_cases = sum(r.get("n_tests", 0) * r.get("n_samples", n_samples) for r in results)
    total_passed_cases = sum(r.get("n_passed_cases", 0) for r in results)
    return {
        "total": total,
        "compiled": compiled,
        "passed": passed,
        "compile_rate": (compiled / total) if total else 0.0,
        "test_pass_rate": (passed / total) if total else 0.0,
        "test_pass_rate_among_compiled": (passed / compiled) if compiled else 0.0,
        "interface_ok": interface_ok,
        "interface_ok_rate": (interface_ok / total) if total else 0.0,
        "non_stdlib_count": non_stdlib,
        "non_stdlib_rate": (non_stdlib / total) if total else 0.0,
        "test_pass_rate_among_interface_ok": (passed / interface_ok) if interface_ok else 0.0,
        "n_samples": n_samples,
        "pass_at_1_avg": (total_passed_samples / (total * n_samples)) if total else 0.0,
        "case_pass_rate": (total_passed_cases / total_cases) if total_cases else 0.0,
    }


def _status_of(r: dict) -> str:
    if r.get("test_pass"):
        return "PASS"
    if not r.get("compile_ok"):
        return "COMPILE-FAIL"
    if r.get("missing_names"):
        return "IFACE-MISS"
    if r.get("non_stdlib_imports"):
        return "NON-STDLIB"
    return "LOGIC-FAIL"


def build_generator(args):
    if args.generator == "reference":
        return ReferenceGenerator(pathlib.Path(args.problems))
    if args.generator == "file":
        return FileGenerator(pathlib.Path(args.solutions_dir))
    if args.generator == "local":
        return LocalModelGenerator(args.model, max_new_tokens=args.max_new_tokens,
                                   load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit,
                                   temperature=args.temperature)
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
    ap.add_argument("--load-in-8bit", action="store_true", help="load model in 8-bit (bitsandbytes)")
    ap.add_argument("--load-in-4bit", action="store_true", help="load model in 4-bit (bitsandbytes)")
    ap.add_argument("--temperature", type=float, default=0.0, help="sampling temperature (0 = greedy)")
    ap.add_argument("--num-samples", type=int, default=1, help="samples per problem (pass@k)")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out-dir", default="../results")
    ap.add_argument("--run-name", default=None, help="run tag; defaults to <generator>_<timestamp>")
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
    base = args.run_name or args.generator
    run_name = f"{base}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    solutions_dir = out_dir / "runs" / run_name

    gen = build_generator(args)
    results = []
    for p in problems:
        eval_fn = evaluate_io if p.get("mode") == "io" else evaluate_one
        samples = []
        for i in range(args.num_samples):
            samples.append(eval_fn(p, gen, args.timeout, solutions_dir, sample_index=i))
            if args.num_samples > 1:
                print(f"    [{p['id']}] sample {i+1}/{args.num_samples}", flush=True)
        r = samples[0]
        r["n_samples"] = args.num_samples
        r["n_passed_samples"] = sum(1 for s in samples if s["test_pass"])
        r["n_passed_cases"] = sum(s.get("n_passed", 0) for s in samples)
        r["test_pass"] = any(s["test_pass"] for s in samples)
        # 保存题目元信息 + 每个样本的明细（之前只留了 samples[0]，其余样本信息全丢了）
        r["dir_name"] = p.get("dir_name")
        r["_bin"] = p.get("_bin")
        r["_ref_lines"] = p.get("_ref_lines")
        r["samples"] = [
            {
                "sample_index": i,
                "n_passed": s.get("n_passed"),
                "n_tests": s.get("n_tests"),
                "test_pass": s.get("test_pass"),
                "compile_ok": s.get("compile_ok"),
                "status": _status_of(s),
                "test_output": (s.get("test_output") or "")[:500],
            }
            for i, s in enumerate(samples)
        ]
        results.append(r)
        status = _status_of(r)
        extra = ""
        if r.get("missing_names"):
            extra += f"  missing={r['missing_names']}"
        if r.get("non_stdlib_imports"):
            extra += f"  non_stdlib={r['non_stdlib_imports']}"
        if args.num_samples > 1:
            extra += f"  pass@k={r['n_passed_samples']}/{args.num_samples}"
        print(f"[{r['id']}] {status}  {r['title']}  ({r['n_tests']} tests){extra}")

    agg = aggregate(results)
    print("\n=== aggregate ===")
    print(json.dumps(agg, indent=2, ensure_ascii=False))

    report = {
        "aggregate": agg,
        "results": results,
        "run_name": run_name,
        "config": {
            "generator": args.generator,
            "model": args.model,
            "temperature": args.temperature,
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_tokens,
            "problems": args.problems,
        },
    }
    out_path = out_dir / f"report_{run_name}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
