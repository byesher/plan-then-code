# -*- coding: utf-8 -*-
"""
两段式评测（全量落盘版）：题目 →(step1 伪代码)→(step2 代码)，prompt 与训练逐字一致。

对比：
  baseline ：base 模型 + 直接写代码
  sft2step ：SFT 模型 + 两步（先伪代码，再照伪代码实现）

【这次重点：全量落盘，供离线分析】
每题都存：
  baseline/<id>/code_0.py          —— baseline 抽出的代码
  sft2step/<id>/plan_0.txt         —— step1 生成的伪代码
  sft2step/<id>/code_0.py          —— step2 生成的代码（抽出）
  report.json                      —— 逐题结果 + 汇总

运行（AutoDL，eval 目录下）：
  ADAPTER_PATH=~/autodl-tmp/plan-then-code-lora-plan python eval_plan.py
输出目录：eval/results/eval_plan/
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/autodl-tmp/Qwen2.5-7B-Instruct"))
DEV_PATH = BASE_DIR / ".." / "data" / "taco_train" / "dev_raw.jsonl"
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "")

NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "1"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2048"))
MAX_EVAL_TESTS = int(os.environ.get("MAX_EVAL_TESTS", "20"))
TEST_TIMEOUT = 10
OUT_ROOT = Path(os.environ.get("OUT_ROOT", BASE_DIR / "results"))
RUN_DIR = OUT_ROOT / "eval_plan"

PLAN_PROMPT = "【题目】\n{question}\n\n请写出分步伪代码（算法计划），不要写代码。"
IMPL_PROMPT = "【题目】\n{question}\n\n【伪代码】\n{plan}\n\n请严格照着伪代码写出完整可运行的 Python 程序。"
BASELINE_PROMPT = "【题目】\n{question}\n\n请写出完整可运行的 Python 程序，直接输出代码，不要解释。"


def load_dev():
    items = []
    with open(DEV_PATH, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                items.append(json.loads(ln))
    return items


def load_model(adapter_path=None):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tok, model


def generate(tok, model, messages, n, temperature, max_new):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = tok(text, return_tensors="pt").to(device)
    outs = []
    with torch.no_grad():
        for _ in range(n):
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=True,
                temperature=temperature, top_p=0.95, pad_token_id=tok.eos_token_id,
            )
            gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            outs.append(gen.strip())
    return outs


def extract_code(text):
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\w*\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _norm_out(text):
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln.lower() if ln.lower() in ("true", "false") else ln for ln in lines]
    return "\n".join(lines)


def run_tests(code, tests):
    td = tempfile.mkdtemp(prefix="eval_")
    try:
        with open(os.path.join(td, "solution.py"), "w", encoding="utf-8") as f:
            f.write(code)
        n_pass = 0
        total = len(tests[:MAX_EVAL_TESTS])
        first_err = ""
        cases = []
        for i, case in enumerate(tests[:MAX_EVAL_TESTS]):
            rec = {"idx": i, "passed": False, "error": "", "actual": "", "expected": (case["output"] or "")[:200]}
            try:
                r = subprocess.run([sys.executable, "-B", "solution.py"], cwd=td,
                                   input=case["input"], capture_output=True, text=True, timeout=TEST_TIMEOUT)
            except subprocess.TimeoutExpired:
                rec["error"] = "timeout"
                first_err = f"case{i}:timeout"
                cases.append(rec)
                break
            rec["actual"] = (r.stdout or "")[:200]
            rec["stderr"] = (r.stderr or "")[:200]
            if r.returncode != 0:
                rec["error"] = "runtime"
                first_err = f"case{i}:runtime({(r.stderr or '').strip()[:80]})"
                cases.append(rec)
                break
            if _norm_out(r.stdout) != _norm_out(case["output"]):
                rec["error"] = "mismatch"
                first_err = f"case{i}:mismatch"
                cases.append(rec)
                break
            rec["passed"] = True
            n_pass += 1
            cases.append(rec)
        return n_pass, total, first_err, cases
    finally:
        shutil.rmtree(td, ignore_errors=True)


def eval_baseline(tok, model, items):
    mode_dir = RUN_DIR / "baseline"
    mode_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for it in items:
        q = it["question"]
        tests = it.get("tests") or []
        pdir = mode_dir / it["id"]
        pdir.mkdir(parents=True, exist_ok=True)
        msgs = [{"role": "user", "content": BASELINE_PROMPT.format(question=q)}]
        gens = generate(tok, model, msgs, NUM_SAMPLES, TEMPERATURE, MAX_NEW_TOKENS)
        best = 0
        best_err = ""
        best_cases = []
        for si, g in enumerate(gens):
            (pdir / f"raw_{si}.txt").write_text(g, encoding="utf-8")   # 原始输出（含围栏）
            code = extract_code(g)
            (pdir / f"code_{si}.py").write_text(code, encoding="utf-8")
            n_pass, total, first_err, cases = run_tests(code, tests)
            if n_pass > best:
                best = n_pass
                best_err = first_err
                best_cases = cases
        results.append({"id": it["id"], "solved": best == len(tests[:MAX_EVAL_TESTS]) and len(tests) > 0,
                        "n_pass": best, "n_tests": len(tests[:MAX_EVAL_TESTS]), "err": best_err,
                        "cases": best_cases})
    return results


def eval_sft2step(tok, model, items):
    mode_dir = RUN_DIR / "sft2step"
    mode_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for it in items:
        q = it["question"]
        tests = it.get("tests") or []
        pdir = mode_dir / it["id"]
        pdir.mkdir(parents=True, exist_ok=True)
        if not tests:
            results.append({"id": it["id"], "solved": False, "n_pass": 0, "n_tests": 0, "err": "no-tests"})
            continue
        # step1：出伪代码
        plan_msgs = [{"role": "user", "content": PLAN_PROMPT.format(question=q)}]
        plan = generate(tok, model, plan_msgs, 1, TEMPERATURE, MAX_NEW_TOKENS)[0]
        (pdir / "plan_0.txt").write_text(plan, encoding="utf-8")
        # step2：照伪代码实现
        impl_msgs = [{"role": "user", "content": IMPL_PROMPT.format(question=q, plan=plan)}]
        code_raws = generate(tok, model, impl_msgs, NUM_SAMPLES, TEMPERATURE, MAX_NEW_TOKENS)
        best = 0
        best_err = ""
        best_cases = []
        for si, raw in enumerate(code_raws):
            (pdir / f"raw_{si}.txt").write_text(raw, encoding="utf-8")  # step2 原始输出（含围栏）
            code = extract_code(raw)
            (pdir / f"code_{si}.py").write_text(code, encoding="utf-8")
            n_pass, total, first_err, cases = run_tests(code, tests)
            if n_pass > best:
                best = n_pass
                best_err = first_err
                best_cases = cases
        results.append({"id": it["id"], "solved": best == len(tests[:MAX_EVAL_TESTS]),
                        "n_pass": best, "n_tests": len(tests[:MAX_EVAL_TESTS]), "err": best_err,
                        "cases": best_cases})
    return results


def summarize(results):
    n = len(results)
    solved = sum(1 for r in results if r["solved"])
    return {"n": n, "solved": solved, "pass@1": solved / n if n else 0.0}


def main():
    items = load_dev()
    print(f"dev 集 {len(items)} 题，NUM_SAMPLES={NUM_SAMPLES}，temperature={TEMPERATURE}\n")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    tok, model = load_model()
    summaries = {}
    all_results = {}

    print("跑 baseline（直接写代码）...")
    r = eval_baseline(tok, model, items)
    summaries["baseline"] = summarize(r)
    all_results["baseline"] = r
    print(f"  baseline: pass@1={summaries['baseline']['pass@1']:.1%} ({summaries['baseline']['solved']}/{summaries['baseline']['n']})\n")

    if ADAPTER_PATH:
        print("跑 sft2step（两步：先伪代码再实现）...")
        del model
        torch.cuda.empty_cache()
        _, model2 = load_model(ADAPTER_PATH)
        r = eval_sft2step(tok, model2, items)
        summaries["sft2step"] = summarize(r)
        all_results["sft2step"] = r
        print(f"  sft2step: pass@1={summaries['sft2step']['pass@1']:.1%} ({summaries['sft2step']['solved']}/{summaries['sft2step']['n']})\n")
    else:
        print("⚠️ 没设 ADAPTER_PATH，跳过 sft2step。")

    print("=" * 50)
    for k, v in summaries.items():
        print(f"{k:10s} pass@1={v['pass@1']:.1%} ({v['solved']}/{v['n']})")
    if "sft2step" in summaries and "baseline" in summaries:
        d = summaries["sft2step"]["pass@1"] - summaries["baseline"]["pass@1"]
        print(f"\nSFT 两步效应 = {d:+.1%}")

    report = {
        "summary": summaries,
        "results": all_results,
        "config": {"adapter": ADAPTER_PATH or None, "num_samples": NUM_SAMPLES,
                   "temperature": TEMPERATURE, "max_new_tokens": MAX_NEW_TOKENS},
    }
    out = RUN_DIR / "report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n全量结果已存到：{RUN_DIR}")
    print(f"  report.json（逐题+汇总）、baseline/<id>/code_0.py、sft2step/<id>/plan_0.txt + code_0.py")


if __name__ == "__main__":
    main()
