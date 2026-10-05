# -*- coding: utf-8 -*-
"""
评测脚本 v2：对比 base /「prompt 伪代码」/「SFT 伪代码」，且【完整落盘】便于排查

【对比口径】
  baseline：base 模型 + 直接写代码（不要求伪代码）
  prompt  ：base 模型 + 「先写伪代码再写代码」
  sft     ：LoRA 微调后 + 「先写伪代码再写代码」
差值：提示词效应 = prompt-baseline；SFT 效应 = sft-prompt。

【这次重点：完整记录，能排查"为什么 0%"】
每题每样本都存：
  raw_{i}.txt    —— 模型原始输出（含伪代码 + 代码，一字不改）
  code_{i}.py    —— 抽出来的代码
  result.json    —— 每个测试用例的实跑结果（passed / error / returncode / actual / expected / stderr）
最后汇总成 report.json。这样 baseline=0% 到底是「抽代码 bug」还是「真不会」，打开文件一看便知。

【运行】（在 AutoDL，eval 目录下）
  python eval_sft.py                                              # baseline + prompt
  ADAPTER_PATH=~/autodl-tmp/plan-then-code-lora python eval_sft.py  # 全三组
输出目录：eval/results/sft_eval_<时间戳>/

依赖：torch transformers peft（无 trl）
"""
import datetime
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

# ── 路径与参数 ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/autodl-tmp/Qwen2.5-7B-Instruct"))
DEV_PATH = BASE_DIR / ".." / "data" / "taco_train" / "dev_raw.jsonl"
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "")          # 空 = 不跑 sft 组
MODE = os.environ.get("MODE", "all")                       # all / baseline / prompt / sft
OUT_ROOT = Path(os.environ.get("OUT_ROOT", BASE_DIR / "results"))

NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "1"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2048"))
MAX_EVAL_TESTS = int(os.environ.get("MAX_EVAL_TESTS", "20"))
TEST_TIMEOUT = 10

# 与训练一致的 prompt 模板
SYSTEM_PROMPT = "你是一个编程助手。收到题目后，先写出分步伪代码（算法计划），再写出完整可运行的 Python 程序。代码必须放在 ```python ... ``` 代码块内。"
PLAIN_USER = "【题目】\n{question}\n\n请写出完整可运行的 Python 程序，直接输出代码，不要解释。"
PLAN_USER = "【题目】\n{question}\n\n请先写出分步伪代码，再写出完整可运行的 Python 程序。"


def load_dev():
    items = []
    with open(DEV_PATH, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
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


def build_messages(mode, question):
    if mode == "baseline":
        return [{"role": "user", "content": PLAIN_USER.format(question=question)}]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PLAN_USER.format(question=question)},
    ]


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
    """抽代码：```python``` 围栏 → ``` 围栏 → 「【代码】」之后 → 整段兜底。"""
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\w*\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    idx = text.find("【代码】")
    if idx != -1:
        return text[idx + len("【代码】"):].strip()
    return text.strip()


def _norm_out(text):
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln.lower() if ln.lower() in ("true", "false") else ln for ln in lines]
    return "\n".join(lines)


def run_tests_detailed(code, tests):
    """跑【所有】样例（只有 timeout 才停），返回 (cases, n_passed)。
    每个 case 记录：passed / error / returncode / actual / expected / stderr。"""
    td = tempfile.mkdtemp(prefix="eval_")
    cases = []
    n_passed = 0
    try:
        with open(os.path.join(td, "solution.py"), "w", encoding="utf-8") as f:
            f.write(code)
        for i, case in enumerate(tests[:MAX_EVAL_TESTS]):
            rec = {"idx": i, "passed": False, "error": "", "actual": "", "expected": (case["output"] or "")[:200]}
            try:
                r = subprocess.run(
                    [sys.executable, "-B", "solution.py"],
                    cwd=td, input=case["input"], capture_output=True, text=True, timeout=TEST_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                rec["error"] = "timeout"
                cases.append(rec)
                break
            rec["returncode"] = r.returncode
            rec["actual"] = (r.stdout or "")[:200]
            rec["stderr"] = (r.stderr or "")[:200]
            if r.returncode != 0:
                rec["error"] = "runtime"
                cases.append(rec)
                continue
            if _norm_out(r.stdout) == _norm_out(case["output"]):
                rec["passed"] = True
                n_passed += 1
            else:
                rec["error"] = "mismatch"
            cases.append(rec)
        return cases, n_passed
    finally:
        shutil.rmtree(td, ignore_errors=True)


def status_of(sample):
    if sample["test_pass"]:
        return "PASS"
    if not sample["code_excerpt"].strip():
        return "EMPTY"
    for c in sample["cases"]:
        if c["error"] == "timeout":
            return "TIMEOUT"
        if c["error"] == "runtime":
            return "RUNTIME-ERR"
    return "MISMATCH"


def evaluate_mode(tok, model, items, mode, mode_dir):
    """跑一个 mode，每题每样本存 raw.txt / code.py / 逐样例结果，返回 results。"""
    mode_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for it in items:
        pid = it["id"]
        pdir = mode_dir / pid
        pdir.mkdir(parents=True, exist_ok=True)
        question = it["question"]
        tests = it.get("tests") or []
        msgs = build_messages(mode, question)
        gens = generate(tok, model, msgs, NUM_SAMPLES, TEMPERATURE, MAX_NEW_TOKENS)
        samples = []
        for si, g in enumerate(gens):
            code = extract_code(g)
            raw_file = pdir / f"raw_{si}.txt"
            code_file = pdir / f"code_{si}.py"
            raw_file.write_text(g, encoding="utf-8")
            code_file.write_text(code, encoding="utf-8")
            cases, n_passed = run_tests_detailed(code, tests)
            sample = {
                "sample_index": si,
                "raw_file": str(raw_file.relative_to(OUT_ROOT / run_name_global)),
                "code_file": str(code_file.relative_to(OUT_ROOT / run_name_global)),
                "raw_excerpt": g[:300],
                "code_excerpt": code[:300],
                "n_passed": n_passed,
                "n_tests": len(cases),
                "test_pass": n_passed == len(cases) and len(cases) > 0,
                "cases": cases,
            }
            sample["status"] = status_of(sample)
            samples.append(sample)
        n_passed_cases = max((s["n_passed"] for s in samples), default=0)
        results.append({
            "id": pid,
            "difficulty": it.get("difficulty"),
            "n_tests": len(tests[:MAX_EVAL_TESTS]),
            "n_passed_cases": n_passed_cases,
            "test_pass": any(s["test_pass"] for s in samples),
            "samples": samples,
        })
    return results


run_name_global = None  # 供 evaluate_mode 里拼相对路径用


def summarize(results):
    n = len(results)
    solved = sum(1 for r in results if r["test_pass"])
    total_cases = sum(r["n_tests"] for r in results)
    passed_cases = sum(r["n_passed_cases"] for r in results)
    return {
        "n": n,
        "pass@1": solved / n if n else 0.0,
        "solved": solved,
        "case_pass": passed_cases / total_cases if total_cases else 0.0,
        "passed_cases": passed_cases,
        "total_cases": total_cases,
    }


def main():
    global run_name_global
    items = load_dev()
    print(f"dev 集 {len(items)} 题，NUM_SAMPLES={NUM_SAMPLES}，temperature={TEMPERATURE}\n")

    run_name = f"sft_eval_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    run_name_global = run_name
    run_dir = OUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tok, model = load_model()
    summaries = {}
    all_results = {}

    if MODE in ("all", "baseline"):
        print("跑 baseline（直接写代码）...")
        res = evaluate_mode(tok, model, items, "baseline", run_dir / "baseline")
        summaries["baseline"] = summarize(res)
        all_results["baseline"] = res
        print(f"  baseline: pass@1={summaries['baseline']['pass@1']:.1%}, case_pass={summaries['baseline']['case_pass']:.1%}\n")

    if MODE in ("all", "prompt"):
        print("跑 prompt（base + 先写伪代码）...")
        res = evaluate_mode(tok, model, items, "prompt", run_dir / "prompt")
        summaries["prompt"] = summarize(res)
        all_results["prompt"] = res
        print(f"  prompt:   pass@1={summaries['prompt']['pass@1']:.1%}, case_pass={summaries['prompt']['case_pass']:.1%}\n")

    if MODE in ("all", "sft") and ADAPTER_PATH:
        print("跑 sft（LoRA 微调 + 先写伪代码）...")
        del model
        torch.cuda.empty_cache()
        _, model2 = load_model(ADAPTER_PATH)
        res = evaluate_mode(tok, model2, items, "sft", run_dir / "sft")
        summaries["sft"] = summarize(res)
        all_results["sft"] = res
        print(f"  sft:      pass@1={summaries['sft']['pass@1']:.1%}, case_pass={summaries['sft']['case_pass']:.1%}\n")
    elif MODE == "sft" and not ADAPTER_PATH:
        print("⚠️ MODE=sft 但没设 ADAPTER_PATH，跳过 sft 组。")

    print("=" * 56)
    for k, v in summaries.items():
        print(f"{k:10s} pass@1={v['pass@1']:.1%} ({v['solved']}/{v['n']})  case_pass={v['case_pass']:.1%} ({v['passed_cases']}/{v['total_cases']})")
    if "prompt" in summaries and "baseline" in summaries:
        d = summaries["prompt"]["pass@1"] - summaries["baseline"]["pass@1"]
        print(f"\n提示词效应 = {d:+.1%}")
    if "sft" in summaries and "prompt" in summaries:
        d = summaries["sft"]["pass@1"] - summaries["prompt"]["pass@1"]
        print(f"SFT 效应   = {d:+.1%}")

    report = {
        "run_name": run_name,
        "config": {
            "model": MODEL_PATH, "adapter": ADAPTER_PATH or None,
            "num_samples": NUM_SAMPLES, "temperature": TEMPERATURE,
            "max_new_tokens": MAX_NEW_TOKENS, "dev_n": len(items),
        },
        "summary": summaries,
        "results": all_results,  # 逐题逐样本的完整结果（含 cases[] 逐样例实跑）
        "note": "results[mode][i] 是每题明细：samples[].cases[] 记录每个测试样例的 passed/error/actual/expected/stderr；raw/code 全文在 <mode>/<题目>/raw_0.txt 和 code_0.py。",
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整记录已存到：{run_dir}")
    print(f"  报告：{report_path}")
    print(f"  排查 baseline=0%：去看 {run_dir}/baseline/<题目>/raw_0.txt 和 code_0.py，以及 report 里的 cases[].error")


if __name__ == "__main__":
    main()
