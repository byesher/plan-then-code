# -*- coding: utf-8 -*-
"""
评测脚本：对比 base vs「prompt 伪代码」vs「SFT 伪代码」的 pass@1（在 dev 集上）

【目的】回答核心问题「伪代码到底有没有用」：
  - baseline：base 模型 + 直接写代码（不要求伪代码）→ 原能力
  - prompt  ：base 模型 + 「先写伪代码再写代码」→ 纯提示词能不能涨（零成本消融）
  - sft     ：LoRA 微调后的模型 + 「先写伪代码再写代码」→ SFT 到底有没有额外收益
差值隔离了「提示词效应」(prompt-baseline) 和「SFT 效应」(sft-prompt)。

【评测口径】
- 用 dev_raw.jsonl（50 题 hold-out，不含 70 题测试集；测试集留作最终评测）。
- 每题生成 NUM_SAMPLES 份代码；prompt/sft 模式抽 ```python``` 块，baseline 模式整段即代码。
- pass@1 = 至少一份代码通过全部样例的题占比；case_pass = 通过样例数/总样例数（best-of-k）。

【运行】（在 AutoDL，eval 目录下）
  cd plan-then-code/eval
  python eval_sft.py                                      # 只跑 baseline + prompt（不加载 adapter，快）
  ADAPTER_PATH=~/autodl-tmp/plan-then-code-lora python eval_sft.py   # 跑全三组

依赖：torch transformers peft（无 trl）
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── 路径与参数 ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/autodl-tmp/Qwen2.5-7B-Instruct"))
DEV_PATH = os.path.join(BASE_DIR, "..", "data", "taco_train", "dev_raw.jsonl")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "")          # 空 = 不跑 sft 组
MODE = os.environ.get("MODE", "all")                       # all / baseline / prompt / sft

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


def generate(tok, model, messages, n, temperature, max_new):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = tok(text, return_tensors="pt").to(device)
    outs = []
    with torch.no_grad():
        for _ in range(n):
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tok.eos_token_id,
            )
            gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            outs.append(gen.strip())
    return outs


def extract_code(text):
    """从「伪代码 + 代码」输出里抽 ```python``` 块；抽不到就当整段是代码。"""
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
    """跑代码，返回 (通过样例数, 总样例数)；首个失败即停（快速失败，同 convert 脚本）。"""
    td = tempfile.mkdtemp(prefix="eval_")
    try:
        with open(os.path.join(td, "solution.py"), "w", encoding="utf-8") as f:
            f.write(code)
        total = len(tests[:MAX_EVAL_TESTS])
        n_pass = 0
        for case in tests[:MAX_EVAL_TESTS]:
            try:
                r = subprocess.run(
                    [sys.executable, "-B", "solution.py"],
                    cwd=td, input=case["input"], capture_output=True, text=True, timeout=TEST_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                return n_pass, total
            if r.returncode != 0:
                return n_pass, total
            if _norm_out(r.stdout) != _norm_out(case["output"]):
                return n_pass, total
            n_pass += 1
        return n_pass, total
    finally:
        shutil.rmtree(td, ignore_errors=True)


def evaluate(tok, model, items, mode):
    solved = 0
    passed_cases = 0
    total_cases = 0
    for it in items:
        question = it["question"]
        tests = it.get("tests") or []
        if not tests:
            continue
        if mode == "baseline":
            msgs = [{"role": "user", "content": PLAIN_USER.format(question=question)}]
        else:
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PLAN_USER.format(question=question)},
            ]
        gens = generate(tok, model, msgs, NUM_SAMPLES, TEMPERATURE, MAX_NEW_TOKENS)
        total = len(tests[:MAX_EVAL_TESTS])
        best_pass = 0
        for g in gens:
            code = g if mode == "baseline" else extract_code(g)
            p, _ = run_tests(code, tests)
            best_pass = max(best_pass, p)
        total_cases += total
        passed_cases += best_pass
        if best_pass == total:
            solved += 1
    n = len(items)
    return {
        "pass@1": solved / n if n else 0.0,
        "case_pass": passed_cases / total_cases if total_cases else 0.0,
        "n": n, "solved": solved, "passed_cases": passed_cases, "total_cases": total_cases,
    }


def main():
    items = load_dev()
    print(f"dev 集 {len(items)} 题，NUM_SAMPLES={NUM_SAMPLES}，temperature={TEMPERATURE}\n")

    tok, model = load_model()
    results = {}

    if MODE in ("all", "baseline"):
        print("跑 baseline（直接写代码）...")
        results["baseline"] = evaluate(tok, model, items, "baseline")
        print(f"  baseline: pass@1={results['baseline']['pass@1']:.1%}, case_pass={results['baseline']['case_pass']:.1%}\n")

    if MODE in ("all", "prompt"):
        print("跑 prompt（base + 先写伪代码）...")
        results["prompt"] = evaluate(tok, model, items, "prompt")
        print(f"  prompt:   pass@1={results['prompt']['pass@1']:.1%}, case_pass={results['prompt']['case_pass']:.1%}\n")

    if MODE in ("all", "sft") and ADAPTER_PATH:
        print("跑 sft（LoRA 微调 + 先写伪代码）...")
        del model
        torch.cuda.empty_cache()
        _, model2 = load_model(ADAPTER_PATH)
        results["sft"] = evaluate(tok, model2, items, "sft")
        print(f"  sft:      pass@1={results['sft']['pass@1']:.1%}, case_pass={results['sft']['case_pass']:.1%}\n")
    elif MODE == "sft" and not ADAPTER_PATH:
        print("⚠️ MODE=sft 但没设 ADAPTER_PATH，跳过 sft 组。")

    print("=" * 56)
    for k, v in results.items():
        print(f"{k:10s} pass@1={v['pass@1']:.1%} ({v['solved']}/{v['n']})  "
              f"case_pass={v['case_pass']:.1%} ({v['passed_cases']}/{v['total_cases']})")
    if "prompt" in results and "baseline" in results:
        d = results["prompt"]["pass@1"] - results["baseline"]["pass@1"]
        print(f"\n提示词效应 = {d:+.1%}")
    if "sft" in results and "prompt" in results:
        d = results["sft"]["pass@1"] - results["prompt"]["pass@1"]
        print(f"SFT 效应   = {d:+.1%}")


if __name__ == "__main__":
    main()
