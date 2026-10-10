# -*- coding: utf-8 -*-
"""单轮评测 v2（激发 baseline 版）：baseline 用「代码补全式」prompt 激发裸模型真实能力，
避免裸 Coder 不跟 chat 指令导致的「格式税」（echo 指令 / 加「代码如下：」前缀 / 截断）。

与 v1（eval_plan.py）的关系：
- v1 baseline 用 chat 指令「请写出完整可运行的 Python 程序，直接输出代码，不要解释。」，
  裸 Coder-7B 会 echo 指令 / 加中文前缀 / 截断 → compile 被人为压低（74.7%）。这是
  「低级失误」，但结果仍有参考价值（说明「格式税」真实存在）。
- v2 baseline 用补全式 prompt「{question}\n\n```python\n」，并且【不走 apply_chat_template】，
  让裸 Coder 直接续写代码，激发其真实代码能力。
- baseline1turn / sft1turn 与 v1 完全一致（同一 prompt、同一 chat 模板、同一温度），可直接对比。
- 输出目录独立（eval_plan_v2/），不覆盖 v1 结果。

两个版本都可信，最后看哪个更有说服力再决定用哪个。

运行：与 v1 相同环境变量（MODEL_PATH / DEV_PATH / ADAPTER_PATH / OUT_ROOT 等）。
"""
import ast
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
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/autodl-tmp/Qwen2.5-Coder-7B"))
DEV_PATH = Path(os.environ.get("DEV_PATH", BASE_DIR / ".." / "data" / "taco_train" / "eval_easy.jsonl"))
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "")

TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2048"))
MAX_EVAL_TESTS = int(os.environ.get("MAX_EVAL_TESTS", "20"))
LIMIT = int(os.environ.get("LIMIT", "0")) or None
TEST_TIMEOUT = 10
OUT_ROOT = Path(os.environ.get("OUT_ROOT", BASE_DIR / "results"))
RUN_DIR = OUT_ROOT / "eval_plan_v2"          # 独立目录，不覆盖 v1

# v2：补全式 baseline prompt（裸 Coder 直接续写代码，不走 chat 指令）
BASELINE_PROMPT_V2 = "{question}\n\n```python\n"
# 以下两种模式与 v1 完全一致
SFT_PROMPT = "【题目】\n{question}\n\n请先写出分步伪代码（算法计划），再写出完整可运行的 Python 程序。"


def load_dev():
    items = []
    with open(DEV_PATH, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                items.append(json.loads(ln))
    if LIMIT:
        items = items[:LIMIT]
    return items


def load_model(adapter_path=None):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.padding_side = "left"          # 左填充（decoder-only 批量生成必须）
    tok.pad_token = tok.eos_token      # 强制 pad=eos，避免填充/生成不一致
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tok, model


def _fim_bad_words(tok):
    fim = ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
           "<|file_sep|>", "<|repo_name|>", "<|file_ext|>"]
    return [[tok.convert_tokens_to_ids(t)] for t in fim
            if tok.convert_tokens_to_ids(t) != tok.unk_token_id]


def generate_batch(tok, model, msg_batches, max_new, temperature):
    """chat 模板 + 批量生成（用于 baseline1turn / sft1turn，与 v1 一致）"""
    bad_words = _fim_bad_words(tok)
    device = next(model.parameters()).device
    results = []
    B = 4
    for i in range(0, len(msg_batches), B):
        batch = msg_batches[i:i + B]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new, do_sample=True,
                temperature=temperature, top_p=0.95,
                pad_token_id=tok.pad_token_id,
                bad_words_ids=bad_words if bad_words else None,
            )
        for j in range(len(batch)):
            gen = tok.decode(out[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            results.append(gen.strip())
    return results


def generate_completion(tok, model, prompts, max_new, temperature):
    """补全式生成（不走 chat 模板）—— 激发裸 Coder 的直接续写能力"""
    bad_words = _fim_bad_words(tok)
    device = next(model.parameters()).device
    results = []
    B = 4
    for i in range(0, len(prompts), B):
        batch = prompts[i:i + B]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new, do_sample=True,
                temperature=temperature, top_p=0.95,
                pad_token_id=tok.pad_token_id,
                bad_words_ids=bad_words if bad_words else None,
            )
        for j in range(len(batch)):
            gen = tok.decode(out[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            results.append(gen.strip())
    return results


def extract_code_completion(gen):
    """补全式输出的代码抽取：整段即代码，去掉模型可能补上的收尾 ```"""
    code = gen.strip()
    # 极少数情况模型又自己包了一层 ```python ... ```，优先取内部
    m = re.search(r"```python\s*\n(.*?)```", code, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\w*\s*\n(.*?)```", code, re.DOTALL)
    if m:
        return m.group(1).strip()
    # 否则整段就是续写出来的代码，去掉收尾围栏
    code = re.sub(r"```\s*$", "", code).strip()
    return code


def split_plan_code(text):
    """单轮响应里，```python 之前是伪代码、之后是代码。返回 (plan, code)。"""
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return text[:m.start()].strip(), m.group(1).strip()
    m = re.search(r"```\w*\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return text[:m.start()].strip(), m.group(1).strip()
    return text.strip(), text.strip()


def is_valid_py3(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


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
                r = subprocess.run([sys.executable, "-u", "-B", "solution.py"], cwd=td,
                                   input=case["input"], capture_output=True, text=True,
                                   timeout=TEST_TIMEOUT, encoding="utf-8", errors="replace")
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


def eval_baseline_v2(tok, model, items):
    """baseline v2：补全式 prompt，激发裸模型真实能力"""
    mode_dir = RUN_DIR / "baseline"
    mode_dir.mkdir(parents=True, exist_ok=True)
    prompts = [BASELINE_PROMPT_V2.format(question=it["question"]) for it in items]
    gens = generate_completion(tok, model, prompts, MAX_NEW_TOKENS, TEMPERATURE)

    results = []
    for it, g in zip(items, gens):
        tests = it.get("tests") or []
        pdir = mode_dir / it["id"]
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "raw_0.txt").write_text(g, encoding="utf-8")
        code = extract_code_completion(g)
        (pdir / "code_0.py").write_text(code, encoding="utf-8")
        compile_ok = is_valid_py3(code)
        n_pass, total, first_err, cases = run_tests(code, tests)
        results.append({
            "id": it["id"], "solved": n_pass == total and total > 0,
            "n_pass": n_pass, "n_tests": total, "err": first_err,
            "compile": compile_ok, "cases": cases,
        })
    return results


def eval_one_pass(tok, model, items, mode, prompt_tpl, save_plan=False):
    """chat 模板单轮（baseline1turn / sft1turn），与 v1 一致"""
    mode_dir = RUN_DIR / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    all_msgs = [[{"role": "user", "content": prompt_tpl.format(question=it["question"])}] for it in items]
    all_gens = generate_batch(tok, model, all_msgs, MAX_NEW_TOKENS, TEMPERATURE)

    results = []
    for it, g in zip(items, all_gens):
        tests = it.get("tests") or []
        pdir = mode_dir / it["id"]
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "raw_0.txt").write_text(g, encoding="utf-8")
        plan, code = split_plan_code(g)
        if save_plan:
            (pdir / "plan_0.txt").write_text(plan, encoding="utf-8")
        (pdir / "code_0.py").write_text(code, encoding="utf-8")
        compile_ok = is_valid_py3(code)
        n_pass, total, first_err, cases = run_tests(code, tests)
        results.append({
            "id": it["id"], "solved": n_pass == total and total > 0,
            "n_pass": n_pass, "n_tests": total, "err": first_err,
            "compile": compile_ok, "cases": cases,
        })
    return results


def summarize(results):
    n = len(results)
    solved = sum(1 for r in results if r["solved"])
    compile_ok = sum(1 for r in results if r.get("compile"))
    n_case_pass = sum(r["n_pass"] for r in results)
    n_case_total = sum(r["n_tests"] for r in results)
    return {"n": n, "solved": solved, "pass@1": solved / n if n else 0.0,
            "compile_rate": compile_ok / n if n else 0.0,
            "case_pass_rate": n_case_pass / n_case_total if n_case_total else 0.0}


def main():
    items = load_dev()
    print(f"dev 集 {len(items)} 题，temperature={TEMPERATURE}（v2：baseline 用补全式 prompt 激发）\n")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    tok, model = load_model()
    summaries = {}
    all_results = {}

    print("跑 baseline_v2（补全式 prompt，激发裸模型）...")
    r = eval_baseline_v2(tok, model, items)
    summaries["baseline"] = summarize(r)
    all_results["baseline"] = r
    print(f"  baseline: pass@1={summaries['baseline']['pass@1']:.1%} "
          f"compile={summaries['baseline']['compile_rate']:.1%} case_pass={summaries['baseline']['case_pass_rate']:.1%}\n")

    print("跑 baseline1turn_v2（裸模型 + 单轮伪代码+代码，与 v1 相同）...")
    r = eval_one_pass(tok, model, items, "baseline1turn", SFT_PROMPT, save_plan=True)
    summaries["baseline1turn"] = summarize(r)
    all_results["baseline1turn"] = r
    print(f"  baseline1turn: pass@1={summaries['baseline1turn']['pass@1']:.1%} "
          f"compile={summaries['baseline1turn']['compile_rate']:.1%} case_pass={summaries['baseline1turn']['case_pass_rate']:.1%}\n")

    if ADAPTER_PATH:
        print("跑 sft1turn_v2（SFT 模型 + 单轮，与 v1 相同）...")
        del model
        torch.cuda.empty_cache()
        _, model2 = load_model(ADAPTER_PATH)
        r = eval_one_pass(tok, model2, items, "sft1turn", SFT_PROMPT, save_plan=True)
        summaries["sft1turn"] = summarize(r)
        all_results["sft1turn"] = r
        print(f"  sft1turn: pass@1={summaries['sft1turn']['pass@1']:.1%} "
              f"compile={summaries['sft1turn']['compile_rate']:.1%} case_pass={summaries['sft1turn']['case_pass_rate']:.1%}\n")
    else:
        print("⚠️ 没设 ADAPTER_PATH，跳过 sft1turn。")

    print("=" * 66)
    for k, v in summaries.items():
        print(f"{k:14s} pass@1={v['pass@1']:.1%}  compile={v['compile_rate']:.1%}  case_pass={v['case_pass_rate']:.1%}")
    if "sft1turn" in summaries and "baseline" in summaries:
        d = summaries["sft1turn"]["pass@1"] - summaries["baseline"]["pass@1"]
        print(f"\nSFT 效应（相对激发后的 baseline）= {d:+.1%}")

    report = {
        "summary": summaries,
        "results": all_results,
        "config": {"adapter": ADAPTER_PATH or None, "temperature": TEMPERATURE,
                   "max_new_tokens": MAX_NEW_TOKENS, "limit": LIMIT,
                   "note": "baseline 用补全式 prompt（不走 chat 模板）激发裸模型"},
    }
    out = RUN_DIR / "report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n全量结果已存到：{RUN_DIR}")


if __name__ == "__main__":
    main()
