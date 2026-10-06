# -*- coding: utf-8 -*-
"""
两段式 plan-then-code 数据构造（重做版，尽善尽美）

【核心变化（相对旧的 build_pseudocode_sft.py）】
1. 伪代码【只看题生成】（不给参考解）—— 教「从题推导」，不教「解释已知解」。
2. 【闭环校验】—— 拿伪代码实现成代码 → 跑测试 → 通过才保留，保证「伪代码能推出正确解」。
3. 【拆两条样本】：
    样本A 题目 → 伪代码          （练「推导计划」）
    样本B 题目 + 伪代码 → 代码    （练「照计划忠实实现」）
4. 伪代码【中等粒度】（每步"做什么+为什么"，5~15 步）。
5. 废掉「直接写代码」普通样本（那是对另一个 prompt 的 rehearsal，不解决本问题）。

【流程】
TACO → 过滤 → 排除已用（测试集+train+dev）→ 参考解自检（3 样例，筛问题质量）
     → 每题：只看题生成伪代码 → 照伪代码实现代码 → 跑测试 → 通过保留，失败重试(≤2)
     → 输出 plan_sft.jsonl + impl_sft.jsonl

【运行】（AutoDL）
  export DASHSCOPE_API_KEY=sk-xxx
  python build_plan_sft.py

依赖：datasets requests
"""
import argparse
import ast
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import Dataset, concatenate_datasets

from llm_api import call_llm, API_KEY

# ── 路径与参数 ─────────────────────────────────────────────────
TACO_DIR = os.path.expanduser("~/autodl-tmp/TACO")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # 本脚本在 data/ 目录下，相对路径从这里算
TEST_PROBLEMS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "eval", "benchmark", "problems_cliff"))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "taco_train"))
OUT_PLAN = os.path.join(DATA_DIR, "plan_sft.jsonl")     # 样本A：题目→伪代码
OUT_IMPL = os.path.join(DATA_DIR, "impl_sft.jsonl")     # 样本B：题目+伪代码→代码

PLAN_N = 1000          # 目标保留样本数（试点；方向对了再扩 3000）
MAX_RETRY = 2          # 每题「伪代码→代码→测试」失败最多重试次数（重新生成伪代码）
CHECK_MAX_TESTS = 3    # 参考解自检（筛问题质量）样例数
CLOSED_LOOP_TESTS = 20  # 闭环校验跑全部样例（≤20），尽善尽美
TEST_TIMEOUT = 10
SEED = 42
CONCURRENCY = 8

# ── 提示词 ─────────────────────────────────────────────────────
PLAN_PROMPT = (
    "你是一名算法助教。下面是编程题。请写出分步伪代码（算法计划），不要写代码。\n\n"
    "要求：\n"
    "1. 第一行先一句话点出整体算法思路（这题的核心观察/技巧）。\n"
    "2. 然后用编号列表（1. 2. 3. ...）分步描述，每一步回答「做什么 + 为什么」，5~15 步为宜。\n"
    "3. 中等粒度：既要体现算法思路，又要足够具体到能直接指导实现（关键数据结构、循环意图、边界处理）。\n"
    "4. 语言中文，每行一条，简洁。\n"
    "5. 只输出伪代码本身，不要输出代码、不要解释、不要多余的话。\n\n"
    "【题目】\n{question}"
)

IMPL_PROMPT = (
    "下面是编程题和它的分步伪代码。请严格照着伪代码，写出完整可运行的 Python 程序。\n\n"
    "要求：只输出代码，放在 ```python ... ``` 代码块内，不要解释、不要伪代码。\n\n"
    "【题目】\n{question}\n\n"
    "【伪代码】\n{plan}"
)


# ── 过滤/工具 ──────────────────────────────────────────────────
def _as_list(v):
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return []
    return v if isinstance(v, list) else []


def _as_io(v):
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return {}
    return v if isinstance(v, dict) else {}


def _norm(x):
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return "\n".join(str(y) for y in x)
    return str(x)


def parse_tests(input_output_val):
    io = _as_io(input_output_val)
    ins = io.get("inputs", [])
    outs = io.get("outputs", [])
    return [{"input": _norm(ins[i]), "output": _norm(outs[i])} for i in range(min(len(ins), len(outs)))]


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


def ref_passes(ref_code, tests):
    td = tempfile.mkdtemp(prefix="plan_refcheck_")
    try:
        with open(os.path.join(td, "solution.py"), "w", encoding="utf-8") as f:
            f.write(ref_code)
        for case in tests[:CHECK_MAX_TESTS]:
            try:
                r = subprocess.run([sys.executable, "-B", "solution.py"], cwd=td,
                                   input=case["input"], capture_output=True, text=True, timeout=TEST_TIMEOUT)
            except subprocess.TimeoutExpired:
                return False
            if r.returncode != 0:
                return False
            if _norm_out(r.stdout) != _norm_out(case["output"]):
                return False
        return True
    finally:
        shutil.rmtree(td, ignore_errors=True)


def run_code(code, tests):
    """跑生成的代码，返回 (通过样例数, 总样例数)。首个失败即停（快）。"""
    td = tempfile.mkdtemp(prefix="plan_run_")
    try:
        with open(os.path.join(td, "solution.py"), "w", encoding="utf-8") as f:
            f.write(code)
        n_pass = 0
        total = len(tests[:CLOSED_LOOP_TESTS])
        for case in tests[:CLOSED_LOOP_TESTS]:
            try:
                r = subprocess.run([sys.executable, "-B", "solution.py"], cwd=td,
                                   input=case["input"], capture_output=True, text=True, timeout=TEST_TIMEOUT)
            except subprocess.TimeoutExpired:
                break
            if r.returncode != 0:
                break
            if _norm_out(r.stdout) != _norm_out(case["output"]):
                break
            n_pass += 1
        return n_pass, total
    finally:
        shutil.rmtree(td, ignore_errors=True)


def extract_code(text):
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\w*\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def load_used_questions():
    qs = set()
    for fp in sorted(glob.glob(os.path.join(TEST_PROBLEMS_DIR, "*", "problem.json"))):
        try:
            p = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        q = (p.get("requirement") or "").strip()
        if q:
            qs.add(q)
    for fn in ["train_raw.jsonl", "dev_raw.jsonl"]:
        fp = os.path.join(DATA_DIR, fn)
        if os.path.isfile(fp):
            for ln in open(fp, encoding="utf-8"):
                ln = ln.strip()
                if ln:
                    q = (json.loads(ln).get("question") or "").strip()
                    if q:
                        qs.add(q)
    return qs


# ── 单条构造（闭环）────────────────────────────────────────────
def build_sample(item, tests):
    question = item.get("question")
    for _ in range(MAX_RETRY):
        try:
            plan = call_llm([{"role": "user", "content": PLAN_PROMPT.format(question=question)}])
            raw = call_llm([{"role": "user", "content": IMPL_PROMPT.format(question=question, plan=plan)}])
            code = extract_code(raw)
            if not code or not is_valid_py3(code):
                continue
            n_pass, total = run_code(code, tests)
            if n_pass == total and total > 0:
                return {"question": question, "plan": plan, "code": code,
                        "difficulty": item.get("difficulty")}
        except Exception:
            continue
    return None


# ── 前置校验（--check，零成本，先跑这个再烧钱）──────────────────
def do_check():
    print("== 前置校验（零成本，只查路径/红线，不碰 LLM 和测试）==\n")
    ok = True

    taco_files = glob.glob(os.path.join(TACO_DIR, "train", "*.arrow"))
    print(f"[{'OK' if taco_files else 'FAIL'}] TACO 数据: {TACO_DIR} 下 {len(taco_files)} 个 arrow")

    test_files = glob.glob(os.path.join(TEST_PROBLEMS_DIR, "*", "problem.json"))
    print(f"[{'OK' if len(test_files) >= 70 else 'FAIL'}] 测试集: {TEST_PROBLEMS_DIR} 下 {len(test_files)} 题（应 70）")

    used = load_used_questions()
    print(f"[{'OK' if len(used) >= 400 else 'FAIL'}] 排除键: {len(used)} 条（应 ≈420 = 70测试+300train+50dev）")

    print(f"[{'OK' if API_KEY else 'FAIL'}] LLM_API_KEY: {'已设置' if API_KEY else '未设置'}")

    print(f"[{'OK' if os.path.isdir(DATA_DIR) else 'FAIL'}] 数据目录: {DATA_DIR}")
    print(f"[{'OK' if os.path.isdir(os.path.dirname(OUT_PLAN)) else 'FAIL'}] 输出目录: {os.path.dirname(OUT_PLAN)}")

    print("\n上面有 FAIL 就修完再跑全量；全 OK 才烧钱。")


# ── 主流程 ─────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-n", type=int, default=PLAN_N, help="目标保留样本数（默认 1000）")
    ap.add_argument("--out-dir", type=str, default="", help="输出目录（默认 data/taco_train；新文件夹如 plan1000）")
    ap.add_argument("--check", action="store_true", help="只做前置校验，不生成数据")
    args = ap.parse_args()
    if args.check:
        do_check()
        return
    plan_n = args.plan_n
    if args.out_dir:
        out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(DATA_DIR, args.out_dir)
    else:
        out_dir = DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_plan = os.path.join(out_dir, "plan_sft.jsonl")
    out_impl = os.path.join(out_dir, "impl_sft.jsonl")

    print("加载 TACO ...")
    files = sorted(glob.glob(os.path.join(TACO_DIR, "train", "*.arrow")))
    if not files:
        raise FileNotFoundError(f"没找到 TACO：{TACO_DIR}/train/*.arrow")
    ds = concatenate_datasets([Dataset.from_file(f) for f in files])
    print(f"  共 {len(ds)} 条")

    used = load_used_questions()
    print(f"  排除键 {len(used)} 条\n")
    if len(used) < 100:
        raise RuntimeError(
            f"⚠️ 排除键只有 {len(used)} 条（应 ≈420 = 70测试集 + 300train + 50dev），"
            f"说明 TEST_PROBLEMS_DIR / DATA_DIR 路径错了，训练集没排除评测题！先修路径再跑。"
        )

    # 过滤 + 参考解自检（筛问题质量）
    cand = []
    for item in ds:
        q = (item.get("question") or "").strip()
        if "interactive" in q.lower():
            continue
        sols = _as_list(item.get("solutions"))
        if not sols or not isinstance(sols[0], str) or not sols[0].strip():
            continue
        if not is_valid_py3(sols[0]):
            continue
        if q in used:
            continue
        tests = parse_tests(item.get("input_output"))
        if not tests:
            continue
        if not ref_passes(sols[0], tests):
            continue
        cand.append((item, tests))

    rng = random.Random(SEED)
    rng.shuffle(cand)
    print(f"  自检后候选 {len(cand)} 条，开始闭环构造 {plan_n} 条（并发 {CONCURRENCY}）...")

    kept = []
    n_done = 0

    def work(i_item):
        item, tests = i_item
        return build_sample(item, tests)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(work, (it, ts)) for it, ts in cand]
        for fut in as_completed(futs):
            n_done += 1
            res = fut.result()
            if res is not None:
                kept.append(res)
            if n_done % 100 == 0:
                print(f"  已处理 {n_done}/{len(cand)}，保留 {len(kept)}", flush=True)
            if len(kept) >= plan_n:
                for f in futs:
                    f.cancel()
                break

    kept = kept[:plan_n]

    # 写两条样本
    with open(out_plan, "w", encoding="utf-8") as fp, open(out_impl, "w", encoding="utf-8") as fi:
        for i, s in enumerate(kept):
            q, plan, code = s["question"], s["plan"], s["code"]
            plan_sample = {
                "id": f"plan_{i:05d}", "difficulty": s["difficulty"], "question": q, "plan": plan,
                "messages": [
                    {"role": "user", "content": f"【题目】\n{q}\n\n请写出分步伪代码（算法计划），不要写代码。"},
                    {"role": "assistant", "content": plan},
                ],
            }
            impl_sample = {
                "id": f"impl_{i:05d}", "difficulty": s["difficulty"], "question": q, "plan": plan, "code": code,
                "messages": [
                    {"role": "user", "content": f"【题目】\n{q}\n\n【伪代码】\n{plan}\n\n请严格照着伪代码写出完整可运行的 Python 程序。"},
                    {"role": "assistant", "content": f"```python\n{code}\n```"},
                ],
            }
            fp.write(json.dumps(plan_sample, ensure_ascii=False) + "\n")
            fi.write(json.dumps(impl_sample, ensure_ascii=False) + "\n")

    print(f"\n✅ 保留 {len(kept)} 条（处理 {n_done}/{len(cand)}）")
    print(f"   样本A（题目→伪代码）：{out_plan}")
    print(f"   样本B（题目+伪代码→代码）：{out_impl}")
    print("下一步：改 train_lora.py 加载这两份文件做两段式 SFT。")


if __name__ == "__main__":
    main()
