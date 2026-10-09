# -*- coding: utf-8 -*-
"""单轮 plan-then-code 数据构造（全量 EASY 版）：
EASY 题 + 伪代码含输入输出契约 + 并行过滤 + 边造边存 + 按总数比例拆分 train/eval

【本次改动（冲 SOTA）】
1. 不限量：抽出所有符合规则的 EASY 题，训练集全部参与闭环构造（去掉 plan_n 截断）。
2. 调 LLM 之前先拿总数：ref_passes 之后 len(cand) 即「可用 EASY 总数」（此步只跑参考解
   subprocess，零 LLM 成本）。总数确定后按比例划分 train/eval，再开始烧钱。
3. 划分方式：eval = round(总数 × EVAL_RATIO)；N_EVAL>0 时用该值手动覆盖；train = 总数 − eval。
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

from datasets import Dataset, concatenate_datasets

from llm_api import call_llm, API_KEY

TACO_DIR = os.path.expanduser("~/autodl-tmp/TACO")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PROBLEMS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "eval", "benchmark", "problems_cliff"))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "taco_train"))

N_EVAL = int(os.environ.get("N_EVAL", "0"))            # 0 = 按 EVAL_RATIO 自动算；>0 = 手动指定评测集数量
EVAL_RATIO = float(os.environ.get("EVAL_RATIO", "0.10"))
MAX_RETRY = 2
CHECK_MAX_TESTS = 3
CLOSED_LOOP_TESTS = 20
TEST_TIMEOUT = 10
SEED = 42
CONCURRENCY = 8
DIFFICULTY = os.environ.get("DIFFICULTY", "EASY")

SFT_PROMPT = (
    "你是一名算法助教。下面是编程题。请先写出分步伪代码（算法计划），再写出完整可运行的 Python 程序。\n\n"
    "要求：\n"
    "1. 伪代码第一行先写【输入怎么读 + 输出什么格式】，再一句话点出算法思路。\n"
    "2. 伪代码用编号列表（1. 2. 3. ...）分步描述，每一步回答「做什么 + 为什么」，3~8 步为宜，只写关键算法步骤，不要写循环变量的初始化/自增这类机械步骤。\n"
    "3. 伪代码写完后，另起一行写代码，代码放在 ```python ... ``` 代码块内。\n"
    "4. 语言中文，简洁。\n\n"
    "【题目】\n{question}"
)


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
    td = tempfile.mkdtemp(prefix="plan_run_")
    try:
        with open(os.path.join(td, "solution.py"), "w", encoding="utf-8") as f:
            f.write(code)
        n_pass = 0
        total = len(tests[:CLOSED_LOOP_TESTS])
        first_err = ""
        for i, case in enumerate(tests[:CLOSED_LOOP_TESTS]):
            try:
                r = subprocess.run([sys.executable, "-B", "solution.py"], cwd=td,
                                   input=case["input"], capture_output=True, text=True, timeout=TEST_TIMEOUT)
            except subprocess.TimeoutExpired:
                first_err = f"case{i}:timeout"
                break
            if r.returncode != 0:
                first_err = f"case{i}:runtime({(r.stderr or '').strip()[:80]})"
                break
            if _norm_out(r.stdout) != _norm_out(case["output"]):
                first_err = f"case{i}:mismatch"
                break
            n_pass += 1
        return n_pass, total, first_err
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


def build_sample(item, tests):
    question = item.get("question")
    for attempt in range(MAX_RETRY):
        try:
            raw = call_llm([{"role": "user", "content": SFT_PROMPT.format(question=question)}])
            code = extract_code(raw)
            if not code or not is_valid_py3(code):
                continue
            n_pass, total, first_err = run_code(code, tests)
            if n_pass == total and total > 0:
                return {"ok": True, "sample": {"question": question, "raw": raw, "code": code,
                                               "difficulty": item.get("difficulty")}}
            return {"ok": False, "failure": {"question": question, "raw": raw, "code": code,
                                             "difficulty": item.get("difficulty"), "err": first_err,
                                             "n_pass": n_pass, "total": total, "attempt": attempt}}
        except Exception as e:
            return {"ok": False, "failure": {"question": question,
                                             "difficulty": item.get("difficulty"),
                                             "err": f"exception:{type(e).__name__}:{e}", "attempt": attempt}}
    return {"ok": False, "failure": {"question": question, "difficulty": item.get("difficulty"),
                                     "err": "max_retry(empty_or_syntax)"}}


def do_check():
    print("== 前置校验 ==\n")
    taco_files = glob.glob(os.path.join(TACO_DIR, "train", "*.arrow"))
    print(f"[{'OK' if taco_files else 'FAIL'}] TACO: {len(taco_files)} 个 arrow")
    test_files = glob.glob(os.path.join(TEST_PROBLEMS_DIR, "*", "problem.json"))
    print(f"[{'OK' if len(test_files) >= 70 else 'FAIL'}] 测试集: {len(test_files)} 题")
    used = load_used_questions()
    print(f"[{'OK' if len(used) >= 400 else 'FAIL'}] 排除键: {len(used)} 条")
    print(f"[{'OK' if API_KEY else 'FAIL'}] LLM_API_KEY: {'已设置' if API_KEY else '未设置'}")
    print(f"[{'OK' if os.path.isdir(DATA_DIR) else 'FAIL'}] 数据目录: {DATA_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        do_check()
        return
    if args.out_dir:
        out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(DATA_DIR, args.out_dir)
    else:
        out_dir = DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_sft = os.path.join(out_dir, "sft.jsonl")
    out_rejected = os.path.join(out_dir, "rejected.jsonl")
    out_manifest = os.path.join(out_dir, "manifest.json")
    eval_out = os.path.join(out_dir, "eval.jsonl")

    print(f"加载 TACO（只抽难度={DIFFICULTY}）...")
    files = sorted(glob.glob(os.path.join(TACO_DIR, "train", "*.arrow")))
    if not files:
        raise FileNotFoundError(f"没找到 TACO：{TACO_DIR}/train/*.arrow")
    ds = concatenate_datasets([Dataset.from_file(f) for f in files])
    print(f"  共 {len(ds)} 条")

    used = load_used_questions()
    if len(used) < 100:
        raise RuntimeError(f"排除键只有 {len(used)} 条，路径可能错了")

    # ── 第一步：快速过滤（纯内存，无 subprocess，秒级）──
    fast = []
    skip = {"interactive": 0, "empty": 0, "py2": 0, "overlap": 0, "no_tests": 0, "ref_fail": 0, "not_easy": 0}
    for item in ds:
        q = (item.get("question") or "").strip()
        if (item.get("difficulty") or "").strip() != DIFFICULTY:
            skip["not_easy"] += 1
            continue
        if "interactive" in q.lower():
            skip["interactive"] += 1
            continue
        sols = _as_list(item.get("solutions"))
        if not sols or not isinstance(sols[0], str) or not sols[0].strip():
            skip["empty"] += 1
            continue
        if not is_valid_py3(sols[0]):
            skip["py2"] += 1
            continue
        if q in used:
            skip["overlap"] += 1
            continue
        tests = parse_tests(item.get("input_output"))
        if not tests:
            skip["no_tests"] += 1
            continue
        fast.append((item, tests))

    print(f"  快速过滤后 {len(fast)} 条，并行做参考解自检（{CONCURRENCY} 并发）...")

    # ── 第二步：并行参考解自检（ref_passes，全量候选，零 LLM）──
    def check_ref(x):
        item, tests = x
        sols = _as_list(item.get("solutions"))
        return (item, tests) if ref_passes(sols[0], tests) else None

    cand = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        for res in ex.map(check_ref, fast):
            if res is not None:
                cand.append(res)
            else:
                skip["ref_fail"] += 1

    rng = random.Random(SEED)
    rng.shuffle(cand)

    # ── ★ 调 LLM 之前：总数已定，按数量划分 train/eval ──
    n_total = len(cand)
    if N_EVAL > 0:
        n_eval = min(N_EVAL, n_total)
    else:
        n_eval = max(1, int(round(n_total * EVAL_RATIO)))
    print(f"  ★ 可用 EASY 题总数（调 LLM 之前已确定）：{n_total} 道")
    print(f"  ★ 评测集 {n_eval} 道（{n_eval / n_total:.1%}），训练集 {n_total - n_eval} 道（{(n_total - n_eval) / n_total:.1%}）")

    eval_items = cand[-n_eval:] if n_eval > 0 else []
    train_items = cand[:-n_eval] if n_eval > 0 else cand
    if eval_items:
        with open(eval_out, "w", encoding="utf-8") as fe:
            for i, (item, tests) in enumerate(eval_items):
                fe.write(json.dumps({
                    "id": f"eval_{i:05d}",
                    "question": item.get("question"),
                    "tests": tests,
                    "difficulty": item.get("difficulty"),
                }, ensure_ascii=False) + "\n")
        print(f"  已预留 {len(eval_items)} 道评测题 -> {eval_out}")

    print(f"  训练候选 {len(train_items)} 条，开始闭环构造（不限量，全部处理）...")

    # ── 第三步：闭环构造 + 边造边存（每出一条立即 flush，崩了不丢）──
    n_kept = 0
    n_rejected = 0
    n_done = 0
    diff_kept = Counter()
    err_dist = Counter()

    def work(i_item):
        item, tests = i_item
        return build_sample(item, tests)

    with open(out_sft, "w", encoding="utf-8") as fs, open(out_rejected, "w", encoding="utf-8") as fr:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futs = [ex.submit(work, (it, ts)) for it, ts in train_items]
            for fut in as_completed(futs):
                n_done += 1
                res = fut.result()
                if res is not None and res.get("ok"):
                    s = res["sample"]
                    sample = {
                        "id": f"sft_{n_kept:05d}", "difficulty": s["difficulty"],
                        "question": s["question"], "code": s["code"],
                        "messages": [
                            {"role": "user", "content": f"【题目】\n{s['question']}\n\n请先写出分步伪代码（算法计划），再写出完整可运行的 Python 程序。"},
                            {"role": "assistant", "content": s["raw"]},
                        ],
                    }
                    fs.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    fs.flush()
                    n_kept += 1
                    diff_kept[s.get("difficulty") or "UNKNOWN"] += 1
                elif res is not None:
                    fr.write(json.dumps(res["failure"], ensure_ascii=False) + "\n")
                    fr.flush()
                    n_rejected += 1
                    err_dist[res["failure"]["err"].split(":")[0]] += 1
                if n_done % 100 == 0:
                    print(f"  已处理 {n_done}/{len(train_items)}，保留 {n_kept}，拒绝 {n_rejected}", flush=True)

    manifest = {
        "difficulty": DIFFICULTY,
        "n_eval": n_eval,
        "eval_ratio": EVAL_RATIO,
        "total_easy": n_total,
        "config": {"check_max_tests": CHECK_MAX_TESTS, "closed_loop_tests": CLOSED_LOOP_TESTS,
                   "max_retry": MAX_RETRY, "seed": SEED, "concurrency": CONCURRENCY},
        "skip_stats": skip,
        "candidates": n_total,
        "train_candidates": len(train_items),
        "eval_reserved": len(eval_items),
        "processed": n_done,
        "kept": n_kept,
        "rejected": n_rejected,
        "acceptance_rate": (n_kept / n_done) if n_done else 0.0,
        "difficulty_kept": dict(diff_kept),
        "reject_err_dist": dict(err_dist),
    }
    with open(out_manifest, "w", encoding="utf-8") as fm:
        json.dump(manifest, fm, ensure_ascii=False, indent=2)

    print(f"\n✅ 保留 {n_kept} 条训练样本（处理 {n_done}/{len(train_items)}，拒绝 {n_rejected}）")
    print(f"   评测集 {len(eval_items)} 道：{eval_out}")
    print(f"   训练样本：{out_sft}")
    print(f"   统计：{out_manifest}")


if __name__ == "__main__":
    main()
