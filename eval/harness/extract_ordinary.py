# -*- coding: utf-8 -*-
"""
从 TACO 抽「普通代码」样本（rehearsal 数据），用于数据混合防灾难性遗忘。

【目的】
策略①（训练侧防遗忘）：训练时把「伪代码样本」和「普通代码样本」混在一起，
让模型学新格式（先伪代码再代码）的同时，不忘记"直接写对代码"的老能力。
本脚本产出「普通代码样本」= (题目 → 直接写代码)，不带伪代码。

【过滤/排除】
- 同 extract_train_set.py：去交互/空解/Python2/无样例。
- 排除：70 题测试集 + 已用的 train_raw(300) + dev(50)，保证与伪代码样本不重叠。
- 轻自检：参考解跑 3 个样例全过才留（与伪代码数据同标准，淘汰约 44% 坏解）。

【输出】
  data/taco_train/ordinary_sft.jsonl  —— messages 格式
    user:      【题目】… 请直接输出代码
    assistant: ```python 代码```

【运行】（AutoDL）
  python extract_ordinary.py
  预计 20~40 分钟（3000 条 × 轻自检，subprocess 跑参考解是主要耗时）
"""
import ast
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

from datasets import Dataset, concatenate_datasets

TACO_DIR = os.path.expanduser("~/autodl-tmp/TACO")
HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PROBLEMS_DIR = os.path.abspath(os.path.join(HARNESS_DIR, "..", "benchmark", "problems_cliff"))
DATA_DIR = os.path.abspath(os.path.join(HARNESS_DIR, "..", "..", "data", "taco_train"))
OUT_PATH = os.path.join(DATA_DIR, "ordinary_sft.jsonl")

ORDINARY_N = 3000
SEED = 42
CHECK_MAX_TESTS = 3   # 与伪代码数据同标准：参考解跑前 3 样例全过才留（能淘汰 ~44% 坏解）
CHECK_TIMEOUT = 10

USER_TEMPLATE = "【题目】\n{question}\n\n请写出完整可运行的 Python 程序，直接输出代码，不要解释。"
ASSISTANT_TEMPLATE = "```python\n{code}\n```"


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
    td = tempfile.mkdtemp(prefix="ord_check_")
    try:
        with open(os.path.join(td, "solution.py"), "w", encoding="utf-8") as f:
            f.write(ref_code)
        for case in tests[:CHECK_MAX_TESTS]:
            try:
                r = subprocess.run([sys.executable, "-B", "solution.py"], cwd=td,
                                   input=case["input"], capture_output=True, text=True, timeout=CHECK_TIMEOUT)
            except subprocess.TimeoutExpired:
                return False
            if r.returncode != 0:
                return False
            if _norm_out(r.stdout) != _norm_out(case["output"]):
                return False
        return True
    finally:
        shutil.rmtree(td, ignore_errors=True)


def load_used_questions():
    """排除 70 测试集 + train_raw(300) + dev(50) 的题目原文。"""
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


def main():
    print("加载 TACO ...")
    files = sorted(glob.glob(os.path.join(TACO_DIR, "train", "*.arrow")))
    if not files:
        raise FileNotFoundError(f"没找到 TACO：{TACO_DIR}/train/*.arrow")
    ds = concatenate_datasets([Dataset.from_file(f) for f in files])
    print(f"  共 {len(ds)} 条")

    used = load_used_questions()
    print(f"  排除键（测试集+train_raw+dev）共 {len(used)} 条唯一题目原文\n")

    kept = []
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
        kept.append((item, sols[0], tests))

    print(f"  候选 {len(kept)} 条，shuffle 后轻自检（{CHECK_MAX_TESTS} 样例）抽 {ORDINARY_N} 条 ...")

    rng = random.Random(SEED)
    order = list(range(len(kept)))
    rng.shuffle(order)

    out = []
    n_fail = 0
    for i in order:
        if len(out) >= ORDINARY_N:
            break
        item, ref, tests = kept[i]
        if not ref_passes(ref, tests):
            n_fail += 1
            continue
        out.append({
            "id": f"ordinary_{i:06d}",
            "difficulty": item.get("difficulty"),
            "question": item.get("question"),
            "code": ref,
            "messages": [
                {"role": "user", "content": USER_TEMPLATE.format(question=item.get("question"))},
                {"role": "assistant", "content": ASSISTANT_TEMPLATE.format(code=ref)},
            ],
        })
        if len(out) % 500 == 0:
            print(f"  已抽 {len(out)}/{ORDINARY_N}（淘汰 {n_fail}）", flush=True)

    if len(out) < ORDINARY_N:
        print(f"  ⚠️ 只凑到 {len(out)} 条（候选不够），就用这 {len(out)} 条。")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"\n✅ 普通代码样本写入 {OUT_PATH}（{len(out)} 条，淘汰 {n_fail}）")
    print("下一步：改 train_lora.py 做数据混合后训练。")


if __name__ == "__main__":
    main()
