# -*- coding: utf-8 -*-
"""
TACO → 能力悬崖测试集 转换脚本

【目的】
从 TACO（I/O 竞赛题，全量 arrow 已下到 ~/autodl-tmp/TACO）里，按「参考解行数」分桶抽题，
生成 eval 能直接跑的 problem.json + reference.py，用来找「代码长度能力悬崖」——
即 7B/14B 的 pass@1 在哪个长度档开始崩。

【为什么选 TACO】
- 全量已下好（25443 题），有参考解（能数行数）+ 五档难度；
- 是 I/O 格式（stdin/stdout + 样例），和我们的评测口径一致。

【过滤规则（必须，否则会混进脏数据）】
1. 交互题：question 含 "interactive"（大小写不敏感）→ 跳过（我们的 one-shot harness 跑不了交互题）
2. 空解：solutions 为空 → 跳过（没参考解就没法数行数 + 自检）
3. Python2 语法：参考解 ast.parse 失败 → 跳过（print 语句等旧语法）

【分桶】按参考解（第 1 个解）的行数：
  [0,20) [20,40) [40,60) [60,80) [80,120) [120,200) [200,∞)
  每桶最多 10 题。

【测试样例】每题取前 3 组 input/output（保证「通过」标准一致 + 省时；最终评测再上全量样例 + 按比例给分）

【输出】
  ../benchmark/problems_cliff/<pXXX_<name>>/problem.json + reference.py
  以及 ../benchmark/problems_cliff/_manifest.json （id -> 行数/桶/难度，供后续分析用）

【运行】
  python convert_taco_cliff.py
"""
import ast
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

from datasets import Dataset, concatenate_datasets

TACO_DIR = os.path.expanduser("~/autodl-tmp/TACO")
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmark", "problems_cliff"))

BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 120), (120, 200), (200, 10**9)]
PER_BIN = 10
MAX_TESTS = 5


def _as_list(v):
    """solutions 字段可能是 JSON 字符串或已解析的 list，统一成 list。"""
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return []
    return v if isinstance(v, list) else []


def _as_io(v):
    """input_output 字段可能是 JSON 字符串或 dict，统一成 {'inputs': [...], 'outputs': [...]}。"""
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return {}
    return v if isinstance(v, dict) else {}


def ref_line_count(solutions_val):
    """返回第 1 个参考解的行数；无解返回 None。"""
    sols = _as_list(solutions_val)
    if not sols or not isinstance(sols[0], str) or not sols[0].strip():
        return None
    return len(sols[0].splitlines())


def _norm(x):
    """把可能是 str 或 list 的样例统一成 str（list 用换行拼接）。"""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return "\n".join(str(y) for y in x)
    return str(x)


def parse_tests(input_output_val):
    """返回前 MAX_TESTS 组 [{'input': ..., 'output': ...}]。"""
    io = _as_io(input_output_val)
    ins = io.get("inputs", [])
    outs = io.get("outputs", [])
    tests = []
    for i in range(min(len(ins), len(outs), MAX_TESTS)):
        tests.append({"input": _norm(ins[i]), "output": _norm(outs[i])})
    return tests


def _norm_out(text):
    """对齐 harness 的输出归一化：去行尾空白 + 单独出现的 true/false 小写化。"""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln.lower() if ln.lower() in ("true", "false") else ln for ln in lines]
    return "\n".join(lines)


def ref_passes(ref_code, tests):
    """参考解写临时文件跑一遍 tests，全部通过才返回 True。
    用于过滤：Python2 运行时错误(raw_input)、错误解、数据不一致的题。"""
    td = tempfile.mkdtemp(prefix="cliff_check_")
    try:
        with open(os.path.join(td, "solution.py"), "w", encoding="utf-8") as f:
            f.write(ref_code)
        for case in tests:
            try:
                r = subprocess.run(
                    [sys.executable, "-B", "solution.py"],
                    cwd=td, input=case["input"], capture_output=True, text=True, timeout=10,
                )
            except subprocess.TimeoutExpired:
                return False
            if r.returncode != 0:
                return False
            if _norm_out(r.stdout) != _norm_out(case["output"]):
                return False
        return True
    finally:
        shutil.rmtree(td, ignore_errors=True)


def is_valid_py3(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:30]


def main():
    print("加载 TACO ...")
    files = sorted(glob.glob(os.path.join(TACO_DIR, "train", "*.arrow")))
    if not files:
        raise FileNotFoundError(f"没找到 TACO arrow：{TACO_DIR}/train/*.arrow")
    ds = concatenate_datasets([Dataset.from_file(f) for f in files])
    print(f"  共 {len(ds)} 条")

    # 第一遍：过滤 + 数行数 + 打印行数分布
    bins_count = {b: 0 for b in BINS}
    kept = []  # (line_count, bin, item)
    skip_interactive = skip_empty = skip_py2 = 0
    for item in ds:
        q = item.get("question") or ""
        if "interactive" in q.lower():
            skip_interactive += 1
            continue
        sols = _as_list(item.get("solutions"))
        if not sols:
            skip_empty += 1
            continue
        n = ref_line_count(item.get("solutions"))
        if n is None:
            skip_empty += 1
            continue
        if not is_valid_py3(sols[0]):
            skip_py2 += 1
            continue
        for b in BINS:
            if b[0] <= n < b[1]:
                bins_count[b] += 1
                kept.append((n, b, item))
                break

    print(f"\n过滤：交互 {skip_interactive}，空解 {skip_empty}，Python2 语法 {skip_py2}")
    print("\n参考解行数分布（过滤后）：")
    for b in BINS:
        lo, hi = b[0], b[1] if b[1] < 10**9 else "∞"
        print(f"  [{lo}, {hi}): {bins_count[b]} 题")

    # 第二遍：每桶按行数均匀取候选，逐个自检参考解，凑满 PER_BIN 个「参考解能通过」的
    selected = []  # (line_count, bin, item)
    for b in BINS:
        in_bin = sorted([(n, item) for (n, bb, item) in kept if bb == b], key=lambda x: x[0])
        if not in_bin:
            print(f"  ⚠️ 桶 [{b[0]}, {b[1]}): 0 题，跳过")
            continue
        n_cand = len(in_bin)
        k = min(PER_BIN * 3, n_cand)  # 取 3 倍缓冲，供自检淘汰
        order = [int(i * (n_cand - 1) / (k - 1)) for i in range(k)] if k > 1 else list(range(n_cand))
        order = sorted(set(order))  # 去重、升序，保持均匀
        got = 0
        for i in order:
            if got >= PER_BIN:
                break
            n, item = in_bin[i]
            sols = _as_list(item.get("solutions"))
            ref = sols[0] if sols else ""
            tests = parse_tests(item.get("input_output"))
            if not tests or not ref or not ref_passes(ref, tests):
                continue
            selected.append((n, b, item))
            got += 1
        print(f"  桶 [{b[0]}, {b[1]}): 抽 {got}/{len(in_bin)}")

    # 写文件（先清空旧目录，避免上次残留）
    import shutil
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {}
    n_written = 0
    for i, (n, b, item) in enumerate(selected):
        sols = _as_list(item.get("solutions"))
        ref = sols[0]
        tests = parse_tests(item.get("input_output"))
        if not tests:
            continue
        name = slugify(item.get("name") or f"taco_{i}")
        did = f"p{100 + i:03d}"
        d = os.path.join(OUT_DIR, f"{did}_{name}")
        os.makedirs(d, exist_ok=True)

        problem = {
            "id": did,
            "title": item.get("name") or did,
            "domain": "algorithm",
            "difficulty": item.get("difficulty") or "unknown",
            "mode": "io",
            "requirement": item.get("question") or "",
            "input_format": "As described in the Input section of the task statement.",
            "output_format": "As described in the Output section of the task statement.",
            "tests": tests,
        }
        with open(os.path.join(d, "problem.json"), "w", encoding="utf-8") as f:
            json.dump(problem, f, ensure_ascii=False, indent=2)
        with open(os.path.join(d, "reference.py"), "w", encoding="utf-8") as f:
            f.write(ref)

        manifest[did] = {"lines": n, "bin": f"[{b[0]},{b[1]})", "difficulty": problem["difficulty"],
                         "name": problem["title"]}
        n_written += 1

    with open(os.path.join(OUT_DIR, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 共写出 {n_written} 题到 {OUT_DIR}")
    print(f"   清单：{os.path.join(OUT_DIR, '_manifest.json')}")
    print("\n下一步：")
    print("  1) 自检参考解（应 100% 通过）：")
    print(f"     python run_eval.py --problems ../benchmark/problems_cliff --generator reference")
    print("  2) 跑 7B / 14B 找悬崖（k=3 抽样）：")
    print("     python run_eval.py --problems ../benchmark/problems_cliff --generator local --model /root/autodl-tmp/Qwen2.5-7B-Instruct --num-samples 3 --temperature 0.7 --run-name 7B_cliff")
    print("     python run_eval.py --problems ../benchmark/problems_cliff --generator local --model /root/autodl-tmp/Qwen2.5-14B-Instruct --num-samples 3 --temperature 0.7 --run-name 14B_cliff")


if __name__ == "__main__":
    main()
