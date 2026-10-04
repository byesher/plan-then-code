# -*- coding: utf-8 -*-
"""
TACO → SFT 训练集原料 抽取脚本（额外题目，与 70 题测试集零重叠）

【目的】
从 TACO（I/O 竞赛题，全量 arrow 在 ~/autodl-tmp/TACO）里，抽取【测试集之外】的题目，
作为 plan-then-code 的小训练集【原料】。后续步骤：
  1) 本脚本：抽 (需求, 参考解, 样例) 原始原料；
  2) 加工脚本（下一步写）：用强 LLM 给参考解反推「伪代码大纲」→ (需求→伪代码→代码) 三元组；
  3) LoRA SFT：训 Qwen2.5-7B 先出伪代码再出代码。
测试集（problems_cliff 70 题）保持不动，留作最终评测。

【红线（必须保证）】
- 抽取时【必须排除】已进测试集的 70 题：按「题目原文 question」精确匹配 +「name/difficulty」双保险。
- 训练集 / dev 集与 70 题测试集零重叠。

【过滤规则（同 convert_taco_cliff.py，保证数据质量）】
1. 交互题：question 含 "interactive" → 跳过
2. 空解：solutions 为空 → 跳过
3. Python2 语法：参考解 ast.parse 失败 → 跳过
4. 无 input_output 测试样例 → 跳过（训练暂时不需要样例，但留样例便于后续 hold-out 自检）
5. 参考解跑不过样例 → 跳过（保证训练用的参考代码是【对的】，这是数据质量底线）

【抽样】
- 从过滤后的候选中，固定随机种子 shuffle，取 (TRAIN_N+DEV_N)*3 缓冲，逐个自检参考解，
  凑满 TRAIN_N + DEV_N 个「干净」题。
- TRAIN_N → train_raw.jsonl（训练原料）；DEV_N → dev_raw.jsonl（pilot 的 hold-out 快速评测，
  不碰 70 题测试集）。

【输出】
  data/taco_train/train_raw.jsonl   —— 训练原料
  data/taco_train/dev_raw.jsonl     —— hold-out 原料
  data/taco_train/_manifest.json    —— 抽取统计 + 难度分布 + 排除审计

【运行】
  python extract_train_set.py

依赖：datasets（读 arrow）。在 AutoDL 上跑（TACO arrow 在 ~/autodl-tmp/TACO）。
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

# ── 路径与参数 ─────────────────────────────────────────────────
TACO_DIR = os.path.expanduser("~/autodl-tmp/TACO")
HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PROBLEMS_DIR = os.path.abspath(os.path.join(HARNESS_DIR, "..", "benchmark", "problems_cliff"))
OUT_DIR = os.path.abspath(os.path.join(HARNESS_DIR, "..", "..", "data", "taco_train"))

TRAIN_N = 300        # 训练原料条数
DEV_N = 50           # 留作 hold-out 的条数（pilot 快速评测，不碰 70 题测试集）
SEED = 42            # 固定随机种子，保证可复现
BUFFER = 3           # 自检缓冲倍数（抽 3 倍，自检淘汰后仍够）
CHECK_MAX_TESTS = 3  # 参考解自检跑前 N 个样例（跑全量太慢）
CHECK_TIMEOUT = 10   # 单个样例超时秒数
DIFFICULTY_FILTER = None  # 例如 {"EASY", "MEDIUM"} 只抽简单题；None = 不过滤


# ── 通用工具（同 convert_taco_cliff.py 的语义）─────────────────
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


def _norm(x):
    """把可能是 str 或 list 的样例统一成 str。"""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return "\n".join(str(y) for y in x)
    return str(x)


def parse_tests(input_output_val):
    """返回 [{'input': ..., 'output': ...}]，不设上限（训练原料里样例越多越利于后续自检）。"""
    io = _as_io(input_output_val)
    ins = io.get("inputs", [])
    outs = io.get("outputs", [])
    tests = []
    for i in range(min(len(ins), len(outs))):
        tests.append({"input": _norm(ins[i]), "output": _norm(outs[i])})
    return tests


def is_valid_py3(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _norm_out(text):
    """对齐 harness 的输出归一化：去行尾空白 + 单独出现的 true/false 小写化。"""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln.lower() if ln.lower() in ("true", "false") else ln for ln in lines]
    return "\n".join(lines)


def ref_passes(ref_code, tests):
    """参考解写临时文件跑一遍 tests，全部通过才返回 True（过滤跑不通/错误的参考解）。"""
    td = tempfile.mkdtemp(prefix="train_check_")
    try:
        with open(os.path.join(td, "solution.py"), "w", encoding="utf-8") as f:
            f.write(ref_code)
        for case in tests:
            try:
                r = subprocess.run(
                    [sys.executable, "-B", "solution.py"],
                    cwd=td, input=case["input"], capture_output=True, text=True, timeout=CHECK_TIMEOUT,
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


def load_test_set_keys():
    """读 70 题测试集的 problem.json，返回 (questions, name_diff, n_files)。
    questions: set[str]  题目原文（精确匹配主键）
    name_diff: set[tuple] (name, difficulty) 双保险副键
    """
    questions = set()
    name_diff = set()
    files = sorted(glob.glob(os.path.join(TEST_PROBLEMS_DIR, "*", "problem.json")))
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                p = json.load(f)
        except Exception as e:
            print(f"  ⚠️ 读失败 {fp}: {e}")
            continue
        q = (p.get("requirement") or "").strip()
        if q:
            questions.add(q)
        name_diff.add((p.get("title"), p.get("difficulty")))
    return questions, name_diff, len(files)


# ── 主流程 ─────────────────────────────────────────────────────
def main():
    print("加载 TACO ...")
    files = sorted(glob.glob(os.path.join(TACO_DIR, "train", "*.arrow")))
    if not files:
        raise FileNotFoundError(
            f"没找到 TACO arrow：{TACO_DIR}/train/*.arrow\n"
            f"先下载：huggingface-cli download BAAI/TACO --repo-type dataset --local-dir ~/autodl-tmp/TACO"
        )
    ds = concatenate_datasets([Dataset.from_file(f) for f in files])
    print(f"  共 {len(ds)} 条")

    questions, name_diff, n_test_files = load_test_set_keys()
    print(f"  测试集 {n_test_files} 题；排除键：question={len(questions)} 条、name/diff={len(name_diff)} 条\n")

    # 第一遍：过滤（不抽样，先得到干净候选池）
    kept = []  # (item, sols, tests)
    skip = {"interactive": 0, "empty": 0, "py2": 0, "no_tests": 0, "overlap": 0, "diff_filter": 0}
    n_name_match = 0  # 审计：与测试集 name/difficulty 相同但题目原文不同的条数（应≈0，非 0 需人工查）
    for item in ds:
        q = (item.get("question") or "").strip()
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
        # 【红线】排除已进测试集的 70 题 —— 只用「题目原文」精确匹配（最稳，绝不误杀）。
        #   name 字段在 TACO 可能为 None，故 (name,difficulty) 只做审计、不做排除键。
        if q in questions:
            skip["overlap"] += 1
            continue
        if (item.get("name"), item.get("difficulty")) in name_diff:
            n_name_match += 1  # 原文不同但 name/diff 撞上 —— 记录下来人工核对
        # 可选难度过滤
        diff = item.get("difficulty") or ""
        if DIFFICULTY_FILTER is not None and diff not in DIFFICULTY_FILTER:
            skip["diff_filter"] += 1
            continue
        tests = parse_tests(item.get("input_output"))
        if not tests:
            skip["no_tests"] += 1
            continue
        kept.append((item, sols, tests))

    print("过滤统计：")
    print(f"  interactive={skip['interactive']}  empty={skip['empty']}  py2={skip['py2']}")
    print(f"  no_tests={skip['no_tests']}  测试集重叠(已排除)={skip['overlap']}  难度过滤={skip['diff_filter']}")
    print(f"  ⚠️ 审计：name/difficulty 撞测试集但原文不同 = {n_name_match} 条（应≈0，非 0 需人工查）")
    print(f"  剩余候选池：{len(kept)} 条\n")

    need = TRAIN_N + DEV_N
    if len(kept) < need * BUFFER:
        print(f"  ⚠️ 候选池 {len(kept)} 小于 {need * BUFFER}，自检淘汰后可能不够；继续自检全部候选。")

    # 第二遍：固定种子 shuffle，取缓冲，逐个自检参考解
    rng = random.Random(SEED)
    order = list(range(len(kept)))
    rng.shuffle(order)
    budget = min(len(kept), need * BUFFER)
    print(f"参考解自检（前 {CHECK_MAX_TESTS} 样例，预算 {budget} 条）...")
    clean = []  # (item, sols, tests, orig_idx)
    n_fail = 0
    for i in order[:budget]:
        item, sols, tests = kept[i]
        ref = sols[0]
        if ref_passes(ref, tests[:CHECK_MAX_TESTS]):
            clean.append((item, sols, tests, i))
        else:
            n_fail += 1
    print(f"  自检通过 {len(clean)}，淘汰 {n_fail}\n")

    if len(clean) < need:
        raise RuntimeError(
            f"干净的题不够：需要 {TRAIN_N}+{DEV_N}={need}，只有 {len(clean)}。\n"
            f"  对策：调小 TRAIN_N/DEV_N、或把 DIFFICULTY_FILTER 设 None、或增大 BUFFER。"
        )

    # 切分：train / dev（clean 已经 shuffle 过，前 TRAIN_N 给 train，接着 DEV_N 给 dev）
    train_items = clean[:TRAIN_N]
    dev_items = clean[TRAIN_N:need]

    os.makedirs(OUT_DIR, exist_ok=True)
    write_split(train_items, "train_raw.jsonl")
    write_split(dev_items, "dev_raw.jsonl")
    write_manifest(train_items, dev_items, skip, n_test_files)

    print("\n下一步：")
    print(f"  1) 加工：写脚本用强 LLM 给 {os.path.join(OUT_DIR, 'train_raw.jsonl')} 反推伪代码大纲")
    print(f"     → (需求→伪代码→代码) 三元组（训练格式）")
    print(f"  2) LoRA SFT：训 Qwen2.5-7B 先出伪代码再出代码")
    print(f"  3) 评测：dev_raw.jsonl 先做 hold-out 快评；最终再用 problems_cliff 70 题测试集")


def to_record(item, sols, tests, orig_idx):
    return {
        "id": f"train_{orig_idx:05d}",
        "name": item.get("name"),
        "source": item.get("source"),
        "difficulty": item.get("difficulty"),
        "question": item.get("question"),          # 需求原文
        "solution": sols[0],                       # 参考解（后续反推伪代码用）
        "all_solutions": sols,                     # 全部参考解（可能有多个）
        "tests": tests,                            # input/output，供 hold-out 自检
        "_ref_lines": len(sols[0].splitlines()),
    }


def write_split(clean_items, fname):
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        for item, sols, tests, orig_idx in clean_items:
            f.write(json.dumps(to_record(item, sols, tests, orig_idx), ensure_ascii=False) + "\n")
    print(f"  ✅ {path}（{len(clean_items)} 条）")


def write_manifest(train_items, dev_items, skip, n_test_files):
    """写抽取统计 + 难度分布 + 排除审计，供后续核对。"""
    def diff_dist(items):
        d = {}
        for item, sols, tests, orig_idx in items:
            k = item.get("difficulty") or "UNKNOWN"
            d[k] = d.get(k, 0) + 1
        return d

    manifest = {
        "seed": SEED,
        "train_n": len(train_items),
        "dev_n": len(dev_items),
        "test_set_excluded_files": n_test_files,
        "filter_stats": skip,
        "train_difficulty": diff_dist(train_items),
        "dev_difficulty": diff_dist(dev_items),
        "note": "训练/开发集已按题目原文排除 70 题测试集，零重叠。",
    }
    path = os.path.join(OUT_DIR, "_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {path}")


if __name__ == "__main__":
    main()
