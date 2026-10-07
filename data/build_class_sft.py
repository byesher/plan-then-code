# -*- coding: utf-8 -*-
"""
结构轴 plan-then-code 数据构造（ClassEval 形状 → 两段式 SFT 样本）

【背景】推理轴（伪代码）在算法题上已被四次 SFT 证伪（4%/6%/8%/14% 均 << baseline），
瓶颈是「算法知识」不是「结构」。结构轴的靶场 = 结构复杂、算法简单的多函数/多方法代码，
瓶颈是「结构分解」（模块税 29pp 的证据）。

【核心思路（结构轴，与推理轴的关键区别）】
1. 中间结构 = 类/函数的「签名 + 契约 / TODO 桩」，它【客观存在于真实代码里】，
   用 AST 确定性抽取（零幻觉、零成本），不像伪代码要靠 LLM 反推。
2. 唯一 AI 步骤 = 上游「从代码反推自然语言需求」（reverse_real_code.py 做），
   本脚本只消费已经成形的「ClassEval 形状三元组」。
3. 两段式：
     样本A（结构） 需求 → 结构骨架（签名+契约，方法体 pass 占位）     —— 练「结构分解」
     样本B（实现） 需求 + 结构骨架 → 完整代码                         —— 练「忠实实现」
4. 闭环校验：solution_code 必须通过自带 unittest 测试才保留（质量闸门）。

【4 种结构轴中间结构（消融变量）】
   ① sig              函数签名（无 docstring）
   ② sig_contract     签名 + 一句话契约
   ③ typed_docstring  签名 + 完整类型/docstring（= ClassEval 原生 skeleton）
   ④ todo             签名 + # TODO 桩

【输入】ClassEval 形状 jsonl，每行字段：
   task_id, skeleton(签名+docstring 空体), solution_code(完整实现),
   test(unittest 测试), class_description, class_name, import_statement(list), methods_info(list)

【运行】
   python build_class_sft.py --input classeval_test.jsonl --out-dir class_train --form all
   python build_class_sft.py --input xxx.jsonl --out-dir class_train --check   # 只校验不写

依赖：标准库 only（ast/re/subprocess/textwrap）。
"""
import argparse
import ast
import json
import os
import subprocess
import sys
import textwrap
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "classeval_data"))

FORMS = ["sig", "sig_contract", "typed_docstring", "todo"]
FORM_CN = {
    "sig": "①函数签名",
    "sig_contract": "②签名+契约",
    "typed_docstring": "③类型标注+docstring",
    "todo": "④TODO桩",
}

VERIFY_TIMEOUT = 30
SEED = 42

# ── 需求模板（评测脚本必须复用同一模板，保持训练/评测一致）────────────
# 需求格式由 build_requirement() 统一生成：类名 + 类描述 + 方法名+一句话用途（不给签名）。

STRUCT_PROMPT = (
    "{requirement}"
    "请先写出这个类的【结构骨架】：根据上面的方法用途，推导每个方法的签名（参数与返回类型），"
    "写出类定义 + 每个方法的签名 + 一句话文档字符串契约，方法体用 pass 占位，不要实现方法体。"
    "只输出代码，放在 ```python ... ``` 内。"
)

IMPL_PROMPT = (
    "{requirement}"
    "【结构骨架】\n{structure}\n\n"
    "请严格按骨架实现每个方法体，输出完整可运行的 Python 类。"
    "只输出代码，放在 ```python ... ``` 内，不要解释、不要伪代码。"
)


# ── 数据清洗 ─────────────────────────────────────────────────────
def _sanitize(code):
    """清洗 ClassEval 常见脏数据，返回干净代码（无法修复就原样返回）。"""
    if code is None:
        return ""
    # 1) 去掉 BOM（U+FEFF，Windows 编辑器/某些仓库文件会带）
    code = code.lstrip("\ufeff")
    # 2) 智能引号 → ASCII 引号（ClassEval_43/46 用了 “ ” ‘ ’）
    code = code.replace("\u201c", '"').replace("\u201d", '"')
    code = code.replace("\u2018", "'").replace("\u2019", "'")
    # 3) 整段被三引号字符串包裹（ClassEval_5 的 skeleton 是 '''...'''），解包取内部代码
    try:
        tree = ast.parse(code)
        if (len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)):
            return tree.body[0].value.value.lstrip("\n")
    except SyntaxError:
        pass
    return code


# ── AST 工具 ─────────────────────────────────────────────────────
def _clean_docstring(text):
    """把 ClassEval 的 class_description（带引号/缩进的 docstring）洗成纯文本。"""
    if text is None:
        return ""
    text = text.strip()
    if text.startswith(('"""', "'''")):
        text = text[3:]
    if text.endswith(('"""', "'''")):
        text = text[:-3]
    return textwrap.dedent(text).strip()


def _doc_summary(node):
    """取 docstring 第一个非空行（一句话契约）。"""
    doc = ast.get_docstring(node, clean=True) or ""
    for ln in doc.splitlines():
        ln = ln.strip()
        if ln:
            return ln
    return ""


def _is_empty_body(fn):
    """判断方法体是否『空』（只有 docstring / pass，没有真实语句）。"""
    meaningful = [s for s in fn.body
                  if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                          and isinstance(s.value.value, str))
                  and not isinstance(s, ast.Pass)]
    return len(meaningful) == 0


def _fn_def(fn):
    """从 ast.FunctionDef 还原『装饰器 + def 行』（不含方法体）。"""
    decs = "\n".join("@" + ast.unparse(d) for d in fn.decorator_list)
    dummy = ast.FunctionDef(name=fn.name, args=fn.args, returns=fn.returns,
                            decorator_list=[], type_comment=None,
                            body=[ast.Pass()], lineno=fn.lineno)
    defline = ast.unparse(ast.fix_missing_locations(dummy)).split("\n")[0]
    return (decs + "\n" + defline) if decs else defline


def _quote_block(text, indent):
    """生成缩进后的三引号 docstring 块。"""
    pad = " " * indent
    lines = text.splitlines()
    if not lines:
        return pad + '"""' + '"""'
    if len(lines) == 1:
        return pad + '"""' + lines[0] + '"""'
    out = [pad + '"""']
    for ln in lines:
        out.append(pad + ln)
    out.append(pad + '"""')
    return "\n".join(out)


def build_class_structure(skeleton, form):
    """从 ClassEval skeleton 构建结构轴 4 种中间结构之一（返回合法 Python 文本）。"""
    tree = ast.parse(skeleton)
    cls = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            cls = node
            break
    if cls is None:
        raise ValueError("skeleton 里没有 ClassDef")

    imports = [ast.get_source_segment(skeleton, n).strip()
               for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    header = ("\n".join(imports) + "\n\n") if imports else ""

    bases = ", ".join(ast.unparse(b) for b in cls.bases) if cls.bases else ""
    cls_line = f"class {cls.name}" + (f"({bases})" if bases else "") + ":"
    cls_summary = _doc_summary(cls)
    cls_full = ast.get_docstring(cls, clean=True) or ""
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)]

    out = []
    if header:
        out.append(header.rstrip("\n"))
    out.append(cls_line)

    # 类级 docstring / TODO
    if form == "typed_docstring" and cls_full:
        out.append(_quote_block(cls_full, 4))
    elif form == "sig_contract" and cls_summary:
        out.append(_quote_block(cls_summary, 4))
    elif form == "todo" and cls_summary:
        out.append("    # TODO: " + cls_summary)

    for fn in methods:
        if not _is_empty_body(fn):
            # 已实现的方法（如 __init__ 初始化字段）保留 def 行 + 真实语句（去掉 docstring）
            defline = _fn_def(fn).replace("\n", "\n    ")
            stmts = [s for s in fn.body
                     if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                             and isinstance(s.value.value, str))]
            out.append("    " + defline)
            if stmts:
                for s in stmts:
                    out.append("        " + ast.unparse(s))
            else:
                out.append("        pass")
            continue
        out.append("    " + _fn_def(fn).replace("\n", "\n    "))
        summary = _doc_summary(fn)
        full = ast.get_docstring(fn, clean=True) or ""
        if form == "sig":
            out.append("        pass")
        elif form == "sig_contract":
            if summary:
                out.append(_quote_block(summary, 8))
            out.append("        pass")
        elif form == "typed_docstring":
            if full:
                out.append(_quote_block(full, 8))
            out.append("        pass")
        elif form == "todo":
            if summary:
                out.append("        # TODO: " + summary)
            out.append("        pass")
    return "\n".join(out) + "\n"


def extract_method_contracts(skeleton):
    """从 skeleton 抽每个方法的 (name, summary)。需求只给方法名+用途，不给签名参数
    （签名是 Stage A 要让模型自己推导出来的结构）。"""
    tree = ast.parse(skeleton)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    out = []
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        summary = _doc_summary(fn)
        if not summary:
            summary = "构造方法，初始化对象" if fn.name == "__init__" else ""
        out.append({"name": fn.name, "summary": summary})
    return out


def build_requirement(class_name, class_description, contracts):
    desc = _clean_docstring(class_description)
    lines = [f"实现一个类 `{class_name}`：", "", desc, "", "需要实现的方法："]
    for c in contracts:
        tail = f"：{c['summary']}" if c["summary"] else ""
        lines.append(f"- `{c['name']}`{tail}")
    return "\n".join(lines) + "\n"


# ── 闭环校验 ─────────────────────────────────────────────────────
def verify_solution(item):
    """合并 import_statement + solution_code + test，跑 unittest，返回 (ok, err)。"""
    imports = item.get("import_statement") or []
    if isinstance(imports, str):
        imports = [imports]
    solution = item.get("solution_code") or ""
    test = item.get("test") or ""
    if not solution or not test:
        return False, "empty(solution_or_test)"

    combined = "\n".join(imports) + "\n\n" + solution + "\n\n" + test
    if "unittest.main" not in test:
        combined += "\n\nif __name__ == '__main__':\n    unittest.main()\n"

    # 固定工作区临时目录（系统 Temp 在沙箱下不可写，tempfile.mkdtemp 会 PermissionError）
    td = os.path.join(DATA_DIR, "_verify_tmp")
    os.makedirs(td, exist_ok=True)
    fp = os.path.join(td, "combined_test.py")
    try:
        with open(fp, "w", encoding="utf-8") as f:
            f.write(combined)
        try:
            r = subprocess.run([sys.executable, "-B", "combined_test.py"], cwd=td,
                               capture_output=True, text=True, timeout=VERIFY_TIMEOUT,
                               encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if r.returncode == 0:
            return True, ""
        err = (r.stderr or r.stdout or "").strip()
        # 取最后一段有意义的报错行，方便定位
        tail = [ln for ln in err.splitlines() if ln.strip()]
        err_short = tail[-1] if tail else "unknown"
        if len(err_short) > 160:
            err_short = err_short[:160]
        return False, err_short
    finally:
        try:
            os.remove(fp)
        except OSError:
            pass


# ── 单条样本构造 ─────────────────────────────────────────────────
def build_samples(item, forms):
    """对一条 ClassEval 形状三元组，产出 {form: (structure_sample, impl_sample)}。"""
    class_name = item.get("class_name") or "Solution"
    skeleton = item.get("skeleton") or ""
    solution = item.get("solution_code") or ""
    contracts = extract_method_contracts(skeleton)
    req = build_requirement(class_name, item.get("class_description"), contracts)

    out = {}
    for form in forms:
        try:
            structure = build_class_structure(skeleton, form)
        except Exception as e:
            continue
        struct_sample = {
            "form": form,
            "task_id": item.get("task_id"),
            "class_name": class_name,
            "requirement": req,
            "structure": structure,
            "messages": [
                {"role": "user", "content": STRUCT_PROMPT.format(requirement=req)},
                {"role": "assistant", "content": f"```python\n{structure}\n```"},
            ],
        }
        impl_sample = {
            "form": form,
            "task_id": item.get("task_id"),
            "class_name": class_name,
            "requirement": req,
            "structure": structure,
            "code": solution,
            "messages": [
                {"role": "user", "content": IMPL_PROMPT.format(requirement=req, structure=structure)},
                {"role": "assistant", "content": f"```python\n{solution}\n```"},
            ],
        }
        out[form] = (struct_sample, impl_sample)
    return out


# ── 前置校验（--check）───────────────────────────────────────────
def do_check(args):
    print("== 前置校验（只查输入格式 + 闭环通过率，不写输出）==\n")
    if not os.path.isfile(args.input):
        print(f"[FAIL] 输入文件不存在: {args.input}")
        return
    items = []
    with open(args.input, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                it = json.loads(ln)
                for key in ("skeleton", "solution_code", "test"):
                    if key in it:
                        it[key] = _sanitize(it[key])
                items.append(it)
    print(f"[OK] 输入 {len(items)} 条")

    need = ["skeleton", "solution_code", "test", "class_name", "class_description"]
    missing = [k for k in need if any(k not in it for it in items)]
    print(f"[{'OK' if not missing else 'FAIL'}] 必需字段齐全（缺: {missing or '无'}）")

    # 方法数分布
    mc = []
    for it in items:
        try:
            tree = ast.parse(it["skeleton"])
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
            mc.append(sum(1 for n in cls.body if isinstance(n, ast.FunctionDef)))
        except Exception:
            mc.append(0)
    if mc:
        print(f"[OK] 方法数分布: min={min(mc)} 中位={sorted(mc)[len(mc)//2]} max={max(mc)} avg={sum(mc)/len(mc):.1f}")

    # 闭环通过率（抽样最多 5 条，快）
    sample = items[:5]
    ok = 0
    for it in sample:
        passed, err = verify_solution(it)
        ok += passed
        if not passed:
            print(f"   ✗ {it.get('task_id')}: {err[:100]}")
    print(f"[{'OK' if ok == len(sample) else 'WARN'}] 闭环校验抽样 {ok}/{len(sample)} 通过")
    print("\n上面有 FAIL 就修完再跑全量。")


# ── 主流程 ───────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str,
                    default=os.path.join(DATA_DIR, "classeval_test.jsonl"),
                    help="ClassEval 形状 jsonl（默认 data/classeval_data/classeval_test.jsonl）")
    ap.add_argument("--out-dir", type=str, default="class_train", help="输出目录（相对 data/ 或绝对路径）")
    ap.add_argument("--form", type=str, default="all",
                    help="结构形式：all | sig | sig_contract | typed_docstring | todo")
    ap.add_argument("--no-verify", action="store_true",
                    help="跳过闭环校验（试水阶段还没抽测试时用；正式数据必须校验）")
    ap.add_argument("--check", action="store_true", help="只做前置校验，不生成数据")
    args = ap.parse_args()

    if args.check:
        do_check(args)
        return

    forms = FORMS if args.form == "all" else [args.form]
    for fm in forms:
        if fm not in FORMS:
            raise ValueError(f"未知形式 {fm}，可选 {FORMS}")

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(DATA_DIR, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    items = []
    with open(args.input, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                it = json.loads(ln)
                for key in ("skeleton", "solution_code", "test"):
                    if key in it:
                        it[key] = _sanitize(it[key])
                items.append(it)
    print(f"加载 {len(items)} 条 ClassEval 形状数据")

    # 每个 form 一个子目录
    form_dirs = {}
    for fm in forms:
        d = os.path.join(out_dir, fm)
        os.makedirs(d, exist_ok=True)
        form_dirs[fm] = d

    kept = {fm: 0 for fm in forms}
    rejected = []
    method_dist = []
    total = len(items)

    # 清空旧输出（避免重跑重复追加）
    for fm in forms:
        for fn in ("structure_sft.jsonl", "impl_sft.jsonl"):
            open(os.path.join(form_dirs[fm], fn), "w", encoding="utf-8").close()
    open(os.path.join(out_dir, "rejected.jsonl"), "w", encoding="utf-8").close()

    for i, it in enumerate(items):
        if args.no_verify:
            passed, err = True, ""
        else:
            passed, err = verify_solution(it)
        if not passed:
            rejected.append({"task_id": it.get("task_id"), "err": err, "stage": "verify"})
            continue
        try:
            tree = ast.parse(it["skeleton"])
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
            method_dist.append(sum(1 for n in cls.body if isinstance(n, ast.FunctionDef)))
        except Exception:
            pass
        samples = build_samples(it, forms)
        for fm in forms:
            if fm not in samples:
                rejected.append({"task_id": it.get("task_id"), "err": f"structure_build_fail:{fm}",
                                 "stage": "structure"})
                continue
            struct_sample, impl_sample = samples[fm]
            with open(os.path.join(form_dirs[fm], "structure_sft.jsonl"), "a", encoding="utf-8") as fa, \
                 open(os.path.join(form_dirs[fm], "impl_sft.jsonl"), "a", encoding="utf-8") as fb:
                fa.write(json.dumps(struct_sample, ensure_ascii=False) + "\n")
                fb.write(json.dumps(impl_sample, ensure_ascii=False) + "\n")
            kept[fm] += 1
        if (i + 1) % 20 == 0:
            print(f"  已处理 {i+1}/{total}，各形式保留 {kept}", flush=True)

    # rejected
    with open(os.path.join(out_dir, "rejected.jsonl"), "w", encoding="utf-8") as fr:
        for r in rejected:
            fr.write(json.dumps(r, ensure_ascii=False) + "\n")

    # manifest
    manifest = {
        "input": args.input,
        "total": total,
        "kept_per_form": kept,
        "rejected": len(rejected),
        "acceptance_rate": {fm: (kept[fm] / total if total else 0.0) for fm in forms},
        "method_dist": {"min": min(method_dist) if method_dist else None,
                        "median": sorted(method_dist)[len(method_dist)//2] if method_dist else None,
                        "max": max(method_dist) if method_dist else None,
                        "avg": round(sum(method_dist)/len(method_dist), 2) if method_dist else None},
        "reject_stage_dist": dict(Counter(r["stage"] for r in rejected)),
        "config": {"verify_timeout": VERIFY_TIMEOUT, "forms": forms, "no_verify": args.no_verify},
        "requirement_format": "类名 + 类描述 + 方法名+一句话用途（不含签名）",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fm:
        json.dump(manifest, fm, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成：输入 {total} 条，闭环通过后各形式保留 {kept}")
    for fm in forms:
        print(f"   {FORM_CN[fm]} ({fm})：{os.path.join(form_dirs[fm], 'structure_sft.jsonl')} + impl_sft.jsonl")
    print(f"   被拒样本+原因：{os.path.join(out_dir, 'rejected.jsonl')}")
    print(f"   统计清单：{os.path.join(out_dir, 'manifest.json')}")
    print("下一步：改 train_lora.py 按 form 目录加载两段式样本做结构轴 SFT 消融。")


if __name__ == "__main__":
    main()
