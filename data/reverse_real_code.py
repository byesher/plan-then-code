# -*- coding: utf-8 -*-
"""
真实类代码逆向生成管线（pilot 版，只做第①步：AST 抽类）

把真实 Python 库里的「多方法类」逆向成 ClassEval 形状的三元组：
    需求(class_description) + 骨架(skeleton) + 实现(solution_code)   [测试 test 留空，后续阶段填]

【为什么要这么做】
结构轴的训练数据要「结构复杂、算法简单、多方法类」的真实代码，且不能是模型背过的库。
这个脚本做整条管线里最基础、零成本、确定性的一步：用 AST 把类抽出来。

【整条数据流（本脚本只做第①步，后面几步陆续补）】
  真实库源码
    → ① AST 抽类（本脚本）：类名 + 类文档(当需求) + 方法签名/docstring(当骨架) + 完整类源码(当实现)
    → ② LLM 反推需求（后续，防模型背库名；pilot 先用类文档顶替，零成本）
    → ③ 抽测试（后续，复用库自带 unittest/pytest）
    → ④ 闭环校验（solution 跑通测试才保留）
    → ⑤ build_class_sft.py 转两段式 SFT 样本（已写好）

【运行（AutoDL，两行）】
  git clone https://github.com/mahmoud/boltons.git
  python reverse_real_code.py --repo boltons --max-classes 50 --out boltons_classes.jsonl
  → 打开 boltons_classes.jsonl，每条 = 一个类 → 需求/骨架/实现

【参数】
  --repo       库地址（git URL 或本地路径）
  --src-dir    库源码子目录（默认自动找；如 boltons 的源码在 boltons/boltons/）
  --min-methods 最少方法数（默认 2，<2 没有「结构分解」可练）
  --max-classes 最多抽多少个类（pilot 先少抽点看看）
  --out        输出 jsonl

依赖：标准库 only（ast/subprocess/os/json/collections）。
"""
import argparse
import ast
import builtins
import json
import os
import subprocess
import sys
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import build_class_sft as B  # 复用它的 AST 工具（_sanitize/_fn_def/_quote_block/...）

# 骨架约定：__init__/__new__ 保留真实方法体（构造函数初始化字段是「给定结构」），
# 其余方法体 pass 占位（这些才是要模型「照着签名实现」的部分）。
KEEP_BODY_METHODS = ("__init__", "__new__")


def is_git_url(s):
    return isinstance(s, str) and (s.startswith(("http://", "https://", "git@", "ssh://")))


def clone_repo(repo, dst):
    """git clone（浅克隆省时间），失败抛异常。"""
    cmd = ["git", "clone", "--depth", "1", repo, dst]
    print("  克隆：", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"git clone 失败：{(r.stderr or r.stdout).strip()[:500]}")


def find_src_dir(repo_dir):
    """找到真正的「包源码」目录（跳过 .git/test/venv）。"""
    # 常见：repo/ 下有个同名包目录，如 boltons/boltons/
    for name in sorted(os.listdir(repo_dir)):
        p = os.path.join(repo_dir, name)
        if os.path.isdir(p) and name not in (".git", "tests", "test", "docs", "venv", "__pycache__"):
            # 目录里有没有 .py 文件？
            has_py = any(f.endswith(".py") for _, _, fs in os.walk(p) for f in fs)
            if has_py and not name.startswith("."):
                return p
    return repo_dir


def walk_py_files(root):
    """遍历所有 .py 文件，跳过测试文件/目录。"""
    out = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "venv", "node_modules")
                   and "test" not in d.lower()]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            if "test" in fn.lower():
                continue
            out.append(os.path.join(dirpath, fn))
    return out


def build_skeleton(src, cls):
    """从真实类 AST 构建「结构骨架」：类定义 + 方法签名 + docstring，
    非构造方法体 pass 占位，构造方法保留真实体。"""
    mod = ast.parse(src)
    imports = [ast.get_source_segment(src, n).strip()
               for n in mod.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    header = ("\n".join(imports) + "\n\n") if imports else ""

    bases = ", ".join(ast.unparse(b) for b in cls.bases) if cls.bases else ""
    cls_line = f"class {cls.name}" + (f"({bases})" if bases else "") + ":"
    cls_doc = ast.get_docstring(cls, clean=True) or ""

    out = []
    if header:
        out.append(header.rstrip("\n"))
    out.append(cls_line)
    if cls_doc:
        out.append(B._quote_block(cls_doc, 4))

    for stmt in cls.body:
        if isinstance(stmt, ast.FunctionDef):
            fn = stmt
            defline = B._fn_def(fn).replace("\n", "\n    ")
            out.append("    " + defline)
            if fn.name in KEEP_BODY_METHODS:
                # 构造函数保留真实语句（去掉 docstring），这是「给定的字段初始化」
                stmts = [s for s in fn.body
                         if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                                 and isinstance(s.value.value, str))]
                if stmts:
                    for s in stmts:
                        out.append("        " + ast.unparse(s))
                else:
                    out.append("        pass")
            else:
                doc = ast.get_docstring(fn, clean=True) or ""
                if doc:
                    out.append(B._quote_block(doc, 8))
                out.append("        pass")
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            # 类级属性（__slots__、类常量）→ 属于结构，保留
            out.append("    " + ast.unparse(stmt).replace("\n", "\n    "))
        # 其余（docstring Expr、嵌套类等）跳过
    return "\n".join(out) + "\n"


def build_solution(src, cls, extra_defs=None):
    """完整类源码（imports + [依赖闭包] + 类定义），作为「实现」ground truth。"""
    mod = ast.parse(src)
    imports = [ast.get_source_segment(src, n).strip()
               for n in mod.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    cls_src = ast.get_source_segment(src, cls) or ast.unparse(cls)
    parts = []
    if imports:
        parts.append("\n".join(imports))
    if extra_defs:
        parts.append("\n\n".join(d for d in extra_defs if d))
    parts.append(cls_src)
    return "\n\n".join(p for p in parts if p)


_BUILTIN_NAMES = set(dir(builtins))


def is_self_contained(solution, class_name):
    """solution 能否独立 exec 并定义出类（抓：缺基类/缺装饰器/类级 NameError）。"""
    try:
        ns = {}
        exec(compile(solution, "<solution>", "exec"), ns)
        return class_name in ns
    except Exception:
        return False


def find_external_deps(solution, class_name):
    """抓「方法体读取了、但 solution 里哪里都没定义、也没 import、也不是内建」的名字。
    闭包后模块级常量/helper 已在 solution 里 → 不再算外部依赖。"""
    try:
        tree = ast.parse(solution)
    except SyntaxError:
        return ["<syntax-error>"]
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    if cls is None:
        return ["<no-class>"]

    # 已定义的名字 = 模块级定义 + 类级定义
    defined = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                defined |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
    for stmt in cls.body:
        if isinstance(stmt, ast.FunctionDef):
            defined.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                defined |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            defined.add(stmt.target.id)

    imported = set()
    for n in tree.body:
        if isinstance(n, ast.Import):
            imported |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            imported |= {(a.asname or a.name) for a in n.names if a.name != "*"}

    referenced = set()
    for stmt in cls.body:
        if isinstance(stmt, ast.FunctionDef):
            # 方法内被赋值(Store) + 参数名(ast.arg)都是局部；只留「读了但方法内没写」的名字
            stored = {n.id for n in ast.walk(stmt)
                      if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            stored |= {a.arg for a in ast.walk(stmt) if isinstance(a, ast.arg)}
            loaded = {n.id for n in ast.walk(stmt)
                      if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            referenced |= (loaded - stored)

    return sorted(referenced - defined - imported - _BUILTIN_NAMES - {"self", "cls"})


def collect_module_defs(src, exclude_name):
    """收集模块级定义（常量/函数/类）的源码，供依赖闭包用。"""
    tree = ast.parse(src)
    defs = {}
    for node in tree.body:
        seg = ast.get_source_segment(src, node)
        if not seg:
            continue
        if isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        defs[n.id] = seg
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defs[node.target.id] = seg
        elif isinstance(node, ast.FunctionDef):
            defs[node.name] = seg
        elif isinstance(node, ast.ClassDef) and node.name != exclude_name:
            defs[node.name] = seg
    return defs


def _should_skip(cls):
    """过滤非目标类，返回跳过原因；None 表示保留。
    目标 = 用户面向的「数据结构/工具」类，不是内部辅助/元编程/异常类。"""
    if cls.name.startswith("_"):
        return "internal(下划线开头)"
    method_names = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    if method_names & {"__get__", "__set__", "__delete__", "__set_name__"}:
        return "descriptor(描述符类)"
    if cls.name.endswith(("Error", "Exception", "Warning", "Mixin")):
        return "exception/mixin(名字)"
    for b in cls.bases:
        bn = ast.unparse(b)
        if bn in ("Exception", "BaseException", "Warning", "UserWarning", "ABC"):
            return f"base({bn})"
    return None


def extract_class(src, cls, src_file):
    """把一个 ast.ClassDef 抽成 ClassEval 形状的 dict；不合格返回 None。"""
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)]
    if len(methods) < MIN_METHODS:
        return None
    cls_doc = ast.get_docstring(cls, clean=True) or ""
    if not cls_doc:
        return None  # 没文档说明，反推不出需求

    skeleton = build_skeleton(src, cls)
    solution = build_solution(src, cls)
    # 骨架里不能有语法错误（自检）
    try:
        ast.parse(skeleton)
    except SyntaxError:
        return None

    # 依赖闭包：方法体引用的模块级常量/helper + 继承的模块级基类，一并带进 solution
    ext_deps = find_external_deps(solution, cls.name)
    module_defs = collect_module_defs(src, cls.name)
    closure = list(dict.fromkeys(module_defs[d] for d in ext_deps
                                 if d in module_defs and not d.startswith("<")))
    for b in cls.bases:  # 基类闭包（如 FastIterOrderedMultiDict 继承 OrderedMultiDict）
        if isinstance(b, ast.Name) and b.id in module_defs and b.id not in _BUILTIN_NAMES:
            closure.append(module_defs[b.id])
    closure = list(dict.fromkeys(closure))
    if closure:
        solution = build_solution(src, cls, extra_defs=closure)
        ext_deps = find_external_deps(solution, cls.name)  # 闭包后重算

    methods_info = []
    for fn in methods:
        methods_info.append({
            "method_name": fn.name,
            "method_description": B._fn_def(fn) + "\n" + (ast.get_docstring(fn, clean=True) or ""),
        })

    return {
        "task_id": src_file.replace("/", "_").replace("\\", "_").replace(".py", "") + "_" + cls.name,
        "skeleton": skeleton,
        "solution_code": solution,
        "test": "",  # 后续阶段填（复用库自带测试，评测集用）
        "class_description": cls_doc,
        "class_name": cls.name,
        "import_statement": [],  # solution_code 已含 imports，这里留空
        "methods_info": methods_info,
        "source_file": src_file,
        "n_methods": len(methods),
        "self_contained": is_self_contained(solution, cls.name),
        "external_deps": ext_deps,
    }


def main():
    ap = argparse.ArgumentParser(description="真实类代码逆向（pilot：AST 抽类）")
    ap.add_argument("--repo", type=str, required=True, help="库 git URL 或本地路径")
    ap.add_argument("--src-dir", type=str, default="", help="源码子目录（默认自动找）")
    ap.add_argument("--min-methods", type=int, default=2, help="最少方法数（默认 2）")
    ap.add_argument("--max-classes", type=int, default=50, help="最多抽多少类（pilot 少抽）")
    ap.add_argument("--out", type=str, default="classes.jsonl", help="输出 jsonl")
    args = ap.parse_args()

    global MIN_METHODS
    MIN_METHODS = args.min_methods

    # 1) 拿到源码目录
    if is_git_url(args.repo):
        repo_dir = os.path.join(".", os.path.basename(args.repo.rstrip("/")).replace(".git", ""))
        if not os.path.isdir(repo_dir):
            clone_repo(args.repo, repo_dir)
    else:
        repo_dir = args.repo
        if not os.path.isdir(repo_dir):
            raise FileNotFoundError(f"找不到路径：{repo_dir}")
    print(f"源码目录：{os.path.abspath(repo_dir)}")

    src_root = args.src_dir if args.src_dir else find_src_dir(repo_dir)
    print(f"包源码目录：{os.path.abspath(src_root)}\n")

    # 2) 遍历 .py，AST 抽类
    py_files = walk_py_files(src_root)
    print(f"扫描 {len(py_files)} 个 .py 文件 ...")
    classes = []
    skip_stats = Counter()
    for fp in py_files:
        try:
            src = B._sanitize(open(fp, encoding="utf-8").read())
            tree = ast.parse(src)
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                skip = _should_skip(node)
                if skip:
                    skip_stats[skip] += 1
                    continue
                rel = os.path.relpath(fp, repo_dir)
                c = extract_class(src, node, rel)
                if c:
                    classes.append(c)
                    if len(classes) >= args.max_classes:
                        break
        if len(classes) >= args.max_classes:
            break

    # 3) 写 jsonl
    out = args.out if os.path.isabs(args.out) else os.path.join(SCRIPT_DIR, args.out)
    with open(out, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 4) 摘要（让用户一眼看懂抽到了什么）
    method_dist = dict(Counter(c["n_methods"] for c in classes))
    n_clean = sum(1 for c in classes if c["self_contained"] and not c["external_deps"])
    print(f"\n✅ 抽出 {len(classes)} 个类（≥{MIN_METHODS} 方法 + 有文档），写到：{out}")
    print(f"   方法数分布：{dict(sorted(method_dist.items()))}")
    print(f"   完全自包含（能独立跑、无模块级依赖）：{n_clean}/{len(classes)}")
    if skip_stats:
        print(f"   过滤掉的类：{dict(skip_stats)}")
    print("\n   示例（前 3 个类名 + 方法数 + 是否自包含）：")
    for c in classes[:3]:
        mark = "✔" if (c["self_contained"] and not c["external_deps"]) else "✘ 依赖:" + ",".join(c["external_deps"][:4])
        print(f"     - {c['class_name']:<28} {c['n_methods']} 方法  {mark}  <- {c['source_file']}")
    print("\n下一步：只有「完全自包含」的类才适合当训练数据（能独立跑）。")
    print("       把 jsonl 里 self_contained=true 且 external_deps=[] 的筛出来做训练。")
    print("转两段式样本：python build_class_sft.py --input classes.jsonl --out-dir xxx --no-verify")


if __name__ == "__main__":
    main()
