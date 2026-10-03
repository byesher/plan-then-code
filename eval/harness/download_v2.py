# -*- coding: utf-8 -*-
"""
补全 / 修正 4 个数据集的抽样下载脚本（v2）

════════════════════════════════════════════════════════════════
【项目背景 —— 为什么需要这些数据】
════════════════════════════════════════════════════════════════
「plan-then-code」代码生成 SFT + 评测项目：
1. 评测端到端：给自然语言需求 → 模型自由写完整代码 → 只测「编译/运行 + 测试样例通过」。
2. 主评测形式 = I/O 题（stdin 读入、stdout 输出、逐样例比对）。
3. 我们在找「能力悬崖」：参考解长到多少行，模型整体崩掉。
4. 所以需要的数据集必须同时具备：
   a. 题目描述（requirement）
   b. 测试样例（input/output，越多越好 —— 后续要「按比例给分」）
   c. 参考解（reference，用于数行数分桶 + 自检）
5. 红线：这些是「探路/校准」数据；最终评测题与 SFT 训练集不得与它们重叠。

════════════════════════════════════════════════════════════════
【本脚本补哪 4 个数据集、为什么】
════════════════════════════════════════════════════════════════
1. BAAI/TACO（竞赛 I/O，元数据最全）
   - 之前失败原因：HF 的 TACO 脚本(tacopy)内部去 Google Drive 下载几十 G 压缩包；
     HF 后台转 parquet 时被 Google 拦截，所以 HF 上【没有】TACO 的 parquet。
   - 本脚本改走 ModelScope（BAAI 自家平台，数据在 OSS，不走 Google Drive）。

2. codeparrot/apps（APPS，竞赛 I/O，三档难度）
   - 之前失败原因：patch.py 直连的 parquet 是「精简版」，只有
     problem_id/question/solutions，【缺 input_output(测试样例) 和 difficulty】。
   - 本脚本重下带全字段的版本，并验证 input_output / difficulty 确实存在。

3. DS-1000（工程向：数据科学，pandas/numpy/sklearn/matplotlib/tensorflow）
   - 之前没下。函数级 + 真实测试样例，是「工程味」最浓、且唯一没抽样的数据集。
   - 从 GitHub 下 JSONL（不是标准 HF 数据集）。

4. LongCodeArena（可选，探索用）
   - 参考解 100~1000 行的「真·长代码」benchmark，但非 I/O、偏长上下文。
   - 只 clone 下来看一眼目录结构，判断要不要深入；不适合直接当评测题。

════════════════════════════════════════════════════════════════
【运行方式】
════════════════════════════════════════════════════════════════
在 AutoDL 的 eval/harness 目录下：
    python download_v2.py

依赖（缺啥装啥；【不要】加 --upgrade，别动已装好的 torch/CUDA）：
    pip install datasets modelscope
"""
import os
import json
import urllib.request
import subprocess

BASE_DIR = "./sampled_datasets"
SAMPLE_SIZE = 20
# 国内访问 HF 的镜像（对 HF 相关下载生效；ModelScope / GitHub 不走这个）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def uniform_indices(total, n):
    """在 [0, total) 上取 n 个尽量均匀的索引。"""
    if total <= n:
        return list(range(total))
    return [int(i * (total - 1) / (n - 1)) for i in range(n)]


def save_jsonl(items, rel_path):
    """把 list[dict] 写成 jsonl，每行一个 JSON，不截断。"""
    path = os.path.join(BASE_DIR, rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"  ✅ 已写入 {path}（{len(items)} 条）")
    return path


def check_fields(items, required, label):
    """验证关键字段是否齐全；缺了就明确报出来，别默默失败。"""
    if not items:
        print(f"  ❌ [{label}] 没抽到任何数据")
        return
    keys = set(items[0].keys())
    missing = [k for k in required if k not in keys]
    if missing:
        print(f"  ❌ [{label}] 缺关键字段: {missing}（实际字段: {sorted(keys)}）")
    else:
        print(f"  ✅ [{label}] 关键字段齐全: {required}")


# ────────────────────────────────────────────────────────────────
# 1. TACO —— 走 ModelScope（HF 没 parquet，脚本走 Google Drive 被墙）
# ────────────────────────────────────────────────────────────────
def download_taco():
    print("\n" + "=" * 60)
    print("[1/4] BAAI/TACO —— 走 ModelScope（HF 无 parquet）")
    print("=" * 60)
    try:
        from modelscope.msdatasets import MsDataset
        ds = MsDataset.load("BAAI/TACO", split="train")
        # MsDataset 不同版本行为不同：可能直接可迭代，也可能要 .to_hf_dataset()
        try:
            all_items = [dict(x) for x in ds]
        except Exception:
            all_items = [dict(x) for x in ds.to_hf_dataset()]
        idx = uniform_indices(len(all_items), SAMPLE_SIZE)
        sampled = [all_items[i] for i in idx]
        save_jsonl(sampled, "BAAI/TACO/train_sampled.jsonl")
        check_fields(sampled, ["question", "solutions", "input_output", "difficulty"], "TACO")
    except Exception as e1:
        print(f"  ❌ ModelScope 方式失败: {type(e1).__name__}: {e1}")
        if isinstance(e1, ModuleNotFoundError) and e1.name:
            print(f"    → 缺依赖，先补：pip install {e1.name}")
        print("    备选 1：命令行下载到本地目录，再手工看结构：")
        print('      modelscope download --model BAAI/TACO --type dataset --local_dir ./TACO')
        print("    备选 2：若你有 Google Drive 通道，可尝试 HF 脚本(会走 Google Drive)：")
        print('      load_dataset("BAAI/TACO", split="train", trust_remote_code=True)')


# ────────────────────────────────────────────────────────────────
# 2. APPS —— 重下带 input_output / difficulty 的版本
# ────────────────────────────────────────────────────────────────
def download_apps():
    print("\n" + "=" * 60)
    print("[2/4] codeparrot/apps —— 重下带 input_output 的版本")
    print("=" * 60)
    from datasets import load_dataset
    # 已知事实：
    #   1) 配置名是 all / introductory / interview / competition（不是 community）；
    #   2) apps.py 脚本有 gzip 解码 bug（UnicodeDecodeError），只能走 parquet 直连；
    #   3) patch.py 之前用 all/test 拿到的是精简版（test split 可能故意去掉 input_output/difficulty），
    #      所以这次优先用 train split，并逐个验证 input_output 是否真的在。
    for cfg in ["introductory", "interview", "competition", "all"]:
        for split in ["train", "test"]:
            url = (f"https://hf-mirror.com/datasets/codeparrot/apps/resolve/"
                   f"refs%2Fconvert%2Fparquet/{cfg}/{split}/0000.parquet")
            try:
                ds = load_dataset("parquet", data_files=url)
                items = [dict(x) for x in ds]
                if items and "input_output" in items[0]:
                    idx = uniform_indices(len(items), SAMPLE_SIZE)
                    sampled = [items[i] for i in idx]
                    save_jsonl(sampled, f"codeparrot/apps/{cfg}_{split}_sampled.jsonl")
                    check_fields(sampled, ["question", "solutions", "input_output", "difficulty"],
                                 f"APPS[{cfg}/{split}]")
                    return
                else:
                    got = sorted(items[0].keys()) if items else "无"
                    print(f"  ⚠️  [{cfg}/{split}] 缺 input_output，实际字段: {got}")
            except Exception as e:
                print(f"  ⚠️  [{cfg}/{split}] 失败: {type(e).__name__}: {e}")
    print("  ❌ APPS 重下失败：以上 parquet 分片都没拿到 input_output。")
    print("     去 https://huggingface.co/datasets/codeparrot/apps/tree/main/refs/convert/parquet")
    print("     看实际有哪些 config/split 分片，再改上面的 URL。")


# ────────────────────────────────────────────────────────────────
# 3. DS-1000 —— 从 GitHub 下 JSONL
# ────────────────────────────────────────────────────────────────
def download_ds1000():
    print("\n" + "=" * 60)
    print("[3/4] DS-1000 —— 从 GitHub 下 JSONL（工程向/数据科学）")
    print("=" * 60)
    # raw.githubusercontent.com 直连被墙（Connection reset），改用 git clone（更稳）
    repo_url = "https://github.com/xlang-ai/DS-1000.git"
    clone_dir = os.path.join(BASE_DIR, "xlangai", "DS-1000-repo")
    if not os.path.isdir(os.path.join(clone_dir, "data")):
        print(f"  clone {repo_url} ...")
        try:
            subprocess.run(["git", "clone", "--depth", "1", repo_url, clone_dir], check=True)
        except Exception as e:
            print(f"  ❌ clone 失败: {type(e).__name__}: {e}")
            print("    备选：用 ghproxy 代理下 raw，或去 https://github.com/xlang-ai/DS-1000 手动下 data/ 目录。")
            return
    else:
        print(f"  已存在 {clone_dir}，跳过 clone")

    data_src = os.path.join(clone_dir, "data", "ds1000_data.jsonl")
    test_src = os.path.join(clone_dir, "data", "ds1000_test_code.py")
    if not os.path.isfile(data_src):
        print(f"  ❌ 找不到 {data_src}，请确认 repo 里 data/ 的文件名。")
        return

    out_dir = os.path.join(BASE_DIR, "xlangai", "DS-1000")
    os.makedirs(out_dir, exist_ok=True)
    import shutil
    shutil.copy(data_src, os.path.join(out_dir, "ds1000_data.jsonl"))
    shutil.copy(test_src, os.path.join(out_dir, "ds1000_test_code.py"))
    print(f"  ✅ 已复制 data 到 {out_dir}")

    with open(data_src, encoding="utf-8") as f:
        lines = [json.loads(ln) for ln in f if ln.strip()]
    idx = uniform_indices(len(lines), SAMPLE_SIZE)
    sampled = [lines[i] for i in idx]
    save_jsonl(sampled, "xlangai/DS-1000/ds1000_sampled.jsonl")
    # 注意：DS-1000 的测试样例不在 jsonl 里，而在同目录的 ds1000_test_code.py（已一并复制）。
    check_fields(sampled, ["prompt", "reference_code", "metadata"], "DS-1000")


# ────────────────────────────────────────────────────────────────
# 4. LongCodeArena —— 可选，clone 下来只看目录结构
# ────────────────────────────────────────────────────────────────
def download_lca():
    print("\n" + "=" * 60)
    print("[4/4] LongCodeArena —— 可选，clone 看一眼（真·长代码）")
    print("=" * 60)
    dst = os.path.join(BASE_DIR, "JetBrains-Research", "LongCodeArena")
    if os.path.exists(dst):
        print(f"  已存在 {dst}，跳过 clone")
        return
    # 该 URL 是 baselines 仓库（含 benchmark 数据或指向）；若 clone 后找不到数据，去
    # https://github.com/JetBrains-Research/lca 或论文仓库确认。
    url = "https://github.com/jetbrains-research/lca-baselines.git"
    try:
        subprocess.run(["git", "clone", "--depth", "1", url, dst], check=True)
        print(f"  ✅ 已 clone 到 {dst}，请自行查看 data/ 目录结构。")
    except Exception as e:
        print(f"  ❌ clone 失败: {type(e).__name__}: {e}")
        print("    手动访问 https://github.com/JetBrains-Research/lca 查看。")


if __name__ == "__main__":
    download_taco()
    download_apps()
    download_ds1000()
    download_lca()
    print("\n全部处理完毕。")
    print("重点核对上面 [TACO] 和 [APPS] 的字段是否齐全；缺了就把报错贴回来，据此改对应函数。")
