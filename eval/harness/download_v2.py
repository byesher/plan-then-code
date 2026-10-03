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
# 1. TACO —— 直接下 HF repo 里的 arrow 数据文件（不是 Google Drive）
# ────────────────────────────────────────────────────────────────
def download_taco():
    print("\n" + "=" * 60)
    print("[1/4] BAAI/TACO —— 直接下 HF repo 里的 arrow 数据文件")
    print("=" * 60)
    # 关键：看 TACO.py 的 _URLS，引用的是相对路径 train/*.arrow / test/*.arrow，
    # 这些 arrow 数据文件就托管在 HF repo 里，【不是 Google Drive】。
    # 所以直接 huggingface-cli download 整个 repo 就能拿到数据，绕开脚本。
    import glob
    from datasets import Dataset, concatenate_datasets
    # 候选位置：① 你手动下到的 ~/autodl-tmp/TACO；② 脚本默认路径
    candidates = [
        os.path.expanduser("~/autodl-tmp/TACO"),
        os.path.join(BASE_DIR, "BAAI", "TACO-repo"),
    ]
    repo_dir = next((c for c in candidates if glob.glob(os.path.join(c, "train", "*.arrow"))), None)
    if repo_dir is None:
        repo_dir = candidates[1]
        print(f"  本地没找到，下载 repo 到 {repo_dir} ...")
        try:
            subprocess.run(
                ["huggingface-cli", "download", "BAAI/TACO",
                 "--repo-type", "dataset", "--local-dir", repo_dir],
                check=True,
            )
        except Exception as e:
            print(f"  ❌ 下载失败: {type(e).__name__}: {e}")
            print("     手动执行：huggingface-cli download BAAI/TACO --repo-type dataset --local-dir ~/autodl-tmp/TACO")
            return
    else:
        print(f"  使用已存在的 {repo_dir}")

    train_files = sorted(glob.glob(os.path.join(repo_dir, "train", "*.arrow")))
    if not train_files:
        print("  ❌ 没找到 train/*.arrow，说明 repo 里没有数据文件。")
        return
    # 关键：Dataset.from_file 直接读 arrow，不写 HF 缓存（load_dataset 会重写一份，把盘写爆）
    ds = concatenate_datasets([Dataset.from_file(f) for f in train_files])
    items = [dict(x) for x in ds]
    idx = uniform_indices(len(items), SAMPLE_SIZE)
    sampled = [items[i] for i in idx]
    save_jsonl(sampled, "BAAI/TACO/train_sampled.jsonl")
    check_fields(sampled, ["question", "solutions", "input_output", "difficulty"], "TACO")


# ────────────────────────────────────────────────────────────────
# 2. APPS —— 重下带 input_output / difficulty 的版本
# ────────────────────────────────────────────────────────────────
def download_apps():
    print("\n" + "=" * 60)
    print("[2/4] codeparrot/apps —— 重下带 input_output 的版本")
    print("=" * 60)
    from datasets import Dataset
    # 已知事实：
    #   1) 配置名是 all / introductory / interview / competition（不是 community）；
    #   2) apps.py 脚本有 gzip 解码 bug（UnicodeDecodeError），只能走 parquet 直连；
    #   3) patch.py 之前用 all/test 拿到的是精简版（test split 可能故意去掉 input_output/difficulty），
    #      所以这次优先用 train split，并逐个验证 input_output 是否真的在。
    for cfg in ["introductory", "interview", "competition", "all"]:
        for split in ["train", "test"]:
            url = (f"https://hf-mirror.com/datasets/codeparrot/apps/resolve/"
                   f"refs%2Fconvert%2Fparquet/{cfg}/{split}/0000.parquet")
            local = os.path.join("/root/autodl-tmp", f"apps_{cfg}_{split}.parquet")
            try:
                if not os.path.isfile(local):
                    print(f"  下载 {url}")
                    urllib.request.urlretrieve(url, local)
                # 关键：Dataset.from_parquet 直接读，不写 HF 缓存
                ds = Dataset.from_parquet(local)
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
    print("[3/4] DS-1000 —— 读 data/ds1000.jsonl.gz（gzip 压缩的 jsonl）")
    print("=" * 60)
    import gzip
    # 真实数据文件是 data/ds1000.jsonl.gz（gzip 压缩），不是 ds1000_data.jsonl。
    # 测试样例也不在单独文件里，而在每题 prompt 的 docstring 例子里 + code_context 字段。
    clone_dir = os.path.join(BASE_DIR, "xlangai", "DS-1000-repo")
    gz_path = os.path.join(clone_dir, "data", "ds1000.jsonl.gz")
    if not os.path.isfile(gz_path):
        print(f"  找不到 {gz_path}，重新 clone ...")
        try:
            subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/xlang-ai/DS-1000.git", clone_dir],
                           check=True, timeout=300)
        except Exception as e:
            print(f"  ❌ clone 失败: {type(e).__name__}: {e}")
            print("    备选：手动去 https://github.com/xlang-ai/DS-1000 下 data/ds1000.jsonl.gz 放到该目录")
            return
    if not os.path.isfile(gz_path):
        print("  ❌ 仍找不到 data/ds1000.jsonl.gz。")
        return

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        lines = [json.loads(ln) for ln in f if ln.strip()]
    idx = uniform_indices(len(lines), SAMPLE_SIZE)
    sampled = [lines[i] for i in idx]
    save_jsonl(sampled, "xlangai/DS-1000/ds1000_sampled.jsonl")
    # 字段：metadata(problem_id/library/perturbation_type) + prompt + reference_code + code_context
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
