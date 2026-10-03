# -*- coding: utf-8 -*-
"""
代码评测数据集 均匀抽样与本地化下载脚本
目标：为宏观规划 AI 提供原汁原味、无截断、均匀分布的 JSONL 数据集。
"""
import os
import json

# 【强制网络代理】确保 AutoDL 环境下连接 HuggingFace 稳定
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

# 所有的 7 大核心权威数据集（一个都不少）
DATASET_NAMES = [
    "BAAI/TACO",
    "codeparrot/apps",
    "deepmind/code_contests",
    "FudanSELab/ClassEval",
    "google-research-datasets/mbpp",
    "openai/openai_humaneval",
    "bigcode/bigcodebench"
]

SAMPLE_SIZE = 20
BASE_DIR = "./sampled_datasets"
REPORT_FILE = "dataset_download_report.md"

def get_uniform_indices(total_len, num_samples):
    """
    数学算法：充分均匀抽样
    如果数据集总量小于需要抽样的数量，则全量返回；
    否则，生成等间距的索引贯穿整个数据集。
    """
    if total_len <= num_samples:
        return list(range(total_len))
    return [int(i * (total_len - 1) / (num_samples - 1)) for i in range(num_samples)]

def main():
    print("🚀 开始执行全量均匀抽样下载...")
    
    # 创建存放真实数据的根目录
    os.makedirs(BASE_DIR, exist_ok=True)
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f_report:
        f_report.write("# 代码大模型评测数据 均匀抽样报告\n\n")
        f_report.write(f"> **抽样策略**: 从每个数据集中充分均匀抽取 **{SAMPLE_SIZE}** 条，不做任何字符串截断，保留原始 `jsonl` 结构。\n\n")
        
        for name in DATASET_NAMES:
            print(f"\n🔄 正在处理: {name} (可能需要几分钟下载全量索引，请耐心等待)...")
            
            try:
                # 必须带 trust_remote_code=True（配合已降级的 datasets 库）加载权威数据集
                dataset_dict = load_dataset(name, trust_remote_code=True)
                
                # 智能寻找主要 Split（优先找 test，没有就找 train，再没有就拿第一个）
                if "test" in dataset_dict:
                    split_name = "test"
                elif "train" in dataset_dict:
                    split_name = "train"
                else:
                    split_name = list(dataset_dict.keys())[0]
                    
                dataset = dataset_dict[split_name]
                total_len = len(dataset)
                
                # 计算均匀抽样的索引
                indices = get_uniform_indices(total_len, SAMPLE_SIZE)
                
                # 抽出这 20 条原汁原味的数据
                sampled_dataset = dataset.select(indices)
                
                # 构造符合原本数据集结构的保存路径
                # 比如： ./sampled_datasets/BAAI/TACO/train_sampled.jsonl
                save_dir = os.path.join(BASE_DIR, name)
                os.makedirs(save_dir, exist_ok=True)
                save_file = os.path.join(save_dir, f"{split_name}_sampled.jsonl")
                
                # 写入 JSONL (一行一个 JSON 对象，绝无长度限制和转义截断)
                with open(save_file, "w", encoding="utf-8") as f_out:
                    for item in sampled_dataset:
                        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                        
                print(f"✅ 成功！[{name}] 原总量 {total_len} 条，抽取的 {len(indices)} 条已落地至 {save_file}")
                
                # 写入 Markdown 报告
                f_report.write(f"## {name}\n")
                f_report.write(f"- **状态**: ✅ 获取成功\n")
                f_report.write(f"- **原始数据总量**: {total_len} 条 (Split: `{split_name}`)\n")
                f_report.write(f"- **均匀抽取数量**: {len(indices)} 条\n")
                f_report.write(f"- **本地保存路径**: `{save_file}`\n\n")
                
            except Exception as e:
                print(f"❌ 失败！[{name}] 报错: {e}")
                # 写入 Markdown 报告
                f_report.write(f"## {name}\n")
                f_report.write(f"- **状态**: ❌ 获取失败\n")
                f_report.write(f"- **报错信息**: `{type(e).__name__}: {e}`\n\n")
                
    print(f"\n🎉 全部完成！")
    print(f"📄 总体状态报告已生成: {REPORT_FILE}")
    print(f"📂 原始 JSONL 数据已按照原结构保存在: {os.path.abspath(BASE_DIR)}")

if __name__ == "__main__":
    main()