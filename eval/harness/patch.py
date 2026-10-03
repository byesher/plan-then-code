# -*- coding: utf-8 -*-
"""
TACO & APPS 专属下载补丁脚本
原理：无视作者带有 Bug 的 python 脚本，直连 Hugging Face 后台转换好的 Parquet 原始数据块。
"""
import os
import json
from datasets import load_dataset

# 还是放在你刚才成功下载的同一个军火库文件夹里
BASE_DIR = "./sampled_datasets"
SAMPLE_SIZE = 20

# 强制使用镜像直连 Parquet 文件（100% 规避 utf-8 解码错误）
FIXED_SOURCES = {
    "BAAI/TACO": {
        "split": "train",
        "url": "https://hf-mirror.com/datasets/BAAI/TACO/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    },
    "codeparrot/apps": {
        "split": "test",
        "url": "https://hf-mirror.com/datasets/codeparrot/apps/resolve/refs%2Fconvert%2Fparquet/all/test/0000.parquet"
    }
}

def get_uniform_indices(total_len, num_samples):
    if total_len <= num_samples: return list(range(total_len))
    return [int(i * (total_len - 1) / (num_samples - 1)) for i in range(num_samples)]

if __name__ == "__main__":
    print("🚀 启动针对 TACO 和 APPS 的专属修复补丁...")

    for name, info in FIXED_SOURCES.items():
        print(f"\n🔄 正在通过 Parquet 协议直连抽取: {name} ...")
        try:
            # 核心魔法：使用 "parquet" 解析器，直接给它后端数据块的 HTTP 链接！
            dataset = load_dataset(
                "parquet", 
                data_files={info["split"]: info["url"]}, 
                split=info["split"]
            )
            
            total_len = len(dataset)
            indices = get_uniform_indices(total_len, SAMPLE_SIZE)
            sampled_dataset = dataset.select(indices)
            
            # 保持目录结构一致
            save_dir = os.path.join(BASE_DIR, name)
            os.makedirs(save_dir, exist_ok=True)
            save_file = os.path.join(save_dir, f"{info['split']}_sampled.jsonl")
            
            # 写入 jsonl
            with open(save_file, "w", encoding="utf-8") as f_out:
                for item in sampled_dataset:
                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    
            print(f"✅ 成功！[{name}] 从后端数据块发现 {total_len} 条数据，抽取的 20 条已完美落地至 {save_file}")
            
        except Exception as e:
            print(f"❌ 失败！[{name}] 报错: {type(e).__name__}: {e}")
            
    print("\n🎉 补丁执行完毕！现在你的 7 大权威数据集已经【全家团聚】，赶紧拿去给规划 AI 分析吧！")