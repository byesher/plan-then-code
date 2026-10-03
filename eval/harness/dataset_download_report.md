# 代码大模型评测数据 均匀抽样报告

> **抽样策略**: 从每个数据集中充分均匀抽取 **20** 条，不做任何字符串截断，保留原始 `jsonl` 结构。

## BAAI/TACO
- **状态**: ❌ 获取失败
- **报错信息**: `UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b in position 1: invalid start byte`

## codeparrot/apps
- **状态**: ❌ 获取失败
- **报错信息**: `UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b in position 1: invalid start byte`

## deepmind/code_contests
- **状态**: ✅ 获取成功
- **原始数据总量**: 165 条 (Split: `test`)
- **均匀抽取数量**: 20 条
- **本地保存路径**: `./sampled_datasets/deepmind/code_contests/test_sampled.jsonl`

## FudanSELab/ClassEval
- **状态**: ✅ 获取成功
- **原始数据总量**: 100 条 (Split: `test`)
- **均匀抽取数量**: 20 条
- **本地保存路径**: `./sampled_datasets/FudanSELab/ClassEval/test_sampled.jsonl`

## google-research-datasets/mbpp
- **状态**: ✅ 获取成功
- **原始数据总量**: 500 条 (Split: `test`)
- **均匀抽取数量**: 20 条
- **本地保存路径**: `./sampled_datasets/google-research-datasets/mbpp/test_sampled.jsonl`

## openai/openai_humaneval
- **状态**: ✅ 获取成功
- **原始数据总量**: 164 条 (Split: `test`)
- **均匀抽取数量**: 20 条
- **本地保存路径**: `./sampled_datasets/openai/openai_humaneval/test_sampled.jsonl`

## bigcode/bigcodebench
- **状态**: ✅ 获取成功
- **原始数据总量**: 1140 条 (Split: `v0.1.0_hf`)
- **均匀抽取数量**: 20 条
- **本地保存路径**: `./sampled_datasets/bigcode/bigcodebench/v0.1.0_hf_sampled.jsonl`

