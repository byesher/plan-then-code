# plan-then-code

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

用 LoRA 指令微调 Qwen2.5-Coder-14B，让它「先规划、再写代码」。在 324 道算法题上，端到端通过率（pass@1）从 **36.7% 提升到 45.7%（+9.0pp，McNemar p < 0.001）**。

## 简介

代码生成模型习惯「一步直接写代码」，遇到需要仔细处理输入输出、分步推理的题目时容易结构混乱、漏边界。本项目用「I/O 契约 + 伪代码 + 代码」的结构化数据微调 Qwen2.5-Coder-14B，让它内化「先规划、再实现」。

## 结果

### 三项指标（324 道 held-out 算法题）

| 设置 | pass@1 | compile | case_pass |
|---|---|---|---|
| 裸 Coder，chat 指令 | 6.2%（20/324） | 26.2%（85/324） | 6.6%（256/3894） |
| 裸 Coder，注释补全 | 36.7%（119/324） | 99.4%（322/324） | 34.1%（1327/3894） |
| **SFT（plan-then-code）** | **45.7%（148/324）** | **100%（324/324）** | **36.7%（1429/3894）** |

相对「注释补全」基线，SFT 的 pass@1 提升 **+9.0pp**（McNemar 检验 p < 0.001）。

### 失败分布（324 题，按不通过原因逐题统计）

| 设置 | 通过 | 编译失败 | 输出错(mismatch) | 运行时错(runtime) | 超时(timeout) |
|---|---|---|---|---|---|
| 裸 Coder，chat 指令 | 20 | 239 | 41 | 22 | 2 |
| 裸 Coder，注释补全 | 119 | 2 | 172 | 26 | 5 |
| **SFT（plan-then-code）** | **148** | **0** | 167 | 8 | 1 |

## 数据

从 TACO 竞赛算法题中筛选可用题目（去除交互题、空题、Python2 代码、无测试用例的题），用强 LLM 生成「规划 + 代码」，只有编译通过 + 跑通原题测试用例的样本才保留（闭环校验），最终得到 2372 条训练样本。

## 数据格式

训练样本是一条「题目 → I/O 契约 + 伪代码 + 代码」，采用 chat 格式的 user / assistant 两条消息：

**Instruction（user）**：

```
【题目】
{题目}

请先写出分步伪代码（算法计划），再写出完整可运行的 Python 程序。
```

**Response（assistant）**——首行是 I/O 契约，再分 3~8 步写算法，最后跟代码：

```
【输入怎么读 + 输出什么格式】读入 T 个测试用例；每组读 N、X 和 N 个 A_i；每组输出一行 YES 或 NO。
思路：判断是否存在某个 A_i ≥ X。

1. 读 T。
2. 每组读 N、X。
3. 读入 N 个 A_i。
4. 判断是否存在 A_i ≥ X。
5. 输出 YES 或 NO。

```python
import sys
...
```
```

**评测样本（eval.jsonl）**——只含题目和测试用例，三种评测路线共用同一份：

```json
{"id": "eval_00000", "question": "{题目}", "tests": [{"input": "...", "output": "..."}], "difficulty": "..."}
```

## 训练

- 底座模型：Qwen2.5-Coder-14B（base 版）。
- LoRA：r=8，alpha=16，lr=1e-4，1 epoch，bf16。
- 单轮生成：一道题一次调用，规划 + 代码一起出。

## 评测

端到端：只给题目，看最终代码能否「编译 + 跑通测试」，规划不进评测。三个指标：**pass@1**（完全通过率）、**compile**（编译通过率）、**case_pass**（样例通过率）。

三种评测路线，共用同一份 eval.jsonl（题目 + 测试用例），区别在「模型」和「如何把题目喂给模型」：

| 路线 | 模型 | 输入形式 |
|---|---|---|
| chat 指令 | 裸 Coder（base） | `【题目】` + 「请先写伪代码再写代码」指令，走 chat 模板 |
| 注释补全 | 裸 Coder（base） | 题目放进 `# 注释`，直接续写代码 |
| **SFT（plan-then-code）** | 微调后的 Coder | `【题目】` + 「请先写伪代码再写代码」指令，走 chat 模板 |

主 baseline 是「注释补全」——把题目放进 `# 注释` 里让裸 Coder 直接续写，这是 Coder 最自然的代码补全用法。

还试过其他补全形式（如代码围栏补全等），稳定性不如「注释补全」，已退役；实现保留在 `eval/evaluate.py` 的注释里备查。

## 推理

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "Qwen/Qwen2.5-Coder-14B"
adapter = "path/to/checkpoints/lora"

tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter)

messages = [{"role": "user", "content": "【题目】\n{题目}\n\n请先写出分步伪代码（算法计划），再写出完整可运行的 Python 程序。"}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=2048)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## 复现

```bash
# 0. 安装依赖
pip install -r requirements.txt

# 1. 下载 TACO
huggingface-cli download BAAI/TACO --repo-type dataset --local-dir ~/autodl-tmp/TACO

# 2. 造数据（需先设置 LLM API 环境变量）
export LLM_API_KEY=... LLM_API_BASE=... LLM_MODEL=...
python data/build_data.py

# 3. 训练
MODEL_PATH=~/autodl-tmp/Qwen2.5-Coder-14B \
DATA_PATH=data/dataset/train.jsonl \
OUT_DIR=~/autodl-tmp/checkpoints/lora \
python train/finetune.py

# 4. 评测
MODEL_PATH=~/autodl-tmp/Qwen2.5-Coder-14B \
DEV_PATH=data/dataset/eval.jsonl \
ADAPTER_PATH=~/autodl-tmp/checkpoints/lora \
python eval/evaluate.py
```

## 目录结构

```
requirements.txt   # 依赖
data/              # 数据构造（build_data.py）
train/             # LoRA SFT（finetune.py）
eval/              # 评测（evaluate.py）
```

## 许可证

Apache-2.0
