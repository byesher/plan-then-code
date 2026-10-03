# plan-then-code — 评测基准（eval）

本目录用于评测模型的**端到端写代码能力**：给定自然语言需求，让模型自由发挥写出完整代码，
然后测 **编译通过率 + 测试通过率**，用于微调前（baseline）vs 微调后的对照。

> **主要口径（重要）**：主指标 = **I/O 题的 pass@1（一次性写对）**；接口题（给定签名+契约的实现题）只作参考。
> 大纲（plan-then-code 的中间结构）是 SFT 注入的**内部手段**，评测端从不要求、也不检查模型是否输出大纲。
> 完整分模式拆解与结论见 [analysis.md](analysis.md)。

## 目录结构

```
eval/
  harness/                # 评测脚本
    run_eval.py           # 主评测：生成 -> 抽代码 -> 编译 -> 跑测试 -> 汇总
    generators.py         # 可插拔生成器（reference / file / local / openai）
    extract.py            # 从模型原始输出里抽 Python 代码
    prompt.py             # 由 problem.json 拼出 prompt
    check_problem.py      # 单题自检（造题时用）
  benchmark/problems/     # 题库，每道题一个目录
    p001_interval/
      problem.json        # 需求 + 接口签名/契约
      tests.py            # pytest 测试
      reference.py        # 参考实现（自检 + 校验测试用）
  results/                # 评测报告输出（report_*.json）
```

## 快速自检（无需模型 / 无需 GPU）

用「参考实现」顶替模型跑一遍，验证 harness + 测试本身没 bug，应得 **100%**：

```bash
cd eval/harness
python run_eval.py --problems ../benchmark/problems --generator reference
```

预期输出：4/4 PASS，`compile_rate = 1.0`，`test_pass_rate = 1.0`。

## 跑 baseline（需要 GPU 或 API）

三种方式，选一种。**正式的前后对比请固定用同一种**（推荐方式 A）。

### 方式 A：本地 transformers（AutoDL，正式 baseline）

```bash
# 在 AutoDL 上，先下载模型（见文末），再：
python run_eval.py --problems ../benchmark/problems \
  --generator local --model /path/to/Qwen2.5-7B-Instruct
```

### 方式 B：OpenAI 兼容 API（DashScope / vLLM）

```bash
python run_eval.py --problems ../benchmark/problems --generator openai \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --model qwen2.5-7b-instruct --api-key <你的KEY>
```

### 方式 C：先在 GPU 上离线生成、再在任意机器评测

生成器已支持 `file` 模式（读 `<solutions_dir>/<题目目录>/solution.py`），
便于「GPU 上生成、本地评测」分离。

## 指标口径

- `compile_rate`：能编译（`import solution` 或脚本能跑）的题数 / 总题数。
- `test_pass_rate`：`--num-samples k` 时为 **pass@k**（k 次里至少一次全过的题数 / 总题数）。
- `pass_at_1_avg`：**pass@1**（主指标），所有样本里测试全过的比例，最稳定。
- **主指标 = I/O 题的 `pass_at_1_avg`**；接口题只作参考（见 [analysis.md](analysis.md)）。
- 微调前后对比必须：同一套题、同一套脚本、同一解码参数（temperature、num-samples）。

## 题目格式（造新题时照抄）

每道题 = 一个目录，含 3 个文件：

1. `problem.json`：
   ```json
   {
     "id": "p001",
     "title": "标题",
     "domain": "data-structure | string-parsing | system-utility | math-geometry",
     "difficulty": "easy | medium | hard",
     "n_functions": 4,
     "requirement": "自然语言需求描述",
     "interface": [
       {"type": "function", "name": "merge", "signature": "intervals", "contract": "..."},
       {"type": "class", "name": "KVStore", "signature": "", "contract": "...",
        "methods": [{"name": "get", "signature": "self, key", "contract": "..."}]}
     ]
   }
   ```
2. `tests.py`：`from solution import ...` 后写 `def test_xxx()`，用 `assert`。
3. `reference.py`：一份能过测试的标准实现（用于自检）。

**I/O 题（主评测题）格式**：`problem.json` 设 `"mode": "io"`，样例直接内嵌在 `tests`，不需要 `interface` 和 `tests.py`：

```json
{
  "id": "p015",
  "title": "INI parser (I/O)",
  "domain": "string-parsing",
  "difficulty": "medium",
  "mode": "io",
  "requirement": "自然语言需求描述",
  "input_format": "输入格式说明（stdin）",
  "output_format": "输出格式说明（stdout）",
  "tests": [
    {"input": "...", "output": "..."}
  ]
}
```

写完后单题自检：

```bash
python check_problem.py ../benchmark/problems/<题目目录>
```

## AutoDL 租卡跑 baseline 的步骤（第 ④ 步）

1. 租一台 **4090（24G）**，选 **PyTorch 2.x + CUDA** 镜像。
2. 装依赖：
   ```bash
   pip install transformers torch accelerate openai pytest
   ```
3. 下载模型（国内建议用 ModelScope 镜像，快）：
   ```bash
   pip install modelscope
   modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./Qwen2.5-7B-Instruct
   ```
4. 把整个 `plan-then-code/` 仓库传到 AutoDL（git clone 或 scp）。
5. 跑 baseline（方式 A，`--model ./Qwen2.5-7B-Instruct`）。
6. 跑完 **立即关机**（不开机不扣费）。

> 原则：**先在本地把 harness + 题目自检通过，再租卡**，避免在计费时段调试代码。
