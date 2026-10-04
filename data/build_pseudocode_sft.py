# -*- coding: utf-8 -*-
"""
训练集加工脚本：强 LLM 反推「伪代码」，把 (需求, 参考解) 加工成 SFT 三元组

【这一步在干嘛（项目的核心亮点 = 数据加工）】
plan-then-code 要教 Qwen2.5-7B「先写伪代码，再写代码」。训练样本的格式是：
    输入  = 题目需求 + 「请先写伪代码，再写代码」
    输出  = 伪代码（步骤计划） + 完整代码

现成的竞赛题只有 (需求, 参考解) 二元组，缺「伪代码」这一环。本脚本用【强 LLM】
给「参考解」反推出一份分步伪代码，拼成 (需求 → 伪代码 → 代码) 三元组：
    - 代码 = 参考解（已自检通过，保证是对的）
    - 伪代码 = 强 LLM 读 (题目 + 参考解) 后写出的步骤计划（保证与代码一致）
这正是 CodePlan(ICLR 2025) 同款做法（"prompting an LLM to obtain the annotation"）。

【为什么代码必须是参考解、伪代码由 LLM 反推】
- 参考解已经过 extract_train_set.py 的「跑样例自检」，是【正确的】，训练不会教坏模型。
- 伪代码由 LLM 对着「题目 + 正确代码」反推，天然与代码对齐，不会出现「计划对、代码错」或反之。

【数据格式（关键：代码必须可被评测抽取）】
assistant 输出里，代码包在 ```python ... ``` 围栏里，伪代码是编号列表。这样评测时
可以直接正则抽 ```python``` 块去跑，不会把伪代码当代码执行。

【输出】
  data/taco_train/train_sft.jsonl    —— SFT 训练集（messages 格式，可直接喂 LLaMA-Factory / trl）
  data/taco_train/_sft_sample.json   —— 前 2 条样例（美化，供人工审一眼）
  data/taco_train/_sft_manifest.json —— 加工统计（成功/失败条数、耗时）

【运行】
  export DASHSCOPE_API_KEY=sk-xxx    # 阿里云百炼 DashScope 的 API Key（Qwen-Max）
  python data/build_pseudocode_sft.py

  换模型/供应商只改底部「强 LLM API」配置块（OpenAI 兼容，DeepSeek/GPT 也能用）。
依赖：requests（pip install requests）
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ── 路径 ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "taco_train", "train_raw.jsonl")          # extract_train_set.py 的产物
OUT_PATH = os.path.join(BASE_DIR, "taco_train", "train_sft.jsonl")
SAMPLE_PATH = os.path.join(BASE_DIR, "taco_train", "_sft_sample.json")
MANIFEST_PATH = os.path.join(BASE_DIR, "taco_train", "_sft_manifest.json")

# ── 强 LLM API（默认通义千问 Qwen-Max，阿里云百炼 DashScope 的 OpenAI 兼容模式）──
# 换模型/供应商只改这几个：DASHSCOPE_API_KEY / DASHSCOPE_API_BASE / QWEN_MODEL
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
API_BASE = os.environ.get("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.environ.get("QWEN_MODEL", "qwen-max")
API_URL = f"{API_BASE.rstrip('/')}/chat/completions"
# Qwen 的思考(reasoning)模式：关掉省 token、更快，且 content 直接是干净伪代码。
# 若 DashScope 报「未知参数 enable_thinking」，把下面改成 None（不发送该字段）即可。
ENABLE_THINKING = False

# ── 加工参数 ──────────────────────────────────────────────────
PSEUDO_LANG = "中文"      # 伪代码语言（可改英文；这是后续可消融的设计变量之一）
CONCURRENCY = 8           # 并发数（300 条 × 8 并发，几分钟搞定）
MAX_RETRIES = 3           # 单条失败重试次数
LLM_TIMEOUT = 120         # 单次请求超时（秒）
LLM_TEMPERATURE = 0.3     # 反推伪代码用低温度（要稳定、可复现，不要创造性）
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0")) or None  # 只加工前 N 条（先测 prompt 质量，设 5）

# ── 模板（改这里即可改 SFT 数据的格式）─────────────────────────
# 1) 给强 LLM 的「反推伪代码」指令
PSEUDO_PROMPT = (
    "你是一名算法助教。下面是编程题和它的一个正确解法。请为这个解法写一份「分步伪代码」——"
    "是【算法计划】，不是代码翻译。一个程序员读完后能自己写出等价代码。\n\n"
    "要求：\n"
    "1. 第一行先一句话点出整体算法思路（这题的核心观察/技巧是什么）。\n"
    "2. 然后用编号列表（1. 2. 3. ...）分步描述，每一步回答「做什么 + 为什么」，5~15 步为宜。\n"
    "3. 覆盖关键数据结构、循环/分支的【意图】、边界情况怎么处理。\n"
    "4. 【禁止】逐行翻译代码、复述具体变量名/数组下标/布尔条件/算术公式。\n"
    "5. 语言用{lang}，每行一条，简洁。\n"
    "6. 只输出伪代码本身，不要输出代码、不要解释、不要多余的话。\n\n"
    "【题目】\n{question}\n\n"
    "【正确解法】\n{code}"
)

# 2) SFT 样本的三个角色
SYSTEM_PROMPT = (
    "你是一个编程助手。收到题目后，先写出分步伪代码（算法计划），再写出完整可运行的 Python 程序。"
    "代码必须放在 ```python ... ``` 代码块内。"
)
USER_TEMPLATE = "【题目】\n{question}\n\n请先写出分步伪代码，再写出完整可运行的 Python 程序。"
ASSISTANT_TEMPLATE = "【伪代码】\n{pseudocode}\n\n【代码】\n```python\n{code}\n```"


# ── LLM 调用（OpenAI 兼容 chat/completions，带重试）────────────
def call_llm(messages):
    if not API_KEY:
        raise RuntimeError("未设置 DASHSCOPE_API_KEY。先 export DASHSCOPE_API_KEY=sk-xxx")
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": 4096,
        "stream": False,
    }
    if ENABLE_THINKING is not None:
        payload["enable_thinking"] = ENABLE_THINKING
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(API_URL, json=payload, headers=headers, timeout=LLM_TIMEOUT)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            if not content:
                raise RuntimeError(f"LLM 返回空 content：{str(r.json())[:300]}")
            return content
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)  # 1, 2, 4 秒退避
    raise RuntimeError(f"LLM 调用失败（重试 {MAX_RETRIES} 次）: {last_err}")


# ── 单条加工 ──────────────────────────────────────────────────
def build_sample(item):
    question = item["question"]
    code = item["solution"]

    # 反推伪代码
    pseudo_prompt = PSEUDO_PROMPT.format(lang=PSEUDO_LANG, question=question, code=code)
    pseudocode = call_llm([{"role": "user", "content": pseudo_prompt}])

    # 拼 SFT 样本（messages 格式，代码可抽取）
    user_content = USER_TEMPLATE.format(question=question)
    assistant_content = ASSISTANT_TEMPLATE.format(pseudocode=pseudocode, code=code)

    return {
        "id": item["id"],
        "name": item.get("name"),
        "difficulty": item.get("difficulty"),
        "question": question,
        "pseudocode": pseudocode,       # 反推出的伪代码（保留，便于后续消融/分析）
        "code": code,                   # 参考解（已自检通过）
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def main():
    if not os.path.isfile(RAW_PATH):
        raise FileNotFoundError(
            f"没找到 {RAW_PATH}\n"
            f"先跑：python eval/harness/extract_train_set.py（在 AutoDL 上，TACO 数据在那）"
        )

    raw = []
    with open(RAW_PATH, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                raw.append(json.loads(ln))
    if MAX_SAMPLES:
        raw = raw[:MAX_SAMPLES]
    print(f"读入 {len(raw)} 条原始训练原料：{RAW_PATH}\n")

    t0 = time.time()
    ok, fail = 0, 0
    results = [None] * len(raw)

    # 并发加工，结果按原顺序放回
    def work(i_item):
        i, item = i_item
        try:
            return i, build_sample(item)
        except Exception as e:
            return i, e

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(work, (i, it)) for i, it in enumerate(raw)]
        for fut in as_completed(futs):
            i, res = fut.result()
            if isinstance(res, Exception):
                fail += 1
                print(f"  ❌ [{i}] {raw[i]['id']} 失败: {res}")
            else:
                ok += 1
                results[i] = res
            if (ok + fail) % 20 == 0:
                print(f"  进度 {ok + fail}/{len(raw)}（成功 {ok} / 失败 {fail}）")

    results = [r for r in results if r is not None]
    elapsed = time.time() - t0

    # 写 SFT 训练集
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 写前 2 条美化样例，供人工审一眼
    with open(SAMPLE_PATH, "w", encoding="utf-8") as f:
        json.dump(results[:2], f, ensure_ascii=False, indent=2)

    # 写统计
    manifest = {
        "raw_n": len(raw),
        "success": ok,
        "fail": fail,
        "model": MODEL,
        "pseudo_lang": PSEUDO_LANG,
        "elapsed_sec": round(elapsed, 1),
        "out": OUT_PATH,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 加工完成：成功 {ok}，失败 {fail}，耗时 {elapsed:.1f}s")
    print(f"   SFT 训练集：{OUT_PATH}")
    print(f"   样例预览：{SAMPLE_PATH}（先打开审一眼伪代码质量）")
    print(f"   统计：{MANIFEST_PATH}")
    if fail:
        print(f"   ⚠️ 有 {fail} 条失败，检查上面的报错；失败的不会进训练集。")
    print("\n下一步：写 LoRA SFT 脚本，用 train_sft.jsonl 微调 Qwen2.5-7B（先出伪代码再出代码）。")


if __name__ == "__main__":
    main()
