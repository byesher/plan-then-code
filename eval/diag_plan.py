# -*- coding: utf-8 -*-
"""
快速诊断：SFT 模型在两步评测里到底输出了什么。

加载 plan-then-code-lora-plan adapter，抽 3 个 dev 题，每题做：
  step1：题目 + 「请写出分步伪代码」 → 打印模型输出的伪代码
  step2：题目 + 【上面那份伪代码】 + 「请实现」 → 打印模型输出的代码
目的是看：① 格式是否混（该出伪代码时夹不夹代码）；② 伪代码对不对；③ 代码对不对。

运行（AutoDL，eval 目录下）：
  python diag_plan.py  > diag_out.txt 2>&1
  # 然后把 diag_out.txt 内容贴给我
"""
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.expanduser("~/autodl-tmp/Qwen2.5-7B-Instruct")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", os.path.expanduser("~/autodl-tmp/plan-then-code-lora-plan"))
DEV_PATH = os.path.join(BASE, "..", "data", "taco_train", "dev_raw.jsonl")

PLAN_PROMPT = "【题目】\n{question}\n\n请写出分步伪代码（算法计划），不要写代码。"
IMPL_PROMPT = "【题目】\n{question}\n\n【伪代码】\n{plan}\n\n请严格照着伪代码写出完整可运行的 Python 程序。"

N_PROBLEMS = 3


def main():
    items = []
    with open(DEV_PATH, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                items.append(json.loads(ln))
    items = items[:N_PROBLEMS]

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    def gen(messages):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        device = next(model.parameters()).device
        inputs = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=2048, do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    for i, it in enumerate(items):
        q = it["question"]
        print("=" * 80)
        print(f"【题 {i+1}】 id={it['id']} 难度={it.get('difficulty')}")
        print(f"题目（前 300 字）：{q[:300]}\n")
        plan = gen([{"role": "user", "content": PLAN_PROMPT.format(question=q)}])
        print(f"--- step1 模型输出的伪代码 ---\n{plan}\n")
        code = gen([{"role": "user", "content": IMPL_PROMPT.format(question=q, plan=plan)}])
        print(f"--- step2 模型输出的代码 ---\n{code}\n")


if __name__ == "__main__":
    main()
