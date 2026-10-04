# -*- coding: utf-8 -*-
"""
LoRA SFT：训 Qwen2.5-7B「先写伪代码，再写代码」

【这一步在干嘛】
读 build_pseudocode_sft.py 产出的 train_sft.jsonl（messages 格式：system/user/assistant），
用 LoRA 微调 Qwen2.5-7B-Instruct，让它学会：看到题目 → 先出伪代码 → 再出代码。

【训练数据格式】
每条样本 messages = [
  {role: system, content: 系统提示},
  {role: user,   content: 题目 + 「请先写伪代码，再写代码」},
  {role: assistant, content: 伪代码 + ```python 代码```},
]
用 tokenizer.apply_chat_template 转成 Qwen2.5 的对话文本后训练。

【关键配置】
- bf16 LoRA（A100）；4090 上把 USE_4BIT 设 True 走 QLoRA（4bit nf4）。
- LoRA target_modules = 全部 7 个投影层（q/k/v/o/gate/up/down）。
- 学习率 2e-4（LoRA 常用区间 1e-4~2e-4）、cosine、warmup 5%、3 epoch。

【运行】
  cd plan-then-code
  export MODEL_PATH=~/autodl-tmp/Qwen2.5-7B-Instruct   # 默认就是这个
  python train/train_lora.py

依赖：torch transformers peft trl datasets（AutoDL 上一般都有，缺啥 pip install）

【先做冒烟测试】
  MAX_SAMPLES=50 python train/train_lora.py   # 只训 50 条，先确认 pipeline 能跑通、能保存 adapter
再跑全量（300 条）。
"""
import json
import os

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# ── 路径与参数 ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/autodl-tmp/Qwen2.5-7B-Instruct"))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "taco_train", "train_sft.jsonl")
OUT_DIR = os.environ.get("OUT_DIR", os.path.expanduser("~/autodl-tmp/plan-then-code-lora"))

USE_4BIT = os.environ.get("USE_4BIT", "0") == "1"   # 4090 上设 1（QLoRA）；A100 上 0（bf16 LoRA）
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0")) or None  # 冒烟测试：设 50 只训 50 条

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

LR = 2e-4
EPOCHS = 3
PER_DEVICE_BATCH = 4
GRAD_ACCUM = 4
MAX_SEQ_LEN = 2048
WARMUP_RATIO = 0.05
LOGGING_STEPS = 10
SAVE_STRATEGY = "epoch"


def load_dataset(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"没找到 {path}\n"
            f"先跑：python data/build_pseudocode_sft.py（要先把 train_raw.jsonl 加工成 train_sft.jsonl）"
        )
    items = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                items.append(json.loads(ln))
    if MAX_SAMPLES:
        items = items[:MAX_SAMPLES]
    # 只保留 messages 列（SFTTrainer 只用它）
    return Dataset.from_list([{"messages": it["messages"]} for it in items])


def main():
    print(f"模型：{MODEL_PATH}")
    print(f"数据：{DATA_PATH}（{MAX_SAMPLES or '全部'} 条）")
    print(f"输出：{OUT_DIR}  4bit={USE_4BIT}\n")

    # ── tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Qwen2.5 无 pad_token，补上避免训练报错

    def formatting_func(examples):
        """messages → Qwen2.5 对话文本。add_generation_prompt=False：训练要学完整 assistant 回话。"""
        return [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in examples["messages"]
        ]

    # ── 模型 ──
    bnb = None
    if USE_4BIT:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # 训练时关 KV cache（配 gradient checkpointing）

    # ── LoRA ──
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── 训练参数 ──
    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        bf16=True,                      # A100 / 4090 都支持 bf16
        logging_steps=LOGGING_STEPS,
        save_strategy=SAVE_STRATEGY,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",               # 不接 wandb
        seed=42,
        optim="adamw_torch",
    )

    ds = load_dataset(DATA_PATH)
    print(f"训练集 {len(ds)} 条，开始训练 ...\n")

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        max_seq_length=MAX_SEQ_LEN,
        peft_config=lora_config,
    )
    trainer.train()

    # ── 保存 adapter + tokenizer + 训练配置（供复现）──
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    with open(os.path.join(OUT_DIR, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": MODEL_PATH,
                "data_path": DATA_PATH,
                "n_samples": len(ds),
                "use_4bit": USE_4BIT,
                "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                         "target_modules": TARGET_MODULES},
                "lr": LR, "epochs": EPOCHS, "max_seq_len": MAX_SEQ_LEN,
                "batch": PER_DEVICE_BATCH, "grad_accum": GRAD_ACCUM,
            },
            f, ensure_ascii=False, indent=2,
        )

    print(f"\n✅ 训练完成，adapter 已存到：{OUT_DIR}")
    print("下一步：写评测脚本，对比 baseline vs SFT 的 pass@1（先在 dev_raw.jsonl 上快评，再上 70 题测试集）。")


if __name__ == "__main__":
    main()
