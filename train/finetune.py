# -*- coding: utf-8 -*-
"""单轮 LoRA SFT：题目 →（伪代码 + 代码），纯 transformers + peft"""
import json
import os
import random

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/autodl-tmp/Qwen2.5-Coder-7B"))
DATA_PATH = os.environ.get("DATA_PATH", os.path.join(BASE_DIR, "..", "data", "dataset", "train.jsonl"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.expanduser("~/autodl-tmp/checkpoints/lora"))
USE_4BIT = os.environ.get("USE_4BIT", "0") == "1"
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0")) or None

LORA_R = int(os.environ.get("LORA_R", "8"))
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LR = float(os.environ.get("LR", "1e-4"))
EPOCHS = int(os.environ.get("EPOCHS", "1"))
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 16
MAX_SEQ_LEN = 2048
WARMUP_RATIO = 0.05
LOGGING_STEPS = 10


def load_dataset(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                items.append(json.loads(ln))
    random.seed(42)
    random.shuffle(items)
    if MAX_SAMPLES:
        items = items[:MAX_SAMPLES]
    return Dataset.from_list([{"messages": it["messages"]} for it in items])


def main():
    print(f"模型：{MODEL_PATH}\n数据：{DATA_PATH}\n输出：{OUT_DIR}\n")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    ))
    model.config.use_cache = False
    model.enable_input_require_grads()

    ds = load_dataset(DATA_PATH)

    def tokenize(examples):
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in examples["messages"]
        ]
        return tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN, padding=False)

    ds = ds.map(tokenize, batched=True, remove_columns=["messages"])
    print(f"训练集 {len(ds)} 条，开始训练 ...\n")

    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=LOGGING_STEPS,
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=42,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"\n✅ 训练完成，adapter 已存到：{OUT_DIR}")


if __name__ == "__main__":
    main()