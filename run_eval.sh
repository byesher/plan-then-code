#!/bin/bash
# 全量评测 FINAL：baseline_fence + baseline_comment + baseline1turn + sft1turn，跑完关机
set -e
cd ~/autodl-tmp/plan-then-code/eval

LOG=~/autodl-tmp/full_eval_final.log

export MODEL_PATH=~/autodl-tmp/Qwen2.5-Coder-14B
export DEV_PATH=~/autodl-tmp/plan-then-code/data/dataset/eval.jsonl
export ADAPTER_PATH=~/autodl-tmp/checkpoints/lora
export OUT_ROOT=~/autodl-tmp/plan-then-code/eval/results/sota_full
# LIMIT 不设 = 全量 324 题
unset LIMIT

echo "[$(date '+%F %T')] 全量评测开始" | tee "$LOG"
python -u evaluate.py 2>&1 | python -c '
import sys, datetime
for line in sys.stdin:
    print(datetime.datetime.now().strftime("%F %T"), line, end="", flush=True)
' | tee -a "$LOG"
echo "[$(date '+%F %T')] 完成，关机" | tee -a "$LOG"
shutdown -h now
