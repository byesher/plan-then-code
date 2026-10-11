#!/bin/bash
# 全量评测：baseline_comment + baseline1turn + sft1turn，评测完成后自动关机（AutoDL 用）。
set -e
cd "$(dirname "$0")"

LOG=~/autodl-tmp/full_eval.log

export MODEL_PATH=~/autodl-tmp/Qwen2.5-Coder-14B
export DEV_PATH=~/autodl-tmp/plan-then-code/data/dataset/eval.jsonl
export ADAPTER_PATH=~/autodl-tmp/checkpoints/lora
export OUT_ROOT=~/autodl-tmp/plan-then-code/eval/results/full
# LIMIT 不设置时评测全量 324 题
unset LIMIT

echo "[$(date '+%F %T')] 全量评测开始" | tee "$LOG"
python -u evaluate.py 2>&1 | python -c '
import sys, datetime
for line in sys.stdin:
    print(datetime.datetime.now().strftime("%F %T"), line, end="", flush=True)
' | tee -a "$LOG"
echo "[$(date '+%F %T')] 完成，关机" | tee -a "$LOG"
shutdown -h now
