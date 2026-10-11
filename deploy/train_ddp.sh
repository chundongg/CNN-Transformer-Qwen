#!/usr/bin/env bash
# 每个进程各一份模型，普通 DDP 不会把一个模型切到多张显卡上。
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
GPU_COUNT="${GPU_COUNT:-2}"
CONFIG="${1:-configs/server.yaml}"
if [[ $# -gt 0 ]]; then shift; fi
# 精度从同一个配置读取，避免 YAML 改为 fp16 后启动器仍强制 bf16。
PRECISION="$("$PROJECT_DIR/.venv/bin/python" -c 'import sys; from music_train.config import load_config; print(load_config(sys.argv[1]).training.mixed_precision)' "$CONFIG")"
"$PROJECT_DIR/.venv/bin/accelerate" launch --multi_gpu --num_processes "$GPU_COUNT" --num_machines 1 --mixed_precision "$PRECISION" --dynamo_backend no -m music_train.train --config "$CONFIG" "$@"
