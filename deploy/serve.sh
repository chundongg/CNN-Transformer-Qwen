#!/usr/bin/env bash
# 同一 GPU 使用单 worker；关闭训练后再启动服务，避免重复占用显存。
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
: "${MUSIC_CHECKPOINT:?Set MUSIC_CHECKPOINT to a completed epoch directory}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
"$PROJECT_DIR/.venv/bin/python" -m uvicorn music_train.serve:app --host 127.0.0.1 --port "${PORT:-8800}" --workers 1
