#!/usr/bin/env bash
# 训练和网页分别启动；默认 CPU 检查模型，监控阶段不占用模型显存。
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
"$PROJECT_DIR/.venv/bin/python" -m music_train.dashboard "$@"
