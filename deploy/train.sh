#!/usr/bin/env bash
# 从任意目录启动；需先按 DEPLOYMENT.md 创建 .venv 并安装服务器依赖。
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
CONFIG="${1:-configs/server.yaml}"
if [[ $# -gt 0 ]]; then shift; fi
"$PROJECT_DIR/.venv/bin/python" -m music_train.train --config "$CONFIG" "$@"
