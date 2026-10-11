"""在服务器训练前检查环境、路径、划分和真实音频解码，不加载完整 Qwen。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

from music_train.audio import load_segments
from music_train.config import load_config
from music_train.data import check_disjoint, read_manifest


def main() -> None:
    """遇到缺失依赖直接报告，不自动联网或安装任何资源。"""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--audio-check", type=int, default=3)
    args: argparse.Namespace = parser.parse_args()
    config = load_config(args.config)
    dependencies: list[str] = ["torch", "transformers", "accelerate", "numpy", "yaml", "soundfile", "scipy", "matplotlib"]
    if config.language.mode == "lora":
        dependencies.append("peft")
    missing: list[str] = [name for name in dependencies if importlib.util.find_spec(name) is None]
    if missing and not config.language.tiny:
        raise SystemExit("服务器缺少依赖：" + ", ".join(missing))
    if not config.language.tiny and not (Path(config.language.model_path) / "config.json").exists():
        raise SystemExit("本地 Qwen 路径无效")
    train = read_manifest(config.train_manifest, config.audio_root)
    val = read_manifest(config.val_manifest, config.audio_root)
    check_disjoint(train, val)
    unique_audio: list[dict[str, str]] = list({row["audio"]: row for row in train}.values())
    checked: list[dict[str, str]] = unique_audio[:max(args.audio_check, 0)]
    for row in checked:
        samples, mask = load_segments(Path(config.audio_root) / row["audio"], config.audio)
        if not samples.isfinite().all() or not mask.any():
            raise RuntimeError("音频预检失败")
    print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False, "missing_dependencies": missing, "train_songs": len({r['song_id'] for r in train}), "val_songs": len({r['song_id'] for r in val}), "audio_checked": len(checked), "tiny_test_only": config.language.tiny}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
