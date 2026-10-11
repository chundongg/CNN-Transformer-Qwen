"""同一推理实现供命令行和服务器复用；不把未经对齐的向量伪装成可检索表示。"""
from __future__ import annotations

import contextlib
import threading
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from music_train.audio import load_segments
from music_train.checkpoint import load_bundle


class Predictor:
    """一个进程加载一个模型，锁避免多个并发请求同时占用推理显存。"""
    def __init__(self, checkpoint: str, base_model: str | None = None, device: str = "auto") -> None:
        """checkpoint 必须来自已完成的训练；不会下载缺失权重。"""
        self.device: str = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.model, self.metadata = load_bundle(checkpoint, base_model, self.device)
        self.lock: threading.RLock = threading.RLock()

    def autocast(self) -> Any:
        """与训练精度一致；CPU 路径保持模块已加载的权重精度。"""
        precision: str = self.model.config.training.mixed_precision
        if self.device.startswith("cuda") and precision != "no":
            return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)
        return contextlib.nullcontext()

    def describe(self, audio_path: str | Path, question: str, max_new_tokens: int = 192) -> str:
        """生成音频条件回答；模型不足以支持的内容仍可能出现幻觉，需评测。"""
        with self.lock, torch.inference_mode(), self.autocast():
            waves, mask = load_segments(Path(audio_path), self.model.config.audio)
            return self.model.describe(waves.unsqueeze(0).to(self.device), mask.unsqueeze(0).to(self.device), question, max_new_tokens)

    def require_retrieval(self) -> None:
        """只有实际启用过检索损失的检查点才开放向量召回。"""
        if not self.metadata.get("retrieval_trained", False):
            raise ValueError("此检查点未训练检索对齐；请使用 retrieval_weight > 0 训练的模型")

    def audio_vector(self, audio_path: str | Path) -> Tensor:
        """离线歌曲向量不依赖用户问题，可预计算后存入索引。"""
        self.require_retrieval()
        with self.lock, torch.inference_mode(), self.autocast():
            waves, mask = load_segments(Path(audio_path), self.model.config.audio)
            hidden, valid = self.model.audio_encoder(waves.unsqueeze(0).to(self.device), mask.unsqueeze(0).to(self.device))
            return self.model.audio_vectors(hidden, valid)[0].float().cpu()

    def text_vectors(self, texts: list[str]) -> Tensor:
        """文本侧不接触音频，训练/检索使用同一个编码模板。"""
        self.require_retrieval()
        with self.lock, torch.inference_mode(), self.autocast():
            return self.model.encode_text(texts).float().cpu()
