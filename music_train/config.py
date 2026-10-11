"""配置合同：路径相对于 YAML 所在目录解析，避免依赖启动目录。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AudioConfig:
    """固定多个短片段覆盖整曲；CNN 和 Transformer 的规模均可修改。"""
    sample_rate: int = 24000
    segment_seconds: float = 8.0
    segments: int = 3
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    channels: list[int] = field(default_factory=lambda: [32, 64, 128])
    hidden_size: int = 384
    heads: int = 6
    layers: int = 4
    tokens_per_segment: int = 16
    dropout: float = 0.1
    trainable: bool = True
    init_checkpoint: str = ""


@dataclass
class LanguageConfig:
    """tiny 只用于无下载测试，绝不能冒充训练好的 Qwen。"""
    model_path: str = ""
    mode: str = "lora"  # frozen / lora / full
    tiny: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True
    question_tokens: int = 160
    answer_tokens: int = 192
    retrieval_text_tokens: int = 192


@dataclass
class TrainConfig:
    """检索损失设为 0 可做问答消融；梯度累积不增加对比损失的负样本数。"""
    seed: int = 42
    epochs: int = 10
    batch_size: int = 4
    eval_batch_size: int = 4
    accumulation_steps: int = 4
    num_workers: int = 4
    mixed_precision: str = "bf16"
    learning_rate: float = 2e-4
    audio_learning_rate: float = 1e-4
    language_learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    retrieval_weight: float = 0.1
    retrieval_dim: int = 256
    temperature: float = 0.07
    one_question_per_song: bool = True
    log_every_updates: int = 10
    diagnostics_every_updates: int = 1  # 每几次优化器更新测量 CNN/Transformer；0 关闭。
    max_batches_per_epoch: int = 0


@dataclass
class Config:
    """所有可迁移配置；模型和音频均显式使用本地文件。"""
    audio_root: str = ""
    train_manifest: str = ""
    val_manifest: str = ""
    output_dir: str = "outputs/run1"
    audio: AudioConfig = field(default_factory=AudioConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    training: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        """在分配显存前拒绝常见配置错误。"""
        a, l, t = self.audio, self.language, self.training
        if a.heads < 1 or a.hidden_size % a.heads:
            raise ValueError("audio.hidden_size 必须能被 heads 整除")
        if min(a.sample_rate, a.segments, a.tokens_per_segment, a.layers, a.n_fft, a.hop_length, a.n_mels, a.hidden_size) <= 0 or a.segment_seconds <= 0:
            raise ValueError("音频长度、模型层数和前端参数必须为正")
        if not a.channels or min(a.channels) < 1 or int(a.sample_rate * a.segment_seconds) <= a.n_fft:
            raise ValueError("CNN 通道不能为空，音频片段必须长于 n_fft")
        if l.mode not in {"frozen", "lora", "full"} or (not l.tiny and not l.model_path):
            raise ValueError("设置本地 model_path，mode 为 frozen/lora/full")
        if l.tiny and l.mode == "lora":
            raise ValueError("tiny 测试请使用 full 或 frozen")
        if min(l.question_tokens, l.answer_tokens, l.retrieval_text_tokens) < 4:
            raise ValueError("文本 token 预算过小")
        if min(t.epochs, t.batch_size, t.eval_batch_size, t.accumulation_steps, t.log_every_updates, t.retrieval_dim) < 1:
            raise ValueError("训练轮数、批量、累积和日志间隔必须为正")
        if t.num_workers < 0 or t.max_batches_per_epoch < 0 or t.diagnostics_every_updates < 0 or t.temperature <= 0 or t.retrieval_weight < 0:
            raise ValueError("非法线程数、温度或检索权重")
        if t.retrieval_weight > 0 and t.batch_size < 2:
            raise ValueError("检索对比训练至少需要每个进程微批 batch_size >= 2；梯度累积不能替代负样本")
        if min(t.learning_rate, t.audio_learning_rate, t.language_learning_rate, t.grad_clip) <= 0:
            raise ValueError("学习率和梯度裁剪阈值必须为正")
        if t.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision 必须为 no/fp16/bf16")
        if not 0 <= t.warmup_ratio < 1 or not 0 <= a.dropout < 1:
            raise ValueError("warmup_ratio/dropout 超出范围")

    def to_dict(self) -> dict[str, Any]:
        """保存解析后的实际配置，用于断点恢复和部署。"""
        return asdict(self)


def from_dict(raw: dict[str, Any]) -> Config:
    """拒绝未知字段，避免拼写错误被静默忽略。"""
    data: dict[str, Any] = dict(raw)
    for key, cls in (("audio", AudioConfig), ("language", LanguageConfig), ("training", TrainConfig)):
        data[key] = cls(**data.get(key, {}))
    result: Config = Config(**data)
    result.validate()
    return result


def load_config(path: str | Path) -> Config:
    """支持 YAML/JSON；相对路径基于配置文件所在目录。"""
    source: Path = Path(path).resolve()
    raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8"))
    config: Config = from_dict(raw)
    for name in ("audio_root", "train_manifest", "val_manifest", "output_dir"):
        value: str = getattr(config, name)
        if value:
            setattr(config, name, str((source.parent / value).resolve()))
    if config.language.model_path:
        config.language.model_path = str((source.parent / config.language.model_path).resolve())
    if config.audio.init_checkpoint:
        config.audio.init_checkpoint = str((source.parent / config.audio.init_checkpoint).resolve())
    return config
