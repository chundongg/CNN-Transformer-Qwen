"""可迁移模型包与训练状态分开保存；部署仍需要同版本本地 Qwen 基座。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from music_train.config import Config, from_dict
from music_train.model import MusicQAModel


def atomic_torch_save(data: Any, path: Path) -> None:
    """先写临时文件再替换，防止中断留下看似完整的 checkpoint。"""
    temporary: Path = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    os.replace(temporary, path)


def save_bundle(model: MusicQAModel, folder: Path, metadata: dict[str, Any]) -> None:
    """保存全部音频/投影参数及语言模型可训练参数；tiny 保存完整基座。"""
    folder.mkdir(parents=True, exist_ok=True)
    # 从较早的 epoch 恢复时可能重写后续目录，旧标记不能让半写入包被认为完整。
    (folder / "READY").unlink(missing_ok=True)
    trainable: set[str] = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    selected: dict[str, Tensor] = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        if not name.startswith("qwen.") or name in trainable or model.config.language.tiny
    }
    atomic_torch_save(selected, folder / "model.pt")
    (folder / "config.json").write_text(json.dumps(model.config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_bundle(model: MusicQAModel, folder: Path, allow_missing_language: bool = False) -> None:
    """严格核对保存权重的名称与形状；冻结的基座参数允许不在包中。"""
    values: dict[str, Tensor] = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    current: dict[str, Tensor] = model.state_dict()
    unexpected: set[str] = set(values) - set(current)
    required: set[str] = {name for name, parameter in model.named_parameters() if parameter.requires_grad or not name.startswith("qwen.")}
    if allow_missing_language:
        required = {name for name in required if not name.startswith("qwen.")}
    if unexpected or required - set(values):
        raise ValueError(f"checkpoint 与模型结构不兼容；多余 {list(unexpected)[:3]}，缺少 {list(required - set(values))[:3]}")
    model.load_state_dict(values, strict=False)


def load_bundle(folder: str | Path, base_model: str | None = None, device: str = "cpu") -> tuple[MusicQAModel, dict[str, Any]]:
    """推理端重建相同网络，可显式覆盖服务器上的基座模型路径。"""
    root: Path = Path(folder)
    config: Config = from_dict(json.loads((root / "config.json").read_text(encoding="utf-8")))
    if base_model:
        config.language.model_path = str(Path(base_model).resolve())
    config.audio.init_checkpoint = ""  # 完整音频参数已在 bundle 内，不再依赖初始化文件。
    config.language.gradient_checkpointing = False
    model: MusicQAModel = MusicQAModel(config)
    apply_bundle(model, root)
    model.to(device).eval()
    return model, json.loads((root / "metadata.json").read_text(encoding="utf-8"))
