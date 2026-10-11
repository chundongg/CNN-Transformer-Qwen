"""逐模块训练诊断：记录真实梯度和优化器前后参数差，而非估算更新量。"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


def finite_number(value: float) -> float | None:
    """JSON 不写 NaN/Infinity；异常数值用 null，另保留 finite 状态。"""
    return value if math.isfinite(value) else None


def vector_norm(values: list[Tensor]) -> float:
    """各参数视为一个大向量求 L2 范数，不拼接成额外的大张量。"""
    return math.sqrt(sum(float(value.detach().double().square().sum().item()) for value in values))


class ModuleDiagnostics:
    """只快照 CNN 和音频 Transformer，避免复制整个 Qwen 的数十亿参数。

    DDP 下在主进程、梯度同步和 AMP unscale 完成后调用。普通 DDP 每卡
    参数与同步梯度相同，因此不再把各 rank 的相同范数相加。
    """

    def __init__(self, modules: dict[str, nn.Module]) -> None:
        """保存参数引用；不改变模型结构，不将诊断数据加入计算图。"""
        self.modules: dict[str, list[nn.Parameter]] = {
            name: list(module.parameters()) for name, module in modules.items()
        }
        self.before: dict[str, list[Tensor]] = {}
        self.metrics: dict[str, dict[str, Any]] = {}

    @torch.no_grad()
    def before_step(self) -> None:
        """在 unscale 后、梯度裁剪前取样，快照仍保持参数原精度。"""
        self.before.clear()
        self.metrics.clear()
        for name, parameters in self.modules.items():
            gradients: list[Tensor] = [p.grad for p in parameters if p.grad is not None]
            trainable: int = sum(p.numel() for p in parameters if p.requires_grad)
            with_grad: int = sum(p.numel() for p in parameters if p.requires_grad and p.grad is not None)
            gradient_elements: int = sum(g.numel() for g in gradients)
            nonzero: int = sum(int(torch.count_nonzero(g).item()) for g in gradients)
            self.before[name] = [p.detach().clone() for p in parameters]
            self.metrics[name] = {
                "total_parameters": sum(p.numel() for p in parameters),
                "trainable_parameters": trainable,
                "gradient_parameter_fraction": with_grad / trainable if trainable else None,
                "grad_nonzero_fraction": nonzero / gradient_elements if gradient_elements else None,
                "grad_l2_before_clip": finite_number(vector_norm(gradients)),
                "grad_max_abs": finite_number(max((float(g.detach().abs().max()) for g in gradients), default=0.0)),
                "grad_finite": all(bool(torch.isfinite(g).all()) for g in gradients),
                "weight_l2_before": finite_number(vector_norm(parameters)),
            }

    @torch.no_grad()
    def after_clip(self) -> None:
        """裁剪后的梯度是交给优化器的梯度，与裁剪前范数分开记录。"""
        for name, parameters in self.modules.items():
            self.metrics[name]["grad_l2_after_clip"] = finite_number(
                vector_norm([p.grad for p in parameters if p.grad is not None])
            )

    @torch.no_grad()
    def after_step(self) -> dict[str, dict[str, Any]]:
        """实测参数差，包含 AdamW 自适应缩放与权重衰减的实际影响。"""
        for name, parameters in self.modules.items():
            differences: list[Tensor] = [p.detach().float() - old.float() for p, old in zip(parameters, self.before[name])]
            change: float = vector_norm(differences)
            weight: float | None = self.metrics[name]["weight_l2_before"]
            self.metrics[name]["update_l2"] = finite_number(change)
            self.metrics[name]["relative_update"] = finite_number(change / max(weight, 1e-12)) if weight is not None else None
            self.metrics[name]["parameter_finite"] = all(bool(torch.isfinite(p).all()) for p in parameters)
        self.before.clear()  # 每次取样后释放快照，避免跨更新积累显存。
        return self.metrics


def write_status(folder: Path, phase: str, **details: Any) -> None:
    """原子写入网页读取的进度快照；只由训练主进程调用。"""
    path: Path = folder / "run_state.json"
    previous: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    state: dict[str, Any] = {**previous, **details, "phase": phase, "updated_at": time.time(), "pid": os.getpid()}
    temporary: Path = folder / "run_state.json.tmp"
    temporary.write_text(json.dumps(state, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)
