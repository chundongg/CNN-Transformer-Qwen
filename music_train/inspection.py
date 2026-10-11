"""将真实音频中间张量转换为有尺寸说明的可视化数据，并保留原始数值下载。"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from music_train.audio import load_segments
from music_train.inference import Predictor


def tensor_summary(tensor: Tensor, matrix: Tensor) -> dict[str, Any]:
    """热力图只等距取部分行列；尺寸、统计与下载使用完整张量。"""
    full: Tensor = tensor.detach().float().cpu()
    view: Tensor = matrix.detach().float().cpu()
    rows: Tensor = torch.linspace(0, view.shape[0] - 1, min(view.shape[0], 128)).round().long().unique()
    columns: Tensor = torch.linspace(0, view.shape[1] - 1, min(view.shape[1], 64)).round().long().unique()
    if not torch.isfinite(full).all():
        raise ValueError("中间特征含 NaN/Inf，请检查模型与训练日志")
    return {
        "shape": list(full.shape), "elements": full.numel(),
        "mean": full.mean().item(), "std": full.std(unbiased=False).item(),
        "min": full.min().item(), "max": full.max().item(), "l2": full.norm().item(),
        "preview": view[:2, :8].tolist(), "token_norms": view.norm(dim=-1).tolist(),
        "heatmap": {"values": view[rows][:, columns].tolist(), "row_indices": rows.tolist(), "column_indices": columns.tolist()},
    }


def inspect_audio(predictor: Predictor, audio_path: Path, question: str, max_new_tokens: int = 192, silence: bool = False) -> tuple[dict[str, Any], bytes]:
    """不把人工参考答案输入模型；同一隐藏序列用于展示与 Qwen 回答。"""
    with predictor.lock, torch.inference_mode(), predictor.autocast():
        waves, segment_mask = load_segments(audio_path, predictor.model.config.audio)
        if silence:
            waves.zero_()  # 保留长度和 mask，仅替换音频内容，便于受控消融。
        waves = waves.unsqueeze(0).to(predictor.device)
        segment_mask = segment_mask.unsqueeze(0).to(predictor.device)
        captured: dict[str, Tensor] = {}

        def capture_cnn(module: nn.Module, arguments: tuple[Any, ...], output: Tensor) -> None:
            """短暂注册 hook 读取真实卷积输出，不替换模型计算。"""
            captured["cnn_raw"] = output.detach()

        def capture_pool(module: nn.Module, arguments: tuple[Any, ...], output: Tensor) -> None:
            """记录投影前的 CNN 池化结果，保留每个片段的时间 token。"""
            captured["cnn_pool"] = output.detach()

        handles = [predictor.model.audio_encoder.cnn.register_forward_hook(capture_cnn), predictor.model.audio_encoder.time_pool.register_forward_hook(capture_pool)]
        try:
            hidden, valid = predictor.model.audio_encoder(waves, segment_mask)
        finally:
            for handle in handles:
                handle.remove()
        cnn: Tensor = captured["cnn_pool"].squeeze(2).transpose(1, 2)
        matrix: Tensor = cnn.reshape(-1, cnn.shape[-1])
        answer: str = predictor.model.describe_hidden(hidden, valid, question, max_new_tokens)
        result: dict[str, Any] = {
            "question": question, "answer": answer, "condition": "silence" if silence else "original",
            "bundle_id": predictor.metadata["bundle_id"], "tiny_test_only": predictor.metadata["tiny_test_only"],
            "segment_mask": segment_mask[0].cpu().tolist(), "token_mask": valid[0].cpu().tolist(),
            "cnn": tensor_summary(cnn, matrix), "cnn_raw_shape": list(captured["cnn_raw"].shape),
            "transformer": tensor_summary(hidden[0], hidden[0]),
            "note": "隐藏向量没有固定的中文维度含义；热力图是数值分布，不能单独证明模型理解音乐。无效片段请结合 mask 阅读。",
        }
        arrays: dict[str, np.ndarray] = {
            "cnn_raw": captured["cnn_raw"].float().cpu().numpy(),
            "cnn_pooled": cnn.float().cpu().numpy(),
            "transformer_hidden": hidden[0].float().cpu().numpy(),
            "segment_mask": segment_mask[0].cpu().numpy(), "token_mask": valid[0].cpu().numpy(),
        }
        if predictor.metadata.get("retrieval_trained"):
            vector: Tensor = predictor.model.audio_vectors(hidden, valid)[0].float().cpu()
            result["audio_embedding"] = vector.tolist()
            arrays["audio_embedding"] = vector.numpy()
        stream: io.BytesIO = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        return result, stream.getvalue()
