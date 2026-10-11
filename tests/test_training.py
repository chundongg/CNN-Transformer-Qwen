"""真实张量与 tiny Qwen 测试：标签掩码、梯度、对齐损失及模型包恢复。"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from music_train.checkpoint import apply_bundle, load_bundle, save_bundle
from music_train.config import AudioConfig, Config, LanguageConfig, TrainConfig
from music_train.model import MusicQAModel, multi_positive_contrastive
from music_train.prepare import extract_pairs, split_rows
from music_train.data import check_disjoint


@pytest.fixture
def tiny_config() -> Config:
    """保留完整 CNN/Transformer/Qwen 链路，仅缩小维度以在 CPU 运行。"""
    torch.set_num_threads(2)
    return Config(audio=AudioConfig(sample_rate=16000, segment_seconds=0.5, segments=2, n_fft=256, hop_length=128, n_mels=32, channels=[8, 16], hidden_size=32, heads=4, layers=1, tokens_per_segment=4, dropout=0.0), language=LanguageConfig(tiny=True, mode="full", gradient_checkpointing=False, question_tokens=32, answer_tokens=32, retrieval_text_tokens=64), training=TrainConfig(mixed_precision="no", retrieval_dim=16, num_workers=0))


def sample_batch() -> dict:
    """不同音频和不同回答，避免对比损失因全部为同歌而退化。"""
    return {"waves": torch.randn(2, 2, 8000), "segment_mask": torch.tensor([[True, True], [True, False]]), "questions": ["人声如何？", "速度如何？"], "answers": ["柔和人声", "快速节奏"], "song_ids": ["a", "b"]}


def test_answer_only_labels_and_audio_gradients(tiny_config: Config) -> None:
    """问答损失确实回传到音频卷积、Transformer、连接模块和 Qwen。"""
    tiny_config.training.retrieval_weight = 0.0
    model: MusicQAModel = MusicQAModel(tiny_config)
    batch: dict = sample_batch()
    hidden, mask = model.audio_encoder(batch["waves"], batch["segment_mask"])
    _, _, labels = model.language_batch(hidden, mask, batch["questions"], batch["answers"])
    for index in range(2):
        first_answer: int = hidden.shape[1] + len(model.question_ids(batch["questions"][index]))
        assert (labels[index, :first_answer] == -100).all()
        assert labels[index, first_answer] != -100
    values = model(batch)
    values["loss"].backward()
    for prefix in ("audio_encoder.cnn", "audio_encoder.transformer", "audio_adapter", "qwen"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for name, p in model.named_parameters() if name.startswith(prefix)), prefix


def test_frozen_qwen_still_passes_audio_gradient(tiny_config: Config) -> None:
    """冻结语言模型权重不会自动切断其输入音频的梯度。"""
    tiny_config.language.mode = "frozen"
    tiny_config.training.retrieval_weight = 0
    model: MusicQAModel = MusicQAModel(tiny_config)
    model(sample_batch())["loss"].backward()
    assert all(p.grad is None for p in model.qwen.parameters())
    assert model.audio_encoder.cnn[0].weight.grad.abs().sum() > 0


def test_gradient_checkpointing_with_joint_loss(tiny_config: Config) -> None:
    """服务器默认开启重算；共享 Qwen 的两条分支仍能一次反传。"""
    tiny_config.language.gradient_checkpointing = True
    model: MusicQAModel = MusicQAModel(tiny_config).train()
    model(sample_batch())["loss"].backward()
    for module in (model.audio_encoder.cnn[0], model.audio_retrieval, model.text_retrieval):
        assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
        assert module.weight.grad.abs().sum() > 0


def test_multi_positive_and_matching_loss() -> None:
    """同歌多个正例不互相惩罚；匹配向量优于交换后的错误配对。"""
    identity: Tensor = torch.eye(2)
    good: Tensor = multi_positive_contrastive(identity, identity, ["a", "b"], 0.1)
    bad: Tensor = multi_positive_contrastive(identity, identity.flip(0), ["a", "b"], 0.1)
    assert good < bad
    assert multi_positive_contrastive(identity, identity, ["same", "same"], 0.1).item() == 0


def test_retrieval_heads_receive_gradients(tiny_config: Config) -> None:
    """检索损失同时训练音频和文本投影头，而非只有描述分支在学习。"""
    model: MusicQAModel = MusicQAModel(tiny_config)
    model(sample_batch())["loss"].backward()
    assert model.audio_retrieval.weight.grad.abs().sum() > 0
    assert model.text_retrieval.weight.grad.abs().sum() > 0


def test_checkpoint_roundtrip(tiny_config: Config, tmp_path: Path) -> None:
    """导出的推理包能够独立重建同样的向量，并保留版本标记。"""
    model: MusicQAModel = MusicQAModel(tiny_config).eval()
    before: Tensor = model.encode_text(["温柔人声"]).detach()
    save_bundle(model, tmp_path, {"bundle_id": "test", "retrieval_trained": True, "tiny_test_only": True})
    restored, metadata = load_bundle(tmp_path)
    after: Tensor = restored.encode_text(["温柔人声"]).detach()
    assert torch.allclose(before, after, atol=1e-6)
    assert metadata["bundle_id"] == "test"


def test_group_split_prevents_song_leakage() -> None:
    """同一录音分组中的不同歌 ID 仍必须落在同一集合。"""
    rows: list[dict[str, str]] = [{"id": f"{i}_{q}", "song_id": str(i), "audio": f"{i}.wav", "group_id": str(i // 2), "question": "问题", "answer": "回答"} for i in range(12) for q in range(2)]
    result = split_rows(rows, 42, 0.2, 0.2)
    check_disjoint(result["train"], result["val"])
    check_disjoint(result["train"], result["test"])
    check_disjoint(result["test"], result["val"])
    with pytest.raises(ValueError, match="泄漏"):
        check_disjoint(result["train"], result["train"])


def test_prepare_ignores_quality_options_and_lyrics_only() -> None:
    """编号选项与默认不支持的歌词字段不会变成虚假的声学标签。"""
    payload: dict = {"ans": [{"id": "q", "desc": "温柔", "options": [{"opid": 5}], "tags": [{"tid": "t", "value": ["悲伤歌词"]}, {"tid": "v", "value": ["男声"]}]}]}
    pairs = extract_pairs(payload, {"q": "这首歌带给你的感受？"}, {"t": "情绪感受（歌词）", "v": "歌唱和人声"})
    assert len(pairs) == 2 and all("悲伤歌词" not in answer for _, answer in pairs)
