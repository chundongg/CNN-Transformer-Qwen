"""真实 tiny 模型部署接口、音频处理和曲线输出测试。"""
from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from music_train.audio import load_segments
from music_train.checkpoint import save_bundle
from music_train.config import AudioConfig, Config, LanguageConfig, TrainConfig
from music_train.inference import Predictor
from music_train.model import MusicQAModel
from music_train.plot import plot_svg
from music_train.serve import create_app


def wav_bytes() -> bytes:
    """生成短于窗口的 PCM WAV，便于同时验证 padding 与无效片段屏蔽。"""
    output: io.BytesIO = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes((torch.sin(torch.arange(1600) * 0.1) * 10000).short().numpy().astype("<i2").tobytes())
    return output.getvalue()


def config() -> Config:
    """部署测试沿用真正的网络模块，尺寸很小且不访问外部模型。"""
    torch.set_num_threads(2)
    return Config(audio=AudioConfig(sample_rate=16000, segment_seconds=0.5, segments=2, n_fft=256, hop_length=128, n_mels=32, channels=[8, 16], hidden_size=32, heads=4, layers=1, tokens_per_segment=4), language=LanguageConfig(tiny=True, mode="full", gradient_checkpointing=False, question_tokens=32, answer_tokens=32, retrieval_text_tokens=64), training=TrainConfig(mixed_precision="no", retrieval_dim=16))


def test_short_audio_mask(tmp_path: Path) -> None:
    """短音频只占一个有效片段，其余槽位全零且被 mask。"""
    path: Path = tmp_path / "short.wav"
    path.write_bytes(wav_bytes())
    waves, mask = load_segments(path, config().audio)
    assert waves.shape == (2, 8000)
    assert mask.tolist() == [True, False]
    assert waves[1].abs().sum() == 0


def test_deployment_audio_text_and_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """API 实际运行描述和双模态编码，并拒绝未认证、空文件与坏音频。"""
    model: MusicQAModel = MusicQAModel(config()).eval()
    save_bundle(model, tmp_path, {"bundle_id": "api-test", "tiny_test_only": True, "retrieval_trained": True})
    predictor: Predictor = Predictor(str(tmp_path), device="cpu")
    monkeypatch.setenv("MUSIC_API_TOKEN", "test-token")
    headers: dict[str, str] = {"Authorization": "Bearer test-token", "Content-Type": "audio/wav"}
    with TestClient(create_app(predictor)) as client:
        assert client.get("/health").json()["tiny_test_only"] is True
        assert client.post("/v1/embed/text", json={"text": "轻柔"}).status_code == 401
        response = client.post("/v1/embed/audio?format=wav", content=wav_bytes(), headers=headers)
        assert response.status_code == 200
        vector = torch.tensor(response.json()["embedding"])
        assert vector.shape == (16,) and torch.allclose(vector.norm(), torch.tensor(1.0), atol=1e-5)
        response = client.post("/v1/embed/text", json={"text": "轻柔"}, headers={"Authorization": "Bearer test-token"})
        assert response.status_code == 200 and len(response.json()["embedding"]) == 16
        response = client.post("/v1/describe?format=wav&max_new_tokens=4", content=wav_bytes(), headers=headers)
        assert response.status_code == 200 and isinstance(response.json()["answer"], str)
        assert client.post("/v1/describe?format=wav", content=b"", headers=headers).status_code == 422
        assert client.post("/v1/describe?format=wav", content=b"bad audio", headers=headers).status_code == 422


def test_untrained_retrieval_is_rejected(tmp_path: Path) -> None:
    """问答模型不能因存在随机投影头就被当成已训练检索模型。"""
    save_bundle(MusicQAModel(config()), tmp_path, {"bundle_id": "qa-only", "tiny_test_only": True, "retrieval_trained": False})
    predictor: Predictor = Predictor(str(tmp_path), device="cpu")
    with pytest.raises(ValueError, match="未训练检索"):
        predictor.text_vectors(["温柔音乐"])


def test_svg_curve_uses_logged_values(tmp_path: Path) -> None:
    """缺少 matplotlib 的环境仍能根据日志输出独立可读曲线。"""
    metrics: dict[str, float] = {"qa_loss": 2.125, "retrieval_loss": 0.25, "total_loss": 2.15}
    log: Path = tmp_path / "metrics.jsonl"
    log.write_text(json.dumps({"type": "epoch", "epoch": 0, "train": metrics, "val": metrics}) + "\n", encoding="utf-8")
    paths: list[str] = plot_svg(log, tmp_path / "curves")
    assert len(paths) == 1
    text: str = Path(paths[0]).read_text(encoding="utf-8")
    assert "2.125000" in text and "Validation" in text and "<polyline" in text
