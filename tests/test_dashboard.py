"""验证诊断数值、实时日志读取和真实小模型的网页检查链路。"""
from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from torch import nn

from music_train.config import load_config
from music_train.dashboard import DashboardSettings, LogTail, create_app
from music_train.diagnostics import ModuleDiagnostics
from music_train.train import run
from scripts.make_smoke_data import main as make_smoke_data
from scripts.package_source import create_archive


def test_diagnostics_measure_clipped_gradient_and_real_update() -> None:
    """已知 SGD 更新的解析值：梯度裁剪到 0.5，学习率 0.2，更新范数为 0.1。"""
    layer: nn.Linear = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[3.0, 4.0]]))
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.2)
    layer(torch.ones(1, 2)).sum().backward()
    recorder: ModuleDiagnostics = ModuleDiagnostics({"cnn": layer})
    recorder.before_step()
    torch.nn.utils.clip_grad_norm_(layer.parameters(), 0.5)
    recorder.after_clip()
    optimizer.step()
    result = recorder.after_step()["cnn"]
    assert result["grad_l2_before_clip"] == pytest.approx(math.sqrt(2))
    assert result["grad_l2_after_clip"] == pytest.approx(0.5, rel=1e-5)
    assert result["weight_l2_before"] == 5
    assert result["update_l2"] == pytest.approx(0.1, rel=1e-5)
    assert result["relative_update"] == pytest.approx(0.02, rel=1e-5)
    assert result["gradient_parameter_fraction"] == 1


def test_frozen_and_nonfinite_diagnostics_are_explicit() -> None:
    """冻结参数应显示零更新；非有限梯度不能写成 JSON 的 NaN。"""
    layer: nn.Linear = nn.Linear(2, 1, bias=False).requires_grad_(False)
    recorder: ModuleDiagnostics = ModuleDiagnostics({"transformer": layer})
    recorder.before_step()
    recorder.after_clip()
    row = recorder.after_step()["transformer"]
    assert row["trainable_parameters"] == 0 and row["update_l2"] == 0
    layer.requires_grad_(True)
    layer.weight.grad = torch.full_like(layer.weight, float("nan"))
    recorder.before_step()
    recorder.after_clip()
    row = recorder.after_step()["transformer"]
    assert row["grad_l2_before_clip"] is None and row["grad_finite"] is False
    json.dumps(row, allow_nan=False)


def test_log_tail_partial_line_and_resume_rewrite(tmp_path: Path) -> None:
    """半行不会被提前解析，回退重写后的事件不会与旧轮次重复。"""
    path: Path = tmp_path / "metrics.jsonl"
    first: str = json.dumps({"type": "train_step", "step": 1}) + "\n"
    second: str = json.dumps({"type": "module_diagnostics", "step": 2})
    path.write_text(first + second[:10], encoding="utf-8")
    reader: LogTail = LogTail(path)
    assert len(reader.read()["train_step"]) == 1
    with path.open("a", encoding="utf-8") as stream:
        stream.write(second[10:] + "\n")
    assert len(reader.read()["module_diagnostics"]) == 1
    assert len(reader.read()["module_diagnostics"]) == 1
    path.write_text(json.dumps({"type": "epoch", "epoch": 0, "note": "new" * 100}) + "\n", encoding="utf-8")
    result = reader.read()
    assert not result["module_diagnostics"] and len(result["epoch"]) == 1


@pytest.fixture
def laboratory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DashboardSettings:
    """合成数据执行真实一轮训练，网页加载它产生的完整检查点。"""
    source: Path = tmp_path / "songs"
    destination: Path = tmp_path / "experiment"
    monkeypatch.setattr(sys, "argv", ["make_smoke_data", "--output", str(source), "--run-dir", str(destination)])
    make_smoke_data()
    run(load_config(source / "config.json"), cpu=True, stop_after_epoch=1)
    return DashboardSettings(destination, source / "test.jsonl", source, device="cpu")


def test_dashboard_real_training_inspection_and_auth(laboratory: DashboardSettings, monkeypatch: pytest.MonkeyPatch) -> None:
    """日志、播放、检查点、原始/静音特征、生成和 NPZ 全部经过实际后端。"""
    monkeypatch.setenv("MUSIC_API_TOKEN", "test-secret")
    headers: dict[str, str] = {"Authorization": "Bearer test-secret"}
    with TestClient(create_app(laboratory)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/state").status_code == 401
        state = client.get("/api/state", headers=headers).json()
        assert state["progress"]["phase"] == "stopped"
        diagnostic = state["history"]["module_diagnostics"][0]
        for name in ("cnn", "transformer"):
            assert diagnostic["modules"][name]["grad_l2_before_clip"] > 0
            assert diagnostic["modules"][name]["update_l2"] > 0
        songs = client.get("/api/songs", headers=headers).json()
        assert songs["total"] == 2 and "不重叠" in songs["split_check"]
        song_id: str = songs["songs"][0]["song_id"]
        assert client.get(f"/api/songs/{song_id}/audio").status_code == 401
        assert client.get(f"/api/songs/{song_id}/audio", headers=headers).content[:4] == b"RIFF"
        assert client.post("/api/load", json={"checkpoint": "../../other"}, headers=headers).status_code == 422
        assert client.post("/api/load", json={"checkpoint": "epoch-0001"}, headers=headers).status_code == 200
        payload: dict = {"song_id": song_id, "question": "有人声吗？", "max_new_tokens": 4}
        response = client.post("/api/inspect", json=payload, headers=headers)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["transformer"]["shape"] == [8, 32]
        assert result["cnn"]["shape"] == [2, 4, 16]
        assert isinstance(result["answer"], str) and result["question"] == payload["question"]
        assert result["tiny_test_only"] is True
        blob = client.get("/api/features/" + result["download_id"], headers=headers).content
        with np.load(io.BytesIO(blob), allow_pickle=False) as arrays:
            assert np.allclose(arrays["transformer_hidden"][:2, :8], result["transformer"]["preview"])
            assert np.allclose(arrays["cnn_pooled"].reshape(-1, 16)[:2, :8], result["cnn"]["preview"])
        silent = client.post("/api/inspect", json={**payload, "condition": "silence"}, headers=headers).json()
        assert silent["condition"] == "silence"
        assert not np.allclose(silent["transformer"]["preview"], result["transformer"]["preview"])
        assert client.post("/api/inspect", json={**payload, "song_id": "not-in-test"}, headers=headers).status_code == 404


def test_dashboard_rejects_training_set_as_test(laboratory: DashboardSettings) -> None:
    """误把训练清单配置成测试清单时，不默默展示为留出集。"""
    laboratory.test_manifest = laboratory.audio_root / "train.jsonl"
    with TestClient(create_app(laboratory)) as client:
        response = client.get("/api/songs")
        assert response.status_code == 422 and "泄漏" in response.text


def test_source_archive_excludes_private_data_and_weights(tmp_path: Path) -> None:
    """真实数据、环境、检查点即使存在，也不能被白名单打包收集。"""
    for name in ("music_train/model.py", "music_train/web/app.js", "MuChin/private.json", "Qwen3-4B-Base/model.safetensors", "outputs/model.pt", ".env", "data/test.jsonl", ".venv/secret.py"):
        path: Path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private" if name.endswith((".json", ".jsonl", ".safetensors", ".pt")) else "# source", encoding="utf-8")
    result = create_archive(tmp_path, tmp_path / "dist/source.zip")
    assert set(result["files"]) == {"music_train/model.py", "music_train/web/app.js", "MuChin/.gitkeep", "Qwen3-4B-Base/.gitkeep", "data/.gitkeep"}


def test_diagnostics_do_not_change_training_weights(laboratory: DashboardSettings) -> None:
    """相同种子和配置只关闭诊断，验证日志采样没有改变优化结果。"""
    config = load_config(laboratory.audio_root / "config.json")
    config.output_dir = str(laboratory.run_dir.with_name("without_diagnostics"))
    config.training.diagnostics_every_updates = 0
    run(config, cpu=True, stop_after_epoch=1)
    measured = torch.load(laboratory.run_dir / "checkpoints/epoch-0001/model.pt", map_location="cpu", weights_only=True)
    plain = torch.load(Path(config.output_dir) / "checkpoints/epoch-0001/model.pt", map_location="cpu", weights_only=True)
    assert set(measured) == set(plain)
    assert all(torch.equal(measured[key], plain[key]) for key in measured)
