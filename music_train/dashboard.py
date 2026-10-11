"""训练网页后端：只读日志、显式加载完整检查点、受控测试清单与中间特征检查。"""
from __future__ import annotations

import argparse
import gc
import hmac
import json
import os
import threading
import time
import uuid
import wave
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from music_train.data import check_disjoint, read_manifest
from music_train.inference import Predictor
from music_train.inspection import inspect_audio


@dataclass
class DashboardSettings:
    """只有启动服务的人能设置服务器路径；HTTP 请求不能任意读取文件。"""
    run_dir: Path
    test_manifest: Path | None = None
    audio_root: Path | None = None
    base_model: str | None = None
    device: str = "cpu"


class LoadRequest(BaseModel):
    """只接受当前实验下可枚举的完整检查点。"""
    checkpoint: str = Field(min_length=1, max_length=100)


class InspectRequest(BaseModel):
    """问题由用户提供，测试参考答案仅供展示，不会作为生成上下文。"""
    song_id: str = Field(min_length=1, max_length=256)
    question: str = Field(min_length=1, max_length=1000)
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    condition: Literal["original", "silence"] = "original"


class LogTail:
    """增量读取完整 JSONL 行；容忍并发写入半行及恢复训练时的日志回退。"""
    def __init__(self, path: Path, limit: int = 3000) -> None:
        """分别限制各类曲线的内存，不让高频梯度事件挤掉 epoch 指标。"""
        self.path: Path = path
        self.offset: int = 0
        self.anchor: bytes = b""
        self.lock: threading.Lock = threading.Lock()
        self.events: dict[str, deque] = {name: deque(maxlen=limit) for name in ("train_step", "epoch", "module_diagnostics")}
        self.invalid_lines: int = 0

    def read(self) -> dict[str, Any]:
        """保留未结束行给下一次轮询；前次读到的位置被改写时重新扫描。"""
        with self.lock:
            if self.path.exists():
                with self.path.open("rb") as stream:
                    stream.seek(max(0, self.offset - len(self.anchor)))
                    changed: bool = stream.read(len(self.anchor)) != self.anchor
                    if self.path.stat().st_size < self.offset or changed:
                        self.offset, self.anchor, self.invalid_lines = 0, b"", 0
                        for history in self.events.values():
                            history.clear()
                    stream.seek(self.offset)
                    while line := stream.readline():
                        if not line.endswith(b"\n"):
                            break
                        self.offset = stream.tell()
                        try:
                            event: dict[str, Any] = json.loads(line)
                            kind: str = event.get("type", "")
                            if kind in self.events:
                                self.events[kind].append(event)
                        except (ValueError, UnicodeDecodeError):
                            self.invalid_lines += 1
                    stream.seek(max(0, self.offset - 128))
                    self.anchor = stream.read(min(self.offset, 128))
            return {**{name: list(rows) for name, rows in self.events.items()}, "invalid_lines": self.invalid_lines, "history_limit_per_type": 3000}


class DashboardService:
    """网页监控不加载模型；用户点加载后，检查请求使用同一个串行 Predictor。"""
    def __init__(self, settings: DashboardSettings) -> None:
        """初始化目录和索引，不占用模型显存、不启动训练进程。"""
        self.settings: DashboardSettings = settings
        self.run_dir: Path = settings.run_dir.resolve()
        self.logs: LogTail = LogTail(self.run_dir / "metrics.jsonl")
        self.lock: threading.RLock = threading.RLock()
        self.predictor: Predictor | None = None
        self.loaded_checkpoint: str | None = None
        self.songs: dict[str, list[dict[str, str]]] = {}
        self.catalog_loaded: bool = False
        self.split_check: str = "未配置测试集"
        self.downloads: OrderedDict[str, bytes] = OrderedDict()

    def checkpoints(self) -> list[dict[str, Any]]:
        """只展示已写完 READY 的目录，避免读取保存到一半的模型。"""
        result: list[dict[str, Any]] = []
        for path in sorted((self.run_dir / "checkpoints").glob("epoch-*")):
            if not path.is_dir() or not (path / "READY").exists():
                continue
            try:
                meta: dict[str, Any] = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
                result.append({"name": path.name, "epoch": meta.get("epoch"), "val": meta.get("val", {}), "tiny_test_only": meta.get("tiny_test_only", False)})
            except (OSError, ValueError):
                continue
        return result

    def state(self) -> dict[str, Any]:
        """状态来自训练进程写入的快照；时间戳不等于进程仍然存活的证明。"""
        path: Path = self.run_dir / "run_state.json"
        try:
            progress: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"phase": "unknown"}
        except (OSError, ValueError):
            progress = {"phase": "unknown"}
        return {"run_name": self.run_dir.name, "progress": progress, "server_time": time.time(), "loaded_checkpoint": self.loaded_checkpoint, "inference_device": self.settings.device, "checkpoints": self.checkpoints(), "history": self.logs.read()}

    def load_catalog(self) -> None:
        """缓存测试清单，若能访问训练/验证清单则同时核对歌曲与录音隔离。"""
        with self.lock:
            if self.catalog_loaded:
                return
            if self.settings.test_manifest is None or self.settings.audio_root is None:
                raise ValueError("启动网页时请提供 --test-manifest 和 --audio-root")
            rows: list[dict[str, str]] = read_manifest(self.settings.test_manifest, self.settings.audio_root)
            resolved: Path = self.run_dir / "resolved_config.json"
            self.split_check = "已读取指定测试清单；缺少原训练清单，未重新核对隔离"
            if resolved.exists():
                config: dict[str, Any] = json.loads(resolved.read_text(encoding="utf-8"))
                paths: list[Path] = [Path(config[key]) for key in ("train_manifest", "val_manifest")]
                if all(path.is_file() for path in paths):
                    for path in paths:
                        check_disjoint(read_manifest(path, self.settings.audio_root), rows)
                    self.split_check = "已核对：测试集与训练、验证集的歌曲/分组/路径均不重叠"
            for row in rows:
                self.songs.setdefault(row["song_id"], []).append(row)
            self.catalog_loaded = True

    def song(self, song_id: str) -> list[dict[str, str]]:
        """歌曲必须出现在配置的清单中；不会把请求值拼接成磁盘路径。"""
        self.load_catalog()
        if song_id not in self.songs:
            raise KeyError("测试集中没有此歌曲")
        return self.songs[song_id]

    def audio_path(self, song_id: str) -> Path:
        """再次检查实际文件路径在音频根内，避免清单加载后的路径替换越界。"""
        row: dict[str, str] = self.song(song_id)[0]
        root: Path = self.settings.audio_root.resolve()
        path: Path = (root / row["audio"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("歌曲音频不存在或超出已配置的音频目录")
        return path

    def load_checkpoint(self, name: str) -> dict[str, Any]:
        """串行更换模型，先释放旧实例；训练期间建议只监控或使用另一设备。"""
        with self.lock:
            if name not in {row["name"] for row in self.checkpoints()}:
                raise ValueError("请选择本实验中已经完整保存的检查点")
            self.predictor = None
            self.loaded_checkpoint = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.predictor = Predictor(str(self.run_dir / "checkpoints" / name), self.settings.base_model, self.settings.device)
            self.loaded_checkpoint = name
            return {"checkpoint": name, "device": self.predictor.device, "tiny_test_only": self.predictor.metadata["tiny_test_only"], "bundle_id": self.predictor.metadata["bundle_id"]}

    def inspect(self, payload: InspectRequest) -> dict[str, Any]:
        """返回真特征、真生成文本和下载 ID；最多缓存最近 5 次张量结果。"""
        with self.lock:
            if self.predictor is None:
                raise ValueError("请先加载一个完整检查点")
            path: Path = self.audio_path(payload.song_id)
            result, archive = inspect_audio(self.predictor, path, payload.question, payload.max_new_tokens, payload.condition == "silence")
            identifier: str = uuid.uuid4().hex
            self.downloads[identifier] = archive
            while len(self.downloads) > 5:
                self.downloads.popitem(last=False)
            return {**result, "song_id": payload.song_id, "checkpoint": self.loaded_checkpoint, "download_id": identifier}


def create_app(settings: DashboardSettings) -> FastAPI:
    """API 统一鉴权；静态页面不含训练数据，令牌不出现在 URL 中。"""
    app: FastAPI = FastAPI(title="音乐训练观察台")
    service: DashboardService = DashboardService(settings)
    app.state.dashboard = service

    async def authorize(request: Request) -> None:
        """可选沿用 MUSIC_API_TOKEN；浏览器请求通过 Authorization 头发送。"""
        token: str = os.getenv("MUSIC_API_TOKEN", "")
        if token and not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {token}"):
            raise HTTPException(401, "请填写正确的访问令牌")

    api: APIRouter = APIRouter(prefix="/api", dependencies=[Depends(authorize)])

    @api.get("/state")
    def state() -> dict[str, Any]:
        """每次只读新增日志；网页默认 3 秒轮询，不重新运行训练。"""
        return service.state()

    @api.get("/songs")
    def songs(query: str = "", offset: int = Query(0, ge=0), limit: int = Query(40, ge=1, le=100)) -> dict[str, Any]:
        """在服务器按 ID 或音频文件名筛选，避免一次返回所有问答。"""
        try:
            service.load_catalog()
            records: list[dict[str, Any]] = [{"song_id": key, "audio": rows[0]["audio"], "questions": len(rows)} for key, rows in service.songs.items() if query.casefold() in (key + " " + rows[0]["audio"]).casefold()]
            return {"total": len(records), "songs": records[offset:offset + limit], "split_check": service.split_check}
        except (OSError, ValueError) as error:
            raise HTTPException(422, str(error)) from error

    @api.get("/songs/{song_id}")
    def song(song_id: str) -> dict[str, Any]:
        """参考答案供人工比较，绝不拼入模型提问。"""
        try:
            rows: list[dict[str, str]] = service.song(song_id)
            return {"song_id": song_id, "references": [{"question": r["question"], "answer": r["answer"], "source": r.get("source", "unknown")} for r in rows]}
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except (OSError, ValueError) as error:
            raise HTTPException(422, str(error)) from error

    @api.get("/songs/{song_id}/audio")
    def audio(song_id: str) -> FileResponse:
        """只播放测试清单中的文件；不能请求任意服务器文件。"""
        try:
            return FileResponse(service.audio_path(song_id))
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except (OSError, ValueError) as error:
            raise HTTPException(422, str(error)) from error

    @api.post("/load")
    async def load(payload: LoadRequest) -> dict[str, Any]:
        """把模型装载放在工作线程中，状态轮询仍可响应。"""
        try:
            return await run_in_threadpool(service.load_checkpoint, payload.checkpoint)
        except (OSError, ValueError, RuntimeError, EOFError, wave.Error) as error:
            raise HTTPException(422, str(error)) from error

    @api.post("/inspect")
    async def inspect(payload: InspectRequest) -> dict[str, Any]:
        """执行一次音频检查，可选静音对照；不会更新模型权重。"""
        try:
            return await run_in_threadpool(service.inspect, payload)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except (OSError, ValueError, RuntimeError, EOFError, wave.Error) as error:
            raise HTTPException(422, str(error)) from error

    @api.get("/features/{identifier}")
    def features(identifier: str) -> Response:
        """下载本会话近期生成的完整数组，不接受文件系统路径。"""
        with service.lock:
            archive: bytes | None = service.downloads.get(identifier)
            if archive is None:
                raise HTTPException(404, "结果已过期，请重新执行检查")
            return Response(archive, media_type="application/octet-stream", headers={"Content-Disposition": 'attachment; filename="audio_features.npz"'})

    app.include_router(api)
    app.mount("/", StaticFiles(directory=Path(__file__).parent / "web", html=True), name="web")
    return app


def main() -> None:
    """只启动查看服务；训练在独立终端运行，不暴露远程执行任意命令功能。"""
    import uvicorn
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--base-model")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8810)
    args: argparse.Namespace = parser.parse_args()
    settings: DashboardSettings = DashboardSettings(args.run_dir, args.test_manifest, args.audio_root, args.base_model, args.device)
    uvicorn.run(create_app(settings), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
