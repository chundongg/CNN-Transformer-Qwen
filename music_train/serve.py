"""部署 API：接收音频字节，返回描述或向量；不接受客户端服务器路径。"""
from __future__ import annotations

import hmac
import os
import tempfile
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from music_train.inference import Predictor


class TextRequest(BaseModel):
    """避免不受控文本长度占满模型上下文。"""
    text: str = Field(min_length=1, max_length=1000)


def create_app(predictor: Predictor | None = None) -> FastAPI:
    """生产从环境变量加载，测试可以注入已经构造的 Predictor。"""
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """启动前完成模型装载；加载失败时服务不会伪报 ready。"""
        checkpoint: str = os.getenv("MUSIC_CHECKPOINT", "")
        if predictor is None and not checkpoint:
            raise RuntimeError("请设置 MUSIC_CHECKPOINT 为训练输出的 epoch 检查点目录")
        application.state.predictor = predictor or Predictor(checkpoint, os.getenv("MUSIC_BASE_MODEL"), os.getenv("MUSIC_DEVICE", "auto"))
        yield

    app: FastAPI = FastAPI(title="Music Audio QA", lifespan=lifespan)

    async def authorize(request: Request) -> None:
        """设置 MUSIC_API_TOKEN 后要求 Bearer 令牌；默认部署只监听本机。"""
        token: str = os.getenv("MUSIC_API_TOKEN", "")
        if token and not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {token}"):
            raise HTTPException(401, "令牌错误或缺失")

    async def audio_call(request: Request, extension: str, question: str | None, max_new_tokens: int = 192) -> dict[str, Any]:
        """流式限制上传大小；请求结束删除临时文件，模型计算在工作线程中进行。"""
        maximum: int = 50 * 1024 * 1024
        filename: str = ""
        try:
            with tempfile.NamedTemporaryFile(suffix="." + extension, delete=False) as stream:
                filename = stream.name
                size: int = 0
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > maximum:
                        raise HTTPException(413, "音频上限为 50 MiB")
                    stream.write(chunk)
            if size == 0:
                raise HTTPException(422, "音频为空")
            current: Predictor = app.state.predictor
            if question is None:
                vector = await run_in_threadpool(current.audio_vector, filename)
                return {"embedding": vector.tolist(), "bundle_id": current.metadata["bundle_id"]}
            answer: str = await run_in_threadpool(current.describe, filename, question, max_new_tokens)
            return {"answer": answer, "bundle_id": current.metadata["bundle_id"], "tiny_test_only": current.metadata["tiny_test_only"]}
        except HTTPException:
            raise
        except (ValueError, RuntimeError, OSError, EOFError, wave.Error) as error:
            raise HTTPException(422, str(error)) from error
        finally:
            if filename:
                Path(filename).unlink(missing_ok=True)

    @app.get("/health")
    def health() -> dict[str, Any]:
        """不暴露磁盘路径；明确报告 tiny 自检模型与检索是否训练。"""
        current: Predictor = app.state.predictor
        return {"ready": True, "device": current.device, "bundle_id": current.metadata["bundle_id"], "tiny_test_only": current.metadata["tiny_test_only"], "retrieval_trained": current.metadata["retrieval_trained"]}

    @app.post("/v1/describe", dependencies=[Depends(authorize)])
    async def describe(request: Request, question: str = Query(default="请描述这首音乐的人声、伴奏、节奏和整体氛围。", min_length=1, max_length=1000), format: Literal["wav", "mp3", "flac", "ogg"] = "mp3", max_new_tokens: int = Query(default=192, ge=1, le=512)) -> dict[str, Any]:
        """请求体是音频原始字节，避免额外 multipart 依赖。"""
        return await audio_call(request, format, question, max_new_tokens)

    @app.post("/v1/embed/audio", dependencies=[Depends(authorize)])
    async def embed_audio(request: Request, format: Literal["wav", "mp3", "flac", "ogg"] = "mp3") -> dict[str, Any]:
        """编码上传音频，用于离线建库或与已有推荐服务集成。"""
        return await audio_call(request, format, None)

    @app.post("/v1/embed/text", dependencies=[Depends(authorize)])
    async def embed_text(payload: TextRequest) -> dict[str, Any]:
        """编码用户需求；返回的向量必须与同 bundle_id 的歌曲索引比较。"""
        current: Predictor = app.state.predictor
        try:
            vector = await run_in_threadpool(current.text_vectors, [payload.text])
            return {"embedding": vector[0].tolist(), "bundle_id": current.metadata["bundle_id"]}
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    return app


app: FastAPI = create_app()
