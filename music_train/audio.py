"""本地音频分段读取；MP3 用 soundfile，PCM WAV 有无额外库的测试通道。"""
from __future__ import annotations

import math
import random
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from music_train.config import AudioConfig


def load_segments(path: Path, config: AudioConfig, seed: int | None = None) -> tuple[Tensor, Tensor]:
    """覆盖全曲抽取多个短片段；只读需要的片段，不将整曲全部解码到内存。

    验证/推理使用均匀位置；训练轻微随机移动位置。短音频只使用一个有效槽，
    其余槽以 mask 屏蔽，避免同一片段重复三次被误当成三个独立片段。
    """
    try:
        import soundfile as sf
        handle: Any = sf.SoundFile(str(path))
        kind: str = "soundfile"
        rate, frames = handle.samplerate, len(handle)
    except ImportError:
        if path.suffix.lower() != ".wav":
            raise RuntimeError("读取 MP3/FLAC 需要服务器安装 soundfile>=0.13")
        handle = wave.open(str(path), "rb")
        kind = "wave"
        rate, frames = handle.getframerate(), handle.getnframes()
    try:
        if rate <= 0 or frames <= 0:
            raise ValueError(f"音频为空：{path}")
        window: int = max(1, round(config.segment_seconds * rate))
        target: int = round(config.segment_seconds * config.sample_rate)
        count: int = 1 if frames <= window else config.segments
        starts: list[int] = [round(i * max(0, frames - window) / max(count - 1, 1)) for i in range(count)]
        if count == 1 and frames > window:
            starts = [(frames - window) // 2]
        if seed is not None and frames > window:
            rng: random.Random = random.Random(seed)
            jitter: int = window // 4
            starts = sorted(max(0, min(frames - window, start + rng.randint(-jitter, jitter))) for start in starts)
        result: Tensor = torch.zeros(config.segments, target, dtype=torch.float32)
        mask: Tensor = torch.zeros(config.segments, dtype=torch.bool)
        for index, start in enumerate(starts):
            if kind == "soundfile":
                handle.seek(start)
                samples: np.ndarray = handle.read(window, dtype="float32", always_2d=True).mean(axis=1)
            else:
                if handle.getsampwidth() != 2:
                    raise ValueError("无 soundfile 的 WAV 通道仅支持 PCM 16-bit")
                handle.setpos(start)
                raw: bytes = handle.readframes(window)
                samples = np.frombuffer(raw, dtype="<i2").reshape(-1, handle.getnchannels()).mean(axis=1).astype(np.float32) / 32768.0
            if rate != config.sample_rate:
                from scipy.signal import resample_poly
                divisor: int = math.gcd(rate, config.sample_rate)
                samples = resample_poly(samples, config.sample_rate // divisor, rate // divisor).astype(np.float32)
            if not np.isfinite(samples).all() or len(samples) == 0:
                raise ValueError(f"音频包含非法样本：{path}")
            values: Tensor = torch.from_numpy(np.asarray(samples, dtype=np.float32))[:target]
            result[index, :len(values)] = values
            mask[index] = True
        return result, mask
    finally:
        handle.close()
