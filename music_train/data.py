"""统一清单、歌曲级隔离和确定性问答采样；不在模型里处理文件格式。"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from music_train.audio import load_segments
from music_train.config import AudioConfig


def read_manifest(path: str | Path, audio_root: str | Path) -> list[dict[str, str]]:
    """每行一条问答；audio 必须是 audio_root 内的相对路径。"""
    root: Path = Path(audio_root).resolve()
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row: dict[str, str] = json.loads(line)
        for key in ("id", "song_id", "audio", "question", "answer"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{path}:{line_number} 缺少有效字段 {key}")
        target: Path = (root / row["audio"]).resolve()
        if Path(row["audio"]).is_absolute() or not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f"{path}:{line_number} 音频越界或不存在：{row['audio']}")
        if row["id"] in seen:
            raise ValueError(f"重复问答 ID：{row['id']}")
        seen.add(row["id"])
        row.setdefault("group_id", row["song_id"])
        records.append(row)
    if not records:
        raise ValueError(f"清单为空：{path}")
    return records


def check_disjoint(train: list[dict[str, str]], val: list[dict[str, str]]) -> None:
    """同一歌曲、录音分组或音频路径不能同时出现在训练与验证中。"""
    for key in ("song_id", "group_id", "audio"):
        overlap: set[str] = {r[key] for r in train} & {r[key] for r in val}
        if overlap:
            raise ValueError(f"训练/验证泄漏：{key}={sorted(overlap)[:5]}")


def fingerprint(records: list[dict[str, str]]) -> str:
    """清单内容签名；恢复训练时检查数据划分是否改变。"""
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class QADataset(Dataset):
    """训练默认每首歌每轮选一道题；验证轮流排列歌曲以减少同曲扎堆。"""
    def __init__(self, records: list[dict[str, str]], audio_root: str, audio: AudioConfig, training: bool, seed: int, one_per_song: bool = True) -> None:
        """只存放标注，音频按样本加载；不把整个曲库放进内存。"""
        self.records: list[dict[str, str]] = records
        self.root: Path = Path(audio_root)
        self.audio: AudioConfig = audio
        self.training: bool = training
        self.seed: int = seed
        self.epoch: int = 0
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in records:
            grouped[row["song_id"]].append(row)
        self.groups: list[list[dict[str, str]]] = [grouped[k] for k in sorted(grouped)]
        self.one_per_song: bool = training and one_per_song
        self.interleaved: list[dict[str, str]] = [group[i] for i in range(max(map(len, self.groups))) for group in self.groups if i < len(group)]

    def __len__(self) -> int:
        """训练采样大小与验证的全部问答大小分别计算。"""
        return len(self.groups) if self.one_per_song else len(self.interleaved)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """题目和音频裁剪由 seed、epoch、样本决定，独立于 worker 调度。"""
        if self.one_per_song:
            group: list[dict[str, str]] = self.groups[index]
            offset: int = int(hashlib.sha256(group[0]["song_id"].encode()).hexdigest()[:8], 16)
            row: dict[str, str] = group[(self.epoch + self.seed + offset) % len(group)]
        else:
            row = self.interleaved[index]
        clip_seed: int | None = self.seed + self.epoch * 1000003 + index if self.training else None
        waves, mask = load_segments(self.root / row["audio"], self.audio, clip_seed)
        return {**row, "waves": waves, "segment_mask": mask}


def collate_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """音频堆叠为张量，文字保留字符串，由模型的 tokenizer 处理。"""
    import torch
    return {
        "waves": torch.stack([row["waves"] for row in examples]),
        "segment_mask": torch.stack([row["segment_mask"] for row in examples]),
        "song_ids": [row["song_id"] for row in examples],
        "questions": [row["question"] for row in examples],
        "answers": [row["answer"] for row in examples],
    }
