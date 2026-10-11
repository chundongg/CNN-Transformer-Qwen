"""生成极小的合成 WAV 与配置，只验证训练管线，不衡量音乐理解质量。"""
from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np


def main() -> None:
    """提供 4 首训练、2 首验证、2 首测试的不同合成信号，默认拒绝覆盖。"""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke_data"))
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/smoke_run"))
    args: argparse.Namespace = parser.parse_args()
    folder: Path = args.output.resolve()
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit("合成数据输出已存在，请换目录")
    folder.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    rate: int = 16000
    for number in range(8):
        name: str = f"synthetic_{number:02d}"
        time: np.ndarray = np.arange(rate) / rate
        values: np.ndarray = (np.sin(2 * math.pi * (180 + number * 50) * time) * 10000).astype("<i2")
        with wave.open(str(folder / f"{name}.wav"), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes(values.tobytes())
        split: str = "train" if number < 4 else "val" if number < 6 else "test"
        for question, answer in (("描述声音。", "持续的合成音。"), ("有人声吗？", "这段合成测试音没有人声。")):
            rows[split].append({"id": f"{name}_{len(rows[split])}", "song_id": name, "group_id": name, "audio": f"{name}.wav", "question": question, "answer": answer, "source": "synthetic_test_only"})
    for split, examples in rows.items():
        (folder / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples), encoding="utf-8")
    config: dict = {"audio_root": str(folder), "train_manifest": str(folder / "train.jsonl"), "val_manifest": str(folder / "val.jsonl"), "output_dir": str(args.run_dir.resolve()), "audio": {"sample_rate": rate, "segment_seconds": 0.5, "segments": 2, "n_fft": 256, "hop_length": 128, "n_mels": 32, "channels": [8, 16], "hidden_size": 32, "heads": 4, "layers": 1, "tokens_per_segment": 4, "dropout": 0.0}, "language": {"tiny": True, "mode": "full", "gradient_checkpointing": False, "question_tokens": 48, "answer_tokens": 48, "retrieval_text_tokens": 64}, "training": {"epochs": 3, "batch_size": 2, "eval_batch_size": 2, "accumulation_steps": 2, "num_workers": 0, "mixed_precision": "no", "retrieval_dim": 16, "retrieval_weight": 0.1, "log_every_updates": 1}}
    (folder / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"仅用于管线自检：{folder / 'config.json'}")


if __name__ == "__main__":
    main()
