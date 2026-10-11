"""将 MuChin 原始问答转为可迁移 JSONL，并按歌曲/自定义分组切分。"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ALLOWED_FIELDS: tuple[str, ...] = ("感受", "人声", "配器", "音效", "乐器", "速度", "节奏", "快慢", "风格", "流派")


def read_json(path: Path) -> Any:
    """原始描述文件没有 .json 后缀，但内容仍是 JSON。"""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def extract_pairs(payload: dict[str, Any], qmap: dict[str, str], smap: dict[str, str]) -> list[tuple[str, str]]:
    """保留可听属性，默认排除纯歌词主题、用途、文化背景与编号选项。"""
    pairs: list[tuple[str, str]] = []
    for answer in payload.get("ans", []):
        question: str = qmap.get(answer.get("id", ""), "")
        if any(word in question for word in ALLOWED_FIELDS) and isinstance(answer.get("desc"), str) and answer["desc"].strip():
            pairs.append((question, answer["desc"].strip()))
        for tag in answer.get("tags", []):
            name: str = smap.get(tag.get("tid", ""), "")
            if not any(word in name for word in ALLOWED_FIELDS) or name == "情绪感受（歌词）":
                continue
            description: str = str(tag.get("desc") or "").strip()
            values: Any = tag.get("value", [])
            tags: str = "、".join(str(v) for v in values if v is not None) if isinstance(values, list) else str(values)
            target: str = description or tags.strip()
            if target:
                pairs.append((f"请根据音频描述这首音乐的{name}。", target))
    return list(dict.fromkeys(pairs))


def convert(dataset: Path, audio_root: Path, sources: list[str], group_map: dict[str, str]) -> tuple[list[dict[str, str]], list[str]]:
    """按实际发现的歌曲处理，不把本地子集强行标为 6066 首。"""
    if (dataset / "MuChin-main" / "dataset").is_dir():
        dataset = dataset / "MuChin-main" / "dataset"
    elif (dataset / "dataset").is_dir():
        dataset = dataset / "dataset"
    qmap, smap = read_json(dataset / "qidmap.json"), read_json(dataset / "subqidmap.json")
    audio_map: dict[str, Path] = {}
    for path in sorted(audio_root.rglob("*")):
        if path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}:
            if path.stem in audio_map:
                raise ValueError(f"重复音频 ID：{path.stem}，请先确定对应录音")
            audio_map[path.stem] = path
    rows: list[dict[str, str]] = []
    warnings: list[str] = []
    for folder in sorted((dataset / "bm-data").iterdir()):
        if not folder.is_dir():
            continue
        if folder.name not in audio_map:
            warnings.append(f"缺少音频：{folder.name}")
            continue
        # 跨来源的完全相同问答只保留一份，避免重复文字被当成独立证据。
        seen: set[tuple[str, str]] = set()
        for source in sources:
            filename: str = "prof_desc_json" if source == "professional" else "amat_desc_json"
            try:
                payload: dict[str, Any] = read_json(folder / filename)
                if payload.get("mid") != folder.name:
                    raise ValueError("mid 与目录不匹配")
                pairs: list[tuple[str, str]] = extract_pairs(payload, qmap, smap)
            except (OSError, ValueError, TypeError) as error:
                warnings.append(f"{folder.name}/{filename}: {error}")
                continue
            for question, answer in pairs:
                if (question, answer) in seen:
                    continue
                seen.add((question, answer))
                key: str = hashlib.sha256(f"{folder.name}\0{question}\0{answer}".encode()).hexdigest()[:24]
                rows.append({"id": key, "song_id": folder.name, "group_id": group_map.get(folder.name, folder.name), "audio": audio_map[folder.name].relative_to(audio_root).as_posix(), "question": question, "answer": answer, "source": source})
    if not rows:
        raise ValueError("没有有效音频问答，检查路径和数据格式")
    return rows, warnings


def split_rows(rows: list[dict[str, str]], seed: int, val_ratio: float, test_ratio: float) -> dict[str, list[dict[str, str]]]:
    """分组后切分，保证同组所有片段和问题一起移动。"""
    if not (0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError("验证/测试比例必须为正且总和小于 1")
    groups: list[str] = sorted({row["group_id"] for row in rows})
    if len(groups) < 3:
        raise ValueError("至少需要 3 个独立分组")
    random.Random(seed).shuffle(groups)
    n_val, n_test = max(1, round(len(groups) * val_ratio)), max(1, round(len(groups) * test_ratio))
    if n_val + n_test >= len(groups):
        raise ValueError("验证/测试分组过多，没有训练集")
    val, test = set(groups[:n_val]), set(groups[n_val:n_val + n_test])
    output: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        subset: str = "val" if row["group_id"] in val else "test" if row["group_id"] in test else "train"
        output[subset].append(row)
    return output


def main() -> None:
    """准备清单；已有输出默认拒绝覆盖，只有显式 --overwrite 才允许。"""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", choices=["professional", "both"], default="professional")
    parser.add_argument("--group-map", type=Path, help="可选：歌曲 ID 到重复录音/艺人分组 ID 的 JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args: argparse.Namespace = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise SystemExit("输出目录已有内容；换目录或显式 --overwrite")
    rows, warnings = convert(args.dataset.resolve(), args.audio_root.resolve(), ["professional", "amateur"] if args.sources == "both" else ["professional"], read_json(args.group_map) if args.group_map else {})
    splits: dict[str, list[dict[str, str]]] = split_rows(rows, args.seed, args.val_ratio, args.test_ratio)
    args.output.mkdir(parents=True, exist_ok=True)
    for subset, records in splits.items():
        (args.output / f"{subset}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    summary: dict[str, Any] = {"seed": args.seed, "subsets": {k: {"songs": len({r['song_id'] for r in v}), "qa_pairs": len(v)} for k, v in splits.items()}, "warnings": warnings, "note": "歌词、用途、文化背景等未纳入默认监督；分组映射需按实际重复录音补充。"}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
