"""描述生成、歌曲建库、文本召回及留出集配对召回评估。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from music_train.checkpoint import atomic_torch_save
from music_train.data import read_manifest
from music_train.inference import Predictor


def create_index(predictor: Predictor, manifest: str, audio_root: str, output: str) -> dict[str, Any]:
    """每首歌只编码一次；索引记录模型包 ID，防止不同版本向量混用。"""
    records: list[dict[str, str]] = read_manifest(manifest, audio_root)
    unique: dict[str, dict[str, str]] = {row["song_id"]: row for row in records}
    items: list[dict[str, str]] = []
    vectors: list[Tensor] = []
    for index, song_id in enumerate(sorted(unique), 1):
        row: dict[str, str] = unique[song_id]
        vectors.append(predictor.audio_vector(Path(audio_root) / row["audio"]))
        items.append({"song_id": song_id, "audio": row["audio"]})
        if index % 50 == 0:
            print(f"已编码 {index}/{len(unique)} 首歌曲", flush=True)
    bundle: dict[str, Any] = {"bundle_id": predictor.metadata["bundle_id"], "items": items, "vectors": torch.stack(vectors)}
    destination: Path = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(bundle, destination)
    return bundle


def read_index(predictor: Predictor, path: str) -> dict[str, Any]:
    """校验索引与当前编码器属于同一个已训练版本。"""
    bundle: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if bundle["bundle_id"] != predictor.metadata["bundle_id"]:
        raise ValueError("索引模型版本不匹配；请用当前检查点重新建立索引")
    return bundle


def search(predictor: Predictor, index: dict[str, Any], query: str, top_k: int) -> list[dict[str, Any]]:
    """6066 首规模可用精确余弦搜索；以后可替换为 FAISS 等近邻索引。"""
    vector: Tensor = predictor.text_vectors([query])[0]
    values, positions = torch.topk(index["vectors"] @ vector, k=min(max(top_k, 1), len(index["items"])))
    return [index["items"][position] | {"score": float(score)} for score, position in zip(values.tolist(), positions.tolist())]


def evaluate(predictor: Predictor, index: dict[str, Any], manifest: str, audio_root: str) -> dict[str, Any]:
    """人工描述→配对音频的 Recall@K；不是实际用户满意度或多相关性 NDCG。"""
    records: list[dict[str, str]] = read_manifest(manifest, audio_root)
    ids: list[str] = [item["song_id"] for item in index["items"]]
    if not {row["song_id"] for row in records}.issubset(set(ids)):
        raise ValueError("索引缺少评测清单中的歌曲")
    hits: dict[int, int] = {1: 0, 5: 0, 10: 0}
    for start in range(0, len(records), 16):
        batch: list[dict[str, str]] = records[start:start + 16]
        vectors: Tensor = predictor.text_vectors([row["answer"] for row in batch])
        ranks: Tensor = torch.topk(vectors @ index["vectors"].T, k=min(10, len(ids)), dim=1).indices
        for row, positions in zip(batch, ranks.tolist()):
            found: list[str] = [ids[position] for position in positions]
            for k in hits:
                hits[k] += row["song_id"] in found[:k]
    return {"queries": len(records), "songs_in_index": len(ids), "recall": {f"R@{k}": count / len(records) for k, count in hits.items()}, "note": "仅评价已知描述与其歌曲的配对；其他语义相关歌曲可能未被标为正样本。"}


def main() -> None:
    """所有子命令共用已训练检查点；只有建库和评测需要数据根目录。"""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["describe", "index", "search", "evaluate"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--audio")
    parser.add_argument("--question", default="请描述这首音乐的人声、伴奏、节奏和整体氛围。")
    parser.add_argument("--query")
    parser.add_argument("--manifest")
    parser.add_argument("--audio-root")
    parser.add_argument("--index")
    parser.add_argument("--output")
    parser.add_argument("--top-k", type=int, default=10)
    args: argparse.Namespace = parser.parse_args()
    predictor: Predictor = Predictor(args.checkpoint, args.base_model, args.device)
    if args.action == "describe":
        if not args.audio:
            parser.error("describe 需要 --audio")
        result: Any = {"answer": predictor.describe(args.audio, args.question), "tiny_test_only": predictor.metadata["tiny_test_only"]}
    elif args.action == "index":
        if not all((args.manifest, args.audio_root, args.output)):
            parser.error("index 需要 --manifest --audio-root --output")
        index = create_index(predictor, args.manifest, args.audio_root, args.output)
        result = {"songs": len(index["items"]), "index": args.output}
    elif args.action == "search":
        if not args.index or not args.query:
            parser.error("search 需要 --index --query")
        result = search(predictor, read_index(predictor, args.index), args.query, args.top_k)
    else:
        if not all((args.index, args.manifest, args.audio_root)):
            parser.error("evaluate 需要 --index --manifest --audio-root")
        result = evaluate(predictor, read_index(predictor, args.index), args.manifest, args.audio_root)
    rendered: str = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output and args.action != "index":
        Path(args.output).write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
