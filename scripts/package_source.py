"""显式白名单打包代码：数据集、标注、权重、日志和虚拟环境不会进入源码包。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def create_archive(root: Path, destination: Path) -> dict[str, object]:
    """只收集指定源码目录的文本文件，数据与基座目录只放空 .gitkeep。"""
    root = root.resolve()
    top_files: tuple[str, ...] = ("README.md", "DEPLOYMENT.md", "WEB_GUIDE.md", "CHECK_REPORT.md", "requirements.txt", "pytest.ini", ".gitignore", ".gitattributes")
    paths: list[Path] = [root / name for name in top_files if (root / name).is_file()]
    extensions: set[str] = {".py", ".yaml", ".yml", ".md", ".html", ".css", ".js", ".svg"}
    for folder in ("music_train", "configs", "deploy", "scripts", "tests"):
        for path in (root / folder).rglob("*"):
            if path.is_file() and (path.suffix in extensions or (folder == "deploy" and path.suffix == ".sh")) and not any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(root / folder).parts):
                if not path.resolve().is_relative_to(root):
                    raise ValueError("源码中存在指向工程外的链接，拒绝打包")
                paths.append(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path = destination.with_suffix(destination.suffix + ".tmp")
    names: list[str] = []
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            payload: bytes = path.read_bytes()
            if len(payload) > 2 * 1024 * 1024:
                raise ValueError(f"源码文本异常大，请人工检查：{path.relative_to(root)}")
            payload.decode("utf-8")  # 拒绝误放入源码目录的二进制内容。
            name: str = path.relative_to(root).as_posix()
            archive.writestr("CNN-Transformer-Qwen/" + name, payload)
            names.append(name)
        for folder in ("MuChin", "Qwen3-4B-Base", "data"):
            name = folder + "/.gitkeep"
            archive.writestr("CNN-Transformer-Qwen/" + name, b"")
            names.append(name)
    os.replace(temporary, destination)
    return {"archive": str(destination), "files": names, "file_count": len(names), "bytes": destination.stat().st_size, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "datasets_and_weights_included": False}


def main() -> None:
    """服务器迁移前可再次生成源码包；不执行任何 Git 推送。"""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dist/CNN-Transformer-Qwen-source.zip"))
    args: argparse.Namespace = parser.parse_args()
    result: dict[str, object] = create_archive(Path(__file__).resolve().parents[1], args.output)
    args.output.with_suffix(".manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
