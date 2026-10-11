"""从真实 JSONL 日志绘制曲线；不生成任何模拟损失或虚构实验结果。"""
from __future__ import annotations

import argparse
import json
import csv
from html import escape
from pathlib import Path
from typing import Any


def plot_svg(log_path: Path, output_dir: Path) -> list[str]:
    """无 matplotlib 时仍输出可查看的 SVG 诊断曲线，数值全部来自真实日志。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    saved: list[str] = []
    for name, kind, x_key in (("epoch_losses", "epoch", "epoch"), ("step_losses", "train_step", "step")):
        records: list[dict[str, Any]] = [row for row in rows if row["type"] == kind]
        if not records:
            continue
        parts: list[str] = ['<svg xmlns="http://www.w3.org/2000/svg" width="1260" height="380" viewBox="0 0 1260 380">', '<rect width="1260" height="380" fill="white"/>', '<g font-family="Arial, sans-serif" font-size="12" fill="#243642">']
        xs: list[float] = [float(row[x_key] + (1 if kind == "epoch" else 0)) for row in records]
        x_min, x_max = min(xs), max(xs)
        if x_min == x_max:
            x_min, x_max = x_min - 0.5, x_max + 0.5
        for panel, (metric, title) in enumerate((("qa_loss", "Answer cross-entropy"), ("retrieval_loss", "Batch contrastive loss"), ("total_loss", "Weighted objective"))):
            series: list[tuple[str, str, list[float]]] = [("Train", "#1769aa", [row["train"][metric] for row in records]), ("Validation", "#d06b29", [row["val"][metric] for row in records])] if kind == "epoch" else [("Train", "#1769aa", [row[metric] for row in records])]
            numbers: list[float] = [number for _, _, values in series for number in values]
            low, high = min(numbers), max(numbers)
            margin: float = max((high - low) * 0.15, abs(high) * 0.002, 1e-4)
            low, high = low - margin, high + margin
            left: int = panel * 420 + 68
            top, width, height = 60, 325, 245
            parts.append(f'<text x="{left}" y="25" font-size="16">{escape(title)}</text>')
            for tick in range(5):
                y: float = top + height * tick / 4
                value: float = high - (high - low) * tick / 4
                parts.append(f'<line x1="{left}" x2="{left + width}" y1="{y}" y2="{y}" stroke="#e5eaf0"/><text x="{left - 7}" y="{y + 4}" text-anchor="end">{value:.4f}</text>')
            parts.append(f'<path d="M {left} {top} V {top + height} H {left + width}" fill="none" stroke="#607482"/>')
            for tick in sorted(set((xs[0], xs[len(xs) // 2], xs[-1]))):
                x: float = left + width * (tick - x_min) / (x_max - x_min)
                parts.append(f'<text x="{x}" y="325" text-anchor="middle">{tick:g}</text>')
            parts.append(f'<text x="{left + width / 2}" y="350" text-anchor="middle">{"Epoch" if kind == "epoch" else "Optimizer update"}</text>')
            for index, (label, color, values) in enumerate(series):
                points: list[tuple[float, float]] = [(left + width * (x - x_min) / (x_max - x_min), top + height * (high - value) / (high - low)) for x, value in zip(xs, values)]
                point_string: str = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
                parts.append(f'<polyline points="{point_string}" fill="none" stroke="{color}" stroke-width="2"/>')
                for (x, y), value in zip(points, values):
                    parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"><title>{label}: {value:.6f}</title></circle>')
                parts.append(f'<text x="{left + index * 125}" y="45" fill="{color}">{label}</text>')
        parts.append('</g></svg>')
        target: Path = output_dir / f"{name}.svg"
        target.write_text("\n".join(parts), encoding="utf-8")
        saved.append(str(target))
    return saved


def plot_history(log_path: str | Path, output_dir: str | Path | None = None) -> list[str]:
    """输出训练/验证损失 PNG 和 PDF；无图形界面的服务器使用 Agg。"""
    source: Path = Path(log_path)
    epoch_rows: list[dict[str, Any]] = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line)["type"] == "epoch"]
    with (source.parent / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        columns: list[str] = ["epoch", "step", "train_qa_loss", "val_qa_loss", "train_retrieval_loss", "val_retrieval_loss", "train_total_loss", "val_total_loss"]
        writer: csv.DictWriter = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in epoch_rows:
            writer.writerow({"epoch": row["epoch"] + 1, "step": row["step"], **{f"{split}_{key}": row[split][key] for key in ("qa_loss", "retrieval_loss", "total_loss") for split in ("train", "val")}})
    try:
        import matplotlib
    except ImportError:
        source: Path = Path(log_path)
        return plot_svg(source, Path(output_dir) if output_dir else source.parent / "curves")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source: Path = Path(log_path)
    destination: Path = Path(output_dir) if output_dir else source.parent / "curves"
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    epochs: list[dict[str, Any]] = [row for row in rows if row["type"] == "epoch"]
    steps: list[dict[str, Any]] = [row for row in rows if row["type"] == "train_step"]
    saved: list[str] = []
    for name, records, x_name, prefix in (("epoch_losses", epochs, "epoch", True), ("step_losses", steps, "step", False)):
        if not records:
            continue
        figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for axis, key, title in zip(axes, ("qa_loss", "retrieval_loss", "total_loss"), ("Answer cross-entropy", "Batch contrastive loss", "Weighted objective")):
            x: list[int] = [row[x_name] + (1 if x_name == "epoch" else 0) for row in records]
            if prefix:
                axis.plot(x, [row["train"][key] for row in records], marker="o", label="Train")
                axis.plot(x, [row["val"][key] for row in records], marker="o", label="Validation")
            else:
                axis.plot(x, [row[key] for row in records], label="Train (logged window)")
            axis.set(title=title, xlabel="Epoch" if prefix else "Optimizer update", ylabel="Loss")
            axis.grid(alpha=0.25)
            axis.legend()
        for suffix in ("png", "pdf"):
            target: Path = destination / f"{name}.{suffix}"
            figure.savefig(target, dpi=160)
            saved.append(str(target))
        plt.close(figure)
    return saved


def main() -> None:
    """绘图可独立重跑，不需要加载音频、模型或 CUDA。"""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--output")
    args: argparse.Namespace = parser.parse_args()
    print(json.dumps(plot_history(args.log, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
