"""单卡 / Accelerate DDP 训练；每个完整 epoch 保存可恢复检查点。"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from music_train.checkpoint import apply_bundle, atomic_torch_save, save_bundle
from music_train.config import Config, from_dict, load_config
from music_train.data import QADataset, check_disjoint, collate_examples, fingerprint, read_manifest
from music_train.diagnostics import ModuleDiagnostics, write_status
from music_train.model import MusicQAModel
from music_train.plot import plot_history


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """验证集不交给分布式 sampler，避免补齐样本造成验证指标重复计数。"""
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def add_metrics(accumulator: Tensor, output: dict[str, Tensor], batch_size: int) -> None:
    """问答按有效回答 token 加权，检索按样本加权，累计值不保留计算图。"""
    tokens: Tensor = output["answer_tokens"].detach()
    accumulator[0] += output["qa_loss"].detach() * tokens
    accumulator[1] += tokens
    accumulator[2] += output["retrieval_loss"].detach() * batch_size
    accumulator[3] += batch_size


def metrics_from(values: Tensor, weight: float) -> dict[str, float]:
    """训练/验证使用相同汇总口径；总损失是两个汇总损失的加权和。"""
    numbers: list[float] = values.detach().double().cpu().tolist()
    qa: float = numbers[0] / max(numbers[1], 1)
    retrieval: float = numbers[2] / max(numbers[3], 1)
    return {"qa_loss": qa, "retrieval_loss": retrieval, "total_loss": qa + weight * retrieval, "answer_tokens": numbers[1], "examples": numbers[3]}


@torch.no_grad()
def validate(model: MusicQAModel, loader: DataLoader, accelerator: Accelerator) -> dict[str, float]:
    """仅主进程完整验证；其他进程等待，不重复或截断验证样本。"""
    model.eval()
    totals: Tensor = torch.zeros(4, device=accelerator.device, dtype=torch.float64)
    for batch in loader:
        batch = move_batch(batch, accelerator.device)
        with accelerator.autocast():
            output: dict[str, Tensor] = model(batch)
        if not torch.isfinite(output["loss"]):
            raise RuntimeError("验证损失出现 NaN/Inf")
        add_metrics(totals, output, len(batch["song_ids"]))
    return metrics_from(totals, model.config.training.retrieval_weight)


def write_event(path: Path, event: dict[str, Any]) -> None:
    """逐行落盘，进程异常退出时仍保留已完成更新的记录。"""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def save_rng(path: Path, accelerator: Accelerator) -> None:
    """各 rank 分别保存随机数状态；epoch 恢复不依赖 worker 的随机序列。"""
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if accelerator.device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(accelerator.device)
    atomic_torch_save(state, path)


def restore_rng(path: Path, accelerator: Accelerator) -> None:
    """只加载本工程生成且可信的训练状态；这些状态含 Python 对象。"""
    state: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if accelerator.device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], accelerator.device)


def run(config: Config, resume: Path | None = None, initialize: Path | None = None, cpu: bool = False, stop_after_epoch: int | None = None) -> None:
    """恢复从下一个完整 epoch 开始；改变训练计划时应另起目录 warm-start。"""
    t = config.training
    accelerator: Accelerator = Accelerator(cpu=cpu, mixed_precision=t.mixed_precision, gradient_accumulation_steps=t.accumulation_steps, kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    if accelerator.distributed_type.name not in {"NO", "MULTI_GPU", "MULTI_CPU"}:
        raise ValueError("本工程支持普通单卡与 DDP；未实现 FSDP/DeepSpeed 分片检查点")
    if cpu and t.mixed_precision != "no":
        raise ValueError("CPU 自检请使用 mixed_precision: no")
    set_seed(t.seed)
    if config.language.tiny:
        torch.set_num_threads(2)
    output_dir: Path = Path(config.output_dir)
    log: Path = output_dir / "metrics.jsonl"
    if log.exists() and resume is None:
        raise ValueError("输出目录已有训练日志；指定 --resume 或使用新的 output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    training_rows = read_manifest(config.train_manifest, config.audio_root)
    validation_rows = read_manifest(config.val_manifest, config.audio_root)
    check_disjoint(training_rows, validation_rows)
    signatures: dict[str, str] = {"train": fingerprint(training_rows), "val": fingerprint(validation_rows)}
    train_set: QADataset = QADataset(training_rows, config.audio_root, config.audio, True, t.seed, t.one_question_per_song)
    val_set: QADataset = QADataset(validation_rows, config.audio_root, config.audio, False, t.seed)
    generator: torch.Generator = torch.Generator()
    train_loader: DataLoader = DataLoader(train_set, batch_size=t.batch_size, shuffle=True, generator=generator, num_workers=t.num_workers, collate_fn=collate_examples, pin_memory=accelerator.device.type == "cuda", persistent_workers=False)
    val_loader: DataLoader = DataLoader(val_set, batch_size=t.eval_batch_size, shuffle=False, num_workers=t.num_workers, collate_fn=collate_examples)
    model: MusicQAModel = MusicQAModel(config)
    groups: dict[str, dict[str, Any]] = {"audio": {"params": [], "lr": t.audio_learning_rate}, "language": {"params": [], "lr": t.language_learning_rate}, "heads": {"params": [], "lr": t.learning_rate}}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group: str = "audio" if name.startswith("audio_encoder.") else "language" if name.startswith("qwen.") else "heads"
            groups[group]["params"].append(parameter)
    optimizer: AdamW = AdamW([group for group in groups.values() if group["params"]], weight_decay=t.weight_decay)
    if resume:
        if not (resume / "READY").exists():
            raise ValueError("检查点未完整保存：缺少 READY")
        saved_config: dict[str, Any] = from_dict(json.loads((resume / "config.json").read_text(encoding="utf-8"))).to_dict()
        if any(saved_config[key] != config.to_dict()[key] for key in ("audio", "language", "training")):
            raise ValueError("恢复时模型/训练配置必须一致；修改参数请用 --initialize-from 开启新实验")
        apply_bundle(model, resume)
    elif initialize:
        apply_bundle(model, initialize, allow_missing_language=True)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    raw: MusicQAModel = accelerator.unwrap_model(model)
    diagnostics: ModuleDiagnostics = ModuleDiagnostics({"cnn": raw.audio_encoder.cnn, "transformer": raw.audio_encoder.transformer})
    batches: int = min(len(train_loader), t.max_batches_per_epoch or len(train_loader))
    # 为了保证累积组尾部正确同步，测试截断必须与真实 loader 长度一致。
    if t.max_batches_per_epoch and t.max_batches_per_epoch < len(train_loader):
        raise ValueError("请通过缩小测试清单限制自检规模；本实现不在累积窗口内截断 DataLoader")
    updates_per_epoch: int = math.ceil(batches / t.accumulation_steps)
    total_updates: int = updates_per_epoch * t.epochs
    warmup: int = int(total_updates * t.warmup_ratio)

    def schedule(step: int) -> float:
        """先线性预热，再余弦下降；按真正的优化器更新次数计数。"""
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress: float = (step - warmup) / max(total_updates - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))

    scheduler: LambdaLR = LambdaLR(optimizer, schedule)
    start_epoch, global_step, best = 0, 0, float("inf")
    retrieval_trained: bool = bool(initialize and json.loads((initialize / "metadata.json").read_text(encoding="utf-8")).get("retrieval_trained", False))
    if resume:
        state: dict[str, Any] = torch.load(resume / "trainer.pt", map_location="cpu", weights_only=False)
        if state["signatures"] != signatures or state["world_size"] != accelerator.num_processes:
            raise ValueError("数据清单或 DDP 进程数改变，不能作为同一次训练恢复")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if accelerator.scaler is not None and state.get("scaler") is not None:
            accelerator.scaler.load_state_dict(state["scaler"])
        start_epoch, global_step, best = state["next_epoch"], state["global_step"], state["best_val_qa"]
        retrieval_trained = json.loads((resume / "metadata.json").read_text(encoding="utf-8")).get("retrieval_trained", False)
        restore_rng(resume / f"rng_rank_{accelerator.process_index}.pt", accelerator)
        if accelerator.is_main_process and log.exists():
            retained: list[str] = [line for line in log.read_text(encoding="utf-8").splitlines() if json.loads(line)["epoch"] < start_epoch]
            log.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")
    if accelerator.is_main_process:
        write_status(output_dir, "starting", epoch=start_epoch, epochs=t.epochs, step=global_step, total_updates=total_updates, tiny_test_only=config.language.tiny, error=None)
        (output_dir / "resolved_config.json").write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        raw_model: MusicQAModel = accelerator.unwrap_model(model)
        print(json.dumps({"parameters": sum(p.numel() for p in raw_model.parameters()), "trainable": sum(p.numel() for p in raw_model.parameters() if p.requires_grad), "train_songs": len({r['song_id'] for r in training_rows}), "val_songs": len({r['song_id'] for r in validation_rows}), "world_size": accelerator.num_processes, "tiny_test_only": config.language.tiny}, ensure_ascii=False), flush=True)
    accelerator.wait_for_everyone()
    for epoch in range(start_epoch, t.epochs):
        if accelerator.is_main_process:
            write_status(output_dir, "training", epoch=epoch + 1, batch=0, batches=batches)
        started: float = time.perf_counter()
        train_set.epoch = epoch
        generator.manual_seed(t.seed + epoch)
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)
        model.train()
        epoch_values: Tensor = torch.zeros(4, device=accelerator.device, dtype=torch.float64)
        window_values: Tensor = torch.zeros_like(epoch_values)
        window_has_negatives: bool = False
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    values: dict[str, Tensor] = model(batch)
                valid_loss: Tensor = accelerator.reduce(torch.isfinite(values["loss"]).long(), reduction="sum")
                if valid_loss.item() != accelerator.num_processes:
                    raise RuntimeError("训练损失出现 NaN/Inf，停止以免保存损坏权重")
                # Accelerator 默认除以累积步数；最后不足一组时补偿，避免缩小梯度。
                window_size: int = min(t.accumulation_steps, batches - (batch_index // t.accumulation_steps) * t.accumulation_steps)
                accelerator.backward(values["loss"] * t.accumulation_steps / window_size)
                window_has_negatives |= t.retrieval_weight > 0 and len(set(batch["song_ids"])) >= 2
                collect: bool = bool(accelerator.sync_gradients and accelerator.is_main_process and t.diagnostics_every_updates and ((global_step + 1) % t.diagnostics_every_updates == 0 or batch_index == batches - 1))
                if accelerator.sync_gradients:
                    # 必须先反缩放再测量；随后直接裁剪，避免 scaler 被重复 unscale。
                    accelerator.unscale_gradients(optimizer)
                    if collect:
                        diagnostics.before_step()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
                    if collect:
                        diagnostics.after_clip()
                optimizer.step()
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                    global_step += 1
                    negative_count: Tensor = accelerator.reduce(torch.tensor(int(window_has_negatives), device=accelerator.device), reduction="sum")
                    retrieval_trained |= negative_count.item() > 0
                if collect:
                    diagnostic_event: dict[str, Any] = {"type": "module_diagnostics", "epoch": epoch, "step": global_step, "timestamp": time.time(), "update_applied": not accelerator.optimizer_step_was_skipped, "modules": diagnostics.after_step()}
                    write_event(log, diagnostic_event)
                    print(json.dumps(diagnostic_event, ensure_ascii=False, allow_nan=False), flush=True)
                if accelerator.sync_gradients:
                    window_has_negatives = False
                    if accelerator.is_main_process:
                        write_status(output_dir, "training", batch=batch_index + 1, step=global_step)
                optimizer.zero_grad(set_to_none=True)
            add_metrics(epoch_values, values, len(batch["song_ids"]))
            add_metrics(window_values, values, len(batch["song_ids"]))
            if accelerator.sync_gradients and (global_step % t.log_every_updates == 0 or batch_index == batches - 1):
                merged: Tensor = accelerator.reduce(window_values, reduction="sum")
                if accelerator.is_main_process:
                    event: dict[str, Any] = {"type": "train_step", "epoch": epoch, "step": global_step, "timestamp": time.time(), **metrics_from(merged, t.retrieval_weight), "learning_rates": scheduler.get_last_lr()}
                    write_event(log, event)
                    print(json.dumps(event, ensure_ascii=False), flush=True)
                window_values.zero_()
        train_metrics: dict[str, float] = metrics_from(accelerator.reduce(epoch_values, reduction="sum"), t.retrieval_weight)
        accelerator.wait_for_everyone()
        checkpoint_dir: Path = output_dir / "checkpoints" / f"epoch-{epoch + 1:04d}"
        if accelerator.is_main_process:
            write_status(output_dir, "validating", batch=batches)
            raw_model = accelerator.unwrap_model(model)
            val_metrics: dict[str, float] = validate(raw_model, val_loader, accelerator)
            is_best: bool = val_metrics["qa_loss"] < best
            best = min(best, val_metrics["qa_loss"])
            event = {"type": "epoch", "epoch": epoch, "step": global_step, "train": train_metrics, "val": val_metrics, "seconds": time.perf_counter() - started}
            write_event(log, event)
            metadata: dict[str, Any] = {"bundle_id": uuid.uuid4().hex, "epoch": epoch + 1, "global_step": global_step, "tiny_test_only": config.language.tiny, "retrieval_trained": retrieval_trained, "val": val_metrics, "signatures": signatures, "world_size": accelerator.num_processes}
            write_status(output_dir, "saving")
            save_bundle(raw_model, checkpoint_dir, metadata)
            atomic_torch_save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": accelerator.scaler.state_dict() if accelerator.scaler is not None else None, "next_epoch": epoch + 1, "global_step": global_step, "best_val_qa": best, "signatures": signatures, "world_size": accelerator.num_processes}, checkpoint_dir / "trainer.pt")
            print(json.dumps(event, ensure_ascii=False), flush=True)
        accelerator.wait_for_everyone()
        save_rng(checkpoint_dir / f"rng_rank_{accelerator.process_index}.pt", accelerator)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            (checkpoint_dir / "READY").write_text("complete\n", encoding="utf-8")
            pointer: dict[str, str] = {"checkpoint": str(checkpoint_dir.resolve())}
            (output_dir / "last.json").write_text(json.dumps(pointer), encoding="utf-8")
            if is_best:
                (output_dir / "best.json").write_text(json.dumps(pointer), encoding="utf-8")
            try:
                plot_history(log)
            except ImportError:
                print("未安装 matplotlib，已保留真实日志；安装后可单独运行 music_train.plot。", flush=True)
        accelerator.wait_for_everyone()
        if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch:
            if accelerator.is_main_process:
                write_status(output_dir, "stopped" if epoch + 1 < t.epochs else "completed")
            break
    else:
        if accelerator.is_main_process:
            write_status(output_dir, "completed")
    accelerator.end_training()


def main() -> None:
    """服务器入口；--stop-after-epoch 用来演练中断恢复，不改变学习率计划。"""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    args: argparse.Namespace = parser.parse_args()
    if args.resume and args.initialize_from:
        raise SystemExit("--resume 与 --initialize-from 不能同时使用")
    config: Config = load_config(args.config)
    try:
        run(config, args.resume, args.initialize_from, args.cpu, args.stop_after_epoch)
    except BaseException as error:
        state: Path = Path(config.output_dir) / "run_state.json"
        # 不覆盖其他进程或尚未由本进程启动的实验状态。
        if os.environ.get("RANK", "0") == "0" and state.exists() and json.loads(state.read_text(encoding="utf-8")).get("pid") == os.getpid():
            write_status(Path(config.output_dir), "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
