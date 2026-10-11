"""CNN + Transformer + Qwen：问答与检索两条分支，共享音频表示。"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from music_train.config import AudioConfig, Config


class ByteTokenizer:
    """仅用于 tiny 离线自检的 UTF-8 tokenizer，不是 Qwen tokenizer。"""
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """每个字节映射到固定 ID，避免下载测试词表。"""
        return [value + 3 for value in text.encode("utf-8")]

    def decode(self, values: list[int], skip_special_tokens: bool = True) -> str:
        """tiny 随机输出可能不是完整 UTF-8；仅用于检查生成接口。"""
        return bytes(value - 3 for value in values if 3 <= value < 259).decode("utf-8", errors="replace")


class AudioEncoder(nn.Module):
    """log-mel → 局部卷积 → 时间压缩 → 多片段 Transformer。

    默认随机初始化。只有加载兼容的 init_checkpoint 后，才属于音频编码器微调。
    """
    def __init__(self, config: AudioConfig) -> None:
        """建立无 torchaudio 依赖的 mel 前端与可训练网络。"""
        super().__init__()
        self.config: AudioConfig = config
        frequencies: Tensor = torch.linspace(0, config.sample_rate / 2, config.n_fft // 2 + 1)
        mel_max: float = 2595 * __import__("math").log10(1 + (config.sample_rate / 2) / 700)
        edges: Tensor = 700 * (10 ** (torch.linspace(0, mel_max, config.n_mels + 2) / 2595) - 1)
        left: Tensor = (frequencies[None] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None]).clamp_min(1e-6)
        right: Tensor = (edges[2:, None] - frequencies[None]) / (edges[2:, None] - edges[1:-1, None]).clamp_min(1e-6)
        self.register_buffer("mel_filter", torch.minimum(left, right).clamp_min(0), persistent=True)
        self.register_buffer("window", torch.hann_window(config.n_fft), persistent=True)
        blocks: list[nn.Module] = []
        previous: int = 1
        for channel in config.channels:
            blocks.extend([nn.Conv2d(previous, channel, 3, stride=2, padding=1), nn.GroupNorm(1, channel), nn.GELU()])
            previous = channel
        self.cnn: nn.Sequential = nn.Sequential(*blocks)
        self.time_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, config.tokens_per_segment))
        self.feature_projection: nn.Linear = nn.Linear(previous, config.hidden_size)
        self.position: nn.Parameter = nn.Parameter(torch.randn(1, config.segments * config.tokens_per_segment, config.hidden_size) * 0.02)
        layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(config.hidden_size, config.heads, config.hidden_size * 4, config.dropout, batch_first=True, norm_first=True)
        self.transformer: nn.TransformerEncoder = nn.TransformerEncoder(layer, config.layers, norm=nn.LayerNorm(config.hidden_size), enable_nested_tensor=False)

    def forward(self, waves: Tensor, segment_mask: Tensor) -> tuple[Tensor, Tensor]:
        """返回音频 token 与有效 mask；回答损失可一路回传到 CNN。"""
        batch, segments, length = waves.shape
        if not segment_mask.any(dim=1).all():
            raise ValueError("每首歌至少需要一个有效片段")
        # STFT 在 float32 执行，后续卷积仍由训练器的 autocast 管理。
        with torch.autocast(device_type=waves.device.type, enabled=False):
            spectrum: Tensor = torch.stft(waves.reshape(batch * segments, length).float(), n_fft=self.config.n_fft, hop_length=self.config.hop_length, window=self.window.float(), return_complex=True).abs().square()
            mel: Tensor = torch.matmul(self.mel_filter.float(), spectrum).clamp_min(1e-6).log()
            mel = (mel - mel.mean(dim=(-2, -1), keepdim=True)) / mel.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        features: Tensor = self.time_pool(self.cnn(mel.unsqueeze(1))).squeeze(2).transpose(1, 2)
        hidden: Tensor = self.feature_projection(features).reshape(batch, segments * self.config.tokens_per_segment, self.config.hidden_size)
        valid: Tensor = segment_mask.repeat_interleave(self.config.tokens_per_segment, dim=1)
        hidden = self.transformer(hidden + self.position.to(hidden.dtype), src_key_padding_mask=~valid)
        return hidden.masked_fill(~valid.unsqueeze(-1), 0), valid


def multi_positive_contrastive(audio: Tensor, text: Tensor, song_ids: list[str], temperature: float) -> Tensor:
    """同歌的多条问答全部视为正样本；仅使用当前进程的实际微批负样本。"""
    scores: Tensor = audio.float() @ text.float().T / temperature
    positives: Tensor = torch.tensor([[left == right for right in song_ids] for left in song_ids], device=scores.device, dtype=torch.bool)
    if len(set(song_ids)) < 2:
        return scores.sum() * 0  # 没有负样本时不伪造有效对比监督。
    positive_scores: Tensor = scores.masked_fill(~positives, -torch.inf)
    a_to_t: Tensor = torch.logsumexp(scores, dim=1) - torch.logsumexp(positive_scores, dim=1)
    t_to_a: Tensor = torch.logsumexp(scores, dim=0) - torch.logsumexp(positive_scores, dim=0)
    return (a_to_t.mean() + t_to_a.mean()) / 2


class MusicQAModel(nn.Module):
    """音频问答模型；用于检索的文本分支只读取文本，不接触配对音频。"""
    def __init__(self, config: Config) -> None:
        """从本地载入 Qwen，按 frozen/lora/full 设置参数；不会自动下载。"""
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
        self.config: Config = config
        if config.language.tiny:
            self.tokenizer: Any = ByteTokenizer()
            self.qwen: Any = Qwen3ForCausalLM(Qwen3Config(vocab_size=259, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=2048, pad_token_id=0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=True))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config.language.model_path, local_files_only=True, trust_remote_code=False)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            dtype: torch.dtype = {"no": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[config.training.mixed_precision]
            if config.language.mode == "full":
                dtype = torch.float32  # AMP 管理计算精度；完整训练保留 FP32 主权重和优化器状态。
            self.qwen = AutoModelForCausalLM.from_pretrained(config.language.model_path, local_files_only=True, trust_remote_code=False, torch_dtype=dtype, attn_implementation="sdpa")
        self.qwen.config.use_cache = False
        mode: str = config.language.mode
        if mode == "frozen":
            self.qwen.requires_grad_(False)
        elif mode == "lora":
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as error:
                raise RuntimeError("lora 模式需要服务器安装 peft；本程序不会自动安装") from error
            self.qwen = get_peft_model(self.qwen, LoraConfig(task_type="CAUSAL_LM", r=config.language.lora_rank, lora_alpha=config.language.lora_alpha, lora_dropout=config.language.lora_dropout, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        if config.language.gradient_checkpointing:
            self.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.audio_encoder: AudioEncoder = AudioEncoder(config.audio)
        self.audio_encoder.requires_grad_(config.audio.trainable)
        if config.audio.init_checkpoint:
            state: dict[str, Tensor] = torch.load(config.audio.init_checkpoint, map_location="cpu", weights_only=True)
            self.audio_encoder.load_state_dict(state, strict=True)
        dim: int = self.qwen.config.hidden_size
        self.audio_adapter: nn.Sequential = nn.Sequential(nn.LayerNorm(config.audio.hidden_size), nn.Linear(config.audio.hidden_size, dim), nn.GELU(), nn.Linear(dim, dim))
        self.audio_retrieval: nn.Linear = nn.Linear(config.audio.hidden_size, config.training.retrieval_dim)
        self.text_retrieval: nn.Linear = nn.Linear(dim, config.training.retrieval_dim)

    def train(self, mode: bool = True) -> MusicQAModel:
        """冻结模块保持 eval，防止 dropout 使所谓固定特征不断变化。"""
        super().train(mode)
        if not self.config.audio.trainable:
            self.audio_encoder.eval()
        if self.config.language.mode == "frozen":
            self.qwen.eval()
        return self

    def question_ids(self, question: str) -> list[int]:
        """训练和推理共用完全相同的提示模板；截取问题，保留回答前缀。"""
        prefix: list[int] = self.tokenizer.encode("根据给定音乐音频回答问题。只描述音频能支持的内容。\n问题：", add_special_tokens=False)
        body: list[int] = self.tokenizer.encode(question, add_special_tokens=False)[:self.config.language.question_tokens]
        return prefix + body + self.tokenizer.encode("\n回答：", add_special_tokens=False)

    def language_batch(self, hidden: Tensor, mask: Tensor, questions: list[str], answers: list[str] | None) -> tuple[Tensor, Tensor, Tensor]:
        """逐样本拼接后再补齐，避免问题 padding 插入问题与答案之间。

        音频和问题位置的标签为 -100。HF 模型内部进行因果右移，调用方不要再移位。
        """
        embedding: nn.Module = self.qwen.get_input_embeddings()
        audio_tokens: Tensor = self.audio_adapter(hidden).to(embedding.weight.dtype)
        sequences: list[Tensor] = []
        masks: list[Tensor] = []
        targets: list[Tensor] = []
        for index, question in enumerate(questions):
            qids: list[int] = self.question_ids(question)
            aids: list[int] = [] if answers is None else self.tokenizer.encode(answers[index], add_special_tokens=False)[:self.config.language.answer_tokens - 1] + [self.tokenizer.eos_token_id]
            ids: Tensor = torch.tensor(qids + aids, device=hidden.device, dtype=torch.long)
            sequences.append(torch.cat([audio_tokens[index], embedding(ids)], dim=0))
            masks.append(torch.cat([mask[index], torch.ones(len(ids), device=hidden.device, dtype=torch.bool)]))
            targets.append(torch.tensor([-100] * (hidden.shape[1] + len(qids)) + aids, device=hidden.device, dtype=torch.long))
        length: int = max(sequence.shape[0] for sequence in sequences)
        if length > self.qwen.config.max_position_embeddings:
            raise ValueError("音频与文本 token 总数超过 Qwen 上下文长度")
        return (torch.stack([F.pad(sequence, (0, 0, 0, length - len(sequence))) for sequence in sequences]),
                torch.stack([F.pad(value, (0, length - len(value)), value=False) for value in masks]),
                torch.stack([F.pad(value, (0, length - len(value)), value=-100) for value in targets]))

    def audio_vectors(self, hidden: Tensor, mask: Tensor) -> Tensor:
        """池化原始音频隐藏层，保留问题无关的离线索引能力。"""
        pooled: Tensor = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return F.normalize(self.audio_retrieval(pooled).float(), dim=-1)

    def encode_text(self, texts: list[str]) -> Tensor:
        """读取 Qwen 的文本隐藏状态；不输入音频、不让模型生成浮点数。"""
        device: torch.device = next(self.parameters()).device
        sequences: list[list[int]] = [self.tokenizer.encode("音乐描述或听歌需求：" + text, add_special_tokens=False)[:self.config.language.retrieval_text_tokens - 1] + [self.tokenizer.eos_token_id] for text in texts]
        length: int = max(map(len, sequences))
        ids: Tensor = torch.full((len(texts), length), self.tokenizer.pad_token_id, device=device, dtype=torch.long)
        mask: Tensor = torch.zeros_like(ids)
        for index, sequence in enumerate(sequences):
            ids[index, :len(sequence)] = torch.tensor(sequence, device=device)
            mask[index, :len(sequence)] = 1
        base: Any = self.qwen.get_base_model() if hasattr(self.qwen, "get_base_model") else self.qwen
        hidden: Tensor = base.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
        last: Tensor = hidden[torch.arange(len(texts), device=device), mask.sum(dim=1) - 1]
        return F.normalize(self.text_retrieval(last.to(self.text_retrieval.weight.dtype)).float(), dim=-1)

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """共同优化回答生成与跨模态检索；两个目标的权重在配置里设置。"""
        hidden, valid = self.audio_encoder(batch["waves"], batch["segment_mask"])
        embeddings, mask, labels = self.language_batch(hidden, valid, batch["questions"], batch["answers"])
        positions: Tensor = (mask.long().cumsum(dim=1) - 1).clamp_min(0)
        output: Any = self.qwen(inputs_embeds=embeddings, attention_mask=mask, position_ids=positions, labels=labels, use_cache=False, return_dict=True)
        qa: Tensor = output.loss.float()
        retrieval: Tensor = qa * 0
        if self.config.training.retrieval_weight > 0:
            retrieval = multi_positive_contrastive(self.audio_vectors(hidden, valid), self.encode_text(batch["answers"]), batch["song_ids"], self.config.training.temperature)
        return {"loss": qa + self.config.training.retrieval_weight * retrieval, "qa_loss": qa, "retrieval_loss": retrieval, "answer_tokens": (labels != -100).sum()}

    @torch.inference_mode()
    def describe(self, waves: Tensor, segment_mask: Tensor, question: str, max_new_tokens: int = 192) -> str:
        """单首音频条件生成；使用与训练相同的连接模块和问题模板。"""
        self.eval()
        hidden, valid = self.audio_encoder(waves, segment_mask)
        return self.describe_hidden(hidden, valid, question, max_new_tokens)

    @torch.inference_mode()
    def describe_hidden(self, hidden: Tensor, valid: Tensor, question: str, max_new_tokens: int = 192) -> str:
        """用同一次编码产生的隐藏序列回答，供网页同时检查特征和最终输出。"""
        embeddings, mask, _ = self.language_batch(hidden, valid, [question], None)
        ids: Tensor = self.qwen.generate(inputs_embeds=embeddings, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(ids[0].tolist(), skip_special_tokens=True).strip()
