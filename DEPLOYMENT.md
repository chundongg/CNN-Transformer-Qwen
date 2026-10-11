# Linux 服务器训练与部署步骤

本说明使用 Python 3.11/3.12、单机单卡或单机多卡的 Linux 环境。路径只是服务器布局示例，按实际情况修改。代码只从本地读取模型和数据，不会自动下载 Qwen 或音频。

新增实时训练网页和测试歌曲交互检查，完整步骤见 [WEB_GUIDE.md](WEB_GUIDE.md)。训练日志现在默认记录 CNN/Transformer 的梯度、裁剪后梯度及真实参数更新量。

## 1. 上传项目、数据、权重

建议目录：

```text
/srv/music/
├── music_audio_training/       # 本工程源代码
├── MuChin/                    # 服务器完整数据；不要求名字为 MuChin1k
│   ├── MuChin-main/dataset/
│   │   ├── qidmap.json
│   │   ├── subqidmap.json
│   │   └── bm-data/<歌曲ID>/prof_desc_json
│   └── muchin-audio/<歌曲ID>.mp3
└── Qwen3-4B-Base/              # 完整本地模型、tokenizer、配置与权重分片
```

上传代码、配置、部署脚本和测试即可；本地 `outputs/`、缓存、Windows 虚拟环境不需要上传。`data/manifests/` 当前对应本地 1000 首子集，建议在服务器生成新的 `data/manifests_full/`。

本次提供的源码 ZIP 已按白名单排除这些文件。解压后工程内有 `MuChin/` 与 `Qwen3-4B-Base/` 占位文件夹，默认配置指向工程内目录；你可以填入文件后使用，也可以像上述布局一样放在工程外，按第 5 节修改路径。源码 ZIP 中的 `data/` 只有占位，需重新转换清单。

如果你的完整 MuChin 与这个结构不同，按第 4 节准备统一 JSONL。没有假设 6066 首的发布版本必然与本地子集结构完全一致。

## 2. 准备服务器依赖

进入工程并建立独立环境：

```bash
cd /srv/music/music_audio_training
python3 -m venv .venv
source .venv/bin/activate
```

先按照服务器 GPU 驱动与 CUDA 环境准备适配的 CUDA 版 PyTorch，再检查：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

已经装好依赖时直接检查即可，不必重复安装。如果服务器缺少依赖，可按你的环境管理方式使用 [requirements.txt](requirements.txt)。联网服务器的命令是：

```bash
python -m pip install -r requirements.txt
python -m pip check
```

离线服务器需先自行备好匹配平台、Python 和 CUDA 的 wheel 包，再使用：

```bash
python -m pip install --no-index --find-links /srv/wheels -r requirements.txt
python -m pip check
```

这些是供服务器执行的步骤，本次没有在你的电脑下载或安装依赖。LoRA 需要 `peft`；真实 MP3/FLAC 解码需要 `soundfile`，重采样需要 `scipy`；PNG/PDF 曲线需要 `matplotlib`。若 soundfile 提示 libsndfile 缺失或不支持 MP3，应在服务器补齐对应系统库/解码支持，然后重新预检。

默认使用 BF16。GPU 不支持 BF16 时将配置改为 `fp16` 或 `"no"`，再预检。代码支持 AMP，但本地验证使用 CPU FP32，并未实测服务器 BF16/FP16 性能。

不要直接沿用 Windows 的 `.venv`。服务器需保持权重和 tokenizer 匹配；训练和推理尽量使用同版本依赖。

## 3. 先运行无需下载的管线自检

```bash
python -m pytest
python -m scripts.make_smoke_data --output outputs/server_smoke_data --run-dir outputs/server_smoke_run
python -m music_train.train --config outputs/server_smoke_data/config.json --cpu --stop-after-epoch 1
python -m music_train.train --config outputs/server_smoke_data/config.json --cpu --resume outputs/server_smoke_run/checkpoints/epoch-0001
```

预期得到 3 轮日志、检查点和曲线。自检使用 8 个合成信号与随机 tiny Qwen3，不加载 4B 权重，不判断音乐质量。命令拒绝覆盖已有自检数据；再次执行请换输出目录。

这里的 `--stop-after-epoch 1` 只在第一轮后停止，配置里的总训练计划仍为 3 轮，适合演练续训。正式训练不需要这个参数。

## 4. 转换完整 MuChin 并划分数据

```bash
python -m music_train.prepare \
  --dataset /srv/music/MuChin \
  --audio-root /srv/music/MuChin/muchin-audio \
  --output data/manifests_full \
  --sources professional \
  --seed 42
```

检查 `summary.json` 的歌曲数、问答数和 `warnings`。转换过程只读取音频文件路径，不代表已经解码并验证了所有音频。存在缺失音频或标注错误时，会报告警告并跳过对应数据，需要先核对原因。

默认划分为 80% 训练、10% 验证、10% 测试，比例按独立分组计算。可用 `--sources both` 加入业余标注，同歌曲完全重复的问答会去重。

如果知道哪些 ID 是同一录音、不同版本或需要按艺人整体隔离，建立 JSON 映射，例如：

```json
{"song_001":"recording_A","song_002":"recording_A","song_003":"recording_B"}
```

加 `--group-map /srv/music/groups.json` 后，同组歌曲会进入相同集合。没有提供映射的歌曲以自己的 ID 分组。按艺人分组时，将值设为稳定艺人 ID，并事先决定合作歌曲如何归组。

若原始格式不同，直接生成 `train.jsonl/val.jsonl/test.jsonl`，每行包含 `id、song_id、audio、question、answer`，可选 `group_id、source`，格式见 README。音频相对路径必须在 `audio_root` 内；同一录音的所有样本要先分组再划分。已有转换输出默认拒绝覆盖，明确需要重做时才加 `--overwrite`。

## 5. 修改训练配置

编辑 [configs/server.yaml](configs/server.yaml) 的路径：

```yaml
audio_root: /srv/music/MuChin/muchin-audio
train_manifest: ../data/manifests_full/train.jsonl
val_manifest: ../data/manifests_full/val.jsonl
output_dir: ../outputs/muchin_joint_v1
```

在现有 `language` 段内设置：

```yaml
model_path: /srv/music/Qwen3-4B-Base
mode: lora
tiny: false
gradient_checkpointing: true
```

这些代码块是对已有字段的修改，不是完整配置文件。**相对路径以 YAML 文件所在目录为基准**，不是以启动命令所在目录为基准。

主要参数：

| 参数 | 初始值 | 调整作用 |
| --- | --- | --- |
| `audio.segments × segment_seconds` | 3 × 8 秒 | 增大整曲覆盖，增加音频计算 |
| `audio.hidden_size/layers` | 384 / 4 | 调整音频 Transformer 容量 |
| `audio.tokens_per_segment` | 16 | 时间细节与 Qwen 输入长度的取舍 |
| `training.batch_size` | 4 | 每个 GPU 的真实微批，也决定本卡检索负样本数 |
| `training.accumulation_steps` | 4 | 降低每次更新的峰值批量压力，不增加负样本池 |
| `training.epochs` | 10 | 初始实验计划，不能视为保证收敛的轮数 |
| `training.audio_learning_rate` | 1e-4 | CNN 与 Transformer 的学习率 |
| `training.language_learning_rate` | 2e-5 | Qwen LoRA/全参学习率 |
| `training.learning_rate` | 2e-4 | 音频连接模块和两侧检索层学习率 |
| `training.retrieval_weight` | 0.1 | 问答和检索目标的相对权重 |
| `training.num_workers` | 4 | 音频读取进程数；排查数据问题可设为 0 |
| `training.diagnostics_every_updates` | 1 | 每次更新记录音频模块梯度/真实权重差；0 关闭，较大值减少开销 |

默认有效优化批量约为 `4 × 4 × GPU 数`，但负样本仍局限于每卡当前 4 条样本。最后不足一个累积窗口时按实际窗口大小补偿。DDP 可能补齐最后的训练批次；验证只在主进程跑完整集合，避免补齐导致重复计数。

全参数训练请设 `mode: full`。该模式保留 FP32 主权重和优化器状态，显存/内存开销明显高于 LoRA；普通 DDP 每卡都有完整模型，不能靠增加 GPU 数让单份模型自动跨卡分摊。本项目没有实现 FSDP、DeepSpeed、4-bit QLoRA。

`max_batches_per_epoch` 是内部保留字段，保持默认 0。小规模试跑请另建缩小的清单，或用 `--stop-after-epoch`，不要依赖中途截断 DataLoader。

## 6. 正式训练前预检

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python -m scripts.preflight --config configs/server.yaml --audio-check 10
```

预检检查依赖、基座配置路径、清单字段、音频存在性、训练/验证隔离，并解码至多 10 个不同训练音频。它不加载完整 Qwen，不能代替显存试跑。将 `--audio-check` 设为大于训练歌曲数可检查全部训练音频；验证/测试音频仍需在相应流程验证。

缺少依赖时程序明确退出，不自动联网安装。初次训练建议先进行一轮试跑，确认输入、损失和保存，再运行完整计划。

## 7. 单卡或多卡训练

单卡：

```bash
bash deploy/train.sh configs/server.yaml
```

同一服务器 2 张 GPU：

```bash
GPU_COUNT=2 bash deploy/train_ddp.sh configs/server.yaml
```

指定可见卡：

```bash
CUDA_VISIBLE_DEVICES=0,1 GPU_COUNT=2 bash deploy/train_ddp.sh configs/server.yaml
```

多卡脚本从 YAML 读取精度，启动普通 Accelerate DDP。这里未实测多卡通信，首次使用请先做短试跑。单卡使用 `train.sh`，不要把多卡脚本的卡数设为 1。

终端会输出参数量、可训练参数量、歌曲数、损失等。可在服务器的终端会话管理器内运行，或由已有作业调度器启动；计算资源申请取决于服务器平台。

训练每个完整 epoch 后验证并保存。所有 epoch 都保留，LoRA 检查点省略冻结基座；全参数模式的权重与优化器状态可能每轮占用很大磁盘空间，应按试跑实际文件大小规划存储。

## 8. 查看损失曲线

```bash
python -m music_train.plot --log outputs/muchin_joint_v1/metrics.jsonl
```

训练每轮会自动调用绘图，上述命令用于重新生成。产物包括：

```text
outputs/muchin_joint_v1/
├── epoch_metrics.csv
└── curves/
    ├── epoch_losses.png
    ├── epoch_losses.pdf
    ├── step_losses.png
    └── step_losses.pdf
```

若缺少 matplotlib，将输出 `epoch_losses.svg/step_losses.svg`，不会编造替代损失。图中分别显示问答交叉熵、检索对比损失和加权总损失。服务器无桌面也能生成图片。

## 9. 断点续训与新实验初始化

同一次训练中断后，从最近包含 `READY` 的检查点恢复：

```bash
bash deploy/train.sh configs/server.yaml \
  --resume outputs/muchin_joint_v1/checkpoints/epoch-0003
```

多卡恢复同理：

```bash
GPU_COUNT=2 bash deploy/train_ddp.sh configs/server.yaml \
  --resume outputs/muchin_joint_v1/checkpoints/epoch-0003
```

恢复点包含模型、优化器、学习率状态、AMP 缩放器及各 rank 随机数状态。只从**下一个完整 epoch**继续；中断所在 epoch 的未保存进度会重跑。模型和训练配置、清单内容、DDP 进程数必须相同；总轮数也必须与原计划相同。

原始音频内容和 Qwen 基座要保持一致。清单签名只覆盖清单内容，没有给每个音频和 4B 基座做全文件哈希。移动服务器后可改顶层数据根路径，但基座路径或 `audio.init_checkpoint` 改动也会触发严格配置检查，最省事的是保持同样的服务器目录布局。

通常只恢复最近完成的检查点。若从更早的 epoch 回退，应备份原实验目录；后续同名 epoch 会被重新训练并覆盖，日志会回退到恢复点。不要让两个进程组同时使用一个 `output_dir`。

改变学习率、训练轮数或训练模式，应新建配置与输出目录，并使用：

```bash
bash deploy/train.sh configs/new_experiment.yaml \
  --initialize-from outputs/muchin_joint_v1/checkpoints/epoch-0010
```

这只初始化模型权重，重新建立优化器和学习率计划，不等同于精确续训。支持同结构初始化；冻结 Qwen 的包可用于同基座的 LoRA 阶段。**LoRA 权重的模块名称与全参 Qwen 不同，不能把 LoRA 检查点直接当成全参检查点加载。** 改音频维度、层数或检索维度也需要显式适配权重。

## 10. 选择检查点并生成描述

训练结束后读取 `best.json`：

```bash
CKPT=$(python -c 'import json; print(json.load(open("outputs/muchin_joint_v1/best.json"))["checkpoint"])')
python -m music_train.infer describe \
  --checkpoint "$CKPT" \
  --base-model /srv/music/Qwen3-4B-Base \
  --audio /srv/music/example.mp3 \
  --question "请描述这首音乐的人声、配器、节奏和整体氛围。"
```

`best.json` 存的是训练机器的绝对路径；迁移检查点后直接把 `CKPT` 设为新的 epoch 目录。它选择的是验证问答损失最佳模型。检索实验应在验证集上比较配对召回指标后再选择模型。

描述生成会使用与训练一致的音频采样和问题模板。新问题可以输入，但超出标注覆盖的知识不能保证可靠；不能从合成自检模型获得有意义的音乐描述。

## 11. 建立音频索引与文本召回

只有实际经历检索对齐更新的检查点才能建库。先对验证集建库、评测用于选择模型，以下以最终测试集为示例：

```bash
python -m music_train.infer index \
  --checkpoint "$CKPT" \
  --base-model /srv/music/Qwen3-4B-Base \
  --manifest data/manifests_full/test.jsonl \
  --audio-root /srv/music/MuChin/muchin-audio \
  --output outputs/muchin_joint_v1/test_index.pt

python -m music_train.infer search \
  --checkpoint "$CKPT" \
  --base-model /srv/music/Qwen3-4B-Base \
  --index outputs/muchin_joint_v1/test_index.pt \
  --query "想听节奏舒缓、柔和人声的歌曲" \
  --top-k 10

python -m music_train.infer evaluate \
  --checkpoint "$CKPT" \
  --base-model /srv/music/Qwen3-4B-Base \
  --index outputs/muchin_joint_v1/test_index.pt \
  --manifest data/manifests_full/test.jsonl \
  --audio-root /srv/music/MuChin/muchin-audio \
  --output outputs/muchin_joint_v1/test_recall.json
```

每首歌只建一个音频向量；6066 首规模使用精确余弦检索。索引记录检查点 `bundle_id`，换模型后必须重建，避免新旧向量混用。`score` 是向量相似度，不是喜欢概率。

`evaluate` 使用留出集人工回答检索其配对歌曲，按问答条数计算 Recall@1/5/10。候选池大小影响结果，报告时需注明索引包含哪些歌曲；不同歌曲也可能语义相关，因此这不是完整的用户需求相关性评估。

用于实际曲库时，可以合并清单再建库：

```bash
cat data/manifests_full/train.jsonl data/manifests_full/val.jsonl data/manifests_full/test.jsonl > data/manifests_full/catalog.jsonl
python -m music_train.infer index \
  --checkpoint "$CKPT" \
  --base-model /srv/music/Qwen3-4B-Base \
  --manifest data/manifests_full/catalog.jsonl \
  --audio-root /srv/music/MuChin/muchin-audio \
  --output outputs/muchin_joint_v1/catalog_index.pt
```

这只是推理索引，不应把合并清单重新用作训练数据后仍报告原测试集独立效果。

## 12. 部署为 HTTP 服务

推理机器需要源代码、依赖、选中的 epoch 目录中的 `model.pt/config.json/metadata.json`，以及匹配的原始 Qwen 基座和 tokenizer。建议复制整个完整 epoch 目录，保留 `READY` 标记；只做推理时不读取 `trainer.pt/rng_rank_*.pt`。LoRA 未合并到基座，加载时由 PEFT 重建，因此仍需要 PEFT 和原基座。

停止占用同一 GPU 的训练后启动：

```bash
export MUSIC_CHECKPOINT="$CKPT"
export MUSIC_BASE_MODEL=/srv/music/Qwen3-4B-Base
export MUSIC_DEVICE=cuda
# 如需请求鉴权，由你设置 MUSIC_API_TOKEN，随后客户端使用同一个值。
bash deploy/serve.sh
```

服务默认监听 `127.0.0.1:8800`，单 worker 只加载一份模型。客户端可使用 SSH 转发：

```bash
ssh -L 8800:127.0.0.1:8800 your_user@your_server
```

然后在本机访问 `http://127.0.0.1:8800/docs`，可查看接口说明。自定义音频上传请求使用原始二进制体，建议用下面的 curl。需要播放器、选测试歌曲提问和中间特征检查时，另外按 [WEB_GUIDE.md](WEB_GUIDE.md) 启动 8810 端口的网页。

健康检查：

```bash
curl http://127.0.0.1:8800/health
```

返回模型是否已加载、设备、版本、是否 tiny 自检模型和检索是否训练。健康接口不要求令牌。

上传音频生成默认问题的回答：

```bash
curl -X POST 'http://127.0.0.1:8800/v1/describe?format=mp3&max_new_tokens=192' \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @/path/to/song.mp3
```

自定义问题时，把 UTF-8 问题做 URL 编码后放入 `question` 查询参数；也可用 Python 标准库调用：

```python
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def describe_audio(audio_path: Path, question: str, token: str = "") -> str:
    """向已有服务上传音频；文件扩展名与实际格式必须一致。"""
    query: str = urlencode({"format": audio_path.suffix.lstrip("."), "question": question})
    headers: dict[str, str] = {"Content-Type": "application/octet-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request: Request = Request(
        f"http://127.0.0.1:8800/v1/describe?{query}",
        data=audio_path.read_bytes(), headers=headers, method="POST",
    )
    with urlopen(request, timeout=300) as response:
        return str(json.load(response)["answer"])


print(describe_audio(Path("song.mp3"), "这首音乐的节奏和人声有什么特点？"))
```

文本与音频向量：

```bash
curl -X POST http://127.0.0.1:8800/v1/embed/text \
  -H 'Content-Type: application/json' \
  -d '{"text":"舒缓、安静、适合放松的音乐"}'

curl -X POST 'http://127.0.0.1:8800/v1/embed/audio?format=mp3' \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @/path/to/song.mp3
```

如果设置了 `MUSIC_API_TOKEN`，以上 POST 请求都加 `-H "Authorization: Bearer ${MUSIC_API_TOKEN}"`，并在客户端环境提供同一令牌。支持 WAV/MP3/FLAC/OGG，上限 50 MiB；文件格式解码依赖服务器 soundfile 能力。临时上传文件在请求结束后清理。

API 返回向量和 `bundle_id`，可以由原推荐系统计算召回与融合；HTTP 服务本身没有内置曲库搜索接口，完整检索使用第 11 节 CLI 或接入已有推荐后端。单进程内的推理串行执行以控制显存，这不是高吞吐推理引擎。

## 13. 常见问题

| 现象 | 检查与处理 |
| --- | --- |
| 找不到模型或尝试解析模型名 | 填写真实本地目录；检查配置、tokenizer、权重分片是否完整 |
| MP3 读取失败 | 用预检定位文件；检查 soundfile/libsndfile MP3 支持和文件完整性 |
| CUDA OOM | 先减微批或 token 长度；保留梯度检查点；LoRA 优先于全参。检索微批仍需至少 2 |
| Windows 复制的 shell 文件不能启动 | 使用 `bash deploy/train.sh`；文件需保持 LF 换行 |
| 恢复提示配置或清单不一致 | 恢复原配置，或新输出目录使用 `--initialize-from`；不要删除检查逻辑硬续训 |
| 向量接口拒绝调用 | 当前模型没有经历有效检索更新，需训练联合目标 |
| 索引版本不匹配 | 使用当前检查点重新运行 `index` |
| 描述流畅但与音乐无关 | 做静音/错配对照，检查数据与采样覆盖，评估音频依赖性；不能仅看语言损失 |
| 训练损失降、验证损失升 | 检查过拟合、题型和分组，降低容量/学习率或选择较早检查点 |

训练结束后保留配置、清单、日志和环境版本，便于复现实验。完整训练与真实召回结论需要你在服务器执行并评估；本地检查结果见 [CHECK_REPORT.md](CHECK_REPORT.md)。
