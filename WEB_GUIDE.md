# 实时训练网页与音乐检查

网页包含两个页面：**训练监控**查看损失、CNN/Transformer 梯度和真实参数更新；**音乐检查**选择测试集歌曲与完整检查点，播放音乐、提问、查看中间特征和 Qwen 回答。前端资源随源码提供，不使用外部 CDN，不需要 Node.js 构建或安装前端依赖。

训练和网页是两个独立进程。网页不会启动训练、改写训练权重或把测试答案输入模型。监控阶段不加载 Qwen；只有点击“加载检查点”才会在网页进程中分配模型内存。

## 1. 服务器启动

先完成 [DEPLOYMENT.md](DEPLOYMENT.md) 的依赖、数据清单和配置步骤。在工程根目录，两个终端分别启动：

```bash
# 终端 A：正常训练
bash deploy/train.sh configs/server.yaml
```

```bash
# 终端 B：实时网页；路径改为本次实验与实际测试集。
bash deploy/dashboard.sh \
  --run-dir outputs/muchin_joint \
  --test-manifest data/manifests_full/test.jsonl \
  --audio-root MuChin/muchin-audio \
  --base-model Qwen3-4B-Base \
  --device cpu \
  --port 8810
```

`--run-dir` 必须与训练配置解析后的 `output_dir` 指向同一目录。这里的命令行路径相对于当前工作目录；YAML 中的路径相对于 YAML 所在目录，两者不要混淆。

默认地址为 `http://127.0.0.1:8810`。远程服务器可在本机终端建立转发：

```bash
ssh -L 8810:127.0.0.1:8810 your_user@your_server
```

随后在本机浏览器访问该地址。训练尚未产生日志时，网页显示等待；日志出现后自动刷新。

只需要监控时可以不配置测试集：

```bash
python -m music_train.dashboard --run-dir outputs/muchin_joint --port 8810
```

模型检查默认用 CPU，4B 模型在 CPU 上可能很慢。训练完成并释放显存后，可以停止网页服务并用同样命令、`--device cuda` 重新启动；也可以通过 `CUDA_VISIBLE_DEVICES` 把网页放到空闲 GPU。普通单卡服务器不建议同时运行完整训练和另一个 4B 推理副本。

如果配置了 `MUSIC_API_TOKEN`，在网页右上角填写相同令牌并点“连接”。令牌只用于当前页面内存中的请求头，不放进 URL，也不写入浏览器持久存储。保持单 worker，避免重复装载模型和分散下载缓存。

## 2. 训练日志新增内容

`metrics.jsonl` 增加 `type: module_diagnostics` 事件。默认每次优化器更新记录一次；梯度累积中的普通微批不单独算一次更新。

```yaml
training:
  diagnostics_every_updates: 1
```

这是配置片段，修改已有 `training` 段中的字段即可。设为 10 可减少取样频率；非零时每个 epoch 的最后一次更新也会记录。设为 0 完全关闭模块诊断。日志取样会复制一份 CNN/Transformer 参数用于比较，并计算范数，存在额外显存和同步开销；不会快照整个 Qwen。

| 字段 | 意义 | 怎么阅读 |
| --- | --- | --- |
| `grad_l2_before_clip` | AMP 反缩放后、裁剪前的梯度 L2 范数 | 观察梯度是否存在、是否异常增大 |
| `grad_l2_after_clip` | 裁剪后梯度 L2 范数 | 实际交给优化器的梯度规模 |
| `grad_max_abs` | 裁剪前单个梯度元素的最大绝对值 | 辅助定位极端数值 |
| `grad_nonzero_fraction` | 已存在的梯度张量中非零元素比例 | 区分稀疏/大量零梯度情况 |
| `gradient_parameter_fraction` | 有 `.grad` 的可训练参数元素占比 | 检查是否有可训练参数没有进入计算图 |
| `weight_l2_before` | 更新前参数 L2 范数 | 为相对变化提供尺度 |
| `update_l2` | 优化器执行前后参数差的 L2 范数 | 真正的参数更新量 |
| `relative_update` | 参数变化范数 / 原参数范数 | 在不同尺度的模块之间辅助比较 |
| `trainable_parameters` | 当前可训练参数元素数 | 0 表示配置上已经冻结 |
| `grad_finite/parameter_finite` | 是否全部为有限值 | false 需要检查数值异常 |
| `update_applied` | 这次是否真正执行更新 | AMP 溢出跳过时为 false |

相对变化计算为：

\[
\text{relative update}=\frac{\lVert\theta_{after}-\theta_{before}\rVert_2}{\max(\lVert\theta_{before}\rVert_2,10^{-12})}
\]

这里记录的是实际权重差，包含 AdamW 的动量、自适应缩放和权重衰减影响，不能简单用 `学习率 × 梯度范数` 替代。梯度为零时，已有动量或权重衰减也可能让参数变化；有梯度时，零学习率或 AMP 跳过更新也可能使参数不变。

当前逐模块范围：`cnn` 包含卷积及对应 GroupNorm；`transformer` 包含音频 Transformer 编码层及其归一化。音频位置编码、特征投影、Qwen 连接层、检索投影和 Qwen 没有加入这两组诊断统计。它们仍按原配置参与训练，不能把这两组更新量当作整个联合模型的更新量。

普通 DDP 下在梯度同步后，仅主进程测量、写日志；不会将多卡上相同的同步范数重复相加。CPU 小模型已验证数值，CUDA AMP 与多卡路径仍需服务器试跑验证。

## 3. 训练监控页面

页面每 3 秒读取一次新增日志，显示：

- 训练阶段、轮次、微批进度、优化器更新次数、最后更新时间。
- 完整 epoch 的训练/验证问答损失，以及日志窗口的总损失和检索损失。
- CNN 与 Transformer 的裁剪前梯度、实际相对参数更新曲线。
- 最近一次诊断的裁剪前后梯度、更新范数、覆盖率和状态。

梯度和相对更新使用对数坐标；精确的 0 不出现在对数曲线上，但会在表格显示。异常数值写成 JSON `null`，同时将相应 finite 标记设为 false。

网页每类曲线保留最近 3000 条记录，原始日志保留完整数据。它能等待还未写完的 JSONL 半行，并在恢复训练导致日志回退时重新读取。高频诊断不会把 epoch 记录挤掉。

`run_state.json` 是训练主进程写入的状态快照；正常退出、受控停止及可捕获的错误会更新状态。断电或强制杀进程无法写入最终状态，因此页面的“训练中”是最后的记录，必须结合更新时间判断。网页不声称能证明训练进程仍然存活。

旧实验没有模块诊断事件时，梯度图显示等待，不补造历史数据；已有完整旧检查点仍可加载检查。

## 4. 训练完成后的音乐检查

1. 切换到“音乐检查”，选择 `epoch-XXXX` 并加载。只有带 `READY` 的完整检查点会出现在列表。
2. 在左侧搜索、选择测试歌曲，浏览器可以播放原音频。
3. 输入自己的问题，或展开人工参考问答，点“使用这道问题”。这个按钮只复制问题。
4. 点击“生成回答并检查”。同一次音频编码产生的隐藏序列既用于展示，也用于 Qwen 生成。
5. 阅读联合模型回答，对照人工标注，并检查中间张量或下载结果。

网页会在能够访问原训练和验证清单时核对测试集隔离；误用训练集会报错。若迁移后找不到旧清单，会明确提示尚未重新核对，不能把“文件名叫 test”当成已经证明无泄漏。

**人工参考只用于人工比较，不会被拼入 Qwen 提示词。** 网页仅使用用户问题和音频。随机 tiny 模型会明确标记，它可能生成无意义字符，这是管线自检，不能用于音乐质量结论。

## 5. 中间表示展示了什么

| 展示/下载项 | 内容 | 默认真实模型下的形状含义 |
| --- | --- | --- |
| CNN 热力图 | CNN 经频率/时间池化后的特征，尚未进入 Transformer | `片段数 × 每片段 token 数 × CNN 通道数` |
| `cnn_raw` | 完整卷积特征图，保存在 NPZ 中 | `片段数 × 通道数 × 频率格数 × 时间格数` |
| Transformer 热力图 | 完成跨 token 编码、屏蔽无效 token 后的音频隐藏序列 | `总音频 token 数 × hidden_size` |
| 均值、标准差、L2、范围 | 对对应完整张量计算的统计 | 没有仅对缩略热力图计算 |
| 有效片段与 token 掩码 | 哪些位置是真实输入，哪些是补零槽 | 下载中保留布尔数组 |
| `audio_embedding` | 对齐训练后的音频召回向量 | 仅检索已训练检查点提供 |
| Qwen 回答 | 使用该音频隐藏序列和问题生成的文字 | 自回归文本输出 |

热力图最多等距选取 128 行 × 64 列，每幅图使用自己的数值色阶。维度编号、形状与数值片段帮助检查网络是否有变化，但没有“第 17 维就是悲伤”这样的固定解释，图案好看也不能证明音乐理解正确。

选择其他歌曲或加载其他检查点时，旧结果会清空，避免把上一首歌的回答误认成当前结果。当前结果还会记录实际歌曲、问题、检查点、模型包 ID 和原始/静音条件。

## 6. 静音与错配对照

先用“原始音频”执行并下载 JSON，再保持相同歌曲和问题，把输入条件改为“静音对照”。静音保留原长度和有效掩码，只有波形内容归零。页面播放器仍播放原歌曲，输入条件标记说明送入模型的是哪一种。

若要做错配对照，选择另一首测试歌曲，保持原问题不变，比较回答与特征。当前页面一次显示一个结果，用下载的 JSON 保留多次实验。

原始输入与静音特征不同，仅说明编码器对输入有响应。判断模型有没有利用音乐，仍要比较属性正确性、错配后质量下降、不同歌曲的回答差异及留出集召回，不能仅看隐藏数值是否变化。

## 7. 下载结果与离线分析

“下载本次结果 JSON”保存文字回答、问题、歌曲 ID、检查点、统计量、热力图抽样和数值片段。“下载完整特征 NPZ”保存完整数值数组：

```python
import numpy as np

with np.load("audio_features.npz", allow_pickle=False) as arrays:
    cnn = arrays["cnn_pooled"]
    hidden = arrays["transformer_hidden"]
    valid = arrays["token_mask"]
    print("CNN:", cnn.shape)
    print("Transformer:", hidden.shape)
    print("有效 token:", int(valid.sum()))
```

服务只在内存保存最近 5 次完整特征，重启或被后续结果淘汰后需要重新检查。前端播放采用携带鉴权头的音频请求再生成本地 Blob，歌曲切换时释放旧 Blob。HTTP 请求只允许访问配置测试清单内的歌曲、当前实验的完整检查点和最近生成的下载结果。

## 8. 代码位置与迁移

```text
music_train/
├── diagnostics.py      # 梯度/实际权重差、训练进度快照
├── dashboard.py        # 日志、检查点、测试集、播放、检查与下载 API
├── inspection.py       # 同次前向的真实张量、摘要和 NPZ
└── web/
    ├── index.html
    ├── style.css
    └── app.js          # 原生 Canvas 曲线/热力图，无外部资源
deploy/dashboard.sh     # Linux 启动入口
tests/test_dashboard.py # 数值、日志、API 和打包排除测试
scripts/package_source.py
```

生成不含数据、标注、权重、日志和环境的源码包：

```bash
python -m scripts.package_source
```

产物在 `dist/CNN-Transformer-Qwen-source.zip`，附文件清单和 SHA-256。`MuChin/`、`Qwen3-4B-Base/`、`data/` 只含空 `.gitkeep`；Git 无法记录完全空目录，因此使用这个占位文件。该操作不会执行 GitHub 推送。

默认 `configs/server.yaml` 已使用工程内占位目录。你也可以把本地已有数据、模型留在工程外，在 YAML 和网页启动参数中填它们的实际路径。
