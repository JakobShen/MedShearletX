# Gemini 3.5 Flash-Lite 的 ImageNet 1000 类实验

本实验先保留原 ShearletX 的 ImageNet 单图分类问题，再把分类器换成远程 VLM。问题始终是从仓库的 **全部 1000 个 ImageNet 类别**中选择一个类别；不得提前只留下狗品种，不得把正确标签告诉模型，也不得在解释图评分时缩小候选集。类别顺序、完整类别描述、prompt 和生成设置在原图、扰动图、最终解释图之间固定。

## 输入图片与出处

用户提供的截图对应 [论文 Figure 1](https://arxiv.org/abs/2211.12857) 的 **English foxhound**，其 ImageNet 0-based ID 为 **167**。该图是站立的短毛猎犬。现有仓库 `code/imgs/ILSVRC2012_val_00017625.JPEG` 则是 500×375 的 **Afghan hound** 头像，ID 160；不能把二者当成同一张输入图。

本次匹配截图的输入文件为 `code/imgs/english_foxhound_paper.png`，从用户提供的 `2211.12857v3.pdf` 首页第 7 个嵌入图像对象直接提取，RGB，484×484。原 JPEG 没有包含在当前仓库中，所以这是论文中的绘图 raster，不能宣称恢复了原始 ImageNet 文件。首页第 9 个嵌入图像对象是截图中的原 ShearletX 结果，仅用于参照；新图必须由 Gemini 的实际查询结果及优化出的 mask 重建，不能直接使用论文解释图。

1000 个类别从 `code/imagenet_utils/imagenet_labels.py` 或 `imagenet_labels.txt` 读取。ID 167 是图片出处的参考标签；实际解释目标取 **Gemini 对原图的预测类别**，随后固定该目标。如果 Gemini 预测不同类别，结果同时记录参考标签与模型预测，不能强行改成 English foxhound。

## 原论文与原示例的参数

论文附录 B.1 和 upstream 代码给出以下设置。论文完整实验使用 300 步，示例 notebook 的狗图使用 150 步，因此需在结果中记录选择的步数。

| 项目 | 原实验设置 |
| --- | --- |
| 图像 | RGB，拉伸为 256×256，各像素在 [0,1] |
| Shearlet system | PyShearLab，4 scales，49 bands |
| Mask | band × 256 × 256，RGB 共享；初始值全 1 |
| Optimizer | Adam，学习率 0.1，其他参数为 PyTorch 默认值 |
| 优化步数 | 论文 300；示例 notebook 150 |
| Monte Carlo noise samples | 每步 16 个 |
| Noise | 各 grayscale shearlet band 的 empirical mean ± standard deviation 范围内采样 uniform noise；RGB 共用 |
| Noise sampling | 原代码每步重新采样 |
| Mask L1 权重 | 1 |
| Spatial L1 权重 | 2，基于 grayscale reconstruction |
| 示例 objective | `maximize_label=True`，概率参考值为 1 |
| 最终显示 | RGB reconstruction clip 至 [0,1]，再除以最大像素值 |

原论文式 (7) 最大化目标类概率的期望，并扣除 mask L1 和重建图像 L1。原代码则最小化 `mean((1 - p_target)^2)` 加两项正则，使用 `maximize_label=True`；这两者不是完全相同的 objective。复制示例代码时需明确使用后者，不能混用“保持原始 score”和“向 1 最大化”两个实验设计。

原 notebook 最终计算的是解释图最大类别概率除以原图最大类别概率，可能换成另一类别。正式结果采用论文文字定义的 **固定目标类** score 比值，另行记录解释图预测类别，以避免把换类误称为 retained probability。

## Gemini score 与 API 限制

模型输出类别是分类决策；模型自己写出的 confidence 数字不进入 score。每次调用仍包含全部 1000 类。可以给类别使用稳定数字 ID，输出必须严格映射回候选类别；无效格式、拒答或 API 失败不作为正确样本，不能静默删除后重新归一化。

Gemini 的 token logprobs 能力必须以指定部署的实际能力检查为准。即使接口返回 top-20 token logprobs，也不能据此得到全部 1000 类的概率分布。缺少目标 token 不能补成 0，多 token ID 的第一个 token 概率不能当作整个类别概率。新模型或接口可能不再支持已有 logprobs 参数；不得用旧文档中曾支持的功能替代实际验证。

2026-10-06 对本次 Vertex AI `gemini-3.5-flash-lite` 部署的 native logprobs probe 返回 HTTP **400**，错误说明 `Logprobs is not supported for this model`。因此本次配置明确使用 sampling frequency，不报告 native probability 或 log margin。[Google 的 GenerateContent reference](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/inference) 仍描述通用 `logprobsResult` 字段；通用 response schema 中存在该字段，不代表这个模型支持它。

此模型使用 provider 默认 temperature **1.0**，自定义 temperature/top-K/top-P 会被忽略。本次 Vertex adapter 不发送 temperature 字段，并分别记录 requested 与 effective temperature；不能声称用温度调整探索了该模型的 confidence。Thinking 设置为 **MINIMAL**，这是该模型默认的最低 reasoning level。[Google 的 3.5 Flash-Lite model documentation](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-5-flash-lite) 说明了这些默认值与参数限制。

当没有完整、可用的 native class probability 时，本实验采用指定温度和生成设置下的 **目标类回答采样频率**：

\[
\hat p_c(x) = \frac{\text{目标类回答次数}}{\text{独立采样次数}}.
\]

该数值是可测量的模型行为，并带有有限采样误差；它不是模型内部完整 softmax，也不是分类正确率或医学诊断置信度。原图与最终解释图应保存计数、采样次数和 95% Wilson interval。频率比值为解释图频率除以原图频率；分母为 0 时比值未定义。比例可以超过 100%，不应截断。

## 黑盒实现与可比性边界

远程 API 没有图像梯度，因此无法直接执行原代码的 `loss.backward()`。当前 `hybrid_adam` 使用 SPSA 的正负扰动估计 **API fidelity 项**的梯度，mask L1 与 spatial L1 的梯度则通过本地 synthesis adjoint 精确计算。二者合并后由 Adam 更新，学习率 0.1，β₁=0.9、β₂=0.999、ε=10⁻⁸，mask 投影回 [0,1]。这样不必用远程随机 score 去近似已经能在本地计算的正则梯度。

Mask 参数采用 band 内粗网格并展开至系数分辨率。每步重新采样 coefficient noise；plus、minus、更新后的 candidate 和之前的 best mask 使用同一批 noise。旧 best 需要重新评估，才能与当前 candidate 在同一 Monte Carlo batch 下比较。该流程每步有四组 noise probes。共同 coefficient noise 可以减少额外图像噪声，但 API 回答本身仍是独立随机采样。网格大小、步数、sampling repeats、noise samples 和总调用预算都写入结果。

Paper profile 对齐原图尺寸、4 scales、全 1 初始化、目标参考值 1、grayscale noise statistics、uniform Monte Carlo noise、正则项和最终显示归一化。SPSA、粗网格和采样频率仍是明确的算法偏差，因此本实验是 **ShearletX 的 API 黑盒适配**，不能称为原论文结果的精确复现，也不能继承原图的 37.29%。

原代码在 inverse transform 之前把扰动后的 shearlet coefficients clamp 到 [0,1]；本实现有意保留 **signed coefficients**，不复制该 legacy clipping，这是另一项明确的实现差异。图像编码还会加入 8-bit 量化，发送给模型评分的 final image 必须与展示图的实际像素一致。显示后的像素不再另行增强、绘制或生成。

API 请求预算需要覆盖原图、所有优化探针和最终验证。16 个 noise samples 与每张图片的重复回答次数是不同维度，不应遗漏其乘积。实验不能把小预算 smoke test 说成完成了 150 或 300 步的原设计。

### 小样本 fidelity 的无偏估计

优化每张扰动图片时使用两个独立回答。直接计算 `(1 - k/n)^2` 会混入 Bernoulli 采样方差：其期望为 `(1-p)^2 + p(1-p)/n`，因此不再只优化原来的平方误差。

`unbiased_sampling_distortion=true` 对参考 score `s` 使用：

\[
\widehat D = s^2 - 2s\frac{k}{n} + \frac{k(k-1)}{n(n-1)}, \qquad n\geq2.
\]

`k(k-1)` 统计不同采样之间的目标类命中配对，因此该项的期望为 `p²`。本次 `s=1`，等价于用 misses 的配对数估计 `(1-p)²`。当 `n=2` 时，只在两个回答都未命中目标类时贡献 1，其余情况贡献 0。它去除了平方频率的偏差，仍有较大方差；四组 coefficient noise 的平均与最终独立验证都有必要。该设置只允许 agreement score 且 `repeats>=2`。

## 已实现的首次运行配置

`configs/vertex-imagenet.json` 对齐原图预处理、表示、初始化、正则及 final normalization，同时缩小首次 API 实验的步数和采样量。

| 项目 | 本次配置 |
| --- | --- |
| Model / provider | `gemini-3.5-flash-lite` / Vertex AI Express mode |
| 输出 | 1000-way 类别 ID，最多 128 output tokens；不要求 confidence |
| Thinking / temperature | MINIMAL / provider default 1.0 |
| 输入 / transform | 256×256 stretch / four-scale 49-band shearlet |
| Mask 参数 | 每 band 8×8 网格，共 49×8×8 = 3136 个；初始全 1，RGB 共享 |
| Optimizer | `hybrid_adam`，30 步，lr 0.1，SPSA perturbation radius 初值 0.1 |
| 正则权重 | fidelity 1、mask 1、spatial 2 |
| Fidelity | `fidelity_reference=one`，无偏 sampling squared error |
| Noise | 每步 4 个 uniform samples，grayscale band mean/sample std，RGB 共享 |
| Spatial regularizer | clipped grayscale reconstruction L1 mean |
| Final image | clip RGB 至 [0,1]，除以正的 maximum，再量化 PNG；该 PNG 用于最终评分 |
| 优化回答次数 | 每张扰动图片 2 次，两个回答并发 |
| Noise 并发 | 4 workers，最多同时 8 个 API calls |
| Target selection | 原图 64 次回答，选择出现最多的类并固定 |
| 独立最终验证 | 原图、解释图、删除图各 128 次，8 workers；不用于选择 mask |
| 调用预算 | 1422 次名义上界 + 全局 32 次网络重试额度 = 1454；全局硬上限 1600；优化部分上限 1100 |

名义上界计算为 `64 + 3×128 + 2×(3 + 4×(1 + 4×30)) = 1422`。其中保守计入一次优化 reference，复用 target-selection 的 reference 时少用 2 个请求。整轮最多额外重试 32 次，每个回答最多调用 3 次，只重试连接故障、timeout、HTTP 429 和指定的临时 5xx；每次实际调用均经过 audit 日志和全局硬限制。能力、鉴权及无效分类回答不会被重试或丢弃后重新归一化。最终计数的分母是有效回答采样数，实际 HTTP 尝试数另行记录。之前的能力 probe 和失败 prototype 单独保存，不混入最终运行上界。首次运行不是论文 150/300 步、每步 16 noise samples 的完整复制。

Hybrid Adam 的分类梯度使用正负探针的实际有符号位移，包含投影到 `[0,1]` 后的边界距离；全 1 初始化时不能仍除以名义 `2×radius`。局部正则项使用精确伴随梯度。每步保存当前最佳 mask 和 history，网络失败时已有 checkpoint 与请求证据仍可检查。

先安装可选本地依赖，将用户提供的密钥安全加载为 `VERTEX_API_KEY`。配置文件只保存环境变量名，不保存密钥。下列命令在 repository root 执行：

```bash
python -m pip install -e '.[shearlet,figures]'
python -m medshearletx imagenet --config configs/vertex-imagenet.json --root . --output output/gemini35-imagenet-first --dry-run
python -m medshearletx imagenet --config configs/vertex-imagenet.json --root . --output output/gemini35-imagenet-first
```

Dry run 检查类别、数据、配置和预算。实际运行要求 output directory 为空；重复实验选择新目录。`classification_prompt.txt` 保存完整 1000-way prompt，`requests.jsonl` 保存去除敏感信息的逐次模型证据，`selection.json`、`reference.json`、`retained.json`、`removed.json` 保存不同阶段的采样统计。Mask、优化 history、preprocessing、effective generation settings 与最终图均另行保存。实验结束后以 `result.json` 的实际数值为准，不预先填写 retained frequency 或结论。

## 导出图

`medshearletx.figures.save_explanation_figure` 直接排版输入与解释的 PIL image，输出 `explanation.png`、`comparison.png` 及对应 PDF。单图使用白色页边与 serif 标题，黑色图像区域来自实际重建像素；comparison 同时呈现原图和解释图。

采样实验标题写作 `Retained sampling frequency: XX.XX%`，脚注给出固定目标、计数与区间、黑盒近似说明。只有确实测量了 native class probability 的实验才允许使用 `Retained probability`。图中的百分比必须来自保存的结果，不能复用论文截图或模型自己输出的信心。
