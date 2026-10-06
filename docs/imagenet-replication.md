# Gemini 3.5 Flash-Lite：完整 ShearletX mask 与历史实验

完整 ImageNet 实验保留原 ShearletX 的单图分类问题，再把分类器换成远程 VLM。每次 API 调用始终从仓库的 **全部 1000 个 ImageNet 类别**中选择一个类别；不会在选出目标之后改成某一品种的二分类，也不会在解释图评分时缩小候选集。少类别实验则明确配置另一组候选，用来比较任务设计。每一轮内部，类别顺序、完整类别描述、prompt 和生成设置在原图、扰动图、最终解释图之间固定；图片的参考品种不作为正确答案提示给模型。

## 输入图片与出处

用户提供的截图对应 [论文 Figure 1](https://arxiv.org/abs/2211.12857) 的 **English foxhound**，其 ImageNet 0-based ID 为 **167**。该图是站立的短毛猎犬。现有仓库 `code/imgs/ILSVRC2012_val_00017625.JPEG` 则是 500×375 的 **Afghan hound** 头像，ID 160；不能把二者当成同一张输入图。

此前匹配截图的输入文件为 `code/imgs/english_foxhound_paper.png`，从用户提供的 `2211.12857v3.pdf` 首页第 7 个嵌入图像对象直接提取，RGB，484×484。原 JPEG 没有包含在当前仓库中，所以这是论文中的绘图 raster，不能宣称恢复了原始 ImageNet 文件。首页第 9 个嵌入图像对象是截图中的原 ShearletX 结果，仅用于参照；新图必须由 Gemini 的实际查询结果及优化出的 mask 重建，不能直接使用论文解释图。

1000 个类别从 `code/imagenet_utils/imagenet_labels.py` 或 `imagenet_labels.txt` 读取。站立猎犬的参考 ID 为 167，Afghan 示例的参考 ID 为 160；实际解释目标取 **Gemini 对原图的预测类别**，随后固定该目标。参考标签及其来源与实际预测分别记录，不强行指定模型答案。

## 原论文与原示例的参数

论文附录 B.1 和 upstream 代码给出以下设置。论文完整实验使用 300 步，示例 notebook 的狗图使用 150 步，因此需在结果中记录选择的步数。

| 项目 | 原实验设置 |
| --- | --- |
| 图像 | 浮点 RGB tensor bilinear 拉伸为 256×256，`align_corners=False`、`antialias=False`，各像素在 [0,1] |
| Shearlet system | PyShearLab，4 scales，49 bands |
| Mask | 49×256×256，共 3,211,264 参数，RGB 共享；初始值全 1，无粗网格或平滑 |
| Optimizer | Adam，学习率 0.1，其他参数为 PyTorch 默认值 |
| 优化步数 | 论文 300；示例 notebook 150 |
| Monte Carlo noise samples | 每步 16 个 |
| Noise | 各 grayscale shearlet band 的 empirical mean ± sample standard deviation（`ddof=1`）范围内采样 uniform noise；RGB 共用 |
| Noise sampling | 原代码每步重新采样 |
| Mask L1 权重 | 1 |
| Spatial L1 权重 | 2，基于 grayscale reconstruction |
| 示例 objective | `maximize_label=True`，平方 fidelity 的参考 score 为 1，L1 penalties 使用 mean normalization |
| 作者代码的扰动路径 | 混合 coefficient noise 后先 clip coefficients 至 [0,1]，再 inverse transform |
| 最终 mask | notebook 返回最后一次迭代的 mask |
| 最终显示 | RGB reconstruction clip 至 [0,1]，再除以最大像素值 |

原论文式 (7) 最大化目标类概率的期望，并扣除 mask L1 和重建图像 L1；换成最小化形式，分类项为 `1 - mean(p_target)`。作者发布的示例代码则最小化 `mean((1 - p_target)^2)` 加按元素数归一化的两项 L1，并包含 coefficient clipping。两者的分类项、L1 单位及实现路径不能混称为同一 objective。本项目将 author-code 与 paper-objective 分成配置；后者采用线性分类项，但明确保留 author-code 的 L1 mean normalization，不能宣称完整复现论文式 (7) 的所有约定。

原 notebook 最终计算的是解释图最大类别概率除以原图最大类别概率，可能换成另一类别。正式结果采用论文文字定义的 **固定目标类** score 比值，另行记录解释图预测类别，以避免把换类误称为 retained probability。

Notebook 的 `retained_information` 则是最大值归一化后的显示解释图 L1 norm 除以原图 L1 norm，`sum(abs(display_image)) / sum(abs(original_image))`。它受亮度与显示归一化影响，与 mask mean、像素面积及信息熵均不同；不能把这些数字互换。本项目保留原来的优化 loss，这条说明不新增 information loss。

## Gemini score 与 API 限制

模型输出类别是分类决策；模型自己写出的 confidence 数字不进入 score。每次完整任务调用仍包含全部 1000 类。类别使用稳定数字 ID；采样分类可把无歧义的数字补零变体映射回同一配置类别，并记录格式统一，例如 `0160` 与 `160`。Native token scoring 不使用这种格式统一来拼造概率。无效回答、拒答或 API 失败不能静默删除后重新归一化。

Gemini 的 token logprobs 能力必须以指定部署的实际能力检查为准。即使接口返回 top-20 token logprobs，也不能据此得到全部 1000 类的概率分布。缺少目标 token 不能补成 0，多 token ID 的第一个 token 概率不能当作整个类别概率。新模型或接口可能不再支持已有 logprobs 参数；不得用旧文档中曾支持的功能替代实际验证。

2026-10-06 对本次 Vertex AI `gemini-3.5-flash-lite` 部署的 native logprobs probe 返回 HTTP **400**，错误说明 `Logprobs is not supported for this model`。因此本次配置明确使用 sampling frequency，不报告 native probability 或 log margin。[Google 的 GenerateContent reference](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/inference) 仍描述通用 `logprobsResult` 字段；通用 response schema 中存在该字段，不代表这个模型支持它。

此模型使用 provider 默认 temperature **1.0**，自定义 temperature/top-K/top-P 会被忽略。本次 Vertex adapter 不发送 temperature 字段，并分别记录 requested 与 effective temperature；不能声称用温度调整探索了该模型的 confidence。Thinking 设置为 **MINIMAL**，这是该模型默认的最低 reasoning level。[Google 的 3.5 Flash-Lite model documentation](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-5-flash-lite) 说明了这些默认值与参数限制。

当没有完整、可用的 native class probability 时，本实验采用指定温度和生成设置下的 **目标类回答采样频率**：

\[
\hat p_c(x) = \frac{\text{目标类回答次数}}{\text{独立采样次数}}.
\]

该数值是可测量的模型行为，并带有有限采样误差；它不是模型内部完整 softmax，也不是分类正确率或医学诊断置信度。原图与最终解释图应保存计数、采样次数和 95% Wilson interval。频率比值为解释图频率除以原图频率；分母为 0 时比值未定义。比例可以超过 100%，不应截断。

## 黑盒实现与可比性边界

远程 API 没有图像梯度，因此无法直接执行原代码的 `loss.backward()`。当前 `hybrid_adam` 使用 SPSA 的正负扰动估计 **API fidelity 项**的梯度，mask L1 与 spatial L1 的梯度则通过本地 synthesis adjoint 精确计算。二者合并后由 Adam 更新，β₁=0.9、β₂=0.999、ε=10⁻⁸，mask 投影回 [0,1]；学习率由各实验配置指定。这样不必用远程随机 score 去近似已经能在本地计算的正则梯度。

纠正后的 paper profiles 对每个 band 的每个空间位置独立优化：**49×256×256 = 3,211,264 参数，RGB 共享**。没有 grid expansion、pooling 或 mask smoothing。此前 8×8、4×4 的 grouped masks 只保留为显式 coarse baselines，不再代表默认论文结构。

每步重新采样 coefficient noise，正负探针与该步 candidate 共用这批 noise；API 回答仍独立采样。Dense author-code profile 使用一组 SPSA 方向与最后一次迭代 mask；每步有 plus、minus、candidate 三组 noise evaluations。历史 coarse profile 另行重新评估 best mask，不能把那个选择规则带入作者代码复现。参数形状、方向数、目标函数、mask 选择规则、采样量和调用预算均记录在结果中。

Dense profile 恢复完整 mask、256 输入、4 scales、全 1 初始化、作者参数与 noise 路径。输入先用 tensor bilinear 在浮点域拉伸，`align_corners=False`、`antialias=False`，浮点结果直接进入 shearlet transform；不会在 transform 前经 PNG 再量化。Gemini 的图像梯度、内部 class probability 仍不可用，故分类梯度使用黑盒估计、score 使用回答频率。这些变化足以使实验属于 **ShearletX 的 API 适配**，不能称为只替换模型后的数值 1:1 复现，也不能继承原图的 37.29%。

Author-code profile 现在明确复制混合 coefficients 后、inverse transform 前的 `[0,1]` clipping。之前保留 signed coefficients 的 coarse 实验是另一条路径，不能作为它的复现结果。远程模型只能接收图像：重建后的值仍需 clip、8-bit 量化并编码为 PNG，这与原白盒模型接收浮点 tensor 的评分路径不同。最终验证使用实际展示 PNG 的同一像素；图像不另行增强或生成。

API 请求预算需要覆盖原图、所有优化探针和最终验证。16 个 noise samples 与每张图片的重复回答次数是不同维度，不应遗漏其乘积。实验不能把小预算 smoke test 说成完成了 150 或 300 步的原设计。

原论文和代码采用固定步数 Adam，没有全局最优或全局收敛保证，也没有在运行结束时证明局部最优。Mask 空间包含非线性分类器，单张图的 loss 下降不能作为证明。Gemini 的有限回答频率可能在多个扰动上都为 1，导致 SPSA 差分为零；零差分不能说明真实梯度为零。完整 mask 下，一两个随机方向也不能代替原 VGG19 对全部参数的图像自动微分。

原 Afghan 示例使用 VGG19，毛发与纹理的稀疏结果属于那个分类器及优化过程。Gemini 图片中鼻子更明显这一视觉现象，本身既不能定位实现错误，也不能证明 Gemini 依赖鼻子。当前 objective 寻找能保留目标回答的稀疏表示，没有直接惩罚移除图的同类回答；移除图仍被识别不违反这个 objective，却阻止“已经隔离必要或唯一证据”的结论。需要稳定的优化结果与鼻子/毛发的受控干预，才能检验模型特征解释。

### 独立作者 FFT 公式对照

[对照测试](../tests/test_transform_parity.py) 没有调用 PyShearLab 的分析/合成函数构造参考值，而是逐式转写 upstream `code/shearletx.py` 的 FFT、shift、RGB/gray 与 dual-frame synthesis，使用已安装的 NumPy/SciPy。输入为实际 Afghan 的 float32 tensor-bilinear 预处理结果；随机 full mask 在 band 和空间上均变化，可检测全 1 round trip 掩盖的索引错误。

[诊断 JSON](results/author-transform-parity.json) 记录六个通过的离线测试。49-band 默认 filters 的最大虚部为 `7.97e-16`；作者 real cast、共轭与 shift 约定在这个偶数尺寸、近乎实数的滤波器组上没有造成结构差异。RGB analysis 的 float64 最大误差 `5.55e-16`，随机 mask synthesis 为 `3.33e-16`；作者 float32 FFT/filter casts 的对应误差分别为 `2.53e-7`、`2.43e-7`，最终 clip/max 后为 `4.68e-7`。Float32 显示 PNG 只有 2/196608 个通道值差 1 LSB。

实际 core 的混合 coefficients 先 clip、final clip/max、移除图与 grayscale 空间正则均与独立 float64 参考相符；三个 PNG 完全相同。把 clipping 错放到 inverse 后会改变 83.37% 的扰动图通道值，故测试能区分这个位置。NumPy 转置数组的 float32 mean reduction 有 `5.42e-6` 累计误差，同一数据改为连续存储或 float64 累计即降至 `5.44e-10`、`2.92e-10`，该数值差异另行保留。这里检查实际 geometry 与输入/输出路径，不是 CUDA FFT bitwise reproduction，也不是整个 VGG 优化器的 parity，更不证明收敛或 Gemini 的特征依赖。

### 小样本 fidelity 的无偏估计

Author-code 配置对每张扰动图片使用两个独立回答。直接计算 `(1 - k/n)^2` 会混入 Bernoulli 采样方差：其期望为 `(1-p)^2 + p(1-p)/n`，因此不再只优化原来的平方误差。

`unbiased_sampling_distortion=true` 对参考 score `s` 使用：

\[
\widehat D = s^2 - 2s\frac{k}{n} + \frac{k(k-1)}{n(n-1)}, \qquad n\geq2.
\]

`k(k-1)` 统计不同采样之间的目标类命中配对，因此该项的期望为 `p²`。Paper profile 使用 `s=1`，等价于用 misses 的配对数估计 `(1-p)²`。当 `n=2` 时，只在两个回答都未命中目标类时贡献 1，其余情况贡献 0。它去除了平方频率的偏差，仍有采样方差；noise 平均与最终独立验证仍有必要。该设置只允许 agreement score 且 `repeats>=2`。

线性 paper-objective 分类项为 `1 - mean(p_target)`，使用频率即可无偏估计，不需要上述二次配对项。

## 历史 coarse 站立猎犬配置

`configs/vertex-imagenet.json` 记录此前 coarse 站立猎犬实验。它使用 8×8 网格、30 步及较小采样量；下面的实测结果只能归属于这个历史配置，不属于新的 dense paper profile。

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

## Dense Afghan hound profiles 与八步 pilot 结果

当前结构验证与后续完整运行使用仓库原始 `code/imgs/ILSVRC2012_val_00017625.JPEG`、全部 1000 个 ImageNet 类。上游示例将其作为 Afghan hound，ID 160，完整名称为 `Afghan hound, Afghan`；这个参考标签及其来源写入 `image_metadata`，不代替 Gemini 的实际分类。

| 配置 | 目标与运行长度 |
| --- | --- |
| `configs/vertex-afghan-imagenet.json` | Full author-code profile：150 步、平方 fidelity、最后一个 mask |
| `configs/vertex-afghan-imagenet-pilot.json` | 同一 full mask 和数据路径，8 步结构验证；不用于声称收敛 |
| `configs/vertex-afghan-paper-objective.json` | 300 步、线性 `1 - mean(p_target)`，关闭混合 coefficient clipping；L1 使用作者代码的 mean normalization，差异明确记录 |
| `configs/vertex-afghan-imagenet-coarse.json`、`configs/vertex-afghan-fiveway-coarse.json` | 之前的 4×4 网格实验，明确作为 coarse baselines 保存 |

Author-code 与 pilot 的共同设置为 49×256×256 dense mask、初始全 1、RGB 共享；每步 16 组重新采样的 shared grayscale uniform noise（sample std，`ddof=1`）；Adam lr 0.1、mask/spatial L1 权重 1/2；平方 fidelity 的参考 score 为 1；混合 coefficients 在 inverse 之前 clip 至 `[0,1]`。分类梯度仍是显式 SPSA 黑盒估计，不是原 classifier 的自动微分。每张扰动图片用两个独立回答估计平方项；这些 repeats 与每步 16 组 coefficient noise 是不同维度。

配置明确设置 `mask_resolution=full`、`mask_selection=last`；author-code 的 `fidelity_loss=squared_error`、`obfuscation_coefficient_clip=true`，paper-objective 则为 `one_minus_score`、`false`。两条路径都不加入 pooling 或平滑。

输入使用浮点 tensor bilinear stretch 到 256×256，关闭 antialias、`align_corners=False`，保留浮点输入进入 transform。最后显示按作者路径 clip RGB reconstruction、再除以正的最大值；实际 API 输入是量化后的 PNG。参数与完整 mask 恢复不能消除这些 API 与 score 差异，故不宣称原论文结果的精确 1:1 复制。

每轮先用原图 32 次回答选择并固定目标；完整 1000 类始终进入每一次 API 调用。原图、最终保留图、移除图各用 128 个独立回答验证，不使用选择或优化探针的计数替代。每个回答最多重试三次，整轮共享 32 次临时故障重试额度；所有实际调用经过 audit 与硬限制，无效分类回答不被静默丢弃。

150 步 author-code profile 的计划上界为 `32 + 3×128 + 2×(3 + 16×(1 + 3×150)) + 32 = 14886` 次物理尝试，硬上限 15000。8 步 pilot 的同类上界为 `32 + 3×128 + 2×(3 + 16×(1 + 3×8)) + 32 = 1254`。上界保守包含一次可复用的优化 reference；最终以 dry-run 计划和实际 `requests.jsonl` 计数为准。**150 步 author-code 和 300 步 paper-objective 尚未运行**。

2026-10-06 的八步 dense pilot 保存在本地 `runs/vertex-afghan-imagenet-full-pilot-20261006-231407/`，源码为 `c7d790ad15180bba55a30277f9e003548e603a1e`。实际 **1221 次物理尝试**，其中一次临时故障重试，耗时 374.7 秒。32/32 原图选择回答为 Afghan hound（ID 160）；下表使用独立于选择和优化的最终回答：

| 验证输入 | Afghan hound 回答 | 目标类频率 | 95% Wilson interval |
| --- | --- | --- | --- |
| 原图 | 128 / 128 | 100% | 97.09–100% |
| 保留图 | 128 / 128 | 100% | 97.09–100% |
| 移除图 | 128 / 128 | 100% | 97.09–100% |

固定目标频率保留率为 100%，移除后的目标频率下降为 0。这轮 **没有隔离必要分类证据**，八步只检查了完整参数形状与实际输入/输出路径，不能声称整个优化器与 VGG 一致、解释成功、收敛或医疗 audit 成功。最后使用 step 8 的 `49×256×256` mask，mean 为 `0.515236`，这不是保留 51.52% 的信息。

![八步完整 mask Gemini 实测对照](assets/gemini35-afghan-full-pilot-20261006/comparison.png)

8 组分类项正负探针有 6 组差分为零；另两组估计梯度 norm 为 `1342.18`、`681.62`，而局部正则梯度 norm 约 `0.00509`。Step 8 的平均 mask 变化仍为 `0.03582`。采样频率的饱和与高维随机差分的噪声都可能影响这个优化过程；不能把零差分或 loss 下降当成驻点证明，也不能仅凭形状判断鼻子或毛发是 Gemini 的决定性特征。[可共享结果与完整 history](results/gemini35-afghan-full-pilot-20261006.json) 保存逐步值及未验证事项，逐步图可在该 run 的 `index.html` 查看。

## 历史 coarse Afghan 实验与必要性检查

这些实验使用 4×4 网格（784 个参数）、50 步、两个 SPSA 方向、每张探针四次回答、每步两组 coefficient noise、lr 0.03、radius 0.15。其计划上界 2868 含重试、硬上限 3000；这些参数和旧结果都不属于纠正后的 full mask profile。

五类候选为 `Afghan hound`、`beagle`、`golden retriever`、`English foxhound`、`bulldog`，输出 A–E，问题询问狗品种。1000 类 baseline 使用完整物体分类问题和数字代码。候选、问题及 verbalizer 均不同，两轮差异不能完全归因于候选数量；五类闭集也没有“看不出狗”的选项。

完成的五类运行保存在本地 `runs/vertex-afghan-fiveway-20261006-224135/`，源码对应 `964c8bc`。实际物理尝试 **2833** 次，其中一次重试；原图、保留图、移除图的独立验证均为 **128/128 Afghan hound**，频率比为 100%。程序执行成功不等于 audit 成功：移除图仍稳定产生同一目标，因此这轮没有隔离必要分类证据。Mask mean 约 0.336 只是软权重均值，不能称为保留 33.6% 或移除 66.4% 的信息。

1000 类初次启动共 184 次尝试，在三次严格格式失败（例如 `0160` 与配置 `160` 的区别）后停止；另一次诊断调用单独保存。格式解析纠正后（`19d33aa`）重新运行，用户在 step 29 后中断，没有完成最终 held-out 评估；checkpoint 和逐次证据保留，不能把进度预览当作完整结果。

最大值归一化会抵消均匀权重的缩放：若 mask 为常数 `a`，保留图 `(a·x)/max(a·x)` 与 `x/max(x)` 相同，而移除图 `(1-a)·x` 仍可能可识别。共同线索、亮度鲁棒性与有限分类任务都可能使两张图继续得到同类回答。必要性检查、采样频率和 mask mean 需要分别报告。

## 通用命令与逐步绘图

`experiment` 是通用单图实验命令，`imagenet` 保留为兼容名称。`task` 配置必须二选一：`imagenet_labels_path` 读取完整有序 ImageNet 类别，或 `labels` 显式列出候选；另可设置 `question`。旧的顶层 `labels_path` 配置仍可使用。

```bash
python -m medshearletx experiment --config configs/vertex-afghan-imagenet-pilot.json --dry-run
python -m medshearletx experiment --config configs/vertex-afghan-imagenet-pilot.json
python -m medshearletx experiment --config configs/vertex-afghan-imagenet.json --dry-run
```

未指定 `--output` 时，每次创建新的 `runs/<run_name>-<timestamp>/`，`run_name` 来自配置。指定 output 时仍要求该目录为空；从其他目录运行可传 `--root`。Dry run 不调用模型，也不创建实验输出目录。

从 step 0 到最后一步，自动保存当前迭代的重建结果：

| 路径 | 内容 |
| --- | --- |
| `figures/f_n/stepNNN.png` | 原图、当前保留图、当前移除图与 loss/mask mean |
| `images/f_n/stepNNN.png` | 当前保留图的原始显示 PNG |
| `images/removed/stepNNN.png` | 当前移除图的 PNG |
| `masks/stepNNN.npy` | 对应 mask；默认 paper profiles 保存完整 dense 参数 |
| `metrics.json`、`index.html` | 逐步指标和可直接在本地浏览器打开的 slider |

运行中刷新 `index.html` 可查看最新完成的 step。这些图片直接从当前 mask 与原始系数重建，绘图不增加 API 调用。图中 loss 来自带 coefficient noise 的优化探针；**没有对每一张显示预览另测 VLM score**，因此不能给 step 图写 retained probability 或把 mask mean 当成 confidence。Author-code profile 使用最后的 mask，历史 coarse profile 可选择 best mask；选择规则与 step 记录在结果中。最终独立验证和 PNG/PDF 图仍单独保存在 `result.json` 及最终图文件中。

## 导出图

`medshearletx.figures.save_explanation_figure` 直接排版输入与解释的 PIL image，输出 `explanation.png`、`comparison.png` 及对应 PDF。单图使用白色页边与 serif 标题，黑色图像区域来自实际重建像素；comparison 同时呈现原图和解释图。

采样实验标题写作 `Retained sampling frequency: XX.XX%`，脚注给出固定目标、计数与区间、黑盒近似说明。只有确实测量了 native class probability 的实验才允许使用 `Retained probability`。图中的百分比必须来自保存的结果，不能复用论文截图或模型自己输出的信心。

## 2026-10-06 历史 coarse 站立猎犬实测结果

实际模型为 Vertex `gemini-3.5-flash-lite`。全部 1000 类参与每次分类，最终固定目标为 **166: Walker hound, Walker foxhound**，与论文图的参考类别 **167: English foxhound** 不同。64 次目标选择中 Walker foxhound 有 40 次；下表使用独立于选择和优化的验证样本。

| 验证输入 | Walker foxhound 回答 | 目标类频率 | 95% Wilson interval |
| --- | --- | --- | --- |
| 原图 | 94 / 128 | 73.44% | 65.18–80.32% |
| 保留图 | 87 / 128 | 67.97% | 59.46–75.43% |
| 移除图 | 17 / 128 | 13.28% | 8.46–20.24% |

固定目标的频率保留率为 `(87/128)/(94/128) = 92.55%`。移除图的 128 次回答中，92 次为 beagle、19 次为 English foxhound、17 次为 Walker foxhound；目标品种的频率下降，狗本身仍可被识别。保留率不是“分类有 92.55% 的把握”。两个频率的区间重叠，也不能把 7 次回答的差异当成确定的性能损失。

![Gemini 实际原图与解释图](assets/gemini35-imagenet-20261006/comparison.png)

本轮实际完成 **1420 次调用**，无失败、无重试，耗时约 7.7 分钟。mask 平均权重为 0.675；这表示软权重均值，不是保留像素面积或非零系数比例。图像没有原论文那样稀疏，30 步预实验不能支持收敛或最小解释结论。论文的 37.29% 使用另一分类器及 native probability，与当前采样频率、预测类别和 mask 不能作数值优劣比较。

完整采样结果保存在本地 `output/gemini35-imagenet-20261006-02/`；[可共享证据摘要](results/gemini35-imagenet-20261006.json) 和实际 PNG/PDF、输入图、保留/移除图、mask 已纳入仓库。原始 optimization diagnostics 包含每张仅 2 次采样的便宜预览，不能替代上表与最终图所用的 128 次验证估计。运行代码对应 commit `2adf84e`。

此前一次 prototype 在第 29 步连接失败，1120 次尝试中 1119 次成功；结果日志单独保留。另有 native 能力检查、单次采样检查及一次未成功保存的 64 次采样检查。总尝试数与这些失败实验均在证据摘要中明确列出，没有挑选 prototype 的有利数值作为最终结论。
