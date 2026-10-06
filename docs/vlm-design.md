# MedShearletX：VLM 适配设计与实验边界

这次实现的目标是把 ShearletX 的图像解释流程接到 VLM 分类上：在固定分类问题下，寻找更稀疏的 shearlet 掩码，使扰动后的图像尽量保留原图的分类分数。第一阶段完成独立、可扩展、可离线测试的实现；真实 API 能力和医疗数据效果需要后续验证。

本文记录实现依据和实验设计，不把生成概率称为诊断正确率，也不预先指定“最适合医疗”的模型。当前没有实现医疗校准流程，没有调用真实 VLM API。

## 1. 原代码结构与迁移点

原项目的 `code/` 是论文实验代码：

| 位置 | 主要职责 | 对这次工作的影响 |
| --- | --- | --- |
| `code/shearletx.py` | PyShearLab 滤波器、PyTorch shearlet 分解与重构、mask 的 Adam 优化、分类器 softmax 分数 | 与本地可微分类器、tensor 布局和设备设置耦合；远程 VLM 不能直接沿用 `loss.backward()` |
| `code/waveletx.py`、`code/cartoonx.py` | wavelet 解释方法 | 保留原论文实验用途 |
| `code/pixelmask.py`、`code/smoothmask.py` | 像素空间解释基线 | 可在后续作为比较对象 |
| `code/imagenet_utils/` | ImageNet 数据与传统分类器实验 | 不适合作为通用医疗图片/API 数据层 |
| `code/*.ipynb`、`code/scatterplot.py` | 示例与论文实验 | 不承担新 VLM provider 的连接逻辑 |

新增 `medshearletx/` 包独立于这些代码，避免为了支持一个 API 而把 provider、数据读取、score 或配置逻辑塞进原优化器。原论文实现仍可以用于本地分类器复现。

### 论文目标与仓库实现的区别

[论文第 4 节，式 (7)](https://arxiv.org/html/2211.12857v3#S4) 的形式是最大化：

\[
\mathbb E_u[\Phi_c(T^{-1}(m\odot T(x)+(1-m)\odot u))]
-\lambda_1\|m\|_1-\lambda_2\|T^{-1}(m\odot T(x))\|_1.
\]

这里 `T` 是 digital shearlet transform，`m` 是 shearlet 系数上的 mask，`u` 是替换被删除信息的扰动；`Phi_c` 是分类器的目标类概率。

仓库 `code/shearletx.py` 默认使用另一种保真形式：最小化原图与扰动图目标类概率的平方差，再加入 mask 与空间正则项；`maximize_label=True` 时才把参考值设为 1。新增 VLM 实现采用**保持原图 score 的平方差**，与仓库默认行为一致，而不是宣称逐字复现论文式 (7)：

\[
L(m)=\lambda_f\mathbb E_u[(s_c(x)-s_c(x_{m,u}))^2]
+\lambda_m\,\operatorname{mean}|m|
+\lambda_x\,\operatorname{mean}|T^{-1}(m\odot T(x))|.
\]

空间项采用 L1 幅度，延续原代码所称的 spatial energy penalty；它不是平方能量。score 更换后量纲也会改变，因此概率与 margin 的实验需要记录正则权重，不能直接用一个未经检查的数值比较优劣。

## 2. VLM 分类 score 的定义

对于生成式 VLM，固定图片与分类 prompt 后，首先观察的是回答 token 的条件分布 `p(token | image, prompt)`。定义候选类别后，可以把这个分布转成分类 score。它回答“模型在这个问题和这些候选项下倾向生成哪个答案”，不直接回答“医学判断正确的概率是多少”。

首版使用单 token 类别编码，例如 `A`、`B`，并在 prompt 中明确其含义。让模型输出类别仍是为了定位回答位置；score 来自 API 返回的概率证据，不来自模型自己写出的一个信心数字。

| score | 定义 | 为什么实现 | 主要限制 |
| --- | --- | --- | --- |
| `probability` | `softmax(class_logprobs)[target]` | 最接近 ShearletX 的目标类概率，默认选择 | 是观测到的候选 token 集合内的条件概率；不是原始完整 vocabulary 概率，也尚未校准 |
| `log_margin` | `logp(target) - max(logp(other classes))` | 提供目标类相对竞争类的差距；数值范围与概率不同，可比较优化行为 | 多分类取最强竞争类；二分类与概率是同一证据的单调变换 |
| `agreement` | 固定设置下重复生成，返回目标类的回答频率 | 对没有 logprobs 的部署仍可定义可测量的黑盒 score | 是指定解码分布下的经验频率，存在采样误差和额外请求成本 |

二分类时 `probability = sigmoid(log_margin)`。实现两个选项是在比较目标函数的尺度与饱和行为，不是得到两份独立的不确定性证据。

对 `agreement`，需要固定采样次数、非零 temperature、prompt 和其他解码设置。首版在第一个无效格式、拒答或请求失败时立即停止该 score，保存失败状态与已尝试的预测次数，不输出部分样本的频率，也不计算无效/拒答比例。不能悄悄删除无效回答后报告提高了的信心。温度 0 下重复生成主要测服务稳定性，并不能充分测出模型的类别分布。小样本的频率只有离散取值，会让黑盒优化更嘈杂。

### 严格的 token 分布处理

- 在同一类别回答位置收集所有候选类别的 logprobs，确认对应单 token。首版不把多 token 标签的第一 token 当作整个答案概率。
- `top_logprobs` 只包含部分 vocabulary；缺少类别表示概率未知，不能补成 0 或一个任意极小值。
- 类别分布覆盖不完整时明确失败，或者由配置显式选择 `agreement`；不要在优化途中无声切换 score。
- 如果模型先输出解释、JSON 标点或其他前缀，不能随意从后文抽取 `A/B` 的 logprob 当作相同条件下的分类证据。先通过能力检查验证回答位置和格式。
- 不分别强制模型生成每个标签，再比较被强制后的生成概率；token bias 或输出约束可能改变分布。
- 保存归一化 class logprobs 与 `log_class_mass`，二者相加可恢复当前观测 token 集合的原始 class logprobs。只有很少概率质量落在候选类别上时，类别内归一化可能产生很高的数值，掩盖格式失败。

Top-k 接口中，同一个类别可能有 `A` 与带前导空格的 ` A` 等不同单 token 变体。首版把实际报告的变体概率相加；即使每个类别都有一个已报告 token，未报告的同类变体仍是未知，因此 `class_mass` 是观测 token 的质量，可能只是全部有效类别回答质量的下界。相应的归一化 score 条件于这个已观测 verbalizer 集合，不应宣称是完整类别分布的精确值。vLLM 显式 token ID 模式使用每类声明的那个 token 作为固定 verbalizer，并记录 ID 是否被实际验证。

闭源 API 不一定开放 logits、hidden states 或图像梯度。可以取得 token logprobs 的服务，也不等于可以读取全部“内部信心”。基于 hidden state 的 confidence probe 是另一个需要本地权重、标签和验证的数据项目。

### 附加诊断与校准

完整类别分布允许计算 `entropy = -sum(p * log(p))`、预测类别及概率差。这些是审计诊断，不能用“最小 entropy”替代保持固定目标类：一个错误的其他类别也可能有很低 entropy。不要把整段解释的平均 token logprob 当作类别 confidence；它同时受语言流畅度、文本长度和回答结构影响。

自报 confidence 可以将来作为实验 baseline，但不是本次默认目标。专门训练或后处理有时能改善其校准，[相关研究](https://arxiv.org/abs/2205.14334) 不支持把所有自报信心一概视为真实概率。

校准需要独立、带标签、与目标场景匹配的验证集。[Temperature scaling 的原始研究](https://proceedings.mlr.press/v70/guo17a.html) 提供一个后续可评估的方法。Brier score、NLL、可靠性图、ECE 与错误检测/拒答表现应在医疗 holdout 上分别测量；当前阶段尚未实现这些校准与临床效果验证。校准在原图上成立，也不能自动保证在扰动解释图上成立。

## 3. 独立模块与扩展方式

数据流：

```text
配置 -> DataLoader -> RGB 图片与样本元数据
                         |
                         v
                    VLM backend -> typed Prediction
                         |               |
                         |               v
                         |          ScoreStrategy
                         |               |
                         v               v
                 Transform <-> SPSA explainer
                                      |
                                      v
                            runner / 图片与诊断结果
```

### Backend 与 Prediction

Backend 只负责构造供应商请求、调用接口、校验返回值，并转换为统一的 `Prediction`：单次采样类别、可选的 class logprobs，以及必要的请求/模型元数据。重复生成计数由 Score 层负责。Score 层不理解 OpenAI 或 Gemini 原始 JSON，解释器也不导入其 SDK。

常见的后续模型部署只需在**一份配置**中填写 provider、model、endpoint、环境变量名称及能力设置。新接口协议才需要新增一个 backend adapter，并在工厂入口注册；不会改动数据读取、score、transform 或优化循环。API key 由环境变量读取，不进入配置、输出或版本控制。

OpenAI-compatible adapter 可以服务 OpenAI 和 vLLM，但兼容请求格式不表示功能完全相同。能力以部署和实际返回值为准：

- [OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) 定义 `logprobs` 与最多 20 个 `top_logprobs`；参数支持因模型而异，不能假定所有视觉模型支持。
- [vLLM Chat 协议](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/protocol/) 在当前版本提供 `logprob_token_ids`，可指定全部类别 token ID，要求 `logprobs=true`；需要验证用户部署版本以及 tokenizer。
- [vLLM sampler](https://docs.vllm.ai/en/stable/api/vllm/v1/sample/sampler/) 区分原始分布和经过 temperature/其他 processors 的分布。实验需记录返回的是何种 logprobs，不能把 processed logits 当成 logprobs。
- [Gemini GenerateContent](https://ai.google.dev/api/generate-content) 定义 `responseLogprobs`、`logprobs`（0 到 20）和 `logprobsResult`。具体模型、接口和账号是否可用，留到能力检查验证。

多 token 候选答案的 teacher forcing 作为将来的独立 capability。当前 [vLLM teacher scoring](https://docs.vllm.ai/en/stable/training/prompt_token_id_logprobs/) 文档规定 V2 runner 等要求，而且不通过 OpenAI-compatible endpoints 暴露；不能把它伪装成所有 `/chat/completions` 部署都有的功能。

### DataLoader

数据层只负责读取和预处理图片，不依赖 PyTorch，不发送 API 请求。首版支持图片文件夹与 CSV manifest，用 Pillow 读入 RGB 图片，并携带样本 ID、可选真实标签和原始路径。真实标签用于未来评估，默认解释目标来自原图预测；这两者不能混淆。

目录读取适合快速试运行，CSV 适合指定样本列表、标签和后续划分。Runner 默认按 `image_size` 保持比例缩放并 padding 到正方形（`letterbox`），记录原始模式、尺寸、缩放后尺寸、padding 与样本元数据；`stretch` 只能由配置显式选择。128px 示例配置用于低成本联调，医疗实验需要另外确定合适分辨率和预处理。运行解释前先检查 transform 和重构误差，再调用 API；dry-run 只做配置、路径和预算检查。

首版输入要求单帧、已明确 windowing 的 8-bit raster。检测到 `I`、`F`、`I;16*` 模式或多帧图片时明确拒绝，避免 Pillow RGB 转换把大于 255 的值静默饱和。部分编码器可能在读取时已经降位深，因此不能把模式检查当作通用医学影像解析。DICOM、医学 windowing、病例多图或其他格式应新增独立数据适配，不改 provider。当前 RGB 路径不会自动解决这些医学数据语义。

### Transform

真实解释使用可选的 PyShearLab shearlet 分解与重构，依赖在选择该 transform 时检查。分解/重构的形状、数值范围和重构误差需要独立验证。

PyShearLab 0.0.1 的系统构造代码把两个尺寸不同的 filter 数组组成 tuple，再除以 NumPy scalar，在当前 NumPy 中会报错。兼容层仅在构造 shearlet system 时，用带逐数组除法的 `_FilterPair` 临时适配上游两个模块的 filter 函数引用；该过程用锁串行化，并在成功或异常后恢复原引用。它不修改安装的依赖源文件，不改变 NumPy 全局行为，也不改变各 filter 的数学除法。已验证 128×128、两尺度的真实分解/重构路径；这仍只是数值正确性验证，不能替代 VLM 解释效果测试。

`identity` 仅用于显式选择的离线 smoke test，验证模块之间的数据流。它既不是 shearlet transform，也不能产生声称属于 ShearletX 的科学结果。PyShearLab 不可用时明确报错，不能偷偷退回 identity。

### Explainer

远程 API 没有从 score 到图片的梯度。首版把每个 shearlet 子带的空间 mask 限制在较粗网格上，再展开到系数大小，并通过 SPSA 的随机方向估计梯度。

每个正负扰动配对使用相同的系数噪声 realization，降低有限差分中无关图像噪声的影响。即使如此，API 解码随机性、图像编码量化与有限采样仍会使 score 有噪声；共同图像噪声不意味着整个 API 响应确定。

优化期间保持原图的目标类别和参考 score 固定，mask 投影到 `[0,1]`，同时施加 mask 稀疏项与重构图 L1 空间项。每次尝试的模型预测都受显式最大 query budget 限制；agreement 的多次生成也计入预算，不能只把一个逻辑 score 当作一个请求。预算必须为原图参考、优化配对和最终验证留下空间。`max_requests` 限制单样本单 score，`max_total_requests` 限制整个运行的计划上界；复用原图 native reference 后实际次数可以更少。能力声明或凭证检查在 HTTP 前失败的尝试也计入 `prediction_attempts`，所以尝试次数是实际外部 HTTP 请求数的上界，不能直接换算成精确费用。

Signed shearlet 系数在分解和 mask 计算中保持原值，重构也先保留 signed pixel 值。空间 L1 惩罚使用这份未截断重构；发送 API 和保存 PNG 时才把像素截断到 `[0,1]` 并量化为 8 bit。诊断记录 raw 重构范围与 clipping 比例，避免把未截断的空间幅度当成可见图的幅度。原始 kept 与 removed 重构相加应等于原图；两幅分别经过 clipping/量化的 PNG 通常不再满足该加法关系。删除图的 score 是这些已处理像素的模型响应，也包含 preprocessing 的影响。

粗网格减少黑盒优化维度和请求需求，也限制了 mask 能表达的细节。因此本实现是 ShearletX 思路的**分组 mask、零阶黑盒近似**。它没有原论文的完整参数空间和原梯度算法，不能未经验证直接继承论文关于解释质量、细节分离或伪影的保证。

### Runner 与比较

Runner 读取配置、创建各模块，按样本运行，并分别保存每个 score 优化出的 mask、解释图、loss/query 记录及共享诊断。比较时保持样本、prompt、目标类、transform、网格、随机种子和预算一致；必要时调整和记录 score 尺度对应的正则权重。

至少检查原图预测、解释图是否保留目标类别、目标类 score 的变化、保留 mask 比例、空间 L1 幅度、预算与采样稳定性。尽可能用一个共同可用的评估 score 比较不同优化目标，不用“用自身 score 优化后自身 score 更好”作为唯一证据。

如果某部署只能提供 agreement，不能虚构 probability/margin 比较结果。无法取得完整分布的实验应标记能力限制，保留失败信息。

## 4. 分阶段验证

1. **当前：离线实现。** 完成独立模块、配置与 CLI，使用 mock backend 和显式 identity smoke test 验证请求预算、固定目标、score 数学、数据读取、异常处理和结果保存。可用时再验证真实 PyShearLab 重构与优化路径。Mock 的“分数”不代表任何真实 VLM 性能。
2. **用户提供 API 后：能力检查与少量图片测试。** 先验证图像输入、类别回答位置、logprobs 完整覆盖、模型/接口参数支持以及请求计数，再用较小预算测试真实 shearlet 解释。通过检查之后才能对模型 score 做有意义的比较。
3. **后续医疗 audit：独立验证任务。** 确定具体影像、标签和分组/划分策略，在未参与选择 score 与调参的 holdout 上比较分类表现、校准、解释稳定性、删除/插入等保真证据及潜在捷径特征。医学 ground truth、模型错误与解释伪影需要分别分析；一张可视化解释不能证明医学因果关系。

当前“可以测试”应理解为代码与离线测试已具备、等待真实 API 的能力检查，不表示医疗效果已验证。
