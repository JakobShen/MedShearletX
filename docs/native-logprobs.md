# Vertex 原生 logprobs 与采样损失

2026-10-06 使用当前密钥的 Vertex Express `v1 generateContent` 接口实测，输入为同一个 256×256 Afghan hound 图像和完整 1000 类 ImageNet prompt。结果和概率证据保存在[审计 JSON](results/vertex-logprobs-20261006.json)。模型和接口的能力分别确认，配置中的 `supports_logprobs=true` 本身不构成能力证明。

| 精确模型 ID | 请求原生 logprobs | 普通分类 |
|---|---|---|
| `gemini-3.5-flash-lite` | 两种请求均 HTTP 400：`Logprobs is not supported for this model` | HTTP 200，编号 `160`，无原生概率字段 |
| `gemini-3.1-flash-lite` | HTTP 400，相同不支持提示 | HTTP 200，编号 `160` |
| `gemini-2.5-flash-lite` | HTTP 200，有 chosen/top token logprobs | 此原生请求回答 `160` |
| `gemini-2.5-flash` | HTTP 200，有 chosen/top token logprobs | 此原生请求回答 `160` |

[官方推理文档](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/inference)将 `responseLogprobs` 和 `logprobs` 标为 Gemini 3.x 已弃用，并将彻底弃用。通用 REST 字段存在，不能推导具体型号支持。[官方生命周期表](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions)当前列出 2.5 Flash/Lite 于 **2026-10-20** 退休，所以这个已测通的前代型号适合短期评分对照，不能作为长期部署承诺。

## 当前“采样损失”如何得到

`agreement` 不让模型写“信心”。它对同一张扰动图独立调用模型 N 次，计数 k 次回答固定目标类，得到频率 `k/N`。这里的采样有两层：M 张随机系数扰动图，以及每张图上的 N 次类别回答。作者的白盒分类器也采样扰动图，但它直接获得每张图的分类概率和反向梯度；当前 API 频率方案还要采样类别回答。

作者代码目标使用 `(1-p_target)^2`。简单代入 `(1-k/N)^2` 会带入有限采样偏差，因此采样方案使用无偏估计：

$$
\widehat D=1-2\frac{k}{N}+\frac{k(k-1)}{N(N-1)}.
$$

当 N=2，这个估计仅在“两个回答都不是目标类”时为 1，其他情况为 0。当前 M=16 的平均保真损失只能取 `0, 1/16, 2/16, …, 1`。因此 plus/minus 两张图的真实目标概率即使不同，也经常得到相同损失，产生零差分。它不意味着内部概率相同，也不证明达到驻点。更完整的梯度分析见[收敛审计](gradient-and-convergence.md)。

## 新增固定目标原生评分

`Scorer(mode="target_probability", target=<label>)` 使用真实返回的原始 logprob，并取 `exp(logprob)`。它需要一次模型请求；目标在优化之前选定，后续不变。`logprob_scope="reported"` 允许 backend 保留实际报告的部分证据；默认 `complete` 的完整候选分布评分仍要求所有类别，不能把 top-20 改称 1000 类 softmax。

缺失类别不补零，不对部分候选重新归一化，不把模型实际回答混成目标标签。结果明确记录 `observed_label`、原始 logprob、概率事件、已报告类别数和是否经过归一化。完整 `probability/log_margin`、部分原生 `target_probability` 与采样 `agreement` 不能互相替代。

2.5 的 `160` 实测拆成三个数字 token。因此评分为：

$$
\log s_{160}(x)=\log P(1\mid x)+\log P(6\mid x,1)+\log P(0\mid x,16).
$$

这里的 x 同时代表图像和固定的完整 prompt。不能使用 `avgLogprobs`，那是 token 平均值。`s` 是指定数字 token 路径的**前缀事件概率**，未包括 EOS，也未汇总所有可能的拼写或 tokenization；它不是正确率或完整语义类别后验。最后一步的候选共享实际测量的 `16` 前缀，因而可以得到已返回的 `160…169` 路径分数；不能把 `P(6|1)` 借给另一个未评估前缀，推算 `260` 的分数。

仅 `reported` scope 支持这种经过严格验证的数字路径：同宽数字编号、每位恰好一个数字 token、实际文本与 token 拼接一致、各步原始分数有效。其他多 token 格式拒绝。优化扰动中的目标若落到未评估的前缀，明确失败并保留已写出的 checkpoint；不会偷偷换成采样损失。最终 clean retained/removed 图缺失目标时保留 `null` 和未知原因，已完成的 mask 仍可保存；独立 held-out 类别采样继续进行。

## 真实分数敏感性探针

对先前 **3.5 Flash-Lite 八步 pilot 保存的图像**重新查询 2.5 Flash-Lite，仍提供完整 1000 类 prompt，温度为 1，三次均回答 Afghan hound：

| 图像 | 原始目标数字路径概率 | `(1-s)^2` |
|---|---:|---:|
| 原图 | 0.998447599 | 0.00000240995 |
| 保留图 | 0.998221982 | 0.00000316135 |
| 剩余图 | 0.995525289 | 0.00002002304 |

这三次实际 API 调用验证了“类别回答不变，仍能读取不同原生分数”。它们不是新的 2.5 mask 优化，也不是扰动损失的 Monte Carlo 平均。分数仍很接近 1，平方保真项仍可能饱和；原生分数解决频率离散化的一部分，不能保证解释必要性、真梯度、局部或全局最优。

## 运行接口

[原生八步配置](../configs/vertex-afghan-native-pilot.json)提供同一 dense mask、1000 类任务、16 张系数扰动图、作者代码的平方目标，关闭类别采样无偏估计。默认实验配置仍为用户指定的 3.5 Flash-Lite。新增配置尚未执行八步优化；可先查看预算：

```bash
python -m medshearletx experiment --config configs/vertex-afghan-native-pilot.json --dry-run
python -m medshearletx experiment --config configs/vertex-afghan-native-pilot.json
```

`experiment` 自动用独立类别采样选择目标，然后固定目标进行原生优化。原图/保留图/剩余图的最终 held-out 采样与原生优化分数分别写入 `reference/retained/removed.json` 和 `optimization_scores.json`；二者数值不混算。每步图像、mask、指标和浏览页面继续自动保存到固定 `runs/<run_name>-<timestamp>/` 结构。

通用 `probe/run` 也支持 `--scores target_probability --target "Afghan hound, Afghan"`，或者在配置顶层设置 `target`。已知 `supports_logprobs=false` 的单图原生实验在付费目标选择前拒绝执行。OpenAI-compatible/vLLM、Gemini、Vertex 的部署信息均独立保存在 model 配置中。
