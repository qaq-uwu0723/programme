# TADS-ICS 项目综合评价

> 评估日期：2026-07-31  
> 评估范围：项目整体架构、训练阶段、生成阶段、协议校验/识别阶段，以及端到端工程闭环。

## 一、总体评价

当前项目已经不是简单的模型 demo，而是一个具备明确研究目标和系统分层的 ICS/Modbus 流量生成框架。整体成熟度可以评价为：

**研究原型较成熟，工程闭环仍需加固。**

项目主线清晰：

```text
PCAP/CSV -> Extractor -> Diffusion Model -> Sampler -> Assembler -> Checker
```

也就是说，项目已经覆盖了从原始流量读取、特征构建、模型训练、特征生成、协议组包到协议校验的完整链条。核心方法上，Type-aware 变量路由、趋势与残差分解、连续/离散混合扩散、低基数字段经验替换、PayloadLookup 协议补偿等设计都比较合理。

从已有实验结果看，`checkpoints/exp_p1p2_300k/results.json` 中 Test Mean KS 约为 `0.0752`，已经低于项目日志中反复提到的 `< 0.10` 目标，说明当前方法在既定数据和实验设置下确实达到过预期效果。

不过，如果项目目标是成为一个稳定可复现、可扩展、可端到端运行的训练生成系统，目前还存在若干工程契约和兜底机制缺口。

## 二、主要优点

### 1. 架构分层清楚

项目按功能拆成以下模块：

| 模块 | 作用 |
|---|---|
| `extractor/` | PCAP/CSV 到训练张量，负责特征提取、窗口化、schema 定义 |
| `diffusion/` | 核心模型，包括 Trend Transformer、Residual DDPM、Masked Diffusion、TypeRouter |
| `assembler/` | 将生成特征还原为 Modbus/TCP PCAP 和 JSONL sidecar |
| `checker/` | 对生成 PCAP 进行协议、事务和期望字段校验 |
| `experiments/` | 统一实验入口、训练日志、监控脚本 |
| `docs/` | 开发日志和实验总结 |

这种分层方式比较适合后续扩展。例如，如果未来要替换模型、增加协议、增强 checker，都可以在对应模块中局部推进。

### 2. Type-aware 设计是项目亮点

项目没有把所有字段都盲目交给高斯扩散模型，而是通过 `FeatureSchema` 和 `TypeRouter` 对变量进行分类：

| 类型 | 含义 | 当前处理方式 |
|---|---|---|
| Type4 | 真正需要建模的连续/离散变量 | 连续字段走 Trend + DDPM，离散字段走 Masked Diffusion |
| Type5 | 协议确定性或派生变量 | 用规则或条件查找表生成 |
| Type6 | 死特征、低基数字段、辅助字段 | 用经验分布采样或默认值替代 |

这个设计非常适合 Modbus/ICS 流量，因为很多字段并不是普通连续随机变量。例如 `payload_size` 本质上由 function code、direction、quantity 决定；低基数字段也不适合用 Gaussian DDPM 学习。

### 3. 训练方法完整且有针对性

训练阶段采用两阶段思路：

1. `TransformerTrend` 学习连续特征的时序趋势 `S`。
2. `ResidualDDPM` 学习残差 `R = X - S`。
3. `MaskedDiffusion` 学习离散 token 的条件分布。

这种拆分避免了单一模型同时处理趋势、残差、离散字段和协议约束的困难。它在建模思路上比简单 GAN、普通 DDPM 或全字段扩散更贴合 ICS 流量特征。

### 4. 实验记录较扎实

`experiments/EXPERIMENT_LOG.md` 和 `docs/DEVELOPMENT_LOG.md` 记录了从 V1.0 到 V2.8.3 的演进，包括：

- payload_size 从 DDPM 中剥离；
- 死特征自动检测；
- 低基数字段经验替换；
- inter_arrival_ns 的 log1p 变换；
- GPU OOM 和显存缓存问题修复；
- PayloadLookup 从二维扩展到三维；
- Min-SNR loss 权重真正接入训练。

这些日志说明项目不是一次性堆代码，而是在持续根据实验结果修正模型假设。

### 5. 协议校验意识较强

checker 不只是计算 KS/JSD，还会检查：

- PCAP 与 JSONL sidecar 是否对齐；
- TCP payload 是否为空；
- MBAP 长度、protocol id 是否正确；
- PDU 是否能按 Modbus descriptor 解析；
- 请求和响应是否能按 transaction_id、unit_id 配对；
- 实际字段是否匹配 metadata 中的 expected 字段。

这对“生成数据是否能成为真实协议流量”非常关键。

## 三、主要缺点和风险

### 1. 工程复现性不足

README 中提到 `requirements.txt`，但当前仓库实际没有该文件。当前默认 Python 环境缺少 `torch` 和 `scapy` 时，`python -m diffusion --help`、`python -m extractor --help` 都无法运行。

这意味着项目虽然有训练代码和 checkpoint，但新环境下不能直接复现。

### 2. 训练产物契约不完整

训练时会根据数据执行 `schema.adapt_to_data()`，动态决定哪些字段是 Type4、Type5、Type6。但 adapted schema 没有作为正式 checkpoint 产物保存。

目前 checkpoint 主要保存：

- `trend_model.pt`
- `ddpm_model.pt`
- `mask_diff_model.pt`
- `ddpm_ema.pt`
- `normalizer.json`
- `results.json`
- `training.log`

但还缺少：

- `schema.json`
- `resolved_config.json`
- `feature_transform.json`
- `payload_lookup.json`
- `stub_sampler.json`
- 数据集指纹和训练数据统计

这会导致一个问题：训练时用的是 adapted schema，但后续独立采样时可能又用 default schema，进而造成路由、模型维度和后处理逻辑不一致。

### 3. CLI 与 README 不完全同步

README 中的部分命令与当前代码入口不一致。例如 assembler 当前需要 `--data` 指向包含 `gen_X.npy` 和 `gen_Y_*.npy` 的目录，而 README 中写的是 `--model`、`--count` 形式。

此外，`python -m diffusion train` 和 `experiments/run_experiment.py` 对训练数据归一化的假设不完全一致，可能存在二次归一化风险。当前更可靠的入口是 `experiments/run_experiment.py`。

### 4. 生成空间和组包能力不完全对齐

默认 schema 的 function code 词表包含：

```text
1, 2, 3, 4, 5, 6, 8, 11, 15, 16, 17, 43
```

但 assembler 目前主要完整支持：

```text
3, 6, 16
```

如果模型生成 FC 1、2、4、5、8、11、15、17、43，assembler 会回退成 FC3 或用较简单逻辑处理。这可能造成：

- function_code 语义偏移；
- metadata expected function code 与实际包不一致；
- checker 报 expected mismatch；
- 生成流量统计上看似合理，但协议语义被削弱。

### 5. 事务级生成兜底不足

当前 direction、transaction_id、unit_id、function_code 很大程度来自模型生成结果。checker 能检查请求/响应是否匹配，但 assembler 生成阶段没有强制状态机保证：

- 请求后必须出现匹配响应；
- response 必须继承 request 的 transaction_id、unit_id、function_code；
- FC6/FC16 response 必须 echo request 字段；
- exception 只能出现在 server-to-client 方向。

这属于“事后发现问题”，不是“生成时预防问题”。

### 6. PayloadLookup 和 StubSampler 未完全产品化

`PayloadLookup` 和 `StubSampler` 是当前结果提升的重要原因，但它们主要在实验脚本评估阶段从训练数据临时构建，并没有作为标准模型产物持久化。

如果用户只有 checkpoint，没有训练数据，就无法重建同样的后处理分布，生成质量可能下降。

### 7. checker 更偏协议验证，不是完整识别系统

如果“识别”指协议校验，当前 checker 已有基础。如果“识别”指异常检测、IDS 评估、生成流量可用性判别，那么项目还缺一个真正的下游识别模块。

当前项目已有：

- 协议合法性 checker；
- KS/JSD/Lag-1 统计指标；
- 实验日志。

但还没有：

- downstream detector；
- anomaly recognition benchmark；
- no-sidecar 模式下的纯 PCAP 识别；
- 与真实攻击/正常流量的识别效果对比。

## 四、训练阶段评价

训练阶段本身是项目中相对成熟的一环。已有实验说明：

- 趋势模型 loss 能下降；
- 连续扩散 loss 能下降；
- 离散 masked diffusion loss 能下降；
- 300K P1P2 实验 Test Mean KS 已达到 `< 0.10`；
- 离散分布 JSD 在部分实验中表现很好；
- GPU OOM、缓存膨胀、AMP、早停等训练工程问题已经经历过修复。

不过，训练阶段仍有两个关键风险：

1. 300K 实验已经达标，但 1M 全配置复训尚未有最终结果支撑。
2. 训练入口不统一，`diffusion.__main__` 与 `experiments/run_experiment.py` 的数据处理契约需要收敛。

结论：训练方法有希望达到预期，且已有局部实验证据支持；但要形成稳定结论，还需要补一次标准环境下的可复现训练。

## 五、生成阶段评价

生成阶段的设计是合理的，但兜底还不够强。

当前 sampler 会执行：

1. 趋势 rollout；
2. DDPM residual sampling；
3. masked diffusion 生成离散字段；
4. 重建完整连续特征；
5. 填充 quantity；
6. 通过 PayloadLookup 填充 payload_size；
7. 反归一化；
8. 对 log 特征做 expm1；
9. clamp 到合法范围；
10. 用 StubSampler 替换低基数/死特征。

这些步骤已经比普通生成器稳健很多。但仍建议增加一层显式 `ProtocolRepair`，把生成结果修成协议一致的事件序列，再交给 assembler。

## 六、识别/校验阶段评价

checker 的协议验证框架有基础，适合作为生成后质量控制模块。但目前它依赖 sidecar metadata，更多是“生成产物一致性检查”，而不是纯 PCAP 识别器。

建议后续拆成两类能力：

1. `ProtocolChecker`：继续负责协议合法性、事务配对、expected 字段一致性。
2. `TrafficEvaluator` 或 `DetectorEvaluator`：负责统计相似度、下游识别效果、异常检测效果。

这样项目中的“识别”概念会更清楚。

## 七、改进方向

### P0：工程闭环优先

1. 增加 `requirements.txt` 或 `pyproject.toml`。
2. 保存并加载 adapted schema。
3. 保存 resolved config、normalizer、feature transform 信息。
4. 将 PayloadLookup 和 StubSampler 持久化为 checkpoint 产物。
5. 统一训练入口，明确 `train_X.npy` 是 raw 还是 normalized。
6. 更新 README，使命令与当前 CLI 一致。

### P1：协议生成兜底

1. 增加 `ProtocolRepair` 层。
2. 强制请求/响应成对。
3. response 继承 request 的 transaction_id、unit_id、function_code。
4. 对 FC3、FC6、FC16 的 request/response 字段做强一致修复。
5. 限制 function code 词表到 assembler 支持范围，或补齐所有 function code builder。
6. 在生成报告中统计 repair_rate、fallback_rate、clamp_rate、unsupported_fc_rate。

### P2：评估与识别增强

1. 增加一条端到端脚本：

```text
checkpoint -> sample tensors -> assemble PCAP -> checker -> metrics report
```

2. 增加 no-sidecar checker 模式，仅基于 PCAP 判断协议合法性。
3. 增加 downstream IDS/anomaly detector 评估。
4. 增加按 function code、direction、quantity 分组的 KS/JSD 指标。
5. 增加事务级指标，例如 request-response match rate、timeout rate、exception rate。

### P3：研究扩展

1. 支持更多 Modbus function code 的完整 PDU 生成。
2. 支持多 flow、多 client、多 server 场景。
3. 增加条件生成能力，例如指定 function code 分布、攻击/正常标签、设备 ID。
4. 增加长序列生成稳定性评估。
5. 对比 GAN、VAE、TimeGAN、普通 DDPM 等 baseline。

## 八、建议的下一步路线

建议不要立刻继续堆模型，而是先完成工程闭环：

1. 修复依赖和 README。
2. 保存完整 checkpoint 契约。
3. 实现 `ProtocolRepair`。
4. 统一一条端到端生成和校验命令。
5. 用当前 P1P2 配置做一次标准复现实验。

完成这些之后，项目会从“可以跑出好结果的研究原型”升级为“可复现、可扩展、可交付的完整系统”。

## 九、最终结论

当前项目的核心思想是成立的，尤其 Type-aware 路由、低基数经验替换、PayloadLookup 和协议 checker 都是非常有价值的设计。它目前最强的是研究思路和实验验证，最弱的是工程契约和端到端兜底。

如果目标是论文复现或研究验证，当前项目已经比较接近成熟。如果目标是稳定训练、稳定生成、稳定校验，并支持他人复现，那么还需要优先补齐 checkpoint 契约、CLI 一致性、协议修复层和端到端评估脚本。
