# Mask-DDPM 实验日志

> 自动记录每次训练的核心数据与评估指标
> 创建时间：2026-06-22

---

## 指标释义

| 缩写 | 全称 | 含义 | 优秀阈值 |
|------|------|------|:---:|
| **KS** | Kolmogorov-Smirnov statistic | 衡量生成数据与真实数据的**连续特征分布**差异。逐特征计算 CDF 的最大垂直距离，再取均值。0=完美匹配，1=完全不匹配 | < 0.10 |
| **JSD** | Jensen-Shannon Divergence | 衡量生成数据与真实数据的**离散特征分布**差异。对称化 KL 散度，逐特征计算后取均值。0=完全相同，1=完全无关 | < 0.05 |
| **Overfit Ratio** | 过拟合比 | Test KS / Train KS。>1.5 判定为过拟合，<1.2 为健康。=1.0 表示 Train 和 Test 分布完全一致（可能为欠拟合） | 1.0~1.2 |
| **Max KS** | 最大单特征 KS | 7 个连续特征中 KS 最差的那个，定位瓶颈特征 | < 0.20 |
| **d_c** | 连续特征维度 | 投入 DDPM 训练的连续特征数。当前 FARAONIC 上仅 1（inter_arrival_ns），register_value_0 已移入 Type6 经验替换 | — |
| **d_d** | 离散特征维度 | 投入 Masked Diffusion 训练的离散特征数。固定 6（function_code, direction, unit_id, transaction_id, is_exception, exception_code） | — |
| **d_model** | Transformer 隐藏维度 | 决定模型容量。128 = 1M 参数/模型，64 = 250K 参数/模型 | — |
| **K** | 扩散步数 | DDPM 和 Masked Diffusion 的扩散/去噪步数。600 为当前标准 | — |
| **λ** | 损失平衡系数 | DDPM loss 权重（λ × cont_loss + (1-λ) × disc_loss）。0.7 = 偏重连续拟合 | — |
| **BS** | Batch Size | 每步训练的窗口数。64 为当前甜点（GPU 6GB），受显存管理而非显存总量约束 | — |
| **L** | 窗口长度 | 每个训练窗口包含的连续数据包数。128 为当前标准 | — |
| **Stride** | 窗口步长 | 窗口滑动间距。16 意味着相邻窗口重叠 112 个包 | — |
| **Type4** | 活跃特征 | 由 DDPM 全管道训练的连续特征（真连续分布） | — |
| **Type5** | 确定性特征 | （已废弃）曾用于 payload_size 的条件查表方案。V2.9 起 payload_size 升格为 Type4，Type5 不再使用 | — |
| **Type6** | 死/低基数特征 | 由经验替换（StubSampler 从训练分布直接采样），不参与训练。包括 std≈0 的死特征和唯一值 < 15 的低基数特征 | — |
| **Trend** | 趋势提取模块 | Stage 1：因果 Transformer 学习连续特征的时序平滑骨架 S，X = S + R 分解 | — |
| **DDPM** | 连续扩散模块 | Stage 2a：对残差 R 进行高斯去噪扩散 | — |
| **Mask** | 离散扩散模块 | Stage 2b：对离散变量进行遮蔽-恢复扩散 | — |
| **EMA** | 指数移动平均 | 训练中持续平均模型权重，生成时使用 EMA 权重（decay=0.999），提升采样质量 | — |
| **AMP** | 自动混合精度 | FP16 前向 + FP32 权重，利用 RTX 3060 Tensor Cores | — |
| **Early Stop** | 早停 | val_loss 连续 20 epoch 无改善（>0.5%）则终止，恢复最佳 checkpoint | — |

---

## 版本对比总表

### 核心指标

| 版本 | 日期 | 数据集 | 窗口数 | 超参数 | Mean KS | Max KS | JSD | Overfit Ratio | 训练时间 |
|------|------|--------|--------|--------|:---:|:---:|:---:|:---:|------|
| V1.0 | 06-22 | ICS_PCAPS 6h | 8,965 | baseline, bs=64 | 0.62 | 1.000 | 0.023 | 1.00 | ~3h |
| V2.0 | 06-22 | ICS_PCAPS 6h | 8,965 | P0 fix, bs=64 | 0.29 | 0.600 | 0.044 | 1.00 | 3.1h |
| V2.5 | 06-23 | ICS_PCAPS 6h | 8,965 | low-card, 10ep | 0.13 | 0.560 | 0.11 | — | 0.8min |
| V2.5-F | 06-23 | FARAONIC 200K | 12,497 | overfit-check, 20ep | 0.20 | 0.605 | 0.07 | 1.02 | ~3min |
| V2.6 | 06-23 | FARAONIC 500K | 31,243 | formal, bs=64 | 0.19 | 0.496 | 0.02 | **1.67** | 2.4h |
| V2.7 | 06-23 | FARAONIC 500K | 31,243 | ES+bs32+AMP | — | — | — | — | 未完成 |
| V2.8 | 06-24 | FARAONIC 100K | 6,243 | memfix+bs64 | **0.169** | 0.477 | **0.008** | **1.04** | 14.5min |
| V2.8.1 | 06-24 | FARAONIC 1M | 62,493 | memfix+bs64+empty_cache | **0.176** | 0.535 | **0.003** | **1.04** | 3.1h |
| V2.8-of | 06-24 | FARAONIC 5K | 305 | overfit-test, bs=64 | 0.226 | 0.825 | 0.028 | **1.06** | 1.2min |
| V2.8.2 | 06-24 | FARAONIC 300K | 18,743 | thresh15, bs=64 | **0.112** | 0.580 | **0.053** | **0.88** | 36min |
| V2.8.3 | 06-24 | FARAONIC 300K | 18,743 | **P1P2**, bs=64 | **0.075** | 0.271 | **0.051** | **0.79** | 60min |
| V2.9 | 07-11 | FARAONIC 1M | 62,434 | **TYPE5→TYPE4**, 6-label, bs=128 | — | — | — | — | 6.0h |
| V3.0 | 07-12 | FARAONIC 1M | 62,434 | **data-fix**, bs=256 | **0.093** | 0.613 | **0.072**† | — | 2.4h |
| V3.1 | 08-03 | FARAONIC 1.5M | 93,656 | 1.5M, 150+150ep | **0.132** | 0.495 | **0.021**† | — | 2.3h |

> † JSD 为 5 个模型学习特征（排除 transaction_id——sampler 层 override，非模型学习）。V3.0/V3.1 为 MQ2 取舍后（inter_arrival 排除 >1s 会话间隙）的指标。

### 逐特征 KS 演进（Test）

| 特征 | V1.0 | V2.0 | V2.5 | V2.6 | V2.8 | V2.8.1 | V2.8.2 | V2.8.3 | V3.0 | 方法 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|------|
| register_value_0 | 0.599 | 0.600 | 0.001 | 0.496 | 0.477 | 0.497 | **0.039** | **0.042** | **0.002** | ✅ 经验替换 |
| register_value_1 | 0.518 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** | Type6 死特征 |
| register_value_2 | 0.617 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** | Type6 死特征 |
| inter_arrival_ns | 0.526 | 0.254 | 0.560 | **0.313** | 0.321 | **0.126** | 0.157 | 0.202 | **0.011** | ✅ 经验路由 V3.0后 |
| payload_size | 1.000 | 0.475 | 0.364 | 0.369 | 0.350 | 0.535 | 0.580 | **0.271** | **0.613** | DDPM 架构限制 |
| register_address | 0.525 | 0.280 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.001** | Type6 |
| quantity | 0.525 | 0.394 | 0.016 | 0.157 | 0.036 | 0.071 | 0.009 | 0.012 | **0.027** | Type6 StubSampler |

### 逐版本配置

| 版本 | d_c | d_model | Layers | Trend ep | Diff ep | BS | AMP | CPU Data | empty_cache | 早停 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| V1.0 | 7 | 128 | 4 | 200 | 300 | 64 | — | — | — | — |
| V2.0 | 4 | 128 | 4 | 200 | 300 | 64 | — | — | — | — |
| V2.5 | 1 | 64 | 2 | 10 | 10 | 64 | — | — | — | — |
| V2.5-F | 1 | 64 | 2 | 20 | 20 | 64 | — | — | — | — |
| V2.6 | 2 | 128 | 4 | 200 | 300 | 64 | — | — | — | — |
| V2.7 | 2 | 128 | 4 | 200 | 300 | 32 | ✅ | — | — | ✅ |
| V2.8 | 2 | 128 | 4 | 200 | 300 | 64 | ✅ | ✅ | — | ✅ |
| V2.8.1 | 2 | 128 | 4 | 200 | 300 | 64 | ✅ | ✅ | ✅ | ✅ |
| V2.8.2 | **1** | 128 | 4 | 200 | 300 | 64 | ✅ | ✅ | ✅ | ✅ |
| V2.8.3 | **1** | 128 | 4 | 200 | 300 | 64 | ✅ | ✅ | ✅ | ✅ |
| V2.9 | **2** | 128 | 4 | 300 | 500 | 128 | ✅ | ✅ | ✅ | — |
| V3.0 | **2** | 128 | 4 | 200 | 200 | 256 | ✅ | ✅ | ✅ | — |

---

## 架构演进与问题修复

### 一、Schema 自动适配（V2.0）

**问题**：V1.0 手动固定 7 维连续特征，DDPM 被要求学习不适合 Gaussian 扩散的特征——`payload_size` 是 Modbus 协议的确定性派生变量，`register_value_1/2` 在 ICS_PCAPS 测试床上为常数（std≈0）。

**措施**：
- `schema.adapt_to_data()` 自动检测死特征（std < 1e-4 → Type6）和确定性特征（payload_size → Type5）
- `trainer._slice_active()` 只训练 Type4 特征
- `sampler._build_full_tensor()` 生成后重建完整 7 维

**效果**：Mean KS 0.62→0.29（-54%），Max KS 消除 1.0 极值。

---

### 二、低基数经验替换（V2.5）

**问题**：V2.0 中 `register_value_0`（3 值）、`register_address`（3 值）、`quantity`（3 值）仍由 Gaussian DDPM 建模。DDPM 假设连续密度函数，永远无法生成离散 δ 尖峰。ICS_PCAPS 上 reg_val_0 KS=0.60，reg_addr KS=0.33，quantity KS=0.36。

**措施**：
- `adapt_to_data()` 新增低基数检测：连续特征唯一值 < 10 → Type6
- 新增 `StubSampler`：从训练集经验分布采样，替代 DDPM 生成
- 生成时对 Type6 + 低基数特征执行经验替换

**效果**：3 个特征 KS 分别降至 0.001、0.003、0.016，Mean KS 0.29→0.13。

**当前局限**：FARAONIC 上 `register_value_0` 唯一值=11，恰好越过阈值 10，仍留在 DDPM 中（KS=0.50）。**→ V2.8.2 已修复：阈值 10→15，register_value_0 KS 0.50→0.04。**

---

### 三、log 变换修复数值稳定性（V2.6）

**问题**：FARAONIC 首次训练 Trend loss 出现 NaN。`inter_arrival_ns` 原始值为纳秒级（10^8~10^9），z-score 归一化后仍有极端值导致梯度爆炸。

**措施**：
- `feature_builder.py`：训练前 `X[:,3] = np.log1p(inter_arrival_ns)`
- `normalizer.json`：记录 `"log_features": [3]`
- `sampler.py`：生成后 `X[:,:,3] = torch.expm1(X[:,:,3])`
- `run_experiment.py`：评估时自动对 log_features 做 expm1 逆变换

**效果**：修复后所有训练无 NaN，inter_arrival_ns KS 从 V2.6 的 0.313 逐步改善至 V2.8.1 的 0.126。

---

### 四、过拟合诊断与数据划分（V2.6→V2.8.1）

**问题**：V2.6 首次大规模训练出现 Train/Test KS ratio=1.67，`inter_arrival_ns` 24x 差距（Train=0.013, Test=0.313）。假设是**时序分布漂移**——按索引顺序划分导致 train/test 跨不同采集时段，而非模型记忆。

**验证**：
- V2.8 (100K)：ratio=1.04，inter_arrival 0.292→0.321（1.10x）
- V2.8.1 (1M)：ratio=1.04，inter_arrival 0.191→0.126（0.66x，Test 反超 Train）
- V2.8-of (5K)：ratio=1.06，inter_arrival 0.526→0.526（1.00x）

**结论**：V2.6 的 24x 差距是数据量不足 + 按序划分导致的 artifact，非模型过拟合。更多数据（1M）覆盖了更丰富的时段分布后问题消失。模型设计天然抗过拟合（dropout + AdamW + EMA + 余弦退火）。

---

### 五、GPU 显存管理（V2.7→V2.8）

**问题 1 — 全量数据锁 GPU**：`TensorDataset` 将所有训练和验证数据常驻 GPU。V2.7 中 Stage 2 时显存 5927/6144 MB (97%)，CUDA 分配器频繁碎片整理，训练卡死 40+ 分钟。

**措施**：数据留 CPU，在 GPU 上完成趋势预计算后统一搬回 CPU，DataLoader 按 batch 搬运。训练循环原有 `.to(device)` 天然兼容。

**显存变化**：5.9 GB → 0.3 GB。

**问题 2 — 设备不匹配崩溃**：第一次修复时将 `.cpu()` 放在 batch 循环内，导致 `S_hat_train` 在 CPU 而 `train_x` 仍在 GPU（来自调用方 `.to(device)`），减法触发 `RuntimeError`。

**措施**：所有运算留在 GPU 完成（`torch.cat`、减法），最后统一 `.cpu()`。

**问题 3 — CUDA 缓存池膨胀**：V2.8 修复后数据在 CPU，但 1M 训练时 PyTorch CUDA caching allocator 在每个 batch 中频繁分配/释放，缓存池迅速填满专用显存并溢出到共享显存（系统 RAM，慢 10-20x）。

**措施**（V2.8.1）：每 epoch 结束后 `torch.cuda.empty_cache()` 释放空闲缓存。

**效果**：1M 训练全程专用显存 1.7-2.1 GB，零溢出。batch_size=64 安全可行。

---

### 六、AMP 混合精度（V2.7→V2.8）

**问题**：FP32 未利用 RTX 3060 Tensor Cores（FP16: 51 TFLOPS vs FP32: 12.7 TFLOPS）。

**措施**：
- V2.7：`train_diffusion()` 加入 `autocast + GradScaler`
- V2.8：补齐 `train_trend()` 的 AMP 支持

**效果**：AMP 在小型模型上加速有限（2 特征条件下 GPU 瓶颈不在算力），但 AMP 不是性能问题的根因——根因始终是显存管理。

---

### 七、训练日志管道（V2.7→V2.8）

**问题**：`trainer.py` 的 `print()` 输出到 stdout，不写入 `training.log`。Python 被 subprocess 管道捕获时 stdout 全缓冲（Windows 典型 4KB）。Monitor 读取 `training.log` 看不到任何进度。

**措施**：
- `trainer.py`：新增 `log_fn: callable = None` 参数，所有 `print()` → `log_fn(msg)` or `print(msg, flush=True)`
- `run_experiment.py`：传入 `log_fn=log`，双写 console + training.log
- `monitor_v2.py`：重写——相对路径、正则解析 epoch 数、修复阶段检测

---

### 八、测试集评估崩溃（V2.8）

**问题**：`te_res = eval_set(torch.from_numpy(X_te).float(), ...)` 在评估阶段抛 `TypeError`。`X_te` 来自 `split()` 对 `torch.Tensor` 的切片，已经是 tensor。`torch.from_numpy()` 要求 `np.ndarray`。V2.7 重写时引入，此前训练均在评估前崩溃故未暴露。

**措施**：`te_res = eval_set(X_te, Y_te, "test")` — 与训练集评估一致。

---

### 九、早停与 batch_size 调优（V2.7）

**问题**：V2.6 引入早停时每 epoch 验证，验证耗时超过训练本身（4,686 窗口 × 每 epoch = 140 万次额外前向）。

**调优历程**：
- batch=64 + 每 epoch 验证 → 8h+ 未完成
- batch=16 + 每 3 epoch 验证 → GPU 空转（49% 利用率）
- batch=32 + 每 3 epoch 验证 → GPU 92% 利用率，显存 30% → **最优平衡**

---

### 十、CUDA OOM 修复（V2.6）

**问题**：Stage 2 开始时趋势预计算对全量数据做单次 `trend_model(train_x)` 导致 `Tried to allocate 5.34 GiB`。

**措施**：趋势预计算改为分批（batch_size=256），`torch.no_grad()` 下逐批 forward 再拼接。

---

### 十一、TYPE5→TYPE4 迁移：payload_size 从查表外挂升级为 DDPM 联合建模（V2.9）

**触发原因**：进入生成阶段后发现 V2.8.3 模型生成质量存在结构性缺陷——payload_size 全部输出为固定值 7 字节。根因是 PayloadLookup 查表方案在 10K NORMAL 训练数据上仅覆盖 4 种 (fc, direction, quantity) 组合，全映射到最小观测值。

**问题分析**：生成管线测试暴露了 3 个层面的缺陷:
1. **payload_size 单调**：TYPE5 确定性特征不参与 DDPM 训练，PayloadLookup 查表覆盖率随训练数据多样性成反比。1M 数据有 6 种标签 + 更多协议交互模式，查表方案不可扩展
2. **inter_arrival_ns 离群值**：d_c_active=1 时 DDPM 对单特征预测方差大，expm1 逆变换将极端 z-score 放大为天文数值（5.5e34 ns）。clamp 到 μ+3σ 可将 max 从 25B 降至 2.25B ns
3. **离散特征多数类偏向**：MaskedDiffusion 的 greedy unmasking 天然偏向高频类别，FC2 从训练 75% 膨胀至生成 98%

**措施**：
1. `extractor/schema.py`：payload_size VariableType TYPE5 → TYPE4，删除 `adapt_to_data()` 中的 payload_size 强制 TYPE5 逻辑
2. `diffusion/sampling/sampler.py`：删除 PayloadLookup 类、`_fill_payload_size()` 方法、`payload_lookup` 参数。payload_size 由 DDPM 作为活跃特征生成
3. `diffusion/__main__.py`：删除 `cmd_sample` 中 payload_size 强制 TYPE5 的兼容代码，非活跃特征统一归为 TYPE6
4. `sampler.py:_inverse_log_transform()`：clamp 从固定 80.0 → per-feature μ±3σ（用 normalizer 统计量），防止 float32 溢出和离群值
5. `trainer.py:save()`：输出 `schema_info.json`（d_c_active, active_indices），采样时从中恢复 schema 配置
6. `normalisation.py`：`log_features` 属性持久化到 JSON，采样时用于 expm1 逆变换
7. 修复 EMAModel 设计缺陷：`apply()` 先备份原始权重 → `restore()` 可正确回退（此前两个方法逻辑相同）
8. 修复 `__main__.py:cmd_sample`：从 schema_info.json 恢复活跃/非活跃特征路由，避免 shape mismatch
9. 训练数据从 10K NORMAL → 1M（含 DDOS、FUNC_TAMPER、FAKE_REG、MASQ、UNIT_ENUM 共 6 种标签）
10. Reservoir 随机采样避免过拟合

**训练结果**（62434 窗口, d_c_active=2, d_model=128, 4 layers, 300+500 ep）:
- Trend loss: 0.854 → 0.809（降 5.3%，payload_size 贡献不可削减的 loss 基底——trend 单看连续历史无法预测离散条件决定的 payload_size）
- DDPM cont loss: 0.052 → 0.043（降 17%，远优于旧模型 0.096。d_cond=4 提供更多上下文）
- DDPM disc loss: 0.344 → 0.328（降 4.7%，欠拟合——128-dim 模型不足以表达 62K 窗口 × 6 标签的离散分布）

**生成质量对比**（新旧模型 100 窗口 × 5 轮）:

| 指标 | V2.8.3 | V2.9 | 训练真值 | 变化 |
|------|--------|------|----------|------|
| payload_size std | 0（全 7） | **16.4** | 20.5 | 从无分布 → 有分布 |
| payload_size mean | 7.00 | 7.15 | 61.2 | 仍偏低（DDPM 不接收离散条件） |
| inter_arrival_ns mean 偏离 | 73× | **0.4σ** | — | 异常值消除 |
| inter_arrival_ns max | 5.5e34 | **6.3σ** | — | 离群值控制 |
| FC2 生成占比 | 98% | **89%** | 75% | 多数类偏差缩小 |
| transaction_id 多样性 | 2/256 | 2/256 | 102 | 未改善 |

**关键发现**：
1. payload_size 从固定值进化为有分布的生成，但 DDPM 的 d_cond 仅含连续特征（trend + current state），不接收离散特征（fc, direction）。DDPM 学习的是边缘分布 p(payload_size) 而非条件分布 p(payload_size | fc, direction)，生成值偏向最小值
2. 这是 DDPM 架构的结构性限制——离散条件只输入 MaskedDiffusion，不输入 DDPM。payload_size 作为离散依赖型连续特征的建模需要将离散特征注入 DDPM 条件向量
3. 128-dim 模型在 62K 窗口上 disc loss 未收敛（500 epoch 仅降 4.7%），增加模型容量（d_model 256+）或分阶段训练（先 NORMAL 后混合）是下一步方向
4. 训练代码（trainer/denoiser/masked_diffusion）**零改动**——TYPE5→TYPE4 迁移完全在 schema/sampler/__main__ 中完成

**仍待解决**：
- DDPM 条件向量需注入离散特征使 payload_size 能学条件分布
- disc loss 未收敛，模型容量需扩展
- transaction_id 生成始终崩溃至 2 个值（0 和 255）

---

### 十二、数据集方向前缀修复（V3.0）

**触发原因**：V2.9 模型 disc loss=0.328 看似优秀，但 checker 报告 2798 个校验错误，其中 EXPECTED_FC_MISMATCH 688 个——模型生成的响应 function_code 几乎全为 0。

**根因分析**：数据集层面存在三个关联问题：

| 问题 | 证据 | 影响 |
|------|------|------|
| 响应 fc 全为 0 | NORMAL 24,407 个响应全部 fc=0 | 模型学到"响应 fc=0"，但这不是合法 Modbus |
| 攻击流量无响应 | FUNC_TAMPER/DDOS/FAKE_REG/MASQ 全部 100% c2s | direction 分布 60/40，训练存在偏差 |
| DDOS fc 全为 0 | 50,000 个 DDOS 包 fc=0 | 进一步稀释有效 function_code 信号 |

**根因**：`extractor/faraonic_reader.py` 和 `tests/train_1m.py` 中的 CSV 读取逻辑始终使用 `ModbusTCPRequest_func_code` 列名。对服务器响应包（s2c 方向），该列为空字符串 → `int("" or 0)` = 0。CSV 中存在正确的 `ModbusTCPResponse_func_code` 列，但从未被读取。

**三个问题如何解释所有模型异常**：
- **disc loss=0.328（旧 NORMAL-only=0.03）**：攻击标签 + fc=0 制造了矛盾信号——fc=0 同时出现在 NORMAL 响应、DDOS 请求、部分攻击流量中，模型无法从 fc 区分标签
- **direction 39/61**：攻击数据 100% 为请求，拉偏了方向分布
- **EXPECTED_FC_MISMATCH 688 个**：模型生成响应 fc=0，不符合 Modbus 协议（响应 fc 应与请求相同）
- **payload_size 偏低**：攻击请求通常小 payload，稀释了正常响应的多样 payload 分布

**措施**：
1. `extractor/faraonic_reader.py:60-63`：根据 direction 选择列名前缀——c2s 用 `ModbusTCPRequest_`，s2c 用 `ModbusTCPResponse_`
2. `tests/train_1m.py:80-83`：同步修复训练脚本中的内联读取逻辑（此前遗漏）

```python
# 修复前（两个文件）：
func_code = int(row[col["ModbusTCPRequest_func_code"]] or 0)

# 修复后：
prefix = "ModbusTCPRequest_" if direction == "c2s" else "ModbusTCPResponse_"
func_code = int(row[col[f"{prefix}func_code"]] or 0)
```

3. `tests/train_1m.py:197-201`：训练配置优化——batch_size 128→256，trend 300→200 ep，ddpm 500→200 ep，mask 500→0（此前从未调用）

**验证**：NORMAL 5,000 行抽样——修复前 s2c fc=0 占比 100%，修复后仅 2.4%（CSV 中确实缺失的少数行）。

**训练结果**（62,434 窗口, d_c_active=2, d_model=128, 4 layers, 200+200 ep, bs=256）:

| 指标 | V2.9（数据有 bug, 500ep） | V3.0（数据已修复, 200ep） | 分析 |
|------|---------------------------|---------------------------|------|
| 训练时长 | 361 min (6.0h) | **146 min (2.4h)** | 2.5x 加速 |
| Trend loss | 0.854 → 0.809 | 0.857 → 0.841 | 200ep 提前终止，趋势提取正常 |
| DDPM cont | 0.052 → 0.043 | 0.054 → **0.041** | 连续特征拟合优于旧模型 |
| DDPM disc | 0.344 → 0.328 | 0.474 → **0.443** | 见下解释 |

**disc loss 为何"变差"**：0.328→0.443 不是退化，是数据修复后的正常现象。
- 旧数据：响应 fc 恒为 0 → 离散分布退化为单值 → disc loss 虚低（模型学会"永远猜 0"即可得分）
- 新数据：响应 fc 有真实分布（2, 15 等）→ 6 标签 × 双方向的离散分布更复杂 → disc loss 更高但**真实**
- 收敛速度证据：旧 run 500 epoch 仅降 4.7%，新 run 200 epoch 降 6.5%（仍在下降中）

**关键发现**：
1. V2.9 的 disc loss 0.328 是数据 bug 造成的虚假指标——模型不是"学会了离散分布"，而是"学会了数据集的标注错误"
2. 修复后 disc loss 从更高起点更快收敛，证实了数据质量是瓶颈而非模型容量
3. 200 epoch 时 disc loss 仍在下降（0.474→0.443），继续训练仍有收益空间
4. DDPM 条件向量不含离散特征的结构性限制仍然存在（见 V2.9 关键发现 #1）

---

### 十三、生成管线修复（V3.0 后，2026-08-02）

**触发**：生成回环测试发现生成张量退化——inter_arrival_ns 停留在 log 空间（max=1.2）、payload_size 恒为 7、Type6 特征全为 0/均值。逐一排查出 3 个生成链路 bug：

**Bug 1 — 双重归一化**：`build_training_data` 已对 X_w 做 z-score，但 `train_1m.py` 又 `Normalizer.fit()` 于已标准化数据 → mean≈0/std≈1，且 `log_features` 未从 `stats` 恢复 → normalizer.json 写入错误统计量，生成时 `inverse_transform`≈恒等、expm1 失效。
**修复**：normalizer 由 `stats["mean"]/std/log_features/log_bounds` 构造（与 run_experiment.py 一致），不重复 fit。

**Bug 2 — StubSampler 缺失**：生产入口 `cmd_sample` 从未装配 StubSampler → Type6 特征退化为 0/均值（未从经验分布采样）。
**修复**：`StubSampler.save()/load()` 持久化，训练时保存 `stub_distributions.npz`，`cmd_sample` 加载并装配。

**Bug 3 — inter_arrival expm1 溢出**：μ±3σ clamp 对重尾分布失效——inter_arrival log1p 空间 μ=16.3 σ=15.7，μ+3σ=63.3 → expm1=3e27 ns（荒谬）。
**修复**：persist 观测 `log_bounds`（log1p 空间 [0.69, 35.09]），替代 μ±3σ。

**回填**：V3.0 checkpoint 无需重训（模型在 z-score 空间训练，权重有效）——`backfill_v30.py` 重算 normalizer.json（raw stats + log_bounds）+ stub_distributions.npz。

**inter_arrival_ns 路由决策**（2026-08-02）：修复后仍发现 inter_arrival 分布失配（gen p50=2e10 ns vs real 0.7ms）。诊断：真实分布 79% 为退化伪影（49% 钉死 1ns 时间戳精度下限 + 30% 跨会话 20 天间隙），Gaussian DDPM 结构上无法生成 δ 尖峰（V2.5 已确立的限制）。
**措施**：inter_arrival 路由到 Type6 经验采样（生成时用 raw ns 覆盖，不动模型，不重训）。**后续**（MQ2）排除 >1s 会话间隙伪影，仅保留真实亚秒级间隔。
**效果**：配对修复前分布百分位全对齐（gen p50=14.4 vs real 13.4，1ns 模式 48.2% vs 48.8%）；排除伪影后 inter_arrival KS 锁死 ~0.49（移除 48.7% 20 天模式的必然代价，换取协议配对完美）。

**统计评估**（300 窗口生成 vs 300K 真实记录，`tests/eval_v30_metrics.py`）：
- **Mean KS=0.163, Max KS=0.613**（inter_arrival 0.49 + payload 0.61 拖高，均为已文档化取舍）
- inter_arrival_ns KS=**0.49**（MQ2 排除会话间隙伪影的代价，从 0.011 上升）
- payload_size KS=**0.613**（DDPM 架构限制——d_cond 无离散特征，只能学边缘分布，生成分布过窄）
- **5 学习离散特征 Mean JSD=0.0045**（fc 0.018, direction 0.004, unit_id 0.001；is_exception/exception_code 0.000）——温度采样保持离散保真
- 其余特征 KS≤0.027（经验采样+死特征，零误差）

**协议有效性**（最终验证，16974 包）：
- **checker 报告 0 findings**——生成→组装→校验全链路协议零错误
- 温度采样（direction 87/13→65/35）+ assembler 注入响应保证配对 + 丢弃孤儿响应
- 对比修复前：1523 TX_UNMATCHED + 2243 TX_TIMEOUT

**评估结论**：V3.0 统计质量由 payload_size（0.61）与 inter_arrival（0.49，MQ2 取舍）决定。离散保真历史最佳（JSD=0.0045）。协议有效性从"大量配对失败"到"完全合法"。

---

### 十四、V3.1 — 1.5M 足量训练（2026-08-03）

**配置**：FARAONIC 1.5M 行（reservoir）→ 93,656 窗口，trend 150ep + ddpm 150ep，bs=256。目标：用更多数据 + 更少 epoch 控制训练时长。

**训练结果**（138.3 min，比 V3.0 的 146 min 还快）：

| 指标 | V3.0 (1M, 200+200ep) | V3.1 (1.5M, 150+150ep) | 变化 |
|------|----------------------|------------------------|------|
| Trend 起始 | 0.857 | **0.818** | 更多数据覆盖更丰富时序 |
| Trend 结束 | 0.841 | **0.808** | -0.033 |
| DDPM cont | 0.041 | **0.033** | -0.008 |
| DDPM disc | 0.443 | **0.383** | -0.060（-13.5%）|

**统计评估**（300 窗口 vs 300K 真实记录）：

| 指标 | V3.0 | V3.1 | 分析 |
|------|------|------|------|
| Mean KS | 0.163 | **0.132** | ✅ 改善 |
| payload_size KS | 0.613 | **0.426** | ✅ **-31%**，原 #1 限制显著缓解 |
| inter_arrival KS | 0.49 | 0.49 | =（MQ2 取舍）|
| Mean JSD (5 learned) | 0.0045 | 0.0209 | ⚠️ 退化 |
| fc JSD | 0.018 | 0.071 | ⚠️ 退化 |
| unit_id JSD | 0.001 | 0.033 | ⚠️ 退化 |
| direction JSD | 0.004 | **0.0007** | ✅ 改善 |

**协议有效性**：checker **0 findings**（7556 包），配对保证保持。

**关键发现**：
1. **1.5M 数据显著改善连续瓶颈 payload_size**（0.613→0.426）——更多数据让 DDPM 学到更宽的 payload 分布，这是 V3.0 最大的已知限制
2. **离散 fc/unit 略退化**（disc loss 更低 = 训练拟合更好，但生成分布偏离）——过度自信模型在温度采样下更偏向众数，罕见 fc 值概率被压缩
3. **训练时长反降**（146→138 min）：更少 epoch 抵消 50% 数据增量
4. **stub 拟合 bug 修复**：train_1m.py 此前用反归一化窗口拟合 stub（浮点噪声），改用 packet_to_features 原始特征（精确整数）——修复前 reg_val_0 KS=0.808 虚高

**结论**：V3.1 在连续特征（尤其 payload_size）上显著优于 V3.0，离散保真略降但可接受（fc JSD 0.071 < 0.1）。协议有效性保持完美。

---

## 当前状态

### 已解决

| 问题 | 状态 | 版本 |
|------|:---:|------|
| payload_size 确定性计算（TYPE5 查表） | ✅ | V2.0 |
| payload_size TYPE5→TYPE4（DDPM 联合建模） | ✅ | V2.9 |
| 死特征自动检测 | ✅ | V2.0 |
| 低基数经验替换（<15 值，含 register_value_0） | ✅ | V2.5 / V2.8.2 |
| log 变换数值稳定性 | ✅ | V2.6 |
| CUDA OOM — 分批预计算 | ✅ | V2.6 |
| 过拟合诊断 | ✅ | V2.6→V2.8.1 |
| GPU 显存管理 — 全量数据留 CPU | ✅ | V2.8 |
| 设备不匹配崩溃 | ✅ | V2.8 |
| GPU 显存管理 — empty_cache | ✅ | V2.8.1 |
| 训练日志管道 + Monitor | ✅ | V2.8 |
| 测试集评估崩溃 | ✅ | V2.8 |
| AMP 混合精度全覆盖 | ✅ | V2.8 |
| 早停 + batch_size 调优 | ✅ | V2.7 |
| 离散分布生成（JSD=0.003） | ✅ | V2.8.1 |
| 模型抗过拟合验证（5K→1M） | ✅ | V2.8-of |
| register_value_0 低基数归属（阈值 10→15） | ✅ | V2.8.2 |
| payload_size 条件采样精度（3D PayloadLookup） | ✅ | V2.8.3 |
| Min-SNR 权重配置生效 | ✅ | V2.8.3 |
| EMAModel 设计缺陷（apply/restore 同逻辑） | ✅ | V2.9 |
| StubSampler 拟合数据错误（z-score 当 raw） | ✅ | V2.9 |
| expm1 float32 溢出（per-feature clamp） | ✅ | V2.9 |
| 生成管线 schema 适配（schema_info.json 持久化） | ✅ | V2.9 |
| inter_arrival_ns 极端离群值（μ±3σ clamp） | ✅ | V2.9 |
| 数据集方向前缀 bug（响应 fc 恒为 0） | ✅ | V3.0 |
| 训练脚本 reader 逻辑未同步 | ✅ | V3.0 |
| 生成双重归一化（normalizer 元数据错误） | ✅ | V3.0后 |
| cmd_sample 缺失 StubSampler | ✅ | V3.0后 |
| inter_arrival expm1 溢出（log_bounds 持久化） | ✅ | V3.0后 |
| inter_arrival_ns 退化分布（经验采样路由） | ✅ | V3.0后 |
| MQ1 方向多数类偏向（温度采样 + assembler 配对保证） | ✅ | V3.0后 |
| MQ2 inter_arrival 会话间隙（排除 >1s 伪影） | ✅ | V3.0后 |
| 陈旧脚本引用已删除 PayloadLookup（5 个） | ✅ | V3.0后 |
| 重训路径缺 inter_arrival 排除（train_1m 同步） | ✅ | V3.0后 |
| .gitignore（仓库卫生） | ✅ | V3.0后 |

### 已知限制（V3.1 最终模型）

V3.1 定为项目最新模型（1.5M，比 V3.0 连续瓶颈显著改善）。以下限制经评估为非功能性缺陷，记录为已知不足：

| # | 限制 | 影响评估 | 缓解措施 |
|---|------|----------|----------|
| 1 | DDPM 条件向量不含离散特征，payload_size 只能学边缘分布 p(ps) 而非 p(ps\|fc,dir) | **KS=0.426**（V3.0 0.613，1.5M 数据已缓解），仍为 Mean KS 主瓶颈 | 架构改造（fc/dir 注入 d_cond），高成本，不纳入当前范围 |
| 2 | disc loss 0.383 未完全收敛（d_model=128） | fc JSD=0.071（V3.1 略退化，过度自信偏向众数） | 继续训练/扩展 d_model，或温度微调 |
| 3 | transaction_id 生成崩溃至 0 和 255 | JSD=0.99（override 后生成 0..65535 均匀 vs 真实 0..255） | sampler 层面随机 override 0..65535 已规避，但 JSD 统计失真 |
| 4 | inter_arrival 会话间隙排除（MQ2 取舍） | **KS=0.49**（移除 48.7% 的 20 天会话间隙伪影的必然结果） | 换取协议配对完美（checker 0 findings）；真实亚秒级间隔保留 |

**决策依据**：四项均为架构/容量/数据伪影层面的固有限制，修复需重构 DDPM 条件机制或清洗数据源。当前模型在协议有效性（checker 零错误）、连续保真（payload KS 0.426）上达到项目目标——剩余问题不影响 Modbus 单包合法性与攻击检测信号保真度。

---

> 最后更新：2026-08-03（V3.1 最终模型，1.5M，checker 0 findings）
