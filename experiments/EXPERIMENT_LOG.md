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
| **Type5** | 确定性特征 | 由启发式规则确定（如 payload_size = f(function_code, direction)），不参与训练 | — |
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

### 逐特征 KS 演进（Test）

| 特征 | V1.0 | V2.0 | V2.5 | V2.6 | V2.8 | V2.8.1 | V2.8.2 | V2.8.3 | 方法 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|------|
| register_value_0 | 0.599 | 0.600 | 0.001 | 0.496 | 0.477 | 0.497 | **0.039** | **0.042** | ✅ 经验替换 |
| register_value_1 | 0.518 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | Type6 死特征 |
| register_value_2 | 0.617 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | Type6 死特征 |
| inter_arrival_ns | 0.526 | 0.254 | 0.560 | **0.313** | 0.321 | **0.126** | 0.157 | 0.202 | DDPM + log |
| payload_size | 1.000 | 0.475 | 0.364 | 0.369 | 0.350 | 0.535 | 0.580 | **0.271** | ✅ 3D 查找表修复 |
| register_address | 0.525 | 0.280 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | Type6 |
| quantity | 0.525 | 0.394 | 0.016 | 0.157 | 0.036 | 0.071 | 0.009 | 0.012 | Type6 StubSampler |

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

## 当前状态

### 已解决

| 问题 | 状态 | 版本 |
|------|:---:|------|
| payload_size 确定性计算 | ✅ | V2.0 |
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

### 待解决

| # | 问题 | 方案 |
|---|------|------|
| — | 暂无明确阻塞项。下一步：1M 全配置复训，验证 Mean KS < 0.08 | — |

---

> 最后更新：2026-06-24（V2.8.3）
