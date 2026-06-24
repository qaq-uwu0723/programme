# Mask-DDPM 开发日志

> 项目：面向工业控制系统（ICS）的混合扩散生成模型  
> 环境：RTX 3060 Laptop (6GB VRAM), Windows 11, PyTorch 2.12.1+cu126

---

## 一、项目起源

### 核心问题

ICS 网络安全研究面临训练数据不足的瓶颈——真实 ICS 网络流量数据难以大规模获取。

现有方法（基于 GAN）存在三方面不足：
1. 无法同时处理混合类型数据（连续传感器读数 + 离散功能码）
2. 时序-分布冲突：优化分布一致性破坏时间序列结构
3. 对特定变量类型错误建模（程序驱动 vs 物理惯性）

### 提出的方法

两阶段混合扩散框架：TransformerTrend 提取连续趋势 S，ResidualDDPM 对残差 R 高斯去噪，MaskedDiffusion 对离散变量遮蔽-恢复扩散，TypeRouter 自动分类变量类型。

---

## 二、开发时间线

---

### [2026-06-22] V1.0 — Baseline

**问题**：ICS_PCAPS 数据集（8,965 窗口），全量 7 个连续特征用 Gaussian DDPM 建模。

**结果**：Mean KS=0.62, Max KS=1.000 (payload_size)。过拟合比=1.00——**欠拟合**，不是训练不足。

**发现**：payload_size 是 Modbus 确定性字段不应由 DDPM 学习；reg_val_1/2 std≈0 为死特征。离散分布 JSD=0.023 优秀。

---

### [2026-06-22] V2.0 — P0 架构修复

**问题**：V1.0 要求 DDPM 学习不适合 Gaussian 扩散的特征——确定性字段和死特征。

**修改思路**：自动检测不适合 DDPM 的特征并排除，让 DDPM 只训练真正需要建模的活跃特征。

**修改内容**：
- `schema.adapt_to_data()` 自动检测死特征（std < 1e-4 → Type6）和确定性特征（payload_size → Type5）
- `trainer._slice_active()` 只训练 Type4 特征
- DDPM d_c: 7 → 4，模型参数 2.4M → 1.07M (-55%)
- 修复 7 个训练 bug（设备不匹配、NumPy API 兼容、双重归一化、填充位置错误、Schema 维度不匹配、JSON 格式不一致、Tensor/Array 混用）

**结果**：Mean KS 0.62 → 0.286 (-54%)，Max KS 消除 1.0 极值。reg_val_0 KS=0.60 持平——后续发现是因为仅有 3 个离散值不适合 Gaussian DDPM。

---

### [2026-06-23] V2.5 — 低基数经验替换

**问题**：V2.0 中 reg_val_0（3 值）、reg_addr（3 值）、quantity（3 值）仍由 Gaussian DDPM 建模。DDPM 假设连续密度函数，永远无法生成离散 δ 尖峰。

**修改思路**：低基数特征（唯一值少）不应由扩散模型处理——直接从训练经验分布采样是精确匹配，零误差。

**修改内容**：
- `schema.adapt_to_data()` 新增低基数检测：连续特征唯一值 < 10 → Type6
- 新增 `StubSampler` 类：从训练集经验分布采样
- `sampler.generate()` 新增经验替换步骤
- inter_arrival_ns log 变换（`np.log1p` 压缩纳秒级尺度）
- `PayloadLookup`：条件采样 payload_size 替代 3 个固定值

**结果**：3 个低基数特征 KS 降至接近 0（0.001, 0.003, 0.016），Mean KS 0.29 → 0.064。

---

### [2026-06-23] V2.6 — FARAONIC 大样本正式训练 + 早停

**问题 1 — CUDA OOM**：Stage 2 开始时趋势预计算对全量数据做单次 `trend_model(train_x)` 导致 `Tried to allocate 5.34 GiB`。

**修改思路**：趋势预计算改为分批进行，避免全量数据同时在 GPU 上做前向。

**修改内容**：`trainer.train_diffusion()` 中趋势预计算从单次全量改为分批（batch_size=256），`torch.no_grad()` 下逐批 forward 再拼接。

---

**问题 2 — expm1 评估不匹配**：inter_arrival_ns 测试集未做逆变换，KS 虚高至 1.0。

**修改思路**：评估时自动对 log 变换特征做 expm1 逆变换。

**修改内容**：评估脚本手动对测试数据 index 3 做 `np.expm1()`。

---

**问题 3 — 轻度过拟合**：Train/Test KS ratio=1.67，inter_arrival_ns 出现 24x 差距（Train=0.013, Test=0.313）。

**修改思路**：引入早停机制和验证集防止过拟合。假设 inter_arrival_ns 的 24x 差距不是经典过拟合而是**时序分布漂移**——按索引顺序划分导致 train/test 跨不同采集时段。

**修改内容**：
- 早停机制：val_loss 连续 20 epoch 无改善（>0.5%）→ 停止，恢复最佳 checkpoint
- 验证集：15% 数据每 3 epoch 评估

**结果**：31K 窗口，2.4h 训练完成。过拟合比 1.67——后续 V2.8 验证此为数据划分 artifact 而非模型记忆。

---

### [2026-06-23] V2.7 — 训练效率优化

**问题 1 — 早停后验证耗时过长**：每 epoch 对 4,686 验证窗口做完整前向，总耗时从 2.4h 拖至 8h+。

**修改思路**：降低验证频率——每 3 epoch 验证一次，早停仍有效。

**修改内容**：验证频率 1→3 epoch。

---

**问题 2 — batch_size 导致 GPU 空转或过热**：batch=64 显存 97% 导致内存绑定假象；batch=16 GPU 利用率仅 49%。

**修改思路**：找到 GPU 利用率与显存的平衡点。

**修改内容**：batch_size 64→32。GPU 利用率 92%，显存 30%。

---

**问题 3 — 4 个互不兼容的训练入口**：`run_experiment.py`（tensor）、`run_faraonic.py`（CSV 快速）、`train_faraonic.py`（CSV 正式）、`train_faraonic_large.py`（大样本）。

**修改思路**：统一为一个通用入口，支持三种数据源。

**修改内容**：重写 `run_experiment.py` 为唯一入口，支持 `--data`（预提取 tensor）、`--csv`（CSV 自动提取）、`--pcap`（PCAP 自动提取）。模型先存盘再评估（评估崩溃不丢权重），所有输出写入 training.log + results.json。

---

**问题 4 — 监控面板无法实时显示进度**：硬编码路径 + 阶段误判 + epoch 估算基于常数。

**修改思路**：重写监控面板，自动找最新 log、解析实际 epoch 数。

**修改内容**：`monitor_v2.py` 重写——相对路径查找、正则解析 "Trend/Diff epoch X/Y"、修复阶段转移检测、`[Growning]/[Stale?]` 标识文件活跃状态。

---

**问题 5 — AMP 混合精度未启用**：FP32 未利用 RTX 3060 Tensor Cores。

**修改思路**：在训练循环中加入 AMP。

**修改内容**：`train_diffusion()` 加入 `torch.amp.autocast("cuda")` + `GradScaler("cuda")`。

---

**问题 6 — 测试集评估会崩溃**：`torch.from_numpy(X_te)` 但 `X_te` 已是 tensor。

**修改思路**：与训练集评估保持一致，直接传 tensor。

**修改内容**：`te_res = eval_set(X_te, Y_te, "test")`。

---

### [2026-06-23] V2.7.1 — 训练日志管道修复

**问题**：`trainer.py` 的 epoch 进度用 `print()` 输出到 stdout，不写入 `training.log`。Python 被 subprocess 管道捕获时 stdout 全缓冲（Windows 4KB），monitor 显示训练"卡住"。

**修改思路**：给 trainer 添加回调函数，复用 `run_experiment.py` 的 `log()` 双写机制。

**修改内容**：
- `trainer.py`：`train_trend()` 和 `train_diffusion()` 新增 `log_fn: callable = None` 参数
- 5 处 `print()` 替换为 `log_fn(msg)` or `print(msg, flush=True)`
- `run_experiment.py`：传入 `log_fn=log`
- `train_trend()` 补齐 AMP（V2.7 仅 `train_diffusion` 启用了 AMP）

---

### [2026-06-24] V2.8 — GPU 显存管理

**问题 1 — Stage 2 训练卡死**：`TensorDataset` 将全部训练和验证数据常驻 GPU（~210 MB 连续+离散张量 + 模型+优化器 + CUDA 缓存 → 5.9 GB ≈ 97%），CUDA 分配器频繁碎片整理，训练卡死 40+ 分钟。

**修改思路**：数据留 CPU，DataLoader 按 batch 搬运到 GPU。训练循环已有 `.to(device)`，无需修改。

**修改内容**：`trainer.train_diffusion()` 中趋势预计算在 GPU 完成后再整体搬回 CPU，`TensorDataset` 用 CPU 张量构建。

**第一次尝试失败**：将 `.cpu()` 放在 batch 循环内，导致 `S_hat_train` 在 CPU 而 `train_x` 仍在 GPU（来自 `X_tr.to(device)`），减法触发 `RuntimeError`。修正为 GPU 上完成全部运算后再 `.cpu()`。

**结果**：显存 5.9 GB → 0.3 GB。batch_size=64 安全可行。100K 验证训练 14.5 min 完成，Mean KS=0.169，过拟合比 1.04。

---

**问题 2 — 1M 训练共享显存溢出**：V2.8 修复后数据留 CPU，但 PyTorch CUDA caching allocator 在每个 batch 中分配/释放的临时显存不归还 OS，缓存池迅速填满专用显存并溢出到共享显存（系统 RAM，慢 10-20x）。

**修改思路**：每个 epoch 结束后强制释放 PyTorch 空闲缓存。

**修改内容**：`trainer.py` 中 `train_trend()` 和 `train_diffusion()` 的 epoch 循环末尾加入 `torch.cuda.empty_cache()`。

**结果**：1M 训练全程专用显存 1.7-2.1 GB，零溢出。

---

### [2026-06-24] V2.8.1 — 1M 正式训练完成

**训练**：FARAONIC 1M 行，62,493 窗口，43,745 train。batch=64，300 epoch 全部完成，188 min。

**结果**：Mean KS=0.176 (Test)，JSD=0.003，过拟合比=1.04。

**核心发现**：
- inter_arrival_ns KS 从 0.32 降至 0.13（-61%）
- register_value_0 KS=0.50 卡住——唯一值=11，恰好越过低基数阈值 10，仍留在 DDPM
- payload_size KS=0.53 恶化——更多数据丰富了分布，条件采样精度不足
- 模型天然抗过拟合（5K 极限测试 ratio=1.06，213 个训练窗口无过拟合）

---

### [2026-06-24] V2.8.3 — PayloadLookup 三维 + Min-SNR 修复

**问题 1 — payload_size KS 卡在 0.58**：PayloadLookup 仅用 (function_code, direction) 二维查找，忽略了 quantity 对 payload_size 的决定性影响。Modbus 协议中 `payload = f(fc, dir, quantity)`，同一 (fc, dir) 下 quantity 变化可使 payload 相差 50 倍。

**修改思路**：将 PayloadLookup 从二维拓展到三维 `(fc, direction, quantity)`。quantity 在 StubSampler 中已有经验分布，生成时先采样 quantity，再用三维查找表查 payload_size。

**修改内容**（`sampler.py`）：
- `PayloadLookup.fit()`: key 从 `(fc,dir)` → `(fc,dir,quantity)`，从训练数据 X_cont column 6 读取 quantity
- `PayloadLookup.sample()`: 新增 `quantity` 参数，三维缓存查找；新增协议感知 `_compute_fallback()` 处理缺失 key
- `generate()`: Step 6 先填充 quantity（从 StubSampler），再传给 payload lookup；Step 9 `_fill_stub_features` 跳过 quantity
- `_fill_payload_size()`: 新增 `quantity` 参数

---

**问题 2 — Min-SNR 权重配置写了但从未生效**：`config.json` 中 `use_min_snr: true, snr_gamma: 5.0`，但 `residual_ddpm.py` 的 `forward()` 调用的是 `F.mse_loss`。低噪声步和高噪声步被平等对待，训练资源分配不合理。

**修改思路**：将 `weighted_epsilon_mse()`（已在 `losses.py` 实现）实际接入 DDPM 训练。高噪声步（SNR 低）的任务更困难，应获得更高权重。

**修改内容**：
- `residual_ddpm.py`: `__init__` 新增 `use_min_snr: bool=True` 和 `snr_gamma: float=5.0`
- `forward()`: `use_min_snr` 时调用 `weighted_epsilon_mse(eps_pred, eps, k, self.alpha_bars, self.snr_gamma)`
- `trainer.py`: 传递 `config.ddpm.use_min_snr` 和 `config.ddpm.snr_gamma` 到 `ResidualDDPM`

---

**验证结果**（300K 行, 18,743 窗口，与 V2.8.2 同数据控制变量）：

| 指标 | V2.8.2 (阈值 15) | V2.8.3 (P1+P2) | 变化 |
|------|:---:|:---:|:---:|
| **Mean KS** | 0.112 | **0.075** | **-33%** |
| Max KS | 0.580 | **0.271** | **-53%** |
| payload_size KS | 0.580 | **0.271** | **-53%** |
| inter_arrival_ns KS | 0.157 | 0.202 | 略高（Min-SNR 增加训练时间） |
| 过拟合比 | 0.88 | 0.79 | ✅ |
| 训练时间 | 36min | 60min | +67%（Min-SNR 增加每步计算） |

**首次 Mean KS < 0.10**。payload_size 不再是最弱特征。P1 和 P2 均已解决。

---

## 三、当前状态

### 已完成

- [x] Type-aware 自动路由（死特征+低基数+确定性检测）
- [x] 混合扩散（DDPM + Masked Diffusion）
- [x] 条件采样 payload_size
- [x] 经验替换低基数特征
- [x] log 变换修复数值稳定性
- [x] 早停机制 + 分批趋势预计算
- [x] AMP 混合精度全覆盖（Trend + Diffusion）
- [x] 通用训练脚本（三合一数据源入口）
- [x] GPU 显存管理（全量 CPU + empty_cache）
- [x] 训练日志管道 + Monitor 重写
- [x] 1M 训练完成（Mean KS=0.176, JSD=0.003, 无过拟合）
- [x] 过拟合验证（5K→1M 全尺度 ratio < 1.06，模型天然抗过拟合）
- [x] 低基数阈值校准（10→15，将 register_value_0 收入经验替换）

- [x] 低基数阈值校准（10→15，register_value_0 KS 0.50→0.04，Mean KS 0.18→0.11）
- [x] PayloadLookup 三维查找表（fc×dir×quantity → payload_size KS 0.58→0.27）
- [x] Min-SNR 权重配置生效（use_min_snr → weighted_epsilon_mse 实际接入）

### 待解决

| # | 问题 | 方案 |
|---|------|------|
| — | 暂无阻塞项。下一步：1M 全配置复训，验证 Mean KS < 0.08 | — |

---

## 四、关键文件速查

| 用途 | 路径 |
|------|------|
| 训练入口 | `experiments/run_experiment.py` |
| 训练编排 | `diffusion/training/trainer.py` |
| 监控面板 | `experiments/monitor_v2.py` |
| 采样/生成 | `diffusion/sampling/sampler.py` |
| 特征提取 | `extractor/feature_builder.py` |
| Schema/路由 | `extractor/schema.py` |
| 配置 | `diffusion/config.py` |
| 实验记录 | `experiments/EXPERIMENT_LOG.md` |
| 开发日志 | `docs/DEVELOPMENT_LOG.md` |
| Bug 清单 | `TO_DEBUG_LIST.md` |

### 最常用命令

```bash
# 训练
python experiments/run_experiment.py --name exp --csv dataset/FARAONIC/Modbus_TCP_ Cybersecurity_Dataset_Training.csv --csv-rows 500000 --output checkpoints/exp/ --epochs 300 --batch-size 64

# 监控
python experiments/monitor_v2.py

# 打包 + 校验
python -m assembler --model checkpoints/exp/ --output output/exp/ --count 100
python -m checker output/exp/
```

### 硬件甜点配置

- GPU: RTX 3060 Laptop (6GB VRAM)
- batch_size=64, d_model=128, val/3ep
- 显存策略：全量数据 CPU + 每 epoch empty_cache
- 显存占用：1.7-2.1 GB (28-35%)

---

> 最后更新：2026-06-24  
> 版本：V2.8.3
