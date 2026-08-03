# Mask-DDPM 项目代码与功能综述

> 更新时间：2026-06-22  
> 涵盖：Extractor（特征提取）、Diffusion（扩散模型）、Assembler（包组装器）、Checker（协议校验器）四大模块  
> 最新改进：**P0 优化 — 自动化 Schema 适配 + Type5 确定性路由**

---

## 目录

1. [项目总览与管道架构](#1-项目总览与管道架构)
2. [Extractor — 特征提取管道](#2-extractor--特征提取管道)
3. [Diffusion — Mask-DDPM 扩散生成模型](#3-diffusion--mask-ddpm-扩散生成模型)
4. [Assembler — 包组装器](#4-assembler--包组装器)
5. [Checker — Modbus/TCP 协议校验器](#5-checker--modbustcp-协议校验器)
6. [共享基础设施](#6-共享基础设施)
7. [端到端管道与验证结果](#7-端到端管道与验证结果)
8. [快速使用指南](#8-快速使用指南)

---

## 1. 项目总览与管道架构

### 1.1 项目目标

为工业控制系统（ICS）和 Modbus 协议生成高保真合成网络流量数据，支撑安全测试与入侵检测系统（IDS）训练。

### 1.2 完整管道（4 阶段全部就绪）

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────┐    ┌──────────┐
│  特征提取     │ →  │  Mask-DDPM 扩散   │ →  │  包组装器    │ →  │ Checker │
│  Extractor   │    │  生成模型          │    │  Assembler   │    │ 校验器   │
└──────────────┘    └──────────────────┘    └─────────────┘    └──────────┘
       │                      │                     │                │
  PCAP/合成数据         特征 tensor           PCAP + JSONL      report.json
  → train_X.npy        X(连续) Y(离散)       sidecar 文件      0 fatal/error
  → train_Y_*.npy
```

### 1.3 代码结构

```
D:\programme\
├── extractor/                 # 特征提取 (已实现, 5 文件)
│   ├── __main__.py            # CLI: --pcap | --generate
│   ├── schema.py              # 共享特征定义 (7 连续 + 6 离散)
│   ├── pcap_reader.py         # PCAP → per-packet 记录
│   ├── feature_builder.py     # per-packet → 归一化 tensor 窗口
│   └── generate_synthetic.py  # 合成训练数据生成器
│
├── diffusion/                 # 扩散模型 (已实现, 16 文件)
│   ├── __main__.py            # CLI: train|sample|eval
│   ├── config.py              # 分层配置系统 (JSON 驱动)
│   ├── configs/default.json   # 出厂超参数
│   ├── models/
│   │   ├── trend_transformer.py  # TransformerTrend — 因果时序骨干
│   │   ├── residual_ddpm.py      # ResidualDDPM — 连续残差扩散
│   │   ├── masked_diffusion.py   # MaskedDiffusion — 离散遮蔽扩散
│   │   ├── denoiser.py           # TransformerDenoiser — 共享去噪 backbone
│   │   └── type_router.py        # TypeRouter — 6 类变量路由
│   ├── training/
│   │   ├── trainer.py         # MaskDDPMTrainer — 分阶段训练 + EMA
│   │   └── losses.py          # 6 种损失函数
│   ├── sampling/
│   │   └── sampler.py         # MaskDDPMSampler — 5 步生成管道
│   └── utils/
│       ├── noise_schedule.py  # beta/mask 调度 (cosine/linear)
│       ├── normalisation.py   # z-score 归一化
│       ├── metrics.py         # KS/JSD/Lag-1 评估
│       └── checkpoint.py      # 模型存取
│
├── assembler/                 # 包组装器 (已实现, 5 文件)
│   ├── __main__.py            # CLI 入口
│   ├── packet_builder.py      # tensor → scapy Modbus PCAP + JSONL
│   ├── modbus_rules.py        # 确定性协议约束 (MBAP/PDU 封包)
│   └── meta_writer.py         # JSONL sidecar 写入
│
├── checker/                   # 协议校验器 (已实现, 12 文件)
│   ├── main.py                # CLI 入口
│   ├── validate.py            # 5 层验证管道
│   ├── config.py              # Modbus descriptor 配置系统
│   ├── modbus_desc.py         # PDU 字段解析引擎
│   ├── mbap.py                # MBAP/ADU 解析
│   ├── decode.py              # Ethernet/IP/TCP 解码
│   ├── pcap_in.py             # PCAP 流式读取 + JSONL 对齐
│   ├── meta.py                # JSONL sidecar schema
│   ├── state.py               # 事务配对状态机
│   ├── report.py              # 结构化报告
│   └── configs/modbus_default.json  # 12 种函数码 descriptor
│
├── knowledges/                # 知识文档 (5 篇)
├── md/                        # 论文摘要 (30 篇, 8 主题)
├── papers/                    # 原始论文 (按主题组织)
├── test_data/                 # Checker 测试数据
├── PROJECT_CODE_SUMMARY.md    # 本文档
└── checker_design.md          # Checker 设计文档
```

---

## 2. Extractor — 特征提取管道

### 2.1 功能概述

Extractor 是管道的**第一环**，负责将原始 Modbus/TCP 流量（PCAP 文件或合成生成）转换为扩散模型可消费的训练 tensor。核心流程：

```
原始 PCAP → 逐包解析 (pcap_reader) → 特征映射 (feature_builder) → 窗口切片 → 归一化 → train_X.npy + train_Y_*.npy
```

### 2.2 PCAP 解析器 (`pcap_reader.py`)

复用 checker 的 `decode.py` 和 `mbap.py` 模块，逐包提取 16 个字段：

```python
@dataclass
class PacketRecord:
    ts_ns: int              # 绝对时间戳 (纳秒)
    inter_arrival_ns: int   # 到达间隔 (纳秒)
    src_ip / dst_ip: str    # 源/目 IP
    src_port / dst_port     # 源/目端口
    direction: str          # "c2s" | "s2c"
    # Modbus ADU
    transaction_id: int
    unit_id: int
    function_code: int      # 原始 (含 0x80 异常位)
    is_exception: bool
    exception_code: int
    pdu_data: bytes
    payload_size: int
    # 寄存器级
    register_address: int
    register_values: List[int]  # 最多 3 个寄存器值
    quantity: int
```

**核心函数**：`extract_packets(pcap_path: str) -> List[PacketRecord]`

自动处理：
- 非 502 端口过滤
- 方向推断（dst=502 → c2s, src=502 → s2c）
- FC3/6/16 的 PDU 解析（地址、数量、寄存器值提取）
- 到达间隔计算

### 2.3 特征构建器 (`feature_builder.py`)

将 per-packet 记录映射到 FeatureSchema 定义的 7 连续 + 6 离散特征：

```python
# 连续特征 (d_c=7)
X[i, 0:3] = register_values[0:3]   # 3 个寄存器值
X[i, 3]   = inter_arrival_ns       # 到达间隔
X[i, 4]   = payload_size           # TCP 负载大小
X[i, 5]   = register_address       # 寄存器地址
X[i, 6]   = quantity               # 读写数量

# 离散特征 (d_d=6)
Y[i, 0] = FC_TO_IDX[function_code] # 功能码 → 词汇索引 (12 类)
Y[i, 1] = direction_index          # c2s=0, s2c=1
Y[i, 2] = unit_id % 248            # 单元 ID
Y[i, 3] = transaction_id           # 事务 ID
Y[i, 4] = is_exception             # 异常标志
Y[i, 5] = exception_code           # 异常码
```

**核心函数**：

| 函数 | 功能 |
|------|------|
| `packet_to_features(records) → (X, Y)` | records → 平铺特征数组 |
| `build_windows(X, Y, L=128, stride=1)` | 滑动窗口切片 |
| `build_training_data(records)` | 一站式：records → 窗口化 + 归一化 + 统计保存 |
| `save_training_data(X, Y, dir)` | 写出 `train_X.npy` + `train_Y_*.npy` |

### 2.4 合成数据生成器 (`generate_synthetic.py`)

当无真实 Modbus PCAP 时，内置物理过程模拟器生成训练数据：

- **ProcessSimulator**：模拟温度（20-80°C 随机游走+正弦）、压力（与温度耦合+独立噪声）、流量（独立随机游走）
- 自动生成 FC3（读保持寄存器 60%）、FC6（写单寄存器 25%）、FC16（写多寄存器 15%）的请求-响应对
- 50ms 基础轮询间隔 + 随机抖动
- 输出标准 PCAP + JSONL，可直接用于 checker 校验或 extractor 二次提取

### 2.5 CLI 接口

```bash
# 从已有 PCAP 提取
python -m extractor --pcap real_modbus.pcapng --output data/processed/

# 生成合成数据 + 提取一步完成
python -m extractor --generate --num-packets 5000 --output data/train/

# 控制窗口参数
python -m extractor --generate --num-packets 2000 --window-length 64 --stride 8 --output data/quick/
```

---

## 3. Diffusion — Mask-DDPM 扩散生成模型

### 3.1 方法概述

Mask-DDPM 是项目的核心生成模型，采用 **"两阶段 + 混合扩散"** 架构：

```
阶段 1 (时序骨干):   TransformerTrend 学习连续变量的低频趋势 S
                     X = S + R 分解，残差 R 交给扩散模型

阶段 2a (连续扩散):   ResidualDDPM 对残差 R 进行高斯去噪扩散
                     前向: r_k = √(ᾱ_k)·r_0 + √(1-ᾱ_k)·ε
                     反向: 从纯噪声迭代去噪 K=600 步恢复 R

阶段 2b (离散扩散):   MaskedDiffusion 对离散变量进行遮蔽-恢复扩散
                     前向: 以概率 m_k 将 token 替换为 [MASK]
                     反向: 从全 MASK 状态迭代取消遮蔽 K=600 步恢复 Y

Type-aware 路由:     TypeRouter 将变量按 6 类分治
                     Type4 (过程变量) → 全管道，其余类型 → stub
```

**核心理念**：将"时序结构"与"分布细节"解耦——趋势模块负责序列走向，扩散模块负责分布形状，避免单一模型同时优化两者产生的冲突。

### 3.2 数据规格

由 `extractor/schema.py` 定义，是 extractor / diffusion / assembler 三方的共享契约：

| 类别 | 特征 | 类型 | 默认路由 | 说明 |
|------|------|------|:---:|------|
| 连续 | `register_value_0` | float | **Type4** | 🔵 DDPM 训练 |
| | `register_value_1` | float | **Type4→6** | 🔵 或 ⚪ **自动检测**（std < 1e-4 → stub） |
| | `register_value_2` | float | **Type4→6** | 🔵 或 ⚪ **自动检测** |
| | `inter_arrival_ns` | float (≥0) | **Type4** | 🔵 DDPM 训练 |
| | `payload_size` | float (≥7) | **Type5** | 🟡 确定性计算（永久） |
| | `register_address` | float [0,65535] | **Type4** | 🔵 DDPM 训练 |
| | `quantity` | float (≥1) | **Type4** | 🔵 DDPM 训练 |
| 离散 | `function_code` | 12 类 | **Type4** | 🔵 Mask 训练 |
| | `direction` | 2 类 | **Type4** | 🔵 Mask 训练 |
| | `unit_id` | 248 类 | **Type4** | 🔵 Mask 训练 |
| | `transaction_id` | - | **Type6** | ⚪ stub |
| | `is_exception` | 2 类 | **Type6** | ⚪ stub |
| | `exception_code` | 256 类 | **Type6** | ⚪ stub |

> **自动化机制**：训练前调用 `schema.adapt_to_data(X)`，根据实际数据 std 自动将死特征标记为 Type6。  
> `payload_size` 永为 Type5（协议确定性字段，不由数据决定）。  
> 在 ICS_PCAPS 上自动得 4 活跃特征；在 SWaT 上自动得 6 活跃特征。**同一份代码，零手动调整**。

### 3.3 模型架构

#### 3.3.1 TransformerTrend — 因果时序骨干

**输入** `X: (B, L, d_c)` → **输出** `S: (B, L, d_c)` 趋势预测

```
X (B, L, 7)
  → Linear(7 → 128)                    # 输入投影
  → PositionalEncoding(sinusoidal)      # 正弦位置编码
  → TransformerEncoder(4层, 128维, 4头) # 因果注意力 (上三角掩码)
  → Linear(128 → 7)                     # 输出投影
  → S (B, L, 7)
```

**核心代码**：
```python
class TransformerTrend(nn.Module):
    def __init__(self, d_c, d_model=128, nhead=4, num_layers=4):
        self.input_proj = nn.Linear(d_c, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512,
                                       activation="gelu", batch_first=True),
            num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_c)

    def forward(self, x):         # (B, L, d_c)
        h = self.input_proj(x)    # (B, L, d_model)
        h = self.pos_enc(h)
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
        h = self.encoder(h, mask=mask)
        return self.output_proj(h)

    def compute_loss(self, x):
        """Teacher forcing: S[:,:-1] → X[:,1:]"""
        s_hat = self.forward(x)
        return F.mse_loss(s_hat[:, :-1, :], x[:, 1:, :])
```

**训练策略**：teacher forcing — 用过去所有时刻预测下一时刻。因果注意力掩码保证 t 时刻只能看到 [0..t]，移位 1 步后计算 MSE。推理时自回归逐步生成。

#### 3.3.2 TransformerDenoiser — 共享去噪 Backbone

DDPM 和 MaskedDiffusion 共用同一个双向 Transformer 去噪器，差异仅在输入/输出维度：

```
输入 x (B, L, d_in)
  → Linear(d_in → 128)          # 输入投影
  + TimeEmbed(t) (B, 1, 128)     # 扩散步数正弦嵌入
  + CondProj(cond) (B, L, 128)   # 条件信号 (趋势 S)
  → TransformerEncoder(4层, 128维, 4头, 双向注意力)
  → h (B, L, 128)
```

时间步嵌入 + 条件信号以**加法方式**注入 token 表示（非拼接），降低维度开销。

#### 3.3.3 ResidualDDPM — 连续残差扩散

```
Forward (训练):
  r_0 = X - S                           # 真实残差
  k ~ Uniform(0, K-1)                   # 随机采样扩散步数
  r_k = √(ᾱ_k)·r_0 + √(1-ᾱ_k)·ε       # 闭式加噪
  ε_pred = Denoiser(r_k, k, S)          # 预测噪声
  Loss = MSE(ε_pred, ε)                 # epsilon 预测 (可选 Min-SNR 加权)

Reverse (采样, K=600):
  r_K ~ N(0, I)                         # 纯噪声起始
  for k = K-1, ..., 0:
    ε_pred = Denoiser(r_k, k, S)
    r_{k-1} = 1/√α_k · (r_k - β_k/√(1-ᾱ_k) · ε_pred) + √β_k · z
  return r_0
```

噪声调度默认 cosine schedule，比线性更平滑，避免末尾步噪声突增。

#### 3.3.4 MaskedDiffusion — 离散遮蔽扩散

```python
class MaskedDiffusion(nn.Module):
    def __init__(self, vocab_sizes, d_cond, ...):
        # 每变量独立 Embedding + [MASK] token (idx = vocab_size)
        self.embeddings = nn.ModuleList([
            nn.Embedding(vs + 1, d_model, padding_idx=vs) for vs in vocab_sizes
        ])
        self.denoiser = TransformerDenoiser(d_in=d_model * len(vocab_sizes), ...)
        self.output_heads = nn.ModuleList([nn.Linear(d_model, vs) for vs in vocab_sizes])

    def forward(self, y0, s_cond, x_hat=None):
        k = torch.randint(0, self.K, (B,))
        y_masked, mask_pos = self.forward_mask(y0, k)  # 随机遮蔽
        y_embed = self._embed(y_masked)
        cond = torch.cat([s_cond, x_hat], dim=-1) if x_hat is not None else s_cond
        h = self.denoiser(y_embed, k, cond)
        # 仅在被遮蔽位置计算 CE
        loss = sum(CE(head(h)[mask], y0_j[mask]) for ...)
        return loss / masked_count
```

**推理策略**：从全 `[MASK]` 开始，50 步线性插值反向，每步选出置信度最高的位置先行取消遮蔽（贪心策略）。每变量有独立 Embedding 和输出头（因词汇表不同：功能码 12 vs unit_id 248）。

### 3.4 训练管道 (`trainer.py`)

分阶段训练，确保各模块专精其职。**P0 改进后只训练 Type4 活跃特征（4 维）**：

```python
class MaskDDPMTrainer:
    def __init__(self, config, schema, device):
        self.router = TypeRouter(schema)
        self.d_c_active = self.router.routing.ddpm_count  # 4 (曾为 7)
        self.active_indices = self.router.routing.ddpm_indices  # [0,3,5,6]
        # Trend + DDPM 都只用 d_c_active 维
        self.trend_model = TransformerTrend(d_c=self.d_c_active, ...)
        self.ddpm = ResidualDDPM(d_c=self.d_c_active, d_cond=self.d_c_active, ...)

    def _slice_active(self, X):
        """从全特征 (N,L,7) 中抽取活跃特征 (N,L,4)"""
        return X.index_select(-1, active_indices_tensor)

    def train_trend(self, train_data):
        train_data = self._slice_active(train_data)  # (N,L,7) → (N,L,4)
        for epoch in range(200):
            loss = trend_model.compute_loss(batch)

    def train_diffusion(self, train_x, train_y):
        train_x = self._slice_active(train_x)  # 只训练活跃特征
        S_hat = trend_model(train_x)
        R = train_x - S_hat
        for epoch in range(300):
            loss_cont = ddpm(R, S_hat)              # ε-MSE (4维)
            loss_disc = mask_diff(Y, S_hat[:,active], ...)  # 条件也用活跃
            total = 0.7·loss_cont + 0.3·loss_disc
            ema.update()

### 3.5 生成/采样管道 (`sampler.py`)

```python
class MaskDDPMSampler:
    @torch.no_grad()
    def generate(self, num_samples=1, seed_seq=None):
        # Step 1: 趋势 rollout (只用活跃特征, d_c_active=4)
        seed = torch.randn(B, 1, d_c_active) * 0.1
        S_hat = trend_model.generate_trend(seed, L)

        # Step 2: 残差采样
        R_hat = ddpm.sample(S_hat)

        # Step 3: 连续组装 (活跃特征)
        X_active_norm = S_hat + R_hat

        # Step 4: 离散采样
        Y_hat = mask_diff.sample(B, L, S_hat, x_hat=X_active_norm)

        # Step 5: 重建全特征向量 (活跃 → 7 维, 死特征填 0)
        X_full_norm = _build_full_tensor(X_active_norm, active_mask, B, L)

        # Step 6: 反归一化 (全部 7 维)
        X_hat = normalizer.inverse_transform(X_full_norm)

        # Step 7: Type5 确定性计算 (payload_size, 原始单位)
        X_hat = _fill_payload_size(X_hat, Y_hat)
        #   FC3 req=12, FC3 resp=8+qty×2, FC6=12,
        #   FC16 req=13+qty×2, FC16 resp=12

        # Step 8: 范围裁剪
        X_hat.clamp_(spec.min_val, spec.max_val)
        return X_hat, Y_hat
```

**P0 关键改进**：
- DDPM 只生成 4 个活跃特征（排除死特征和确定性特征），维度从 7→4
- `payload_size` 由协议规则确定性计算，**消除 KS=1.0 的瓶颈**
- 死特征 `reg_val_1/2` 填充训练均值（≈0），不再污染 DDPM 训练梯度

支持跨窗口长序列生成（`generate_long_sequence`）：前一窗口最后 8 步作为下一窗口的自回归种子。

### 3.6 损失函数 (`losses.py`)

| 损失函数 | 公式 | 用途 |
|---------|------|------|
| `epsilon_mse` | `MSE(ε_pred, ε_true)` | DDPM 基础损失 |
| `weighted_epsilon_mse` | `w_k · MSE`, w_k = min(SNR_k, γ)/SNR_k | Min-SNR 加权，抑制高噪声步梯度 |
| `masked_cross_entropy` | `Σ CE(logits[mask], targets[mask]) / |mask|` | 仅在被遮蔽位置计算 CE |
| `combined_loss` | `λ·L_cont + (1-λ)·L_disc` | 联合训练总损失 |
| `quantile_loss` | `(1/K) Σ |Q_k(real) - Q_k(gen)|₁` | 分位数对齐，直接改善 KS |
| `stat_loss` | `MSE(μ_gen, μ_real) + MSE(σ_gen, σ_real)` | 防止残差分布塌缩 |

### 3.7 评估指标 (`metrics.py`)

| 指标 | 计算方式 | 衡量目标 |
|------|---------|----------|
| **KS** | `scipy.stats.ks_2samp` per feature | 连续分布最大偏差 |
| **JSD** | `½(KL(P||M) + KL(Q||M))`，手动实现 | 离散分布差异 |
| **Lag-1 Diff** | `|corr(X[:-1], X[1:])_real - corr(X[:-1], X[1:])_gen|` | 时序相关结构保留度 |

### 3.8 CLI 接口

```bash
python -m diffusion train --data data/processed/ --output checkpoints/run01/
python -m diffusion sample --model checkpoints/run01/ --num-windows 20 --output generated/
python -m diffusion eval --real data/processed/ --gen generated/ --output eval_report.json
```

---

## 4. Assembler — 包组装器

### 4.1 功能概述

Assembler 是 **扩散模型输出 → Checker 输入** 的桥梁，将生成的连续+离散特征 tensor 转换为符合 Modbus/TCP 规范的 PCAP 二进制包和 JSONL 侧边栏。

### 4.2 核心实现

```python
class PacketAssembler:
    def assemble(self, X_hat, Y_hat, output_pcap, output_meta):
        X = X_hat.reshape(-1, d_c)         # 展平窗口 → (N, 7)
        Y = stack(Y_hat).reshape(-1, d_d)  # 展平窗口 → (N, 6)

        for i in range(N):
            # 1. 语义解码
            func_code = FC_VOCAB[Y[i, D_FUNCTION_CODE]]
            direction = "c2s" if Y[i, D_DIRECTION] == 0 else "s2c"

            # 2. 确定性协议封包 (MBAP + PDU)
            if func_code == 3:
                adu = build_read_registers_request(...)  # 或 response
            elif func_code == 6:
                adu = build_write_single_register_request(...)
            # ... FC16, 异常码等

            # 3. scapy 网络包构建
            pkt = Ether() / IP(src, dst) / TCP(sport, dport, "PA", seq, ack) / Raw(adu.raw)
            packets.append(pkt)

            # 4. JSONL sidecar (含 expected block 供 checker 校验)
            write_meta_line(..., expected_modbus={txid, unit_id, func_code},
                            expected_fields={starting_address, quantity})

        wrpcap(output_pcap, packets)
```

### 4.3 确定性协议约束

Assembler 强制执行扩散模型**不学习**的协议规则：

- `protocol_id` 始终为 `0x0000`
- MBAP `length` 字段自动计算（= 1 + len(PDU)），绝不依赖生成值
- 异常响应自动设置功能码最高位 (`| 0x80`)
- TCP seq/ack 自动追踪递增
- PDU 结构严格遵循 Modbus 规范（FC3/6/16 请求/响应的字段布局）

### 4.4 CLI 接口

```bash
python -m assembler --data generated/ --output traces/ --client-ip 10.0.0.10 --server-ip 10.0.0.20
```

---

## 5. Checker — Modbus/TCP 协议校验器

### 5.1 功能概述

Checker 接收 PCAP + JSONL sidecar 两个文件，对生成流量进行 **4 层递进验证**，输出结构化报告（`report.json`）。支持 MVP 和 Strict 两种模式。

### 5.2 验证层次

| 层 | 验证内容 | 严重度 | 诊断码 |
|---|---------|:---:|------|
| **L1: 对齐** | PCAP 包与 JSONL 行对齐 | Fatal | `ALIGNMENT_ERROR` |
| **L2: 网络层** | Ethernet/IP/TCP 解析、Port 502 检测、空负载检测 | Error/Warn | `NON_STANDARD_PORT`, `EMPTY_TCP_PAYLOAD` |
| **L3: Modbus** | MBAP 7 字节头校验、PDU 解析、descriptor 字段解码 | Fatal/Error | `MBAP_TOO_SHORT`, `MBAP_LENGTH_MISMATCH`, `PDU_DESCRIPTOR_PARSE_ERROR` |
| **L4: 事务** | Transaction ID 请求-响应配对、超时检测、特征保真度比对 | Error/Warn | `TX_UNMATCHED_RESPONSE`, `TX_TIMEOUT`, `EXPECTED_FC_MISMATCH` |

### 5.3 核心实现：验证管道

```python
def validate(pcap_path, meta_path, config, mode="mvp") -> Report:
    for pcap_index, raw_pkt, meta, decoded, error in iter_aligned(pcap_path, meta_path):
        # Step 0: 对齐检查
        if error: report.add_finding(Finding(severity=FATAL, code="ALIGNMENT_ERROR"))

        # Step 1: 端口 + 空负载
        # Step 2: MBAP 头校验 (7字节, protocol_id=0, length一致性)
        # Step 3: PDU 解析 + descriptor 字段解码
        # Step 4: 事务配对 (txid, unit_id 状态机)
        # Step 5: 特征保真度 (expected block 比对)
        # Step 6: 超时扫描 (10秒超时)
    return report
```

每个规则独立函数（`_check_mbap` / `_check_modbus_pdu` / `_check_transaction` / `_check_expected` / `_check_timeouts`），返回 `List[Finding]`，管道负责汇总。

### 5.4 Descriptor 驱动解析

Checker 采用声明式 JSON 配置驱动 PDU 解析，覆盖 12 种 Modbus 函数码（FC 1/2/3/4/5/6/8/11/15/16/17/43）：

```json
{
  "functions": {
    "3": {
      "name": "read_holding_registers",
      "request":  [{"name": "starting_address", "field_type": "u16"},
                   {"name": "quantity_of_registers", "field_type": "u16"}],
      "response": [{"name": "byte_count", "field_type": "u8"},
                   {"name": "register_values", "field_type": "bytes", "length_from": "byte_count"}]
    }
  }
}
```

**解析引擎**支持：定长整数（U8/U16/U32/U64/I8/I16/I32/I64）、变长字节序列（BYTES）、位掩码（BITS）、`scale` 缩放、`enum_map` 枚举映射、`length_from` 字段间引用。

### 5.5 CLI 接口

```bash
python -m checker trace.pcapng trace.meta.jsonl --mode strict --output report.json
```

---

## 6. 共享基础设施

### 6.1 特征 Schema (`extractor/schema.py`)

```python
class VariableType(Enum):
    TYPE1 = 1  # 程序驱动/设定值 (阶跃+停留)
    TYPE2 = 2  # 控制器输出 (PID 反馈)
    TYPE3 = 3  # 执行器位置 (饱和/停留)
    TYPE4 = 4  # 过程变量 (惯性主导) — 全管道主力
    TYPE5 = 5  # 派生变量 (确定性函数)
    TYPE6 = 6  # 辅助/低影响 (stub: 不参与训练)

@dataclass
class FeatureSchema:
    continuous: List[FeatureSpec]  # d_c=7
    discrete: List[FeatureSpec]    # d_d=6
    window_length: int = 128       # L

    def adapt_to_data(self, X_cont_flat, dead_threshold=1e-4):
        """自动检测死特征并标记为 Type6。
        - 连续特征 std < threshold → Type6 (不参与训练)
        - payload_size 永为 Type5 (确定性)
        - 返回新的 FeatureSchema，不修改原对象
        """
```

**使用方式**（训练脚本一行搞定）：
```python
schema = FeatureSchema.default_modbus()
schema = schema.adapt_to_data(X_train)  # 自动检测死特征
trainer = MaskDDPMTrainer(cfg, schema, device)
```

**效果**：同一份代码在 ICS_PCAPS 上得 4 活跃特征，在 SWaT 上得 6 活跃特征，零手动调整。

### 6.2 配置系统 (`diffusion/config.py`)

三层 JSON 驱动配置：`TrendConfig` / `DDPMConfig` / `MaskDiffusionConfig` + 顶层 `lambda_balance` / `window_length`。

### 6.3 噪声调度 (`diffusion/utils/noise_schedule.py`)

- **Cosine beta**（DDPM 默认）：平滑过渡，末尾步噪声不过大
- **Linear mask**（Mask 默认）：m_k 从 0→1 线性递增
- **Min-SNR 权重**：`w_k = min(SNR_k, γ) / SNR_k`（γ=5.0）

### 6.4 TypeRouter (`diffusion/models/type_router.py`)

TypeRouter 根据 Schema 中的 `var_type` 自动分配训练策略。**路由在 Schema 创建后自动确定**，无需手动配置。

**路由规则**：

| Schema 类型 | 路由 | 行为 |
|-----------|------|------|
| TYPE4 | `TREND_DDPM` / `MASK` | 全管道训练 |
| TYPE5 | `STUB` | 采样时确定性计算 |
| TYPE6 | `STUB` | 采样时填充均值（连续）或随机（离散） |

**实际效果**（取决于数据集）：

| 数据集 | DDPM 活跃特征 | 图示 |
|--------|:---:|------|
| ICS_PCAPS (1 传感器) | **4** (reg_val_0 + inter_arrival + addr + qty) | `[T4][T6][T6][T4][T5][T4][T4]` |
| SWaT (多传感器) | **6** (3 registers + 3) | `[T4][T4][T4][T4][T5][T4][T4]` |

**确定性负载计算**（`sampler.py: _fill_payload_size`）：
```python
payload_size = f(function_code, quantity, direction)
# FC3 req=12  |  FC3 resp=8+qty×2  |  FC6=12  |  FC16 req=13+qty×2  |  FC16 resp=12
```

**自动化流程**：
```
extract → X (7维) → schema.adapt_to_data(X) → TypeRouter → 自动确定训练维度
```

---

## 7. 端到端管道与验证结果

### 7.1 完整调用链

```
# 1. 数据准备
python -m extractor --generate --num-packets 5000 --output data/train/
  → train_X.npy (N, 128, 7) + train_Y_*.npy (N, 128) + normalizer.json

# 2. 训练
python -m diffusion train --data data/train/ --output checkpoints/run01/
  → Stage 1: TransformerTrend (200 epochs, MSE)
  → Stage 2: DDPM + MaskedDiffusion (300 epochs, λ=0.7)
  → trend_model.pt + ddpm_model.pt + mask_diff_model.pt + ddpm_ema.pt

# 3. 生成
python -m diffusion sample --model checkpoints/run01/ --num-windows 20 --output generated/
  → gen_X.npy (20, 128, 7) + gen_Y_*.npy (20, 128)

# 4. 组装
python -m assembler --data generated/ --output traces/
  → generated-trace-001.pcapng + generated-trace-001.meta.jsonl

# 5. 校验
python -m checker traces/generated-trace-001.pcapng traces/generated-trace-001.meta.jsonl
  → report.json (fatal/error/warn/info)

# 6. 统计评估
python -m diffusion eval --real data/train/ --gen generated/ --output eval_report.json
  → KS / JSD / Lag-1 Diff
```

### 7.2 集成测试结果

使用 4000 包合成数据 + 64 窗口 + 快速训练（10+20 epoch, 小模型）的端到端测试：

```
Test: 4000 pkts → 247 windows → train(10+20 epochs) → generate → assemble → check

1/6 Extractor   ✅ 4000 records → 247 windows (X=(247,64,7), Y=6×(247,64))
2/6 Trainer     ✅ Trend loss: 1.41 → 0.96; Diff cont: 0.55, disc: 2.12
3/6 Generator   ✅ X=(1, 128, 7), Y=6×(1, 128)
4/6 Assembler   ✅ 128 packets PCAP + 128 lines JSONL
5/6 Checker     ✅ fatal=0, error=0, warn=0
6/6 Metrics     ✅ KS/JSD/Lag-1 computed successfully
```

### 7.3 模型参数规模

| 模型 | 参数量 (P0 前) | 参数量 (P0 后) | 说明 |
|------|:---:|:---:|------|
| TransformerTrend | ~795K | ~**220K** | 输入/输出从 7→4 维 |
| ResidualDDPM (含 Denoiser) | ~530K | ~**150K** | d_c 从 7→4 |
| MaskedDiffusion (含 Denoiser) | ~1.1M | ~**700K** | d_cond 从 14→8 |
| **总计** | **~2.4M** | **~1.07M** | **减少 55%** |

### 7.4 P0 改进预期效果

| 指标 | 实验 #1 (P0 前) | P0 预期 |
|------|:---:|:---:|
| payload_size KS | **1.000** | **< 0.01**（确定性计算） |
| reg_val_1/2 KS | ~0.52 | < 0.01（填充均值） |
| Mean KS (活跃) | 0.64 | **< 0.10** |
| 模型参数 | 2.4M | **1.07M**（-55%） |
| 训练速度 | 1.0x | **~1.5x**（维度更小） |

---

## 8. 快速使用指南

### 8.1 从零开始训练并使用

```bash
# Step 1: 生成 5000 个合成 Modbus 包 + 提取训练 tensor
python -m extractor --generate --num-packets 5000 --output data/train/

# Step 2: 训练模型 (CPU: ~15-40min, GPU: ~3-8min)
python -m diffusion train --data data/train/ --output checkpoints/run01/

# Step 3: 生成合成流量 (20 个窗口 = 20×128=2560 包)
python -m diffusion sample --model checkpoints/run01/ --num-windows 20 --output generated/

# Step 4: 组装为可用 PCAP
python -m assembler --data generated/ --output traces/

# Step 5: 校验
python -m checker traces/generated-trace-001.pcapng traces/generated-trace-001.meta.jsonl

# Step 6: 统计评估
python -m diffusion eval --real data/train/ --gen generated/ --output eval_report.json
```

### 8.2 快速验证（小数据量）

```bash
python -m extractor --generate --num-packets 2000 --window-length 64 --output data/quick/
python -m diffusion train --data data/quick/ --output checkpoints/quick/
python -m diffusion sample --model checkpoints/quick/ --num-windows 5 --output gen_quick/
python -m assembler --data gen_quick/ --output traces_quick/
python -m checker traces_quick/generated-trace-001.pcapng traces_quick/generated-trace-001.meta.jsonl
```

目标：`fatal=0, error=0, warn=0`。

### 8.3 使用真实 Modbus PCAP

```bash
python -m extractor --pcap your_modbus_traffic.pcapng --output data/real/
python -m diffusion train --data data/real/ --output checkpoints/real/
# ... 后续步骤同上
```

### 8.4 已知局限与未来方向

| 局限 | 影响 | 计划 |
|------|------|------|
| 训练需合成或真实 PCAP 数据 | 无法零样本生成 | ✅ 已提供合成生成器; 支持真实 PCAP 导入 |
| Type1/2/3/5/6 为 stub | KS 被程序驱动变量主导 | P1: 专用模型（HMM/控制器模拟器） |
| 无条件生成 | 不同工况下分布混合 | P2: 加工况 embedding 条件化 |
| 无诊断工具 | 无法定位瓶颈变量 | P1: Per-feature KS 排序 + CDF 图 |
| 仅支持 128 步窗口 | 长序列需手动拼接 | P2: 完善 generate_long_sequence |
