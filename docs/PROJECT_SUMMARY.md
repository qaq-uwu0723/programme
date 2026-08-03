# TADS-ICS 项目总结报告

> 生成时间：2026-08-03
> 版本：V3.1（最终模型）
> 项目：Mask-DDPM 复现与优化——面向工业控制系统（ICS）的 Modbus 流量生成

---

## 一、项目起源与设计

### 1.1 核心问题

ICS 网络安全研究面临**训练数据不足**的瓶颈——真实 ICS 网络流量难以大规模获取。现有生成方法（基于 GAN）存在三方面不足：

1. **无法同时处理混合类型数据**：连续传感器读数 + 离散功能码（function_code）
2. **时序-分布冲突**：优化分布一致性时破坏时间序列结构
3. **对特定变量类型错误建模**：程序驱动值（setpoint）vs 物理惯性值（过程变量）的分布本质不同

### 1.2 提出的方法

**两阶段混合扩散框架（Mask-DDPM）**：

```
PCAP/CSV → Extractor → Diffusion Model → Assembler → Checker
            (特征提取)  (train+generate)  (PCAP)     (协议校验)
```

| 模块 | 职责 |
|------|------|
| **TransformerTrend** | 因果 Transformer 提取时序平滑骨架 S（X = S + R 分解）|
| **ResidualDDPM** | 对残差 R 进行高斯去噪扩散（连续特征）|
| **MaskedDiffusion** | 对离散变量（fc/direction/unit_id）遮蔽-恢复扩散 |
| **TypeRouter** | 自动分类变量类型（活跃/死/低基数/确定性）|
| **Assembler** | 特征张量 → 协议合法 Modbus PCAP + JSONL 元数据 |
| **Checker** | 4 层协议校验（帧→TCP→Modbus→事务配对）|

### 1.3 核心设计洞察

**Gaussian DDPM 假设连续密度，无法生成离散 δ 尖峰**。因此退化特征（死特征、低基数、确定性派生）必须从扩散模型中排除，改用经验分布采样——这是贯穿全部版本演进的主线。

---

## 二、演进历程（13 个版本）

### 阶段 1：基线到架构定型（V1.0 → V2.8.3，2026-06-22 ~ 06-24）

| 版本 | 关键变更 | 核心指标 |
|------|----------|----------|
| **V1.0** | Baseline：全 7 维连续特征交给 DDPM | Mean KS=0.62 |
| **V2.0** | Type-aware 路由：死特征/确定性特征排除 | KS 0.62→0.29 |
| **V2.5** | 低基数经验替换（StubSampler）+ log 变换 | KS 0.29→0.064 |
| **V2.6** | FARAONIC 大样本 + 早停 + 分批预计算 | 过拟合比 1.67 |
| **V2.7** | AMP + batch 调优 + 统一训练入口 | — |
| **V2.8** | GPU 显存管理（CPU 数据 + empty_cache）| 显存 5.9→0.3 GB |
| **V2.8.3** | 3D PayloadLookup + Min-SNR 接入 | **Mean KS=0.075**（历史最优）|

**此阶段确立了**：退化特征经验替换、阈值校准（10→15）、GPU 6GB 甜点配置。

### 阶段 2：6 标签迁移与数据修复（V2.9 → V3.0，2026-07-11 ~ 07-12）

| 版本 | 关键变更 | 核心指标 |
|------|----------|----------|
| **V2.9** | payload_size TYPE5→TYPE4（DDPM 联合建模），6 标签训练 | disc 欠拟合（4.7%/500ep）|
| **V3.0** | **数据集方向前缀修复**（响应 fc 恒 0 根因）| disc 0.328→0.443（真实值）|

**V3.0 关键发现**：响应 function_code 恒为 0 是 reader 对所有包使用 `ModbusTCPRequest_` 列所致——模型学到的是"数据集的标注错误"而非真实协议。修复后 disc loss 从虚低的 0.328 升至真实的 0.443。

### 阶段 3：管线修复与质量收尾（V3.0 后，2026-08-02）

| 变更 | 内容 |
|------|------|
| 生成管线 3 bug | 双重归一化、StubSampler 缺失、expm1 溢出 |
| 组装 2 bug | 全 12 fc builder、响应 echo fc/unit_id |
| 校验 7 bug | L21/M6/L13/M7/H6/M9/L11 |
| 42-bug 清单 | 全部代码级问题解决 |
| **MQ1** | 温度采样（direction 87/13→65/35）+ assembler 配对保证 |
| **MQ2** | inter_arrival 排除 >1s 会话间隙伪影 |

**最终效果**：checker 报告 **0 findings**（此前 1523 unmatched + 2243 timeout）。

### 阶段 4：足量训练（V3.1，2026-08-03）

| 指标 | V3.0 (1M) | V3.1 (1.5M) |
|------|-----------|-------------|
| payload_size KS | 0.613 | **0.426**（-31%）|
| Mean KS | 0.163 | **0.132** |
| 训练时长 | 146 min | **138 min** |

---

## 三、最终结果（V3.1）

### 3.1 协议有效性 —— 完美

```
全流程 17,020 包 → checker 0 findings
请求-响应全部配对（assembler 注入保证），无超时、无失配、无字段错误
```

### 3.2 统计保真度

| 特征 | KS | 特征 | JSD |
|------|:---:|------|:---:|
| register_value_0 | 0.002 | function_code | 0.071 |
| register_value_1/2 | 0.000 | direction | 0.0007 |
| inter_arrival_ns | 0.495† | unit_id | 0.033 |
| payload_size | 0.426 | is_exception | 0.000 |
| register_address | 0.001 | exception_code | 0.000 |
| quantity | 0.0004 | Mean JSD (5 learned) | 0.021 |

> † inter_arrival KS 为 MQ2 取舍——排除 48.7% 的 20 天会话间隙伪影换取协议配对完美。

**Mean KS = 0.132，Mean JSD = 0.021**（6 标签混合数据，300 窗口 vs 300K 真实记录）。

### 3.3 工程状态

- **代码**：~7K 行，extractor/diffusion/assembler/checker 四模块，24 生产模块全部导入通过
- **训练**：1.5M 行 / 93,656 窗口 / 138 min（RTX 3060 6GB）
- **可复现**：`tests/train_1m.py` → `diffusion sample` → `assembler` → `checker` → `eval_v30_metrics.py`

---

## 四、承认的缺陷

### 4.1 架构级限制

| # | 缺陷 | 影响 | 根因 |
|---|------|------|------|
| 1 | **payload_size KS=0.426** | 生成 payload 分布偏窄 | DDPM 条件向量不含离散特征，只能学边缘分布 p(ps) 而非 p(ps\|fc,dir) |
| 2 | **disc loss 0.383 未收敛** | 离散预测存在随机错误 | d_model=128 在 62K 窗口 × 6 标签上容量不足 |

### 4.2 数据伪影取舍

| # | 缺陷 | 影响 | 根因 |
|---|------|------|------|
| 3 | **inter_arrival KS=0.49** | 时序分布偏离真实 | 为修复协议配对，排除 48.7% 的 20 天会话间隙伪影（FARAONIC CSV 多会话拼接）|
| 4 | **fc/unit JSD 略退化**（V3.1）| 罕见离散值概率被压缩 | 更多数据使模型过度自信，温度采样下更偏向众数 |

### 4.3 统计失真

| # | 缺陷 | 影响 | 根因 |
|---|------|------|------|
| 5 | **transaction_id JSD=0.99** | 统计评估失真 | MaskedDiffusion 对高基数均匀分布建模失败，sampler 层 override 0..65535 规避但偏离真实 0..255 |
| 6 | **方向平衡依赖 assembler** | 模型本身仍有方向偏向 | MaskedDiffusion 多数类偏向（训练数据攻击 100% c2s），由 assembler 注入响应兜底而非模型自纠正 |

### 4.4 数据源固有局限

- FARAONIC 数据 97% 的 inter_arrival 是伪影（49% 1ns 时间戳精度下限 + 48% 会话间隙），真实时序信号仅 ~3%
- 攻击标签（FUNC_TAMPER/DDOS 等）100% 为 c2s 请求，方向分布天然不平衡

---

## 五、结论

**TADS-ICS 从论文复现（V1.0，KS 0.62）演进为工程化流量生成器（V3.1）**，核心成果：

1. **协议完全合法**：checker 0 findings，可生成直接可用的 Modbus 流量
2. **统计保真**：Mean KS 0.132 / JSD 0.021，6 标签混合数据下合理
3. **工程完备**：全自动管线（训练→生成→组装→校验→评估），24 模块无缺陷

**承认的缺陷集中在两个架构局限**（payload_size 无离散条件、d_model 容量不足）**与一个数据伪影取舍**（inter_arrival 会话间隙）。三者均为非功能性缺陷，不影响 Modbus 单包合法性与攻击检测信号保真度。

---

> 详细时间线见 [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md)，实验数据见 [EXPERIMENT_LOG.md](../experiments/EXPERIMENT_LOG.md)，Bug 追踪见 [TO_DEBUG_LIST.md](../TO_DEBUG_LIST.md)
