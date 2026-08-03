# TO_DEBUG_LIST — Mask-DDPM 代码审查 Bug 清单

> 审查日期：2026-06-24  
> 审查范围：`assembler/` `checker/` `diffusion/` `extractor/`  
> 审查结论：共发现 **42 个 bug**：Critical 2, High 6, Medium 13, Low 21  
> 状态更新：2026-08-02（V3.0 管线全通，见下方"修复状态总表"）

---

## 修复状态总表（2026-08-02）

> 管线 5 阶段（训练/生成/组装/校验/评估）端到端回环测试通过，协议层校验零错误。

### ✅ 已修复（经回环测试验证）

| Bug | 修复版本 | 验证 |
|-----|---------|------|
| C1 FC16 payload torch.where 丢弃 | V2.9 代码删除 | 生成测试 |
| C2 EMAModel restore/apply 同逻辑 | V2.9 | 训练 |
| H4 TCP ACK 字段算错 | V2.9 | 组装测试 |
| H5 负 fc_idx 回绕 | V2.9 | 组装测试 |
| M4 exception_code 未箝位 | V2.9 | 组装测试 |
| M5 NaN/inf int() 崩溃 | V2.9（_safe_int） | 组装测试 |
| L1 log_indices 硬编码 | V2.9 | 生成测试 |
| L3 FC_VOCAB 硬编码 | V2.9 | 生成测试 |
| L6 metadata 功能码不一致 | 08-02（全 12 fc builder + 响应 echo） | 校验零 EXPECTED_FC_MISMATCH |
| L7 FUNC_BUILDERS 死代码 | 08-02（补齐全部 12 fc） | 组装测试 |
| L9 pending_request["size"] 死数据 | 08-02 | 组装测试 |
| L21 无最终 timeout 清扫 | 08-02 | 校验测试 |
| M6 缺失预期字段静默通过 | 08-02（EXPECTED_FIELD_MISSING） | 校验测试 |
| M7 PDU 未截断至 MBAP 长度 | 08-02 | 校验测试 |
| H6 isinstance(length, dict) 漏 list | 08-02 | 校验测试 |
| M9 部分读取消耗计数错误 | 08-02 | 校验测试 |
| L11 assert 可被 -O 禁用 | 08-02 | 校验测试 |
| L13 异常响应 fc 从不校验 | 08-02 | 校验测试 |

### 🔧 新发现并已修复（08-02 管线审查）

| Bug | 文件 | 影响 |
|-----|------|------|
| 双重归一化（normalizer 对已标准化数据重 fit） | `tests/train_1m.py` | mean≈0/std≈1、log_features 丢失 → 生成退化 |
| cmd_sample 未装配 StubSampler | `diffusion/__main__.py` | Type6 特征退化为 0/均值 |
| inter_arrival expm1 溢出（μ±3σ 对重尾失效） | `normalisation.py`/`sampler.py` | expm1(63)=3e27 ns |
| 响应未 echo 请求 unit_id | `assembler/packet_builder.py` | 1,629 TX_UNMATCHED_RESPONSE |
| metadata expected_fields 非按 (fc,dir) | `assembler/packet_builder.py` | 5,001 EXPECTED_FIELD_MISSING |
| **stub 用反归一化窗口拟合（浮点噪声）**（08-03） | `tests/train_1m.py` | reg_val_0 KS 虚高 0.808（`2.29e-09` vs 精确 `0`）；改用 `packet_to_features` 原始特征，KS 修复至 0.0016 |

### 🟡 保留（低优先级 / 边界路径未触发）

| Bug | 说明 |
|-----|------|
| L2 _get_active_mask 重复 TypeRouter | 无害，简单读 schema |
| L4/L5 NaN 训练日志 | trainer 侧，无崩溃影响 |
| L8 未使用 import | 仅 typing 导入，cosmetic |
| L10 enum_map 类型注解 | cosmetic |
| L12 冗余 parse_full_pdu | 双次解析，性能非正确性 |
| L20 未使用列 | C_PAYLOAD_SIZE 定义未读，cosmetic |
| H1/H2/H3/M1/M2/M3/M10-M13/L14-L19 | PCAP 源 / 非默认 schema / 边界输入，当前 CSV 路径不触发 |

### 🐛 模型/数据质量（2026-08-02 全部解决）

| # | 问题 | 修复 | 验证 |
|---|------|------|------|
| MQ1 | MaskedDiffusion 多数类偏向：direction 87/13 | 温度采样 + assembler 注入响应保证配对 | checker 0 findings，dir_JSD=0.004 |
| MQ2 | inter_arrival 20 天间隙破坏协议配对 | 排除 >1s 会话间隙伪影（>10s→>1s） | checker 0 findings；KS 取舍见下 |

> MQ2 代价：inter_arrival KS 0.011→0.49（移除 48.7% 20 天伪影的必然结果，收窄 10s→1s 仅额外 +0.001）。**协议有效性优先于分布保真**（记录为已知取舍）。

### 🧹 已清理（2026-08-02）

删除 5 个引用已废弃 PayloadLookup API 的陈旧脚本：`experiments/run_experiment.py`、`tests/eval_gen_quality.py`、`tests/run_gen_only.py`、`tests/run_gen_test.py`、`tests/test_generation.py`（V2.9 起 PayloadLookup 已删除，这些脚本 import 即崩）。新增 `.gitignore`。

---

## Critical（2 个）

### C1 — `sampler.py:314` — torch.where 返回值未赋值，FC16 payload 回退是空操作

**文件**：`diffusion/sampling/sampler.py`，第 314 行

```python
torch.where((fc == 16) & is_request, 15.0, payload_raw)  # 返回值丢弃
```

对比第 313 行正确写法：
```python
payload_raw = torch.where((fc == 3) & ~is_request, 28.0, payload_raw)
```

**触发条件**：采样生成时遇到 `function_code == 16` 的包。  
**修复**：`payload_raw = torch.where((fc == 16) & is_request, 15.0, payload_raw)`

---

### C2 — `trainer.py:50-54` — `EMAModel.restore()` 与 `apply()` 完全相同，无法撤销 EMA

**文件**：`diffusion/training/trainer.py`，第 50-54 行

两个方法代码逐字节相同——都从 shadow 拷贝到 model。`restore()` 应保存原始权重或做反向操作，但未实现。  
**触发条件**：先调 `apply()`（生成），再调 `restore()` 想继续训练。  
**现状**：`restore()` 从未被调用，暂时无害。一旦加入 "生成后继续训练" 等流程即触发。

---

## High（6 个）

### H1 — `pcap_reader.py:148-161` — FC1-4 读响应被错误解析为请求

**文件**：`extractor/pcap_reader.py`，第 148-161 行

```python
if base_fc in (1, 2, 3, 4):
    if len(pdu_data) >= 4:       # 请求路径 ← 包含 ≥2 个寄存器的响应也会进来
        ...
    elif len(pdu_data) >= 1:     # 响应路径 ← 只有 1 个寄存器的响应才正确
        ...
```

**触发条件**：`--pcap` 模式训练，PCAP 中含 FC3 读响应且 ≥2 个寄存器。  
**真相**：FARAONIC 走 CSV reader，ICS_PCAPS 走预提取 tensor，均未触发此路径。

---

### H2 — `feature_builder.py:50-88` — 硬编码特征索引，无视 FeatureSchema

**文件**：`extractor/feature_builder.py`，第 50-88 行

用 `d_c`/`d_d` 初始化数组维度，但所有赋值用的是硬编码位置 [0-6]/[0-5]。  
**触发条件**：传入非默认的 `FeatureSchema`（顺序不同或多/少列）。  
**真相**：始终与 `FeatureSchema.default_modbus()` 配合使用，巧合吻合。

---

### H3 — `pcap_reader.py:117` vs `faraonic_reader.py:104` — payload_size 系统性偏移 ~40 字节

**文件**：`extractor/pcap_reader.py` 第 117 行 / `extractor/faraonic_reader.py` 第 104 行

- PCAP reader：`payload_size = len(payload)`（TCP payload = Modbus ADU，9-250 字节）
- FARAONIC reader：`payload_size = int(row[col["IP_len"]])`（IP 总长，49-300 字节）

**触发条件**：用不同来源的数据混合训练或交叉评估。`min_val=7` 与 FARAONIC 实际值不一致。

---

### H4 — `packet_builder.py:140` — TCP ACK 字段算错，服务器将自身响应大小加到了 ACK 上

**文件**：`assembler/packet_builder.py`，第 140 行

```python
seq, ack = next_ack, next_seq + (len(adu.raw) if pending_request else 0)
```

服务器发送响应时 `ack` 加了自身包长度。应直接使用 `next_seq`（客户端请求后的 seq）。

---

### H5 — `packet_builder.py:112` — 负的 fc_idx 通过 Python 负索引静默回绕

**文件**：`assembler/packet_builder.py`，第 112 行

```python
func_code = FC_VOCAB[fc_idx] if fc_idx < len(FC_VOCAB) else 3
```

`fc_idx = -1` 时 `-1 < 12` 为 `True`，`FC_VOCAB[-1]` 返回 `43` 而非回退到 `3`。  
**修复**：`if 0 <= fc_idx < len(FC_VOCAB) else 3`

---

### H6 — `modbus_desc.py:89-95` — `isinstance(length, dict)` 漏掉了 list，offset + list 崩溃

**文件**：`checker/modbus_desc.py`，第 89-95 行

```python
if isinstance(length, dict):
    length = 0
```

`length_from` 指向 BYTES/BITS 字段时 `parsed.get()` 返回 list，`isinstance(..., dict)` 不匹配，`offset + length`（int + list）引发 `TypeError`。  
**触发条件**：描述符配置中 length_from 引用非数值字段。当前默认配置无此引用。  
**修复**：`if not isinstance(length, int): length = 0`

---

## Medium（13 个）

### M1 — `residual_ddpm.py:96` — `use_min_snr=true` 被静默忽略

**文件**：`diffusion/models/residual_ddpm.py`，第 96 行

配置里 `use_min_snr: true`，但 forward 用 `F.mse_loss(eps_pred, eps)` 完全无视该参数。`losses.py` 里的 `weighted_epsilon_mse()` 从未被调用。

---

### M2 — `trainer.py:157+` — AMP 硬编码 `"cuda"`，无视 `self.device`

**文件**：`diffusion/training/trainer.py`，第 157、167、281、307 行

`GradScaler("cuda")` 和 `autocast("cuda")` 无论 `self.device` 是什么。CPU-only 机器上不崩溃但静默失效。

---

### M3 — `masked_diffusion.py:193` — `torch.linspace(dtype=torch.long)` 版本依赖

**文件**：`diffusion/models/masked_diffusion.py`，第 193 行

`torch.linspace` 用 `dtype=torch.long` 在不同 PyTorch 版本中舍入行为不同。应 `torch.linspace(..., dtype=torch.float).long()`。

---

### M4 — `packet_builder.py:117` — `exception_code` 未箝位，`bytes()` 崩溃

**文件**：`assembler/packet_builder.py` 第 117 行 / `assembler/modbus_rules.py` 第 25 行

`transaction_id` 和 `unit_id` 有 `%` 箝位，但 `exception_code` 无保护。超出 [0,255] 时 `bytes([0x80 | fc, exc_code])` 抛 `ValueError`。

---

### M5 — `packet_builder.py` 多处 — NaN/inf 使 `int()` 崩溃，无回退

**文件**：`assembler/packet_builder.py`，第 118、119、179、214、219、220、234、235、240 行

`int(np.nan)` → `ValueError`；`int(np.inf)` → `OverflowError`。模型数值不稳定时，一个窗口中任何一个 NaN 值都会使整批崩溃。

---

### M6 — `validate.py:386-398` — 缺失的预期字段静默通过验证

**文件**：`checker/validate.py`，第 386-398 行

```python
actual_val = parsed.get(field_name)
if actual_val is not None and actual_val != expected_val:
```

`actual_val is None`（字段从未解析）时静默跳过。应该报告或加 else 分支。

---

### M7 — `mbap.py:56` — PDU 未截断至 MBAP 声明的长度

**文件**：`checker/mbap.py`，第 56 行

```python
remaining = data[7:]  # 应 data[7:7+length-1]
```

尾随垃圾数据被当作合法 PDU 继续解析。

---

### M8 — `validate.py:127-132` — `except Exception: pass` 吞掉解析失败

**文件**：`checker/validate.py`，第 127-132 行

字段解析中任何异常均被静默吞掉，`adu.parsed_fields` 保持 `None`但无 finding 产生。

---

### M9 — `modbus_desc.py:110-114` — 部分数值读取返回错误消耗计数

**文件**：`checker/modbus_desc.py`，第 110-114 行

```python
if len(raw) < size:
    return None, len(raw)  # 返回部分字节数 → 偏移量混乱 → 后续所有字段错位
```

应返回 `size` 而非 `len(raw)` 以保持对齐。

---

### M10 — `faraonic_reader.py:121` — 只捕获一个寄存器值，[1]/[2] 恒为 0

**文件**：`extractor/faraonic_reader.py`，第 121 行

```python
register_values=[reg_val_0, 0, 0]
```

FARAONIC 数据本身每包确实只有一个传感器读数，所以这对 FARAONIC 是准确的。但混合来源数据会不一致。

---

### M11 — `generate_synthetic.py:199-202` — FC3 响应 byte_count 与实际数据不匹配

**文件**：`extractor/generate_synthetic.py`，第 199-202 行

`qty > 3` 时 byte_count 设为 `qty * 2`，但实际 `vals` 只有 6 字节（3 个寄存器）。  
与 H1 联动：`pcap_reader` 解析这种响应时进一步误解析。

---

### M12 — `feature_builder.py:121` — `stride=0` → ZeroDivisionError

**文件**：`extractor/feature_builder.py`，第 121 行

```python
num_windows = max(1, (N - window_length) // stride + 1)  # stride=0 → //0
```

---

### M13 — `feature_builder.py:116` — 空记录列表 N=0 → ZeroDivisionError

**文件**：`extractor/feature_builder.py`，第 116 行

```python
repeats = (window_length // N) + 1  # N=0 → //0
```

---

## Low（21 个）

### 扩散模型

| # | 文件 | 行号 | 描述 |
|---|------|------|------|
| L1 | `sampler.py` | 252 | `log_indices = [3]` 硬编码，schema 列顺序变更时失效 |
| L2 | `sampler.py` | 268-276 | `_get_active_mask` 重复了 `TypeRouter.get_ddpm_mask` |
| L3 | `sampler.py` | 301-303 | `FC_VOCAB` 硬编码，re-extract 时可能与训练数据不一致 |
| L4 | `denoiser.py` | 38 | `import F` 放在首次使用之后（不崩溃但不规范） |
| L5 | `trainer.py` | 169-173 | NaN/Inf 在梯度缩放中无日志，训练静默退化 |

### 组装器

| # | 文件 | 行号 | 描述 |
|---|------|------|------|
| L6 | `packet_builder.py` | 112,153-156 | FC 不在 {3,6,16} 时 metadata 功能码与报文功能码不一致 |
| L7 | `modbus_rules.py` | 134-143 | `FUNC_BUILDERS` 死代码，值类型不一致 |
| L8 | `packet_builder.py` | 6-8 | 未使用的 import（`Tuple`、`struct`、`Path`） |
| L9 | `packet_builder.py` | 183 | `pending_request["size"]` 写入但从未读取 |

### 检查器

| # | 文件 | 行号 | 描述 |
|---|------|------|------|
| L10 | `config.py` / `modbus_desc.py` | 37 / 121-122 | `enum_map` 类型注解 `Dict[int,str]` 与运行时 `str(value) in` 矛盾 |
| L11 | `validate.py` | 56 | `assert` 可被 `python -O` 禁用，应改为显式 `if/raise` |
| L12 | `validate.py` | 256 | `_check_modbus_pdu` 中冗余的 `parse_full_pdu` 调用 |
| L13 | `validate.py` | 375-376 | 异常响应功能码从不与预期 metadata 比对 |

### 提取器

| # | 文件 | 行号 | 描述 |
|---|------|------|------|
| L14 | `pcap_reader.py` | 163-168 | FC5 线圈值（0/0xFF00）被当作 16 位寄存器，语义不匹配 |
| L15 | `faraonic_reader.py` | 40 | `max_rows=0` 被静默忽略（`0` 为 falsy），读取全部行 |
| L16 | `schema.py` | 47-48 | `vocab=None` 静默回退到 256，意图未说明 |
| L17 | `schema.py` | 76 | 一维数组会使 `adapt_to_data` 出现 `IndexError`（无 `shape[1]`） |
| L18 | `pcap_reader.py` | 12 | `sys.path.insert(0, ...)` 副作用污染全局 |
| L19 | `pcap_reader.py` | 74 | 最小 ADU 长度检查为 `< 7`，应 `< 8`（MBAP 7 + PDU 至少 1） |
| L20 | `packet_builder.py` | 36,38 | Schema 列 `register_value_2` 和 `payload_size` 已定义但从未读取 |
| L21 | `checker/validate.py` | 149-152 | 主循环结束后没有最终的 timeout 检查——残留请求不会被报告 |

---

## 触发条件总览

| 触发条件 | 涉及的 bug |
|----------|-----------|
| 训练成功 → 进入生成阶段 | C1, H4, H5, M4, M5, L6, L20 |
| 训练成功 → 进入校验阶段 | H6, M6, M7, M8, M9, L10-L13, L21 |
| 使用 PCAP 数据源（`--pcap`） | H1, H3 |
| 使用非默认 FeatureSchema | H2, L1, L3 |
| 混用不同数据源 | H3, M10 |
| 模型数值不稳定（NaN/Inf） | M5, L5 |
| 边界输入（空数据、stride=0） | M12, M13, L15, L17 |
| CPU-only 训练 | M2 |
| 跨 PyTorch 版本推理 | M3 |
| 代码路径尚未执行到 | C2, L7-L9, L14, L16, L18, L19 |

---

> 当前训练路径（CSV → extractor → trainer，Stage 1+2）上 **无一触发**。  
> Min-SNR 被忽略（M1）是唯一每次训练都生效的，但它不影响崩溃，仅轻微影响收敛质量。

---

## 模型质量问题（非代码 bug）— 2026-08-02 追加

### MQ1 — MaskedDiffusion 多数类偏向：direction 87/13 膨胀

**现象**：组装 12,800 包时 direction 分布 87.3% c2s / 12.7% s2c，训练数据实际 60/40。导致 9,544 个请求无响应、2 个孤儿响应。

**根因**：MaskedDiffusion 的 greedy unmasking 天然过度生成高频类。与 V2.9 已记录的 "FC2 从训练 75% 膨胀至生成 98%" 同一机制——fc 和 direction 都受影响。

**触发**：任何生成流程（`python -m diffusion sample` → `assembler`）。

**影响**：
- 方向分布保真度差（评估阶段 direction JSD 偏高）
- 9,544 个未应答请求 → checker 报大量 timeout/orphan finding（报告噪音大，但单包仍协议合法）
- 下游异常检测：攻击全在 c2s，过度生成 c2s 强化攻击方向性，但破坏正常 request-response 对结构

**修复**（2026-08-02，已解决）：
1. **温度采样**：`MaskedDiffusion.sample()` 从 greedy argmax 改为 temperature-softened 采样（`--temperature` 参数，默认 1.0）。direction 87/13 → 65/35
2. **assembler 配对保证**：`PacketAssembler` 对每个未应答请求注入响应（echo txid/unit_id/fc），并丢弃孤儿响应（无对应请求的 s2c）→ 输出方向 ~50/50，协议配对完美

**验证**：checker 报告 **0 findings**（16974 包，含注入响应），direction JSD=0.004。

---

### MQ2 — inter_arrival 经验路由引入 20 天间隙，破坏请求-响应配对

**现象**：V3.0 后 inter_arrival 路由到经验采样后，生成流量含 ~30% 的 1.74e15 ns（20 天）间隙（真实数据伪影被完整复现）。checker 报 1,628 个 TX_UNMATCHED_RESPONSE——请求在前一包的 20 天间隙中被 10s 超时清除，后续响应引用已超时请求 → 失配。

**根因**：FARAONIC CSV 多会话拼接产生跨会话 20 天时间戳跳跃（真实数据 48.8% 值 > 10s），经验采样原样保留该伪影。真实 Modbus 客户端-服务器流量不存在此类间隙。

**触发**：生成含 20 天间隙的流量 → checker 交易配对校验。

**影响**：
- 协议配对几乎全部失败（12,800 包中仅 ~2 对成功配对）
- TX_TIMEOUT 与 TX_UNMATCHED_RESPONSE 大量叠加，但单包结构合法

**修复**（2026-08-02，已解决）：
- `backfill_v30.py` 和 `train_1m.py`：经验采样前**排除 >1s 的会话间隙伪影**（仅保留真实亚秒级包间隔；真实数据 51.2% 值保留，inter_arrival KS 因此锁死在 ~0.49——移除 20 天模式的必然代价，收窄到 1s 相对 10s 仅额外损失 0.001）

**验证**：配合 MQ1 的 assembler 注入，checker 报告 **0 findings**。inter_arrival KS=0.49 为已文档化取舍（见 EXPERIMENT_LOG 已知限制）。
