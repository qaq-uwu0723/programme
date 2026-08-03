# TADS-ICS

NOTION:This is a reimplementation based on https://gitea.markyan04.cn/ModuFlow/internal-docs. For original information, please turn to the [release page]( https://gitea.markyan04.cn/ModuFlow/internal-docs).

Type-Aware Diffusion Synthesis for Industrial Control Systems — a Python reimplementation and optimization of Mask-DDPM for ICS/Modbus traffic generation.

Based on: [Mask-DDPM: A Two-Stage Hybrid Diffusion Framework for ICS Data Generation] *— original paper link TBD*

## Structure

```
├── extractor/          PCAP/CSV → training tensors (feature extraction + auto-schema)
├── diffusion/          Core model: TransformerTrend, DDPM, Masked Diffusion, TypeRouter
├── assembler/          Tensors → scapy Modbus/TCP PCAP + JSONL sidecar (guarantees pairing)
├── checker/            4-layer protocol validator (frame → TCP → Modbus → transaction)
├── tests/              training / backfill / evaluation scripts
├── experiments/        experiment log, monitor
├── docs/               development log, project summary, code summary, knowledge base
├── checkpoints/        trained models (V2.x → V3.0 → V3.1 evolution)
└── requirements.txt
```

## Pipeline

```
PCAP/CSV → Extractor → Diffusion Model → Assembler → Checker
            (features)   (train+generate)    (PCAP)     (report)
```

## Quick Start

### 训练

```bash
# 1.5M 行 / 6 标签混合训练（FARAONIC，~2.3h）
.venv/Scripts/python.exe tests/train_1m.py
```

### 监控

```bash
python experiments/monitor_v2.py                                    # 自动找最新 log
python experiments/monitor_v2.py checkpoints/exp_15m_type4/training.log   # 指定 log
```

### 生成 & 校验（V3.1 模型）

```bash
# 生成（扩散输出 → 特征张量，--temperature 1.0）
python -m diffusion sample --model checkpoints/exp_15m_type4/ --output generated/ --num-windows 100 --temperature 1.0

# 打包（张量 → PCAP + JSONL，自动注入响应保证请求-响应配对）
python -m assembler --data generated/ --output traces/

# 校验
python -m checker traces/gen-trace-001.pcapng traces/gen-trace-001.meta.jsonl --output traces/report.json

# 统计评估（KS/JSD）
.venv/Scripts/python.exe tests/eval_v30_metrics.py checkpoints/exp_15m_type4
```

## Requirements

```bash
pip install -r requirements.txt
```

```
Python 3.10+, PyTorch 2.0+ (CUDA recommended), numpy, scipy, scapy
```

## Results

> V2.x 为 NORMAL-only 评估（无 MQ2 取舍）；V3.x 为 6 标签混合数据 + inter_arrival 会话间隙排除（MQ2 取舍）。指标口径不同，非直接可比。

| 版本 | Mean KS | Key Change |
|---------|:------:|------|
| V1.0 baseline | 0.62 | Full DDPM on all 7 features |
| V2.0 type routing | 0.29 | Exclude dead/deterministic features |
| V2.5 + empirical | 0.13 | Low-cardinality empirical replacement |
| V2.8.3 | 0.075 | 3D PayloadLookup + Min-SNR weighting (NORMAL-only) |
| V2.9 | — | TYPE5→TYPE4, 6-label mixed training |
| V3.0 | 0.163 | Data fix (response fc), pipeline repairs |
| **V3.1** | **0.132** | **1.5M rows, payload_size KS 0.613→0.426, checker 0 findings** |

**协议有效性：100%**（checker 0 findings，请求-响应完全配对）。详细结果 → [docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md)、[docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md) 和 [experiments/EXPERIMENT_LOG.md](experiments/EXPERIMENT_LOG.md)。


