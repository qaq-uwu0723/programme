# TADS-ICS

NOTION:This is a reimplementation based on https://gitea.markyan04.cn/ModuFlow/internal-docs. For original information, please turn to the [release page]( https://gitea.markyan04.cn/ModuFlow/internal-docs).

Type-Aware Diffusion Synthesis for Industrial Control Systems — a Python reimplementation and optimization of Mask-DDPM for ICS/Modbus traffic generation.

Based on: [Mask-DDPM: A Two-Stage Hybrid Diffusion Framework for ICS Data Generation] *— original paper link TBD*

## Structure

```
├── extractor/          PCAP/CSV → training tensors (feature extraction + auto-schema)
├── diffusion/          Core model: TransformerTrend, DDPM, Masked Diffusion, TypeRouter
├── assembler/          Tensors → scappy Modbus/TCP PCAP + JSONL sidecar
├── checker/            4-layer protocol validator (frame → TCP → Modbus → transaction)
├── experiments/        Training scripts, experiment log, monitor
├── checkpoints/        Trained model weights (multiple experiment runs)
├── docs/               Development log, code summary, design docs, paper notes
└── requirements.txt
```

## Pipeline

```
PCAP/CSV → Extractor → Diffusion Model → Assembler → Checker
            (features)   (train+generate)    (PCAP)     (report)
```

## Quick Start

### 训练（通用入口：支持 CSV / PCAP）

```bash
# 从 CSV 训练（FARAONIC 数据集）
python experiments/run_experiment.py --name exp01 --csv dataset/FARAONIC/Modbus_TCP_ Cybersecurity_Dataset_Training.csv --csv-rows 500000 --output checkpoints/exp01/ --epochs 300 --batch-size 64

# 从 PCAP 训练（原始抓包）
python experiments/run_experiment.py --name exp02 --pcap dataset/ICS_PACPS/clean/traffic.pcap --output checkpoints/exp02/ --epochs 300 --batch-size 64
```

### 监控

```bash
python experiments/monitor_v2.py                                    # 自动找最新 log
python experiments/monitor_v2.py checkpoints/exp01/training.log     # 指定 log
```

### 生成 & 校验

```bash
# 打包（扩散输出 → PCAP + JSONL）
python -m assembler --model checkpoints/exp01/ --output output/exp01/ --count 100

# 校验
python -m checker output/exp01/
```

## Requirements

```
Python 3.10+, PyTorch 2.12+ (CUDA recommended), numpy, scipy, scapy
```

## Results

| 版本 | Mean KS | Key Change |
|---------|:------:|------|
| V1.0 baseline | 0.62 | Full DDPM on all 7 features |
| V2.0 type routing | 0.29 | Exclude dead/deterministic features |
| V2.5 + empirical | 0.13 | Low-cardinality empirical replacement |
| V2.8 GPU mem fix | 0.17 | CPU data + empty_cache, batch=64 |
| V2.8.1 1M | 0.18 | 1M-row training (62K windows), 3.1h |
| V2.8.2 thresh15 | 0.11 | Low-card threshold 10→15, register_value_0 KS→0.04 |
| **V2.8.3** | **0.075** | **3D PayloadLookup + Min-SNR weighting** |

Protocol validity: 100%. Detailed results → `docs/DEVELOPMENT_LOG.md` and `experiments/EXPERIMENT_LOG.md`.

## Detail Info
[Training result](experiments/EXPERIMENT_LOG.md)
[Developing process](docs/DEVELOPMENT_LOG.md)
