# TADS-ICS

NOTION:This is a reimplementation based on https://gitea.markyan04.cn/ModuFlow/internal-docs. For original information, please turn to that release page.

Type-Aware Diffusion Synthesis for Industrial Control Systems — a Python reimplementation and optimization of Mask-DDPM for ICS/Modbus traffic generation.

Based on: [Mask-DDPM: A Two-Stage Hybrid Diffusion Framework for ICS Data Generation] *— original paper link TBD*

## Structure

```
├── extractor/          PCAP/CSV → training tensors (feature extraction + auto-schema)
├── diffusion/          Core model: TransformerTrend, DDPM, Masked Diffusion, TypeRouter
├── assembler/          Tensors → scappy Modbus/TCP PCAP + JSONL sidecar
├── checker/            4-layer protocol validator (frame → TCP → Modbus → transaction)
├── experiments/        Training scripts, experiment log, monitor
├── checkpoints/        Trained model weights (4 experiment runs)
├── docs/               Development log, code summary, design docs, paper notes
└── requirements.txt
```

## Pipeline

```
PCAP/CSV → Extractor → Diffusion Model → Assembler → Checker
            (features)   (train+generate)    (PCAP)     (report)
```

## Quick Start

```bash
# 1. Prepare data
python -m extractor --pcap traffic.pcapng --output data/

# 2. Train
python -m diffusion train --data data/ --output checkpoints/

# 3. Generate synthetic traffic
python -m diffusion sample --model checkpoints/ --num-windows 20 --output gen/
python -m assembler --data gen/ --output traces/

# 4. Validate
python -m checker traces/trace.pcapng traces/trace.meta.jsonl
```

## Requirements

```
Python 3.10+, PyTorch 2.12+ (CUDA recommended), numpy, scipy, scapy
```

## Results

| Version | Mean KS | Key Change |
|---------|:------:|------|
| V1.0 baseline | 0.62 | Full DDPM on all 7 features |
| V2.0 type routing | 0.29 | Exclude dead/deterministic features |
| V2.5 + empirical | **0.064** | Low-cardinality empirical replacement |

Protocol validity: 100%. Detailed results → `docs/DEVELOPMENT_LOG.md` and `experiments/EXPERIMENT_LOG.md`.
