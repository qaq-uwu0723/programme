"""CLI entry point for the Mask-DDPM diffusion model.

Usage:
    python -m diffusion train --data <path> --output <path>
    python -m diffusion sample --model <path> --output <path>
    python -m diffusion eval --real <path> --gen <path>
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np

from .config import DiffusionConfig
from .training.trainer import MaskDDPMTrainer
from .sampling.sampler import MaskDDPMSampler
from .utils.normalisation import Normalizer
from .utils.metrics import evaluate_all
from extractor.schema import FeatureSchema


def cmd_train(args) -> None:
    """Train the full Mask-DDPM pipeline."""
    schema = FeatureSchema.default_modbus()
    config = DiffusionConfig.load(args.config) if args.config else DiffusionConfig()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_path = Path(args.data)
    train_x = torch.from_numpy(np.load(data_path / "train_X.npy")).float()
    train_y = [
        torch.from_numpy(np.load(data_path / f"train_Y_{j}.npy")).long()
        for j in range(schema.d_d)
    ]

    # Normalize
    normalizer = Normalizer(schema.d_c).fit(train_x)
    train_x = normalizer.transform(train_x)

    print(f"Train X: {train_x.shape}, d_c={schema.d_c}, d_d={schema.d_d}")

    # Train
    trainer = MaskDDPMTrainer(config, schema, device=device)

    print("=== Stage 1: Trend Training ===")
    trend_history = trainer.train_trend(train_x.to(device))

    print("=== Stage 2: Diffusion Training ===")
    diff_history = trainer.train_diffusion(train_x.to(device), train_y)

    # Save
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save(str(out))
    normalizer.save(str(out / "normalizer.json"))
    np.savez(out / "history.npz", trend=trend_history, diffusion=diff_history)
    print(f"Training complete. Output saved to {out}/")


def cmd_sample(args) -> None:
    """Generate synthetic data from a trained model."""
    schema = FeatureSchema.default_modbus()
    config = DiffusionConfig.load(args.config) if args.config else DiffusionConfig()
    device = torch.device("cpu")

    # Load normalizer
    normalizer = Normalizer.load(Path(args.model) / "normalizer.json")

    # Build and load models
    trainer = MaskDDPMTrainer(config, schema, device=device)
    trainer.load(args.model)

    # Use EMA weights for DDPM
    trainer.ddpm_ema.apply()

    sampler = MaskDDPMSampler(
        trainer.trend_model, trainer.ddpm, trainer.mask_diff,
        normalizer, schema, device=device,
    )

    num_windows = args.num_windows
    X_hat, Y_hat = sampler.generate(num_samples=num_windows)

    # Save
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "gen_X.npy", X_hat.cpu().numpy())
    for j, y in enumerate(Y_hat):
        np.save(out / f"gen_Y_{j}.npy", y.cpu().numpy())

    print(f"Generated {num_windows} windows of shape ({num_windows}, {schema.window_length}, {schema.d_c})")
    print(f"Saved to {out}/")


def cmd_eval(args) -> None:
    """Evaluate generated data against real data."""
    schema = FeatureSchema.default_modbus()

    real_x = np.load(Path(args.real) / "test_X.npy")
    gen_x = np.load(Path(args.gen) / "gen_X.npy")
    real_y_list = [np.load(Path(args.real) / f"test_Y_{j}.npy") for j in range(schema.d_d)]
    gen_y_list = [np.load(Path(args.gen) / f"gen_Y_{j}.npy") for j in range(schema.d_d)]

    real_y = np.stack(real_y_list, axis=-1)
    gen_y = np.stack(gen_y_list, axis=-1)

    results = evaluate_all(real_x, gen_x, real_y, gen_y, schema.vocab_sizes)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nReport saved to {out_path.resolve()}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Mask-DDPM Diffusion Model")
    sub = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train", help="Train Mask-DDPM")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--config")
    p_train.add_argument("--output", default="checkpoints/")
    p_train.add_argument("--device", default="cuda")

    # sample
    p_sample = sub.add_parser("sample", help="Generate synthetic data")
    p_sample.add_argument("--model", required=True)
    p_sample.add_argument("--config")
    p_sample.add_argument("--output", default="generated/")
    p_sample.add_argument("--num-windows", type=int, default=10)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate generated data")
    p_eval.add_argument("--real", required=True)
    p_eval.add_argument("--gen", required=True)
    p_eval.add_argument("--output", default="eval_report.json")

    args = parser.parse_args(argv)
    if args.command == "train":
        cmd_train(args)
    elif args.command == "sample":
        cmd_sample(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
