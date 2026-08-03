import sys; sys.path.insert(0,'.')
import torch
print("imports done", flush=True)
from extractor.schema import FeatureSchema
from diffusion.config import DiffusionConfig
from diffusion.training.trainer import MaskDDPMTrainer
from diffusion.utils.normalisation import Normalizer
print("modules loaded", flush=True)

schema = FeatureSchema.default_modbus()
config = DiffusionConfig()
device = torch.device('cuda')
print(f"Loading checkpoint on {device}...", flush=True)
trainer = MaskDDPMTrainer(config, schema, device=device)
trainer.load('checkpoints/exp_p1p2_300k')
print("Checkpoint loaded OK", flush=True)
normalizer = Normalizer.load('checkpoints/exp_p1p2_300k/normalizer.json')
print(f"Normalizer mean={normalizer.mean.tolist()}", flush=True)
print(f"log_features={normalizer.log_features}", flush=True)
print("DONE", flush=True)
