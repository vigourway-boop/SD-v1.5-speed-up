"""Shared configuration for the SD 1.5 PYNQ acceleration experiment."""

import os

import torch


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Model and compute device.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Image generation settings.
PROMPT = (
    "a realistic full-body adult horse standing on a grassland, single horse, "
    "one horse only, horse anatomy, four legs, hooves, long mane, natural "
    "horse head, photorealistic, detailed body, sharp focus"
)
NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, deformed, malformed animal, bad "
    "anatomy, extra legs, missing legs, extra head, fused body, distorted "
    "face, dog, cow, deer, goat, bird, cartoon, multiple animals, foal, "
    "second horse"
)
GUIDANCE = 7.5
SEED = None
RUN_SEED = int.from_bytes(os.urandom(4), "big") if SEED is None else SEED

# Diffusion and dynamic-skip settings.
BASE_STEPS = 100
TARGET_STEPS = 30
WARMUP_STEPS = 15
DPS_ENABLED = True
SIMILARITY_THRESHOLD = 0.999
MAX_CONSECUTIVE_SKIPS = 3

# PYNQ-Z2 network service. Environment variables can override these defaults.
PYNQ_HOST = os.environ.get("PYNQ_HOST", "192.168.2.99")
PYNQ_PORT = int(os.environ.get("PYNQ_PORT", "9000"))


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def make_generator():
    """Create a fresh generator so both strategies start from the same noise."""
    return torch.Generator(device=DEVICE).manual_seed(RUN_SEED)


def create_pipe(use_dpm_solver: bool = True):
    """Create the Stable Diffusion 1.5 pipeline."""
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        local_files_only=False,
        resume_download=True,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(DEVICE)

    if use_dpm_solver:
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )

    return pipeline
