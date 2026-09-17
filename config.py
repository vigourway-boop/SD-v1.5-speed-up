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
_seed_override = os.environ.get("SD_SEED")
SEED = int(_seed_override) if _seed_override else None
RUN_SEED = int.from_bytes(os.urandom(4), "big") if SEED is None else SEED

# Diffusion and dynamic-skip settings.
BASE_STEPS = 100
TARGET_STEPS = 30
WARMUP_STEPS = 15
DPS_ENABLED = True
SIMILARITY_THRESHOLD = 0.999
MAX_CONSECUTIVE_SKIPS = 3
SKIP_PREDICTOR_MODE = os.environ.get("SD_SKIP_PREDICTOR", "linear").lower()
SKIP_PREDICTOR_DAMPING = float(os.environ.get("SD_SKIP_PREDICTOR_DAMPING", "0.5"))
SKIP_PREDICTOR_MAX_FACTOR = float(
    os.environ.get("SD_SKIP_PREDICTOR_MAX_FACTOR", "1.5")
)

# Dynamic threshold schedule derived from eight fixed-threshold CSK3 runs.
DYNAMIC_THRESHOLD_ENABLED = os.environ.get(
    "SD_DYNAMIC_THRESHOLD", "1"
).lower() not in {"0", "false", "no"}
WARMUP_SIMILARITY_THRESHOLD = 0.99995
MIDDLE_SIMILARITY_THRESHOLD = 0.99960
LATE_THRESHOLD_START_RATIO = 0.70
LATE_THRESHOLD_MARGIN = 0.00035
LATE_THRESHOLD_MIN = 0.99945
LATE_THRESHOLD_MAX = 0.99965
THRESHOLD_EMA_ALPHA = 0.20
DISTANCE_THRESHOLD = float(os.environ.get("SD_DISTANCE_THRESHOLD", "2.0"))
DECISION_BACKEND = os.environ.get("SD_DECISION_BACKEND", "auto").lower()

# PYNQ-Z2 network service. Environment variables can override these defaults.
PYNQ_HOST = os.environ.get("PYNQ_HOST", "192.168.2.99")
PYNQ_PORT = int(os.environ.get("PYNQ_PORT", "9000"))
PYNQ_CONNECT_ATTEMPTS = int(os.environ.get("PYNQ_CONNECT_ATTEMPTS", "5"))
PYNQ_RETRY_DELAY_SECONDS = float(os.environ.get("PYNQ_RETRY_DELAY_SECONDS", "2"))
PYNQ_GENERATION_RETRIES = int(os.environ.get("PYNQ_GENERATION_RETRIES", "1"))


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def make_generator(seed=None):
    """Create a fresh generator so both strategies start from the same noise."""
    effective_seed = RUN_SEED if seed is None else seed
    return torch.Generator(device=DEVICE).manual_seed(effective_seed)


def create_pipe(use_dpm_solver: bool = True):
    """Create the Stable Diffusion 1.5 pipeline."""
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        local_files_only=os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"},
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
