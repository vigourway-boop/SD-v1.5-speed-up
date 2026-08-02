# config.py
"""SRTP 项目统一配置文件"""

import os
import torch

# 修复环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ---------------------------
# 基本设备 & 模型
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# ---------------------------
# 文本提示相关
# ---------------------------
PROMPT = "a realistic full-body adult horse standing on a grassland, single horse, one horse only, horse anatomy, four legs, hooves, long mane, natural horse head, photorealistic, detailed body, sharp focus"
NEGATIVE_PROMPT = "blurry, low quality, low resolution, deformed, malformed animal, bad anatomy, extra legs, missing legs, extra head, fused body, distorted face, dog, cow, deer, goat, bird, cartoon, multiple animals, foal, second horse"
GUIDANCE = 7.5
SEED = None
RUN_SEED = int.from_bytes(os.urandom(4), "big") if SEED is None else SEED

# ---------------------------
# 采样步数设置
# ---------------------------
BASE_STEPS =100
TARGET_STEPS =30
WARMUP_STEPS = 15   # 前15步构建画面框架，绝对不跳，保证质量

# ---------------------------
# DPS动态跳步配置 (针对 PYNQ 硬件设计)
# ---------------------------
DPS_ENABLED = True
SIMILARITY_THRESHOLD = 0.999   # 余弦相似度阈值（非常严苛，保证质量）
MAX_CONSECUTIVE_SKIPS = 3     # 最多连续跳过2步，防止误差雪球效应导致画面变脏

# PYNQ-Z2 network service. Override with PYNQ_HOST/PYNQ_PORT if needed.
PYNQ_HOST = os.environ.get("PYNQ_HOST", "192.168.2.99")
PYNQ_PORT = int(os.environ.get("PYNQ_PORT", "9000"))

# ------------- --------------
# 通用小工具
# ---------------------------
def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()

def make_generator():
    """Create a fresh generator so each strategy starts from the same noise."""
    return torch.Generator(device=DEVICE).manual_seed(RUN_SEED)

def create_pipe(use_dpm_solver: bool = True):
    """创建 Stable Diffusion pipeline"""
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        local_files_only=False,
        resume_download=True,
        use_safetensors=True,
        safety_checker=None,        # 【绝对关键】彻底关闭安全检查，防止黑图
        requires_safety_checker=False
    ).to(DEVICE)

    if use_dpm_solver:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    return pipe
