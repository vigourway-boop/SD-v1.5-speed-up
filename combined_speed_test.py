import argparse
import csv
import gc
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch

from config import *
from experiment_report import generate_experiment_report
from pynq_cosine_client import FEATURE_DOWNSAMPLE_FACTOR, PynqCosineClient
from skip_predictor import SkipNoisePredictor


pipe = None
STEP_FIELDS = [
    "step",
    "timestep",
    "similarity",
    "threshold",
    "threshold_requested",
    "threshold_q15",
    "threshold_phase",
    "adjacent_similarity_ema",
    "threshold_passed",
    "cosine_passed",
    "distance_passed",
    "normalized_distance",
    "distance_threshold",
    "distance_threshold_q20",
    "should_skip",
    "action",
    "skip_streak",
    "feature_bytes",
    "prepare_ms",
    "pynq_total_ms",
    "kernel_ms",
    "server_ms",
    "round_trip_ms",
    "prediction_mode",
    "prediction_factor",
    "prediction_ms",
    "unet_ms",
    "scheduler_ms",
    "step_total_ms",
]
TIMING_FIELDS = [
    "阶段 / Stage",
    "耗时（ms） / Time (ms)",
    "计入生成总耗时 / Included in generation total",
    "说明 / Notes",
]


def run_original_steps(num_steps, filename):
    print(f"\n=== Baseline: {num_steps} UNet steps ===")
    sync()
    started = time.perf_counter()
    image = pipe(
        PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=num_steps,
        guidance_scale=GUIDANCE,
        generator=make_generator(RUN_SEED),
    ).images[0]
    sync()
    elapsed = time.perf_counter() - started
    save_started = time.perf_counter()
    image.save(filename)
    save_ms = (time.perf_counter() - save_started) * 1000.0
    print(f"Baseline finished in {elapsed:.2f} s: {filename}")
    return elapsed, image, {
        "generation_ms": elapsed * 1000.0,
        "image_save_ms": save_ms,
    }


def run_with_dynamic_steps(
    filename,
    pynq_client,
    distance_threshold,
    predictor_mode,
    predictor_damping,
    predictor_max_factor,
):
    """Run SD while PYNQ makes every cosine-based skip decision."""
    print("\n=== Dynamic diffusion with PYNQ-Z2 decisions ===")

    pynq_client.reset()
    pynq_client.configure(
        enabled=DYNAMIC_THRESHOLD_ENABLED,
        total_steps=BASE_STEPS,
        warmup_steps=WARMUP_STEPS,
        max_consecutive_skips=MAX_CONSECUTIVE_SKIPS,
        fixed_threshold=SIMILARITY_THRESHOLD,
        warmup_threshold=WARMUP_SIMILARITY_THRESHOLD,
        middle_threshold=MIDDLE_SIMILARITY_THRESHOLD,
        late_start_ratio=LATE_THRESHOLD_START_RATIO,
        late_margin=LATE_THRESHOLD_MARGIN,
        late_min=LATE_THRESHOLD_MIN,
        late_max=LATE_THRESHOLD_MAX,
        ema_alpha=THRESHOLD_EMA_ALPHA,
        distance_threshold=distance_threshold,
    )
    sync()
    started = time.perf_counter()

    text_started = time.perf_counter()
    text_inputs = pipe.tokenizer(
        [NEGATIVE_PROMPT, PROMPT],
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        prompt_embeds = pipe.text_encoder(text_inputs.input_ids)[0]
    sync()
    text_encoder_ms = (time.perf_counter() - text_started) * 1000.0

    pipe.scheduler.set_timesteps(BASE_STEPS, device=DEVICE)
    latent_started = time.perf_counter()
    latents = torch.randn(
        (1, 4, 64, 64),
        generator=make_generator(RUN_SEED),
        device=DEVICE,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    latents = latents * pipe.scheduler.init_noise_sigma
    sync()
    latent_init_ms = (time.perf_counter() - latent_started) * 1000.0

    predictor = SkipNoisePredictor(
        mode=predictor_mode,
        damping=predictor_damping,
        max_factor=predictor_max_factor,
    )
    executed_steps = 0
    skipped_steps = 0
    step_rows = []
    for step_index, timestep in enumerate(pipe.scheduler.timesteps):
        step_started = time.perf_counter()
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(
            latent_model_input, timestep
        )

        decision = pynq_client.decide(
            latent_model_input,
            step_index=step_index,
        )
        should_skip = decision.should_skip
        skip_streak = decision.skip_streak
        action = "SKIP" if should_skip else "UNET"
        similarity_text = (
            "n/a" if math.isnan(decision.similarity) else f"{decision.similarity:.6f}"
        )
        print(
            f"[PYNQ] step {step_index + 1:03d}/{BASE_STEPS}: {action} "
            f"similarity={similarity_text} "
            f"distance={decision.normalized_distance:.6f} "
            f"threshold={decision.effective_threshold:.6f} "
            f"phase={decision.threshold_phase} "
            f"kernel={decision.kernel_ms:.3f} ms "
            f"round-trip={decision.round_trip_ms:.3f} ms"
        )

        unet_ms = 0.0
        prediction_ms = 0.0
        prediction_mode = "unet"
        prediction_factor = 0.0
        if should_skip:
            sync()
            prediction_started = time.perf_counter()
            prediction = predictor.predict(timestep)
            noise_pred = prediction.value
            sync()
            prediction_ms = (time.perf_counter() - prediction_started) * 1000.0
            prediction_mode = prediction.mode
            prediction_factor = prediction.factor
            skipped_steps += 1
        else:
            sync()
            unet_started = time.perf_counter()
            with torch.no_grad():
                noise_pred = pipe.unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                ).sample
            sync()
            unet_ms = (time.perf_counter() - unet_started) * 1000.0
            predictor.record(timestep, noise_pred)
            executed_steps += 1

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        guided_noise_pred = noise_pred_uncond + GUIDANCE * (
            noise_pred_text - noise_pred_uncond
        )
        sync()
        scheduler_started = time.perf_counter()
        latents = pipe.scheduler.step(
            guided_noise_pred, timestep, latents
        ).prev_sample
        sync()
        scheduler_ms = (time.perf_counter() - scheduler_started) * 1000.0

        step_rows.append(
            {
                "step": step_index + 1,
                "timestep": int(timestep.item()),
                "similarity": decision.similarity,
                "threshold": decision.effective_threshold,
                "threshold_requested": decision.threshold_requested,
                "threshold_q15": decision.threshold_q15,
                "threshold_phase": decision.threshold_phase,
                "adjacent_similarity_ema": decision.adjacent_similarity_ema,
                "threshold_passed": int(decision.threshold_passed),
                "cosine_passed": int(decision.cosine_passed),
                "distance_passed": int(decision.distance_passed),
                "normalized_distance": decision.normalized_distance,
                "distance_threshold": decision.effective_distance_threshold,
                "distance_threshold_q20": decision.distance_threshold_q20,
                "should_skip": int(should_skip),
                "action": action,
                "skip_streak": skip_streak,
                "feature_bytes": decision.feature_bytes,
                "prepare_ms": decision.prepare_ms,
                "pynq_total_ms": decision.prepare_ms + decision.round_trip_ms,
                "kernel_ms": decision.kernel_ms,
                "server_ms": decision.server_ms,
                "round_trip_ms": decision.round_trip_ms,
                "prediction_mode": prediction_mode,
                "prediction_factor": prediction_factor,
                "prediction_ms": prediction_ms,
                "unet_ms": unet_ms,
                "scheduler_ms": scheduler_ms,
                "step_total_ms": (time.perf_counter() - step_started) * 1000.0,
            }
        )

    vae_started = time.perf_counter()
    with torch.no_grad():
        image = pipe.vae.decode(
            latents / pipe.vae.config.scaling_factor
        ).sample
        image = (
            (image / 2 + 0.5)
            .clamp(0, 1)
            .cpu()
            .permute(0, 2, 3, 1)
            .float()
            .numpy()
        )
        image = pipe.numpy_to_pil(image)[0]

    sync()
    vae_ms = (time.perf_counter() - vae_started) * 1000.0
    elapsed = time.perf_counter() - started
    save_started = time.perf_counter()
    image.save(filename)
    image_save_ms = (time.perf_counter() - save_started) * 1000.0
    average_network_ms = (
        pynq_client.total_round_trip_ms / pynq_client.step_calls
        if pynq_client.step_calls
        else 0.0
    )
    diffusion_loop_ms = sum(row["step_total_ms"] for row in step_rows)
    unet_total_ms = sum(row["unet_ms"] for row in step_rows)
    scheduler_total_ms = sum(row["scheduler_ms"] for row in step_rows)
    prediction_total_ms = sum(row["prediction_ms"] for row in step_rows)
    pynq_total_ms = sum(row["pynq_total_ms"] for row in step_rows)
    timing = {
        "generation_ms": elapsed * 1000.0,
        "text_encoder_ms": text_encoder_ms,
        "latent_init_ms": latent_init_ms,
        "diffusion_loop_ms": diffusion_loop_ms,
        "unet_ms": unet_total_ms,
        "scheduler_ms": scheduler_total_ms,
        "skip_prediction_ms": prediction_total_ms,
        "pynq_feature_prepare_ms": sum(row["prepare_ms"] for row in step_rows),
        "pynq_total_ms": pynq_total_ms,
        "network_round_trip_ms": sum(row["round_trip_ms"] for row in step_rows),
        "fpga_kernel_ms": sum(row["kernel_ms"] for row in step_rows),
        "pynq_server_ms": sum(row["server_ms"] for row in step_rows),
        "diffusion_loop_other_ms": max(
            0.0,
            diffusion_loop_ms
            - unet_total_ms
            - scheduler_total_ms
            - pynq_total_ms
            - prediction_total_ms,
        ),
        "vae_ms": vae_ms,
        "image_save_ms": image_save_ms,
    }
    print(
        f"Dynamic run finished: UNet={executed_steps}, skipped={skipped_steps}, "
        f"elapsed={elapsed:.2f} s"
    )
    print(
        f"PYNQ feature traffic={pynq_client.total_bytes_sent / 1024:.1f} KiB, "
        f"average round-trip={average_network_ms:.3f} ms"
    )
    print(f"Image saved to: {filename}")
    return elapsed, executed_steps, skipped_steps, image, step_rows, timing


def run_dynamic_with_recovery(
    filename,
    pynq_client,
    distance_threshold,
    predictor_mode,
    predictor_damping,
    predictor_max_factor,
    generation_retries,
    connect_attempts,
    retry_delay_seconds,
):
    """Restart the deterministic dynamic run after a connection failure."""
    for recovery_attempt in range(generation_retries + 1):
        try:
            connection_attempt = pynq_client.connect_with_retry(
                attempts=connect_attempts,
                delay_seconds=retry_delay_seconds,
            )
            if connection_attempt > 1:
                print(f"PYNQ connected on attempt {connection_attempt}.")
            result = run_with_dynamic_steps(
                filename,
                pynq_client,
                distance_threshold,
                predictor_mode,
                predictor_damping,
                predictor_max_factor,
            )
            return (*result, recovery_attempt)
        except ConnectionError as exc:
            pynq_client.close()
            if recovery_attempt >= generation_retries:
                raise
            print(
                "PYNQ connection was interrupted. Restarting the complete "
                f"Dynamic run ({recovery_attempt + 1}/{generation_retries}): {exc}"
            )
            time.sleep(retry_delay_seconds)

    raise RuntimeError("Dynamic recovery loop exited unexpectedly")


def warm_up_pipeline():
    print("\nWarming up the Stable Diffusion pipeline...")
    started = time.perf_counter()
    try:
        pipe(
            "warmup",
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=max(2, min(WARMUP_STEPS, 5)),
            guidance_scale=GUIDANCE,
        )
        sync()
        return (time.perf_counter() - started) * 1000.0
    except Exception as exc:
        print(f"Warmup was skipped: {exc}")
        return 0.0


def create_experiment_directory(root):
    root = Path(root)
    stem = datetime.now().strftime(f"%Y%m%d_%H%M%S_seed_{RUN_SEED}")
    candidate = root / stem
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stem}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def write_step_metrics(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=STEP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_timing_summary(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        for row in rows:
            rendered = dict(row)
            rendered["耗时（ms） / Time (ms)"] = (
                f"{float(row['耗时（ms） / Time (ms)']):.3f}"
            )
            writer.writerow(rendered)


def build_timing_rows(model_load_ms, warmup_ms, baseline_timing, dynamic_timing,
                      quality_evaluation_ms):
    rows = [
        {
            "阶段 / Stage": "模型加载 / Model loading",
            "耗时（ms） / Time (ms)": model_load_ms,
            "计入生成总耗时 / Included in generation total": "否 / No",
            "说明 / Notes": "加载 SD 1.5 pipeline",
        },
        {
            "阶段 / Stage": "预热 / Warm-up",
            "耗时（ms） / Time (ms)": warmup_ms,
            "计入生成总耗时 / Included in generation total": "否 / No",
            "说明 / Notes": "避免首次 CUDA 调用干扰",
        },
        {
            "阶段 / Stage": "Baseline 扩散生成 / Baseline diffusion",
            "耗时（ms） / Time (ms)": baseline_timing["generation_ms"],
            "计入生成总耗时 / Included in generation total": "是 / Yes",
            "说明 / Notes": "100 个完整 UNet 时间步",
        },
        {
            "阶段 / Stage": "Dynamic 总生成 / Dynamic generation total",
            "耗时（ms） / Time (ms)": dynamic_timing["generation_ms"],
            "计入生成总耗时 / Included in generation total": "是 / Yes",
            "说明 / Notes": "包含特征处理、传输、FPGA 和扩散过程",
        },
        {
            "阶段 / Stage": "文本编码器 / Text encoder",
            "耗时（ms） / Time (ms)": dynamic_timing["text_encoder_ms"],
            "计入生成总耗时 / Included in generation total": "是 / Yes",
            "说明 / Notes": "CLIP text encoder",
        },
        {
            "阶段 / Stage": "Latent 初始化 / Latent initialization",
            "耗时（ms） / Time (ms)": dynamic_timing["latent_init_ms"],
            "计入生成总耗时 / Included in generation total": "是 / Yes",
            "说明 / Notes": "生成初始噪声",
        },
        {
            "阶段 / Stage": "UNet 推理 / UNet inference",
            "耗时（ms） / Time (ms)": dynamic_timing["unet_ms"],
            "计入生成总耗时 / Included in generation total": "是 / Yes",
            "说明 / Notes": "实际执行的 UNet 时间步",
        },
        {
            "阶段 / Stage": "扩散循环总计 / Diffusion loop total",
            "耗时（ms） / Time (ms)": dynamic_timing["diffusion_loop_ms"],
            "计入生成总耗时 / Included in generation total": "是 / Yes",
            "说明 / Notes": "UNet、Scheduler、PYNQ 和循环开销的总计",
        },
        {
            "阶段 / Stage": "Scheduler 更新 / Scheduler update",
            "耗时（ms） / Time (ms)": dynamic_timing["scheduler_ms"],
            "计入生成总耗时 / Included in generation total": "是 / Yes",
            "说明 / Notes": "包含跳步后的 latent 更新",
        },
        {
            "阶段 / Stage": "跳步噪声预测 / Skip noise prediction",
            "耗时（ms） / Time (ms)": dynamic_timing["skip_prediction_ms"],
            "计入生成总耗时 / Included in generation total": "是 / Yes",
            "说明 / Notes": "双历史阻尼线性外推或旧值复用",
        },
        {
            "阶段 / Stage": "PYNQ 特征准备 / PYNQ feature preparation",
            "耗时（ms） / Time (ms)": dynamic_timing["pynq_feature_prepare_ms"],
            "计入生成总耗时 / Included in generation total": "是 / Yes",
            "说明 / Notes": "下采样和 int8 量化",
        },
        {
            "阶段 / Stage": "PYNQ 总处理 / PYNQ total",
            "耗时（ms） / Time (ms)": dynamic_timing["pynq_total_ms"],
            "计入生成总耗时 / Included in generation total": "是 / Yes",
            "说明 / Notes": "特征准备与网络往返的总计",
        },
        {
            "阶段 / Stage": "网络往返 / Network round-trip",
            "耗时（ms） / Time (ms)": dynamic_timing["network_round_trip_ms"],
            "计入生成总耗时 / Included in generation total": "是 / Yes",
            "说明 / Notes": "PC 与 PYNQ-Z2 TCP 通信",
        },
        {
            "阶段 / Stage": "FPGA 内核 / FPGA kernel",
            "耗时（ms） / Time (ms)": dynamic_timing["fpga_kernel_ms"],
            "计入生成总耗时 / Included in generation total": "是 / Yes",
            "说明 / Notes": "余弦、距离和跳步判断",
        },
        {
            "阶段 / Stage": "PYNQ 服务端 / PYNQ server",
            "耗时（ms） / Time (ms)": dynamic_timing["pynq_server_ms"],
            "计入生成总耗时 / Included in generation total": "是 / Yes",
            "说明 / Notes": "ARM MMIO 和协议处理",
        },
        {
            "阶段 / Stage": "扩散循环其他开销 / Diffusion loop other",
            "耗时（ms） / Time (ms)": dynamic_timing["diffusion_loop_other_ms"],
            "计入生成总耗时 / Included in generation total": "是 / Yes",
            "说明 / Notes": "CFG、张量拼接、缩放和 Python 循环",
        },
        {
            "阶段 / Stage": "VAE 解码 / VAE decode",
            "耗时（ms） / Time (ms)": dynamic_timing["vae_ms"],
            "计入生成总耗时 / Included in generation total": "是 / Yes",
            "说明 / Notes": "latent 转 RGB 图片",
        },
        {
            "阶段 / Stage": "Baseline 图片保存 / Baseline image save",
            "耗时（ms） / Time (ms)": baseline_timing["image_save_ms"],
            "计入生成总耗时 / Included in generation total": "否 / No",
            "说明 / Notes": "Baseline PNG 文件写入",
        },
        {
            "阶段 / Stage": "Dynamic 图片保存 / Dynamic image save",
            "耗时（ms） / Time (ms)": dynamic_timing["image_save_ms"],
            "计入生成总耗时 / Included in generation total": "否 / No",
            "说明 / Notes": "Dynamic PNG 文件写入",
        },
        {
            "阶段 / Stage": "质量评估 / Quality evaluation",
            "耗时（ms） / Time (ms)": quality_evaluation_ms,
            "计入生成总耗时 / Included in generation total": "否 / No",
            "说明 / Notes": "PSNR、SSIM、LPIPS、CLIP",
        },
    ]
    return rows


def print_timing_summary(rows):
    print("\n=== 耗时汇总 / Timing breakdown ===")
    for row in rows:
        value = row["耗时（ms） / Time (ms)"]
        print(f"{row['阶段 / Stage']}: {value:.3f} ms")


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stable Diffusion with PYNQ-Z2 cosine skip decisions"
    )
    parser.add_argument("--pynq-host", default=PYNQ_HOST)
    parser.add_argument("--pynq-port", type=int, default=PYNQ_PORT)
    parser.add_argument("--seed", type=int, default=RUN_SEED)
    parser.add_argument(
        "--distance-threshold", type=float, default=DISTANCE_THRESHOLD
    )
    parser.add_argument(
        "--skip-predictor",
        choices=("linear", "reuse"),
        default=SKIP_PREDICTOR_MODE,
    )
    parser.add_argument(
        "--predictor-damping", type=float, default=SKIP_PREDICTOR_DAMPING
    )
    parser.add_argument(
        "--predictor-max-factor", type=float, default=SKIP_PREDICTOR_MAX_FACTOR
    )
    parser.add_argument(
        "--connect-attempts", type=int, default=PYNQ_CONNECT_ATTEMPTS
    )
    parser.add_argument(
        "--retry-delay", type=float, default=PYNQ_RETRY_DELAY_SECONDS
    )
    parser.add_argument(
        "--generation-retries", type=int, default=PYNQ_GENERATION_RETRIES
    )
    parser.add_argument("--output", default="combo_step_dynamic_pynq.png")
    parser.add_argument("--experiment-root", default="experiments")
    parser.add_argument(
        "--with-baseline",
        action="store_true",
        help="also generate a full-step baseline image before the PYNQ run",
    )
    parser.add_argument(
        "--skip-quality-eval",
        action="store_true",
        help="skip PSNR, SSIM, LPIPS and CLIP Score evaluation",
    )
    parser.add_argument("--skip-warmup", action="store_true")
    return parser.parse_args()


def main():
    global pipe, RUN_SEED
    args = parse_args()
    if not 0 <= args.seed <= 0xFFFFFFFF:
        raise SystemExit("--seed must be between 0 and 4294967295")
    if not math.isfinite(args.distance_threshold) or not 0.0 <= args.distance_threshold <= 2.0:
        raise SystemExit("--distance-threshold must be a finite value in [0, 2]")
    if not math.isfinite(args.predictor_damping) or not 0.0 <= args.predictor_damping <= 1.0:
        raise SystemExit("--predictor-damping must be a finite value in [0, 1]")
    if not math.isfinite(args.predictor_max_factor) or args.predictor_max_factor < 0.0:
        raise SystemExit("--predictor-max-factor must be finite and non-negative")
    if args.connect_attempts < 1:
        raise SystemExit("--connect-attempts must be at least 1")
    if not math.isfinite(args.retry_delay) or args.retry_delay < 0.0:
        raise SystemExit("--retry-delay must be finite and non-negative")
    if args.generation_retries < 0:
        raise SystemExit("--generation-retries cannot be negative")
    RUN_SEED = args.seed
    experiment_dir = create_experiment_directory(args.experiment_root)
    print("=" * 68)
    print("Stable Diffusion + PYNQ-Z2 cosine skip control")
    print(f"Compute device: {DEVICE}")
    print(f"Random seed: {RUN_SEED}")
    print(f"PYNQ server: {args.pynq_host}:{args.pynq_port}")
    print(f"Experiment: {experiment_dir}")
    print("=" * 68)

    client = PynqCosineClient(args.pynq_host, args.pynq_port)
    try:
        connection_attempt = client.connect_with_retry(
            attempts=args.connect_attempts,
            delay_seconds=args.retry_delay,
        )
    except (ConnectionError, OSError) as exc:
        raise SystemExit(
            "Cannot connect to the PYNQ cosine server. "
            "Check the board IP and service status.\n"
            f"Details: {exc}"
        ) from exc

    print(
        "PYNQ connection and CSK4 protocol check passed "
        f"(attempt {connection_attempt}/{args.connect_attempts})."
    )
    client.close()

    baseline_elapsed = None
    baseline_image = None
    baseline_timing = {"generation_ms": 0.0, "image_save_ms": 0.0}
    dynamic_timing = {}
    model_load_started = time.perf_counter()
    model_load_ms = 0.0
    warmup_ms = 0.0
    quality_evaluation_ms = 0.0
    quality = None
    try:
        pipe = create_pipe(use_dpm_solver=True)
        sync()
        model_load_ms = (time.perf_counter() - model_load_started) * 1000.0
        if not args.skip_warmup:
            warmup_ms = warm_up_pipeline()
        if args.with_baseline:
            baseline_path = experiment_dir / "baseline.png"
            baseline_elapsed, baseline_image, baseline_timing = run_original_steps(
                BASE_STEPS, baseline_path
            )
            alias_save_started = time.perf_counter()
            baseline_image.save("combo_step100_baseline.png")
            baseline_timing["image_save_ms"] += (
                time.perf_counter() - alias_save_started
            ) * 1000.0

        dynamic_path = experiment_dir / "pynq_dynamic.png"
        (
            dynamic_elapsed,
            executed_steps,
            skipped_steps,
            dynamic_image,
            step_rows,
            dynamic_timing,
            recovery_attempts,
        ) = run_dynamic_with_recovery(
            dynamic_path,
            client,
            args.distance_threshold,
            args.skip_predictor,
            args.predictor_damping,
            args.predictor_max_factor,
            args.generation_retries,
            args.connect_attempts,
            args.retry_delay,
        )
        alias_save_started = time.perf_counter()
        dynamic_image.save(args.output)
        dynamic_timing["image_save_ms"] += (
            time.perf_counter() - alias_save_started
        ) * 1000.0
        write_step_metrics(experiment_dir / "step_metrics.csv", step_rows)

        speedup = (
            baseline_elapsed / dynamic_elapsed
            if baseline_elapsed is not None and dynamic_elapsed
            else None
        )
        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seed": RUN_SEED,
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "device": DEVICE,
            "pynq_host": args.pynq_host,
            "pynq_port": args.pynq_port,
            "base_steps": BASE_STEPS,
            "executed_unet_steps": executed_steps,
            "skipped_steps": skipped_steps,
            "similarity_threshold": (
                MIDDLE_SIMILARITY_THRESHOLD
                if DYNAMIC_THRESHOLD_ENABLED
                else SIMILARITY_THRESHOLD
            ),
            "threshold_mode": (
                "dynamic" if DYNAMIC_THRESHOLD_ENABLED else "fixed"
            ),
            "threshold_schedule": {
                "warmup_steps": WARMUP_STEPS,
                "warmup_threshold": WARMUP_SIMILARITY_THRESHOLD,
                "middle_threshold": MIDDLE_SIMILARITY_THRESHOLD,
                "late_start_step": round(BASE_STEPS * LATE_THRESHOLD_START_RATIO) + 1,
                "late_rule": "adjacent_similarity_ema - margin",
                "late_margin": LATE_THRESHOLD_MARGIN,
                "late_min": LATE_THRESHOLD_MIN,
                "late_max": LATE_THRESHOLD_MAX,
                "ema_alpha": THRESHOLD_EMA_ALPHA,
            } if DYNAMIC_THRESHOLD_ENABLED else None,
            "warmup_steps": WARMUP_STEPS,
            "max_consecutive_skips": MAX_CONSECUTIVE_SKIPS,
            "distance_threshold": args.distance_threshold,
            "skip_predictor": {
                "mode": args.skip_predictor,
                "damping": args.predictor_damping,
                "max_factor": args.predictor_max_factor,
            },
            "connection_recovery": {
                "connect_attempts": args.connect_attempts,
                "retry_delay_seconds": args.retry_delay,
                "generation_retries_allowed": args.generation_retries,
                "generation_retries_used": recovery_attempts,
            },
            "baseline_seconds": baseline_elapsed,
            "dynamic_seconds": dynamic_elapsed,
            "speedup": speedup,
            "pynq_feature_bytes": client.total_bytes_sent,
            "pynq_feature_dtype": "int8",
            "pynq_downsample_factor": FEATURE_DOWNSAMPLE_FACTOR,
            "dynamic_threshold_controller_location": "PYNQ ARM",
            "cosine_similarity_location": "FPGA",
            "feature_distance_location": "FPGA",
            "skip_controller_location": "FPGA",
            "skip_rule": "cosine_passed AND distance_passed",
            "average_round_trip_ms": (
                client.total_round_trip_ms / client.step_calls
                if client.step_calls
                else 0.0
            ),
            "pynq_quantization_transfer_and_fpga_included_in_dynamic_time": True,
            "quality_evaluation_included_in_dynamic_time": False,
        }

        if baseline_image is not None and not args.skip_quality_eval:
            print("\n=== Full image quality evaluation (outside generation timing) ===")
            pipe = None
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            from quality_metrics import evaluate_quality

            quality_started = time.perf_counter()
            quality = evaluate_quality(
                baseline_image, dynamic_image, PROMPT, DEVICE
            )
            quality_evaluation_ms = (time.perf_counter() - quality_started) * 1000.0
            write_json(experiment_dir / "quality_metrics.json", quality)
            print(f"PSNR       : {quality['psnr_dynamic_vs_baseline_db']:.4f} dB")
            print(f"SSIM       : {quality['ssim_dynamic_vs_baseline']:.6f}")
            print(f"LPIPS      : {quality['lpips_dynamic_vs_baseline']:.6f}")
            print(f"CLIP base   : {quality['clip_score_baseline']:.4f}")
            print(f"CLIP dynamic: {quality['clip_score_dynamic']:.4f}")
        elif not args.skip_quality_eval:
            print("Quality evaluation needs --with-baseline; it was not run.")

        summary["timing_ms"] = {
            "model_load_ms": model_load_ms,
            "warmup_ms": warmup_ms,
            "baseline_generation_ms": baseline_timing["generation_ms"],
            "baseline_image_save_ms": baseline_timing["image_save_ms"],
            "dynamic_generation_ms": dynamic_timing["generation_ms"],
            "dynamic_text_encoder_ms": dynamic_timing["text_encoder_ms"],
            "dynamic_latent_init_ms": dynamic_timing["latent_init_ms"],
            "dynamic_diffusion_loop_ms": dynamic_timing["diffusion_loop_ms"],
            "dynamic_unet_ms": dynamic_timing["unet_ms"],
            "dynamic_scheduler_ms": dynamic_timing["scheduler_ms"],
            "dynamic_skip_prediction_ms": dynamic_timing["skip_prediction_ms"],
            "dynamic_pynq_feature_prepare_ms": dynamic_timing["pynq_feature_prepare_ms"],
            "dynamic_pynq_total_ms": dynamic_timing["pynq_total_ms"],
            "dynamic_network_round_trip_ms": dynamic_timing["network_round_trip_ms"],
            "dynamic_fpga_kernel_ms": dynamic_timing["fpga_kernel_ms"],
            "dynamic_pynq_server_ms": dynamic_timing["pynq_server_ms"],
            "dynamic_diffusion_loop_other_ms": dynamic_timing["diffusion_loop_other_ms"],
            "dynamic_vae_ms": dynamic_timing["vae_ms"],
            "dynamic_image_save_ms": dynamic_timing["image_save_ms"],
            "quality_evaluation_ms": quality_evaluation_ms,
        }
        timing_rows = build_timing_rows(
            model_load_ms,
            warmup_ms,
            baseline_timing,
            dynamic_timing,
            quality_evaluation_ms,
        )
        summary["report_file"] = "report.html"
        write_json(experiment_dir / "summary.json", summary)
        write_timing_summary(experiment_dir / "timing_summary.csv", timing_rows)
        report_path = generate_experiment_report(
            experiment_dir,
            summary,
            quality,
            timing_rows,
            step_rows,
        )
        print_timing_summary(timing_rows)
        print(f"HTML report        : {report_path}")

        print("\n" + "=" * 68)
        print("Baseline vs PYNQ dynamic result")
        print(f"Random seed       : {RUN_SEED}")
        if baseline_elapsed is not None:
            print(f"Baseline {BASE_STEPS} steps: {baseline_elapsed:.2f} s")
        print(
            f"PYNQ dynamic       : {dynamic_elapsed:.2f} s "
            f"({executed_steps} UNet steps, {skipped_steps} skipped)"
        )
        if speedup is not None:
            print(f"Speedup            : {speedup:.2f}x")
        print(f"Experiment files   : {experiment_dir}")
        print("=" * 68)
    finally:
        client.close()


if __name__ == "__main__":
    main()
