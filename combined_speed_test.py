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
from pynq_cosine_client import FEATURE_DOWNSAMPLE_FACTOR, PynqCosineClient


pipe = None
STEP_FIELDS = [
    "step",
    "timestep",
    "similarity",
    "threshold",
    "threshold_passed",
    "should_skip",
    "action",
    "skip_streak",
    "feature_bytes",
    "prepare_ms",
    "pynq_total_ms",
    "kernel_ms",
    "server_ms",
    "round_trip_ms",
    "unet_ms",
    "scheduler_ms",
    "step_total_ms",
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
        generator=make_generator(),
    ).images[0]
    sync()
    elapsed = time.perf_counter() - started
    image.save(filename)
    print(f"Baseline finished in {elapsed:.2f} s: {filename}")
    return elapsed, image


def run_with_dynamic_steps(filename, pynq_client):
    """Run SD while PYNQ makes every cosine-based skip decision."""
    print("\n=== Dynamic diffusion with PYNQ-Z2 decisions ===")

    pynq_client.reset()
    sync()
    started = time.perf_counter()

    text_inputs = pipe.tokenizer(
        [NEGATIVE_PROMPT, PROMPT],
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        prompt_embeds = pipe.text_encoder(text_inputs.input_ids)[0]

    pipe.scheduler.set_timesteps(BASE_STEPS, device=DEVICE)
    latents = torch.randn(
        (1, 4, 64, 64),
        generator=make_generator(),
        device=DEVICE,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    latents = latents * pipe.scheduler.init_noise_sigma

    prev_noise_pred = None
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
            threshold=SIMILARITY_THRESHOLD,
            warmup_steps=WARMUP_STEPS,
            max_consecutive_skips=MAX_CONSECUTIVE_SKIPS,
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
            f"kernel={decision.kernel_ms:.3f} ms "
            f"round-trip={decision.round_trip_ms:.3f} ms"
        )

        unet_ms = 0.0
        if should_skip:
            if prev_noise_pred is None:
                raise RuntimeError("PYNQ requested a skip before any UNet result exists")
            noise_pred = prev_noise_pred
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
            prev_noise_pred = noise_pred
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
                "threshold": SIMILARITY_THRESHOLD,
                "threshold_passed": int(decision.threshold_passed),
                "should_skip": int(should_skip),
                "action": action,
                "skip_streak": skip_streak,
                "feature_bytes": decision.feature_bytes,
                "prepare_ms": decision.prepare_ms,
                "pynq_total_ms": decision.prepare_ms + decision.round_trip_ms,
                "kernel_ms": decision.kernel_ms,
                "server_ms": decision.server_ms,
                "round_trip_ms": decision.round_trip_ms,
                "unet_ms": unet_ms,
                "scheduler_ms": scheduler_ms,
                "step_total_ms": (time.perf_counter() - step_started) * 1000.0,
            }
        )

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
    elapsed = time.perf_counter() - started
    image.save(filename)
    average_network_ms = (
        pynq_client.total_round_trip_ms / pynq_client.step_calls
        if pynq_client.step_calls
        else 0.0
    )
    print(
        f"Dynamic run finished: UNet={executed_steps}, skipped={skipped_steps}, "
        f"elapsed={elapsed:.2f} s"
    )
    print(
        f"PYNQ feature traffic={pynq_client.total_bytes_sent / 1024:.1f} KiB, "
        f"average round-trip={average_network_ms:.3f} ms"
    )
    print(f"Image saved to: {filename}")
    return elapsed, executed_steps, skipped_steps, image, step_rows


def warm_up_pipeline():
    print("\nWarming up the Stable Diffusion pipeline...")
    try:
        pipe(
            "warmup",
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=max(2, min(WARMUP_STEPS, 5)),
            guidance_scale=GUIDANCE,
        )
    except Exception as exc:
        print(f"Warmup was skipped: {exc}")


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
    global pipe
    args = parse_args()
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
        client.connect()
    except (ConnectionError, OSError) as exc:
        raise SystemExit(
            "Cannot connect to the PYNQ cosine server. "
            "Check the board IP and service status.\n"
            f"Details: {exc}"
        ) from exc

    print("PYNQ connection and CSK3 protocol check passed.")
    client.close()

    baseline_elapsed = None
    baseline_image = None
    try:
        pipe = create_pipe(use_dpm_solver=True)
        if not args.skip_warmup:
            warm_up_pipeline()
        if args.with_baseline:
            baseline_path = experiment_dir / "baseline.png"
            baseline_elapsed, baseline_image = run_original_steps(
                BASE_STEPS, baseline_path
            )
            baseline_image.save("combo_step100_baseline.png")

        client.connect()
        dynamic_path = experiment_dir / "pynq_dynamic.png"
        (
            dynamic_elapsed,
            executed_steps,
            skipped_steps,
            dynamic_image,
            step_rows,
        ) = run_with_dynamic_steps(dynamic_path, client)
        dynamic_image.save(args.output)
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
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "warmup_steps": WARMUP_STEPS,
            "max_consecutive_skips": MAX_CONSECUTIVE_SKIPS,
            "baseline_seconds": baseline_elapsed,
            "dynamic_seconds": dynamic_elapsed,
            "speedup": speedup,
            "pynq_feature_bytes": client.total_bytes_sent,
            "pynq_feature_dtype": "int8",
            "pynq_downsample_factor": FEATURE_DOWNSAMPLE_FACTOR,
            "skip_controller_location": "FPGA",
            "average_round_trip_ms": (
                client.total_round_trip_ms / client.step_calls
                if client.step_calls
                else 0.0
            ),
            "pynq_quantization_transfer_and_fpga_included_in_dynamic_time": True,
            "quality_evaluation_included_in_dynamic_time": False,
        }
        write_json(experiment_dir / "summary.json", summary)

        if baseline_image is not None and not args.skip_quality_eval:
            print("\n=== Full image quality evaluation (outside generation timing) ===")
            pipe = None
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            from quality_metrics import evaluate_quality

            quality = evaluate_quality(
                baseline_image, dynamic_image, PROMPT, DEVICE
            )
            write_json(experiment_dir / "quality_metrics.json", quality)
            print(f"PSNR       : {quality['psnr_dynamic_vs_baseline_db']:.4f} dB")
            print(f"SSIM       : {quality['ssim_dynamic_vs_baseline']:.6f}")
            print(f"LPIPS      : {quality['lpips_dynamic_vs_baseline']:.6f}")
            print(f"CLIP base   : {quality['clip_score_baseline']:.4f}")
            print(f"CLIP dynamic: {quality['clip_score_dynamic']:.4f}")
        elif not args.skip_quality_eval:
            print("Quality evaluation needs --with-baseline; it was not run.")

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
