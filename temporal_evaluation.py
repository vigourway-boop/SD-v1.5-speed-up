"""Reproducible calibration, held-out SD evaluation and real-board parity."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import platform
import random
import shutil
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import combined_speed_test as sd
from pc_skip_controller import PcCosineClient
from pynq_cosine_client import PynqCosineClient, compress_feature
from quality_metrics import QualityEvaluator


NEGATIVE = "blurry, low quality, deformed, distorted, bad anatomy, watermark, text"
CALIBRATION = [
    ("horse", "a photograph of a single brown horse standing in a green field, full body, daylight"),
    ("cabin", "a photograph of a small wooden cabin beside a mountain lake, daylight"),
    ("teapot", "a studio photograph of a white ceramic teapot on a gray table, soft light"),
]
VALIDATION = [
    ("cat", "a photograph of a cute orange kitten sitting on a windowsill, big bright eyes"),
    ("portrait", "a portrait photograph of a woman wearing a red scarf, daylight, neutral background"),
    ("street", "a photograph of a narrow street in a European town, colorful buildings, afternoon"),
    ("car", "a photograph of a red sports car parked beside a modern building, daylight"),
    ("flowers", "a still life photograph of sunflowers in a blue glass vase on a white table"),
    ("food", "a food photograph of a bowl of noodles with vegetables on a restaurant table"),
    ("robot", "a watercolor painting of a small friendly robot watering plants in a garden"),
    ("beach", "a photograph of a lighthouse above a rocky beach, ocean waves, clear daylight"),
]
SEEDS = {"calibration": [12345, 54321], "validation": [202601, 202602]}
BASE_PROFILE = {
    "middle_threshold": 0.9996, "late_margin": 0.00035,
    "late_min": 0.99945, "late_max": 0.99965,
    "warmup_ratio": 0.15, "max_consecutive_skips": 3,
    "distance_threshold": 2.0, "predictor_damping": 0.5,
}
PROFILES = {
    "current": dict(BASE_PROFILE),
    "balanced": {**BASE_PROFILE, "middle_threshold": 0.999,
                 "late_margin": 0.0008, "late_min": 0.9988, "late_max": 0.9992},
    "fast": {**BASE_PROFILE, "middle_threshold": 0.9985,
             "late_margin": 0.0012, "late_min": 0.998, "late_max": 0.9988},
}
GLOBAL_NAMES = ["PROMPT", "NEGATIVE_PROMPT", "RUN_SEED", "BASE_STEPS", "WARMUP_STEPS",
                "MIDDLE_SIMILARITY_THRESHOLD", "LATE_THRESHOLD_MARGIN",
                "LATE_THRESHOLD_MIN", "LATE_THRESHOLD_MAX", "MAX_CONSECUTIVE_SKIPS",
                "DYNAMIC_THRESHOLD_ENABLED"]


def stats(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": min(values), "max": max(values)}


def write_csv(path, rows, labels=None):
    if not rows:
        return
    keys = list(rows[0])
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([labels.get(k, k) if labels else k for k in keys])
        writer.writerows([[row.get(k) for k in keys] for row in rows])


def controller_config(profile, steps):
    return dict(enabled=True, total_steps=steps,
                warmup_steps=max(1, round(steps * profile["warmup_ratio"])),
                max_consecutive_skips=profile["max_consecutive_skips"],
                fixed_threshold=0.999, warmup_threshold=0.99995,
                middle_threshold=profile["middle_threshold"], late_start_ratio=0.7,
                late_margin=profile["late_margin"], late_min=profile["late_min"],
                late_max=profile["late_max"], ema_alpha=0.2,
                distance_threshold=profile["distance_threshold"])


@contextlib.contextmanager
def settings(prompt, seed, steps, profile):
    previous = {name: getattr(sd, name) for name in GLOBAL_NAMES}
    sd.PROMPT, sd.NEGATIVE_PROMPT, sd.RUN_SEED = prompt, NEGATIVE, seed
    sd.BASE_STEPS = steps
    sd.WARMUP_STEPS = max(1, round(steps * profile["warmup_ratio"]))
    sd.MIDDLE_SIMILARITY_THRESHOLD = profile["middle_threshold"]
    sd.LATE_THRESHOLD_MARGIN = profile["late_margin"]
    sd.LATE_THRESHOLD_MIN = profile["late_min"]
    sd.LATE_THRESHOLD_MAX = profile["late_max"]
    sd.MAX_CONSECUTIVE_SKIPS = profile["max_consecutive_skips"]
    sd.DYNAMIC_THRESHOLD_ENABLED = True
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(sd, name, value)


def feature_diagnostics(features):
    """Analyze raw adjacent latents separately from timed generation."""
    rows = []
    previous = previous_q = None
    for index, feature in enumerate(features):
        values = feature[0].float()
        pooled = F.avg_pool2d(values.unsqueeze(0), 2).squeeze(0).numpy().ravel()
        quantized = compress_feature(feature)
        scale = float(np.abs(pooled).max()) / 127.0
        reconstructed = quantized.astype(np.float64) * scale
        q_error = float(np.mean((pooled - reconstructed) ** 2))
        q_relative = q_error / max(float(np.mean(pooled ** 2)), 1e-30)
        if previous is not None:
            x, y = values.numpy().ravel().astype(np.float64), previous
            delta = x - y
            q_delta = quantized.astype(np.int16) - previous_q.astype(np.int16)
            bits = np.where(q_delta == 0, 0,
                            np.ceil(np.log2(np.abs(q_delta).astype(float) + 1)) + 1)
            rows.append({
                "step": index + 1,
                "raw_adjacent_cosine": float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))),
                "raw_normalized_distance": float(np.dot(delta, delta) / (np.dot(x, x) + np.dot(y, y))),
                "raw_delta_rms": float(np.sqrt(np.mean(delta ** 2))),
                "raw_exact_zero_fraction": float(np.mean(delta == 0)),
                "int8_delta_zero_fraction": float(np.mean(q_delta == 0)),
                "int8_delta_signed_bits_p95": float(np.percentile(bits, 95)),
                "quantization_relative_mse": q_relative,
                "feature_bytes_unique_fp16": values.numel() * 2,
                "feature_bytes_int8": quantized.nbytes,
            })
        previous, previous_q = values.numpy().ravel().astype(np.float64), quantized
    return rows


def parity_check(features, profile, board):
    pc = PcCosineClient()
    config = controller_config(profile, len(features))
    pc.configure(**config)
    board.close()
    board.connect()
    board.reset()
    board.configure(**config)
    records = []
    exact = ("should_skip", "cosine_passed", "distance_passed", "skip_streak",
             "threshold_q15", "distance_threshold_q20", "threshold_phase")
    numeric = ("similarity", "normalized_distance", "threshold_requested", "adjacent_similarity_ema")
    for index, feature in enumerate(features):
        # Alternate order; both backends receive exactly the same CPU tensor.
        if index % 2:
            hardware, software = board.decide(feature, index), pc.decide(feature, index)
        else:
            software, hardware = pc.decide(feature, index), board.decide(feature, index)
        mismatch = [name for name in exact if getattr(software, name) != getattr(hardware, name)]
        for name in numeric:
            x, y = getattr(software, name), getattr(hardware, name)
            if not ((math.isnan(x) and math.isnan(y)) or math.isclose(x, y, abs_tol=1e-12, rel_tol=1e-12)):
                mismatch.append(name)
        records.append({"step": index + 1, "mismatch": ",".join(mismatch),
                        "pc_ms": software.prepare_ms + software.decision_ms,
                        "pynq_ms": hardware.prepare_ms + hardware.decision_ms,
                        "network_round_trip_ms": hardware.round_trip_ms,
                        "arm_ip_call_ms": hardware.kernel_ms})
    return records


def summarize(rows):
    grouped = {}
    for row in rows:
        key = f"{row['split']}/{row['steps']}/{row['variant']}"
        grouped.setdefault(key, []).append(row)
    return {
        key: {"count": len(items),
              **{metric: stats(item[metric] for item in items)
                 for metric in ("speedup", "seconds", "psnr_db", "ssim", "lpips",
                                "clip_cosine_delta", "peak_allocated_mib", "executed")},
              "two_x_count": sum(item["speedup"] >= 2 for item in items)}
        for key, items in grouped.items()
    }


def select_profile(rows):
    """Select on calibration only. Quality limits are engineering heuristics."""
    groups = summarize(rows)
    candidates = []
    for name in PROFILES:
        g = groups[f"calibration/100/{name}"]
        eligible = (g["lpips"]["mean"] <= 0.10 and g["lpips"]["max"] <= 0.25
                    and g["clip_cosine_delta"]["mean"] >= -0.01)
        candidates.append((name, g, eligible))
    eligible = [item for item in candidates if item[2]]
    if eligible:
        best = max(eligible, key=lambda item: item[1]["speedup"]["mean"])
        reason = "Fastest calibration mean meeting the predeclared diagnostic quality limits"
    else:
        best = min(candidates, key=lambda item: item[1]["lpips"]["mean"])
        reason = "No candidate met diagnostic quality limits; lowest mean LPIPS retained for evaluation"
    return best[0], {"reason": reason, "quality_limits_met": best[2],
                     "calibration_only": True, "held_out_data_used": False,
                     "diagnostic_limits": {"mean_lpips_max": 0.10, "max_lpips_max": 0.25,
                                           "mean_clip_cosine_delta_min": -0.01}}


class Evaluation:
    def __init__(self, root, board, evaluator):
        self.root, self.board, self.evaluator = root, board, evaluator
        self.results = json.loads((root / "results.json").read_text(encoding="utf-8")) if (root / "results.json").exists() else []

    def checkpoint(self):
        sd.write_json(self.root / "results.json", self.results)
        sd.write_json(self.root / "aggregate.json", summarize(self.results))
        labels = {"split": "数据划分 / Split", "prompt_id": "提示词 / Prompt",
                  "seed": "种子 / Seed", "steps": "时间步 / Steps", "variant": "方案 / Variant",
                  "backend": "控制后端 / Backend", "seconds": "生成秒 / Seconds",
                  "baseline_seconds": "基准秒 / Baseline seconds", "speedup": "加速比 / Speedup",
                  "executed": "UNet执行 / UNet calls", "skipped": "跳步 / Skipped",
                  "psnr_db": "峰值信噪比 / PSNR dB", "ssim": "结构相似性 / SSIM",
                  "lpips": "感知距离 / LPIPS", "clip_cosine_delta": "文本对齐变化 / CLIP cosine delta",
                  "peak_allocated_mib": "显存分配峰值 / Peak allocated MiB",
                  "peak_reserved_mib": "显存保留峰值 / Peak reserved MiB",
                  "predictor_cache_bytes": "预测缓存字节 / Predictor cache bytes",
                  "network_feature_bytes": "网络特征字节 / Network feature bytes",
                  "distance_rejections": "距离拒绝次数 / Distance rejections",
                  "path": "结果目录 / Result directory"}
        write_csv(self.root / "results.csv", self.results, labels)

    def group(self, split, prompt_id, prompt, seed, steps, variants, include_half=False):
        group = self.root / split / f"{prompt_id}_{seed}_{steps}"
        group.mkdir(parents=True, exist_ok=True)
        tasks = [("baseline", None, None, steps)] + list(variants)
        if include_half:
            tasks.append(("original50", None, None, 50))
        expected = {t[0] for t in tasks if t[0] != "baseline"}
        existing = {r["variant"] for r in self.results if
                    (r["split"], r["prompt_id"], r["seed"], r["steps"]) == (split, prompt_id, seed, steps)}
        if existing == expected:
            print(f"Already complete: {split}/{prompt_id}/{seed}/{steps}", flush=True)
            return
        self.results = [r for r in self.results if
                        (r["split"], r["prompt_id"], r["seed"], r["steps"]) != (split, prompt_id, seed, steps)]
        order_seed = int.from_bytes(hashlib.sha256(f"{prompt_id}:{seed}:{steps}".encode()).digest()[:8], "big")
        random.Random(order_seed).shuffle(tasks)
        generated = {}
        print(f"{split}: {prompt_id} seed={seed} steps={steps}, variants={[t[0] for t in tasks]}", flush=True)
        for name, profile, backend, count in tasks:
            target = group / name
            target.mkdir(exist_ok=True)
            if backend == "pynq":
                self.board.close()
                self.board.connect()
            with settings(prompt, seed, count, profile or BASE_PROFILE):
                # Release unused allocator blocks equally before each measured run.
                if sd.DEVICE == "cuda":
                    torch.cuda.empty_cache()
                with (target / "run.log").open("w", encoding="utf-8") as log:
                    with contextlib.redirect_stdout(log):
                        if profile is None:
                            seconds, image, timing = sd.run_original_steps(count, target / "image.png")
                            executed, skipped, step_rows = count, 0, []
                        else:
                            client = self.board if backend == "pynq" else PcCosineClient()
                            seconds, executed, skipped, image, step_rows, timing = sd.run_with_dynamic_steps(
                                target / "image.png", client, profile["distance_threshold"],
                                "linear", profile["predictor_damping"], 1.5)
                        sd.sync()
                if step_rows:
                    sd.write_step_metrics(target / "step_metrics.csv", step_rows)
                record = {"split": split, "prompt_id": prompt_id, "prompt": prompt,
                          "negative_prompt": NEGATIVE, "seed": seed, "steps": steps,
                          "actual_scheduler_steps": count, "variant": name, "backend": backend or "none",
                          "seconds": seconds, "executed": executed, "skipped": skipped,
                          "profile": profile, "timing": timing, "order": [t[0] for t in tasks],
                          "image_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
                sd.write_json(target / "run.json", record)
                generated[name] = (image, record, target, step_rows)
                print(f"  {name}: {seconds:.3f}s, UNet={executed}", flush=True)
        baseline, base_record, _, _ = generated["baseline"]
        for name, (image, record, target, step_rows) in generated.items():
            if name == "baseline":
                continue
            quality = self.evaluator.evaluate(baseline, image, prompt)
            sd.write_json(target / "quality.json", quality)
            timing = record["timing"]
            row = {"split": split, "prompt_id": prompt_id, "seed": seed, "steps": steps,
                   "variant": name, "backend": record["backend"], "seconds": record["seconds"],
                   "baseline_seconds": base_record["seconds"],
                   "speedup": base_record["seconds"] / record["seconds"],
                   "executed": record["executed"], "skipped": record["skipped"],
                   "psnr_db": quality["psnr_dynamic_vs_baseline_db"],
                   "ssim": quality["ssim_dynamic_vs_baseline"],
                   "lpips": quality["lpips_dynamic_vs_baseline"],
                   "clip_cosine_delta": quality["clip_cosine_delta_dynamic_minus_baseline"],
                   "peak_allocated_mib": timing["cuda_peak_allocated_bytes"] / 2**20,
                   "peak_reserved_mib": timing["cuda_peak_reserved_bytes"] / 2**20,
                   "predictor_cache_bytes": timing.get("predictor_cache_bytes", 0),
                   "network_feature_bytes": sum(r["feature_bytes"] for r in step_rows) if record["backend"] == "pynq" else 0,
                   "distance_rejections": sum(r["cosine_passed"] and not r["distance_passed"] for r in step_rows),
                   "path": str(target.relative_to(self.root))}
            self.results.append(row)
            print(f"  metrics {name}: {row['speedup']:.3f}x, LPIPS={row['lpips']:.4f}, PSNR={row['psnr_db']:.2f}", flush=True)
        self.checkpoint()


def collect_traces(root, board, trace_root=None):
    all_features, all_diagnostics = [], []
    for prompt_id, prompt in CALIBRATION:
        features = []
        def capture(module, args):
            features.append(args[0][:1].detach().cpu().clone())
        target = root / "diagnostics" / prompt_id
        target.mkdir(parents=True, exist_ok=True)
        if trace_root is not None:
            source = Path(trace_root) / "diagnostics" / prompt_id
            with np.load(source / "latent_trace.npz") as stored:
                features = list(torch.from_numpy(stored["features"]))
            if len(features) != 100:
                raise ValueError("Expected a complete 100-step calibration trace")
            for filename in ("latent_trace.npz", "trace_image.png"):
                if (source / filename).resolve() != (target / filename).resolve():
                    shutil.copy2(source / filename, target / filename)
        else:
            handle = sd.pipe.unet.register_forward_pre_hook(capture)
            try:
                with settings(prompt, SEEDS["calibration"][0], 100, BASE_PROFILE):
                    sd.run_original_steps(100, target / "trace_image.png")
            finally:
                handle.remove()
            np.savez_compressed(target / "latent_trace.npz", features=torch.stack(features).numpy())
        diagnostics = feature_diagnostics(features)
        write_csv(target / "feature_statistics.csv", diagnostics,
                  {"step": "时间步 / Step", "raw_adjacent_cosine": "原始相邻余弦 / Raw adjacent cosine",
                   "raw_normalized_distance": "原始归一化距离 / Raw normalized distance",
                   "raw_delta_rms": "差值均方根 / Delta RMS",
                   "raw_exact_zero_fraction": "原始差值零比例 / Raw delta zero fraction",
                   "int8_delta_zero_fraction": "INT8差值零比例 / INT8 delta zero fraction",
                   "int8_delta_signed_bits_p95": "差值有符号位宽P95 / Delta signed bits P95",
                   "quantization_relative_mse": "量化相对均方误差 / Quantization relative MSE",
                   "feature_bytes_unique_fp16": "单份FP16字节 / Unique FP16 bytes",
                   "feature_bytes_int8": "INT8字节 / INT8 bytes"})
        all_diagnostics.extend(diagnostics)
        all_features.append(features)
    # Learn a gate from actual candidate-skip distances on calibration traces.
    learned = {}
    for name, profile in PROFILES.items():
        if name == "current":
            continue
        distances = []
        for features in all_features:
            client = PcCosineClient()
            client.configure(**controller_config(profile, len(features)))
            for index, feature in enumerate(features):
                decision = client.decide(feature, index)
                if decision.should_skip and math.isfinite(decision.normalized_distance):
                    distances.append(decision.normalized_distance)
        if not distances:
            raise RuntimeError("No candidate-skip distances available for calibration")
        profile["distance_threshold"] = float(np.quantile(distances, 0.90))
        learned[name] = {"distance_samples": len(distances), "quantile": 0.90,
                         "distance_threshold": profile["distance_threshold"],
                         "stats": stats(distances)}
    parity = []
    if board is not None:
        for name, profile in PROFILES.items():
            for trace_index, features in enumerate(all_features):
                records = parity_check(features, profile, board)
                parity.extend({"profile": name, "trace": trace_index, **r} for r in records)
        generator = torch.Generator().manual_seed(414)
        base = torch.randn(1, 4, 64, 64, generator=generator)
        edges = [torch.zeros_like(base), base, base, base, base, -base,
                 torch.zeros_like(base), torch.ones(1, 4, 32, 32)]
        parity.extend({"profile": "edge_cases", "trace": -1, **r}
                      for r in parity_check(edges, BASE_PROFILE, board))
        write_csv(root / "parity.csv", parity,
                  {"profile": "参数 / Profile", "trace": "轨迹 / Trace", "step": "时间步 / Step",
                   "mismatch": "不一致字段 / Mismatch", "pc_ms": "电脑毫秒 / PC ms",
                   "pynq_ms": "板端往返毫秒 / PYNQ ms", "network_round_trip_ms": "网络往返毫秒 / RTT ms",
                   "arm_ip_call_ms": "ARM调用IP毫秒 / ARM IP call ms"})
    summary = {"learned_distance_gates": learned,
               "feature_statistics": {k: stats(r[k] for r in all_diagnostics) for k in all_diagnostics[0]},
               "parity": {"performed": board is not None, "steps": len(parity),
                          "mismatches": sum(bool(r["mismatch"]) for r in parity),
                          "pc_ms": stats(r["pc_ms"] for r in parity),
                          "pynq_ms": stats(r["pynq_ms"] for r in parity)},
               "note": "Instrumented traces are excluded from speed claims. Independently scaled int8 deltas are not raw activation sparsity."}
    sd.write_json(root / "diagnostics.json", summary)
    if any(r["mismatch"] for r in parity):
        raise RuntimeError("PC/PYNQ parity failed; see parity.csv. Performance evaluation stopped.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("pynq", "pc"), default="pynq")
    parser.add_argument("--host", default=sd.PYNQ_HOST)
    parser.add_argument("--port", type=int, default=sd.PYNQ_PORT)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_root or f"experiments/temporal_{datetime.now():%Y%m%d_%H%M%S}")
    root.mkdir(parents=True, exist_ok=args.resume)
    if args.resume and not (root / "manifest.json").exists():
        raise ValueError("Resume requires an existing manifest")
    torch.set_num_threads(4)
    board = PynqCosineClient(args.host, args.port, timeout=5) if args.backend == "pynq" else None
    try:
        if board:
            board.connect_with_retry(1, 0)
        manifest = {"created": datetime.now().isoformat(), "backend": args.backend,
                    "calibration_prompts": CALIBRATION, "validation_prompts": VALIDATION,
                    "seeds": SEEDS, "negative_prompt": NEGATIVE,
                    "torch": torch.__version__, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                    "python": platform.python_version(), "model_id": sd.MODEL_ID,
                    "resolution": [512, 512], "guidance": sd.GUIDANCE,
                    "initial_profiles": PROFILES,
                    "selection_limits": {"mean_lpips_max": 0.10, "max_lpips_max": 0.25, "mean_clip_cosine_delta_min": -0.01},
                    "measurement": "Wall clock with CUDA synchronization, text encoding through VAE/image conversion; includes controller reset/config, compression and network; excludes load, warmup, PNG save and quality evaluation. No retries or auto fallback.",
                    "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('.').glob('*.py')}}
        if args.resume:
            previous = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            for key in ("backend", "calibration_prompts", "validation_prompts", "seeds", "negative_prompt", "source_sha256"):
                # JSON converts tuple prompt entries into lists.
                if previous[key] != json.loads(json.dumps(manifest[key])):
                    raise ValueError(f"Cannot resume with changed {key}; use a new output directory")
        else:
            manifest["trace_source"] = str(args.trace_root) if args.trace_root else None
            sd.write_json(root / "manifest.json", manifest)
        print(f"Output: {root.resolve()}", flush=True)
        sd.pipe = sd.create_pipe()
        sd.pipe.set_progress_bar_config(disable=True)
        sd.write_json(root / "scheduler.json", dict(sd.pipe.scheduler.config))
        if sd.warm_up_pipeline() <= 0:
            raise RuntimeError("Warmup failed; refusing to publish performance results")
        collect_traces(root, board, root if args.resume else args.trace_root)
        sd.write_json(root / "calibrated_candidates.json", PROFILES)
        # Quality models remain on CPU so they do not alter measured CUDA peaks.
        with QualityEvaluator("cpu") as evaluator:
            run = Evaluation(root, board, evaluator)
            for prompt_id, prompt in CALIBRATION:
                for seed in SEEDS["calibration"]:
                    variants = [(name, profile, args.backend, 100) for name, profile in PROFILES.items()]
                    run.group("calibration", prompt_id, prompt, seed, 100, variants)
            selected, selection = select_profile(run.results)
            profile = PROFILES[selected]
            sd.write_json(root / "selected_profile.json", {"name": selected, "profile": profile, **selection})
            print(f"Selected profile: {selected}: {selection}", flush=True)
            for prompt_index, (prompt_id, prompt) in enumerate(VALIDATION):
                for seed_index, seed in enumerate(SEEDS["validation"]):
                    variants = [("selected", profile, args.backend, 100),
                                ("current", BASE_PROFILE, args.backend, 100)]
                    if board and prompt_index < 3:
                        variants.append(("selected_pc", profile, "pc", 100))
                    run.group("validation", prompt_id, prompt, seed, 100, variants, include_half=True)
            for prompt_id, prompt in VALIDATION[:3]:
                for steps in (20, 30, 50, 75):
                    run.group("step_sweep", prompt_id, prompt, SEEDS["validation"][0], steps,
                              [("selected", profile, args.backend, steps)])
        sd.write_json(root / "completion.json", {"complete": True, "result_rows": len(run.results),
                                                 "selected": selected, "finished": datetime.now().isoformat()})
        print(f"Completed: {root.resolve()}", flush=True)
    except Exception as exc:
        sd.write_json(root / "failure.json", {"error": repr(exc), "finished": datetime.now().isoformat()})
        raise
    finally:
        if board:
            board.close()
        sd.pipe = None


if __name__ == "__main__":
    main()
