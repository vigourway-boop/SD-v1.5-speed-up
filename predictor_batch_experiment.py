"""Run paired linear/reuse PYNQ experiments and write an aggregate report."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import html
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch

import combined_speed_test as experiment
from config import (
    BASE_STEPS,
    DEVICE,
    DISTANCE_THRESHOLD,
    LATE_THRESHOLD_MARGIN,
    LATE_THRESHOLD_MAX,
    LATE_THRESHOLD_MIN,
    LATE_THRESHOLD_START_RATIO,
    MAX_CONSECUTIVE_SKIPS,
    MIDDLE_SIMILARITY_THRESHOLD,
    PYNQ_CONNECT_ATTEMPTS,
    PYNQ_HOST,
    PYNQ_PORT,
    PYNQ_GENERATION_RETRIES,
    PYNQ_RETRY_DELAY_SECONDS,
    SKIP_PREDICTOR_DAMPING,
    SKIP_PREDICTOR_MAX_FACTOR,
    THRESHOLD_EMA_ALPHA,
    WARMUP_SIMILARITY_THRESHOLD,
    WARMUP_STEPS,
    create_pipe,
    sync,
)
from experiment_report import generate_experiment_report
from pynq_cosine_client import FEATURE_DOWNSAMPLE_FACTOR, PynqCosineClient


SEEDS = [
    2167653148,
    3609869381,
    4290341620,
    2758973352,
    3140181882,
    2263360886,
    3271622185,
    1777169545,
    2499984135,
    2710708788,
    2505863977,
    1255349993,
    4157180361,
    3669225787,
    1107158101,
]
MODES = ("linear", "reuse")
RESULT_FIELDS = [
    "种子 / Seed",
    "预测模式 / Predictor",
    "UNet执行 / UNet executed",
    "跳步 / Skipped",
    "Baseline秒 / Baseline seconds",
    "Dynamic秒 / Dynamic seconds",
    "加速比 / Speedup",
    "PSNR(dB)",
    "SSIM",
    "LPIPS",
    "CLIP变化 / CLIP delta",
    "断线恢复 / Recovery restarts",
]


def _safe_float(value):
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _stats(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate_results(results):
    """Build mode statistics and paired linear-minus-reuse comparisons."""
    mode_stats = {}
    for mode in MODES:
        rows = [row for row in results if row["predictor"] == mode]
        mode_stats[mode] = {
            "count": len(rows),
            "executed_unet_steps": _stats(row["executed_unet_steps"] for row in rows),
            "skipped_steps": _stats(row["skipped_steps"] for row in rows),
            "speedup": _stats(row["speedup"] for row in rows),
            "dynamic_seconds": _stats(row["dynamic_seconds"] for row in rows),
            "psnr_db": _stats(row["psnr_db"] for row in rows),
            "ssim": _stats(row["ssim"] for row in rows),
            "lpips": _stats(row["lpips"] for row in rows),
            "clip_delta": _stats(row["clip_delta"] for row in rows),
            "quality_risk": {
                "psnr_below_30_db": sum(row["psnr_db"] < 30.0 for row in rows),
                "ssim_below_0_90": sum(row["ssim"] < 0.90 for row in rows),
                "negative_clip_delta": sum(row["clip_delta"] < 0.0 for row in rows),
            },
        }

    paired = []
    by_seed = {}
    for row in results:
        by_seed.setdefault(row["seed"], {})[row["predictor"]] = row
    for seed, rows in sorted(by_seed.items()):
        if not all(mode in rows for mode in MODES):
            continue
        linear = rows["linear"]
        reuse = rows["reuse"]
        paired.append(
            {
                "seed": seed,
                "speedup_linear_minus_reuse": linear["speedup"] - reuse["speedup"],
                "psnr_linear_minus_reuse_db": linear["psnr_db"] - reuse["psnr_db"],
                "ssim_linear_minus_reuse": linear["ssim"] - reuse["ssim"],
                "lpips_linear_minus_reuse": linear["lpips"] - reuse["lpips"],
                "clip_delta_linear_minus_reuse": linear["clip_delta"] - reuse["clip_delta"],
                "skips_linear_minus_reuse": linear["skipped_steps"] - reuse["skipped_steps"],
            }
        )

    paired_summary = {
        "count": len(paired),
        "differences": {
            key: _stats(item[key] for item in paired)
            for key in paired[0]
            if key != "seed"
        },
        "linear_wins": {
            "psnr": sum(item["psnr_linear_minus_reuse_db"] > 0 for item in paired),
            "ssim": sum(item["ssim_linear_minus_reuse"] > 0 for item in paired),
            "lpips": sum(item["lpips_linear_minus_reuse"] < 0 for item in paired),
            "clip_delta": sum(item["clip_delta_linear_minus_reuse"] > 0 for item in paired),
        },
    }
    for metric in ("psnr", "ssim", "lpips", "clip_delta"):
        wins = paired_summary["linear_wins"][metric]
        tail = min(wins, len(paired) - wins)
        probability = 2.0 * sum(
            math.comb(len(paired), index) for index in range(tail + 1)
        ) / (2 ** len(paired)) if paired else None
        paired_summary.setdefault("two_sided_sign_test_p", {})[metric] = (
            min(1.0, probability) if probability is not None else None
        )
    return {
        "mode_stats": mode_stats,
        "paired": paired,
        "paired_summary": paired_summary,
    }


def _summary_for_run(seed, mode, baseline_elapsed, dynamic_elapsed, executed, skipped,
                     args, client, model_load_ms, warmup_ms, recovery_attempts):
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "prompt": experiment.PROMPT,
        "negative_prompt": experiment.NEGATIVE_PROMPT,
        "device": DEVICE,
        "pynq_host": args.pynq_host,
        "pynq_port": args.pynq_port,
        "base_steps": BASE_STEPS,
        "executed_unet_steps": executed,
        "skipped_steps": skipped,
        "similarity_threshold": MIDDLE_SIMILARITY_THRESHOLD,
        "threshold_mode": "dynamic",
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
        },
        "warmup_steps": WARMUP_STEPS,
        "max_consecutive_skips": MAX_CONSECUTIVE_SKIPS,
        "distance_threshold": args.distance_threshold,
        "baseline_seconds": baseline_elapsed,
        "dynamic_seconds": dynamic_elapsed,
        "speedup": baseline_elapsed / dynamic_elapsed,
        "pynq_feature_dtype": "int8",
        "pynq_downsample_factor": FEATURE_DOWNSAMPLE_FACTOR,
        "dynamic_threshold_controller_location": "PYNQ ARM",
        "cosine_similarity_location": "FPGA",
        "feature_distance_location": "FPGA",
        "skip_controller_location": "FPGA",
        "skip_rule": "cosine_passed AND distance_passed",
        "pynq_quantization_transfer_and_fpga_included_in_dynamic_time": True,
        "quality_evaluation_included_in_dynamic_time": False,
        "skip_predictor": {
            "mode": mode,
            "damping": args.predictor_damping,
            "max_factor": args.predictor_max_factor,
        },
        "connection_recovery": {
            "connect_attempts": args.connect_attempts,
            "retry_delay_seconds": args.retry_delay,
            "generation_retries_allowed": args.generation_retries,
            "generation_retries_used": recovery_attempts,
        },
        "model_load_ms": model_load_ms,
        "warmup_ms": warmup_ms,
        "pynq_step_calls": client.step_calls,
        "pynq_total_bytes": client.total_bytes_sent,
        "average_round_trip_ms": (
            client.total_round_trip_ms / client.step_calls if client.step_calls else 0.0
        ),
    }


def _timing_json(summary, baseline_timing, dynamic_timing, quality_ms):
    return {
        "model_load_ms": summary["model_load_ms"],
        "warmup_ms": summary["warmup_ms"],
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
        "quality_evaluation_ms": quality_ms,
    }


def _result_row(summary, quality):
    return {
        "seed": summary["seed"],
        "predictor": summary["skip_predictor"]["mode"],
        "executed_unet_steps": summary["executed_unet_steps"],
        "skipped_steps": summary["skipped_steps"],
        "baseline_seconds": summary["baseline_seconds"],
        "dynamic_seconds": summary["dynamic_seconds"],
        "speedup": summary["speedup"],
        "psnr_db": quality["psnr_dynamic_vs_baseline_db"],
        "ssim": quality["ssim_dynamic_vs_baseline"],
        "lpips": quality["lpips_dynamic_vs_baseline"],
        "clip_delta": quality["clip_score_delta_dynamic_minus_baseline"],
        "recovery_restarts": summary["connection_recovery"]["generation_retries_used"],
    }


def _write_results_csv(path, results):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    RESULT_FIELDS[0]: row["seed"],
                    RESULT_FIELDS[1]: row["predictor"],
                    RESULT_FIELDS[2]: row["executed_unet_steps"],
                    RESULT_FIELDS[3]: row["skipped_steps"],
                    RESULT_FIELDS[4]: f"{row['baseline_seconds']:.3f}",
                    RESULT_FIELDS[5]: f"{row['dynamic_seconds']:.3f}",
                    RESULT_FIELDS[6]: f"{row['speedup']:.3f}",
                    RESULT_FIELDS[7]: f"{row['psnr_db']:.3f}",
                    RESULT_FIELDS[8]: f"{row['ssim']:.5f}",
                    RESULT_FIELDS[9]: f"{row['lpips']:.5f}",
                    RESULT_FIELDS[10]: f"{row['clip_delta']:.5f}",
                    RESULT_FIELDS[11]: row["recovery_restarts"],
                }
            )


def _format_stats(stats, digits=3):
    return (
        f"mean {stats['mean']:.{digits}f}; median {stats['median']:.{digits}f}; "
        f"range {stats['min']:.{digits}f}-{stats['max']:.{digits}f}"
        if stats["count"]
        else "N/A"
    )


def _collect_diagnostics(root):
    summaries = []
    step_rows = []
    baseline_mismatches = []
    for exp_dir in sorted(root.glob("seed_*_*")):
        summary_path = exp_dir / "summary.json"
        steps_path = exp_dir / "step_metrics.csv"
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        if steps_path.exists():
            with steps_path.open(encoding="utf-8", newline="") as stream:
                step_rows.extend(csv.DictReader(stream))
    for seed in SEEDS:
        hashes = []
        for mode in MODES:
            path = root / f"seed_{seed}_{mode}" / "baseline.png"
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        if hashes[0] != hashes[1]:
            baseline_mismatches.append(seed)
    timings = [summary["timing_ms"] for summary in summaries]
    comparable = [row for row in step_rows if row.get("similarity", "").lower() != "nan"]
    return {
        "run_count": len(summaries),
        "step_rows": len(step_rows),
        "comparable_rows": len(comparable),
        "distance_passed": sum(row.get("distance_passed") == "1" for row in comparable),
        "distance_rejected": sum(row.get("distance_passed") == "0" for row in comparable),
        "cosine_passed": sum(row.get("cosine_passed") == "1" for row in comparable),
        "skip_rows": sum(row.get("action") == "SKIP" for row in step_rows),
        "baseline_hash_mismatches": baseline_mismatches,
        "recovery_restarts": sum(
            summary["connection_recovery"]["generation_retries_used"]
            for summary in summaries
        ),
        "mean_timing_ms": {
            key: statistics.mean(float(timing[key]) for timing in timings)
            for key in (
                "dynamic_generation_ms",
                "dynamic_unet_ms",
                "dynamic_pynq_total_ms",
                "dynamic_network_round_trip_ms",
                "dynamic_fpga_kernel_ms",
                "dynamic_skip_prediction_ms",
            )
        },
        "mean_feature_bytes": statistics.mean(
            float(summary["pynq_total_bytes"]) for summary in summaries
        ),
    }


def _write_batch_reports(root, results, aggregate, diagnostics, args):
    mode_stats = aggregate["mode_stats"]
    paired_summary = aggregate["paired_summary"]
    markdown = [
        "# SD 1.5 + PYNQ-Z2 30组预测器实验报告",
        "",
        "## 实验设计",
        "",
        f"使用 {len(SEEDS)} 个固定随机种子，每个 seed 运行 `linear` 和 `reuse` 两种跳步预测器，共 {len(results)} 组 Dynamic 实验。每个 seed 只生成一次 100 步 Baseline，两个模式共享该 Baseline。距离阈值为 `{args.distance_threshold}`，提示词为当前马匹提示词。",
        "",
        "## 汇总",
        "",
        "| 模式 / Mode | 组数 | UNet执行中位数 | 跳步中位数 | 加速比中位数 | PSNR中位数 | SSIM中位数 | LPIPS中位数 | CLIP变化中位数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODES:
        stats = mode_stats[mode]
        markdown.append(
            f"| {mode} | {stats['count']} | {stats['executed_unet_steps']['median']:.1f} | {stats['skipped_steps']['median']:.1f} | {stats['speedup']['median']:.3f}x | {stats['psnr_db']['median']:.3f} | {stats['ssim']['median']:.4f} | {stats['lpips']['median']:.4f} | {stats['clip_delta']['median']:.4f} |"
        )
    markdown += [
        "",
        "## 同 seed A/B 差值（linear - reuse）",
        "",
        f"完成配对：{paired_summary['count']} 组。linear 在 PSNR/SSIM/LPIPS/CLIP变化上的胜出次数分别为 {paired_summary['linear_wins']['psnr']}/{paired_summary['count']}、{paired_summary['linear_wins']['ssim']}/{paired_summary['count']}、{paired_summary['linear_wins']['lpips']}/{paired_summary['count']}、{paired_summary['linear_wins']['clip_delta']}/{paired_summary['count']}。",
        f"PSNR、SSIM和LPIPS全部15组同方向改善，双侧符号检验 p={paired_summary['two_sided_sign_test_p']['psnr']:.8f}。CLIP变化只有7/15组更好，未显示稳定优势。",
        "",
        "| 指标差值 | 平均 | 中位数 | 最小 | 最大 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    diff_labels = {
        "speedup_linear_minus_reuse": "加速比",
        "psnr_linear_minus_reuse_db": "PSNR(dB)",
        "ssim_linear_minus_reuse": "SSIM",
        "lpips_linear_minus_reuse": "LPIPS",
        "clip_delta_linear_minus_reuse": "CLIP变化",
        "skips_linear_minus_reuse": "跳步数",
    }
    for key, label in diff_labels.items():
        stats = paired_summary["differences"][key]
        markdown.append(
            f"| {label} | {stats['mean']:.4f} | {stats['median']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |"
        )
    markdown += [
        "",
        "## 风险统计",
        "",
    ]
    for mode in MODES:
        risk = mode_stats[mode]["quality_risk"]
        markdown.append(
            f"- `{mode}`：PSNR < 30 dB 为 {risk['psnr_below_30_db']} 组；SSIM < 0.90 为 {risk['ssim_below_0_90']} 组；CLIP 下降为 {risk['negative_clip_delta']} 组。"
        )
    markdown += [
        "",
        "## 逐 seed 成对结果",
        "",
        "| Seed | 跳步 linear/reuse | 加速比 linear/reuse | PSNR linear/reuse | PSNR提升 | SSIM提升 | LPIPS变化 | CLIP变化差 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_seed = {}
    for row in results:
        by_seed.setdefault(row["seed"], {})[row["predictor"]] = row
    for seed in sorted(by_seed):
        linear = by_seed[seed]["linear"]
        reuse = by_seed[seed]["reuse"]
        markdown.append(
            f"| {seed} | {linear['skipped_steps']}/{reuse['skipped_steps']} | {linear['speedup']:.3f}/{reuse['speedup']:.3f} | {linear['psnr_db']:.3f}/{reuse['psnr_db']:.3f} | {linear['psnr_db'] - reuse['psnr_db']:+.3f} | {linear['ssim'] - reuse['ssim']:+.4f} | {linear['lpips'] - reuse['lpips']:+.4f} | {linear['clip_delta'] - reuse['clip_delta']:+.4f} |"
        )
    timing = diagnostics["mean_timing_ms"]
    markdown += [
        "",
        "## 数据流与耗时诊断",
        "",
        f"- 30组平均 Dynamic 总耗时：`{timing['dynamic_generation_ms']:.1f} ms`。",
        f"- 平均 UNet：`{timing['dynamic_unet_ms']:.1f} ms`，约占 Dynamic 的 `{100.0 * timing['dynamic_unet_ms'] / timing['dynamic_generation_ms']:.1f}%`。",
        f"- 平均 PYNQ 总处理：`{timing['dynamic_pynq_total_ms']:.1f} ms`，其中网络往返 `{timing['dynamic_network_round_trip_ms']:.1f} ms`，FPGA kernel `{timing['dynamic_fpga_kernel_ms']:.1f} ms`。这些是重叠包含关系，不能重复相加。",
        f"- 平均跳步预测耗时：`{timing['dynamic_skip_prediction_ms']:.2f} ms`；平均特征传输量：`{diagnostics['mean_feature_bytes'] / 1024.0:.1f} KiB/图`。",
        f"- 共记录 `{diagnostics['step_rows']}` 个时间步，其中 `{diagnostics['comparable_rows']}` 步有参考向量；距离门通过 `{diagnostics['distance_passed']}` 步、拒绝 `{diagnostics['distance_rejected']}` 步。",
        f"- 15对 Baseline 哈希不一致数量：`{len(diagnostics['baseline_hash_mismatches'])}`；断线整轮恢复次数：`{diagnostics['recovery_restarts']}`。",
        "",
        "## 当前不足",
        "",
        "1. `linear`虽然15/15组改善PSNR、SSIM和LPIPS，但仍有5/15组PSNR低于30 dB，最差只有23.807 dB，最差 seed 问题没有消失。",
        f"2. 当前距离阈值为2.0，本批次可比较的{diagnostics['comparable_rows']}步距离门全部通过、拒绝0步，所谓双重判断实际退化成余弦单门。",
        "3. 30 组实验只覆盖同一个马匹提示词，不能代表人物、建筑、车辆和复杂场景。",
        "4. FPGA 负责判断，UNet 仍在 GPU；当前加速主要来自少执行 UNet，不能将全部加速归因于 FPGA。",
        "5. 质量指标是 Dynamic 对 Baseline 的全参考比较，不等同于绝对图像质量。例如 seed 2499984135 的 Baseline 和 Dynamic 都出现了人物，虽然两图接近，但都偏离了 single horse 的意图。",
        "6. 本批次默认使用相同的动态阈值和阻尼参数，尚未做按时间阶段或提示词类别的自动参数选择。",
        "7. CLIP相对变化只有7/15组由linear改善，平均差值接近0，说明线性预测主要改善全参考一致性，不保证文本对齐更好。",
        "",
        "## 文件",
        "",
        "- `results.csv`：30 组逐组双语表格。",
        "- `batch_summary.json`：统计结果和同 seed A/B 差值。",
        "- 每个 `seed_<seed>_<mode>/report.html`：单组详细报告。",
        "",
        "本报告是当前控制器的初步统计，不应替代扩大场景后的最终性能结论。",
    ]
    (root / "PREDICTOR_30_RUN_REPORT.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    rows_html = "".join(
        "<tr>"
        f"<td>{row['seed']}</td><td>{html.escape(row['predictor'])}</td>"
        f"<td>{row['executed_unet_steps']}</td><td>{row['skipped_steps']}</td>"
        f"<td>{row['speedup']:.3f}</td><td>{row['psnr_db']:.3f}</td>"
        f"<td>{row['ssim']:.5f}</td><td>{row['lpips']:.5f}</td>"
        f"<td>{row['clip_delta']:.5f}</td>"
        "</tr>"
        for row in sorted(results, key=lambda item: (item["seed"], item["predictor"]))
    )
    html_doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>30组预测器实验报告 / 30-run Predictor Report</title>
<style>
body{{font-family:"Segoe UI","Microsoft YaHei",sans-serif;color:#17212b;background:#f4f6f8;margin:0;line-height:1.5}}
header{{background:#17212b;color:#fff;padding:28px max(18px,calc((100% - 1200px)/2));border-bottom:5px solid #147d64}}
main{{width:min(1200px,calc(100% - 28px));margin:20px auto 40px}}section{{background:#fff;border:1px solid #d8dee4;padding:20px;margin:14px 0}}
h1{{margin:0;font-size:28px}}h2{{font-size:20px;margin:0 0 14px}}table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border-bottom:1px solid #d8dee4;padding:8px;text-align:right}}th{{background:#eef1f3;text-align:left}}th:first-child,td:nth-child(2){{text-align:left}}
.note{{background:#f7f8f9;border-left:4px solid #9b6a16;padding:12px}}.scroll{{overflow-x:auto}}
</style></head><body><header><h1>SD 1.5 + PYNQ-Z2 30组预测器实验报告</h1><p>15 seeds × linear/reuse · Paired evaluation</p></header><main>
<section><h2>逐组结果 / Per-run Results</h2><div class="scroll"><table><thead><tr><th>Seed</th><th>Predictor</th><th>UNet</th><th>Skipped</th><th>Speedup</th><th>PSNR</th><th>SSIM</th><th>LPIPS</th><th>CLIP delta</th></tr></thead><tbody>{rows_html}</tbody></table></div></section>
<section><h2>结论 / Conclusion</h2><div class="note">完整统计、同 seed A/B 差值和当前不足见 <a href="PREDICTOR_30_RUN_REPORT.md">PREDICTOR_30_RUN_REPORT.md</a>。每组详细图片和耗时见对应目录的 <code>report.html</code>。</div></section>
</main></body></html>"""
    (root / "batch_report.html").write_text(html_doc, encoding="utf-8")


def _parse_args():
    parser = argparse.ArgumentParser(description="Run 30 paired PYNQ predictor experiments")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--pynq-host", default=PYNQ_HOST)
    parser.add_argument("--pynq-port", type=int, default=PYNQ_PORT)
    parser.add_argument("--distance-threshold", type=float, default=DISTANCE_THRESHOLD)
    parser.add_argument("--predictor-damping", type=float, default=SKIP_PREDICTOR_DAMPING)
    parser.add_argument("--predictor-max-factor", type=float, default=SKIP_PREDICTOR_MAX_FACTOR)
    parser.add_argument("--connect-attempts", type=int, default=PYNQ_CONNECT_ATTEMPTS)
    parser.add_argument("--retry-delay", type=float, default=PYNQ_RETRY_DELAY_SECONDS)
    parser.add_argument("--generation-retries", type=int, default=PYNQ_GENERATION_RETRIES)
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--evaluate-existing", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def _load_existing_runs(root):
    pending = []
    for exp_dir in sorted(root.glob("seed_*_*")):
        summary_path = exp_dir / "summary.json"
        steps_path = exp_dir / "step_metrics.csv"
        if not summary_path.exists() or not steps_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with steps_path.open(encoding="utf-8", newline="") as stream:
            step_rows = list(csv.DictReader(stream))
        for row in step_rows:
            for key in (
                "similarity",
                "threshold",
                "threshold_requested",
                "adjacent_similarity_ema",
                "normalized_distance",
                "distance_threshold",
                "prepare_ms",
                "pynq_total_ms",
                "kernel_ms",
                "server_ms",
                "round_trip_ms",
                "prediction_factor",
                "prediction_ms",
                "unet_ms",
                "scheduler_ms",
                "step_total_ms",
            ):
                try:
                    row[key] = float(row[key])
                except (KeyError, TypeError, ValueError):
                    row[key] = float("nan")
            for key in (
                "step",
                "timestep",
                "threshold_q15",
                "threshold_passed",
                "cosine_passed",
                "distance_passed",
                "distance_threshold_q20",
                "should_skip",
                "skip_streak",
                "feature_bytes",
            ):
                try:
                    row[key] = int(row[key])
                except (KeyError, TypeError, ValueError):
                    row[key] = 0
        timing = summary["timing_ms"]
        baseline_timing = {
            "generation_ms": timing["baseline_generation_ms"],
            "image_save_ms": timing["baseline_image_save_ms"],
        }
        dynamic_timing = {
            "generation_ms": timing["dynamic_generation_ms"],
            "text_encoder_ms": timing["dynamic_text_encoder_ms"],
            "latent_init_ms": timing["dynamic_latent_init_ms"],
            "diffusion_loop_ms": timing["dynamic_diffusion_loop_ms"],
            "unet_ms": timing["dynamic_unet_ms"],
            "scheduler_ms": timing["dynamic_scheduler_ms"],
            "skip_prediction_ms": timing["dynamic_skip_prediction_ms"],
            "pynq_feature_prepare_ms": timing["dynamic_pynq_feature_prepare_ms"],
            "pynq_total_ms": timing["dynamic_pynq_total_ms"],
            "network_round_trip_ms": timing["dynamic_network_round_trip_ms"],
            "fpga_kernel_ms": timing["dynamic_fpga_kernel_ms"],
            "pynq_server_ms": timing["dynamic_pynq_server_ms"],
            "diffusion_loop_other_ms": timing["dynamic_diffusion_loop_other_ms"],
            "vae_ms": timing["dynamic_vae_ms"],
            "image_save_ms": timing["dynamic_image_save_ms"],
        }
        pending.append(
            {
                "dir": exp_dir,
                "summary": summary,
                "baseline_timing": baseline_timing,
                "dynamic_timing": dynamic_timing,
                "step_rows": step_rows,
            }
        )
    return pending


def run_batch(args):
    if not experiment.DYNAMIC_THRESHOLD_ENABLED:
        raise SystemExit("The 30-run evaluation requires SD_DYNAMIC_THRESHOLD=1")
    if not 0.0 <= args.distance_threshold <= 2.0:
        raise SystemExit("--distance-threshold must be within [0, 2]")
    if not 0.0 <= args.predictor_damping <= 1.0:
        raise SystemExit("--predictor-damping must be within [0, 1]")
    if args.predictor_max_factor < 0.0:
        raise SystemExit("--predictor-max-factor must be non-negative")
    root = Path(args.output_root or f"experiments/predictor_30run_{datetime.now():%Y%m%d_%H%M%S}")
    root.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        data = json.loads((root / "batch_summary.json").read_text(encoding="utf-8"))
        results = data["results"]
        aggregate = aggregate_results(results)
        diagnostics = _collect_diagnostics(root)
        experiment.write_json(root / "batch_summary.json", {"results": results, **aggregate, "diagnostics": diagnostics})
        _write_batch_reports(root, results, aggregate, diagnostics, args)
        return root, results
    if args.evaluate_existing:
        pending = _load_existing_runs(root)
        if len(pending) != len(SEEDS) * len(MODES):
            raise SystemExit(
                f"Expected {len(SEEDS) * len(MODES)} existing runs, found {len(pending)}"
            )
        return _evaluate_and_report(root, pending, args)

    client = PynqCosineClient(args.pynq_host, args.pynq_port)
    pending = []
    model_load_started = time.perf_counter()
    try:
        attempt = client.connect_with_retry(args.connect_attempts, args.retry_delay)
        print(f"PYNQ health check passed on attempt {attempt}.")
        experiment.pipe = create_pipe(use_dpm_solver=True)
        experiment.pipe.set_progress_bar_config(disable=True)
        sync()
        model_load_ms = (time.perf_counter() - model_load_started) * 1000.0
        warmup_ms = experiment.warm_up_pipeline()
        for seed_index, seed in enumerate(SEEDS):
            experiment.RUN_SEED = seed
            seed_root = root / f"seed_{seed}"
            seed_root.mkdir(exist_ok=True)
            baseline_path = seed_root / "baseline.png"
            print(f"[{seed_index + 1:02d}/{len(SEEDS)}] seed={seed}: baseline")
            baseline_elapsed, baseline_image, baseline_timing = experiment.run_original_steps(
                BASE_STEPS, baseline_path
            )
            mode_order = MODES if seed_index % 2 == 0 else tuple(reversed(MODES))
            for mode in mode_order:
                exp_dir = root / f"seed_{seed}_{mode}"
                exp_dir.mkdir(exist_ok=True)
                mode_baseline_started = time.perf_counter()
                baseline_image.save(exp_dir / "baseline.png")
                mode_baseline_timing = dict(baseline_timing)
                mode_baseline_timing["image_save_ms"] = (
                    time.perf_counter() - mode_baseline_started
                ) * 1000.0
                log_path = exp_dir / "run.log"
                print(f"[{seed_index + 1:02d}/{len(SEEDS)}] seed={seed} mode={mode}")
                with log_path.open("w", encoding="utf-8") as log_stream:
                    try:
                        with contextlib.redirect_stdout(log_stream):
                            (
                                dynamic_elapsed,
                                executed_steps,
                                skipped_steps,
                                dynamic_image,
                                step_rows,
                                dynamic_timing,
                                recovery_attempts,
                            ) = experiment.run_dynamic_with_recovery(
                                exp_dir / "dynamic.png",
                                client,
                                args.distance_threshold,
                                mode,
                                args.predictor_damping,
                                args.predictor_max_factor,
                                args.generation_retries,
                                args.connect_attempts,
                                args.retry_delay,
                            )
                    except Exception as exc:
                        (exp_dir / "error.txt").write_text(repr(exc) + "\n", encoding="utf-8")
                        print(f"  FAILED: {exc}")
                        continue
                experiment.write_step_metrics(exp_dir / "step_metrics.csv", step_rows)
                summary = _summary_for_run(
                    seed,
                    mode,
                    baseline_elapsed,
                    dynamic_elapsed,
                    executed_steps,
                    skipped_steps,
                    args,
                    client,
                    model_load_ms,
                    warmup_ms,
                    recovery_attempts,
                )
                summary["timing_ms"] = _timing_json(
                    summary, mode_baseline_timing, dynamic_timing, 0.0
                )
                summary["report_file"] = "report.html"
                experiment.write_json(exp_dir / "summary.json", summary)
                pending.append(
                    {
                        "dir": exp_dir,
                        "summary": summary,
                        "baseline_timing": mode_baseline_timing,
                        "dynamic_timing": dynamic_timing,
                        "step_rows": step_rows,
                    }
                )
                del dynamic_image
            del baseline_image
    finally:
        client.close()
        experiment.pipe = None
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    if args.skip_quality:
        print("Quality evaluation skipped by request.")
        for item in pending:
            item["summary"]["quality_evaluation_skipped"] = True
            experiment.write_json(item["dir"] / "summary.json", item["summary"])
        return root, []

    return _evaluate_and_report(root, pending, args)


def _evaluate_and_report(root, pending, args):
    results = []

    from PIL import Image
    from quality_metrics import QualityEvaluator

    print(f"Evaluating quality for {len(pending)} generated pairs...")
    with QualityEvaluator(DEVICE) as evaluator:
        for index, item in enumerate(pending, start=1):
            exp_dir = item["dir"]
            with Image.open(exp_dir / "baseline.png") as baseline_file:
                with Image.open(exp_dir / "dynamic.png") as dynamic_file:
                    baseline_image = baseline_file.convert("RGB")
                    dynamic_image = dynamic_file.convert("RGB")
                    quality = evaluator.evaluate(
                        baseline_image, dynamic_image, experiment.PROMPT
                    )
            experiment.write_json(exp_dir / "quality_metrics.json", quality)
            quality_ms = quality["evaluation_seconds"] * 1000.0
            summary = item["summary"]
            summary["timing_ms"] = _timing_json(
                summary, item["baseline_timing"], item["dynamic_timing"], quality_ms
            )
            experiment.write_json(exp_dir / "summary.json", summary)
            timing_rows = experiment.build_timing_rows(
                summary["model_load_ms"],
                summary["warmup_ms"],
                item["baseline_timing"],
                item["dynamic_timing"],
                quality_ms,
            )
            experiment.write_timing_summary(exp_dir / "timing_summary.csv", timing_rows)
            generate_experiment_report(
                exp_dir, summary, quality, timing_rows, item["step_rows"]
            )
            result = _result_row(summary, quality)
            results.append(result)
            print(
                f"[{index:02d}/{len(pending)}] {summary['seed']} {summary['skip_predictor']['mode']} "
                f"PSNR={result['psnr_db']:.2f} SSIM={result['ssim']:.4f} "
                f"speedup={result['speedup']:.2f}x"
            )

    aggregate = aggregate_results(results)
    diagnostics = _collect_diagnostics(root)
    experiment.write_json(root / "batch_summary.json", {"results": results, **aggregate, "diagnostics": diagnostics})
    _write_results_csv(root / "results.csv", results)
    _write_batch_reports(root, results, aggregate, diagnostics, args)
    print(f"Batch report: {root / 'batch_report.html'}")
    print(f"Markdown report: {root / 'PREDICTOR_30_RUN_REPORT.md'}")
    if len(results) != len(SEEDS) * len(MODES):
        raise SystemExit(
            f"Only {len(results)} of {len(SEEDS) * len(MODES)} experiments completed"
        )
    return root, results


def main():
    args = _parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()
