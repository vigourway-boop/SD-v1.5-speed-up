"""Evaluate conservative skip profiles on known failures and fresh prompts."""

import argparse
import hashlib
import json
import statistics
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import combined_speed_test as sd
from controller_profile import load_controller_profile
from pynq_cosine_client import PynqCosineClient
from quality_metrics import QualityEvaluator
from temporal_evaluation import Evaluation, VALIDATION, parity_check, summarize, write_csv


CALIBRATION = [item for item in VALIDATION if item[0] in {"car", "robot"}]
FRESH_PROMPTS = [
    ("dog", "a photograph of a golden retriever sitting on a park path, full body, daylight"),
    ("athlete", "a photograph of a runner wearing a blue jacket standing beside a running track, full body"),
    ("market", "a photograph of a busy outdoor vegetable market, shoppers walking between stalls, daylight"),
    ("bridge", "a photograph of a stone bridge crossing a river in the mountains, clear daylight"),
    ("motorcycle", "a photograph of a black motorcycle parked outside a cafe, side view, daylight"),
    ("fruit", "a still life photograph of apples and grapes in a woven basket on a kitchen table"),
]
FRESH_SEEDS = [20260915, 20260916]


def select_candidate(rows, names):
    """Minimize worst calibration LPIPS among candidates with mean speed >=2x."""
    groups = summarize(rows)
    eligible = [name for name in names if groups[f"calibration/100/{name}"]["speedup"]["mean"] >= 2]
    pool = eligible or list(names)
    selected = min(pool, key=lambda name: (
        groups[f"calibration/100/{name}"]["lpips"]["max"],
        groups[f"calibration/100/{name}"]["lpips"]["mean"]))
    return selected, bool(eligible)


def report(root):
    rows = json.loads((root / "results.json").read_text(encoding="utf-8"))
    selected = json.loads((root / "selected_profile.json").read_text(encoding="utf-8"))
    groups = summarize(rows)
    fresh = [r for r in rows if r["split"] == "fresh_validation" and r["variant"] == "selected"]
    fast = {(r["prompt_id"], r["seed"]): r for r in rows
            if r["split"] == "fresh_validation" and r["variant"] == "fast"}
    wins = sum(r["lpips"] < fast[r["prompt_id"], r["seed"]]["lpips"] for r in fresh)
    summary = {"selected": selected["name"], "fresh_cases": len(fresh),
               "lpips_improved_vs_fast": wins,
               "mean_speedup": statistics.mean(r["speedup"] for r in fresh),
               "two_x_count": sum(r["speedup"] >= 2 for r in fresh),
               "mean_lpips": statistics.mean(r["lpips"] for r in fresh),
               "worst_lpips": max(r["lpips"] for r in fresh),
               "lpips_over_025_count": sum(r["lpips"] > .25 for r in fresh),
               "mean_clip_cosine_delta": statistics.mean(r["clip_cosine_delta"] for r in fresh),
               "mean_executed": statistics.mean(r["executed"] for r in fresh),
               "original50_mean_speedup": groups["fresh_validation/100/original50"]["speedup"]["mean"]}
    sd.write_json(root / "evidence.json", summary)
    lines = ["# 画质稳定性补充实验 / Robustness Evaluation", "",
             "本轮将上一轮已见过的汽车和机器人样本明确作为调参集；独立验证改为6种新提示词×2个新种子。不能把已用于调参的失败样本继续称为独立验证。", "",
             "选参规则在生成前写入manifest：候选平均速度达到2倍时，选择调参集最差LPIPS最小者；若都达不到则保留最差LPIPS最小者并记录未满足速度条件。未根据验证集重新调整参数。", "",
             "使用同一SD1.5、DPM-Solver、FP16、512×512、100步及种子；每组执行顺序由提示词/种子散列打乱。严格PYNQ模式不自动后备、不重试；计时包含压缩、初始化、网络往返、扩散和VAE，不含加载/预热/PNG保存/画质评价。", "",
             "| 数据集 / Split | 方案 / Variant | 样本 / N | 平均加速 / Mean speedup | 平均UNet / Calls | 平均LPIPS / Mean | 最差LPIPS / Max | CLIP余弦变化 / Delta |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key, g in sorted(groups.items()):
        split, _, variant = key.split("/")
        lines.append(f"| {split} | {variant} | {g['count']} | {g['speedup']['mean']:.3f}x | {g['executed']['mean']:.1f} | {g['lpips']['mean']:.4f} | {g['lpips']['max']:.4f} | {g['clip_cosine_delta']['mean']:+.4f} |")
    lines += ["", f"选定参数：`{selected['name']}`。独立验证平均 **{summary['mean_speedup']:.3f}x**，达到2倍 **{summary['two_x_count']}/{len(fresh)}**。",
              f"LPIPS相对fast改善 **{wins}/{len(fresh)}**，平均 **{summary['mean_lpips']:.4f}**，最差 **{summary['worst_lpips']:.4f}**。超过0.25的样本 **{summary['lpips_over_025_count']}/{len(fresh)}**。0.25仅为诊断线，不是课题验收标准。", "",
              "以下图库包含选定方案最差LPIPS的案例。画质指标只衡量与原始100步的接近程度，不能保证人体、动物结构正确；原始50步也必须共同比较。", "",
              "![按选定方案LPIPS由差到好 / Worst cases first](worst_cases.jpg)", "",
              "## 使用边界 / Limits", "",
              "本轮只针对100步、512×512、当前采样器和显卡。不将结果外推到20步、其他模型或所有提示词。更保守的参数不能保证每个样本都改善；若独立验证未通过，应作为实验配置而非新的默认设置。",
              "PC/PYNQ固定轨迹核对记录在parity.csv。硬件未重综合，因为这次只改变通过CSK4配置接口下发的控制参数。未测量板卡功耗或能量。", "",
              "源代码快照source_snapshot.zip、提示词和参数manifest.json、逐组results.csv以及每组图像/日志/step_metrics.csv/run.json/quality.json均保留。"]
    worst = sorted(fresh, key=lambda r: r["lpips"], reverse=True)[:4]
    sheet = Image.new("RGB", (1024, 32 + 286 * len(worst)), "white")
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(("Original 100", "Fast", "Selected conservative", "Original 50")):
        draw.text((256 * col + 8, 8), label, fill="black")
    for index, row in enumerate(worst):
        parent = (root / row["path"]).parent
        for col, name in enumerate(("baseline", "fast", "selected", "original50")):
            with Image.open(parent / name / "image.png") as image:
                sheet.paste(image.resize((256, 256)), (col * 256, 32 + index * 286))
        draw.text((8, 290 + index * 286),
                  f"{row['prompt_id']} seed={row['seed']} LPIPS={row['lpips']:.3f} speedup={row['speedup']:.2f}x", fill="black")
    sheet.save(root / "worst_cases.jpg", quality=92)
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(f"experiments/robustness_{datetime.now():%Y%m%d_%H%M%S}"))
    parser.add_argument("--host", default=sd.PYNQ_HOST)
    parser.add_argument("--port", type=int, default=sd.PYNQ_PORT)
    args = parser.parse_args()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=False)
    fast = load_controller_profile("profiles/temporal_fast_100.json")
    profiles = {"fast": fast,
                "guarded": {**fast, "warmup_ratio": .25, "max_consecutive_skips": 2},
                "balanced": {**fast, "warmup_ratio": .20, "max_consecutive_skips": 2,
                             "middle_threshold": .999, "late_margin": .0008,
                             "late_min": .9988, "late_max": .9992,
                             "distance_threshold": .0011062812454257125}}
    manifest = {"created": datetime.now().isoformat(), "profiles": profiles,
                "calibration": CALIBRATION, "calibration_seeds": [202601, 202602],
                "validation": FRESH_PROMPTS, "validation_seeds": FRESH_SEEDS,
                "selection": "Minimum worst calibration LPIPS among guarded/balanced with mean speedup >=2; if none, minimum worst LPIPS and flag speed failure",
                "backend": "pynq", "host": args.host, "port": args.port,
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('.').glob('*.py')}}
    sd.write_json(root / "manifest.json", manifest)
    with zipfile.ZipFile(root / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for source in manifest["source_sha256"]:
            archive.write(source)
    board = PynqCosineClient(args.host, args.port, timeout=5)
    torch.set_num_threads(4)
    try:
        board.connect()
        board.close()
        # These saved raw inputs test parameters, not quality or performance.
        trace = Path("experiments/temporal_evaluation_20260914_v2/diagnostics/horse/latent_trace.npz")
        with np.load(trace) as source:
            features = list(torch.from_numpy(source["features"]))
        checks = []
        for name, profile in profiles.items():
            checks.extend({"profile": name, **row} for row in parity_check(features, profile, board))
        write_csv(root / "parity.csv", checks)
        if any(row["mismatch"] for row in checks):
            raise RuntimeError("Controller parity failed")
        board.close()
        sd.pipe = sd.create_pipe()
        sd.pipe.set_progress_bar_config(disable=True)
        if sd.warm_up_pipeline() <= 0:
            raise RuntimeError("Warmup failed")
        with QualityEvaluator("cpu") as evaluator:
            run = Evaluation(root, board, evaluator)
            for prompt_id, prompt in CALIBRATION:
                for seed in (202601, 202602):
                    run.group("calibration", prompt_id, prompt, seed, 100,
                              [(name, p, "pynq", 100) for name, p in profiles.items()])
            selected, speed_met = select_candidate(run.results, ("guarded", "balanced"))
            sd.write_json(root / "selected_profile.json", {"name": selected, "profile": profiles[selected],
                                                          "calibration_mean_speed_condition_met": speed_met})
            print(f"Selected {selected}, calibration mean >=2x: {speed_met}", flush=True)
            for prompt_id, prompt in FRESH_PROMPTS:
                for seed in FRESH_SEEDS:
                    run.group("fresh_validation", prompt_id, prompt, seed, 100,
                              [("fast", fast, "pynq", 100), ("selected", profiles[selected], "pynq", 100)],
                              include_half=True)
        report(root)
        sd.write_json(root / "completion.json", {"complete": True, "rows": len(run.results)})
        print(f"Completed {root.resolve()}", flush=True)
    except Exception as exc:
        sd.write_json(root / "failure.json", {"error": repr(exc)})
        raise
    finally:
        board.close()
        sd.pipe = None


if __name__ == "__main__":
    main()
