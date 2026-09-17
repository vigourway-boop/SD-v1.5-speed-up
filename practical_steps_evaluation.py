"""Compare ordinary 20/30/50-step SD against dynamic 100 on equal prompts."""

import argparse
import hashlib
import json
import statistics
import zipfile
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageDraw

import combined_speed_test as sd
from controller_profile import load_controller_profile
from pynq_cosine_client import PynqCosineClient
from quality_metrics import QualityEvaluator
from robustness_evaluation import FRESH_PROMPTS, FRESH_SEEDS
from temporal_evaluation import Evaluation, write_csv


def report(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads((root / "results.json").read_text(encoding="utf-8"))
    groups = {}
    for row in rows:
        groups.setdefault(row["variant"], []).append(row)
    case_rows = {(r["prompt_id"], r["seed"], r["variant"]): r for r in rows}
    summary = []
    for name, values in groups.items():
        quality = [json.loads((root / r["path"] / "quality.json").read_text(encoding="utf-8")) for r in values]
        summary.append({"variant": name, "cases": len(values),
                        "mean_seconds": statistics.mean(r["seconds"] for r in values),
                        "mean_executed": statistics.mean(r["executed"] for r in values),
                        "mean_speedup_vs100": statistics.mean(r["speedup"] for r in values),
                        "mean_clip_cosine": statistics.mean(q["clip_score_dynamic"] / 100 for q in quality),
                        "mean_lpips_vs100": statistics.mean(r["lpips"] for r in values),
                        "worst_lpips_vs100": max(r["lpips"] for r in values)})
    paired = []
    for r in groups["fast100"]:
        for ordinary in (20, 30, 50):
            other = case_rows[r["prompt_id"], r["seed"], f"original{ordinary}"]
            paired.append({"prompt": r["prompt_id"], "seed": r["seed"], "ordinary_steps": ordinary,
                           "ordinary_seconds": other["seconds"], "dynamic100_seconds": r["seconds"],
                           "dynamic_speedup_vs_ordinary": other["seconds"] / r["seconds"]})
    sd.write_json(root / "practical_summary.json", {"variants": summary, "paired": paired})
    write_csv(root / "practical_summary.csv", summary,
              {"variant": "方案 / Variant", "cases": "样本 / Cases", "mean_seconds": "平均秒 / Mean seconds",
               "mean_executed": "平均UNet次数 / Mean UNet calls", "mean_speedup_vs100": "相对100步加速 / Speedup vs100",
               "mean_clip_cosine": "平均文本图像余弦 / Mean CLIP cosine",
               "mean_lpips_vs100": "与100步感知差异 / LPIPS vs100", "worst_lpips_vs100": "最大差异 / Max LPIPS vs100"})
    order = ["original20", "original30", "original50", "fast100", "conservative100"]
    by_variant = {r["variant"]: r for r in summary}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    colors = ["#5083a0", "#54735b", "#888888", "#bd4a40", "#977c26"]
    axes[0].bar(range(5), [by_variant[k]["mean_seconds"] for k in order], color=colors)
    axes[0].set(xticks=range(5), xticklabels=order, ylabel="Mean generation seconds")
    axes[0].tick_params(axis="x", labelrotation=25)
    for name, color in zip(order, colors):
        axes[1].scatter([r["seconds"] for r in groups[name]],
                        [r["lpips"] for r in groups[name]], label=name, color=color)
    axes[1].set(xlabel="Generation seconds", ylabel="LPIPS vs 100 steps (difference, not absolute quality)")
    axes[1].legend(fontsize=8)
    fig.savefig(root / "practical_tradeoff.png", dpi=160)
    plt.close(fig)
    cases = sorted({(r["prompt_id"], r["seed"]) for r in rows})
    columns = ["baseline"] + order
    sheet = Image.new("RGB", (256 * len(columns), 32 + 284 * len(cases)), "white")
    draw = ImageDraw.Draw(sheet)
    for column, label in enumerate(["Original 100 reference"] + order):
        draw.text((256 * column + 6, 8), label, fill="black")
    for index, (prompt, seed) in enumerate(cases):
        group = root / "practical" / f"{prompt}_{seed}_100"
        for column, variant in enumerate(columns):
            with Image.open(group / variant / "image.png") as image:
                sheet.paste(image.resize((256, 256)), (column * 256, 32 + index * 284))
        draw.text((8, 289 + index * 284), f"{prompt} seed={seed}", fill="black")
    sheet.save(root / "all_cases.jpg", quality=92)
    lines = ["# 常用步数与动态跳步对照 / Practical Step Baselines", "",
             "SD1.5常见生成范围约20～50步；当前DPM-Solver应优先考虑20～30步，具体画质由采样器、提示词、种子和引导强度决定。100步是本项目的历史算法基准，不是标准画质或日常必需设置。步数增加不保证画质持续提高。", "",
             "本实验用3种提示词（狗、市场、摩托车）×2个种子，同组比较原始20/30/50/100步、快速动态100步与保守动态100步。生成顺序按提示词和种子打乱。完整提示词、种子、参数、源码快照已存档。", "",
             "| 方案 / Variant | 平均秒 / Mean s | UNet次数 / Calls | 相对100步加速 / Speedup | CLIP余弦 / Cosine | 与100步LPIPS / Difference |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in order:
        r = by_variant[name]
        lines.append(f"| {name} | {r['mean_seconds']:.3f} | {r['mean_executed']:.1f} | {r['mean_speedup_vs100']:.3f}x | {r['mean_clip_cosine']:.4f} | {r['mean_lpips_vs100']:.4f} |")
    lines += ["", "## 实际速度收益", ""]
    for steps in (20, 30, 50):
        values = [r["dynamic_speedup_vs_ordinary"] for r in paired if r["ordinary_steps"] == steps]
        lines.append(f"- 快速动态100步相对原始{steps}步的平均加速比：**{statistics.mean(values):.3f}x**。大于1才是更快，小于1表示更慢。")
    lines += ["", "不能把‘相对原始100步达到2倍’写成‘相对常用SD1.5生成达到2倍’。应先按目标画质选普通20/30/50步基准，再比较动态方案；没有统一画质验收阈值时，本报告只给速度和画质取舍，不宣布某方案全面更好。", "",
              "LPIPS/PSNR/SSIM以原始100步为参考，只表示图像差异。更接近100步不代表视觉上更好；CLIP也只衡量文本图像对齐，不能判断手指、腿部、文字等结构是否正确。", "",
              "![耗时与差异 / Time and difference](practical_tradeoff.png)", "",
              "![全部样本 / All cases](all_cases.jpg)", "",
              "## 下一步技术重点", "",
              "如果目标是超过日常20～30步DPM-Solver，应该重新研究短采样轨迹下的缓存/预测，或UNet内部的局部计算复用，并与更少步数基准比较。仅把100步跳到约36～46次UNet，不足以说明优于20次完整UNet。不能把FID用于这6个样本并作可靠分布质量结论。", "",
              "计时包含压缩、控制器初始化、网络往返、UNet、scheduler及VAE，排除模型加载、预热、PNG保存和质量评价。严格PYNQ无自动后备，且不计入其他批次的速度均值。功耗未测。"]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robustness-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(f"experiments/practical_{datetime.now():%Y%m%d_%H%M%S}"))
    args = parser.parse_args()
    if not (args.robustness_root / "completion.json").exists():
        raise ValueError("Finish robustness evaluation before the practical comparison")
    root = args.output_root
    root.mkdir(parents=True, exist_ok=False)
    fast = load_controller_profile("profiles/temporal_fast_100.json")
    conservative = load_controller_profile(args.robustness_root / "selected_profile.json")
    prompts = [p for p in FRESH_PROMPTS if p[0] in {"dog", "market", "motorcycle"}]
    manifest = {"prompts": prompts, "seeds": FRESH_SEEDS, "fast": fast, "conservative": conservative,
                "backend": "pynq", "scheduler": "DPMSolverMultistepScheduler", "base_steps": 100,
                "steps": [20, 30, 50, 100], "tuning_performed": False,
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('.').glob('*.py')}}
    sd.write_json(root / "manifest.json", manifest)
    with zipfile.ZipFile(root / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for source in manifest["source_sha256"]:
            archive.write(source)
    board = PynqCosineClient(sd.PYNQ_HOST, sd.PYNQ_PORT, timeout=5)
    torch.set_num_threads(4)
    try:
        board.connect()
        board.close()
        sd.pipe = sd.create_pipe()
        sd.pipe.set_progress_bar_config(disable=True)
        if sd.warm_up_pipeline() <= 0:
            raise RuntimeError("Warmup failed")
        with QualityEvaluator("cpu") as evaluator:
            run = Evaluation(root, board, evaluator)
            for prompt_id, prompt in prompts:
                for seed in FRESH_SEEDS:
                    run.group("practical", prompt_id, prompt, seed, 100,
                              [("original20", None, None, 20), ("original30", None, None, 30),
                               ("fast100", fast, "pynq", 100), ("conservative100", conservative, "pynq", 100)],
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
