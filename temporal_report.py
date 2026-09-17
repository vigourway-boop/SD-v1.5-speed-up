"""Render bilingual tables and figures from a completed temporal evaluation."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cluster_interval(rows, metric):
    groups = {}
    for row in rows:
        groups.setdefault(row["prompt_id"], []).append(row[metric])
    means = np.array([statistics.mean(values) for values in groups.values()])
    rng = np.random.default_rng(763)
    samples = rng.choice(means, size=(5000, len(means)), replace=True).mean(axis=1)
    return np.quantile(samples, [.025, .975]).tolist()


def build_report(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load(root / "results.json")
    aggregates = load(root / "aggregate.json")
    diagnostics = load(root / "diagnostics.json")
    selection = load(root / "selected_profile.json")
    manifest = load(root / "manifest.json")
    selected = [r for r in rows if r["split"] == "validation" and r["variant"] == "selected"]
    current = [r for r in rows if r["split"] == "validation" and r["variant"] == "current"]
    original50 = [r for r in rows if r["split"] == "validation" and r["variant"] == "original50"]
    fifty_by_case = {(r["prompt_id"], r["seed"]): r for r in original50}
    pc = [r for r in rows if r["split"] == "validation" and r["variant"] == "selected_pc"]
    interval = cluster_interval(selected, "speedup")
    speedup = statistics.mean(r["speedup"] for r in selected)
    ratio_totals = sum(r["baseline_seconds"] for r in selected) / sum(r["seconds"] for r in selected)
    match = {(r["prompt_id"], r["seed"]): r for r in selected}
    pc_pairs = [(r, match[(r["prompt_id"], r["seed"])]) for r in pc]
    baseline_peaks = []
    hash_matches, step_matches = [], []
    for row in selected:
        baseline = load((root / row["path"]).parent / "baseline" / "run.json")
        baseline_peaks.append(baseline["timing"]["cuda_peak_allocated_bytes"] / 2**20)
    for software, hardware in pc_pairs:
        pc_dir, hw_dir = root / software["path"], root / hardware["path"]
        hash_matches.append(load(pc_dir / "run.json")["image_sha256"] == load(hw_dir / "run.json")["image_sha256"])
        def actions(path):
            with path.open(encoding="utf-8") as stream:
                return [r["should_skip"] for r in csv.DictReader(stream)]
        step_matches.append(actions(pc_dir / "step_metrics.csv") == actions(hw_dir / "step_metrics.csv"))
    evidence = {"validation_mean_speedup": speedup, "ratio_of_total_times": ratio_totals,
                "prompt_cluster_bootstrap_95_interval": interval,
                "two_x_fraction": sum(r["speedup"] >= 2 for r in selected) / len(selected),
                "pc_pynq_pairs": len(pc_pairs), "pc_pynq_identical_images": sum(hash_matches),
                "pc_pynq_identical_decision_sequences": sum(step_matches),
                "pynq_minus_pc_seconds_mean": statistics.mean(h["seconds"] - p["seconds"] for p, h in pc_pairs) if pc_pairs else None,
                "baseline_peak_allocated_mib_mean": statistics.mean(baseline_peaks)}
    evidence["held_out_quality_diagnostic_limits_met"] = (
        statistics.mean(r["lpips"] for r in selected) <= .10
        and max(r["lpips"] for r in selected) <= .25
        and statistics.mean(r["clip_cosine_delta"] for r in selected) >= -.01
    )
    evidence["held_out_lpips_over_025_count"] = sum(r["lpips"] > .25 for r in selected)
    evidence["selected_lpips_better_than_original50_count"] = sum(
        r["lpips"] < fifty_by_case[(r["prompt_id"], r["seed"])]["lpips"] for r in selected)
    evidence["mean_speedup_vs_original50"] = statistics.mean(
        fifty_by_case[(r["prompt_id"], r["seed"])]["seconds"] / r["seconds"] for r in selected)
    verification_path = root / "verification" / "summary.json"
    verification = load(verification_path) if verification_path.exists() else None
    evidence["counterbalanced_verification"] = verification
    (root / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    styles = [(current, "Current", "#777777"), (selected, "Selected", "#12755f"),
              (original50, "Original 50 steps", "#b4473b")]
    for items, label, color in styles:
        axes[0].scatter([r["lpips"] for r in items], [r["speedup"] for r in items],
                        label=label, color=color, alpha=.75, s=25)
    axes[0].axhline(2, color="black", linestyle="--", linewidth=.8)
    axes[0].set(xlabel="LPIPS vs original 100 (lower is closer)", ylabel="Speedup vs original 100")
    axes[0].legend(fontsize=8)
    sweep = [r for r in rows if r["split"] == "step_sweep"]
    points = sorted(set(r["steps"] for r in sweep))
    for prompt_id in sorted(set(r["prompt_id"] for r in sweep)):
        items = sorted([r for r in sweep if r["prompt_id"] == prompt_id], key=lambda r: r["steps"])
        axes[1].plot([r["steps"] for r in items], [r["speedup"] for r in items], "o-", label=prompt_id)
    axes[1].axhline(2, color="black", linestyle="--", linewidth=.8)
    axes[1].set(xlabel="Total scheduler steps", ylabel="Speedup vs same-step original", xticks=points)
    axes[1].legend(fontsize=8)
    fig.savefig(root / "speed_quality.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
    for path in sorted((root / "diagnostics").glob("*/feature_statistics.csv")):
        with path.open(encoding="utf-8-sig") as stream:
            data = list(csv.reader(stream))[1:]
        data = np.asarray(data, dtype=float)
        for ax, column, ylabel in zip(axes, (1, 2, 5), ("Raw adjacent cosine", "Raw normalized distance", "Int8 delta zero fraction")):
            ax.plot(data[:, 0], data[:, column], label=path.parent.name)
            ax.set(xlabel="Step", ylabel=ylabel)
    axes[0].legend(fontsize=8)
    fig.savefig(root / "feature_curves.png", dpi=180)
    plt.close(fig)

    # Include the best and worst fidelity cases, then other prompts, without cherry-picking.
    ordered = sorted(selected, key=lambda r: r["lpips"])
    gallery_rows = [ordered[-1], ordered[0]]
    seen = {r["prompt_id"] for r in gallery_rows}
    for row in ordered:
        if row["prompt_id"] not in seen:
            gallery_rows.append(row)
            seen.add(row["prompt_id"])
        if len(gallery_rows) == 6:
            break
    sheet = Image.new("RGB", (4 * 256, 6 * 286 + 32), "white")
    draw = ImageDraw.Draw(sheet)
    for column, title in enumerate(("Original 100", "Current dynamic", "Selected dynamic", "Original 50")):
        draw.text((column * 256 + 10, 8), title, fill="black")
    for row_index, row in enumerate(gallery_rows):
        group = (root / row["path"]).parent
        for column, variant in enumerate(("baseline", "current", "selected", "original50")):
            with Image.open(group / variant / "image.png") as image:
                sheet.paste(image.resize((256, 256)), (column * 256, row_index * 286 + 32))
        draw.text((8, row_index * 286 + 291),
                  f"{row['prompt_id']} seed={row['seed']} | selected LPIPS={row['lpips']:.3f}, {row['speedup']:.2f}x", fill="black")
    sheet.save(root / "comparison_gallery.jpg", quality=92)

    lines = ["# 时间步相似性加速实验报告 / Temporal Similarity Evaluation", "",
             f"日期：{manifest['created']}。GPU：{manifest['gpu']}。控制后端：{manifest['backend']}。", "",
             "## 实验范围与计时", "",
             "- 调参：3种提示词 × 2个种子 × 3种动态参数；另有3次100步原始轨迹采集。",
             "- 独立验证：8种新提示词 × 2个新种子；比较原始100步、现有动态、选定动态和原始50步。",
             "- 其中3种提示词 × 2个种子另跑电脑控制器；步数扫描为3种提示词 × 20/30/50/75步。",
             "- 每组方案执行顺序按固定种子打乱；所有方案统一预热、提示词、种子、512×512分辨率和DPM-Solver采样器。",
             "- 生成计时包含文本编码、扩散、VAE、控制器初始化、特征压缩及板端通信；不包含模型加载、预热、保存PNG、画质评估。",
             "- 轨迹采集运行不计入速度结果；正式运行不自动降级、不重试，异常直接记为失败。",
             "- CLIP变化采用原始余弦尺度；旧报告的CLIP Score乘过100，不能混用。", "",
             "## 结果 / Results", "",
             "| 数据集 / Dataset | 时间步 / Steps | 方案 / Variant | 样本 / N | 平均加速 / Mean speedup | 平均UNet / Mean calls | 平均LPIPS / Mean LPIPS | 最差LPIPS / Max LPIPS | 平均CLIP变化 / Mean CLIP delta |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for key, g in sorted(aggregates.items()):
        split, steps, variant = key.split("/")
        lines.append(f"| {split} | {steps} | {variant} | {g['count']} | {g['speedup']['mean']:.3f}x | {g['executed']['mean']:.1f} | {g['lpips']['mean']:.4f} | {g['lpips']['max']:.4f} | {g['clip_cosine_delta']['mean']:+.4f} |")
    parity = diagnostics["parity"]
    lines += ["", f"选定参数：`{selection['name']}`，仅使用调参集选择。筛选条件是工程诊断规则（平均LPIPS≤0.10、最差≤0.25、平均CLIP余弦下降≤0.01），不是课题规定的画质验收标准。调参集是否满足：{selection['quality_limits_met']}。", "",
              f"独立验证平均加速 **{speedup:.3f}x**；总基准时间/总动态时间 **{ratio_totals:.3f}x**；达到2x的样本 **{sum(r['speedup'] >= 2 for r in selected)}/{len(selected)}**。",
              f"按提示词分组重采样的平均加速95%区间：**[{interval[0]:.3f}, {interval[1]:.3f}]x**。该区间反映本批提示词差异，不覆盖不同机器和运行日期。", "",
              f"独立验证是否满足调参时的画质诊断条件：**{evidence['held_out_quality_diagnostic_limits_met']}**；LPIPS超过0.25的样本：**{evidence['held_out_lpips_over_025_count']}/{len(selected)}**。通过调参筛选不等于通过独立验证，不据此替换原有默认参数。",
              f"相对原始50步，新动态平均快 **{evidence['mean_speedup_vs_original50']:.3f}x**；与原始100步的LPIPS比原始50步更好的样本为 **{evidence['selected_lpips_better_than_original50_count']}/{len(selected)}**。", "",
              "## FPGA实际收益 / FPGA Contribution", "",
              f"逐步回放核对：{parity['steps']}步，{parity['mismatches']}处不一致。PC平均判断含压缩 {parity['pc_ms']['mean']:.3f}ms；PYNQ平均判断含压缩与往返 {parity['pynq_ms']['mean']:.3f}ms。" if parity['performed'] else "本轮没有板端核对。",
              f"完整生成PC/PYNQ配对：{len(pc_pairs)}组；决策序列相同{sum(step_matches)}组；图片像素哈希相同{sum(hash_matches)}组。",
              f"PYNQ减PC的平均生成耗时：{evidence['pynq_minus_pc_seconds_mean']:.4f}秒（正数表示PYNQ较慢）。" if pc_pairs else "无PC/PYNQ生成时间对照。",
              "FPGA确实负责整数相似度/距离判断和跳步状态；整体加速主要来自少执行GPU UNet。ARM报告的kernel时间包含MMIO和轮询，不能称为纯FPGA运算延迟。", "",
              "## 存储与差异 / Storage and Differences", "",
              f"原始100步平均PyTorch分配显存峰值：{statistics.mean(baseline_peaks):.1f} MiB；选定动态：{statistics.mean(r['peak_allocated_mib'] for r in selected):.1f} MiB。此指标不是整机GPU显存占用。",
              f"动态预测缓存：{max(r['predictor_cache_bytes'] for r in selected)}字节；量化参考向量4096字节。未做模型权重量化，不能宣称模型存储缩小。",
              "每步单份FP16 latent为32768字节，去重、2×2池化和INT8后为4096字节，特征载荷减少8倍；相对重复CFG输入为16倍。网络字节统计只含特征载荷，不含TCP/IP头。",
              f"选定动态中余弦通过而距离拒绝的次数合计：{sum(r['distance_rejections'] for r in selected)}。",
              "INT8向量逐步独立缩放，幅值变化可能被消除，因此INT8距离不是原始latent的真实幅值距离；其差值零比例也不能直接当作UNet可跳过的稀疏计算比例。", "",
              "## 图表 / Figures", "", "![速度与画质 / Speed and quality](speed_quality.png)", "",
              "![特征曲线 / Feature curves](feature_curves.png)", "",
              "![包含最差与最佳一致性样本 / Includes worst and best fidelity cases](comparison_gallery.jpg)", "",
              "## 限制 / Limitations", "",
              "1. 画质指标衡量与原始图的接近程度，不能证明图像本身正确；应结合图库检查主体、结构和提示词符合程度。",
              "2. 当前验证只有8种独立提示词和每词2个种子；尚不代表所有场景、采样器、分辨率和GPU。",
              "3. 原始50步对照必须共同报告，不能仅以原始100步作为唯一参照。不同步数扫描采用15%预热比例，不能承诺每种步数都达到2倍。",
              "4. 没有实测整机或板卡能量，不宣称节能；FPGA纯运算周期仍需硬件计数器/RTL验证。",
              "5. 动态阈值不等于模型训练；本轮离线调参后固定配置，运行时EMA适应变化。距离门限由调参轨迹候选跳步距离的90%分位数确定。",
              "6. 未实现UNet卷积差分计算、MAC阵列或完整CacheQuant/Ditto复现；这些不是本简介强制指定的路线。", "",
              "## 复现 / Reproduction", "", "```powershell", "conda activate sd_accel",
              "python temporal_evaluation.py --backend pynq",
              "python temporal_report.py experiments/<本次输出目录>", "```", "",
              "`manifest.json`保存环境、提示词、种子、初始候选和源文件哈希；`selected_profile.json`保存选定参数；每组目录保存图像、逐步记录、耗时与画质数据。"]
    if verification:
        lines += ["", "## 补充顺序验证 / Counterbalanced Verification", "",
                  "主批次随机顺序只由种子和步数决定，恰好把验证集基准放在最后。补充测试对3种提示词各交换一次先后顺序，且不重新调参；后续实验脚本将提示词也加入顺序种子。",
                  f"6组补充平均加速 **{verification['mean_speedup']:.3f}x**，范围 **{verification['min_speedup']:.3f}–{verification['max_speedup']:.3f}x**。原始先跑平均 **{verification['baseline_first_mean']:.3f}x**，动态先跑平均 **{verification['dynamic_first_mean']:.3f}x**。",
                  f"重复运行图像哈希一致：{verification['identical_repeat_images']}。关闭跳步的20步手工扩散与原始pipeline图片像素一致：{verification['no_skip']['identical_pixels']}。",
                  "主批次测量源码另存于 `source_snapshot.zip`；补充测试源码哈希见 `verification/summary.json`。补充数据单独报告，不混入独立验证集的平均数。"]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "REPORT.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    build_report(parser.parse_args().root)
