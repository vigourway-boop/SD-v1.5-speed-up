"""Generate a self-contained bilingual HTML report for one experiment."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path


def _safe(value):
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_for_html(value):
    return json.dumps(_safe(value), ensure_ascii=False).replace("<", "\\u003c")


def _format_number(value, digits=3, suffix=""):
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not math.isfinite(numeric):
        return "N/A"
    return f"{numeric:.{digits}f}{suffix}"


def _metric(label, value):
    return (
        '<div class="metric"><div class="metric-label">'
        + html.escape(label)
        + '</div><div class="metric-value">'
        + html.escape(value)
        + "</div></div>"
    )


def _table_row(label, value):
    return (
        "<tr><th>"
        + html.escape(label)
        + "</th><td>"
        + html.escape(str(value))
        + "</td></tr>"
    )


def generate_experiment_report(
    experiment_dir,
    summary,
    quality,
    timing_rows,
    step_rows,
):
    experiment_dir = Path(experiment_dir)
    report_path = experiment_dir / "report.html"
    speedup = _format_number(summary.get("speedup"), 2, "x")
    baseline_seconds = _format_number(summary.get("baseline_seconds"), 2, " s")
    dynamic_seconds = _format_number(summary.get("dynamic_seconds"), 2, " s")
    predictor = summary.get("skip_predictor") or {}
    recovery = summary.get("connection_recovery") or {}

    metrics = "".join(
        [
            _metric("加速比 / Speedup", speedup),
            _metric("Baseline 耗时 / Time", baseline_seconds),
            _metric("Dynamic 耗时 / Time", dynamic_seconds),
            _metric(
                "UNet 执行 / Executed",
                str(summary.get("executed_unet_steps", "N/A")),
            ),
            _metric("跳步 / Skipped", str(summary.get("skipped_steps", "N/A"))),
            _metric("随机种子 / Seed", str(summary.get("seed", "N/A"))),
        ]
    )

    image_blocks = []
    for filename, label in (
        ("baseline.png", "原始生成 / Baseline"),
        ("pynq_dynamic.png", "PYNQ 动态生成 / Dynamic"),
    ):
        if (experiment_dir / filename).exists():
            image_blocks.append(
                '<figure><img src="'
                + filename
                + '" alt="'
                + html.escape(label)
                + '"><figcaption>'
                + html.escape(label)
                + "</figcaption></figure>"
            )
    images_html = "".join(image_blocks) or "<p>本次实验没有保存对比图片。</p>"

    quality_rows = []
    if quality:
        quality_fields = (
            ("PSNR（动态对比原始） / Dynamic vs baseline", "psnr_dynamic_vs_baseline_db", " dB"),
            ("SSIM（动态对比原始） / Dynamic vs baseline", "ssim_dynamic_vs_baseline", ""),
            ("LPIPS（越低越接近） / Lower is closer", "lpips_dynamic_vs_baseline", ""),
            ("Baseline CLIP 分数 / Score", "clip_score_baseline", ""),
            ("Dynamic CLIP 分数 / Score", "clip_score_dynamic", ""),
        )
        for label, key, suffix in quality_fields:
            quality_rows.append(_table_row(label, _format_number(quality.get(key), 4, suffix)))
    quality_html = (
        '<table class="detail-table"><tbody>' + "".join(quality_rows) + "</tbody></table>"
        if quality_rows
        else "<p>本次实验未运行完整质量评估。</p>"
    )

    timing_html = "".join(
        "<tr><td>"
        + html.escape(str(row["阶段 / Stage"]))
        + "</td><td class=\"number\">"
        + _format_number(row["耗时（ms） / Time (ms)"], 3)
        + "</td><td>"
        + html.escape(str(row["计入生成总耗时 / Included in generation total"]))
        + "</td><td>"
        + html.escape(str(row["说明 / Notes"]))
        + "</td></tr>"
        for row in timing_rows
    )

    configuration_rows = "".join(
        [
            _table_row("模型 / Model", "Stable Diffusion v1.5"),
            _table_row("设备 / Device", summary.get("device", "N/A")),
            _table_row(
                "PYNQ 服务 / Server",
                f"{summary.get('pynq_host', 'N/A')}:{summary.get('pynq_port', 'N/A')}",
            ),
            _table_row("基础步数 / Base steps", summary.get("base_steps", "N/A")),
            _table_row("预热步数 / Warm-up steps", summary.get("warmup_steps", "N/A")),
            _table_row("阈值模式 / Threshold mode", summary.get("threshold_mode", "N/A")),
            _table_row("距离阈值 / Distance threshold", summary.get("distance_threshold", "N/A")),
            _table_row("跳步预测器 / Skip predictor", predictor.get("mode", "N/A")),
            _table_row("预测阻尼 / Predictor damping", predictor.get("damping", "N/A")),
            _table_row("最大外推系数 / Max factor", predictor.get("max_factor", "N/A")),
            _table_row(
                "断线恢复次数 / Recovery restarts",
                recovery.get("generation_retries_used", 0),
            ),
        ]
    )

    raw_files = (
        ("summary.json", "summary.json"),
        ("timing_summary.csv", "timing_summary.csv"),
        ("step_metrics.csv", "step_metrics.csv"),
        ("quality_metrics.json", "quality_metrics.json"),
    )
    file_links = "".join(
        f'<a href="{filename}">{html.escape(label)}</a>'
        for filename, label in raw_files
        if (experiment_dir / filename).exists()
    )

    prompt = html.escape(str(summary.get("prompt", "")))
    negative_prompt = html.escape(str(summary.get("negative_prompt", "")))
    timestamp = html.escape(str(summary.get("timestamp", "")))
    step_json = _json_for_html(step_rows)

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SD 1.5 + PYNQ-Z2 实验报告 / Experiment Report</title>
<style>
:root {{ color-scheme: light; --ink:#17212b; --muted:#5d6874; --line:#d8dee4; --paper:#fff; --soft:#f4f6f8; --green:#147d64; --red:#b4443e; --blue:#2c64a5; --gold:#9b6a16; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--soft); color:var(--ink); font-family:"Segoe UI","Microsoft YaHei",sans-serif; line-height:1.5; letter-spacing:0; }}
header {{ background:#17212b; color:#fff; padding:30px max(24px,calc((100% - 1180px)/2)); border-bottom:5px solid var(--green); }}
h1 {{ margin:0 0 6px; font-size:30px; letter-spacing:0; }}
header p {{ margin:0; color:#cbd3da; }}
main {{ width:min(1180px,calc(100% - 32px)); margin:24px auto 48px; }}
section {{ background:var(--paper); border:1px solid var(--line); margin:16px 0; padding:22px; }}
h2 {{ margin:0 0 16px; font-size:20px; letter-spacing:0; }}
.metrics {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); border:1px solid var(--line); background:var(--paper); }}
.metric {{ min-width:0; padding:16px; border-right:1px solid var(--line); }}
.metric:last-child {{ border-right:0; }}
.metric-label {{ color:var(--muted); font-size:12px; min-height:38px; }}
.metric-value {{ font-size:24px; font-weight:700; overflow-wrap:anywhere; }}
.images {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
figure {{ margin:0; }}
figure img {{ display:block; width:100%; aspect-ratio:1; object-fit:contain; background:#eef1f3; border:1px solid var(--line); }}
figcaption {{ padding-top:8px; font-weight:600; }}
.charts {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
.chart {{ border:1px solid var(--line); padding:14px; min-width:0; }}
.chart h3 {{ margin:0 0 8px; font-size:15px; }}
canvas {{ width:100%; height:220px; display:block; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ border-bottom:1px solid var(--line); padding:9px 10px; text-align:left; vertical-align:top; }}
thead th {{ background:#eef1f3; font-weight:650; }}
.detail-table th {{ width:46%; color:var(--muted); font-weight:600; }}
.number {{ text-align:right; font-variant-numeric:tabular-nums; }}
.scroll {{ overflow-x:auto; }}
.prompt {{ margin:10px 0; padding:12px; background:#f7f8f9; border-left:4px solid var(--gold); overflow-wrap:anywhere; }}
.files {{ display:flex; flex-wrap:wrap; gap:10px; }}
.files a {{ color:var(--blue); text-decoration:none; border-bottom:1px solid currentColor; }}
.legend {{ color:var(--muted); font-size:12px; margin-top:6px; }}
@media (max-width:900px) {{ .metrics {{ grid-template-columns:repeat(3,1fr); }} .metric:nth-child(3) {{ border-right:0; }} .charts {{ grid-template-columns:1fr; }} }}
@media (max-width:640px) {{ h1 {{ font-size:24px; }} main {{ width:min(100% - 20px,1180px); }} section {{ padding:16px; }} .metrics {{ grid-template-columns:repeat(2,1fr); }} .metric:nth-child(3) {{ border-right:1px solid var(--line); }} .metric:nth-child(even) {{ border-right:0; }} .images {{ grid-template-columns:1fr; }} }}
@media print {{ body {{ background:#fff; }} main {{ width:100%; margin:0; }} section {{ break-inside:avoid; }} }}
</style>
</head>
<body>
<header><h1>SD 1.5 + PYNQ-Z2 实验报告</h1><p>Experiment Report · {timestamp}</p></header>
<main>
<div class="metrics">{metrics}</div>
<section><h2>生成结果 / Generated Images</h2><div class="images">{images_html}</div></section>
<section><h2>逐步行为 / Step Behavior</h2><div class="charts">
<div class="chart"><h3>UNet 与跳步 / UNet vs Skip</h3><canvas id="decisionChart"></canvas><div class="legend">绿色 Green = SKIP，红色 Red = UNET</div></div>
<div class="chart"><h3>余弦相似度与阈值 / Similarity vs Threshold</h3><canvas id="similarityChart"></canvas><div class="legend">蓝色 Blue = similarity，金色 Gold = threshold</div></div>
</div></section>
<section><h2>质量指标 / Quality Metrics</h2>{quality_html}</section>
<section><h2>耗时汇总 / Timing Breakdown</h2><div class="scroll"><table><thead><tr><th>阶段 / Stage</th><th>耗时（ms） / Time (ms)</th><th>计入总耗时 / Included</th><th>说明 / Notes</th></tr></thead><tbody>{timing_html}</tbody></table></div></section>
<section><h2>实验配置 / Configuration</h2><table class="detail-table"><tbody>{configuration_rows}</tbody></table>
<div class="prompt"><strong>正向提示词 / Prompt</strong><br>{prompt}</div>
<div class="prompt"><strong>负向提示词 / Negative prompt</strong><br>{negative_prompt}</div></section>
<section><h2>原始文件 / Raw Files</h2><div class="files">{file_links}</div></section>
</main>
<script type="application/json" id="stepData">{step_json}</script>
<script>
const steps=JSON.parse(document.getElementById('stepData').textContent);
function setup(canvas){{const dpr=window.devicePixelRatio||1;const rect=canvas.getBoundingClientRect();canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.round(220*dpr);const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);return [ctx,rect.width,220];}}
function axes(ctx,w,h){{ctx.strokeStyle='#d8dee4';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(34,10);ctx.lineTo(34,h-24);ctx.lineTo(w-8,h-24);ctx.stroke();ctx.fillStyle='#5d6874';ctx.font='11px Segoe UI';ctx.fillText('1',32,h-7);ctx.fillText(String(steps.length),Math.max(34,w-28),h-7);}}
function drawDecisions(){{const [ctx,w,h]=setup(document.getElementById('decisionChart'));ctx.clearRect(0,0,w,h);axes(ctx,w,h);const usable=Math.max(1,w-44);const bar=usable/Math.max(1,steps.length);steps.forEach((s,i)=>{{ctx.fillStyle=s.action==='SKIP'?'#147d64':'#b4443e';ctx.fillRect(35+i*bar,18,Math.max(1,bar),h-43);}});}}
function drawSimilarity(){{const [ctx,w,h]=setup(document.getElementById('similarityChart'));ctx.clearRect(0,0,w,h);axes(ctx,w,h);const values=steps.flatMap(s=>[s.similarity,s.threshold]).filter(v=>Number.isFinite(v));if(!values.length)return;let min=Math.min(...values),max=Math.max(...values);if(max-min<1e-8){{min-=1e-5;max+=1e-5;}}const x=i=>35+i*Math.max(1,w-44)/Math.max(1,steps.length-1);const y=v=>10+(max-v)*(h-35)/(max-min);function line(key,color){{ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();let active=false;steps.forEach((s,i)=>{{const v=s[key];if(!Number.isFinite(v)){{active=false;return;}}if(!active){{ctx.moveTo(x(i),y(v));active=true;}}else ctx.lineTo(x(i),y(v));}});ctx.stroke();}}line('similarity','#2c64a5');line('threshold','#9b6a16');ctx.fillStyle='#5d6874';ctx.font='11px Segoe UI';ctx.fillText(max.toFixed(6),2,18);ctx.fillText(min.toFixed(6),2,h-28);}}
function drawAll(){{drawDecisions();drawSimilarity();}}window.addEventListener('resize',drawAll);drawAll();
</script>
</body></html>
"""
    report_path.write_text(document, encoding="utf-8")
    return report_path
