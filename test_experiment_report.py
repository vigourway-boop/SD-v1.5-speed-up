import shutil
import unittest
from pathlib import Path

from experiment_report import generate_experiment_report


class ExperimentReportTest(unittest.TestCase):
    def test_generates_bilingual_offline_report(self):
        directory = Path("test_experiment_report_output")
        directory.mkdir(exist_ok=True)
        try:
            path = generate_experiment_report(
                directory,
                {
                    "timestamp": "2026-08-15T20:00:00",
                    "seed": 123,
                    "prompt": "horse <test>",
                    "negative_prompt": "blurry",
                    "device": "cuda",
                    "pynq_host": "192.168.2.99",
                    "pynq_port": 9000,
                    "base_steps": 100,
                    "warmup_steps": 15,
                    "executed_unet_steps": 51,
                    "skipped_steps": 49,
                    "baseline_seconds": 10.0,
                    "dynamic_seconds": 5.5,
                    "speedup": 10.0 / 5.5,
                    "skip_predictor": {"mode": "linear", "damping": 0.5, "max_factor": 1.5},
                    "connection_recovery": {"generation_retries_used": 0},
                },
                {"ssim_dynamic_vs_baseline": 0.95},
                [
                    {
                        "阶段 / Stage": "UNet 推理 / UNet inference",
                        "耗时（ms） / Time (ms)": 5000.0,
                        "计入生成总耗时 / Included in generation total": "是 / Yes",
                        "说明 / Notes": "test",
                    }
                ],
                [
                    {"step": 1, "action": "UNET", "similarity": float("nan"), "threshold": 0.9999},
                    {"step": 2, "action": "SKIP", "similarity": 0.99995, "threshold": 0.9999},
                ],
            )
            content = path.read_text(encoding="utf-8")
        finally:
            shutil.rmtree(directory, ignore_errors=True)

        self.assertIn("实验报告 / Experiment Report", content)
        self.assertIn("耗时汇总 / Timing Breakdown", content)
        self.assertIn('id="decisionChart"', content)
        self.assertIn("horse &lt;test&gt;", content)
        self.assertNotIn("NaN", content)


if __name__ == "__main__":
    unittest.main()

