import csv
import unittest
from pathlib import Path

from combined_speed_test import (
    TIMING_FIELDS,
    build_timing_rows,
    write_timing_summary,
)


class TimingSummaryTest(unittest.TestCase):
    def test_bilingual_timing_csv(self):
        baseline = {"generation_ms": 1000.0, "image_save_ms": 10.0}
        dynamic = {
            "generation_ms": 600.0,
            "text_encoder_ms": 10.0,
            "latent_init_ms": 2.0,
            "diffusion_loop_ms": 500.0,
            "unet_ms": 400.0,
            "scheduler_ms": 20.0,
            "skip_prediction_ms": 3.0,
            "pynq_feature_prepare_ms": 10.0,
            "pynq_total_ms": 50.0,
            "network_round_trip_ms": 40.0,
            "fpga_kernel_ms": 5.0,
            "pynq_server_ms": 8.0,
            "diffusion_loop_other_ms": 30.0,
            "vae_ms": 80.0,
            "image_save_ms": 12.0,
        }
        rows = build_timing_rows(2000.0, 300.0, baseline, dynamic, 5000.0)

        self.assertEqual(len(rows), 20)
        self.assertTrue(all(" / " in field for field in TIMING_FIELDS))
        self.assertTrue(all(" / " in row["阶段 / Stage"] for row in rows))

        path = Path("test_timing_summary_output.csv")
        try:
            write_timing_summary(path, rows)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                loaded = list(reader)
                fieldnames = reader.fieldnames
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(fieldnames, TIMING_FIELDS)
        self.assertEqual(len(loaded), 20)
        self.assertEqual(loaded[0]["阶段 / Stage"], "模型加载 / Model loading")
        self.assertEqual(loaded[0]["耗时（ms） / Time (ms)"], "2000.000")


if __name__ == "__main__":
    unittest.main()
