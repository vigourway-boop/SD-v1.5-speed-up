import unittest

import torch

from temporal_evaluation import (BASE_PROFILE, PROFILES, controller_config,
                                 feature_diagnostics, select_profile, settings)
import combined_speed_test as sd


class TemporalEvaluationTest(unittest.TestCase):
    def test_settings_restore_even_after_failure(self):
        previous = (sd.PROMPT, sd.BASE_STEPS, sd.WARMUP_STEPS)
        with self.assertRaises(RuntimeError):
            with settings("test prompt", 33, 20, BASE_PROFILE):
                self.assertEqual(sd.WARMUP_STEPS, 3)
                self.assertEqual(controller_config(BASE_PROFILE, 20)["warmup_steps"], 3)
                raise RuntimeError("interrupted")
        self.assertEqual((sd.PROMPT, sd.BASE_STEPS, sd.WARMUP_STEPS), previous)

    def test_scaled_features_expose_int8_scale_information_loss(self):
        feature = torch.arange(64, dtype=torch.float32).reshape(1, 4, 4, 4)
        row = feature_diagnostics([feature, feature * 2])[0]
        self.assertAlmostEqual(row["raw_adjacent_cosine"], 1.0)
        self.assertGreater(row["raw_normalized_distance"], 0.0)
        self.assertEqual(row["int8_delta_zero_fraction"], 1.0)
        self.assertGreater(row["raw_delta_rms"], 0.0)

    def test_selection_ignores_held_out_speed_and_rejects_poor_quality(self):
        rows = []
        for name, speed, lpips in [("current", 1.8, 0.02), ("balanced", 2.1, 0.05), ("fast", 3.0, 0.3)]:
            for split in ("calibration", "validation"):
                rows.append(dict(split=split, steps=100, variant=name,
                                 speedup=speed if split == "calibration" else 100.0,
                                 seconds=5, psnr_db=30, ssim=.95, lpips=lpips,
                                 clip_cosine_delta=0, peak_allocated_mib=100, executed=40))
        selected, details = select_profile(rows)
        self.assertEqual(selected, "balanced")
        self.assertFalse(details["held_out_data_used"])


if __name__ == "__main__":
    unittest.main()
