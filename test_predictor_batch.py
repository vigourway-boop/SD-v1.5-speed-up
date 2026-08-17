import unittest

from predictor_batch_experiment import aggregate_results


def row(seed, mode, psnr, ssim, lpips, clip_delta, speedup, skipped):
    return {
        "seed": seed,
        "predictor": mode,
        "executed_unet_steps": 100 - skipped,
        "skipped_steps": skipped,
        "baseline_seconds": 10.0,
        "dynamic_seconds": 5.0,
        "speedup": speedup,
        "psnr_db": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "clip_delta": clip_delta,
        "recovery_restarts": 0,
    }


class PredictorBatchTest(unittest.TestCase):
    def test_aggregate_has_mode_and_paired_statistics(self):
        results = [
            row(1, "linear", 31.0, 0.95, 0.02, 0.1, 2.0, 50),
            row(1, "reuse", 29.0, 0.90, 0.04, -0.1, 2.1, 51),
            row(2, "linear", 28.0, 0.91, 0.03, -0.1, 1.9, 49),
            row(2, "reuse", 30.0, 0.93, 0.02, 0.0, 1.8, 48),
        ]

        aggregate = aggregate_results(results)

        self.assertEqual(aggregate["mode_stats"]["linear"]["count"], 2)
        self.assertEqual(aggregate["paired_summary"]["count"], 2)
        self.assertEqual(aggregate["paired_summary"]["linear_wins"]["psnr"], 1)
        self.assertAlmostEqual(
            aggregate["paired"][0]["psnr_linear_minus_reuse_db"], 2.0
        )
        self.assertEqual(
            aggregate["paired_summary"]["two_sided_sign_test_p"]["psnr"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
