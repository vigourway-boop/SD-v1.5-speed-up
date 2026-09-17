import unittest

from robustness_evaluation import select_candidate


def row(split, variant, speedup, lpips):
    return dict(split=split, steps=100, variant=variant, speedup=speedup,
                lpips=lpips, seconds=1, psnr_db=30, ssim=.95, clip_cosine_delta=0,
                peak_allocated_mib=100, executed=40)


class RobustnessSelectionTest(unittest.TestCase):
    def test_falls_back_to_quality_and_reports_speed_failure(self):
        rows = [row("calibration", "guarded", 1.8, .1), row("calibration", "balanced", 1.9, .2)]
        self.assertEqual(select_candidate(rows, ("guarded", "balanced")), ("guarded", False))

    def test_fresh_validation_cannot_change_selection(self):
        rows = [row("calibration", "guarded", 2.1, .1), row("calibration", "balanced", 2.2, .2),
                row("fresh_validation", "guarded", 1, .9)]
        self.assertEqual(select_candidate(rows, ("guarded", "balanced")), ("guarded", True))


if __name__ == "__main__":
    unittest.main()
