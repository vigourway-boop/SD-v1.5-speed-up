import unittest

from dynamic_threshold import DynamicThresholdController


def make_controller(**overrides):
    settings = {
        "total_steps": 100,
        "warmup_steps": 15,
        "enabled": True,
        "fixed_threshold": 0.999,
        "warmup_threshold": 0.99995,
        "middle_threshold": 0.99960,
        "late_start_ratio": 0.70,
        "late_margin": 0.00035,
        "late_min": 0.99945,
        "late_max": 0.99965,
        "ema_alpha": 1.0,
    }
    settings.update(overrides)
    return DynamicThresholdController(**settings)


class DynamicThresholdTest(unittest.TestCase):
    def test_schedule_phases(self):
        controller = make_controller()
        self.assertEqual(controller.point_for(0).phase, "warmup")
        self.assertEqual(controller.point_for(14).value, 0.99995)
        self.assertEqual(controller.point_for(15).phase, "middle")
        self.assertEqual(controller.point_for(69).value, 0.99960)
        self.assertEqual(controller.point_for(70).phase, "late_adaptive")

    def test_late_threshold_uses_adjacent_similarity(self):
        controller = make_controller()
        controller.observe(float("nan"), should_skip=False)
        controller.observe(0.99990, should_skip=True)
        point = controller.point_for(70)
        self.assertAlmostEqual(point.adjacent_similarity_ema, 0.99990)
        self.assertAlmostEqual(point.value, 0.99955)

        # This comparison spans a skipped step, so it must not move the anchor.
        controller.observe(0.99910, should_skip=False)
        self.assertAlmostEqual(
            controller.point_for(71).adjacent_similarity_ema, 0.99990
        )

        # The previous step executed UNet, so this adjacent comparison updates it.
        controller.observe(0.99970, should_skip=False)
        self.assertEqual(controller.point_for(72).value, 0.99945)

    def test_fixed_mode_preserves_legacy_threshold(self):
        controller = make_controller(enabled=False)
        point = controller.point_for(80)
        self.assertEqual(point.phase, "fixed")
        self.assertEqual(point.value, 0.999)

    def test_invalid_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            make_controller(late_min=0.9998, late_max=0.9997)
        with self.assertRaises(ValueError):
            make_controller(ema_alpha=0.0)


if __name__ == "__main__":
    unittest.main()
