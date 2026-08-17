import unittest

import torch

from skip_predictor import SkipNoisePredictor


class SkipNoisePredictorTest(unittest.TestCase):
    def test_linear_prediction_uses_timestep_spacing(self):
        predictor = SkipNoisePredictor(mode="linear", damping=0.5, max_factor=1.5)
        predictor.record(1000, torch.tensor([1.0, 2.0]))
        predictor.record(900, torch.tensor([3.0, 6.0]))

        result = predictor.predict(800)

        self.assertEqual(result.mode, "linear")
        self.assertAlmostEqual(result.factor, 0.5)
        torch.testing.assert_close(result.value, torch.tensor([4.0, 8.0]))

    def test_factor_is_capped_after_consecutive_skips(self):
        predictor = SkipNoisePredictor(mode="linear", damping=1.0, max_factor=1.5)
        predictor.record(1000, torch.tensor([0.0]))
        predictor.record(900, torch.tensor([2.0]))

        result = predictor.predict(600)

        self.assertAlmostEqual(result.factor, 1.5)
        torch.testing.assert_close(result.value, torch.tensor([5.0]))

    def test_reuse_mode_preserves_legacy_behavior(self):
        predictor = SkipNoisePredictor(mode="reuse")
        predictor.record(1000, torch.tensor([1.0]))
        predictor.record(900, torch.tensor([3.0]))

        result = predictor.predict(800)

        self.assertEqual(result.mode, "reuse")
        self.assertEqual(result.factor, 0.0)
        torch.testing.assert_close(result.value, torch.tensor([3.0]))

    def test_first_real_output_falls_back_to_reuse(self):
        predictor = SkipNoisePredictor(mode="linear")
        predictor.record(1000, torch.tensor([4.0]))

        result = predictor.predict(900)

        self.assertEqual(result.mode, "reuse")
        torch.testing.assert_close(result.value, torch.tensor([4.0]))

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            SkipNoisePredictor(mode="unknown")
        with self.assertRaises(ValueError):
            SkipNoisePredictor(damping=1.1)
        with self.assertRaises(ValueError):
            SkipNoisePredictor(max_factor=-1.0)


if __name__ == "__main__":
    unittest.main()

