import math
import unittest
from unittest import mock

import torch

from decision_backend import AutoDecisionClient, create_decision_client
from pc_skip_controller import PcCosineClient


def configure(client, *, max_skips=3, distance_threshold=2.0):
    client.configure(
        enabled=False,
        total_steps=10,
        warmup_steps=0,
        max_consecutive_skips=max_skips,
        fixed_threshold=0.99,
        warmup_threshold=0.99,
        middle_threshold=0.99,
        late_start_ratio=0.7,
        late_margin=0.00035,
        late_min=0.98,
        late_max=0.999,
        ema_alpha=0.2,
        distance_threshold=distance_threshold,
    )


class PcSkipControllerTest(unittest.TestCase):
    def test_zero_norm_distance_flag_matches_fpga(self):
        client = PcCosineClient()
        configure(client, distance_threshold=2.0)
        client.decide(torch.zeros(1, 1, 4, 4), 0)
        result = client.decide(torch.ones(1, 1, 4, 4), 1)
        self.assertTrue(result.distance_passed)
        self.assertFalse(result.cosine_passed)
        self.assertFalse(result.should_skip)

    def test_wire_precision_controls_late_phase_rounding(self):
        client = PcCosineClient()
        configure(client)
        import struct
        self.assertEqual(client._controller.middle_threshold,
                         struct.unpack("!f", struct.pack("!f", 0.99))[0])

    def test_identical_feature_skips_after_reference_is_created(self):
        client = PcCosineClient()
        configure(client)
        feature = torch.arange(64, dtype=torch.float32).reshape(1, 4, 4, 4)

        first = client.decide(feature, 0)
        second = client.decide(feature, 1)

        self.assertFalse(first.should_skip)
        self.assertTrue(math.isnan(first.similarity))
        self.assertTrue(second.should_skip)
        self.assertTrue(second.cosine_passed)
        self.assertTrue(second.distance_passed)
        self.assertAlmostEqual(second.similarity, 1.0)
        self.assertEqual(second.feature_bytes, 16)
        self.assertGreaterEqual(second.decision_ms, 0.0)

    def test_maximum_skip_streak_forces_reference_update(self):
        client = PcCosineClient()
        configure(client, max_skips=2)
        feature = torch.arange(64, dtype=torch.float32).reshape(1, 4, 4, 4)

        decisions = [client.decide(feature, step).should_skip for step in range(4)]

        self.assertEqual(decisions, [False, True, True, False])

    def test_distance_gate_can_reject_cosine_match(self):
        client = PcCosineClient()
        configure(client, distance_threshold=0.0001)
        reference = torch.ones(1, 1, 4, 4)
        candidate = reference.clone()
        candidate[:, :, 2:, 2:] = 0.9

        client.decide(reference, 0)
        result = client.decide(candidate, 1)

        self.assertTrue(result.cosine_passed)
        self.assertFalse(result.distance_passed)
        self.assertFalse(result.should_skip)
        self.assertGreater(result.normalized_distance, 0.0001)

    def test_auto_backend_falls_back_when_pynq_is_unreachable(self):
        client = AutoDecisionClient("192.0.2.1", 9000)
        with mock.patch.object(
            client._client,
            "connect_with_retry",
            side_effect=ConnectionError("board offline"),
        ):
            attempt = client.connect_with_retry(1, 0.0)

        self.assertEqual(attempt, 1)
        self.assertEqual(client.backend_name, "pc")
        self.assertTrue(client.fallback_used)
        self.assertIn("board offline", client.fallback_reason)

    def test_factory_supports_forced_pc_backend(self):
        client = create_decision_client("pc", "unused", 0)
        self.assertIsInstance(client, PcCosineClient)


if __name__ == "__main__":
    unittest.main()
