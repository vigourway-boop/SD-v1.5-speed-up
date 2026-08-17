import socket
import threading
import unittest
from unittest import mock

import torch

from pynq_cosine_client import (
    PynqCosineClient,
    compress_feature,
    encode_distance_threshold,
    encode_threshold,
    quantize_feature,
)
from pynq_protocol import (
    CMD_CONFIG,
    CMD_PING,
    CMD_RESET,
    CMD_STEP,
    CONFIG,
    MAGIC,
    PHASE_MIDDLE,
    REQUEST,
    RESPONSE,
)


def recv_exact(connection, size):
    result = bytearray(size)
    view = memoryview(result)
    received = 0
    while received < size:
        count = connection.recv_into(view[received:])
        if count == 0:
            raise ConnectionError("client disconnected")
        received += count
    return bytes(result)


class FakePynqServer(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(("127.0.0.1", 0))
        self.port = self.server.getsockname()[1]
        self.received_lengths = []
        self.received_config = None
        self.error = None

    def run(self):
        try:
            self.server.listen(1)
            connection, _ = self.server.accept()
            with connection:
                while True:
                    raw = recv_exact(connection, REQUEST.size)
                    magic, command, step, length = REQUEST.unpack(raw)
                    if magic != MAGIC:
                        raise AssertionError("wrong protocol magic")
                    if command == CMD_CONFIG:
                        payload = recv_exact(connection, length)
                        self.received_config = CONFIG.unpack(payload)
                        decision = 0
                        cosine_passed = 0
                        distance_passed = 0
                        skip_streak = 0
                    elif command == CMD_STEP:
                        payload = recv_exact(connection, length)
                        self.received_lengths.append((length, len(payload)))
                        decision = 1
                        cosine_passed = 1
                        distance_passed = 1
                        skip_streak = 2
                    else:
                        decision = 0
                        cosine_passed = 0
                        distance_passed = 0
                        skip_streak = 0
                    connection.sendall(
                        RESPONSE.pack(
                            MAGIC,
                            0,
                            decision,
                            cosine_passed,
                            distance_passed,
                            skip_streak,
                            PHASE_MIDDLE,
                            step,
                            321,
                            400,
                            int(0.9996 * 32768),
                            2 << 20,
                            0.9995,
                            0.0004,
                            0.9996,
                            0.9998,
                        )
                    )
                    if command == CMD_STEP:
                        return
                    if command not in (CMD_PING, CMD_RESET, CMD_CONFIG):
                        raise AssertionError(f"unexpected command {command}")
        except Exception as exc:
            self.error = exc
        finally:
            self.server.close()


class PynqProtocolTest(unittest.TestCase):
    def test_connect_with_retry_reports_successful_attempt(self):
        client = PynqCosineClient("127.0.0.1", 9000)
        with mock.patch.object(
            client,
            "connect",
            side_effect=[ConnectionError("not ready"), None],
        ) as connect:
            attempt = client.connect_with_retry(attempts=2, delay_seconds=0.0)

        self.assertEqual(attempt, 2)
        self.assertEqual(connect.call_count, 2)

    def test_connect_with_retry_validates_settings(self):
        client = PynqCosineClient("127.0.0.1", 9000)
        with self.assertRaises(ValueError):
            client.connect_with_retry(attempts=0)
        with self.assertRaises(ValueError):
            client.connect_with_retry(delay_seconds=-1.0)

    def test_cfg_feature_is_reduced_and_sent(self):
        server = FakePynqServer()
        server.start()
        client = PynqCosineClient("127.0.0.1", server.port)
        try:
            client.connect()
            client.reset()
            client.configure(
                enabled=True,
                total_steps=100,
                warmup_steps=15,
                max_consecutive_skips=3,
                fixed_threshold=0.999,
                warmup_threshold=0.99995,
                middle_threshold=0.9996,
                late_start_ratio=0.7,
                late_margin=0.00035,
                late_min=0.99945,
                late_max=0.99965,
                ema_alpha=0.2,
                distance_threshold=2.0,
            )
            single = torch.linspace(-1.0, 1.0, 4 * 64 * 64).reshape(1, 4, 64, 64)
            feature = torch.cat([single, single], dim=0)
            result = client.decide(feature, 7)
        finally:
            client.close()
        server.join(timeout=2)

        self.assertIsNone(server.error)
        self.assertEqual(server.received_lengths, [(4096, 4096)])
        self.assertIsNotNone(server.received_config)
        self.assertTrue(result.should_skip)
        self.assertTrue(result.threshold_passed)
        self.assertTrue(result.cosine_passed)
        self.assertTrue(result.distance_passed)
        self.assertEqual(result.step_index, 7)
        self.assertEqual(client.total_bytes_sent, 4096)
        self.assertAlmostEqual(result.similarity, 0.9995)
        self.assertGreaterEqual(result.prepare_ms, 0.0)
        self.assertEqual(result.feature_bytes, 4096)
        self.assertEqual(result.skip_streak, 2)
        self.assertEqual(result.threshold_q15, int(0.9996 * 32768))
        self.assertEqual(
            result.effective_threshold, result.threshold_q15 / 32768.0
        )
        self.assertEqual(result.distance_threshold_q20, 2 << 20)
        self.assertEqual(result.effective_distance_threshold, 2.0)
        self.assertAlmostEqual(result.normalized_distance, 0.0004)
        self.assertEqual(result.threshold_phase, "middle")
        self.assertAlmostEqual(result.threshold_requested, 0.9996)
        self.assertAlmostEqual(result.adjacent_similarity_ema, 0.9998)

    def test_zero_feature_quantization(self):
        result = quantize_feature(torch.zeros(4, 64, 64))
        self.assertEqual(result.dtype, "int8")
        self.assertEqual(result.size, 4096)
        self.assertEqual(int(result.max()), 0)

    def test_invalid_downsample_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            compress_feature(torch.zeros(4, 63, 64))

    def test_threshold_encoding_is_bounded(self):
        self.assertEqual(encode_threshold(-1.0), (0, 0.0))
        self.assertEqual(encode_threshold(1.0), (32767, 32767 / 32768.0))
        with self.assertRaises(ValueError):
            encode_threshold(float("nan"))

    def test_distance_threshold_encoding_is_q20(self):
        self.assertEqual(encode_distance_threshold(2.0), (2 << 20, 2.0))
        self.assertEqual(encode_distance_threshold(-1.0), (0, 0.0))
        with self.assertRaises(ValueError):
            encode_distance_threshold(float("inf"))


if __name__ == "__main__":
    unittest.main()
