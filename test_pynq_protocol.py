import socket
import threading
import unittest

import torch

from pynq_cosine_client import (
    CMD_PING,
    CMD_RESET,
    CMD_STEP,
    MAGIC,
    REQUEST,
    RESPONSE,
    PynqCosineClient,
    compress_feature,
    encode_threshold,
    quantize_feature,
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
        self.error = None

    def run(self):
        try:
            self.server.listen(1)
            connection, _ = self.server.accept()
            with connection:
                while True:
                    raw = recv_exact(connection, REQUEST.size)
                    magic, command, step, length, _, _, _ = REQUEST.unpack(raw)
                    if magic != MAGIC:
                        raise AssertionError("wrong protocol magic")
                    if command == CMD_STEP:
                        payload = recv_exact(connection, length)
                        self.received_lengths.append((length, len(payload)))
                        decision = 1
                        threshold_passed = 1
                        skip_streak = 2
                    else:
                        decision = 0
                        threshold_passed = 0
                        skip_streak = 0
                    connection.sendall(
                        RESPONSE.pack(
                            MAGIC,
                            0,
                            decision,
                            threshold_passed,
                            skip_streak,
                            step,
                            321,
                            400,
                            0.9995,
                        )
                    )
                    if command == CMD_STEP:
                        return
                    if command not in (CMD_PING, CMD_RESET):
                        raise AssertionError(f"unexpected command {command}")
        except Exception as exc:
            self.error = exc
        finally:
            self.server.close()


class PynqProtocolTest(unittest.TestCase):
    def test_cfg_feature_is_reduced_and_sent(self):
        server = FakePynqServer()
        server.start()
        client = PynqCosineClient("127.0.0.1", server.port)
        try:
            client.connect()
            client.reset()
            single = torch.linspace(-1.0, 1.0, 4 * 64 * 64).reshape(1, 4, 64, 64)
            feature = torch.cat([single, single], dim=0)
            result = client.decide(feature, 7, 0.999, 5, 3)
        finally:
            client.close()
        server.join(timeout=2)

        self.assertIsNone(server.error)
        self.assertEqual(server.received_lengths, [(4096, 4096)])
        self.assertTrue(result.should_skip)
        self.assertTrue(result.threshold_passed)
        self.assertEqual(result.step_index, 7)
        self.assertEqual(client.total_bytes_sent, 4096)
        self.assertAlmostEqual(result.similarity, 0.9995)
        self.assertGreaterEqual(result.prepare_ms, 0.0)
        self.assertEqual(result.feature_bytes, 4096)
        self.assertEqual(result.skip_streak, 2)
        self.assertEqual(result.threshold_q15, int(0.999 * 32768))
        self.assertEqual(
            result.effective_threshold, result.threshold_q15 / 32768.0
        )

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


if __name__ == "__main__":
    unittest.main()
