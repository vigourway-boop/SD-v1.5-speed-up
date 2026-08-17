import unittest
from unittest import mock

from combined_speed_test import run_dynamic_with_recovery


class FakeClient:
    def __init__(self):
        self.connect_calls = 0
        self.close_calls = 0

    def connect_with_retry(self, attempts, delay_seconds):
        self.connect_calls += 1
        return 1

    def close(self):
        self.close_calls += 1


class DynamicRecoveryTest(unittest.TestCase):
    def test_connection_failure_restarts_the_complete_dynamic_run(self):
        client = FakeClient()
        completed = (5.0, 51, 49, object(), [], {"generation_ms": 5000.0})
        with mock.patch(
            "combined_speed_test.run_with_dynamic_steps",
            side_effect=[ConnectionError("link lost"), completed],
        ) as dynamic_run:
            result = run_dynamic_with_recovery(
                "output.png",
                client,
                2.0,
                "linear",
                0.5,
                1.5,
                generation_retries=1,
                connect_attempts=3,
                retry_delay_seconds=0.0,
            )

        self.assertEqual(dynamic_run.call_count, 2)
        self.assertEqual(client.connect_calls, 2)
        self.assertEqual(client.close_calls, 1)
        self.assertEqual(result[:-1], completed)
        self.assertEqual(result[-1], 1)

    def test_failure_is_raised_after_retry_budget_is_exhausted(self):
        client = FakeClient()
        with mock.patch(
            "combined_speed_test.run_with_dynamic_steps",
            side_effect=ConnectionError("link lost"),
        ):
            with self.assertRaises(ConnectionError):
                run_dynamic_with_recovery(
                    "output.png",
                    client,
                    2.0,
                    "linear",
                    0.5,
                    1.5,
                    generation_retries=0,
                    connect_attempts=1,
                    retry_delay_seconds=0.0,
                )


if __name__ == "__main__":
    unittest.main()

