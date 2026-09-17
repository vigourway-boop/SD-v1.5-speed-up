import json
import tempfile
import unittest
from pathlib import Path

from controller_profile import load_controller_profile, profile_globals
from temporal_evaluation import BASE_PROFILE


class ControllerProfileTest(unittest.TestCase):
    def load(self, profile):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profile.json"
            path.write_text(json.dumps({"name": "test", "profile": profile}), encoding="utf-8")
            return load_controller_profile(path)

    def test_evaluation_profile_can_be_used_without_changing_defaults(self):
        result = self.load(BASE_PROFILE)
        self.assertEqual(result, BASE_PROFILE)
        self.assertEqual(profile_globals(result, 20)["WARMUP_STEPS"], 3)

    def test_non_finite_or_fractional_skip_count_is_rejected(self):
        for patch in ({"distance_threshold": float("nan")},
                      {"max_consecutive_skips": 1.5}, {"late_min": 1.0},
                      {"unexpected": 3}):
            with self.assertRaises(ValueError):
                self.load({**BASE_PROFILE, **patch})


if __name__ == "__main__":
    unittest.main()
