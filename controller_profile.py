"""Validate and load an explicitly selected, versioned controller profile."""

import json
import math
from pathlib import Path


FIELDS = {"middle_threshold", "late_margin", "late_min", "late_max",
          "warmup_ratio", "max_consecutive_skips", "distance_threshold",
          "predictor_damping"}


def load_controller_profile(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Controller profile must be a JSON object")
    profile = data.get("profile", data)
    if not isinstance(profile, dict) or set(profile) != FIELDS:
        raise ValueError("Controller profile must contain exactly: " + ", ".join(sorted(FIELDS)))
    for name, value in profile.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    for name in ("middle_threshold", "late_min", "late_max", "warmup_ratio", "predictor_damping"):
        if not 0 <= profile[name] <= 1:
            raise ValueError(f"{name} must be within [0, 1]")
    if profile["late_min"] > profile["late_max"]:
        raise ValueError("late_min cannot exceed late_max")
    if profile["late_margin"] < 0 or not 0 <= profile["distance_threshold"] <= 2:
        raise ValueError("Invalid margin or distance threshold")
    if not isinstance(profile["max_consecutive_skips"], int) or profile["max_consecutive_skips"] < 0:
        raise ValueError("max_consecutive_skips must be a non-negative integer")
    return profile


def profile_globals(profile, steps):
    return {
        "DYNAMIC_THRESHOLD_ENABLED": True,
        "WARMUP_STEPS": max(1, round(steps * profile["warmup_ratio"])),
        "MIDDLE_SIMILARITY_THRESHOLD": profile["middle_threshold"],
        "LATE_THRESHOLD_MARGIN": profile["late_margin"],
        "LATE_THRESHOLD_MIN": profile["late_min"],
        "LATE_THRESHOLD_MAX": profile["late_max"],
        "MAX_CONSECUTIVE_SKIPS": profile["max_consecutive_skips"],
        "WARMUP_SIMILARITY_THRESHOLD": 0.99995,
        "LATE_THRESHOLD_START_RATIO": 0.7,
        "THRESHOLD_EMA_ALPHA": 0.2,
    }
