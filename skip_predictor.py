"""Noise-prediction reuse strategies for skipped diffusion steps."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SkipPrediction:
    value: object
    mode: str
    factor: float


class SkipNoisePredictor:
    """Predict a skipped UNet output from the two latest real outputs."""

    def __init__(self, mode="linear", damping=0.5, max_factor=1.5):
        if mode not in {"linear", "reuse"}:
            raise ValueError("mode must be 'linear' or 'reuse'")
        if not math.isfinite(damping) or not 0.0 <= damping <= 1.0:
            raise ValueError("damping must be finite and within [0, 1]")
        if not math.isfinite(max_factor) or max_factor < 0.0:
            raise ValueError("max_factor must be finite and non-negative")
        self.mode = mode
        self.damping = float(damping)
        self.max_factor = float(max_factor)
        self._history = []

    @staticmethod
    def _timestep_value(timestep):
        return float(timestep.item() if hasattr(timestep, "item") else timestep)

    def record(self, timestep, prediction):
        self._history.append((self._timestep_value(timestep), prediction.detach()))
        if len(self._history) > 2:
            self._history.pop(0)

    @property
    def cache_bytes(self):
        return sum(value.numel() * value.element_size() for _, value in self._history)

    def predict(self, timestep):
        if not self._history:
            raise RuntimeError("Cannot predict before a real UNet output is recorded")

        last_timestep, last_prediction = self._history[-1]
        if self.mode == "reuse" or len(self._history) < 2:
            return SkipPrediction(last_prediction, "reuse", 0.0)

        previous_timestep, previous_prediction = self._history[-2]
        denominator = last_timestep - previous_timestep
        if abs(denominator) < 1e-12:
            return SkipPrediction(last_prediction, "reuse", 0.0)

        current_timestep = self._timestep_value(timestep)
        raw_factor = (current_timestep - last_timestep) / denominator
        factor = min(self.max_factor, max(0.0, raw_factor * self.damping))
        if factor == 0.0:
            return SkipPrediction(last_prediction, "reuse", 0.0)

        predicted = last_prediction + factor * (last_prediction - previous_prediction)
        return SkipPrediction(predicted, "linear", factor)
