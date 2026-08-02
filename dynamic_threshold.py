"""Time-aware adaptive cosine threshold scheduling."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdPoint:
    value: float
    phase: str
    adjacent_similarity_ema: float | None


class DynamicThresholdController:
    """Schedule thresholds and learn late-step thresholds from adjacent steps."""

    def __init__(
        self,
        *,
        total_steps: int,
        warmup_steps: int,
        enabled: bool,
        fixed_threshold: float,
        warmup_threshold: float,
        middle_threshold: float,
        late_start_ratio: float,
        late_margin: float,
        late_min: float,
        late_max: float,
        ema_alpha: float,
    ):
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0 <= warmup_steps <= total_steps:
            raise ValueError("warmup_steps must be within total_steps")
        if not 0.0 <= late_start_ratio <= 1.0:
            raise ValueError("late_start_ratio must be between 0 and 1")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if late_min > late_max:
            raise ValueError("late_min cannot exceed late_max")

        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.enabled = enabled
        self.fixed_threshold = fixed_threshold
        self.warmup_threshold = warmup_threshold
        self.middle_threshold = middle_threshold
        self.late_start_step = max(
            warmup_steps, min(total_steps, round(total_steps * late_start_ratio))
        )
        self.late_margin = late_margin
        self.late_min = late_min
        self.late_max = late_max
        self.ema_alpha = ema_alpha
        self._adjacent_similarity_ema: float | None = None
        self._previous_should_skip: bool | None = None

    def point_for(self, step_index: int) -> ThresholdPoint:
        if not 0 <= step_index < self.total_steps:
            raise ValueError(f"step_index {step_index} is outside the schedule")
        if not self.enabled:
            return ThresholdPoint(
                self.fixed_threshold, "fixed", self._adjacent_similarity_ema
            )
        if step_index < self.warmup_steps:
            return ThresholdPoint(
                self.warmup_threshold, "warmup", self._adjacent_similarity_ema
            )
        if step_index < self.late_start_step:
            return ThresholdPoint(
                self.middle_threshold, "middle", self._adjacent_similarity_ema
            )

        if self._adjacent_similarity_ema is None:
            threshold = self.middle_threshold
        else:
            threshold = self._adjacent_similarity_ema - self.late_margin
        threshold = min(self.late_max, max(self.late_min, threshold))
        return ThresholdPoint(threshold, "late_adaptive", self._adjacent_similarity_ema)

    def observe(self, similarity: float, should_skip: bool) -> None:
        # If the previous step executed UNet, FPGA's reference is exactly one
        # timestep old. Those adjacent comparisons form a stable online anchor.
        if (
            self._previous_should_skip is False
            and math.isfinite(similarity)
        ):
            if self._adjacent_similarity_ema is None:
                self._adjacent_similarity_ema = similarity
            else:
                alpha = self.ema_alpha
                self._adjacent_similarity_ema = (
                    alpha * similarity
                    + (1.0 - alpha) * self._adjacent_similarity_ema
                )
        self._previous_should_skip = bool(should_skip)

    @property
    def adjacent_similarity_ema(self) -> float | None:
        return self._adjacent_similarity_ema
